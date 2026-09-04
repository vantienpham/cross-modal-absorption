"""Dropping visual tokens inside a running decoder.

Pruning happens at one decoder layer. Up to that layer the model sees the whole
sequence; from the next layer on it sees only the kept positions, and the
key/value cache from that point is correspondingly shorter, which is where the
memory saving comes from. This is the arrangement the training-free pruning
literature uses, and it is reproduced here rather than approximated so that
accuracy numbers mean what they usually mean.

The implementation wraps each decoder layer instead of patching the parent
module's forward. The parent hands *every* layer the same full-length
``attention_mask``, ``position_embeddings`` and ``cache_position``, so a wrapper
sitting on a downstream layer can narrow those to the kept positions on the way
in, while the hidden states arrive already narrowed from the layer before. No
private method of the modelling code is overridden, so a transformers upgrade
that changes the parent loop cannot silently produce wrong numbers here; it
produces a shape error.

Two facts about the arrangement are worth stating because they bound what may be
claimed from it.

*Positions are not renumbered.* Kept tokens carry their original rotary
positions, so the geometry the later layers see is the original one with gaps,
not a compacted sequence. Renumbering would change the relative distances the
model was trained on.

*The attention signal is computed, not materialised.* Selection needs the mass
flowing from post-image text queries to visual keys at the pruning layer. Rather
than switching the whole model to an eager kernel to read an attention matrix
out of it, :class:`AttentionProbe` re-derives that one block from the layer's own
projections. The text side is a few tens of rows, so the block is small even
when the full matrix would not be, and the model keeps whichever kernel it was
loaded with.

Correctness here is not argued, it is tested: at a budget equal to the number of
visual tokens the pruned path must reproduce the unpruned generation token for
token. ``tests/test_prune.py`` asserts exactly that, and it is the check that
catches a mask sliced along the wrong axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .models import decoder_layers


@dataclass
class PruningPlan:
    """Mutable state shared by every wrapper in one model.

    Set by the caller before a forward pass, read by the wrappers during it.
    ``keep`` is in *sequence* coordinates, not visual-token coordinates: it lists
    every position the layers after the pruning layer should retain, visual and
    text alike, so a wrapper can slice with it directly.
    """

    layer: int
    keep: torch.Tensor | None = None
    prompt_len: int | None = None
    enabled: bool = False
    #: Filled in by the probe at the pruning layer during the forward pass.
    attention: torch.Tensor | None = field(default=None, repr=False)
    hidden: torch.Tensor | None = field(default=None, repr=False)


def _slice_kwargs(kw: dict[str, Any], keep: torch.Tensor, q_len: int,
                  prompt_len: int) -> dict[str, Any]:
    """Narrow a decoder layer's positional kwargs to the kept positions.

    During prefill (``q_len > 1``) queries and keys are the same positions, so
    both axes of a 4-D mask are narrowed. During decode there is a single new
    query and the keys are this layer's own shortened cache followed by whatever
    has been generated since, so only the key axis moves and it is narrowed by
    the kept prompt positions plus every position after the prompt.
    """
    out = dict(kw)

    pe = out.get("position_embeddings")
    if pe is not None and q_len > 1:
        cos, sin = pe
        out["position_embeddings"] = (cos[:, keep], sin[:, keep])

    cp = out.get("cache_position")
    if cp is not None and q_len > 1:
        out["cache_position"] = cp[keep]

    pi = out.get("position_ids")
    if pi is not None and q_len > 1:
        out["position_ids"] = pi[:, keep]

    am = out.get("attention_mask")
    if am is not None and am.dim() == 4:
        if q_len > 1:
            out["attention_mask"] = am[:, :, keep, :][:, :, :, keep]
        else:
            total = am.shape[-1]
            tail = torch.arange(prompt_len, total, device=keep.device)
            out["attention_mask"] = am[:, :, :, torch.cat([keep, tail])]
    return out


class PrunedLayer(nn.Module):
    """One decoder layer, narrowing its positional kwargs when pruning is active."""

    def __init__(self, inner: nn.Module, plan: PruningPlan, index: int) -> None:
        super().__init__()
        self.inner = inner
        self.plan = plan
        self.index = index

    def forward(self, hidden_states: torch.Tensor, **kw: Any):
        plan = self.plan
        active = plan.enabled and plan.keep is not None and self.index > plan.layer
        if active:
            keep = plan.keep.to(hidden_states.device)
            kw = _slice_kwargs(kw, keep, hidden_states.shape[1], plan.prompt_len or 0)
        out = self.inner(hidden_states, **kw)

        if plan.enabled and self.index == plan.layer:
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:
                # Prefill only. During decode this layer sees one token, and
                # overwriting the recorded states with it would leave the
                # selection inputs holding the last generated token instead of
                # the prompt they were computed from.
                plan.hidden = h.detach()
            if plan.keep is not None and h.shape[1] > 1:
                keep = plan.keep.to(h.device)
                h = h[:, keep]
                out = (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return out


class AttentionProbe:
    """Text-query-to-visual-key attention mass at one layer, without the matrix.

    Registered as a forward pre-hook on the layer's attention module. It reads
    the same hidden states the module is about to consume, applies that module's
    own query and key projections and rotary embedding, and forms only the
    ``(n_text, n_all)`` logit block for the post-image text queries. The softmax
    still runs over every key, because an attention *probability* is only
    defined against the full row; the visual columns are summed afterwards.

    Restricted to decoders whose rotary embedding takes the LLaMA form, which
    covers the LLaVA family used here. Families with a different rotary scheme
    (Qwen2.5-VL's multimodal rope, for instance) would need their own variant,
    and are used in this project only for measurements that need no attention.
    """

    def __init__(self, attn_module: nn.Module, n_heads_keep: int = 12) -> None:
        self.mod = attn_module
        self.n_heads_keep = n_heads_keep
        self.result: torch.Tensor | None = None
        self._text: torch.Tensor | None = None
        self._visual: torch.Tensor | None = None
        self._handle = None

    def arm(self, text_idx: torch.Tensor, visual_idx: torch.Tensor) -> None:
        self._text, self._visual, self.result = text_idx, visual_idx, None
        if self._handle is None:
            self._handle = self.mod.register_forward_pre_hook(self, with_kwargs=True)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __call__(self, module, args, kwargs):
        if self._text is None:
            return None
        h = kwargs.get("hidden_states")
        if h is None and args:
            h = args[0]
        pe = kwargs.get("position_embeddings")
        if h is None or pe is None or h.shape[1] <= 1:
            return None
        self.result = self._mass(module, h, pe)
        self._text = None      # one shot: later layers must not overwrite it
        return None

    @torch.no_grad()
    def _mass(self, module, h: torch.Tensor, pe) -> torch.Tensor:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        cos, sin = pe
        b, L, _ = h.shape
        hd = getattr(module, "head_dim", None)
        if hd is None:
            raise AttributeError("attention module exposes no head_dim; probe needs it")
        q = module.q_proj(h).view(b, L, -1, hd).transpose(1, 2)
        k = module.k_proj(h).view(b, L, -1, hd).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # Grouped-query attention: expand the key heads to match the query heads
        # so that head h of q is scored against the key head it actually shares.
        if k.shape[1] != q.shape[1]:
            rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(rep, dim=1)

        t = self._text.to(h.device)
        qt = q[:, :, t]                                    # (b, H, n_text, hd)
        logits = torch.matmul(qt, k.transpose(-1, -2)) / (hd ** 0.5)
        # Causal: a text query may not see a key after it. The text positions are
        # all after every visual one, so this only masks text-on-later-text, but
        # leaving it out would put probability mass on the future and change the
        # normalisation of the visual columns.
        keys = torch.arange(L, device=h.device)
        logits = logits.masked_fill(keys.view(1, 1, 1, -1) > t.view(1, 1, -1, 1), float("-inf"))
        p = torch.softmax(logits.float(), dim=-1)

        v = self._visual.to(h.device)
        per_head = p[:, :, :, v].sum(dim=2)[0]             # (H, n_visual)
        # Heads differ enormously in how much they look at the image at all;
        # averaging over all of them buries the signal in heads that are doing
        # something else. Keep the ones with the most visual mass.
        mass = per_head.sum(dim=1)
        keep_h = torch.topk(mass, min(self.n_heads_keep, mass.numel())).indices
        return per_head[keep_h].sum(dim=0).to("cpu", torch.float32)


class PrunableModel:
    """Context manager installing pruning wrappers on a loaded VLM.

    ``with PrunableModel(model, layer=..) as p:`` swaps every decoder layer for a
    :class:`PrunedLayer` and restores the originals on exit, so a failure inside
    the block cannot leave a model that silently prunes afterwards.
    """

    def __init__(self, model: nn.Module, layer: int, n_heads_keep: int = 12) -> None:
        self.model = model
        self.layers = decoder_layers(model)
        self.plan = PruningPlan(layer=layer)
        self.probe = AttentionProbe(self.layers[layer].self_attn, n_heads_keep)
        self._original: list[nn.Module] = []

    def __enter__(self) -> "PrunableModel":
        self._original = list(self.layers)
        for i, layer in enumerate(self._original):
            self.layers[i] = PrunedLayer(layer, self.plan, i)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.probe.remove()
        for i, layer in enumerate(self._original):
            self.layers[i] = layer
        self._original = []

    def disable(self) -> None:
        self.plan.enabled = False
        self.plan.keep = None

    def enable(self, keep: torch.Tensor, prompt_len: int) -> None:
        self.plan.enabled = True
        self.plan.keep = keep.to(torch.long)
        self.plan.prompt_len = prompt_len
