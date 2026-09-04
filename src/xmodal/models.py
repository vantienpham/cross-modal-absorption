"""Loading VLMs and locating the tensors this study operates on.

Kept deliberately thin. The only real content is :func:`attention_sites`, which
decides *which* linear layers the statistics are collected for, and
:func:`describe`, which records enough of the architecture in every run that a
number can be traced back to the shape that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch
import torch.nn as nn

#: Checkpoints this project has been exercised against. Others may work; these
#: are the ones whose token-expansion behaviour has been checked.
KNOWN_MODELS = {
    "llava-hf/llava-1.5-7b-hf": {"visual_tokens": 576, "n_heads": 32},
    "llava-hf/llava-1.5-13b-hf": {"visual_tokens": 576, "n_heads": 40},
    "llava-hf/llava-v1.6-vicuna-7b-hf": {"visual_tokens": None, "n_heads": 32},
}


@dataclass
class ModelInfo:
    name: str
    n_layers: int
    n_heads: int
    #: Grouped-query attention gives K and V fewer heads than Q. Splitting a
    #: projection by n_heads would then be silently wrong for K/V, so every
    #: per-head operation keys off head_dim instead and derives the count as
    #: d_out // head_dim.
    n_kv_heads: int
    hidden_size: int
    head_dim: int
    image_token_id: int
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_vlm(
    name: str,
    dtype: torch.dtype = torch.float16,
    device_map: str | dict | None = None,
    attn_implementation: str = "eager",
):
    """Load a LLaVA-family checkpoint and its processor.

    ``attn_implementation`` selects the attention kernel. It does **not** affect
    what this project measures: the statistics are hooked on the Q/K/V
    ``nn.Linear`` modules, and the gradient at a linear layer's output is the
    same quantity whichever kernel consumes it downstream. ``sdpa`` is therefore
    safe and is much cheaper for long visual-token sequences, where ``eager``
    materialises an $L \times L$ attention matrix per layer and keeps it for
    backward -- around 17 GB for a 2880-token LLaVA-Next sequence.

    The architecture class is resolved rather than hardcoded, because LLaVA-1.5
    and LLaVA-Next need different ones (``LlavaForConditionalGeneration`` vs
    ``LlavaNextForConditionalGeneration``) and the point of running both is that
    they differ in visual token budget.
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(name)
    kwargs = dict(torch_dtype=dtype, device_map=device_map,
                  attn_implementation=attn_implementation)
    try:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(name, **kwargs)
    except (ImportError, ValueError):
        # Older transformers, or a config the auto-class does not map: fall back
        # to the class the config names for itself.
        from transformers import AutoConfig
        import transformers as tf

        cfg = AutoConfig.from_pretrained(name)
        arch = (getattr(cfg, "architectures", None) or [None])[0]
        if arch is None or not hasattr(tf, arch):
            raise
        model = getattr(tf, arch).from_pretrained(name, **kwargs)
    model.eval()
    return model, processor


def language_model(model: nn.Module) -> nn.Module:
    """Return the text decoder, across the transformers naming churn.

    ``LlavaForConditionalGeneration`` moved the decoder between
    ``model.language_model`` and ``model.model.language_model`` across releases;
    both spellings are probed rather than assumed.
    """
    for path in (
        ("model", "language_model"),
        ("language_model", "model"),
        ("language_model",),
        ("model", "text_model"),
        ("text_model",),
        ("model", "decoder"),
        ("model",),
    ):
        obj: Any = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if hasattr(obj, "layers"):
            return obj
    raise AttributeError(
        "could not locate the text decoder (no attribute path ended in .layers); "
        f"top-level attributes are {[n for n, _ in model.named_children()]}"
    )


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    return language_model(model).layers


def attention_sites(model: nn.Module, projections: tuple[str, ...] = ("q", "k", "v")) -> dict[str, nn.Linear]:
    """Map ``"layer{i}.{proj}_proj"`` to the module, for every decoder layer.

    Q/K/V are the target because they are where the KV-cache win lives and where
    every method this work compares against operates. They also share one input
    activation per layer, so a single input covariance serves all three.
    """
    sites: dict[str, nn.Linear] = {}
    for i, layer in enumerate(decoder_layers(model)):
        attn = layer.self_attn
        for p in projections:
            mod = getattr(attn, f"{p}_proj", None)
            if mod is None:
                raise AttributeError(
                    f"layer {i} self_attn has no {p}_proj; attributes are "
                    f"{[n for n, _ in attn.named_children()]}"
                )
            sites[f"layer{i}.{p}_proj"] = mod
    return sites


def enable_activation_grads(model: nn.Module) -> torch.utils.hooks.RemovableHandle:
    """Make decoder activations differentiable without storing parameter grads.

    We need ``dL/dy`` at every linear layer, but no ``dL/dW`` at all -- carrying
    parameter gradients for a 7B model costs a second copy of the weights for
    statistics we never read.

    Freezing every parameter alone leaves the graph empty (``element 0 of
    tensors does not require grad``). Instead the token-embedding output is
    turned into a leaf that requires grad: it has no grad-requiring inputs, so
    ``requires_grad_(True)`` makes it a graph root, and every activation
    downstream of it -- including the visual positions, which are scattered into
    the same tensor -- becomes a non-leaf carrying ``grad_fn``. Backward then
    populates activation gradients everywhere and parameter gradients nowhere.
    """
    emb = model.get_input_embeddings()
    if emb is None:  # pragma: no cover - would mean a very unusual architecture
        raise AttributeError("model has no input embedding module")

    def hook(module, inputs, output):
        if isinstance(output, torch.Tensor) and not output.requires_grad:
            output.requires_grad_(True)

    return emb.register_forward_hook(hook)


def describe(model: nn.Module, name: str) -> ModelInfo:
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    n_heads = text_cfg.num_attention_heads
    n_kv_heads = getattr(text_cfg, "num_key_value_heads", None) or n_heads
    hidden = text_cfg.hidden_size
    # Prefer an explicit head_dim; several configs now carry one that is not
    # hidden_size // num_attention_heads.
    head_dim = getattr(text_cfg, "head_dim", None) or (hidden // n_heads)
    image_token_id = getattr(cfg, "image_token_index", None)
    if image_token_id is None:
        image_token_id = getattr(cfg, "image_token_id", None)
    if image_token_id is None:
        raise AttributeError(
            "config carries neither image_token_index nor image_token_id; the "
            "visual mask cannot be built without it"
        )
    return ModelInfo(
        name=name,
        n_layers=text_cfg.num_hidden_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        hidden_size=hidden,
        head_dim=head_dim,
        image_token_id=int(image_token_id),
        dtype=str(next(model.parameters()).dtype),
    )
