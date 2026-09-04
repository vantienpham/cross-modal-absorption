"""Selecting visual tokens for one example and generating from the pruned model.

Two forward passes per example. The first runs the prompt to the pruning layer
to obtain the representations selection needs; the second generates with the
chosen tokens kept. A deployed implementation would fuse them, since the first
pass only needs the layers up to the pruning point and its result is thrown away
afterwards. **No wall-clock claim in this project comes from this path**: it is
built for measurement, where being able to run every criterion against identical
inputs matters more than the prefill it repeats. Cost claims are made
analytically and from the separate timing harness.

The comparison this supports is a paired one. Every criterion sees the same
examples, the same layer, the same budget and the same residual matrix, so the
difference between two of them is a difference of ranking and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .absorption import residual_ratio, tikhonov_reconstruct
from .calib import Sample, make_batch
from .modality import ROLE_VISUAL
from .prune import PrunableModel
from .select import select


@dataclass
class SelectionInputs:
    """Everything a criterion needs at the pruning layer, for one example."""

    residual: torch.Tensor        # (n_visual, d), text-explained part removed
    ratio: torch.Tensor           # (n_visual,), relative residual norm
    attention: torch.Tensor       # (n_visual,), mass from post-image text
    visual_pos: torch.Tensor      # (n_visual,), positions in the sequence
    text_pos: torch.Tensor        # (n_text,), post-image text positions
    seq_len: int


@torch.no_grad()
def selection_inputs(pm: PrunableModel, batch, eta: float = 0.75) -> SelectionInputs:
    """Run the prompt to the pruning layer and assemble the selection inputs."""
    roles = batch.mask.roles[0]
    valid = batch.mask.valid[0]
    vis = roles == ROLE_VISUAL
    if not bool(vis.any()):
        raise ValueError("no visual positions; prompt style does not match the checkpoint")
    last = int(vis.nonzero()[-1].item())
    txt = valid & ~vis
    txt[: last + 1] = False
    if not bool(txt.any()):
        raise ValueError("no post-image text positions; the prompt template has no suffix")

    visual_pos = vis.nonzero().squeeze(-1)
    text_pos = txt.nonzero().squeeze(-1)

    pm.disable()
    pm.plan.enabled = True
    pm.probe.arm(text_pos, visual_pos)
    pm.model(**batch.inputs, use_cache=False)
    attn = pm.probe.result
    pm.probe.remove()
    if pm.plan.hidden is None:
        raise RuntimeError("the pruning layer recorded no hidden states")
    if attn is None:
        raise RuntimeError("the attention probe did not fire; check the layer index")

    h = pm.plan.hidden[0].to("cpu", torch.float32)
    V, T = h[visual_pos], h[text_pos]
    # Centred on the text mean, the convention the criteria being compared
    # against were defined under. xmodal.controls documents what that choice
    # does and measures the alternative.
    mu = T.mean(dim=0, keepdim=True)
    Vc, Tc = V - mu, T - mu
    V_hat, _, _ = tikhonov_reconstruct(Vc, Tc, eta=eta)
    return SelectionInputs(
        residual=(Vc.double() - V_hat),
        ratio=residual_ratio(Vc, V_hat),
        attention=attn,
        visual_pos=visual_pos,
        text_pos=text_pos,
        seq_len=int(valid.sum()),
    )


def positions_for(chosen: torch.Tensor, visual_pos: torch.Tensor,
                  seq_len: int) -> torch.Tensor:
    """Sequence positions to retain: the chosen visual tokens plus all text.

    Text positions are never dropped. They are few, they carry the question, and
    every method being compared keeps them, so dropping any would make the
    comparison about something other than visual token selection.
    """
    keep_visual = visual_pos[chosen.to(torch.long)]
    is_visual = torch.zeros(seq_len, dtype=torch.bool)
    is_visual[visual_pos] = True
    non_visual = (~is_visual).nonzero().squeeze(-1)
    return torch.cat([keep_visual, non_visual]).sort().values


def keep_positions(inp: SelectionInputs, criterion: str, budget: int,
                   seq_len: int, seed: int = 0) -> torch.Tensor:
    """Apply one criterion and return the sequence positions it retains."""
    chosen = select(criterion, R=inp.residual, attn=inp.attention,
                    residual_ratio=inp.ratio, budget=budget, seed=seed)
    return positions_for(chosen, inp.visual_pos, seq_len)


@torch.no_grad()
def generate_pruned(pm: PrunableModel, processor, sample: Sample,
                    image_token_id: int, device, criterion: str, budget: int,
                    max_new_tokens: int = 16, eta: float = 0.75, seed: int = 0,
                    prompt_style: str = "llava_v1") -> tuple[str, dict[str, Any]]:
    """Generate one answer with visual tokens pruned by ``criterion``.

    ``criterion="none"`` skips selection entirely and generates from the
    unpruned model, which is the reference every retention rate is measured
    against and which must be produced by the same code path as the rest.
    """
    batch = make_batch(sample, processor, image_token_id, device,
                       with_answer=False, prompt_style=prompt_style)
    prompt_len = int(batch.inputs["input_ids"].shape[-1])
    info: dict[str, Any] = {"prompt_len": prompt_len}

    if criterion == "none":
        pm.disable()
    else:
        inp = selection_inputs(pm, batch, eta=eta)
        keep = keep_positions(inp, criterion, budget, prompt_len, seed=seed)
        info.update(n_visual=int(inp.visual_pos.numel()), n_kept=int(budget),
                    kept_len=int(keep.numel()))
        pm.enable(keep, prompt_len)

    out = pm.model.generate(
        **batch.inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    )
    pm.disable()
    new = out[0, prompt_len:]
    return processor.tokenizer.decode(new, skip_special_tokens=True).strip(), info
