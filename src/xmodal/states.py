"""Per-layer hidden states, split into visual and text rows.

One forward pass per example with ``output_hidden_states=True``, then the
modality mask from :mod:`xmodal.modality` selects rows. Nothing here is
approximate: since transformers 4.47 the processor expands the image
placeholder to one ``input_ids`` entry per visual token, so position ``i`` of
the decoder's hidden states carries the same role as position ``i`` of
``input_ids`` and the split is an equality test rather than an estimate of where
the image "must" have landed.

The text rows used throughout are the **post-image** ones. Under causal
attention only those positions can have attended to the image, so they are the
only ones that could have absorbed anything from it; including the pre-image
system prompt would add rows that are constant across examples and would dilute
the very pairing the controls test.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .calib import IGNORE_INDEX, Batch
from .modality import ROLE_VISUAL


@dataclass
class LayerStates:
    """Visual and text rows at every layer, for one example.

    ``visual[l]`` is ``(n_visual, d)`` and ``text[l]`` is ``(n_text, d)``, both
    float32 on the CPU. Layer 0 is the embedding output, so ``len(visual)`` is
    one more than the number of decoder blocks and index ``l`` means "after
    ``l`` blocks".
    """

    visual: list[torch.Tensor]
    text: list[torch.Tensor]

    def __len__(self) -> int:
        return len(self.visual)


@torch.no_grad()
def capture(model, batch: Batch, post_image_text_only: bool = True) -> LayerStates:
    """Run one example forward and return its per-layer modality split.

    Args:
        model: a loaded VLM.
        batch: one tokenised example from :func:`xmodal.calib.make_batch`.
        post_image_text_only: keep only text positions after the last visual
            one. Leave this on; see the module docstring.

    Hidden states are moved to the CPU in float32 as they are collected. Keeping
    33 layers of a 2880-token LLaVA-Next sequence on the device in addition to
    the model is the difference between fitting on a 40 GB card and not, and
    every consumer of these promotes to float64 anyway.
    """
    out = model(**batch.inputs, output_hidden_states=True, use_cache=False)
    roles = batch.mask.roles[0]
    vis = roles == ROLE_VISUAL
    txt = batch.mask.valid[0] & ~vis
    if post_image_text_only and bool(vis.any()):
        last = int(vis.nonzero()[-1].item())
        after = torch.zeros_like(txt)
        after[last + 1:] = True
        txt = txt & after
    if not bool(vis.any()):
        raise ValueError(
            "no visual positions in this batch; the processor did not expand "
            "the image placeholder, which usually means the prompt style does "
            "not match the checkpoint family"
        )
    if not bool(txt.any()):
        raise ValueError("no post-image text positions; the prompt template has no suffix")

    visual, text = [], []
    for h in out.hidden_states:
        h = h[0].to("cpu", torch.float32)
        visual.append(h[vis])
        text.append(h[txt])
    del out
    return LayerStates(visual=visual, text=text)


def make_eval_batch(sample, processor, image_token_id: int, device,
                    prompt_style: str = "llava_v1") -> Batch:
    """Tokenise one sample as it appears at *inference* time: prompt, no answer.

    The absorption statistic is a property of the prefill, which is what a
    pruning method sees, so the answer must not be in the sequence. Building the
    batch with the answer appended would put the gold string inside the text
    rows and let the text subspace explain visual tokens using information the
    model does not have when it prunes.
    """
    from .calib import make_batch

    b = make_batch(sample, processor, image_token_id, device, with_answer=False,
                   prompt_style=prompt_style)
    if int((b.labels != IGNORE_INDEX).sum()) != 0:
        raise ValueError("prompt-only batch still carries answer labels")
    return b
