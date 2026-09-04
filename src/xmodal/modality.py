"""Per-token modality labelling for vision-language sequences.

The entire diagnosis in this project rests on being able to say, for every
position in the decoder's hidden-state sequence, whether that position carries a
*visual* token, a *prompt text* token, or an *answer text* token. Everything
downstream -- the split covariances, the eta diagnostic, the mixture-of-Kronecker
curvature -- is an expectation conditioned on that label, so a mask that is off
by even a few positions quietly corrupts every number in the paper.

Why the ``llava-hf`` checkpoints rather than the original LLaVA fork: since
transformers 4.47 the LLaVA processor *expands* the single ``<image>`` placeholder
into ``num_image_tokens`` copies of ``config.image_token_index`` **inside
input_ids**, and ``forward`` then scatters the vision-tower features into exactly
those positions. So ``input_ids`` and the decoder hidden states are aligned 1:1
and the visual mask is an exact equality test rather than an arithmetic
reconstruction of where the image "must" have landed. The original fork splices
embeddings around a single ``IMAGE_TOKEN_INDEX`` sentinel, where the mask has to
be recomputed by hand and drifts whenever the prompt template changes.

:func:`assert_mask_consistent` exists because that alignment is an assumption
about someone else's library, and a silent change to it would not produce an
error -- it would produce plausible, wrong numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Role codes. Kept as small ints so masks are cheap to carry around and index.
ROLE_PAD = 0
ROLE_VISUAL = 1
ROLE_TEXT_PROMPT = 2
ROLE_TEXT_ANSWER = 3

ROLE_NAMES: dict[int, str] = {
    ROLE_PAD: "pad",
    ROLE_VISUAL: "visual",
    ROLE_TEXT_PROMPT: "text_prompt",
    ROLE_TEXT_ANSWER: "text_answer",
}

#: Covariance accumulators are kept for these two coarse classes only. Three-way
#: splits of a 4096x4096 float64 Gram matrix across 32 layers do not fit
#: alongside a 7B model, and the prompt/answer distinction matters for the
#: *gradient mass* argument (a scalar) rather than for the whitening geometry.
COV_CLASSES: tuple[str, ...] = ("visual", "text")

#: Scalar energy accumulators are cheap, so they keep the finer split.
SCALAR_CLASSES: tuple[str, ...] = ("visual", "text_prompt", "text_answer")


@dataclass(frozen=True)
class ModalityMask:
    """Per-position role labels for one batch.

    Attributes:
        roles: ``(B, L)`` int8 tensor of ``ROLE_*`` codes.
        n_visual / n_text_prompt / n_text_answer: token counts, padding excluded.
    """

    roles: torch.Tensor

    @property
    def visual(self) -> torch.Tensor:
        return self.roles == ROLE_VISUAL

    @property
    def text(self) -> torch.Tensor:
        """Prompt and answer text together -- the complement of visual."""
        return (self.roles == ROLE_TEXT_PROMPT) | (self.roles == ROLE_TEXT_ANSWER)

    @property
    def text_prompt(self) -> torch.Tensor:
        return self.roles == ROLE_TEXT_PROMPT

    @property
    def text_answer(self) -> torch.Tensor:
        return self.roles == ROLE_TEXT_ANSWER

    @property
    def valid(self) -> torch.Tensor:
        """Every non-padding position."""
        return self.roles != ROLE_PAD

    def by_name(self, name: str) -> torch.Tensor:
        try:
            return getattr(self, name)
        except AttributeError as exc:  # pragma: no cover - programmer error
            raise KeyError(f"unknown modality class {name!r}") from exc

    def counts(self) -> dict[str, int]:
        return {
            "visual": int(self.visual.sum()),
            "text_prompt": int(self.text_prompt.sum()),
            "text_answer": int(self.text_answer.sum()),
            "text": int(self.text.sum()),
            "valid": int(self.valid.sum()),
        }


def build_modality_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    image_token_id: int,
    ignore_index: int = -100,
) -> ModalityMask:
    """Label every position as visual / prompt text / answer text / padding.

    Args:
        input_ids: ``(B, L)``, with image placeholders already expanded by the
            processor to one id per visual token.
        attention_mask: ``(B, L)``, 1 for real tokens.
        labels: ``(B, L)``, ``ignore_index`` everywhere the loss is not taken.
            Positions where this differs from ``ignore_index`` are the answer.
        image_token_id: ``config.image_token_index`` for the checkpoint.
        ignore_index: the label value that means "no loss here".

    Returns:
        A :class:`ModalityMask` over the same ``(B, L)`` grid.
    """
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            f"input_ids {tuple(input_ids.shape)} and attention_mask "
            f"{tuple(attention_mask.shape)} must have the same shape"
        )
    if input_ids.shape != labels.shape:
        raise ValueError(
            f"input_ids {tuple(input_ids.shape)} and labels "
            f"{tuple(labels.shape)} must have the same shape"
        )

    roles = torch.full_like(input_ids, ROLE_PAD, dtype=torch.int8)
    real = attention_mask.bool()

    is_image = (input_ids == image_token_id) & real
    is_answer = (labels != ignore_index) & real
    # An image position that is also a label position would make the split
    # ambiguous. It should never happen (the loss is taken on generated text),
    # but resolving it silently in favour of "visual" would hide a broken
    # collator, so we surface it instead.
    overlap = int((is_image & is_answer).sum())
    if overlap:
        raise ValueError(
            f"{overlap} position(s) are labelled both visual and answer; the "
            "label mask is probably not aligned with the expanded input_ids"
        )

    roles[real] = ROLE_TEXT_PROMPT
    roles[is_answer] = ROLE_TEXT_ANSWER
    roles[is_image] = ROLE_VISUAL
    return ModalityMask(roles=roles)


def assert_mask_consistent(
    mask: ModalityMask,
    hidden_len: int,
    expected_visual_per_image: int | None = None,
    n_images: int = 1,
) -> None:
    """Fail loudly if the mask does not line up with the decoder's sequence.

    This guards the one assumption the whole study inherits from transformers:
    that image tokens are expanded in ``input_ids`` so that position ``i`` of the
    mask is position ``i`` of the decoder hidden state. If a library upgrade
    changed that, no exception would be raised anywhere -- the covariances would
    simply be accumulated against the wrong tokens.
    """
    if mask.roles.shape[-1] != hidden_len:
        raise AssertionError(
            f"modality mask length {mask.roles.shape[-1]} != decoder hidden "
            f"length {hidden_len}. Image tokens are probably not being expanded "
            "in input_ids; check the transformers version and the processor."
        )
    n_visual = int(mask.visual.sum())
    if n_visual == 0:
        raise AssertionError(
            "no visual tokens found. Either the batch carries no image or "
            "image_token_id does not match config.image_token_index."
        )
    if expected_visual_per_image is not None:
        expected = expected_visual_per_image * n_images
        if n_visual != expected:
            raise AssertionError(
                f"found {n_visual} visual tokens, expected {expected} "
                f"({expected_visual_per_image} x {n_images} image(s))"
            )
