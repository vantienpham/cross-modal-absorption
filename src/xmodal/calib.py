"""Calibration and evaluation batches for VLM compression.

One batch is a single image-question-answer triple rendered into the model's
chat template, with the label mask covering the answer only. The answer boundary
is found by tokenising the *prompt alone with the same image* and taking its
length: the processor expands image placeholders deterministically, so the
prompt tokenisation is a genuine prefix of the full tokenisation and the
boundary is exact rather than estimated.

Batch size is 1 by design. VLM sequences are long and highly variable, and
padding several together would put ``pad`` positions into the very covariances
this project is measuring. Throughput is not the constraint here -- a few
hundred calibration samples is the whole budget.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch

from .modality import build_modality_mask, ModalityMask

IGNORE_INDEX = -100

# LLaVA-1.5 / v1 conversation template. Written out rather than taken from
# processor.apply_chat_template because that helper's output has changed across
# transformers releases, and a template change silently moves the answer
# boundary.
LLAVA_V1_TEMPLATE = "USER: <image>\n{question} ASSISTANT:"

#: Prompt styles. ``llava_v1`` is the literal template above and is the default
#: so that every LLaVA-1.5/Next number in this project stays reproducible
#: byte-for-byte. ``chat`` defers to the processor's own chat template, which is
#: required for families whose image placeholder and turn markers differ
#: entirely (Qwen2-VL, Idefics2/3) -- a LLaVA prompt on those models inserts no
#: image token at all, which the mask consistency check then catches.
PROMPT_STYLES = ("llava_v1", "chat")


def build_prompt(processor, question: str, style: str = "llava_v1") -> str:
    """Render one question into the model's prompt format."""
    if style == "llava_v1":
        return LLAVA_V1_TEMPLATE.format(question=question)
    if style != "chat":
        raise ValueError(f"unknown prompt style {style!r}; expected {PROMPT_STYLES}")

    apply = getattr(processor, "apply_chat_template", None)
    has_template = (getattr(processor, "chat_template", None)
                    or getattr(getattr(processor, "tokenizer", None), "chat_template", None))
    if apply is None or not has_template:
        raise ValueError(
            "prompt style 'chat' requested but the processor has no chat "
            "template; pass --prompt-style llava_v1 or use a processor that "
            "declares one"
        )
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": question}]}]
    return apply(messages, add_generation_prompt=True, tokenize=False)


@dataclass
class Sample:
    """One rendered example, before tokenisation."""

    image: Any
    question: str
    answer: str
    meta: dict[str, Any]


@dataclass
class Batch:
    inputs: dict[str, torch.Tensor]
    labels: torch.Tensor
    mask: ModalityMask
    sample: Sample


# --- dataset adapters -------------------------------------------------------
# Each adapter turns one raw HF row into a Sample. Registered by name so a run
# records which adapter produced its calibration set.

Adapter = Callable[[dict[str, Any]], Sample | None]
ADAPTERS: dict[str, Adapter] = {}


def register(name: str) -> Callable[[Adapter], Adapter]:
    def deco(fn: Adapter) -> Adapter:
        ADAPTERS[name] = fn
        return fn

    return deco


def _letter(i: int) -> str:
    return chr(ord("A") + i)


@register("scienceqa")
def _scienceqa(row: dict[str, Any]) -> Sample | None:
    """ScienceQA-IMG: multiple choice with an image, hint and lettered options."""
    image = row.get("image")
    if image is None:
        return None
    choices = row.get("choices") or []
    hint = (row.get("hint") or "").strip()
    q = (row.get("question") or "").strip()
    opts = "\n".join(f"{_letter(i)}. {c}" for i, c in enumerate(choices))
    parts = [p for p in (hint, q, opts) if p]
    question = "\n".join(parts) + "\nAnswer with the option's letter from the given choices directly."
    ans_idx = row.get("answer")
    if ans_idx is None or not isinstance(ans_idx, int) or ans_idx >= len(choices):
        return None
    return Sample(
        image=image,
        question=question,
        answer=_letter(ans_idx),
        meta={"task": "scienceqa", "n_choices": len(choices), "gold": _letter(ans_idx)},
    )


@register("seedbench")
def _seedbench(row: dict[str, Any]) -> Sample | None:
    """SEED-Bench-IMG: four-way multiple choice with a lettered answer.

    Included because it is the *other* benchmark prior low-rank VLM work
    reports. If compression damage is invisible here too, the claim is about
    the literature's protocol rather than about ScienceQA specifically.
    """
    image = row.get("image")
    if image is None:
        return None
    if row.get("data_type") not in (None, "image"):
        return None  # the video split cannot be scored here
    letters = ["A", "B", "C", "D"]
    choices = [row.get(f"choice_{c}") for c in ("a", "b", "c", "d")]
    if any(c is None for c in choices):
        return None
    gold = (row.get("answer") or "").strip().upper()
    if gold not in letters:
        return None
    q = (row.get("question") or "").strip()
    opts = "\n".join(f"{L}. {c}" for L, c in zip(letters, choices))
    return Sample(
        image=image if not isinstance(image, list) else image[0],
        question=f"{q}\n{opts}\nAnswer with the option's letter from the given "
                 "choices directly.",
        answer=gold,
        meta={"task": "seedbench", "n_choices": 4, "gold": gold},
    )


@register("pope")
def _pope(row: dict[str, Any]) -> Sample | None:
    """POPE: binary object-hallucination probes; the answer is yes/no."""
    image = row.get("image")
    if image is None:
        return None
    q = (row.get("question") or "").strip()
    ans = (row.get("answer") or "").strip()
    if not q or not ans:
        return None
    return Sample(
        image=image,
        question=q + "\nAnswer the question using a single word or phrase.",
        answer=ans.capitalize(),
        meta={
            "task": "pope",
            "gold": ans.lower(),
            "category": row.get("category") or row.get("pope_category"),
        },
    )


@register("textvqa")
def _textvqa(row: dict[str, Any]) -> Sample | None:
    """TextVQA: reading text in the image; answers are short free-form strings."""
    image = row.get("image")
    if image is None:
        return None
    q = (row.get("question") or "").strip()
    answers = row.get("answers") or []
    if isinstance(answers, str):
        answers = [answers]
    if not q or not answers:
        return None
    return Sample(
        image=image,
        question=q + "\nAnswer the question using a single word or phrase.",
        answer=str(answers[0]),
        meta={"task": "textvqa", "gold": [str(a) for a in answers]},
    )


def load_mixture(specs: list[tuple[str, str | None, str, str]],
                 limit: int, seed: int = 0) -> list[Sample]:
    """Draw an equal number of samples from each ``(dataset, config, split, adapter)``.

    Used for the mixed-calibration condition. The curvature statistics depend on
    what the answers look like -- calibrating on single-letter answers puts
    almost all gradient mass on one token -- so a claim about the compression
    method has to be shown not to be a claim about the calibration set.
    """
    per = max(1, limit // len(specs))
    out: list[Sample] = []
    for ds, cfg, split, adapter in specs:
        out += load_samples(dataset=ds, adapter=adapter, split=split,
                            limit=per, config=cfg, seed=seed)
    random.Random(seed).shuffle(out)
    return out[:limit]


def load_samples(
    dataset: str,
    adapter: str,
    split: str,
    limit: int | None = None,
    config: str | None = None,
    seed: int = 0,
    shuffle: bool = True,
) -> list[Sample]:
    """Load and render up to ``limit`` usable samples from an HF dataset."""
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split) if config else load_dataset(dataset, split=split)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    fn = ADAPTERS[adapter]
    out: list[Sample] = []
    skipped = 0
    for row in ds:
        s = fn(row)
        if s is None:
            skipped += 1
            continue
        out.append(s)
        if limit is not None and len(out) >= limit:
            break
    if not out:
        raise RuntimeError(
            f"adapter {adapter!r} produced no usable samples from {dataset}:{split} "
            f"({skipped} rows skipped) -- the schema has probably changed"
        )
    return out


def make_batch(
    sample: Sample,
    processor,
    image_token_id: int,
    device: torch.device | str,
    with_answer: bool = True,
    prompt_style: str = "llava_v1",
) -> Batch:
    """Tokenise one sample and build its label and modality masks."""
    prompt = build_prompt(processor, sample.question, prompt_style)
    image = sample.image.convert("RGB") if hasattr(sample.image, "convert") else sample.image

    prompt_enc = processor(images=image, text=prompt, return_tensors="pt")
    n_prompt = int(prompt_enc["input_ids"].shape[-1])

    if with_answer:
        full_text = f"{prompt} {sample.answer}"
        enc = processor(images=image, text=full_text, return_tensors="pt")
    else:
        enc = prompt_enc

    input_ids = enc["input_ids"]
    labels = input_ids.clone()
    labels[:, :n_prompt] = IGNORE_INDEX
    if not with_answer:
        # Nothing to score; every position is prompt.
        labels[:] = IGNORE_INDEX

    attn = enc.get("attention_mask")
    if attn is None:
        attn = torch.ones_like(input_ids)

    mask = build_modality_mask(
        input_ids=input_ids,
        attention_mask=attn,
        labels=labels,
        image_token_id=image_token_id,
        ignore_index=IGNORE_INDEX,
    )
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in enc.items()}
    return Batch(inputs=inputs, labels=labels.to(device), mask=mask, sample=sample)


def iter_batches(
    samples: list[Sample],
    processor,
    image_token_id: int,
    device: torch.device | str,
    with_answer: bool = True,
    prompt_style: str = "llava_v1",
) -> Iterator[Batch]:
    for s in samples:
        yield make_batch(s, processor, image_token_id, device,
                         with_answer=with_answer, prompt_style=prompt_style)


# --- image ablations -------------------------------------------------------
#
# The paper argues multiple-choice accuracy survives compression because it is
# substantially carried by the language prior. That is a claim about how much
# each benchmark depends on the image at all, and it is testable directly:
# degrade the image and see which benchmarks notice.
#
#   blank    a uniform mid-grey image
#   noise    uniform random pixels
#   shuffle  a *different* sample's image, so image statistics are unchanged
#            and only the question-image correspondence is destroyed
#
# ``shuffle`` is the sharpest of the three: blank and noise also change the
# activation distribution the model sees, whereas shuffle leaves it in
# distribution and removes only the information.

IMAGE_ABLATIONS = ("none", "blank", "noise", "shuffle")


def _grey_like(img):
    from PIL import Image
    return Image.new("RGB", img.size if hasattr(img, "size") else (336, 336),
                     (127, 127, 127))


def _noise_like(img):
    import numpy as np
    from PIL import Image
    w, h = img.size if hasattr(img, "size") else (336, 336)
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype="uint8"), "RGB")


def ablate_images(samples: list[Sample], mode: str, seed: int = 0) -> list[Sample]:
    """Return samples with their images degraded, leaving questions untouched."""
    if mode not in IMAGE_ABLATIONS:
        raise ValueError(f"unknown image ablation {mode!r}; expected {IMAGE_ABLATIONS}")
    if mode == "none":
        return samples
    if mode == "shuffle":
        n = len(samples)
        if n < 2:
            raise ValueError("shuffle ablation needs at least two samples")
        rng = random.Random(seed)
        perm = list(range(n))
        rng.shuffle(perm)
        # A derangement: no sample may keep its own image, or the ablation is
        # partially a no-op and the control is weakened without saying so.
        for i in range(n):
            if perm[i] == i:
                j = (i + 1) % n
                perm[i], perm[j] = perm[j], perm[i]
        assert all(perm[i] != i for i in range(n)), "shuffle left a fixed point"
        return [Sample(image=samples[perm[i]].image, question=s.question,
                       answer=s.answer, meta=s.meta) for i, s in enumerate(samples)]
    fn = _grey_like if mode == "blank" else _noise_like
    return [Sample(image=fn(s.image), question=s.question, answer=s.answer,
                   meta=s.meta) for s in samples]
