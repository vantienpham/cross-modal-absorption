"""Benchmark evaluation for compressed VLMs.

Greedy generation followed by task-specific matching, which is what the papers
this work compares against report. Deliberately covers two *kinds* of task:

* **Multiple choice** (ScienceQA-IMG) -- the benchmark prior low-rank VLM work
  is tuned on. The model emits one letter, so the metric is close to a
  four-way classification and is fairly forgiving of degraded visual features.
* **Open-ended, vision-grounded** (POPE, TextVQA) -- short free-form answers
  that cannot be produced from the language prior alone. POPE in particular is
  a hallucination probe, so it measures whether compression has quietly
  detached the answer from the image.

The gap between the two is itself a result: a method can look lossless on the
first while degrading on the second.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field

import torch

from .calib import Sample, make_batch

log = logging.getLogger(__name__)


@dataclass
class EvalResult:
    task: str
    n: int
    correct: int
    extra: dict = field(default_factory=dict)
    #: Per-item score, in dataset order: 1/0 for the binary tasks, the VQA soft
    #: score for TextVQA. Kept so that comparisons between two compressed models
    #: can use a *paired* test. The z-tests reported without it treat paired
    #: samples as independent, which is conservative but wastes power on the
    #: paper's central claim.
    per_item: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.n, 1)

    def to_dict(self) -> dict:
        return {"task": self.task, "n": self.n, "correct": self.correct,
                "accuracy": self.accuracy, **self.extra}


# --- answer normalisation ---------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = re.compile(r"[^\w\s]")


def normalise(text: str) -> str:
    """VQA-style normalisation: lowercase, strip punctuation and articles."""
    text = text.lower().strip()
    text = _PUNCT.sub(" ", text)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def extract_choice(text: str, n_choices: int) -> str | None:
    """Pull an option letter out of a generated multiple-choice answer."""
    letters = [chr(ord("A") + i) for i in range(n_choices)]
    t = text.strip()
    # Most common case: the reply begins with the letter, optionally punctuated.
    m = re.match(r"\s*\(?([A-Z])\)?[\.\):,]?\s*", t)
    if m and m.group(1) in letters:
        return m.group(1)
    for L in letters:  # noqa: E741 - option letters are genuinely single chars
        if re.search(rf"\b{L}\b", t):
            return L
    return None


def vqa_accuracy(pred: str, golds: list[str]) -> float:
    """Standard VQA metric: min(#humans agreeing / 3, 1)."""
    p = normalise(pred)
    counts = Counter(normalise(g) for g in golds)
    return min(counts.get(p, 0) / 3.0, 1.0)


# --- generation -------------------------------------------------------------

@torch.no_grad()
def generate(model, processor, sample: Sample, image_token_id: int,
             device, max_new_tokens: int = 16, prompt_style: str = "llava_v1") -> str:
    """Greedy-decode one answer."""
    batch = make_batch(sample, processor, image_token_id, device,
                       with_answer=False, prompt_style=prompt_style)
    out = model.generate(
        **batch.inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    )
    new = out[0, batch.inputs["input_ids"].shape[-1]:]
    return processor.tokenizer.decode(new, skip_special_tokens=True).strip()


@torch.no_grad()
def evaluate(model, processor, samples: list[Sample], task: str,
             image_token_id: int, device, max_new_tokens: int = 16,
             log_every: int = 250, prompt_style: str = "llava_v1") -> EvalResult:
    """Run one benchmark and score it with the task's own metric."""
    model.eval()
    n = correct = 0
    score = 0.0
    yes_pred = 0
    per_item: list[float] = []
    for i, s in enumerate(samples):
        text = generate(model, processor, s, image_token_id, device, max_new_tokens,
                        prompt_style=prompt_style)
        n += 1
        if task in ("scienceqa", "seedbench"):
            got = extract_choice(text, s.meta.get("n_choices", 4))
            hit = float(got == s.meta["gold"])
            correct += int(hit); per_item.append(hit)
        elif task == "pope":
            got = normalise(text)
            got = "yes" if got.startswith("yes") else ("no" if got.startswith("no") else got)
            hit = float(got == s.meta["gold"])
            correct += int(hit); per_item.append(hit)
            yes_pred += int(got == "yes")
        elif task == "textvqa":
            v = vqa_accuracy(text, s.meta["gold"])
            score += v; per_item.append(v)
            correct = int(round(score))
        else:
            raise ValueError(f"no metric defined for task {task!r}")
        if log_every and (i + 1) % log_every == 0:
            log.info("  %s %d/%d  running acc %.4f", task, i + 1, len(samples),
                     (score / n if task == "textvqa" else correct / n))

    extra: dict = {}
    if task == "textvqa":
        return EvalResult(task=task, n=n, correct=int(round(score)),
                          extra={"vqa_accuracy": score / max(n, 1)},
                          per_item=per_item)
    if task == "pope":
        # The yes-rate exposes the failure mode POPE is built to catch: a
        # degraded model drifting toward answering "yes" regardless of image.
        extra["yes_rate"] = yes_pred / max(n, 1)
    return EvalResult(task=task, n=n, correct=correct, extra=extra,
                      per_item=per_item)
