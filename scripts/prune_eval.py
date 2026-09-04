#!/usr/bin/env python3
"""Benchmark accuracy under visual-token pruning, one criterion per run.

    uv run --no-sync python scripts/prune_eval.py \\
      --model llava-hf/llava-1.5-7b-hf --dataset lmms-lab/POPE --adapter pope \\
      --split test --limit 1000 --criterion leverage --budget 64 \\
      --run-dir out/runs/prune-llava15-pope-leverage-64

Every criterion is scored on the *same* samples in the same order at the same
budget, and per-item scores are kept, so two runs can be compared with a paired
test rather than by the gap between two summary numbers. ``--criterion none``
produces the unpruned reference that retention rates are quoted against.

Two things are measured beyond accuracy. ``recon_error`` is the energy of the
text residual that the kept tokens cannot span, which is the quantity the
perturbation argument bounds pruning damage by and which needs no generation to
evaluate. ``overlap`` records how much the chosen set shares with the
attention-only and residual-only sets, which is what makes a claim about two
signals being complementary checkable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xmodal.calib import make_batch, load_samples        # noqa: E402
from xmodal.evaluate import EvalResult, normalise, extract_choice, vqa_accuracy  # noqa: E402
from xmodal.models import describe, load_vlm             # noqa: E402
from xmodal.pipeline import positions_for, selection_inputs  # noqa: E402
from xmodal.prune import PrunableModel                   # noqa: E402
from xmodal.runlog import RunDir                         # noqa: E402
from xmodal.select import CRITERIA, reconstruction_error, select  # noqa: E402

log = logging.getLogger("prune")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--dataset", default="lmms-lab/POPE")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--adapter", default="pope")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--criterion", default="leverage",
                   choices=(*CRITERIA, "none"))
    p.add_argument("--budget", type=int, default=64,
                   help="visual tokens kept; ignored when --criterion none")
    p.add_argument("--layer", type=int, default=15, help="pruning layer")
    p.add_argument("--eta", type=float, default=0.75)
    p.add_argument("--heads", type=int, default=12,
                   help="image-attending heads aggregated for the attention signal")
    p.add_argument("--rank", type=int, default=512,
                   help="sketch rank for leverage; 0 forces the exact decomposition")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--prompt-style", default="llava_v1", choices=("llava_v1", "chat"))
    p.add_argument("--run-dir", default="out/runs/prune")
    return p.parse_args()


def score_one(task: str, text: str, meta: dict) -> float:
    if task in ("scienceqa", "seedbench"):
        return float(extract_choice(text, meta.get("n_choices", 4)) == meta["gold"])
    if task == "pope":
        got = normalise(text)
        got = "yes" if got.startswith("yes") else ("no" if got.startswith("no") else got)
        return float(got == meta["gold"])
    if task == "textvqa":
        return vqa_accuracy(text, meta["gold"])
    raise ValueError(f"no metric defined for task {task!r}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    run = RunDir(args.run_dir)
    run.write_config(vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_vlm(args.model, dtype=torch.float16,
                                device_map={"": 0} if device == "cuda" else None,
                                attn_implementation="sdpa")
    if device == "cpu":
        model = model.to(device)
    info = describe(model, args.model)
    run.write_env({"model": info.to_dict()})

    samples = load_samples(dataset=args.dataset, adapter=args.adapter,
                           split=args.split, limit=args.limit,
                           config=args.config, seed=args.seed)
    task = samples[0].meta["task"]
    log.info("%s: %d samples, criterion=%s budget=%s layer=%d",
             task, len(samples), args.criterion, args.budget, args.layer)

    rank = None if args.rank == 0 else args.rank
    per_item: list[float] = []
    predictions: list[str] = []
    recon: list[float] = []
    overlap_attn: list[float] = []
    overlap_res: list[float] = []
    n_visual_seen: list[int] = []
    yes_pred = 0
    skipped = 0

    with PrunableModel(model, layer=args.layer, n_heads_keep=args.heads) as pm:
        for i, s in enumerate(samples):
            try:
                text, extra = _run_one(pm, processor, s, info.image_token_id, device,
                                       args, rank)
            except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                log.warning("sample %d skipped: %s", i, exc)
                torch.cuda.empty_cache()
                skipped += 1
                continue
            per_item.append(score_one(task, text, s.meta))
            predictions.append(text)
            if task == "pope":
                yes_pred += int(normalise(text).startswith("yes"))
            for dst, key in ((recon, "recon"), (overlap_attn, "ov_attn"),
                             (overlap_res, "ov_res")):
                if key in extra:
                    dst.append(extra[key])
            if "n_visual" in extra:
                n_visual_seen.append(extra["n_visual"])
            if (i + 1) % 200 == 0:
                log.info("  %d/%d  running %.4f", i + 1, len(samples),
                         sum(per_item) / len(per_item))

    n = len(per_item)
    if n == 0:
        log.error("every sample was skipped")
        return 1
    acc = sum(per_item) / n
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731

    result = EvalResult(task=task, n=n, correct=int(round(sum(per_item))),
                        per_item=per_item)
    metrics = {
        "task": task, "n": n, "skipped": skipped, "accuracy": acc,
        "criterion": args.criterion, "budget": args.budget, "layer": args.layer,
        "n_visual": mean(n_visual_seen),
        "retention": (args.budget / mean(n_visual_seen)) if n_visual_seen else None,
        "recon_error": mean(recon),
        "overlap_attention": mean(overlap_attn),
        "overlap_residual": mean(overlap_res),
    }
    if task == "pope":
        metrics["yes_rate"] = yes_pred / n
    if task == "textvqa":
        metrics["vqa_accuracy"] = acc
    run.log(**metrics)
    # Per-item scores are the input to the paired tests; they stay beside the
    # run rather than in the summary, which sync.sh pulls back on its own.
    run.artifact("per_item.json").write_text(json.dumps(result.per_item))
    # The generated strings, not just whether they scored. Accuracy is a blunt
    # probe of how much pruning perturbed the model: two different answers can
    # both be wrong, so a score that does not flip is not evidence that nothing
    # changed. Agreement with the unpruned run's strings is the sharper measure,
    # and it is the one the perturbation bound in the paper is about.
    run.artifact("predictions.json").write_text(json.dumps(predictions))
    run.write_metrics()
    log.info("%s", json.dumps(metrics, indent=2))
    return 0


@torch.no_grad()
def _run_one(pm, processor, sample, image_token_id, device, args, rank):
    batch = make_batch(sample, processor, image_token_id, device,
                       with_answer=False, prompt_style=args.prompt_style)
    prompt_len = int(batch.inputs["input_ids"].shape[-1])
    extra: dict = {}

    if args.criterion == "none":
        pm.disable()
    else:
        inp = selection_inputs(pm, batch, eta=args.eta)
        chosen = select(args.criterion, R=inp.residual, attn=inp.attention,
                        residual_ratio=inp.ratio, budget=args.budget,
                        seed=args.seed, rank=rank)
        extra["n_visual"] = int(inp.visual_pos.numel())
        extra["recon"] = reconstruction_error(inp.residual, chosen)
        # What the two single-signal criteria would have chosen, for the
        # complementarity claim. Cheap: both are top-k over vectors we hold.
        ref_a = select("attention", R=inp.residual, attn=inp.attention,
                       residual_ratio=inp.ratio, budget=args.budget)
        ref_r = select("residual", R=inp.residual, attn=inp.attention,
                       residual_ratio=inp.ratio, budget=args.budget)
        k = max(chosen.numel(), 1)
        sel = set(chosen.tolist())
        extra["ov_attn"] = len(sel & set(ref_a.tolist())) / k
        extra["ov_res"] = len(sel & set(ref_r.tolist())) / k

        pm.enable(positions_for(chosen, inp.visual_pos, prompt_len), prompt_len)

    out = pm.model.generate(
        **batch.inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
        num_beams=1,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    )
    pm.disable()
    new = out[0, prompt_len:]
    return processor.tokenizer.decode(new, skip_special_tokens=True).strip(), extra


if __name__ == "__main__":
    raise SystemExit(main())
