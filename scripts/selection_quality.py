#!/usr/bin/env python3
"""Reconstruction error of each criterion's chosen subset, without generating.

    uv run --no-sync python scripts/selection_quality.py \\
      --model llava-hf/llava-1.5-7b-hf --dataset lmms-lab/POPE --adapter pope \\
      --split test --limit 200 --run-dir out/runs/selq-pope

Proposition 2 bounds the perturbation that dropping a visual token set induces by
the *attention-weighted* reconstruction error of the residual matrix from the
retained rows. That is the quantity the leverage criterion minimises and the one
a comparison of selection rules should be made on, and it needs no decoding: one
forward pass per example produces the residual matrix, after which every
criterion and every budget can be scored from the same inputs.

Both weighted and unweighted errors are reported. The unweighted number asks how
well the kept tokens span the visual residual as such; the weighted one asks how
well they span the part of it the text queries actually look at. A criterion can
win on one and lose on the other, and which of the two tracks downstream accuracy
is itself a result.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xmodal.calib import load_samples, make_batch          # noqa: E402
from xmodal.models import describe, load_vlm               # noqa: E402
from xmodal.pipeline import selection_inputs               # noqa: E402
from xmodal.prune import PrunableModel                     # noqa: E402
from xmodal.runlog import RunDir                           # noqa: E402
from xmodal.select import CRITERIA, select                 # noqa: E402

log = logging.getLogger("selq")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--dataset", default="lmms-lab/POPE")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--adapter", default="pope")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--eta", type=float, default=0.75)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--rank", type=int, default=512)
    p.add_argument("--budgets", type=int, nargs="+", default=[192, 128, 64])
    p.add_argument("--prompt-style", default="llava_v1", choices=("llava_v1", "chat"))
    p.add_argument("--run-dir", default="out/runs/selq")
    return p.parse_args()


def pct_rank(values: torch.Tensor, keep: torch.Tensor) -> float:
    """Mean percentile rank, within the example, of the selected rows.

    A criterion that scores tokens by residual *ratio* divides by the row norm,
    so it is scale-free and can prefer atypical low-energy rows over the ones
    the model reads. That is a property of the score's definition, and this
    turns it into something measured: 0.5 means the selection is indifferent to
    the quantity, values near 0 mean it systematically prefers the smallest.
    """
    v = values.detach().to("cpu", torch.float64)
    n = v.numel()
    if n < 2:
        return 0.5
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[torch.argsort(v)] = torch.arange(n, dtype=torch.float64)
    return float(ranks[keep.to(torch.long)].mean() / (n - 1))


def errors(R: torch.Tensor, w: torch.Tensor, keep: torch.Tensor) -> tuple[float, float]:
    """Unweighted and attention-weighted relative reconstruction error.

    Both measure the energy of ``R`` outside the span of the kept rows; they
    differ only in how rows are weighted in the norm. The span is the same in
    both cases, since scaling a row does not change the subspace it spans.
    """
    Rd = R.detach().to("cpu", torch.float64)
    S = Rd[keep.to(torch.long)]
    Q, _ = torch.linalg.qr(S.T)
    resid = Rd - (Rd @ Q) @ Q.T

    def rel(weights: torch.Tensor | None) -> float:
        if weights is None:
            num, den = (resid * resid).sum(), (Rd * Rd).sum()
        else:
            a = weights.detach().to("cpu", torch.float64).clamp_min(0).unsqueeze(1)
            num, den = (a * resid * resid).sum(), (a * Rd * Rd).sum()
        return float(num / den) if float(den) > 0 else 0.0

    return rel(None), rel(w)


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
    rank = None if args.rank == 0 else args.rank
    rows: list[dict] = []
    overlaps: list[dict] = []

    with PrunableModel(model, layer=args.layer, n_heads_keep=args.heads) as pm:
        for i, s in enumerate(samples):
            try:
                batch = make_batch(s, processor, info.image_token_id, device,
                                   with_answer=False, prompt_style=args.prompt_style)
                inp = selection_inputs(pm, batch, eta=args.eta)
            except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                log.warning("sample %d skipped: %s", i, exc)
                torch.cuda.empty_cache()
                continue
            n_v = int(inp.visual_pos.numel())
            for budget in args.budgets:
                if budget >= n_v:
                    continue
                chosen: dict[str, torch.Tensor] = {}
                for crit in CRITERIA:
                    keep = select(crit, R=inp.residual, attn=inp.attention,
                                  residual_ratio=inp.ratio, budget=budget,
                                  seed=args.seed, rank=rank)
                    chosen[crit] = keep
                    unw, wtd = errors(inp.residual, inp.attention, keep)
                    rows.append({"example": i, "budget": budget, "criterion": crit,
                                 "n_visual": n_v, "recon": unw, "recon_weighted": wtd,
                                 "norm_pct": pct_rank(inp.residual.norm(dim=1), keep),
                                 "attn_pct": pct_rank(inp.attention, keep)})
                # Pairwise agreement between criteria, which is what a claim that
                # two signals are complementary has to be measured against.
                names = list(chosen)
                for a in range(len(names)):
                    for b in range(a + 1, len(names)):
                        ka, kb = set(chosen[names[a]].tolist()), set(chosen[names[b]].tolist())
                        overlaps.append({"example": i, "budget": budget,
                                         "a": names[a], "b": names[b],
                                         "overlap": len(ka & kb) / max(len(ka), 1)})
            if (i + 1) % 50 == 0:
                log.info("  %d/%d", i + 1, len(samples))

    if not rows:
        log.error("no examples produced measurements")
        return 1

    for path, data in ((run.artifact("selection.csv"), rows),
                       (run.artifact("overlap.csv"), overlaps)):
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    summary = _summarise(rows, overlaps)
    run.log(n_examples=len({r["example"] for r in rows}), **summary)
    run.write_metrics()

    tables = Path("results/tables")
    tables.mkdir(parents=True, exist_ok=True)
    name = Path(args.run_dir).name
    (tables / f"{name}.json").write_text(json.dumps(
        {"config": vars(args), **summary}, indent=2))
    log.info("wrote %s", tables / f"{name}.json")
    return 0


def _summarise(rows: list[dict], overlaps: list[dict]) -> dict:
    out: dict = {"recon": {}, "recon_weighted": {}, "norm_pct": {},
                 "attn_pct": {}, "overlap": {}}
    for key in ("recon", "recon_weighted", "norm_pct", "attn_pct"):
        by: dict[str, list[float]] = {}
        for r in rows:
            by.setdefault(f"{r['criterion']}@{r['budget']}", []).append(r[key])
        out[key] = {k: sum(v) / len(v) for k, v in sorted(by.items())}
    by_ov: dict[str, list[float]] = {}
    for o in overlaps:
        by_ov.setdefault(f"{o['a']}|{o['b']}@{o['budget']}", []).append(o["overlap"])
    out["overlap"] = {k: sum(v) / len(v) for k, v in sorted(by_ov.items())}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
