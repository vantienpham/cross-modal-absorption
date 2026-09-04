#!/usr/bin/env python3
"""Aggregate the pruning runs into one table, with paired comparisons.

    uv run --no-sync python scripts/collect_results.py --runs out/runs \\
      --out results/tables/pruning.json

Runs on the cluster, where the run directories are. Reads ``metrics.json`` and
``per_item.json`` from every ``prune-*`` and ``layer-*`` run and writes a single
summary that ``slurm/sync.sh pull`` brings back.

Two runs at the same budget on the same benchmark scored the same examples in the
same order, so they can be compared *paired*: the quantity of interest is the
mean per-item difference and a confidence interval for it, not the gap between
two independent-looking averages. On a benchmark of a thousand binary items the
unpaired interval is wide enough to swallow every difference the literature
reports, while the paired one is not, because the two systems agree on most items
and the variance of the difference is far smaller than the variance of either
score. The bootstrap here resamples items, which is the unit that varies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

TITLES = {"pope": "POPE", "scienceqa": "ScienceQA-IMG", "textvqa": "TextVQA"}


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 10000,
                     seed: int = 0) -> dict:
    """Mean of ``a - b`` with a percentile bootstrap interval over items.

    Resampling is done as one ``(n_boot, n)`` index draw rather than a Python
    loop: with a thousand items and dozens of comparisons the loop version takes
    minutes, which is long enough that it stops being run.
    """
    import numpy as np

    if len(a) != len(b) or not a:
        return {}
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    obs = float(diff.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boots = diff[idx].mean(axis=1)
    lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    # A two-sided bootstrap p-value: the smaller tail mass on either side of zero.
    below = float((boots <= 0).mean())
    return {"delta": obs, "ci_low": lo, "ci_high": hi,
            "p_two_sided": min(1.0, 2 * min(below, 1 - below)),
            "significant": (lo > 0) or (hi < 0)}


def read_runs(root: Path, prefix: str) -> list[dict]:
    out = []
    for d in sorted(root.glob(f"{prefix}*")):
        m = d / "metrics.json"
        if not m.is_file():
            continue
        rec = json.loads(m.read_text())
        for field in ("per_item", "predictions"):
            p = d / f"{field}.json"
            if p.is_file():
                rec[field] = json.loads(p.read_text())
        rec["run"] = d.name
        out.append(rec)
    return out


def agreement(pred: list[str], ref: list[str]) -> float | None:
    """Fraction of items where the pruned model produced the reference string.

    A sharper probe than accuracy of how far pruning moved the model: a score
    that does not flip is not evidence that the answer did not change, because
    two different answers can both be wrong. Comparison is on the normalised
    string, so trailing punctuation and case do not count as disagreement.
    """
    if not pred or not ref or len(pred) != len(ref):
        return None
    norm = lambda s: " ".join(s.lower().split()).rstrip(".")  # noqa: E731
    return mean([float(norm(a) == norm(b)) for a, b in zip(pred, ref)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("out/runs"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/pruning.json"))
    ap.add_argument("--baseline", default="leverage",
                    help="criterion every other one is compared against")
    args = ap.parse_args()

    runs = read_runs(args.runs, "prune-")
    reference = {r["task"]: r.get("predictions") for r in runs if r["criterion"] == "none"}
    tasks: dict[str, dict] = {}
    for r in runs:
        t = r["task"]
        d = tasks.setdefault(t, {"title": TITLES.get(t, t), "criteria": {},
                                 "none": None, "n": r.get("n"),
                                 "n_visual": r.get("n_visual"), "recon": {},
                                 "yes_rate": {}, "agreement": {}})
        if r["criterion"] == "none":
            d["none"] = r["accuracy"]
            continue
        b = str(r["budget"])
        d["criteria"].setdefault(r["criterion"], {})[b] = r["accuracy"]
        if r.get("recon_error") is not None:
            d["recon"].setdefault(r["criterion"], {})[b] = r["recon_error"]
        if r.get("yes_rate") is not None:
            d["yes_rate"].setdefault(r["criterion"], {})[b] = r["yes_rate"]
        agr = agreement(r.get("predictions") or [], reference.get(t) or [])
        if agr is not None:
            d["agreement"].setdefault(r["criterion"], {})[b] = agr

    # Paired comparisons, within (task, budget), against the named criterion.
    by_key: dict[tuple[str, int, str], list[float]] = {
        (r["task"], r["budget"], r["criterion"]): r["per_item"]
        for r in runs if "per_item" in r and r["criterion"] != "none"
    }
    comparisons = []
    for (task, budget, crit), scores in sorted(by_key.items()):
        if crit == args.baseline:
            continue
        ref = by_key.get((task, budget, args.baseline))
        if ref is None or len(ref) != len(scores):
            continue
        res = paired_bootstrap(ref, scores)
        if res:
            comparisons.append({"task": task, "budget": budget,
                                "baseline": args.baseline, "against": crit, **res})

    # The layer sweep reuses the same benchmark, split, limit and seed as the
    # budget campaign, so the unpruned run of the matching task is a valid
    # reference for agreement without rerunning it.
    layers: dict[str, dict] = {}
    for r in read_runs(args.runs, "layer-"):
        t = r["task"]
        d = layers.setdefault(t, {"title": TITLES.get(t, t), "accuracy": {},
                                  "agreement": {},
                                  "none": tasks.get(t, {}).get("none")})
        lay = str(r["layer"])
        d["accuracy"].setdefault(r["criterion"], {})[lay] = r["accuracy"]
        agr = agreement(r.get("predictions") or [], reference.get(t) or [])
        if agr is not None:
            d["agreement"].setdefault(r["criterion"], {})[lay] = agr

    payload = {**tasks, "_comparisons": comparisons, "_layers": layers}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {args.out} from {len(runs)} pruning runs")
    for c in comparisons:
        star = "*" if c["significant"] else " "
        print(f"  {c['task']:8s} b={c['budget']:<4d} "
              f"{c['baseline']} - {c['against']:9s} "
              f"{c['delta']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] {star}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
