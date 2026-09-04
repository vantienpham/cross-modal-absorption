#!/usr/bin/env python3
"""Layer-wise cross-modal absorption, with its null and its controls.

One forward pass per example, then at every layer: how much of the visual token
energy the example's own text subspace explains, how much a *different*
example's text subspace of the same dimension explains, how much a same-modality
subspace of the same dimension explains, and where the random-subspace null sits
for that layer's spectrum.

    uv run --no-sync python scripts/measure_absorption.py \\
      --model llava-hf/llava-1.5-7b-hf --dataset lmms-lab/POPE --adapter pope \\
      --split test --limit 200 --run-dir out/runs/absorb-llava15-pope

The mismatched partner for example ``i`` is example ``i-1``, which is a random
pairing because the sample order is shuffled by seed. Holding one example's text
states at a time keeps memory flat in the number of examples; holding all of
them would be 16 MB per example across 33 layers.
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

from xmodal.absorption import measure                     # noqa: E402
from xmodal.calib import ablate_images, load_samples      # noqa: E402
from xmodal.controls import CENTRINGS, centre, compare, summarise  # noqa: E402
from xmodal.models import describe, load_vlm              # noqa: E402
from xmodal.runlog import RunDir                          # noqa: E402
from xmodal.states import capture, make_eval_batch        # noqa: E402

log = logging.getLogger("absorb")


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
    p.add_argument("--eta", type=float, default=0.75,
                   help="energy shell for the regularised reconstruction")
    p.add_argument("--centring", default="text", choices=CENTRINGS,
                   help="primary convention; 'joint' is always measured alongside")
    p.add_argument("--prompt-style", default="llava_v1", choices=("llava_v1", "chat"))
    p.add_argument("--swap-image", action="store_true",
                   help="add the same-question, different-image control; doubles "
                        "the forward passes")
    p.add_argument("--layer-stride", type=int, default=1,
                   help="measure every n-th layer; 1 measures all of them")
    p.add_argument("--run-dir", default="out/runs/absorb")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    run = RunDir(args.run_dir)
    run.write_config(vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("loading %s on %s", args.model, device)
    model, processor = load_vlm(args.model, dtype=torch.float16,
                                device_map={"": 0} if device == "cuda" else None,
                                attn_implementation="sdpa")
    if device == "cpu":
        model = model.to(device)
    info = describe(model, args.model)
    run.write_env({"model": info.to_dict()})
    log.info("model: %d layers, hidden %d, image token %d",
             info.n_layers, info.hidden_size, info.image_token_id)

    samples = load_samples(dataset=args.dataset, adapter=args.adapter,
                           split=args.split, limit=args.limit,
                           config=args.config, seed=args.seed)
    log.info("loaded %d samples from %s", len(samples), args.dataset)

    rows: list[dict] = []
    stat_rows: list[dict] = []
    prev_text: list[torch.Tensor] | None = None
    n_done = 0

    # Same question, different image. On a templated benchmark such as POPE, a
    # different example's text is nearly the same string, so the mismatched
    # control cannot separate "different content" from "different wording".
    # Running each question a second time over another example's image gives a
    # text subspace whose wording is identical and whose absorbed image is not.
    swapped = ablate_images(samples, "shuffle", seed=args.seed) if args.swap_image else None

    for i, s in enumerate(samples):
        try:
            batch = make_eval_batch(s, processor, info.image_token_id, device,
                                    prompt_style=args.prompt_style)
            st = capture(model, batch)
            st_swap = None
            if swapped is not None:
                b2 = make_eval_batch(swapped[i], processor, info.image_token_id,
                                     device, prompt_style=args.prompt_style)
                st_swap = capture(model, b2)
        except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            log.warning("example %d skipped: %s", i, exc)
            torch.cuda.empty_cache()
            continue

        layers = range(0, len(st), args.layer_stride)
        for l in layers:
            V, T = st.visual[l], st.text[l]
            # The headline statistic, under the literature's own convention.
            for how in (args.centring, "joint") if args.centring != "joint" else ("joint",):
                Vc, Tc = centre(V, T, how)
                stats, _ = measure(Vc, Tc, layer=l, eta=args.eta)
                d = stats.to_dict()
                d.update(example=i, centring=how)
                stat_rows.append(d)
            # The controls need a partner example at the same layer.
            if prev_text is not None and l < len(prev_text):
                T_swap = st_swap.text[l] if st_swap is not None else None
                for r in compare(V, T, prev_text[l], layer=l, example=i,
                                 centring=args.centring, seed=args.seed + i,
                                 T_swapped=T_swap):
                    rows.append(r.to_dict())

        prev_text = st.text
        n_done += 1
        if n_done % 25 == 0:
            log.info("  %d/%d examples", n_done, len(samples))

    if not stat_rows:
        log.error("no examples produced measurements")
        return 1

    _write_csv(run.artifact("absorption.csv"), stat_rows)
    if rows:
        _write_csv(run.artifact("controls.csv"), rows)

    summary = _summarise(stat_rows, args.centring)
    if rows:
        from xmodal.controls import ControlRow
        summary["controls"] = summarise([ControlRow(**r) for r in rows])
    run.log(n_examples=n_done, n_layers=len(set(r["layer"] for r in stat_rows)),
            summary=summary)
    run.write_metrics()

    tables = Path("results/tables")
    tables.mkdir(parents=True, exist_ok=True)
    name = Path(args.run_dir).name
    (tables / f"{name}.json").write_text(json.dumps(
        {"config": vars(args), "n_examples": n_done, **summary}, indent=2))
    log.info("wrote %s", tables / f"{name}.json")
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _summarise(stat_rows: list[dict], centring: str) -> dict:
    """Per-layer means of every measured quantity, for the primary centring."""
    keys = ("frac_orth", "frac_tik", "erank_visual", "null_mean", "null_sd",
            "z_orth", "sub_dim")
    by: dict[int, list[dict]] = {}
    for r in stat_rows:
        if r["centring"] == centring:
            by.setdefault(r["layer"], []).append(r)
    layers = sorted(by)
    out: dict = {"centring": centring, "layers": layers}
    for k in keys:
        out[k] = [sum(r[k] for r in by[l]) / len(by[l]) for l in layers]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
