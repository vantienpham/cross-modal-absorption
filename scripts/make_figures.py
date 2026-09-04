#!/usr/bin/env python3
"""Figures for the manuscript, from the summary tables pulled off the cluster.

    python3 scripts/make_figures.py --tables results/tables --out redaction/figures

Runs on the laptop, not the cluster: it reads only the small JSON summaries that
``slurm/sync.sh pull`` brings back, so the figures can be regenerated without a
GPU and without the run artifacts.

Everything is drawn in one grey-and-black scheme with distinguishable markers,
because the journal prints in colour but is read in both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

STYLE = {
    "matched":    dict(color="#1a1a1a", marker="o", ms=3, lw=1.4, label="matched text"),
    "mismatched": dict(color="#c1121f", marker="s", ms=3, lw=1.4, ls="--",
                       label="mismatched text"),
    "swapped":    dict(color="#d97706", marker="D", ms=2.6, lw=1.3, ls=(0, (3, 2)),
                       label="same question, other image"),
    "visual":     dict(color="#4a7ba7", marker="^", ms=3, lw=1.2, ls=":",
                       label="visual subspace"),
}

CRIT_STYLE = {
    "random":    dict(color="#9a9a9a", marker="x", ms=4, lw=1.2, ls=":"),
    "attention": dict(color="#4a7ba7", marker="s", ms=3.5, lw=1.3, ls="--"),
    "residual":  dict(color="#6a9955", marker="^", ms=3.5, lw=1.3, ls="-."),
    "greedy":    dict(color="#c1121f", marker="d", ms=3.5, lw=1.3, ls="--"),
    "leverage":  dict(color="#8a5fa8", marker="v", ms=3.5, lw=1.3, ls="--"),
    "pivot":     dict(color="#1a1a1a", marker="o", ms=3.5, lw=1.6, ls="-"),
}

TITLES = {
    "swap-llava15-pope": "LLaVA-1.5-7B, POPE",
    "swap-llava15-textvqa": "LLaVA-1.5-7B, TextVQA",
    "swap-llava15-sqa": "LLaVA-1.5-7B, ScienceQA-IMG",
    "swap-qwen25-pope": "Qwen2.5-VL-7B, POPE",
    "absorb-llava15-pope": "LLaVA-1.5-7B, POPE",
    "absorb-llava15-sqa": "LLaVA-1.5-7B, ScienceQA-IMG",
    "absorb-llava15-textvqa": "LLaVA-1.5-7B, TextVQA",
    "absorb-next-pope": "LLaVA-NeXT-7B, POPE",
    "absorb-next-textvqa": "LLaVA-NeXT-7B, TextVQA",
    "absorb-nextlite-pope": "LLaVA-NeXT-7B, POPE",
    "absorb-nextlite-textvqa": "LLaVA-NeXT-7B, TextVQA",
    "absorb-qwen25-pope": "Qwen2.5-VL-7B, POPE",
}


def load(tables: Path, name: str) -> dict | None:
    p = tables / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def fig_absorption(tables: Path, out: Path, names: list[str]) -> str | None:
    """The statistic and its controls above, the decomposition below.

    The upper row alone is misleading in one direction: matched and mismatched
    are so close that the gap between them is hard to read, while both are so far
    above the null that the null looks like the floor of the axis. The lower row
    plots the share of the excess over the null that is content-specific, which
    is the quantity the cross-modal reading of the statistic depends on.
    """
    have = [(n, load(tables, n)) for n in names]
    have = [(n, d) for n, d in have if d and "controls" in d]
    if not have:
        return None
    fig, axes = plt.subplots(2, len(have), figsize=(2.9 * len(have), 4.0),
                             sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.18})
    for j, (name, d) in enumerate(have):
        ax, ax2 = axes[0][j], axes[1][j]
        c = d["controls"]
        L = c["layers"]
        for cond in ("matched", "swapped", "mismatched", "visual"):
            if cond in c["conditions"]:
                ax.plot(L, c["conditions"][cond], **STYLE[cond])
        mu, sd = d["null_mean"], d["null_sd"]
        ax.fill_between(d["layers"], [m - 3 * s for m, s in zip(mu, sd)],
                        [m + 3 * s for m, s in zip(mu, sd)],
                        color="#d9a441", alpha=0.9, lw=0,
                        label=r"random subspace, $\pm 3\sigma$")
        ax.set_title(TITLES.get(name, name))
        ax.set_ylim(0, 0.92)

        # Prefer the image-specific share when the swapped-image control ran:
        # it holds the question fixed, so the gap cannot be the wording.
        if "excess_image" in c:
            share = [100 * e / max(m - n, 1e-12) for e, m, n
                     in zip(c["excess_image"], c["conditions"]["matched"], c["null_mean"])]
        else:
            share = [100 * a / max(a + b, 1e-12)
                     for a, b in zip(c["excess_matched"], c["excess_generic"])]
        ax2.plot(L, share, color="#1a1a1a", marker="o", ms=2.5, lw=1.3)
        ax2.axhline(0, color="#999999", lw=0.7)
        ax2.axhline(100, color="#999999", lw=0.7, ls=":")
        ax2.set_ylim(-12, 108)
        ax2.set_xlabel("decoder layer")
        if j:
            ax.tick_params(labelleft=False)
            ax2.tick_params(labelleft=False)
    axes[0][0].set_ylabel("explained fraction")
    axes[1][0].set_ylabel("image-specific\nshare of excess (\\%)")
    # One legend above the grid rather than inside a panel: every panel is full
    # enough that an inset legend covers data in at least one of them.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.5, ncol=5,
               loc="upper center", bbox_to_anchor=(0.5, 1.05))
    path = out / "absorption_controls.pdf"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def fig_erank(tables: Path, out: Path, names: list[str]) -> str | None:
    """Participation rank against depth, beside the null spread it produces."""
    have = [(n, load(tables, n)) for n in names]
    have = [(n, d) for n, d in have if d]
    if not have:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.5),
                               gridspec_kw={"wspace": 0.30})
    marks = ["o", "s", "^", "d", "v", "P"]
    for (name, d), m in zip(have, marks):
        ax1.plot(d["layers"], d["erank_visual"], marker=m, ms=2.5, lw=1.2,
                 label=TITLES.get(name, name))
        ax2.plot(d["layers"], d["null_sd"], marker=m, ms=2.5, lw=1.2)
    ax1.set_xlabel("decoder layer")
    ax1.set_ylabel("participation rank of $V$")
    ax1.set_yscale("log")
    ax2.set_xlabel("decoder layer")
    ax2.set_ylabel(r"null s.d. of $\mathcal{A}$")
    # Legend above both panels: the curves fill the axes and an inset legend
    # covers data in one of them whichever corner it is put in.
    h, la = ax1.get_legend_handles_labels()
    fig.legend(h, la, frameon=False, fontsize=6.5, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 1.10))
    path = out / "effective_rank.pdf"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def fig_pruning(tables: Path, out: Path) -> str | None:
    """Accuracy against budget, one panel per benchmark, with the unpruned line."""
    p = tables / "pruning.json"
    if not p.is_file():
        return None
    data = json.loads(p.read_text())
    tasks = [t for t in ("pope", "textvqa", "scienceqa") if t in data]
    if not tasks:
        return None
    fig, axes = plt.subplots(1, len(tasks), figsize=(2.9 * len(tasks), 2.4),
                             squeeze=False)
    for ax, task in zip(axes[0], tasks):
        d = data[task]
        for crit, style in CRIT_STYLE.items():
            if crit not in d["criteria"]:
                continue
            xs = sorted(int(b) for b in d["criteria"][crit])
            ys = [d["criteria"][crit][str(b)] for b in xs]
            ax.plot(xs, ys, label=crit, **style)
        if d.get("none") is not None:
            ax.axhline(d["none"], color="#666666", lw=0.8, ls=(0, (4, 3)))
            ax.text(0.98, d["none"], "unpruned", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=6.5, color="#666666")
        ax.set_xlabel("visual tokens kept")
        ax.set_title(d.get("title", task))
    axes[0][0].set_ylabel("accuracy")
    axes[0][0].legend(frameon=False, fontsize=6.5, loc="lower right")
    path = out / "pruning_accuracy.pdf"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def fig_layers(tables: Path, out: Path) -> str | None:
    """Accuracy against pruning layer at a fixed budget, one panel per benchmark.

    The paper's sharpest comparison. Where the criteria separate at all is a
    property of depth, and the panel makes it visible without a table.
    """
    p = tables / "pruning.json"
    if not p.is_file():
        return None
    data = json.loads(p.read_text()).get("_layers") or {}
    tasks = [t for t in ("pope", "textvqa", "scienceqa") if t in data]
    if not tasks:
        return None
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.0 * len(tasks), 2.5),
                             squeeze=False)
    for ax, task in zip(axes[0], tasks):
        d = data[task]
        for crit, style in CRIT_STYLE.items():
            acc = d["accuracy"].get(crit)
            if not acc:
                continue
            xs = sorted(int(k) for k in acc)
            ax.plot(xs, [acc[str(x)] for x in xs], label=crit, **style)
        if d.get("none") is not None:
            ax.axhline(d["none"], color="#666666", lw=0.8, ls=(0, (4, 3)))
            ax.text(0.02, d["none"], "unpruned", transform=ax.get_yaxis_transform(),
                    ha="left", va="bottom", fontsize=6.5, color="#666666")
        ax.set_xlabel("pruning layer")
        ax.set_title(d.get("title", task))
    axes[0][0].set_ylabel("accuracy")
    axes[0][0].legend(frameon=False, fontsize=6.5, loc="lower right")
    path = out / "layer_sweep.pdf"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def fig_selection(tables: Path, out: Path) -> str | None:
    """Reconstruction error by criterion: the objective the bound names."""
    names = [n for n in ("selq-pope", "selq-sqa", "selq-textvqa")
             if (tables / f"{n}.json").is_file()]
    if not names:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.4), sharex=True)
    d = json.loads((tables / names[0]).with_suffix(".json").read_text())
    budgets = sorted({int(k.split("@")[1]) for k in d["recon"]})
    for ax, key, title in ((axes[0], "recon", "unweighted"),
                           (axes[1], "recon_weighted", "attention-weighted")):
        for crit, style in CRIT_STYLE.items():
            ys = [d[key].get(f"{crit}@{b}") for b in budgets]
            if any(y is None for y in ys):
                continue
            ax.plot(budgets, ys, label=crit, **style)
        ax.set_xlabel("visual tokens kept")
        ax.set_title(title)
    axes[0].set_ylabel("relative reconstruction error")
    axes[0].legend(frameon=False, fontsize=6.5)
    path = out / "selection_quality.pdf"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=Path, default=Path("results/tables"))
    ap.add_argument("--out", type=Path, default=Path("redaction/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    absorb = ["swap-llava15-pope", "swap-llava15-textvqa", "swap-qwen25-pope"]
    # One entry per distinct (model, benchmark): the swap-* runs measure the
    # same thing as the absorb-* ones and listing both would draw each curve
    # twice. LLaVA-Next falls back to the reduced-depth run if the full one did
    # not finish.
    all_absorb = ["swap-llava15-pope", "swap-llava15-textvqa", "swap-llava15-sqa",
                  "swap-qwen25-pope"]
    for full, lite in (("absorb-next-pope", "absorb-nextlite-pope"),
                       ("absorb-next-textvqa", "absorb-nextlite-textvqa")):
        all_absorb.append(full if (args.tables / f"{full}.json").is_file() else lite)
    made = [
        fig_absorption(args.tables, args.out, absorb),
        fig_erank(args.tables, args.out, all_absorb),
        fig_pruning(args.tables, args.out),
        fig_layers(args.tables, args.out),
        fig_selection(args.tables, args.out),
    ]
    for m in made:
        print("wrote" if m else "skipped (no data)", m or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
