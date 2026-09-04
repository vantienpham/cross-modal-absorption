"""Controls that say whether absorption is cross-modal or merely geometric.

A rising explained fraction across depth is consistent with two very different
stories. Under the first, text representations accumulate the visual content of
*this* image, so the text subspace of an example comes to explain that example's
visual tokens specifically. Under the second, representations of both modalities
drift into a shared, low-dimensional, strongly anisotropic region of the hidden
space, and any subspace of the right dimension drawn from that region explains
them equally well, whatever image it came from.

The matched measurement alone cannot separate these, because it is the same
number under both. Three comparisons at the same subspace dimension can:

``matched``
    Visual tokens against the text of the *same* example. What the literature
    reports.
``mismatched``
    Visual tokens against the text of a *different* example, drawn at the same
    layer. Keeps the modality, the layer and the dimension; destroys only the
    pairing. Whatever this explains is not content-specific.
``visual``
    Visual tokens against a subspace spanned by other visual tokens of the same
    example. Fixes the pairing and the dimension and changes the modality, so it
    asks whether text is special or merely low-dimensional.
``swapped``
    Visual tokens against the text of a forward pass carrying the *same
    question* over a *different image*. This is the control the mismatched one
    cannot be on a templated benchmark: POPE asks "Is there a ... in the image?"
    of every example, so a different example's text is very nearly the same
    string, and the mismatched condition then understates how much of absorption
    is image-specific. Holding the question fixed and changing only the image
    removes that objection, because any gap between ``matched`` and ``swapped``
    is attributable to the image alone.

The excess of ``matched`` over ``mismatched`` is the part of absorption that is
about this image and this question. The excess of ``mismatched`` over the random
null (:mod:`xmodal.absorption`) is shared anisotropy. The two answer different
questions and only the first supports a claim about cross-modal information
flow.

Every comparison is made at a **common subspace dimension**, truncating both
bases to the smaller of the two numerical ranks. Explained fraction is monotone
in dimension, so comparing a rank-31 subspace against a rank-24 one would
recover the ranks rather than the pairing.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import torch

from .absorption import (
    explained_fraction,
    null_mean,
    null_sd,
    null_z,
    orthogonal_project,
    orthonormal_basis,
    participation_rank,
    spectrum,
)

#: How the two modalities are put on a common origin before anything is
#: measured. ``text`` reproduces the convention of the pruning literature, which
#: subtracts the mean text vector from both. ``joint`` subtracts the mean over
#: all tokens of the example. The choice is not cosmetic: under ``text`` the
#: offset between the modality means stays inside the visual matrix and is
#: mostly orthogonal to the text subspace, which *lowers* the explained
#: fraction. Both are measured so that the headline number can be checked
#: against its own convention.
CENTRINGS = ("text", "joint", "none")


def centre(V: torch.Tensor, T: torch.Tensor, how: str = "text"
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Put visual and text matrices on a common origin."""
    if how == "text":
        mu = T.mean(dim=0, keepdim=True)
    elif how == "joint":
        mu = torch.cat([V, T], dim=0).mean(dim=0, keepdim=True)
    elif how == "none":
        mu = torch.zeros(1, V.shape[1], dtype=V.dtype, device=V.device)
    else:
        raise ValueError(f"unknown centring {how!r}; expected one of {CENTRINGS}")
    return V - mu, T - mu


def truncate(B: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the first ``k`` rows of an orthonormal basis, which are its strongest.

    :func:`xmodal.absorption.orthonormal_basis` returns rows ordered by the
    energy of the direction they came from, so truncation keeps the dominant
    part of the subspace rather than an arbitrary slice of it.
    """
    if k > B.shape[0]:
        raise ValueError(f"cannot truncate a rank-{B.shape[0]} basis to {k}")
    return B[:k]


def visual_control_basis(V: torch.Tensor, k: int, seed: int = 0) -> torch.Tensor:
    """Orthonormal basis from ``k`` visual rows chosen at random, excluding none.

    The same-modality control. Drawn from the example's own visual tokens, so it
    shares their anisotropy exactly; if this explains as much as the text
    subspace does, then the text subspace is doing nothing a subspace of that
    dimension would not.

    Rows are sampled without replacement, then orthonormalised, so the returned
    rank can come out below ``k`` when the sampled rows are collinear. The
    caller re-truncates to the common dimension afterwards.
    """
    n = V.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(n, generator=g)[: min(k, n)]
    B, _ = orthonormal_basis(V[idx])
    return B


@dataclass
class ControlRow:
    """One layer of one example, under one control condition."""

    layer: int
    example: int
    condition: str
    centring: str
    sub_dim: int
    hidden_dim: int
    n_visual: int
    erank_visual: float
    frac: float
    null_mean: float
    null_sd: float
    z: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare(
    V: torch.Tensor,
    T_self: torch.Tensor,
    T_other: torch.Tensor,
    layer: int,
    example: int,
    centring: str = "text",
    seed: int = 0,
    T_swapped: torch.Tensor | None = None,
) -> list[ControlRow]:
    """Measure every condition at one layer, at a common subspace dimension.

    Args:
        V: visual token rows for this example at this layer, uncentred.
        T_self: text token rows for the same example at the same layer.
        T_other: text token rows for a different example at the same layer.
        layer: index, recorded only.
        example: index, recorded only.
        centring: one of :data:`CENTRINGS`.
        seed: for the visual control's row sample.
        T_swapped: text rows from a pass with this example's question over a
            different image. Optional; when absent the ``swapped`` condition is
            not reported.

    Returns one row per condition. The visual control is skipped when the
    example has too few visual tokens to span the common dimension, which does
    not happen for the models here but would for a heavily pre-pruned input.
    """
    Vc, Tc_self = centre(V, T_self, centring)
    # The other example's text is centred on its own statistics, not this
    # example's: the control is "a text subspace from elsewhere", and moving it
    # to this example's origin would smuggle part of the pairing back in.
    _, Tc_other = centre(T_other, T_other, centring if centring != "text" else "text")

    B_self, _ = orthonormal_basis(Tc_self)
    B_other, _ = orthonormal_basis(Tc_other)
    ranks = [B_self.shape[0], B_other.shape[0]]
    B_swap = None
    if T_swapped is not None:
        _, Tc_swap = centre(T_swapped, T_swapped, centring)
        B_swap, _ = orthonormal_basis(Tc_swap)
        ranks.append(B_swap.shape[0])
    k = int(min(ranks))
    if k == 0:
        return []

    D = int(V.shape[1])
    erank = participation_rank(spectrum(Vc))
    mu, sd = null_mean(D, k), null_sd(D, k, erank)

    def row(condition: str, B: torch.Tensor) -> ControlRow:
        frac = explained_fraction(Vc, orthogonal_project(Vc, B))
        return ControlRow(
            layer=layer, example=example, condition=condition, centring=centring,
            sub_dim=k, hidden_dim=D, n_visual=int(V.shape[0]), erank_visual=erank,
            frac=frac, null_mean=mu, null_sd=sd, z=null_z(frac, D, k, erank),
        )

    rows = [row("matched", truncate(B_self, k)), row("mismatched", truncate(B_other, k))]
    if B_swap is not None:
        rows.append(row("swapped", truncate(B_swap, k)))
    B_vis = visual_control_basis(Vc, k, seed=seed)
    if B_vis.shape[0] >= k:
        rows.append(row("visual", truncate(B_vis, k)))
    return rows


def summarise(rows: Sequence[ControlRow]) -> dict[str, Any]:
    """Mean explained fraction per (layer, condition), and the matched excess.

    ``excess_matched`` is the quantity a claim about cross-modal information
    flow has to rest on: matched minus mismatched, at equal dimension.
    ``excess_generic`` is mismatched minus the random null, which is shared
    anisotropy and is not evidence of anything cross-modal.
    """
    by: dict[tuple[int, str], list[float]] = {}
    nulls: dict[int, list[float]] = {}
    for r in rows:
        by.setdefault((r.layer, r.condition), []).append(r.frac)
        nulls.setdefault(r.layer, []).append(r.null_mean)

    layers = sorted({layer for layer, _ in by})
    out: dict[str, Any] = {"layers": layers, "conditions": {}}
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731 - local, and used repeatedly
    for cond in ("matched", "mismatched", "swapped", "visual"):
        vals = [mean(by[(l, cond)]) for l in layers if (l, cond) in by]
        if vals:
            out["conditions"][cond] = vals
    out["null_mean"] = [mean(nulls[l]) for l in layers]
    m, mm = out["conditions"].get("matched"), out["conditions"].get("mismatched")
    if m and mm:
        out["excess_matched"] = [a - b for a, b in zip(m, mm)]
        out["excess_generic"] = [b - n for b, n in zip(mm, out["null_mean"])]
    sw = out["conditions"].get("swapped")
    if m and sw:
        # The image-specific excess: same question, different image, so the gap
        # cannot be attributed to the question's wording.
        out["excess_image"] = [a - b for a, b in zip(m, sw)]
    return out
