"""Properties the selection criteria are supposed to have, stated as tests.

The interesting ones concern leverage. The argument for it over the greedy
pipeline is that redundancy is priced into the score instead of being removed by
a later pass, and that is a checkable claim: duplicate a row and its score must
fall, roughly by half.
"""

from __future__ import annotations

import pytest
import torch

from xmodal.select import (
    CRITERIA,
    gram_eigh,
    reconstruction_error,
    ridge_for_budget,
    ridge_leverage,
    select,
    select_leverage,
    select_random,
)


def _rows(n: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=torch.float64)


def test_ridge_for_budget_hits_the_effective_dimension():
    ev = torch.tensor([10.0 ** (-0.1 * i) for i in range(200)], dtype=torch.float64)
    for budget in (5, 20, 64):
        lam = ridge_for_budget(ev, budget)
        eff = float((ev / (ev + lam)).sum())
        assert eff == pytest.approx(budget, rel=1e-6)


def test_ridge_for_budget_is_zero_when_rank_fits():
    ev = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64)
    assert ridge_for_budget(ev, 3) == 0.0
    assert ridge_for_budget(ev, 10) == 0.0


def test_leverage_splits_exactly_between_duplicated_rows():
    """Two copies of a direction share the leverage one copy would have had.

    Exact once the budget covers the rank, where the ridge term drops out: the
    scores then sum to the rank and a duplicated row's two copies take half
    each. This is the redundancy pricing the greedy pipeline installs by hand.
    """
    R = _rows(30, 50, seed=1)
    solo = ridge_leverage(R, budget=30, rank=None)
    Rd = torch.cat([R, R[:1]], dim=0)          # row 0 now appears twice
    dup = ridge_leverage(Rd, budget=31, rank=None)
    assert float(solo.sum()) == pytest.approx(30.0, rel=1e-9)
    assert float(dup[0]) == pytest.approx(float(dup[-1]), rel=1e-9)
    assert float(dup[0]) == pytest.approx(float(solo[0]) / 2, rel=1e-9)


def test_leverage_of_a_duplicate_falls_under_regularisation_too():
    """The same ordering survives the ridge term, without the exact factor."""
    R = _rows(30, 50, seed=1)
    solo = float(ridge_leverage(R, budget=10, rank=None)[0])
    dup = ridge_leverage(torch.cat([R, R[:1]], dim=0), budget=10, rank=None)
    assert float(dup[0]) == pytest.approx(float(dup[-1]), rel=1e-8)
    assert float(dup[0]) < solo


def test_leverage_prefers_the_unique_direction():
    """A lone direction outranks any member of a redundant cluster."""
    d = 40
    g = torch.Generator().manual_seed(2)
    base = torch.randn(1, d, generator=g, dtype=torch.float64)
    cluster = base.repeat(12, 1) + 0.01 * torch.randn(12, d, generator=g, dtype=torch.float64)
    lone = torch.randn(1, d, generator=g, dtype=torch.float64)
    R = torch.cat([cluster, lone], dim=0)
    tau = ridge_leverage(R, budget=3, rank=None)
    assert int(torch.argmax(tau)) == 12, "the unique row should carry the most leverage"


def test_randomised_gram_eigh_matches_exact_on_the_top_spectrum():
    R = _rows(1200, 300, seed=3)
    exact, _ = gram_eigh(R, rank=None)
    approx, _ = gram_eigh(R, rank=256, seed=0)
    k = 64
    rel = (exact[:k] - approx[:k]).abs() / exact[:k]
    assert float(rel.max()) < 0.05


def test_reconstruction_error_is_zero_when_everything_is_kept():
    R = _rows(24, 40, seed=4)
    assert reconstruction_error(R, torch.arange(24)) == pytest.approx(0.0, abs=1e-12)


def test_reconstruction_error_prefers_leverage_over_random():
    """The quantity the perturbation argument bounds damage by, on a hard matrix.

    The matrix is a few strong directions plus a long tail, which is the shape a
    text residual actually has. A uniform sample spends its budget in proportion
    to cluster size; leverage spends it on directions.
    """
    g = torch.Generator().manual_seed(5)
    d, budget = 60, 8
    basis = torch.randn(8, d, generator=g, dtype=torch.float64)
    weights = torch.zeros(200, 8, dtype=torch.float64)
    weights[:, 0] = torch.randn(200, generator=g, dtype=torch.float64)   # dominant cluster
    for j in range(1, 8):
        weights[j * 2, j] = 3.0                                          # rare directions
    R = weights @ basis + 0.01 * torch.randn(200, d, generator=g, dtype=torch.float64)

    attn = torch.ones(200, dtype=torch.float64)
    lev = reconstruction_error(R, select_leverage(R, attn, budget, rank=None))
    rnd = [reconstruction_error(R, select_random(200, budget, seed=s)) for s in range(20)]
    assert lev < sum(rnd) / len(rnd)


def _weighted_error(R: torch.Tensor, w: torch.Tensor, keep: torch.Tensor) -> float:
    """The quantity the perturbation bound names, as a relative energy."""
    B = R * w.clamp_min(0).sqrt().unsqueeze(1)
    Q, _ = torch.linalg.qr(R[keep.to(torch.long)].T)
    resid = B - (B @ Q) @ Q.T
    return float((resid * resid).sum() / (B * B).sum())


def test_pivoted_beats_leverage_and_greedy_on_the_weighted_objective():
    """The bound names one objective; the pivoted rule descends on it directly.

    Top-$k$ leverage is not the sampling the theory licenses, and it degrades
    when the weights are concentrated, which is the regime attention mass is
    actually in. The matrix here reproduces that regime: a heavy-tailed weight
    vector over a few strong directions plus a long tail.
    """
    g = torch.Generator().manual_seed(11)
    n, d, budget = 150, 60, 12
    basis = torch.randn(10, d, generator=g, dtype=torch.float64)
    coef = torch.randn(n, 10, generator=g, dtype=torch.float64)
    R = coef @ basis + 0.05 * torch.randn(n, d, generator=g, dtype=torch.float64)
    # Attention mass is concentrated: a few tokens carry most of it.
    w = torch.rand(n, generator=g, dtype=torch.float64) ** 6

    ratio = R.norm(dim=1) / R.norm(dim=1).max()
    piv = _weighted_error(R, w, select("pivot", R=R, attn=w, residual_ratio=ratio,
                                       budget=budget))
    lev = _weighted_error(R, w, select("leverage", R=R, attn=w, residual_ratio=ratio,
                                       budget=budget, rank=None))
    grd = _weighted_error(R, w, select("greedy", R=R, attn=w, residual_ratio=ratio,
                                       budget=budget))
    assert piv < lev, f"pivot {piv:.5f} should beat leverage {lev:.5f}"
    assert piv < grd, f"pivot {piv:.5f} should beat greedy {grd:.5f}"


def test_pivoted_is_deterministic_and_exact_when_the_budget_covers_the_rank():
    g = torch.Generator().manual_seed(12)
    basis = torch.randn(6, 30, generator=g, dtype=torch.float64)
    R = torch.randn(40, 6, generator=g, dtype=torch.float64) @ basis  # rank 6
    w = torch.rand(40, generator=g, dtype=torch.float64)
    ratio = R.norm(dim=1) / R.norm(dim=1).max()
    a = select("pivot", R=R, attn=w, residual_ratio=ratio, budget=10)
    b = select("pivot", R=R, attn=w, residual_ratio=ratio, budget=10)
    assert torch.equal(a, b)
    assert a.numel() == 10
    # Ten rows of a rank-six matrix span it, so nothing is left unreconstructed.
    assert _weighted_error(R, w, a) == pytest.approx(0.0, abs=1e-12)


def test_every_criterion_returns_a_valid_sorted_subset():
    n, d, budget = 80, 50, 16
    R = _rows(n, d, seed=6)
    attn = torch.rand(n, dtype=torch.float64)
    ratio = torch.rand(n, dtype=torch.float64)
    for name in CRITERIA:
        keep = select(name, R=R, attn=attn, residual_ratio=ratio, budget=budget, rank=None)
        assert keep.numel() == budget, name
        assert keep.unique().numel() == budget, name
        assert bool((keep == keep.sort().values).all()), name
        assert int(keep.min()) >= 0 and int(keep.max()) < n, name


def test_unknown_criterion_is_rejected():
    R = _rows(10, 10)
    with pytest.raises(ValueError, match="unknown criterion"):
        select("nope", R=R, attn=torch.ones(10), residual_ratio=torch.ones(10), budget=2)
