"""The null distribution is the paper's load-bearing claim; these check it.

``null_mean`` and ``null_var`` are exact, not asymptotic, so a Monte Carlo
simulation of the same quantity is a test of the *implementation* and can be
held to a tight tolerance rather than a hand-waved one.
"""

from __future__ import annotations

import math

import pytest
import torch

from xmodal.absorption import (
    explained_fraction,
    null_mean,
    null_monte_carlo,
    null_sd,
    null_var,
    orthogonal_project,
    orthonormal_basis,
    participation_rank,
    residual_ratio,
    spectrum,
    tikhonov_reconstruct,
)


def _spectral_matrix(n: int, d: int, decay: float, seed: int = 0) -> torch.Tensor:
    """A matrix with a controlled singular value decay, for known effective rank."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, d, generator=g, dtype=torch.float64)
    U, _, Vh = torch.linalg.svd(A, full_matrices=False)
    s = torch.tensor([decay ** i for i in range(min(n, d))], dtype=torch.float64)
    return (U * s) @ Vh


def test_participation_rank_bounds():
    flat = torch.ones(50, dtype=torch.float64)
    assert participation_rank(flat) == pytest.approx(50.0)
    spike = torch.tensor([1.0] + [0.0] * 49, dtype=torch.float64)
    assert participation_rank(spike) == pytest.approx(1.0)


def test_null_mean_matches_simulation():
    """E[explained fraction] = k/D exactly, whatever the matrix looks like."""
    D, k = 128, 8
    for decay in (1.0, 0.85):
        V = _spectral_matrix(40, D, decay)
        sim = null_monte_carlo(V, k, trials=400, seed=1)
        se = float(sim.std()) / math.sqrt(sim.numel())
        assert abs(float(sim.mean()) - null_mean(D, k)) < 4 * se


def test_null_variance_matches_simulation():
    """The closed-form variance tracks the simulated one across anisotropy."""
    D, k, trials = 96, 6, 3000
    for decay in (1.0, 0.9, 0.7):
        V = _spectral_matrix(30, D, decay)
        erank = participation_rank(spectrum(V))
        sim = null_monte_carlo(V, k, trials=trials, seed=2)
        got, want = float(sim.var()), null_var(D, k, erank)
        # The variance of a sample variance is itself O(var^2 / trials), so the
        # tolerance has to scale with the quantity rather than be absolute.
        assert got == pytest.approx(want, rel=0.15), f"decay={decay} erank={erank:.1f}"


def test_null_variance_vanishes_for_isotropic_matrix():
    """A random subspace explains exactly k/D of an isotropic cloud, with no spread."""
    D = 64
    assert null_var(D, 4, erank=float(D)) == pytest.approx(0.0, abs=1e-15)
    assert null_var(D, 4, erank=1.0) > null_var(D, 4, erank=8.0) > 0


def test_null_variance_grows_as_effective_rank_falls():
    """The claim that makes depth trends hard to read: the null moves too."""
    D, k = 4096, 30
    eranks = [512.0, 64.0, 8.0, 2.0]
    sds = [null_sd(D, k, e) for e in eranks]
    assert sds == sorted(sds), "null spread must increase as the spectrum concentrates"


def test_orthogonal_projection_is_idempotent_and_exact_in_span():
    g = torch.Generator().manual_seed(3)
    T = torch.randn(12, 40, generator=g, dtype=torch.float64)
    B, _ = orthonormal_basis(T)
    assert B.shape[0] == 12
    assert torch.allclose(B @ B.T, torch.eye(12, dtype=torch.float64), atol=1e-10)

    # A matrix built inside the span is explained perfectly, and projecting it
    # twice changes nothing.
    C = torch.randn(7, 12, generator=g, dtype=torch.float64)
    V = C @ T
    P1 = orthogonal_project(V, B)
    assert explained_fraction(V, P1) == pytest.approx(1.0, abs=1e-10)
    assert torch.allclose(orthogonal_project(P1, B), P1, atol=1e-10)


def test_orthogonal_projection_drops_rank_deficient_directions():
    """A repeated text row must not inflate the reported subspace dimension."""
    g = torch.Generator().manual_seed(4)
    T = torch.randn(5, 30, generator=g, dtype=torch.float64)
    T = torch.cat([T, T[:2]], dim=0)          # rank 5, seven rows
    B, _ = orthonormal_basis(T)
    assert B.shape[0] == 5


def test_tikhonov_explains_no_more_than_orthogonal():
    """The regularised reconstruction is a shrunk projection, so it explains less."""
    g = torch.Generator().manual_seed(5)
    T = torch.randn(16, 64, generator=g, dtype=torch.float64)
    V = torch.randn(50, 64, generator=g, dtype=torch.float64)
    B, _ = orthonormal_basis(T)
    f_orth = explained_fraction(V, orthogonal_project(V, B))
    V_hat, k, lam = tikhonov_reconstruct(V, T, eta=0.75)
    assert 0 < k <= 16 and lam > 0
    assert explained_fraction(V, V_hat) <= f_orth + 1e-12


def test_residual_ratio_range_and_zero_rows():
    g = torch.Generator().manual_seed(6)
    T = torch.randn(8, 32, generator=g, dtype=torch.float64)
    V = torch.randn(20, 32, generator=g, dtype=torch.float64)
    V[3] = 0.0
    V_hat, _, _ = tikhonov_reconstruct(V, T)
    r = residual_ratio(V, V_hat)
    assert r.shape == (20,)
    assert float(r[3]) == 0.0
    assert bool(((r >= 0) & (r <= 1.0 + 1e-9)).all())


def test_controls_report_the_swapped_condition_at_a_common_dimension():
    """The same-question-different-image control, and the excess it defines."""
    from xmodal.controls import compare, summarise

    g = torch.Generator().manual_seed(7)
    V = torch.randn(40, 64, generator=g, dtype=torch.float32)
    T_self = torch.randn(10, 64, generator=g, dtype=torch.float32)
    T_other = torch.randn(14, 64, generator=g, dtype=torch.float32)
    T_swap = torch.randn(12, 64, generator=g, dtype=torch.float32)

    rows = compare(V, T_self, T_other, layer=3, example=0, T_swapped=T_swap)
    conds = {r.condition for r in rows}
    assert {"matched", "mismatched", "swapped", "visual"} <= conds
    # Every condition must be measured at the same subspace dimension, or the
    # comparison recovers the ranks rather than the pairing.
    assert len({r.sub_dim for r in rows}) == 1
    # Nine, not ten: centring subtracts the mean row, which costs one dimension,
    # so the text subspace of n tokens has rank n-1 at most. The null is
    # conditioned on the rank that is actually there.
    assert rows[0].sub_dim == 9

    s = summarise(rows)
    assert "swapped" in s["conditions"]
    assert "excess_image" in s
    assert s["excess_image"][0] == pytest.approx(
        s["conditions"]["matched"][0] - s["conditions"]["swapped"][0])


def test_controls_omit_the_swapped_condition_when_not_supplied():
    from xmodal.controls import compare

    g = torch.Generator().manual_seed(8)
    V = torch.randn(30, 48, generator=g, dtype=torch.float32)
    rows = compare(V, torch.randn(9, 48, generator=g, dtype=torch.float32),
                   torch.randn(9, 48, generator=g, dtype=torch.float32),
                   layer=0, example=0)
    assert "swapped" not in {r.condition for r in rows}
