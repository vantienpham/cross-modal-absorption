"""How much of a visual token matrix a text subspace explains, and how much of
that is worth believing.

The quantity under study is the fraction of visual-token energy lying in the
span of the text tokens of the same example,

.. math::   A = 1 - \\|V_c - \\Pi V_c\\|_F^2 / \\|V_c\\|_F^2 ,

reported layer by layer. Read naively it looks like a number on ``[0, 1]``, and
a rise from 0.01 to 0.21 across depth looks like a large effect. It is not
readable that way, for two reasons that this module exists to handle.

**The scale is set by the dimensions, not by the modalities.** A text subspace
has dimension at most the number of text tokens, typically a few tens against a
hidden size of thousands. A *random* subspace of that dimension already explains
a nonzero fraction of any cloud. :func:`null_mean` gives that fraction exactly,
and :func:`null_sd` gives the spread around it, so an observed value converts to
a z-score instead of being compared against zero.

**The null spread is not constant across depth.** It grows as the visual token
matrix becomes more anisotropic (:func:`participation_rank`), and transformer
representations do become more anisotropic with depth. A rise in explained
fraction across layers is therefore not by itself evidence of anything
cross-modal: part of it is the null moving. Separating the two is what
:mod:`xmodal.controls` measures and what the null here makes quantitative.

Both projections are provided and both are reported. :func:`orthogonal_project`
is idempotent and is the one the null distribution below describes exactly.
:func:`tikhonov_reconstruct` is the regularised variant used by the pruning
literature; it is a shrunk projection, so its explained fraction is bounded
above by the orthogonal one and the null does not apply to it unchanged.

Numerics: every decomposition here runs in float64 on the CPU. The matrices are
small (the text Gram is a few tens square) so it costs nothing, and it avoids
the two documented cuSOLVER failures on this cluster's older cards, one of which
returns ``info == 0`` beside a factor full of NaN. See ``cluster.local.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch


# --- spectra ----------------------------------------------------------------


def participation_rank(energies: torch.Tensor) -> float:
    """Effective rank of a spectrum, as a participation ratio.

    For squared singular values :math:`\\sigma_r^2` with normalised weights
    :math:`p_r`, this is :math:`1 / \\sum_r p_r^2`. It equals the true rank for
    a flat spectrum and falls toward 1 as one direction takes over, which is the
    regime deep transformer representations drift into.

    This is the only property of the visual matrix the null distribution needs.
    """
    e = energies.double()
    e = e[e > 0]
    if e.numel() == 0:
        return 0.0
    p = e / e.sum()
    return float(1.0 / (p * p).sum())


def spectrum(X: torch.Tensor) -> torch.Tensor:
    """Squared singular values of ``X`` (n, d), descending, float64 on CPU.

    Computed as the eigenvalues of the smaller Gram matrix rather than by SVD.
    ``torch.linalg.svd`` on this cluster's older cards can fail to converge on
    ill-conditioned input even after the internal fallback, and the Gram route
    needs only a symmetric eigendecomposition of a matrix whose side is
    ``min(n, d)``.
    """
    Xd = X.detach().to("cpu", torch.float64)
    n, d = Xd.shape
    G = Xd @ Xd.T if n <= d else Xd.T @ Xd
    ev = torch.linalg.eigvalsh(G)
    if not torch.isfinite(ev).all():
        raise FloatingPointError(
            "non-finite eigenvalues from the Gram matrix; the hidden states "
            "reaching this function are already corrupted"
        )
    return ev.flip(0).clamp_min(0.0)


# --- the null distribution --------------------------------------------------
#
# For a subspace of dimension k drawn uniformly (Haar) in R^D and a *fixed*
# matrix V with normalised squared singular values p_r, the explained fraction
# is  A = sum_r p_r * beta_r,  where beta_r = ||U^T b_r||^2 for the right
# singular vectors b_r. Each beta_r is Beta(k/2, (D-k)/2) by rotational
# invariance, and the beta_r over a full orthonormal basis sum to k, which fixes
# their covariance. Both moments below follow, exactly rather than
# asymptotically.


def null_mean(hidden_dim: int, sub_dim: int) -> float:
    """Expected explained fraction under a uniformly random subspace.

    Exactly ``sub_dim / hidden_dim``, for any ``V`` whatsoever: rotational
    invariance gives :math:`\\mathbb{E}[UU^\\top] = (k/D) I`, so the spectrum
    drops out of the first moment. It does not drop out of the second.
    """
    if not 0 <= sub_dim <= hidden_dim:
        raise ValueError(f"sub_dim {sub_dim} outside [0, {hidden_dim}]")
    return sub_dim / hidden_dim


def null_var(hidden_dim: int, sub_dim: int, erank: float) -> float:
    """Variance of the explained fraction under a uniformly random subspace.

    .. math::
        \\mathrm{Var} = \\frac{2k(D-k)}{D^2(D+2)}
                        \\cdot \\frac{D/\\mathrm{erank} - 1}{D - 1}

    with ``erank`` the participation rank of the matrix being explained. The
    first factor is the variance of a single ``Beta(k/2, (D-k)/2)``; the second
    is the reduction from averaging over the spectrum, which vanishes for an
    isotropic matrix (``erank = D``, variance 0, because a random subspace then
    explains exactly ``k/D`` with no fluctuation) and is largest for a rank-one
    one.

    The dependence on ``erank`` is the part that matters in practice: the same
    observed value is more or less surprising depending on how anisotropic the
    representations at that layer have become.
    """
    D, k = hidden_dim, sub_dim
    if D < 2:
        raise ValueError("hidden_dim must be at least 2")
    if erank <= 0:
        return 0.0
    beta_var = 2.0 * k * (D - k) / (D * D * (D + 2))
    shrink = (D / erank - 1.0) / (D - 1.0)
    return max(beta_var * shrink, 0.0)


def null_sd(hidden_dim: int, sub_dim: int, erank: float) -> float:
    return null_var(hidden_dim, sub_dim, erank) ** 0.5


def null_z(observed: float, hidden_dim: int, sub_dim: int, erank: float) -> float:
    """Observed explained fraction as a z-score against the random-subspace null.

    Returns ``inf`` when the null has no spread (an isotropic matrix, where any
    excess over ``k/D`` is deterministic evidence). Callers that report this
    should also report the raw fraction: a z-score of 200 and one of 20 are both
    "significant" and the distinction that matters is the effect size.
    """
    sd = null_sd(hidden_dim, sub_dim, erank)
    if sd == 0.0:
        return float("inf") if observed > null_mean(hidden_dim, sub_dim) else 0.0
    return (observed - null_mean(hidden_dim, sub_dim)) / sd


def null_monte_carlo(V: torch.Tensor, sub_dim: int, trials: int = 64,
                     seed: int = 0) -> torch.Tensor:
    """Simulate the null by drawing Haar subspaces and projecting.

    Present to check :func:`null_mean` and :func:`null_var` against something
    that makes no distributional assumption at all. The closed forms are exact,
    so this is a test of the implementation rather than of the mathematics, and
    it is what ``tests/test_absorption.py`` compares against.
    """
    Vd = V.detach().to("cpu", torch.float64)
    D = Vd.shape[1]
    total = (Vd * Vd).sum()
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = torch.empty(trials, dtype=torch.float64)
    for t in range(trials):
        # A Haar-random orthonormal basis: QR of a Gaussian matrix.
        Q, _ = torch.linalg.qr(torch.randn(D, sub_dim, generator=g, dtype=torch.float64))
        out[t] = ((Vd @ Q) ** 2).sum() / total
    return out


# --- projections ------------------------------------------------------------


def orthonormal_basis(T: torch.Tensor, rel_tol: float = 1e-10
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthonormal basis for the row space of ``T`` (n_t, d), and its spectrum.

    Returns ``(B, energies)`` with ``B`` of shape ``(k, d)``, ``B B^T = I_k``,
    and ``k`` the numerical rank. Directions whose energy falls below
    ``rel_tol`` times the largest are dropped rather than inverted: they are the
    ones that make the least-squares solution blow up, and keeping them would
    make the "dimension" of the text subspace a fiction.

    Via ``eigh`` of the text Gram in float64 on the CPU. The Gram is ``n_t``
    square, a few tens on a side, so this is free.
    """
    Td = T.detach().to("cpu", torch.float64)
    G = Td @ Td.T
    ev, U = torch.linalg.eigh(G)
    if not torch.isfinite(ev).all() or not torch.isfinite(U).all():
        raise FloatingPointError("eigh returned non-finite factors for the text Gram")
    ev = ev.flip(0)
    U = U.flip(1)
    keep = ev > rel_tol * ev[0].clamp_min(torch.finfo(torch.float64).tiny)
    ev, U = ev[keep], U[:, keep]
    # Rows of B span the same space as the rows of T and are orthonormal.
    B = (U / ev.sqrt()).T @ Td
    return B, ev


def orthogonal_project(V: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Project the rows of ``V`` onto the row space of an orthonormal ``B``."""
    Vd = V.detach().to("cpu", torch.float64)
    return (Vd @ B.T) @ B


def tikhonov_reconstruct(V: torch.Tensor, T: torch.Tensor, eta: float = 0.75,
                         eps: float = 1e-8) -> tuple[torch.Tensor, int, float]:
    """Reconstruct rows of ``V`` from rows of ``T`` by ridge least squares.

    The regularised variant used by the token-pruning literature: each visual
    row is written as a coefficient vector over text rows, with the ridge
    parameter set from the text spectrum itself as the eigenvalue at which a
    fraction ``eta`` of the text energy has accumulated. Directions weaker than
    that are damped rather than inverted.

    Returns ``(reconstruction, k, lambda)`` where ``k`` is the number of text
    directions inside the ``eta`` energy shell.

    This is a *shrunk* projection, not a projection: it is not idempotent, and
    the explained fraction it produces is bounded above by the one from
    :func:`orthogonal_project` on the same subspace. The random-subspace null in
    this module describes the orthogonal case exactly and this case only as an
    upper bound, which is why both are measured.
    """
    Vd = V.detach().to("cpu", torch.float64)
    Td = T.detach().to("cpu", torch.float64)
    G = Td @ Td.T
    ev = torch.linalg.eigvalsh(G).flip(0).clamp_min(0.0)
    if not torch.isfinite(ev).all():
        raise FloatingPointError("non-finite text Gram spectrum")
    total = ev.sum()
    if total <= 0:
        return torch.zeros_like(Vd), 0, eps
    csum = torch.cumsum(ev, 0) / total
    k = int((csum >= eta).nonzero()[0].item()) + 1 if bool((csum >= eta).any()) else ev.numel()
    lam = float(ev[k - 1].item() + eps)
    n_t = G.shape[0]
    A = G + lam * torch.eye(n_t, dtype=torch.float64)
    # Solve rather than invert, and fall back to the pseudo-inverse if the
    # factorisation is not usable. cluster.local.md documents cholesky_ex
    # returning info == 0 beside a NaN factor on this hardware, so the factor is
    # checked rather than the status code.
    try:
        coef = torch.linalg.solve(A.T, (Vd @ Td.T).T).T
        if not torch.isfinite(coef).all():
            raise torch.linalg.LinAlgError("non-finite solve")
    except (torch.linalg.LinAlgError, RuntimeError):
        coef = (Vd @ Td.T) @ torch.linalg.pinv(A)
    return coef @ Td, k, lam


# --- the statistic itself ---------------------------------------------------


def explained_fraction(V: torch.Tensor, V_hat: torch.Tensor) -> float:
    """Fraction of the energy of ``V`` captured by the reconstruction ``V_hat``."""
    Vd = V.detach().to("cpu", torch.float64)
    Rd = Vd - V_hat.detach().to("cpu", torch.float64)
    total = float((Vd * Vd).sum())
    if total <= 0:
        return 0.0
    return 1.0 - float((Rd * Rd).sum()) / total


def residual_ratio(V: torch.Tensor, V_hat: torch.Tensor) -> torch.Tensor:
    """Per-row relative residual norm, ``||v - v_hat|| / ||v||``.

    The token-level score the pruning literature calls the cross-modal residual:
    low where a row is well explained by the text subspace, high where it is
    not. Rows of zero norm score 0, which keeps them out of a top-k selection
    rather than putting them at an undefined position in it.
    """
    Vd = V.detach().to("cpu", torch.float64)
    Rd = Vd - V_hat.detach().to("cpu", torch.float64)
    num = Rd.norm(dim=1)
    den = Vd.norm(dim=1)
    return torch.where(den > 0, num / den.clamp_min(1e-30), torch.zeros_like(num))


@dataclass
class AbsorptionStats:
    """Everything measured about one (layer, example) pair.

    ``frac_orth`` and ``frac_tik`` are the two explained fractions;
    ``z_orth`` places the first against the random-subspace null. ``sub_dim`` is
    the numerical rank of the text subspace, which is what the null is
    conditioned on and is usually smaller than the text token count.
    """

    layer: int
    hidden_dim: int
    n_visual: int
    n_text: int
    sub_dim: int
    erank_visual: float
    frac_orth: float
    frac_tik: float
    tik_k: int
    null_mean: float
    null_sd: float
    z_orth: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure(V: torch.Tensor, T: torch.Tensor, layer: int, eta: float = 0.75
            ) -> tuple[AbsorptionStats, torch.Tensor]:
    """Measure absorption of visual rows ``V`` by text rows ``T`` at one layer.

    Both matrices must already be centred, and centred *the same way*: the
    statistic is about the position of the visual cloud relative to the text
    subspace, so a different origin for each modality would make the two
    quantities incomparable. :func:`xmodal.controls.centre` does this.

    Returns the statistics and the per-token residual ratio from the
    regularised reconstruction, which is what token selection consumes.
    """
    B, _ = orthonormal_basis(T)
    V_orth = orthogonal_project(V, B)
    V_tik, k_tik, _ = tikhonov_reconstruct(V, T, eta=eta)
    erank = participation_rank(spectrum(V))
    D = V.shape[1]
    sub_dim = int(B.shape[0])
    frac = explained_fraction(V, V_orth)
    stats = AbsorptionStats(
        layer=layer,
        hidden_dim=D,
        n_visual=int(V.shape[0]),
        n_text=int(T.shape[0]),
        sub_dim=sub_dim,
        erank_visual=erank,
        frac_orth=frac,
        frac_tik=explained_fraction(V, V_tik),
        tik_k=k_tik,
        null_mean=null_mean(D, sub_dim),
        null_sd=null_sd(D, sub_dim, erank),
        z_orth=null_z(frac, D, sub_dim, erank),
    )
    return stats, residual_ratio(V, V_tik)
