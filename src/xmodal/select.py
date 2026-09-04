"""Which visual tokens to keep.

Every criterion here takes the same inputs at a pruning layer, so they differ
only in how they rank: the text-residual matrix ``R`` (visual rows with the part
the text subspace explains removed), a per-token attention mass ``a`` from the
post-image text queries, and a budget ``K``. All return sorted keep-indices.

The criteria fall into two groups.

*Pointwise* rankings score each token on its own and take the top ``K``:
:func:`select_random`, :func:`select_attention`, :func:`select_residual_norm`.
They are cheap and they all share one failure: a set of tokens each individually
informative can be collectively redundant, so the budget gets spent several
times on the same content.

*Set-aware* rankings account for what has already been kept.
:func:`select_greedy_diversity` is the pipeline the pruning literature converges
on: a product score, then a greedy pass that discounts remaining candidates by
their residual-direction overlap with each pick. It works, and it costs a
``N_v x N_v`` similarity matrix and a sequential loop.

The last two come from the perturbation argument rather than from intuition. The
quantity that controls the damage from dropping a set is how badly the dropped
rows are reconstructed from the kept ones, which makes selection a
row-subset-selection problem rather than a ranking problem, and attention enters
as a *row weighting* of the residual matrix rather than as a separate factor
multiplied in afterwards: the error a dropped row contributes is its
reconstruction residual scaled by the attention mass that would have flowed
through it.

:func:`select_leverage` scores rows by ridge leverage, the standard sampling
distribution for that problem, with the ridge parameter fixed by requiring the
effective dimension to equal the budget (:func:`ridge_for_budget`) rather than
tuned. It is the obvious way to turn the argument into a criterion and it does
not work well here, for a reason worth keeping in view: taking the top ``K`` is
not the sampling those results license, and attention mass is concentrated enough
that the weighted matrix is dominated by a few rows, so the scores collapse onto
them and the criterion drifts toward attention-only selection.

:func:`select_pivoted` is the deterministic procedure that actually descends on
the objective, taking at each step the row whose residual against the
already-chosen span is largest. It minimises the weighted reconstruction error
best of everything here, which is what it is built to do. Both are kept, and both
are reported.
"""

from __future__ import annotations

import torch

CRITERIA = ("random", "attention", "residual", "greedy", "leverage", "pivot")


def _as64(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu", torch.float64)


def _topk(scores: torch.Tensor, budget: int) -> torch.Tensor:
    k = min(int(budget), int(scores.numel()))
    return torch.topk(scores, k).indices.sort().values


# --- pointwise --------------------------------------------------------------


def select_random(n: int, budget: int, seed: int = 0) -> torch.Tensor:
    """Uniform sample without replacement.

    The floor every other criterion has to clear. It is a stronger baseline than
    it sounds: visual tokens are highly redundant, so a uniform sample is
    unbiased coverage of the image, and criteria that concentrate on one region
    can fall below it.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(n, generator=g)[: min(budget, n)].sort().values


def select_attention(attn: torch.Tensor, budget: int) -> torch.Tensor:
    """Top ``K`` by attention mass received from post-image text queries."""
    return _topk(_as64(attn), budget)


def select_residual_norm(residual_ratio: torch.Tensor, budget: int) -> torch.Tensor:
    """Top ``K`` by relative residual against the text subspace.

    Keeps what text explains least. Reported on its own because the literature's
    claim is that this signal carries information attention does not, and that
    claim is separable from the pipeline built on top of it.
    """
    return _topk(_as64(residual_ratio), budget)


# --- set-aware --------------------------------------------------------------


def select_greedy_diversity(R: torch.Tensor, attn: torch.Tensor,
                            residual_ratio: torch.Tensor, budget: int) -> torch.Tensor:
    """Product score, then greedy selection discounting residual-direction overlap.

    Score starts at ``(a_i * cmr_i)^2``; each pick multiplies every remaining
    candidate by ``1 - cos^2`` of its residual direction against the pick. This
    is the published pipeline, reimplemented here so that it runs under the same
    harness, on the same samples, at the same budgets as everything else. It is
    a baseline in this project, not a contribution of it.

    Cost is the ``N_v x N_v`` cosine matrix plus a sequential loop of length
    ``K``, against one eigendecomposition for :func:`select_leverage`.
    """
    Rd = _as64(R)
    a = _as64(attn).clamp_min(0)
    cmr = _as64(residual_ratio).clamp_min(0)
    n = Rd.shape[0]
    k = min(int(budget), n)

    norms = Rd.norm(dim=1, keepdim=True).clamp_min(1e-30)
    U = Rd / norms
    score = (a * cmr) ** 2

    keep: list[int] = []
    alive = score.clone()
    for _ in range(k):
        i = int(torch.argmax(alive).item())
        keep.append(i)
        alive[i] = -1.0
        g = (U @ U[i]) ** 2          # squared cosine to the pick, in [0, 1]
        live = alive > 0
        alive[live] = alive[live] * (1.0 - g[live])
    return torch.tensor(sorted(keep), dtype=torch.long)


def gram_eigh(R: torch.Tensor, rank: int | None = None, seed: int = 0
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigendecomposition of ``R R^T``: eigenvalues descending, eigenvectors.

    Exact for matrices small enough that ``eigh`` is cheap. Above that it uses a
    randomised range finder with one power iteration, which is what makes the
    high-resolution models affordable: an exact float64 ``eigh`` of a 2880-square
    Gram at every pruning layer of every example dominates the whole evaluation,
    and only the top of the spectrum is used.

    float64 on the CPU throughout, deliberately. ``cluster.local.md`` records
    both cuSOLVER failures this avoids, one of which returns a success code
    beside a factor full of NaN.
    """
    Rd = _as64(R)
    n = Rd.shape[0]
    if rank is None or rank >= n or n <= 1024:
        G = Rd @ Rd.T
        ev, U = torch.linalg.eigh(G)
        if not torch.isfinite(ev).all() or not torch.isfinite(U).all():
            raise FloatingPointError("eigh returned non-finite factors for the residual Gram")
        return ev.flip(0).clamp_min(0.0), U.flip(1)

    r = min(int(rank), n)
    g = torch.Generator(device="cpu").manual_seed(seed)
    Om = torch.randn(n, r, generator=g, dtype=torch.float64)
    Y = Rd @ (Rd.T @ Om)             # one power iteration: (R R^T) Om
    Q, _ = torch.linalg.qr(Y)
    B = Q.T @ Rd
    ev, W = torch.linalg.eigh(B @ B.T)
    if not torch.isfinite(ev).all() or not torch.isfinite(W).all():
        raise FloatingPointError("randomised range finder produced non-finite factors")
    return ev.flip(0).clamp_min(0.0), (Q @ W).flip(1)


def ridge_for_budget(eigenvalues: torch.Tensor, budget: int) -> float:
    """Ridge parameter whose effective dimension equals the budget.

    The effective dimension ``sum_j s_j / (s_j + lam)`` falls from ``rank`` at
    ``lam -> 0`` to 0 as ``lam`` grows, so there is exactly one ``lam`` at which
    it equals ``K``; bisection finds it. Fixing ``lam`` this way is what keeps
    the criterion free of a tuned constant: the question "which rows matter" is
    being asked at the resolution the budget can actually represent, and the
    answer moves with the budget instead of being pinned to a value chosen at
    one operating point.

    Returns 0 when the matrix has rank at or below the budget, where no
    regularisation is needed because every direction can be kept.
    """
    s = _as64(eigenvalues).clamp_min(0.0)
    s = s[s > 0]
    if s.numel() == 0 or budget >= s.numel():
        return 0.0
    lo, hi = 1e-18, float(s[0].item()) * s.numel() + 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if float((s / (s + mid)).sum()) > budget:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def ridge_leverage(R: torch.Tensor, budget: int, weights: torch.Tensor | None = None,
                   rank: int | None = 512, seed: int = 0) -> torch.Tensor:
    """Ridge leverage scores of the rows of ``R``, optionally row-weighted.

    With ``R = U S^{1/2} W^T``, the score of row ``i`` is
    ``sum_j (s_j / (s_j + lam)) u_ij^2``. It measures how much of the matrix's
    row space that row is uniquely responsible for: a row lying in a direction
    many other rows also cover shares its score with them, which is exactly the
    redundancy penalty the greedy pipeline installs by hand.

    ``weights`` scale each row before the decomposition, so a row that carries
    little attention mass is judged as contributing proportionally less of the
    structure worth preserving. The returned scores are for the weighted matrix.
    """
    Rd = _as64(R)
    if weights is not None:
        w = _as64(weights).clamp_min(0.0)
        Rd = Rd * w.sqrt().unsqueeze(1)
    ev, U = gram_eigh(Rd, rank=rank, seed=seed)
    lam = ridge_for_budget(ev, budget)
    filt = ev / (ev + lam) if lam > 0 else (ev > 0).double()
    return ((U * U) * filt.unsqueeze(0)).sum(dim=1)


def select_leverage(R: torch.Tensor, attn: torch.Tensor, budget: int,
                    rank: int | None = 512, seed: int = 0) -> torch.Tensor:
    """Top ``K`` rows of the attention-weighted text residual by ridge leverage.

    Kept because it is the obvious way to turn the leverage argument into a
    criterion, and because it does not work well: taking the top ``K`` is not
    what the sampling theory licenses, and when the weights are concentrated,
    as attention mass is, the weighted matrix is dominated by a few rows and the
    scores collapse onto them. :func:`select_pivoted` is the deterministic
    procedure that actually optimises the objective; the comparison between the
    two is reported rather than hidden.
    """
    tau = ridge_leverage(R, budget, weights=attn, rank=rank, seed=seed)
    return _topk(tau, budget)


def select_pivoted(R: torch.Tensor, attn: torch.Tensor, budget: int,
                   tol: float = 1e-12) -> torch.Tensor:
    """Greedy row subset selection on the attention-weighted residual.

    Pivoted Cholesky of the weighted Gram, which is greedy column subset
    selection written in the form that costs ``O(K N^2)`` instead of
    ``O(K N d)``. At each step it takes the row whose residual against the
    already-chosen span is largest, which is the row that reduces
    ``||W^{1/2}(R - Pi_S R)||_F^2`` most, so it descends directly on the
    quantity the perturbation bound names rather than on a proxy for it.

    Scaling a row does not change the subspace it spans, so weighting before the
    decomposition changes which rows are worth taking without changing what
    taking them buys. Selection stops early if the remaining residual falls
    below ``tol``, which happens when the retained rows already span the matrix;
    the budget is then filled by the largest remaining residuals, so the returned
    set always has the requested size.
    """
    Rd = _as64(R)
    a = _as64(attn).clamp_min(0.0)
    B = Rd * a.sqrt().unsqueeze(1)
    n = B.shape[0]
    k = min(int(budget), n)

    G = B @ B.T
    d = torch.diagonal(G).clone()
    L = torch.zeros(n, k, dtype=torch.float64)
    chosen: list[int] = []
    taken = torch.zeros(n, dtype=torch.bool)

    for step in range(k):
        d_masked = d.masked_fill(taken, float("-inf"))
        i = int(torch.argmax(d_masked).item())
        if not torch.isfinite(d_masked[i]) or float(d_masked[i]) <= tol:
            break
        chosen.append(i)
        taken[i] = True
        col = (G[:, i] - L[:, :step] @ L[i, :step]) / abs(float(d[i])) ** 0.5
        L[:, step] = col
        d = (d - col * col).clamp_min(0.0)

    if len(chosen) < k:
        # The span is already complete. Fill deterministically by residual, then
        # by index, so the result does not depend on tie-breaking order.
        rest = (~taken).nonzero().squeeze(-1)
        extra = rest[torch.argsort(d[rest], descending=True)][: k - len(chosen)]
        chosen.extend(int(x) for x in extra)
    return torch.tensor(sorted(chosen), dtype=torch.long)


# --- dispatch ---------------------------------------------------------------


def select(criterion: str, *, R: torch.Tensor, attn: torch.Tensor,
           residual_ratio: torch.Tensor, budget: int, seed: int = 0,
           rank: int | None = 512) -> torch.Tensor:
    """Apply one named criterion. Names are :data:`CRITERIA`."""
    if criterion == "random":
        return select_random(int(R.shape[0]), budget, seed=seed)
    if criterion == "attention":
        return select_attention(attn, budget)
    if criterion == "residual":
        return select_residual_norm(residual_ratio, budget)
    if criterion == "greedy":
        return select_greedy_diversity(R, attn, residual_ratio, budget)
    if criterion == "leverage":
        return select_leverage(R, attn, budget, rank=rank, seed=seed)
    if criterion == "pivot":
        return select_pivoted(R, attn, budget)
    raise ValueError(f"unknown criterion {criterion!r}; expected one of {CRITERIA}")


def reconstruction_error(R: torch.Tensor, keep: torch.Tensor) -> float:
    """Relative energy of ``R`` not reachable from the span of the kept rows.

    The quantity the perturbation argument bounds the pruning damage by, and the
    one every criterion here is implicitly competing on. Measuring it directly
    separates "this criterion picks a good subset" from "this criterion happens
    to help this benchmark", and it needs no generation to evaluate.
    """
    Rd = _as64(R)
    total = float((Rd * Rd).sum())
    if total <= 0:
        return 0.0
    S = Rd[keep.to(torch.long)]
    # Orthonormalise the kept rows, then measure what the rest projects onto.
    Q, _ = torch.linalg.qr(S.T)
    resid = Rd - (Rd @ Q) @ Q.T
    return float((resid * resid).sum()) / total
