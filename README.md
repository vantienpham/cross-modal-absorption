# Cross-modal absorption, calibrated

Code and measured results for *Calibrating cross-modal absorption]{Calibrating cross-modal absorption in vision-language models: a null model and its controls*.

Recent visual-token pruning work uses a geometric statistic as a signal: the
fraction of visual-token energy lying in the subspace spanned by the text
tokens, which rises with depth and is read as text absorbing visual content.
This repository contains the code that calibrates that statistic against an
exact null, the controls that test whether it is cross-modal at all, and the
evaluation of six token-selection criteria.

## What is here

```
src/xmodal/
  absorption.py   the statistic, its exact null distribution, the projections
  controls.py     matched / mismatched / swapped-image / same-modality controls
  select.py       the six selection criteria
  prune.py        the decoder surgery, and the identity test that validates it
  pipeline.py     selection inputs, and generation from a pruned model
  states.py       per-layer visual and text hidden states
  models.py modality.py calib.py evaluate.py runlog.py
                  model loading, the modality mask, data adapters, metrics

scripts/
  measure_absorption.py   the statistic and its controls, layer by layer
  prune_eval.py           benchmark accuracy under pruning
  selection_quality.py    reconstruction error, without decoding
  collect_results.py      aggregation and the paired bootstrap
  make_figures.py         figures, from results/tables alone

results/tables/           every number reported in the paper, as JSON
slurm/                    job scripts for a Slurm cluster
tests/                    the test suite, including the null-distribution checks
```

`results/tables/` is the important directory for anyone checking the paper: each
JSON there is the summary a single run produced, and every figure and table in
the manuscript is derived from those files by `scripts/make_figures.py` and
`scripts/collect_results.py`.

## Two things worth reading the code for

**The null distribution** (`src/xmodal/absorption.py`). For a uniformly random
subspace of dimension *k* in *D* dimensions, the explained fraction has mean
exactly *k/D* for any matrix, and a closed-form variance depending on the matrix
only through the participation rank of its spectrum. Both moments are checked
against Monte Carlo in `tests/test_absorption.py`, so the implementation is
verified rather than asserted.

**The pruning surgery** (`src/xmodal/prune.py`). Dropping tokens mid-stack means
narrowing an attention mask, a rotary embedding and a cache position along axes
that are easy to confuse, and every wrong version still emits fluent text. The
check that licenses the accuracy numbers is an identity: at a budget equal to the
visual token count, the pruned path must reproduce the unpruned generation token
for token, for every criterion. `tests/test_prune.py` asserts exactly that.

## Running it

```bash
uv sync
uv run python -m pytest tests/ -q          # 27 pass; 1 GPU test skipped
```

The GPU identity test needs a real checkpoint and a warm cache, so it requires an
explicit opt-in and is skipped otherwise:

```bash
XMODAL_GPU_TESTS=1 uv run python -m pytest tests/test_prune.py -q
```

Measurements, on one GPU:

```bash
# the statistic, its null and its controls; forward passes only
uv run python scripts/measure_absorption.py \
  --model llava-hf/llava-1.5-7b-hf --dataset lmms-lab/POPE --adapter pope \
  --split test --limit 200 --swap-image --run-dir out/runs/absorb-pope

# benchmark accuracy under pruning, one criterion per run
uv run python scripts/prune_eval.py \
  --dataset lmms-lab/POPE --adapter pope --split test --limit 1000 \
  --criterion pivot --budget 64 --layer 15 --run-dir out/runs/prune-pope-pivot-64

# reconstruction error for every criterion, no decoding needed
uv run python scripts/selection_quality.py --limit 200 --run-dir out/runs/selq-pope
```

`slurm/submit_campaign.sh` runs the three campaigns behind the paper and
self-throttles against a per-user job limit. Set the placeholders in
`slurm/sync.sh` first; `cluster.example.md` lists what a cluster-specific
configuration needs to record.

## A note on the numerics

Every eigendecomposition, projection and reconstruction runs in float64 on the
CPU. The matrices are small enough that this costs little, and it avoids two
failure modes seen on older GPUs: a cuSOLVER SVD that fails to converge on
ill-conditioned input even after the library's internal fallback, and a Cholesky
that returns a success code alongside a factor containing NaN. Outputs are
checked for finiteness rather than status codes.

## Attribution

The statistic studied here was proposed by Ou, Song, Zhou, Sun, Zhang and Luo,
*When Vision Becomes Text: Visual Token Pruning via Cross-Modal Residual
Guidance in VLMs* (arXiv:2608.10489). Their selection rule is reimplemented in
`src/xmodal/select.py` as the `greedy` criterion, for comparison under a common
harness; no code or measurement of theirs is included or reproduced here.

Parts of `models.py`, `modality.py`, `calib.py`, `evaluate.py` and `runlog.py`
are adapted from an earlier project by the same author.

## Licence

MIT. See [LICENSE](LICENSE).
