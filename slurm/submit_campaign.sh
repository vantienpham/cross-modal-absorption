#!/usr/bin/env bash
# Submit a campaign, staying under the per-user job limits.
#
#   bash slurm/submit_campaign.sh absorption     # phase 1: measurement
#   bash slurm/submit_campaign.sh pruning        # phase 2: benchmark accuracy
#   bash slurm/submit_campaign.sh layers         # phase 3: pruning-layer sweep
#   bash slurm/submit_campaign.sh <name> --dry-run
#
# Run ON the login node, from <REMOTE_DIR>.
#
# cluster.local.md records MaxJobs=12 and MaxSubmit=40. Over the *submit* limit
# sbatch is rejected outright, so a campaign of sixty jobs cannot simply be fired
# at the queue; over the *job* limit the extra ones pend with
# AssocMaxJobsLimit, which is a cap rather than contention and clears only as
# your own jobs drain. This script therefore blocks until there is room, rather
# than submitting and hoping.

set -euo pipefail
cd "$(dirname "$0")/.."

# The usage string carries no braces on purpose: a '}' inside ${x:?...} closes
# the expansion, and the remainder becomes part of the variable's value.
CAMPAIGN="${1:?usage: $0 absorption|pruning|layers [--dry-run]}"
DRY=""
[[ "${2:-}" == "--dry-run" ]] && DRY=1

# Leave headroom under MaxSubmit=40 so an interactive srun still gets through.
MAX_QUEUED=28
POLL=60

# 7B in fp16 needs 40 GB and up; the 16 GB partitions would take the job and
# then kill it after the queue wait. cluster.local.md, "Which list to pass".
PARTS="gpu40G,gpu80G,prismgpup,gpul40s,gpuh100p,gpuh200p"

LLAVA15="llava-hf/llava-1.5-7b-hf"
LLAVANEXT="llava-hf/llava-v1.6-vicuna-7b-hf"
QWEN="Qwen/Qwen2.5-VL-7B-Instruct"

wait_for_room() {
  while true; do
    local n
    n=$(squeue -u "$USER" -h | wc -l)
    [[ "$n" -lt "$MAX_QUEUED" ]] && break
    echo "  queue at ${n}; waiting for room" >&2
    sleep "$POLL"
  done
}

submit() {
  local name="$1" time="$2"; shift 2
  if [[ -n "$DRY" ]]; then
    echo "sbatch --job-name=$name -p $PARTS --time=$time slurm/run.slurm $*"
    return
  fi
  wait_for_room
  sbatch --job-name="$name" -p "$PARTS" --time="$time" slurm/run.slurm "$@" \
    | sed "s/^/  ${name}: /"
}

# --- phase 1: absorption, its null and its controls --------------------------
# Forward passes only, no generation. Cheap, and it is the measurement the
# paper's central claim rests on.

absorption() {
  local n=200
  submit absorb-llava15-pope 0-02:00:00 scripts/measure_absorption.py \
    --model "$LLAVA15" --dataset lmms-lab/POPE --adapter pope --split test \
    --limit $n --run-dir out/runs/absorb-llava15-pope
  submit absorb-llava15-sqa 0-02:00:00 scripts/measure_absorption.py \
    --model "$LLAVA15" --dataset lmms-lab/ScienceQA --config ScienceQA-IMG \
    --adapter scienceqa --split test --limit $n \
    --run-dir out/runs/absorb-llava15-sqa
  submit absorb-llava15-textvqa 0-02:00:00 scripts/measure_absorption.py \
    --model "$LLAVA15" --dataset lmms-lab/textvqa --adapter textvqa \
    --split validation --limit $n --run-dir out/runs/absorb-llava15-textvqa

  # LLaVA-Next carries five times the visual tokens, so the same statistic is
  # measured at a very different sequence length; if absorption were an artefact
  # of having few visual tokens relative to the hidden size, it would move here.
  submit absorb-next-pope 0-04:00:00 scripts/measure_absorption.py \
    --model "$LLAVANEXT" --dataset lmms-lab/POPE --adapter pope --split test \
    --limit 150 --run-dir out/runs/absorb-next-pope
  submit absorb-next-textvqa 0-04:00:00 scripts/measure_absorption.py \
    --model "$LLAVANEXT" --dataset lmms-lab/textvqa --adapter textvqa \
    --split validation --limit 150 --run-dir out/runs/absorb-next-textvqa

  # A different family entirely, with its own visual encoder and prompt format.
  submit absorb-qwen25-pope 0-04:00:00 scripts/measure_absorption.py \
    --model "$QWEN" --dataset lmms-lab/POPE --adapter pope --split test \
    --limit 150 --prompt-style chat --run-dir out/runs/absorb-qwen25-pope
}

# --- phase 2: what the criteria are worth ------------------------------------
# Budgets are the retention rates the pruning literature reports for LLaVA-1.5's
# 576 visual tokens: 33.3%, 22.2% and 11.1%.

pruning() {
  local n=1000 layer=15
  local -a benches=(
    "pope|lmms-lab/POPE||pope|test"
    "sqa|lmms-lab/ScienceQA|ScienceQA-IMG|scienceqa|test"
    "textvqa|lmms-lab/textvqa||textvqa|validation"
  )
  for spec in "${benches[@]}"; do
    IFS='|' read -r tag ds cfg adapter split <<< "$spec"
    local cfgarg=()
    [[ -n "$cfg" ]] && cfgarg=(--config "$cfg")

    submit "prune-${tag}-none" 0-03:00:00 scripts/prune_eval.py \
      --model "$LLAVA15" --dataset "$ds" "${cfgarg[@]}" --adapter "$adapter" \
      --split "$split" --limit $n --criterion none --layer $layer \
      --run-dir "out/runs/prune-${tag}-none"

    for budget in 192 128 64; do
      for crit in random attention residual greedy leverage; do
        submit "prune-${tag}-${crit}-${budget}" 0-03:00:00 scripts/prune_eval.py \
          --model "$LLAVA15" --dataset "$ds" "${cfgarg[@]}" --adapter "$adapter" \
          --split "$split" --limit $n --criterion "$crit" --budget "$budget" \
          --layer $layer --run-dir "out/runs/prune-${tag}-${crit}-${budget}"
      done
    done
  done
}

# --- phase 3: where to prune -------------------------------------------------
# The layer is the one free choice the criteria share, and absorption is a
# function of depth, so the two interact. Swept on one benchmark at the tightest
# budget, where the criteria separate most.

# All five criteria, so the sweep answers "does the criterion matter at this
# depth" rather than only "which of two is better". Run on POPE and on TextVQA:
# POPE is binary with a strong prior and tolerates aggressive pruning, TextVQA
# needs to read the image and does not, so a criterion that matters anywhere
# should matter there.
layers() {
  for layer in 2 8 15 22; do
    for crit in random attention residual greedy leverage; do
      submit "layer-pope-${crit}-${layer}" 0-03:00:00 scripts/prune_eval.py \
        --model "$LLAVA15" --dataset lmms-lab/POPE --adapter pope --split test \
        --limit 1000 --criterion "$crit" --budget 64 --layer "$layer" \
        --run-dir "out/runs/layer-pope-${crit}-${layer}"
      submit "layer-tvqa-${crit}-${layer}" 0-03:00:00 scripts/prune_eval.py \
        --model "$LLAVA15" --dataset lmms-lab/textvqa --adapter textvqa \
        --split validation --limit 1000 --criterion "$crit" --budget 64 \
        --layer "$layer" --run-dir "out/runs/layer-tvqa-${crit}-${layer}"
    done
  done
}

case "$CAMPAIGN" in
  absorption) absorption ;;
  pruning)    pruning ;;
  layers)     layers ;;
  *) echo "unknown campaign $CAMPAIGN" >&2; exit 2 ;;
esac
echo "submitted: $CAMPAIGN"
