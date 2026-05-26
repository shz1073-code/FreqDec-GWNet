#!/usr/bin/env bash
# Step 3 — strengthen the GRU state branch with physiological priors.
#
# Diagnosis (state_quality_gru_diag_v1): the GRU state branch overfits
# the 29 real training sequences — train Pearson 0.78 vs val Pearson 0.23.
# Label QC ruled out bad Phase-B labels (worst val seqs have anchorQ 0.99).
# It is a genuine generalization failure: too-large hypothesis space.
#
# Fix under test: the §III.E physiological priors (phase smoothness,
# amplitude smoothness, spectral concentration) are *unsupervised*
# regularizers that shrink the hypothesis space. Baseline for comparison
# is stage2_gru_chrono_v5_best (no priors), val Pearson 0.175.
#
# 3 cells, fair vs v5 (100 ep, lr 5e-4, GRU, chronological_carry, T=64):
#   light  — gentle regularization
#   med    — balanced
#   spec   — spectral-concentration heavy (band-limited prior dominant)
#
# Usage:
#   nohup bash scripts/run_gru_prior_sweep.sh > /tmp/gru_prior_sweep.log 2>&1 &
#   tail -f /tmp/gru_prior_sweep.log

set -uo pipefail
cd "$(dirname "$0")/.."

CKPT_DIR=experiments/checkpoints
REPORTS=reports/gru_prior_sweep_v1
mkdir -p "$REPORTS"

DATA_ROOT=/workspace/shz/clean_data
STATE_LABELS=/workspace/FreqDec-GWNet/reports/state_labels/clean_v2
STAGE1_CKPT=$CKPT_DIR/stage1_v1_best.pth

log() { echo "[$(date +'%H:%M:%S')] $*"; }

attempt() {
    local desc="$1"; shift
    local check="$1"; shift
    [ "${1:-}" = "--" ] && shift          # consume readability separator
    if [ -f "$check" ]; then log "SKIP [$desc]"; return 0; fi
    for i in 1 2 3; do
        log "[$desc] attempt $i/3"
        local resume_arg=""
        [ "$i" -gt 1 ] && resume_arg="--resume"
        timeout 4h "$@" $resume_arg >> "$REPORTS/log_${desc}.txt" 2>&1
        local rc=$?
        if [ $rc -eq 0 ] && [ -f "$check" ]; then log "[$desc] OK"; return 0; fi
        log "[$desc] failed rc=$rc; sleep 20s"
        sleep 20
    done
    log "[$desc] EXHAUSTED"
    return 1
}

train_cell() {
    local name="$1"; local lph="$2"; local lamp="$3"; local lspec="$4"
    local prefix="stage2_gru_prior_${name}_v1"
    attempt "$prefix" "$CKPT_DIR/${prefix}_best.pth" -- \
        python scripts/train_stage2_state.py \
            --data-root "$DATA_ROOT" --state-labels-dir "$STATE_LABELS" \
            --stage1-ckpt "$STAGE1_CKPT" \
            --epochs 100 --batch-size 1 --T-window 64 \
            --state-core-type gru --state-protocol chronological_carry \
            --lr 5e-4 \
            --lambda-phase-smooth "$lph" \
            --lambda-amp-smooth "$lamp" \
            --lambda-spectral "$lspec" \
            --save-prefix "$prefix"
}

# ===================================================================
log "Step 3: GRU physiological-prior sweep"
#          name   phase  amp    spectral
train_cell light  0.10   0.05   0.10
train_cell med    0.30   0.10   0.30
train_cell spec   0.10   0.05   0.50

# ===================================================================
log "Step 3b: evaluate all prior cells + v5 baseline on real val/train"
EVAL_CKPTS=()
for name in light med spec; do
    ck="$CKPT_DIR/stage2_gru_prior_${name}_v1_best.pth"
    [ -f "$ck" ] && EVAL_CKPTS+=("$ck")
done
EVAL_CKPTS+=("$CKPT_DIR/stage2_gru_chrono_v5_best.pth")

python scripts/evaluate_state_quality.py \
    --data-root "$DATA_ROOT" \
    --state-labels-dir "$STATE_LABELS" \
    --ckpts "${EVAL_CKPTS[@]}" \
    --state-core-types $(printf 'gru %.0s' $(seq ${#EVAL_CKPTS[@]})) \
    --splits train val \
    --output-dir "$REPORTS/state_quality" \
    >> "$REPORTS/log_eval.txt" 2>&1 || log "WARN eval failed"

log "=== SCOREBOARD ==="
cat "$REPORTS/state_quality/state_quality_summary.csv" 2>/dev/null
log "GRU PRIOR SWEEP v1 DONE"
