#!/usr/bin/env bash
# Week-2 background pipeline — addresses v5's two weaknesses:
#
#   (a) state branch underperformed (phase MAE 80-90° on val).
#       Probable causes: too few epochs (Mamba best @ epoch 43/100),
#       too small T_window, or labels themselves noisy. We try the
#       cheapest fix first: train MUCH longer (300 epochs) plus a
#       larger window (T=96) so the SSM sees more cycles per window.
#
#   (b) headline numbers are single-seed.  JBHI reviewers will ask for
#       error bars.  We launch 2 additional seeds of the final
#       configuration so we can report mean ± std on (n=3).
#
# Wall-clock estimate on a single 4090: ~35-50 hours.  Safe for 1 week.
#
# Usage (inside docker, repo root):
#   nohup bash scripts/background_pipeline_v6.sh > pipeline_v6.log 2>&1 &
#   tail -f pipeline_v6.log
#
# All steps are idempotent + resumable. If pipeline crashes mid-step,
# just re-run: --resume reads the latest {prefix}_last.pth.

set -uo pipefail
cd "$(dirname "$0")/.."

CKPT_DIR=experiments/checkpoints
REPORTS=reports/paper_v6
mkdir -p "$REPORTS"

DATA_ROOT=/workspace/shz/clean_data
STATE_LABELS=/workspace/FreqDec-GWNet/reports/state_labels/clean_v2
STAGE1_CKPT=$CKPT_DIR/stage1_v1_best.pth

MAX_RETRIES=3
STEP_TIMEOUT_HOURS=36

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

attempt() {
    local desc="$1"; shift
    local check="$1"; shift
    local prefix="$1"; shift
    [ "$1" = "--" ] && shift

    if [ -f "$check" ]; then
        log "SKIP [$desc] — $check exists"
        return 0
    fi
    for i in $(seq 1 "$MAX_RETRIES"); do
        log "[$desc] attempt $i/$MAX_RETRIES"
        local resume_arg=""
        [ "$i" -gt 1 ] && resume_arg="--resume"
        timeout "${STEP_TIMEOUT_HOURS}h" "$@" $resume_arg \
            >> "$REPORTS/log_${prefix}.txt" 2>&1
        local rc=$?
        if [ $rc -eq 0 ] && [ -f "$check" ]; then
            log "[$desc] OK"
            return 0
        fi
        log "[$desc] failed (rc=$rc); sleep 60"
        sleep 60
    done
    log "[$desc] EXHAUSTED"
    return 1
}

# ========================================================================
# A. Long-train state branches (T=96, 300 epoch) — fix v5 weakness (a)
# ========================================================================
S2_M_LONG=$CKPT_DIR/stage2_mamba_long_v6_best.pth
attempt "stage2_mamba_long" "$S2_M_LONG" "s2_m_long" -- \
    python scripts/train_stage2_state.py \
        --data-root "$DATA_ROOT" --state-labels-dir "$STATE_LABELS" \
        --stage1-ckpt "$STAGE1_CKPT" \
        --epochs 300 --batch-size 1 --T-window 96 \
        --state-core-type mamba --state-protocol chronological_carry \
        --lr 3e-4 \
        --save-prefix stage2_mamba_long_v6

S2_G_LONG=$CKPT_DIR/stage2_gru_long_v6_best.pth
attempt "stage2_gru_long" "$S2_G_LONG" "s2_g_long" -- \
    python scripts/train_stage2_state.py \
        --data-root "$DATA_ROOT" --state-labels-dir "$STATE_LABELS" \
        --stage1-ckpt "$STAGE1_CKPT" \
        --epochs 300 --batch-size 1 --T-window 96 \
        --state-core-type gru --state-protocol chronological_carry \
        --lr 3e-4 \
        --save-prefix stage2_gru_long_v6

# ========================================================================
# B. State eval on long-trained ckpts (paper §IV.C update)
# ========================================================================
log "Step B: state-branch evaluation on long ckpts"
for tag in "mamba_long:mamba" "gru_long:gru"; do
    NAME=${tag%%:*}; CORE=${tag##*:}
    CKPT=$CKPT_DIR/stage2_${NAME}_v6_best.pth
    if [ ! -f "$CKPT" ]; then continue; fi
    for split in val test; do
        python scripts/evaluate_state.py \
            --data-root "$DATA_ROOT" --state-labels-dir "$STATE_LABELS" \
            --ckpt "$CKPT" --state-core-type "$CORE" \
            --split "$split" \
            --T-window 96 \
            --output-csv "$REPORTS/state_${NAME}_${split}.csv" \
            >> "$REPORTS/log_state_eval.txt" 2>&1 || log "  WARN state $NAME $split"
    done
done

# ========================================================================
# C. Final stage 3 with best state core + 3 seeds (multi-seed for §IV)
# ========================================================================
# Use best state branch we have (re-evaluate after long training).
# Two seed cohorts: stage3_main_seed{1,2,3} (relative + zm).
for seed in 1 2 3; do
    PREFIX="stage3_main_seed${seed}_v6"
    CKPT=$CKPT_DIR/${PREFIX}_best.pth
    attempt "stage3_seed${seed}" "$CKPT" "$PREFIX" -- \
        python scripts/train_stage3_joint.py \
            --data-root "$DATA_ROOT" --state-labels-dir "$STATE_LABELS" \
            --stage2-ckpt "$S2_M_LONG" \
            --epochs 80 --batch-size 1 --T-window 64 \
            --state-core-type mamba \
            --motion-field-type relative \
            --lambda-state 1.0 --lambda-motion 0.5 \
            --lambda-zero-mean 0.5 \
            --lambda-drift-cap 0.3 --drift-cap-ratio 1.5 \
            --lr 3e-4 \
            --save-prefix "$PREFIX"
done

# ========================================================================
# D. Evaluate every seed on val + test with reset=16 (the operating point)
# ========================================================================
log "Step D: multi-seed eval"
for seed in 1 2 3; do
    CKPT=$CKPT_DIR/stage3_main_seed${seed}_v6_best.pth
    if [ ! -f "$CKPT" ]; then continue; fi
    for split in val test; do
        for reset in 0 16 32; do
            tag="seed${seed}_${split}_reset${reset}"
            outdir="$REPORTS/eval_${tag}"
            reset_flag=""
            [ "$reset" -gt 0 ] && reset_flag="--reset-every $reset"
            python scripts/evaluate_drift.py \
                --dataset dense_v1 --split "$split" --all \
                --ckpt "$CKPT" --state-core-type mamba \
                --motion-field-type relative \
                $reset_flag --cycle-consistency \
                --output-dir "$outdir" \
                >> "$REPORTS/log_eval.txt" 2>&1 || log "  WARN $tag"
        done
    done
done

# ========================================================================
# E. Final paper-quality videos (seed 1, reset_every=16)
# ========================================================================
log "Step E: paper videos"
S3_S1=$CKPT_DIR/stage3_main_seed1_v6_best.pth
if [ -f "$S3_S1" ]; then
    python scripts/visualize_compensation.py \
        --dataset dense_v1 --split test \
        --seq cmu_seq_094 \
        --ckpt "$S3_S1" --state-core-type mamba \
        --motion-field-type relative \
        --format mp4 --dpi 200 \
        --output-dir "$REPORTS/viz_test" \
        >> "$REPORTS/log_viz.txt" 2>&1 || log "  WARN viz test"
fi

log "============================================="
log "PIPELINE v6 DONE"
log "  Stage 2 long-trained: $S2_M_LONG, $S2_G_LONG"
log "  Stage 3 multi-seed:   $CKPT_DIR/stage3_main_seed{1,2,3}_v6_best.pth"
log "  State eval:           $REPORTS/state_*.csv"
log "  Drift eval matrix:    $REPORTS/eval_seed{1,2,3}_{val,test}_reset{0,16,32}/"
log "  Final test video:     $REPORTS/viz_test/cmu_seq_094.mp4"
log "After you return:"
log "  1. python scripts/aggregate_drift_results.py --input-dir $REPORTS \\"
log "     → look at seed{1,2,3} consistency to estimate std"
log "  2. Compare state_*.csv vs v5 — has phase MAE improved?"
log "============================================="
