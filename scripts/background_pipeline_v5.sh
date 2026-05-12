#!/usr/bin/env bash
# Background pipeline v5 — B+ framing per GPT's 5 guardrails.
#
# Differences vs v4:
#   1. Full 2x2 ablation matrix (zm vs no_zm × reset vs no_reset).
#   2. Stage 2 GRU baseline trained too (for §IV.E Mamba-vs-GRU table).
#   3. Auto-retry: each step retries up to 3 times with --resume.
#   4. Watchdog: per-step wall-clock limit, kill+retry if hung.
#   5. reset_every choice tuned on dense_v1/val, reported once on test.
#   6. Multi-sequence inter-burst test (not just one case study).
#   7. Phase MAE + amp RMSE on validation (state-estimation positive endpoint).
#
# Wall-clock estimate on a single 4090: ~60-80 hours.
# Safe to run for up to a week — every step is idempotent and resumable.
#
# Usage (inside the docker container, from FreqDec-GWNet repo root):
#   nohup bash scripts/background_pipeline_v5.sh > pipeline_v5.log 2>&1 &
#   tail -f pipeline_v5.log

set -uo pipefail
cd "$(dirname "$0")/.."

CKPT_DIR=experiments/checkpoints
REPORTS=reports/paper_v5
mkdir -p "$REPORTS" "$CKPT_DIR"

DATA_ROOT=/workspace/shz/clean_data
STATE_LABELS=/workspace/FreqDec-GWNet/reports/state_labels/clean_v2
STAGE1_CKPT=$CKPT_DIR/stage1_v1_best.pth

MAX_RETRIES=3                  # how many times to retry a failed step
STEP_TIMEOUT_HOURS=24          # kill any single step that hangs > 24h

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

# Retry wrapper: invokes the command, monitors timeout, retries with --resume.
# Args: <description> <expected_ckpt_path> <prefix> -- <command args...>
attempt() {
    local desc="$1"; shift
    local check="$1"; shift
    local prefix="$1"; shift
    [ "$1" = "--" ] && shift

    # If already done, skip.
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
        log "[$desc] failed (rc=$rc); waiting 60s before retry"
        sleep 60
    done
    log "[$desc] EXHAUSTED — leaving placeholder, continuing pipeline"
    return 1
}

# ========================================================================
# Stage 2 — both Mamba and GRU at 100 epoch (so §IV.E Mamba-vs-GRU is fair)
# ========================================================================
STAGE2_MAMBA=$CKPT_DIR/stage2_mamba_chrono_v5_best.pth
attempt "stage2_mamba" "$STAGE2_MAMBA" "stage2_mamba_v5" -- \
    python scripts/train_stage2_state.py \
        --data-root "$DATA_ROOT" --state-labels-dir "$STATE_LABELS" \
        --stage1-ckpt "$STAGE1_CKPT" \
        --epochs 100 --batch-size 1 --T-window 64 \
        --state-core-type mamba --state-protocol chronological_carry \
        --lr 5e-4 --save-prefix stage2_mamba_chrono_v5

STAGE2_GRU=$CKPT_DIR/stage2_gru_chrono_v5_best.pth
attempt "stage2_gru" "$STAGE2_GRU" "stage2_gru_v5" -- \
    python scripts/train_stage2_state.py \
        --data-root "$DATA_ROOT" --state-labels-dir "$STATE_LABELS" \
        --stage1-ckpt "$STAGE1_CKPT" \
        --epochs 100 --batch-size 1 --T-window 64 \
        --state-core-type gru --state-protocol chronological_carry \
        --lr 5e-4 --save-prefix stage2_gru_chrono_v5

# ========================================================================
# Stage 3 — full 2x2 ablation training matrix (Mamba state branch)
# Eval-time reset is orthogonal so only 2 trainings needed for the
# 4-cell matrix:
#       zm + no_reset:     zm trained, evaluate WITHOUT reset
#       zm + reset:        zm trained, evaluate WITH reset
#       no_zm + no_reset:  no_zm trained, evaluate WITHOUT reset
#       no_zm + reset:     no_zm trained, evaluate WITH reset
# Plus the absolute baseline trained with the same zm config.
# ========================================================================
S3_REL_ZM=$CKPT_DIR/stage3_rel_zm_v5_best.pth
attempt "stage3_rel_zm" "$S3_REL_ZM" "stage3_rel_zm_v5" -- \
    python scripts/train_stage3_joint.py \
        --data-root "$DATA_ROOT" --state-labels-dir "$STATE_LABELS" \
        --stage2-ckpt "$STAGE2_MAMBA" \
        --epochs 80 --batch-size 1 --T-window 64 \
        --state-core-type mamba \
        --motion-field-type relative \
        --motion-state-dim 3 --motion-input-mode delta \
        --lambda-state 1.0 --lambda-motion 0.5 \
        --lambda-zero-mean 0.5 \
        --lambda-drift-cap 0.3 --drift-cap-ratio 1.5 \
        --lr 3e-4 --save-prefix stage3_rel_zm_v5

S3_REL_NOZM=$CKPT_DIR/stage3_rel_nozm_v5_best.pth
attempt "stage3_rel_nozm" "$S3_REL_NOZM" "stage3_rel_nozm_v5" -- \
    python scripts/train_stage3_joint.py \
        --data-root "$DATA_ROOT" --state-labels-dir "$STATE_LABELS" \
        --stage2-ckpt "$STAGE2_MAMBA" \
        --epochs 80 --batch-size 1 --T-window 64 \
        --state-core-type mamba \
        --motion-field-type relative \
        --lambda-state 1.0 --lambda-motion 0.5 \
        --lambda-zero-mean 0.0 --lambda-drift-cap 0.0 \
        --lr 3e-4 --save-prefix stage3_rel_nozm_v5

S3_ABS_ZM=$CKPT_DIR/stage3_abs_zm_v5_best.pth
attempt "stage3_abs_zm" "$S3_ABS_ZM" "stage3_abs_zm_v5" -- \
    python scripts/train_stage3_joint.py \
        --data-root "$DATA_ROOT" --state-labels-dir "$STATE_LABELS" \
        --stage2-ckpt "$STAGE2_MAMBA" \
        --epochs 80 --batch-size 1 --T-window 64 \
        --state-core-type mamba \
        --motion-field-type absolute \
        --lambda-state 1.0 --lambda-motion 0.5 \
        --lambda-zero-mean 0.5 \
        --lambda-drift-cap 0.3 --drift-cap-ratio 1.5 \
        --lr 3e-4 --save-prefix stage3_abs_zm_v5

# ========================================================================
# Evaluation — pre-registered reset_every choices:
#     reset_every=16  ≈ 1 s @ 15 fps   (sub-cycle)
#     reset_every=32  ≈ 2 s @ 15 fps   (~1 respiratory cycle)
#     reset_every= 0  full-burst (cumulative drift naturally exposed)
# Run all three on val to PICK the operating point; report on test ONCE.
# ========================================================================
log "Step: Evaluation matrix (Metric A + B, reset sweep)"
for ckpt_tag in \
    "rel_zm:relative" "rel_nozm:relative" "abs_zm:absolute"; do
    NAME=${ckpt_tag%%:*}
    FIELD=${ckpt_tag##*:}
    CKPT=$CKPT_DIR/stage3_${NAME}_v5_best.pth
    if [ ! -f "$CKPT" ]; then
        log "  skip $NAME — ckpt missing"; continue
    fi
    for split in train val test; do
        for reset in 0 16 32; do
            tag="${NAME}_${split}_reset${reset}"
            outdir="$REPORTS/eval_${tag}"
            if [ -d "$outdir" ] && [ -f "$outdir/cmu_seq_007.csv" -o \
                                   -f "$outdir/cmu_seq_084.csv" -o \
                                   -f "$outdir/cmu_seq_094.csv" ]; then
                log "  skip $tag (already evaluated)"; continue
            fi
            reset_flag=""
            [ "$reset" -gt 0 ] && reset_flag="--reset-every $reset"
            log "  eval $tag"
            python scripts/evaluate_drift.py \
                --dataset dense_v1 --split "$split" --all \
                --ckpt "$CKPT" --state-core-type mamba \
                --motion-field-type "$FIELD" \
                $reset_flag --cycle-consistency \
                --output-dir "$outdir" \
                >> "$REPORTS/log_eval.txt" 2>&1 || \
                log "  WARN — eval $tag returned non-zero"
        done
    done
done

# ========================================================================
# Inter-burst continuity — multi-sequence, Mamba vs GRU vs no-carry
# This is now the second positive endpoint (§IV.E).
# ========================================================================
log "Step: Inter-burst continuity (Mamba vs GRU on long sequences)"
INTER_SEQS="cmu_seq_002 cmu_seq_005 zhu_seq_015"
if [ -f "$STAGE2_MAMBA" ]; then
    python scripts/evaluate_inter_burst.py \
        --dataset clean --split train --seq $INTER_SEQS \
        --ckpt "$STAGE2_MAMBA" --state-core-type mamba \
        --K 3 --gap-frames 30 \
        --output-dir "$REPORTS/inter_burst_mamba_v5" \
        >> "$REPORTS/log_inter_burst.txt" 2>&1 || log "  WARN inter-burst mamba"
fi
if [ -f "$STAGE2_GRU" ]; then
    python scripts/evaluate_inter_burst.py \
        --dataset clean --split train --seq $INTER_SEQS \
        --ckpt "$STAGE2_GRU" --state-core-type gru \
        --K 3 --gap-frames 30 \
        --output-dir "$REPORTS/inter_burst_gru_v5" \
        >> "$REPORTS/log_inter_burst.txt" 2>&1 || log "  WARN inter-burst gru"
fi

# ========================================================================
# Compensation videos for each variant — all using reset_every=16 for
# the deployable bounded-horizon look the paper claims.
# ========================================================================
log "Step: Compensation videos"
for ckpt_tag in "rel_zm:relative" "rel_nozm:relative" "abs_zm:absolute"; do
    NAME=${ckpt_tag%%:*}
    FIELD=${ckpt_tag##*:}
    CKPT=$CKPT_DIR/stage3_${NAME}_v5_best.pth
    if [ ! -f "$CKPT" ]; then continue; fi
    python scripts/visualize_compensation.py \
        --dataset dense_v1 --split train \
        --seq cmu_seq_007 cmu_seq_015 cmu_seq_017 \
        --ckpt "$CKPT" --state-core-type mamba \
        --motion-field-type "$FIELD" \
        --format mp4 --dpi 200 \
        --output-dir "$REPORTS/viz_${NAME}_v5" \
        >> "$REPORTS/log_viz.txt" 2>&1 || log "  WARN viz $NAME"
done

log "============================================="
log "PIPELINE v5 DONE"
log "  Checkpoints: $CKPT_DIR/stage{2,3}_*_v5_*"
log "  Eval matrix: $REPORTS/eval_*/"
log "  Inter-burst: $REPORTS/inter_burst_{mamba,gru}_v5/"
log "  Videos     : $REPORTS/viz_*/"
log ""
log "Next step for you:"
log "  1. python scripts/aggregate_drift_results.py --input-dir $REPORTS/ \\"
log "                                                --output-csv $REPORTS/summary.csv"
log "  2. Open summary.csv to see the full 3x3 grid (run × split × reset)."
log "  3. Pick best reset on dense_v1/val rows; report dense_v1/test row once."
log "============================================="
