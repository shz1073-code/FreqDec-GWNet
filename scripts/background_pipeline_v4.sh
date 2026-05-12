#!/usr/bin/env bash
# A-lite + B+ background pipeline (paper §IV).
# Run once before leaving the computer; produces all paper figures/tables.
#
# Estimated wall-clock on a single 4090: ~45-50 hours.
#
# Usage (from FreqDec-GWNet repo root, inside the docker container):
#   nohup bash scripts/background_pipeline_v4.sh > pipeline_v4.log 2>&1 &
#   tail -f pipeline_v4.log
#
# Each step is idempotent — skips if {save_prefix}_best.pth already exists.

set -uo pipefail
cd "$(dirname "$0")/.."

CKPT_DIR=experiments/checkpoints
REPORTS=reports/paper_v4
mkdir -p "$REPORTS"

DATA_ROOT=/workspace/shz/clean_data
STATE_LABELS=/workspace/FreqDec-GWNet/reports/state_labels/clean_v2
STAGE1_CKPT=$CKPT_DIR/stage1_v1_best.pth

log() {
    echo "[$(date +'%H:%M:%S')] $*"
}
skip_done() {
    if [ -f "$1" ]; then log "SKIP $1"; return 0; fi
    return 1
}

# ------------------------------------------------------------ Stage 2
STAGE2=$CKPT_DIR/stage2_mamba_chrono_v4_best.pth
if ! skip_done "$STAGE2"; then
    log "Step 1: Stage 2 Mamba 100 epoch"
    python scripts/train_stage2_state.py \
        --data-root "$DATA_ROOT" --state-labels-dir "$STATE_LABELS" \
        --stage1-ckpt "$STAGE1_CKPT" \
        --epochs 100 --batch-size 1 --T-window 64 \
        --state-core-type mamba \
        --state-protocol chronological_carry \
        --lr 5e-4 \
        --save-prefix stage2_mamba_chrono_v4 || log "WARN stage2"
fi

# ------------------------------------------------------------ Stage 3 main
S3_MAIN=$CKPT_DIR/stage3_main_v4_best.pth
if ! skip_done "$S3_MAIN"; then
    log "Step 2: Stage 3 main_v4 (relative + zero-mean)"
    python scripts/train_stage3_joint.py \
        --data-root "$DATA_ROOT" --state-labels-dir "$STATE_LABELS" \
        --stage2-ckpt "$STAGE2" \
        --epochs 80 --batch-size 1 --T-window 64 \
        --state-core-type mamba \
        --motion-field-type relative --motion-state-dim 3 --motion-input-mode delta \
        --lambda-state 1.0 --lambda-motion 0.5 \
        --lambda-zero-mean 0.5 \
        --lambda-drift-cap 0.3 --drift-cap-ratio 1.5 \
        --lr 3e-4 \
        --save-prefix stage3_main_v4 || log "WARN main_v4"
fi

# ------------------------------------------------------------ Stage 3 absolute
S3_ABS=$CKPT_DIR/stage3_abs_v3_best.pth
if ! skip_done "$S3_ABS"; then
    log "Step 3: Stage 3 abs_v3 (absolute baseline)"
    python scripts/train_stage3_joint.py \
        --data-root "$DATA_ROOT" --state-labels-dir "$STATE_LABELS" \
        --stage2-ckpt "$STAGE2" \
        --epochs 80 --batch-size 1 --T-window 64 \
        --state-core-type mamba \
        --motion-field-type absolute \
        --lambda-state 1.0 --lambda-motion 0.5 \
        --lambda-zero-mean 0.5 \
        --lambda-drift-cap 0.3 --drift-cap-ratio 1.5 \
        --lr 3e-4 \
        --save-prefix stage3_abs_v3 || log "WARN abs_v3"
fi

# ------------------------------------------------------------ Stage 3 no_zm
S3_NOZM=$CKPT_DIR/stage3_no_zm_v1_best.pth
if ! skip_done "$S3_NOZM"; then
    log "Step 4: Stage 3 no_zm_v1 (ablation, no regularizer)"
    python scripts/train_stage3_joint.py \
        --data-root "$DATA_ROOT" --state-labels-dir "$STATE_LABELS" \
        --stage2-ckpt "$STAGE2" \
        --epochs 80 --batch-size 1 --T-window 64 \
        --state-core-type mamba \
        --motion-field-type relative \
        --lambda-state 1.0 --lambda-motion 0.5 \
        --lambda-zero-mean 0.0 \
        --lambda-drift-cap 0.0 \
        --lr 3e-4 \
        --save-prefix stage3_no_zm_v1 || log "WARN no_zm"
fi

# ------------------------------------------------------------ Eval
log "Step 5: Evaluation (full + reset_every=32)"
for variant in "main_v4:relative" "abs_v3:absolute" "no_zm_v1:relative"; do
    NAME=${variant%%:*}
    FIELD=${variant##*:}
    CKPT=$CKPT_DIR/stage3_${NAME}_best.pth
    if [ ! -f "$CKPT" ]; then log "skip $NAME (no ckpt)"; continue; fi
    python scripts/evaluate_drift.py \
        --dataset dense_v1 --split train --all \
        --ckpt "$CKPT" --state-core-type mamba \
        --motion-field-type "$FIELD" --cycle-consistency \
        --output-dir "$REPORTS/drift_${NAME}_full" || log "WARN drift $NAME full"
    python scripts/evaluate_drift.py \
        --dataset dense_v1 --split train --all \
        --ckpt "$CKPT" --state-core-type mamba \
        --motion-field-type "$FIELD" --reset-every 32 --cycle-consistency \
        --output-dir "$REPORTS/drift_${NAME}_reset32" || log "WARN drift $NAME reset"
done

# ------------------------------------------------------------ Inter-burst
if [ -f "$S3_MAIN" ]; then
    log "Step 6: Inter-burst continuity"
    python scripts/evaluate_inter_burst.py \
        --dataset clean --split train \
        --seq cmu_seq_002 cmu_seq_005 zhu_seq_015 \
        --ckpt "$S3_MAIN" --state-core-type mamba \
        --K 3 --gap-frames 30 \
        --output-dir "$REPORTS/inter_burst_v4" || log "WARN inter-burst"
fi

# ------------------------------------------------------------ Videos
log "Step 7: Compensation videos"
for variant in "main_v4:relative" "abs_v3:absolute" "no_zm_v1:relative"; do
    NAME=${variant%%:*}
    FIELD=${variant##*:}
    CKPT=$CKPT_DIR/stage3_${NAME}_best.pth
    if [ ! -f "$CKPT" ]; then continue; fi
    python scripts/visualize_compensation.py \
        --dataset dense_v1 --split train \
        --seq cmu_seq_007 cmu_seq_015 cmu_seq_017 \
        --ckpt "$CKPT" --state-core-type mamba \
        --motion-field-type "$FIELD" \
        --format mp4 --dpi 200 \
        --output-dir "$REPORTS/viz_${NAME}" || log "WARN viz $NAME"
done

log "PIPELINE DONE."
log "Read $REPORTS/drift_main_v4_reset32/*.csv for the headline numbers."
log "If main_v4 beats abs_v3 on Metric A reset32 -> direction A salvaged."
log "If not -> use B+ framing with inter-burst + zero-mean ablation as positive endpoint."
