#!/usr/bin/env bash
# Master ablation runner — sweeps every cell of the paper §8 ablation tables.
#
# Usage:
#   bash scripts/run_ablation_tables.sh <eval_split> <output_root>
#
# Example:
#   bash scripts/run_ablation_tables.sh train reports/ablation_v1
#
# Assumes:
#   * Stage 1 / Stage 2 / Stage 3 checkpoints already exist under
#     experiments/checkpoints/ following the names from the hand-off list.
#   * dense_v1 split is the evaluation target (where tip landmarks live).
#
# Notes:
#   * This is the *evaluation* sweep (no training). For training-time
#     ablations (state protocol, GRU vs Mamba, motion-input mode), train
#     each variant separately, save its ckpt, then add it to the list
#     below by overriding the env vars at the top.
#   * Each cell writes per-sequence CSVs under
#     $OUTPUT_ROOT/$run_name/{seq}.csv so the aggregator can pivot them.

set -euo pipefail

EVAL_SPLIT="${1:-train}"
OUTPUT_ROOT="${2:-reports/ablation_v1}"
DATASET="${DATASET:-dense_v1}"

mkdir -p "$OUTPUT_ROOT"
echo "[ablation] eval_split=$EVAL_SPLIT  output_root=$OUTPUT_ROOT  dataset=$DATASET"

# Common eval-side defaults (override via env vars if needed)
WIDTH_MULT="${WIDTH_MULT:-1.0}"
STATE_CORE="${STATE_CORE:-mamba}"

# Stage-3 main paper checkpoint
MAIN_CKPT="${MAIN_CKPT:-experiments/checkpoints/stage3_main_v2_best.pth}"

# Sibling ckpts — each ablation row that needs its own training run
# points at a separate ckpt path. Override empty rows when training done.
GRU_CKPT="${GRU_CKPT:-experiments/checkpoints/stage2_gru_chrono_v2_best.pth}"
MAMBA_CKPT="${MAMBA_CKPT:-experiments/checkpoints/stage2_mamba_chrono_v2_best.pth}"
ABS_CKPT="${ABS_CKPT:-experiments/checkpoints/stage3_abs_v1_best.pth}"
PROT_RANDOM_CKPT="${PROT_RANDOM_CKPT:-experiments/checkpoints/stage2_random_init_best.pth}"
PROT_CONTEXT_CKPT="${PROT_CONTEXT_CKPT:-experiments/checkpoints/stage2_context_prefix_best.pth}"
DET_NO_PREV_CKPT="${DET_NO_PREV_CKPT:-experiments/checkpoints/stage3_no_detach_prev_best.pth}"
NINE_D_CKPT="${NINE_D_CKPT:-experiments/checkpoints/stage3_9d_best.pth}"
FOUR_D_CKPT="${FOUR_D_CKPT:-experiments/checkpoints/stage3_4d_best.pth}"

# Helper to run one evaluation row with the given ckpt + extra args
run_drift() {
    local run_name="$1"; shift
    local ckpt="$1"; shift
    local out_dir="$OUTPUT_ROOT/$run_name"
    if [ ! -f "$ckpt" ]; then
        echo "[skip] $run_name  (ckpt missing: $ckpt)"
        return 0
    fi
    echo "[run]  $run_name  (ckpt: $ckpt)"
    python scripts/evaluate_drift.py \
        --dataset "$DATASET" --split "$EVAL_SPLIT" --all \
        --ckpt "$ckpt" \
        --width-mult "$WIDTH_MULT" \
        --state-core-type "$STATE_CORE" \
        --cycle-consistency \
        --output-dir "$out_dir" \
        "$@" || echo "[fail] $run_name"
}

# ========================================================================
# Sec 8.1 Main ablations — 5 rows
# ========================================================================

# #1 absolute vs relative
run_drift "main_relative"           "$MAIN_CKPT"  --motion-field-type relative
run_drift "main_absolute"           "$ABS_CKPT"   --motion-field-type absolute

# #2 state training protocol — needs sibling training runs
run_drift "main_proto_chrono"       "$MAIN_CKPT"
run_drift "main_proto_random_init"  "$PROT_RANDOM_CKPT"
run_drift "main_proto_context"      "$PROT_CONTEXT_CKPT"

# #3 motion loss detach policy — needs a sibling training run
run_drift "main_detach_default"     "$MAIN_CKPT"
run_drift "main_detach_no_prev"     "$DET_NO_PREV_CKPT"

# #4 reference selection — same ckpt, different runtime strategy
run_drift "main_ref_auto"           "$MAIN_CKPT" --reference-strategy auto
run_drift "main_ref_first"          "$MAIN_CKPT" --reference-strategy first_frame
run_drift "main_ref_random"         "$MAIN_CKPT" --reference-strategy random --reference-random-seed 0

# #5 GRU vs Mamba — stage-2 only (no motion field), evaluated with
# zero-init R3 to isolate the state-branch contribution.
run_drift "main_state_gru"          "$GRU_CKPT"   --state-core-type gru
run_drift "main_state_mamba"        "$MAMBA_CKPT" --state-core-type mamba

# ========================================================================
# Sec 8.2 Appendix ablations
# ========================================================================

# #A1 9D / 6D / 4D motion input
run_drift "appendix_motion_6d"      "$MAIN_CKPT" --motion-state-dim 3 --motion-input-mode delta
run_drift "appendix_motion_9d"      "$NINE_D_CKPT" --motion-state-dim 3 --motion-input-mode with_prev
run_drift "appendix_motion_4d"      "$FOUR_D_CKPT" --motion-state-dim 2 --motion-input-mode delta

# ========================================================================
# Aggregate
# ========================================================================

echo ""
echo "[aggregate] consolidating CSVs..."
python scripts/aggregate_drift_results.py \
    --input-dir "$OUTPUT_ROOT" \
    --output-csv "$OUTPUT_ROOT/summary.csv" \
    --pivot-csv "$OUTPUT_ROOT/pivot_improvement.csv" \
    --pivot-metric improvement_mean
python scripts/aggregate_drift_results.py \
    --input-dir "$OUTPUT_ROOT" \
    --output-csv "$OUTPUT_ROOT/summary.csv" \
    --pivot-csv "$OUTPUT_ROOT/pivot_drift_max.csv" \
    --pivot-metric drift_max

echo ""
echo "ablation sweep complete. See $OUTPUT_ROOT/"
echo "  long-form table:   $OUTPUT_ROOT/summary.csv"
echo "  improvement pivot: $OUTPUT_ROOT/pivot_improvement.csv"
echo "  drift_max pivot:   $OUTPUT_ROOT/pivot_drift_max.csv"
