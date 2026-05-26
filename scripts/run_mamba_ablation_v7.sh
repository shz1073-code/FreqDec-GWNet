#!/usr/bin/env bash
# Step B — Targeted Mamba ablation on synthetic-augmented data (paper §IV.E).
#
# Hypothesis: Mamba's failure on the 27-seq real dataset was data scarcity.
# With ~290 synthetic train sequences (perfect labels, varied amplitudes/
# frequencies), Mamba should be able to learn the respiratory state
# mapping. We sweep (lr × d_state × epochs × detach) to find the regime
# in which Mamba converges, then fine-tune on real for generalization.
#
# Naming:
#   stage2_mamba_aug_lr{lr}_d{d_state}_ep{epochs}_det{detach}
#
# Wall-clock estimate on a single 4090:
#   8 cells × ~90 min = ~12 hours; with --resume + skip-done idempotent.
#
# Usage:
#   nohup bash scripts/run_mamba_ablation_v7.sh > /tmp/mamba_ablation_v7.log 2>&1 &
#   tail -f /tmp/mamba_ablation_v7.log

set -uo pipefail
cd "$(dirname "$0")/.."

CKPT_DIR=experiments/checkpoints
REPORTS=reports/mamba_ablation_v7
mkdir -p "$REPORTS"

DATA_ROOT=/workspace/shz/synthetic_aug_v1
STATE_LABELS=/workspace/FreqDec-GWNet/reports/state_labels/synthetic_aug_v1
REAL_DATA=/workspace/shz/clean_data
REAL_STATE_LABELS=/workspace/FreqDec-GWNet/reports/state_labels/clean_v2
STAGE1_CKPT=$CKPT_DIR/stage1_v1_best.pth

log() { echo "[$(date +'%H:%M:%S')] $*"; }

attempt() {
    local desc="$1"; shift
    local check="$1"; shift
    if [ -f "$check" ]; then log "SKIP [$desc]"; return 0; fi
    for i in 1 2 3; do
        log "[$desc] attempt $i/3"
        local resume_arg=""
        [ "$i" -gt 1 ] && resume_arg="--resume"
        timeout 6h "$@" $resume_arg >> "$REPORTS/log_${desc}.txt" 2>&1
        local rc=$?
        if [ $rc -eq 0 ] && [ -f "$check" ]; then
            log "[$desc] OK"
            return 0
        fi
        log "[$desc] failed rc=$rc; sleep 30s"
        sleep 30
    done
    log "[$desc] EXHAUSTED"
    return 1
}

train_cell() {
    local lr="$1"
    local d_state="$2"
    local epochs="$3"
    local prefix="stage2_mamba_aug_lr${lr/./p}_d${d_state}_ep${epochs}"
    local ckpt="$CKPT_DIR/${prefix}_best.pth"
    attempt "$prefix" "$ckpt" -- \
        python scripts/train_stage2_state.py \
            --data-root "$DATA_ROOT" --state-labels-dir "$STATE_LABELS" \
            --stage1-ckpt "$STAGE1_CKPT" \
            --epochs "$epochs" --batch-size 1 --T-window 64 \
            --state-core-type mamba --state-protocol chronological_carry \
            --state-d-state "$d_state" \
            --lr "$lr" \
            --save-prefix "$prefix"
}

# ===================================================================
# Phase 1 — coarse sweep: 6 cells, find which lr/d_state actually learns
# ===================================================================
log "Phase 1: coarse Mamba sweep"
train_cell "2e-4" "16" "30"   # baseline retry with more data
train_cell "5e-5" "16" "30"   # smaller LR (Mamba often prefers this)
train_cell "2e-4" "32" "30"   # larger d_state
train_cell "2e-4" "64" "30"   # even larger d_state
train_cell "5e-5" "32" "30"   # both knobs
train_cell "1e-4" "32" "50"   # longer training

# ===================================================================
# Phase 2 — evaluate every cell on REAL val (the cell that wins)
# ===================================================================
log "Phase 2: evaluate each cell on real val"
for ckpt in "$CKPT_DIR"/stage2_mamba_aug_*_best.pth; do
    [ -f "$ckpt" ] || continue
    name=$(basename "$ckpt" .pth)
    python scripts/evaluate_state_quality.py \
        --data-root "$REAL_DATA" \
        --state-labels-dir "$REAL_STATE_LABELS" \
        --ckpts "$ckpt" \
        --splits train val \
        --output-dir "$REPORTS/eval_${name}" \
        >> "$REPORTS/log_eval.txt" 2>&1 || log "  WARN eval $name"
done

# ===================================================================
# Phase 3 — aggregate scoreboard
# ===================================================================
log "Phase 3: aggregate scoreboard"
python -c "
import csv, glob, math
from pathlib import Path
from collections import defaultdict
root = Path('$REPORTS')
rows = []
for csv_path in sorted(root.glob('eval_*/state_quality_summary.csv')):
    cell = csv_path.parent.name.replace('eval_', '')
    for r in csv.DictReader(open(csv_path)):
        r['cell'] = cell
        rows.append(r)
print(f'cells found: {len({r[chr(34)+chr(99)+chr(101)+chr(108)+chr(108)+chr(34)]: 1 for r in rows}.keys()) if rows else 0}')
print()
print(f'{\"cell\":50s} {\"split\":6s} {\"phase_MAE_deg\":>14s} {\"Pearson\":>10s}')
for r in sorted(rows, key=lambda x: (x['cell'], x['split'])):
    try:
        mae = float(r['phase_MAE_deg_mean'])
        rho = float(r['phase_pearson_mean'])
        print(f'{r[chr(34)+chr(99)+chr(101)+chr(108)+chr(108)+chr(34)]:50s} {r[chr(34)+chr(115)+chr(112)+chr(108)+chr(105)+chr(116)+chr(34)]:6s} {mae:>14.2f} {rho:>+10.3f}')
    except (KeyError, ValueError):
        pass
" >> "$REPORTS/scoreboard.txt" 2>&1
cat "$REPORTS/scoreboard.txt"

log "MAMBA ABLATION v7 DONE"
log "Best cell: read scoreboard, pick lowest val phase_MAE_deg"
log "Then run Step C with that ckpt to train stage 3 + paper metrics"
