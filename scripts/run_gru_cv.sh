#!/usr/bin/env bash
# Step 4 — GRU state branch: 5-fold cross-validation, with vs without
# real-data augmentation.
#
# Why CV: the fixed 6-sequence val split is high-variance (2/6 already
# near-perfect, the split is arbitrary). 5-fold CV over the pooled 35
# sequences gives an honest generalization estimate where every sequence
# is held out exactly once, and every model trains on ~28 sequences
# instead of 29 — using all the data.
#
# Why the aug vs no-aug arm: it is the augmentation ablation the paper
# needs. Augmentation is burst-consistent (geometry + photometric applied
# identically across all T frames) so respiratory labels stay valid.
#
# 10 trainings (5 folds x 2 arms), 100 epochs each, ~22 min each
# => ~4 h wall-clock on one 4090. Idempotent: skips folds whose
# checkpoint already exists; --resume on retry.
#
# Usage:
#   nohup bash scripts/run_gru_cv.sh > /tmp/gru_cv.log 2>&1 &
#   tail -f /tmp/gru_cv.log

set -uo pipefail
cd "$(dirname "$0")/.."

# --- Stability fix (root cause of the day-long crash storm) -----------------
# Heavy imports (torch/numpy/scipy) crash nondeterministically (~30-50% of
# process starts) due to a BLAS/OpenMP thread-pool init race that corrupts
# process state — surfacing downstream as bogus CUDA asserts, segfaults and
# garbled Python errors. Pinning thread counts to 1 makes heavy imports
# 100% reliable (verified 10/10). GPU training is GPU-bound so the CPU-side
# single-threading costs ~nothing. Exported here so dataloader workers
# inherit it too.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CKPT_DIR=experiments/checkpoints
REPORTS=reports/gru_cv_v1
FOLDS=reports/cv_folds_v1
DATA_ROOT=/workspace/shz/clean_data
SL=reports/state_labels/clean_v2_cv
STAGE1=$CKPT_DIR/stage1_v1_best.pth
mkdir -p "$REPORTS"

log() { echo "[$(date +'%H:%M:%S')] $*"; }

train_eval() {
    local fold="$1"; local aug="$2"          # aug = "" or "aug"
    local augflag="" augname=""
    if [ "$aug" = "aug" ]; then augflag="--augment"; augname="_aug"; fi
    local prefix="stage2_gru_cv${augname}_fold${fold}_v1"
    local ckpt="$CKPT_DIR/${prefix}_best.pth"
    local done_marker="$REPORTS/${prefix}.done"

    if [ -f "$done_marker" ]; then
        log "SKIP train [$prefix] (already complete)"
    else
        # A '.done' marker is written ONLY on a clean rc==0 exit. A bare
        # checkpoint file is NOT proof of completion — a segfault mid-run
        # leaves a partial one. So: retry until clean exit, and --resume
        # whenever a partial checkpoint is present (intermittent CUDA /
        # cv2-worker crashes on this box recur ~every other fold).
        local ok=0
        for i in $(seq 1 10); do
            local resume=""
            [ -f "$CKPT_DIR/${prefix}_last.pth" ] && resume="--resume"
            log "[train $prefix] attempt $i/10 ${resume}"
            timeout 2h python scripts/train_stage2_state.py \
                --data-root "$DATA_ROOT" --state-labels-dir "$SL" \
                --cv-split cv_pool \
                --cv-train-file "$FOLDS/fold${fold}_train.txt" \
                --cv-val-file "$FOLDS/fold${fold}_val.txt" \
                --stage1-ckpt "$STAGE1" \
                --epochs 100 --batch-size 1 --T-window 64 \
                --num-workers 2 \
                --state-core-type gru --state-protocol chronological_carry \
                --lr 5e-4 $augflag $resume \
                --save-prefix "$prefix" >> "$REPORTS/log_${prefix}.txt" 2>&1
            local rc=$?
            if [ $rc -eq 0 ] && [ -f "$ckpt" ]; then
                log "[train $prefix] OK"; ok=1; touch "$done_marker"; break
            fi
            # The box's CUDA context wedges intermittently under sustained
            # load. Empirically it recovers after a few idle minutes but NOT
            # after 60s. Reap any lingering trainer and give the GPU a full
            # 5-minute cooldown before the next --resume.
            pkill -9 -f train_stage2_state 2>/dev/null
            log "[train $prefix] crashed rc=$rc; reaped; 5-min GPU cooldown"
            sleep 300
        done
        [ $ok -eq 1 ] || { log "FAIL [$prefix] after 10 attempts"; return 1; }
    fi
    [ -f "$ckpt" ] || { log "FAIL [$prefix] — no checkpoint"; return 1; }

    # evaluate this fold's model ONLY on its held-out sequences
    python scripts/evaluate_state_quality.py \
        --data-root "$DATA_ROOT" --state-labels-dir "$SL" \
        --ckpts "$ckpt" --state-core-types gru \
        --splits cv_pool \
        --include-sequences-file "$FOLDS/fold${fold}_val.txt" \
        --output-dir "$REPORTS/eval_${prefix}" \
        >> "$REPORTS/log_eval.txt" 2>&1 || log "WARN eval [$prefix]"
    log "[eval $prefix] done"
}

log "=== GRU 5-fold CV — arm A: no augmentation ==="
for k in 0 1 2 3 4; do train_eval "$k" ""; done

log "=== GRU 5-fold CV — arm B: with augmentation ==="
for k in 0 1 2 3 4; do train_eval "$k" "aug"; done

log "=== aggregate honest CV scoreboard ==="
python scripts/_cv_aggregate.py 2>&1 | tee "$REPORTS/scoreboard.txt"
log "GRU CV v1 DONE"
