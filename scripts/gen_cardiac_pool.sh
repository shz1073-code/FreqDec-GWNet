#!/usr/bin/env bash
# Generate cardiac (0.8–2.0 Hz) Phase-B labels on clean + ori, then
# symlink them into the expanded_pool layout that matches the
# respiratory pool. Result: reports/state_labels/cardiac_pool/expanded_pool/
# with renamed npz files (cl_tr_..., or_te_..., etc.).

set -uo pipefail
cd "$(dirname "$0")/.."

# Single-thread (proven stability fix from earlier in the day)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

OUT=reports/state_labels/cardiac_v1
POOL=reports/state_labels/cardiac_pool/expanded_pool
mkdir -p "$POOL" "$OUT"
CKPT=experiments/checkpoints/stage1_v1_best.pth

log() { echo "[$(date +'%H:%M:%S')] $*"; }

gen_split() {
    local ds="$1"; local split="$2"
    log "$ds/$split (cardiac 0.8-2.0 Hz)"
    python scripts/generate_state_labels.py \
        --dataset "$ds" --split "$split" --all --min-frames 32 \
        --band-low-hz 0.8 --band-high-hz 2.0 \
        --output-dir "$OUT/$ds" \
        --seg-checkpoint "$CKPT" \
        >> "$OUT/log_${ds}_${split}.txt" 2>&1
    local rc=$?
    local n=$(ls "$OUT/$ds/$split" 2>/dev/null | wc -l)
    log "$ds/$split done (rc=$rc) -> $n labels"
}

# Source-data label generation
for ds in clean ori; do
    for split in train val test; do
        gen_split "$ds" "$split"
    done
done

# Pool symlinks with the same naming as respiratory pool
# Format: <ds-prefix>_<split-prefix>_<seqname>.npz
log "symlinking into expanded_pool layout"
pref() {
    case "$1" in
        clean) echo "cl";;
        ori)   echo "or";;
    esac
}
spref() {
    case "$1" in
        train) echo "tr";;
        val)   echo "va";;
        test)  echo "te";;
    esac
}
n_total=0
for ds in clean ori; do
    for split in train val test; do
        src_dir="$OUT/$ds/$split"
        [ -d "$src_dir" ] || continue
        for f in "$src_dir"/*.npz; do
            [ -f "$f" ] || continue
            seq=$(basename "$f" .npz)
            link_name="$(pref $ds)_$(spref $split)_${seq}.npz"
            ln -sf "$(realpath "$f")" "$POOL/$link_name"
            n_total=$((n_total + 1))
        done
    done
done
log "DONE. cardiac pool -> $POOL  ($n_total npz)"
