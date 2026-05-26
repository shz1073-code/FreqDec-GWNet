"""Build a pooled 'cv_pool' split + stratified 5-fold assignment.

The 35 state-labelled bursts are physically split across clean_data/train
(29) and clean_data/val (6), and a few sequence NAMES collide across the
two (clean_data/train/images/cmu_seq_001 is a different burst from
clean_data/val/images/cmu_seq_001). To run sequence-level k-fold CV we
pool all 35 into one directory via symlinks, renamed '{origin}_{seq}' so
there are no collisions:

    {data_root}/cv_pool/images/{tr|va}_{seq}/        -> original frames
    {data_root}/cv_pool/labels/{tr|va}_{seq}/        -> original wire masks
    {sl_root}_cv/cv_pool/{tr|va}_{seq}.npz           -> original StateLabel

Folds are stratified by source dataset (cmu / zhu) so each fold has a
balanced patient mix. Writes, per fold k:
    reports/cv_folds_v1/fold{k}_val.txt    held-out sequences
    reports/cv_folds_v1/fold{k}_train.txt  the other four folds
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path("/workspace/shz/clean_data")
SL_ROOT = Path("/workspace/FreqDec-GWNet/reports/state_labels/clean_v2")
SL_CV = Path("/workspace/FreqDec-GWNet/reports/state_labels/clean_v2_cv")
FOLD_DIR = Path("/workspace/FreqDec-GWNet/reports/cv_folds_v1")
N_FOLDS = 5
SEED = 42


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def main() -> None:
    pool_img = DATA_ROOT / "cv_pool" / "images"
    pool_lbl = DATA_ROOT / "cv_pool" / "labels"
    pool_sl = SL_CV / "cv_pool"
    for d in (pool_img, pool_lbl, pool_sl, FOLD_DIR):
        d.mkdir(parents=True, exist_ok=True)

    pooled = []                                   # (new_name, dataset)
    for split, tag in (("train", "tr"), ("val", "va")):
        sl_dir = SL_ROOT / split
        for npz in sorted(sl_dir.glob("*.npz")):
            seq = npz.stem
            img_src = DATA_ROOT / split / "images" / seq
            lbl_src = DATA_ROOT / split / "labels" / seq
            if not img_src.is_dir():
                print(f"  [skip] no images for {split}/{seq}")
                continue
            new = f"{tag}_{seq}"
            _link(img_src, pool_img / new)
            if lbl_src.is_dir():
                _link(lbl_src, pool_lbl / new)
            else:
                print(f"  [warn] no wire-mask dir for {split}/{seq}")
            _link(npz, pool_sl / f"{new}.npz")
            dataset = "cmu" if seq.startswith("cmu") else "zhu"
            pooled.append((new, dataset))

    print(f"pooled {len(pooled)} sequences "
          f"({sum(d=='cmu' for _,d in pooled)} cmu / "
          f"{sum(d=='zhu' for _,d in pooled)} zhu)")

    # stratified fold assignment: round-robin within each dataset after a
    # deterministic shuffle
    import random
    folds = [[] for _ in range(N_FOLDS)]
    for dataset in ("cmu", "zhu"):
        names = sorted(n for n, d in pooled if d == dataset)
        random.Random(SEED).shuffle(names)
        for i, n in enumerate(names):
            folds[i % N_FOLDS].append(n)

    all_names = sorted(n for n, _ in pooled)
    for k in range(N_FOLDS):
        val = sorted(folds[k])
        train = sorted(set(all_names) - set(val))
        (FOLD_DIR / f"fold{k}_val.txt").write_text("\n".join(val) + "\n")
        (FOLD_DIR / f"fold{k}_train.txt").write_text("\n".join(train) + "\n")
        print(f"  fold{k}: val={len(val)}  train={len(train)}  "
              f"(val cmu={sum(n.split('_')[1]=='cmu' for n in val)})")
    print(f"fold files -> {FOLD_DIR}")


if __name__ == "__main__":
    main()
