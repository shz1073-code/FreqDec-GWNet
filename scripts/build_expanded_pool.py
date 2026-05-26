"""Build the expanded training pool from clean + ori datasets.

Mirrors :file:`build_cv_pool.py` but uses ALL sequences whose Phase-B
state labels were just regenerated under ``reports/state_labels/expanded_v1/``
with ``--min-frames 32``. The pool unifies four origin axes:

    cl_tr_*  cl_va_*  cl_te_*   from clean_data {train, val, test}
    or_tr_*  or_va_*  or_te_*   from ori_data   {train, val, test}

After this script, the dataset reads:

    /workspace/shz/clean_data/expanded_pool/images/{new_name}/
    /workspace/shz/clean_data/expanded_pool/labels/{new_name}/
    reports/state_labels/clean_v2_expanded/expanded_pool/{new_name}.npz

Stratified k-fold (k=5) is written to ``reports/cv_folds_expanded/``.
"""
from __future__ import annotations

import os
from pathlib import Path

# clean lives here
CLEAN_ROOT = Path("/workspace/shz/clean_data")
# ori lives here
ORI_ROOT = Path("/workspace/shz/ori_data")

SL_ROOT = Path("/workspace/FreqDec-GWNet/reports/state_labels/expanded_v1")
POOL_DATA_ROOT = CLEAN_ROOT                          # symlinks go here
POOL_SL_ROOT = Path("/workspace/FreqDec-GWNet/reports/state_labels/clean_v2_expanded")
FOLD_DIR = Path("/workspace/FreqDec-GWNet/reports/cv_folds_expanded")
N_FOLDS = 5
SEED = 42

KINDS = (
    ("clean", CLEAN_ROOT, "cl"),
    ("ori",   ORI_ROOT,   "or"),
)
SPLIT_TAGS = (("train", "tr"), ("val", "va"), ("test", "te"))


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def main() -> None:
    pool_img = POOL_DATA_ROOT / "expanded_pool" / "images"
    pool_lbl = POOL_DATA_ROOT / "expanded_pool" / "labels"
    pool_sl = POOL_SL_ROOT / "expanded_pool"
    for d in (pool_img, pool_lbl, pool_sl, FOLD_DIR):
        d.mkdir(parents=True, exist_ok=True)

    pooled = []   # (new_name, dataset_origin: cmu/zhu)
    for kind, root, ktag in KINDS:
        for split, stag in SPLIT_TAGS:
            sl_dir = SL_ROOT / kind / split
            if not sl_dir.is_dir():
                continue
            for npz in sorted(sl_dir.glob("*.npz")):
                seq = npz.stem
                img_src = root / split / "images" / seq
                lbl_src = root / split / "labels" / seq
                if not img_src.is_dir():
                    print(f"  [skip] no images for {kind}/{split}/{seq}")
                    continue
                new = f"{ktag}_{stag}_{seq}"
                _link(img_src, pool_img / new)
                if lbl_src.is_dir():
                    _link(lbl_src, pool_lbl / new)
                else:
                    print(f"  [warn] no wire-mask dir for {kind}/{split}/{seq}")
                _link(npz, pool_sl / f"{new}.npz")
                ds = "cmu" if seq.startswith("cmu") else (
                    "zhu" if seq.startswith("zhu") else "other"
                )
                pooled.append((new, ds))

    cmu_n = sum(d == "cmu" for _, d in pooled)
    zhu_n = sum(d == "zhu" for _, d in pooled)
    oth_n = sum(d == "other" for _, d in pooled)
    print(f"pooled {len(pooled)} sequences  (cmu={cmu_n} / zhu={zhu_n} / other={oth_n})")

    # 5-fold stratified by patient family
    import random
    folds = [[] for _ in range(N_FOLDS)]
    for dataset in ("cmu", "zhu", "other"):
        names = sorted(n for n, d in pooled if d == dataset)
        if not names:
            continue
        random.Random(SEED).shuffle(names)
        for i, n in enumerate(names):
            folds[i % N_FOLDS].append(n)

    all_names = sorted(n for n, _ in pooled)
    for k in range(N_FOLDS):
        val = sorted(folds[k])
        train = sorted(set(all_names) - set(val))
        (FOLD_DIR / f"fold{k}_val.txt").write_text("\n".join(val) + "\n")
        (FOLD_DIR / f"fold{k}_train.txt").write_text("\n".join(train) + "\n")
        print(f"  fold{k}: val={len(val)}  train={len(train)}")

    # also write all-N.txt for final-model training (uses every sequence)
    (FOLD_DIR / "all_n.txt").write_text("\n".join(all_names) + "\n")
    print(f"\nfold files + all_n.txt -> {FOLD_DIR}")


if __name__ == "__main__":
    main()
