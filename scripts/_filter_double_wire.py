"""Scan expanded_pool sequences for double-wire contamination.

The `ori` source is documented as "may contain double-wire" in data_paths.yaml.
We detect contamination by counting large connected components in each wire
mask: a single-wire frame has 1 connected component (sometimes 2 if the wire
crosses itself or tip detaches briefly); a double-wire frame consistently
has >=2 large components.

A sequence is flagged as contaminated if >= CONTAM_FRAC of its sampled frames
show >=2 components each with area >= MIN_COMPONENT_AREA px.

Writes:
  reports/cv_folds_expanded_clean/all_clean.txt   (uncontaminated seqs)
  reports/cv_folds_expanded_clean/all_dirty.txt   (flagged seqs)
  reports/cv_folds_expanded_clean/fold{k}_{train,val}.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POOL_LABELS = Path("/workspace/shz/clean_data/expanded_pool/labels")
ALL_N = ROOT / "reports/cv_folds_expanded/all_n.txt"
OUT_DIR = ROOT / "reports/cv_folds_expanded_clean"

MIN_COMPONENT_AREA = 50            # px
CONTAM_FRAC = 0.30                 # fraction of frames with >=2 large CCs to flag
N_SAMPLE = 12                      # frames sampled per sequence (evenly spaced)


def count_large_ccs(mask_path: Path) -> int:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return 0
    m = (m > 50).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA)


def main():
    seqs = [s.strip() for s in ALL_N.read_text().splitlines() if s.strip()]
    print(f"scanning {len(seqs)} pool sequences for double-wire contamination")
    clean, dirty = [], []
    summary_rows = []
    for seq in seqs:
        seq_dir = POOL_LABELS / seq
        if not seq_dir.is_dir():
            print(f"  WARN no labels dir for {seq}; keeping (treated as clean)")
            clean.append(seq)
            continue
        files = sorted(seq_dir.glob("*.png"))
        if not files:
            clean.append(seq); continue
        step = max(1, len(files) // N_SAMPLE)
        sample = files[::step][:N_SAMPLE]
        cc_per_frame = [count_large_ccs(f) for f in sample]
        n_multi = sum(1 for c in cc_per_frame if c >= 2)
        frac_multi = n_multi / max(1, len(cc_per_frame))
        avg_cc = float(np.mean(cc_per_frame)) if cc_per_frame else 0.0
        is_dirty = frac_multi >= CONTAM_FRAC
        (dirty if is_dirty else clean).append(seq)
        summary_rows.append((seq, len(files), avg_cc, frac_multi, "DIRTY" if is_dirty else "clean"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "all_clean.txt").write_text("\n".join(clean) + "\n")
    (OUT_DIR / "all_dirty.txt").write_text("\n".join(dirty) + "\n")

    # Build 5-fold splits (stratified-ish by dataset prefix cmu/zhu)
    rng = np.random.default_rng(42)
    cmu = sorted([s for s in clean if "_cmu_" in s])
    zhu = sorted([s for s in clean if "_zhu_" in s])
    rng.shuffle(cmu); rng.shuffle(zhu)
    folds = [[] for _ in range(5)]
    for i, s in enumerate(cmu): folds[i % 5].append(s)
    for i, s in enumerate(zhu): folds[i % 5].append(s)
    for k in range(5):
        val = sorted(folds[k])
        train = sorted([s for s in clean if s not in val])
        (OUT_DIR / f"fold{k}_train.txt").write_text("\n".join(train) + "\n")
        (OUT_DIR / f"fold{k}_val.txt").write_text("\n".join(val) + "\n")

    print()
    print(f"{'seq':22s} {'frames':>7s} {'avg_cc':>7s} {'frac_multi':>11s}  status")
    print("-" * 64)
    for r in sorted(summary_rows, key=lambda x: (-x[3], -x[2])):
        if r[4] == "DIRTY":
            print(f"{r[0]:22s} {r[1]:>7d} {r[2]:>7.2f} {r[3]:>11.2f}  {r[4]}")
    print()
    print(f"CLEAN: {len(clean)}    DIRTY (flagged): {len(dirty)}")
    print(f"fold sizes: " + ", ".join(str(len(f)) for f in folds))
    print(f"\noutput -> {OUT_DIR}")


if __name__ == "__main__":
    main()
