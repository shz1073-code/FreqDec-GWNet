"""Build the cardiac expanded_pool by mirroring the respiratory pool naming.

For each respiratory-pool symlink `<renamed>.npz` at
  reports/state_labels/clean_v2_expanded/expanded_pool/<renamed>.npz
we figure out (dataset, split, seqname) from `<renamed>` and create
the parallel cardiac symlink at
  reports/state_labels/cardiac_pool/expanded_pool/<renamed>.npz
pointing to
  reports/state_labels/cardiac_v1/<dataset>/<seqname>.npz

generate_state_labels.py wrote cardiac NPZs flat under `<output-dir>/<seq>.npz`
regardless of split, so split prefixing is purely a renaming step here.
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESP_POOL = ROOT / "reports/state_labels/clean_v2_expanded/expanded_pool"
CARDIAC_SRC = ROOT / "reports/state_labels/cardiac_v1"
CARDIAC_POOL = ROOT / "reports/state_labels/cardiac_pool/expanded_pool"

DS_PREFIX = {"cl": "clean", "or": "ori"}


def parse_renamed(name: str):
    """`cl_tr_cmu_seq_001` -> ('clean', 'train', 'cmu_seq_001')."""
    parts = name.split("_", 2)
    if len(parts) != 3:
        return None
    ds, _, seq = parts
    return DS_PREFIX.get(ds), None, seq         # split unused; flat output


def main():
    CARDIAC_POOL.mkdir(parents=True, exist_ok=True)
    found = 0
    miss = 0
    for resp_npz in sorted(RESP_POOL.glob("*.npz")):
        renamed = resp_npz.stem
        parsed = parse_renamed(renamed)
        if parsed is None:
            continue
        ds, _, seq = parsed
        if ds is None:
            continue
        src = CARDIAC_SRC / ds / f"{seq}.npz"
        if not src.is_file():
            miss += 1
            continue
        dst = CARDIAC_POOL / f"{renamed}.npz"
        dst.unlink(missing_ok=True)
        dst.symlink_to(src.resolve())
        found += 1
    print(f"linked {found} cardiac labels into {CARDIAC_POOL}")
    print(f"missing source npz (seq filtered out of cardiac band): {miss}")


if __name__ == "__main__":
    main()
