#!/usr/bin/env python
"""Generate synthetic augmented bursts from real fluoroscopy seeds.

For each existing real burst with a Phase B state label:

    1. Pick a "stable" reference frame (the burst's reference_index from
       its StateLabel, falling back to the warmup-window minimum-state-
       change frame).
    2. Spawn K virtual bursts by applying randomized respiratory motion
       via :class:`SyntheticBurstGenerator`. Each virtual burst has
       perfect (cos φ, sin φ, amplitude) labels.
    3. Persist each virtual burst as:
            {output_root}/{split}/images/{seq_name}_aug_{i}/frame_*.png
            {output_root}/{split}/labels/{seq_name}_aug_{i}/frame_*.png
              (copies the seed-frame's wire mask, optionally jittered)
            {state_labels_dir}/{split}/{seq_name}_aug_{i}.npz
       so :class:`FluoroSequenceWindowDataset` can consume them
       without code changes.

Usage:
    python scripts/generate_synthetic_augmented.py \\
        --source-data-root /workspace/shz/clean_data \\
        --source-state-labels-dir reports/state_labels/clean_v2 \\
        --output-data-root /workspace/shz/synthetic_aug_v1 \\
        --output-state-labels-dir reports/state_labels/synthetic_aug_v1 \\
        --splits train val \\
        --K-per-seed 10 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freqdec_gwnet.data.state_labels import StateLabel                 # noqa: E402
from freqdec_gwnet.data.synthetic_burst import (                       # noqa: E402
    SyntheticBurstGenerator,
)


def _pick_seed_frame(
    state_label_path: Path, images_dir: Path,
) -> tuple:
    """Return the seed (anchor) frame plus optional wire mask from disk."""
    sl = StateLabel.load_npz(state_label_path)
    ref_idx = int(sl.reference_index)
    frame_names = sorted([
        p.name for p in images_dir.iterdir() if p.suffix.lower() == ".png"
    ])
    if not frame_names:
        raise FileNotFoundError(f"no frames under {images_dir}")
    ref_idx = min(ref_idx, len(frame_names) - 1)
    frame_path = images_dir / frame_names[ref_idx]
    img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cv2.imread failed: {frame_path}")
    return img, ref_idx, frame_names[ref_idx], sl


def _write_burst(
    out_imgs_dir: Path, out_lbls_dir: Optional[Path],
    burst,
    wire_mask: Optional[np.ndarray],
) -> None:
    out_imgs_dir.mkdir(parents=True, exist_ok=True)
    if out_lbls_dir is not None:
        out_lbls_dir.mkdir(parents=True, exist_ok=True)
    T = burst.num_frames
    for t in range(T):
        fname = f"frame_{t:04d}.png"
        cv2.imwrite(str(out_imgs_dir / fname), burst.images[t])
        if out_lbls_dir is not None and wire_mask is not None:
            # Copy the seed-frame wire mask (could be jittered with a similar
            # warp in a future revision; for now keep it static — the state
            # branch ignores the wire region via SpatialLowPass anyway)
            cv2.imwrite(str(out_lbls_dir / fname), wire_mask)


def _state_label_for_burst(
    seq_name: str, dataset: str, split: str, burst,
    fs: float, amp_scale: float,
) -> StateLabel:
    """Wrap the synthetic burst's perfect labels in a StateLabel object."""
    T = burst.num_frames
    return StateLabel(
        seq_name=seq_name,
        dataset=dataset, split=split,
        cos_phi=burst.cos_phi,
        sin_phi=burst.sin_phi,
        amplitude=burst.amplitude,
        amp_scale=float(burst.amp_scale),
        valid_mask=burst.valid_mask,
        anchor_quality=np.full(T, 1.0, dtype=np.float32),
        reference_index=0,
        reference_strategy="synthetic",
        metadata={
            "fs": fs,
            "synthetic": True,
            **{k: (float(v) if isinstance(v, np.floating) else v)
               for k, v in burst.params.items()},
        },
    )


def process_one_seed(
    *,
    src_data_root: Path, src_state_labels: Path,
    split: str, seq_name: str,
    out_data_root: Path, out_state_labels: Path,
    K: int, generator: SyntheticBurstGenerator,
    base_seed: int,
) -> List[str]:
    src_images_dir = src_data_root / split / "images" / seq_name
    src_labels_dir = src_data_root / split / "labels" / seq_name
    sl_path = src_state_labels / split / f"{seq_name}.npz"
    if not sl_path.is_file():
        return []
    if not src_images_dir.is_dir():
        return []

    seed_img, ref_idx, ref_fname, sl = _pick_seed_frame(sl_path, src_images_dir)
    seed_wire = None
    if src_labels_dir.is_dir():
        wire_path = src_labels_dir / ref_fname
        if wire_path.is_file():
            seed_wire = cv2.imread(str(wire_path), cv2.IMREAD_GRAYSCALE)

    out_imgs_root = out_data_root / split / "images"
    out_lbls_root = out_data_root / split / "labels"
    out_sl_dir = out_state_labels / split
    out_sl_dir.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    for i in range(K):
        new_name = f"{seq_name}_aug_{i:02d}"
        burst = generator.generate(seed_img, seed=base_seed + i)
        _write_burst(
            out_imgs_root / new_name,
            out_lbls_root / new_name if seed_wire is not None else None,
            burst,
            wire_mask=seed_wire,
        )
        new_sl = _state_label_for_burst(
            seq_name=new_name, dataset="synthetic_aug", split=split,
            burst=burst, fs=generator.fs, amp_scale=burst.amp_scale,
        )
        new_sl.save_npz(out_sl_dir / f"{new_name}.npz")
        written.append(new_name)
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source-data-root", type=Path, required=True)
    p.add_argument("--source-state-labels-dir", type=Path, required=True)
    p.add_argument("--output-data-root", type=Path, required=True)
    p.add_argument("--output-state-labels-dir", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--seq", nargs="+", default=None,
                   help="restrict to specific seq names")
    p.add_argument("--K-per-seed", type=int, default=10,
                   help="virtual bursts per real seed frame")
    p.add_argument("--fs", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T-range", nargs=2, type=int, default=[120, 300])
    p.add_argument("--resp-freq-range", nargs=2, type=float, default=[0.20, 0.40])
    p.add_argument("--resp-amp-y-range", nargs=2, type=float, default=[4.0, 22.0])
    p.add_argument("--resp-amp-x-range", nargs=2, type=float, default=[0.0, 4.0])
    p.add_argument("--amp-mod-range", nargs=2, type=float, default=[0.10, 0.40])
    p.add_argument("--cardiac-amp-range", nargs=2, type=float, default=[0.0, 0.0])
    p.add_argument("--drift-range", nargs=2, type=float, default=[0.0, 0.0])
    p.add_argument("--noise-sigma-range", nargs=2, type=float,
                   default=[0.005, 0.020])
    args = p.parse_args(list(argv) if argv is not None else None)

    generator = SyntheticBurstGenerator(
        fs=args.fs,
        resp_freq_hz_range=tuple(args.resp_freq_range),
        resp_amp_y_px_range=tuple(args.resp_amp_y_range),
        resp_amp_x_px_range=tuple(args.resp_amp_x_range),
        amp_modulation_range=tuple(args.amp_mod_range),
        cardiac_amp_px_range=tuple(args.cardiac_amp_range),
        drift_px_range=tuple(args.drift_range),
        noise_sigma_range=tuple(args.noise_sigma_range),
        T_range=tuple(args.T_range),
        seed=args.seed,
    )

    total_written = 0
    seed_counter = args.seed * 1000
    for split in args.splits:
        sl_split = args.source_state_labels_dir / split
        if not sl_split.is_dir():
            print(f"[skip] {sl_split} not found")
            continue
        sl_files = sorted(sl_split.glob("*.npz"))
        seeds = [p.stem for p in sl_files]
        if args.seq:
            seeds = [s for s in seeds if s in args.seq]
        print(f"[{split}] {len(seeds)} seeds → generating {len(seeds) * args.K_per_seed} virtual bursts")
        for sname in seeds:
            written = process_one_seed(
                src_data_root=args.source_data_root,
                src_state_labels=args.source_state_labels_dir,
                split=split, seq_name=sname,
                out_data_root=args.output_data_root,
                out_state_labels=args.output_state_labels_dir,
                K=args.K_per_seed, generator=generator,
                base_seed=seed_counter,
            )
            seed_counter += args.K_per_seed
            if written:
                print(f"  {sname:18s} → {len(written)} bursts")
                total_written += len(written)
            else:
                print(f"  {sname:18s} (skipped — missing files)")
    print(f"\n[done] wrote {total_written} virtual bursts to {args.output_data_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
