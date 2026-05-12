#!/usr/bin/env python
"""Generate R2 weak-supervision state labels for a dataset split.

Runs the full pipeline (B.1 + B.2 + B.3 + B.4) per sequence and writes
``{output_dir}/{seq_name}.npz`` plus an optional diagnostic PNG. Stage 2
and stage 3 training will read the .npz files to obtain
``cos_phi / sin_phi / amplitude / valid_mask / amp_scale / anchor_quality``.

Usage::

    # Single sequence (great for inspection):
    python scripts/generate_state_labels.py \\
        --dataset dense_v1 --split train --seq cmu_seq_007 \\
        --visualize

    # Whole split:
    python scripts/generate_state_labels.py \\
        --dataset dense_v1 --split train --all

The script does *not* require a trained stage-1 segmentation model — when
no ``--seg-checkpoint`` is given, the wire mask is omitted and the entire
frame is treated as background. This produces a usable but slightly noisier
state label and is the right choice for the very first end-to-end run.

Once stage-1 is trained, re-run with ``--seg-checkpoint`` to inject wire
masks into B.1 and tighten the labels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freqdec_gwnet.data import (                                       # noqa: E402
    DataPaths,
    FluoroSequenceLoader,
)
from freqdec_gwnet.data.state_labels import (                          # noqa: E402
    StateLabel,
    generate_state_label,
)
from freqdec_gwnet.data.state_labels.visualize import (                # noqa: E402
    render_state_label_png,
)


# ---------------------------------------------------------------------------
# Single-sequence orchestration
# ---------------------------------------------------------------------------


def process_sequence(
    paths: DataPaths,
    *,
    dataset: str,
    split: str,
    seq_name: str,
    output_dir: Path,
    fs: float,
    band_hz: tuple,
    bp_order: int,
    grid: tuple,
    grid_overlap: float,
    quality_threshold: float,
    min_valid_anchors: int,
    warmup_min: int,
    warmup_max: int,
    visualize: bool,
    seg_checkpoint: Optional[Path],
    max_frames: Optional[int],
    wire_mask_inferencer=None,
) -> dict:
    loader = FluoroSequenceLoader(paths)
    seq = loader.load_sequence(
        dataset, split, seq_name, max_frames=max_frames,
    )

    # Frames as uint8 [T, H, W] — phase-corr expects greyscale
    images = (seq.frames.numpy()[:, 0] * 255.0).astype(np.uint8)

    # Optional wire masks via stage-1 inference (Phase B.5).
    # When wire_mask_inferencer is provided we run frame-by-frame inference
    # with feature_bank carry-over.
    wire_masks = None
    if wire_mask_inferencer is not None:
        wire_masks = wire_mask_inferencer.infer_burst(images)
        # 形状自检：[T, H, W] uint8
        assert wire_masks.shape == images.shape, (
            f"wire mask shape {wire_masks.shape} != images {images.shape}"
        )

    label = generate_state_label(
        images,
        wire_masks=wire_masks,
        seq_name=seq.seq_name,
        dataset=seq.dataset,
        split=seq.split,
        fs=fs,
        band_hz=band_hz,
        bp_order=bp_order,
        grid=grid,
        grid_overlap=grid_overlap,
        quality_threshold=quality_threshold,
        min_valid_anchors=min_valid_anchors,
        warmup_min=warmup_min,
        warmup_max=warmup_max,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{seq_name}.npz"
    label.save_npz(npz_path)

    png_path = None
    if visualize:
        png_path = output_dir / f"{seq_name}.png"
        render_state_label_png(label, images, png_path, fs=fs)

    return {
        "seq_name": seq_name,
        "npz_path": npz_path,
        "png_path": png_path,
        "n_frames": label.num_frames,
        "n_valid": int(label.valid_mask.sum()),
        "amp_scale": float(label.amp_scale),
        "reference_index": label.reference_index,
        "reference_strategy": label.reference_strategy,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_seqs(
    paths: DataPaths,
    dataset: str,
    split: str,
    explicit: Optional[List[str]],
    all_flag: bool,
) -> List[str]:
    if all_flag and explicit:
        raise SystemExit("--all and --seq are mutually exclusive")
    if all_flag:
        return paths.list_sequences(dataset, split)
    if not explicit:
        raise SystemExit("specify --seq <name> [<name>...] or --all")
    available = set(paths.list_sequences(dataset, split))
    missing = [s for s in explicit if s not in available]
    if missing:
        raise SystemExit(
            f"sequences not found in {dataset}/{split}: {missing}"
        )
    return list(explicit)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate R2 weak-supervision state labels (Phase B).",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="path to data_paths.yaml (defaults to project's)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True,
                        choices=("train", "val", "test"))
    parser.add_argument("--seq", nargs="+", default=None,
                        help="one or more sequence names")
    parser.add_argument("--all", action="store_true",
                        help="process every sequence in the split")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "reports" / "state_labels")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="cap T per sequence (debug only)")
    parser.add_argument("--min-frames", type=int, default=64,
                        help="skip sequences shorter than this. Default 64 "
                             "covers ≥1 respiratory cycle at 15 fps; bursts "
                             "<32 cannot pass the order-4 filtfilt minimum")
    parser.add_argument("--skip-failures", action="store_true",
                        help="continue on per-sequence errors instead of aborting")
    parser.add_argument("--visualize", action="store_true",
                        help="emit a diagnostic PNG alongside each .npz")
    parser.add_argument("--seg-checkpoint", type=Path, default=None,
                        help="FAST-LiteNet stage-1 ckpt for wire-mask "
                             "inference (Phase B.5). When omitted, the "
                             "pipeline treats the full frame as background.")
    parser.add_argument("--seg-width-mult", type=float, default=1.0,
                        help="must match the stage-1 ckpt's width_mult")
    parser.add_argument("--seg-freq-mode", default="global",
                        choices=("global", "local_sff", "ms_local_sff"))
    parser.add_argument("--seg-threshold", type=float, default=0.5)
    parser.add_argument("--seg-device", default="cuda")

    # Pipeline knobs (defaults match PROJECT_CONSTRAINTS_19.6 §1)
    parser.add_argument("--fs", type=float, default=15.0,
                        help="sampling rate in Hz (typical fluoro: 15)")
    parser.add_argument("--band-low-hz", type=float, default=0.15)
    parser.add_argument("--band-high-hz", type=float, default=0.50)
    parser.add_argument("--bp-order", type=int, default=4)
    parser.add_argument("--grid-h", type=int, default=2)
    parser.add_argument("--grid-w", type=int, default=2)
    parser.add_argument("--grid-overlap", type=float, default=0.10)
    parser.add_argument("--quality-threshold", type=float, default=0.50)
    parser.add_argument("--min-valid-anchors", type=int, default=3)
    parser.add_argument("--warmup-min", type=int, default=5)
    parser.add_argument("--warmup-max", type=int, default=10)

    args = parser.parse_args(list(argv) if argv is not None else None)

    paths = (
        DataPaths.from_yaml(args.config)
        if args.config
        else DataPaths.from_default_config()
    )

    seqs = _resolve_seqs(paths, args.dataset, args.split, args.seq, args.all)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Phase B.5: optionally build the wire-mask inferencer once, reuse for all seqs
    wire_mask_inferencer = None
    if args.seg_checkpoint is not None:
        from freqdec_gwnet.data.state_labels.wire_mask_inference import (
            WireMaskInferencer,
        )
        print(
            f"[B.5] loading stage-1 ckpt for wire-mask inference: "
            f"{args.seg_checkpoint}"
        )
        wire_mask_inferencer = WireMaskInferencer(
            ckpt_path=str(args.seg_checkpoint),
            device=args.seg_device,
            threshold=args.seg_threshold,
            width_mult=args.seg_width_mult,
            freq_mode=args.seg_freq_mode,
        )

    print(
        f"[generate_state_labels] dataset={args.dataset} split={args.split} "
        f"sequences={len(seqs)} fs={args.fs} band={args.band_low_hz}-"
        f"{args.band_high_hz} Hz output={args.output_dir} "
        f"wire_masks={'on' if wire_mask_inferencer else 'off'}"
    )
    summary = []
    skipped_short: list = []
    failures: list = []
    for sname in seqs:
        # 先快速看一下帧数（只数文件名），太短直接跳过
        try:
            frames = paths.list_frames(args.dataset, args.split, sname)
            full_T = len(frames)
        except Exception as exc:
            if args.skip_failures:
                failures.append((sname, f"list_frames: {exc}"))
                continue
            raise

        if full_T < args.min_frames:
            skipped_short.append((sname, full_T))
            continue

        try:
            result = process_sequence(
                paths,
                dataset=args.dataset, split=args.split, seq_name=sname,
                output_dir=args.output_dir,
                fs=args.fs,
                band_hz=(args.band_low_hz, args.band_high_hz),
                bp_order=args.bp_order,
                grid=(args.grid_h, args.grid_w),
                grid_overlap=args.grid_overlap,
                quality_threshold=args.quality_threshold,
                min_valid_anchors=args.min_valid_anchors,
                warmup_min=args.warmup_min, warmup_max=args.warmup_max,
                visualize=args.visualize,
                seg_checkpoint=args.seg_checkpoint,
                max_frames=args.max_frames,
                wire_mask_inferencer=wire_mask_inferencer,
            )
        except Exception as exc:
            if args.skip_failures:
                failures.append((sname, str(exc)))
                continue
            raise
        summary.append(result)
        print(
            f"  {sname:15s}  T={result['n_frames']:4d}  "
            f"valid={result['n_valid']:4d}/{result['n_frames']:4d}  "
            f"amp_scale={result['amp_scale']:.3f}  "
            f"r={result['reference_index']:3d} "
            f"({result['reference_strategy']:14s})"
        )
    if skipped_short:
        print(
            f"\n[skipped {len(skipped_short)} short sequences "
            f"(< {args.min_frames} frames):"
        )
        for sname, T in skipped_short:
            print(f"   - {sname}: {T} frames")
    if failures:
        print(f"\n[failures: {len(failures)}]")
        for sname, msg in failures:
            print(f"   - {sname}: {msg}")
    print(
        f"\nDone: produced={len(summary)} skipped_short={len(skipped_short)} "
        f"failures={len(failures)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
