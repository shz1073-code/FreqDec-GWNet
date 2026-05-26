#!/usr/bin/env python
"""Classical phase-correlation baseline for the paper §IV comparison row.

GPT's review pointed out that the paper needs to compare against a
non-learned, textbook motion-estimation baseline so readers see that
the learning is doing more than what cv2 + a Hilbert envelope already
do. This implements the classical pipeline:

    1. For each consecutive frame pair (t-1, t), compute the per-frame
       background-only translation Δb_t via masked phase correlation
       (the same routine Phase B already uses).
    2. Accumulate U_t = cumsum(Δb_t) relative to the reference frame r.
       Optionally apply periodic reset every N frames (paper §III-C
       bounded-horizon).

This U is then fed into the same Metric A + Metric B evaluator we
report in `evaluate_paper_metrics.py`, so the comparison row is
1-to-1 with the learned methods.

Reference: cv2.phaseCorrelate is the de-facto standard for sub-pixel
translation estimation in fluoroscopy registration (used as a baseline
in Shechter 2004, Ambrosini 2015, and most subsequent works).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freqdec_gwnet.data import DataPaths, FluoroSequenceLoader        # noqa: E402
from freqdec_gwnet.data.state_labels import (                          # noqa: E402
    estimate_background_translation, cumulative_position,
    project_to_principal_axis,
)
from freqdec_gwnet.utils.background_patch_tracking import (            # noqa: E402
    detect_background_features, patch_alignment_residual,
    track_features_across_burst,
)
from freqdec_gwnet.utils.drift_metrics import (                        # noqa: E402
    landmark_residual, respiratory_projected_residual,
)
from freqdec_gwnet.utils.reference_selector import ReferenceSelector   # noqa: E402


def _apply_reset(U: np.ndarray, reset: int) -> np.ndarray:
    if reset is None or reset <= 0:
        return U
    T = U.shape[0]
    delta_u = np.zeros_like(U)
    delta_u[1:] = U[1:] - U[:-1]
    out = np.zeros_like(U)
    for s in range(0, T, reset):
        e = min(T, s + reset)
        chunk = delta_u[s:e].copy()
        chunk[0] = 0.0
        out[s:e] = np.cumsum(chunk, axis=0)
    return out


def classical_motion_field(
    images_uint8: np.ndarray,
    wire_masks_uint8,
    reference_index: int,
    method: str = "phase_corr",
) -> np.ndarray:
    """U_t for each classical baseline method.

    Methods:
        ``phase_corr`` — masked 2D phase correlation (Phase B baseline).
        ``ecc``        — cv2.findTransformECC translation per frame pair,
                          cumulated to absolute trajectory.
        ``lk_farneback`` — dense Farnebäck flow, mean over background ROI,
                            cumulated.

    All methods are pairwise (t-1, t) → translation, then cumsum + re-anchor.
    """
    import cv2
    T, H, W = images_uint8.shape
    if method == "phase_corr":
        deltas, _ = estimate_background_translation(
            images_uint8, wire_masks=wire_masks_uint8,
        )
    elif method == "ecc":
        deltas = np.zeros((T, 2), dtype=np.float32)
        # Build dilated background mask per frame
        if wire_masks_uint8 is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))
            bg_masks = np.stack(
                [(1 - cv2.dilate(w, kernel)).astype(np.uint8)
                 for w in wire_masks_uint8], axis=0,
            )
        else:
            bg_masks = np.ones((T, H, W), dtype=np.uint8)
        for t in range(1, T):
            f0 = images_uint8[t - 1].astype(np.float32) / 255.0
            f1 = images_uint8[t].astype(np.float32) / 255.0
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                # MOTION_TRANSLATION (2D shift only)
                _, warp = cv2.findTransformECC(
                    f0, f1, warp, motionType=cv2.MOTION_TRANSLATION,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        50, 1e-4,
                    ),
                    inputMask=bg_masks[t],
                )
                # warp[:, 2] is (tx, ty) such that f1 ≈ shift(f0, (tx, ty))
                deltas[t] = (float(warp[0, 2]), float(warp[1, 2]))
            except cv2.error:
                deltas[t] = 0.0
    elif method == "lk_farneback":
        deltas = np.zeros((T, 2), dtype=np.float32)
        if wire_masks_uint8 is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))
            bg_masks_bool = np.stack(
                [(1 - cv2.dilate(w, kernel)).astype(bool)
                 for w in wire_masks_uint8], axis=0,
            )
        else:
            bg_masks_bool = np.ones((T, H, W), dtype=bool)
        for t in range(1, T):
            f0 = images_uint8[t - 1]
            f1 = images_uint8[t]
            flow = cv2.calcOpticalFlowFarneback(
                f0, f1, None,
                pyr_scale=0.5, levels=3, winsize=21,
                iterations=3, poly_n=5, poly_sigma=1.1, flags=0,
            )
            # Mean flow over the joint background ROI
            joint = bg_masks_bool[t - 1] & bg_masks_bool[t]
            if joint.sum() < 0.05 * H * W:
                deltas[t] = 0.0
                continue
            deltas[t] = (
                float(flow[..., 0][joint].mean()),
                float(flow[..., 1][joint].mean()),
            )
    else:
        raise ValueError(f"unknown classical method '{method}'")

    pos = cumulative_position(deltas)
    pos = pos - pos[reference_index]
    return pos.astype(np.float32)


def evaluate_one(
    paths: DataPaths,
    dataset: str, split: str, seq_name: str,
    reset_every: int,
    method: str = "phase_corr",
) -> dict:
    loader = FluoroSequenceLoader(paths)
    seq = loader.load_sequence(dataset, split, seq_name)
    T = seq.num_frames
    images_uint8 = (seq.frames[:, 0].cpu().numpy() * 255).astype(np.uint8)
    wire_mask_uint8 = None
    if seq.masks is not None:
        wire_mask_uint8 = (seq.masks[:, 0].cpu().numpy() > 0.5).astype(np.uint8)

    selector = ReferenceSelector(warmup_min=5, warmup_max=10)
    ref = selector.select(images=seq.frames[:10, 0], total_frames=T)

    U_np = classical_motion_field(
        images_uint8, wire_mask_uint8, ref.reference_index, method=method,
    )
    U_np = _apply_reset(U_np, reset_every)
    U = torch.from_numpy(U_np)

    deltas, _ = estimate_background_translation(images_uint8, wire_masks=wire_mask_uint8)
    pos = cumulative_position(deltas)
    _, resp_axis, _ = project_to_principal_axis(pos)
    resp_axis_t = torch.from_numpy(resp_axis).float()

    # Metric A
    metric_a = None
    if seq.tip_xy is not None and bool(seq.tip_present[0]):
        lm = seq.tip_xy.clone()
        miss = ~seq.tip_present
        lm[miss] = float("nan")
        try:
            ra = respiratory_projected_residual(
                lm, U, reference_index=ref.reference_index,
                respiratory_axis=resp_axis_t,
            )
            metric_a = {
                "improvement": float(ra["improvement_mean"].item()),
                "reduction": float(ra["reduction_ratio"].item()),
            }
        except (ValueError, IndexError):
            pass

    # Metric B
    metric_b = None
    wire_ref = (
        wire_mask_uint8[ref.reference_index] if wire_mask_uint8 is not None
        else np.zeros(images_uint8.shape[1:], dtype=np.uint8)
    )
    corners = detect_background_features(
        images_uint8[ref.reference_index], wire_ref,
        max_features=40, wire_dilation_px=32,
    )
    if corners.shape[0] >= 4:
        track = track_features_across_burst(
            images_uint8, corners, reference_index=ref.reference_index,
        )
        pb = patch_alignment_residual(track, U_np, reference_index=ref.reference_index)
        metric_b = {
            "improvement": pb["global_improvement_mean"],
            "reduction": pb["reduction_ratio"],
        }

    # Legacy tip
    legacy = float("nan")
    if seq.tip_xy is not None and bool(seq.tip_present[0]):
        lm = seq.tip_xy.clone()
        miss = ~seq.tip_present
        lm[miss] = float("nan")
        try:
            lr = landmark_residual(lm, U, reference_index=ref.reference_index)
            if bool(lr["valid"].any().item()):
                legacy = float(lr["improvement_mean"].item())
        except (ValueError, IndexError):
            pass

    return {
        "method": method,
        "dataset": dataset, "split": split, "seq": seq_name, "T": T,
        "reset_every": reset_every,
        "metric_a_improvement": metric_a["improvement"] if metric_a else float("nan"),
        "metric_a_reduction": metric_a["reduction"] if metric_a else float("nan"),
        "metric_b_improvement": metric_b["improvement"] if metric_b else float("nan"),
        "metric_b_reduction": metric_b["reduction"] if metric_b else float("nan"),
        "legacy_tip_improvement": legacy,
        "U_max": float(np.linalg.norm(U_np, axis=-1).max()),
    }


_COLUMNS = [
    "method", "dataset", "split", "seq", "T", "reset_every",
    "metric_a_improvement", "metric_a_reduction",
    "metric_b_improvement", "metric_b_reduction",
    "legacy_tip_improvement", "U_max",
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset", default="dense_v1")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--seq", nargs="+", default=None)
    p.add_argument("--reset-every", nargs="+", type=int, default=[0, 16, 32])
    p.add_argument("--methods", nargs="+",
                   default=["phase_corr", "ecc", "lk_farneback"],
                   help="classical baselines to evaluate "
                        "(phase_corr / ecc / lk_farneback)")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(list(argv) if argv is not None else None)

    paths = (
        DataPaths.from_yaml(args.config) if args.config
        else DataPaths.from_default_config()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for method in args.methods:
        for split in args.splits:
            seqs = paths.list_sequences(args.dataset, split)
            if args.seq:
                seqs = [s for s in seqs if s in args.seq]
            for sname in seqs:
                for reset in args.reset_every:
                    row = evaluate_one(
                        paths, args.dataset, split, sname, reset, method=method,
                    )
                    rows.append(row)
                    print(
                        f"  {method:14s} {split:6s} {sname:18s} reset={reset:3d}  "
                        f"A_imp={row['metric_a_improvement']:+7.2f}  "
                        f"B_imp={row['metric_b_improvement']:+7.2f}  "
                        f"legacy={row['legacy_tip_improvement']:+7.2f}  "
                        f"U_max={row['U_max']:.1f}"
                    )

    out = args.output_dir / "classical_baseline.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_COLUMNS)
        for r in rows:
            w.writerow([
                "" if (isinstance(v, float) and v != v) else
                (f"{v:.6g}" if isinstance(v, float) else str(v))
                for v in (r.get(c, "") for c in _COLUMNS)
            ])
    print(f"\n[csv] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
