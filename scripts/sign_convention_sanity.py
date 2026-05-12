#!/usr/bin/env python
"""Diagnose sign convention between phase correlation and LK tracking.

The oracle ceiling test exposed a contradiction: phase-correlation
background-motion estimates (Phase B) made Metric B (background patch
alignment, computed via LK tracking) WORSE, not better. If both
geometric methods agreed on the bg motion direction, feeding one as U
should make the other's residual go to zero.

This script does a controlled synthetic experiment: shift a real
frame by a known (dx, dy), then:

    1. Run phase correlation between original and shifted frame.
    2. Run LK tracking on corners in original → shifted frame.
    3. Compare the recovered shifts.

If they agree → Metric B oracle should work and the failure is
elsewhere. If they disagree on sign → we know exactly what to fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freqdec_gwnet.data import DataPaths, FluoroSequenceLoader
from freqdec_gwnet.data.state_labels import estimate_background_translation
from freqdec_gwnet.utils.background_patch_tracking import (
    detect_background_features, track_features_across_burst,
)


def shift_image(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    H, W = img.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )


def diagnose(images_uint8: np.ndarray, wire_masks: np.ndarray | None) -> None:
    print(f"  image shape = {images_uint8.shape}, dtype={images_uint8.dtype}")

    # Take frame 0 and create synthetic shifted version
    ref = images_uint8[0]
    test_shifts = [(0, 5), (0, -5), (10, 0), (-10, 0), (3, 7)]

    print("\n  Known synthetic shift → recovered shift comparison:")
    print(f"  {'truth dx,dy':>13s}  {'phase_corr dx,dy':>20s}  {'LK mean dx,dy':>20s}")
    for dx, dy in test_shifts:
        shifted = shift_image(ref, dx, dy)
        burst = np.stack([ref, shifted], axis=0)
        wm = wire_masks[:2] if wire_masks is not None else None

        # Phase correlation
        pc_deltas, _ = estimate_background_translation(burst, wire_masks=wm)
        pc_dx, pc_dy = float(pc_deltas[1, 0]), float(pc_deltas[1, 1])

        # LK tracking
        wm_ref = wm[0] if wm is not None else np.zeros_like(ref, dtype=np.uint8)
        corners = detect_background_features(
            ref, wm_ref, max_features=20, wire_dilation_px=8,
        )
        if corners.shape[0] >= 4:
            track = track_features_across_burst(burst, corners, reference_index=0)
            displacements = track.feature_xy_per_frame[1] - corners      # [K, 2]
            valid_disp = displacements[track.valid[1]]
            if valid_disp.size:
                lk_dx, lk_dy = float(valid_disp[:, 0].mean()), float(valid_disp[:, 1].mean())
            else:
                lk_dx, lk_dy = float("nan"), float("nan")
        else:
            lk_dx, lk_dy = float("nan"), float("nan")
        print(f"  ({dx:+3d},{dy:+3d})         ({pc_dx:+6.2f},{pc_dy:+6.2f})       "
              f"({lk_dx:+6.2f},{lk_dy:+6.2f})")


def main():
    paths = DataPaths.from_default_config()
    loader = FluoroSequenceLoader(paths)
    seq = loader.load_sequence("dense_v1", "train", "cmu_seq_015")
    images_uint8 = (seq.frames[:5, 0].cpu().numpy() * 255).astype(np.uint8)
    wire_masks = None
    if seq.masks is not None:
        wire_masks = (seq.masks[:5, 0].cpu().numpy() > 0.5).astype(np.uint8)

    print("=== Sign convention diagnostic ===")
    print("Truth: apply known affine shift to frame 0 → frame 1.")
    print("Phase correlation and LK should recover the SAME (dx, dy).\n")
    diagnose(images_uint8, wire_masks)

    print("\n=== Real-frame pair (frame 0 vs frame 1 of same sequence) ===")
    print("(Different sources of motion: phase corr sees full image, LK sees corners)")
    burst2 = images_uint8[:2]
    wm2 = wire_masks[:2] if wire_masks is not None else None
    pc_deltas, _ = estimate_background_translation(burst2, wire_masks=wm2)
    print(f"  phase_corr (0→1) = ({pc_deltas[1, 0]:+.2f}, {pc_deltas[1, 1]:+.2f})")
    wm_ref = wm2[0] if wm2 is not None else np.zeros_like(images_uint8[0], dtype=np.uint8)
    corners = detect_background_features(
        images_uint8[0], wm_ref, max_features=20, wire_dilation_px=8,
    )
    if corners.shape[0] >= 4:
        track = track_features_across_burst(burst2, corners, reference_index=0)
        disp = track.feature_xy_per_frame[1] - corners
        valid_disp = disp[track.valid[1]]
        if valid_disp.size:
            print(f"  LK mean (0→1)    = ({valid_disp[:, 0].mean():+.2f}, "
                  f"{valid_disp[:, 1].mean():+.2f})")


if __name__ == "__main__":
    main()
