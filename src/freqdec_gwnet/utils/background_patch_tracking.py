"""Background-patch alignment metric (paper §IV.D, Metric B).

The naive landmark residual on the wire tip mixes respiratory motion with
active push. A cleaner test of "does compensation remove respiratory
drift" is to track **anatomical features that are *not* on the
instrument** — small high-contrast patches such as vessel bifurcations,
spine corners, or diaphragm edges. These features are physiologically
stationary modulo respiration, so their cross-frame displacement
isolates the respiratory component of motion.

Implementation choices:

    * **Feature selection** — use ``cv2.goodFeaturesToTrack`` (Shi-Tomasi
      corner detector) on the reference frame's **background-only** mask.
      The background mask is the inverse of the dilated wire mask, so
      no feature can land on the instrument.
    * **Cross-frame matching** — Lucas-Kanade pyramidal optical flow
      (``cv2.calcOpticalFlowPyrLK``) tracks each reference feature into
      every other frame of the burst. Lost / divergent features are
      dropped (back-tracking RMSE > 1.5 px).
    * **Before / after** — for each tracked feature, ``before_drift_t =
      |pos_t - pos_r|`` and ``after_drift_t = |(pos_t - U_t) - pos_r|``.
      Mean over features per frame; mean over frames for the headline.

Reference: Shechter et al. (IEEE TMI 2004), Ablitt et al. 2004, and the
2D-3D registration literature evaluate motion compensation this way —
not on the moving instrument tip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class BackgroundPatchTrack:
    """Tracked anatomical features across a burst.

    Attributes:
        feature_xy_ref: ``[K, 2]`` reference-frame anchor positions.
        feature_xy_per_frame: ``[T, K, 2]`` tracked positions, ``NaN``
            where tracking failed.
        valid: ``[T, K]`` bool — True if both forward + backward LK
            converged within the configured tolerance.
        n_features: ``K``.
    """
    feature_xy_ref: np.ndarray
    feature_xy_per_frame: np.ndarray
    valid: np.ndarray
    n_features: int


def detect_background_features(
    image_ref: np.ndarray,                 # [H, W] uint8 or float
    wire_mask_ref: np.ndarray,             # [H, W] 0/1
    *,
    max_features: int = 50,
    quality_level: float = 0.01,
    min_distance: int = 20,
    wire_dilation_px: int = 32,
) -> np.ndarray:
    """Shi-Tomasi corners on the *background-only* part of the reference frame.

    Returns:
        ``[K, 2]`` float32 array of (x, y) corner coordinates. ``K`` may
        be < ``max_features`` if not enough strong corners exist.
    """
    import cv2
    if image_ref.dtype != np.uint8:
        img = (np.clip(image_ref, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        img = image_ref

    # Dilate the wire mask aggressively so no corner can sit on the
    # instrument or its immediate shadow.
    k = max(3, 2 * wire_dilation_px + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    wm = (wire_mask_ref > 0).astype(np.uint8)
    wm_dil = cv2.dilate(wm, kernel, iterations=1)
    bg_mask = (1 - wm_dil).astype(np.uint8)
    if bg_mask.sum() < 0.1 * bg_mask.size:
        # Background almost gone — skip wire mask entirely
        bg_mask = np.ones_like(bg_mask)

    corners = cv2.goodFeaturesToTrack(
        img, maxCorners=max_features,
        qualityLevel=quality_level, minDistance=min_distance,
        mask=bg_mask * 255,
    )
    if corners is None:
        return np.zeros((0, 2), dtype=np.float32)
    return corners.reshape(-1, 2).astype(np.float32)


def track_features_across_burst(
    images: np.ndarray,                    # [T, H, W] uint8 or float
    feature_xy_ref: np.ndarray,             # [K, 2]
    reference_index: int = 0,
    *,
    pyramid_levels: int = 3,
    bidirectional_tolerance: float = 1.5,
) -> BackgroundPatchTrack:
    """Track each reference feature across every other frame via LK.

    Bidirectional check: track forward (ref → t) then backward (t → ref)
    and reject features whose forward+backward round-trip error exceeds
    ``bidirectional_tolerance`` pixels. This is the standard robustness
    trick (Kalal et al. 2010, "Forward-Backward Error").
    """
    import cv2
    T, H, W = images.shape
    K = feature_xy_ref.shape[0]
    if images.dtype != np.uint8:
        imgs = (np.clip(images, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        imgs = images

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=pyramid_levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    out = np.full((T, K, 2), np.nan, dtype=np.float32)
    valid = np.zeros((T, K), dtype=bool)
    out[reference_index] = feature_xy_ref
    valid[reference_index] = True

    ref_img = imgs[reference_index]
    pts_ref = feature_xy_ref.reshape(-1, 1, 2).astype(np.float32)
    for t in range(T):
        if t == reference_index:
            continue
        pts_fwd, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
            ref_img, imgs[t], pts_ref, None, **lk_params,
        )
        if pts_fwd is None:
            continue
        # Backward check
        pts_back, st_back, _ = cv2.calcOpticalFlowPyrLK(
            imgs[t], ref_img, pts_fwd, None, **lk_params,
        )
        if pts_back is None:
            continue
        diff = np.linalg.norm(
            (pts_back.reshape(-1, 2) - feature_xy_ref), axis=-1,
        )
        ok = (st_fwd.reshape(-1) == 1) & (st_back.reshape(-1) == 1)
        ok = ok & (diff < bidirectional_tolerance)
        for k in range(K):
            if ok[k]:
                out[t, k] = pts_fwd[k, 0]
                valid[t, k] = True
    return BackgroundPatchTrack(
        feature_xy_ref=feature_xy_ref,
        feature_xy_per_frame=out,
        valid=valid,
        n_features=K,
    )


def patch_alignment_residual(
    track: BackgroundPatchTrack,
    U: np.ndarray,                          # [T, 2] cumulative motion
    reference_index: int = 0,
) -> Dict[str, np.ndarray]:
    """Compute per-frame mean patch drift before/after compensation.

    For each tracked feature ``k`` and frame ``t``:

        before_t,k = ||p_t,k − p_ref,k||
        after_t,k  = ||(p_t,k − U_t) − p_ref,k||

    Aggregated over features (mean) per frame, then over frames for the
    headline reduction ratio.
    """
    feat = track.feature_xy_per_frame             # [T, K, 2]
    ref = track.feature_xy_ref                    # [K, 2]
    valid = track.valid                            # [T, K]
    T, K, _ = feat.shape

    drift_uncomp = feat - ref[None, :, :]          # [T, K, 2]
    drift_comp = drift_uncomp - U[:, None, :]      # [T, K, 2]

    norm_before = np.linalg.norm(drift_uncomp, axis=-1)   # [T, K]
    norm_after = np.linalg.norm(drift_comp, axis=-1)      # [T, K]

    # Masked mean per frame
    safe_valid = valid.astype(np.float32)
    cnt = safe_valid.sum(axis=-1).clip(min=1.0)
    mean_before = (np.where(valid, norm_before, 0.0).sum(axis=-1)) / cnt
    mean_after = (np.where(valid, norm_after, 0.0).sum(axis=-1)) / cnt
    # Frames with zero valid features → NaN
    mean_before = np.where(safe_valid.sum(axis=-1) > 0, mean_before, np.nan)
    mean_after = np.where(safe_valid.sum(axis=-1) > 0, mean_after, np.nan)

    valid_frames = ~np.isnan(mean_before)
    if valid_frames.any():
        global_before = float(mean_before[valid_frames].mean())
        global_after = float(mean_after[valid_frames].mean())
        global_imp = global_before - global_after
        reduction = (
            global_after / global_before if global_before > 1e-8
            else float("nan")
        )
    else:
        global_before = global_after = global_imp = reduction = float("nan")

    return {
        "mean_before_per_frame": mean_before,         # [T]
        "mean_after_per_frame": mean_after,           # [T]
        "global_mean_before": global_before,
        "global_mean_after": global_after,
        "global_improvement_mean": global_imp,
        "reduction_ratio": reduction,
        "n_features": K,
        "n_valid_frames": int(valid_frames.sum()),
    }
