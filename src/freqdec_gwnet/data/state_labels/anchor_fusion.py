"""B.3 — multi-region anchor fusion + per-frame quality / valid_mask.

PROJECT_CONSTRAINTS_19.6 §1 R2 mandates "multi-anchor quality fusion" and
"quality_report / valid_mask". The interpretation we adopt:

    * Anchors are *spatial* — different sub-regions of the X-ray image
      (lung, heart, spine, abdomen) carry physiological motion at different
      amplitudes and slightly different phases. Running B.1+B.2 separately
      per anchor produces K independent state estimates.
    * Consensus across the K estimates per frame yields a quality score:
      tightly-agreeing anchors → high confidence; widely-disagreeing
      anchors → low confidence (likely noise, partial occlusion, or
      C-arm motion).

We pick a 2×2 spatial grid by default (4 anchors). Anatomical priors
could replace the grid in a future revision but the grid is robust and
automatic.

Output:
    cos_phi[T], sin_phi[T], amplitude[T]: median across anchors after
        circular phase alignment (anchors are aligned to the consensus
        phase before averaging amplitudes).
    anchor_quality[T]: 1 − circular variance of the K phase estimates,
        scaled to [0, 1]. Higher = anchors agree.
    valid_mask[T]: boolean, True where anchor_quality exceeds the
        configured threshold AND the underlying B.1 estimate was valid
        for at least ``min_valid_anchors`` of the K anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .background_motion import (
    cumulative_position,
    estimate_background_translation,
)
from .band_filter import (
    band_pass_filter,
    hilbert_phase_amplitude,
    project_to_principal_axis,
)


# ---------------------------------------------------------------------------
# Per-anchor result container
# ---------------------------------------------------------------------------


@dataclass
class AnchorEstimate:
    """Per-anchor outputs of B.1 + B.2 chained together.

    Attributes:
        cos_phi / sin_phi: ``[T]`` instantaneous phase as ``(cos, sin)``.
        amplitude: ``[T]`` Hilbert envelope after band-pass filtering.
        principal_axis: ``[2]`` PCA axis used for the 1-D projection.
        explained_ratio: variance ratio captured by the principal axis.
        delta_valid: ``[T]`` boolean from :func:`estimate_background_translation`.
        roi_box: ``(y0, y1, x0, x1)`` half-open slice that defines the anchor.
    """

    cos_phi: np.ndarray
    sin_phi: np.ndarray
    amplitude: np.ndarray
    principal_axis: np.ndarray
    explained_ratio: float
    delta_valid: np.ndarray
    roi_box: Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Spatial grid of anchor ROIs
# ---------------------------------------------------------------------------


def _grid_rois(
    H: int, W: int, grid_h: int, grid_w: int, overlap: float = 0.0,
) -> List[Tuple[int, int, int, int]]:
    """Tile the image into a (grid_h × grid_w) grid of (possibly) overlapping ROIs.

    Each ROI is a half-open ``(y0, y1, x0, x1)`` slice. Overlap is the
    fractional extension of each tile in both directions: 0.0 means a
    plain non-overlapping tiling, 0.25 means each tile is 25% wider/taller
    than its base size and overlaps neighbours by half of that on each side.
    """
    if not 0.0 <= overlap < 0.5:
        raise ValueError("overlap must be in [0, 0.5)")
    base_h = H / grid_h
    base_w = W / grid_w
    pad_h = int(round(base_h * overlap / 2))
    pad_w = int(round(base_w * overlap / 2))

    rois: List[Tuple[int, int, int, int]] = []
    for gy in range(grid_h):
        for gx in range(grid_w):
            y0 = max(0, int(round(gy * base_h)) - pad_h)
            y1 = min(H, int(round((gy + 1) * base_h)) + pad_h)
            x0 = max(0, int(round(gx * base_w)) - pad_w)
            x1 = min(W, int(round((gx + 1) * base_w)) + pad_w)
            rois.append((y0, y1, x0, x1))
    return rois


# ---------------------------------------------------------------------------
# Single-anchor estimation: chain B.1 → B.2 inside one ROI
# ---------------------------------------------------------------------------


def _estimate_one_anchor(
    images: np.ndarray,                # [T, H, W]
    wire_masks: np.ndarray,            # [T, H, W] or None
    roi: Tuple[int, int, int, int],
    fs: float,
    band: Tuple[float, float],
    bp_order: int,
) -> AnchorEstimate:
    """Run B.1 + B.2 inside a single spatial ROI."""
    y0, y1, x0, x1 = roi
    sub_imgs = images[:, y0:y1, x0:x1]
    sub_wire = (
        wire_masks[:, y0:y1, x0:x1] if wire_masks is not None else None
    )
    deltas, delta_valid = estimate_background_translation(
        sub_imgs, sub_wire,
    )
    pos = cumulative_position(deltas)
    proj, axis, ratio = project_to_principal_axis(pos)

    bp = band_pass_filter(proj, fs=fs, low_hz=band[0], high_hz=band[1], order=bp_order)
    phase, amp = hilbert_phase_amplitude(bp)

    return AnchorEstimate(
        cos_phi=np.cos(phase).astype(np.float32),
        sin_phi=np.sin(phase).astype(np.float32),
        amplitude=amp,
        principal_axis=axis,
        explained_ratio=ratio,
        delta_valid=delta_valid,
        roi_box=roi,
    )


# ---------------------------------------------------------------------------
# Multi-anchor consensus
# ---------------------------------------------------------------------------


def _circular_align(anchors: List[AnchorEstimate]) -> np.ndarray:
    """Align per-anchor phases to a shared reference by removing a global offset.

    The Hilbert phase has an arbitrary global phase shift per anchor (it
    depends on where in the cycle the recording happened to start in that
    sub-region). We pick anchor 0 as reference and rotate all others to
    minimize their mean phase distance to it.

    Returns ``cos_aligned[K, T]`` and ``sin_aligned[K, T]`` as a single
    ``[K, T, 2]`` array stacked along the last axis.
    """
    K = len(anchors)
    T = anchors[0].cos_phi.shape[0]
    aligned = np.empty((K, T, 2), dtype=np.float32)
    aligned[0, :, 0] = anchors[0].cos_phi
    aligned[0, :, 1] = anchors[0].sin_phi

    ref_complex = anchors[0].cos_phi + 1j * anchors[0].sin_phi  # [T]
    for k in range(1, K):
        cur = anchors[k].cos_phi + 1j * anchors[k].sin_phi
        # 全局相位差 = arg(mean(ref * conj(cur)))
        rot = np.angle(np.mean(ref_complex * np.conj(cur)))
        aligned_complex = cur * np.exp(1j * rot)
        aligned[k, :, 0] = aligned_complex.real
        aligned[k, :, 1] = aligned_complex.imag
    return aligned


def _circular_variance(aligned: np.ndarray) -> np.ndarray:
    """1 − |mean(unit phasors)| across the anchor axis.

    Args:
        aligned: ``[K, T, 2]`` (cos, sin) per anchor.

    Returns:
        ``[T]`` circular variance in ``[0, 1]``. 0 = perfect agreement,
        1 = uniform disagreement.
    """
    mean_complex = aligned[..., 0].mean(axis=0) + 1j * aligned[..., 1].mean(axis=0)
    R = np.abs(mean_complex)        # mean resultant length per frame
    return 1.0 - R                  # circular variance


def multi_region_consensus(
    images: np.ndarray,                # [T, H, W]
    wire_masks: np.ndarray = None,     # [T, H, W] or None
    *,
    fs: float = 15.0,
    band_hz: Tuple[float, float] = (0.15, 0.50),
    bp_order: int = 4,
    grid: Tuple[int, int] = (2, 2),
    grid_overlap: float = 0.10,
    quality_threshold: float = 0.50,
    min_valid_anchors: int = 3,
) -> dict:
    """Run B.1+B.2 over a spatial grid of anchors and fuse their outputs.

    Args:
        images: ``[T, H, W]`` greyscale frames (uint8 or float).
        wire_masks: ``[T, H, W]`` binary wire masks (1=wire). If ``None``,
            the full image is treated as background — useful before stage 1
            is trained.
        fs: sampling rate (Hz).
        band_hz: respiratory band edges. Default (0.15, 0.50) covers
            9-30 BPM.
        bp_order: Butterworth order.
        grid: ``(grid_h, grid_w)`` anchor grid. Default 2×2 = 4 anchors.
        grid_overlap: fractional overlap between adjacent tiles (0.10 = 10%).
        quality_threshold: mean-resultant-length threshold above which a
            frame is marked as ``valid``. Lower values keep more frames at
            the cost of noisier labels.
        min_valid_anchors: a frame is also marked invalid if fewer than
            this many anchors had a valid B.1 increment estimate.

    Returns:
        Dict with keys::

            cos_phi[T], sin_phi[T]:      consensus phase
            amplitude[T]:                 consensus envelope (median across anchors)
            anchor_quality[T]:            1 − circular variance, in [0, 1]
            valid_mask[T]:                bool, after threshold + min_valid check
            per_anchor:                   list[AnchorEstimate], for diagnostics
            grid_rois:                    list[(y0,y1,x0,x1)]
    """
    if images.ndim != 3:
        raise ValueError(f"images must be [T, H, W]; got {images.shape}")
    T, H, W = images.shape

    rois = _grid_rois(H, W, grid[0], grid[1], overlap=grid_overlap)
    anchors: List[AnchorEstimate] = []
    for roi in rois:
        anchors.append(_estimate_one_anchor(
            images, wire_masks, roi, fs=fs, band=band_hz, bp_order=bp_order,
        ))

    aligned = _circular_align(anchors)                      # [K, T, 2]
    cos_consensus = aligned[..., 0].mean(axis=0)             # [T]
    sin_consensus = aligned[..., 1].mean(axis=0)             # [T]
    # 单位化以保证 (cos² + sin²) = 1（多锚点平均后会偏离单位圆）
    norm = np.sqrt(cos_consensus ** 2 + sin_consensus ** 2)
    norm = np.maximum(norm, 1e-8)
    cos_consensus = (cos_consensus / norm).astype(np.float32)
    sin_consensus = (sin_consensus / norm).astype(np.float32)

    # 振幅取跨锚点中位数（鲁棒于单个区域噪声）
    amps = np.stack([a.amplitude for a in anchors], axis=0)  # [K, T]
    amplitude = np.median(amps, axis=0).astype(np.float32)

    quality = (1.0 - _circular_variance(aligned)).astype(np.float32)  # [T]

    # 有效帧数（B.1 没回退为零的锚点数）
    delta_valid_stack = np.stack([a.delta_valid for a in anchors], axis=0)  # [K, T] bool
    valid_anchor_count = delta_valid_stack.sum(axis=0)        # [T]
    valid_mask = (
        (quality >= quality_threshold)
        & (valid_anchor_count >= min_valid_anchors)
    )

    return {
        "cos_phi": cos_consensus,
        "sin_phi": sin_consensus,
        "amplitude": amplitude,
        "anchor_quality": quality,
        "valid_mask": valid_mask,
        "per_anchor": anchors,
        "grid_rois": rois,
    }
