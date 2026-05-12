"""B.1 — per-frame background-only translation via phase correlation.

The wire mask (dilated to absorb a safety margin) is removed from both frames
before phase correlation so that the active push of the catheter does not
contaminate the physiological background estimate. The per-frame increment
``Δbg_motion[t] = bg(t) − bg(t−1)`` is what phase correlation returns; the
cumulative position ``bg_motion[t]`` is recovered via :func:`cumulative_position`.

Sign convention (pinned by tests):
    ``estimate_background_translation`` returns ``shift[t] = (dx, dy)`` such
    that frame ``t`` looks like frame ``t-1`` translated by ``+shift[t]`` —
    i.e. ``f_t(x, y) ≈ f_{t-1}(x - dx, y - dy)``.

This matches OpenCV's :func:`cv2.phaseCorrelate` documented behavior. We
verify it explicitly in :mod:`tests.test_state_labels`.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_float32_grayscale(images: np.ndarray) -> np.ndarray:
    """Coerce ``[T,H,W]`` images to ``float32`` in ``[0, 1]``.

    Phase correlation expects float32; uint8 inputs are normalized.
    """
    if images.ndim != 3:
        raise ValueError(
            f"images must be [T, H, W]; got shape {images.shape}"
        )
    if images.dtype == np.uint8:
        return images.astype(np.float32) / 255.0
    if not np.issubdtype(images.dtype, np.floating):
        raise TypeError(
            f"images dtype {images.dtype} not supported; pass uint8 or float"
        )
    return images.astype(np.float32, copy=False)


def _dilate_wire_masks(
    wire_masks: np.ndarray, dilate_radius: int
) -> np.ndarray:
    """Return ``[T,H,W]`` uint8 background-only ROI masks (1=background)."""
    if wire_masks.shape != wire_masks.shape:
        raise ValueError("wire_masks shape mismatch")  # pragma: no cover
    if dilate_radius < 0:
        raise ValueError("dilate_radius must be >= 0")
    T, H, W = wire_masks.shape
    if dilate_radius == 0:
        return (wire_masks <= 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilate_radius + 1, 2 * dilate_radius + 1),
    )
    out = np.empty((T, H, W), dtype=np.uint8)
    for t in range(T):
        wire = (wire_masks[t] > 0).astype(np.uint8)
        dilated = cv2.dilate(wire, kernel)
        out[t] = 1 - dilated      # invert: background = 1
    return out


def _hann_window(H: int, W: int) -> np.ndarray:
    """2D Hann window used to suppress spectral leakage in phase correlation."""
    wy = np.hanning(H).astype(np.float32)
    wx = np.hanning(W).astype(np.float32)
    return np.outer(wy, wx)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_background_translation(
    images: np.ndarray,                    # [T, H, W] uint8 or float in [0,1]
    wire_masks: Optional[np.ndarray] = None,  # [T, H, W] binary; None = full image
    *,
    dilate_radius: int = 16,
    min_background_fraction: float = 0.10,
    use_hann_window: bool = True,
) -> np.ndarray:
    """Estimate per-frame background translation via masked phase correlation.

    Args:
        images: ``[T, H, W]`` greyscale frames.
        wire_masks: ``[T, H, W]`` binary wire masks (1 = wire). If ``None``,
            the entire frame is treated as background — useful for sanity
            checks before stage 1 is trained.
        dilate_radius: morphological dilation radius applied to the wire mask
            before exclusion. Defaults to 16 px (≈3% of a 512×512 image).
        min_background_fraction: if the joint-background ROI between two
            consecutive frames covers less than this fraction of the image,
            the increment is set to zero and the estimate is flagged as
            unreliable in the returned ``valid`` array.
        use_hann_window: whether to multiply the masked frames by a 2D Hann
            window before phase correlation. This suppresses ringing from
            the rectangular image boundary.

    Returns:
        ``Δbg[T, 2]`` per-frame translation increments in pixels (Δbg[0]=0)
        and a ``valid[T]`` boolean array marking frames where the estimate
        passed the joint-background-fraction check.

        Sign convention: ``frame_t(x, y) ≈ frame_{t-1}(x - dx, y - dy)``,
        i.e. ``shift[t]`` is what you would apply to frame ``t-1`` to make
        it overlap frame ``t``.
    """
    imgs = _to_float32_grayscale(images)
    T, H, W = imgs.shape

    if wire_masks is None:
        bg_masks = np.ones((T, H, W), dtype=np.uint8)
    else:
        if wire_masks.shape != imgs.shape:
            raise ValueError(
                f"wire_masks shape {wire_masks.shape} does not match "
                f"images shape {imgs.shape}"
            )
        bg_masks = _dilate_wire_masks(wire_masks, dilate_radius)

    deltas = np.zeros((T, 2), dtype=np.float32)
    valid = np.zeros(T, dtype=bool)
    valid[0] = True                     # 第一帧无前帧，约定 valid

    hann = _hann_window(H, W) if use_hann_window else np.ones((H, W), dtype=np.float32)

    img_size = H * W
    min_pixels = int(min_background_fraction * img_size)

    for t in range(1, T):
        # 联合背景：前后帧都要在背景里才算可比像素
        joint = bg_masks[t - 1] & bg_masks[t]
        if int(joint.sum()) < min_pixels:
            # 背景太少（导丝几乎覆盖全图）→ 这一帧增量不可信
            deltas[t] = 0.0
            valid[t] = False
            continue

        # 只在联合背景内做 phase correlation：mask 之外用各自帧的均值替换，
        # 减少边界伪影对 FFT 的影响。
        m = joint.astype(np.float32)
        f0 = imgs[t - 1]
        f1 = imgs[t]
        m0 = (f0 * m).sum() / m.sum()
        m1 = (f1 * m).sum() / m.sum()
        # f0 在背景区保留原值，非背景区填均值（这样 FFT 看到的是连续场而非阶跃）
        f0_norm = (f0 - m0) * m
        f1_norm = (f1 - m1) * m
        f0_norm *= hann
        f1_norm *= hann

        try:
            (dx, dy), _ = cv2.phaseCorrelate(
                np.ascontiguousarray(f0_norm),
                np.ascontiguousarray(f1_norm),
            )
        except cv2.error:
            # OpenCV 偶发数值不稳，回退为零增量
            deltas[t] = 0.0
            valid[t] = False
            continue

        deltas[t] = (dx, dy)
        valid[t] = True

    return deltas, valid


def cumulative_position(deltas: np.ndarray) -> np.ndarray:
    """Convert per-frame increments to absolute positions via cumulative sum.

    Args:
        deltas: ``[T, 2]`` increments with ``deltas[0] == 0`` (or any value;
            the function does not enforce the convention).

    Returns:
        ``[T, 2]`` positions with ``pos[0] = deltas[0]``.
    """
    if deltas.ndim != 2 or deltas.shape[1] != 2:
        raise ValueError(
            f"deltas must be [T, 2]; got shape {deltas.shape}"
        )
    return np.cumsum(deltas, axis=0).astype(np.float32, copy=False)
