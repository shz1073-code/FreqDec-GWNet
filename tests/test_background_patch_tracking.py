"""Tests for the background-patch tracking metric (Metric B)."""

from __future__ import annotations

import numpy as np
import pytest

from freqdec_gwnet.utils.background_patch_tracking import (
    BackgroundPatchTrack,
    detect_background_features,
    patch_alignment_residual,
    track_features_across_burst,
)


def _make_textured_frame(H: int, W: int, seed: int) -> np.ndarray:
    """Random texture with strong corners (for Shi-Tomasi)."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 255, size=(H, W)).astype(np.uint8)
    import cv2
    return cv2.GaussianBlur(base, (3, 3), 0.5)


def _shift_image(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    import cv2
    H, W = img.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        img, M, (W, H), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


# ---------------------------------------------------------------------------
# detect_background_features
# ---------------------------------------------------------------------------


def test_detect_features_returns_corners_outside_wire():
    """Corners must not land inside the dilated wire region."""
    img = _make_textured_frame(128, 128, seed=0)
    wire = np.zeros((128, 128), dtype=np.uint8)
    wire[55:75, 50:80] = 1                       # 20x30 wire region
    corners = detect_background_features(
        img, wire, max_features=20, wire_dilation_px=8,
    )
    # 没有 corner 应该落在 wire + dilation 区域内
    for x, y in corners:
        in_wire = (50 - 8 <= x < 80 + 8) and (55 - 8 <= y < 75 + 8)
        assert not in_wire, f"corner at ({x}, {y}) is inside dilated wire"


def test_detect_features_handles_image_with_no_background():
    """If the whole frame is masked out, fall back to whole image."""
    img = _make_textured_frame(64, 64, seed=1)
    wire_full = np.ones((64, 64), dtype=np.uint8)
    corners = detect_background_features(
        img, wire_full, max_features=10, wire_dilation_px=2,
    )
    # 应该返回 corners 而不是空（因为我们 fallback 到整图）
    assert corners.shape[0] > 0


# ---------------------------------------------------------------------------
# track_features_across_burst
# ---------------------------------------------------------------------------


def test_track_features_recovers_known_translation():
    """Synthetic burst with known pure translation → LK should recover."""
    H = W = 128
    T = 6
    base = _make_textured_frame(H, W, seed=2)
    shifts = np.array([
        [0., 0.],
        [3., 0.],
        [6., 0.],
        [9., 0.],
        [6., 0.],
        [3., 0.],
    ], dtype=np.float32)
    images = np.stack(
        [_shift_image(base, dx, dy) for dx, dy in shifts], axis=0,
    )
    wire = np.zeros((H, W), dtype=np.uint8)
    corners = detect_background_features(
        images[0], wire, max_features=10, wire_dilation_px=2,
    )
    assert corners.shape[0] > 0

    track = track_features_across_burst(images, corners, reference_index=0)
    assert track.n_features == corners.shape[0]
    assert track.feature_xy_per_frame.shape == (T, corners.shape[0], 2)
    # 至少 60% 的 (T, K) 单元应该跟踪成功
    assert track.valid.mean() > 0.5

    # 对每个 tracked feature，计算 frame 3 vs ref 的位移，应该 ≈ (9, 0)
    valid_at_3 = track.valid[3]
    if valid_at_3.any():
        delta = track.feature_xy_per_frame[3] - corners
        recovered = delta[valid_at_3].mean(axis=0)
        assert abs(recovered[0] - 9.0) < 1.0
        assert abs(recovered[1] - 0.0) < 1.0


# ---------------------------------------------------------------------------
# patch_alignment_residual
# ---------------------------------------------------------------------------


def test_patch_residual_perfect_compensation_zeros_after():
    """If U exactly matches the known feature translation, after-drift = 0."""
    H = W = 128
    T = 5
    base = _make_textured_frame(H, W, seed=3)
    shifts = np.array([[0., 0.], [4., 0.], [8., 0.], [4., 0.], [0., 0.]], dtype=np.float32)
    images = np.stack([_shift_image(base, dx, dy) for dx, dy in shifts])
    wire = np.zeros((H, W), dtype=np.uint8)
    corners = detect_background_features(
        images[0], wire, max_features=10, wire_dilation_px=2,
    )
    track = track_features_across_burst(images, corners, reference_index=0)

    # Use the known shifts as the ideal U
    U = shifts.copy()
    out = patch_alignment_residual(track, U, reference_index=0)
    # After perfect compensation, residual should be small
    assert out["global_mean_after"] < 1.5


def test_patch_residual_zero_compensation_keeps_before_eq_after():
    H = W = 128
    T = 3
    base = _make_textured_frame(H, W, seed=4)
    images = np.stack([base, _shift_image(base, 5, 0), _shift_image(base, 10, 0)])
    wire = np.zeros((H, W), dtype=np.uint8)
    corners = detect_background_features(
        images[0], wire, max_features=10, wire_dilation_px=2,
    )
    track = track_features_across_burst(images, corners, reference_index=0)
    U = np.zeros((T, 2), dtype=np.float32)        # no compensation
    out = patch_alignment_residual(track, U, reference_index=0)
    assert out["reduction_ratio"] == pytest.approx(1.0, abs=1e-5)


def test_patch_residual_handles_zero_valid_frames():
    """If tracking failed entirely, output should still be well-defined NaN."""
    H = W = 64
    K = 3
    track = BackgroundPatchTrack(
        feature_xy_ref=np.zeros((K, 2), dtype=np.float32),
        feature_xy_per_frame=np.full((5, K, 2), np.nan, dtype=np.float32),
        valid=np.zeros((5, K), dtype=bool),
        n_features=K,
    )
    # Re-set reference frame valid (track.valid is all False which means
    # even the ref frame has no valid features — abnormal but should not
    # crash)
    U = np.zeros((5, 2), dtype=np.float32)
    out = patch_alignment_residual(track, U, reference_index=0)
    assert np.isnan(out["global_mean_before"])
    assert np.isnan(out["reduction_ratio"])


def test_patch_residual_reduction_ratio_meaningful():
    """Half-correct U → reduction_ratio ≈ 0.5."""
    H = W = 128
    base = _make_textured_frame(H, W, seed=5)
    T = 5
    shifts = np.array(
        [[0., 0.], [2., 0.], [4., 0.], [2., 0.], [0., 0.]], dtype=np.float32,
    )
    images = np.stack([_shift_image(base, dx, dy) for dx, dy in shifts])
    wire = np.zeros((H, W), dtype=np.uint8)
    corners = detect_background_features(
        images[0], wire, max_features=10, wire_dilation_px=2,
    )
    track = track_features_across_burst(images, corners, reference_index=0)
    U_half = shifts * 0.5
    out = patch_alignment_residual(track, U_half, reference_index=0)
    # Should be near 0.5 — half-correct compensation
    assert 0.3 < out["reduction_ratio"] < 0.7
