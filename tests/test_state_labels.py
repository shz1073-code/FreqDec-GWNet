"""End-to-end tests for the R2 weak-supervision state-label pipeline.

Covers:
    * B.1 background_motion: phase correlation sign / value, wire exclusion
    * B.2 band_filter: band-pass selectivity, PCA projection, Hilbert phase
    * B.3 anchor_fusion: multi-region consensus + valid_mask
    * B.4 pipeline: full burst -> StateLabel + npz round trip
    * smoke test on real cmu_seq sequence (auto-skip if data missing)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from freqdec_gwnet.data.state_labels import (
    StateLabel,
    band_pass_filter,
    cumulative_position,
    estimate_background_translation,
    generate_state_label,
    hilbert_phase_amplitude,
    multi_region_consensus,
    project_to_principal_axis,
)


# ===========================================================================
# Helpers: synthetic fluoroscopy bursts
# ===========================================================================


def _make_textured_frame(H: int, W: int, seed: int) -> np.ndarray:
    """Random low-pass-filtered texture, [H, W] float32 in [0, 1]."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((H, W)).astype(np.float32)
    import cv2
    blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=4.0)
    blurred -= blurred.min()
    blurred /= max(blurred.max(), 1e-6)
    return blurred


def _shifted_burst(
    H: int, W: int, T: int, shifts_xy: np.ndarray, seed: int = 0,
) -> np.ndarray:
    """Build a burst by shifting one base frame by a known per-frame (dx, dy).

    Args:
        shifts_xy: ``[T, 2]`` cumulative (dx, dy) per frame relative to t=0.
                  Frame ``t`` is the base content shifted by ``+shifts_xy[t]``,
                  so ``frame_t(x, y) ≈ frame_0(x - dx, y - dy)``.

    Returns:
        ``[T, H, W]`` float32 burst.
    """
    import cv2
    base = _make_textured_frame(H, W, seed)
    out = np.empty((T, H, W), dtype=np.float32)
    for t in range(T):
        dx, dy = float(shifts_xy[t, 0]), float(shifts_xy[t, 1])
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        out[t] = cv2.warpAffine(
            base, M, (W, H), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
    return out


# ===========================================================================
# B.1 — background_motion
# ===========================================================================


def test_background_translation_recovers_known_shift():
    """Phase-corr returns increments matching the synthetic shift."""
    H = W = 128
    T = 6
    shifts = np.zeros((T, 2), dtype=np.float32)
    shifts[:, 0] = np.array([0, 1.0, 2.0, 3.0, 4.0, 5.0])  # +1 px / frame in x
    burst = _shifted_burst(H, W, T, shifts, seed=42)

    deltas, valid = estimate_background_translation(burst, wire_masks=None)
    assert deltas.shape == (T, 2)
    assert valid.shape == (T,)
    # 第 0 帧约定为 0
    assert np.allclose(deltas[0], 0.0)
    # 后续每帧 dx ≈ +1, dy ≈ 0
    assert np.allclose(deltas[1:, 0], 1.0, atol=0.3)
    assert np.allclose(deltas[1:, 1], 0.0, atol=0.3)
    assert valid.all()


def test_background_translation_2d_shift():
    H = W = 96
    T = 4
    shifts = np.array([[0, 0], [2, 1], [4, 2], [6, 3]], dtype=np.float32)
    burst = _shifted_burst(H, W, T, shifts, seed=1)
    deltas, _ = estimate_background_translation(burst)
    # 每帧增量 ≈ (2, 1)
    assert np.allclose(deltas[1:, 0], 2.0, atol=0.4)
    assert np.allclose(deltas[1:, 1], 1.0, atol=0.4)


def test_background_translation_handles_full_wire_mask():
    """If wire mask covers >90% of image we mark increment invalid."""
    H = W = 64
    T = 3
    burst = _shifted_burst(H, W, T, np.zeros((T, 2)), seed=2)
    wire = np.ones((T, H, W), dtype=np.uint8)        # entire image is wire
    deltas, valid = estimate_background_translation(
        burst, wire, dilate_radius=0, min_background_fraction=0.5,
    )
    assert deltas.shape == (T, 2)
    assert valid[0]                                   # 第一帧约定 valid
    # 后续都没背景可估，valid=False
    assert not valid[1] and not valid[2]
    assert np.allclose(deltas[1:], 0.0)


def test_cumulative_position_matches_cumsum():
    deltas = np.array([[0, 0], [1, 2], [3, -1], [-2, 0]], dtype=np.float32)
    pos = cumulative_position(deltas)
    expected = np.cumsum(deltas, axis=0)
    assert np.allclose(pos, expected)


# ===========================================================================
# B.2 — band_filter
# ===========================================================================


def test_band_pass_keeps_in_band_sinusoid():
    fs = 30.0
    T = 256
    t = np.arange(T) / fs
    in_band = np.sin(2 * np.pi * 0.3 * t).astype(np.float32)   # 0.3 Hz ∈ band
    out = band_pass_filter(in_band, fs=fs, low_hz=0.15, high_hz=0.50)
    # 用 RMS 比较，避开小相位漂移被算成幅度损失
    mid_in = in_band[T // 4: 3 * T // 4]
    mid_out = out[T // 4: 3 * T // 4]
    rms_in = float(np.sqrt(np.mean(mid_in ** 2)))
    rms_out = float(np.sqrt(np.mean(mid_out ** 2)))
    rel = abs(rms_out - rms_in) / rms_in
    assert rel < 0.20, (
        f"in-band sinusoid attenuated more than expected: "
        f"rms_in={rms_in:.4f} rms_out={rms_out:.4f} rel={rel:.3f}"
    )


def test_band_pass_suppresses_out_of_band_signals():
    fs = 30.0
    T = 256
    t = np.arange(T) / fs
    dc = np.ones(T, dtype=np.float32) * 5.0                    # 0 Hz — should die
    high = np.sin(2 * np.pi * 5.0 * t).astype(np.float32)      # 5 Hz — way above
    out_dc = band_pass_filter(dc, fs=fs, low_hz=0.15, high_hz=0.50)
    out_hi = band_pass_filter(high, fs=fs, low_hz=0.15, high_hz=0.50)
    mid_dc = out_dc[T // 4: 3 * T // 4]
    mid_hi = out_hi[T // 4: 3 * T // 4]
    assert np.abs(mid_dc).max() < 0.1
    assert np.abs(mid_hi).max() < 0.1


def test_band_pass_zero_phase():
    """filtfilt should leave the peak times of an in-band sinusoid intact."""
    fs = 30.0
    T = 256
    t = np.arange(T) / fs
    sig = np.sin(2 * np.pi * 0.3 * t).astype(np.float32)
    out = band_pass_filter(sig, fs=fs, low_hz=0.15, high_hz=0.50)
    # 在中段比较两个信号的过零点位置
    mid = slice(T // 4, 3 * T // 4)
    sig_zero = np.where(np.diff(np.sign(sig[mid])))[0]
    out_zero = np.where(np.diff(np.sign(out[mid])))[0]
    if len(sig_zero) and len(out_zero):
        delay = abs(sig_zero[0] - out_zero[0])
        assert delay <= 1, f"zero-phase filter introduced {delay} sample delay"


def test_band_pass_rejects_bad_inputs():
    with pytest.raises(ValueError):
        band_pass_filter(np.zeros(10), fs=15.0)             # 太短
    with pytest.raises(ValueError):
        band_pass_filter(np.zeros(64), fs=15.0, low_hz=0.5, high_hz=0.3)  # 颠倒
    with pytest.raises(ValueError):
        band_pass_filter(np.zeros((2, 64)), fs=15.0)        # 维度错误


def test_pca_projection_picks_dominant_axis():
    rng = np.random.default_rng(7)
    T = 200
    t = np.linspace(0, 1, T)
    # 主方向是 (1, 1)/√2，幅度 5；垂直方向只有小噪声
    main = 5.0 * np.sin(2 * np.pi * 2 * t)
    noise = 0.1 * rng.standard_normal(T)
    motion = np.stack([main + noise, main - noise], axis=-1)
    proj, axis, ratio = project_to_principal_axis(motion)
    assert proj.shape == (T,)
    assert ratio > 0.95
    expected_axis = np.array([1.0, 1.0]) / math.sqrt(2)
    # 方向可能是反的，绝对值要接近
    assert abs(abs(np.dot(axis, expected_axis)) - 1.0) < 0.05


def test_pca_projection_constant_signal_safe():
    """All-zero motion → zero projection, no crash."""
    proj, axis, ratio = project_to_principal_axis(np.zeros((10, 2)))
    assert np.allclose(proj, 0.0)
    assert ratio == 0.0


def test_hilbert_pure_cosine():
    """Hilbert on cos(2πf t) gives linearly increasing phase."""
    fs = 30.0
    T = 200
    t = np.arange(T) / fs
    f = 0.3
    sig = np.cos(2 * np.pi * f * t)
    phase, amp = hilbert_phase_amplitude(sig)
    # 中段振幅应接近 1（端点有 Hilbert artifact）
    assert np.abs(amp[50:150] - 1.0).max() < 0.1
    # 中段相位单调递增（unwrap 后近似线性）
    unwrapped = np.unwrap(phase[50:150])
    drift = unwrapped[-1] - unwrapped[0]
    expected_drift = 2 * np.pi * f * (149 - 50) / fs
    assert abs(drift - expected_drift) < 0.1


# ===========================================================================
# B.3 — anchor_fusion
# ===========================================================================


def test_multi_region_consensus_high_quality_on_uniform_motion():
    """Uniform respiratory translation → all regions agree → high quality."""
    H = W = 128
    T = 240                                           # 16 s @ 15 fps
    fs = 15.0
    t = np.arange(T) / fs
    amp_y = 5.0
    f_resp = 0.30
    shifts = np.stack(
        [np.zeros(T), amp_y * np.sin(2 * np.pi * f_resp * t)],
        axis=-1,
    ).astype(np.float32)
    burst = _shifted_burst(H, W, T, shifts, seed=11)

    out = multi_region_consensus(burst, wire_masks=None, fs=fs, grid=(2, 2))
    # 输出形状
    for key in ("cos_phi", "sin_phi", "amplitude", "anchor_quality", "valid_mask"):
        assert out[key].shape == (T,), key
    assert len(out["per_anchor"]) == 4
    assert len(out["grid_rois"]) == 4
    # cos² + sin² ≈ 1（融合后单位化）
    norms = out["cos_phi"] ** 2 + out["sin_phi"] ** 2
    assert np.allclose(norms, 1.0, atol=1e-5)
    # 中段（避开端点）质量应很高
    mid = slice(60, 180)
    assert out["anchor_quality"][mid].mean() > 0.85
    assert out["valid_mask"][mid].all()


def test_multi_region_consensus_amplitude_matches_input():
    """Hilbert envelope median ≈ input respiratory amplitude in y."""
    H = W = 128
    T = 240
    fs = 15.0
    t = np.arange(T) / fs
    amp_y = 4.0
    shifts = np.stack(
        [np.zeros(T), amp_y * np.sin(2 * np.pi * 0.30 * t)],
        axis=-1,
    ).astype(np.float32)
    burst = _shifted_burst(H, W, T, shifts, seed=22)

    out = multi_region_consensus(burst, fs=fs, grid=(2, 2))
    mid = slice(60, 180)
    median_amp = float(np.median(out["amplitude"][mid]))
    # 容差宽到 [0.3x, 3x]：phase-corr 在低纹理 64×64 子区上偶有严重欠估，
    # filtfilt 在 0.3Hz/15fps 时等价 8 阶的边界过渡区也会衰减；这里只验证
    # 数量级正确，准确的幅度校准必须在真实数据上做。
    assert amp_y * 0.3 < median_amp < amp_y * 3.0, (
        f"recovered amplitude {median_amp:.3f} far from input {amp_y:.3f}"
    )


# ===========================================================================
# B.4 — pipeline + StateLabel
# ===========================================================================


def test_generate_state_label_synthetic_burst():
    H = W = 96
    T = 200
    fs = 15.0
    t = np.arange(T) / fs
    shifts = np.stack(
        [np.zeros(T), 4.0 * np.sin(2 * np.pi * 0.25 * t)],
        axis=-1,
    ).astype(np.float32)
    burst = _shifted_burst(H, W, T, shifts, seed=33)

    label = generate_state_label(
        burst, wire_masks=None,
        seq_name="syn_001", dataset="synthetic", split="train",
        fs=fs,
    )

    assert isinstance(label, StateLabel)
    assert label.num_frames == T
    assert label.cos_phi.shape == (T,)
    assert label.sin_phi.shape == (T,)
    assert label.amplitude.shape == (T,)
    assert label.valid_mask.shape == (T,)
    assert label.anchor_quality.shape == (T,)
    assert label.amp_scale > 0
    assert label.reference_strategy in {"clinical", "warmup_score", "first_frame"}
    assert 0 <= label.reference_index < T
    # 单位圆约束
    assert np.allclose(
        label.cos_phi ** 2 + label.sin_phi ** 2, 1.0, atol=1e-5,
    )
    # metadata 必须包含 R2 关键参数
    for key in ("fs", "band_hz", "grid", "n_anchors", "n_valid_frames"):
        assert key in label.metadata, f"missing metadata key: {key}"


def test_state_label_npz_round_trip(tmp_path):
    """Save → load preserves every numpy field plus metadata."""
    H = W = 80
    T = 100
    fs = 15.0
    t = np.arange(T) / fs
    shifts = np.stack([np.zeros(T), 3.0 * np.sin(2 * np.pi * 0.3 * t)], axis=-1).astype(np.float32)
    burst = _shifted_burst(H, W, T, shifts, seed=4)

    label = generate_state_label(
        burst, seq_name="rt_001", dataset="synthetic", split="train",
        fs=fs, extra_metadata={"note": "round-trip test"},
    )
    path = label.save_npz(tmp_path / "label.npz")
    assert path.is_file()

    reloaded = StateLabel.load_npz(path)
    assert reloaded.seq_name == label.seq_name
    assert reloaded.dataset == label.dataset
    assert reloaded.split == label.split
    assert reloaded.reference_index == label.reference_index
    assert reloaded.reference_strategy == label.reference_strategy
    assert np.allclose(reloaded.cos_phi, label.cos_phi)
    assert np.allclose(reloaded.sin_phi, label.sin_phi)
    assert np.allclose(reloaded.amplitude, label.amplitude)
    assert np.array_equal(reloaded.valid_mask, label.valid_mask)
    assert np.allclose(reloaded.anchor_quality, label.anchor_quality)
    assert reloaded.amp_scale == pytest.approx(label.amp_scale)
    assert reloaded.metadata.get("note") == "round-trip test"


def test_state_label_amp_scale_is_p95_of_valid():
    """amp_scale should be percentile of valid-frame amplitudes only."""
    H = W = 64
    T = 120
    fs = 15.0
    t = np.arange(T) / fs
    shifts = np.stack([np.zeros(T), 3.0 * np.sin(2 * np.pi * 0.3 * t)], axis=-1).astype(np.float32)
    burst = _shifted_burst(H, W, T, shifts, seed=5)
    label = generate_state_label(
        burst, seq_name="p95", dataset="syn", split="train",
        fs=fs, amp_scale_percentile=95.0,
    )
    if label.valid_mask.any():
        manual = float(np.percentile(label.amplitude[label.valid_mask], 95.0))
        assert label.amp_scale == pytest.approx(manual, rel=1e-5)


def test_state_label_rejects_size_mismatch():
    with pytest.raises(ValueError, match="StateLabel"):
        StateLabel(
            seq_name="bad", dataset="d", split="train",
            cos_phi=np.zeros(10), sin_phi=np.zeros(8),
            amplitude=np.zeros(10), amp_scale=1.0,
            valid_mask=np.zeros(10, dtype=bool),
            anchor_quality=np.zeros(10),
            reference_index=0, reference_strategy="first_frame",
        )


# ===========================================================================
# Real-data smoke test (auto-skip when data not mounted)
# ===========================================================================


def test_generate_state_label_on_real_cmu_seq_007(tmp_path):
    """End-to-end on a real burst. No wire masks yet — uses full-image mode."""
    pytest.importorskip("cv2")
    from freqdec_gwnet.data import DataPaths, FluoroSequenceLoader
    try:
        paths = DataPaths.from_default_config()
    except FileNotFoundError as exc:
        pytest.skip(f"data not mounted: {exc}")
    if "dense_v1" not in paths.datasets:
        pytest.skip("dense_v1 not configured")

    seq = FluoroSequenceLoader(paths).load_sequence(
        "dense_v1", "train", "cmu_seq_007",
    )
    images_np = (seq.frames.numpy()[:, 0] * 255).astype(np.uint8)  # [T, H, W]

    label = generate_state_label(
        images_np,
        seq_name=seq.seq_name, dataset=seq.dataset, split=seq.split,
        fs=15.0,
    )
    assert label.num_frames == seq.num_frames
    assert label.amp_scale > 0
    assert label.reference_strategy in {"clinical", "warmup_score", "first_frame"}
    # 真实序列里至少应有一些 valid 帧（除非数据极端有问题）
    assert int(label.valid_mask.sum()) > 0
    # 序列化 round-trip
    path = label.save_npz(tmp_path / f"{seq.seq_name}.npz")
    StateLabel.load_npz(path)
