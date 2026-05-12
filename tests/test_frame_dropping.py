"""Tests for frame-drop simulation + Δt embedding (paper §6.2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from freqdec_gwnet.utils.frame_dropping import (
    DropPlan,
    apply_drop_plan,
    concat_delta_t_embedding,
    downsample_fps,
    random_drop_rate,
    remap_reference_index,
)


# ---------------------------------------------------------------------------
# random_drop_rate
# ---------------------------------------------------------------------------


def test_random_drop_rate_keeps_index_0_by_default():
    plan = random_drop_rate(T=100, drop_rate=0.5, seed=0)
    # Frame 0 必须被保留以便后续 cumsum
    assert plan.keep_mask[0]
    assert plan.kept_indices[0] == 0


def test_random_drop_rate_actual_close_to_requested():
    plan = random_drop_rate(T=2000, drop_rate=0.3, seed=42)
    # 大样本：actual ≈ requested ± 5%
    assert abs(plan.actual_drop_rate - 0.3) < 0.05


def test_random_drop_rate_seedable():
    a = random_drop_rate(T=100, drop_rate=0.3, seed=7)
    b = random_drop_rate(T=100, drop_rate=0.3, seed=7)
    assert np.array_equal(a.kept_indices, b.kept_indices)


def test_random_drop_rate_min_kept_4():
    """即使 drop_rate=0.99 也至少保留 4 帧（下游能算 phase / drift）。"""
    plan = random_drop_rate(T=200, drop_rate=0.99, seed=3)
    assert plan.T_kept >= 4


def test_random_drop_rate_invalid_inputs():
    with pytest.raises(ValueError):
        random_drop_rate(T=100, drop_rate=1.0)
    with pytest.raises(ValueError):
        random_drop_rate(T=1, drop_rate=0.5)


# ---------------------------------------------------------------------------
# downsample_fps
# ---------------------------------------------------------------------------


def test_downsample_fps_factor_2():
    plan = downsample_fps(T=20, factor=2)
    # 期望保留索引 0, 2, 4, ..., 18 → 10 帧
    expected = np.arange(0, 20, 2)
    assert np.array_equal(plan.kept_indices, expected)
    assert plan.T_kept == 10


def test_downsample_fps_factor_4():
    plan = downsample_fps(T=20, factor=4)
    expected = np.arange(0, 20, 4)
    assert np.array_equal(plan.kept_indices, expected)


def test_downsample_fps_invalid():
    with pytest.raises(ValueError):
        downsample_fps(T=20, factor=1)
    with pytest.raises(ValueError):
        downsample_fps(T=2, factor=4)


# ---------------------------------------------------------------------------
# delta_t computation
# ---------------------------------------------------------------------------


def test_delta_t_for_uniform_factor_is_constant():
    plan = downsample_fps(T=20, factor=2)
    # delta_t_frames[0] = 0, the rest should be 2
    assert plan.delta_t_frames[0] == 0
    assert (plan.delta_t_frames[1:] == 2).all()
    # delta_t_norm = delta_t / median(>0) = 2 / 2 = 1
    assert (plan.delta_t_norm[1:] == 1.0).all()


def test_delta_t_norm_handles_irregular_intervals():
    """Random drop produces irregular Δt; norm should center around 1."""
    plan = random_drop_rate(T=300, drop_rate=0.4, seed=11)
    nonzero = plan.delta_t_frames[plan.delta_t_frames > 0]
    if nonzero.size > 4:
        median = float(np.median(nonzero))
        nz_norm = plan.delta_t_norm[plan.delta_t_frames > 0]
        # 至少有一个 norm = 1.0（最常见的 Δt 就是 median）
        assert np.allclose(nz_norm.min(), 1.0 / median * nonzero.min(), rtol=1e-4)


# ---------------------------------------------------------------------------
# apply_drop_plan
# ---------------------------------------------------------------------------


def test_apply_drop_plan_index_select():
    plan = downsample_fps(T=10, factor=2)              # keep 0, 2, 4, 6, 8
    seq = torch.arange(10).unsqueeze(-1).float()        # [10, 1]
    out = apply_drop_plan(seq, plan, dim=0)
    assert out.shape == (5, 1)
    assert torch.equal(out.flatten().long(), torch.tensor([0, 2, 4, 6, 8]))


def test_apply_drop_plan_works_on_image_tensor():
    plan = downsample_fps(T=8, factor=2)
    images = torch.randn(8, 1, 32, 32)
    out = apply_drop_plan(images, plan, dim=0)
    assert out.shape == (4, 1, 32, 32)
    assert torch.equal(out[0], images[0])
    assert torch.equal(out[1], images[2])


# ---------------------------------------------------------------------------
# remap_reference_index
# ---------------------------------------------------------------------------


def test_remap_reference_index_kept_frame():
    plan = downsample_fps(T=20, factor=2)              # keep 0, 2, 4, ..., 18
    # 原参考 = 4 → 在 kept 里位于索引 2
    assert remap_reference_index(plan, 4) == 2


def test_remap_reference_index_dropped_frame_picks_next_kept():
    plan = downsample_fps(T=20, factor=2)              # keep 0, 2, 4, ..., 18
    # 原参考 = 5 (被 drop 了) → 取下一个 kept = 6 → kept 内索引 3
    assert remap_reference_index(plan, 5) == 3


def test_remap_reference_index_past_last_kept():
    plan = downsample_fps(T=10, factor=2)              # keep 0, 2, 4, 6, 8
    # 原参考 = 9 (drop 了，且后面没 kept) → 退回最后一帧
    assert remap_reference_index(plan, 9) == plan.T_kept - 1


# ---------------------------------------------------------------------------
# concat_delta_t_embedding
# ---------------------------------------------------------------------------


def test_concat_delta_t_embedding_shape():
    plan = downsample_fps(T=10, factor=2)
    z = torch.randn(2, plan.T_kept, 8)
    z_with_dt = concat_delta_t_embedding(z, plan)
    assert z_with_dt.shape == (2, plan.T_kept, 9)
    # 最后一列 = delta_t_norm
    assert torch.allclose(
        z_with_dt[0, :, -1].cpu(),
        torch.from_numpy(plan.delta_t_norm),
        atol=1e-5,
    )


def test_concat_delta_t_embedding_rejects_mismatch():
    plan = downsample_fps(T=10, factor=2)
    bad = torch.randn(2, plan.T_kept + 1, 4)            # T 不一致
    with pytest.raises(ValueError, match="T_kept"):
        concat_delta_t_embedding(bad, plan)


def test_concat_delta_t_embedding_rejects_wrong_rank():
    plan = downsample_fps(T=10, factor=2)
    with pytest.raises(ValueError, match=r"\[B, T, D\]"):
        concat_delta_t_embedding(torch.randn(2, plan.T_kept), plan)
