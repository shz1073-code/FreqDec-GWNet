"""Tests for the inter-burst continuity sandbox."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from freqdec_gwnet.utils.inter_burst_continuity import (
    BurstChunkPlan,
    build_chunk_plan,
    run_inter_burst_continuity,
)


# ---------------------------------------------------------------------------
# build_chunk_plan
# ---------------------------------------------------------------------------


def test_build_chunk_plan_default_partition_balances_chunks():
    plan = build_chunk_plan(T_full=240, K=3, gap_frames=15)
    assert plan.K == 3
    # 240 - 2*15 = 210 → chunk_length = 70 per chunk
    assert all(L == 70 for L in plan.chunk_lengths)
    assert plan.kept_indices.size == 70 * 3
    # 跨 chunk 位置：原始索引应该跳过 gap
    assert plan.chunk_starts == [0, 85, 170]


def test_build_chunk_plan_delta_t_is_1_within_chunk_and_gap_at_boundary():
    plan = build_chunk_plan(T_full=150, K=2, gap_frames=20, chunk_length=50)
    # dt 内部 = 1，第一帧 = 0，跨 chunk = gap+1 = 21
    inside_dts = plan.delta_t_frames[1:50]
    assert (inside_dts == 1.0).all()
    assert plan.delta_t_frames[50] == pytest.approx(21.0)
    assert plan.delta_t_frames[0] == 0.0


def test_build_chunk_plan_boundary_indices():
    plan = build_chunk_plan(T_full=180, K=3, gap_frames=10, chunk_length=50)
    boundaries = plan.boundary_kept_indices
    # 三个 chunk → 两个边界
    assert len(boundaries) == 2
    # 第一边界：chunk 0 最后一帧（idx 49）→ chunk 1 第一帧（idx 50）
    assert boundaries[0] == (49, 50)
    assert boundaries[1] == (99, 100)


def test_build_chunk_plan_rejects_too_short_T_full():
    with pytest.raises(ValueError):
        build_chunk_plan(T_full=3, K=3)


def test_build_chunk_plan_rejects_negative_gap():
    with pytest.raises(ValueError):
        build_chunk_plan(T_full=200, K=2, gap_frames=-5)


def test_build_chunk_plan_rejects_too_aggressive_gap():
    # gap 太大让 chunk_length < 4 → 应该报错
    with pytest.raises(ValueError, match="chunk_length"):
        build_chunk_plan(T_full=20, K=3, gap_frames=10, chunk_length=2)


# ---------------------------------------------------------------------------
# Continuity protocols (no real ckpt — use untrained tiny model)
# ---------------------------------------------------------------------------


def _tiny_model():
    from freqdec_gwnet.models.freqdec_gwnet import FreqDecGWNet
    return FreqDecGWNet(
        in_channels=1, num_classes=1, width_mult=0.5,
        state_proj_dim=16, state_hidden_dim=16, state_lp_kernel=3,
        state_core_type="gru",                       # CPU 友好
        motion_hidden_dim=16,
    )


def test_run_inter_burst_returns_four_protocols():
    """Smoke test: all 4 protocols return ContinuityResult with proper shape."""
    model = _tiny_model().eval()
    images = torch.randn(120, 1, 32, 32)
    plan = build_chunk_plan(T_full=120, K=3, gap_frames=5, chunk_length=30)
    results = run_inter_burst_continuity(model, images, plan, device="cpu")
    assert set(results.keys()) == {"naive", "carry", "carry_dt", "oracle"}
    for proto, res in results.items():
        # T_kept = K * chunk_length = 90
        assert res.cos_phi.shape == (90,)
        assert res.sin_phi.shape == (90,)
        # K-1 = 2 boundary jumps
        assert res.boundary_phase_jumps.shape == (2,)
        # mean is finite when boundaries exist
        assert np.isfinite(res.mean_boundary_jump)


def test_oracle_boundary_jumps_are_bounded():
    """Oracle is the unchunked baseline; its boundary jumps should be in
    a sensible range (cos phi differences in [0, 2]). For an untrained
    model the absolute value relationship to other protocols is noise-
    dominated; the protocol comparison only becomes meaningful after the
    state branch is trained — that's the experiment the paper reports.
    """
    model = _tiny_model().eval()
    images = torch.randn(150, 1, 32, 32)
    plan = build_chunk_plan(T_full=150, K=3, gap_frames=10, chunk_length=40)
    results = run_inter_burst_continuity(model, images, plan, device="cpu")
    for proto, res in results.items():
        # cos_phi 在 untrained 模型下也应在 [-1.5, 1.5] 之内
        assert np.abs(res.cos_phi).max() < 1.5
        assert 0 <= res.mean_boundary_jump < 0.5, (
            f"{proto} boundary jump out of sanity range: "
            f"{res.mean_boundary_jump}"
        )


def test_carry_changes_chunk_n_plus_1_predictions_vs_naive():
    """carry 和 naive 在第 2/3 个 chunk 上应该给出不同的预测。

    第一个 chunk 完全相同（h₀=0 in both cases），后续因为 carry h 而分歧。
    """
    torch.manual_seed(0)
    model = _tiny_model().eval()
    images = torch.randn(150, 1, 32, 32)
    plan = build_chunk_plan(T_full=150, K=3, gap_frames=5, chunk_length=40)
    results = run_inter_burst_continuity(model, images, plan, device="cpu")
    # 第 0 个 chunk 内部 (前 40 帧) 应该一致
    n0_naive = results["naive"].cos_phi[:40]
    n0_carry = results["carry"].cos_phi[:40]
    assert np.allclose(n0_naive, n0_carry, atol=1e-4), \
        "first chunk should match exactly (both started h0=0)"
    # 第 1 个 chunk (40..80) 应该开始分歧
    n1_naive = results["naive"].cos_phi[40:80]
    n1_carry = results["carry"].cos_phi[40:80]
    diff = float(np.abs(n1_naive - n1_carry).max())
    assert diff > 1e-4, (
        f"second chunk should diverge from naive (carry h0≠0); diff={diff}"
    )


def test_carry_dt_differs_from_plain_carry_when_gap_is_long():
    """Δt 重缩放只在 dt > 1 时影响 z_t，所以 carry_dt vs carry 在跨 gap 处不同。"""
    model = _tiny_model().eval()
    images = torch.randn(150, 1, 32, 32)
    plan = build_chunk_plan(T_full=150, K=3, gap_frames=30, chunk_length=40)
    results = run_inter_burst_continuity(model, images, plan, device="cpu")
    carry = results["carry"].cos_phi
    carry_dt = results["carry_dt"].cos_phi
    # 第二/三个 chunk 的第一帧（kept idx 40, 80）应该不同
    boundaries = [pair[1] for pair in plan.boundary_kept_indices]
    diffs = [abs(carry[i] - carry_dt[i]) for i in boundaries]
    assert max(diffs) > 1e-5, (
        f"carry_dt should differ from carry at gap boundaries (diffs={diffs})"
    )
