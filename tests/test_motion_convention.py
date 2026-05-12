"""Motion convention tests — single source of truth for R3 + R4 behavior.

These tests pin down the protected behavior that PROJECT_CONSTRAINTS_19.6.md
requires: the cumulative-sum semantics, the warping sign convention, the
detach policy of ``MotionLoss``, and the wire-dilated background exclusion.
Breaking any test here means a red-line in the paper has been crossed.
"""

import torch
import pytest

from freqdec_gwnet.losses.motion_losses import MotionLoss
from freqdec_gwnet.models.relative_motion_field import (
    RelativeMotionField,
    apply_compensation,
)


# ---------------------------------------------------------------------------
# Δs / Δu / U cumulative-sum semantics
# ---------------------------------------------------------------------------


def test_compute_increments_pads_first_frame_zero():
    s = torch.tensor([[
        [1.0, 0.0, 0.0],
        [2.0, 1.0, 0.5],
        [4.0, 3.0, 1.0],
    ]])
    ds = RelativeMotionField.compute_increments(s)
    assert ds.shape == s.shape
    # Δs_0 必须为 0：第一帧没有"上一帧"
    assert torch.allclose(ds[:, 0], torch.zeros(1, 3))
    assert torch.allclose(ds[:, 1], torch.tensor([[1.0, 1.0, 0.5]]))
    assert torch.allclose(ds[:, 2], torch.tensor([[2.0, 2.0, 0.5]]))


def test_cumulative_displacement_zero_at_reference_int():
    delta_u = torch.tensor([[
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
    ]])  # [1, 4, 2]
    U = RelativeMotionField.cumulative_displacement(delta_u, reference_index=1)
    # U[r] == 0
    assert torch.allclose(U[:, 1], torch.zeros(1, 2))
    # U[t] = sum_{i=r+1}^{t} Δu[i]，r=1 时:
    assert torch.allclose(U[:, 2], torch.tensor([[3.0, 0.0]]))
    assert torch.allclose(U[:, 3], torch.tensor([[7.0, 0.0]]))
    # warm-up: t < r 必须为 0
    assert torch.allclose(U[:, 0], torch.zeros(1, 2))


def test_cumulative_displacement_per_sample_reference():
    delta_u = torch.tensor([
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
    ])  # [2, 4, 2]
    r = torch.tensor([0, 2])
    U = RelativeMotionField.cumulative_displacement(delta_u, reference_index=r)
    # 样本 0, r=0:
    #   U[0, 0] = 0 (reference)
    #   U[0, 1] = Δu[1] = 2
    #   U[0, 2] = Δu[1] + Δu[2] = 5
    #   U[0, 3] = Δu[1] + Δu[2] + Δu[3] = 9
    expected_0 = torch.tensor([
        [0.0, 0.0], [2.0, 0.0], [5.0, 0.0], [9.0, 0.0]
    ])
    assert torch.allclose(U[0], expected_0)
    # 样本 1, r=2: warm-up (t<2) 全部置 0；U[2]=0；U[3]=Δu[3]=4
    expected_1 = torch.tensor([
        [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [4.0, 0.0]
    ])
    assert torch.allclose(U[1], expected_1)


# ---------------------------------------------------------------------------
# RelativeMotionField forward — 零初始化 / detach_state_input 行为
# ---------------------------------------------------------------------------


def test_relative_motion_field_zero_init_gives_zero_motion():
    """末层零初始化保证训练前 Δu ≡ 0，cumsum 不会乱跳。"""
    rmf = RelativeMotionField()
    s = torch.randn(2, 5, 3)
    out = rmf(s, reference_index=0)
    assert torch.allclose(out["delta_u"], torch.zeros_like(out["delta_u"]))
    assert torch.allclose(out["U"], torch.zeros_like(out["U"]))
    assert torch.allclose(out["delta_u"][:, 0], torch.zeros(2, 2))


def test_relative_motion_field_detach_state_input():
    """``detach_state_input=True`` 必须阻断梯度回到 state。"""
    s = torch.randn(2, 5, 3, requires_grad=True)
    rmf = RelativeMotionField()
    # 让末层非零，否则 Δu 恒等于 0，无法考察梯度
    with torch.no_grad():
        rmf.g_delta.net[-1].weight.normal_(0.0, 0.1)
        rmf.g_delta.net[-1].bias.normal_(0.0, 0.1)
    out = rmf(s, reference_index=0, detach_state_input=True)
    out["delta_u"].sum().backward()
    assert s.grad is None, "detach_state_input=True must block state gradient"


def test_relative_motion_field_default_propagates_to_state():
    """默认配置 ``detach_state_input=False``：梯度能流回 state。"""
    s = torch.randn(2, 5, 3, requires_grad=True)
    rmf = RelativeMotionField()
    with torch.no_grad():
        rmf.g_delta.net[-1].weight.normal_(0.0, 0.1)
        rmf.g_delta.net[-1].bias.normal_(0.0, 0.1)
    out = rmf(s, reference_index=0)
    out["delta_u"].sum().backward()
    assert s.grad is not None
    assert s.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# 方向约定：apply_compensation 的符号 (R3 红线)
# ---------------------------------------------------------------------------


def test_compensation_recovers_reference_with_negative_sign():
    """current = ref shifted by (+5, 0); U = (5, 0).
    apply_compensation(current, U, 'current_to_reference') 应能复原 ref。
    """
    H, W = 32, 64
    ref = torch.zeros(1, 1, H, W)
    ref[0, 0, H // 2, W // 2] = 1.0
    current = torch.zeros_like(ref)
    current[0, 0, H // 2, W // 2 + 5] = 1.0

    U = torch.tensor([[5.0, 0.0]])
    recovered = apply_compensation(
        current, U, mode="current_to_reference", align_corners=True
    )
    flat = recovered.view(-1)
    peak = int(torch.argmax(flat).item())
    py, px = divmod(peak, W)
    assert (py, px) == (H // 2, W // 2), f"got ({py},{px})"


def test_compensation_aligns_roadmap_with_positive_sign():
    """warp(ref, +U) 应把参考路图推到当前帧坐标系。"""
    H, W = 32, 64
    ref = torch.zeros(1, 1, H, W)
    ref[0, 0, H // 2, W // 2] = 1.0
    U = torch.tensor([[5.0, 0.0]])
    aligned = apply_compensation(
        ref, U, mode="roadmap_to_current", align_corners=True
    )
    flat = aligned.view(-1)
    peak = int(torch.argmax(flat).item())
    py, px = divmod(peak, W)
    assert (py, px) == (H // 2, W // 2 + 5), f"got ({py},{px})"


def test_compensation_y_direction():
    """y 方向（行）符号也必须正确：dy>0 = 向下。"""
    H, W = 32, 32
    ref = torch.zeros(1, 1, H, W)
    ref[0, 0, H // 2, W // 2] = 1.0
    U = torch.tensor([[0.0, 4.0]])  # dy=4，正方向 = 行号增加
    aligned = apply_compensation(
        ref, U, mode="roadmap_to_current", align_corners=True
    )
    peak = int(torch.argmax(aligned.view(-1)).item())
    py, px = divmod(peak, W)
    assert (py, px) == (H // 2 + 4, W // 2), f"got ({py},{px})"


# ---------------------------------------------------------------------------
# MotionLoss detach policy + wire mask exclusion (R4)
# ---------------------------------------------------------------------------


def test_motion_loss_default_detach_blocks_feature_gradient():
    """默认 detach_prev/curr=True：梯度不能回到 encoder 特征。"""
    B, T, C, H, W = 1, 3, 4, 16, 16
    feat = torch.randn(B, T, C, H, W, requires_grad=True)
    # delta_u 必须是叶子张量，否则 .grad 不会被填充
    delta_u = torch.randn(B, T, 2).mul_(0.5).requires_grad_(True)
    wire_mask = torch.zeros(B, T, 1, H, W)  # 全背景

    loss_fn = MotionLoss(detach_prev_feature=True, detach_curr_feature=True)
    out = loss_fn(feat, delta_u, wire_mask)
    out["loss_motion"].backward()

    assert feat.grad is None, \
        "detach_prev/curr=True must block feat gradient (R5 protected default)"
    # 但 Δu 仍然要可训练
    assert delta_u.grad is not None
    assert delta_u.grad.abs().sum().item() > 0


def test_motion_loss_no_detach_allows_feature_gradient():
    """消融配置 detach_prev_feature=False：梯度可回流到 encoder 特征。"""
    B, T, C, H, W = 1, 3, 4, 16, 16
    feat = torch.randn(B, T, C, H, W, requires_grad=True)
    delta_u = torch.randn(B, T, 2).mul_(0.5).requires_grad_(True)
    wire_mask = torch.zeros(B, T, 1, H, W)

    loss_fn = MotionLoss(detach_prev_feature=False, detach_curr_feature=False)
    out = loss_fn(feat, delta_u, wire_mask)
    out["loss_motion"].backward()
    assert feat.grad is not None
    assert feat.grad.abs().sum().item() > 0


def test_motion_loss_wire_mask_excludes_active_motion():
    """导丝膨胀掩码内的残差不能进入 L_motion (R4 主动/背景解耦)。"""
    B, T, C, H, W = 1, 2, 1, 16, 16
    feat = torch.zeros(B, T, C, H, W)
    # 唯一变化集中在 (H/2, W/2)：t=1 多一个亮点，t=0 没有
    feat[:, 1, 0, H // 2, W // 2] = 1.0
    delta_u = torch.zeros(B, T, 2)  # 不需要平移即可暴露差异

    wire_mask_full = torch.zeros(B, T, 1, H, W)
    wire_mask_full[:, 1, 0, H // 2, W // 2] = 1.0

    loss_fn = MotionLoss(
        detach_prev_feature=True, detach_curr_feature=True, wire_dilate_k=2
    )
    out_with_mask = loss_fn(feat.clone(), delta_u, wire_mask_full)
    out_no_mask = loss_fn(
        feat.clone(), delta_u, torch.zeros(B, T, 1, H, W)
    )

    assert out_with_mask["loss_motion"].item() < 1e-6, (
        f"wire region must be excluded, got "
        f"{out_with_mask['loss_motion'].item()}"
    )
    assert out_no_mask["loss_motion"].item() > 1e-4, (
        "without mask the residual should be visible"
    )


def test_motion_loss_short_sequence_safe():
    """T<2 时 L_motion 应安全返回 0，不抛错。"""
    feat = torch.randn(1, 1, 4, 8, 8)
    delta_u = torch.randn(1, 1, 2)
    wire_mask = torch.zeros(1, 1, 1, 8, 8)
    loss_fn = MotionLoss()
    out = loss_fn(feat, delta_u, wire_mask)
    assert float(out["loss_motion"]) == 0.0


# ===========================================================================
# §8.2.1 appendix ablation — 4D / 6D / 9D motion-input variants
# ===========================================================================


def test_motion_field_input_mode_delta_default_is_6d_when_state_dim_3():
    """state_dim=3 + input_mode=delta → ψ width 6 (default main config)."""
    rmf = RelativeMotionField(state_dim_for_motion=3, input_mode="delta")
    s = torch.randn(2, 5, 3)
    out = rmf(s, reference_index=0)
    first = next(m for m in rmf.g_delta.net if isinstance(m, torch.nn.Linear))
    assert first.in_features == 6
    assert out["delta_u"].shape == (2, 5, 2)


def test_motion_field_input_mode_delta_4d_with_state_dim_2():
    """state_dim=2 + input_mode=delta → ψ width 4 (paper §8.2.1 4D)."""
    rmf = RelativeMotionField(state_dim_for_motion=2, input_mode="delta")
    first = next(m for m in rmf.g_delta.net if isinstance(m, torch.nn.Linear))
    assert first.in_features == 4
    s = torch.randn(2, 5, 2)
    out = rmf(s, reference_index=0)
    assert out["delta_u"].shape == (2, 5, 2)


def test_motion_field_input_mode_with_prev_is_9d_when_state_dim_3():
    """state_dim=3 + input_mode=with_prev → ψ width 9 (paper §8.2.1 9D)."""
    rmf = RelativeMotionField(
        state_dim_for_motion=3, input_mode="with_prev",
    )
    first = next(m for m in rmf.g_delta.net if isinstance(m, torch.nn.Linear))
    assert first.in_features == 9
    s = torch.randn(2, 5, 3)
    out = rmf(s, reference_index=0)
    assert out["delta_u"].shape == (2, 5, 2)


def test_motion_field_with_prev_first_frame_uses_self_as_prev():
    """t=0 时 s_{-1} := s_0 约定 → Δu_0 仍按规范强制为 0。"""
    rmf = RelativeMotionField(
        state_dim_for_motion=3, input_mode="with_prev",
    )
    with torch.no_grad():
        rmf.g_delta.net[-1].weight.normal_(0, 0.01)
        rmf.g_delta.net[-1].bias.normal_(0, 0.01)
    s = torch.randn(1, 6, 3)
    out = rmf(s, reference_index=0)
    assert torch.allclose(out["delta_u"][:, 0], torch.zeros(1, 2))


def test_motion_field_with_prev_changes_delta_u_vs_delta():
    """两种 input_mode 在同一 s 输入下应产生不同的 Δu 分布——证明 9D 真的喂入了 s_{t-1}。"""
    torch.manual_seed(0)
    s = torch.randn(1, 8, 3)
    rmf_delta = RelativeMotionField(
        state_dim_for_motion=3, input_mode="delta",
    )
    rmf_prev = RelativeMotionField(
        state_dim_for_motion=3, input_mode="with_prev",
    )
    with torch.no_grad():
        for m in (rmf_delta.g_delta, rmf_prev.g_delta):
            for layer in m.net:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.normal_(0, 0.1)
                    layer.bias.normal_(0, 0.1)
    out_delta = rmf_delta(s, reference_index=0)
    out_prev = rmf_prev(s, reference_index=0)
    diff = (out_delta["delta_u"] - out_prev["delta_u"]).abs().max().item()
    assert diff > 1e-3


def test_motion_field_unknown_input_mode_rejected():
    with pytest.raises(ValueError, match="input_mode"):
        RelativeMotionField(input_mode="bogus")


# ===========================================================================
# Zero-mean + drift-cap regularizers (paper §7 over-compensation fix)
# ===========================================================================


def test_motion_loss_zero_mean_is_zero_for_oscillating_delta_u():
    """Δu that oscillates around 0 → loss_zero_mean ≈ 0.

    Use T=9 so the oscillation t=1..8 covers an even count → mean = 0
    exactly. The point is to show the regularizer doesn't punish
    well-behaved oscillating motion.
    """
    T = 9
    feat = torch.randn(1, T, 4, 8, 8)
    du = torch.zeros(1, T, 2)
    for t in range(1, T):
        du[0, t, 0] = (-1.0) ** t
    wire_mask = torch.zeros(1, T, 1, 8, 8)
    loss_fn = MotionLoss(lambda_zero_mean=1.0)
    out = loss_fn(feat, du, wire_mask)
    assert out["loss_zero_mean"].item() < 1e-3


def test_motion_loss_zero_mean_punishes_monotonic_bias():
    """Δu with persistent positive bias → loss_zero_mean is large.

    This is exactly the failure mode we observed in stage3_main_v2:
    G_delta shortcut to ~+1.5 px constant → monotonic U drift.
    """
    T = 8
    feat = torch.randn(1, T, 4, 8, 8)
    # Δu_t = +1.5 for all t≥1 (the observed pathology)
    du = torch.zeros(1, T, 2)
    du[0, 1:, 0] = 1.5
    wire_mask = torch.zeros(1, T, 1, 8, 8)
    loss_fn = MotionLoss(lambda_zero_mean=1.0)
    out = loss_fn(feat, du, wire_mask)
    # mean over t=1..T-1 of 1.5 = 1.5, squared = 2.25
    assert abs(out["loss_zero_mean"].item() - 2.25) < 0.1


def test_motion_loss_zero_mean_defaults_to_zero_weight():
    """Default lambda_zero_mean=0 → preserves historical loss exactly."""
    feat = torch.randn(1, 4, 4, 8, 8)
    du = torch.full((1, 4, 2), 2.0)                    # large constant bias
    wire_mask = torch.zeros(1, 4, 1, 8, 8)
    out_default = MotionLoss()(feat, du, wire_mask)
    out_legacy = MotionLoss(lambda_zero_mean=0.0)(feat, du, wire_mask)
    # 默认权重 = 0 → 总 loss 跟显式给 0 必须完全相同
    assert torch.allclose(
        out_default["loss_motion"], out_legacy["loss_motion"]
    )


def test_motion_loss_zero_mean_changes_total_when_weight_nonzero():
    feat = torch.randn(1, 4, 4, 8, 8)
    du = torch.full((1, 4, 2), 2.0)                    # bias
    du[:, 0] = 0
    wire_mask = torch.zeros(1, 4, 1, 8, 8)
    out_off = MotionLoss(lambda_zero_mean=0.0)(feat, du, wire_mask)
    out_on = MotionLoss(lambda_zero_mean=1.0)(feat, du, wire_mask)
    assert out_on["loss_motion"].item() > out_off["loss_motion"].item() + 0.1


def test_motion_loss_drift_cap_inactive_under_threshold():
    """When ‖U‖_max is within cap, loss_drift_cap should be 0."""
    feat = torch.randn(1, 4, 4, 8, 8)
    du = torch.zeros(1, 4, 2)
    U = torch.tensor([[[1.0, 0], [1.0, 0], [1.0, 0], [1.0, 0]]])    # constant U
    wire_mask = torch.zeros(1, 4, 1, 8, 8)
    out = MotionLoss(lambda_drift_cap=1.0, drift_cap_ratio=2.0)(
        feat, du, wire_mask, U=U,
    )
    assert out["loss_drift_cap"].item() == 0.0


def test_motion_loss_drift_cap_punishes_blowup():
    feat = torch.randn(1, 8, 4, 8, 8)
    du = torch.zeros(1, 8, 2)
    # U 单调累加到 100：median≈50, cap=100, max=100，正好临界
    U = torch.zeros(1, 8, 2)
    U[0, :, 0] = torch.arange(0, 8) * 50.0              # 0, 50, 100, ... 350
    wire_mask = torch.zeros(1, 8, 1, 8, 8)
    out = MotionLoss(lambda_drift_cap=1.0, drift_cap_ratio=2.0)(
        feat, du, wire_mask, U=U,
    )
    # max=350, median=175, cap=2*175=350, excess=0
    # 给 ratio=1.0 应该触发
    out_strict = MotionLoss(lambda_drift_cap=1.0, drift_cap_ratio=1.0)(
        feat, du, wire_mask, U=U,
    )
    assert out_strict["loss_drift_cap"].item() > 0


def test_freqdec_gwnet_motion_input_mode_passed_through():
    """FreqDecGWNet 必须把 motion_input_mode 透传到 RelativeMotionField。"""
    from freqdec_gwnet.models.freqdec_gwnet import FreqDecGWNet
    model = FreqDecGWNet(
        width_mult=0.5, state_proj_dim=16, state_hidden_dim=16,
        state_lp_kernel=3, motion_hidden_dim=16,
        motion_input_mode="with_prev",
    )
    assert model.motion_field.input_mode == "with_prev"
    first = next(
        m for m in model.motion_field.g_delta.net
        if isinstance(m, torch.nn.Linear)
    )
    assert first.in_features == 9
