"""Tests for AbsoluteMotionField (§6.1 ablation #1 baseline).

Two distinct concerns:
    * Output dict shape parity with RelativeMotionField (drop-in compatible).
    * Structural distinction: U is ``u_t − u_r`` (anchor-rebased), NOT a
      cumulative sum. This is the very property the ablation table is
      designed to expose.
"""

from __future__ import annotations

import pytest
import torch

from freqdec_gwnet.models.absolute_motion_field import (
    AbsoluteMotionField,
    GMotionMLP,
)
from freqdec_gwnet.models.relative_motion_field import RelativeMotionField


# ---------------------------------------------------------------------------
# Output shape parity with RelativeMotionField
# ---------------------------------------------------------------------------


def test_absolute_field_output_dict_matches_relative_field():
    rel = RelativeMotionField()
    abs_ = AbsoluteMotionField()
    rel.eval(); abs_.eval()

    state = torch.randn(2, 8, 3)
    with torch.no_grad():
        rel_out = rel(state, reference_index=0)
        abs_out = abs_(state, reference_index=0)

    # 完全一致的 keys 才能在 stage3 trainer 里互换
    assert set(rel_out.keys()) == set(abs_out.keys())
    # 每个张量的 shape 也要一一对应
    for k in rel_out:
        assert rel_out[k].shape == abs_out[k].shape, (
            f"{k}: rel={rel_out[k].shape} abs={abs_out[k].shape}"
        )


def test_absolute_field_zero_init_gives_zero_U():
    """末层零初始化 → 训练前 U ≡ 0（与 RelativeMotionField 同）."""
    field = AbsoluteMotionField()
    field.eval()
    state = torch.randn(2, 6, 3)
    with torch.no_grad():
        out = field(state)
    assert torch.allclose(out["U"], torch.zeros_like(out["U"]))
    assert torch.allclose(out["delta_u"], torch.zeros_like(out["delta_u"]))


# ---------------------------------------------------------------------------
# Structural distinction: U is anchor-rebased, NOT a cumsum
# ---------------------------------------------------------------------------


def test_absolute_field_U_is_anchor_relative_not_cumulative():
    """注入已知 G_motion 权重，验证 U[t] = G(s_t) − G(s_r)（而非 cumsum）。"""
    field = AbsoluteMotionField(state_dim_for_motion=3, output_dim=2)
    # 把 G_motion 替换为线性映射 W·s（无 ReLU 干扰）：
    # 用一个线性层模拟整个 MLP 的"等价线性"行为，方便预测输出。
    field.g_motion.net = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        field.g_motion.net.weight.fill_(0.0)
        field.g_motion.net.weight[0, 0] = 1.0           # u_x = s[0]
        field.g_motion.net.weight[1, 1] = 2.0           # u_y = 2 * s[1]

    state = torch.tensor([[
        [1.0, 0.5, 0.0],
        [2.0, 1.0, 0.0],
        [4.0, 2.0, 0.0],
        [8.0, 4.0, 0.0],
    ]])                                                  # [1, 4, 3]

    with torch.no_grad():
        out = field(state, reference_index=1)            # ref = frame 1

    # 期望：u_t = (s[t,0], 2*s[t,1])
    expected_u = torch.tensor([[
        [1.0, 1.0],
        [2.0, 2.0],
        [4.0, 4.0],
        [8.0, 8.0],
    ]])
    expected_U = expected_u - expected_u[:, 1:2]         # 减 u_ref
    # warm-up 帧 t<1 屏蔽为 0
    expected_U[:, 0] = 0
    assert torch.allclose(out["U"], expected_U, atol=1e-6)


def test_absolute_field_does_not_grow_with_constant_delta_s():
    """关键结构差异：恒定 Δs 时，AbsoluteMotionField 的 U 不会线性增长。

    RelativeMotionField 在 G_delta 输出非零时，U = cumsum(Δu) 必然累加；
    AbsoluteMotionField 的 U = u_t − u_r 不存在累加项。
    """
    abs_field = AbsoluteMotionField()
    # 非零初始化最后一层，让 G_motion 输出非零
    with torch.no_grad():
        abs_field.g_motion.net[-1].weight.fill_(0.1)
        abs_field.g_motion.net[-1].bias.fill_(0.5)
    abs_field.eval()

    # 构造 s_t = t * unit (恒定 Δs)
    T = 10
    state = torch.linspace(0.0, 1.0, T).unsqueeze(0).unsqueeze(-1).expand(1, T, 3)

    with torch.no_grad():
        out = abs_field(state.contiguous(), reference_index=0)

    # 比较 U[5] 和 2 * U[2]：若是 cumsum 行为，比例 ≈ 5/2；
    # 若是 anchor-rebased，则比例 ≈ G(s[5])/G(s[2])，与 5/2 不同
    # （取决于 ReLU 非线性，但**结构上不应是恰好的 cumsum 比例**）
    # 这里验证一个更稳的性质：U_t 完全由 s_t 决定，与历史无关。
    # 把 s_seq 在中间做一次置换：state[:, 5] 的 U 应保持不变。
    state2 = state.clone()
    state2[:, 1:5] = 0.0                                  # 历史完全改变
    with torch.no_grad():
        out2 = abs_field(state2.contiguous(), reference_index=0)

    # 关键断言：absolute 是无记忆的，U[t] 只看 s[t] 和 s[r]
    # state[:, 5] 在两种情况下相同，且 reference_index 都是 0（都用 s[0]=0）；
    # state2[:, 0] = state[:, 0]，所以 U[5] 应该相等
    assert torch.allclose(out["U"][:, 5], out2["U"][:, 5], atol=1e-6), (
        "AbsoluteMotionField.U[t] should depend only on s[t] and s[r], "
        "not on intermediate history (the very property RelativeMotionField "
        "lacks — that's what the ablation is designed to expose)."
    )


def test_absolute_field_per_sample_reference():
    """与 RelativeMotionField 相同：reference_index 可以是 [B] LongTensor。"""
    field = AbsoluteMotionField()
    field.eval()
    state = torch.randn(3, 8, 3)
    ref = torch.tensor([0, 3, 7])
    with torch.no_grad():
        out = field(state, reference_index=ref)
    assert out["U"].shape == (3, 8, 2)
    # 每个样本在自己的 ref 处 U 应为 0（anchor-rebased 性质）
    for b in range(3):
        assert torch.allclose(out["U"][b, ref[b]], torch.zeros(2), atol=1e-6)


def test_absolute_field_warmup_zeroed():
    """t < r 的帧 U 被屏蔽为 0（与 RelativeMotionField 同）."""
    field = AbsoluteMotionField()
    with torch.no_grad():
        field.g_motion.net[-1].weight.fill_(0.5)
        field.g_motion.net[-1].bias.fill_(1.0)
    field.eval()
    state = torch.randn(1, 10, 3)
    with torch.no_grad():
        out = field(state, reference_index=4)
    assert torch.allclose(out["U"][:, :4], torch.zeros(1, 4, 2), atol=1e-6)
    # ref 处 U 必须正好是 0
    assert torch.allclose(out["U"][:, 4], torch.zeros(1, 2), atol=1e-6)


# ---------------------------------------------------------------------------
# delta_u API parity
# ---------------------------------------------------------------------------


def test_absolute_field_delta_u_is_implicit_increment_of_U():
    field = AbsoluteMotionField()
    with torch.no_grad():
        field.g_motion.net[-1].weight.fill_(0.3)
        field.g_motion.net[-1].bias.fill_(-0.1)
    field.eval()
    state = torch.randn(2, 6, 3)
    with torch.no_grad():
        out = field(state, reference_index=0)
    # delta_u[t] = U[t] - U[t-1]，且 delta_u[0] = 0
    expected = torch.zeros_like(out["U"])
    expected[:, 1:] = out["U"][:, 1:] - out["U"][:, :-1]
    assert torch.allclose(out["delta_u"], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_absolute_field_rejects_wrong_input_rank():
    field = AbsoluteMotionField()
    with pytest.raises(ValueError, match=r"\[B, T, D\]"):
        field(torch.randn(8, 3))                        # missing batch dim


# ---------------------------------------------------------------------------
# G_motion MLP standalone
# ---------------------------------------------------------------------------


def test_gmotion_mlp_zero_init_last_layer():
    mlp = GMotionMLP(state_dim=3, hidden_dim=16, output_dim=2)
    assert torch.all(mlp.net[-1].weight == 0)
    assert torch.all(mlp.net[-1].bias == 0)
