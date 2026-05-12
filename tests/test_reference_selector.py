"""ReferenceSelector tests — pin down the 3-tier policy from §2 of the
constraints document and the warm-up scoring formula from §1.3 of the
technical route.
"""

import math
import torch
import pytest

from freqdec_gwnet.utils.reference_selector import (
    ReferenceSelection,
    ReferenceSelector,
    laplacian_variance,
    select_reference,
    state_change_norm,
)


# ---------------------------------------------------------------------------
# helpers: build synthetic warm-up frames with known sharpness ranking
# ---------------------------------------------------------------------------


def _make_blurred_frame(H: int, W: int, sigma: float, seed: int) -> torch.Tensor:
    """Random texture filtered with a Gaussian kernel of width ``sigma``.

    Larger sigma -> blurrier image -> lower Laplacian variance.
    """
    torch.manual_seed(seed)
    base = torch.randn(1, 1, H, W)
    if sigma <= 0:
        return base.squeeze(0).squeeze(0)
    # 用 1D 高斯核做可分离卷积，避免引入 SciPy 依赖
    radius = max(1, int(3 * sigma))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k1d_h = g.view(1, 1, 1, -1)
    k1d_v = g.view(1, 1, -1, 1)
    blurred = torch.nn.functional.conv2d(base, k1d_h, padding=(0, radius))
    blurred = torch.nn.functional.conv2d(blurred, k1d_v, padding=(radius, 0))
    return blurred.squeeze(0).squeeze(0)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def test_laplacian_variance_higher_for_sharper_image():
    sharp = _make_blurred_frame(32, 32, sigma=0.0, seed=0).unsqueeze(0)
    blurry = _make_blurred_frame(32, 32, sigma=3.0, seed=0).unsqueeze(0)
    v_sharp = laplacian_variance(sharp).item()
    v_blurry = laplacian_variance(blurry).item()
    assert v_sharp > v_blurry, (
        f"sharp({v_sharp:.4f}) should exceed blurry({v_blurry:.4f})"
    )


def test_state_change_norm_first_frame_is_zero():
    s = torch.tensor([
        [1.0, 0.0],
        [2.0, 0.0],
        [4.0, 0.0],
    ])
    n = state_change_norm(s)
    assert n.shape == (3,)
    assert n[0].item() == 0.0
    assert math.isclose(n[1].item(), 1.0, abs_tol=1e-6)
    assert math.isclose(n[2].item(), 2.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Tier 1: clinical roadmap short-circuits everything else
# ---------------------------------------------------------------------------


def test_clinical_priority_overrides_warmup():
    # 即使 warm-up 评分会选别的帧，只要给了 clinical_roadmap_index 就直接返回它
    images = torch.stack([
        _make_blurred_frame(16, 16, sigma=s, seed=i)
        for i, s in enumerate([3.0, 0.1, 3.0, 3.0, 3.0])  # frame 1 最锐利
    ], dim=0)
    sel = ReferenceSelector(warmup_min=5, warmup_max=5)
    out = sel.select(images=images, clinical_roadmap_index=3)
    assert out.strategy == "clinical"
    assert out.reference_index == 3


def test_clinical_index_out_of_range_raises():
    sel = ReferenceSelector()
    with pytest.raises(ValueError):
        sel.select(total_frames=10, clinical_roadmap_index=15)


# ---------------------------------------------------------------------------
# Tier 2: warm-up scoring
# ---------------------------------------------------------------------------


def test_warmup_picks_sharpest_when_other_terms_constant():
    # 5 帧，frame 2 最锐利；states/anchor 都不给 -> 只看 sharpness
    sigmas = [3.0, 3.0, 0.1, 3.0, 3.0]
    images = torch.stack([
        _make_blurred_frame(32, 32, sigma=s, seed=i)
        for i, s in enumerate(sigmas)
    ], dim=0)
    sel = ReferenceSelector(warmup_min=5, warmup_max=5)
    out = sel.select(images=images)
    assert out.strategy == "warmup_score"
    assert out.reference_index == 2
    assert out.warmup_window == (0, 5)


def test_warmup_penalizes_state_change():
    # 全部清晰度相同（用同一 seed 同一 sigma），区分仅来自 state change
    images = torch.stack([
        _make_blurred_frame(16, 16, sigma=1.0, seed=0)
        for _ in range(5)
    ], dim=0)
    # state_seq 设计：frame 3 的 ||Δs|| 显著最大（最不稳定）
    states = torch.tensor([
        [0.0, 0.0],
        [0.05, 0.0],
        [0.10, 0.0],
        [5.00, 0.0],   # 大跳变
        [5.05, 0.0],
    ])
    sel = ReferenceSelector(warmup_min=5, warmup_max=5)
    out = sel.select(images=images, state_seq=states)
    # frame 3 应该被惩罚到最低分
    score = out.score_components["score"]
    assert torch.argmin(score).item() == 3
    assert out.reference_index != 3


def test_warmup_uses_anchor_quality_to_break_ties():
    # 全图相同 sharpness、相同 state；只有 anchor_quality 区分
    images = torch.stack([
        _make_blurred_frame(16, 16, sigma=1.0, seed=0)
        for _ in range(5)
    ], dim=0)
    states = torch.zeros(5, 2)
    anchor = torch.tensor([0.1, 0.2, 0.9, 0.3, 0.4])  # frame 2 最高
    sel = ReferenceSelector(warmup_min=5, warmup_max=5)
    out = sel.select(images=images, state_seq=states, anchor_quality=anchor)
    assert out.reference_index == 2
    assert out.strategy == "warmup_score"


def test_warmup_window_capped_by_warmup_max():
    # 序列 20 帧，但 warmup_max=5：只能在前 5 帧打分
    sigmas = [3.0] * 20
    sigmas[10] = 0.0  # 10 帧之后最锐利的帧不应被选中
    sigmas[2] = 0.0   # 窗口内最锐利的帧
    images = torch.stack([
        _make_blurred_frame(16, 16, sigma=s, seed=i)
        for i, s in enumerate(sigmas)
    ], dim=0)
    sel = ReferenceSelector(warmup_min=5, warmup_max=5)
    out = sel.select(images=images)
    assert out.reference_index == 2
    assert out.warmup_window == (0, 5)


def test_warmup_min_floor_for_short_sequences():
    # 序列只有 3 帧，warmup_min=5：实际窗口只能取 3
    images = torch.stack([
        _make_blurred_frame(16, 16, sigma=s, seed=i)
        for i, s in enumerate([3.0, 0.0, 3.0])
    ], dim=0)
    sel = ReferenceSelector(warmup_min=5, warmup_max=10)
    out = sel.select(images=images)
    assert out.warmup_window == (0, 3)
    assert out.reference_index == 1


def test_warmup_score_components_reported():
    images = torch.stack([
        _make_blurred_frame(16, 16, sigma=s, seed=i)
        for i, s in enumerate([3.0, 1.0, 0.5, 3.0, 3.0])
    ], dim=0)
    states = torch.linspace(0, 1, 10).view(5, 2)
    anchor = torch.tensor([0.5, 0.6, 0.7, 0.4, 0.3])
    out = select_reference(
        images=images, state_seq=states, anchor_quality=anchor,
        warmup_min=5, warmup_max=5,
    )
    expected_keys = {
        "sharpness_norm", "anchor_norm", "state_change_norm", "score"
    }
    assert set(out.score_components.keys()) == expected_keys
    for v in out.score_components.values():
        assert v.shape == (5,)


# ---------------------------------------------------------------------------
# Tier 3: fallback
# ---------------------------------------------------------------------------


def test_fallback_first_frame_when_no_signals():
    out = select_reference(total_frames=8)
    assert out.strategy == "first_frame"
    assert out.reference_index == 0
    assert out.warmup_window == (0, 8)


def test_fallback_first_frame_zero_total_raises():
    sel = ReferenceSelector()
    with pytest.raises(ValueError):
        sel.select(total_frames=0)


# ---------------------------------------------------------------------------
# Strategy 字段始终被填充（PROJECT_CONSTRAINTS §2 末尾的硬要求）
# ---------------------------------------------------------------------------


def test_strategy_field_always_populated():
    # 三种调用路径都必须填 strategy
    images = torch.stack([
        _make_blurred_frame(16, 16, sigma=1.0, seed=0)
        for _ in range(5)
    ], dim=0)
    out_clinical = select_reference(total_frames=5, clinical_roadmap_index=2)
    out_warmup = select_reference(images=images, warmup_min=5, warmup_max=5)
    out_first = select_reference(total_frames=5)
    for out in (out_clinical, out_warmup, out_first):
        assert isinstance(out, ReferenceSelection)
        assert out.strategy in {"clinical", "warmup_score", "first_frame"}


# ---------------------------------------------------------------------------
# 整合检查：与 RelativeMotionField.cumulative_displacement 的接口对齐
# ---------------------------------------------------------------------------


def test_reference_selection_plugs_into_cumulative_displacement():
    """选出的 r 直接喂给 cumulative_displacement，下游链路不爆。"""
    from freqdec_gwnet.models.relative_motion_field import RelativeMotionField

    images = torch.stack([
        _make_blurred_frame(16, 16, sigma=s, seed=i)
        for i, s in enumerate([3.0, 0.1, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    ], dim=0)
    out = select_reference(images=images, warmup_min=5, warmup_max=5)
    r = out.reference_index
    # 构造一个假的 Δu 序列喂进 cumulative_displacement，长度 = 8
    delta_u = torch.randn(1, 8, 2)
    U = RelativeMotionField.cumulative_displacement(delta_u, reference_index=r)
    assert torch.allclose(U[:, r], torch.zeros(1, 2))


# ===========================================================================
# §6.1 main ablation #4 — force_strategy override
# ===========================================================================


def test_force_strategy_first_frame_ignores_warmup_score():
    """force_strategy='first_frame' 必须返回 r=0，无视清晰度评分。"""
    sigmas = [3.0, 3.0, 0.1, 3.0, 3.0]                  # frame 2 最锐
    images = torch.stack([
        _make_blurred_frame(32, 32, sigma=s, seed=i)
        for i, s in enumerate(sigmas)
    ], dim=0)
    sel = ReferenceSelector(warmup_min=5, warmup_max=5,
                            force_strategy="first_frame")
    out = sel.select(images=images)
    assert out.strategy == "first_frame"
    assert out.reference_index == 0


def test_force_strategy_random_uniform_within_warmup_window():
    """random 必须落在 warm-up window [0, N) 之内，且对种子敏感。"""
    images = torch.stack([
        _make_blurred_frame(16, 16, sigma=1.0, seed=i)
        for i in range(8)
    ], dim=0)
    sel_a = ReferenceSelector(warmup_min=5, warmup_max=8,
                              force_strategy="random", random_seed=7)
    sel_b = ReferenceSelector(warmup_min=5, warmup_max=8,
                              force_strategy="random", random_seed=7)
    sel_c = ReferenceSelector(warmup_min=5, warmup_max=8,
                              force_strategy="random", random_seed=99)

    out_a = sel_a.select(images=images)
    out_b = sel_b.select(images=images)
    out_c = sel_c.select(images=images)

    assert out_a.strategy == "random"
    assert 0 <= out_a.reference_index < 8
    # 同种子可复现
    assert out_a.reference_index == out_b.reference_index
    # 不同种子应该（大概率）选不同帧
    assert out_a.reference_index != out_c.reference_index or 8 == 1


def test_force_strategy_warmup_score_uses_scoring_branch():
    """force_strategy='warmup_score' 强制评分分支，结果与 auto 模式一致。"""
    sigmas = [3.0, 3.0, 0.1, 3.0, 3.0]
    images = torch.stack([
        _make_blurred_frame(32, 32, sigma=s, seed=i)
        for i, s in enumerate(sigmas)
    ], dim=0)
    sel_auto = ReferenceSelector(warmup_min=5, warmup_max=5)
    sel_force = ReferenceSelector(warmup_min=5, warmup_max=5,
                                   force_strategy="warmup_score")
    out_auto = sel_auto.select(images=images)
    out_force = sel_force.select(images=images)
    assert out_auto.reference_index == out_force.reference_index
    assert out_force.strategy == "warmup_score"


def test_force_strategy_clinical_requires_index():
    """force_strategy='clinical' 没给 clinical_roadmap_index 必须报错。"""
    sel = ReferenceSelector(force_strategy="clinical")
    with pytest.raises(ValueError, match="clinical_roadmap_index"):
        sel.select(total_frames=10)


def test_force_strategy_clinical_uses_provided_index():
    sel = ReferenceSelector(force_strategy="clinical")
    out = sel.select(total_frames=20, clinical_roadmap_index=7)
    assert out.strategy == "clinical"
    assert out.reference_index == 7


def test_force_strategy_clinical_index_out_of_range():
    sel = ReferenceSelector(force_strategy="clinical")
    with pytest.raises(ValueError, match="out of"):
        sel.select(total_frames=10, clinical_roadmap_index=15)


def test_invalid_force_strategy_rejected():
    with pytest.raises(ValueError, match="force_strategy"):
        ReferenceSelector(force_strategy="bogus")


def test_force_strategy_auto_equals_none():
    """'auto' 必须等价于不传 force_strategy。"""
    sel_a = ReferenceSelector()
    sel_b = ReferenceSelector(force_strategy="auto")
    assert sel_a.force_strategy is None and sel_b.force_strategy is None


def test_force_strategy_first_frame_overrides_clinical_when_both_given():
    """强制 first_frame 时即便给了 clinical_roadmap_index 也忽略它。"""
    sel = ReferenceSelector(force_strategy="first_frame")
    out = sel.select(total_frames=20, clinical_roadmap_index=12)
    assert out.strategy == "first_frame"
    assert out.reference_index == 0


def test_force_strategy_random_warmup_window_reported():
    """random 模式仍然要报告 warmup_window 用于诊断。"""
    sel = ReferenceSelector(warmup_min=5, warmup_max=8,
                            force_strategy="random", random_seed=0)
    out = sel.select(total_frames=20)
    assert out.warmup_window is not None
    assert out.warmup_window == (0, 8)
