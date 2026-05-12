"""Tests for the end-to-end FreqDecGWNet glue model.

Coverage:
    * stage1_seg / stage2_state / stage3_joint / inference forward modes
    * R2 -> R3 state-dim contract (slice [..., :3] is mandatory)
    * stage-1 checkpoint partial-loading (FAST_LiteNet weights -> self.r1)
    * encoder gradient detachment honored under stage2 default policy
    * shape sanity for all output tensors
"""

from __future__ import annotations

import torch
import pytest

from freqdec_gwnet.models.freqdec_gwnet import FreqDecGWNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_model(**overrides) -> FreqDecGWNet:
    """Small model for fast unit tests (32×32 input, width_mult=0.5)."""
    defaults = dict(
        in_channels=1, num_classes=1, width_mult=0.5,
        state_proj_dim=16, state_hidden_dim=16, state_lp_kernel=3,
        motion_hidden_dim=16,
    )
    defaults.update(overrides)
    return FreqDecGWNet(**defaults)


# ---------------------------------------------------------------------------
# Stage-1: pass-through to FAST-LiteNet
# ---------------------------------------------------------------------------


def test_stage1_forward_returns_seg_and_feature_bank():
    model = _tiny_model()
    model.eval()
    x = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        out = model(x, mode="stage1_seg")
    assert "seg" in out and "feature_bank" in out
    assert out["seg"].shape == (2, 1, 32, 32)
    # bottleneck is at 1/32 scale; feature_bank shape check is approximate
    assert out["feature_bank"].dim() == 4


def test_stage1_feature_bank_carry_over():
    """Stage-1 successive forwards should accept the feature_bank produced
    by the previous step (the FAST-LiteNet contract)."""
    model = _tiny_model()
    model.eval()
    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        out0 = model(x, mode="stage1_seg", feature_bank=None, reset_flag=True)
        out1 = model(
            x, mode="stage1_seg",
            feature_bank=out0["feature_bank"], reset_flag=False,
        )
    assert out1["seg"].shape == out0["seg"].shape


# ---------------------------------------------------------------------------
# Stage-2: window forward through encoder + state branch
# ---------------------------------------------------------------------------


def test_stage2_window_shape_and_state_dim():
    model = _tiny_model()
    model.eval()
    images = torch.randn(2, 6, 1, 32, 32)
    with torch.no_grad():
        out = model(images, mode="stage2_state")
    assert "state" in out and "feat_18_seq" in out
    assert out["state"].shape == (2, 6, 4)        # [cos, sin, amp, conf]
    # 1/8 of 32 = 4
    assert out["feat_18_seq"].shape[:2] == (2, 6)
    assert out["feat_18_seq"].shape[3:] == (4, 4)


def test_stage2_default_detaches_state_loss_gradient_from_encoder():
    """detach_encoder=True (default) → L_state grads should not reach encoder."""
    model = _tiny_model(state_detach_encoder=True)
    model.train()
    images = torch.randn(2, 4, 1, 32, 32, requires_grad=True)
    out = model(images, mode="stage2_state")
    fake_loss = out["state"].sum()
    fake_loss.backward()
    # 输入图像梯度被 detach 截断（state branch 内部 detach 了 1/8 features）
    assert images.grad is None or images.grad.abs().sum() == 0


def test_stage2_no_detach_lets_grad_flow():
    """detach_encoder=False → grad flows back through encoder (ablation mode)."""
    model = _tiny_model(state_detach_encoder=False)
    model.train()
    images = torch.randn(2, 4, 1, 32, 32, requires_grad=True)
    out = model(images, mode="stage2_state")
    out["state"].sum().backward()
    assert images.grad is not None
    assert images.grad.abs().sum() > 0


def test_stage2_rejects_wrong_input_rank():
    model = _tiny_model()
    with pytest.raises(ValueError, match=r"\[B, T, 1, H, W\]"):
        model(torch.randn(2, 1, 32, 32), mode="stage2_state")


# ---------------------------------------------------------------------------
# Stage-3: state branch + motion field
# ---------------------------------------------------------------------------


def test_stage3_full_outputs_and_state_dim_slice():
    model = _tiny_model()
    model.eval()
    images = torch.randn(2, 5, 1, 32, 32)
    with torch.no_grad():
        out = model(images, mode="stage3_joint", reference_index=0)
    expected_keys = {"state", "feat_18_seq", "delta_s", "delta_u", "U"}
    assert expected_keys.issubset(out.keys())
    assert out["state"].shape == (2, 5, 4)
    # R3 contract: motion field consumed [..., :3] → delta_s width = 3
    assert out["delta_s"].shape == (2, 5, 3)
    # 默认 output_dim=2 (2D translation)
    assert out["delta_u"].shape == (2, 5, 2)
    assert out["U"].shape == (2, 5, 2)


def test_stage3_zero_init_motion_field_gives_zero_U():
    """Untrained R3 (zero-init last layer) → U ≡ 0 regardless of state."""
    model = _tiny_model()
    model.eval()
    images = torch.randn(2, 5, 1, 32, 32)
    with torch.no_grad():
        out = model(images, mode="stage3_joint", reference_index=0)
    assert torch.allclose(out["delta_u"], torch.zeros_like(out["delta_u"]))
    assert torch.allclose(out["U"], torch.zeros_like(out["U"]))


def test_stage3_per_sample_reference_supported():
    """reference_index can be a per-sample LongTensor [B]."""
    model = _tiny_model()
    model.eval()
    images = torch.randn(3, 6, 1, 32, 32)
    ref = torch.tensor([0, 2, 5], dtype=torch.long)
    with torch.no_grad():
        out = model(images, mode="stage3_joint", reference_index=ref)
    assert out["U"].shape == (3, 6, 2)


def test_stage3_with_seg_returns_segmentation():
    model = _tiny_model()
    model.eval()
    images = torch.randn(2, 4, 1, 32, 32)
    with torch.no_grad():
        out = model(images, mode="stage3_joint", return_seg=True)
    assert "seg" in out
    assert out["seg"].shape == (2, 4, 1, 32, 32)


# ---------------------------------------------------------------------------
# Inference mode: stage3 + seg always
# ---------------------------------------------------------------------------


def test_inference_returns_full_pipeline():
    model = _tiny_model()
    model.eval()
    images = torch.randn(2, 4, 1, 32, 32)
    with torch.no_grad():
        out = model(images, mode="inference")
    for key in ("seg", "state", "delta_u", "U"):
        assert key in out, f"inference output missing '{key}'"


# ---------------------------------------------------------------------------
# Mode dispatch / error handling
# ---------------------------------------------------------------------------


def test_unknown_mode_raises():
    model = _tiny_model()
    with pytest.raises(ValueError, match="unknown mode"):
        model(torch.randn(1, 4, 1, 32, 32), mode="bogus")


def test_motion_state_dim_too_large_for_state_branch_rejected():
    with pytest.raises(ValueError, match="motion_state_dim"):
        FreqDecGWNet(motion_state_dim=10)


# ---------------------------------------------------------------------------
# Stage-1 checkpoint partial loading
# ---------------------------------------------------------------------------


def test_load_stage1_checkpoint_partial_load(tmp_path):
    """stage-1 ckpt loads into self.r1; R2/R3 stay untouched."""
    src = _tiny_model()
    # Mutate R1 with a known signature so we can verify it transferred
    with torch.no_grad():
        src.r1.seg_head.weight.fill_(0.7)
        src.r1.seg_head.bias.fill_(-0.3)
    ckpt = {
        "epoch": 1,
        "global_step": 100,
        "model_state_dict": src.r1.state_dict(),
    }
    path = tmp_path / "stage1.pth"
    torch.save(ckpt, path)

    dst = _tiny_model()
    # 验证目标模型初始 seg_head 权重确实不同
    assert not torch.allclose(
        dst.r1.seg_head.weight, src.r1.seg_head.weight,
    )
    missing, unexpected = dst.load_stage1_checkpoint(path, strict=False)
    # 全部 R1 键都应该匹配上
    assert missing == [], f"unexpected missing keys: {missing}"
    assert unexpected == [], f"unexpected extra keys: {unexpected}"
    # 权重已成功迁移
    assert torch.allclose(dst.r1.seg_head.weight, src.r1.seg_head.weight)
    assert torch.allclose(dst.r1.seg_head.bias, src.r1.seg_head.bias)


def test_load_stage1_checkpoint_strips_module_prefix(tmp_path):
    """DataParallel/DDP-saved checkpoints have 'module.' prefix → must strip."""
    src = _tiny_model()
    # 模拟 DDP 保存：在所有键名前加 module. 前缀
    state = {f"module.{k}": v for k, v in src.r1.state_dict().items()}
    path = tmp_path / "stage1_ddp.pth"
    torch.save(state, path)

    dst = _tiny_model()
    missing, unexpected = dst.load_stage1_checkpoint(path, strict=False)
    assert missing == []
    assert unexpected == []


def test_load_stage1_checkpoint_with_real_sanity_run_ckpt():
    """Smoke test: real stage-1 sanity_run checkpoint should partial-load."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    ckpt_path = (
        repo_root / "experiments" / "checkpoints" / "sanity_run_best.pth"
    )
    if not ckpt_path.is_file():
        pytest.skip(f"sanity_run checkpoint not found at {ckpt_path}")

    # The sanity run was trained with width_mult=1.0; build the matching size
    model = FreqDecGWNet(width_mult=1.0)
    missing, unexpected = model.load_stage1_checkpoint(ckpt_path, strict=False)
    # All R1 keys should match; R2/R3 weights stay at init.
    # We tolerate at most a small number of head differences (multi_task etc).
    assert len(unexpected) == 0, f"unexpected keys: {unexpected[:5]}..."
    # missing 可以是 0（若 ckpt 是同等结构）；如果有，应来自 multi-task 头未启用
    print(f"  partial-load: {len(missing)} missing, {len(unexpected)} unexpected")


# ---------------------------------------------------------------------------
# Module discovery / introspection
# ---------------------------------------------------------------------------


def test_freezedec_gwnet_exposes_three_redline_modules():
    model = _tiny_model()
    # R1 / R2 / R3 模块都能被命名直接访问，方便冻结/解冻控制
    assert hasattr(model, "r1")
    assert hasattr(model, "state_branch")
    assert hasattr(model, "motion_field")
    # 类型检查
    from freqdec_gwnet.models.fast_litenet import FAST_LiteNet
    from freqdec_gwnet.models.global_state_branch import GlobalStateBranch
    from freqdec_gwnet.models.relative_motion_field import RelativeMotionField
    assert isinstance(model.r1, FAST_LiteNet)
    assert isinstance(model.state_branch, GlobalStateBranch)
    assert isinstance(model.motion_field, RelativeMotionField)


# ---------------------------------------------------------------------------
# motion_field_type ablation switch (§6.1 ablation #1)
# ---------------------------------------------------------------------------


def test_motion_field_type_default_is_relative():
    model = _tiny_model()
    from freqdec_gwnet.models.relative_motion_field import RelativeMotionField
    assert model.motion_field_type == "relative"
    assert isinstance(model.motion_field, RelativeMotionField)


def test_motion_field_type_absolute_swaps_module():
    model = _tiny_model(motion_field_type="absolute")
    from freqdec_gwnet.models.absolute_motion_field import AbsoluteMotionField
    from freqdec_gwnet.models.relative_motion_field import RelativeMotionField
    assert model.motion_field_type == "absolute"
    assert isinstance(model.motion_field, AbsoluteMotionField)
    assert not isinstance(model.motion_field, RelativeMotionField)


def test_motion_field_type_unknown_raises():
    with pytest.raises(ValueError, match="motion_field_type"):
        _tiny_model(motion_field_type="diffusion")


def test_motion_field_absolute_path_works_in_stage3():
    """Absolute baseline 必须能完整跑 stage3_joint forward（API parity 红线）。"""
    model = _tiny_model(motion_field_type="absolute")
    model.eval()
    images = torch.randn(2, 5, 1, 32, 32)
    with torch.no_grad():
        out = model(images, mode="stage3_joint", reference_index=0)
    expected_keys = {"state", "feat_18_seq", "delta_s", "delta_u", "U"}
    assert expected_keys.issubset(out.keys())
    # 同一窗口、同一种子下，relative vs absolute 都返回零（zero-init）
    assert torch.allclose(out["U"], torch.zeros_like(out["U"]))


def test_motion_field_absolute_vs_relative_are_structurally_distinct():
    """注入相同的非零权重，恒定 Δs 输入下 U[t] 的依赖不同。"""
    rel_model = _tiny_model(motion_field_type="relative")
    abs_model = _tiny_model(motion_field_type="absolute")
    # 给两边 G 的最后一层注入相同非零权重
    with torch.no_grad():
        rel_model.motion_field.g_delta.net[-1].weight.fill_(0.05)
        rel_model.motion_field.g_delta.net[-1].bias.fill_(0.1)
        abs_model.motion_field.g_motion.net[-1].weight.fill_(0.05)
        abs_model.motion_field.g_motion.net[-1].bias.fill_(0.1)

    rel_model.eval(); abs_model.eval()
    images = torch.randn(1, 6, 1, 32, 32)
    with torch.no_grad():
        # 共享 state branch 输出，确保差异只来自 motion 模块
        s2 = rel_model._forward_stage2(images)["state"]
        s3 = s2[..., :3]
        rel_out = rel_model.motion_field(s3, reference_index=0)
        abs_out = abs_model.motion_field(s3, reference_index=0)

    # absolute U[t] 完全由 s[t]、s[r] 决定 → 改变 s 中间帧不影响 U[T-1]
    s3_perturbed = s3.clone()
    s3_perturbed[:, 1:5] = 0.0
    with torch.no_grad():
        abs_perturb = abs_model.motion_field(
            s3_perturbed, reference_index=0,
        )
    # absolute: U[5] 不变（无记忆性）
    assert torch.allclose(
        abs_out["U"][:, 5], abs_perturb["U"][:, 5], atol=1e-6,
    )
    # relative: U[5] 会随中间历史变（cumsum 性质）
    with torch.no_grad():
        rel_perturb = rel_model.motion_field(
            s3_perturbed, reference_index=0,
        )
    # 仅当 G_delta 实际产出非零时才严格不等；这里至少要有一个分量明显不同
    # （如果完全一致就说明 cumsum 没有起作用，红线被破坏）
    diff = (rel_out["U"][:, 5] - rel_perturb["U"][:, 5]).abs().sum()
    assert diff > 1e-6, (
        "RelativeMotionField.U[5] should depend on intermediate history "
        "via cumsum — found no difference, red-line might be broken."
    )
