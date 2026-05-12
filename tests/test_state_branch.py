"""GlobalStateBranch + StateLoss + R2->R3 state-dim contract.

Migrates the ad-hoc ``if __name__ == '__main__'`` blocks in
``models/global_state_branch.py`` and ``losses/state_losses.py`` into proper
pytest cases, and adds the previously-unprotected interface contract between
:class:`GlobalStateBranch` (4D output) and :class:`RelativeMotionField`
(3D state input — confidence is intentionally dropped).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from freqdec_gwnet.models.global_state_branch import (
    GlobalStateBranch,
    MambaStateCore,
)
from freqdec_gwnet.models.relative_motion_field import RelativeMotionField
from freqdec_gwnet.losses.state_losses import (
    StateLoss,
    amplitude_loss,
    circular_phase_loss,
    make_burn_in_mask,
)


# ---------------------------------------------------------------------------
# GlobalStateBranch shape & gradient contract
# ---------------------------------------------------------------------------


def _branch(in_channels: int = 32, **kw) -> GlobalStateBranch:
    return GlobalStateBranch(
        in_channels=in_channels, proj_dim=32, hidden_dim=16, **kw,
    )


def test_state_branch_output_shape_is_BT4():
    branch = _branch()
    feat = torch.randn(2, 5, 32, 8, 8)
    out = branch.forward_window(feat)
    assert out.shape == (2, 5, 4), \
        "GlobalStateBranch must emit 4-D state (cos, sin, amp, conf)"


def test_state_branch_detach_default_blocks_encoder_grad():
    """Default ``detach_encoder=True`` must stop L_state gradient at the branch.

    Constraints §5: encoder is only optimized by L_seg-family losses.
    """
    branch = _branch(detach_encoder=True)
    feat = torch.randn(2, 5, 32, 8, 8, requires_grad=True)
    branch.forward_window(feat).sum().backward()
    assert feat.grad is None, \
        "detach_encoder=True must leave feat.grad as None"


def test_state_branch_no_detach_lets_grad_flow():
    """Ablation switch — detach_encoder=False routes gradient back."""
    branch = _branch(detach_encoder=False)
    feat = torch.randn(2, 5, 32, 8, 8, requires_grad=True)
    branch.forward_window(feat).sum().backward()
    assert feat.grad is not None
    assert feat.grad.abs().sum() > 0


def test_state_branch_step_matches_window_for_gru():
    """Per-frame ``forward_step`` must match ``forward_window`` output."""
    torch.manual_seed(0)
    branch = _branch(state_core_type="gru")
    branch.eval()
    feat = torch.randn(1, 5, 32, 8, 8)
    with torch.no_grad():
        win = branch.forward_window(feat)
        step_outs = []
        h = None
        for t in range(5):
            s_t, h, _ = branch.forward_step(feat[0:1, t], h_prev=h)
            step_outs.append(s_t)
        step = torch.stack(step_outs, dim=1)
    assert torch.allclose(win, step, atol=1e-5)


# ---------------------------------------------------------------------------
# Mamba state core (with/without mamba_ssm installed)
# ---------------------------------------------------------------------------


def _mamba_ssm_installed() -> bool:
    try:
        import mamba_ssm  # noqa: F401
        return True
    except ImportError:
        return False


def test_state_branch_mamba_constructor_does_not_crash():
    """state_core_type='mamba' must work whether or not mamba_ssm installed.

    Without mamba_ssm the MambaStateCore fallback to GRU activates, with a
    warning. With it, the real Mamba kernel is used.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # 接受 fallback 警告
        branch = _branch(state_core_type="mamba")
    assert branch.state_core_type == "mamba"
    assert hasattr(branch, "state_core")


def test_state_branch_mamba_window_forward_shape():
    """Whether real Mamba or GRU fallback, window forward must return BT4."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        branch = _branch(state_core_type="mamba")
    branch.eval()
    feat = torch.randn(2, 6, 32, 8, 8)
    # Mamba CUDA kernel 只能在 GPU 上跑；GRU fallback 可以 CPU。
    if _mamba_ssm_installed() and torch.cuda.is_available():
        branch = branch.cuda()
        feat = feat.cuda()
    with torch.no_grad():
        out = branch.forward_window(feat)
    assert out.shape == (2, 6, 4)


def test_state_branch_mamba_actually_uses_mamba_when_installed():
    """If mamba_ssm is installed, MambaStateCore should not fall back."""
    if not _mamba_ssm_installed():
        pytest.skip("mamba_ssm not installed in this environment")
    branch = _branch(state_core_type="mamba")
    # 内部标志：_use_mamba 为 True 说明真的用了 Mamba 而非 GRU fallback
    assert branch.state_core._use_mamba is True


def test_freqdec_gwnet_mamba_state_core_smoke():
    """End-to-end: FreqDecGWNet with state_core_type='mamba' forward works.

    Critical for the §6.1 ablation table 5 (GRU vs Mamba). Smoke-only —
    accepts either real Mamba or GRU fallback.
    """
    from freqdec_gwnet.models.freqdec_gwnet import FreqDecGWNet
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = FreqDecGWNet(
            in_channels=1, num_classes=1, width_mult=0.5,
            state_proj_dim=16, state_hidden_dim=16, state_lp_kernel=3,
            motion_hidden_dim=16,
            state_core_type="mamba",
        )
    model.eval()
    images = torch.randn(2, 5, 1, 32, 32)
    if _mamba_ssm_installed() and torch.cuda.is_available():
        model = model.cuda()
        images = images.cuda()
    with torch.no_grad():
        out = model(images, mode="stage3_joint", reference_index=0)
    assert out["state"].shape == (2, 5, 4)
    assert out["U"].shape == (2, 5, 2)


def test_get_state_components_keys():
    branch = _branch()
    state = torch.randn(2, 5, 4)
    comps = branch.get_state_components(state)
    assert set(comps.keys()) == {
        "cos_phi", "sin_phi", "amplitude", "confidence",
    }
    assert torch.equal(comps["cos_phi"], state[..., 0])
    assert torch.equal(comps["confidence"], state[..., 3])


# ---------------------------------------------------------------------------
# State-dim contract: GlobalStateBranch -> RelativeMotionField
# ---------------------------------------------------------------------------


def test_state_branch_output_first3_feeds_relative_motion_field():
    """The 3D slice ``[cos, sin, amp]`` is the contract between R2 and R3.

    R3 deliberately drops the confidence channel — pinning that here so a
    future refactor cannot silently widen the input dim and corrupt G_delta.
    """
    torch.manual_seed(1)
    branch = _branch()
    rmf = RelativeMotionField()                         # state_dim_for_motion=3 by default
    branch.eval()
    rmf.eval()
    feat = torch.randn(2, 7, 32, 8, 8)
    with torch.no_grad():
        state = branch.forward_window(feat)            # [2, 7, 4]
        s_seq = state[..., :3]                          # drop confidence
        out = rmf(s_seq, reference_index=0)
    assert s_seq.shape == (2, 7, 3)
    assert out["U"].shape == (2, 7, 2)
    assert out["delta_u"].shape == (2, 7, 2)
    # Zero-init G_delta -> all-zero output before training
    assert torch.allclose(out["U"], torch.zeros_like(out["U"]))


def test_relative_motion_field_state_dim_matches_branch_minus_confidence():
    """If GlobalStateBranch widens its output, R3 default must be revisited."""
    branch = _branch()
    rmf = RelativeMotionField()                         # state_dim_for_motion=3 default
    state = branch.forward_window(torch.randn(1, 4, 32, 8, 8))
    expected_in_dim = state.shape[-1] - 1               # drop confidence
    # 第一层 Linear 的 in_features 必须等于 2*state_dim (s_t || Δs_t)
    first_linear = next(m for m in rmf.g_delta.net.modules()
                        if isinstance(m, torch.nn.Linear))
    assert first_linear.in_features == 2 * expected_in_dim, (
        f"R3 expects 2*state_dim={2*expected_in_dim} input features, "
        f"got {first_linear.in_features}; either branch widened or "
        f"R3.state_dim drifted"
    )
    assert rmf.state_dim == expected_in_dim


# ---------------------------------------------------------------------------
# StateLoss — circular phase, amplitude scaling, burn-in
# ---------------------------------------------------------------------------


def test_circular_phase_loss_handles_wrap_around():
    """φ ≈ 2π should score nearly the same as φ ≈ 0 — MSE cannot do this."""
    phi_gt = torch.tensor([0.05])
    cos_gt, sin_gt = torch.cos(phi_gt), torch.sin(phi_gt)
    # Prediction near 2π — physically equivalent to phi_gt
    phi_wrap = torch.tensor([2 * math.pi - 0.05])
    loss_wrap = circular_phase_loss(
        torch.cos(phi_wrap), torch.sin(phi_wrap), cos_gt, sin_gt,
    )
    # Prediction at π — physically opposite
    phi_far = torch.tensor([math.pi])
    loss_far = circular_phase_loss(
        torch.cos(phi_far), torch.sin(phi_far), cos_gt, sin_gt,
    )
    assert loss_wrap.item() < 0.05, \
        f"wrap-around loss should be tiny, got {loss_wrap.item()}"
    assert loss_far.item() > 1.5, \
        f"opposite-phase loss should be near 2.0, got {loss_far.item()}"


def test_circular_phase_loss_normalizes_pred_magnitude():
    """Network can't cheat by scaling cos/sin large — F.normalize protects."""
    phi_gt = torch.tensor([0.5])
    cos_gt, sin_gt = torch.cos(phi_gt), torch.sin(phi_gt)
    # 同方向但幅度任意大
    cos_big = cos_gt * 10.0
    sin_big = sin_gt * 10.0
    loss_unit = circular_phase_loss(cos_gt, sin_gt, cos_gt, sin_gt)
    loss_big = circular_phase_loss(cos_big, sin_big, cos_gt, sin_gt)
    assert torch.allclose(loss_unit, loss_big, atol=1e-6)


def test_amplitude_loss_normalization_collapses_units():
    """Per-sequence amp_scale removes pixel-unit dependence."""
    # 同样的相对误差，不同的绝对量级
    pred_a = torch.tensor([10.0])
    gt_a = torch.tensor([12.0])
    scale_a = torch.tensor([12.0])

    pred_b = torch.tensor([100.0])
    gt_b = torch.tensor([120.0])
    scale_b = torch.tensor([120.0])

    loss_a = amplitude_loss(pred_a, gt_a, scale_a)
    loss_b = amplitude_loss(pred_b, gt_b, scale_b)
    assert torch.allclose(loss_a, loss_b, atol=1e-6), \
        "same relative error -> same normalized loss"


def test_make_burn_in_mask_min_floor_and_ratio():
    # 短序列：min_frames 主导
    m = make_burn_in_mask(T=10, burn_in_ratio=0.1, burn_in_min_frames=3,
                          burn_in_weight=0.1)
    assert torch.allclose(m[:3], torch.full((3,), 0.1), atol=1e-5)
    assert (m[3:] == 1.0).all()

    # 长序列：ratio 主导
    m2 = make_burn_in_mask(T=50, burn_in_ratio=0.2, burn_in_min_frames=3,
                           burn_in_weight=0.0)
    assert (m2[:10] == 0.0).all()
    assert (m2[10:] == 1.0).all()


def test_state_loss_full_forward_and_backward():
    B, T = 2, 12
    loss_fn = StateLoss(lambda_amp=0.5)
    state_pred = torch.randn(B, T, 4, requires_grad=True)
    phase = torch.randn(B, T)
    out = loss_fn(
        state_pred,
        phase_cos_gt=torch.cos(phase),
        phase_sin_gt=torch.sin(phase),
        amp_gt=torch.abs(torch.randn(B, T)) * 15.0,
        amp_scale=torch.tensor([18.0, 18.0]),
        valid_mask=torch.ones(B, T),
    )
    assert set(out.keys()) >= {
        "loss_state", "loss_phase", "loss_amp", "n_valid_frames",
    }
    out["loss_state"].backward()
    assert state_pred.grad is not None
    assert state_pred.grad.abs().sum() > 0


def test_state_loss_respects_valid_mask():
    """Sequences with valid_mask=0 must not pull the loss."""
    B, T = 2, 8
    loss_fn = StateLoss(lambda_amp=0.5,
                        burn_in_ratio=0.0, burn_in_min_frames=0)
    state_pred_a = torch.zeros(B, T, 4)
    state_pred_b = torch.zeros(B, T, 4)
    state_pred_b[1] = 1e6                                # 第二个序列预测灾难
    phase = torch.zeros(B, T)
    cos = torch.cos(phase)
    sin = torch.sin(phase)
    amp = torch.zeros(B, T)
    amp_scale = torch.tensor([1.0, 1.0])
    # 把第二个序列整段标记为 invalid
    valid = torch.ones(B, T)
    valid[1, :] = 0.0

    loss_a = loss_fn(state_pred_a, cos, sin, amp, amp_scale, valid)["loss_state"]
    loss_b = loss_fn(state_pred_b, cos, sin, amp, amp_scale, valid)["loss_state"]
    assert torch.allclose(loss_a, loss_b, atol=1e-5), \
        "invalid-masked sequence must not influence the loss"
