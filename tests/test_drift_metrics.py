"""Tests for ``freqdec_gwnet.utils.drift_metrics``.

Pure-tensor metric tests with synthetic motions where the expected value can
be computed by hand. No real data needed.
"""

from __future__ import annotations

import math

import pytest
import torch

from freqdec_gwnet.utils.drift_metrics import (
    compensation_error,
    cumulative_drift_curve,
    landmark_residual,
    respiratory_projected_residual,
)


# ---------------------------------------------------------------------------
# cumulative_drift_curve
# ---------------------------------------------------------------------------


def test_cumulative_drift_curve_single_seq():
    U = torch.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 5.0]])
    curve = cumulative_drift_curve(U)
    assert curve.shape == (3,)
    assert torch.allclose(curve, torch.tensor([0.0, 5.0, 5.0]))


def test_cumulative_drift_curve_batched():
    U = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [0.0, 2.0]],
    ])  # [2, 2, 2]
    curve = cumulative_drift_curve(U)
    assert curve.shape == (2, 2)
    assert torch.allclose(curve[0], torch.tensor([0.0, 1.0]))
    assert torch.allclose(curve[1], torch.tensor([0.0, 2.0]))


def test_cumulative_drift_curve_rejects_wrong_shape():
    with pytest.raises(ValueError):
        cumulative_drift_curve(torch.zeros(5))


# ---------------------------------------------------------------------------
# compensation_error
# ---------------------------------------------------------------------------


def test_compensation_error_zero_when_pred_equals_gt():
    U = torch.randn(7, 2)
    err = compensation_error(U, U.clone())
    assert torch.allclose(err, torch.zeros(7), atol=1e-7)


def test_compensation_error_known_value():
    pred = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    gt = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    err = compensation_error(pred, gt)
    assert torch.allclose(err, torch.tensor([0.0, 3.0, 4.0]))


def test_compensation_error_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        compensation_error(torch.zeros(3, 2), torch.zeros(4, 2))


# ---------------------------------------------------------------------------
# landmark_residual
# ---------------------------------------------------------------------------


def test_landmark_residual_perfect_compensation_zeros_after():
    """If U exactly matches the apparent landmark motion, after = 0."""
    landmark = torch.tensor([
        [10.0, 10.0],
        [13.0, 14.0],
        [10.0, 14.0],
    ])
    # 这里参考帧是 0；apparent 位移就是 landmark - landmark[0]
    U = torch.tensor([
        [0.0, 0.0],
        [3.0, 4.0],
        [0.0, 4.0],
    ])
    out = landmark_residual(landmark, U, reference_index=0)
    assert torch.allclose(out["after"], torch.zeros(3), atol=1e-6)
    assert torch.allclose(out["before"],
                          torch.tensor([0.0, 5.0, 4.0]), atol=1e-6)
    assert torch.allclose(out["improvement"],
                          torch.tensor([0.0, 5.0, 4.0]), atol=1e-6)
    assert math.isclose(out["improvement_mean"].item(), 3.0, abs_tol=1e-6)


def test_landmark_residual_zero_compensation_keeps_before_equal_after():
    landmark = torch.tensor([
        [10.0, 10.0],
        [12.0, 13.0],
    ])
    U = torch.zeros_like(landmark)
    out = landmark_residual(landmark, U, reference_index=0)
    assert torch.allclose(out["before"], out["after"])
    assert torch.allclose(out["improvement"], torch.zeros(2))


def test_landmark_residual_skips_nan_frames():
    nan = float("nan")
    landmark = torch.tensor([
        [10.0, 10.0],
        [nan, nan],          # 没标注 -> 应跳过
        [12.0, 12.0],
    ])
    U = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],          # 这帧 U 给了，但因为没 landmark，不能算 residual
        [2.0, 2.0],
    ])
    out = landmark_residual(landmark, U, reference_index=0)
    assert out["valid"].tolist() == [True, False, True]
    assert torch.isnan(out["before"][1])
    assert torch.isnan(out["after"][1])
    assert torch.isnan(out["improvement"][1])
    # mean 只对 valid 帧求
    valid_improvement = out["improvement"][out["valid"]]
    assert math.isclose(out["improvement_mean"].item(),
                         valid_improvement.mean().item(), abs_tol=1e-7)


def test_landmark_residual_reference_must_be_annotated():
    landmark = torch.tensor([
        [float("nan"), float("nan")],
        [10.0, 10.0],
    ])
    U = torch.zeros_like(landmark)
    with pytest.raises(ValueError, match="no landmark annotation"):
        landmark_residual(landmark, U, reference_index=0)


def test_landmark_residual_reference_out_of_range():
    landmark = torch.tensor([[10.0, 10.0], [12.0, 12.0]])
    U = torch.zeros_like(landmark)
    with pytest.raises(IndexError):
        landmark_residual(landmark, U, reference_index=5)


def test_landmark_residual_pipeline_with_relative_motion_field():
    """End-to-end: feed RelativeMotionField outputs into the metric."""
    from freqdec_gwnet.models.relative_motion_field import RelativeMotionField

    # 构造一段 5 帧的"假状态"，让 RelativeMotionField 的 zero-init 给出 U=0
    s = torch.randn(1, 5, 3)
    rmf = RelativeMotionField()
    U = rmf(s, reference_index=0)["U"][0]      # [T, 2]

    landmark = torch.tensor([
        [50.0, 50.0],
        [51.0, 50.0],
        [52.0, 50.0],
        [53.0, 50.0],
        [54.0, 50.0],
    ])
    out = landmark_residual(landmark, U, reference_index=0)
    # 未训练情况下 U=0，after 必须等于 before
    assert torch.allclose(out["before"], out["after"])


# ---------------------------------------------------------------------------
# respiratory_projected_residual — paper headline metric
# ---------------------------------------------------------------------------


def test_respiratory_projected_pure_active_push_gives_zero_resp_residual():
    """Wire pushed purely along x; respiratory axis is y; the projected
    residual along y is therefore zero regardless of U. Confirms the
    metric does NOT punish active push that's orthogonal to respiration.
    """
    T = 5
    # Active push purely along +x
    landmark = torch.tensor([
        [50.0, 50.0],
        [60.0, 50.0],
        [70.0, 50.0],
        [80.0, 50.0],
        [90.0, 50.0],
    ])
    U = torch.zeros(T, 2)
    e_resp = torch.tensor([0.0, 1.0])      # y is respiratory
    out = respiratory_projected_residual(
        landmark, U, reference_index=0, respiratory_axis=e_resp,
    )
    assert torch.allclose(out["before"], torch.zeros(T))
    assert torch.allclose(out["after"], torch.zeros(T))


def test_respiratory_projected_perfect_compensation_zeros_after():
    """Landmark moves ONLY along respiratory axis; U perfectly tracks it;
    the after-projection must be all zero, improvement = before sum.
    """
    T = 5
    e_resp = torch.tensor([0.0, 1.0])
    # Pure respiratory drift along y
    landmark = torch.tensor([
        [50.0, 50.0],
        [50.0, 55.0],
        [50.0, 60.0],
        [50.0, 55.0],
        [50.0, 50.0],
    ])
    # Perfect U cancels respiratory drift (same direction)
    U = landmark - landmark[0]
    out = respiratory_projected_residual(
        landmark, U, reference_index=0, respiratory_axis=e_resp,
    )
    assert torch.allclose(
        out["after"], torch.zeros(T), atol=1e-5,
    )
    assert out["reduction_ratio"].item() < 1e-5


def test_respiratory_projected_axis_decomposition():
    """Landmark moves diagonally; respiratory axis is y; only the y-component
    is reported. A non-respiratory U should leave the y-residual untouched.
    """
    T = 4
    e_resp = torch.tensor([0.0, 1.0])
    landmark = torch.tensor([
        [50.0, 50.0],
        [60.0, 60.0],          # diagonal: 10 in x, 10 in y
        [70.0, 70.0],
        [80.0, 80.0],
    ])
    # U only along x (does not address respiratory)
    U = torch.zeros(T, 2)
    U[:, 0] = torch.tensor([0.0, 10.0, 20.0, 30.0])
    out = respiratory_projected_residual(
        landmark, U, reference_index=0, respiratory_axis=e_resp,
    )
    # before y-drift: 0, 10, 20, 30
    assert torch.allclose(out["before"], torch.tensor([0., 10., 20., 30.]))
    # after y-drift unchanged (U is orthogonal to e_resp)
    assert torch.allclose(out["after"], torch.tensor([0., 10., 20., 30.]))


def test_respiratory_projected_nan_landmark_skipped():
    nan = float("nan")
    T = 5
    landmark = torch.tensor([
        [50.0, 50.0],
        [nan, nan],
        [50.0, 60.0],
        [nan, nan],
        [50.0, 50.0],
    ])
    U = torch.zeros(T, 2)
    e_resp = torch.tensor([0.0, 1.0])
    out = respiratory_projected_residual(
        landmark, U, reference_index=0, respiratory_axis=e_resp,
    )
    assert torch.isnan(out["before"][1])
    assert torch.isnan(out["after"][3])
    # 只在 valid 帧上算 mean
    expected_mean_before = (0.0 + 10.0 + 0.0) / 3
    assert out["mean_before"].item() == pytest.approx(expected_mean_before)


def test_respiratory_projected_axis_normalization():
    """Non-unit axis must be normalized internally — the answer should not
    depend on the magnitude of ``respiratory_axis``.
    """
    landmark = torch.tensor([[0., 0.], [0., 10.], [0., 20.]])
    U = torch.zeros(3, 2)
    e1 = torch.tensor([0.0, 1.0])
    e2 = torch.tensor([0.0, 5.0])           # same direction, 5× magnitude
    out1 = respiratory_projected_residual(
        landmark, U, reference_index=0, respiratory_axis=e1,
    )
    out2 = respiratory_projected_residual(
        landmark, U, reference_index=0, respiratory_axis=e2,
    )
    assert torch.allclose(out1["before"], out2["before"])
    assert torch.allclose(out1["mean_before"], out2["mean_before"])


def test_respiratory_projected_reference_must_be_annotated():
    nan = float("nan")
    landmark = torch.tensor([[nan, nan], [50.0, 60.0]])
    U = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="reference frame"):
        respiratory_projected_residual(
            landmark, U, reference_index=0,
            respiratory_axis=torch.tensor([0., 1.]),
        )


def test_respiratory_projected_reduction_ratio_meaningful():
    """Reduction ratio < 1 means compensation helped along respiratory axis;
    > 1 means it made things worse along that direction.
    """
    T = 5
    e = torch.tensor([0.0, 1.0])
    # Pure y drift
    landmark = torch.tensor([
        [0., 0.], [0., 5.], [0., 10.], [0., 5.], [0., 0.],
    ])
    # Half-correct compensation (only removes 50% of resp drift)
    U_half = (landmark - landmark[0]) * 0.5
    out = respiratory_projected_residual(
        landmark, U_half, reference_index=0, respiratory_axis=e,
    )
    assert 0.0 < out["reduction_ratio"].item() < 1.0
    assert math.isclose(out["reduction_ratio"].item(), 0.5, abs_tol=1e-5)
