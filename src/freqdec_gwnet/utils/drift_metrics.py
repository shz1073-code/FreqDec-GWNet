"""Drift evaluation metrics required by PROJECT_CONSTRAINTS_19.6.md §3
and tech route §2.

Functions provided:

    * ``cumulative_drift_curve(U)``  — ``‖U_t‖₂`` over time (R3 cumsum monitor)
    * ``compensation_error(U_pred, U_gt)``  — pred-vs-GT distance
    * ``landmark_residual(...)``  — naive TRE-style before/after.  WARNING:
      mixes respiratory motion (which compensation *should* remove) with
      active instrument push (which it *should not*); kept for diagnostic
      use only — the headline number in the paper should be the next one.
    * ``respiratory_projected_residual(...)``  — TRE-style metric projected
      onto the respiratory principal axis.  This isolates the respiratory
      component of motion and matches the gold-standard methodology used in
      e.g. Shechter et al. 2004 (IEEE TMI), Ablitt et al. 2004, and
      Ambrosini et al. 2015.  **Use this for paper results.**

The functions are tensor-only (no plotting, no I/O); rendering is the
concern of the eval script.
"""

from __future__ import annotations

from typing import Dict

import torch


# ---------------------------------------------------------------------------
# 1. Cumulative drift curve — monitors R3's cumulative sum stability
# ---------------------------------------------------------------------------


def cumulative_drift_curve(U: torch.Tensor) -> torch.Tensor:
    """Return ``‖U_t‖₂`` along time.

    Args:
        U: ``[T, D]`` (single sequence) or ``[B, T, D]`` (batched). ``D`` is
            usually 2 (translation) but the function is dim-agnostic.

    Returns:
        ``[T]`` or ``[B, T]`` non-negative magnitudes.
    """
    if U.dim() not in (2, 3):
        raise ValueError(
            f"U must be [T, D] or [B, T, D], got {tuple(U.shape)}"
        )
    return U.norm(dim=-1)


# ---------------------------------------------------------------------------
# 2. Compensation error — when ground-truth motion is available
# ---------------------------------------------------------------------------


def compensation_error(
    U_pred: torch.Tensor,
    U_gt: torch.Tensor,
) -> torch.Tensor:
    """Per-frame Euclidean distance between predicted and GT cumulative motion.

    Args:
        U_pred: ``[T, D]`` or ``[B, T, D]``.
        U_gt: same shape as ``U_pred``.

    Returns:
        Per-frame error tensor with the leading dims of ``U_pred``.
    """
    if U_pred.shape != U_gt.shape:
        raise ValueError(
            f"shape mismatch: U_pred {tuple(U_pred.shape)} vs "
            f"U_gt {tuple(U_gt.shape)}"
        )
    return (U_pred - U_gt).norm(dim=-1)


# ---------------------------------------------------------------------------
# 3. Landmark residual — observed motion vs motion remaining after Δu warp
# ---------------------------------------------------------------------------


def landmark_residual(
    landmark_xy: torch.Tensor,        # [T, 2]; NaN where unannotated
    U: torch.Tensor,                  # [T, 2]
    reference_index: int,
) -> Dict[str, torch.Tensor]:
    """Measure how much of the tip's apparent motion is removed by ``U``.

    For each annotated frame ``t``:

        before_t = ‖landmark_t − landmark_ref‖
        after_t  = ‖(landmark_t − U_t) − landmark_ref‖

    Frames that are missing a landmark annotation (``NaN`` in ``landmark_xy``)
    are excluded from both vectors and from the summary statistic. If the
    reference frame itself is unannotated the function raises — landmark
    residual cannot be defined without a reference position.

    Args:
        landmark_xy: ``[T, 2]`` tip coordinates per frame; ``NaN`` for
            unannotated frames.
        U: ``[T, 2]`` predicted cumulative motion per frame.
        reference_index: ``r``; ``U[r]`` should be zero for the metric to be
            meaningful (this is enforced by RelativeMotionField).

    Returns:
        ``dict`` with:
            ``before``: ``[T]`` distance before compensation; ``NaN`` where
                landmark missing.
            ``after``: ``[T]`` distance after compensation; ``NaN`` where
                landmark missing.
            ``valid``: ``[T]`` bool mask of frames that contributed.
            ``improvement``: ``[T]`` of ``before − after``; ``NaN`` where
                ``valid`` is False.
            ``improvement_mean``: scalar mean of ``improvement`` over valid
                frames; ``NaN`` if there are zero valid frames.
    """
    if landmark_xy.dim() != 2 or landmark_xy.shape[-1] != 2:
        raise ValueError(
            f"landmark_xy must be [T, 2], got {tuple(landmark_xy.shape)}"
        )
    if U.shape != landmark_xy.shape:
        raise ValueError(
            f"U must match landmark_xy shape, got {tuple(U.shape)}"
        )
    T = landmark_xy.shape[0]
    if not (0 <= reference_index < T):
        raise IndexError(
            f"reference_index {reference_index} out of range [0, {T})"
        )

    landmark_present = ~torch.isnan(landmark_xy).any(dim=-1)   # [T] bool
    if not bool(landmark_present[reference_index]):
        raise ValueError(
            f"reference frame {reference_index} has no landmark annotation"
        )

    ref_xy = landmark_xy[reference_index]                       # [2]
    delta_before = landmark_xy - ref_xy                         # [T, 2]
    delta_after = (landmark_xy - U) - ref_xy                    # [T, 2]

    before = delta_before.norm(dim=-1)                          # [T]
    after = delta_after.norm(dim=-1)                            # [T]

    # 没标注的帧不能定义 before/after，置 NaN 让下游清晰
    nan_t = torch.full((), float("nan"),
                       device=before.device, dtype=before.dtype)
    before = torch.where(landmark_present, before, nan_t)
    after = torch.where(landmark_present, after, nan_t)
    improvement = before - after                                # NaN-aware

    if landmark_present.any():
        improvement_mean = improvement[landmark_present].mean()
    else:
        improvement_mean = nan_t.clone()

    return {
        "before": before,
        "after": after,
        "valid": landmark_present,
        "improvement": improvement,
        "improvement_mean": improvement_mean,
    }


# ---------------------------------------------------------------------------
# 4. Respiratory-projected residual — the *correct* paper metric
# ---------------------------------------------------------------------------


def respiratory_projected_residual(
    landmark_xy: torch.Tensor,        # [T, 2]; NaN where unannotated
    U: torch.Tensor,                  # [T, 2]
    reference_index: int,
    respiratory_axis: torch.Tensor,    # [2] unit vector (e.g. PCA axis from Phase B)
) -> Dict[str, torch.Tensor]:
    """Landmark drift projected onto the respiratory principal axis.

    The plain landmark residual conflates two sources of motion:

        1. respiratory drift of the patient (what compensation should remove)
        2. active push of the instrument by the operator (what it must *not*
           remove — that motion is intentional and clinically meaningful)

    On 2-D fluoroscopy the two are approximately orthogonal: respiratory
    motion is dominantly AP (perpendicular to the imaging plane projection
    of the spine), while catheter push is dominantly along the vessel
    direction. Projecting the landmark drift onto the respiratory axis
    isolates component (1), which is what the §III-C motion field aims to
    compensate, and matches the methodology used by Shechter et al. 2004,
    Ablitt et al. 2004, and Ambrosini et al. 2015.

    Args:
        landmark_xy: ``[T, 2]`` tip coordinates per frame; ``NaN`` where
            unannotated.
        U: ``[T, 2]`` predicted cumulative motion per frame.
        reference_index: ``r``.
        respiratory_axis: ``[2]`` unit vector. In FreqDec-GWNet the natural
            choice is the principal-axis vector emitted by the Phase B
            background-motion pipeline (see
            ``state_labels.band_filter.project_to_principal_axis``).

    Returns:
        ``dict`` with:
            ``before``: ``[T]`` ``|drift_t · e_resp|`` without compensation;
                NaN where landmark missing.
            ``after``: ``[T]`` ``|(drift_t − U_t) · e_resp|``; NaN where
                landmark missing.
            ``valid``: ``[T]`` bool mask.
            ``improvement``: ``before − after`` (positive = compensation
                reduced respiratory-direction drift).
            ``improvement_mean``: scalar mean over valid frames.
            ``reduction_ratio``: scalar ``after.mean() / before.mean()``
                — lower is better; 0 = perfect compensation along resp axis.
    """
    if landmark_xy.dim() != 2 or landmark_xy.shape[-1] != 2:
        raise ValueError(
            f"landmark_xy must be [T, 2], got {tuple(landmark_xy.shape)}"
        )
    if U.shape != landmark_xy.shape:
        raise ValueError("U must match landmark_xy shape")
    if respiratory_axis.shape[-1] != 2 or respiratory_axis.dim() != 1:
        raise ValueError(
            f"respiratory_axis must be [2], got {tuple(respiratory_axis.shape)}"
        )

    T = landmark_xy.shape[0]
    if not (0 <= reference_index < T):
        raise IndexError(
            f"reference_index {reference_index} out of range [0, {T})"
        )

    # Normalize the axis (defensive — should already be unit length)
    e = respiratory_axis / respiratory_axis.norm().clamp(min=1e-8)

    landmark_present = ~torch.isnan(landmark_xy).any(dim=-1)
    if not bool(landmark_present[reference_index]):
        raise ValueError(
            f"reference frame {reference_index} has no landmark annotation"
        )

    ref_xy = landmark_xy[reference_index]
    drift_uncomp = landmark_xy - ref_xy                         # [T, 2]
    drift_comp = drift_uncomp - U                                # [T, 2]

    # Project onto the respiratory axis (signed → take abs for magnitude)
    proj_before = (drift_uncomp @ e).abs()                       # [T]
    proj_after = (drift_comp @ e).abs()                          # [T]

    nan_t = torch.full((), float("nan"),
                       device=proj_before.device, dtype=proj_before.dtype)
    proj_before = torch.where(landmark_present, proj_before, nan_t)
    proj_after = torch.where(landmark_present, proj_after, nan_t)
    improvement = proj_before - proj_after

    if landmark_present.any():
        mean_before = proj_before[landmark_present].mean()
        mean_after = proj_after[landmark_present].mean()
        improvement_mean = improvement[landmark_present].mean()
        reduction_ratio = mean_after / mean_before.clamp(min=1e-8)
    else:
        mean_before = mean_after = improvement_mean = nan_t.clone()
        reduction_ratio = nan_t.clone()

    return {
        "before": proj_before,
        "after": proj_after,
        "valid": landmark_present,
        "improvement": improvement,
        "improvement_mean": improvement_mean,
        "mean_before": mean_before,
        "mean_after": mean_after,
        "reduction_ratio": reduction_ratio,
        "respiratory_axis": e,
    }
