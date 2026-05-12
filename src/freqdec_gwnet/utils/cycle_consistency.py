"""Cycle-consistency analysis (PROJECT_CONSTRAINTS_19.6 §3 + paper §5.2).

When a fluoroscopy burst contains a frame ``K`` whose physiological state is
near the reference frame's state, the cumulative displacement should return
to zero — any residual is genuine drift. We quantify this via:

    E_cycle = ‖U_K − U_ref‖

where ``U_ref`` is zero by R3 convention. The constraints document is
explicit: **if no near-cycle exists in the burst, report "not available"
and do not fake one**. This module therefore separates detection from
computation, and the detection step can return ``None`` cleanly.

Two detection strategies are supported (matching paper §5.2):

    1. ``state_distance`` — ``‖s_K − s_ref‖₂ < threshold``. Simple, robust;
       requires comparable amplitude scales.
    2. ``phase_bin``     — phase bin of ``K`` returns to bin of ``ref`` AND
       amplitude is within ``amplitude_tolerance`` of the reference.
       Decouples respiratory phase from amplitude drift.

Both are exposed as standalone functions plus a high-level dispatcher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CycleConsistencyResult:
    """Outcome of a cycle-consistency analysis on one burst.

    Attributes:
        available: True if at least one near-cycle frame K was detected.
        K: index of the chosen near-cycle frame (or None if not available).
        E_cycle: ``‖U_K − U_ref‖₂`` for the chosen K (or None).
        all_candidates: list of ``(K, distance)`` pairs that passed
            detection — useful for ablation analysis.
        strategy: the detection strategy used ("state_distance" / "phase_bin").
        details: free-form metadata (threshold, n_phase_bins, ...).
    """

    available: bool
    K: Optional[int]
    E_cycle: Optional[float]
    all_candidates: List[Tuple[int, float]] = field(default_factory=list)
    strategy: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """Human-readable status for CSV/PNG headers."""
        if not self.available:
            return "not_available"
        return f"E_cycle={self.E_cycle:.4f}"


# ---------------------------------------------------------------------------
# Strategy 1: state-distance
# ---------------------------------------------------------------------------


def _state_distance_candidates(
    state_seq: torch.Tensor,                # [T, D]
    reference_index: int,
    threshold: float,
    *,
    min_gap: int,
) -> List[Tuple[int, float]]:
    """Frames whose state is within ``threshold`` of the reference state.

    Args:
        state_seq: shape ``[T, D]`` (D usually 3 = cos, sin, amp).
        reference_index: index of the reference frame.
        threshold: maximum Euclidean distance ``‖s_K − s_ref‖₂``.
        min_gap: minimum frame distance ``|K − ref|`` to qualify; small
            gaps are excluded because trivially-near frames are not
            informative cycle returns.

    Returns:
        Sorted list of ``(K, distance)`` pairs where distance < threshold,
        excluding the reference itself and frames within ``min_gap``.
    """
    s_ref = state_seq[reference_index]
    diff = state_seq - s_ref.unsqueeze(0)        # [T, D]
    dist = diff.norm(dim=-1)                      # [T]
    out: List[Tuple[int, float]] = []
    T = state_seq.shape[0]
    for t in range(T):
        if abs(t - reference_index) < min_gap:
            continue
        d = float(dist[t].item())
        if d < threshold:
            out.append((t, d))
    out.sort(key=lambda kv: kv[1])
    return out


# ---------------------------------------------------------------------------
# Strategy 2: phase-bin return
# ---------------------------------------------------------------------------


def _phase_bin_candidates(
    cos_phi: torch.Tensor,                  # [T]
    sin_phi: torch.Tensor,                  # [T]
    amplitude: torch.Tensor,                # [T]
    reference_index: int,
    *,
    n_phase_bins: int,
    amplitude_tolerance: float,
    min_gap: int,
) -> List[Tuple[int, float]]:
    """Frames whose phase returns to the reference bin AND amp is near.

    Args:
        cos_phi / sin_phi: ``[T]`` phase components on the unit circle.
        amplitude: ``[T]`` Hilbert envelope (raw px units).
        reference_index: ``ref`` frame index.
        n_phase_bins: how many bins to discretize ``[0, 2π)``.
        amplitude_tolerance: relative tolerance ``|A_K − A_ref| / A_ref``.
        min_gap: skip frames within this many of ``ref``.

    Returns:
        Sorted ``(K, score)`` pairs where score is a combined distance
        metric (smaller = better). Score = phase_bin_distance + amp_distance.
    """
    T = cos_phi.shape[0]
    phase = torch.atan2(sin_phi, cos_phi)         # [-π, π]
    bin_size = 2 * torch.pi / n_phase_bins
    bins = ((phase + torch.pi) / bin_size).floor().long() % n_phase_bins

    ref_bin = int(bins[reference_index].item())
    ref_amp = max(float(amplitude[reference_index].item()), 1e-6)

    out: List[Tuple[int, float]] = []
    for t in range(T):
        if abs(t - reference_index) < min_gap:
            continue
        if int(bins[t].item()) != ref_bin:
            continue
        amp_t = float(amplitude[t].item())
        rel_amp_dist = abs(amp_t - ref_amp) / ref_amp
        if rel_amp_dist > amplitude_tolerance:
            continue
        # 复合得分：phase bin 是 0（同 bin），主导项是相对振幅差
        out.append((t, rel_amp_dist))
    out.sort(key=lambda kv: kv[1])
    return out


# ---------------------------------------------------------------------------
# High-level entry: pick best K and compute E_cycle
# ---------------------------------------------------------------------------


def cycle_consistency_error(
    state_seq: torch.Tensor,                          # [T, D]
    U: torch.Tensor,                                   # [T, 2]
    reference_index: int,
    *,
    strategy: Literal["state_distance", "phase_bin"] = "state_distance",
    state_distance_threshold: float = 0.30,
    n_phase_bins: int = 12,
    amplitude_tolerance: float = 0.20,
    min_gap_frames: int = 8,
) -> CycleConsistencyResult:
    """Detect a near-cycle frame K and compute ``E_cycle = ‖U_K − U_ref‖₂``.

    Args:
        state_seq: ``[T, D]`` state sequence. For ``state_distance`` strategy
            ``D`` is unrestricted; for ``phase_bin`` strategy ``D >= 3`` and
            the first three columns are interpreted as
            ``(cos_phi, sin_phi, amplitude)``.
        U: ``[T, 2]`` cumulative displacement from
            :class:`RelativeMotionField`.
        reference_index: ``r`` from R3 (typically the chosen reference).
        strategy: detection strategy (see module docstring).
        state_distance_threshold: threshold for ``state_distance`` strategy.
            Default 0.30 fits a normalized 3-D state with cos/sin in
            ``[-1, 1]`` and amplitude near 1 (after dividing by amp_scale).
        n_phase_bins: bin count for ``phase_bin`` strategy.
        amplitude_tolerance: relative amplitude tolerance for phase-bin.
        min_gap_frames: minimum ``|K − ref|`` so the candidate is
            non-trivial. Default 8 frames (≈0.5 s at 15 fps).

    Returns:
        :class:`CycleConsistencyResult` with ``available=False`` if no K
        passed detection — never silently substitutes a fake cycle.
    """
    if state_seq.dim() != 2:
        raise ValueError(f"state_seq must be [T, D]; got {state_seq.shape}")
    if U.dim() != 2 or U.shape[1] != 2:
        raise ValueError(f"U must be [T, 2]; got {U.shape}")
    if state_seq.shape[0] != U.shape[0]:
        raise ValueError("state_seq and U must agree on T")
    T = state_seq.shape[0]
    if not 0 <= reference_index < T:
        raise ValueError(
            f"reference_index={reference_index} out of [0, {T})"
        )

    if strategy == "state_distance":
        candidates = _state_distance_candidates(
            state_seq, reference_index,
            threshold=state_distance_threshold,
            min_gap=min_gap_frames,
        )
        details = {
            "threshold": state_distance_threshold,
            "min_gap_frames": min_gap_frames,
            "n_candidates": len(candidates),
        }
    elif strategy == "phase_bin":
        if state_seq.shape[1] < 3:
            raise ValueError(
                "phase_bin strategy needs state_seq[:, :3] = "
                "(cos, sin, amp); got width "
                f"{state_seq.shape[1]}"
            )
        candidates = _phase_bin_candidates(
            cos_phi=state_seq[:, 0],
            sin_phi=state_seq[:, 1],
            amplitude=state_seq[:, 2],
            reference_index=reference_index,
            n_phase_bins=n_phase_bins,
            amplitude_tolerance=amplitude_tolerance,
            min_gap=min_gap_frames,
        )
        details = {
            "n_phase_bins": n_phase_bins,
            "amplitude_tolerance": amplitude_tolerance,
            "min_gap_frames": min_gap_frames,
            "n_candidates": len(candidates),
        }
    else:
        raise ValueError(f"unknown strategy '{strategy}'")

    if not candidates:
        # 严格遵守 §5.2 红线：找不到 near-cycle 就报告 not available
        return CycleConsistencyResult(
            available=False,
            K=None,
            E_cycle=None,
            all_candidates=[],
            strategy=strategy,
            details=details,
        )

    K_best = candidates[0][0]
    # E_cycle = ‖U_K − U_ref‖₂ ；R3 约定 U_ref = 0，但写成显式差更稳
    delta = U[K_best] - U[reference_index]
    e_cycle = float(delta.norm().item())

    return CycleConsistencyResult(
        available=True,
        K=K_best,
        E_cycle=e_cycle,
        all_candidates=candidates,
        strategy=strategy,
        details=details,
    )
