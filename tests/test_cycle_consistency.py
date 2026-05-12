"""Tests for cycle-consistency analysis (paper §5.2 + constraints §3).

Coverage:
    * state_distance strategy detects near-cycle frame & computes E_cycle
    * phase_bin strategy detects bin-return + amplitude match
    * NO available cycle → ``available=False``, no fake cycle returned
    * E_cycle == 0 when U is constant (perfect compensation)
    * min_gap excludes trivially-near frames
    * shape / dtype error handling
"""

from __future__ import annotations

import math

import pytest
import torch

from freqdec_gwnet.utils.cycle_consistency import (
    CycleConsistencyResult,
    cycle_consistency_error,
)


# ---------------------------------------------------------------------------
# state_distance strategy
# ---------------------------------------------------------------------------


def test_state_distance_detects_near_cycle():
    """Frame K with same state as ref → low distance, picked as best K."""
    T = 100
    state = torch.randn(T, 3)
    ref = 5
    K = 60
    state[K] = state[ref] + 0.05 * torch.randn(3)        # near-cycle
    U = torch.randn(T, 2)

    out = cycle_consistency_error(
        state, U, reference_index=ref,
        strategy="state_distance",
        state_distance_threshold=0.30,
        min_gap_frames=8,
    )
    assert out.available
    assert out.K == K
    expected = float((U[K] - U[ref]).norm().item())
    assert out.E_cycle == pytest.approx(expected, rel=1e-5)
    assert out.strategy == "state_distance"


def test_state_distance_excludes_min_gap_frames():
    """Frames within min_gap of ref must NOT be eligible."""
    T = 50
    state = torch.zeros(T, 3)                            # all states identical
    state[ref := 10] = torch.tensor([0.0, 0.0, 0.0])
    U = torch.zeros(T, 2)

    out = cycle_consistency_error(
        state, U, reference_index=ref,
        strategy="state_distance",
        state_distance_threshold=0.10,
        min_gap_frames=10,
    )
    # 候选必须远离 ref 至少 10 帧
    if out.available:
        assert abs(out.K - ref) >= 10
    # 候选列表里不应有 |K - ref| < 10
    for K, _ in out.all_candidates:
        assert abs(K - ref) >= 10


def test_no_near_cycle_returns_not_available():
    """If every frame is far from reference, must report not_available."""
    T = 60
    state = torch.zeros(T, 3)
    # Reference is at a unique location no other frame approaches
    ref = 30
    state[ref] = torch.tensor([10.0, 10.0, 10.0])
    U = torch.randn(T, 2)

    out = cycle_consistency_error(
        state, U, reference_index=ref,
        strategy="state_distance",
        state_distance_threshold=0.10,
    )
    assert out.available is False
    assert out.K is None
    assert out.E_cycle is None
    assert out.status == "not_available"


def test_e_cycle_zero_for_constant_U():
    """If U is constant everywhere, E_cycle should be exactly 0."""
    T = 50
    state = torch.zeros(T, 3)                            # all states equal
    U = torch.ones(T, 2) * 3.14                          # constant
    out = cycle_consistency_error(
        state, U, reference_index=10,
        strategy="state_distance",
        state_distance_threshold=0.10,
        min_gap_frames=5,
    )
    assert out.available
    assert out.E_cycle == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# phase_bin strategy
# ---------------------------------------------------------------------------


def test_phase_bin_detects_bin_return():
    """Frame K in same phase bin and matching amplitude is detected."""
    T = 200
    fs = 15.0
    t = torch.arange(T) / fs
    f_resp = 0.30                                         # 0.3 Hz
    phase = 2 * math.pi * f_resp * t
    cos_phi = torch.cos(phase)
    sin_phi = torch.sin(phase)
    amp = torch.full((T,), 5.0)
    state = torch.stack([cos_phi, sin_phi, amp], dim=-1)

    ref = 30
    U = torch.zeros(T, 2)
    out = cycle_consistency_error(
        state, U, reference_index=ref,
        strategy="phase_bin",
        n_phase_bins=12,
        amplitude_tolerance=0.10,
        min_gap_frames=8,
    )
    assert out.available
    # 至少存在多个回到同 bin 的候选（一周期 = 1/0.3 ≈ 3.3 s ≈ 50 帧）
    assert len(out.all_candidates) >= 2


def test_phase_bin_rejects_amplitude_drift():
    """Same phase but big amplitude change → rejected."""
    T = 100
    cos_phi = torch.cos(torch.linspace(0, 2 * math.pi * 3, T))
    sin_phi = torch.sin(torch.linspace(0, 2 * math.pi * 3, T))
    amp = torch.linspace(5.0, 50.0, T)                    # 振幅严重漂移
    state = torch.stack([cos_phi, sin_phi, amp], dim=-1)
    U = torch.zeros(T, 2)

    out = cycle_consistency_error(
        state, U, reference_index=10,
        strategy="phase_bin",
        amplitude_tolerance=0.10,                         # 严格：10%
        min_gap_frames=10,
    )
    # 振幅相差太大 → 没有 K 能通过振幅过滤
    assert out.available is False


def test_phase_bin_rejects_short_state_seq():
    """phase_bin strategy needs state_seq with width >= 3."""
    state = torch.randn(10, 2)                            # 只有 2 维
    U = torch.zeros(10, 2)
    with pytest.raises(ValueError, match="phase_bin strategy"):
        cycle_consistency_error(
            state, U, reference_index=0,
            strategy="phase_bin",
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_unknown_strategy_raises():
    state = torch.zeros(10, 3)
    U = torch.zeros(10, 2)
    with pytest.raises(ValueError, match="unknown strategy"):
        cycle_consistency_error(state, U, reference_index=0, strategy="bogus")


def test_shape_mismatches_caught():
    with pytest.raises(ValueError, match=r"\[T, D\]"):
        cycle_consistency_error(
            torch.zeros(10), torch.zeros(10, 2), reference_index=0,
        )
    with pytest.raises(ValueError, match=r"\[T, 2\]"):
        cycle_consistency_error(
            torch.zeros(10, 3), torch.zeros(10, 3), reference_index=0,
        )
    with pytest.raises(ValueError, match="agree on T"):
        cycle_consistency_error(
            torch.zeros(10, 3), torch.zeros(8, 2), reference_index=0,
        )


def test_reference_index_out_of_range_raises():
    state = torch.zeros(10, 3)
    U = torch.zeros(10, 2)
    with pytest.raises(ValueError, match="out of"):
        cycle_consistency_error(state, U, reference_index=15)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


def test_result_status_strings():
    not_avail = CycleConsistencyResult(
        available=False, K=None, E_cycle=None,
    )
    assert not_avail.status == "not_available"

    avail = CycleConsistencyResult(
        available=True, K=42, E_cycle=1.234,
    )
    assert "1.2340" in avail.status
