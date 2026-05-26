"""Unit tests for physiological priors (paper §III.E)."""

from __future__ import annotations

import math

import pytest
import torch

from freqdec_gwnet.losses.physiological_priors import (
    amplitude_smoothness_loss,
    phase_smoothness_loss,
    physiological_prior_loss,
    spectral_concentration_loss,
)


# ---------------------------------------------------------------------------
# Phase smoothness
# ---------------------------------------------------------------------------


def test_phase_smoothness_zero_for_constant_rate():
    """Perfectly periodic phase advance → Δ²φ ≡ 0 → loss = 0."""
    T = 100
    fs = 15.0
    f_resp = 0.30
    t = torch.arange(T).float() / fs
    phi = 2 * math.pi * f_resp * t
    cos_p = torch.cos(phi).unsqueeze(0)                       # [1, T]
    sin_p = torch.sin(phi).unsqueeze(0)
    loss = phase_smoothness_loss(cos_p, sin_p)
    assert loss.item() < 1e-8


def test_phase_smoothness_positive_for_jagged_signal():
    """Random phases → high second-order difference → loss > 0."""
    torch.manual_seed(0)
    T = 100
    phi = torch.rand(1, T) * 2 * math.pi
    cos_p = torch.cos(phi)
    sin_p = torch.sin(phi)
    loss = phase_smoothness_loss(cos_p, sin_p)
    assert loss.item() > 0.1


def test_phase_smoothness_handles_valid_mask():
    T = 30
    cos_p = torch.ones(1, T)
    sin_p = torch.zeros(1, T)                                  # phase ≡ 0
    valid = torch.ones(1, T)
    valid[:, :5] = 0                                            # mask out beginning
    loss = phase_smoothness_loss(cos_p, sin_p, valid_mask=valid)
    assert loss.item() == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Amplitude smoothness
# ---------------------------------------------------------------------------


def test_amplitude_smoothness_zero_for_constant_amp():
    amp = torch.full((1, 50), 5.0)
    loss = amplitude_smoothness_loss(amp)
    assert loss.item() < 1e-8


def test_amplitude_smoothness_normalized_by_scale():
    """Scale-invariance: dividing amp by scale gives the same loss as
    dividing scale by 2 and the amp jumps by 2x."""
    amp_a = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    amp_b = torch.tensor([[0.0, 2.0, 0.0, 2.0]])
    scale_a = torch.tensor([1.0])
    scale_b = torch.tensor([2.0])
    L_a = amplitude_smoothness_loss(amp_a, scale_a)
    L_b = amplitude_smoothness_loss(amp_b, scale_b)
    assert L_a.item() == pytest.approx(L_b.item(), rel=1e-5)


# ---------------------------------------------------------------------------
# Spectral concentration
# ---------------------------------------------------------------------------


def test_spectral_concentration_low_for_pure_resp_signal():
    """Pure sinusoid at 0.3 Hz → most energy in band → loss near 0."""
    T = 200
    fs = 15.0
    f_resp = 0.30
    t = torch.arange(T).float() / fs
    phi = 2 * math.pi * f_resp * t
    cos_p = torch.cos(phi).unsqueeze(0)
    sin_p = torch.sin(phi).unsqueeze(0)
    loss = spectral_concentration_loss(cos_p, sin_p, fs=fs)
    assert loss.item() < 0.2                                   # 20% leakage tolerated for finite T


def test_spectral_concentration_high_for_out_of_band():
    """High-freq signal (2 Hz) → all energy out of band → loss near 1."""
    T = 200
    fs = 15.0
    f_high = 2.5
    t = torch.arange(T).float() / fs
    phi = 2 * math.pi * f_high * t
    cos_p = torch.cos(phi).unsqueeze(0)
    sin_p = torch.sin(phi).unsqueeze(0)
    loss = spectral_concentration_loss(cos_p, sin_p, fs=fs)
    assert loss.item() > 0.9


def test_spectral_concentration_short_signal_safe():
    """Signals < 8 frames return 0 without error."""
    loss = spectral_concentration_loss(
        torch.zeros(1, 4), torch.zeros(1, 4), fs=15.0,
    )
    assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# Composite wrapper
# ---------------------------------------------------------------------------


def test_composite_returns_diagnostics_and_total():
    T = 100
    cos_p = torch.cos(torch.linspace(0, 2 * math.pi * 3, T)).unsqueeze(0)
    sin_p = torch.sin(torch.linspace(0, 2 * math.pi * 3, T)).unsqueeze(0)
    amp_p = torch.full((1, T), 5.0)
    amp_scale = torch.tensor([5.0])
    out = physiological_prior_loss(
        cos_p, sin_p, amp_p, amp_scale,
        lambda_phase_smooth=1.0,
        lambda_amp_smooth=0.5,
        lambda_spectral=0.3,
    )
    assert set(out.keys()) == {
        "loss_phys_prior", "L_phase_smooth", "L_amp_smooth", "L_spectral",
    }
    assert out["loss_phys_prior"].requires_grad is False  # all components from no_grad inputs


def test_composite_grad_flows_when_inputs_require_grad():
    """Gradient should propagate through composite to the parameter."""
    T = 60
    fs = 15.0
    cos_p = torch.randn(1, T, requires_grad=True)
    sin_p = torch.randn(1, T, requires_grad=True)
    amp_p = torch.randn(1, T, requires_grad=True)
    out = physiological_prior_loss(
        cos_p, sin_p, amp_p, None,
        lambda_phase_smooth=1.0,
        lambda_amp_smooth=1.0,
        lambda_spectral=1.0,
        fs=fs,
    )
    out["loss_phys_prior"].backward()
    assert cos_p.grad is not None and cos_p.grad.abs().sum() > 0
    assert sin_p.grad is not None and sin_p.grad.abs().sum() > 0
    assert amp_p.grad is not None and amp_p.grad.abs().sum() > 0
