"""Physiological priors for state-branch training (paper §III.E).

These auxiliary losses encode prior knowledge about respiratory dynamics
to reduce the effective hypothesis space the state branch must fit. With
only ~27 real training sequences, the unconstrained branch overfits;
adding physiologically-motivated constraints shrinks parameter degrees
of freedom and improves generalization.

Three priors are exposed; the StateLoss caller chooses which to enable:

1. ``phase_smoothness_loss`` — second-order finite difference on the
   predicted unwrapped phase. Penalizes "jagged" phase trajectories;
   regular respiration has near-constant phase advance Δφ ≈ 2π f_resp / fs
   per frame.

2. ``amplitude_smoothness_loss`` — first-order finite difference on the
   predicted amplitude. Real respiratory amplitude varies slowly (deep
   breath / shallow breath drift), not frame-to-frame chaos.

3. ``spectral_concentration_loss`` — fraction of phase-trajectory energy
   that falls *outside* the respiratory band (0.15-0.50 Hz at 15 fps).
   Out-of-band energy is penalized; this is the "band-limited" assumption
   that mirrors the Phase B label generation pipeline.

All three are *unsupervised* — they need only the model's own output and
do NOT need ground-truth labels — so they work in semi-supervised /
low-label regimes too.

Sign convention:
    cos_pred[..., t] = cos φ_pred(t)
    sin_pred[..., t] = sin φ_pred(t)
    phase φ recovered via atan2; unwrap via cumulative complex product
    to stay differentiable.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Phase smoothness — second-order finite difference
# ---------------------------------------------------------------------------


def phase_smoothness_loss(
    cos_pred: torch.Tensor,            # [..., T]
    sin_pred: torch.Tensor,            # [..., T]
    valid_mask: Optional[torch.Tensor] = None,  # [..., T] float in {0, 1}
) -> torch.Tensor:
    """Penalize non-uniform phase advance (φ'' should be small).

    For each consecutive pair (t-1, t), the phase advance is

        Δφ_t = atan2(sin(φ_t − φ_{t-1}), cos(φ_t − φ_{t-1}))

    which is computed without unwrapping by using the trig identities

        sin(Δφ) = sin φ_t cos φ_{t-1} − cos φ_t sin φ_{t-1}
        cos(Δφ) = cos φ_t cos φ_{t-1} + sin φ_t sin φ_{t-1}

    The second-order finite difference Δ²φ_t = Δφ_t − Δφ_{t-1} is then
    penalized via mean squared error. For perfectly periodic respiration
    Δφ is constant, so Δ²φ ≡ 0.

    Args:
        cos_pred / sin_pred: ``[B, T]`` or ``[..., T]`` unit-circle outputs.
        valid_mask: optional ``[..., T]`` mask. Frames marked invalid do
            not contribute. Must broadcast against ``cos_pred``.

    Returns:
        Scalar loss tensor.
    """
    if cos_pred.shape != sin_pred.shape:
        raise ValueError("cos_pred / sin_pred shape mismatch")
    if cos_pred.shape[-1] < 3:
        return cos_pred.new_zeros(())

    cos_prev = cos_pred[..., :-1]
    sin_prev = sin_pred[..., :-1]
    cos_cur = cos_pred[..., 1:]
    sin_cur = sin_pred[..., 1:]

    # Δφ_t between consecutive frames, using trig identity (differentiable,
    # avoids atan2-based wrap discontinuity)
    sin_dphi = sin_cur * cos_prev - cos_cur * sin_prev      # [..., T-1]
    cos_dphi = cos_cur * cos_prev + sin_cur * sin_prev      # [..., T-1]

    # Second-order: Δ²φ_t = Δφ_t − Δφ_{t-1}. We measure (sin(Δ²φ))^2 as
    # the penalty — a smooth, bounded surrogate that is small iff Δφ is
    # constant. Using the same identity:
    sin_ddphi = (
        sin_dphi[..., 1:] * cos_dphi[..., :-1]
        - cos_dphi[..., 1:] * sin_dphi[..., :-1]
    )                                                        # [..., T-2]

    if valid_mask is not None:
        # We need a valid mask aligned to T-2 (both endpoints of the second
        # difference must be valid)
        vm = valid_mask[..., 2:] * valid_mask[..., 1:-1] * valid_mask[..., :-2]
        denom = vm.sum().clamp(min=1.0)
        return (sin_ddphi ** 2 * vm).sum() / denom
    return (sin_ddphi ** 2).mean()


# ---------------------------------------------------------------------------
# 2. Amplitude smoothness — first-order finite difference
# ---------------------------------------------------------------------------


def amplitude_smoothness_loss(
    amp_pred: torch.Tensor,           # [..., T]
    amp_scale: Optional[torch.Tensor] = None,  # broadcastable to [..., 1]
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Penalize sudden amplitude jumps.

    Real respiratory amplitude modulates on a multi-second timescale, so
    frame-to-frame jumps |Δamp| should be small. We optionally normalize
    by ``amp_scale`` (the per-sequence P95 from Phase B) so the loss is
    scale-invariant across patients.

    Args:
        amp_pred: ``[B, T]`` or ``[..., T]``.
        amp_scale: ``[B]`` or scalar; used to normalize Δamp.
        valid_mask: ``[..., T]`` float mask.

    Returns:
        Scalar loss tensor.
    """
    if amp_pred.shape[-1] < 2:
        return amp_pred.new_zeros(())
    diff = amp_pred[..., 1:] - amp_pred[..., :-1]            # [..., T-1]
    if amp_scale is not None:
        scale = amp_scale
        # broadcast to leading dims
        while scale.dim() < diff.dim():
            scale = scale.unsqueeze(-1)
        diff = diff / scale.clamp(min=1e-3)
    if valid_mask is not None:
        vm = valid_mask[..., 1:] * valid_mask[..., :-1]
        denom = vm.sum().clamp(min=1.0)
        return (diff.abs() * vm).sum() / denom
    return diff.abs().mean()


# ---------------------------------------------------------------------------
# 3. Spectral concentration — most phase energy in respiratory band
# ---------------------------------------------------------------------------


def spectral_concentration_loss(
    cos_pred: torch.Tensor,           # [..., T]
    sin_pred: torch.Tensor,           # [..., T]
    fs: float = 15.0,
    band_hz: tuple = (0.15, 0.50),
    eps: float = 1e-8,
) -> torch.Tensor:
    """Loss = fraction of phase-trajectory FFT energy *outside* the
    respiratory band (0.15-0.50 Hz at 15 fps default).

    The predicted phase trajectory ``z_t = cos φ + i sin φ`` is interpreted
    as an analytic signal whose dominant frequency should be respiratory.
    Energy outside the band is "leakage" we penalize.

    Note: rFFT is applied along the LAST axis. Works with ``[B, T]`` or
    ``[..., T]`` shapes. The frequency grid is derived from ``fs``.

    Args:
        cos_pred / sin_pred: ``[..., T]`` predictions.
        fs: sampling rate.
        band_hz: ``(low, high)`` band edges.
        eps: numerical floor for the division.

    Returns:
        Scalar loss tensor in ``[0, 1]``. 0 means all energy is in-band
        (fully respiratory), 1 means all energy is outside the band.
    """
    if cos_pred.shape[-1] < 8:
        return cos_pred.new_zeros(())
    T = cos_pred.shape[-1]
    # 复信号
    z = cos_pred + 1j * sin_pred                              # [..., T]
    # FFT 沿时间轴；非 real 输入用 fft（不是 rfft）
    Z = torch.fft.fft(z, dim=-1)                              # [..., T]
    power = (Z.real ** 2 + Z.imag ** 2)                       # [..., T]
    freqs = torch.fft.fftfreq(T, d=1.0 / fs, device=cos_pred.device)
    band_mask = (freqs.abs() >= band_hz[0]) & (freqs.abs() <= band_hz[1])
    band_mask = band_mask.to(power.dtype)                     # [T]
    while band_mask.dim() < power.dim():
        band_mask = band_mask.unsqueeze(0)
    band_energy = (power * band_mask).sum(dim=-1)
    total_energy = power.sum(dim=-1).clamp(min=eps)
    out_of_band_frac = 1.0 - band_energy / total_energy       # [...]
    return out_of_band_frac.mean()


# ---------------------------------------------------------------------------
# Composite loss wrapper — convenience for StateLoss
# ---------------------------------------------------------------------------


def physiological_prior_loss(
    cos_pred: torch.Tensor,           # [B, T]
    sin_pred: torch.Tensor,
    amp_pred: torch.Tensor,
    amp_scale: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    *,
    lambda_phase_smooth: float = 0.0,
    lambda_amp_smooth: float = 0.0,
    lambda_spectral: float = 0.0,
    fs: float = 15.0,
    band_hz: tuple = (0.15, 0.50),
) -> dict:
    """Aggregate the three priors with configurable weights.

    Returns a dict with the total loss + each component for diagnostic
    logging.
    """
    if valid_mask is not None:
        valid_mask = valid_mask.to(cos_pred.dtype)

    L_phase = phase_smoothness_loss(cos_pred, sin_pred, valid_mask)
    L_amp = amplitude_smoothness_loss(amp_pred, amp_scale, valid_mask)
    L_spec = spectral_concentration_loss(
        cos_pred, sin_pred, fs=fs, band_hz=band_hz,
    )
    total = (
        lambda_phase_smooth * L_phase
        + lambda_amp_smooth * L_amp
        + lambda_spectral * L_spec
    )
    return {
        "loss_phys_prior": total,
        "L_phase_smooth": L_phase.detach(),
        "L_amp_smooth": L_amp.detach(),
        "L_spectral": L_spec.detach(),
    }
