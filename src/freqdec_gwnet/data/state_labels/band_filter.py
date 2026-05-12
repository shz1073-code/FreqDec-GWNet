"""B.2 — band-limited filtering, principal-axis projection, Hilbert envelope.

Three primitives that turn a noisy 2-D background-motion signal into the
state quantities the constraints document mandates:

    1. :func:`band_pass_filter` — Butterworth zero-phase band-pass to keep
       only the respiratory band (default 0.15-0.5 Hz). PROJECT_CONSTRAINTS_19.6
       §1 R2 makes "band-limited filtering" a hard requirement.
    2. :func:`project_to_principal_axis` — PCA projection of the 2-D motion
       trajectory onto its dominant axis. The respiratory direction in
       fluoroscopy varies per patient (AP vs LR), so we let the data choose.
    3. :func:`hilbert_phase_amplitude` — instantaneous phase ``φ(t)`` and
       amplitude ``A(t)`` via the analytic signal. Phase is wrapped in
       ``[-π, π]``; amplitude is the envelope.

Sampling rate ``fs`` (Hz) is exposed at every entry point because the band
edges only make sense in physical units.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


# ---------------------------------------------------------------------------
# Band-pass filter
# ---------------------------------------------------------------------------


def band_pass_filter(
    signal: np.ndarray,
    fs: float,
    *,
    low_hz: float = 0.15,
    high_hz: float = 0.50,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter for a 1-D physiological signal.

    Zero phase is critical: causal IIR filters introduce a frequency-dependent
    delay that would scramble the alignment between Hilbert phase and the
    raw signal, breaking downstream multi-anchor consensus.

    Args:
        signal: ``[T]`` 1-D signal (length must be at least ``3*max(b, a)``;
            for order 4 that's roughly 12 samples — we enforce 16 to give
            ``filtfilt`` enough padding to avoid edge artifacts).
        fs: sampling rate in Hz.
        low_hz: low band edge in Hz (must be > 0).
        high_hz: high band edge in Hz (must be < fs/2).
        order: Butterworth order. Default 4 trades sharpness vs stability.

    Returns:
        ``[T]`` filtered signal of the same length.
    """
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D; got shape {signal.shape}")
    if signal.shape[0] < 16:
        raise ValueError(
            f"signal too short ({signal.shape[0]} samples) for stable "
            "filtfilt at order 4. Need ≥16 samples (≈1 s at 15 fps)."
        )
    if not 0.0 < low_hz < high_hz < fs / 2.0:
        raise ValueError(
            f"band edges out of range: low={low_hz}, high={high_hz}, "
            f"Nyquist={fs/2}. Required: 0 < low < high < fs/2."
        )

    nyq = fs / 2.0
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    return filtfilt(b, a, signal).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Principal-axis projection
# ---------------------------------------------------------------------------


def project_to_principal_axis(
    motion_2d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Project a 2-D trajectory onto its dominant axis (PCA).

    The respiratory direction in fluoroscopy is patient-specific — the AP
    (anterior-posterior) component is usually dominant but the exact axis
    in image coordinates varies with the C-arm angle. Using PCA lets the
    pipeline pick the right axis automatically.

    Args:
        motion_2d: ``[T, 2]`` 2-D trajectory (e.g. cumulative background
            position from :func:`cumulative_position`).

    Returns:
        ``projected[T]``: 1-D projection onto the principal axis (centered).
        ``axis[2]``: unit vector of the principal axis (image coords).
        ``explained_variance_ratio``: variance captured by the principal
            axis, in ``[0, 1]``. Values near 1 indicate the trajectory lies
            on a single line; values near 0.5 indicate isotropic noise.

    Edge case:
        If both eigenvalues are zero (constant signal), returns a zero
        projection and ``axis = (1, 0)`` by convention with ratio = 0.
    """
    if motion_2d.ndim != 2 or motion_2d.shape[1] != 2:
        raise ValueError(
            f"motion_2d must be [T, 2]; got shape {motion_2d.shape}"
        )

    centered = motion_2d - motion_2d.mean(axis=0, keepdims=True)
    # 协方差矩阵 (2x2)
    cov = (centered.T @ centered) / max(motion_2d.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh 返回升序；最大特征值在索引 -1
    total = float(eigvals.sum())
    if total < 1e-12:
        return (
            np.zeros(motion_2d.shape[0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
            0.0,
        )
    axis = eigvecs[:, -1].astype(np.float32)
    ratio = float(eigvals[-1] / total)

    # 让 axis 的方向稳定：要求第一个非零分量为正
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis

    projected = (centered @ axis).astype(np.float32, copy=False)
    return projected, axis, ratio


# ---------------------------------------------------------------------------
# Hilbert envelope + phase
# ---------------------------------------------------------------------------


def hilbert_phase_amplitude(
    signal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Instantaneous phase and amplitude via the analytic signal.

    Args:
        signal: ``[T]`` real-valued 1-D signal, ideally already band-pass
            filtered. Hilbert on broadband signals gives meaningless phase.

    Returns:
        ``phase[T]``: instantaneous phase in ``[-π, π]`` (wrapped).
        ``amplitude[T]``: envelope ``|analytic|`` (always non-negative).

    Note: the Hilbert transform has end-effect artifacts at both ends of
    the sequence (typically a few samples). Downstream code should rely on
    the multi-anchor consensus to mark these regions as low-quality rather
    than special-case here.
    """
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D; got shape {signal.shape}")
    analytic = hilbert(signal)
    phase = np.angle(analytic).astype(np.float32, copy=False)
    amplitude = np.abs(analytic).astype(np.float32, copy=False)
    return phase, amplitude
