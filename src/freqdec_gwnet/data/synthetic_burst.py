"""Synthetic respiratory burst generator (paper §IV.B augmentation).

Builds a virtual fluoroscopy burst from a single static frame by applying
a known, controllable respiratory motion to the whole image. The generator
emits both the warped frames AND the ground-truth (cos φ_t, sin φ_t,
amplitude_t, valid_mask_t, amp_scale) — i.e. perfect Phase-B-style labels
without the noise of real-anchor estimation.

Usage in the paper:
    * **Augmentation for state-branch training**: each real reference frame
      seeds K virtual bursts with randomized respiratory frequency,
      amplitude, phase offset, noise. We document this clearly in §IV.B
      ("synthetic motion is a data-augmentation strategy; the model is
      evaluated on real data only"). This is standard practice.
    * **Diagnostic for Mamba**: with perfect supervision and effectively
      unlimited training data, we can isolate whether Mamba's failure on
      real data was due to (a) wrong inductive bias or (b) data scarcity.

The motion model has three components, all switchable:
    1. **Respiratory sinusoid** — primary: cos(2π f_resp t + φ_0) with
       slow amplitude modulation A(t).
    2. **Cardiac harmonic** (optional, default off) — smaller, higher-
       frequency oscillation at 0.8-1.5 Hz.
    3. **Drift** (optional) — slow trend across the burst, useful to test
       the cumulative-drift regularizer.

We only translate (no rotation/affine) so the ground-truth motion can be
expressed cleanly as a 2-D offset per frame, matching the cumulative
displacement formulation R3 expects.

Sign convention (pinned by ``tests/test_synthetic_burst.py``):
    ``frame_t(x, y) = static(x − dx_t, y − dy_t)``, where (dx_t, dy_t) is
    the per-frame translation. This matches Phase-B's phase-correlation
    output convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass
class SyntheticBurst:
    """Output of one synthesis call.

    Attributes:
        images: ``[T, H, W]`` uint8 — the warped frames.
        cos_phi / sin_phi / amplitude: ``[T]`` float32 — perfect ground-truth
            labels, ready to drop into a :class:`StateLabel`.
        valid_mask: ``[T]`` bool — all True for synthetic (no noisy anchor
            consensus failure).
        amp_scale: float — P95(amplitude), matches the StateLabel field.
        offsets_xy: ``[T, 2]`` float32 — the actual per-frame (dx, dy)
            applied; useful for unit tests and downstream debugging.
        params: dict — record of every parameter used (RNG seed, freq, amp,
            etc.) so the burst is fully reproducible.
    """
    images: np.ndarray
    cos_phi: np.ndarray
    sin_phi: np.ndarray
    amplitude: np.ndarray
    valid_mask: np.ndarray
    amp_scale: float
    offsets_xy: np.ndarray
    params: dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.images.shape[0])


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class SyntheticBurstGenerator:
    """Parameterized synthetic burst factory.

    Args:
        fs: sampling rate in Hz (e.g. 15 fps for typical fluoroscopy).
        resp_freq_hz_range: ``(low, high)`` for uniform sampling of
            respiratory frequency in Hz. Default (0.20, 0.40) covers
            12-24 BPM.
        resp_amp_y_px_range: vertical (AP-direction) amplitude in pixels.
            Default (4, 22) matches the amp_scale ranges observed in our
            real bursts (Phase B P95 of 5-30 px).
        resp_amp_x_px_range: horizontal amplitude. Default (0, 4) — small
            because respiratory motion is dominantly AP on PA-projection.
        amp_modulation_range: relative depth of amplitude breathing (deep
            breath vs shallow breath). Default (0.10, 0.40) means
            A(t) = A0 * (1 + r * sin(2π f_mod t + ψ_mod)).
        cardiac_amp_px_range: cardiac harmonic amplitude. Default (0, 0) =
            disabled. Enable for cardiac-pace stress tests.
        drift_px_range: monotonic drift over the burst. Default (0, 0) =
            disabled. Enable to stress-test zero-mean regularizer.
        noise_sigma_range: Gaussian intensity noise added per frame
            (in [0, 1] scale). Default (0.005, 0.020) — mild, matches
            real fluoroscopy.
        T_range: number of frames per burst.
        seed: base RNG seed.
    """

    def __init__(
        self,
        fs: float = 15.0,
        resp_freq_hz_range: Tuple[float, float] = (0.20, 0.40),
        resp_amp_y_px_range: Tuple[float, float] = (4.0, 22.0),
        resp_amp_x_px_range: Tuple[float, float] = (0.0, 4.0),
        amp_modulation_range: Tuple[float, float] = (0.10, 0.40),
        amp_modulation_freq_hz_range: Tuple[float, float] = (0.02, 0.05),
        cardiac_amp_px_range: Tuple[float, float] = (0.0, 0.0),
        cardiac_freq_hz_range: Tuple[float, float] = (0.8, 1.4),
        drift_px_range: Tuple[float, float] = (0.0, 0.0),
        noise_sigma_range: Tuple[float, float] = (0.005, 0.020),
        T_range: Tuple[int, int] = (120, 300),
        border_mode: int = cv2.BORDER_REFLECT,
        seed: int = 0,
    ):
        self.fs = float(fs)
        self.resp_freq_range = resp_freq_hz_range
        self.amp_y_range = resp_amp_y_px_range
        self.amp_x_range = resp_amp_x_px_range
        self.amp_mod_range = amp_modulation_range
        self.amp_mod_freq_range = amp_modulation_freq_hz_range
        self.cardiac_amp_range = cardiac_amp_px_range
        self.cardiac_freq_range = cardiac_freq_hz_range
        self.drift_range = drift_px_range
        self.noise_sigma_range = noise_sigma_range
        self.T_range = T_range
        self.border_mode = int(border_mode)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Core synthesis
    # ------------------------------------------------------------------

    def generate(
        self,
        static_image: np.ndarray,
        *,
        T: Optional[int] = None,
        params_override: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> SyntheticBurst:
        """Synthesize one burst from a single static frame.

        Args:
            static_image: ``[H, W]`` uint8 or float32 in [0, 1]. The "anchor"
                background; respiratory motion is applied around it.
            T: explicit burst length; if None, sampled from ``T_range``.
            params_override: dict overriding any of the sampled parameters
                (useful for unit tests). Keys: ``resp_freq, amp_y, amp_x,
                amp_mod, amp_mod_freq, phase_offset, cardiac_amp,
                cardiac_freq, drift_x, drift_y, noise_sigma``.
            seed: optional per-call seed (separate from the generator's
                internal RNG); fully determines the synthesis.

        Returns:
            :class:`SyntheticBurst`.
        """
        rng = (
            self._rng if seed is None
            else np.random.default_rng(int(seed))
        )

        if static_image.ndim != 2:
            raise ValueError(
                f"static_image must be [H, W]; got {static_image.shape}"
            )
        if static_image.dtype == np.uint8:
            base = static_image.astype(np.float32) / 255.0
        else:
            base = np.asarray(static_image, dtype=np.float32)

        H, W = base.shape
        if T is None:
            T = int(rng.integers(self.T_range[0], self.T_range[1] + 1))

        # Sample parameters
        def _u(lo, hi):
            return float(rng.uniform(lo, hi))

        params = {
            "T": int(T), "fs": self.fs,
            "resp_freq":      _u(*self.resp_freq_range),
            "amp_y":          _u(*self.amp_y_range),
            "amp_x":          _u(*self.amp_x_range),
            "amp_mod":        _u(*self.amp_mod_range),
            "amp_mod_freq":   _u(*self.amp_mod_freq_range),
            "phase_offset":   _u(0.0, 2.0 * np.pi),
            "cardiac_amp":    _u(*self.cardiac_amp_range),
            "cardiac_freq":   _u(*self.cardiac_freq_range),
            "drift_x":        _u(-self.drift_range[1], self.drift_range[1]) if self.drift_range[1] > 0 else 0.0,
            "drift_y":        _u(-self.drift_range[1], self.drift_range[1]) if self.drift_range[1] > 0 else 0.0,
            "noise_sigma":    _u(*self.noise_sigma_range),
        }
        if params_override:
            params.update(params_override)

        t = np.arange(T, dtype=np.float32) / self.fs                          # [T]
        phi = 2.0 * np.pi * params["resp_freq"] * t + params["phase_offset"]   # [T]
        cos_phi = np.cos(phi).astype(np.float32)
        sin_phi = np.sin(phi).astype(np.float32)

        # Amplitude (slowly modulated)
        mod = 1.0 + params["amp_mod"] * np.sin(
            2.0 * np.pi * params["amp_mod_freq"] * t
        )
        amplitude = (params["amp_y"] * mod).astype(np.float32)                # [T]

        # Per-frame translation
        dy = params["amp_y"] * mod * sin_phi
        dx = params["amp_x"] * mod * sin_phi
        # Cardiac harmonic (optional)
        if params["cardiac_amp"] > 0:
            cardiac_phi = 2.0 * np.pi * params["cardiac_freq"] * t
            dy = dy + params["cardiac_amp"] * np.sin(cardiac_phi)
        # Drift (optional, linear in time)
        if params["drift_x"] != 0 or params["drift_y"] != 0:
            ramp = t / max(1.0, float(t[-1]))   # 0..1
            dx = dx + params["drift_x"] * ramp
            dy = dy + params["drift_y"] * ramp
        offsets = np.stack([dx, dy], axis=-1).astype(np.float32)              # [T, 2]

        # Render the warped sequence
        images = np.empty((T, H, W), dtype=np.uint8)
        for k in range(T):
            M = np.array([[1.0, 0.0, float(dx[k])],
                          [0.0, 1.0, float(dy[k])]], dtype=np.float32)
            warped = cv2.warpAffine(
                base, M, (W, H), flags=cv2.INTER_LINEAR,
                borderMode=self.border_mode,
            )
            if params["noise_sigma"] > 0:
                warped = warped + rng.normal(
                    0.0, params["noise_sigma"], size=warped.shape,
                ).astype(np.float32)
                warped = np.clip(warped, 0.0, 1.0)
            images[k] = (warped * 255.0 + 0.5).clip(0, 255).astype(np.uint8)

        valid_mask = np.ones(T, dtype=bool)
        amp_scale = float(np.percentile(amplitude, 95.0))

        return SyntheticBurst(
            images=images,
            cos_phi=cos_phi, sin_phi=sin_phi, amplitude=amplitude,
            valid_mask=valid_mask, amp_scale=amp_scale,
            offsets_xy=offsets, params=params,
        )
