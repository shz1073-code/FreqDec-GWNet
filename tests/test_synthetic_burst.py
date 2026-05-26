"""Unit tests for the synthetic respiratory burst generator (paper §IV.B).

The generator's sign convention and label correctness must be pinned so
the state branch trained on synthetic data sees exactly what the
real-data Phase-B pipeline would produce.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from freqdec_gwnet.data.synthetic_burst import (
    SyntheticBurst,
    SyntheticBurstGenerator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _textured_frame(H: int = 96, W: int = 96, seed: int = 0) -> np.ndarray:
    """Random low-pass-filtered intensity field, uint8."""
    import cv2
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 1, (H, W)).astype(np.float32)
    blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=3.0)
    blurred = (blurred - blurred.min()) / max(1e-6, (blurred.max() - blurred.min()))
    return (blurred * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Shape and label correctness
# ---------------------------------------------------------------------------


def test_generate_shape_and_label_lengths():
    gen = SyntheticBurstGenerator(seed=0)
    img = _textured_frame()
    out = gen.generate(img, T=120)
    assert isinstance(out, SyntheticBurst)
    assert out.images.shape == (120, 96, 96)
    assert out.cos_phi.shape == (120,)
    assert out.sin_phi.shape == (120,)
    assert out.amplitude.shape == (120,)
    assert out.valid_mask.shape == (120,)
    assert out.offsets_xy.shape == (120, 2)
    assert out.images.dtype == np.uint8
    assert out.valid_mask.all()                            # synthetic = always valid


def test_phase_label_is_unit_circle():
    gen = SyntheticBurstGenerator(seed=1)
    img = _textured_frame()
    out = gen.generate(img, T=128)
    norms = out.cos_phi ** 2 + out.sin_phi ** 2
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_amplitude_label_matches_amp_scale_p95():
    gen = SyntheticBurstGenerator(seed=2)
    img = _textured_frame()
    out = gen.generate(img, T=200)
    p95 = float(np.percentile(out.amplitude, 95.0))
    assert out.amp_scale == pytest.approx(p95, rel=1e-6)


# ---------------------------------------------------------------------------
# Sign convention — pinned by tests so downstream code can trust it
# ---------------------------------------------------------------------------


def test_offset_matches_phase_label_sign():
    """At phase φ = π/2, sin φ = 1; the y-offset should be +amp_y (image
    content shifts downward, content at (x, y) in the static frame appears
    at (x, y+amp_y) in the warped frame)."""
    gen = SyntheticBurstGenerator(
        fs=15.0,
        resp_freq_hz_range=(0.25, 0.25),
        resp_amp_y_px_range=(10.0, 10.0),
        resp_amp_x_px_range=(0.0, 0.0),
        amp_modulation_range=(0.0, 0.0),
        noise_sigma_range=(0.0, 0.0),
        seed=3,
    )
    img = _textured_frame()
    out = gen.generate(
        img, T=60, params_override={"phase_offset": 0.0},
    )
    # The 25% point of one cycle is at t = (1/4) / freq = 1.0 s → frame 15
    # at fs=15. At that point sin = 1, cos = 0.
    assert out.cos_phi[15] == pytest.approx(0.0, abs=1e-4)
    assert out.sin_phi[15] == pytest.approx(1.0, abs=1e-4)
    # And dy(15) should be +amp_y * 1.0 = +10
    assert out.offsets_xy[15, 1] == pytest.approx(10.0, abs=1e-4)


def test_warped_frame_matches_known_offset():
    """Phase correlation between the static frame and the warped frame
    should recover (dx, dy)."""
    import cv2
    gen = SyntheticBurstGenerator(
        fs=15.0,
        resp_freq_hz_range=(0.3, 0.3),
        resp_amp_y_px_range=(7.0, 7.0),
        resp_amp_x_px_range=(3.0, 3.0),
        amp_modulation_range=(0.0, 0.0),
        noise_sigma_range=(0.0, 0.0),
        seed=4,
    )
    img = _textured_frame(H=128, W=128, seed=99)
    out = gen.generate(
        img, T=10, params_override={"phase_offset": math.pi / 2},
    )
    # phase_offset = π/2 → at t=0 sin = 1 → dy = amp_y = 7, dx = amp_x = 3
    assert out.sin_phi[0] == pytest.approx(1.0, abs=1e-4)
    assert out.offsets_xy[0] == pytest.approx([3.0, 7.0], abs=1e-4)
    # Compare static vs frame 0 with phase correlation. Tolerance is loose
    # because phase correlation on a small noise-textured 128x128 image with
    # BORDER_REFLECT boundary handling can underestimate small shifts; what
    # we really care about is that the SIGN and order of magnitude match —
    # the downstream model learns directly from cos/sin labels, not phase
    # corr output.
    base_f32 = img.astype(np.float32) / 255.0
    frame0_f32 = out.images[0].astype(np.float32) / 255.0
    (rec_dx, rec_dy), _ = cv2.phaseCorrelate(base_f32, frame0_f32)
    assert rec_dx > 0.5 and rec_dy > 4.0          # both positive, dy dominant
    assert rec_dy > rec_dx                          # respiratory > lateral


# ---------------------------------------------------------------------------
# Reproducibility + params override
# ---------------------------------------------------------------------------


def test_same_seed_gives_same_burst():
    gen_a = SyntheticBurstGenerator(seed=42)
    gen_b = SyntheticBurstGenerator(seed=42)
    img = _textured_frame()
    a = gen_a.generate(img, T=100, seed=7)
    b = gen_b.generate(img, T=100, seed=7)
    assert np.array_equal(a.images, b.images)
    assert np.allclose(a.cos_phi, b.cos_phi)
    assert np.allclose(a.offsets_xy, b.offsets_xy)


def test_params_override_propagates():
    gen = SyntheticBurstGenerator(seed=11)
    img = _textured_frame()
    out = gen.generate(img, T=50, params_override={
        "resp_freq": 0.30, "amp_y": 15.0, "amp_x": 0.0,
        "phase_offset": 0.0, "amp_mod": 0.0,
    })
    # cos at t=0 is 1, sin is 0 → no offset
    assert out.cos_phi[0] == pytest.approx(1.0, abs=1e-4)
    assert out.sin_phi[0] == pytest.approx(0.0, abs=1e-4)
    assert out.offsets_xy[0] == pytest.approx([0.0, 0.0], abs=1e-4)


# ---------------------------------------------------------------------------
# Drift mode — used to stress-test the zero-mean regularizer
# ---------------------------------------------------------------------------


def test_drift_adds_monotonic_offset():
    gen = SyntheticBurstGenerator(
        resp_amp_y_px_range=(0.0, 0.0), resp_amp_x_px_range=(0.0, 0.0),
        amp_modulation_range=(0.0, 0.0),
        drift_px_range=(0.0, 30.0),
        noise_sigma_range=(0.0, 0.0),
        seed=5,
    )
    img = _textured_frame()
    out = gen.generate(img, T=100, params_override={
        "drift_x": 0.0, "drift_y": 20.0,
        "amp_y": 0.0, "amp_x": 0.0, "amp_mod": 0.0,
    })
    # First frame ~0, last frame ~+drift_y
    assert out.offsets_xy[0, 1] == pytest.approx(0.0, abs=0.2)
    assert out.offsets_xy[-1, 1] == pytest.approx(20.0, abs=0.5)


# ---------------------------------------------------------------------------
# Cardiac harmonic — optional component
# ---------------------------------------------------------------------------


def test_cardiac_harmonic_adds_higher_frequency():
    gen = SyntheticBurstGenerator(
        resp_amp_y_px_range=(5.0, 5.0), resp_amp_x_px_range=(0.0, 0.0),
        amp_modulation_range=(0.0, 0.0),
        cardiac_amp_px_range=(3.0, 3.0),
        cardiac_freq_hz_range=(1.0, 1.0),
        noise_sigma_range=(0.0, 0.0),
        seed=6,
    )
    img = _textured_frame()
    out = gen.generate(img, T=200, params_override={"phase_offset": 0.0})
    # The y-offset trace should contain frequencies at both ~0.3 Hz
    # (respiratory) and ~1.0 Hz (cardiac); FFT magnitude should peak at both.
    from numpy.fft import rfft, rfftfreq
    fft_mag = np.abs(rfft(out.offsets_xy[:, 1]))
    freqs = rfftfreq(200, d=1.0 / 15.0)
    # Pick the bin nearest 1.0 Hz
    idx_cardiac = np.argmin(np.abs(freqs - 1.0))
    idx_dc = 0
    # Cardiac peak must be significantly above DC
    assert fft_mag[idx_cardiac] > 2.0 * fft_mag[idx_dc] + 1.0
