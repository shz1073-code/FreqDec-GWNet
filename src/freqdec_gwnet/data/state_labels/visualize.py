"""Diagnostic visualization for a generated :class:`StateLabel`.

Produces a single PNG with five panels designed to make it obvious whether
the R2 pipeline extracted a clean physiological signal:

    1. raw 2-D background trajectory (after PCA projection, pre-bandpass)
    2. band-pass filtered signal + Hilbert envelope
    3. cos_phi / sin_phi overlaid (sanity: should trace a unit circle in time)
    4. amplitude + valid_mask (invalid frames shaded red)
    5. anchor_quality (1 − circular variance across anchors)

Used by ``scripts/generate_state_labels.py`` and easy to call ad-hoc when
debugging a specific sequence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .anchor_fusion import multi_region_consensus
from .background_motion import (
    cumulative_position,
    estimate_background_translation,
)
from .band_filter import (
    band_pass_filter,
    project_to_principal_axis,
)
from .pipeline import StateLabel


def render_state_label_png(
    label: StateLabel,
    images: np.ndarray,
    path: Path,
    *,
    fs: Optional[float] = None,
) -> Path:
    """Render a 5-panel diagnostic figure for ``label``.

    Args:
        label: a :class:`StateLabel` produced by :func:`generate_state_label`.
        images: ``[T, H, W]`` original frames (used to recompute the raw
            projection for panel 1 — kept out of the StateLabel itself to
            avoid bloating the .npz with redundant data).
        path: output PNG path.
        fs: sampling rate in Hz; falls back to ``label.metadata['fs']``.

    Returns:
        The output path (so callers can chain assertions).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if fs is None:
        fs = float(label.metadata.get("fs", 15.0))

    T = label.num_frames
    t_axis = np.arange(T) / fs

    # ---- Recompute the global (single-anchor) raw projection for panel 1 ----
    # 这里只用全图（不分 grid）做一次估计，方便看到信号原始形状
    deltas, _ = estimate_background_translation(images, wire_masks=None)
    pos = cumulative_position(deltas)
    raw_proj, axis, ratio = project_to_principal_axis(pos)

    # 对应的 band-pass 后信号（用于 panel 2）
    band = label.metadata.get("band_hz", [0.15, 0.50])
    bp_order = int(label.metadata.get("bp_order", 4))
    filtered = band_pass_filter(
        raw_proj, fs=fs, low_hz=band[0], high_hz=band[1], order=bp_order,
    )
    envelope = label.amplitude

    # ---- Build figure ----
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    title = (
        f"{label.dataset}/{label.split}/{label.seq_name}    "
        f"T={T}    fs={fs:.1f}Hz    "
        f"ref={label.reference_index} ({label.reference_strategy})    "
        f"amp_scale={label.amp_scale:.2f}    "
        f"valid={int(label.valid_mask.sum())}/{T}"
    )
    fig.suptitle(title, fontsize=11)

    # Panel 1: raw projection (pre-filter)
    ax = axes[0]
    ax.plot(t_axis, raw_proj, color="C7", lw=0.8, label="raw projection")
    ax.set_ylabel("raw position (px)")
    ax.legend(loc="upper right", fontsize=8)
    ax.text(
        0.01, 0.97,
        f"PCA axis = ({axis[0]:+.2f}, {axis[1]:+.2f})  ratio={ratio:.2f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox=dict(facecolor="white", edgecolor="0.7"),
    )
    ax.grid(alpha=0.3)

    # Panel 2: filtered + envelope
    ax = axes[1]
    ax.plot(t_axis, filtered, color="C0", lw=1.0, label="band-pass")
    ax.plot(t_axis, envelope, color="C3", lw=1.0, label="Hilbert |·|")
    ax.plot(t_axis, -envelope, color="C3", lw=1.0)
    ax.fill_between(t_axis, -envelope, envelope, color="C3", alpha=0.07)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_ylabel("filtered (px)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: cos/sin
    ax = axes[2]
    ax.plot(t_axis, label.cos_phi, color="C2", lw=0.8, label="cos φ")
    ax.plot(t_axis, label.sin_phi, color="C4", lw=0.8, label="sin φ")
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_ylabel("phase components")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 4: amplitude + invalid shading
    ax = axes[3]
    ax.plot(t_axis, envelope, color="C3", lw=1.0, label="amplitude")
    invalid = ~label.valid_mask
    if invalid.any():
        # 把所有 invalid 帧用红色阴影标出
        ymax = max(float(envelope.max()) * 1.05, 0.5)
        ax.fill_between(
            t_axis, 0, ymax, where=invalid,
            color="red", alpha=0.10, step="mid",
            label="invalid",
        )
        ax.set_ylim(0, ymax)
    ax.set_ylabel("amplitude (px)")
    ax.axhline(label.amp_scale, color="C1", lw=0.6, ls="--",
               label=f"amp_scale={label.amp_scale:.2f}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 5: anchor quality
    ax = axes[4]
    ax.plot(t_axis, label.anchor_quality, color="C5", lw=1.0,
            label="anchor_quality (1 − circ.var)")
    th = label.metadata.get("quality_threshold", 0.50)
    ax.axhline(th, color="C8", lw=0.8, ls="--",
               label=f"threshold={th:.2f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("quality")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
