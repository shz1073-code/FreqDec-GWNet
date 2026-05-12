"""B.4 — full-burst orchestrator that produces a per-sequence state label.

Wraps B.1 + B.2 + B.3 into a single :func:`generate_state_label` call and
packs the outputs into a :class:`StateLabel` dataclass that is directly
serializable to ``.npz`` for offline storage.

The output schema is the contract consumed by :class:`StateLoss` during
stage-2 / stage-3 training:

    cos_phi[T]              : float32, phase = cos(φ)
    sin_phi[T]              : float32, phase = sin(φ)
    amplitude[T]             : float32, raw envelope (pixel units)
    amp_scale (scalar)       : float32, per-sequence P95(amplitude) for
                               StateLoss normalization
    valid_mask[T]            : bool, frames eligible for L_state
    anchor_quality[T]        : float32, 1 − circular variance across anchors
    metadata (dict)          : reference_strategy, fs, band_hz, grid, ...

PROJECT_CONSTRAINTS_19.6 §2 also requires the reference strategy to be
reported. The pipeline runs :class:`ReferenceSelector` once per burst and
stores its decision in the metadata block.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ...utils.reference_selector import ReferenceSelector
from .anchor_fusion import multi_region_consensus


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class StateLabel:
    """Per-burst weak supervision label produced by the R2 pipeline."""

    seq_name: str
    dataset: str
    split: str

    cos_phi: np.ndarray            # [T]
    sin_phi: np.ndarray            # [T]
    amplitude: np.ndarray          # [T]
    amp_scale: float               # scalar P95(amplitude)
    valid_mask: np.ndarray         # [T] bool
    anchor_quality: np.ndarray     # [T]

    reference_index: int
    reference_strategy: str        # "clinical" | "warmup_score" | "first_frame"

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- shape sanity ----

    def __post_init__(self) -> None:
        T = self.cos_phi.shape[0]
        for name, arr in (
            ("sin_phi", self.sin_phi),
            ("amplitude", self.amplitude),
            ("valid_mask", self.valid_mask),
            ("anchor_quality", self.anchor_quality),
        ):
            if arr.shape[0] != T:
                raise ValueError(
                    f"StateLabel.{name} length {arr.shape[0]} != T={T}"
                )

    @property
    def num_frames(self) -> int:
        return int(self.cos_phi.shape[0])

    # ---- serialization ----

    def save_npz(self, path: Path) -> Path:
        """Persist to a single ``.npz`` file (numpy arrays + JSON metadata)."""
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            seq_name=np.array(self.seq_name),
            dataset=np.array(self.dataset),
            split=np.array(self.split),
            cos_phi=self.cos_phi,
            sin_phi=self.sin_phi,
            amplitude=self.amplitude,
            amp_scale=np.float32(self.amp_scale),
            valid_mask=self.valid_mask,
            anchor_quality=self.anchor_quality,
            reference_index=np.int64(self.reference_index),
            reference_strategy=np.array(self.reference_strategy),
            metadata_json=np.array(json.dumps(self.metadata)),
        )
        return path

    @classmethod
    def load_npz(cls, path: Path) -> "StateLabel":
        """Load a StateLabel from a ``.npz`` file."""
        import json
        d = np.load(path, allow_pickle=False)
        return cls(
            seq_name=str(d["seq_name"]),
            dataset=str(d["dataset"]),
            split=str(d["split"]),
            cos_phi=d["cos_phi"],
            sin_phi=d["sin_phi"],
            amplitude=d["amplitude"],
            amp_scale=float(d["amp_scale"]),
            valid_mask=d["valid_mask"].astype(bool),
            anchor_quality=d["anchor_quality"],
            reference_index=int(d["reference_index"]),
            reference_strategy=str(d["reference_strategy"]),
            metadata=json.loads(str(d["metadata_json"])),
        )


# ---------------------------------------------------------------------------
# Reference frame selection adapter
# ---------------------------------------------------------------------------


def _select_reference(
    images: np.ndarray,
    state_seq: np.ndarray,
    quality: np.ndarray,
    warmup_min: int,
    warmup_max: int,
) -> Tuple[int, str]:
    """Pick the reference frame using ReferenceSelector with the just-computed
    state sequence as the ``state_change`` signal.

    Until anchor_quality is normalized into [0,1] it can be passed straight
    in. The selector internally min-max normalizes per-window so absolute
    scale does not matter.
    """
    selector = ReferenceSelector(
        warmup_min=warmup_min, warmup_max=warmup_max,
    )
    # ReferenceSelector takes torch.Tensor inputs; convert.
    images_t = torch.from_numpy(images.astype(np.float32))
    if images_t.max() > 1.5:                       # heuristic for uint8 input
        images_t = images_t / 255.0
    state_t = torch.from_numpy(state_seq.astype(np.float32))
    quality_t = torch.from_numpy(quality.astype(np.float32))
    sel = selector.select(
        images=images_t,
        state_seq=state_t,
        anchor_quality=quality_t,
    )
    return int(sel.reference_index), sel.strategy


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def generate_state_label(
    images: np.ndarray,                          # [T, H, W]
    wire_masks: Optional[np.ndarray] = None,     # [T, H, W] or None
    *,
    seq_name: str,
    dataset: str,
    split: str,
    fs: float = 15.0,
    band_hz: Tuple[float, float] = (0.15, 0.50),
    bp_order: int = 4,
    grid: Tuple[int, int] = (2, 2),
    grid_overlap: float = 0.10,
    quality_threshold: float = 0.50,
    min_valid_anchors: int = 3,
    warmup_min: int = 5,
    warmup_max: int = 10,
    amp_scale_percentile: float = 95.0,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> StateLabel:
    """Full R2 pipeline: images (+ wire masks) → :class:`StateLabel`.

    All keyword arguments are passed through to the underlying components
    so that one call captures the full configuration. Defaults match the
    decisions documented in ``PROJECT_CONSTRAINTS_19.6.md``.
    """
    # ---- 1. Multi-anchor B.1 + B.2 + B.3 ----
    consensus = multi_region_consensus(
        images=images,
        wire_masks=wire_masks,
        fs=fs,
        band_hz=band_hz,
        bp_order=bp_order,
        grid=grid,
        grid_overlap=grid_overlap,
        quality_threshold=quality_threshold,
        min_valid_anchors=min_valid_anchors,
    )

    cos_phi = consensus["cos_phi"]
    sin_phi = consensus["sin_phi"]
    amplitude = consensus["amplitude"]
    quality = consensus["anchor_quality"]
    valid_mask = consensus["valid_mask"]

    # ---- 2. amp_scale: per-sequence P95 for StateLoss normalization ----
    # 只用 valid 帧上的振幅，避免边界 artifact 把 P95 抬上去
    valid_amps = amplitude[valid_mask] if valid_mask.any() else amplitude
    amp_scale = float(np.percentile(valid_amps, amp_scale_percentile))
    amp_scale = max(amp_scale, 1e-3)               # 防止下游除零

    # ---- 3. Reference frame selection ----
    state_seq = np.stack([cos_phi, sin_phi, amplitude], axis=-1)  # [T, 3]
    ref_index, ref_strategy = _select_reference(
        images=images,
        state_seq=state_seq,
        quality=quality,
        warmup_min=warmup_min,
        warmup_max=warmup_max,
    )

    # ---- 4. Bundle metadata ----
    metadata: Dict[str, Any] = {
        "fs": float(fs),
        "band_hz": [float(band_hz[0]), float(band_hz[1])],
        "bp_order": int(bp_order),
        "grid": [int(grid[0]), int(grid[1])],
        "grid_overlap": float(grid_overlap),
        "quality_threshold": float(quality_threshold),
        "min_valid_anchors": int(min_valid_anchors),
        "amp_scale_percentile": float(amp_scale_percentile),
        "n_anchors": len(consensus["per_anchor"]),
        "n_valid_frames": int(valid_mask.sum()),
        "wire_masks_provided": wire_masks is not None,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return StateLabel(
        seq_name=seq_name,
        dataset=dataset,
        split=split,
        cos_phi=cos_phi,
        sin_phi=sin_phi,
        amplitude=amplitude,
        amp_scale=amp_scale,
        valid_mask=valid_mask,
        anchor_quality=quality,
        reference_index=ref_index,
        reference_strategy=ref_strategy,
        metadata=metadata,
    )
