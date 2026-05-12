"""Frame-drop simulation + Δt embedding (paper §6.2 + tech route §3).

Two dropping protocols are supported, both seedable for reproducibility:

    * ``random_drop_rate`` — independent Bernoulli mask per frame (excluding
      the reference). The drop rate is the *expected* fraction; actual
      counts vary slightly. This is what the technical route §3.2 calls
      "random frame dropping".
    * ``downsample_fps``   — keep every k-th frame (deterministic). This
      simulates a lower acquisition rate, e.g. 15 fps → 7.5 fps with k=2.

After dropping, the kept indices are returned along with a ``delta_t`` array
that is exactly what §3.4 asks the state branch to optionally consume:

    delta_t[k] = (kept_indices[k] − kept_indices[k-1]) frames
    delta_t_norm = delta_t / median(delta_t)
    z'_t = concat(z_t, delta_t_norm)            # (handled by callers)

We do *not* impute dropped frames (that would mask the very robustness
question the experiment is designed to ask). Downstream evaluators must
either rerun on the kept-only sub-sequence or use the kept-mask directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class DropPlan:
    """The frames a stress test will keep, plus per-pair Δt metadata.

    Attributes:
        protocol: "random_drop_rate" or "downsample_fps".
        T_full: original full-sequence length.
        kept_indices: ``np.ndarray[int]`` of frames to keep, sorted ascending.
        keep_mask: ``np.ndarray[bool]`` of length ``T_full``, True at kept.
        delta_t_frames: ``[len(kept)]`` per-frame gap in *frame units*. The
            first entry is 0 by convention.
        delta_t_norm: ``[len(kept)]`` ``delta_t_frames / median(>0 entries)``.
            Use this directly as the Δt embedding to concatenate onto
            ``z_t``.
        details: protocol-specific metadata for logs and ablation tables.
    """

    protocol: str
    T_full: int
    kept_indices: np.ndarray
    keep_mask: np.ndarray
    delta_t_frames: np.ndarray
    delta_t_norm: np.ndarray
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def T_kept(self) -> int:
        return int(self.kept_indices.size)

    @property
    def actual_drop_rate(self) -> float:
        return 1.0 - self.T_kept / max(self.T_full, 1)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


def random_drop_rate(
    T: int,
    drop_rate: float,
    *,
    seed: int = 0,
    keep_indices_fixed: Optional[np.ndarray] = None,
) -> DropPlan:
    """Bernoulli per-frame drop with the given expected rate.

    Args:
        T: full-sequence length.
        drop_rate: expected fraction of frames to drop, ``[0, 1)``.
        seed: RNG seed for reproducibility.
        keep_indices_fixed: indices that must always be kept (e.g. the
            reference frame). Defaults to ``[0]`` so downstream cumulative
            sums always start from a real frame.

    Returns:
        :class:`DropPlan`.
    """
    if not 0.0 <= drop_rate < 1.0:
        raise ValueError(f"drop_rate={drop_rate} must be in [0, 1)")
    if T < 2:
        raise ValueError(f"T={T} too small for dropping experiment")

    rng = np.random.default_rng(seed)
    keep_mask = rng.uniform(0, 1, size=T) >= drop_rate

    forced = (
        np.array([0], dtype=int)
        if keep_indices_fixed is None
        else np.asarray(keep_indices_fixed, dtype=int)
    )
    keep_mask[forced] = True

    # 至少要保留 4 帧，否则下游 phase / drift 计算无意义
    if int(keep_mask.sum()) < 4:
        # 强制再保留几帧，从均匀采样里补
        n_extra = 4 - int(keep_mask.sum())
        order = rng.permutation(np.where(~keep_mask)[0])[:n_extra]
        keep_mask[order] = True

    kept = np.where(keep_mask)[0].astype(np.int64)
    dt_frames, dt_norm = _delta_t_from_kept(kept)
    return DropPlan(
        protocol="random_drop_rate",
        T_full=T,
        kept_indices=kept,
        keep_mask=keep_mask,
        delta_t_frames=dt_frames,
        delta_t_norm=dt_norm,
        details={"requested_drop_rate": drop_rate, "seed": seed},
    )


def downsample_fps(
    T: int,
    factor: int,
    *,
    keep_indices_fixed: Optional[np.ndarray] = None,
) -> DropPlan:
    """Deterministic ``factor:1`` downsample (15 fps → 7.5 fps with factor=2).

    Args:
        T: full-sequence length.
        factor: keep every ``factor``-th frame. Must be >= 2.
        keep_indices_fixed: indices forced to be kept (default ``[0]``).
    """
    if factor < 2:
        raise ValueError(f"factor={factor} must be >= 2")
    if T < factor + 1:
        raise ValueError(f"T={T} too short for factor={factor}")

    keep_mask = np.zeros(T, dtype=bool)
    keep_mask[::factor] = True
    forced = (
        np.array([0], dtype=int)
        if keep_indices_fixed is None
        else np.asarray(keep_indices_fixed, dtype=int)
    )
    keep_mask[forced] = True

    kept = np.where(keep_mask)[0].astype(np.int64)
    dt_frames, dt_norm = _delta_t_from_kept(kept)
    return DropPlan(
        protocol="downsample_fps",
        T_full=T,
        kept_indices=kept,
        keep_mask=keep_mask,
        delta_t_frames=dt_frames,
        delta_t_norm=dt_norm,
        details={"factor": int(factor)},
    )


# ---------------------------------------------------------------------------
# Δt computation (technical route §3.4)
# ---------------------------------------------------------------------------


def _delta_t_from_kept(kept: np.ndarray) -> tuple:
    """Per-step frame gap and its median-normalized version."""
    if kept.size < 1:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    dt = np.zeros(kept.size, dtype=np.float32)
    dt[1:] = np.diff(kept).astype(np.float32)
    nonzero = dt[dt > 0]
    median = float(np.median(nonzero)) if nonzero.size else 1.0
    dt_norm = dt / max(median, 1e-6)
    return dt, dt_norm.astype(np.float32)


# ---------------------------------------------------------------------------
# Apply a DropPlan to a sequence tensor
# ---------------------------------------------------------------------------


def apply_drop_plan(
    seq: torch.Tensor, plan: DropPlan, *, dim: int = 0,
) -> torch.Tensor:
    """Index-select the kept frames out of a sequence tensor.

    Args:
        seq: any tensor whose ``dim`` axis indexes frames.
        plan: a :class:`DropPlan` produced by one of the protocols above.
        dim: the temporal axis (default 0).

    Returns:
        A new tensor with ``plan.T_kept`` entries along ``dim``.
    """
    idx = torch.from_numpy(plan.kept_indices).to(seq.device).long()
    return torch.index_select(seq, dim=dim, index=idx)


def remap_reference_index(
    plan: DropPlan, reference_index_full: int,
) -> int:
    """Translate a full-sequence reference index into the kept-only space.

    If the original reference frame was dropped (only happens if the caller
    forgot to add it to ``keep_indices_fixed``), the closest kept frame
    *after* it is chosen — preserving the constraints document's "no fake
    cycle" discipline.
    """
    if plan.keep_mask[reference_index_full]:
        return int(np.searchsorted(plan.kept_indices, reference_index_full))
    after = plan.kept_indices[plan.kept_indices > reference_index_full]
    if after.size:
        return int(np.searchsorted(plan.kept_indices, after[0]))
    return int(plan.T_kept - 1)


# ---------------------------------------------------------------------------
# Δt embedding helper (paper §6.2)
# ---------------------------------------------------------------------------


def concat_delta_t_embedding(
    z_seq: torch.Tensor, plan: DropPlan,
) -> torch.Tensor:
    """Append the median-normalized Δt as an extra feature on each frame.

    Args:
        z_seq: ``[B, T_kept, D]`` per-frame state-branch input.
        plan: drop plan whose ``delta_t_norm`` matches ``T_kept``.

    Returns:
        ``[B, T_kept, D+1]`` with the Δt feature appended last.
    """
    if z_seq.dim() != 3:
        raise ValueError(f"z_seq must be [B, T, D]; got {z_seq.shape}")
    B, T, _ = z_seq.shape
    if T != plan.T_kept:
        raise ValueError(
            f"z_seq T={T} does not match plan.T_kept={plan.T_kept}"
        )
    dt = torch.from_numpy(plan.delta_t_norm).to(
        device=z_seq.device, dtype=z_seq.dtype,
    )
    dt = dt.view(1, T, 1).expand(B, T, 1)
    return torch.cat([z_seq, dt], dim=-1)
