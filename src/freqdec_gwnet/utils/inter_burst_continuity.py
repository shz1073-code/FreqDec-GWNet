"""Inter-burst continuity test — does the state core remember across gaps?

Real fluoroscopy clinical practice produces **multiple bursts per surgery**
separated by 5-30 second pauses (when the operator releases the pedal).
Patient physiology (respiration / heartbeat) continues across those gaps.
A state core that can carry its hidden state with a Δt embedding should
predict the phase at the start of burst N+1 better than a fresh-h₀
baseline; this is the canonical scenario where Mamba's continuous-time
SSM interpretation pays off — but it's only visible if the experiment is
**designed** to exercise it.

This module builds a *sandbox* using a single long burst from the public
clean_data: chop the burst into K chunks with a synthetic gap between
each, then compare four inference protocols:

    1. naive    — each chunk runs with h₀ = 0
    2. carry    — h_final from chunk N becomes h₀ of chunk N+1
    3. carry+dt — carry h₀ + concatenate a Δt embedding onto z_t
    4. oracle   — process the *unchunked* burst as a single forward pass

The discontinuity metric is the phase jump at chunk boundaries:

    delta_cos_at_boundary = | cos_phi[T_chunk_end] − cos_phi[T_chunk_start+1] |

In the oracle pass this is naturally tiny; the question is how close
naive / carry / carry+dt come to oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


@dataclass
class BurstChunkPlan:
    """How a single long burst is split into pseudo-multi-burst segments.

    Attributes:
        T_full: original burst length.
        chunk_lengths: ``[K]`` frames per chunk after gap dropping.
        chunk_starts: ``[K]`` start index of each chunk in the original burst.
        gap_frames: number of original frames removed between chunks
            (uniform). Use 0 for "no gap"; typical clinical gap ≈ 5-30 s
            which at 15 fps = 75-450 frames.
        kept_indices: ``[sum(chunk_lengths)]`` flat indices into the
            original burst. Adjacent kept frames within a chunk are
            consecutive; the gap drops appear as jumps at chunk
            boundaries.
        delta_t_frames: ``[sum(chunk_lengths)]`` per-step gap in frame
            units. Inside a chunk = 1; at chunk boundaries = gap_frames + 1.
    """
    T_full: int
    chunk_lengths: List[int]
    chunk_starts: List[int]
    gap_frames: int
    kept_indices: np.ndarray
    delta_t_frames: np.ndarray

    @property
    def K(self) -> int:
        return len(self.chunk_lengths)

    @property
    def boundary_kept_indices(self) -> List[Tuple[int, int]]:
        """``(end_of_chunk_N, start_of_chunk_N+1)`` in *kept* coordinate
        — useful for evaluating phase discontinuity per boundary.
        """
        out: List[Tuple[int, int]] = []
        cursor = 0
        for L in self.chunk_lengths[:-1]:
            cursor += L
            out.append((cursor - 1, cursor))
        return out


def build_chunk_plan(
    T_full: int,
    *,
    K: int = 3,
    gap_frames: int = 15,
    chunk_length: Optional[int] = None,
) -> BurstChunkPlan:
    """Split ``T_full`` frames into ``K`` chunks with synthetic gaps.

    Args:
        T_full: original burst length (must be ≥ ``K*(chunk_length+gap_frames)``).
        K: number of chunks. Default 3.
        gap_frames: how many original frames to drop between chunks.
            15 frames @ 15 fps = 1 s pause.
        chunk_length: per-chunk frame count. Defaults to fitting all
            ``K`` chunks + ``K-1`` gaps within ``T_full``.

    Returns:
        :class:`BurstChunkPlan`.
    """
    if T_full < 2 * K:
        raise ValueError(f"T_full={T_full} too short for K={K} chunks")
    if gap_frames < 0:
        raise ValueError(f"gap_frames={gap_frames} must be ≥ 0")

    if chunk_length is None:
        chunk_length = (T_full - (K - 1) * gap_frames) // K
    if chunk_length < 4:
        raise ValueError(
            f"chunk_length={chunk_length} too small after carving out "
            f"K-1={K-1} gaps of {gap_frames} frames"
        )

    chunk_starts: List[int] = []
    kept: List[int] = []
    delta_t: List[float] = []
    pos = 0
    for k in range(K):
        if pos + chunk_length > T_full:
            break
        chunk_starts.append(pos)
        for t in range(chunk_length):
            kept.append(pos + t)
            if not delta_t:
                delta_t.append(0.0)              # 第一帧约定 dt=0
            elif t == 0:
                delta_t.append(float(gap_frames + 1))   # 跨 chunk 跳变
            else:
                delta_t.append(1.0)
        pos += chunk_length + gap_frames

    return BurstChunkPlan(
        T_full=T_full,
        chunk_lengths=[chunk_length] * len(chunk_starts),
        chunk_starts=chunk_starts,
        gap_frames=gap_frames,
        kept_indices=np.asarray(kept, dtype=np.int64),
        delta_t_frames=np.asarray(delta_t, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@dataclass
class ContinuityResult:
    """Per-protocol output of the inter-burst test.

    Attributes:
        protocol: ``naive`` / ``carry`` / ``carry_dt`` / ``oracle``.
        cos_phi: ``[T_kept]`` predicted cos(φ) at kept frames.
        sin_phi: ``[T_kept]`` predicted sin(φ).
        boundary_phase_jumps: per-boundary ``|cos_phi_end − cos_phi_start|``,
            length ``K-1``.
        mean_boundary_jump: scalar — primary metric (lower = smoother).
    """
    protocol: str
    cos_phi: np.ndarray
    sin_phi: np.ndarray
    boundary_phase_jumps: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_boundary_jump: float = float("nan")


def _phase_jumps_at_boundaries(
    cos_phi: np.ndarray, plan: BurstChunkPlan,
) -> np.ndarray:
    out: List[float] = []
    for (end_idx, start_idx) in plan.boundary_kept_indices:
        if end_idx < cos_phi.shape[0] and start_idx < cos_phi.shape[0]:
            out.append(float(abs(cos_phi[end_idx] - cos_phi[start_idx])))
    return np.asarray(out, dtype=np.float32)


def _forward_window_with_optional_dt(
    branch,
    feat_18_seq: torch.Tensor,           # [B=1, T, C, H, W]
    h0: Optional[torch.Tensor] = None,
    delta_t: Optional[torch.Tensor] = None,   # [T] (normalized)
) -> torch.Tensor:
    """Forward through GlobalStateBranch with optional Δt fusion at z_t.

    Δt fusion strategy: after ``_extract_z`` builds ``z_seq`` (which has
    width ``proj_dim``), we modulate it by a learned gating that depends
    on Δt — but training never saw such a gate. To keep the test honest
    we instead **scale z_t by 1/sqrt(1 + dt_norm)** so that frames after
    long gaps decay slightly. This is a known approximation; the proper
    way (concat Δt as extra channel + re-train) is documented but not
    needed to demonstrate the qualitative effect.

    Returns:
        ``[1, T, 4]`` state output.
    """
    feat_lp = branch.low_pass(
        feat_18_seq.reshape(-1, *feat_18_seq.shape[2:])
    )
    feat_proj = branch.channel_proj(feat_lp)
    z_flat = branch.gap(feat_proj).squeeze(-1).squeeze(-1)  # [B*T, proj_dim]
    B, T = feat_18_seq.shape[:2]
    z_seq = z_flat.reshape(B, T, -1)
    if delta_t is not None:
        # 简单 decay：dt 大 → 信号衰减，让模型不要过度信任跨 gap 的特征
        scale = (1.0 / (1.0 + delta_t.to(z_seq.dtype).to(z_seq.device))).view(1, T, 1)
        z_seq = z_seq * scale
    if branch.state_core_type == "gru":
        h_seq = branch.state_core.scan(z_seq, h0)
    else:
        h_seq = branch.state_core.scan(z_seq)        # Mamba: no explicit h0
    return branch.output_head(h_seq)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_inter_burst_continuity(
    model,
    images_full: torch.Tensor,               # [T_full, 1, H, W]
    plan: BurstChunkPlan,
    *,
    device: str = "cuda",
) -> Dict[str, ContinuityResult]:
    """Run all four protocols on the same model + plan.

    Args:
        model: a :class:`FreqDecGWNet` with state_branch trained (stage2/3).
        images_full: full original burst.
        plan: chunking plan produced by :func:`build_chunk_plan`.
        device: where to run.

    Returns:
        Dict ``{protocol: ContinuityResult}``.
    """
    model.eval()
    images_full = images_full.to(device)

    # ---- 1) Encoder over the full burst once (reuse features) ----
    with torch.no_grad():
        flat = images_full.unsqueeze(0)              # [1, T_full, 1, H, W]
        B, T_full, C_in, H, W = flat.shape
        encoded = model.r1.encoder(
            flat.reshape(B * T_full, C_in, H, W)
        )
        feat_18 = encoded[2]                         # 1/8 res
        feat_18_seq = feat_18.reshape(
            B, T_full, *feat_18.shape[1:],
        )
        feat_18_kept = feat_18_seq[
            :, torch.from_numpy(plan.kept_indices).to(device).long()
        ]

    out: Dict[str, ContinuityResult] = {}

    # ---- 2) oracle: process unchunked full burst ----
    with torch.no_grad():
        state_full = model.state_branch.forward_window(feat_18_seq)[0]
        oracle_kept = state_full[
            torch.from_numpy(plan.kept_indices).to(device).long()
        ].cpu().numpy()
    out["oracle"] = ContinuityResult(
        protocol="oracle",
        cos_phi=oracle_kept[..., 0],
        sin_phi=oracle_kept[..., 1],
    )

    # ---- 3) naive: each chunk fresh h₀=0 ----
    cos_list: List[np.ndarray] = []
    sin_list: List[np.ndarray] = []
    cursor = 0
    with torch.no_grad():
        for L in plan.chunk_lengths:
            sub = feat_18_kept[:, cursor:cursor + L]
            s = model.state_branch.forward_window(sub)[0].cpu().numpy()
            cos_list.append(s[..., 0])
            sin_list.append(s[..., 1])
            cursor += L
    naive_cos = np.concatenate(cos_list, axis=0)
    naive_sin = np.concatenate(sin_list, axis=0)
    out["naive"] = ContinuityResult(
        protocol="naive",
        cos_phi=naive_cos, sin_phi=naive_sin,
    )

    # ---- 4) carry: final h of chunk N → h₀ of chunk N+1 ----
    cos_list, sin_list = [], []
    cursor = 0
    h_prev: Optional[torch.Tensor] = None
    with torch.no_grad():
        for L in plan.chunk_lengths:
            sub = feat_18_kept[:, cursor:cursor + L]
            # GRU 支持 h0; Mamba 当前 scan 不直接吃 h0，但实战中通过
            # _use_mamba=True 时 forward_window 仍可 carry hidden state
            # 的近似——这里我们对两种 core 走相同路径
            s_full = model.state_branch.forward_window(sub, h0=h_prev)[0]
            s_np = s_full.cpu().numpy()
            cos_list.append(s_np[..., 0])
            sin_list.append(s_np[..., 1])
            # 把最后一帧的隐状态拿出来（用 forward_step）
            last_feat = sub[:, -1]
            _, h_prev, _ = model.state_branch.forward_step(
                last_feat, h_prev=h_prev,
            )
            h_prev = h_prev.detach()
            cursor += L
    out["carry"] = ContinuityResult(
        protocol="carry",
        cos_phi=np.concatenate(cos_list),
        sin_phi=np.concatenate(sin_list),
    )

    # ---- 5) carry+dt: carry + Δt scaling on z_t ----
    cos_list, sin_list = [], []
    cursor = 0
    h_prev = None
    dt_kept = torch.from_numpy(plan.delta_t_frames).to(device)
    # 归一化 Δt：内部帧 dt=1 → norm=1；跨 gap dt=gap+1 → norm 较大
    dt_norm = dt_kept / max(float(dt_kept[dt_kept > 0].median().item()), 1.0)
    with torch.no_grad():
        for L in plan.chunk_lengths:
            sub = feat_18_kept[:, cursor:cursor + L]
            sub_dt = dt_norm[cursor:cursor + L]
            s_full = _forward_window_with_optional_dt(
                model.state_branch, sub, h0=h_prev, delta_t=sub_dt,
            )[0]
            s_np = s_full.cpu().numpy()
            cos_list.append(s_np[..., 0])
            sin_list.append(s_np[..., 1])
            last_feat = sub[:, -1]
            _, h_prev, _ = model.state_branch.forward_step(
                last_feat, h_prev=h_prev,
            )
            h_prev = h_prev.detach()
            cursor += L
    out["carry_dt"] = ContinuityResult(
        protocol="carry_dt",
        cos_phi=np.concatenate(cos_list),
        sin_phi=np.concatenate(sin_list),
    )

    # ---- 6) Boundary discontinuity scores ----
    for proto, res in out.items():
        jumps = _phase_jumps_at_boundaries(res.cos_phi, plan)
        res.boundary_phase_jumps = jumps
        res.mean_boundary_jump = float(jumps.mean()) if jumps.size else float("nan")

    return out
