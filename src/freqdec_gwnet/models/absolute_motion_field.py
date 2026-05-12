"""Absolute motion field — main-paper §6.1 ablation #1 ONLY.

This module is the explicitly forbidden formulation that PROJECT_CONSTRAINTS_19.6
§1 R3 calls out:

    u_phys(t) = G_motion(s_t)        # forbidden as the main method

It is implemented here on purpose, in a *separate file*, so it can serve
as the head-to-head comparison in main ablation table 1 (paper §8.1.1):

    "absolute displacement vs sequence-relative increment"

Design discipline:
    * AbsoluteMotionField is **only** consumable as ``motion_field_type='absolute'``
      in :class:`FreqDecGWNet` — the default remains ``'relative'``.
    * It exposes the *same* output dict as :class:`RelativeMotionField`
      (``delta_s``, ``delta_u``, ``U``) so MotionLoss / drift evaluation
      can be A/B-compared without rewriting downstream code.
    * It does NOT cumsum: ``U_t = u_t − u_r`` (anchor-relative, not
      time-cumulative). This is the structural distinction the ablation
      is designed to expose.
"""

from __future__ import annotations

from typing import Dict, Union

import torch
import torch.nn as nn


class GMotionMLP(nn.Module):
    """Small MLP implementing the forbidden ``G_motion(s_t) -> u_t``.

    Mirrors :class:`GDeltaMLP` in width / depth so the ablation comparison
    is parameter-matched. Last layer is zero-init for the same reason —
    untrained model produces ``U ≡ 0``.

    Args:
        state_dim: width of the state slice consumed (default 3 = cos/sin/amp).
        hidden_dim: hidden layer width.
        output_dim: motion parameter dimension (default 2 = global translation).
    """

    def __init__(
        self,
        state_dim: int = 3,
        hidden_dim: int = 64,
        output_dim: int = 2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """``s: [..., state_dim]`` -> ``[..., output_dim]``."""
        return self.net(s)


class AbsoluteMotionField(nn.Module):
    """Baseline R3-ablation: ``u_t = G_motion(s_t)`` with anchor-relative U.

    The output dict mirrors :class:`RelativeMotionField` so MotionLoss and
    drift reporting work unchanged. The structural difference is:

        ``U_t := u_t − u_r``

    (no cumulative sum). For ``t < r`` we set ``U_t := 0`` to match the
    R3-side convention that warm-up frames are excluded from compensation —
    this keeps the ablation comparison apples-to-apples.

    The ``delta_u`` returned is the implicit per-frame increment of this U
    (i.e. ``delta_u[t] = U[t] − U[t−1]``) so :class:`MotionLoss` can warp
    consecutive frames consistently in either ablation arm.

    Args:
        state_dim_for_motion: state slice width consumed (default 3).
        hidden_dim / output_dim: parameter-matched to RelativeMotionField.
    """

    def __init__(
        self,
        state_dim_for_motion: int = 3,
        hidden_dim: int = 64,
        output_dim: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim_for_motion
        self.output_dim = output_dim
        self.g_motion = GMotionMLP(state_dim_for_motion, hidden_dim, output_dim)

    def forward(
        self,
        s_seq: torch.Tensor,                            # [B, T, state_dim]
        reference_index: Union[int, torch.Tensor] = 0,
        detach_state_input: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Apply ``G_motion`` per-frame and rebase to the reference.

        Args:
            s_seq: ``[B, T, state_dim]`` state sequence (cos / sin / amp).
            reference_index: ``int`` (shared) or ``LongTensor[B]`` (per-sample).
            detach_state_input: detach gradient at G_motion's input. Default
                False to match the §5 main-config behavior.

        Returns:
            ``dict`` with keys
                * ``delta_s``: ``[B, T, state_dim]`` Δs_t = s_t − s_{t−1}
                  (with Δs_0 := 0). Returned for API parity; the absolute
                  formulation does not actually consume Δs.
                * ``delta_u``: ``[B, T, output_dim]`` implicit per-frame
                  motion increment ``U_t − U_{t−1}`` (with Δu_0 := 0).
                * ``U``: ``[B, T, output_dim]`` ``u_t − u_r`` masked to
                  zero for warm-up frames ``t < r``.
        """
        if s_seq.dim() != 3:
            raise ValueError(
                f"s_seq must be [B, T, D]; got shape {tuple(s_seq.shape)}"
            )

        B, T, _ = s_seq.shape
        s_in = s_seq.detach() if detach_state_input else s_seq

        # 1) per-frame absolute u_t = G_motion(s_t)
        u_seq = self.g_motion(s_in)                     # [B, T, output_dim]

        # 2) anchor-relative U_t = u_t − u_r (NOT cumsum — that's the point)
        if isinstance(reference_index, int):
            r = torch.full((B,), reference_index, dtype=torch.long,
                           device=u_seq.device)
        else:
            r = reference_index.to(device=u_seq.device, dtype=torch.long)
        idx = r.view(B, 1, 1).expand(-1, 1, self.output_dim)
        u_at_r = torch.gather(u_seq, dim=1, index=idx)  # [B, 1, output_dim]
        U_full = u_seq - u_at_r                         # [B, T, output_dim]

        # warm-up frames (t < r) are excluded from compensation, same as R3
        time_idx = torch.arange(T, device=u_seq.device).view(1, T, 1)
        warmup_mask = (time_idx >= r.view(B, 1, 1)).to(u_seq.dtype)
        U = U_full * warmup_mask                        # [B, T, output_dim]

        # 3) implicit per-frame increment for L_motion compatibility
        delta_u = torch.zeros_like(U)
        delta_u[:, 1:] = U[:, 1:] - U[:, :-1]            # Δu_0 := 0

        # 4) Δs returned for API parity; this baseline doesn't consume it
        delta_s = torch.zeros_like(s_seq)
        delta_s[:, 1:] = s_seq[:, 1:] - s_seq[:, :-1]

        return {"delta_s": delta_s, "delta_u": delta_u, "U": U}
