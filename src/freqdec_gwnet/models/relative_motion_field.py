"""Sequence-relative incremental physiological motion field (R3 main path).

This module implements the protected R3 formulation of FreqDec-GWNet 19.6:

    delta_u_phys(t) = G_delta(s_t, s_t - s_{t-1})
    U_t = cumulative_sum(delta_u_phys, reference=r),  with U_r := 0

Δu_t is parameterized as a 2D global translation in pixel units. The cumulative
sum along time is then a literal element-wise vector summation, which keeps the
R3 wording exact and defensible. Affine and dense-flow variants are deferred so
that ``cumulative_sum`` remains semantically valid (matrix composition is not
addition).

The sign convention is pinned by ``tests/test_motion_convention.py``:

    Δu_t and U_t describe how the physiological background moved in image
    coordinates between the reference frame ``r`` and frame ``t``.
        - To align the reference roadmap with the current frame, warp the
          roadmap by ``+U`` (mode='roadmap_to_current').
        - To pull the current frame back to reference coordinates, warp it by
          ``-U`` (mode='current_to_reference').
"""

from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


_INPUT_MODE_MULTIPLIER = {"delta": 2, "with_prev": 3}


def _input_dim(state_dim: int, input_mode: str) -> int:
    """Return ``len(ψ)`` for the given input mode (paper §3.3 + §8.2.1).

    * ``delta`` (default, 6D when state_dim=3): ``ψ = [s_t, Δs_t]``.
    * ``with_prev`` (9D when state_dim=3, appendix ablation):
      ``ψ = [s_{t−1}, s_t, Δs_t]``.
    """
    if input_mode not in _INPUT_MODE_MULTIPLIER:
        raise ValueError(
            f"unknown input_mode='{input_mode}'; "
            f"expected one of {sorted(_INPUT_MODE_MULTIPLIER)}"
        )
    return _INPUT_MODE_MULTIPLIER[input_mode] * state_dim


class GDeltaMLP(nn.Module):
    """Small MLP implementing ``G_delta(ψ_t) -> Δu_t``.

    Args:
        state_dim: dimensionality of the state slice used for motion. Default
            is 3 = ``(cos_phi, sin_phi, amplitude)``; the confidence channel
            from :class:`GlobalStateBranch` is intentionally excluded.
        hidden_dim: width of internal hidden layers.
        output_dim: motion parameter dimension. Default 2 = global translation.
        input_mode: shape of the input concatenation:
            * ``delta`` (default): ``ψ = [s_t, Δs_t]`` → in_dim = 2*state_dim
              (6D when state_dim=3, 4D when state_dim=2 — appendix #1 reduces
              to 4D by trimming amplitude from the state slice).
            * ``with_prev``: ``ψ = [s_{t-1}, s_t, Δs_t]`` → 3*state_dim
              (9D when state_dim=3 — appendix ablation per paper §8.2.1).

    The final linear layer is initialized to zero so that, prior to training,
    Δu ≡ 0 and the cumulative sum U_t ≡ 0 — preventing wild compensation from
    untrained parameters.
    """

    def __init__(
        self,
        state_dim: int = 3,
        hidden_dim: int = 64,
        output_dim: int = 2,
        input_mode: str = "delta",
    ):
        super().__init__()
        in_dim = _input_dim(state_dim, input_mode)
        self.input_mode = input_mode
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        # 末层零初始化：训练前 Δu ≡ 0，避免未训练时 cumsum 产生大幅伪补偿
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """psi: ``[..., 2*state_dim]`` -> ``[..., output_dim]``."""
        return self.net(psi)


class RelativeMotionField(nn.Module):
    """R3 main-path module: sequence-relative incremental physiological motion.

    Returns a dict per forward call:
        - ``delta_s``: incremental state Δs_t = s_t − s_{t−1}, with Δs_0 := 0.
        - ``delta_u``: incremental motion Δu_t = G_delta([s_t, Δs_t]),
          with Δu_0 := 0 (no previous frame to compare against).
        - ``U``: cumulative displacement; U_r := 0 and
          U_t = Σ_{i=r+1}^{t} Δu_i for t ≥ r. For t < r (warm-up frames before
          the chosen reference) we set U_t := 0 — these frames are treated as
          un-compensable in the main path; reverse cumulation is left to a
          dedicated ablation rather than being baked into the protected core.

    The reference index ``r`` may be a Python ``int`` (shared across the batch)
    or a 1-D ``LongTensor`` of shape ``[B]`` for per-sample selection (matches
    :class:`ReferenceSelector` output planned for the next milestone).
    """

    def __init__(
        self,
        state_dim_for_motion: int = 3,   # s = (cos_phi, sin_phi, amp)
        hidden_dim: int = 64,
        output_dim: int = 2,             # 2D translation in pixels
        input_mode: str = "delta",       # "delta" (6D/4D) | "with_prev" (9D)
    ):
        super().__init__()
        self.state_dim = state_dim_for_motion
        self.output_dim = output_dim
        self.input_mode = input_mode
        self.g_delta = GDeltaMLP(
            state_dim_for_motion, hidden_dim, output_dim,
            input_mode=input_mode,
        )

    @staticmethod
    def compute_increments(s_seq: torch.Tensor) -> torch.Tensor:
        """Compute Δs_t = s_t − s_{t−1} with Δs_0 := 0.

        Args:
            s_seq: ``[B, T, state_dim]``.

        Returns:
            ``[B, T, state_dim]`` with the first time step zeroed.
        """
        # 第一帧没有"上一帧"，Δs_0 强制为 0 与 R3 表述一致
        delta_s = torch.zeros_like(s_seq)
        delta_s[:, 1:] = s_seq[:, 1:] - s_seq[:, :-1]
        return delta_s

    @staticmethod
    def cumulative_displacement(
        delta_u: torch.Tensor,
        reference_index: Union[int, torch.Tensor] = 0,
    ) -> torch.Tensor:
        """Cumulative sum of Δu starting from frame ``r``.

        Args:
            delta_u: ``[B, T, D]`` incremental motion per frame.
            reference_index: ``int`` (shared) or ``LongTensor`` of shape
                ``[B]`` (per-sample).

        Returns:
            ``[B, T, D]`` such that
                * ``U[:, r] == 0``;
                * ``U[:, t] == Σ_{i=r+1}^{t} delta_u[:, i]`` for ``t ≥ r``;
                * ``U[:, t] == 0`` for ``t < r`` (warm-up region).
        """
        B, T, D = delta_u.shape
        # 沿时间累加，再减去参考帧处的累加值，使 U[r] = 0
        cs = torch.cumsum(delta_u, dim=1)              # [B, T, D]

        if isinstance(reference_index, int):
            r = torch.full((B,), reference_index, dtype=torch.long,
                           device=delta_u.device)
        else:
            r = reference_index.to(device=delta_u.device, dtype=torch.long)

        # gather cs[b, r[b]] for each sample
        idx = r.view(B, 1, 1).expand(-1, 1, D)
        cs_at_r = torch.gather(cs, dim=1, index=idx)   # [B, 1, D]
        U_full = cs - cs_at_r                          # [B, T, D]

        # warm-up 帧 (t < r) 不参与补偿，按 R3 文字约束置 0
        time_idx = torch.arange(T, device=delta_u.device).view(1, T, 1)
        warmup_mask = (time_idx >= r.view(B, 1, 1)).to(delta_u.dtype)
        return U_full * warmup_mask

    def forward(
        self,
        s_seq: torch.Tensor,                     # [B, T, state_dim_for_motion]
        reference_index: Union[int, torch.Tensor] = 0,
        detach_state_input: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run G_delta over a window and accumulate displacement.

        Args:
            s_seq: state slice used for motion, ``[B, T, state_dim_for_motion]``.
                Caller is responsible for slicing ``(cos, sin, amp)`` out of
                the 4-D state vector returned by :class:`GlobalStateBranch`.
            reference_index: per-sample reference frame index.
            detach_state_input: if ``True``, stop gradient at G_delta's input.
                Default ``False`` matches the protected main config
                (``detach_state_input=False``).

        Returns:
            ``dict`` with keys ``delta_s``, ``delta_u``, ``U``.
        """
        s_in = s_seq.detach() if detach_state_input else s_seq

        delta_s = self.compute_increments(s_in)             # [B, T, D_s]
        if self.input_mode == "delta":
            psi = torch.cat([s_in, delta_s], dim=-1)         # [B, T, 2*D_s]
        elif self.input_mode == "with_prev":
            # ψ_t = [s_{t-1}, s_t, Δs_t]；s_{-1} := s_0 让 t=0 行 ψ_0 良定义
            prev = torch.zeros_like(s_in)
            prev[:, 1:] = s_in[:, :-1]
            prev[:, 0] = s_in[:, 0]
            psi = torch.cat([prev, s_in, delta_s], dim=-1)   # [B, T, 3*D_s]
        else:
            raise ValueError(
                f"unknown input_mode='{self.input_mode}'"
            )
        delta_u_raw = self.g_delta(psi)                       # [B, T, D_out]

        # 强制 Δu_0 = 0：第一帧没有上一帧，没有合法的"增量"概念
        zero_first = torch.zeros_like(delta_u_raw[:, :1])
        delta_u = torch.cat([zero_first, delta_u_raw[:, 1:]], dim=1)

        U = self.cumulative_displacement(delta_u, reference_index)
        return {"delta_s": delta_s, "delta_u": delta_u, "U": U}


def apply_compensation(
    image: torch.Tensor,
    U: torch.Tensor,
    mode: str = "roadmap_to_current",
    align_corners: bool = False,
) -> torch.Tensor:
    """Translate ``image`` by ±U along the image plane (differentiable).

    Args:
        image: ``[B, C, H, W]`` or ``[B, T, C, H, W]``. If 5-D, ``U`` must be
            ``[B, T, 2]``. If 4-D, ``U`` must be ``[B, 2]``.
        U: translation in pixels, last dim 2 = ``(dx, dy)`` where ``x`` is the
            horizontal axis (column index, positive = right) and ``y`` is the
            vertical axis (row index, positive = down).
        mode:
            ``"roadmap_to_current"`` warps by ``+U`` so that a reference image
            is moved to where the current frame's background is observed.

            ``"current_to_reference"`` warps by ``−U`` so that the current
            frame is brought back to reference coordinates.
        align_corners: forwarded to :func:`torch.nn.functional.grid_sample`.

    Returns:
        Warped image with the same shape as the input.
    """
    if mode not in ("roadmap_to_current", "current_to_reference"):
        raise ValueError(f"unknown mode: {mode}")
    sign = 1.0 if mode == "roadmap_to_current" else -1.0

    # 统一处理 4D / 5D 输入
    squeeze_T = False
    if image.dim() == 4:
        image = image.unsqueeze(1)   # [B, 1, C, H, W]
        U = U.unsqueeze(1)           # [B, 1, 2]
        squeeze_T = True
    B, T, C, H, W = image.shape

    image_flat = image.reshape(B * T, C, H, W)
    U_flat = U.reshape(B * T, 2)

    # 用 affine_grid 取标准 identity base，与 grid_sample 的 align_corners 语义对齐
    theta = torch.eye(2, 3, device=image.device, dtype=image.dtype)
    theta = theta.unsqueeze(0).expand(B * T, -1, -1).contiguous()
    base_grid = F.affine_grid(
        theta, [B * T, C, H, W], align_corners=align_corners
    )  # [B*T, H, W, 2]

    # 像素位移转 normalized 偏移
    if align_corners:
        scale_x = 2.0 / max(W - 1, 1)
        scale_y = 2.0 / max(H - 1, 1)
    else:
        scale_x = 2.0 / W
        scale_y = 2.0 / H
    U_norm = torch.stack(
        [U_flat[..., 0] * scale_x, U_flat[..., 1] * scale_y], dim=-1
    )                                            # [B*T, 2]
    U_norm = U_norm.view(B * T, 1, 1, 2).expand(B * T, H, W, 2)

    # grid_sample: output[h, w] = input[grid[h, w]]
    # 想让图像整体平移 +U（输出在 (h,w) 处的值 = 输入在 (h,w)−U 处的值），
    # 采样位置应是 base − U_norm
    grid = base_grid - sign * U_norm
    warped = F.grid_sample(
        image_flat, grid, mode="bilinear",
        padding_mode="zeros", align_corners=align_corners,
    )
    warped = warped.view(B, T, C, H, W)
    if squeeze_T:
        warped = warped.squeeze(1)
    return warped
