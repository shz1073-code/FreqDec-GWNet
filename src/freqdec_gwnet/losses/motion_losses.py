"""Motion consistency loss with active/background decoupling (R4 protected).

L_motion warps the previous frame's feature forward by the predicted Δu_t and
compares it against the current frame's feature, but only on background pixels
— i.e., outside a dilated guidewire mask. This implements the R4 red lines:

    * motion loss only on background;
    * wire dilated exclusion mask;
    * compensation only of the physiological component.

Detach policy follows §5 of PROJECT_CONSTRAINTS_19.6.md:

    detach_prev_feature = True   (default)
    detach_curr_feature = True   (default)
    detach_state_input  = False  (handled inside RelativeMotionField)

so that L_motion optimizes the state-conditioned motion mapping (Δu via
G_delta and the state branch) but does not flow gradient back into the
encoder/decoder feature representations — those are governed by the
segmentation losses.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.relative_motion_field import apply_compensation


def dilate_binary(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Morphological dilation via ``max_pool2d``.

    Args:
        mask: ``[B, 1, H, W]`` (binary or in [0, 1]).
        k: half-size of the dilation kernel; effective kernel is ``2k+1``.

    Returns:
        Dilated mask with the same shape.
    """
    pad = k
    kernel = 2 * k + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=pad)


class MotionLoss(nn.Module):
    """Photometric residual on background pixels after Δu warp.

    For each consecutive pair ``(t-1, t)``, the previous-frame feature is
    warped by ``+Δu_t`` (i.e. moved forward toward the current frame's
    coordinate system) and compared against the current-frame feature. The
    squared residual is averaged across channels and then reduced over
    background pixels only — the dilated current-frame guidewire region is
    excluded so that L_motion never tries to align active wire motion.

    Args:
        detach_prev_feature: detach the previous-frame feature before warp.
        detach_curr_feature: detach the current-frame feature before residual.
        wire_dilate_k: half-size of the wire-mask dilation kernel.
        eps: numerical floor when dividing by the background pixel count.
    """

    def __init__(
        self,
        detach_prev_feature: bool = True,
        detach_curr_feature: bool = True,
        wire_dilate_k: int = 3,
        eps: float = 1e-6,
        lambda_zero_mean: float = 0.0,
        lambda_drift_cap: float = 0.0,
        drift_cap_ratio: float = 2.0,
    ):
        """Args:
            detach_prev_feature / detach_curr_feature: §5 detach policy.
            wire_dilate_k: dilation kernel half-size for the exclusion mask.
            eps: numerical floor.
            lambda_zero_mean: weight on the zero-mean Δu regularizer
                ``‖mean_t(Δu_t)‖²``. Defaults to 0 to preserve historical
                behavior; recommend ~1.0 to suppress monotonic drift bias
                (see paper §7 rationale: respiratory/cardiac motion is a
                bounded periodic signal so over a long-enough window mean
                Δu ≈ 0). Without this, G_delta can shortcut to outputting
                a near-constant value that minimises pairwise photometric
                residual while letting cumsum drift unbounded.
            lambda_drift_cap: weight on the soft cap ``max(0, ‖U‖_max −
                drift_cap_ratio × amp_scale_proxy)²`` where the proxy is
                the per-batch median of ``‖U‖``. Defaults 0. Use with
                care — too aggressive a cap can collapse G_delta to 0.
            drift_cap_ratio: multiplier for the soft cap threshold.
        """
        super().__init__()
        self.detach_prev = detach_prev_feature
        self.detach_curr = detach_curr_feature
        self.wire_dilate_k = wire_dilate_k
        self.eps = eps
        self.lambda_zero_mean = float(lambda_zero_mean)
        self.lambda_drift_cap = float(lambda_drift_cap)
        self.drift_cap_ratio = float(drift_cap_ratio)

    def forward(
        self,
        feat_seq: torch.Tensor,                        # [B, T, C, H, W]
        delta_u: torch.Tensor,                         # [B, T, 2] pixel units
        wire_mask_seq: torch.Tensor,                   # [B, T, 1, H, W] in {0,1}
        valid_pair_mask: Optional[torch.Tensor] = None,  # [B, T-1] frame-pair valid
        U: Optional[torch.Tensor] = None,              # [B, T, 2] cumulative — for drift cap
    ) -> Dict[str, torch.Tensor]:
        """Compute the masked motion residual.

        Args:
            feat_seq: feature maps over a window.
            delta_u: incremental motion predicted by RelativeMotionField. Only
                ``delta_u[:, 1:]`` is used (one Δu per consecutive pair).
            wire_mask_seq: binary guidewire masks per frame (1 inside wire).
            valid_pair_mask: optional per-pair validity (e.g. drop frame-pair
                spanning a sequence boundary).

        Returns:
            ``dict(loss_motion, n_valid)``. ``n_valid`` reports the effective
            number of background pixel-pairs that contributed to the loss.
        """
        B, T, C, H, W = feat_seq.shape
        if T < 2:
            zero = feat_seq.new_zeros(())
            return {"loss_motion": zero, "n_valid": zero}

        prev = feat_seq[:, :-1]                         # [B, T-1, C, H, W]
        curr = feat_seq[:, 1:]                          # [B, T-1, C, H, W]
        # 红线：detach 默认开启，L_motion 不优化 encoder 特征
        if self.detach_prev:
            prev = prev.detach()
        if self.detach_curr:
            curr = curr.detach()

        delta_u_pair = delta_u[:, 1:]                   # [B, T-1, 2]
        # 把 prev 按 +Δu 推到 curr 的坐标系（对齐到当前帧背景）
        prev_warped = apply_compensation(
            prev.contiguous(), delta_u_pair,
            mode="roadmap_to_current",
        )

        # 背景掩码 = 1 - 膨胀(curr 帧的 wire mask)
        wire_curr = wire_mask_seq[:, 1:].reshape(B * (T - 1), 1, H, W)
        wire_dilated = dilate_binary(wire_curr.float(), self.wire_dilate_k)
        bg_mask = 1.0 - wire_dilated.clamp(0.0, 1.0)
        bg_mask = bg_mask.view(B, T - 1, 1, H, W)

        residual = (prev_warped - curr).pow(2)          # [B, T-1, C, H, W]
        # 跨通道求均值，再用 bg_mask 加权
        per_pixel = residual.mean(dim=2, keepdim=True)  # [B, T-1, 1, H, W]
        weighted = per_pixel * bg_mask

        if valid_pair_mask is not None:
            vmask = valid_pair_mask.view(B, T - 1, 1, 1, 1).to(weighted.dtype)
            weighted = weighted * vmask
            denom = (bg_mask * vmask).sum().clamp(min=self.eps)
        else:
            denom = bg_mask.sum().clamp(min=self.eps)

        loss_photo = weighted.sum() / denom

        # ---- zero-mean regularizer (paper §7) -----------------------------
        # 长时序累计 Δu 的均值应 ≈ 0 (呼吸是周期信号)。
        # 这一项直接堵死"G_delta 输出常数 bias"的捷径。
        loss_zero_mean = delta_u.new_zeros(())
        if self.lambda_zero_mean > 0 and T >= 2:
            # 只对 t>=1 求均值（t=0 强制为 0 不参与）
            mean_du = delta_u[:, 1:].mean(dim=1)            # [B, D_out]
            loss_zero_mean = mean_du.pow(2).sum(dim=-1).mean()

        # ---- drift cap regularizer ---------------------------------------
        loss_drift_cap = delta_u.new_zeros(())
        if self.lambda_drift_cap > 0 and U is not None:
            U_mag = U.norm(dim=-1)                          # [B, T]
            U_max = U_mag.max(dim=1).values                 # [B]
            U_median = U_mag.median(dim=1).values.clamp(min=1.0)
            cap = self.drift_cap_ratio * U_median
            excess = (U_max - cap).clamp(min=0)
            loss_drift_cap = excess.pow(2).mean()

        total = (
            loss_photo
            + self.lambda_zero_mean * loss_zero_mean
            + self.lambda_drift_cap * loss_drift_cap
        )
        return {
            "loss_motion": total,
            "loss_photo": loss_photo.detach(),
            "loss_zero_mean": loss_zero_mean.detach(),
            "loss_drift_cap": loss_drift_cap.detach(),
            "n_valid": denom.detach(),
        }
