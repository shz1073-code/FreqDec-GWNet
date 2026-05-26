"""
phys_dec_gwnet/losses/state_losses.py

生理状态损失函数。

包含：
1. circular_phase_loss：圆周相位损失（不用角度 MSE 的原因见注释）
2. amplitude_loss：归一化振幅损失
3. make_burn_in_mask：burn-in 权重掩码（解决冷启动噪声梯度）
4. StateLoss：完整状态损失模块（组合上述三个）
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def circular_phase_loss(
    cos_pred: torch.Tensor,    # [...] 预测余弦
    sin_pred: torch.Tensor,    # [...] 预测正弦
    phase_cos_gt: torch.Tensor,   # [...] GT 余弦
    phase_sin_gt: torch.Tensor,   # [...] GT 正弦
    weight_mask: Optional[torch.Tensor] = None,  # [...] 权重（burn-in + valid）
) -> torch.Tensor:
    """
    圆周相位损失：L = 1 - cos(φ_pred - φ_gt) = 1 - (e_pred · e_gt)

    为什么不用 MSE(φ_pred, φ_gt)：
        呼吸相位是圆周变量，φ=0 和 φ=2π 是同一个状态。
        如果 φ_gt = 0.1 rad，φ_pred = 6.18 rad（接近 2π ≈ 6.28），
        两者物理上几乎相同，但 MSE = (6.18-0.1)² ≈ 37，产生巨大错误梯度。
        本损失通过余弦相似度避免 wrap 问题，值域 [0, 2]，0 表示完全对齐。

    e_pred 在计算前用 F.normalize 归一化到单位圆，
    避免网络通过缩放 cos/sin 幅度来"欺骗"损失。
    """
    # 归一化预测向量到单位圆（数值稳定）
    e_pred = F.normalize(
        torch.stack([cos_pred, sin_pred], dim=-1), dim=-1, eps=1e-8
    )  # [..., 2]

    e_gt = torch.stack([phase_cos_gt, phase_sin_gt], dim=-1)  # [..., 2]

    # 点积 = cos(φ_pred - φ_gt)，范围 [-1, 1]
    dot = (e_pred * e_gt).sum(dim=-1)  # [...]

    # 损失 = 1 - 余弦相似度，范围 [0, 2]
    loss_per_element = 1.0 - dot  # [...]

    # 加权平均
    if weight_mask is not None:
        # 避免除零：如果所有权重都是 0（不应该发生，但防御性编程）
        denom = weight_mask.sum().clamp(min=1e-6)
        return (loss_per_element * weight_mask).sum() / denom
    else:
        return loss_per_element.mean()


def amplitude_loss(
    amp_pred: torch.Tensor,   # [...] 预测振幅（raw，未归一化）
    amp_gt: torch.Tensor,     # [...] GT 振幅（raw，像素单位）
    amp_scale: torch.Tensor,  # [B] or broadcast-able，per-sequence 归一化尺度
    weight_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    归一化振幅损失。

    为什么必须归一化：
        呼吸振幅是像素位移量，通常 5–30 px。
        圆周损失值域是 [0, 2]。
        如果直接把 SmoothL1(amp_pred, amp_gt) 加入总损失，
        振幅项量级（~5-30）会完全压制相位项量级（~0-2），
        导致 L_state 实际上只优化振幅，相位学不动。

        归一化到 [0, ~1] 后两项量级一致。

    amp_scale 推荐：per-sequence P95(|u_filtered|)，
    保存在 state label 的 .npz 文件中，每次从文件读取。
    """
    # amp_scale broadcast：[B] → [...] 形状
    # 调用前需保证 amp_scale 的形状可 broadcast 到 amp_pred
    amp_pred_norm = amp_pred / (amp_scale + 1e-6)
    amp_gt_norm = amp_gt / (amp_scale + 1e-6)

    loss_per_element = F.smooth_l1_loss(
        amp_pred_norm, amp_gt_norm, reduction='none', beta=0.1
    )  # [...]

    if weight_mask is not None:
        denom = weight_mask.sum().clamp(min=1e-6)
        return (loss_per_element * weight_mask).sum() / denom
    else:
        return loss_per_element.mean()


def make_burn_in_mask(
    T: int,
    burn_in_ratio: float = 0.2,
    burn_in_min_frames: int = 3,
    burn_in_weight: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    生成 burn-in 权重掩码：[T]

    为什么需要 burn-in：
        GRU/Mamba 从零隐状态 (h_0=0) 开始，前几帧处于"冷启动"状态，
        相位预测天然不准。如果这些帧参与全权 L_state，
        会引入大量噪声梯度，且会使 Mamba 看起来比 GRU 更差
        （但实际上只是初始化策略的问题，而非 Mamba 本身的劣势）。

        burn-in 不把这些帧的权重设为 0（完全忽略），
        而是设为小值 0.1（保留弱监督），
        避免网络完全不学习如何从零状态启动。

    Args:
        T: 窗口长度
        burn_in_ratio: burn-in 帧占窗口比例（如 0.2 = 前 20%）
        burn_in_min_frames: 最少 burn-in 帧数（短窗口保护）
        burn_in_weight: burn-in 帧的损失权重（0=完全忽略，0.1=弱监督）

    Returns:
        mask: [T]，前 B 帧为 burn_in_weight，其余为 1.0
    """
    B = max(burn_in_min_frames, int(T * burn_in_ratio))
    mask = torch.ones(T, device=device)
    mask[:B] = burn_in_weight
    return mask


class StateLoss(nn.Module):
    """
    完整的生理状态损失模块。

    组合：
    - 圆周相位损失 L_phase（主项）
    - 归一化振幅损失 L_amp（辅助项）
    - burn-in 掩码（过滤冷启动帧）
    - valid_mask（过滤低质量 state label 序列）

    用法：
        loss_fn = StateLoss(lambda_amp=0.5)
        result = loss_fn(
            state_pred,    # [B, T, 4]
            phase_cos_gt,  # [B, T]
            phase_sin_gt,  # [B, T]
            amp_gt,        # [B, T]
            amp_scale,     # [B]
            valid_mask,    # [B, T]
        )
        loss = result["loss_state"]
    """

    def __init__(
        self,
        lambda_amp: float = 0.5,        # 振幅损失权重
        burn_in_ratio: float = 0.2,     # burn-in 比例
        burn_in_min_frames: int = 3,    # 最少 burn-in 帧数
        burn_in_weight: float = 0.1,    # burn-in 帧权重
        lambda_cardiac: float = 0.0,    # 心跳分支总权重 (0=不启用)
        lambda_cardiac_amp: float = 0.5,  # 心跳分支内部振幅子权重
    ):
        super().__init__()
        self.lambda_amp = lambda_amp
        self.burn_in_ratio = burn_in_ratio
        self.burn_in_min_frames = burn_in_min_frames
        self.burn_in_weight = burn_in_weight
        self.lambda_cardiac = float(lambda_cardiac)
        self.lambda_cardiac_amp = float(lambda_cardiac_amp)

    def forward(
        self,
        state_pred: torch.Tensor,       # [B, T, 4] resp-only 或 [B, T, 8] resp+card
        phase_cos_gt: torch.Tensor,     # [B, T]
        phase_sin_gt: torch.Tensor,     # [B, T]
        amp_gt: torch.Tensor,           # [B, T]
        amp_scale: torch.Tensor,        # [B] per-sequence 振幅归一化尺度
        valid_mask: torch.Tensor,       # [B, T] 0=低质量label，1=正常
        # Dual-cycle (cardiac) GT —— 启用前提: state_pred 是 8 维 且 lambda_cardiac > 0
        cardiac_cos_gt: torch.Tensor = None,
        cardiac_sin_gt: torch.Tensor = None,
        cardiac_amp_gt: torch.Tensor = None,
        cardiac_amp_scale: torch.Tensor = None,
        cardiac_valid_mask: torch.Tensor = None,
    ) -> dict:
        """
        返回字典：
        {
            "loss_state":   总损失（用于 backward）
            "loss_phase":   相位损失（detach，用于日志记录）
            "loss_amp":     振幅损失（detach，用于日志记录）
            "n_valid_frames": 参与计算的有效帧数（用于调试）
        }
        """
        B, T = phase_cos_gt.shape
        device = state_pred.device

        # 1. 生成 burn-in 掩码 [T]
        burn_in = make_burn_in_mask(
            T, self.burn_in_ratio, self.burn_in_min_frames,
            self.burn_in_weight, device
        )

        # 2. 合并 burn-in 和 valid_mask
        # burn_in: [T] → [1, T]，valid_mask: [B, T]
        effective_weight = valid_mask * burn_in.unsqueeze(0)  # [B, T]

        # 3. amp_scale broadcast: [B] → [B, T]
        amp_scale_bt = amp_scale.unsqueeze(1).expand(B, T)

        # 4. 计算各项损失
        loss_phase = circular_phase_loss(
            state_pred[:, :, 0],    # cos_pred [B, T]
            state_pred[:, :, 1],    # sin_pred [B, T]
            phase_cos_gt,
            phase_sin_gt,
            effective_weight
        )

        loss_amp = amplitude_loss(
            state_pred[:, :, 2],    # amp_pred [B, T]
            amp_gt,
            amp_scale_bt,
            effective_weight
        )

        # 5. 总损失（呼吸部分）
        loss_resp = loss_phase + self.lambda_amp * loss_amp

        # 6. 可选：心跳分支
        loss_card_phase = state_pred.new_zeros(())
        loss_card_amp = state_pred.new_zeros(())
        if (self.lambda_cardiac > 0.0
                and state_pred.shape[-1] >= 8
                and cardiac_cos_gt is not None):
            card_valid = (cardiac_valid_mask if cardiac_valid_mask is not None
                          else valid_mask)
            card_weight = card_valid * burn_in.unsqueeze(0)
            card_scale = (cardiac_amp_scale if cardiac_amp_scale is not None
                          else amp_scale)
            card_scale_bt = card_scale.unsqueeze(1).expand(B, T)
            loss_card_phase = circular_phase_loss(
                state_pred[:, :, 4], state_pred[:, :, 5],
                cardiac_cos_gt, cardiac_sin_gt, card_weight,
            )
            loss_card_amp = amplitude_loss(
                state_pred[:, :, 6],
                cardiac_amp_gt,
                card_scale_bt,
                card_weight,
            )

        loss_card = loss_card_phase + self.lambda_cardiac_amp * loss_card_amp
        loss_total = loss_resp + self.lambda_cardiac * loss_card

        return {
            "loss_state": loss_total,
            "loss_phase": loss_phase.detach(),
            "loss_amp": loss_amp.detach(),
            "loss_card_phase": loss_card_phase.detach(),
            "loss_card_amp": loss_card_amp.detach(),
            "n_valid_frames": effective_weight.sum().detach(),
        }


# ===========================================================================
# 单元测试
# ===========================================================================

def _run_unit_tests():
    print("=" * 60)
    print("StateLoss 单元测试")
    print("=" * 60)

    B, T = 3, 20  # 3个样本，20帧
    device = torch.device("cpu")

    # ---------- 测试 1：圆周损失 wrap 问题 ----------
    print("\n[测试 1] 圆周损失：φ_pred ≈ 2π ≈ φ_gt = 0 时损失应该接近 0")
    phi_gt = torch.tensor([0.05])   # GT 相位：接近 0
    phi_pred_good = torch.tensor([0.10])   # 预测：接近 GT
    phi_pred_bad_mse = torch.tensor([6.20])  # 预测：接近 2π（物理上≈GT，但 MSE 会很大）

    cos_gt = torch.cos(phi_gt)
    sin_gt = torch.sin(phi_gt)
    cos_good = torch.cos(phi_pred_good)
    sin_good = torch.sin(phi_pred_good)
    cos_bad = torch.cos(phi_pred_bad_mse)
    sin_bad = torch.sin(phi_pred_bad_mse)

    loss_good = circular_phase_loss(cos_good, sin_good, cos_gt, sin_gt)
    loss_bad = circular_phase_loss(cos_bad, sin_bad, cos_gt, sin_gt)
    mse_good = F.mse_loss(phi_pred_good, phi_gt)
    mse_bad = F.mse_loss(phi_pred_bad_mse, phi_gt)

    print(f"  圆周损失（好预测）:    {loss_good.item():.4f}  ← 应该小")
    print(f"  圆周损失（wrap 预测）: {loss_bad.item():.4f}   ← 应该也小（物理上近似）")
    print(f"  MSE（好预测）:         {mse_good.item():.4f}  ← 小")
    print(f"  MSE（wrap 预测）:      {mse_bad.item():.4f}  ← 很大！这就是问题所在")

    if loss_bad < 0.1 and mse_bad > 10:
        print("  ✓ PASS：圆周损失正确处理 wrap，MSE 不能处理 wrap")
    else:
        print("  ✗ 结果有些异常，请检查")

    # ---------- 测试 2：burn-in mask 形状和权重 ----------
    print("\n[测试 2] Burn-in mask 正确性")
    T_test = 20
    mask = make_burn_in_mask(T_test, burn_in_ratio=0.2, burn_in_min_frames=3,
                              burn_in_weight=0.1)
    B_expected = max(3, int(T_test * 0.2))  # = max(3, 4) = 4
    burn_in_correct = (mask[:B_expected] == 0.1).all() and (mask[B_expected:] == 1.0).all()
    print(f"  期望 burn-in 帧数: {B_expected}，mask[:4]={mask[:4].tolist()}, mask[4:6]={mask[4:6].tolist()}")
    print(f"  ✓ PASS" if burn_in_correct else "  ✗ FAIL")

    # ---------- 测试 3：振幅归一化消除量纲差异 ----------
    print("\n[测试 3] 振幅归一化：raw amp（像素）vs 归一化 amp 量级比较")
    amp_gt_pixels = torch.tensor([15.0, 20.0, 12.0])   # 像素量级
    amp_pred_pixels = torch.tensor([14.5, 21.0, 11.5])
    amp_scale_val = torch.tensor([18.0, 18.0, 18.0])   # P95 尺度

    loss_raw = F.smooth_l1_loss(amp_pred_pixels, amp_gt_pixels)
    loss_norm = amplitude_loss(amp_pred_pixels, amp_gt_pixels, amp_scale_val)
    print(f"  raw SmoothL1: {loss_raw.item():.4f}  ← 量级约 0.1-1（像素）")
    print(f"  归一化后:     {loss_norm.item():.4f}  ← 量级约 0.001-0.1（无量纲）")
    print(f"  比值: {loss_raw.item() / loss_norm.item():.1f}x  ← 归一化前大得多")

    # ---------- 测试 4：StateLoss 完整前向 ----------
    print("\n[测试 4] StateLoss 完整前向")
    loss_fn = StateLoss(lambda_amp=0.5)
    state_pred = torch.randn(B, T, 4)
    phase_cos_gt = torch.cos(torch.randn(B, T))
    phase_sin_gt = torch.sin(torch.randn(B, T))
    amp_gt = torch.abs(torch.randn(B, T)) * 15.0
    amp_scale = torch.tensor([18.0] * B)
    valid_mask = torch.ones(B, T)
    valid_mask[0, :] = 0.0  # 第一个序列质量差，全部降权

    result = loss_fn(state_pred, phase_cos_gt, phase_sin_gt,
                     amp_gt, amp_scale, valid_mask)

    print(f"  loss_state:   {result['loss_state'].item():.4f}")
    print(f"  loss_phase:   {result['loss_phase'].item():.4f}")
    print(f"  loss_amp:     {result['loss_amp'].item():.4f}")
    print(f"  n_valid_frames: {result['n_valid_frames'].item():.0f}")
    print(f"  ✓ PASS：前向通过，loss 值合理" if result['loss_state'].item() > 0 else "  ✗ FAIL")

    # ---------- 测试 5：梯度反传 ----------
    print("\n[测试 5] StateLoss 可反传梯度")
    state_pred_req = torch.randn(B, T, 4, requires_grad=True)
    result2 = loss_fn(state_pred_req, phase_cos_gt, phase_sin_gt,
                      amp_gt, amp_scale, valid_mask)
    result2["loss_state"].backward()
    if state_pred_req.grad is not None:
        print(f"  ✓ PASS：梯度成功反传，grad 范数 = {state_pred_req.grad.norm().item():.4f}")
    else:
        print("  ✗ FAIL：梯度未反传")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    _run_unit_tests()
