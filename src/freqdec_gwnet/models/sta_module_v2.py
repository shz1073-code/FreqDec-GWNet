import torch
import torch.nn as nn
import torch.nn.functional as F

# 频域分支（继承自 19.x 第一阶段；保持完整路径，避免相对导入歧义）
from freqdec_gwnet.models.frequency import (
    FrequencyBranch,
    LocalSpatialFrequencyFusion,
    MultiScaleLocalSpatialFrequencyFusion,
)


class TemporalBranchV2(nn.Module):
    """
    升级版的时序分支，专门处理与 Feature Bank 的 Cross-Attention
    """

    def __init__(self, channels):
        super(TemporalBranchV2, self).__init__()
        self.q_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.k_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_curr, prev_feat, fuse_threshold=0.5):
        B, C, H, W = x_curr.shape

        # 用 bottleneck 特征相似度做一个轻量置信估计。
        curr_flat = x_curr.view(B, -1)
        prev_flat = prev_feat.view(B, -1)
        cos_sim = F.cosine_similarity(curr_flat, prev_flat, dim=1).view(B, 1, 1, 1)
        sim_mask = (cos_sim > fuse_threshold).float()

        # 轻量级 Channel Cross-Attention
        q = self.pool(x_curr).squeeze(-1).squeeze(-1).unsqueeze(1)
        k = self.pool(prev_feat).squeeze(-1).squeeze(-1).unsqueeze(1)

        q = self.q_conv(q)
        k = self.k_conv(k)

        attn_weight = F.softmax(torch.bmm(q.permute(0, 2, 1), k), dim=-1)
        v = prev_feat.view(B, C, H * W)

        out_temp = torch.bmm(attn_weight, v).view(B, C, H, W)
        return out_temp * sim_mask, cos_sim


class STA_Block_V2(nn.Module):
    def __init__(
        self,
        channels,
        h,
        w,
        freq_mode="global",
        local_window_size=4,
        local_window_sizes=(4, 8),
        enable_conf_gate=False,
        fuse_threshold=0.5,
        freeze_threshold=0.45,
        reinit_threshold=0.20,
        bank_momentum=0.7,
        freq_gate_type="none",
        freq_gate_reduction=4,
    ):
        super(STA_Block_V2, self).__init__()
        self.freq_mode = freq_mode
        # 默认保留原来的全局 FFT 分支；只有显式切到 local_sff 时，才启用局部空间-频率融合。
        if freq_mode == "global":
            self.frequency_branch = FrequencyBranch(
                channels,
                h,
                w,
                gate_type=freq_gate_type,
                gate_reduction=freq_gate_reduction,
            )
        elif freq_mode == "local_sff":
            self.frequency_branch = LocalSpatialFrequencyFusion(
                channels,
                window_size=local_window_size,
                gate_reduction=freq_gate_reduction,
            )
        elif freq_mode == "ms_local_sff":
            self.frequency_branch = MultiScaleLocalSpatialFrequencyFusion(
                channels,
                window_sizes=local_window_sizes,
                gate_reduction=freq_gate_reduction,
            )
        else:
            raise ValueError(f"Unsupported freq_mode: {freq_mode}")
        self.temporal_branch = TemporalBranchV2(channels)
        self.enable_conf_gate = enable_conf_gate
        self.fuse_threshold = fuse_threshold
        self.freeze_threshold = freeze_threshold
        self.reinit_threshold = reinit_threshold
        self.bank_momentum = bank_momentum

    def forward(self, x_curr, feature_bank=None, reset_flag=False):
        """
        前向传播
        x_curr: 当前帧特征
        feature_bank: 保存的 T-1 时刻的 Bottleneck 特征
        reset_flag: 模拟医生松开踏板，强制清空时序记忆
        """
        # 1. 频域增强 (无论何时都执行，保证基础抗噪能力)
        x_freq = self.frequency_branch(x_curr)

        # 2. 临床中断重置逻辑 (极其重要！)
        if reset_flag:
            feature_bank = None

        # 3. 时序融合
        if feature_bank is not None:
            x_temp, cos_sim = self.temporal_branch(
                x_curr,
                feature_bank,
                fuse_threshold=self.fuse_threshold,
            )
            out = x_freq + x_temp

            if self.enable_conf_gate:
                # 高相似度：用 EMA 更新 feature bank
                # 中等相似度：冻结 feature bank，避免一点点偏差逐帧累积
                # 低相似度：直接用当前频域特征重建 bank，相当于 re-init
                update_mask = (cos_sim > self.freeze_threshold).float()
                reinit_mask = (cos_sim < self.reinit_threshold).float()
                ema_bank = self.bank_momentum * feature_bank + (1.0 - self.bank_momentum) * out
                frozen_bank = feature_bank
                reinit_bank = x_freq
                next_feature_bank = update_mask * ema_bank + (1.0 - update_mask) * frozen_bank
                next_feature_bank = reinit_mask * reinit_bank + (1.0 - reinit_mask) * next_feature_bank
            else:
                next_feature_bank = out
        else:
            # 如果是序列首帧，或者医生重新踩下踏板 (reset_flag=True)
            # 网络退化为单帧推理，仅依赖频域抗噪
            out = x_freq
            next_feature_bank = out

        return out, next_feature_bank


# ================= 测试代码 =================
if __name__ == "__main__":
    channels, h, w = 160, 16, 16
    sta_v2 = STA_Block_V2(channels=channels, h=h, w=w, enable_conf_gate=True)

    frame_1 = torch.randn(1, channels, h, w)
    frame_2 = torch.randn(1, channels, h, w)

    print("--- 场景 A: 医生持续踩住踏板 (连续透视) ---")
    out_1, feature_bank = sta_v2(frame_1, feature_bank=None, reset_flag=False)
    print("Frame 1 (首帧): 无 Feature Bank 参与融合。")

    out_2, feature_bank = sta_v2(frame_2, feature_bank=feature_bank, reset_flag=False)
    print("Frame 2 (连续帧): Feature Bank 成功参与融合。")

    print("\n--- 场景 B: 医生松开踏板后再次踩下 (触发 Reset) ---")
    out_3, feature_bank = sta_v2(frame_2, feature_bank=feature_bank, reset_flag=True)
    print("Frame 3 (重置帧): reset_flag=True, 网络主动丢弃旧记忆，退化为单帧推理。")
