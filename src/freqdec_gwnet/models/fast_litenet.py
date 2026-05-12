import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入我们亲手搭建的组件
from freqdec_gwnet.models.ghost_encoder import GhostEncoder
from freqdec_gwnet.models.sta_module_v2 import STA_Block_V2


class DecoderBlock(nn.Module):
    """
    轻量级解码器模块：上采样 -> 拼接 (Skip Connection) -> 卷积融合
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # 采用双线性插值进行上采样，比转置卷积更节省参数且不易产生棋盘效应
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            # 输入通道数是 上采样后的通道 + 跳跃连接的通道
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class LargeKernelRefineBlock(nn.Module):
    """
    轻量大核精修块：在最终高分辨率特征上补一层局部+大感受野融合。

    直觉上，它不负责“从零找导丝”，而是负责把已经较好的粗分割进一步提纯：
    - 小核分支保留局部边缘
    - 大核深度卷积分支补充长条结构的上下文
    - 残差连接保证不会轻易破坏原本已经学到的结果
    """

    def __init__(self, channels, kernel_size=7):
        super(LargeKernelRefineBlock, self).__init__()
        padding = kernel_size // 2
        self.local_branch = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.global_branch = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        local_feat = self.local_branch(x)
        global_feat = self.global_branch(x)
        fused = self.fuse(torch.cat([local_feat, global_feat], dim=1))
        return x + fused


class FAST_LiteNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        width_mult=1.0,
        multi_task=False,
        edge_aux=False,
        use_refine_head=False,
        refine_kernel_size=7,
        freq_mode="global",
        local_window_size=4,
        local_window_sizes=(4, 8),
        enable_conf_gate=False,
        gate_fuse_threshold=0.5,
        gate_freeze_threshold=0.45,
        gate_reinit_threshold=0.20,
        bank_momentum=0.7,
        freq_gate_type="none",
        freq_gate_reduction=4,
    ):
        super(FAST_LiteNet, self).__init__()
        self.width_mult = width_mult
        self.multi_task = multi_task
        self.edge_aux = edge_aux
        self.use_refine_head = use_refine_head

        # 1. 编码器 (GhostNet Backbone)
        self.encoder = GhostEncoder(in_channels=in_channels, width_mult=width_mult)
        feat_1_2, feat_1_4, feat_1_8, feat_1_16, feat_1_32 = self.encoder.out_channels

        # 2. 核心模块：STA Module 放在最底层。
        # 空间尺寸仍然固定是 16x16，通道数则跟随 width_mult 扩展。
        self.sta_module = STA_Block_V2(
            channels=feat_1_32,
            h=16,
            w=16,
            freq_mode=freq_mode,
            local_window_size=local_window_size,
            local_window_sizes=local_window_sizes,
            enable_conf_gate=enable_conf_gate,
            fuse_threshold=gate_fuse_threshold,
            freeze_threshold=gate_freeze_threshold,
            reinit_threshold=gate_reinit_threshold,
            bank_momentum=bank_momentum,
            freq_gate_type=freq_gate_type,
            freq_gate_reduction=freq_gate_reduction,
        )

        # 3. 解码器 (逐层恢复分辨率并融合特征)
        # 这里也统一跟随编码器通道扩展，保证 S / M / L 三档结构一致。
        self.dec4 = DecoderBlock(in_channels=feat_1_32, skip_channels=feat_1_16, out_channels=feat_1_16)
        self.dec3 = DecoderBlock(in_channels=feat_1_16, skip_channels=feat_1_8, out_channels=feat_1_8)
        self.dec2 = DecoderBlock(in_channels=feat_1_8, skip_channels=feat_1_4, out_channels=feat_1_4)
        self.dec1 = DecoderBlock(in_channels=feat_1_4, skip_channels=feat_1_2, out_channels=feat_1_2)

        # 4. 最终输出头
        # 先统一上采样，再从同一份高分辨率特征里分出 segmentation / auxiliary heads。
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if self.use_refine_head:
            self.refine_head = LargeKernelRefineBlock(feat_1_2, kernel_size=refine_kernel_size)
        self.seg_head = nn.Conv2d(feat_1_2, num_classes, kernel_size=1)
        if self.multi_task:
            self.centerline_head = nn.Conv2d(feat_1_2, 1, kernel_size=1)
            self.tip_head = nn.Sequential(
                nn.Conv2d(feat_1_2, feat_1_2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feat_1_2),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_1_2, 1, kernel_size=1),
            )
        if self.edge_aux:
            # 这个边界头不改变主干推理流程，只在训练时提供更干净的边缘监督。
            self.edge_head = nn.Sequential(
                nn.Conv2d(feat_1_2, feat_1_2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feat_1_2),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_1_2, 1, kernel_size=1),
            )

    def forward(self, x_curr, feature_bank=None, reset_flag=False):
        """
        x_curr: 当前帧 X光图像 [B, 1, 512, 512]
        feature_bank: 来自上一帧的 Bottleneck 特征 [B, C, 16, 16]
        reset_flag: 序列首帧时清空时序记忆
        """
        # --- Encoder ---
        features = self.encoder(x_curr)
        feat_1_2, feat_1_4, feat_1_8, feat_1_16, feat_1_32 = features

        # --- Bottleneck (Spatio-Temporal Attention) ---
        bottleneck_feat, next_feature_bank = self.sta_module(
            feat_1_32,
            feature_bank=feature_bank,
            reset_flag=reset_flag,
        )

        # --- Decoder ---
        d4 = self.dec4(bottleneck_feat, feat_1_16)
        d3 = self.dec3(d4, feat_1_8)
        d2 = self.dec2(d3, feat_1_4)
        d1 = self.dec1(d2, feat_1_2)

        # --- Final Output ---
        final_feat = self.final_up(d1)
        if self.use_refine_head:
            final_feat = self.refine_head(final_feat)
        out_mask = self.seg_head(final_feat)

        if not self.multi_task and not self.edge_aux:
            # 默认保持旧接口不变，避免训练/评估脚本全都跟着改。
            return out_mask, next_feature_bank

        outputs = {
            "seg": out_mask,
        }
        if self.multi_task:
            outputs["centerline"] = self.centerline_head(final_feat)
            # 当前数据没有真 tip 标注，这里先用 endpoint / tip-proxy 监督。
            outputs["tip"] = self.tip_head(final_feat)
        if self.edge_aux:
            outputs["edge"] = self.edge_head(final_feat)
        return outputs, next_feature_bank


# ================= 测试代码 =================
if __name__ == "__main__":
    batch_size = 1
    h, w = 512, 512

    frame_t_minus_1 = torch.randn(batch_size, 1, h, w)
    frame_t = torch.randn(batch_size, 1, h, w)

    # 这里直接测试 M 档，方便确认 width scaling 没有破坏时序接口。
    model = FAST_LiteNet(in_channels=1, num_classes=1, width_mult=1.5)

    print("--- 模拟视频流前向传播 ---")

    mask_t_minus_1, bottom_feat_t_minus_1 = model(frame_t_minus_1, feature_bank=None, reset_flag=True)
    print(f"Frame t-1 Output Mask: {mask_t_minus_1.shape}")
    print(f"Frame t-1 Bottleneck Feature saved: {bottom_feat_t_minus_1.shape}")

    mask_t, bottom_feat_t = model(frame_t, feature_bank=bottom_feat_t_minus_1, reset_flag=False)
    print(f"Frame t Output Mask: {mask_t.shape}")
    print(f"Frame t Bottleneck Feature saved: {bottom_feat_t.shape}")
