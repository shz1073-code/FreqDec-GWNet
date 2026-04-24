import torch
import torch.nn as nn
from freqdec_gwnet.models.ghost_blocks import GhostBottleneck


def make_divisible(value, divisor=8, min_value=None):
    """
    将通道数调整到更适合 GPU / Tensor Core 的整数。
    这里统一按 8 对齐，后面做 width scaling 时更稳。
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class GhostEncoder(nn.Module):
    def __init__(self, in_channels=1, width_mult=1.0):
        super(GhostEncoder, self).__init__()
        self.width_mult = width_mult

        # 按比例统一扩宽每一级通道，为后续的 S / M / L 版本提供入口。
        c2 = make_divisible(16 * width_mult)
        c4 = make_divisible(24 * width_mult)
        c8 = make_divisible(40 * width_mult)
        c16 = make_divisible(80 * width_mult)
        c32 = make_divisible(160 * width_mult)

        h1 = make_divisible(16 * width_mult)
        h2a = make_divisible(72 * width_mult)
        h2b = make_divisible(120 * width_mult)
        h3a = make_divisible(240 * width_mult)
        h3b = make_divisible(200 * width_mult)
        h4a = make_divisible(480 * width_mult)
        h4b = make_divisible(960 * width_mult)

        # 记录每一级输出通道，解码器和 STA 模块会直接复用这组尺寸。
        self.out_channels = [c2, c4, c8, c16, c32]

        # 初始卷积层 (下采样 1/2) -> feat_1/2 (256x256)
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )

        # Stage 1 (下采样 1/4) -> feat_1/4 (128x128)
        self.stage1 = GhostBottleneck(inp=c2, hidden_dim=h1, oup=c4, kernel_size=3, stride=2, use_se=False)

        # Stage 2 (下采样 1/8) -> feat_1/8 (64x64)
        self.stage2 = nn.Sequential(
            GhostBottleneck(inp=c4, hidden_dim=h2a, oup=c8, kernel_size=3, stride=2, use_se=False),
            GhostBottleneck(inp=c8, hidden_dim=h2b, oup=c8, kernel_size=3, stride=1, use_se=False)
        )

        # Stage 3 (下采样 1/16) -> feat_1/16 (32x32)
        self.stage3 = nn.Sequential(
            GhostBottleneck(inp=c8, hidden_dim=h3a, oup=c16, kernel_size=3, stride=2, use_se=False),
            GhostBottleneck(inp=c16, hidden_dim=h3b, oup=c16, kernel_size=3, stride=1, use_se=False)
        )

        # Stage 4 (下采样 1/32) -> feat_1/32 (16x16)
        self.stage4 = nn.Sequential(
            GhostBottleneck(inp=c16, hidden_dim=h4a, oup=c32, kernel_size=3, stride=2, use_se=False),
            GhostBottleneck(inp=c32, hidden_dim=h4b, oup=c32, kernel_size=3, stride=1, use_se=False)
        )

    def forward(self, x):
        # 严格按照技术路线要求的输出列表进行特征收集
        feat_1_2 = self.init_conv(x)
        feat_1_4 = self.stage1(feat_1_2)
        feat_1_8 = self.stage2(feat_1_4)
        feat_1_16 = self.stage3(feat_1_8)
        feat_1_32 = self.stage4(feat_1_16)

        # 返回列表，用于后续的 Skip Connection 和 STA 模块输入
        return [feat_1_2, feat_1_4, feat_1_8, feat_1_16, feat_1_32]


# ================= 测试代码 =================
if __name__ == "__main__":
    # 模拟单通道 X 光图像输入: [Batch, Channel, H, W] = [1, 1, 512, 512]
    dummy_xray = torch.randn(1, 1, 512, 512)

    encoder = GhostEncoder(in_channels=1, width_mult=1.0)
    features = encoder(dummy_xray)

    print("GhostNet Encoder Output Feature Shapes:")
    for i, feat in enumerate(features):
        print(f"feat_1/{2**(i+1)} shape: {feat.shape}")
