import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # 处理奇数尺寸导致的对齐问题
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetBaseline(nn.Module):
    """
    标准 2D U-Net baseline (无时序记忆)。
    为了和现有时序训练/评估脚本兼容，forward 返回 (logits, None)。
    """

    def __init__(self, in_channels=1, num_classes=1, base_channels=32):
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2

        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)

        self.up1 = Up(c5, c4, c4)
        self.up2 = Up(c4, c3, c3)
        self.up3 = Up(c3, c2, c2)
        self.up4 = Up(c2, c1, c1)
        self.outc = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x_curr=None, feature_bank=None, reset_flag=False, **kwargs):
        _ = feature_bank
        _ = reset_flag
        if x_curr is None:
            # 兼容少数调用方可能传入的 x=... 写法。
            x_curr = kwargs.pop("x", None)
        if x_curr is None:
            raise TypeError("UNetBaseline.forward() requires an input image tensor named x_curr (or x).")

        x1 = self.inc(x_curr)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits, None


if __name__ == "__main__":
    model = UNetBaseline(in_channels=1, num_classes=1, base_channels=32)
    x = torch.randn(2, 1, 512, 512)
    y, memory = model(x, feature_bank=None, reset_flag=True)
    print("Output:", y.shape)
    print("Feature bank:", memory)
