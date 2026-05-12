import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyBranch(nn.Module):
    def __init__(self, channels, h, w, gate_type="none", gate_reduction=4):
        super(FrequencyBranch, self).__init__()
        # 在 PyTorch 中，对于实数输入的 FFT (rfft2)，
        # 频域特征图在宽度维度上会减半，尺寸为 W // 2 + 1
        w_freq = w // 2 + 1

        # 对应路线图 Step 2: 定义一个和频域图一样大的可训练权重参数 Weight_mask
        # 初始化为 1，形状与频域图一致。
        self.weight_mask = nn.Parameter(torch.ones(1, channels, h, w_freq))

        # 自适应频域残差融合：
        # 不是每次都把 iFFT 结果原封不动加回去，而是让网络自己决定
        # 当前通道到底需不需要这份频域增强。
        self.gate_type = gate_type
        if gate_type == "channel":
            hidden = max(channels // gate_reduction, 8)
            self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
        elif gate_type != "none":
            raise ValueError(f"Unsupported gate_type: {gate_type}")

    def forward(self, x):
        # x shape: [Batch, Channels, H, W]

        # FFT 在半精度复数路径上支持不稳定，固定到 float32 更稳，也避免 ComplexHalf 警告。
        orig_dtype = x.dtype
        x_fp32 = x.float()

        # 1. FFT 变换：对 feature map 做 rfft2，得到复数张量。
        fft_x = torch.fft.rfft2(x_fp32)

        # 2. 频域滤波：通过可学习 mask 重标定频率分量强度。
        fft_filtered = fft_x * self.weight_mask.float()

        # 3. iFFT 逆变换：回到空间域。
        ifft_x = torch.fft.irfft2(fft_filtered, s=(x.size(-2), x.size(-1)))

        # 4. 频域残差融合。
        if self.gate_type == "channel":
            gate = self.channel_gate(x_fp32)
            output = x_fp32 + gate * ifft_x
        else:
            output = x_fp32 + ifft_x

        return output.to(dtype=orig_dtype)


class LocalSpatialFrequencyFusion(nn.Module):
    """
    局部空间-频率融合模块。

    设计动机：
    - 当前全局 FFT 分支更像“整张图频率重标定”，对导丝这种细长局部结构不够精细。
    - 这里把 bottleneck 特征切成多个局部窗口，每个窗口单独做 FFT，
      再与一个轻量空间分支融合，尽量同时保留：
      1) 局部频域响应
      2) 空间定位和边界信息
    """

    def __init__(self, channels, window_size=4, gate_reduction=4):
        super(LocalSpatialFrequencyFusion, self).__init__()
        self.window_size = window_size
        w_freq = window_size // 2 + 1

        # 每个局部窗口共享同一个频域 mask，保持参数量可控。
        self.weight_mask = nn.Parameter(torch.ones(1, channels, window_size, w_freq))

        # 轻量空间分支，负责补局部边界和位置细节。
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        hidden = max(channels // gate_reduction, 8)
        # 让网络自己决定每个通道该吸收多少“局部频域增强”。
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def _window_fft(self, x):
        """
        对局部窗口分别做 FFT，再重组回原始空间布局。
        """
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        _, _, hp, wp = x.shape
        nh = hp // ws
        nw = wp // ws

        windows = (
            x.view(b, c, nh, ws, nw, ws)
            .permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .view(-1, c, ws, ws)
        )

        fft_windows = torch.fft.rfft2(windows.float())
        fft_filtered = fft_windows * self.weight_mask.float()
        ifft_windows = torch.fft.irfft2(fft_filtered, s=(ws, ws))

        out = (
            ifft_windows.view(b, nh, nw, c, ws, ws)
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(b, c, hp, wp)
        )

        return out[:, :, :h, :w]

    def forward(self, x):
        orig_dtype = x.dtype
        x_fp32 = x.float()

        spatial_feat = self.spatial_branch(x_fp32)
        local_freq_feat = self._window_fft(x_fp32)
        freq_gate = self.freq_gate(torch.cat([spatial_feat, local_freq_feat], dim=1))
        gated_freq = freq_gate * local_freq_feat

        fused = self.fuse(torch.cat([x_fp32, spatial_feat, gated_freq], dim=1))
        return (x_fp32 + fused).to(dtype=orig_dtype)


class MultiScaleLocalSpatialFrequencyFusion(nn.Module):
    """
    多尺度局部空间-频率融合模块。

    设计动机：
    - 单一窗口大小的局部 FFT 仍然只覆盖一个固定尺度。
    - 导丝既有很细的局部边缘，也有更长的连续走向，因此更合理的做法是并行看多个局部尺度。
    - 这里把多个局部频域分支与空间分支一起融合，让网络自己学习“该在哪个尺度上更相信频域信息”。
    """

    def __init__(self, channels, window_sizes=(4, 8), gate_reduction=4):
        super(MultiScaleLocalSpatialFrequencyFusion, self).__init__()
        if len(window_sizes) == 0:
            raise ValueError("window_sizes 不能为空。")

        self.window_sizes = tuple(window_sizes)
        self.weight_masks = nn.ParameterList(
            [nn.Parameter(torch.ones(1, channels, ws, ws // 2 + 1)) for ws in self.window_sizes]
        )

        self.spatial_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        hidden = max(channels // gate_reduction, 8)
        # 这里输出的是“每个尺度、每个通道”的门控系数，而不是单一一组通道门控。
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * (1 + len(self.window_sizes)), hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels * len(self.window_sizes), kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * (2 + len(self.window_sizes)), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def _window_fft(self, x, window_size: int, weight_mask: nn.Parameter):
        b, c, h, w = x.shape
        ws = window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        _, _, hp, wp = x.shape
        nh = hp // ws
        nw = wp // ws

        windows = (
            x.view(b, c, nh, ws, nw, ws)
            .permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .view(-1, c, ws, ws)
        )

        fft_windows = torch.fft.rfft2(windows.float())
        fft_filtered = fft_windows * weight_mask.float()
        ifft_windows = torch.fft.irfft2(fft_filtered, s=(ws, ws))

        out = (
            ifft_windows.view(b, nh, nw, c, ws, ws)
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(b, c, hp, wp)
        )
        return out[:, :, :h, :w]

    def forward(self, x):
        orig_dtype = x.dtype
        x_fp32 = x.float()

        spatial_feat = self.spatial_branch(x_fp32)
        freq_feats = [
            self._window_fft(x_fp32, ws, mask)
            for ws, mask in zip(self.window_sizes, self.weight_masks)
        ]

        gate = self.freq_gate(torch.cat([spatial_feat] + freq_feats, dim=1))
        gate_chunks = gate.chunk(len(freq_feats), dim=1)
        gated_freq_feats = [g * f for g, f in zip(gate_chunks, freq_feats)]

        fused = self.fuse(torch.cat([x_fp32, spatial_feat] + gated_freq_feats, dim=1))
        return (x_fp32 + fused).to(dtype=orig_dtype)


# ================= 测试代码 =================
if __name__ == "__main__":
    batch_size = 1
    channels = 64
    height, width = 32, 32

    dummy_input = torch.randn(batch_size, channels, height, width)
    freq_branch = FrequencyBranch(channels=channels, h=height, w=width, gate_type="channel")

    output = freq_branch(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
