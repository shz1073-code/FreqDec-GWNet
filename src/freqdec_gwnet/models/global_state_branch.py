"""
phys_dec_gwnet/models/global_state_branch.py

全局生理状态分支。

功能：
- 从 1/8 分辨率特征中提取低频全局生理状态
- 用 GRU 或 Mamba 建模跨帧呼吸/心动节律
- 输出 [cos(φ), sin(φ), amplitude, confidence]，供运动场模块使用

关键设计决策：
1. detach_encoder=True（默认）：阻止 L_state 梯度反向影响 encoder 中的
   导丝高频边缘特征。encoder 只被 L_seg 系列损失优化。
2. SpatialLowPass：在 GAP 之前压制导丝局部高频响应，
   让 GAP 后的向量主要反映背景低频结构。
3. 训练用 forward_window（整窗口），推理用 forward_step（单帧递推）。

作者约定：本文件不依赖 fast-litenet 的任何代码，只依赖标准 PyTorch。
"""

import warnings
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 子模块 1：空间低通滤波器
# ===========================================================================

class SpatialLowPass(nn.Module):
    """
    空间低通滤波器（可学习 depthwise blur conv）。

    为什么需要它：
        导丝是细长结构，在特征图上产生局部高频激活。
        如果直接对 1/8 特征做 GAP，导丝的局部激活会混入全局向量，
        干扰"背景低频结构 → 生理状态"的映射。
        低通滤波先把这些局部高频响应模糊掉，再做 GAP，
        使全局向量更干净地反映背景（横膈膜、肋骨等）的缓慢变化。

    实现：
        Depthwise conv（每通道独立模糊，不混通道）+ Pointwise proj（混通道）
        初始化为均值核，允许被小幅度学习（但不应该学出高通滤波器）。
    """

    def __init__(self, channels: int, kernel_size: int = 5,
                 learnable: bool = True):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 必须是奇数"
        padding = kernel_size // 2

        # Depthwise blur conv：groups=channels，每通道独立做空间模糊
        self.blur = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,   # ← depthwise，不混通道
            bias=False,
            padding_mode='reflect'  # 边缘用反射 padding，避免边界伪影
        )

        # 初始化为均值核（真正的低通）
        with torch.no_grad():
            self.blur.weight.fill_(1.0 / (kernel_size * kernel_size))

        if not learnable:
            # 固定为均值核，不参与学习
            for p in self.blur.parameters():
                p.requires_grad = False

        # Pointwise proj：混合通道信息，保持通道数不变
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x_blurred = self.blur(x)    # 空间模糊：压制高频局部响应
        x_proj = self.proj(x_blurred)  # 通道投影+归一化
        return x_proj  # [B, C, H, W]


# ===========================================================================
# 子模块 2a：GRU 状态核心（baseline）
# ===========================================================================

class GRUStateCore(nn.Module):
    """
    GRU 状态核心。作为对照 baseline，与 Mamba 做受控对比。

    单帧接口（推理时逐帧调用）：
        h_t = core.step(z_t, h_prev)

    整窗口接口（训练时并行处理）：
        h_seq = core.scan(z_seq)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # GRUCell：单步处理，方便逐帧递推
        self.cell = nn.GRUCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def step(self, z_t: torch.Tensor,
             h_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        单帧处理（推理时使用）。
        z_t:    [B, input_dim]
        h_prev: [B, hidden_dim] 或 None（自动初始化为零）
        返回:   h_t: [B, hidden_dim]
        """
        if h_prev is None:
            h_prev = torch.zeros(
                z_t.size(0), self.hidden_dim, device=z_t.device, dtype=z_t.dtype
            )
        return self.cell(z_t, h_prev)  # [B, hidden_dim]

    def scan(self, z_seq: torch.Tensor,
             h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        整序列处理（训练时使用）。
        z_seq: [B, T, input_dim]
        h0:    [B, hidden_dim] 或 None
        返回:  h_seq: [B, T, hidden_dim]
        """
        B, T, _ = z_seq.shape
        h = h0
        h_list = []
        for t in range(T):
            h = self.step(z_seq[:, t], h)
            h_list.append(h)
        return torch.stack(h_list, dim=1)  # [B, T, hidden_dim]


# ===========================================================================
# 子模块 2b：Mamba 状态核心
# ===========================================================================

class MambaStateCore(nn.Module):
    """
    Mamba 状态核心。

    本项目 Mamba 的用法与其他医学图像 Mamba 论文（VM-UNet, SegMamba 等）不同：
    - 其他论文：Mamba 沿空间维度扫描单帧图像的 patch，建模空间依赖
    - 本论文：Mamba 沿时间维度扫描帧序列的状态向量，建模生理节律

    这是 Mamba 在时序信号建模上的经典应用方式（类似原始论文的语言建模）。

    安装：pip install mamba-ssm
    如果未安装，自动 fallback 到 GRU，并打印警告。

    接口：
        scan(z_seq)  -> h_seq（训练，整窗口）
        step(z_t, cache) -> h_t, cache（推理，单帧，需要 mamba_ssm 支持）
    """

    def __init__(self, input_dim: int, hidden_dim: int, d_state: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self._use_mamba = False

        try:
            from mamba_ssm import Mamba
            # Mamba 输入输出维度必须一致（d_model），所以先投影到 d_model
            d_model = input_dim
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,   # SSM 状态维度，越大记忆越长（计算量也越大）
                d_conv=4,          # 局部卷积核大小
                expand=2,          # 内部扩展比例
            )
            # 输出投影：Mamba 输出维度 = d_model，需要投影到 hidden_dim
            self.output_proj = nn.Linear(d_model, hidden_dim)
            self._use_mamba = True
        except ImportError:
            warnings.warn(
                "[MambaStateCore] mamba_ssm 未安装，自动 fallback 到 GRU。\n"
                "安装命令：pip install mamba-ssm\n"
                "注意：fallback 后论文不能声称使用了 Mamba。"
            )
            # Fallback：用 GRU 代替，接口保持一致
            self._fallback_gru = GRUStateCore(input_dim, hidden_dim)

    def scan(self, z_seq: torch.Tensor,
             h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        整序列处理（训练时使用）。
        z_seq: [B, T, input_dim]
        返回:  h_seq: [B, T, hidden_dim]
        """
        if not self._use_mamba:
            return self._fallback_gru.scan(z_seq, h0)

        # Mamba 标准前向：[B, T, D] -> [B, T, D]（因果，当前帧只看历史）
        mamba_out = self.mamba(z_seq)     # [B, T, input_dim]
        h_seq = self.output_proj(mamba_out)  # [B, T, hidden_dim]
        return h_seq

    def step(self, z_t: torch.Tensor,
             inference_params=None) -> Tuple[torch.Tensor, object]:
        """
        单帧处理（推理时使用）。
        需要 mamba_ssm 的 inference_params 对象来维护 KV 缓存。

        使用方法：
            from mamba_ssm.utils.generation import InferenceParams
            cache = InferenceParams(max_seqlen=1000, max_batch_size=1)
            for frame in frames:
                h_t, cache = state_core.step(z_t, cache)

        z_t:            [B, input_dim]
        inference_params: mamba_ssm InferenceParams 对象
        返回: (h_t [B, hidden_dim], updated inference_params)
        """
        if not self._use_mamba:
            # GRU fallback：inference_params 里存 h_prev
            h_prev = inference_params if isinstance(inference_params, torch.Tensor) else None
            h_t = self._fallback_gru.step(z_t, h_prev)
            return h_t, h_t  # 返回 h_t 同时作为下一帧的 cache

        # Mamba stateful step（需要 mamba_ssm >= 1.2）
        # z_t 需要 unsqueeze 成 [B, 1, D] 再 squeeze 回来
        z_t_seq = z_t.unsqueeze(1)  # [B, 1, input_dim]
        out = self.mamba(z_t_seq, inference_params=inference_params)
        h_t = self.output_proj(out.squeeze(1))  # [B, hidden_dim]
        return h_t, inference_params


# ===========================================================================
# 主模块：GlobalStateBranch
# ===========================================================================

class GlobalStateBranch(nn.Module):
    """
    全局生理状态分支（PhysDec-GWNet 的核心新模块之一）。

    输入：Shared GhostEncoder 的 1/8 分辨率特征图（每帧一张）
    输出：[cos(φ_t), sin(φ_t), amplitude_t, confidence_t]
         其中 φ_t 是呼吸相位（圆周变量），amplitude_t 是呼吸幅度

    数据流（训练，整窗口模式）：
        [B, T, C, H, W]
        → detach（阻断梯度到 encoder）
        → SpatialLowPass（压制导丝高频）   [B*T, C, H, W]
        → ChannelProj（降维）              [B*T, proj_dim, H, W]
        → GAP（全局平均池化）              [B*T, proj_dim]
        → reshape                         [B, T, proj_dim]
        → GRU/Mamba scan                  [B, T, hidden_dim]
        → OutputHead                      [B, T, 4]

    数据流（推理，逐帧模式）：
        [B, C, H, W]  +  h_prev
        → 同上前四步                      [B, proj_dim]
        → GRU/Mamba step                  [B, hidden_dim]  + new h
        → OutputHead                      [B, 4]
    """

    def __init__(
        self,
        in_channels: int,           # 来自 encoder 1/8 特征的通道数
        proj_dim: int = 128,        # 通道投影维度（降维加速）
        hidden_dim: int = 64,       # GRU/Mamba 隐状态维度
        lp_kernel: int = 5,         # 低通核大小（5=覆盖约 1/8 分辨率下的导丝宽度）
        state_core_type: str = "gru",   # "gru" 或 "mamba"
        d_state: int = 16,          # Mamba SSM 状态维度（仅 Mamba 时有效）
        detach_encoder: bool = True,    # 关键：默认断开梯度（见 R1/P1）
        learnable_lp: bool = True,  # 低通核是否允许微调
        output_dim: int = 4,        # 4 = resp-only [cos,sin,amp,conf];
                                    # 8 = dual-cycle [resp_4, card_4]
    ):
        super().__init__()

        self.detach_encoder = detach_encoder
        self.state_core_type = state_core_type
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim

        # 1. 空间低通滤波
        self.low_pass = SpatialLowPass(
            in_channels, kernel_size=lp_kernel, learnable=learnable_lp
        )

        # 2. 通道压缩投影（降维：in_channels → proj_dim）
        self.channel_proj = nn.Sequential(
            nn.Conv2d(in_channels, proj_dim, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, proj_dim), proj_dim),
            nn.ReLU(inplace=True)
        )

        # 3. 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)  # [B, proj_dim, H, W] → [B, proj_dim, 1, 1]

        # 4. 时序状态核心
        if state_core_type == "gru":
            self.state_core = GRUStateCore(proj_dim, hidden_dim)
        elif state_core_type == "mamba":
            self.state_core = MambaStateCore(proj_dim, hidden_dim, d_state=d_state)
        else:
            raise ValueError(
                f"Unknown state_core_type: '{state_core_type}'，"
                f"请选择 'gru' 或 'mamba'"
            )

        # 5. 输出头：隐状态 → output_dim 维
        # output_dim=4: 单周期 [cos_phi, sin_phi, amplitude, confidence]
        # output_dim=8: 双周期解耦 [resp_4, card_4]; 下游切片使用
        self.output_dim = int(output_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.output_dim)
            # 注意：不在这里加 tanh/sigmoid，由外部损失函数处理
            # cos/sin 在损失计算前做 F.normalize，amplitude 在损失中除以 amp_scale
        )

    def _extract_z(self, feat: torch.Tensor) -> torch.Tensor:
        """
        从单帧或批量帧特征中提取低频全局向量 z。
        内部辅助函数，被 forward_window 和 forward_step 共用。

        feat: [B, C, H, W]（单帧）或 [B*T, C, H, W]（批量帧）
        返回: z: [B, proj_dim] 或 [B*T, proj_dim]
        """
        feat_lp = self.low_pass(feat)                        # [*, C, H, W]
        feat_proj = self.channel_proj(feat_lp)               # [*, proj_dim, H, W]
        z = self.gap(feat_proj).squeeze(-1).squeeze(-1)      # [*, proj_dim]
        return z

    # -------------------------------------------------------------------
    # 训练接口：整窗口处理
    # -------------------------------------------------------------------

    def forward_window(
        self,
        feat_18_seq: torch.Tensor,    # [B, T, C, H, W]
        h0: Optional[torch.Tensor] = None,   # 初始隐状态（通常为 None）
    ) -> torch.Tensor:
        """
        训练专用：处理整个时序窗口。

        feat_18_seq: [B, T, C, H, W]  来自 encoder 1/8 特征序列
        h0:          [B, hidden_dim]  初始隐状态，通常为 None（从零开始）

        返回: state_seq [B, T, 4]
              每帧的 [cos_phi, sin_phi, amplitude, confidence]
        """
        B, T, C, H, W = feat_18_seq.shape

        # ============ 关键 1：stop-gradient ============
        # 阻止 L_state 的梯度反向传播到 GhostEncoder
        # GhostEncoder 只被 L_seg / L_clDice 等分割损失优化
        if self.detach_encoder:
            feat_seq = feat_18_seq.detach()
        else:
            feat_seq = feat_18_seq  # 消融实验用：允许梯度流回
        # ===============================================

        # 展平时间维度，批量处理所有帧
        feat_flat = feat_seq.reshape(B * T, C, H, W)   # [B*T, C, H, W]
        z_flat = self._extract_z(feat_flat)             # [B*T, proj_dim]
        z_seq = z_flat.reshape(B, T, self.proj_dim)     # [B, T, proj_dim]

        # 时序状态建模
        if self.state_core_type == "gru":
            h_seq = self.state_core.scan(z_seq, h0)    # [B, T, hidden_dim]
        elif self.state_core_type == "mamba":
            h_seq = self.state_core.scan(z_seq)        # [B, T, hidden_dim]

        # 输出状态预测
        state_seq = self.output_head(h_seq)             # [B, T, 4]
        return state_seq

    # -------------------------------------------------------------------
    # 推理接口：单帧递推
    # -------------------------------------------------------------------

    def forward_step(
        self,
        feat_18: torch.Tensor,          # [B, C, H, W]  当前帧
        h_prev: Optional[torch.Tensor] = None,  # [B, hidden_dim]  上一帧隐状态
        mamba_cache=None,               # Mamba 专用推理缓存（GRU 不用）
    ) -> Tuple[torch.Tensor, torch.Tensor, object]:
        """
        推理专用：单帧递推。

        feat_18: [B, C, H, W]
        h_prev:  [B, hidden_dim] 或 None
        mamba_cache: InferenceParams（仅 Mamba 时使用）

        返回:
            state_t:  [B, 4]  当前帧的生理状态
            h_out:    [B, hidden_dim]  新的隐状态（GRU 用）
            new_cache: 更新后的 mamba_cache（Mamba 用）
        """
        # stop-gradient（推理时通常不需要梯度，但保持接口一致）
        if self.detach_encoder:
            feat = feat_18.detach()
        else:
            feat = feat_18

        z_t = self._extract_z(feat)  # [B, proj_dim]

        # 单帧状态更新
        if self.state_core_type == "gru":
            h_out = self.state_core.step(z_t, h_prev)  # [B, hidden_dim]
            new_cache = None
        elif self.state_core_type == "mamba":
            h_out, new_cache = self.state_core.step(z_t, mamba_cache)

        state_t = self.output_head(h_out)  # [B, 4]
        return state_t, h_out, new_cache

    def get_state_components(
        self, state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        把输出向量拆分成有名字的组件，方便使用。

        state: [B, 4] 或 [B, T, 4]
        返回 dict:
            cos_phi:    呼吸相位余弦
            sin_phi:    呼吸相位正弦
            amplitude:  呼吸幅度（未归一化的预测值）
            confidence: 预测置信度（可用于下游加权）
        """
        return {
            "cos_phi":    state[..., 0],
            "sin_phi":    state[..., 1],
            "amplitude":  state[..., 2],
            "confidence": state[..., 3],
        }


# ===========================================================================
# 单元测试（运行此文件可直接验证）
# ===========================================================================

def _run_unit_tests():
    """
    运行基本正确性测试。
    不需要 GPU，CPU 上运行。
    """
    print("=" * 60)
    print("GlobalStateBranch 单元测试")
    print("=" * 60)

    # ---------- 配置 ----------
    B, T, C, H, W = 2, 10, 128, 64, 64   # 2个样本，10帧，128通道，64x64
    in_channels = C
    device = torch.device("cpu")

    # 创建假的 encoder 输出（需要 requires_grad=True 来测试 detach）
    feat_seq = torch.randn(B, T, C, H, W, requires_grad=True)

    # ---------- 测试 1：detach 模式下梯度不流回 feat_seq ----------
    print("\n[测试 1] detach_encoder=True 时，L_state 梯度不流回 encoder")
    branch = GlobalStateBranch(
        in_channels=in_channels, proj_dim=64, hidden_dim=32,
        state_core_type="gru", detach_encoder=True
    )
    state_seq = branch.forward_window(feat_seq)  # [B, T, 4]
    fake_loss = state_seq.sum()
    fake_loss.backward()

    if feat_seq.grad is None:
        print("  ✓ PASS: feat_seq.grad 为 None，梯度被 detach 正确阻断")
    else:
        print("  ✗ FAIL: feat_seq.grad 不为 None，detach 未生效！")

    # ---------- 测试 2：no-detach 消融，梯度可以流回 ----------
    print("\n[测试 2] detach_encoder=False 时，梯度可以流回（消融实验用）")
    feat_seq2 = torch.randn(B, T, C, H, W, requires_grad=True)
    branch_no_detach = GlobalStateBranch(
        in_channels=in_channels, proj_dim=64, hidden_dim=32,
        state_core_type="gru", detach_encoder=False
    )
    state_seq2 = branch_no_detach.forward_window(feat_seq2)
    state_seq2.sum().backward()

    if feat_seq2.grad is not None:
        print("  ✓ PASS: feat_seq2.grad 不为 None，梯度正确流回")
    else:
        print("  ✗ FAIL: feat_seq2.grad 为 None，no-detach 模式失效！")

    # ---------- 测试 3：输出形状正确 ----------
    print("\n[测试 3] 输出形状检查")
    with torch.no_grad():
        feat_seq3 = torch.randn(B, T, C, H, W)
        branch3 = GlobalStateBranch(in_channels=C, proj_dim=64, hidden_dim=32)
        state_seq3 = branch3.forward_window(feat_seq3)

    expected_shape = (B, T, 4)
    if state_seq3.shape == expected_shape:
        print(f"  ✓ PASS: 输出形状 {state_seq3.shape} 符合预期 {expected_shape}")
    else:
        print(f"  ✗ FAIL: 输出形状 {state_seq3.shape}，预期 {expected_shape}")

    # ---------- 测试 4：推理单帧接口与训练接口输出维度一致 ----------
    print("\n[测试 4] 推理 forward_step 输出维度检查")
    with torch.no_grad():
        feat_single = torch.randn(B, C, H, W)
        state_t, h_out, _ = branch3.forward_step(feat_single, h_prev=None)

    if state_t.shape == (B, 4) and h_out.shape == (B, 32):
        print(f"  ✓ PASS: state_t {state_t.shape}, h_out {h_out.shape}")
    else:
        print(f"  ✗ FAIL: state_t {state_t.shape}, h_out {h_out.shape}")

    # ---------- 测试 5：burn-in mask 是否正确（间接验证） ----------
    print("\n[测试 5] 多帧递推一致性（GRU 状态递推）")
    with torch.no_grad():
        branch5 = GlobalStateBranch(in_channels=C, proj_dim=64, hidden_dim=32)
        # 用 forward_window 的结果和逐帧 forward_step 的结果对比
        feat_seq5 = torch.randn(1, 5, C, H, W)
        window_out = branch5.forward_window(feat_seq5)  # [1, 5, 4]

        h = None
        step_outs = []
        for t in range(5):
            s_t, h, _ = branch5.forward_step(feat_seq5[0:1, t], h_prev=h)
            step_outs.append(s_t)
        step_out = torch.stack(step_outs, dim=1)  # [1, 5, 4]

        diff = (window_out - step_out).abs().max().item()
        if diff < 1e-5:
            print(f"  ✓ PASS: window 和逐帧 step 结果一致，最大差异 {diff:.2e}")
        else:
            print(f"  ✗ FAIL: window 和逐帧 step 结果不一致，最大差异 {diff:.2e}")

    # ---------- 测试 6：get_state_components 接口 ----------
    print("\n[测试 6] get_state_components 拆分接口")
    with torch.no_grad():
        dummy_state = torch.randn(B, T, 4)
        branch6 = GlobalStateBranch(in_channels=C, proj_dim=64, hidden_dim=32)
        comps = branch6.get_state_components(dummy_state)
    keys = set(comps.keys())
    expected_keys = {"cos_phi", "sin_phi", "amplitude", "confidence"}
    if keys == expected_keys:
        print(f"  ✓ PASS: 返回 keys 正确 {keys}")
    else:
        print(f"  ✗ FAIL: 返回 keys {keys}，预期 {expected_keys}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    _run_unit_tests()
