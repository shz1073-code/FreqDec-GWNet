import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBranch(nn.Module):
    def __init__(self, channels):
        super(TemporalBranch, self).__init__()
        # 对应路线图 Step 3: 为了轻量化，使用 Channel Attention 提取特征
        # 使用 1D 卷积处理通道间的交互，极大地节省了参数量
        self.q_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.k_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_curr, x_prev):
        # 输入接收两个张量 Current_Feat 和 Prev_Feat
        B, C, H, W = x_curr.shape
        
        # 1. 门控融合 (Gate Fusion): 计算两帧的相似度 (Cosine Similarity)
        # 将特征图展平后计算余弦相似度
        curr_flat = x_curr.view(B, -1)
        prev_flat = x_prev.view(B, -1)
        # 相似度形状转为 [B, 1, 1, 1] 以便后续直接做广播乘法
        cos_sim = F.cosine_similarity(curr_flat, prev_flat, dim=1).view(B, 1, 1, 1)
        
        # 相似度过低则置 0：这里我们设一个阈值(比如0.5)，低于此值说明可能是镜头切换
        # sim_mask 的值为 0 或 1
        sim_mask = (cos_sim > 0.5).float()

        # 2. Cross-Attention 核心逻辑 (Channel 维度)
        # Query (Q) = Current_Feat; Key (K) & Value (V) = Prev_Feat
        # 先做全局池化拿到通道描述符: [B, C, 1, 1] -> [B, 1, C]
        q = self.pool(x_curr).squeeze(-1).squeeze(-1).unsqueeze(1) 
        k = self.pool(x_prev).squeeze(-1).squeeze(-1).unsqueeze(1)
        
        # 通过 1D 卷积生成真正的 Q 和 K
        q = self.q_conv(q) # [B, 1, C]
        k = self.k_conv(k) # [B, 1, C]
        
        # 计算通道注意力权重: Softmax(Q * K^T)
        attn_weight = F.softmax(torch.bmm(q.permute(0, 2, 1), k), dim=-1) # [B, C, C]
        
        # Value (V) = Prev_Feat, 将其重塑为 [B, C, H*W] 以便矩阵相乘
        v = x_prev.view(B, C, H * W) 
        
        # 将注意力权重施加到上一帧的 Value 上，并恢复空间形状
        out_temp = torch.bmm(attn_weight, v).view(B, C, H, W)

        # 3. 输出前乘以相似度掩码，防止错误信息传递
        return out_temp * sim_mask

# ================= 测试代码 =================
if __name__ == "__main__":
    batch_size = 1
    channels = 64  
    height, width = 32, 32 
    
    # 模拟当前帧特征和上一帧特征
    current_feat = torch.randn(batch_size, channels, height, width)
    # 为了测试，我们让上一帧和当前帧加一点微小噪声，模拟连续帧的高相似度
    prev_feat = current_feat + torch.randn(batch_size, channels, height, width) * 0.1
    
    # 实例化时序分支
    temporal_branch = TemporalBranch(channels=channels)
    
    # 跑通 Forward
    output = temporal_branch(current_feat, prev_feat)
    
    print(f"Current Frame Input shape: {current_feat.shape}")
    print(f"Previous Frame Input shape: {prev_feat.shape}")
    print(f"Temporal Output shape: {output.shape}")