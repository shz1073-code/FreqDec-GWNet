import torch
import torch.nn as nn

class MomentumTTA:
    """
    动量测试时自适应 (Momentum Test-Time Adaptation)
    针对真实测试环境，实现零样本 (Zero-shot) 风格迁移
    """
    def __init__(self, model, momentum=0.1):
        self.model = model
        self.momentum = momentum
        self.original_momentums = {} # 用于保存原本的 momentum

    def __enter__(self):
        """进入上下文：强制开启 BN 层的更新"""
        # 确保总体模型处于 eval() 模式（关闭 Dropout 等）
        self.model.eval() 
        
        # 遍历模型的所有层，精准打击 BatchNorm2d
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # 保存原本的 momentum
                self.original_momentums[name] = module.momentum
                
                # 核心机制：强制 BN 层进入 train 模式
                # 这样它就会计算当前测试数据的 Batch 均值/方差，
                # 并以我们设定的动量 (0.1) 悄悄更新 running_stats
                module.train() 
                module.track_running_stats = True
                module.momentum = self.momentum
                
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文：恢复现场"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # 恢复 eval 模式和原本的 momentum
                module.eval()
                module.momentum = self.original_momentums[name]

# ================= 测试代码 =================
if __name__ == "__main__":
    # 我们用一个简单的包含 BN 层的网络来测试
    dummy_model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.BatchNorm2d(16)
    )
    
    # 初始化网络并进入标准的评估模式 (eval)
    dummy_model.eval()
    
    # 模拟一张完全不同于训练集分布的真实测试图像 (比如整体特别亮)
    real_test_img = torch.ones(1, 1, 64, 64) * 100.0 
    
    print("--- 传统推理模式 (不更新统计量) ---")
    with torch.no_grad():
        out_normal = dummy_model(real_test_img)
        bn_layer = dummy_model[1]
        print(f"推理后 BN 层的 running_mean (期望接近 0): {bn_layer.running_mean[0].item():.4f}")

    print("\n--- TTA 推理模式 (零样本自适应) ---")
    # 使用我们的上下文管理器
    with torch.no_grad(): # 依然不需要计算梯度，极其省显存
        with MomentumTTA(dummy_model, momentum=0.1) as tta_model:
            out_tta = tta_model(real_test_img)
            
    print(f"TTA后 BN 层的 running_mean (期望发生偏移，向新数据靠拢): {bn_layer.running_mean[0].item():.4f}")