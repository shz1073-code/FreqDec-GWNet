import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.fft

def visualize_frequency_spectrum():
    # 1. 制造一张模拟图
    h, w = 512, 512
    img = np.zeros((h, w), dtype=np.float32)
    
    # A. 模拟骨骼 (大面积、边缘模糊的块状物)
    cv2.circle(img, (200, 200), 80, 0.5, -1)
    cv2.rectangle(img, (300, 100), (450, 400), 0.6, -1)
    # 对骨骼做强高斯模糊 -> 模拟低频背景
    img_bone = cv2.GaussianBlur(img, (45, 45), 0)
    
    # B. 模拟导丝 (细长、锐利、高频)
    img_wire = np.zeros((h, w), dtype=np.float32)
    # 画一条斜着的导丝
    cv2.line(img_wire, (100, 100), (400, 400), 1.0, 2) 
    
    # C. 合成图
    img_combined = img_bone + img_wire
    
    # 2. 转为 Tensor 并进行 FFT 变换
    # Shape: [1, 1, H, W]
    tensor_img = torch.from_numpy(img_combined).unsqueeze(0).unsqueeze(0)
    
    # FFT 变换 (得到复数)
    fft_x = torch.fft.rfft2(tensor_img)
    
    # 计算频谱幅值 (Log Magnitude) 用于可视化
    # Shift 频谱中心，让低频在中间
    fft_abs = torch.abs(fft_x)
    # log 压缩以便人眼观察
    fft_log = torch.log(fft_abs + 1e-6)
    
    # 转换为 Numpy 绘图
    spec_img = fft_log[0, 0].numpy()
    
    # 3. 模拟“理想的高频滤波器” (假设网络学到了只保留高频方向)
    # 导丝是 45度，其频谱特征应该在 -45度 方向有一条亮线
    # 这里我们简单模拟：把中心低频区域挖掉 (High Pass Filter)
    mask = torch.ones_like(fft_x)
    cy, cx = mask.shape[2]//2, mask.shape[3]//2
    # 暴力屏蔽低频 (中心区域)
    # 注意：rfft2 的宽度是 W/2+1，频域中心在左侧，这里简化处理演示原理
    mask[:, :, :20, :20] = 0 
    
    # 逆变换查看效果
    ifft_filtered = torch.fft.irfft2(fft_x * mask, s=(h, w))
    img_filtered = ifft_filtered[0, 0].numpy()

    # ================= 画图 =================
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_combined, cmap='gray')
    plt.title("Input (Blurry Bone + Sharp Wire)")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(spec_img, cmap='jet') # jet 伪彩能看清能量分布
    plt.title("Frequency Spectrum (Log)")
    plt.axis('off')
    # 注释：频谱图中心最亮的是低频(骨骼)，
    # 发散出去的星芒状亮线是高频(导丝边缘)
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_filtered, cmap='gray')
    plt.title("Filtered (Simulated High-Pass)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_frequency_spectrum()