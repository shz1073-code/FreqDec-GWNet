import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import glob
import random
from scipy import interpolate

# 导入之前的物理伪影引擎
from freqdec_gwnet.data.physics_aug import PhysicalArtifactsInjector

class SyntheticVideoDataset(Dataset):
    def __init__(self, bg_dir, num_samples=2000, seq_len=5, img_size=(512, 512)):
        """
        bg_dir: 刚才生成的 ARCADE 背景库路径
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.img_size = img_size
        self.injector = PhysicalArtifactsInjector()
        
        # 1. 加载所有背景图片路径
        self.bg_files = glob.glob(os.path.join(bg_dir, "*.png"))
        if len(self.bg_files) == 0:
            raise ValueError(f"No .png files found in {bg_dir}. Did you run prepare_arcade_v2.py?")
        print(f"Dataset initialized with {len(self.bg_files)} ARCADE background frames.")

    def __len__(self):
        return self.num_samples

    def _generate_base_control_points(self):
        """生成基础 B-Spline 控制点"""
        w, h = self.img_size
        num_points = np.random.randint(5, 8) # 随机控制点数量
        # 随机导丝走向 (从左到右，或从上到下)
        if random.random() > 0.5:
            x_coords = np.linspace(w * 0.1, w * 0.9, num_points)
            y_coords = np.random.randint(h * 0.2, h * 0.8, size=num_points)
        else:
            x_coords = np.random.randint(w * 0.2, w * 0.8, size=num_points)
            y_coords = np.linspace(h * 0.1, h * 0.9, num_points)
            
        # 加扰动
        x_coords += np.random.randint(-20, 20, size=num_points)
        y_coords += np.random.randint(-20, 20, size=num_points)
        return x_coords, y_coords

    def __getitem__(self, idx):
        # 1. 随机挑选一张 ARCADE 背景
        bg_path = random.choice(self.bg_files)
        background_base = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
        
        # 2. 准备序列容器
        images_seq = torch.zeros((self.seq_len, 1, self.img_size[0], self.img_size[1]), dtype=torch.float32)
        masks_seq = torch.zeros((self.seq_len, 1, self.img_size[0], self.img_size[1]), dtype=torch.float32)

        # 生成导丝基准点
        base_x, base_y = self._generate_base_control_points()
        
        # 随机模拟呼吸运动参数 (频率和幅度)
        resp_amp = np.random.uniform(2.0, 8.0) # 呼吸幅度
        resp_freq = np.random.uniform(0.2, 0.5) # 呼吸频率

        for t in range(self.seq_len):
            # A. 背景形变 (模拟心脏/血管随呼吸的跳动) - TMI 高级加分项
            # 简单起见，我们对背景做微小的仿射变换
            shift_x = resp_amp * np.sin(t * resp_freq)
            shift_y = resp_amp * np.cos(t * resp_freq)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            bg_t = cv2.warpAffine(background_base, M, self.img_size, borderMode=cv2.BORDER_REFLECT)
            
            # B. 导丝生成 (微小蠕动)
            current_x = base_x + np.random.randint(-2, 2, size=base_x.shape) + (t * 3) # 推进
            current_y = base_y + np.random.randint(-2, 2, size=base_y.shape)
            
            # B-Spline 插值
            try:
                tck, _ = interpolate.splprep([current_x, current_y], s=0, k=min(3, len(base_x)-1))
                u_fine = np.linspace(0, 1, 1000)
                x_fine, y_fine = interpolate.splev(u_fine, tck)
            except:
                # 容错处理：如果插值失败，退化为直线
                x_fine, y_fine = base_x, base_y

            # 绘制 Mask
            mask = np.zeros(self.img_size, dtype=np.uint8)
            curve_points = np.vstack((x_fine, y_fine)).T.astype(np.int32)
            cv2.polylines(mask, [curve_points], isClosed=False, color=255, thickness=np.random.randint(2, 4), lineType=cv2.LINE_AA)
            
            # C. 物理融合 (X-ray Absorption Model) - 关键修改！
            # 导丝是金属，吸收 X 光，所以在反色图(白血管)中，导丝应该是更黑或者更白？
            # 通常：数字减影血管造影 (DSA) 中，血管和导丝都是黑色的 (背景是灰白)。
            # 或者：普通透视中，骨头是白的，空气是黑的，金属导丝是极黑的。
            # 这里我们采用通用的 "Inverted" 模式：假设背景是暗色，导丝是亮色 (Mask)，
            # 或者是：背景是亮色 X 光，导丝是黑色。
            # 为了适配我们的网络 (通常预测高亮 Mask)，我们让导丝在图像中呈现为"黑色"(吸收)，
            # 但 Label 是白色的。
            
            bg_float = bg_t.astype(np.float32)
            
            # 模拟导丝遮挡：将导丝区域的像素值 减去 一个值 (变黑)
            # 或者做乘法：I_out = I_bg * (1 - alpha * Mask)
            alpha = np.random.uniform(0.4, 0.7) # 导丝不透明度
            
            # 归一化 mask 到 0-1
            mask_norm = mask.astype(np.float32) / 255.0
            
            # 融合公式：在背景上"减去"导丝的亮度 (模拟金属阻挡射线，图像变暗)
            # 假设 ARCADE 是普通 X 光 (背景亮，血管暗)，导丝应该更暗
            img_fused = bg_float * (1.0 - alpha * mask_norm)
            
            # D. 注入噪声和模糊
            blurred = self.injector.add_motion_blur(img_fused.astype(np.uint8), degree=np.random.randint(5, 10))
            noisy_img = self.injector.add_poisson_noise(blurred, dose_level=np.random.uniform(8.0, 15.0))
            
            # E. 转 Tensor
            images_seq[t, 0, :, :] = torch.from_numpy(noisy_img).float() / 255.0
            masks_seq[t, 0, :, :] = torch.from_numpy(mask).float() / 255.0 # Label 依然是白色的

        return images_seq, masks_seq
