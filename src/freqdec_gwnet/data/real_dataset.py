import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# OpenCV 在多进程 DataLoader 下偶发段错误时，先关闭内部线程和 OpenCL 更稳。
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class RealGuidewireVideoDataset(Dataset):
    def __init__(
        self,
        data_root,
        split="train",
        seq_len=5,
        img_size=(512, 512),
        exclude_sequences_file=None,
        augmentation_profile="fluoro_hard",
        hflip_p=0.5,
        vflip_p=0.5,
    ):
        """
        data_root: 也就是你整理好的 ori_data 文件夹路径
        split: "train", "val", 或 "test"
        seq_len: 时序长度 T=5
        """
        self.seq_len = seq_len
        self.img_size = img_size
        self.split = split
        self.augmentation_profile = augmentation_profile
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.exclude_sequences = self._load_excluded_sequences(exclude_sequences_file)
        
        self.img_dir = os.path.join(data_root, split, "images")
        self.ann_dir = os.path.join(data_root, split, "labels")
        
        # 1. 扫描所有的有效滑动窗口 (Sliding Windows)
        self.samples = [] # 存放 (seq_name, start_frame_idx, list_of_frames)
        self.skipped_windows = 0
        self.skipped_sequences = 0
        
        # 遍历所有的序列文件夹 (如 zhu_seq_001)
        for seq_name in sorted(os.listdir(self.img_dir)):
            seq_img_path = os.path.join(self.img_dir, seq_name)
            seq_ann_path = os.path.join(self.ann_dir, seq_name)
            if not os.path.isdir(seq_img_path):
                continue
            if seq_name in self.exclude_sequences:
                self.skipped_sequences += 1
                continue
            if not os.path.isdir(seq_ann_path):
                print(f"[{split.upper()} Set] 跳过序列 {seq_name}: 缺少标签目录 {seq_ann_path}")
                continue
            
            # 获取该序列下所有的图片并排序 (确保按时间顺序)
            frames = sorted(os.listdir(seq_img_path))
            
            # 如果视频总帧数 >= 我们需要的 seq_len (比如 5)，则构建滑动窗口
            if len(frames) >= self.seq_len:
                # 例如 10 帧的视频，T=5，可以切出 6 个片段: 起点分别为 0, 1, 2, 3, 4, 5
                for start_idx in range(len(frames) - self.seq_len + 1):
                    window_frames = frames[start_idx : start_idx + self.seq_len]

                    # 训练前先过滤掉标签缺失的窗口，避免把缺失标注误当成全背景。
                    if not all(
                        os.path.isfile(os.path.join(seq_ann_path, frame_name))
                        for frame_name in window_frames
                    ):
                        self.skipped_windows += 1
                        continue

                    self.samples.append({
                        "seq_name": seq_name,
                        "frame_names": window_frames
                    })
                    
        print(f"[{split.upper()} Set] 扫描完毕: 共提取出 {len(self.samples)} 个有效时序片段 (T={seq_len})")
        if self.skipped_sequences > 0:
            print(f"[{split.upper()} Set] 根据排除名单跳过了 {self.skipped_sequences} 个序列")
        if self.skipped_windows > 0:
            print(f"[{split.upper()} Set] 额外跳过了 {self.skipped_windows} 个标签不完整的窗口")
        
        # 2. 定义【时空一致性】的数据增强 Pipeline
        self.transform = self._get_transforms()

    @staticmethod
    def _load_excluded_sequences(exclude_sequences_file):
        if not exclude_sequences_file:
            return set()
        if not os.path.isfile(exclude_sequences_file):
            raise FileNotFoundError(f"找不到排除序列名单: {exclude_sequences_file}")

        excluded = set()
        with open(exclude_sequences_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                excluded.add(line)
        return excluded

    def _get_transforms(self):
        if self.split in ["val", "test"]:
            # 推理阶段不再依赖 albumentations，避免在纯评估/定位流程里触发 scipy 导入链。
            return None

        from freqdec_gwnet.data.physics_aug import build_fluoro_train_transform
        return build_fluoro_train_transform(
            seq_len=self.seq_len,
            img_size=self.img_size,
            profile=self.augmentation_profile,
            hflip_p=self.hflip_p,
            vflip_p=self.vflip_p,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        seq_name = sample_info["seq_name"]
        frame_names = sample_info["frame_names"]
        
        images_np = []
        masks_np = []
        
        # 1. 连续加载 T 帧图片和标签
        for f_name in frame_names:
            img_path = os.path.join(self.img_dir, seq_name, f_name)
            mask_path = os.path.join(self.ann_dir, seq_name, f_name)
            
            # X光图读灰度即可
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                raise FileNotFoundError(f"无法读取图像文件: {img_path}")
            if mask is None:
                raise FileNotFoundError(f"无法读取标签文件: {mask_path}")
            
            images_np.append(img)
            masks_np.append(mask)

        if self.split in ["val", "test"]:
            images_seq = torch.zeros((self.seq_len, 1, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            masks_seq = torch.zeros((self.seq_len, 1, self.img_size[0], self.img_size[1]), dtype=torch.float32)

            for i in range(self.seq_len):
                img_resized = cv2.resize(images_np[i], (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(masks_np[i], (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
                images_seq[i, 0] = torch.from_numpy(img_resized).float() / 255.0
                masks_seq[i, 0] = torch.from_numpy((mask_resized.astype(np.float32) / 255.0 > 0.5).astype(np.float32))

            return images_seq, masks_seq
            
        # 2. 组装送入 Albumentations 的字典
        transform_data = {'image': images_np[0], 'mask': masks_np[0]}
        for i in range(1, self.seq_len):
            transform_data[f'image{i}'] = images_np[i]
            transform_data[f'mask{i}'] = masks_np[i]
            
        # 3. 执行同步数据增强！
        augmented = self.transform(**transform_data)
        
        # 4. 取出增强后的数据，转为 PyTorch 需要的 [T, C, H, W] 张量
        images_seq = torch.zeros((self.seq_len, 1, self.img_size[0], self.img_size[1]), dtype=torch.float32)
        masks_seq = torch.zeros((self.seq_len, 1, self.img_size[0], self.img_size[1]), dtype=torch.float32)
        
        for i in range(self.seq_len):
            key_img = 'image' if i == 0 else f'image{i}'
            key_mask = 'mask' if i == 0 else f'mask{i}'
            
            # 归一化到 0~1 之间
            images_seq[i, 0] = torch.from_numpy(augmented[key_img]).float() / 255.0
            
            # 确保 mask 也是 0~1 的纯净浮点数 (有些开源数据集的 mask 是 255 代表前景)
            mask_aug = augmented[key_mask].astype(np.float32) / 255.0
            masks_seq[i, 0] = torch.from_numpy((mask_aug > 0.5).astype(np.float32))

        return images_seq, masks_seq

# ================= 测试代码 =================
# ================= 测试与可视化代码 =================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    
    # 请确保这里的路径是你服务器上的真实路径
    DATA_ROOT = "/home/amax/data/media/rmlab_dataset/SHZ/ori_data" 
    
    print("正在测试真实视频 Dataset 并生成可视化预览...")
    train_dataset = RealGuidewireVideoDataset(
        data_root=DATA_ROOT,
        split="train",
        seq_len=5,
        augmentation_profile="fluoro_hard",
    )
    
    if len(train_dataset) > 0:
        # 随机抽取一个样本 (包含 5 帧)
        import random
        sample_idx = random.randint(0, len(train_dataset) - 1)
        images, masks = train_dataset[sample_idx]
        
        print(f"✅ 成功提取样本 {sample_idx}，正在绘制可视化图像...")
        
        # 创建一个 2 行 5 列的画布
        # 第一行放原始 X 光图像，第二行放对应的 Ground Truth Mask
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f"Spatio-Temporal Consistent Augmentation (Sequence {sample_idx})", fontsize=16)
        
        for t in range(5):
            # 取出第 t 帧的图像和掩膜，并去掉 Channel 维度转为 numpy
            img_t = images[t, 0].numpy()
            mask_t = masks[t, 0].numpy()
            
            # 绘制原图
            axes[0, t].imshow(img_t, cmap='gray', vmin=0, vmax=1)
            axes[0, t].set_title(f"Frame t+{t}")
            axes[0, t].axis('off')
            
            # 绘制 Mask
            axes[1, t].imshow(mask_t, cmap='gray')
            axes[1, t].set_title(f"Mask t+{t}")
            axes[1, t].axis('off')
            
        plt.tight_layout()
        
        # 将可视化结果保存到当前目录下
        save_path = "sample_batch_vis.png"
        plt.savefig(save_path, dpi=150)
        print(f"🎉 可视化结果已保存至: {os.path.abspath(save_path)}")
        print("请将该图片下载到本地查看，确认导丝标注是否与原图精准对齐！")
        
    else:
        print("❌ 未能提取到样本，请检查路径。")
