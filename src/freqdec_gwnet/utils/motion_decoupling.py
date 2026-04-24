import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

class MotionDecoupler:
    def __init__(self):
        pass

    def get_skeleton_and_normal(self, mask):
        """
        Step 4.2: 提取骨架并计算每个点的法向量
        mask: 二值化的导丝预测图 (0 或 255)
        """
        # 1. 骨架化 (Skeletonization)
        # 将 mask 转为布尔型 (0, 1)
        binary_mask = mask > 100 # 阈值稍微放宽以适应模糊
        skeleton = skeletonize(binary_mask)
        
        # 获取骨架点的坐标 (y, x)
        y_indices, x_indices = np.where(skeleton)
        points = np.vstack((x_indices, y_indices)).T # shape: [N, 2]
        
        # 2. 计算法向量 (Normal Vector)
        # 对每个点，取其前后邻域点计算切向量，然后旋转90度得到法向量
        if len(points) < 5:
            return points, np.array([]) # 骨架太短，无法计算
            
        # 计算切向量 (dx, dy)
        gradients = np.gradient(points, axis=0)
        dx, dy = gradients[:, 0], gradients[:, 1]
        
        # 法向量 = (-dy, dx)
        # 归一化
        norms = np.sqrt(dx**2 + dy**2) + 1e-6
        nx = -dy / norms
        ny = dx / norms
        
        normal_vectors = np.vstack((nx, ny)).T
        return points, normal_vectors

    def calculate_respiratory_offset(self, prev_img, curr_img, mask_curr):
        """
        Step 4.3: 核心解耦算法
        prev_img, curr_img: 前后两帧灰度图
        mask_curr: 当前帧的导丝 Mask (用于限定计算区域)
        """
        # 1. 计算稠密光流 (Farneback 光流法)
        # 参数微调：winsize=20 以捕捉更大运动
        flow = cv2.calcOpticalFlowFarneback(prev_img, curr_img, None, 
                                            pyr_scale=0.5, levels=3, winsize=20, 
                                            iterations=5, poly_n=7, poly_sigma=1.5, flags=0)
        
        # 2. 提取骨架和法向量
        points, normals = self.get_skeleton_and_normal(mask_curr)
        
        if len(points) == 0:
            return (0, 0) # 没检测到导丝，不校准
            
        # 3. 几何投影解耦
        # 提取骨架位置的光流向量 V_obs
        # 注意: flow 的坐标是 (y, x, 2)
        v_obs_x = flow[points[:, 1], points[:, 0], 0]
        v_obs_y = flow[points[:, 1], points[:, 0], 1]
        
        # 投影到法向量上: V_resp = V_obs · n
        # 点积: (vx * nx) + (vy * ny)
        v_resp_magnitude = v_obs_x * normals[:, 0] + v_obs_y * normals[:, 1]
        
        # 恢复为法向运动向量: V_resp_vec = magnitude * n
        v_resp_x = v_resp_magnitude * normals[:, 0]
        v_resp_y = v_resp_magnitude * normals[:, 1]
        
        # 4. 全局呼吸补偿量估计 (取中位数鲁棒性最强)
        global_shift_x = np.median(v_resp_x)
        global_shift_y = np.median(v_resp_y)
        
        return global_shift_x, global_shift_y

    def apply_compensation(self, roadmap, shift_x, shift_y):
        """
        对静态路图应用反向平移补偿
        """
        M = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
        h, w = roadmap.shape
        corrected_roadmap = cv2.warpAffine(roadmap, M, (w, h))
        return corrected_roadmap

# ================= 临床场景模拟测试 (Sim-to-Real 修正版) =================
if __name__ == "__main__":
    decoupler = MotionDecoupler()
    
    # 1. 模拟场景：
    # 导丝向右猛插 (切向运动, dx=5) + 
    # 血管向下微动 (呼吸运动, dy=3)
    # 注意：为了配合 winsize，我们将位移量稍微减小一点点，或者增加模糊度
    
    dx_move = 5.0
    dy_move = 3.0
    
    # 创建带噪声背景 (模拟 X 光散射)
    h, w = 200, 200
    frame_prev = np.random.randint(50, 100, (h, w)).astype(np.uint8)
    frame_curr = frame_prev.copy() # 背景不动 (假设背景只有噪声)
    
    # 绘制导丝函数 (带高斯模糊，模拟真实 X 光的边缘衰减)
    def draw_fuzzy_wire(img, start, end):
        mask = np.zeros_like(img)
        cv2.line(mask, start, end, 200, 5) # 画粗一点
        # 关键步骤：高斯模糊！这能产生梯度，让光流算法能"抓"住它
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        # 叠加到图像上
        return cv2.add(img, mask)

    # T-1 帧
    frame_prev = draw_fuzzy_wire(frame_prev, (50, 100), (150, 100))
    
    # T 帧 (移动后)
    frame_curr = draw_fuzzy_wire(frame_curr, (50+int(dx_move), 100+int(dy_move)), (150+int(dx_move), 100+int(dy_move)))
    
    # 当前帧 Mask (用于骨架提取)
    # 我们用稍微干净一点的 Mask 传给骨架提取算法，但光流计算用模糊的原图
    mask_curr = np.zeros((h, w), dtype=np.uint8)
    cv2.line(mask_curr, (50+int(dx_move), 100+int(dy_move)), (150+int(dx_move), 100+int(dy_move)), 255, 3)

    # 2. 运行解耦算法
    shift_x, shift_y = decoupler.calculate_respiratory_offset(frame_prev, frame_curr, mask_curr)
    
    print(f"--- 运动解耦结果验证 (模糊+噪声增强版) ---")
    print(f"真实物理运动: 切向(医生操作)={dx_move}, 法向(呼吸)={dy_move}")
    print(f"算法解耦结果: Shift_X={shift_x:.4f}, Shift_Y={shift_y:.4f}")
    
    # 3. 结果分析
    # 对于水平导丝，法向量是垂直的(0,1)。
    # 理论预期：Shift_X (切向投影) 应接近 0；Shift_Y (法向投影) 应接近 3.0。
    
    if abs(shift_x) < 1.0 and abs(shift_y - dy_move) < 1.0:
        print("✅ 验证成功！算法成功滤除了医生的切向操作，只保留了呼吸引起的法向位移。")
    else:
        print(f"❌ 验证需检查：Shift_Y 误差 {abs(shift_y - dy_move):.4f}。")