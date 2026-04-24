import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_from_logits(logits, targets, smooth=1e-5):
    """
    通用 Dice Loss，既能给主分割头用，也能给 centerline 这种辅助头用。
    """
    probs = torch.sigmoid(logits)
    intersection = torch.sum(probs * targets)
    return 1.0 - (2.0 * intersection + smooth) / (torch.sum(probs) + torch.sum(targets) + smooth)


def soft_erode(img):
    """可微的形态学腐蚀 (使用 Min Pooling 的等效实现)"""
    p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
    p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
    return torch.min(p1, p2)

def soft_dilate(img):
    """可微的形态学膨胀 (使用 Max Pooling 的等效实现)"""
    return F.max_pool2d(img, (3,3), (1,1), (1,1))

def soft_open(img):
    """可微的形态学开运算"""
    return soft_dilate(soft_erode(img))

def soft_skel(img, iter_=15):
    """
    软骨架化提取核心算法
    通过迭代地求原图与开运算图的差值来逼近骨架
    """
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel

class Soft_clDiceLoss(nn.Module):
    def __init__(self, iter_=15, smooth=1e-5):
        super(Soft_clDiceLoss, self).__init__()
        self.iter_ = iter_
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        """
        y_pred: 网络的输出概率图 (经过 Sigmoid) Shape: [B, C, H, W]
        y_true: Ground Truth Mask, Shape: [B, C, H, W]
        """
        # 1. 提取预测图和真实图的软骨架
        skel_pred = soft_skel(y_pred, self.iter_)
        skel_true = soft_skel(y_true, self.iter_)
        
        # 2. 计算拓扑精确度 (Topology Precision)
        tprec = (torch.sum(torch.mul(skel_pred, y_true)) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        
        # 3. 计算拓扑敏感度 (Topology Sensitivity)
        tsens = (torch.sum(torch.mul(skel_true, y_pred)) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        
        # 4. 计算 clDice 调和平均数并转为 Loss
        cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens)
        return 1.0 - cl_dice
class BoundaryLoss(nn.Module):
    """
    边界感知损失 (Boundary Loss)
    通过形态学膨胀和腐蚀计算预测和真实 Mask 的边界，促使网络关注边缘细节
    """
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # 1. 计算真实 Mask 的边界 (膨胀 - 腐蚀)
        true_dilate = F.max_pool2d(y_true, kernel_size=3, stride=1, padding=1)
        true_erode = -F.max_pool2d(-y_true, kernel_size=3, stride=1, padding=1)
        true_boundary = true_dilate - true_erode
        
        # 2. 计算预测 Mask 的边界
        pred_dilate = F.max_pool2d(y_pred, kernel_size=3, stride=1, padding=1)
        pred_erode = -F.max_pool2d(-y_pred, kernel_size=3, stride=1, padding=1)
        pred_boundary = pred_dilate - pred_erode
        
        # 3. 计算边界区域的均方误差 (MSE)
        boundary_loss = F.mse_loss(pred_boundary, true_boundary)
        return boundary_loss


def build_boundary_target(mask):
    """
    从 GT mask 直接生成边界监督。
    和 centerline/tip 不同，这个监督与主分割目标更一致，噪声也更小。
    """
    true_dilate = F.max_pool2d(mask.float(), kernel_size=3, stride=1, padding=1)
    true_erode = -F.max_pool2d(-mask.float(), kernel_size=3, stride=1, padding=1)
    boundary = (true_dilate - true_erode).clamp(0.0, 1.0)
    return (boundary > 0.0).float()


class FAST_LiteNet_Loss(nn.Module):
    """
    按照技术路线 2.0 组装的终极混合 Loss
    L_total = L_Dice + lambda_1 * L_clDice + lambda_2 * L_Boundary
    """
    def __init__(self, lambda_1=0.5, lambda_2=0.5):
        super(FAST_LiteNet_Loss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.clDice_loss = Soft_clDiceLoss(iter_=15)
        self.boundary_loss = BoundaryLoss()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def forward(self, y_pred, y_true):
        # 训练时直接输入 logits，先转概率再计算形态学相关项
        y_prob = torch.sigmoid(y_pred)

        # 1. BCEWithLogits 保证前期收敛稳定
        bce_loss = self.bce_loss(y_pred, y_true)

        # 2. 基础 Dice Loss (为了面积重叠)
        smooth = 1e-5
        intersection = torch.sum(y_prob * y_true)
        dice_loss = 1.0 - (2. * intersection + smooth) / (torch.sum(y_prob) + torch.sum(y_true) + smooth)
        
        # 3. 拓扑连续性损失 (防断裂，防变短)
        cl_loss = self.clDice_loss(y_prob, y_true)
        
        # 4. 边界感知损失 (强化锐利边缘)
        bound_loss = self.boundary_loss(y_prob, y_true)
        
        # 5. 混合相加
        total_loss = bce_loss + dice_loss + self.lambda_1 * cl_loss + self.lambda_2 * bound_loss
        return total_loss, dice_loss, cl_loss, bound_loss


def build_centerline_target(mask, iter_=15):
    """
    用 GT mask 自身生成 soft centerline target。
    这里直接复用 soft skeleton，省掉额外的离线骨架标注步骤。
    """
    centerline = soft_skel(mask.float(), iter_=iter_)
    normalizer = centerline.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (centerline / normalizer).clamp(0.0, 1.0)


def _gaussian_kernel2d(kernel_size=9, sigma=2.0, device="cpu", dtype=torch.float32):
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum().clamp_min(1e-6)
    return kernel.view(1, 1, kernel_size, kernel_size)


def build_tip_heatmap_target(mask, iter_=15, threshold=0.1, kernel_size=9, sigma=2.0):
    """
    当前数据没有真实 tip 标注，这里先从骨架端点生成一个 tip-proxy heatmap。
    论文里可以诚实写成 endpoint supervision / tip proxy supervision。
    """
    centerline = build_centerline_target(mask, iter_=iter_)
    binary_centerline = (centerline > threshold).float()

    # 端点检测：骨架像素在 8 邻域里只有 1 个相邻像素时，认为它是一个端点。
    neighbor_kernel = torch.ones((1, 1, 3, 3), device=mask.device, dtype=mask.dtype)
    neighbor_kernel[:, :, 1, 1] = 0.0
    neighbor_count = F.conv2d(binary_centerline, neighbor_kernel, padding=1)
    endpoints = binary_centerline * ((neighbor_count > 0.0) & (neighbor_count <= 1.5)).float()

    # 极少数情况下 skeleton 没有稳定端点，就退化到局部极大值作为弱监督，避免整张图都是 0。
    empty_mask = endpoints.sum(dim=(1, 2, 3), keepdim=True) <= 0
    if empty_mask.any():
        local_max = (centerline == F.max_pool2d(centerline, kernel_size=3, stride=1, padding=1)).float()
        fallback = local_max * binary_centerline
        endpoints = torch.where(empty_mask, fallback, endpoints)

    blur_kernel = _gaussian_kernel2d(
        kernel_size=kernel_size,
        sigma=sigma,
        device=mask.device,
        dtype=mask.dtype,
    )
    tip_heatmap = F.conv2d(endpoints, blur_kernel, padding=kernel_size // 2)
    normalizer = tip_heatmap.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (tip_heatmap / normalizer).clamp(0.0, 1.0)


class EdgeAuxiliaryLoss(nn.Module):
    """
    边界辅助监督：直接让网络学会“哪里是导丝边缘”。
    这个任务比 tip-proxy 更贴近主分割，因此更适合作为下一步结构优化。
    """

    def __init__(self):
        super(EdgeAuxiliaryLoss, self).__init__()
        self.edge_bce = nn.BCEWithLogitsLoss()

    def forward(self, edge_logits, y_true):
        edge_target = build_boundary_target(y_true).detach()
        edge_loss = self.edge_bce(edge_logits, edge_target)
        edge_loss = edge_loss + dice_loss_from_logits(edge_logits, edge_target)
        return edge_loss, edge_target


class GuidewireAuxiliaryLoss(nn.Module):
    """
    辅助监督：
    - centerline: 让网络学会“线在哪”
    - tip-proxy(endpoint): 让网络学会“线的末端在哪”
    """

    def __init__(self, centerline_iter=15, tip_threshold=0.1, tip_kernel_size=9, tip_sigma=2.0):
        super(GuidewireAuxiliaryLoss, self).__init__()
        self.centerline_iter = centerline_iter
        self.tip_threshold = tip_threshold
        self.tip_kernel_size = tip_kernel_size
        self.tip_sigma = tip_sigma
        self.centerline_bce = nn.BCEWithLogitsLoss()
        self.tip_bce = nn.BCEWithLogitsLoss()

    def forward(self, centerline_logits, tip_logits, y_true):
        centerline_target = build_centerline_target(y_true, iter_=self.centerline_iter).detach()
        tip_target = build_tip_heatmap_target(
            y_true,
            iter_=self.centerline_iter,
            threshold=self.tip_threshold,
            kernel_size=self.tip_kernel_size,
            sigma=self.tip_sigma,
        ).detach()

        centerline_loss = self.centerline_bce(centerline_logits, centerline_target)
        centerline_loss = centerline_loss + dice_loss_from_logits(centerline_logits, centerline_target)
        tip_loss = self.tip_bce(tip_logits, tip_target)

        return centerline_loss, tip_loss, centerline_target, tip_target

# ================= 终极 Loss 测试代码 (请替换原来的测试部分) =================
if __name__ == "__main__":
    # 模拟 [Batch=1, Channel=1, H=64, W=64] 的输出
    y_true = torch.zeros(1, 1, 64, 64)
    y_true[0, 0, 10:54, 31:34] = 1.0 
    
    # 稍微带有噪声和偏差的预测
    y_pred = y_true * 0.9 
    
    criterion = FAST_LiteNet_Loss(lambda_1=0.5, lambda_2=0.5)
    
    total, d_loss, cl_loss, b_loss = criterion(y_pred, y_true)
    
    print(f"Total Loss: {total.item():.4f}")
    print(f"  |- Dice Loss: {d_loss.item():.4f}")
    print(f"  |- clDice Loss: {cl_loss.item():.4f}")
    print(f"  |- Boundary Loss: {b_loss.item():.4f}")
