import argparse
import os
import random
import sys

# ================= 硬件指定 (必须放在所有 torch 导入之前) =================
# 这里不再写死默认 GPU。
# 约定：
# - 外部若设置了 CUDA_VISIBLE_DEVICES，则严格尊重外部设置
# - 外部若未设置，则交给 PyTorch / Docker 自己决定可见设备

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --- 解决工程内包导入路径问题 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BASE_DIR 用于在 train() 内部拼接 experiments/checkpoints 等路径；
# 19.x 旧版本依赖一个全局 BASE_DIR 但没定义，这里显式补上。
BASE_DIR = PROJECT_ROOT
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from freqdec_gwnet.data.real_dataset import RealGuidewireVideoDataset
from freqdec_gwnet.losses.segmentation_losses import FAST_LiteNet_Loss, GuidewireAuxiliaryLoss, EdgeAuxiliaryLoss
from freqdec_gwnet.models.fast_litenet import FAST_LiteNet


def seed_everything(seed=42, rank=0):
    # 各个 rank 用不同种子，避免 DDP 进程拿到完全一样的随机序列。
    final_seed = seed + rank
    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)
    torch.cuda.manual_seed_all(final_seed)


def batch_dice_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def extract_seg_logits(model_output):
    """
    兼容单头和多头输出。
    - 单头: 直接返回 segmentation logits
    - 多头: 从字典里取 seg logits
    """
    if isinstance(model_output, dict):
        return model_output["seg"]
    return model_output


def parse_args():
    parser = argparse.ArgumentParser(description="Train or resume FAST_LiteNet deployment model.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/amax/data/media/rmlab_dataset/SHZ/ori_data",
        help="Dataset root containing train/val/test folders.",
    )
    parser.add_argument(
        "--exclude-sequences-file",
        type=str,
        default=None,
        help="Optional text file listing sequence names to exclude.",
    )
    parser.add_argument(
        "--augmentation-profile",
        type=str,
        default="fluoro_hard",
        choices=["none", "standard", "fluoro_hard", "fluoro_extreme"],
        help="Train-time fluoroscopy augmentation profile for clean_data/train.",
    )
    parser.add_argument(
        "--aug-data-root",
        type=str,
        default=None,
        help="Optional augmented dataset root. Only aug_data/train will be merged into training.",
    )
    parser.add_argument(
        "--aug-window-ratio",
        type=float,
        default=1.0,
        help="How much of aug_data/train to keep. 1.0 means full aug_data, 0.5 means use half of the augmented windows.",
    )
    parser.add_argument(
        "--aug-augmentation-profile",
        type=str,
        default="none",
        choices=["none", "standard", "fluoro_hard", "fluoro_extreme"],
        help="Optional online augmentation profile applied when loading aug_data/train.",
    )
    parser.add_argument(
        "--train-hflip-p",
        type=float,
        default=0.5,
        help="Horizontal flip probability used for online training augmentation.",
    )
    parser.add_argument(
        "--train-vflip-p",
        type=float,
        default=0.5,
        help="Vertical flip probability used for online training augmentation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a full training checkpoint. Defaults to experiments/checkpoints/fast_litenet_last.pth",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="Path to a full training checkpoint containing model/optimizer/scheduler/scaler state.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Load model weights only. Useful for continuing an older run that only saved .pth weights.",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="Epoch to start from when using --model-path. Example: 2 means continue from epoch 2.",
    )
    parser.add_argument(
        "--global-step",
        type=int,
        default=0,
        help="Global step to resume from when using --model-path.",
    )
    parser.add_argument(
        "--best-val-loss",
        type=float,
        default=None,
        help="Best validation loss so far when using --model-path.",
    )
    parser.add_argument(
        "--best-val-dice",
        type=float,
        default=0.0,
        help="Best validation Dice so far when using --model-path.",
    )
    parser.add_argument("--epochs", type=int, default=150, help="Total epochs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Global batch size across all GPUs. Under DDP it will be divided evenly per rank.",
    )
    parser.add_argument("--seq-len", type=int, default=5, help="Sequence length.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Total dataloader workers. Under DDP it will be divided across ranks.",
    )
    parser.add_argument(
        "--disable-persistent-workers",
        action="store_true",
        help="Disable DataLoader persistent_workers to reduce worker-related crashes.",
    )
    parser.add_argument(
        "--disable-pin-memory",
        action="store_true",
        help="Disable DataLoader pin_memory for stability debugging.",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable AMP mixed precision. Disabled by default because the FFT branch can be unstable with AMP on some CUDA/PyTorch stacks.",
    )
    parser.add_argument(
        "--width-mult",
        type=float,
        default=1.0,
        help="Width multiplier for FAST-LiteNet. 1.0=S, 1.5=M, 2.0=L is a good starting point.",
    )
    parser.add_argument(
        "--lambda-cldice",
        type=float,
        default=0.5,
        help="Weight of the clDice topology term.",
    )
    parser.add_argument(
        "--lambda-boundary",
        type=float,
        default=0.5,
        help="Weight of the boundary term.",
    )
    parser.add_argument(
        "--multi-task",
        action="store_true",
        help="Enable extra centerline and tip-proxy heads on top of the main segmentation head.",
    )
    parser.add_argument(
        "--edge-aux",
        action="store_true",
        help="Enable an auxiliary boundary head to sharpen guidewire edges.",
    )
    parser.add_argument(
        "--use-refine-head",
        action="store_true",
        help="Enable a lightweight large-kernel refinement head before the final segmentation prediction.",
    )
    parser.add_argument(
        "--refine-kernel-size",
        type=int,
        default=7,
        help="Kernel size used by the lightweight refinement head.",
    )
    parser.add_argument(
        "--lambda-centerline",
        type=float,
        default=0.2,
        help="Weight of the centerline auxiliary loss when --multi-task is enabled.",
    )
    parser.add_argument(
        "--lambda-tip",
        type=float,
        default=0.1,
        help="Weight of the tip-proxy auxiliary loss when --multi-task is enabled.",
    )
    parser.add_argument(
        "--lambda-edge",
        type=float,
        default=0.1,
        help="Weight of the boundary auxiliary loss when --edge-aux is enabled.",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default="fast_litenet",
        help="Experiment name prefix used for checkpoints and TensorBoard logs.",
    )
    parser.add_argument(
        "--enable-conf-gate",
        action="store_true",
        help="Enable confidence-gated temporal feature-bank updates.",
    )
    parser.add_argument(
        "--gate-fuse-threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for allowing temporal fusion.",
    )
    parser.add_argument(
        "--gate-freeze-threshold",
        type=float,
        default=0.45,
        help="If similarity is below this value, freeze the feature bank instead of updating it.",
    )
    parser.add_argument(
        "--gate-reinit-threshold",
        type=float,
        default=0.20,
        help="If similarity is below this value, reinitialize the feature bank from the current frame.",
    )
    parser.add_argument(
        "--bank-momentum",
        type=float,
        default=0.7,
        help="EMA momentum used when updating the temporal feature bank under high confidence.",
    )
    parser.add_argument(
        "--freq-mode",
        type=str,
        default="global",
        choices=["global", "local_sff", "ms_local_sff"],
        help="Frequency modeling mode. 'global' keeps the original FFT residual, 'local_sff' enables single-scale local spatial-frequency fusion, and 'ms_local_sff' enables multi-scale local spatial-frequency fusion.",
    )
    parser.add_argument(
        "--local-window-size",
        type=int,
        default=4,
        help="Local FFT window size used when --freq-mode local_sff.",
    )
    parser.add_argument(
        "--local-window-sizes",
        type=int,
        nargs="+",
        default=[4, 8],
        help="Window sizes used when --freq-mode ms_local_sff. Example: --local-window-sizes 4 8",
    )
    parser.add_argument(
        "--freq-gate-type",
        type=str,
        default="none",
        choices=["none", "channel"],
        help="Optional channel gate for the FFT residual branch. Use 'channel' to make frequency enhancement more selective.",
    )
    parser.add_argument(
        "--freq-gate-reduction",
        type=int,
        default=4,
        help="Reduction ratio used by the FFT channel gate. Smaller values mean a slightly stronger gate.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable DistributedDataParallel. Usually torchrun will auto-enable this via WORLD_SIZE.",
    )
    parser.add_argument(
        "--sync-bn",
        action="store_true",
        help="Convert BatchNorm to SyncBatchNorm when using DDP.",
    )
    parser.add_argument(
        "--max-train-iters",
        type=int,
        default=0,
        help="If > 0, stop the training loop after this many batches per epoch. "
             "Use for fast sanity checks (e.g. --max-train-iters 2).",
    )
    parser.add_argument(
        "--max-val-iters",
        type=int,
        default=0,
        help="If > 0, stop validation after this many batches per epoch. "
             "Pair with --max-train-iters for end-to-end sanity runs.",
    )
    return parser.parse_args()


def is_dist_ready():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return (not is_dist_ready()) or dist.get_rank() == 0


def print0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def get_model_to_save(model):
    return model.module if hasattr(model, "module") else model


def strip_module_prefix(state_dict):
    """
    兼容 DataParallel / DDP 保存出来的 module.* 前缀。
    """
    if not state_dict:
        return state_dict
    if all(not key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def remap_fast_litenet_state_dict_for_current_model(state_dict, model_state_dict):
    """
    兼容 FAST-LiteNet 从单头版本迁移到多任务版本的权重命名变化。

    以前:
      final_up = [Upsample, Conv2d]
      卷积权重保存在 final_up.1.*

    现在:
      final_up 只负责上采样
      分割头改成 seg_head.*

    所以这里会把旧 checkpoint 的 final_up.1.{weight,bias}
    自动映射到新模型的 seg_head.{weight,bias}。
    """
    state_dict = strip_module_prefix(state_dict)
    remapped = dict(state_dict)

    legacy_to_new = {
        "final_up.1.weight": "seg_head.weight",
        "final_up.1.bias": "seg_head.bias",
    }
    new_to_legacy = {
        "seg_head.weight": "final_up.1.weight",
        "seg_head.bias": "final_up.1.bias",
    }

    for old_key, new_key in legacy_to_new.items():
        if old_key in remapped and new_key in model_state_dict and new_key not in remapped:
            remapped[new_key] = remapped.pop(old_key)

    for new_key, old_key in new_to_legacy.items():
        if new_key in remapped and old_key in model_state_dict and old_key not in remapped:
            remapped[old_key] = remapped.pop(new_key)

    return remapped


def filter_state_dict_by_shape(state_dict, model_state_dict, log_fn=print0):
    """
    过滤掉“名字能对上但张量尺寸已经不兼容”的权重。

    典型场景：
    - 从 global FFT 分支切到 local_sff / ms_local_sff
    - 调整 local window size
    - 修改 refine / auxiliary 结构后继续用旧权重 warm start

    这样可以尽量复用 backbone / decoder 等稳定部分的参数，
    只让真正结构变了的模块从头初始化。
    """
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state_dict:
            filtered[key] = value
            continue
        if model_state_dict[key].shape == value.shape:
            filtered[key] = value
        else:
            skipped.append((key, tuple(value.shape), tuple(model_state_dict[key].shape)))

    if skipped:
        log_fn("ℹ️ Shape-mismatched keys skipped during warm start:")
        for key, old_shape, new_shape in skipped:
            log_fn(f"   - {key}: ckpt{old_shape} -> model{new_shape}")
    return filtered


def setup_distributed(args):
    # torchrun 会注入 WORLD_SIZE / RANK / LOCAL_RANK；有这些环境变量就自动切 DDP。
    args.distributed = args.distributed or int(os.environ.get("WORLD_SIZE", "1")) > 1
    args.rank = 0
    args.local_rank = 0
    args.world_size = 1

    if not args.distributed:
        return args

    if not torch.cuda.is_available():
        raise RuntimeError("DDP 训练需要 CUDA 环境。")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError("检测到 --distributed，但缺少 torchrun 注入的 RANK/WORLD_SIZE/LOCAL_RANK。")

    args.rank = int(os.environ["RANK"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return args


def cleanup_distributed():
    if is_dist_ready():
        dist.barrier()
        dist.destroy_process_group()


def split_global_batch_size(global_value, world_size, field_name):
    if world_size <= 1:
        return global_value
    if global_value < world_size:
        raise ValueError(f"{field_name}={global_value} 小于 world_size={world_size}，无法均分到每张卡。")
    if global_value % world_size != 0:
        raise ValueError(f"{field_name}={global_value} 不能被 world_size={world_size} 整除，请改成可整除的数。")
    return global_value // world_size


def split_num_workers(total_workers, world_size):
    if world_size <= 1:
        return total_workers
    if total_workers <= 0:
        return 0
    return max(1, total_workers // world_size)


def sync_metric_sums(values, device):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if is_dist_ready():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def save_resume_checkpoint(path, model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, best_val_dice):
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "best_val_dice": best_val_dice,
        "model_state_dict": get_model_to_save(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    torch.save(checkpoint, path)


def load_training_state(path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model_to_load = get_model_to_save(model)
    model_state_dict = model_to_load.state_dict()

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        loaded_state_dict = remap_fast_litenet_state_dict_for_current_model(
            checkpoint["model_state_dict"],
            model_state_dict,
        )
        loaded_state_dict = filter_state_dict_by_shape(loaded_state_dict, model_state_dict, log_fn=print0)
        missing_keys, unexpected_keys = model_to_load.load_state_dict(loaded_state_dict, strict=False)
        if missing_keys:
            print0(f"ℹ️ Resume load: missing keys initialized from scratch: {missing_keys}")
        if unexpected_keys:
            print0(f"ℹ️ Resume load: unexpected checkpoint keys ignored: {unexpected_keys}")
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        return {
            "start_epoch": checkpoint.get("epoch", 0) + 1,
            "global_step": checkpoint.get("global_step", 0),
            "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
            "best_val_dice": checkpoint.get("best_val_dice", 0.0),
            "mode": "full",
        }

    loaded_state_dict = remap_fast_litenet_state_dict_for_current_model(checkpoint, model_state_dict)
    loaded_state_dict = filter_state_dict_by_shape(loaded_state_dict, model_state_dict, log_fn=print0)
    missing_keys, unexpected_keys = model_to_load.load_state_dict(loaded_state_dict, strict=False)
    if missing_keys:
        print0(f"ℹ️ Partial weight loading: missing keys initialized from scratch: {missing_keys}")
    if unexpected_keys:
        print0(f"ℹ️ Partial weight loading: unexpected checkpoint keys ignored: {unexpected_keys}")
    return {
        "start_epoch": 1,
        "global_step": 0,
        "best_val_loss": float("inf"),
        "best_val_dice": 0.0,
        "mode": "weights_only",
    }


def train(args):
    args = setup_distributed(args)

    EPOCHS = args.epochs
    SEQ_LEN = args.seq_len
    INITIAL_LR = args.lr
    WEIGHT_DECAY = args.weight_decay

    per_rank_batch_size = split_global_batch_size(args.batch_size, args.world_size, "batch_size")
    per_rank_workers = split_num_workers(args.num_workers, args.world_size)
    pin_memory = (not args.disable_pin_memory) and torch.cuda.is_available()
    persistent_workers = (not args.disable_persistent_workers) and per_rank_workers > 0

    DATA_ROOT = args.data_root
    SAVE_DIR = os.path.join(BASE_DIR, "experiments", "checkpoints")
    LOG_DIR = os.path.join(BASE_DIR, "experiments", "logs", args.save_prefix)
    LAST_RESUME_PATH = os.path.join(SAVE_DIR, f"{args.save_prefix}_last.pth")
    BEST_RESUME_PATH = os.path.join(SAVE_DIR, f"{args.save_prefix}_best_resume.pth")
    BEST_WEIGHT_PATH = os.path.join(SAVE_DIR, f"{args.save_prefix}_best.pth")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    seed_everything(42, rank=args.rank)
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        if args.distributed:
            device = torch.device("cuda", args.local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print0(f"🔥 Starting Training... backend={device}")
    print0(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    if args.distributed:
        print0(
            f"DDP enabled | world_size={args.world_size} | global_batch={args.batch_size} | "
            f"per_rank_batch={per_rank_batch_size} | per_rank_workers={per_rank_workers}"
        )
    print0(
        f"Model config | save_prefix={args.save_prefix} | width_mult={args.width_mult:.2f} | "
        f"lambda_clDice={args.lambda_cldice:.3f} | lambda_boundary={args.lambda_boundary:.3f} | "
        f"multi_task={args.multi_task} | refine_head={args.use_refine_head} | refine_k={args.refine_kernel_size} | "
        f"lambda_centerline={args.lambda_centerline:.3f} | "
        f"lambda_tip={args.lambda_tip:.3f} | conf_gate={args.enable_conf_gate} | "
        f"fuse={args.gate_fuse_threshold:.2f} | freeze={args.gate_freeze_threshold:.2f} | "
        f"reinit={args.gate_reinit_threshold:.2f} | bank_momentum={args.bank_momentum:.2f} | "
        f"freq_mode={args.freq_mode} | local_ws={args.local_window_size} | local_wss={args.local_window_sizes} | "
        f"freq_gate={args.freq_gate_type} | freq_red={args.freq_gate_reduction} | amp={args.amp} | "
        f"pin_memory={pin_memory} | persistent_workers={persistent_workers} | "
        f"hflip={args.train_hflip_p:.2f} | vflip={args.train_vflip_p:.2f} | aug_ratio={args.aug_window_ratio:.2f}"
    )

    print0("Initializing Real Guidewire Video Datasets...")
    clean_train_dataset = RealGuidewireVideoDataset(
        data_root=DATA_ROOT,
        split="train",
        seq_len=SEQ_LEN,
        exclude_sequences_file=args.exclude_sequences_file,
        augmentation_profile=args.augmentation_profile,
        hflip_p=args.train_hflip_p,
        vflip_p=args.train_vflip_p,
    )
    train_dataset = clean_train_dataset
    if args.aug_data_root:
        # aug_data 只并入训练集，验证和测试始终保持真实 clean_data。
        aug_train_dataset = RealGuidewireVideoDataset(
            data_root=args.aug_data_root,
            split="train",
            seq_len=SEQ_LEN,
            exclude_sequences_file=None,
            augmentation_profile=args.aug_augmentation_profile,
            hflip_p=args.train_hflip_p,
            vflip_p=args.train_vflip_p,
        )
        if not (0.0 < args.aug_window_ratio <= 1.0):
            raise ValueError("--aug-window-ratio 必须在 (0, 1] 范围内。")
        if args.aug_window_ratio < 1.0:
            keep_count = max(1, int(len(aug_train_dataset) * args.aug_window_ratio))
            rng = np.random.default_rng(42)
            keep_indices = sorted(rng.choice(len(aug_train_dataset), size=keep_count, replace=False).tolist())
            aug_train_dataset = Subset(aug_train_dataset, keep_indices)
        train_dataset = ConcatDataset([clean_train_dataset, aug_train_dataset])
        print0(
            f"📦 Merged clean_data + aug_data | clean_windows={len(clean_train_dataset)} "
            f"+ aug_windows={len(aug_train_dataset)} (ratio={args.aug_window_ratio:.2f}) = total={len(train_dataset)}"
        )

    val_dataset = RealGuidewireVideoDataset(
        data_root=DATA_ROOT,
        split="val",
        seq_len=SEQ_LEN,
        exclude_sequences_file=args.exclude_sequences_file,
        augmentation_profile=args.augmentation_profile,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("训练集或验证集为空，请先检查 clean_data/train 和 clean_data/val。")

    train_sampler = None
    val_sampler = None
    if args.distributed:
        # DDP 下每个 rank 只看自己那份样本，避免三张卡重复吃同一批数据。
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
            drop_last=True,
        )

    multi_gpu_fallback = (not args.distributed) and torch.cuda.device_count() > 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_rank_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=per_rank_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        # 非 DDP 的 DataParallel 下，最后一个 batch 太小会导致副卡拿不到输入。
        drop_last=multi_gpu_fallback,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=per_rank_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=per_rank_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=multi_gpu_fallback,
    )

    model = FAST_LiteNet(
        in_channels=1,
        num_classes=1,
        width_mult=args.width_mult,
        multi_task=args.multi_task,
        edge_aux=args.edge_aux,
        use_refine_head=args.use_refine_head,
        refine_kernel_size=args.refine_kernel_size,
        freq_mode=args.freq_mode,
        local_window_size=args.local_window_size,
        local_window_sizes=tuple(args.local_window_sizes),
        enable_conf_gate=args.enable_conf_gate,
        gate_fuse_threshold=args.gate_fuse_threshold,
        gate_freeze_threshold=args.gate_freeze_threshold,
        gate_reinit_threshold=args.gate_reinit_threshold,
        bank_momentum=args.bank_momentum,
        freq_gate_type=args.freq_gate_type,
        freq_gate_reduction=args.freq_gate_reduction,
    ).to(device)
    if args.distributed and args.sync_bn and device.type == "cuda":
        # 每张卡上 batch 很小时，同步 BN 会更稳。
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print0("SyncBatchNorm enabled.")

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        print0(f"🚀 DDP is using {args.world_size} visible GPUs.")
    elif torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print0(f"🚀 DataParallel fallback is using {torch.cuda.device_count()} visible GPUs.")

    criterion = FAST_LiteNet_Loss(lambda_1=args.lambda_cldice, lambda_2=args.lambda_boundary).to(device)
    aux_criterion = GuidewireAuxiliaryLoss().to(device) if args.multi_task else None
    edge_aux_criterion = EdgeAuxiliaryLoss().to(device) if args.edge_aux else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # FFT + autocast 在部分 CUDA/PyTorch 组合上会触发不稳定的内部错误，
    # 默认关闭 AMP，只有显式传 --amp 时才启用混合精度。
    use_amp = device.type == "cuda" and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    writer = SummaryWriter(log_dir=LOG_DIR) if is_main_process() else None

    best_val_loss = float("inf")
    best_val_dice = 0.0
    global_step = 0
    start_epoch = 1

    resume_path = args.resume_path or LAST_RESUME_PATH
    if args.resume and os.path.exists(resume_path):
        state = load_training_state(
            resume_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        start_epoch = state["start_epoch"]
        global_step = state["global_step"]
        best_val_loss = state["best_val_loss"]
        best_val_dice = state["best_val_dice"]
        print0(
            f"🔁 Resumed full training state from {resume_path} | "
            f"start_epoch={start_epoch}, global_step={global_step}, best_val_dice={best_val_dice:.4f}"
        )
    elif args.model_path:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"找不到模型权重: {args.model_path}")
        _ = load_training_state(args.model_path, model, device=device)
        start_epoch = args.start_epoch
        global_step = args.global_step
        best_val_loss = args.best_val_loss if args.best_val_loss is not None else float("inf")
        best_val_dice = args.best_val_dice
        print0(
            f"🔁 Loaded model weights from {args.model_path} | "
            f"start_epoch={start_epoch}, global_step={global_step}, best_val_dice={best_val_dice:.4f}"
        )

    for epoch in range(start_epoch, EPOCHS + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        model.train()
        train_epoch_loss = 0.0
        train_epoch_dice = 0.0
        train_epoch_batches = 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]") if is_main_process() else train_loader

        for batch_idx, (images_seq, masks_seq) in enumerate(pbar_train):
            if args.max_train_iters > 0 and batch_idx >= args.max_train_iters:
                # sanity-check 早退：跳出训练循环但仍走完后面的 val + 保存逻辑
                break
            images_seq = images_seq.to(device, non_blocking=True)
            masks_seq = masks_seq.to(device, non_blocking=True)
            _, T, _, _, _ = images_seq.shape

            optimizer.zero_grad(set_to_none=True)
            feature_bank = None
            batch_total_loss = 0.0
            batch_total_dice = 0.0
            batch_centerline_loss = 0.0
            batch_tip_loss = 0.0
            batch_edge_loss = 0.0

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                # 当前实现是逐帧递推，所以一个 batch 内会做 T 次前向。
                for t in range(T):
                    img_t = images_seq[:, t, ...]
                    mask_t = masks_seq[:, t, ...]
                    reset_flag = t == 0

                    pred_t, feature_bank = model(img_t, feature_bank=feature_bank, reset_flag=reset_flag)
                    seg_logits = extract_seg_logits(pred_t)
                    loss_t, _, _, _ = criterion(seg_logits, mask_t)
                    if args.multi_task:
                        centerline_loss_t, tip_loss_t, _, _ = aux_criterion(
                            pred_t["centerline"],
                            pred_t["tip"],
                            mask_t,
                        )
                        loss_t = loss_t + args.lambda_centerline * centerline_loss_t + args.lambda_tip * tip_loss_t
                        batch_centerline_loss += centerline_loss_t
                        batch_tip_loss += tip_loss_t
                    if args.edge_aux:
                        edge_loss_t, _ = edge_aux_criterion(pred_t["edge"], mask_t)
                        loss_t = loss_t + args.lambda_edge * edge_loss_t
                        batch_edge_loss += edge_loss_t
                    batch_total_loss += loss_t
                    batch_total_dice += batch_dice_from_logits(seg_logits.detach(), mask_t)

                batch_total_loss = batch_total_loss / T
                batch_total_dice = batch_total_dice / T
                if args.multi_task:
                    batch_centerline_loss = batch_centerline_loss / T
                    batch_tip_loss = batch_tip_loss / T
                if args.edge_aux:
                    batch_edge_loss = batch_edge_loss / T

            scaler.scale(batch_total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_epoch_loss += batch_total_loss.item()
            train_epoch_dice += batch_total_dice.item()
            train_epoch_batches += 1.0
            if is_main_process():
                global_step += 1
                postfix = {"Loss": f"{batch_total_loss.item():.4f}", "Dice": f"{batch_total_dice.item():.4f}"}
                if args.multi_task:
                    postfix["Ctr"] = f"{batch_centerline_loss.item():.4f}"
                    postfix["Tip"] = f"{batch_tip_loss.item():.4f}"
                if args.edge_aux:
                    postfix["Edge"] = f"{batch_edge_loss.item():.4f}"
                pbar_train.set_postfix(postfix)
                writer.add_scalar("Train/Loss", batch_total_loss.item(), global_step)
                writer.add_scalar("Train/Dice", batch_total_dice.item(), global_step)
                if args.multi_task:
                    writer.add_scalar("Train/Centerline_Loss", batch_centerline_loss.item(), global_step)
                    writer.add_scalar("Train/Tip_Loss", batch_tip_loss.item(), global_step)
                if args.edge_aux:
                    writer.add_scalar("Train/Edge_Loss", batch_edge_loss.item(), global_step)
                writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)

        train_sum_loss, train_sum_dice, train_sum_batches = sync_metric_sums(
            [train_epoch_loss, train_epoch_dice, train_epoch_batches], device=device
        )
        avg_train_loss = train_sum_loss / max(train_sum_batches, 1.0)
        avg_train_dice = train_sum_dice / max(train_sum_batches, 1.0)

        model.eval()
        val_epoch_loss = 0.0
        val_cldice_loss = 0.0
        val_epoch_dice = 0.0
        val_epoch_batches = 0.0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]") if is_main_process() else val_loader

        with torch.no_grad():
            for batch_idx, (images_seq, masks_seq) in enumerate(pbar_val):
                if args.max_val_iters > 0 and batch_idx >= args.max_val_iters:
                    break
                images_seq = images_seq.to(device, non_blocking=True)
                masks_seq = masks_seq.to(device, non_blocking=True)
                _, T, _, _, _ = images_seq.shape
                feature_bank = None
                batch_val_loss = 0.0
                batch_cldice = 0.0
                batch_val_dice = 0.0
                batch_val_centerline = 0.0
                batch_val_tip = 0.0
                batch_val_edge = 0.0

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    for t in range(T):
                        img_t = images_seq[:, t, ...]
                        mask_t = masks_seq[:, t, ...]
                        reset_flag = t == 0

                        pred_t, feature_bank = model(img_t, feature_bank=feature_bank, reset_flag=reset_flag)
                        seg_logits = extract_seg_logits(pred_t)
                        loss_t, _, cl_loss, _ = criterion(seg_logits, mask_t)
                        if args.multi_task:
                            centerline_loss_t, tip_loss_t, _, _ = aux_criterion(
                                pred_t["centerline"],
                                pred_t["tip"],
                                mask_t,
                            )
                            loss_t = loss_t + args.lambda_centerline * centerline_loss_t + args.lambda_tip * tip_loss_t
                            batch_val_centerline += centerline_loss_t
                            batch_val_tip += tip_loss_t
                        if args.edge_aux:
                            edge_loss_t, _ = edge_aux_criterion(pred_t["edge"], mask_t)
                            loss_t = loss_t + args.lambda_edge * edge_loss_t
                            batch_val_edge += edge_loss_t
                        batch_val_loss += loss_t
                        batch_cldice += cl_loss
                        batch_val_dice += batch_dice_from_logits(seg_logits, mask_t)

                    batch_val_loss = batch_val_loss / T
                    batch_cldice = batch_cldice / T
                    batch_val_dice = batch_val_dice / T
                    if args.multi_task:
                        batch_val_centerline = batch_val_centerline / T
                        batch_val_tip = batch_val_tip / T
                    if args.edge_aux:
                        batch_val_edge = batch_val_edge / T

                val_epoch_loss += batch_val_loss.item()
                val_cldice_loss += batch_cldice.item()
                val_epoch_dice += batch_val_dice.item()
                val_epoch_batches += 1.0
                if is_main_process():
                    postfix = {"Val Loss": f"{batch_val_loss.item():.4f}", "Val Dice": f"{batch_val_dice.item():.4f}"}
                    if args.multi_task:
                        postfix["ValCtr"] = f"{batch_val_centerline.item():.4f}"
                        postfix["ValTip"] = f"{batch_val_tip.item():.4f}"
                    if args.edge_aux:
                        postfix["ValEdge"] = f"{batch_val_edge.item():.4f}"
                    pbar_val.set_postfix(postfix)

        val_sum_loss, val_sum_cldice, val_sum_dice, val_sum_batches = sync_metric_sums(
            [val_epoch_loss, val_cldice_loss, val_epoch_dice, val_epoch_batches], device=device
        )
        avg_val_loss = val_sum_loss / max(val_sum_batches, 1.0)
        avg_val_cldice = val_sum_cldice / max(val_sum_batches, 1.0)
        avg_val_dice = val_sum_dice / max(val_sum_batches, 1.0)

        if writer is not None:
            writer.add_scalar("Val/Loss", avg_val_loss, epoch)
            writer.add_scalar("Val/clDice_Loss", avg_val_cldice, epoch)
            writer.add_scalar("Val/Dice", avg_val_dice, epoch)

        print0(
            f"🏁 Epoch {epoch} Summary: "
            f"Train Loss: {avg_train_loss:.4f} | Train Dice: {avg_train_dice:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f} | Val clDice Loss: {avg_val_cldice:.4f}"
        )

        scheduler.step()

        if is_main_process() and avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_val_loss = avg_val_loss
            model_to_save = get_model_to_save(model)
            torch.save(model_to_save.state_dict(), BEST_WEIGHT_PATH)
            save_resume_checkpoint(
                BEST_RESUME_PATH,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                best_val_loss,
                best_val_dice,
            )
            print0(f"🌟 New Best Model saved! Val Dice improved to {best_val_dice:.4f} (Val Loss {best_val_loss:.4f})")

        if is_main_process():
            save_resume_checkpoint(
                LAST_RESUME_PATH,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                best_val_loss,
                best_val_dice,
            )

            if epoch % 20 == 0:
                model_to_save = get_model_to_save(model)
                torch.save(model_to_save.state_dict(), os.path.join(SAVE_DIR, f"{args.save_prefix}_epoch_{epoch}.pth"))

    if writer is not None:
        writer.close()
    print0("🎉 Phase 3 completed! Your model has fully converged on real datasets!")


if __name__ == "__main__":
    try:
        train(parse_args())
    finally:
        cleanup_distributed()
