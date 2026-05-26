#!/usr/bin/env python
"""Stage 3 — joint training of GlobalStateBranch + RelativeMotionField.

Default training protocol (PROJECT_CONSTRAINTS_19.6.md §1 R3 + R4 + §5):

    * R1 (encoder + sta_module + decoder + seg_head) stays frozen — its
      weights came from stage 1 + stage 2 and we do not perturb them.
    * R2 (state_branch) and R3 (motion_field) train jointly.
    * Loss = ``λ_state · L_state + λ_motion · L_motion``.
    * L_motion default detach policy: ``detach_prev = detach_curr = True``,
      ``detach_state_input = False`` (the third is enforced inside
      :class:`RelativeMotionField`). This is the §5 main configuration —
      motion loss optimizes the state-conditioned mapping but does not
      flow gradient into encoder feature representation. CLI flags expose
      the feature-adaptive ablation.

Inputs:
    * data_root with split-then-kind layout (e.g. /workspace/shz/clean_data)
    * pre-generated state labels (Phase B output)
    * stage-2 checkpoint to seed state_branch and frozen R1

Outputs:
    * ``experiments/checkpoints/{save_prefix}_last.pth``
    * ``experiments/checkpoints/{save_prefix}_best.pth``
    * ``experiments/logs/{save_prefix}/`` (TensorBoard)

Usage::

    python scripts/train_stage3_joint.py \\
        --data-root /workspace/shz/clean_data \\
        --state-labels-dir /workspace/FreqDec-GWNet/reports/state_labels \\
        --stage2-ckpt experiments/checkpoints/stage2_state_best.pth \\
        --epochs 30 --batch-size 1 --T-window 64
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from freqdec_gwnet.data import FluoroSequenceWindowDataset            # noqa: E402
from freqdec_gwnet.losses.motion_losses import MotionLoss             # noqa: E402
from freqdec_gwnet.losses.state_losses import StateLoss               # noqa: E402
from freqdec_gwnet.models.freqdec_gwnet import FreqDecGWNet           # noqa: E402


# ===========================================================================
# CLI
# ===========================================================================


def parse_args():
    p = argparse.ArgumentParser(description="Stage 3: joint state + motion.")
    # Data
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--state-labels-dir", required=True, type=Path)
    p.add_argument("--exclude-sequences-file", type=str, default=None)
    # Pooled split (mirrors train_stage2_state.py): use one virtual split
    # dir and pass explicit train/val sequence lists. Enables training on
    # the expanded 137-seq pool.
    p.add_argument("--cv-split", type=str, default=None,
                   help="virtual split name under data_root (e.g. 'expanded_pool')")
    p.add_argument("--cv-train-file", type=str, default=None)
    p.add_argument("--cv-val-file", type=str, default=None)
    p.add_argument("--T-window", type=int, default=64)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--img-size", type=int, default=512)

    # Stage seeds
    p.add_argument("--stage2-ckpt", type=Path, default=None,
                   help="stage-2 checkpoint with state_branch trained")
    p.add_argument("--stage1-ckpt", type=Path, default=None,
                   help="optional fallback if no stage-2 ckpt is given; "
                        "loads R1 weights from stage 1 only")

    # Model
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--state-core-type", choices=("gru", "mamba"), default="gru")
    p.add_argument("--state-proj-dim", type=int, default=128)
    p.add_argument("--state-hidden-dim", type=int, default=64)
    p.add_argument(
        "--motion-field-type",
        choices=("relative", "absolute"),
        default="relative",
        help="§6.1 main ablation #1: 'absolute' uses G_motion(s_t) baseline "
             "(forbidden as main method by R3, allowed only as ablation)",
    )
    p.add_argument(
        "--motion-state-dim", type=int, default=3,
        help="§8.2.1 appendix ablation: trim G_delta input by reducing the "
             "state slice. 3 (default, 6D/9D) | 2 (4D, drops amplitude)",
    )
    p.add_argument(
        "--motion-input-mode",
        choices=("delta", "with_prev"),
        default="delta",
        help="§8.2.1 appendix ablation: 'delta' (default) ψ=[s_t, Δs_t] (6D); "
             "'with_prev' ψ=[s_{t-1}, s_t, Δs_t] (9D)",
    )

    # Loss weights
    p.add_argument("--lambda-state", type=float, default=1.0)
    p.add_argument("--lambda-motion", type=float, default=0.5)
    p.add_argument("--lambda-amp", type=float, default=0.5)
    p.add_argument("--burn-in-ratio", type=float, default=0.2)
    p.add_argument("--burn-in-min-frames", type=int, default=3)
    p.add_argument("--burn-in-weight", type=float, default=0.1)

    # §5 detach policy (defaults match the main configuration)
    p.add_argument("--detach-prev-feature", dest="detach_prev",
                   action="store_true", default=True)
    p.add_argument("--no-detach-prev-feature", dest="detach_prev",
                   action="store_false")
    p.add_argument("--detach-curr-feature", dest="detach_curr",
                   action="store_true", default=True)
    p.add_argument("--no-detach-curr-feature", dest="detach_curr",
                   action="store_false")
    p.add_argument("--wire-dilate-k", type=int, default=3)
    # §7 over-compensation fix (recommended after stage3_main_v2 negative
    # improvement_mean diagnosis; see docs and ablation_v1 results).
    p.add_argument("--lambda-zero-mean", type=float, default=0.0,
                   help="weight on ‖mean_t(Δu_t)‖²  zero-mean regularizer; "
                        "recommend 1.0 to suppress monotonic-bias shortcut")
    p.add_argument("--lambda-drift-cap", type=float, default=0.0,
                   help="weight on max(0, ‖U‖_max − cap)² soft cap; "
                        "recommend 0.5 with --drift-cap-ratio 2.0 for "
                        "extra protection against runaway cumsum")
    p.add_argument("--drift-cap-ratio", type=float, default=2.0,
                   help="multiplier of per-batch median ‖U‖ to define cap")

    # Freeze policy
    p.add_argument("--freeze-state-branch", action="store_true",
                   help="ablation: freeze state branch, train motion only")
    p.add_argument("--unfreeze-r1", action="store_true",
                   help="ablation: also train R1 (encoder + decoder + seg_head)")

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--max-train-iters", type=int, default=0)
    p.add_argument("--max-val-iters", type=int, default=0)
    p.add_argument("--save-prefix", default="stage3_joint")
    p.add_argument("--resume", action="store_true",
                   help="auto-resume from {save_prefix}_last.pth if present")

    # Distributed
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--sync-bn", action="store_true")
    return p.parse_args()


# ===========================================================================
# Distributed helpers (mirror stage1 / stage2)
# ===========================================================================


def is_dist_ready():
    return dist.is_available() and dist.is_initialized()


def is_main():
    return (not is_dist_ready()) or dist.get_rank() == 0


def print0(*a, **kw):
    if is_main():
        print(*a, **kw)


def setup_distributed(args):
    args.distributed = args.distributed or int(os.environ.get("WORLD_SIZE", "1")) > 1
    args.rank = 0
    args.local_rank = 0
    args.world_size = 1
    if not args.distributed:
        return args
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


def split_global_batch(value, world_size, name):
    if world_size <= 1:
        return value
    if value < world_size or value % world_size != 0:
        raise ValueError(f"{name}={value} must be multiple of world_size={world_size}")
    return value // world_size


def split_workers(total, world_size):
    if world_size <= 1 or total <= 0:
        return total
    return max(1, total // world_size)


def sync_metric_sums(values, device):
    t = torch.tensor(values, dtype=torch.float64, device=device)
    if is_dist_ready():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


def seed_everything(seed=42, rank=0):
    s = seed + rank
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ===========================================================================
# Freezing helpers
# ===========================================================================


def freeze_module(module, on=True):
    n = 0
    for p in module.parameters():
        if p.requires_grad == on:
            p.requires_grad = not on
            n += 1
    return n


# ===========================================================================
# Wire mask down-sampling for L_motion at 1/8 feature resolution
# ===========================================================================


def downsample_wire_mask(
    wire_mask_seq: torch.Tensor,            # [B, T, 1, H, W]
    target_h: int, target_w: int,
) -> torch.Tensor:
    """Nearest-neighbour downsample to match feat_18_seq spatial size.

    We use ``adaptive_max_pool2d`` so that any pixel hit by wire at full
    resolution remains 1 at low resolution — important because the dilated
    wire region must be conservative (over-include rather than under-include).
    """
    B, T, _, H, W = wire_mask_seq.shape
    if H == target_h and W == target_w:
        return wire_mask_seq
    flat = wire_mask_seq.view(B * T, 1, H, W).float()
    pooled = F.adaptive_max_pool2d(flat, output_size=(target_h, target_w))
    return pooled.view(B, T, 1, target_h, target_w)


def rescale_delta_u_to_feature(
    delta_u: torch.Tensor, full_size: int, feat_size: int,
) -> torch.Tensor:
    """Convert Δu from full-image pixel units to feature-map pixel units.

    RelativeMotionField predicts Δu at the network's nominal pixel scale
    (i.e. full input resolution). MotionLoss applies a warp at the feature
    resolution, which interprets Δu in *its own* pixel grid — so we must
    rescale by ``feat_size / full_size``.
    """
    if full_size == feat_size:
        return delta_u
    return delta_u * (float(feat_size) / float(full_size))


# ===========================================================================
# Train / val loops
# ===========================================================================


def run_train_epoch(
    model, loader, criterion_state, criterion_motion, optimizer, scaler,
    device, use_amp, args, epoch, writer, global_step,
):
    model.train()
    if not args.unfreeze_r1:
        # 即便 train()，被冻结的 R1 子模块要保持 eval 以稳住 BN 统计量
        for m in (model.module if hasattr(model, "module") else model).r1.modules():
            m.eval()

    sums = [0.0] * 5      # loss, loss_state, loss_motion, n_valid, iters

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]") if is_main() else loader
    for batch_idx, batch in enumerate(pbar):
        if args.max_train_iters > 0 and batch_idx >= args.max_train_iters:
            break

        images = batch["images"].to(device, non_blocking=True)
        wire = batch["wire_mask"].to(device, non_blocking=True)
        cos_gt = batch["cos_phi"].to(device, non_blocking=True)
        sin_gt = batch["sin_phi"].to(device, non_blocking=True)
        amp_gt = batch["amplitude"].to(device, non_blocking=True)
        amp_scale = batch["amp_scale"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True).float()
        ref_idx = batch["reference_index"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(
                images, mode="stage3_joint", reference_index=ref_idx,
            )
            state_pred = out["state"]                      # [B, T, 4]
            delta_u = out["delta_u"]                       # [B, T, 2] full-px
            feat_18_seq = out["feat_18_seq"]               # [B, T, C, H/8, W/8]

            # ---- L_state ----
            state_losses = criterion_state(
                state_pred,
                phase_cos_gt=cos_gt, phase_sin_gt=sin_gt,
                amp_gt=amp_gt, amp_scale=amp_scale,
                valid_mask=valid,
            )
            loss_state = state_losses["loss_state"]

            # ---- L_motion (背景 ROI 的 photometric residual) ----
            _, _, _, h_feat, w_feat = feat_18_seq.shape
            wire_18 = downsample_wire_mask(wire, h_feat, w_feat)
            delta_u_18 = rescale_delta_u_to_feature(
                delta_u, full_size=args.img_size, feat_size=h_feat,
            )
            # 把 U 也喂给 motion loss (drift cap 需要)；同样 rescale 到 1/8 单位。
            U_18 = rescale_delta_u_to_feature(
                out["U"], full_size=args.img_size, feat_size=h_feat,
            )
            motion_losses = criterion_motion(
                feat_seq=feat_18_seq,
                delta_u=delta_u_18,
                wire_mask_seq=wire_18,
                valid_pair_mask=None,
                U=U_18,
            )
            loss_motion = motion_losses["loss_motion"]

            loss = args.lambda_state * loss_state + args.lambda_motion * loss_motion

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()

        sums[0] += float(loss.item())
        sums[1] += float(loss_state.item())
        sums[2] += float(loss_motion.item())
        sums[3] += float(state_losses["n_valid_frames"].item())
        sums[4] += 1.0

        if is_main():
            global_step += 1
            pbar.set_postfix({
                "L": f"{loss.item():.4f}",
                "Ls": f"{loss_state.item():.4f}",
                "Lm": f"{loss_motion.item():.4f}",
            })
            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/loss_state", loss_state.item(), global_step)
                writer.add_scalar("train/loss_motion", loss_motion.item(), global_step)

    sums = sync_metric_sums(sums, device=device)
    n = max(sums[4], 1.0)
    return {
        "loss": sums[0] / n,
        "loss_state": sums[1] / n,
        "loss_motion": sums[2] / n,
        "n_valid_avg": sums[3] / n,
        "global_step": global_step,
    }


@torch.no_grad()
def run_val_epoch(
    model, loader, criterion_state, criterion_motion, device, use_amp, args,
):
    model.eval()
    sums = [0.0] * 5
    pbar = tqdm(loader, desc="[Val]") if is_main() else loader
    for batch_idx, batch in enumerate(pbar):
        if args.max_val_iters > 0 and batch_idx >= args.max_val_iters:
            break

        images = batch["images"].to(device, non_blocking=True)
        wire = batch["wire_mask"].to(device, non_blocking=True)
        cos_gt = batch["cos_phi"].to(device, non_blocking=True)
        sin_gt = batch["sin_phi"].to(device, non_blocking=True)
        amp_gt = batch["amplitude"].to(device, non_blocking=True)
        amp_scale = batch["amp_scale"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True).float()
        ref_idx = batch["reference_index"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(
                images, mode="stage3_joint", reference_index=ref_idx,
            )
            state_losses = criterion_state(
                out["state"],
                phase_cos_gt=cos_gt, phase_sin_gt=sin_gt,
                amp_gt=amp_gt, amp_scale=amp_scale,
                valid_mask=valid,
            )
            _, _, _, h_feat, w_feat = out["feat_18_seq"].shape
            wire_18 = downsample_wire_mask(wire, h_feat, w_feat)
            delta_u_18 = rescale_delta_u_to_feature(
                out["delta_u"], full_size=args.img_size, feat_size=h_feat,
            )
            motion_losses = criterion_motion(
                feat_seq=out["feat_18_seq"],
                delta_u=delta_u_18,
                wire_mask_seq=wire_18,
            )
            loss = (
                args.lambda_state * state_losses["loss_state"]
                + args.lambda_motion * motion_losses["loss_motion"]
            )

        sums[0] += float(loss.item())
        sums[1] += float(state_losses["loss_state"].item())
        sums[2] += float(motion_losses["loss_motion"].item())
        sums[3] += float(state_losses["n_valid_frames"].item())
        sums[4] += 1.0

    sums = sync_metric_sums(sums, device=device)
    n = max(sums[4], 1.0)
    return {
        "loss": sums[0] / n,
        "loss_state": sums[1] / n,
        "loss_motion": sums[2] / n,
        "n_valid_avg": sums[3] / n,
    }


# ===========================================================================
# Main
# ===========================================================================


def train(args):
    args = setup_distributed(args)
    seed_everything(42, rank=args.rank)
    torch.backends.cudnn.benchmark = True

    per_rank_bs = split_global_batch(args.batch_size, args.world_size, "batch_size")
    per_rank_workers = split_workers(args.num_workers, args.world_size)

    device = (
        torch.device("cuda", args.local_rank) if (args.distributed and torch.cuda.is_available())
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    save_dir = BASE_DIR / "experiments" / "checkpoints"
    log_dir = BASE_DIR / "experiments" / "logs" / args.save_prefix
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    last_path = save_dir / f"{args.save_prefix}_last.pth"
    best_path = save_dir / f"{args.save_prefix}_best.pth"

    print0(f"🔥 Stage-3 joint training | device={device} | per_rank_bs={per_rank_bs}")
    print0(
        f"   motion_field={args.motion_field_type}  "
        f"state_core={args.state_core_type}  "
        f"λ_state={args.lambda_state}  λ_motion={args.lambda_motion}  "
        f"detach_prev={args.detach_prev}  detach_curr={args.detach_curr}  "
        f"unfreeze_r1={args.unfreeze_r1}  freeze_state={args.freeze_state_branch}"
    )

    # ---- Datasets ----
    excluded = None
    if args.exclude_sequences_file:
        excluded = [
            ln.strip() for ln in Path(args.exclude_sequences_file).read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]

    def _read_list(path):
        return [ln.strip() for ln in Path(path).read_text().splitlines()
                if ln.strip() and not ln.strip().startswith("#")]

    # Pooled-split mode (mirrors train_stage2_state.py): when --cv-split is
    # set, both datasets read from data_root/<cv_split>/... and the
    # included sequences come from explicit .txt files.
    if getattr(args, "cv_split", None):
        if not args.cv_train_file or not args.cv_val_file:
            raise SystemExit("--cv-split requires --cv-train-file and --cv-val-file")
        train_seqs = _read_list(args.cv_train_file)
        val_seqs = _read_list(args.cv_val_file)
        train_set = FluoroSequenceWindowDataset(
            data_root=args.data_root, state_labels_dir=args.state_labels_dir,
            split=args.cv_split, T_window=args.T_window, stride=args.stride,
            img_size=(args.img_size, args.img_size),
            included_sequences=train_seqs,
        )
        val_set = FluoroSequenceWindowDataset(
            data_root=args.data_root, state_labels_dir=args.state_labels_dir,
            split=args.cv_split, T_window=args.T_window, stride=args.stride,
            img_size=(args.img_size, args.img_size),
            included_sequences=val_seqs,
        )
    else:
        train_set = FluoroSequenceWindowDataset(
            data_root=args.data_root, state_labels_dir=args.state_labels_dir,
            split="train", T_window=args.T_window, stride=args.stride,
            img_size=(args.img_size, args.img_size),
            excluded_sequences=excluded,
        )
        val_set = FluoroSequenceWindowDataset(
            data_root=args.data_root, state_labels_dir=args.state_labels_dir,
            split="val", T_window=args.T_window, stride=args.stride,
            img_size=(args.img_size, args.img_size),
            excluded_sequences=excluded,
        )

    train_sampler = (
        DistributedSampler(train_set, args.world_size, args.rank, shuffle=True, drop_last=True)
        if args.distributed else None
    )
    val_sampler = (
        DistributedSampler(val_set, args.world_size, args.rank, shuffle=False, drop_last=False)
        if args.distributed else None
    )
    train_loader = DataLoader(
        train_set, batch_size=per_rank_bs, sampler=train_sampler,
        shuffle=train_sampler is None, num_workers=per_rank_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=per_rank_bs, sampler=val_sampler,
        shuffle=False, num_workers=per_rank_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )

    # ---- Model ----
    model = FreqDecGWNet(
        in_channels=1, num_classes=1, width_mult=args.width_mult,
        state_proj_dim=args.state_proj_dim,
        state_hidden_dim=args.state_hidden_dim,
        state_core_type=args.state_core_type,
        motion_field_type=args.motion_field_type,
        motion_state_dim=args.motion_state_dim,
        motion_input_mode=args.motion_input_mode,
    ).to(device)

    # 加载顺序：优先 stage2 ckpt（包含 stage1 的 R1）；缺则只装 stage1
    if args.stage2_ckpt is not None:
        if not args.stage2_ckpt.is_file():
            raise FileNotFoundError(f"stage2_ckpt not found: {args.stage2_ckpt}")
        ckpt = torch.load(args.stage2_ckpt, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print0(
            f"🔁 Loaded stage-2 ckpt {args.stage2_ckpt.name} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    elif args.stage1_ckpt is not None:
        missing, unexpected = model.load_stage1_checkpoint(
            args.stage1_ckpt, strict=False,
        )
        print0(
            f"🔁 Loaded stage-1 ckpt only {args.stage1_ckpt.name} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    # Freeze policy
    if not args.unfreeze_r1:
        n_frozen = freeze_module(model.r1, on=True)
        print0(f"❄️  Froze {n_frozen} R1 parameters.")
    if args.freeze_state_branch:
        n_frozen = freeze_module(model.state_branch, on=True)
        print0(f"❄️  Froze {n_frozen} state_branch parameters.")

    if args.distributed:
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # ---- Loss / optimizer ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print0(f"🧠 trainable parameters: {n_trainable:,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    criterion_state = StateLoss(
        lambda_amp=args.lambda_amp,
        burn_in_ratio=args.burn_in_ratio,
        burn_in_min_frames=args.burn_in_min_frames,
        burn_in_weight=args.burn_in_weight,
    ).to(device)
    criterion_motion = MotionLoss(
        detach_prev_feature=args.detach_prev,
        detach_curr_feature=args.detach_curr,
        wire_dilate_k=args.wire_dilate_k,
        lambda_zero_mean=args.lambda_zero_mean,
        lambda_drift_cap=args.lambda_drift_cap,
        drift_cap_ratio=args.drift_cap_ratio,
    ).to(device)

    writer = SummaryWriter(log_dir=str(log_dir)) if is_main() else None
    best_loss = float("inf")
    global_step = 0
    start_epoch = 1

    # Auto-resume — lets the v5 background pipeline retry crashed runs
    # without losing wall-clock progress.
    if args.resume:
        if last_path.is_file():
            ck = torch.load(last_path, map_location="cpu", weights_only=False)
            sd = ck.get("model_state_dict", ck)
            sd = {
                (k[len("module."):] if k.startswith("module.") else k): v
                for k, v in sd.items()
            }
            inner = model.module if hasattr(model, "module") else model
            miss, unex = inner.load_state_dict(sd, strict=False)
            if "optimizer_state_dict" in ck:
                optimizer.load_state_dict(ck["optimizer_state_dict"])
            if "scheduler_state_dict" in ck:
                scheduler.load_state_dict(ck["scheduler_state_dict"])
            if "scaler_state_dict" in ck:
                scaler.load_state_dict(ck["scaler_state_dict"])
            start_epoch = int(ck.get("epoch", 0)) + 1
            global_step = int(ck.get("global_step", 0))
            best_loss = float(ck.get("best_val_loss", float("inf")))
            print0(
                f"🔁 RESUME — start_epoch={start_epoch}, "
                f"global_step={global_step}, best_val={best_loss:.4f}"
            )
        else:
            print0(f"⚠️  --resume given but {last_path} missing; fresh start")

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_stats = run_train_epoch(
            model, train_loader, criterion_state, criterion_motion,
            optimizer, scaler, device,
            use_amp=args.amp and device.type == "cuda",
            args=args, epoch=epoch, writer=writer, global_step=global_step,
        )
        global_step = train_stats["global_step"]
        val_stats = run_val_epoch(
            model, val_loader, criterion_state, criterion_motion, device,
            use_amp=args.amp and device.type == "cuda", args=args,
        )

        scheduler.step()

        if is_main():
            print0(
                f"🏁 Epoch {epoch}/{args.epochs}  "
                f"train: L={train_stats['loss']:.4f} "
                f"Ls={train_stats['loss_state']:.4f} Lm={train_stats['loss_motion']:.4f}   "
                f"val: L={val_stats['loss']:.4f} "
                f"Ls={val_stats['loss_state']:.4f} Lm={val_stats['loss_motion']:.4f}"
            )
            if writer is not None:
                writer.add_scalar("val/loss", val_stats["loss"], epoch)
                writer.add_scalar("val/loss_state", val_stats["loss_state"], epoch)
                writer.add_scalar("val/loss_motion", val_stats["loss_motion"], epoch)

            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": (
                    model.module.state_dict() if hasattr(model, "module")
                    else model.state_dict()
                ),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_loss": best_loss,
                "args": vars(args),
            }
            torch.save(ckpt, last_path)
            if val_stats["loss"] < best_loss:
                best_loss = val_stats["loss"]
                torch.save(ckpt, best_path)
                print0(f"🌟 New best val loss = {best_loss:.4f}")

    if writer is not None:
        writer.close()
    print0("🎉 Stage 3 joint training complete.")


if __name__ == "__main__":
    try:
        train(parse_args())
    finally:
        cleanup_distributed()
