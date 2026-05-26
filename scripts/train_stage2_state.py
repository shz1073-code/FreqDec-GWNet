#!/usr/bin/env python
"""Stage 2 — train the GlobalStateBranch on top of a frozen R1 backbone.

Per PROJECT_CONSTRAINTS_19.6.md §5, ``L_state`` must not flow back into the
encoder; the default ``state_detach_encoder=True`` already enforces that
inside :class:`GlobalStateBranch`. As a belt-and-braces measure we *also*
freeze every R1 parameter here (``requires_grad=False``) so even an
accidental future change to the detach flag cannot leak gradients into
seg-trained weights.

Inputs:
    * data_root with split-then-kind layout (e.g. /workspace/shz/clean_data)
    * pre-generated state labels under ``--state-labels-dir`` (Phase B)
    * stage-1 checkpoint to seed the encoder + segmentation modules

Outputs:
    * ``experiments/checkpoints/{save_prefix}_last.pth``  (resume-capable)
    * ``experiments/checkpoints/{save_prefix}_best.pth``  (state branch only)
    * ``experiments/logs/{save_prefix}/`` (TensorBoard)

Usage::

    python scripts/train_stage2_state.py \\
        --data-root /workspace/shz/clean_data \\
        --state-labels-dir /workspace/FreqDec-GWNet/reports/state_labels \\
        --stage1-ckpt /workspace/FreqDec-GWNet/experiments/checkpoints/stage1_v1_best.pth \\
        --epochs 30 --batch-size 1 --T-window 64
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from freqdec_gwnet.data import (                                     # noqa: E402
    ChronologicalSampler,
    FluoroSequenceWindowDataset,
    is_sequence_boundary,
)
from freqdec_gwnet.losses.physiological_priors import (              # noqa: E402
    physiological_prior_loss,
)
from freqdec_gwnet.losses.state_losses import StateLoss               # noqa: E402
from freqdec_gwnet.models.freqdec_gwnet import FreqDecGWNet           # noqa: E402


# ===========================================================================
# Checkpoint I/O
# ===========================================================================


def _atomic_save(obj, path) -> None:
    """Write a checkpoint atomically: torch.save to a temp file, then
    os.replace into place. A crash mid-save (this box segfaults
    intermittently under cv2-augmented dataloading) can corrupt a
    plain torch.save target to 0 bytes, which then makes --resume fail
    with EOFError. os.replace is atomic on the same filesystem, so the
    real checkpoint is never left half-written.
    """
    path = Path(path)
    tmp = path.parent / (path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ===========================================================================
# CLI
# ===========================================================================


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: train GlobalStateBranch.")
    # Data
    p.add_argument("--data-root", required=True, type=Path,
                   help="root with {split}/{images,labels}/{seq}/...")
    p.add_argument("--state-labels-dir", required=True, type=Path,
                   help="dir with per-sequence StateLabel .npz files")
    p.add_argument("--exclude-sequences-file", type=str, default=None)
    # ---- k-fold cross-validation (sequence-level) ----
    p.add_argument("--cv-split", type=str, default=None,
                   help="if set, BOTH train and val datasets read this one "
                        "(pooled) split; the train/val sequence partition "
                        "is then given explicitly by --cv-train-file / "
                        "--cv-val-file. Use for k-fold CV.")
    p.add_argument("--cv-train-file", type=str, default=None,
                   help="text file, one sequence name per line — the "
                        "training fold (used only with --cv-split)")
    p.add_argument("--cv-val-file", type=str, default=None,
                   help="text file, one sequence name per line — the "
                        "held-out fold (used only with --cv-split)")
    p.add_argument("--augment", action="store_true",
                   help="enable burst-consistent real-data augmentation "
                        "on the training set (geometry + photometric)")
    p.add_argument("--T-window", type=int, default=64)
    p.add_argument("--stride", type=int, default=None,
                   help="default = T_window // 2")
    p.add_argument("--img-size", type=int, default=512)

    # Stage-1 seed
    p.add_argument("--stage1-ckpt", type=Path, default=None,
                   help="FAST_LiteNet checkpoint to load into self.r1; "
                        "if omitted, encoder starts from scratch")

    # Model
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--state-core-type", choices=("gru", "mamba"),
                   default="gru")
    p.add_argument("--state-proj-dim", type=int, default=128)
    p.add_argument("--state-hidden-dim", type=int, default=64)
    p.add_argument("--state-d-state", type=int, default=16)
    p.add_argument("--no-state-detach-encoder", action="store_true",
                   help="ablation: allow L_state grad to flow into encoder "
                        "(violates §5; only for table 3 ablation)")

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-amp", type=float, default=0.5)
    p.add_argument("--burn-in-ratio", type=float, default=0.2)
    p.add_argument("--burn-in-min-frames", type=int, default=3)
    p.add_argument("--burn-in-weight", type=float, default=0.1)
    p.add_argument("--amp", action="store_true",
                   help="enable mixed precision")
    p.add_argument("--max-train-iters", type=int, default=0,
                   help=">0 caps train iterations per epoch (sanity)")
    p.add_argument("--max-val-iters", type=int, default=0)
    p.add_argument("--save-prefix", default="stage2_state")
    p.add_argument("--resume", action="store_true",
                   help="if {save_prefix}_last.pth exists, load it and "
                        "continue from the next epoch (background "
                        "pipelines depend on this)")
    # §III.E physiological priors (paper §IV.E ablation)
    p.add_argument("--lambda-phase-smooth", type=float, default=0.0,
                   help=">0 enables phase smoothness prior")
    p.add_argument("--lambda-amp-smooth", type=float, default=0.0,
                   help=">0 enables amplitude smoothness prior")
    p.add_argument("--lambda-spectral", type=float, default=0.0,
                   help=">0 enables spectral concentration prior")
    p.add_argument("--prior-fs", type=float, default=15.0)
    p.add_argument("--prior-band-low", type=float, default=0.15)
    p.add_argument("--prior-band-high", type=float, default=0.50)

    # Dual-cycle (cardiac) extension — option C: extract respiratory AND
    # cardiac phase jointly with an 8-dim state branch output.
    p.add_argument("--cardiac-labels-dir", type=str, default=None,
                   help="path to cardiac Phase-B labels. Enables dual-head.")
    p.add_argument("--state-output-dim", type=int, default=4,
                   choices=[4, 8],
                   help="4=resp-only (default), 8=dual-cycle (resp+card)")
    p.add_argument("--lambda-cardiac", type=float, default=0.0,
                   help="cardiac branch weight in the total loss")
    p.add_argument("--lambda-cardiac-amp", type=float, default=0.5,
                   help="cardiac sub-weight for the amplitude term")
    p.add_argument("--warm-start-ckpt", type=Path, default=None,
                   help="path to a 4-dim stage-2 ckpt to warm-start an "
                        "8-dim dual-head model (expands output head)")

    # §6.1 main ablation #2: state training protocol
    p.add_argument(
        "--state-protocol",
        choices=("random_zero_init", "context_prefix", "chronological_carry"),
        default="random_zero_init",
        help=(
            "random_zero_init: shuffled windows, fresh h0 per window (default).\n"
            "context_prefix:   uses --burn-in-min-frames as a hard cold-start "
            "floor — e.g. --burn-in-min-frames 16 --burn-in-weight 0 to "
            "discard supervision on the first 16 frames per window.\n"
            "chronological_carry: ChronologicalSampler walks each sequence in "
            "order; h0 from window k carries (detached) into window k+1 "
            "of the same sequence. Single-rank only — DDP not supported."
        ),
    )

    # Distributed
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--sync-bn", action="store_true")
    return p.parse_args()


# ===========================================================================
# Distributed helpers (mirror train_stage1_seg.py)
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
    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA")
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
        raise ValueError(f"{name}={value} must be a positive multiple of world_size={world_size}")
    return value // world_size


def split_workers(total, world_size):
    if world_size <= 1:
        return total
    return max(1, total // world_size) if total > 0 else 0


def sync_metric_sums(values, device):
    t = torch.tensor(values, dtype=torch.float64, device=device)
    if is_dist_ready():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


def seed_everything(seed=42, rank=0):
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ===========================================================================
# Encoder + decoder freezing
# ===========================================================================


def freeze_r1(model: FreqDecGWNet) -> int:
    """Set ``requires_grad=False`` on every R1 parameter; return frozen count."""
    n = 0
    for p in model.r1.parameters():
        if p.requires_grad:
            p.requires_grad = False
            n += 1
    return n


# ===========================================================================
# Train / val loops
# ===========================================================================


def run_train_epoch(
    model, loader, criterion, optimizer, scaler, device,
    use_amp, args, epoch, writer, global_step,
):
    model.train()
    # Belt-and-braces：即使 train() 把 BN 切回训练态，R1 是冻结的 forward-only
    for m in model.r1.modules():
        m.eval()                                   # 冻结 BN/Dropout 统计量

    sum_loss = 0.0
    sum_phase = 0.0
    sum_amp = 0.0
    sum_n_valid = 0.0
    n_iters = 0

    # chronological_carry: 跟踪上 batch 的 seq_name 和 h0，在序列边界 reset
    chrono_mode = args.state_protocol == "chronological_carry"
    h_carry: Optional[torch.Tensor] = None
    prev_seq_names: Optional[List[str]] = None

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]") if is_main() else loader
    for batch_idx, batch in enumerate(pbar):
        if args.max_train_iters > 0 and batch_idx >= args.max_train_iters:
            break

        images = batch["images"].to(device, non_blocking=True)
        cos_gt = batch["cos_phi"].to(device, non_blocking=True)
        sin_gt = batch["sin_phi"].to(device, non_blocking=True)
        amp_gt = batch["amplitude"].to(device, non_blocking=True)
        amp_scale = batch["amp_scale"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True).float()
        # Dual-cycle (cardiac) GT — optional.
        card_kwargs = {}
        if "cardiac_cos_phi" in batch:
            card_kwargs = {
                "cardiac_cos_gt":     batch["cardiac_cos_phi"].to(device, non_blocking=True),
                "cardiac_sin_gt":     batch["cardiac_sin_phi"].to(device, non_blocking=True),
                "cardiac_amp_gt":     batch["cardiac_amplitude"].to(device, non_blocking=True),
                "cardiac_amp_scale":  batch["cardiac_amp_scale"].to(device, non_blocking=True),
                "cardiac_valid_mask": batch["cardiac_valid_mask"].to(device, non_blocking=True).float(),
            }
        curr_seq_names: List[str] = (
            batch["seq_name"] if isinstance(batch["seq_name"], list)
            else list(batch["seq_name"])
        )

        # Build h0 according to protocol
        h0 = None
        if chrono_mode:
            boundaries = is_sequence_boundary(prev_seq_names, curr_seq_names)
            if h_carry is None or all(boundaries):
                h0 = None                          # 全是新序列 → fresh zeros
            else:
                # 把碰到序列边界的样本的 h_carry 行清零，其余保留
                mask = torch.tensor(
                    [0.0 if b else 1.0 for b in boundaries],
                    device=h_carry.device, dtype=h_carry.dtype,
                ).view(-1, 1)
                h0 = (h_carry.detach() * mask).contiguous()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(images, mode="stage2_state", h0=h0)
            state_pred = out["state"]              # [B, T, 4 or 8]
            losses = criterion(
                state_pred,
                phase_cos_gt=cos_gt,
                phase_sin_gt=sin_gt,
                amp_gt=amp_gt,
                amp_scale=amp_scale,
                valid_mask=valid,
                **card_kwargs,
            )
            loss = losses["loss_state"]

            # ---- §III.E physiological priors (paper §IV.E ablation) ----
            if (
                args.lambda_phase_smooth > 0
                or args.lambda_amp_smooth > 0
                or args.lambda_spectral > 0
            ):
                prior = physiological_prior_loss(
                    cos_pred=state_pred[..., 0],
                    sin_pred=state_pred[..., 1],
                    amp_pred=state_pred[..., 2],
                    amp_scale=amp_scale,
                    valid_mask=valid,
                    lambda_phase_smooth=args.lambda_phase_smooth,
                    lambda_amp_smooth=args.lambda_amp_smooth,
                    lambda_spectral=args.lambda_spectral,
                    fs=args.prior_fs,
                    band_hz=(args.prior_band_low, args.prior_band_high),
                )
                loss = loss + prior["loss_phys_prior"]
                losses["L_phase_smooth"] = prior["L_phase_smooth"]
                losses["L_amp_smooth"] = prior["L_amp_smooth"]
                losses["L_spectral"] = prior["L_spectral"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()

        # chronological_carry：抓出每个样本最后一帧的 hidden state 作为下一窗的 h0。
        # 这里用 GRU 路径下的简单近似——读取 state 输出的最后一帧 4-D 向量
        # 反推到 hidden_dim 不直观，所以改为：让模型再前向一次 forward_step
        # 取最终隐状态。代价是一次额外 forward；好处是与单步推理路径完全等价。
        if chrono_mode:
            with torch.no_grad():
                # 用最后一帧 1/8 特征再走一次 forward_step 拿真隐状态
                # （比解析 4-D state 反推更可靠）
                feat_last = out["feat_18_seq"][:, -1].detach()
                model_to_use = (
                    model.module if hasattr(model, "module") else model
                )
                _, h_last, _ = model_to_use.state_branch.forward_step(
                    feat_last, h_prev=h0,
                )
                h_carry = h_last.detach()
            prev_seq_names = curr_seq_names

        sum_loss += float(loss.item())
        sum_phase += float(losses["loss_phase"].item())
        sum_amp += float(losses["loss_amp"].item())
        sum_n_valid += float(losses["n_valid_frames"].item())
        n_iters += 1

        if is_main():
            global_step += 1
            pbar.set_postfix({
                "L": f"{loss.item():.4f}",
                "Lp": f"{losses['loss_phase'].item():.4f}",
                "La": f"{losses['loss_amp'].item():.4f}",
                "valid": f"{int(losses['n_valid_frames'].item())}",
            })
            if writer is not None:
                writer.add_scalar("train/loss_state", loss.item(), global_step)
                writer.add_scalar("train/loss_phase", losses["loss_phase"].item(), global_step)
                writer.add_scalar("train/loss_amp", losses["loss_amp"].item(), global_step)

    sums = sync_metric_sums(
        [sum_loss, sum_phase, sum_amp, sum_n_valid, float(n_iters)],
        device=device,
    )
    n = max(sums[4], 1.0)
    return {
        "loss_state": sums[0] / n,
        "loss_phase": sums[1] / n,
        "loss_amp": sums[2] / n,
        "n_valid_avg": sums[3] / n,
        "global_step": global_step,
    }


@torch.no_grad()
def run_val_epoch(model, loader, criterion, device, use_amp, args):
    model.eval()
    sum_loss = sum_phase = sum_amp = sum_n_valid = 0.0
    n_iters = 0
    pbar = tqdm(loader, desc="[Val]") if is_main() else loader
    for batch_idx, batch in enumerate(pbar):
        if args.max_val_iters > 0 and batch_idx >= args.max_val_iters:
            break
        images = batch["images"].to(device, non_blocking=True)
        cos_gt = batch["cos_phi"].to(device, non_blocking=True)
        sin_gt = batch["sin_phi"].to(device, non_blocking=True)
        amp_gt = batch["amplitude"].to(device, non_blocking=True)
        amp_scale = batch["amp_scale"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True).float()
        card_kwargs = {}
        if "cardiac_cos_phi" in batch:
            card_kwargs = {
                "cardiac_cos_gt":     batch["cardiac_cos_phi"].to(device, non_blocking=True),
                "cardiac_sin_gt":     batch["cardiac_sin_phi"].to(device, non_blocking=True),
                "cardiac_amp_gt":     batch["cardiac_amplitude"].to(device, non_blocking=True),
                "cardiac_amp_scale":  batch["cardiac_amp_scale"].to(device, non_blocking=True),
                "cardiac_valid_mask": batch["cardiac_valid_mask"].to(device, non_blocking=True).float(),
            }

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(images, mode="stage2_state")
            losses = criterion(
                out["state"],
                phase_cos_gt=cos_gt, phase_sin_gt=sin_gt,
                amp_gt=amp_gt, amp_scale=amp_scale,
                valid_mask=valid,
                **card_kwargs,
            )
            loss = losses["loss_state"]

        sum_loss += float(loss.item())
        sum_phase += float(losses["loss_phase"].item())
        sum_amp += float(losses["loss_amp"].item())
        sum_n_valid += float(losses["n_valid_frames"].item())
        n_iters += 1

    sums = sync_metric_sums(
        [sum_loss, sum_phase, sum_amp, sum_n_valid, float(n_iters)], device,
    )
    n = max(sums[4], 1.0)
    return {
        "loss_state": sums[0] / n,
        "loss_phase": sums[1] / n,
        "loss_amp": sums[2] / n,
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

    print0(f"🔥 Stage-2 training | device={device} | per_rank_bs={per_rank_bs}")
    print0(f"   data_root={args.data_root}  state_labels={args.state_labels_dir}")
    print0(f"   T_window={args.T_window} stride={args.stride}  width_mult={args.width_mult}")

    # ---- Datasets ----
    def _read_seq_file(path):
        return [
            ln.strip() for ln in Path(path).read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]

    excluded = None
    if args.exclude_sequences_file:
        excluded = _read_seq_file(args.exclude_sequences_file)

    if args.cv_split:
        # k-fold CV: both datasets read one pooled split; the train/val
        # partition is given explicitly by the fold files.
        if not (args.cv_train_file and args.cv_val_file):
            raise SystemExit("--cv-split requires --cv-train-file and --cv-val-file")
        cv_train = _read_seq_file(args.cv_train_file)
        cv_val = _read_seq_file(args.cv_val_file)
        print0(f"📂 CV mode: split={args.cv_split}  "
               f"train={len(cv_train)} seqs  val={len(cv_val)} seqs  "
               f"augment={args.augment}")
        train_set = FluoroSequenceWindowDataset(
            data_root=args.data_root, state_labels_dir=args.state_labels_dir,
            split=args.cv_split, T_window=args.T_window, stride=args.stride,
            img_size=(args.img_size, args.img_size),
            included_sequences=cv_train, augment=args.augment,
            cardiac_labels_dir=args.cardiac_labels_dir,
        )
        val_set = FluoroSequenceWindowDataset(
            data_root=args.data_root, state_labels_dir=args.state_labels_dir,
            split=args.cv_split, T_window=args.T_window, stride=args.stride,
            img_size=(args.img_size, args.img_size),
            included_sequences=cv_val, augment=False,
            cardiac_labels_dir=args.cardiac_labels_dir,
        )
    else:
        train_set = FluoroSequenceWindowDataset(
            data_root=args.data_root, state_labels_dir=args.state_labels_dir,
            split="train", T_window=args.T_window, stride=args.stride,
            img_size=(args.img_size, args.img_size),
            excluded_sequences=excluded, augment=args.augment,
            cardiac_labels_dir=args.cardiac_labels_dir,
        )
        val_set = FluoroSequenceWindowDataset(
            data_root=args.data_root, state_labels_dir=args.state_labels_dir,
            split="val", T_window=args.T_window, stride=args.stride,
            img_size=(args.img_size, args.img_size),
            excluded_sequences=excluded,
            cardiac_labels_dir=args.cardiac_labels_dir,
        )

    # State training protocol (paper §8.1.2 main ablation #2)
    if args.state_protocol == "chronological_carry":
        if args.distributed:
            raise SystemExit(
                "--state-protocol chronological_carry is single-rank only; "
                "DDP across sequences would break the carry-over contract"
            )
        train_sampler = ChronologicalSampler(
            train_set, shuffle_sequences=True, seed=42,
        )
        print0(
            f"⏱  ChronologicalSampler over {train_sampler.num_sequences} "
            f"sequences (h0 carries within each sequence)"
        )
    elif args.distributed:
        train_sampler = DistributedSampler(
            train_set, args.world_size, args.rank,
            shuffle=True, drop_last=True,
        )
    else:
        train_sampler = None

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
        state_d_state=args.state_d_state,
        state_detach_encoder=not args.no_state_detach_encoder,
        state_output_dim=args.state_output_dim,
    ).to(device)

    if args.stage1_ckpt is not None:
        if not args.stage1_ckpt.is_file():
            raise FileNotFoundError(f"stage1_ckpt not found: {args.stage1_ckpt}")
        missing, unexpected = model.load_stage1_checkpoint(
            args.stage1_ckpt, strict=False,
        )
        print0(
            f"🔁 Loaded stage-1 ckpt {args.stage1_ckpt.name} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    # Warm-start from a previously-trained 4-dim stage-2 checkpoint, expanding
    # the output head 4 -> 8 (resp block initialized from the source, cardiac
    # block initialized to small random values). All other weights are loaded
    # verbatim. The expanded resp block keeps the proven resp performance;
    # the cardiac block starts from scratch and learns alongside.
    if args.warm_start_ckpt is not None:
        if not args.warm_start_ckpt.is_file():
            raise FileNotFoundError(
                f"warm_start_ckpt not found: {args.warm_start_ckpt}"
            )
        wsck = torch.load(args.warm_start_ckpt, map_location="cpu",
                          weights_only=False)
        src_sd = wsck.get("model_state_dict", wsck)
        src_sd = {(k[7:] if k.startswith("module.") else k): v
                  for k, v in src_sd.items()}
        inner = model.module if hasattr(model, "module") else model
        dst_sd = inner.state_dict()
        # Expand output_head.2 if source is 4-dim and target is 8-dim.
        head_w_key = "state_branch.output_head.2.weight"
        head_b_key = "state_branch.output_head.2.bias"
        if (head_w_key in src_sd and head_w_key in dst_sd
                and src_sd[head_w_key].shape[0] == 4
                and dst_sd[head_w_key].shape[0] == 8):
            new_w = dst_sd[head_w_key].clone()
            new_w[:4] = src_sd[head_w_key]
            new_w[4:] = src_sd[head_w_key] * 0.0 + torch.randn_like(
                src_sd[head_w_key]) * 0.01
            src_sd[head_w_key] = new_w
            new_b = dst_sd[head_b_key].clone()
            new_b[:4] = src_sd[head_b_key]
            new_b[4:] = 0.0
            src_sd[head_b_key] = new_b
            print0("🌱 expanded output_head 4 -> 8 (resp block warm-start, "
                   "cardiac block small-random init)")
        miss, unex = inner.load_state_dict(src_sd, strict=False)
        print0(f"🌱 warm-start from {args.warm_start_ckpt.name} "
               f"(missing={len(miss)}, unexpected={len(unex)})")

    n_frozen = freeze_r1(model)
    print0(f"❄️  Froze {n_frozen} R1 parameters; only state_branch trains.")

    if args.distributed:
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # ---- Optimizer / scheduler / loss ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    criterion = StateLoss(
        lambda_amp=args.lambda_amp,
        burn_in_ratio=args.burn_in_ratio,
        burn_in_min_frames=args.burn_in_min_frames,
        burn_in_weight=args.burn_in_weight,
        lambda_cardiac=args.lambda_cardiac,
        lambda_cardiac_amp=args.lambda_cardiac_amp,
    ).to(device)

    writer = SummaryWriter(log_dir=str(log_dir)) if is_main() else None
    best_loss = float("inf")
    global_step = 0
    start_epoch = 1

    # ---- Auto-resume from {prefix}_last.pth if --resume and the file exists.
    # 这条路径让长时间运行的训练在脚本崩溃/重启后能从上次 epoch 接着跑。
    if args.resume:
        resume_path = last_path
        ck = None
        if resume_path.is_file() and resume_path.stat().st_size > 0:
            try:
                ck = torch.load(resume_path, map_location="cpu", weights_only=False)
            except Exception as exc:                       # corrupt checkpoint
                print0(f"⚠️  --resume checkpoint {resume_path} unreadable "
                       f"({type(exc).__name__}: {exc}); starting fresh")
                ck = None
        if ck is not None:
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
                f"global_step={global_step}, best_val={best_loss:.4f} "
                f"(missing={len(miss)}, unexpected={len(unex)})"
            )
        else:
            print0(f"⚠️  --resume given but {resume_path} missing; starting fresh")

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_stats = run_train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            use_amp=args.amp and device.type == "cuda",
            args=args, epoch=epoch, writer=writer, global_step=global_step,
        )
        global_step = train_stats["global_step"]
        val_stats = run_val_epoch(
            model, val_loader, criterion, device,
            use_amp=args.amp and device.type == "cuda", args=args,
        )

        scheduler.step()

        if is_main():
            print0(
                f"🏁 Epoch {epoch}/{args.epochs}  "
                f"train: L={train_stats['loss_state']:.4f} "
                f"Lp={train_stats['loss_phase']:.4f} La={train_stats['loss_amp']:.4f}   "
                f"val: L={val_stats['loss_state']:.4f} "
                f"Lp={val_stats['loss_phase']:.4f} La={val_stats['loss_amp']:.4f}"
            )
            if writer is not None:
                writer.add_scalar("val/loss_state", val_stats["loss_state"], epoch)
                writer.add_scalar("val/loss_phase", val_stats["loss_phase"], epoch)
                writer.add_scalar("val/loss_amp", val_stats["loss_amp"], epoch)

            # Save last
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
            _atomic_save(ckpt, last_path)

            if val_stats["loss_state"] < best_loss:
                best_loss = val_stats["loss_state"]
                _atomic_save(ckpt, best_path)
                print0(f"🌟 New best val loss = {best_loss:.4f}")

    if writer is not None:
        writer.close()
    print0("🎉 Stage 2 training complete.")


if __name__ == "__main__":
    try:
        train(parse_args())
    finally:
        cleanup_distributed()
