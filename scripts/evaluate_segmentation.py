import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 和训练脚本保持一致，方便直接在当前工程目录下运行。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 19.x 旧版本依赖一个全局 BASE_DIR 但没定义，这里显式补上。
BASE_DIR = PROJECT_ROOT
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from freqdec_gwnet.data.real_dataset import RealGuidewireVideoDataset
from freqdec_gwnet.losses.segmentation_losses import Soft_clDiceLoss
from freqdec_gwnet.models.fast_litenet import FAST_LiteNet
from freqdec_gwnet.models.unet_baseline import UNetBaseline


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FAST_LiteNet on val/test split.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/checkpoints/fast_litenet_best.pth",
        help="Path to model checkpoint or state_dict.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="fast_litenet",
        choices=["fast_litenet", "unet_baseline"],
        help="Model architecture used for loading/evaluation.",
    )
    parser.add_argument(
        "--width-mult",
        type=float,
        default=1.0,
        help="Width multiplier for FAST-LiteNet evaluation. Keep it consistent with training.",
    )
    parser.add_argument(
        "--multi-task",
        action="store_true",
        help="Load FAST-LiteNet with segmentation + centerline + tip-proxy heads. Evaluation still uses seg output only.",
    )
    parser.add_argument(
        "--edge-aux",
        action="store_true",
        help="Load FAST-LiteNet with an auxiliary edge head. Evaluation still uses seg output only.",
    )
    parser.add_argument(
        "--use-refine-head",
        action="store_true",
        help="Enable the lightweight large-kernel refinement head during evaluation. Keep it consistent with training.",
    )
    parser.add_argument(
        "--refine-kernel-size",
        type=int,
        default=7,
        help="Kernel size used by the lightweight refinement head. Keep it consistent with training.",
    )
    parser.add_argument(
        "--enable-conf-gate",
        action="store_true",
        help="Enable confidence-gated temporal feature-bank updates during evaluation.",
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
        help="Frequency modeling mode. Keep it consistent with training.",
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
        help="Window sizes used when --freq-mode ms_local_sff. Keep it consistent with training.",
    )
    parser.add_argument(
        "--freq-gate-type",
        type=str,
        default="none",
        choices=["none", "channel"],
        help="Optional channel gate for the FFT residual branch. Keep it consistent with training.",
    )
    parser.add_argument(
        "--freq-gate-reduction",
        type=int,
        default=4,
        help="Reduction ratio used by the FFT channel gate. Keep it consistent with training.",
    )
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
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which split to evaluate.",
    )
    parser.add_argument("--seq-len", type=int, default=5, help="Sequence length.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="If > 0, stop evaluation after this many batches. Use for sanity checks.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for binary mask.")
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=8,
        help="How many prediction previews to save.",
    )
    parser.add_argument(
        "--max-worst-visualizations",
        type=int,
        default=16,
        help="How many worst-case sequence previews to save.",
    )
    parser.add_argument(
        "--worst-metric",
        type=str,
        default="dice",
        choices=["dice", "iou", "cldice"],
        help="Metric used to rank worst-case previews.",
    )
    return parser.parse_args()


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if all(not key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def remap_fast_litenet_state_dict_for_current_model(state_dict, model_state_dict):
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


def filter_state_dict_by_shape(state_dict, model_state_dict):
    """
    评估时也允许跨结构 warm start：
    如果当前模型只是局部模块结构变了，就跳过尺寸不兼容的旧参数，
    避免直接因为 shape mismatch 终止。
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
        print("[Checkpoint Load] shape-mismatched keys skipped:")
        for key, old_shape, new_shape in skipped:
            print(f"  - {key}: ckpt{old_shape} -> model{new_shape}")
    return filtered


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = {}

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        metadata = {
            "epoch": checkpoint.get("epoch"),
            "best_val_loss": checkpoint.get("best_val_loss"),
            "best_val_dice": checkpoint.get("best_val_dice"),
        }
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    remapped_state_dict = remap_fast_litenet_state_dict_for_current_model(state_dict, model.state_dict())
    remapped_state_dict = filter_state_dict_by_shape(remapped_state_dict, model.state_dict())
    missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
    if missing_keys:
        print(f"[Checkpoint Load] missing keys initialized from scratch: {missing_keys}")
    if unexpected_keys:
        print(f"[Checkpoint Load] unexpected keys ignored: {unexpected_keys}")
    return metadata


def count_parameters(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params, trainable_params


def compute_batch_metrics(preds, targets, cldice_metric, eps=1e-6):
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    pred_sum = preds.sum(dim=(1, 2, 3))
    target_sum = targets.sum(dim=(1, 2, 3))
    union = pred_sum + target_sum - intersection

    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    precision = (intersection + eps) / (pred_sum + eps)
    recall = (intersection + eps) / (target_sum + eps)

    # 总指标仍然按帧级平均，与之前保持一致。
    cldice_score = 1.0 - cldice_metric(preds.float(), targets.float())

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "cldice": cldice_score.item(),
    }


def compute_sequence_metrics(preds_seq, targets_seq, cldice_metric, eps=1e-6):
    # preds_seq / targets_seq: [T, 1, H, W]
    intersection = (preds_seq * targets_seq).sum(dim=(1, 2, 3))
    pred_sum = preds_seq.sum(dim=(1, 2, 3))
    target_sum = targets_seq.sum(dim=(1, 2, 3))
    union = pred_sum + target_sum - intersection

    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    precision = (intersection + eps) / (pred_sum + eps)
    recall = (intersection + eps) / (target_sum + eps)
    cldice_score = 1.0 - cldice_metric(preds_seq.float(), targets_seq.float())

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "cldice": cldice_score.item(),
    }


def extract_seg_logits(model_output):
    if isinstance(model_output, dict):
        return model_output["seg"]
    return model_output


def to_gray_u8(img_tensor):
    img = img_tensor.cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def to_mask_u8(mask_tensor):
    mask = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8) * 255
    return mask


def build_sequence_preview(images_seq, masks_seq, preds_seq):
    # 每一帧做成 3 行：原图 / GT / Pred，所有时间步横向拼起来。
    columns = []
    seq_len = images_seq.shape[0]

    for t in range(seq_len):
        image = to_gray_u8(images_seq[t, 0])
        gt_mask = to_mask_u8(masks_seq[t, 0])
        pred_mask = to_mask_u8(preds_seq[t, 0])

        image_rgb = np.stack([image, image, image], axis=-1)
        gt_rgb = np.stack([gt_mask, gt_mask, gt_mask], axis=-1)
        pred_rgb = np.stack([pred_mask, pred_mask, pred_mask], axis=-1)

        tile = np.concatenate([image_rgb, gt_rgb, pred_rgb], axis=0)
        columns.append(tile)

    return np.concatenate(columns, axis=1)


def save_sequence_preview(images_seq, masks_seq, preds_seq, save_path):
    canvas = build_sequence_preview(images_seq, masks_seq, preds_seq)
    cv2.imwrite(save_path, canvas)


def encode_sequence_preview(images_seq, masks_seq, preds_seq):
    canvas = build_sequence_preview(images_seq, masks_seq, preds_seq)
    success, encoded = cv2.imencode(".png", canvas)
    if not success:
        raise RuntimeError("Failed to encode preview image.")
    return encoded.tobytes()


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "fast_litenet":
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
    elif args.model == "unet_baseline":
        model = UNetBaseline(in_channels=1, num_classes=1, base_channels=32).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    checkpoint_meta = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    total_params, trainable_params = count_parameters(model)

    dataset = RealGuidewireVideoDataset(
        data_root=args.data_root,
        split=args.split,
        seq_len=args.seq_len,
        exclude_sequences_file=args.exclude_sequences_file,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    cldice_metric = Soft_clDiceLoss(iter_=15).to(device)

    metrics_sum = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "cldice": 0.0,
    }
    frame_count = 0

    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    save_dir = os.path.join(BASE_DIR, "experiments", "eval", f"{args.split}_{ckpt_name}")
    preview_dir = os.path.join(save_dir, "previews")
    worst_preview_dir = os.path.join(save_dir, f"worst_{args.worst_metric}")
    os.makedirs(preview_dir, exist_ok=True)
    os.makedirs(worst_preview_dir, exist_ok=True)

    saved_visualizations = 0
    per_sequence_records = []
    worst_cases = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Evaluating [{args.split}]")
        for batch_idx, (images_seq, masks_seq) in enumerate(pbar):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            images_seq = images_seq.to(device, non_blocking=True)
            masks_seq = masks_seq.to(device, non_blocking=True)
            batch_size, seq_len, _, _, _ = images_seq.shape

            feature_bank = None
            pred_seq_list = []
            pred_seq_device_list = []

            for t in range(seq_len):
                model_output_t, feature_bank = model(
                    images_seq[:, t, ...],
                    feature_bank=feature_bank,
                    reset_flag=(t == 0),
                )
                logits_t = extract_seg_logits(model_output_t)
                probs_t = torch.sigmoid(logits_t)
                preds_t = (probs_t > args.threshold).float()
                pred_seq_list.append(preds_t.cpu())
                pred_seq_device_list.append(preds_t)

                batch_metrics = compute_batch_metrics(preds_t, masks_seq[:, t, ...], cldice_metric)
                for key, value in batch_metrics.items():
                    metrics_sum[key] += value * batch_size
                frame_count += batch_size

            pred_seq_tensor = torch.stack(pred_seq_list, dim=1)
            pred_seq_device_tensor = torch.stack(pred_seq_device_list, dim=1)

            if saved_visualizations < args.max_visualizations:
                num_to_save = min(batch_size, args.max_visualizations - saved_visualizations)
                for i in range(num_to_save):
                    save_path = os.path.join(preview_dir, f"sample_{saved_visualizations:03d}.png")
                    save_sequence_preview(
                        images_seq[i].cpu(),
                        masks_seq[i].cpu(),
                        pred_seq_tensor[i],
                        save_path,
                    )
                    saved_visualizations += 1

            sample_start_idx = batch_idx * args.batch_size
            for i in range(batch_size):
                sample_idx = sample_start_idx + i
                sample_info = dataset.samples[sample_idx]
                sequence_record = {
                    "sample_index": sample_idx,
                    "seq_name": sample_info["seq_name"],
                    "frame_names": "|".join(sample_info["frame_names"]),
                }
                sequence_record.update(
                    compute_sequence_metrics(
                        pred_seq_device_tensor[i],
                        masks_seq[i],
                        cldice_metric,
                    )
                )
                per_sequence_records.append(sequence_record)

                if args.max_worst_visualizations > 0:
                    worst_case = {
                        **sequence_record,
                        "preview_png_bytes": encode_sequence_preview(
                            images_seq[i].cpu(),
                            masks_seq[i].cpu(),
                            pred_seq_tensor[i],
                        ),
                    }
                    worst_cases.append(worst_case)
                    worst_cases.sort(key=lambda item: item[args.worst_metric])
                    if len(worst_cases) > args.max_worst_visualizations:
                        worst_cases.pop()

    mean_metrics = {key: value / max(frame_count, 1) for key, value in metrics_sum.items()}

    per_sequence_path = os.path.join(save_dir, "per_sequence_metrics.csv")
    fieldnames = [
        "sample_index",
        "seq_name",
        "frame_names",
        "dice",
        "iou",
        "precision",
        "recall",
        "cldice",
    ]
    with open(per_sequence_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(per_sequence_records, key=lambda item: item[args.worst_metric]):
            writer.writerow(record)

    saved_worst_paths = []
    for rank, record in enumerate(sorted(worst_cases, key=lambda item: item[args.worst_metric]), start=1):
        filename = (
            f"rank_{rank:03d}_idx_{record['sample_index']:05d}_"
            f"{args.worst_metric}_{record[args.worst_metric]:.4f}_{record['seq_name']}.png"
        )
        save_path = os.path.join(worst_preview_dir, filename)
        with open(save_path, "wb") as f:
            f.write(record["preview_png_bytes"])
        saved_worst_paths.append(os.path.abspath(save_path))

    results = {
        "split": args.split,
        "model": args.model,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_meta": checkpoint_meta,
        "num_sequences": len(dataset),
        "num_frames_evaluated": frame_count,
        "threshold": args.threshold,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_size_mb_fp32": round(trainable_params * 4 / 1024 / 1024, 3),
        "metrics": mean_metrics,
        "preview_dir": os.path.abspath(preview_dir),
        "worst_metric": args.worst_metric,
        "worst_preview_dir": os.path.abspath(worst_preview_dir),
        "num_worst_visualizations_saved": len(saved_worst_paths),
        "per_sequence_metrics_csv": os.path.abspath(per_sequence_path),
    }

    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n===== Evaluation Summary =====")
    print(f"Split: {results['split']}")
    print(f"Model: {results['model']}")
    print(f"Checkpoint: {results['checkpoint']}")
    if checkpoint_meta:
        print(f"Checkpoint Epoch: {checkpoint_meta.get('epoch')}")
        print(f"Checkpoint Best Val Dice: {checkpoint_meta.get('best_val_dice')}")
    print(f"Sequences: {results['num_sequences']}")
    print(f"Frames Evaluated: {results['num_frames_evaluated']}")
    print(f"Params: {results['trainable_params']} ({results['param_size_mb_fp32']} MB fp32)")
    print(f"Dice: {mean_metrics['dice']:.4f}")
    print(f"IoU: {mean_metrics['iou']:.4f}")
    print(f"Precision: {mean_metrics['precision']:.4f}")
    print(f"Recall: {mean_metrics['recall']:.4f}")
    print(f"clDice: {mean_metrics['cldice']:.4f}")
    print(f"Preview Dir: {results['preview_dir']}")
    print(f"Worst Preview Dir: {results['worst_preview_dir']}")
    print(f"Per-sequence CSV: {results['per_sequence_metrics_csv']}")
    print(f"Metrics JSON: {os.path.abspath(metrics_path)}")


if __name__ == "__main__":
    evaluate(parse_args())
