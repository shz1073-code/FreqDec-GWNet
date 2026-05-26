#!/usr/bin/env python
"""Stage-2 state estimation quality (paper §IV.C — positive endpoint).

Compares the state branch's predicted physiological state against the
Phase-B weak-supervision label, on a per-sequence basis. Reports:

    * phase_mae_rad     — circular MAE between predicted and label phase
    * phase_pearson     — Pearson correlation of (cos pred, cos gt)
    * amp_mae_norm      — |A_pred - A_gt| / A_scale, mean over valid frames
    * amp_rmse_norm     — sqrt(mean(((A_pred - A_gt) / A_scale)^2))
    * frac_valid        — fraction of frames that passed the label's valid_mask

This is the clean "positive endpoint" the paper §III-B claim relies on
(GPT's recommendation): it is independent of the motion-field
controversy — high state quality stands as a contribution on its own.

Used for:
    * §IV.C "state estimation quality" table
    * Mamba vs GRU comparison (run with two different stage-2 ckpts)

Usage:
    python scripts/evaluate_state_quality.py \\
        --ckpts experiments/checkpoints/stage2_mamba_chrono_v5_best.pth \\
                experiments/checkpoints/stage2_gru_chrono_v5_best.pth \\
        --state-labels-dir reports/state_labels/clean_v2 \\
        --data-root /workspace/shz/clean_data \\
        --splits train val \\
        --output-dir reports/paper_v6_state_quality
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freqdec_gwnet.data import FluoroSequenceWindowDataset             # noqa: E402
from freqdec_gwnet.data.state_labels import StateLabel                 # noqa: E402
from freqdec_gwnet.models.freqdec_gwnet import FreqDecGWNet            # noqa: E402


def _circular_mae(cos_pred, sin_pred, cos_gt, sin_gt, valid):
    """Mean unsigned angular distance over valid frames (radians)."""
    # Cosine of angular difference: (cos_p cos_g + sin_p sin_g)
    cos_diff = (cos_pred * cos_gt + sin_pred * sin_gt).clamp(-1.0, 1.0)
    angular = torch.acos(cos_diff)                            # [T] in [0, π]
    if not valid.any():
        return float("nan")
    return float(angular[valid].mean().item())


def _phase_pearson(cos_pred, cos_gt, valid):
    """Pearson correlation between cos_pred and cos_gt on valid frames."""
    if not valid.any() or valid.sum() < 4:
        return float("nan")
    x = cos_pred[valid].float()
    y = cos_gt[valid].float()
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.norm() * y.norm()).clamp(min=1e-8)
    return float((x @ y / denom).item())


def _amp_stats(amp_pred, amp_gt, amp_scale, valid):
    """Normalized amplitude MAE and RMSE."""
    if not valid.any():
        return float("nan"), float("nan")
    diff = (amp_pred - amp_gt) / max(amp_scale, 1e-6)
    diff = diff[valid].float()
    mae = float(diff.abs().mean().item())
    rmse = float((diff ** 2).mean().sqrt().item())
    return mae, rmse


def evaluate_one(
    model: FreqDecGWNet,
    seq_window_ds: FluoroSequenceWindowDataset,
    seq_name: str,
    burn_in_frames: int,
    device: str,
) -> Optional[dict]:
    """Build a full-burst window for seq_name + forward + compute stats."""
    # Find the windows for this sequence in the dataset
    matching = [w for w in seq_window_ds.windows if w.seq_name == seq_name]
    if not matching:
        return None
    # Stitch the full-burst forward by concatenating overlapping windows is
    # complex; simpler: load the full burst directly from the StateLabel + images.
    state_label = StateLabel.load_npz(matching[0].state_label_path)
    T = state_label.num_frames

    # Load all frames
    import cv2
    H, W = seq_window_ds.img_size
    images = np.zeros((T, 1, H, W), dtype=np.float32)
    for k, fname in enumerate(matching[0].frame_names[:T]):
        img_path = matching[0].seq_dir_images / fname
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if (img.shape[0], img.shape[1]) != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        images[k, 0] = img.astype(np.float32) / 255.0
    images_t = torch.from_numpy(images)                              # [T, 1, H, W]

    # Encoder is chunked (no temporal dependency) to avoid OOM on long bursts.
    # State branch then forward_window on the FULL concatenated 1/8 feature
    # stack — this preserves the Mamba/GRU temporal carry that the model
    # was trained under (chronological_carry).
    T_chunk = 32
    feat_chunks = []
    inner = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for s in range(0, T, T_chunk):
            e = min(T, s + T_chunk)
            x = images_t[s:e].to(device)                              # [≤T_chunk, 1, H, W]
            feats = inner.r1.encoder(x)                               # 5-tuple
            feat_18 = feats[2].cpu()                                  # [≤T_chunk, C, H/8, W/8]
            feat_chunks.append(feat_18)
            del x, feats, feat_18
            if device == "cuda":
                torch.cuda.empty_cache()
    feat_18_seq = torch.cat(feat_chunks, dim=0).unsqueeze(0).to(device)  # [1, T, C, H/8, W/8]
    with torch.no_grad():
        state_pred = inner.state_branch.forward_window(feat_18_seq)[0].cpu()  # [T, 4]
    del feat_18_seq
    if device == "cuda":
        torch.cuda.empty_cache()
    cos_pred = state_pred[:, 0]
    sin_pred = state_pred[:, 1]
    # Normalize to unit circle (matching StateLoss's F.normalize)
    norm = torch.sqrt(cos_pred ** 2 + sin_pred ** 2).clamp(min=1e-6)
    cos_pred = cos_pred / norm
    sin_pred = sin_pred / norm
    amp_pred = state_pred[:, 2]

    # GT from label
    cos_gt = torch.from_numpy(state_label.cos_phi)
    sin_gt = torch.from_numpy(state_label.sin_phi)
    amp_gt = torch.from_numpy(state_label.amplitude)
    valid_gt = torch.from_numpy(state_label.valid_mask)

    # Exclude burn-in frames from evaluation (state branch is warming up there)
    valid_eval = valid_gt.clone()
    valid_eval[:burn_in_frames] = False

    phase_mae = _circular_mae(cos_pred, sin_pred, cos_gt, sin_gt, valid_eval)
    pearson = _phase_pearson(cos_pred, cos_gt, valid_eval)
    amp_mae, amp_rmse = _amp_stats(amp_pred, amp_gt, state_label.amp_scale, valid_eval)

    return {
        "seq": seq_name,
        "T": T,
        "n_valid": int(valid_eval.sum().item()),
        "frac_valid": float(valid_eval.float().mean().item()),
        "amp_scale": float(state_label.amp_scale),
        "phase_mae_rad": phase_mae,
        "phase_mae_deg": phase_mae * 180.0 / math.pi if phase_mae == phase_mae else float("nan"),
        "phase_pearson": pearson,
        "amp_mae_norm": amp_mae,
        "amp_rmse_norm": amp_rmse,
    }


_COLUMNS = [
    "ckpt", "split", "seq", "T", "n_valid", "frac_valid", "amp_scale",
    "phase_mae_rad", "phase_mae_deg", "phase_pearson",
    "amp_mae_norm", "amp_rmse_norm",
]


def _fmt(v):
    if isinstance(v, float):
        if v != v:
            return ""
        return f"{v:.6g}"
    return str(v)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--state-labels-dir", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--ckpts", nargs="+", type=Path, required=True)
    p.add_argument("--state-core-types", nargs="+", default=None,
                   help="one per ckpt; default infers from ckpt name "
                        "('mamba' / 'gru'). The stage-2/3 ckpt's state "
                        "core must match.")
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--burn-in-frames", type=int, default=16,
                   help="exclude this many leading frames per sequence "
                        "(state branch warm-up)")
    p.add_argument("--T-window", type=int, default=64,
                   help="for dataset construction; full burst is processed")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--include-sequences-file", type=Path, default=None,
                   help="text file, one sequence name per line; restrict "
                        "evaluation to these sequences (for k-fold CV)")
    args = p.parse_args(list(argv) if argv is not None else None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    include_seqs = None
    if args.include_sequences_file:
        include_seqs = [
            ln.strip() for ln in args.include_sequences_file.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]

    # Resolve state core type
    cores = (
        args.state_core_types if args.state_core_types
        else ["gru" if "gru" in c.name.lower() else "mamba" for c in args.ckpts]
    )
    if len(cores) != len(args.ckpts):
        raise SystemExit("--state-core-types must match --ckpts length")

    rows: List[dict] = []
    for ckpt, core in zip(args.ckpts, cores):
        if not ckpt.is_file():
            print(f"[skip] {ckpt} missing")
            continue
        print(f"[ckpt] {ckpt.name}  state_core={core}")
        model = FreqDecGWNet(
            width_mult=args.width_mult, state_core_type=core,
        ).to(device)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck)
        sd = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in sd.items()
        }
        model.load_state_dict(sd, strict=False)
        model.eval()

        for split in args.splits:
            sl_dir = args.state_labels_dir / split
            if not sl_dir.is_dir():
                # If users point at parent dir (clean_v2), descend automatically
                if not (args.state_labels_dir / f"{split}").is_dir():
                    print(f"  no state labels for split={split}")
                    continue
            ds = FluoroSequenceWindowDataset(
                data_root=args.data_root, state_labels_dir=args.state_labels_dir,
                split=split, T_window=args.T_window,
                img_size=(args.img_size, args.img_size),
                included_sequences=include_seqs,
            )
            seq_names = sorted({w.seq_name for w in ds.windows})
            for sname in seq_names:
                r = evaluate_one(model, ds, sname, args.burn_in_frames, device)
                if r is None:
                    continue
                r["ckpt"] = ckpt.stem
                r["split"] = split
                rows.append(r)
                print(
                    f"  {ckpt.stem[:30]:30s} {split:6s} {sname:18s} "
                    f"T={r['T']:3d} valid={r['frac_valid']:.2f}  "
                    f"phase_MAE={r['phase_mae_deg']:5.1f}deg  "
                    f"phase_r={r['phase_pearson']:+.3f}  "
                    f"amp_MAE={r['amp_mae_norm']:.3f}"
                )

    out = args.output_dir / "state_quality.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_COLUMNS)
        for r in rows:
            w.writerow([_fmt(r.get(c, "")) for c in _COLUMNS])
    print(f"\n[csv] {out}")

    # Aggregate per (ckpt, split)
    from collections import defaultdict
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["ckpt"], r["split"])].append(r)
    summary_path = args.output_dir / "state_quality_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ckpt", "split", "n_seqs",
                    "phase_MAE_deg_mean", "phase_MAE_deg_std",
                    "phase_pearson_mean",
                    "amp_MAE_norm_mean", "amp_RMSE_norm_mean",
                    "frac_valid_mean"])
        for (ckpt, split), rr in sorted(by_key.items()):
            def m(field):
                vals = [r[field] for r in rr if r.get(field) is not None
                        and r[field] == r[field]]
                return sum(vals) / len(vals) if vals else float("nan")
            def s(field):
                vals = [r[field] for r in rr if r.get(field) is not None
                        and r[field] == r[field]]
                if len(vals) < 2:
                    return float("nan")
                mu = sum(vals) / len(vals)
                return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))
            w.writerow([ckpt, split, len(rr),
                        _fmt(m("phase_mae_deg")), _fmt(s("phase_mae_deg")),
                        _fmt(m("phase_pearson")),
                        _fmt(m("amp_mae_norm")), _fmt(m("amp_rmse_norm")),
                        _fmt(m("frac_valid"))])
    print(f"[csv] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
