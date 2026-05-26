#!/usr/bin/env python
"""Stage-2 state-branch quality evaluator (paper §IV.C positive endpoint).

Computes per-sequence and aggregated metrics on the held-out split:

    phase_MAE_deg       — circular mean absolute error of the predicted
                          phase angle vs the Phase B pseudo-label, in
                          degrees. Computed via arccos(<pred · gt>) so
                          wrap-around is handled correctly.
    amplitude_RMSE_px   — RMSE of predicted amplitude vs Phase B
                          pseudo-label amplitude.
    amplitude_NRMSE     — amplitude RMSE normalized by per-sequence
                          P95(amplitude) (=amp_scale) to make the
                          metric scale-invariant across patients.
    n_valid_frames      — number of frames with v_t = 1 used.

Reports a per-sequence CSV plus a split-level mean row, suitable for
the paper's §IV.C row showing the ECG-free state estimator is
quantitatively sound regardless of the §IV.D compensation outcome.

Usage::

    python scripts/evaluate_state.py \\
        --data-root /workspace/shz/clean_data \\
        --state-labels-dir /workspace/FreqDec-GWNet/reports/state_labels/clean_v2 \\
        --ckpt experiments/checkpoints/stage2_mamba_chrono_v5_best.pth \\
        --state-core-type mamba \\
        --split val \\
        --output-csv reports/paper_v5/state_eval_mamba_v5_val.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freqdec_gwnet.data import FluoroSequenceWindowDataset             # noqa: E402
from freqdec_gwnet.data.state_labels.pipeline import StateLabel        # noqa: E402
from freqdec_gwnet.models.freqdec_gwnet import FreqDecGWNet            # noqa: E402


def _circular_mae_deg(
    cos_pred: np.ndarray, sin_pred: np.ndarray,
    cos_gt: np.ndarray, sin_gt: np.ndarray,
    valid: np.ndarray,
) -> float:
    """Mean abs phase angle error, handling 0/2π wrap correctly.

    For two unit vectors e_pred=(cos_p, sin_p) and e_gt=(cos_g, sin_g),
    the angular distance is acos(clamp(<e_pred · e_gt>, -1, 1)).
    """
    if not valid.any():
        return float("nan")
    norm_p = np.sqrt(cos_pred ** 2 + sin_pred ** 2).clip(min=1e-8)
    cp = cos_pred / norm_p
    sp = sin_pred / norm_p
    dot = (cp * cos_gt + sp * sin_gt).clip(-1.0, 1.0)
    err_rad = np.arccos(dot)
    return float(np.degrees(err_rad[valid].mean()))


def _amp_metrics(
    amp_pred: np.ndarray, amp_gt: np.ndarray,
    amp_scale: float, valid: np.ndarray,
) -> tuple:
    if not valid.any():
        return float("nan"), float("nan")
    err = amp_pred - amp_gt
    rmse = float(np.sqrt((err[valid] ** 2).mean()))
    nrmse = rmse / max(amp_scale, 1e-6)
    return rmse, nrmse


@torch.no_grad()
def evaluate_one_sequence(
    model, dataset, seq_name: str, device: str, T_window: int,
) -> dict:
    """Run model.forward_window-style inference on the full burst and
    compare to the StateLabel saved by Phase B.

    Because the dataset returns chunked windows, we collect all windows
    of the same sequence and concatenate their non-overlapping portions.
    The simplest version: just take the windows whose start indices form
    a non-overlapping cover (the dataset uses stride < T_window so most
    frames appear in 2 windows). To stay honest we evaluate only the
    *first occurrence* of each frame across the burst.
    """
    # Find windows for this seq
    window_indices = [
        i for i, w in enumerate(dataset.windows) if w.seq_name == seq_name
    ]
    if not window_indices:
        return {"seq_name": seq_name, "skipped": "no_windows"}

    # Aggregate per-frame predictions taking earliest window vote
    sl_path = dataset.windows[window_indices[0]].state_label_path
    sl = StateLabel.load_npz(sl_path)
    T_total = sl.num_frames
    pred_cos = np.full(T_total, np.nan, dtype=np.float32)
    pred_sin = np.full(T_total, np.nan, dtype=np.float32)
    pred_amp = np.full(T_total, np.nan, dtype=np.float32)

    for wi in window_indices:
        sample = dataset[wi]
        images = sample["images"].unsqueeze(0).to(device)            # [1, T, 1, H, W]
        out = model(images, mode="stage2_state")
        state = out["state"][0].cpu().numpy()                         # [T, 4]
        win_start = int(sample["window_start"])
        for k in range(T_window):
            gi = win_start + k
            if gi >= T_total:
                break
            if np.isnan(pred_cos[gi]):           # first occurrence wins
                pred_cos[gi] = state[k, 0]
                pred_sin[gi] = state[k, 1]
                pred_amp[gi] = state[k, 2]

    valid = sl.valid_mask & ~np.isnan(pred_cos)
    phase_mae = _circular_mae_deg(
        pred_cos, pred_sin, sl.cos_phi, sl.sin_phi, valid,
    )
    amp_rmse, amp_nrmse = _amp_metrics(
        pred_amp, sl.amplitude, sl.amp_scale, valid,
    )
    return {
        "seq_name": seq_name,
        "T_total": T_total,
        "n_valid_frames": int(valid.sum()),
        "phase_MAE_deg": phase_mae,
        "amplitude_RMSE_px": amp_rmse,
        "amplitude_NRMSE": amp_nrmse,
        "amp_scale_px": sl.amp_scale,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Stage-2 state branch evaluator (paper §IV.C)",
    )
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--state-labels-dir", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--state-core-type",
                   choices=("gru", "mamba"), default="mamba")
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--T-window", type=int, default=64)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--output-csv", type=Path, default=None)

    args = p.parse_args(list(argv) if argv is not None else None)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    dataset = FluoroSequenceWindowDataset(
        data_root=args.data_root, state_labels_dir=args.state_labels_dir,
        split=args.split, T_window=args.T_window, stride=args.stride,
        img_size=(args.img_size, args.img_size),
    )

    model = FreqDecGWNet(
        width_mult=args.width_mult,
        state_core_type=args.state_core_type,
    ).to(device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    sd = {(k[len("module."):] if k.startswith("module.") else k): v
          for k, v in sd.items()}
    miss, unex = model.load_state_dict(sd, strict=False)
    print(f"[evaluate_state] loaded {args.ckpt.name}  "
          f"missing={len(miss)} unexpected={len(unex)}")
    model.eval()

    seqs = sorted({w.seq_name for w in dataset.windows})
    rows = []
    for sname in seqs:
        r = evaluate_one_sequence(
            model, dataset, sname, device=device, T_window=args.T_window,
        )
        rows.append(r)
        if "skipped" in r:
            print(f"  {sname:18s}  SKIPPED ({r['skipped']})")
        else:
            print(f"  {sname:18s}  phase_MAE={r['phase_MAE_deg']:6.2f} deg  "
                  f"amp_RMSE={r['amplitude_RMSE_px']:5.2f} px  "
                  f"amp_NRMSE={r['amplitude_NRMSE']:.3f}  "
                  f"n_valid={r['n_valid_frames']}/{r['T_total']}")

    # Aggregate
    valid_rows = [r for r in rows if "skipped" not in r
                  and not math.isnan(r["phase_MAE_deg"])]
    if valid_rows:
        agg_phase = sum(r["phase_MAE_deg"] for r in valid_rows) / len(valid_rows)
        agg_rmse = sum(r["amplitude_RMSE_px"] for r in valid_rows) / len(valid_rows)
        agg_nrmse = sum(r["amplitude_NRMSE"] for r in valid_rows) / len(valid_rows)
        print(f"\n[aggregate {args.split}]  "
              f"phase_MAE = {agg_phase:5.2f}°  "
              f"amp_RMSE = {agg_rmse:5.2f} px  "
              f"amp_NRMSE = {agg_nrmse:.3f}  "
              f"({len(valid_rows)} sequences)")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as fh:
            cols = ["seq_name", "T_total", "n_valid_frames",
                    "phase_MAE_deg", "amplitude_RMSE_px",
                    "amplitude_NRMSE", "amp_scale_px"]
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                if "skipped" in r:
                    continue
                w.writerow({c: r.get(c, "") for c in cols})
        print(f"📄 wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
