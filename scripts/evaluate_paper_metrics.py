#!/usr/bin/env python
"""Evaluate trained FreqDecGWNet with PAPER-GRADE metrics (Metric A + B).

This is what we report in the §IV experiments — supersedes the tip-only
``improvement_mean`` from evaluate_drift.py (which conflates respiratory
drift with the operator's active wire push).

Per Shechter et al. 2004 TMI, Ablitt et al. 2004, Ambrosini et al. 2015,
the field-standard evaluation for fluoroscopy motion compensation is:

    (A) Respiratory-projected residual on landmarks, NOT raw TRE.
    (B) Background patch alignment (Shi-Tomasi corners outside the wire).

The script also computes the old tip residual for backward-compat in a
separate column so we can clearly show readers that "the wrong metric
hid the bias" — that becomes an honest §IV.A note in the paper.

Output:
    {output_dir}/per_sequence.csv  — one row per (ckpt, split, reset, seq)
    {output_dir}/summary.csv        — aggregated stats per (ckpt × reset)

Usage:
    python scripts/evaluate_paper_metrics.py \\
        --ckpts experiments/checkpoints/stage3_{rel_zm,abs_zm,rel_nozm}_v5_best.pth \\
        --splits train val test \\
        --reset-every 0 16 32 \\
        --output-dir reports/paper_v6_metrics
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freqdec_gwnet.data import (                                       # noqa: E402
    DataPaths, FluoroSequenceLoader,
)
from freqdec_gwnet.data.state_labels import (                          # noqa: E402
    estimate_background_translation, cumulative_position,
    project_to_principal_axis,
)
from freqdec_gwnet.models.freqdec_gwnet import FreqDecGWNet            # noqa: E402
from freqdec_gwnet.utils.background_patch_tracking import (            # noqa: E402
    detect_background_features, patch_alignment_residual,
    track_features_across_burst,
)
from freqdec_gwnet.utils.drift_metrics import (                        # noqa: E402
    landmark_residual, respiratory_projected_residual,
)
from freqdec_gwnet.utils.reference_selector import ReferenceSelector   # noqa: E402


def _apply_reset(U: torch.Tensor, reset: int) -> torch.Tensor:
    if reset is None or reset <= 0:
        return U
    T = U.shape[0]
    # Recover delta_u from U (numerical re-cumulation: Δu_0 = 0; Δu_t = U_t - U_{t-1})
    delta_u = torch.zeros_like(U)
    delta_u[1:] = U[1:] - U[:-1]
    out = torch.zeros_like(U)
    for s in range(0, T, reset):
        e = min(T, s + reset)
        chunk = delta_u[s:e].clone()
        chunk[0] = 0.0
        out[s:e] = torch.cumsum(chunk, dim=0)
    return out


def _resp_axis_from_phase_b(
    images_uint8: np.ndarray, wire_mask_uint8: Optional[np.ndarray],
) -> np.ndarray:
    """Use Phase B's PCA axis as the respiratory axis."""
    deltas, _ = estimate_background_translation(images_uint8, wire_masks=wire_mask_uint8)
    pos = cumulative_position(deltas)
    _, axis, _ = project_to_principal_axis(pos)
    return axis


def evaluate_one(
    model: Optional[FreqDecGWNet],
    paths: DataPaths,
    dataset: str, split: str, seq_name: str,
    reset_every: int,
    device: str,
) -> dict:
    loader = FluoroSequenceLoader(paths)
    seq = loader.load_sequence(dataset, split, seq_name)
    T = seq.num_frames
    images_uint8 = (seq.frames[:, 0].cpu().numpy() * 255).astype(np.uint8)
    wire_mask_uint8 = None
    if seq.masks is not None:
        wire_mask_uint8 = (seq.masks[:, 0].cpu().numpy() > 0.5).astype(np.uint8)

    # Reference frame
    selector = ReferenceSelector(warmup_min=5, warmup_max=10)
    warmup_imgs = seq.frames[: 10, 0]
    ref = selector.select(images=warmup_imgs, total_frames=T)

    # Model forward — full burst
    if model is None:
        # Zero baseline (no compensation)
        U = torch.zeros(T, 2)
    else:
        with torch.no_grad():
            x = seq.frames.unsqueeze(0).to(device)            # [1, T, 1, H, W]
            out = model(x, mode="stage3_joint", reference_index=ref.reference_index)
            U = out["U"][0].cpu()

    # Apply reset
    U = _apply_reset(U, reset_every)

    # Respiratory axis from Phase B
    resp_axis = _resp_axis_from_phase_b(images_uint8, wire_mask_uint8)
    resp_axis_t = torch.from_numpy(resp_axis).float()

    # ---- Metric A: respiratory-projected residual on tip ----
    metric_a = None
    if seq.tip_xy is not None and bool(seq.tip_present[0]):
        lm = seq.tip_xy.clone()
        miss = ~seq.tip_present
        lm[miss] = float("nan")
        try:
            ra = respiratory_projected_residual(
                lm, U, reference_index=ref.reference_index,
                respiratory_axis=resp_axis_t,
            )
            metric_a = {
                "before": float(ra["mean_before"].item()),
                "after": float(ra["mean_after"].item()),
                "improvement": float(ra["improvement_mean"].item()),
                "reduction": float(ra["reduction_ratio"].item()),
            }
        except (ValueError, IndexError):
            metric_a = None

    # ---- Metric B: background patch alignment ----
    metric_b = None
    ref_idx = ref.reference_index
    wire_ref = (
        wire_mask_uint8[ref_idx] if wire_mask_uint8 is not None
        else np.zeros(images_uint8.shape[1:], dtype=np.uint8)
    )
    corners = detect_background_features(
        images_uint8[ref_idx], wire_ref,
        max_features=40, wire_dilation_px=32,
    )
    if corners.shape[0] >= 4:
        track = track_features_across_burst(
            images_uint8, corners, reference_index=ref_idx,
        )
        pb = patch_alignment_residual(track, U.cpu().numpy(),
                                      reference_index=ref_idx)
        metric_b = {
            "before": pb["global_mean_before"],
            "after": pb["global_mean_after"],
            "improvement": pb["global_improvement_mean"],
            "reduction": pb["reduction_ratio"],
            "n_features": int(pb["n_features"]),
        }

    # ---- Legacy tip residual (the misleading one — kept for §IV.A note) ----
    legacy = None
    if seq.tip_xy is not None and bool(seq.tip_present[0]):
        lm = seq.tip_xy.clone()
        miss = ~seq.tip_present
        lm[miss] = float("nan")
        try:
            lr = landmark_residual(lm, U, reference_index=ref.reference_index)
            valid = bool(lr["valid"].any().item())
            if valid:
                legacy = float(lr["improvement_mean"].item())
        except (ValueError, IndexError):
            legacy = None

    return {
        "dataset": dataset, "split": split, "seq": seq_name, "T": T,
        "reset_every": reset_every,
        "reference_index": ref.reference_index,
        "reference_strategy": ref.strategy,
        "resp_axis_x": float(resp_axis[0]), "resp_axis_y": float(resp_axis[1]),
        "U_max": float(U.norm(dim=-1).max().item()),
        "metric_a_before": metric_a["before"] if metric_a else float("nan"),
        "metric_a_after": metric_a["after"] if metric_a else float("nan"),
        "metric_a_reduction": metric_a["reduction"] if metric_a else float("nan"),
        "metric_a_improvement": metric_a["improvement"] if metric_a else float("nan"),
        "metric_b_before": metric_b["before"] if metric_b else float("nan"),
        "metric_b_after": metric_b["after"] if metric_b else float("nan"),
        "metric_b_reduction": metric_b["reduction"] if metric_b else float("nan"),
        "metric_b_improvement": metric_b["improvement"] if metric_b else float("nan"),
        "metric_b_n_features": metric_b["n_features"] if metric_b else 0,
        "legacy_tip_improvement": legacy if legacy is not None else float("nan"),
    }


_COLUMNS = [
    "ckpt", "dataset", "split", "seq", "T", "reset_every",
    "reference_index", "reference_strategy",
    "resp_axis_x", "resp_axis_y", "U_max",
    "metric_a_before", "metric_a_after", "metric_a_reduction", "metric_a_improvement",
    "metric_b_before", "metric_b_after", "metric_b_reduction", "metric_b_improvement",
    "metric_b_n_features",
    "legacy_tip_improvement",
]


def _fmt(v) -> str:
    if isinstance(v, float):
        if v != v:
            return ""
        return f"{v:.6g}"
    return str(v)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset", default="dense_v1")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--seq", nargs="+", default=None,
                   help="restrict to specific seq names")
    p.add_argument("--ckpts", nargs="+", required=True, type=Path,
                   help="one or more FreqDecGWNet checkpoints")
    p.add_argument("--state-core-type", default="mamba",
                   choices=("gru", "mamba"))
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--motion-field-types", nargs="+",
                   default=None,
                   help="one type per ckpt; default infers from ckpt name "
                        "(contains 'abs' -> absolute else relative)")
    p.add_argument("--reset-every", nargs="+", type=int, default=[0, 16, 32])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--include-baseline", action="store_true",
                   help="also evaluate no-compensation baseline (U=0)")
    args = p.parse_args(list(argv) if argv is not None else None)

    paths = (
        DataPaths.from_yaml(args.config) if args.config
        else DataPaths.from_default_config()
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve motion field types
    if args.motion_field_types is None:
        motion_types = [
            "absolute" if "abs" in c.name.lower() else "relative"
            for c in args.ckpts
        ]
    else:
        if len(args.motion_field_types) != len(args.ckpts):
            raise SystemExit("--motion-field-types must match --ckpts length")
        motion_types = args.motion_field_types

    all_rows: List[dict] = []

    # Optional baseline: no compensation
    if args.include_baseline:
        print(f"[baseline] U=0 (no compensation)")
        for split in args.splits:
            seqs = paths.list_sequences(args.dataset, split)
            if args.seq:
                seqs = [s for s in seqs if s in args.seq]
            for sname in seqs:
                for reset in args.reset_every:
                    if reset != 0:
                        continue  # baseline doesn't depend on reset
                    row = evaluate_one(
                        None, paths, args.dataset, split, sname,
                        reset_every=0, device=device,
                    )
                    row["ckpt"] = "baseline_no_comp"
                    all_rows.append(row)

    # Each trained ckpt
    for ckpt_path, mtype in zip(args.ckpts, motion_types):
        if not ckpt_path.is_file():
            print(f"[skip] {ckpt_path} missing")
            continue
        print(f"[ckpt] {ckpt_path.name}  motion={mtype}")
        model = FreqDecGWNet(
            width_mult=args.width_mult,
            state_core_type=args.state_core_type,
            motion_field_type=mtype,
        ).to(device)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck)
        sd = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in sd.items()
        }
        model.load_state_dict(sd, strict=False)
        model.eval()

        for split in args.splits:
            seqs = paths.list_sequences(args.dataset, split)
            if args.seq:
                seqs = [s for s in seqs if s in args.seq]
            for sname in seqs:
                for reset in args.reset_every:
                    row = evaluate_one(
                        model, paths, args.dataset, split, sname,
                        reset_every=reset, device=device,
                    )
                    row["ckpt"] = ckpt_path.stem
                    all_rows.append(row)
                    print(
                        f"  {ckpt_path.stem[:32]:32s} {split:6s} {sname:18s} "
                        f"reset={reset:3d}  "
                        f"A_imp={row['metric_a_improvement']:+7.2f}  "
                        f"B_imp={row['metric_b_improvement']:+7.2f}  "
                        f"legacy={row['legacy_tip_improvement']:+7.2f}"
                    )

    # ---- Per-sequence CSV ----
    per_seq_path = args.output_dir / "per_sequence.csv"
    with per_seq_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_COLUMNS)
        for r in all_rows:
            w.writerow([_fmt(r.get(c, "")) for c in _COLUMNS])
    print(f"\n[csv] per_sequence: {per_seq_path}")

    # ---- Summary (mean per ckpt × split × reset) ----
    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        cols = ["ckpt", "split", "reset_every", "n_seqs",
                "A_red", "A_imp", "B_red", "B_imp", "legacy_imp"]
        w.writerow(cols)
        from collections import defaultdict
        agg = defaultdict(list)
        for r in all_rows:
            key = (r["ckpt"], r["split"], r["reset_every"])
            agg[key].append(r)
        for (ckpt, split, reset), rows in sorted(agg.items()):
            def m(field):
                vals = [r[field] for r in rows
                        if r.get(field) is not None and r[field] == r[field]]
                return sum(vals) / len(vals) if vals else float("nan")
            w.writerow([
                ckpt, split, reset, len(rows),
                _fmt(m("metric_a_reduction")), _fmt(m("metric_a_improvement")),
                _fmt(m("metric_b_reduction")), _fmt(m("metric_b_improvement")),
                _fmt(m("legacy_tip_improvement")),
            ])
    print(f"[csv] summary:      {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
