#!/usr/bin/env python
"""Oracle ceiling test — what's the upper bound of our task?

Uses Phase B's background motion estimate (pure geometric phase-correlation
on dilated background, no learned model involved) as the oracle U_t and
measures Metric A + B reduction ratios. This is the ceiling our learned
model could plausibly reach if it perfectly tracked the visible
background motion.

Three outcomes:

    * reduction_ratio  ≪ 1 (e.g. < 0.4):
        Background motion is a real and useful signal — our task is
        feasible and the model just needs better training.

    * reduction_ratio  ≈ 0.7-0.9:
        Marginally useful — even perfect bg-tracking only modestly
        helps because tip motion is dominated by active push.

    * reduction_ratio  ≈ 1.0 or > 1:
        The task as currently framed is essentially unsolvable on this
        data — bg motion does not predict tip drift, so we need to
        re-frame the paper around state estimation / background
        stability rather than tip-level compensation.

Usage::

    python scripts/oracle_ceiling_diagnostic.py \\
        --dataset dense_v1 --split train --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freqdec_gwnet.data import DataPaths, FluoroSequenceLoader            # noqa: E402
from freqdec_gwnet.data.state_labels import (                              # noqa: E402
    cumulative_position, estimate_background_translation,
    project_to_principal_axis,
)
from freqdec_gwnet.utils.background_patch_tracking import (                # noqa: E402
    detect_background_features, patch_alignment_residual,
    track_features_across_burst,
)
from freqdec_gwnet.utils.drift_metrics import respiratory_projected_residual  # noqa: E402


def evaluate_oracle(seq) -> dict:
    images = (seq.frames[:, 0].cpu().numpy() * 255).astype(np.uint8)
    T = images.shape[0]
    if T < 32:
        return {"seq": seq.seq_name, "T": T, "skipped": "too_short"}

    # ---- Phase B oracle: pure geometric background motion ----
    wire_masks_np = None
    if seq.masks is not None:
        wire_masks_np = (seq.masks[:, 0].cpu().numpy() > 0.5).astype(np.uint8)
    deltas, valid = estimate_background_translation(
        images, wire_masks=wire_masks_np,
    )
    U_oracle = cumulative_position(deltas)                  # [T, 2]
    proj_1d, axis, ratio = project_to_principal_axis(U_oracle)

    # ---- Metric A: respiratory-projected residual on tip ----
    metric_a = None
    if seq.tip_xy is not None and bool(seq.tip_present[0]):
        lm = seq.tip_xy.clone()
        miss = ~seq.tip_present
        lm[miss] = float("nan")
        metric_a = respiratory_projected_residual(
            lm,
            torch.from_numpy(U_oracle),
            reference_index=0,
            respiratory_axis=torch.from_numpy(axis),
        )

    # ---- Metric B: background patch alignment ----
    wire_ref = (
        wire_masks_np[0] if wire_masks_np is not None
        else np.zeros(images.shape[1:], dtype=np.uint8)
    )
    corners = detect_background_features(
        images[0], wire_ref, max_features=40, wire_dilation_px=32,
    )
    metric_b = None
    if corners.shape[0] >= 4:
        track = track_features_across_burst(
            images, corners, reference_index=0,
        )
        metric_b = patch_alignment_residual(track, U_oracle, reference_index=0)

    return {
        "seq": seq.seq_name,
        "T": T,
        "axis": axis.tolist(),
        "pca_ratio": float(ratio),
        "U_oracle_max": float(np.linalg.norm(U_oracle, axis=-1).max()),
        "U_oracle_mean": float(np.linalg.norm(U_oracle, axis=-1).mean()),
        "metric_a": metric_a,
        "metric_b": metric_b,
    }


def _format_metric(r) -> str:
    a = r.get("metric_a")
    b = r.get("metric_b")
    parts = [f"T={r['T']:3d}", f"axis=({r['axis'][0]:+.2f},{r['axis'][1]:+.2f})",
             f"PCA={r['pca_ratio']:.2f}",
             f"U_max={r['U_oracle_max']:.1f}"]
    if a is not None:
        parts.append(
            f"A: before={a['mean_before'].item():5.2f}  "
            f"after={a['mean_after'].item():5.2f}  "
            f"reduction={a['reduction_ratio'].item():.3f}"
        )
    if b is not None:
        parts.append(
            f"B: before={b['global_mean_before']:5.2f}  "
            f"after={b['global_mean_after']:5.2f}  "
            f"reduction={b['reduction_ratio']:.3f}"
        )
    return "  ".join(parts)


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Phase-B oracle ceiling test — is the task solvable at all?",
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", required=True, choices=("train", "val", "test"))
    p.add_argument("--seq", nargs="+", default=None)
    p.add_argument("--all", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    paths = (
        DataPaths.from_yaml(args.config) if args.config
        else DataPaths.from_default_config()
    )
    seqs = args.seq if args.seq else paths.list_sequences(args.dataset, args.split)
    if args.all:
        seqs = paths.list_sequences(args.dataset, args.split)

    loader = FluoroSequenceLoader(paths)
    print(f"[oracle] dataset={args.dataset} split={args.split} sequences={len(seqs)}")

    rows = []
    for sname in seqs:
        seq = loader.load_sequence(args.dataset, args.split, sname)
        r = evaluate_oracle(seq)
        rows.append(r)
        if "skipped" in r:
            print(f"  {sname:18s}  SKIPPED ({r['skipped']})")
        else:
            print(f"  {sname:18s}  {_format_metric(r)}")

    # ---- Aggregate verdict ----
    a_reductions = [r["metric_a"]["reduction_ratio"].item()
                    for r in rows
                    if r.get("metric_a") is not None]
    b_reductions = [r["metric_b"]["reduction_ratio"]
                    for r in rows
                    if r.get("metric_b") is not None
                    and not np.isnan(r["metric_b"]["reduction_ratio"])]
    if a_reductions:
        a_mean = sum(a_reductions) / len(a_reductions)
        print(f"\n[oracle aggregate] Metric A mean reduction = {a_mean:.3f}  "
              f"(< 0.5 = feasible, > 0.9 = task infeasible)")
    if b_reductions:
        b_mean = sum(b_reductions) / len(b_reductions)
        print(f"[oracle aggregate] Metric B mean reduction = {b_mean:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
