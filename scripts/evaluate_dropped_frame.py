#!/usr/bin/env python
"""Dropped-frame stress test (paper §6.2 + tech route §3).

For each sequence × drop_rate combination, report four metrics from the
technical route §3.2:

    * phase_correlation  — Pearson correlation between full-sequence phase
                           and the dropped-then-reinterpreted phase at the
                           kept frames. Higher = state branch tracks
                           respiration despite missing frames.
    * centerline_jitter  — placeholder until the wire centerline extractor
                           is wired into the eval pipeline. We currently
                           record per-frame Δ‖U‖ as a structural proxy.
    * compensation_error — drift-residual at the dropped frames after
                           re-running the cumulative sum on the kept-only
                           subset.
    * drift_reduction    — ratio of cumulative drift between full vs
                           dropped runs; near 1.0 means dropping does not
                           amplify drift.

Output:
    reports/dropped_frame/{seq}.csv  (one row per drop_rate × strategy)
    reports/dropped_frame/{seq}_summary.csv (aggregated rows)

This is the *framework* — until the model is trained the placeholder state
generator is reused, so the absolute numbers are meaningless. After
stage-2/3 training, plug a real forward pass into ``run_one_sequence``
and the same script becomes a §6.2 result table.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freqdec_gwnet.data import (                                     # noqa: E402
    DataPaths, FluoroSequence, FluoroSequenceLoader,
)
from freqdec_gwnet.models.freqdec_gwnet import FreqDecGWNet          # noqa: E402
from freqdec_gwnet.models.relative_motion_field import (             # noqa: E402
    RelativeMotionField,
)
from freqdec_gwnet.utils.drift_metrics import (                      # noqa: E402
    cumulative_drift_curve,
)
from freqdec_gwnet.utils.frame_dropping import (                     # noqa: E402
    DropPlan, apply_drop_plan, downsample_fps, random_drop_rate,
    remap_reference_index,
)
from freqdec_gwnet.utils.reference_selector import (                 # noqa: E402
    ReferenceSelector,
)


# ---------------------------------------------------------------------------
# Stress-test core
# ---------------------------------------------------------------------------


def _placeholder_state(T: int, mode: str = "sinusoidal") -> torch.Tensor:
    """Same placeholder generator as evaluate_drift.py — keeps the two
    scripts comparable until the trained state branch is plugged in.
    """
    if mode == "zero":
        return torch.zeros(1, T, 3)
    if mode == "sinusoidal":
        t = torch.linspace(0.0, 2 * math.pi, T)
        return torch.stack(
            [torch.cos(t), torch.sin(t), torch.full_like(t, 0.5)],
            dim=-1,
        ).unsqueeze(0)
    if mode == "random":
        g = torch.Generator().manual_seed(0)
        return torch.randn(1, T, 3, generator=g)
    raise ValueError(f"unknown state-source mode '{mode}'")


def _phase_correlation(
    cos_full: torch.Tensor, sin_full: torch.Tensor,
    cos_kept: torch.Tensor, sin_kept: torch.Tensor,
    kept_indices: np.ndarray,
) -> float:
    """Pearson correlation between full-burst phase and kept-only phase
    *at the kept indices*. Both inputs are unit-circle (cos, sin) pairs;
    we compare ``arctan2`` after circular alignment."""
    if cos_kept.shape[0] < 4:
        return float("nan")
    phi_full = torch.atan2(sin_full[kept_indices], cos_full[kept_indices])
    phi_kept = torch.atan2(sin_kept, cos_kept)
    # 球状均值对齐
    delta = torch.atan2(
        torch.sin(phi_full - phi_kept).mean(),
        torch.cos(phi_full - phi_kept).mean(),
    )
    phi_kept_aligned = phi_kept + delta
    # 直接 cos 距离：1 = 完美对齐
    return float(torch.cos(phi_full - phi_kept_aligned).mean().item())


def _drift_at_dropped_frames(
    U_full: torch.Tensor,                 # [T_full, 2]
    U_kept: torch.Tensor,                 # [T_kept, 2]
    plan: DropPlan,
) -> float:
    """Mean ‖U_full − U_kept_at_neighbour‖ at *dropped* frames.

    For each dropped frame, we compare the full-sequence U value at that
    frame against the kept run's U at the **nearest kept frame** (left
    neighbour). This is the residual a downstream system would experience
    if it tried to reconstruct the dropped frame's compensation from the
    kept-only model.
    """
    if (~plan.keep_mask).sum() == 0:
        return 0.0
    drops = np.where(~plan.keep_mask)[0]
    # 对每个被 drop 的 t，找最近的 kept_indices 中 ≤ t 的索引
    insert = np.searchsorted(plan.kept_indices, drops, side="right") - 1
    insert = np.clip(insert, 0, plan.T_kept - 1)
    diffs = U_full[drops] - U_kept[insert]
    return float(diffs.norm(dim=-1).mean().item())


def _drift_reduction(
    U_full: torch.Tensor, U_kept: torch.Tensor,
) -> float:
    """Ratio of cumulative drift magnitudes (kept_run / full_run).

    Near 1.0 = dropping does not amplify drift; >> 1.0 = much worse;
    near 0.0 = dropping somehow reduced drift (unrealistic; usually
    indicates pathological U).
    """
    full_mag = float(cumulative_drift_curve(U_full).max().item())
    kept_mag = float(cumulative_drift_curve(U_kept).max().item())
    if full_mag < 1e-6:
        return float("nan")
    return kept_mag / full_mag


# ---------------------------------------------------------------------------
# Per-sequence × drop-rate orchestration
# ---------------------------------------------------------------------------


def run_one_sequence(
    paths: DataPaths,
    *,
    dataset: str,
    split: str,
    seq_name: str,
    drop_rates: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.5),
    fps_factors: Tuple[int, ...] = (2,),
    state_source: str = "sinusoidal",
    warmup_min: int = 5, warmup_max: int = 10,
    seed: int = 0,
    max_frames: Optional[int] = None,
    model: Optional[FreqDecGWNet] = None,
    device: str = "cpu",
) -> List[dict]:
    """Run the stress test on one full burst across all configured rates.

    Returns:
        list of per-condition rows ready for CSV serialization.
    """
    loader = FluoroSequenceLoader(paths)
    seq: FluoroSequence = loader.load_sequence(
        dataset, split, seq_name, max_frames=max_frames,
    )
    T_full = seq.num_frames
    if T_full < 16:
        return []                                       # too short to drop

    # ---- Reference frame on the full burst (shared across stress runs) ----
    selector = ReferenceSelector(warmup_min=warmup_min, warmup_max=warmup_max)
    warmup_imgs = seq.frames[: max(warmup_max, warmup_min), 0]
    ref_full = selector.select(images=warmup_imgs)

    # ---- Full-sequence baseline run ----
    if model is not None:
        # Real inference path — use trained FreqDecGWNet end-to-end on full burst
        images_t = seq.frames.unsqueeze(0).to(device)            # [1, T_full, 1, H, W]
        with torch.no_grad():
            out_full = model(
                images_t, mode="stage3_joint",
                reference_index=ref_full.reference_index,
            )
        delta_u_full = out_full["delta_u"][0].cpu()
        U_full = out_full["U"][0].cpu()
        # state for phase correlation (use [cos, sin, amp])
        s_full = out_full["state"][..., :3].cpu()                # [1, T_full, 3]
    else:
        # Placeholder path — only useful for pipeline sanity, not real numbers
        s_full = _placeholder_state(T_full, mode=state_source)
        rmf = RelativeMotionField()
        rmf.eval()
        with torch.no_grad():
            out_full = rmf(s_full, reference_index=ref_full.reference_index)
        delta_u_full = out_full["delta_u"][0]
        U_full = out_full["U"][0]
    cos_full = s_full[0, :, 0]
    sin_full = s_full[0, :, 1]

    rows: List[dict] = []

    # ---- Random drop conditions ----
    for rate in drop_rates:
        plan = random_drop_rate(T_full, drop_rate=rate, seed=seed)
        ref_kept = remap_reference_index(plan, ref_full.reference_index)

        # Re-run the cumulative pipeline on the kept-only subset.
        # When a trained model is given, do a full FreqDecGWNet forward on
        # the kept frames; the encoder + state branch see only the kept
        # frames, mimicking what the deployed system would see if frames
        # were dropped at acquisition time.
        if model is not None:
            kept_imgs = seq.frames[plan.kept_indices].unsqueeze(0).to(device)
            with torch.no_grad():
                out_kept = model(
                    kept_imgs, mode="stage3_joint",
                    reference_index=ref_kept,
                )
            U_kept = out_kept["U"][0].cpu()
            s_kept = out_kept["state"][..., :3].cpu()
        else:
            s_kept = s_full[:, plan.kept_indices, :]
            with torch.no_grad():
                out_kept = rmf(s_kept, reference_index=ref_kept)
            U_kept = out_kept["U"][0]

        cos_kept = s_kept[0, :, 0]
        sin_kept = s_kept[0, :, 1]
        rows.append({
            "seq_name": seq_name, "T_full": T_full,
            "protocol": "random_drop_rate",
            "rate_or_factor": float(rate),
            "T_kept": int(plan.T_kept),
            "actual_drop_rate": float(plan.actual_drop_rate),
            "phase_correlation": _phase_correlation(
                cos_full, sin_full, cos_kept, sin_kept, plan.kept_indices,
            ),
            "compensation_error": _drift_at_dropped_frames(
                U_full, U_kept, plan,
            ),
            "drift_reduction": _drift_reduction(U_full, U_kept),
            "ref_full": int(ref_full.reference_index),
            "ref_kept": int(ref_kept),
            "ref_strategy": ref_full.strategy,
            "seed": int(seed),
        })

    # ---- Deterministic FPS-downsample conditions ----
    for factor in fps_factors:
        if factor + 1 >= T_full:
            continue
        plan = downsample_fps(T_full, factor=factor)
        ref_kept = remap_reference_index(plan, ref_full.reference_index)
        if model is not None:
            kept_imgs = seq.frames[plan.kept_indices].unsqueeze(0).to(device)
            with torch.no_grad():
                out_kept = model(
                    kept_imgs, mode="stage3_joint",
                    reference_index=ref_kept,
                )
            U_kept = out_kept["U"][0].cpu()
            s_kept = out_kept["state"][..., :3].cpu()
        else:
            s_kept = s_full[:, plan.kept_indices, :]
            with torch.no_grad():
                out_kept = rmf(s_kept, reference_index=ref_kept)
            U_kept = out_kept["U"][0]
        cos_kept = s_kept[0, :, 0]
        sin_kept = s_kept[0, :, 1]
        rows.append({
            "seq_name": seq_name, "T_full": T_full,
            "protocol": "downsample_fps",
            "rate_or_factor": float(factor),
            "T_kept": int(plan.T_kept),
            "actual_drop_rate": float(plan.actual_drop_rate),
            "phase_correlation": _phase_correlation(
                cos_full, sin_full, cos_kept, sin_kept, plan.kept_indices,
            ),
            "compensation_error": _drift_at_dropped_frames(
                U_full, U_kept, plan,
            ),
            "drift_reduction": _drift_reduction(U_full, U_kept),
            "ref_full": int(ref_full.reference_index),
            "ref_kept": int(ref_kept),
            "ref_strategy": ref_full.strategy,
            "seed": -1,
        })

    return rows


# ---------------------------------------------------------------------------
# CSV serialization
# ---------------------------------------------------------------------------


_CSV_COLUMNS = [
    "seq_name", "T_full", "protocol", "rate_or_factor", "T_kept",
    "actual_drop_rate", "phase_correlation", "compensation_error",
    "drift_reduction", "ref_full", "ref_kept", "ref_strategy", "seed",
]


def _format_float(v) -> str:
    if isinstance(v, float):
        if v != v:                                       # NaN
            return ""
        return f"{v:.6g}"
    return str(v)


def write_dropped_frame_csv(rows: List[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_COLUMNS)
        for row in rows:
            writer.writerow([_format_float(row.get(k, "")) for k in _CSV_COLUMNS])
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_seqs(
    paths, dataset, split, explicit, all_flag,
):
    if all_flag and explicit:
        raise SystemExit("--all and --seq are mutually exclusive")
    if all_flag:
        return paths.list_sequences(dataset, split)
    if not explicit:
        raise SystemExit("specify --seq <name> [<name>...] or --all")
    available = set(paths.list_sequences(dataset, split))
    missing = [s for s in explicit if s not in available]
    if missing:
        raise SystemExit(f"sequences not found in {dataset}/{split}: {missing}")
    return list(explicit)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dropped-frame stress test (paper §6.2 + tech route §3)",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True,
                        choices=("train", "val", "test"))
    parser.add_argument("--seq", nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "reports" / "dropped_frame")
    parser.add_argument("--drop-rates", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--fps-factors", type=int, nargs="+",
                        default=[2, 3])
    parser.add_argument("--state-source", default="sinusoidal",
                        choices=("zero", "sinusoidal", "random"))
    parser.add_argument("--warmup-min", type=int, default=5)
    parser.add_argument("--warmup-max", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    # Trained-model integration
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="trained FreqDecGWNet ckpt; without it the "
                             "script falls back to a placeholder state "
                             "(numbers will be meaningless)")
    parser.add_argument("--device", default="cuda",
                        choices=("cuda", "cpu"))
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--state-core-type",
                        choices=("gru", "mamba"), default="mamba")
    parser.add_argument("--motion-state-dim", type=int, default=3)
    parser.add_argument("--motion-input-mode",
                        choices=("delta", "with_prev"), default="delta")
    parser.add_argument("--motion-field-type",
                        choices=("relative", "absolute"), default="relative")

    args = parser.parse_args(list(argv) if argv is not None else None)

    paths = (
        DataPaths.from_yaml(args.config) if args.config
        else DataPaths.from_default_config()
    )
    seqs = _resolve_seqs(paths, args.dataset, args.split, args.seq, args.all)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = None
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if args.ckpt is not None:
        if not args.ckpt.is_file():
            raise SystemExit(f"ckpt not found: {args.ckpt}")
        print(f"[evaluate_dropped_frame] loading ckpt: {args.ckpt}")
        model = FreqDecGWNet(
            width_mult=args.width_mult,
            state_core_type=args.state_core_type,
            motion_state_dim=args.motion_state_dim,
            motion_input_mode=args.motion_input_mode,
            motion_field_type=args.motion_field_type,
        ).to(device)
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck)
        sd = {(k[len("module."):] if k.startswith("module.") else k): v
              for k, v in sd.items()}
        miss, unex = model.load_state_dict(sd, strict=False)
        print(f"   loaded missing={len(miss)} unexpected={len(unex)}")
        model.eval()

    print(
        f"[evaluate_dropped_frame] {args.dataset}/{args.split} "
        f"sequences={len(seqs)} drop_rates={args.drop_rates} "
        f"fps_factors={args.fps_factors} "
        f"model={'trained' if model else 'placeholder'}"
    )
    all_rows: List[dict] = []
    for sname in seqs:
        rows = run_one_sequence(
            paths,
            dataset=args.dataset, split=args.split, seq_name=sname,
            drop_rates=tuple(args.drop_rates),
            fps_factors=tuple(args.fps_factors),
            state_source=args.state_source,
            warmup_min=args.warmup_min, warmup_max=args.warmup_max,
            seed=args.seed,
            max_frames=args.max_frames,
            model=model,
            device=device,
        )
        if not rows:
            print(f"  {sname:15s}  skipped (T_full too short)")
            continue
        seq_csv = args.output_dir / f"{sname}.csv"
        write_dropped_frame_csv(rows, seq_csv)
        all_rows.extend(rows)
        # 一行总结
        for r in rows:
            print(
                f"  {sname:15s}  {r['protocol']:18s} "
                f"x={r['rate_or_factor']:.2f}  "
                f"T_kept={r['T_kept']:4d}  "
                f"phase_corr={r['phase_correlation']:.3f}  "
                f"comp_err={r['compensation_error']:.3f}  "
                f"drift_ratio={r['drift_reduction']:.3f}"
            )

    if all_rows:
        summary_csv = args.output_dir / "summary.csv"
        write_dropped_frame_csv(all_rows, summary_csv)
        print(f"\n📊 wrote summary: {summary_csv} ({len(all_rows)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
