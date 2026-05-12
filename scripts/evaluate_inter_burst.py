#!/usr/bin/env python
"""Inter-burst continuity evaluator — the Mamba sandbox.

Real clinical fluoroscopy produces multi-burst sessions with 5-30 s
inter-burst pauses. The public open-source datasets we use only ship
**single-burst** sequences. This script bridges that gap by synthetically
chunking one long burst into K pseudo-bursts with configurable gap, then
runs four state-branch inference protocols (paper §IV.E ablation):

    1. naive    — every chunk starts with h₀ = 0
    2. carry    — carry final hidden state from chunk N to chunk N+1
    3. carry_dt — carry h₀ + Δt-modulated z_t inputs
    4. oracle   — process the unchunked burst (upper bound)

Outputs:
    reports/inter_burst/{seq}.csv         per-protocol cos/sin + jumps
    reports/inter_burst/{seq}_jumps.csv   compact table of mean boundary jump
    reports/inter_burst/{seq}.png         4-panel plot of cos_phi over time

Usage::

    python scripts/evaluate_inter_burst.py \\
        --dataset clean --split val --seq zhu_seq_015 \\
        --ckpt experiments/checkpoints/stage3_main_v3_full_best.pth \\
        --state-core-type mamba \\
        --K 3 --gap-frames 30 \\
        --output-dir reports/inter_burst_v1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freqdec_gwnet.data import DataPaths, FluoroSequenceLoader        # noqa: E402
from freqdec_gwnet.models.freqdec_gwnet import FreqDecGWNet           # noqa: E402
from freqdec_gwnet.utils.inter_burst_continuity import (              # noqa: E402
    build_chunk_plan, run_inter_burst_continuity,
)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def render_continuity_png(
    results, plan, path: Path, *, seq_name: str,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    colors = {"naive": "C3", "carry": "C0", "carry_dt": "C2", "oracle": "C7"}
    titles = {
        "naive": "(a) naive — each chunk starts h₀=0",
        "carry": "(b) carry — h_final carried across",
        "carry_dt": "(c) carry+Δt — h_final + Δt-modulated z",
        "oracle": "(d) oracle — unchunked baseline",
    }
    K = plan.K
    chunk_starts_in_kept = [0]
    cursor = 0
    for L in plan.chunk_lengths[:-1]:
        cursor += L
        chunk_starts_in_kept.append(cursor)

    for ax, key in zip(axes, ("naive", "carry", "carry_dt", "oracle")):
        if key not in results:
            continue
        cos = results[key].cos_phi
        ax.plot(np.arange(cos.size), cos, color=colors[key], lw=0.9)
        # 标出 chunk 边界
        for x in chunk_starts_in_kept[1:]:
            ax.axvline(x, color="0.5", lw=0.6, ls="--")
        # 边界跳变值
        jumps = results[key].boundary_phase_jumps
        for i, (end_idx, start_idx) in enumerate(plan.boundary_kept_indices):
            ax.text(
                start_idx, 1.05, f"Δ={jumps[i]:.3f}",
                fontsize=8, ha="left", va="bottom", color="0.3",
            )
        mean_jump = results[key].mean_boundary_jump
        ax.set_title(
            f"{titles[key]}    mean boundary jump = {mean_jump:.4f}",
            fontsize=10, loc="left",
        )
        ax.set_ylim(-1.3, 1.3)
        ax.set_ylabel("cos φ")
        ax.grid(alpha=0.3)
        ax.axhline(0, color="0.7", lw=0.5)
    axes[-1].set_xlabel(f"kept-frame index  ({seq_name}: K={K} chunks, "
                        f"gap={plan.gap_frames} frames each)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def write_continuity_csv(results, plan, path: Path) -> Path:
    """Per-frame protocol predictions, all 4 protocols side-by-side."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    T = results["oracle"].cos_phi.size
    columns = ["kept_idx"]
    for p in ("naive", "carry", "carry_dt", "oracle"):
        columns += [f"{p}_cos", f"{p}_sin"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        for i in range(T):
            row = [i]
            for p in ("naive", "carry", "carry_dt", "oracle"):
                row += [
                    f"{results[p].cos_phi[i]:.6g}",
                    f"{results[p].sin_phi[i]:.6g}",
                ]
            w.writerow(row)
    return path


def write_jump_summary_csv(results, plan, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in ("naive", "carry", "carry_dt", "oracle"):
        rows.append({
            "protocol": p,
            "K": plan.K,
            "gap_frames": plan.gap_frames,
            "n_boundaries": len(plan.boundary_kept_indices),
            "mean_boundary_jump": results[p].mean_boundary_jump,
            "max_boundary_jump": (
                float(results[p].boundary_phase_jumps.max())
                if results[p].boundary_phase_jumps.size else float("nan")
            ),
        })
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Inter-burst continuity sandbox (paper §IV.E)",
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", required=True,
                   choices=("train", "val", "test"))
    p.add_argument("--seq", nargs="+", required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "reports" / "inter_burst")
    # Chunking
    p.add_argument("--K", type=int, default=3,
                   help="number of pseudo-bursts to split into")
    p.add_argument("--gap-frames", type=int, default=15,
                   help="frames to drop between chunks (≈ 1 s at 15 fps)")
    p.add_argument("--chunk-length", type=int, default=None,
                   help="frames per chunk (default = even split)")
    # Model
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--state-core-type",
                   choices=("gru", "mamba"), default="mamba")
    p.add_argument("--device", default="cuda",
                   choices=("cuda", "cpu"))

    args = p.parse_args(list(argv) if argv is not None else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = (
        DataPaths.from_yaml(args.config) if args.config
        else DataPaths.from_default_config()
    )
    loader = FluoroSequenceLoader(paths)

    device = (
        args.device if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    print(f"[inter_burst] device={device} ckpt={args.ckpt}")
    model = FreqDecGWNet(
        width_mult=args.width_mult,
        state_core_type=args.state_core_type,
    ).to(device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    sd = {(k[len("module."):] if k.startswith("module.") else k): v
          for k, v in sd.items()}
    miss, unex = model.load_state_dict(sd, strict=False)
    print(f"  loaded with missing={len(miss)} unexpected={len(unex)}")
    model.eval()

    all_jumps: List[dict] = []
    for sname in args.seq:
        seq = loader.load_sequence(args.dataset, args.split, sname)
        T_full = seq.num_frames
        plan = build_chunk_plan(
            T_full, K=args.K, gap_frames=args.gap_frames,
            chunk_length=args.chunk_length,
        )
        print(
            f"\n  {sname}  T_full={T_full}  K={plan.K}  "
            f"chunk_length={plan.chunk_lengths[0]}  gap={args.gap_frames}"
        )
        results = run_inter_burst_continuity(
            model, seq.frames, plan, device=device,
        )
        # Persist
        write_continuity_csv(
            results, plan, args.output_dir / f"{sname}.csv",
        )
        write_jump_summary_csv(
            results, plan, args.output_dir / f"{sname}_jumps.csv",
        )
        try:
            render_continuity_png(
                results, plan, args.output_dir / f"{sname}.png",
                seq_name=sname,
            )
        except Exception as exc:
            print(f"    (plot skipped: {exc})")

        # Print compact summary
        for proto in ("naive", "carry", "carry_dt", "oracle"):
            res = results[proto]
            print(
                f"    {proto:9s} mean_boundary_jump = "
                f"{res.mean_boundary_jump:.4f}"
            )
            all_jumps.append({
                "seq": sname, "protocol": proto,
                "mean_boundary_jump": res.mean_boundary_jump,
            })

    # Cross-sequence summary
    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["seq", "protocol", "mean_boundary_jump"])
        w.writeheader()
        for r in all_jumps:
            r["mean_boundary_jump"] = f"{r['mean_boundary_jump']:.6g}"
            w.writerow(r)
    print(f"\n📊 summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
