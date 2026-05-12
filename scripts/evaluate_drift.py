#!/usr/bin/env python
"""Drift evaluation entry point (PROJECT_CONSTRAINTS_19.6.md §3 + tech route §2).

Loads one fluoroscopy sequence, picks a reference frame via
:class:`ReferenceSelector`, runs :class:`RelativeMotionField` over the burst,
and writes ``reports/drift/{seq}.csv`` + ``reports/drift/{seq}.png``.

Until a state branch checkpoint is wired in, the state input fed to G_delta
is a placeholder; the script clearly labels the chosen ``state_source`` in
the CSV header so untrained runs are not mistaken for real performance
numbers. Once stage-2 produces a checkpoint, swap the state-source flag and
the same script will report real drift.

Usage::

    python scripts/evaluate_drift.py \\
        --dataset dense_v1 --split train --seq cmu_seq_007

    # All sequences in a split:
    python scripts/evaluate_drift.py --dataset dense_v1 --split train --all
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import torch

# 让脚本可以在没装包的情况下从 src/ 跑起来
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freqdec_gwnet.data import (                                       # noqa: E402
    DataPaths,
    FluoroSequence,
    FluoroSequenceLoader,
)
from freqdec_gwnet.models.freqdec_gwnet import FreqDecGWNet            # noqa: E402
from freqdec_gwnet.models.relative_motion_field import (               # noqa: E402
    RelativeMotionField,
)
from freqdec_gwnet.utils.cycle_consistency import (                    # noqa: E402
    cycle_consistency_error,
)
from freqdec_gwnet.utils.drift_reporting import (                      # noqa: E402
    assemble_drift_report,
    render_drift_png,
    write_drift_csv,
)
from freqdec_gwnet.utils.reference_selector import (                   # noqa: E402
    ReferenceSelector,
)


# ---------------------------------------------------------------------------
# Placeholder state generators (until stage-2 checkpoint is available)
# ---------------------------------------------------------------------------


def make_state_placeholder(
    T: int,
    mode: str,
    amplitude: float = 0.5,
    seed: int = 0,
) -> torch.Tensor:
    """Generate a length-T state proxy ``[1, T, 3]`` for the motion field.

    Until the global state branch is trained, the script needs *something* to
    feed G_delta. The choice does not affect zero-init Δu (G_delta's last
    layer is zeroed), but it lets us inspect the cumulative pipeline shape.

    Modes:
        ``zero``: all zeros — minimal, deterministic.
        ``sinusoidal``: cos / sin / amplitude over one cycle of length T;
            simulates a single respiratory cycle.
        ``random``: torch.randn fixed by ``seed``; for sanity stress-tests.
    """
    if mode == "zero":
        return torch.zeros(1, T, 3)
    if mode == "sinusoidal":
        t = torch.linspace(0.0, 2 * math.pi, T)
        return torch.stack(
            [torch.cos(t), torch.sin(t), torch.full_like(t, amplitude)],
            dim=-1,
        ).unsqueeze(0)
    if mode == "random":
        g = torch.Generator().manual_seed(seed)
        return torch.randn(1, T, 3, generator=g)
    raise ValueError(f"unknown state-source mode '{mode}'")


# ---------------------------------------------------------------------------
# Single-sequence orchestration
# ---------------------------------------------------------------------------


def evaluate_sequence(
    paths: DataPaths,
    dataset: str,
    split: str,
    seq_name: str,
    output_dir: Path,
    *,
    max_frames: Optional[int] = None,
    state_source: str = "sinusoidal",
    warmup_min: int = 5,
    warmup_max: int = 10,
    reference_strategy: str = "auto",
    reference_random_seed: int = 0,
    clinical_roadmap_index: Optional[int] = None,
    cycle_consistency: bool = False,
    cycle_strategy: str = "state_distance",
    cycle_state_threshold: float = 0.30,
    cycle_n_phase_bins: int = 12,
    cycle_amplitude_tolerance: float = 0.20,
    cycle_min_gap_frames: int = 8,
    reset_every: int = 0,
    model: Optional[FreqDecGWNet] = None,
    device: str = "cpu",
) -> dict:
    """Run the full drift pipeline on one sequence and persist outputs.

    Returns:
        ``dict`` with keys ``csv_path``, ``png_path``, ``reference_index``,
        ``reference_strategy``, ``improvement_mean``.
    """
    loader = FluoroSequenceLoader(paths)
    seq: FluoroSequence = loader.load_sequence(
        dataset, split, seq_name, max_frames=max_frames,
    )
    T = seq.num_frames

    # ---- 1. 选 reference frame ----
    selector = ReferenceSelector(
        warmup_min=warmup_min, warmup_max=warmup_max,
        force_strategy=reference_strategy,
        random_seed=reference_random_seed,
    )
    # warm-up 评分需要 [N, H, W] 灰度，从 seq.frames 第 0 通道切出
    warmup_imgs = seq.frames[: max(warmup_max, warmup_min), 0]
    ref_sel = selector.select(
        images=warmup_imgs,
        clinical_roadmap_index=clinical_roadmap_index,
        total_frames=T,
    )

    # ---- 2. State → Δu → U：优先用训练好的 FreqDecGWNet，其次退回 placeholder ----
    if model is not None:
        # 真实推理路径：encoder + state branch + motion field 端到端
        images_t = seq.frames.unsqueeze(0).to(device)          # [1, T, 1, H, W]
        with torch.no_grad():
            out = model(
                images_t, mode="stage3_joint",
                reference_index=ref_sel.reference_index,
            )
        delta_u = out["delta_u"][0].cpu()                       # [T, 2]
        U = out["U"][0].cpu()
        # state 用于 cycle-consistency 检测：取 [cos, sin, amp]
        s_seq = out["state"][..., :3].cpu()                     # [1, T, 3]
        state_source_used = "freqdec_gwnet_ckpt"
        # Reference-reset: cumulative formulation accumulates drift over long
        # bursts; in a deployed system the operator periodically re-anchors,
        # which we simulate by zeroing U every ``reset_every`` frames.
        # 这是 A-lite 列表的第 4 项 "relative reset"。
        if reset_every and reset_every > 0:
            new_U = torch.zeros_like(U)
            for chunk_start in range(0, T, reset_every):
                chunk_end = min(T, chunk_start + reset_every)
                if chunk_end <= chunk_start:
                    continue
                # within-chunk cumulative sum of Δu, reset at boundaries
                chunk_delta = delta_u[chunk_start:chunk_end].clone()
                chunk_delta[0] = 0.0           # reset Δu_0 = 0 at each anchor
                new_U[chunk_start:chunk_end] = torch.cumsum(chunk_delta, dim=0)
            U = new_U
            state_source_used += f"+reset_every_{reset_every}"
    else:
        # 旧的 placeholder 路径（保留用于无 ckpt 时的流水线检查）
        s_seq = make_state_placeholder(T, mode=state_source)
        rmf = RelativeMotionField()
        rmf.eval()
        with torch.no_grad():
            out = rmf(s_seq, reference_index=ref_sel.reference_index)
        delta_u = out["delta_u"][0]
        U = out["U"][0]
        state_source_used = f"placeholder/{state_source}"

    # ---- 2.5 Cycle-consistency analysis (paper §5.2) ----
    cycle_extra = {}
    if cycle_consistency:
        # 把 placeholder state 当成 [T, 3] 喂给 detector
        # 真正训完之后这里应该用 model 预测的 state
        s_for_cycle = s_seq[0]                  # [T, 3]
        result = cycle_consistency_error(
            state_seq=s_for_cycle,
            U=U,
            reference_index=ref_sel.reference_index,
            strategy=cycle_strategy,
            state_distance_threshold=cycle_state_threshold,
            n_phase_bins=cycle_n_phase_bins,
            amplitude_tolerance=cycle_amplitude_tolerance,
            min_gap_frames=cycle_min_gap_frames,
        )
        cycle_extra = {
            "cycle_available": result.available,
            "cycle_K": result.K if result.K is not None else -1,
            "cycle_E_cycle": result.E_cycle if result.E_cycle is not None else float("nan"),
            "cycle_strategy": result.strategy,
            "cycle_status": result.status,
            "cycle_n_candidates": result.details.get("n_candidates", 0),
        }

    # ---- 3. 装配报告 ----
    report = assemble_drift_report(
        seq_name=seq.seq_name,
        dataset=seq.dataset,
        split=seq.split,
        frame_names=seq.frame_names,
        reference_index=ref_sel.reference_index,
        reference_strategy=ref_sel.strategy,
        delta_u=delta_u,
        U=U,
        landmark_xy=seq.tip_xy,
        extra={
            "state_source": state_source_used,
            "warmup_min": warmup_min,
            "warmup_max": warmup_max,
            "model": (
                "FreqDecGWNet(trained)" if model is not None
                else "RelativeMotionField(zero_init,untrained)"
            ),
            "n_frames": T,
            "n_landmark_frames": int(seq.tip_present.sum().item()),
            **cycle_extra,
        },
    )

    # ---- 4. 写 csv + png ----
    csv_path = output_dir / f"{seq.seq_name}.csv"
    png_path = output_dir / f"{seq.seq_name}.png"
    write_drift_csv(report, csv_path)
    render_drift_png(report, png_path)

    return {
        "csv_path": csv_path,
        "png_path": png_path,
        "reference_index": report.reference_index,
        "reference_strategy": report.reference_strategy,
        "improvement_mean": float(report.improvement_mean.item()),
        "n_frames": T,
        "n_landmark_frames": int(seq.tip_present.sum().item()),
        **cycle_extra,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_seqs(
    paths: DataPaths,
    dataset: str,
    split: str,
    explicit: Optional[List[str]],
    all_flag: bool,
) -> List[str]:
    if all_flag and explicit:
        raise SystemExit("--all and --seq are mutually exclusive")
    if all_flag:
        return paths.list_sequences(dataset, split)
    if not explicit:
        raise SystemExit(
            "specify --seq <name> [<name>...] or --all"
        )
    available = set(paths.list_sequences(dataset, split))
    missing = [s for s in explicit if s not in available]
    if missing:
        raise SystemExit(
            f"sequences not found in {dataset}/{split}: {missing}"
        )
    return list(explicit)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="FreqDec-GWNet drift evaluator (R3 pipeline)",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="path to data_paths.yaml (defaults to project's)")
    parser.add_argument("--dataset", required=True,
                        help="dataset key in data_paths.yaml")
    parser.add_argument("--split", required=True,
                        choices=("train", "val", "test"))
    parser.add_argument("--seq", nargs="+", default=None,
                        help="one or more sequence names")
    parser.add_argument("--all", action="store_true",
                        help="evaluate every sequence in the split")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "reports" / "drift",
                        help="where to write {seq}.csv and {seq}.png")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="cap T for quick smoke runs")
    parser.add_argument("--state-source",
                        choices=("zero", "sinusoidal", "random"),
                        default="sinusoidal",
                        help="placeholder state used for G_delta input "
                             "(swap when state-branch ckpt is available)")
    parser.add_argument("--warmup-min", type=int, default=5)
    parser.add_argument("--warmup-max", type=int, default=10)
    parser.add_argument("--reset-every", type=int, default=0,
                        help="if > 0, zero-out U every N frames at eval — "
                             "mimics a deployed system that re-anchors "
                             "periodically (paper §III.C reference reset)")
    # Trained-model integration: when --ckpt is given, run actual inference
    # via FreqDecGWNet instead of the zero-init placeholder.
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="path to a trained FreqDecGWNet checkpoint "
                             "(stage2 or stage3); when omitted, uses "
                             "RelativeMotionField with zero-init (numbers "
                             "will be meaningless)")
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
    # §6.1 main ablation #4: reference selection strategy
    parser.add_argument(
        "--reference-strategy",
        choices=("auto", "warmup_score", "first_frame", "random", "clinical"),
        default="auto",
        help="force a specific reference-frame selection strategy. "
             "'auto' (default) follows the 3-tier priority. "
             "'first_frame' is the BAD baseline (r=0). "
             "'random' picks a uniform-random frame in the warm-up window. "
             "'clinical' requires --clinical-roadmap-index."
    )
    parser.add_argument("--reference-random-seed", type=int, default=0)
    parser.add_argument("--clinical-roadmap-index", type=int, default=None,
                        help="external clinical roadmap frame index "
                             "(required when --reference-strategy clinical)")
    # Cycle-consistency analysis (paper §5.2)
    parser.add_argument("--cycle-consistency", action="store_true",
                        help="also compute E_cycle = ‖U_K − U_ref‖₂ if a "
                             "near-cycle frame is detected; reports "
                             "'not_available' when none qualifies")
    parser.add_argument("--cycle-strategy", default="state_distance",
                        choices=("state_distance", "phase_bin"))
    parser.add_argument("--cycle-state-threshold", type=float, default=0.30)
    parser.add_argument("--cycle-n-phase-bins", type=int, default=12)
    parser.add_argument("--cycle-amplitude-tolerance", type=float, default=0.20)
    parser.add_argument("--cycle-min-gap-frames", type=int, default=8)
    args = parser.parse_args(list(argv) if argv is not None else None)

    paths = (
        DataPaths.from_yaml(args.config)
        if args.config else
        DataPaths.from_default_config()
    )

    seqs = _resolve_seqs(paths, args.dataset, args.split, args.seq, args.all)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 加载训练好的 FreqDecGWNet 一次，复用给所有序列
    model = None
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if args.ckpt is not None:
        if not args.ckpt.is_file():
            raise SystemExit(f"ckpt not found: {args.ckpt}")
        print(f"[evaluate_drift] loading ckpt: {args.ckpt}")
        model = FreqDecGWNet(
            width_mult=args.width_mult,
            state_core_type=args.state_core_type,
            motion_state_dim=args.motion_state_dim,
            motion_input_mode=args.motion_input_mode,
            motion_field_type=args.motion_field_type,
        ).to(device)
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in sd.items()
        }
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"   loaded with missing={len(missing)} unexpected={len(unexpected)}")
        model.eval()

    print(f"[evaluate_drift] dataset={args.dataset} split={args.split} "
          f"sequences={len(seqs)} output={args.output_dir} "
          f"model={'trained' if model else 'untrained-placeholder'}")
    for sname in seqs:
        result = evaluate_sequence(
            paths,
            dataset=args.dataset,
            split=args.split,
            seq_name=sname,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            state_source=args.state_source,
            warmup_min=args.warmup_min,
            warmup_max=args.warmup_max,
            reference_strategy=args.reference_strategy,
            reference_random_seed=args.reference_random_seed,
            clinical_roadmap_index=args.clinical_roadmap_index,
            cycle_consistency=args.cycle_consistency,
            cycle_strategy=args.cycle_strategy,
            cycle_state_threshold=args.cycle_state_threshold,
            cycle_n_phase_bins=args.cycle_n_phase_bins,
            cycle_amplitude_tolerance=args.cycle_amplitude_tolerance,
            model=model,
            device=device,
            cycle_min_gap_frames=args.cycle_min_gap_frames,
            reset_every=args.reset_every,
        )
        line = (
            f"  {sname:15s}  "
            f"T={result['n_frames']:4d}  "
            f"landmarks={result['n_landmark_frames']:4d}  "
            f"r={result['reference_index']:3d} "
            f"({result['reference_strategy']:14s})  "
            f"improvement_mean={result['improvement_mean']:.3f} px"
        )
        if args.cycle_consistency:
            line += f"  cycle={result.get('cycle_status', 'n/a')}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
