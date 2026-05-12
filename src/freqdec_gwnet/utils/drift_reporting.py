"""Drift evaluation reporting (CSV + PNG).

Companion to :mod:`freqdec_gwnet.utils.drift_metrics`. The metrics module
produces tensors; this module persists them to disk in the layout the
constraints document requires::

    reports/drift/{sequence_id}.csv
    reports/drift/{sequence_id}.png

Both writers operate on a :class:`DriftReport` value object so that the eval
script and the (future) batched reporting tool share the same schema.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class DriftReport:
    """Everything ``evaluate_drift.py`` needs to produce a CSV+PNG report.

    Attributes:
        seq_name: e.g. ``"cmu_seq_007"``.
        dataset / split: source identifiers, written into the CSV header.
        frame_names: per-frame filenames; aligned with all per-frame tensors.
        reference_index: chosen ``r`` from :class:`ReferenceSelector`.
        reference_strategy: ``"clinical"`` / ``"warmup_score"`` /
            ``"first_frame"``; required by PROJECT_CONSTRAINTS §2.
        delta_u: ``[T, 2]`` predicted incremental motion (pixels).
        U: ``[T, 2]`` cumulative motion ``U_t`` (pixels), with ``U[r] = 0``.
        drift_curve: ``[T]`` ``‖U_t‖₂`` — equals ``U.norm(dim=-1)``.
        before_residual: ``[T]`` landmark distance from reference *before*
            compensation; ``NaN`` for frames without a tip annotation.
        after_residual: ``[T]`` landmark distance from reference *after*
            subtracting ``U_t``; ``NaN`` where unannotated.
        landmark_present: ``[T]`` boolean mask of frames that contributed.
        improvement_mean: scalar mean of ``before − after`` over annotated
            frames; ``NaN`` when no frame had a tip annotation.
        extra: free-form metadata to drop into the CSV header (model name,
            checkpoint id, motion-source mode, …).
    """

    seq_name: str
    dataset: str
    split: str
    frame_names: List[str]

    reference_index: int
    reference_strategy: str

    delta_u: torch.Tensor
    U: torch.Tensor
    drift_curve: torch.Tensor
    before_residual: torch.Tensor
    after_residual: torch.Tensor
    landmark_present: torch.Tensor
    improvement_mean: torch.Tensor

    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        T = len(self.frame_names)
        for name, t in (
            ("delta_u", self.delta_u),
            ("U", self.U),
            ("drift_curve", self.drift_curve),
            ("before_residual", self.before_residual),
            ("after_residual", self.after_residual),
            ("landmark_present", self.landmark_present),
        ):
            if t.shape[0] != T:
                raise ValueError(
                    f"DriftReport.{name} has T={t.shape[0]}; "
                    f"frame_names has {T}"
                )


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


_CSV_COLUMNS = [
    "frame_index",
    "frame_name",
    "delta_u_x",
    "delta_u_y",
    "U_x",
    "U_y",
    "drift_norm",
    "landmark_present",
    "before_residual",
    "after_residual",
]


def _format_float(x: torch.Tensor) -> str:
    """Print 6 sig figs; render NaN as empty cell so spreadsheets cope."""
    v = float(x.item())
    if v != v:                      # NaN
        return ""
    return f"{v:.6g}"


def write_drift_csv(report: DriftReport, path: Path) -> Path:
    """Persist ``report`` as a one-row-per-frame CSV.

    Header lines (prefixed with ``#``) record the sequence identity,
    reference-frame choice, the strategy used, and ``improvement_mean`` so
    the file is interpretable in isolation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header_lines = [
        f"# seq_name={report.seq_name}",
        f"# dataset={report.dataset}",
        f"# split={report.split}",
        f"# reference_index={report.reference_index}",
        f"# reference_strategy={report.reference_strategy}",
        f"# improvement_mean={_format_float(report.improvement_mean)}",
    ]
    for k, v in report.extra.items():
        header_lines.append(f"# {k}={v}")

    with path.open("w", encoding="utf-8", newline="") as fh:
        for line in header_lines:
            fh.write(line + "\n")
        writer = csv.writer(fh)
        writer.writerow(_CSV_COLUMNS)
        T = len(report.frame_names)
        for i in range(T):
            writer.writerow([
                i,
                report.frame_names[i],
                _format_float(report.delta_u[i, 0]),
                _format_float(report.delta_u[i, 1]),
                _format_float(report.U[i, 0]),
                _format_float(report.U[i, 1]),
                _format_float(report.drift_curve[i]),
                int(bool(report.landmark_present[i].item())),
                _format_float(report.before_residual[i]),
                _format_float(report.after_residual[i]),
            ])
    return path


# ---------------------------------------------------------------------------
# PNG renderer
# ---------------------------------------------------------------------------


def render_drift_png(report: DriftReport, path: Path) -> Path:
    """Render the canonical 3-panel drift figure.

    Panel layout:

        1. Cumulative drift curve ``‖U_t‖₂`` (R3 stability monitor).
        2. Landmark residual before vs after (only on annotated frames).
        3. Δu_x and Δu_y per-frame (sanity check on the increment magnitude).

    A reference-frame guideline is drawn on every panel so the reader can
    line up where the cumulative sum starts.
    """
    # 延迟 import：避免单纯加载本模块时强制依赖 matplotlib
    import matplotlib
    matplotlib.use("Agg")            # 服务器/容器无图形界面时必需
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    T = len(report.frame_names)
    t = list(range(T))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # ---- Panel 1: cumulative drift ----
    ax = axes[0]
    ax.plot(t, report.drift_curve.cpu().numpy(), color="C0",
            label=r"$\|U_t\|_2$")
    ax.axvline(report.reference_index, color="k", linestyle="--",
               alpha=0.6, label=f"r = {report.reference_index}")
    ax.set_ylabel("cumulative drift (pix)")
    ax.set_title(
        f"{report.dataset}/{report.split}/{report.seq_name}  "
        f"(ref strategy: {report.reference_strategy})"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # ---- Panel 2: landmark residual ----
    ax = axes[1]
    before = report.before_residual.cpu().numpy()
    after = report.after_residual.cpu().numpy()
    valid = report.landmark_present.cpu().numpy().astype(bool)
    if valid.any():
        # 用散点 + 线 同时画，方便看到 NaN 间断
        ax.plot([t[i] for i in range(T) if valid[i]],
                before[valid], "o-", label="before", color="C1", ms=4)
        ax.plot([t[i] for i in range(T) if valid[i]],
                after[valid], "o-", label="after", color="C2", ms=4)
        ax.legend(loc="upper left")
        improvement = report.improvement_mean
        ax.text(0.99, 0.97,
                f"mean improvement = {_format_float(improvement)} px",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9,
                bbox=dict(facecolor="white", edgecolor="0.7"))
    else:
        ax.text(0.5, 0.5, "no landmark annotations in this sequence",
                ha="center", va="center", transform=ax.transAxes,
                color="0.5")
    ax.axvline(report.reference_index, color="k", linestyle="--", alpha=0.6)
    ax.set_ylabel("landmark residual (pix)")
    ax.grid(alpha=0.3)

    # ---- Panel 3: per-frame Δu ----
    ax = axes[2]
    du = report.delta_u.cpu().numpy()
    ax.plot(t, du[:, 0], label=r"$\Delta u_x$", color="C3")
    ax.plot(t, du[:, 1], label=r"$\Delta u_y$", color="C4")
    ax.axvline(report.reference_index, color="k", linestyle="--", alpha=0.6)
    ax.axhline(0.0, color="0.7", linewidth=0.7)
    ax.set_ylabel(r"$\Delta u_t$ (pix)")
    ax.set_xlabel("frame index")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Convenience: build a DriftReport from raw tensors
# ---------------------------------------------------------------------------


def assemble_drift_report(
    *,
    seq_name: str,
    dataset: str,
    split: str,
    frame_names: List[str],
    reference_index: int,
    reference_strategy: str,
    delta_u: torch.Tensor,                     # [T, 2]
    U: torch.Tensor,                            # [T, 2]
    landmark_xy: Optional[torch.Tensor],        # [T, 2] with NaN, or None
    extra: Optional[Dict[str, Any]] = None,
) -> DriftReport:
    """Build a :class:`DriftReport` from R3 outputs and tip annotations.

    Wraps the metric calls so the eval script just hands over raw tensors.
    """
    from .drift_metrics import cumulative_drift_curve, landmark_residual

    drift = cumulative_drift_curve(U)
    if landmark_xy is None:
        T = U.shape[0]
        nan = torch.full((T,), float("nan"))
        before = nan.clone()
        after = nan.clone()
        present = torch.zeros(T, dtype=torch.bool)
        improvement_mean = torch.tensor(float("nan"))
    else:
        # 如果参考帧本身没标注，landmark_residual 会抛错；这里捕获并做安全降级
        try:
            res = landmark_residual(
                landmark_xy, U, reference_index=reference_index,
            )
            before = res["before"]
            after = res["after"]
            present = res["valid"]
            improvement_mean = res["improvement_mean"]
        except ValueError:
            T = U.shape[0]
            nan = torch.full((T,), float("nan"))
            before = nan.clone()
            after = nan.clone()
            present = torch.zeros(T, dtype=torch.bool)
            improvement_mean = torch.tensor(float("nan"))

    return DriftReport(
        seq_name=seq_name,
        dataset=dataset,
        split=split,
        frame_names=list(frame_names),
        reference_index=int(reference_index),
        reference_strategy=reference_strategy,
        delta_u=delta_u.detach().cpu(),
        U=U.detach().cpu(),
        drift_curve=drift.detach().cpu(),
        before_residual=before.detach().cpu(),
        after_residual=after.detach().cpu(),
        landmark_present=present.detach().cpu(),
        improvement_mean=improvement_mean.detach().cpu(),
        extra=dict(extra or {}),
    )
