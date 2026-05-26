#!/usr/bin/env python
"""Produce the paper §IV.D headline table from v6 evaluation outputs.

Reads:
    reports/paper_v6_metrics/per_sequence.csv     (learned methods)
    reports/paper_v6_metrics/classical_baseline.csv

Emits:
    reports/paper_v6_metrics/paper_table.csv
    reports/paper_v6_metrics/paper_table.md   (latex-friendly markdown)

The headline columns match what §IV.D needs:
    method | reset | n_seqs | A_imp ± std | A_reduction | B_imp ± std | B_reduction
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_per_sequence(path: Path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def read_classical(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open() as fh:
        for r in csv.DictReader(fh):
            r["ckpt"] = "classical_phase_corr"
            rows.append(r)
    return rows


def _to_float(x):
    if x is None or x == "":
        return float("nan")
    try:
        return float(x)
    except ValueError:
        return float("nan")


def aggregate(rows, ckpt_filter=None, split="test"):
    by_method_reset = defaultdict(list)
    for r in rows:
        if r.get("split") != split:
            continue
        ckpt = r.get("ckpt", "?")
        if ckpt_filter and ckpt != ckpt_filter:
            continue
        reset = int(r.get("reset_every", 0))
        by_method_reset[(ckpt, reset)].append(r)
    return by_method_reset


def _stat(rows, field):
    vals = [_to_float(r.get(field)) for r in rows]
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan"), 0
    mean = sum(vals) / len(vals)
    if len(vals) > 1:
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
    else:
        std = 0.0
    return mean, std, len(vals)


def render_table(per_seq_rows, classical_rows, output_dir: Path):
    all_rows = list(per_seq_rows) + list(classical_rows)

    output_dir.mkdir(parents=True, exist_ok=True)

    md = ["# Paper §IV.D — drift-aware compensation, FreqDec-GWNet v5 ckpts\n"]
    md.append(
        "Three metrics per row (mean ± std over evaluation sequences):\n\n"
        "- **A_imp**  = respiratory-projected residual reduction (px). "
        "Positive = compensation reduces respiratory drift along the PCA axis.\n"
        "- **A_red**  = ratio after/before (lower better; 1.0 = no effect).\n"
        "- **B_imp**  = background-patch alignment reduction (px). "
        "Positive = background features re-align after compensation.\n"
        "- **B_red**  = same ratio for Metric B.\n"
        "- **legacy** = tip TRE residual reduction (px) — kept for "
        "transparency; conflates respiratory motion with active push and "
        "is NOT the headline.\n\n"
        "See §IV.A for metric definitions (Shechter 2004 TMI, "
        "Ablitt 2004 TMI, Ambrosini 2015 TMI).\n"
    )

    method_labels = [
        ("classical_phase_corr", "Classical phase-correlation"),
        ("stage3_abs_zm_v5_best", "Absolute + zero-mean (ours, baseline head)"),
        ("stage3_rel_zm_v5_best", "Relative + zero-mean (ours, main)"),
        ("stage3_rel_nozm_v5_best", "Relative, no zero-mean (ablation)"),
        ("baseline_no_comp", "No compensation (baseline)"),
    ]

    csv_path = output_dir / "paper_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "method", "split", "reset_every", "n_seqs",
            "A_imp_mean", "A_imp_std", "A_red_mean",
            "B_imp_mean", "B_imp_std", "B_red_mean",
            "legacy_imp_mean",
        ])
        for split in ("train", "val", "test"):
            md.append(f"\n## Split: dense_v1/{split}\n")
            md.append(
                "| Method | reset | n | A_imp (px) | A_red | "
                "B_imp (px) | B_red | legacy (px) |\n"
                "|---|---:|---:|---:|---:|---:|---:|---:|\n"
            )
            grouped = aggregate(all_rows, split=split)
            seen_methods = []
            for ckpt_key, label in method_labels:
                for reset in (0, 16, 32):
                    key = (ckpt_key, reset)
                    rows = grouped.get(key, [])
                    if not rows:
                        continue
                    a_imp, a_std, n_a = _stat(rows, "metric_a_improvement")
                    b_imp, b_std, n_b = _stat(rows, "metric_b_improvement")
                    a_red, _, _ = _stat(rows, "metric_a_reduction")
                    b_red, _, _ = _stat(rows, "metric_b_reduction")
                    leg, _, _ = _stat(rows, "legacy_tip_improvement")
                    n = max(n_a, n_b, len(rows))
                    w.writerow([
                        label, split, reset, n,
                        f"{a_imp:.3f}", f"{a_std:.3f}", f"{a_red:.3f}",
                        f"{b_imp:.3f}", f"{b_std:.3f}", f"{b_red:.3f}",
                        f"{leg:.3f}",
                    ])
                    md.append(
                        f"| {label} | {reset} | {n} | "
                        f"{a_imp:+.2f}±{a_std:.2f} | {a_red:.2f} | "
                        f"{b_imp:+.2f}±{b_std:.2f} | {b_red:.2f} | "
                        f"{leg:+.2f} |\n"
                    )
                    seen_methods.append(label)

    md_path = output_dir / "paper_table.md"
    md_path.write_text("".join(md), encoding="utf-8")
    print(f"[csv] {csv_path}")
    print(f"[md]  {md_path}")


def main():
    metrics_dir = ROOT / "reports" / "paper_v6_metrics"
    per_seq = read_per_sequence(metrics_dir / "per_sequence.csv")
    classical = read_classical(metrics_dir / "classical_baseline.csv")
    render_table(per_seq, classical, metrics_dir)


if __name__ == "__main__":
    main()
