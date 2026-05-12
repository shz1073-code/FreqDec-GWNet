#!/usr/bin/env python
"""Aggregate per-sequence drift CSVs into a paper-ready ablation table.

Reads every ``{run}/{seq}.csv`` under ``--input-dir`` and produces a single
``summary.csv`` whose rows are ``(run, seq)`` and whose columns are the
metrics most consumed by a §5/§8 ablation table:

    n_frames, n_landmark_frames,
    reference_index, reference_strategy,
    improvement_mean,
    drift_max, drift_mean, drift_endpoint,
    cycle_available, cycle_E_cycle,
    state_source, model

Optionally also emits a wide ``pivot.csv`` where each row is a sequence
and each column is the ``improvement_mean`` (or another configurable
metric) of one run, so the table can be pasted straight into a paper.

Usage::

    # aggregate everything under reports/ablation/
    python scripts/aggregate_drift_results.py --input-dir reports/ablation \\
        --output-csv reports/ablation_summary.csv \\
        --pivot-metric improvement_mean \\
        --pivot-csv  reports/ablation_pivot.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional


def parse_drift_csv(path: Path) -> Dict[str, object]:
    """Parse one drift CSV (produced by evaluate_drift.py) into a flat dict."""
    meta: Dict[str, object] = {}
    drift_norms: List[float] = []
    last_U_norm: Optional[float] = None
    with path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header_seen = False
        col_idx: Dict[str, int] = {}
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                # 形如 "# key=value"
                kv = row[0][1:].strip().split("=", 1)
                if len(kv) == 2:
                    meta[kv[0].strip()] = kv[1].strip()
                continue
            if not header_seen:
                col_idx = {name: i for i, name in enumerate(row)}
                header_seen = True
                continue
            # 数据行
            try:
                drift_norm = float(row[col_idx["drift_norm"]] or "nan")
            except (KeyError, ValueError):
                drift_norm = float("nan")
            if not math.isnan(drift_norm):
                drift_norms.append(drift_norm)
                last_U_norm = drift_norm
    n = len(drift_norms)
    meta["drift_max"] = max(drift_norms) if n else float("nan")
    meta["drift_mean"] = (sum(drift_norms) / n) if n else float("nan")
    meta["drift_endpoint"] = last_U_norm if last_U_norm is not None else float("nan")
    return meta


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------


_LONG_COLUMNS = [
    "run", "seq",
    "n_frames", "n_landmark_frames",
    "reference_index", "reference_strategy",
    "improvement_mean",
    "drift_max", "drift_mean", "drift_endpoint",
    "cycle_available", "cycle_E_cycle",
    "state_source", "model",
]


def _format(v) -> str:
    if isinstance(v, float):
        if v != v:
            return ""
        return f"{v:.6g}"
    return str(v)


def write_long_csv(rows: List[Dict[str, object]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_LONG_COLUMNS)
        for r in rows:
            w.writerow([_format(r.get(c, "")) for c in _LONG_COLUMNS])
    return path


def write_pivot_csv(
    rows: List[Dict[str, object]], path: Path, metric: str,
) -> Path:
    """Wide-format ``seq × run`` table for paper appendix tables."""
    seqs = sorted({r["seq"] for r in rows})
    runs = sorted({r["run"] for r in rows})
    table: Dict[str, Dict[str, object]] = {s: {} for s in seqs}
    for r in rows:
        table[r["seq"]][r["run"]] = r.get(metric, "")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seq"] + runs)
        for s in seqs:
            w.writerow([s] + [_format(table[s].get(run, "")) for run in runs])
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Aggregate per-sequence drift CSVs into paper-ready tables",
    )
    p.add_argument("--input-dir", type=Path, required=True,
                   help="root dir containing {run_name}/{seq}.csv files")
    p.add_argument("--output-csv", type=Path,
                   default=Path("reports/ablation_summary.csv"))
    p.add_argument("--pivot-csv", type=Path, default=None,
                   help="optional wide-format output")
    p.add_argument("--pivot-metric", default="improvement_mean")
    args = p.parse_args(list(argv) if argv is not None else None)

    if not args.input_dir.is_dir():
        raise SystemExit(f"input-dir not found: {args.input_dir}")

    rows: List[Dict[str, object]] = []
    for run_dir in sorted(args.input_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for csv_path in sorted(run_dir.glob("*.csv")):
            if csv_path.name in ("summary.csv", "pivot.csv"):
                continue
            try:
                meta = parse_drift_csv(csv_path)
            except Exception as exc:
                print(f"⚠️  skipping {csv_path}: {exc}")
                continue
            meta["run"] = run_dir.name
            meta["seq"] = csv_path.stem
            rows.append(meta)

    if not rows:
        raise SystemExit(f"no parseable CSVs under {args.input_dir}")

    write_long_csv(rows, args.output_csv)
    print(f"📊 wrote long-format summary: {args.output_csv} ({len(rows)} rows)")

    if args.pivot_csv is not None:
        write_pivot_csv(rows, args.pivot_csv, args.pivot_metric)
        print(f"📊 wrote pivot ({args.pivot_metric}): {args.pivot_csv}")

    # Quick stdout overview
    print("\n=== quick view (run, mean improvement_mean) ===")
    by_run: Dict[str, List[float]] = {}
    for r in rows:
        try:
            v = float(r.get("improvement_mean", ""))
        except (TypeError, ValueError):
            continue
        by_run.setdefault(r["run"], []).append(v)
    for run, vals in sorted(by_run.items()):
        if vals:
            print(f"  {run:36s}  mean improvement_mean = "
                  f"{sum(vals)/len(vals):+.3f}  (over {len(vals)} seqs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
