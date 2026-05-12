import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate temporal stability from localization_results.csv.")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to localization_results.csv")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Defaults to <csv-dir>/stability_metrics.json",
    )
    return parser.parse_args()


def _maybe_point(x_str, y_str):
    if x_str == "" or y_str == "":
        return None
    return (float(x_str), float(y_str))


def _dist(point_a, point_b):
    if point_a is None or point_b is None:
        return None
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def _mean_or_none(values):
    return float(np.mean(values)) if values else None


def _median_or_none(values):
    return float(np.median(values)) if values else None


def _p90_or_none(values):
    return float(np.percentile(values, 90)) if values else None


def _safe_std(values):
    return float(np.std(values)) if values else None


def _match_endpoint_order(prev_pair, current_pair):
    if prev_pair is None or current_pair is None:
        return current_pair

    curr_a, curr_b = current_pair
    prev_a, prev_b = prev_pair
    direct = _dist(prev_a, curr_a) + _dist(prev_b, curr_b)
    swapped = _dist(prev_a, curr_b) + _dist(prev_b, curr_a)
    return current_pair if direct <= swapped else (curr_b, curr_a)


def evaluate(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["seq_name"]].append(row)

    total_rows = len(rows)
    tip_success = 0
    tip_jitter_values = []
    endpoint_jitter_values = []
    centerline_len_deltas = []
    sequence_success_flags = []
    per_sequence = []

    for seq_name, seq_rows in grouped.items():
        seq_rows = sorted(seq_rows, key=lambda item: item["target_frame"])
        seq_tip_success = 0
        seq_tip_jitter = []
        seq_endpoint_jitter = []
        seq_centerline_lengths = []
        prev_tip = None
        prev_endpoints = None

        for row in seq_rows:
            tip = _maybe_point(row["tip_x"], row["tip_y"])
            ep1 = _maybe_point(row["endpoint_1_x"], row["endpoint_1_y"])
            ep2 = _maybe_point(row["endpoint_2_x"], row["endpoint_2_y"])
            endpoints = (ep1, ep2) if ep1 is not None and ep2 is not None else None
            centerline_points = int(row["centerline_points"])
            seq_centerline_lengths.append(centerline_points)

            if tip is not None:
                seq_tip_success += 1
                tip_success += 1

            tip_delta = _dist(prev_tip, tip)
            if tip_delta is not None:
                seq_tip_jitter.append(tip_delta)
                tip_jitter_values.append(tip_delta)
            prev_tip = tip if tip is not None else prev_tip

            if endpoints is not None:
                endpoints = _match_endpoint_order(prev_endpoints, endpoints)
                if prev_endpoints is not None:
                    d1 = _dist(prev_endpoints[0], endpoints[0])
                    d2 = _dist(prev_endpoints[1], endpoints[1])
                    if d1 is not None and d2 is not None:
                        endpoint_delta = 0.5 * (d1 + d2)
                        seq_endpoint_jitter.append(endpoint_delta)
                        endpoint_jitter_values.append(endpoint_delta)
                prev_endpoints = endpoints

        if len(seq_centerline_lengths) >= 2:
            diffs = np.abs(np.diff(seq_centerline_lengths)).tolist()
            centerline_len_deltas.extend(diffs)

        success_ratio = seq_tip_success / max(len(seq_rows), 1)
        sequence_success_flags.append(success_ratio >= 0.9)
        per_sequence.append(
            {
                "seq_name": seq_name,
                "num_windows": len(seq_rows),
                "tip_success_ratio": success_ratio,
                "mean_tip_jitter": _mean_or_none(seq_tip_jitter),
                "mean_endpoint_jitter": _mean_or_none(seq_endpoint_jitter),
                "centerline_points_std": _safe_std(seq_centerline_lengths),
            }
        )

    metrics = {
        "csv_path": os.path.abspath(csv_path),
        "num_windows": total_rows,
        "num_sequences": len(grouped),
        "tip_success_ratio": tip_success / max(total_rows, 1),
        "mean_tip_jitter": _mean_or_none(tip_jitter_values),
        "median_tip_jitter": _median_or_none(tip_jitter_values),
        "p90_tip_jitter": _p90_or_none(tip_jitter_values),
        "mean_endpoint_jitter": _mean_or_none(endpoint_jitter_values),
        "median_endpoint_jitter": _median_or_none(endpoint_jitter_values),
        "p90_endpoint_jitter": _p90_or_none(endpoint_jitter_values),
        "mean_centerline_length_delta": _mean_or_none(centerline_len_deltas),
        "median_centerline_length_delta": _median_or_none(centerline_len_deltas),
        "sequence_success_rate_ge_0.9_tip": float(np.mean(sequence_success_flags)) if sequence_success_flags else None,
        "per_sequence": per_sequence,
    }
    return metrics


def main():
    args = parse_args()
    metrics = evaluate(args.csv_path)
    output_json = args.output_json or os.path.join(os.path.dirname(args.csv_path), "stability_metrics.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_sequence"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
