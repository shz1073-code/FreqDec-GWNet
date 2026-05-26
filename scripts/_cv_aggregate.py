"""Aggregate the 5-fold GRU CV results into one honest scoreboard.

Each fold model is evaluated only on its held-out sequences, so pooling
all fold evals gives exactly one held-out measurement per sequence (35
total). We report that pooled mean for the no-aug and aug arms and
compare against the fixed-split v5 baseline (val Pearson 0.175).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

REPORTS = Path("reports/gru_cv_v1")
BASELINE = "stage2_gru_chrono_v5_best (fixed 6-seq val): Pearson 0.175  MAE 75.7deg"


def _load(arm: str):
    """arm = 'noaug' or 'aug' -> list of per-seq rows."""
    rows = []
    for csv_path in sorted(REPORTS.glob("eval_*/state_quality.csv")):
        name = csv_path.parent.name
        is_aug = "_cv_aug_" in name
        if (arm == "aug") != is_aug:
            continue
        for r in csv.DictReader(open(csv_path)):
            try:
                rows.append({
                    "seq": r["seq"],
                    "mae": float(r["phase_mae_deg"]),
                    "r": float(r["phase_pearson"]),
                })
            except (KeyError, ValueError):
                pass
    return rows


def _stats(rows):
    if not rows:
        return None
    mae = [x["mae"] for x in rows]
    r = [x["r"] for x in rows]
    n = len(rows)
    mu_mae = sum(mae) / n
    mu_r = sum(r) / n
    sd_r = math.sqrt(sum((v - mu_r) ** 2 for v in r) / (n - 1)) if n > 1 else 0.0
    return {
        "n": n,
        "mae": mu_mae,
        "r": mu_r,
        "r_sd": sd_r,
        "abs_r": sum(abs(v) for v in r) / n,
        "good": sum(v > 0.5 for v in r),
        "fail": sum(v < 0.2 for v in r),
    }


def main():
    print("=" * 66)
    print("GRU 5-fold cross-validation — honest held-out scoreboard")
    print("=" * 66)
    print(f"baseline: {BASELINE}\n")

    arms = {}
    for arm in ("noaug", "aug"):
        s = _stats(_load(arm))
        arms[arm] = s
        if s is None:
            print(f"[{arm}] no eval data found")
            continue
        print(f"[{arm:6s}] n={s['n']:2d}  "
              f"phase_MAE={s['mae']:6.2f}deg  "
              f"Pearson={s['r']:+.3f} (sd {s['r_sd']:.3f})  "
              f"|Pearson|={s['abs_r']:.3f}  "
              f"good(r>.5)={s['good']:2d}  fail(r<.2)={s['fail']:2d}")

    if arms.get("noaug") and arms.get("aug"):
        d = arms["aug"]["r"] - arms["noaug"]["r"]
        print(f"\naugmentation effect on Pearson: {d:+.3f}")

    # per-sequence aug vs no-aug
    no = {x["seq"]: x for x in _load("noaug")}
    au = {x["seq"]: x for x in _load("aug")}
    common = sorted(set(no) & set(au))
    if common:
        print(f"\n{'sequence':22s} {'noaug_r':>9s} {'aug_r':>8s} {'delta':>8s}")
        print("-" * 50)
        for s in common:
            dr = au[s]["r"] - no[s]["r"]
            print(f"{s:22s} {no[s]['r']:+9.3f} {au[s]['r']:+8.3f} {dr:+8.3f}")


if __name__ == "__main__":
    main()
