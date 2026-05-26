"""Phase-MAE comparison: GRU (DL) vs classical baselines vs Phase-B gold.

Pulls together three sources of per-frame respiratory phase on dense_v1:

  gold       — reports/state_labels/dense_v1/<split>/<seq>.npz
               (offline batch with anchor optimization; treated as ground truth)

  classical  — reports/classical_phase_v1/<method>/<dataset>__<split>__<seq>.npz
               (frame-by-frame online methods; from classical_phase_baselines.py)

  DL         — stage-2 GRU checkpoints; we run inference here per sequence

For each method, computes circular phase MAE (degrees) and Pearson on cos:

  raw       — direct comparison (penalizes per-sequence phase offset)
  aligned   — minimize over a constant per-sequence Δφ before measuring
              (separates "tracks respiration but with reference offset"
              from "fails to track")

Outputs CSV + a stdout table.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freqdec_gwnet.data.data_paths import DataPaths            # noqa: E402
from freqdec_gwnet.data.state_labels.pipeline import StateLabel  # noqa: E402
from freqdec_gwnet.models.freqdec_gwnet import FreqDecGWNet     # noqa: E402


# ---------------------------------------------------------------------------
# Phase comparison metrics
# ---------------------------------------------------------------------------


def circular_mae_deg(cos_p, sin_p, cos_g, sin_g, valid):
    """Per-frame circular distance |φ_p − φ_g| in degrees, averaged."""
    cos_p, sin_p, cos_g, sin_g, valid = [
        np.asarray(x, dtype=np.float64) for x in (cos_p, sin_p, cos_g, sin_g, valid)
    ]
    n_p = np.sqrt(cos_p ** 2 + sin_p ** 2).clip(min=1e-6)
    n_g = np.sqrt(cos_g ** 2 + sin_g ** 2).clip(min=1e-6)
    inner = (cos_p * cos_g + sin_p * sin_g) / (n_p * n_g)
    inner = np.clip(inner, -1.0, 1.0)
    delta_rad = np.arccos(inner)                          # [0, π]
    m = valid.astype(bool)
    if m.sum() == 0:
        return float("nan")
    return float(np.degrees(delta_rad[m]).mean())


def best_offset_mae_deg(cos_p, sin_p, cos_g, sin_g, valid):
    """Aligned MAE after subtracting the optimal constant phase offset.

    Optimal Δ = angle of Σ (z_p · conj(z_g)) over valid frames.
    """
    m = np.asarray(valid).astype(bool)
    z_p = (cos_p + 1j * sin_p)[m]
    z_g = (cos_g + 1j * sin_g)[m]
    cross = (z_p * np.conj(z_g)).sum()
    delta = np.angle(cross)                               # the phase offset
    # rotate predictions by -delta
    cos_pa = cos_p * math.cos(-delta) - sin_p * math.sin(-delta)
    sin_pa = sin_p * math.cos(-delta) + cos_p * math.sin(-delta)
    return circular_mae_deg(cos_pa, sin_pa, cos_g, sin_g, valid), math.degrees(delta)


def pearson_cos(cos_p, cos_g, valid):
    m = np.asarray(valid).astype(bool)
    if m.sum() < 3:
        return float("nan")
    a, b = np.asarray(cos_p)[m], np.asarray(cos_g)[m]
    a = a - a.mean(); b = b - b.mean()
    den = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / max(den, 1e-12))


# ---------------------------------------------------------------------------
# DL inference on a sequence
# ---------------------------------------------------------------------------


def dl_predict(model, paths, dataset, split, seq, img_size=512, T_chunk=32,
               branch: str = "resp"):
    """Run DL inference and return (cos_pred, sin_pred) for the chosen branch.

    branch="resp"     -> state output slice [0:2] (cos_r, sin_r)
    branch="cardiac"  -> state output slice [4:6] (cos_c, sin_c) — requires
                         the model to have been trained with state_output_dim=8
    """
    spec = paths.datasets[dataset]
    seq_dir = spec.split_kind_dir(split, spec.images_subdir) / seq
    files = sorted(seq_dir.glob("*.png")) + sorted(seq_dir.glob("*.jpg"))
    if not files:
        return None
    H = W = img_size
    imgs = np.zeros((len(files), 1, H, W), dtype=np.float32)
    for k, f in enumerate(files):
        im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        if im.shape != (H, W):
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
        imgs[k, 0] = im.astype(np.float32) / 255.0

    imgs_t = torch.from_numpy(imgs)
    device = next(model.parameters()).device
    feats = []
    with torch.no_grad():
        for s in range(0, len(files), T_chunk):
            e = min(len(files), s + T_chunk)
            f = model.r1.encoder(imgs_t[s:e].to(device))[2].cpu()
            feats.append(f)
    feat = torch.cat(feats, 0).unsqueeze(0).to(device)
    with torch.no_grad():
        st = model.state_branch.forward_window(feat)[0].cpu().numpy()
    if branch == "cardiac":
        if st.shape[1] < 8:
            return None                                # not a dual-head model
        cos_p, sin_p = st[:, 4], st[:, 5]
    else:
        cos_p, sin_p = st[:, 0], st[:, 1]
    norm = np.sqrt(cos_p ** 2 + sin_p ** 2).clip(min=1e-6)
    return cos_p / norm, sin_p / norm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dense_v1")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--gold-dir", type=Path,
                   default=ROOT / "reports/state_labels/dense_v1")
    p.add_argument("--classical-dir", type=Path,
                   default=ROOT / "reports/classical_phase_v1")
    p.add_argument("--classical-methods", nargs="+", default=["DSH", "PCA", "PCM"])
    p.add_argument("--stage2-ckpts", nargs="*", type=Path, default=[])
    p.add_argument("--burn-in", type=int, default=8,
                   help="skip first N frames (band-pass warm-up)")
    p.add_argument("--branch", choices=["resp", "cardiac"], default="resp",
                   help="which DL output branch to compare against gold")
    p.add_argument("--state-output-dim", type=int, default=4,
                   help="output dim of the DL model (set 8 to load dual-head)")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = DataPaths.from_default_config()
    spec = paths.datasets[args.dataset]

    # Discover sequences with gold labels. Support both split-subdir
    # layout (state_labels/<dataset>/<split>/<seq>.npz) and flat
    # layout (state_labels/<dataset>/<seq>.npz, which is what
    # generate_state_labels.py produces for dense_v1).
    seqs = []
    for split in args.splits:
        if split not in spec.splits:
            continue
        img_root = spec.split_kind_dir(split, spec.images_subdir)
        if not img_root.is_dir():
            continue
        for seq_dir in sorted(img_root.iterdir()):
            if not seq_dir.is_dir():
                continue
            seq = seq_dir.name
            split_npz = args.gold_dir / split / f"{seq}.npz"
            flat_npz = args.gold_dir / f"{seq}.npz"
            npz = split_npz if split_npz.is_file() else (flat_npz if flat_npz.is_file() else None)
            if npz is None:
                continue
            seqs.append((split, seq, npz))

    # Optionally load DL models.
    dl_models = {}
    if args.stage2_ckpts:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for ck_path in args.stage2_ckpts:
            model = FreqDecGWNet(
                width_mult=1.0, state_core_type="gru",
                state_output_dim=args.state_output_dim,
            ).to(device)
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            sd = ck.get("model_state_dict", ck)
            sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
            model.eval()
            dl_models[ck_path.stem] = model

    rows = []
    for split, seq, gold_npz in seqs:
        gold = StateLabel.load_npz(gold_npz)
        cos_g, sin_g = gold.cos_phi, gold.sin_phi
        valid = gold.valid_mask.copy()
        if args.burn_in > 0:
            valid[:args.burn_in] = False

        candidates = []   # (method_name, cos_p, sin_p)
        # classical
        for cm in args.classical_methods:
            cl_npz = args.classical_dir / cm / f"{args.dataset}__{split}__{seq}.npz"
            if not cl_npz.is_file():
                continue
            d = np.load(cl_npz)
            T_use = min(len(d["cos_phi"]), len(cos_g))
            candidates.append((cm, d["cos_phi"][:T_use], d["sin_phi"][:T_use]))
        # DL
        for name, model in dl_models.items():
            out = dl_predict(model, paths, args.dataset, split, seq,
                             branch=args.branch)
            if out is None:
                continue
            cos_p, sin_p = out
            T_use = min(len(cos_p), len(cos_g))
            candidates.append((f"DL:{name}", cos_p[:T_use], sin_p[:T_use]))

        T_use = min(len(cos_g), valid.shape[0])
        cos_gold = cos_g[:T_use]; sin_gold = sin_g[:T_use]; valid_use = valid[:T_use]
        for method, cp, sp in candidates:
            cp = cp[:T_use]; sp = sp[:T_use]
            raw_mae = circular_mae_deg(cp, sp, cos_gold, sin_gold, valid_use)
            aligned_mae, delta = best_offset_mae_deg(cp, sp, cos_gold, sin_gold, valid_use)
            r = pearson_cos(cp, cos_gold, valid_use)
            rows.append({
                "split": split, "seq": seq, "method": method,
                "T": T_use, "n_valid": int(valid_use.sum()),
                "raw_mae_deg": raw_mae, "aligned_mae_deg": aligned_mae,
                "best_delta_deg": delta, "pearson_cos": r,
            })

    # Write CSV
    import csv as _csv
    out_csv = args.output_dir / "phase_mae_comparison.csv"
    with out_csv.open("w") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[csv] {out_csv}")

    # Pretty print
    print()
    print(f"{'split':5s} {'seq':14s} {'method':35s} {'raw_MAE':>10s} {'align_MAE':>10s} {'best_Δ':>8s} {'rho':>7s}")
    print("-" * 96)
    for r in sorted(rows, key=lambda x: (x["split"], x["seq"], x["method"])):
        print(f"{r['split']:5s} {r['seq']:14s} {r['method']:35s} {r['raw_mae_deg']:>10.2f} "
              f"{r['aligned_mae_deg']:>10.2f} {r['best_delta_deg']:>+8.1f} {r['pearson_cos']:>+7.3f}")

    # Aggregate by method
    print()
    print(f"{'method':35s} {'n_seqs':>7s} {'raw_MAE_mean':>14s} {'align_MAE_mean':>16s} {'rho_mean':>9s}")
    print("-" * 88)
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    for m, rs in by_method.items():
        rmae = np.mean([r["raw_mae_deg"] for r in rs])
        amae = np.mean([r["aligned_mae_deg"] for r in rs])
        rho = np.mean([r["pearson_cos"] for r in rs])
        print(f"{m:35s} {len(rs):>7d} {rmae:>14.2f} {amae:>16.2f} {rho:>+9.3f}")


if __name__ == "__main__":
    main()
