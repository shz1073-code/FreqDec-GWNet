"""Classical respiratory-phase-estimation baselines.

For option-A paper story we claim: even though DL doesn't beat classical
methods at *pixel-level alignment* (Metric A), it does beat them at extracting
an *explicit respiratory phase* — the actual deliverable for ECG-free gating.

To make that claim we need apples-to-apples phase MAE between:
  * GRU stage-2 prediction (online per-frame inference)
  * Classical pure-signal baselines (this script)
  * Phase-B "gold" labels (offline batch with anchor optimization)

Methods implemented:

  DSH  — Diaphragm-Strip Hilbert. Mean intensity of a horizontal strip
         per frame -> band-pass 0.15-0.50 Hz -> Hilbert -> phase.
         Frame-by-frame inference (the simplest realistic online method).

  PCA  — PCA-on-strip. Top temporal principal component of the strip
         pixel matrix, then band-pass + Hilbert.

  PCM  — Phase-Correlation Motion. Frame-to-frame phase correlation gives
         (du, dv); cumulative trajectory projected onto its principal
         axis, then band-pass + Hilbert. Closest to what optical-flow
         compensation pipelines actually use.

Per sequence we save (cos, sin, amplitude) as a StateLabel-shaped NPZ
into <output-dir>/<method>/<dataset>__<split>__<seq>.npz so downstream
evaluation can load any method uniformly.

Usage:
    python scripts/classical_phase_baselines.py \
        --dataset dense_v1 --splits train val test \
        --methods DSH PCA PCM --output-dir reports/classical_phase_v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freqdec_gwnet.data.data_paths import DataPaths      # noqa: E402

# ---------------------------------------------------------------------------
# Common: 1D respiratory phase extraction
# ---------------------------------------------------------------------------


def _bandpass(signal: np.ndarray, fs: float, lo: float, hi: float,
              order: int = 4) -> np.ndarray:
    nyq = fs * 0.5
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    if len(signal) <= 3 * max(len(a), len(b)):
        return signal - np.mean(signal)
    return filtfilt(b, a, signal)


def _hilbert_phase(filtered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """analytic signal -> (cos phi, sin phi, amplitude)."""
    z = hilbert(filtered)
    amp = np.abs(z)
    amp = np.clip(amp, 1e-6, None)
    return z.real / amp, z.imag / amp, amp


def _scalar_to_state(signal: np.ndarray, fs: float, lo: float, hi: float):
    filt = _bandpass(signal, fs, lo, hi)
    cos_p, sin_p, amp = _hilbert_phase(filt)
    return cos_p, sin_p, amp


# ---------------------------------------------------------------------------
# Frame loader
# ---------------------------------------------------------------------------


def _load_strip_signal(seq_img_dir: Path, strip_rel=(0.65, 0.95)) -> np.ndarray:
    """Return [T] mean-intensity time series over a bottom-strip of frames."""
    files = sorted(seq_img_dir.glob("*.png")) + sorted(seq_img_dir.glob("*.jpg"))
    if not files:
        return np.zeros(0, dtype=np.float32)
    signal = []
    h0 = h1 = None
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            signal.append(signal[-1] if signal else 0.0)
            continue
        H = img.shape[0]
        if h0 is None:
            h0 = int(H * strip_rel[0]); h1 = int(H * strip_rel[1])
        signal.append(float(img[h0:h1].mean()))
    return np.asarray(signal, dtype=np.float32)


def _load_strip_matrix(seq_img_dir: Path, strip_rel=(0.65, 0.95),
                       downscale: int = 4) -> np.ndarray:
    """Return [T, H_strip*W/downscale^2] for PCA."""
    files = sorted(seq_img_dir.glob("*.png")) + sorted(seq_img_dir.glob("*.jpg"))
    rows = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            rows.append(rows[-1] if rows else None)
            continue
        H = img.shape[0]
        h0 = int(H * strip_rel[0]); h1 = int(H * strip_rel[1])
        strip = img[h0:h1]
        if downscale > 1:
            strip = cv2.resize(
                strip,
                (strip.shape[1] // downscale, strip.shape[0] // downscale),
                interpolation=cv2.INTER_AREA,
            )
        rows.append(strip.flatten().astype(np.float32))
    return np.stack(rows, axis=0) if rows else np.zeros((0, 0))


def _phase_correlation_displacement(seq_img_dir: Path) -> np.ndarray:
    """Return [T, 2] cumulative (u, v) trajectory via OpenCV phase correlation."""
    files = sorted(seq_img_dir.glob("*.png")) + sorted(seq_img_dir.glob("*.jpg"))
    if not files:
        return np.zeros((0, 2), dtype=np.float32)
    prev = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    cum = np.zeros(2, dtype=np.float64)
    trace = [cum.copy()]
    win = cv2.createHanningWindow(prev.shape[::-1], cv2.CV_32F)
    for f in files[1:]:
        cur = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if cur is None or cur.shape != prev.shape:
            trace.append(cum.copy())
            continue
        (du, dv), _ = cv2.phaseCorrelate(prev, cur, win)
        cum += [du, dv]
        trace.append(cum.copy())
        prev = cur
    return np.asarray(trace, dtype=np.float32)


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


def method_DSH(seq_img_dir: Path, fs: float, lo: float, hi: float):
    sig = _load_strip_signal(seq_img_dir)
    if len(sig) < 8:
        return None
    return _scalar_to_state(sig, fs, lo, hi)


def method_PCA(seq_img_dir: Path, fs: float, lo: float, hi: float):
    M = _load_strip_matrix(seq_img_dir)
    if M.size == 0 or M.shape[0] < 8:
        return None
    M = M - M.mean(axis=0, keepdims=True)
    # Top temporal PC: largest singular value over time.
    # SVD on [T, D] gives U[T,k] * S * V[k,D].  Top temporal score = U[:, 0] * S[0].
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    pc1 = U[:, 0] * S[0]
    return _scalar_to_state(pc1.astype(np.float32), fs, lo, hi)


def method_PCM(seq_img_dir: Path, fs: float, lo: float, hi: float):
    traj = _phase_correlation_displacement(seq_img_dir)
    if len(traj) < 8:
        return None
    traj_centered = traj - traj.mean(axis=0)
    _, _, Vt = np.linalg.svd(traj_centered, full_matrices=False)
    axis = Vt[0]                                  # dominant 2D direction
    proj = traj_centered @ axis
    return _scalar_to_state(proj.astype(np.float32), fs, lo, hi)


METHODS = {"DSH": method_DSH, "PCA": method_PCA, "PCM": method_PCM}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--seq", nargs="+", default=None)
    p.add_argument("--methods", nargs="+", default=list(METHODS.keys()),
                   choices=list(METHODS.keys()))
    p.add_argument("--fs", type=float, default=15.0)
    p.add_argument("--band-low-hz", type=float, default=0.15)
    p.add_argument("--band-high-hz", type=float, default=0.50)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    paths = DataPaths.from_default_config() if args.config is None \
        else DataPaths.from_yaml(args.config)
    if args.dataset not in paths.datasets:
        raise SystemExit(f"unknown dataset '{args.dataset}'")
    spec = paths.datasets[args.dataset]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for split in args.splits:
        if split not in spec.splits:
            continue
        img_root = spec.split_kind_dir(split, spec.images_subdir)
        if not img_root.is_dir():
            continue
        seqs = sorted([s.name for s in img_root.iterdir() if s.is_dir()])
        if args.seq:
            seqs = [s for s in seqs if s in args.seq]
        for seq in seqs:
            seq_dir = img_root / seq
            for method_name in args.methods:
                fn = METHODS[method_name]
                out = fn(seq_dir, args.fs, args.band_low_hz, args.band_high_hz)
                if out is None:
                    print(f"  SKIP {method_name} {split}/{seq} (too short)")
                    continue
                cos_p, sin_p, amp = out
                method_dir = args.output_dir / method_name
                method_dir.mkdir(parents=True, exist_ok=True)
                np.savez(
                    method_dir / f"{args.dataset}__{split}__{seq}.npz",
                    cos_phi=cos_p.astype(np.float32),
                    sin_phi=sin_p.astype(np.float32),
                    amplitude=amp.astype(np.float32),
                    num_frames=np.int64(len(cos_p)),
                )
                summary.append(
                    (method_name, args.dataset, split, seq, len(cos_p))
                )
                print(f"  {method_name:4s} {split:5s} {seq:14s} T={len(cos_p)}")

    print(f"\nwrote {len(summary)} classical phase estimates -> {args.output_dir}")


if __name__ == "__main__":
    main()
