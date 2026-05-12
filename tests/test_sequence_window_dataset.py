"""Tests for FluoroSequenceWindowDataset.

Builds a tiny synthetic dataset on disk (a few short bursts + matching
StateLabel .npz files) and verifies windowing, label slicing, and the
reference-index window-local remapping.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from freqdec_gwnet.data import FluoroSequenceWindowDataset
from freqdec_gwnet.data.state_labels.pipeline import StateLabel


# ---------------------------------------------------------------------------
# Synthetic dataset builder
# ---------------------------------------------------------------------------


def _build_synthetic_dataset(
    root: Path, state_dir: Path, n_seqs: int, frames_per_seq: int,
    H: int = 64, W: int = 64,
):
    """Create images/labels/.npz files for ``n_seqs`` synthetic bursts."""
    images_root = root / "train" / "images"
    labels_root = root / "train" / "labels"
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    seq_names = []
    for i in range(n_seqs):
        seq_name = f"seq_{i:03d}"
        seq_names.append(seq_name)
        seq_img = images_root / seq_name
        seq_lbl = labels_root / seq_name
        seq_img.mkdir(exist_ok=True)
        seq_lbl.mkdir(exist_ok=True)

        # 写帧 + label
        for t in range(frames_per_seq):
            img = (rng.uniform(0, 255, size=(H, W))).astype(np.uint8)
            mask = ((rng.uniform(0, 1, size=(H, W)) > 0.95) * 255).astype(np.uint8)
            cv2.imwrite(str(seq_img / f"frame_{t:04d}.png"), img)
            cv2.imwrite(str(seq_lbl / f"frame_{t:04d}.png"), mask)

        # 写一份合成 StateLabel
        T = frames_per_seq
        cos_phi = np.cos(np.linspace(0, 2 * np.pi * 3, T)).astype(np.float32)
        sin_phi = np.sin(np.linspace(0, 2 * np.pi * 3, T)).astype(np.float32)
        amplitude = (5.0 + 1.0 * np.sin(np.linspace(0, 2 * np.pi * 2, T))).astype(np.float32)
        valid = np.ones(T, dtype=bool)
        valid[:5] = False                                          # warmup invalid
        sl = StateLabel(
            seq_name=seq_name, dataset="synthetic", split="train",
            cos_phi=cos_phi, sin_phi=sin_phi, amplitude=amplitude,
            amp_scale=float(np.percentile(amplitude[valid], 95.0)),
            valid_mask=valid,
            anchor_quality=np.full(T, 0.8, dtype=np.float32),
            reference_index=10,
            reference_strategy="warmup_score",
            metadata={"fs": 15.0, "n_frames": T},
        )
        sl.save_npz(state_dir / f"{seq_name}.npz")

    return seq_names


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dataset_discovers_windows(tmp_path):
    root = tmp_path / "data"
    state_dir = tmp_path / "state_labels"
    _build_synthetic_dataset(root, state_dir, n_seqs=3, frames_per_seq=80)

    ds = FluoroSequenceWindowDataset(
        data_root=root, state_labels_dir=state_dir,
        split="train", T_window=32, stride=16, img_size=(64, 64),
    )
    # 3 seqs × 80 frames, T=32 stride=16 → windows per seq ≈ (80-32)/16 + 1 = 4
    # 末窗保底可能再加 0 或 1（这里 (80-32)/16 = 3 整除 → 末窗已覆盖）
    assert len(ds) >= 3 * 4
    assert len(ds) <= 3 * 6


def test_dataset_sample_shape_and_keys(tmp_path):
    root = tmp_path / "data"
    state_dir = tmp_path / "state_labels"
    _build_synthetic_dataset(root, state_dir, n_seqs=2, frames_per_seq=64)

    ds = FluoroSequenceWindowDataset(
        data_root=root, state_labels_dir=state_dir,
        split="train", T_window=32, stride=16, img_size=(64, 64),
    )
    sample = ds[0]
    expected_keys = {
        "images", "wire_mask", "cos_phi", "sin_phi", "amplitude",
        "valid_mask", "amp_scale", "reference_index", "seq_name",
        "window_start",
    }
    assert expected_keys.issubset(sample.keys())
    # Shapes
    assert sample["images"].shape == (32, 1, 64, 64)
    assert sample["wire_mask"].shape == (32, 1, 64, 64)
    assert sample["cos_phi"].shape == (32,)
    assert sample["sin_phi"].shape == (32,)
    assert sample["amplitude"].shape == (32,)
    assert sample["valid_mask"].shape == (32,)
    assert sample["valid_mask"].dtype == torch.bool
    assert sample["amp_scale"].dim() == 0
    # Range checks
    assert sample["images"].min() >= 0.0 and sample["images"].max() <= 1.0
    assert sample["wire_mask"].min() >= 0.0 and sample["wire_mask"].max() <= 1.0


def test_dataset_reference_index_remapped_into_window(tmp_path):
    """全局 reference_index=10；当窗口起点 = 0 时窗内 ref = 10；起点 = 16 时 ref = max(0, 10-16) = 0."""
    root = tmp_path / "data"
    state_dir = tmp_path / "state_labels"
    _build_synthetic_dataset(root, state_dir, n_seqs=1, frames_per_seq=64)

    ds = FluoroSequenceWindowDataset(
        data_root=root, state_labels_dir=state_dir,
        split="train", T_window=32, stride=16, img_size=(64, 64),
    )
    starts = sorted({int(ds[i]["window_start"]) for i in range(len(ds))})
    # 找 window_start=0 和 window_start>=16 的两个样本
    s0 = next(ds[i] for i in range(len(ds)) if int(ds[i]["window_start"]) == 0)
    s_after = next(
        ds[i] for i in range(len(ds))
        if int(ds[i]["window_start"]) >= 16
    )
    assert int(s0["reference_index"]) == 10              # 仍在窗口内
    # ref 落在窗口前 → 应被夹到 0
    assert int(s_after["reference_index"]) == 0


def test_dataset_skips_too_short_sequences(tmp_path):
    root = tmp_path / "data"
    state_dir = tmp_path / "state_labels"
    # 2 条短序列 (16 帧) + 1 条够长 (64 帧)
    _build_synthetic_dataset(root, state_dir, n_seqs=1, frames_per_seq=64)
    # 单独写一条短的
    short_img = root / "train" / "images" / "short_seq"
    short_lbl = root / "train" / "labels" / "short_seq"
    short_img.mkdir()
    short_lbl.mkdir()
    for t in range(16):
        cv2.imwrite(
            str(short_img / f"frame_{t:04d}.png"),
            np.zeros((64, 64), dtype=np.uint8),
        )
        cv2.imwrite(
            str(short_lbl / f"frame_{t:04d}.png"),
            np.zeros((64, 64), dtype=np.uint8),
        )

    ds = FluoroSequenceWindowDataset(
        data_root=root, state_labels_dir=state_dir,
        split="train", T_window=32, stride=16, img_size=(64, 64),
    )
    # 只剩长序列里的窗口；short_seq 因为缺 state label + 太短被跳掉
    seq_names_in_ds = {ds.windows[i].seq_name for i in range(len(ds))}
    assert "short_seq" not in seq_names_in_ds


def test_dataset_excluded_sequences(tmp_path):
    root = tmp_path / "data"
    state_dir = tmp_path / "state_labels"
    _build_synthetic_dataset(root, state_dir, n_seqs=3, frames_per_seq=64)

    ds = FluoroSequenceWindowDataset(
        data_root=root, state_labels_dir=state_dir,
        split="train", T_window=32, stride=16, img_size=(64, 64),
        excluded_sequences=["seq_001"],
    )
    seqs = {ds.windows[i].seq_name for i in range(len(ds))}
    assert "seq_001" not in seqs
    assert "seq_000" in seqs and "seq_002" in seqs


def test_dataset_rejects_too_small_T_window(tmp_path):
    root = tmp_path / "data"
    state_dir = tmp_path / "state_labels"
    _build_synthetic_dataset(root, state_dir, n_seqs=1, frames_per_seq=64)
    with pytest.raises(ValueError, match="T_window"):
        FluoroSequenceWindowDataset(
            data_root=root, state_labels_dir=state_dir,
            split="train", T_window=8,
        )


def test_dataset_amp_scale_matches_state_label(tmp_path):
    """amp_scale 来自 .npz，应该与 StateLabel 字段一致。"""
    root = tmp_path / "data"
    state_dir = tmp_path / "state_labels"
    _build_synthetic_dataset(root, state_dir, n_seqs=1, frames_per_seq=64)

    ds = FluoroSequenceWindowDataset(
        data_root=root, state_labels_dir=state_dir,
        split="train", T_window=32, stride=16, img_size=(64, 64),
    )
    sl = StateLabel.load_npz(state_dir / "seq_000.npz")
    sample = ds[0]
    assert float(sample["amp_scale"]) == pytest.approx(sl.amp_scale, rel=1e-5)
