"""Tests for the §6.1 main ablation #2 — state training protocol.

Three protocols are exposed by ``train_stage2_state.py``:
    * ``random_zero_init``      — default; standard shuffled DataLoader.
    * ``context_prefix``        — implemented via burn_in_min_frames /
                                  burn_in_weight knobs of StateLoss.
    * ``chronological_carry``   — ChronologicalSampler + h0 carry-over.

This file covers:
    * ChronologicalSampler grouping + ordering invariants.
    * Reproducibility under set_epoch.
    * is_sequence_boundary helper edge cases.
    * End-to-end: when chronological_carry is active, the predicted state
      at frame 0 of window k+1 differs from the prediction at the same
      frame given a fresh h0 — proving the carry actually flows.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest
import torch

from freqdec_gwnet.data import (
    ChronologicalSampler,
    FluoroSequenceWindowDataset,
    is_sequence_boundary,
)
from freqdec_gwnet.data.state_labels.pipeline import StateLabel
from freqdec_gwnet.models.freqdec_gwnet import FreqDecGWNet


# ---------------------------------------------------------------------------
# Synthetic dataset on disk (3 sequences × 80 frames each)
# ---------------------------------------------------------------------------


def _build_dataset(tmp_path: Path, n_seqs: int = 3, frames_per_seq: int = 80):
    """Mirror the helper in test_sequence_window_dataset.py."""
    images_root = tmp_path / "data" / "train" / "images"
    labels_root = tmp_path / "data" / "train" / "labels"
    state_dir = tmp_path / "state_labels"
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    for i in range(n_seqs):
        seq_name = f"seq_{i:03d}"
        seq_img = images_root / seq_name
        seq_lbl = labels_root / seq_name
        seq_img.mkdir()
        seq_lbl.mkdir()
        for t in range(frames_per_seq):
            cv2.imwrite(
                str(seq_img / f"f_{t:04d}.png"),
                (rng.uniform(0, 255, size=(48, 48))).astype(np.uint8),
            )
            cv2.imwrite(
                str(seq_lbl / f"f_{t:04d}.png"),
                ((rng.uniform(0, 1, size=(48, 48)) > 0.95) * 255).astype(np.uint8),
            )
        T = frames_per_seq
        cos_phi = np.cos(np.linspace(0, 2 * np.pi * 3, T)).astype(np.float32)
        sin_phi = np.sin(np.linspace(0, 2 * np.pi * 3, T)).astype(np.float32)
        amp = np.full(T, 5.0, dtype=np.float32)
        valid = np.ones(T, dtype=bool)
        StateLabel(
            seq_name=seq_name, dataset="synthetic", split="train",
            cos_phi=cos_phi, sin_phi=sin_phi, amplitude=amp,
            amp_scale=5.0, valid_mask=valid,
            anchor_quality=np.full(T, 0.8, dtype=np.float32),
            reference_index=5, reference_strategy="warmup_score",
            metadata={"fs": 15.0},
        ).save_npz(state_dir / f"{seq_name}.npz")
    return tmp_path / "data", state_dir


def _make_window_dataset(tmp_path: Path) -> FluoroSequenceWindowDataset:
    data_root, state_dir = _build_dataset(tmp_path)
    return FluoroSequenceWindowDataset(
        data_root=data_root, state_labels_dir=state_dir,
        split="train", T_window=32, stride=16, img_size=(48, 48),
    )


# ---------------------------------------------------------------------------
# ChronologicalSampler grouping + ordering
# ---------------------------------------------------------------------------


def test_sampler_groups_windows_by_sequence(tmp_path):
    ds = _make_window_dataset(tmp_path)
    sampler = ChronologicalSampler(ds, shuffle_sequences=False, seed=0)
    indices: List[int] = list(sampler)
    # 每条序列的所有窗口必须连续出现在采样序列中（不被打散）
    seqs_in_order = [ds.windows[i].seq_name for i in indices]
    # 把 [s0, s0, s0, s1, s1, ...] 简化成 [s0, s1, ...]
    runs = [s for k, s in enumerate(seqs_in_order)
            if k == 0 or s != seqs_in_order[k - 1]]
    # 每个 seq_name 应该只出现 1 次（连续区间）
    assert len(runs) == len(set(runs))


def test_sampler_orders_windows_chronologically(tmp_path):
    ds = _make_window_dataset(tmp_path)
    sampler = ChronologicalSampler(ds, shuffle_sequences=False, seed=0)
    indices = list(sampler)
    # 逐序列检查 window_start 是非递减的
    last_seq = None
    last_start = -1
    for i in indices:
        w = ds.windows[i]
        if w.seq_name != last_seq:
            last_seq = w.seq_name
            last_start = -1
        assert w.window_start > last_start, (
            f"窗口在序列 {w.seq_name} 内不是严格递增："
            f"prev={last_start} curr={w.window_start}"
        )
        last_start = w.window_start


def test_sampler_set_epoch_reproducible(tmp_path):
    ds = _make_window_dataset(tmp_path)
    a = ChronologicalSampler(ds, shuffle_sequences=True, seed=42)
    b = ChronologicalSampler(ds, shuffle_sequences=True, seed=42)
    a.set_epoch(7); b.set_epoch(7)
    assert list(a) == list(b)


def test_sampler_set_epoch_differs_across_epochs(tmp_path):
    ds = _make_window_dataset(tmp_path)
    s = ChronologicalSampler(ds, shuffle_sequences=True, seed=42)
    s.set_epoch(0); ord_0 = list(s)
    s.set_epoch(1); ord_1 = list(s)
    # epoch 切换至少要改变序列起点的顺序（窗口数足够时）
    seq_starts_0 = [ds.windows[i].seq_name for i in ord_0
                    if i == ord_0[0] or ds.windows[i].seq_name
                    != ds.windows[ord_0[ord_0.index(i) - 1]].seq_name]
    if len(set(seq_starts_0)) >= 2:
        # 多于 1 条序列时，shuffle 应该至少有时改变顺序
        assert ord_0 != ord_1 or seq_starts_0 != seq_starts_0


def test_sampler_no_shuffle_is_alphabetical(tmp_path):
    ds = _make_window_dataset(tmp_path)
    s = ChronologicalSampler(ds, shuffle_sequences=False, seed=0)
    seqs_seen: List[str] = []
    for i in s:
        n = ds.windows[i].seq_name
        if not seqs_seen or seqs_seen[-1] != n:
            seqs_seen.append(n)
    assert seqs_seen == sorted(seqs_seen)


def test_sampler_len_matches_total_windows(tmp_path):
    ds = _make_window_dataset(tmp_path)
    s = ChronologicalSampler(ds)
    assert len(s) == len(ds)
    assert s.num_sequences == 3


def test_sampler_windows_per_sequence_matches(tmp_path):
    ds = _make_window_dataset(tmp_path)
    s = ChronologicalSampler(ds)
    counts = s.windows_per_sequence()
    assert sum(counts.values()) == len(ds)
    assert all(v > 0 for v in counts.values())


# ---------------------------------------------------------------------------
# is_sequence_boundary helper
# ---------------------------------------------------------------------------


def test_is_sequence_boundary_first_batch():
    """No previous batch -> all elements are boundaries."""
    assert is_sequence_boundary(None, ["a", "b", "c"]) == [True, True, True]


def test_is_sequence_boundary_no_change():
    assert is_sequence_boundary(["a", "b"], ["a", "b"]) == [False, False]


def test_is_sequence_boundary_partial_change():
    assert is_sequence_boundary(["a", "b", "c"], ["a", "x", "c"]) == [
        False, True, False,
    ]


def test_is_sequence_boundary_size_mismatch_resets_all():
    """If batch sizes differ (e.g. last partial batch), reset everything."""
    assert is_sequence_boundary(["a", "b", "c"], ["a", "b"]) == [True, True]


# ---------------------------------------------------------------------------
# End-to-end: h0 carry actually flows through GRU forward
# ---------------------------------------------------------------------------


def _tiny_model() -> FreqDecGWNet:
    return FreqDecGWNet(
        in_channels=1, num_classes=1, width_mult=0.5,
        state_proj_dim=16, state_hidden_dim=16, state_lp_kernel=3,
        motion_hidden_dim=16,
    )


def test_h0_carry_changes_state_prediction_at_window_start():
    """Same image input, different h0 (zero vs carried) => different state[0].

    This proves the chronological_carry mechanism actually flows hidden
    state across windows; if the GRU ignored h0, fresh-vs-carried would
    give identical predictions.
    """
    model = _tiny_model()
    model.eval()
    images = torch.randn(1, 6, 1, 32, 32)

    with torch.no_grad():
        # Fresh h0=None
        out_fresh = model(images, mode="stage2_state", h0=None)
        # Carried h0=non-zero (simulates state from a previous window)
        h0_carry = torch.randn(1, 16) * 0.5
        out_carry = model(images, mode="stage2_state", h0=h0_carry)

    state_fresh = out_fresh["state"]
    state_carry = out_carry["state"]
    # 第 0 帧明显不同（h0 直接进入 GRUCell 的第一次计算）
    diff = (state_fresh[:, 0] - state_carry[:, 0]).abs().max().item()
    assert diff > 1e-4, (
        f"h0 carry-over had no effect on state[0] (max diff = {diff})"
    )


def test_h0_carry_per_sample_zero_mask():
    """When mask[i]=0 for sample i, that row's h_carry must be zeroed."""
    h_carry = torch.tensor([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]])
    boundaries = [True, False]                               # 第 0 个 reset
    mask = torch.tensor(
        [0.0 if b else 1.0 for b in boundaries], dtype=h_carry.dtype,
    ).view(-1, 1)
    h0 = h_carry * mask
    assert torch.allclose(h0[0], torch.zeros(3))
    assert torch.allclose(h0[1], h_carry[1])
