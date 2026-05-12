"""Tests for ``FluoroSequenceLoader``.

Synthetic tests (with ``tmp_path``) check the I/O correctness without needing
real data; ``test_real_*`` tests run against the live dataset and skip
gracefully when not mounted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from freqdec_gwnet.data import (
    DataPaths,
    FluoroSequence,
    FluoroSequenceLoader,
)


# ---------------------------------------------------------------------------
# helpers — build a tiny fake sequence on disk
# ---------------------------------------------------------------------------


def _save_grey(path: Path, value: int, hw=(8, 8)) -> None:
    """Save a constant-value uint8 image."""
    arr = np.full(hw, value, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path)


def _save_binary(path: Path, on: bool, hw=(8, 8)) -> None:
    """Save a fully-on or fully-off binary mask (0 / 255)."""
    arr = np.full(hw, 255 if on else 0, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path)


def _build_loader_sandbox(tmp_path: Path) -> tuple[DataPaths, str]:
    """Project sandbox: 1 dataset 'tiny' / split 'train' / seq 'seq_a' /
    3 frames with constant grey and alternating mask coverage.
    Tip CSV annotates frames 0 and 2 only."""
    root = tmp_path / "tiny"
    seq = root / "train" / "images" / "seq_a"
    lab = root / "train" / "labels" / "seq_a"
    for i in range(3):
        _save_grey(seq / f"frame_{i:03d}.png", value=10 + i * 10)
        _save_binary(lab / f"frame_{i:03d}.png", on=(i % 2 == 0))

    tip_csv = tmp_path / "tip.csv"
    with tip_csv.open("w", encoding="utf-8") as fh:
        fh.write("annotation_id,split,seq_name,frame_name,image_relpath,"
                 "reference_mask_relpath,tip_index,tip_x,tip_y,"
                 "num_tips_in_frame,visibility,quality,annotator_notes,"
                 "source_dataset\n")
        fh.write("a,train,seq_a,frame_000.png,_,_,0,3.0,4.0,1,single_tip,ok,,tiny\n")
        fh.write("a,train,seq_a,frame_002.png,_,_,0,7.0,1.0,1,single_tip,ok,,tiny\n")

    cfg = {
        "datasets": {"tiny": {
            "root": str(root),
            "splits": ["train"],
            "images_subdir": "images",
            "labels_subdir": "labels",
            "layout": "split_then_kind",
        }},
        "annotations": {
            "tip_csv": str(tip_csv),
            "tip_source_datasets": {"tiny": "tiny"},
        },
    }
    cfg_path = tmp_path / "data_paths.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return DataPaths.from_yaml(cfg_path), "tiny"


# ---------------------------------------------------------------------------
# Synthetic tests
# ---------------------------------------------------------------------------


def test_load_sequence_shapes_and_dtypes(tmp_path):
    paths, ds = _build_loader_sandbox(tmp_path)
    loader = FluoroSequenceLoader(paths)
    seq = loader.load_sequence(ds, "train", "seq_a")

    assert isinstance(seq, FluoroSequence)
    assert seq.num_frames == 3
    assert seq.frames.shape == (3, 1, 8, 8)
    assert seq.masks.shape == (3, 1, 8, 8)
    assert seq.frames.dtype == torch.float32
    assert seq.masks.dtype == torch.float32


def test_load_sequence_normalizes_image_to_unit_range(tmp_path):
    paths, ds = _build_loader_sandbox(tmp_path)
    seq = FluoroSequenceLoader(paths).load_sequence(ds, "train", "seq_a")
    # 帧 0 全是 10/255，帧 1 全是 20/255，帧 2 全是 30/255
    assert torch.allclose(seq.frames[0], torch.full_like(seq.frames[0], 10 / 255))
    assert torch.allclose(seq.frames[2], torch.full_like(seq.frames[2], 30 / 255))


def test_load_sequence_mask_is_binary(tmp_path):
    paths, ds = _build_loader_sandbox(tmp_path)
    seq = FluoroSequenceLoader(paths).load_sequence(ds, "train", "seq_a")
    # 帧 0 和 2 mask 全 1，帧 1 mask 全 0
    assert torch.equal(seq.masks[0], torch.ones_like(seq.masks[0]))
    assert torch.equal(seq.masks[1], torch.zeros_like(seq.masks[1]))
    assert torch.equal(seq.masks[2], torch.ones_like(seq.masks[2]))


def test_tip_xy_nan_for_unannotated_frames(tmp_path):
    paths, ds = _build_loader_sandbox(tmp_path)
    seq = FluoroSequenceLoader(paths).load_sequence(ds, "train", "seq_a")

    assert seq.tip_present.tolist() == [True, False, True]
    assert seq.tip_xy[0].tolist() == [3.0, 4.0]
    # frame 1 没有标注，xy 必须是 NaN
    assert torch.isnan(seq.tip_xy[1]).all().item()
    assert seq.tip_xy[2].tolist() == [7.0, 1.0]


def test_max_frames_truncates(tmp_path):
    paths, ds = _build_loader_sandbox(tmp_path)
    seq = FluoroSequenceLoader(paths).load_sequence(
        ds, "train", "seq_a", max_frames=2,
    )
    assert seq.num_frames == 2
    assert seq.frame_names == ["frame_000.png", "frame_001.png"]


def test_load_sequence_with_no_frames_raises(tmp_path):
    """An empty sequence directory should fail loudly, not return T=0."""
    paths, ds = _build_loader_sandbox(tmp_path)
    # 把 seq_a 的所有帧都"掏空" labels，让 list_frames 返回空
    lab = paths.datasets[ds].root / "train" / "labels" / "seq_a"
    for f in lab.iterdir():
        f.unlink()
    with pytest.raises(RuntimeError, match="no usable frames"):
        FluoroSequenceLoader(paths).load_sequence(ds, "train", "seq_a")


# ---------------------------------------------------------------------------
# Real-data tests (auto-skip)
# ---------------------------------------------------------------------------


def _real_paths_or_skip() -> DataPaths:
    try:
        return DataPaths.from_default_config()
    except FileNotFoundError as exc:
        pytest.skip(f"real config missing: {exc}")


def test_real_load_dense_v1_cmu_seq_007():
    paths = _real_paths_or_skip()
    if "dense_v1" not in paths.datasets:
        pytest.skip("dense_v1 not configured")
    seq = FluoroSequenceLoader(paths).load_sequence(
        "dense_v1", "train", "cmu_seq_007", max_frames=20,
    )
    assert seq.num_frames == 20
    assert seq.frames.shape[2:] == (512, 512)
    assert seq.masks.shape[2:] == (512, 512)
    # tip CSV 在 dense_v1 几乎每帧都有标注
    assert int(seq.tip_present.sum().item()) >= 15


def test_real_dense_v1_tip_coords_in_image_bounds():
    paths = _real_paths_or_skip()
    if "dense_v1" not in paths.datasets:
        pytest.skip("dense_v1 not configured")
    seq = FluoroSequenceLoader(paths).load_sequence(
        "dense_v1", "train", "cmu_seq_007", max_frames=10,
    )
    H, W = seq.frames.shape[-2], seq.frames.shape[-1]
    valid = seq.tip_present.nonzero(as_tuple=True)[0]
    for i in valid.tolist():
        x, y = seq.tip_xy[i].tolist()
        assert 0 <= x < W and 0 <= y < H, (
            f"tip ({x}, {y}) out of bounds for {seq.frame_names[i]}"
        )
