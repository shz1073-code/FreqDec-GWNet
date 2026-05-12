"""Tests for ``freqdec_gwnet.data.data_paths``.

Most tests build synthetic dataset trees with ``tmp_path`` so they run on any
machine; a small ``test_real_*`` group hits the actual ``configs/data_paths.yaml``
and is auto-skipped when the data isn't mounted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from freqdec_gwnet.data import DataPaths, DatasetSpec, TipAnnotation


# ---------------------------------------------------------------------------
# helpers: build a fake project tree
# ---------------------------------------------------------------------------


def _touch(p: Path, content: str = "") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def _make_split_then_kind_dataset(
    root: Path,
    splits=("train", "val"),
    seqs=("seq_a", "seq_b"),
    n_frames=3,
    images_subdir="images",
    labels_subdir="labels",
) -> None:
    """Create ``root/<split>/{images,labels}/<seq>/frame_NNN.png`` skeleton."""
    for sp in splits:
        for seq in seqs:
            for i in range(n_frames):
                fname = f"frame_{i:03d}.png"
                _touch(root / sp / images_subdir / seq / fname, "img")
                _touch(root / sp / labels_subdir / seq / fname, "msk")


def _make_kind_then_split_dataset(
    root: Path,
    splits=("train",),
    seqs=("seq_a",),
    n_frames=3,
    images_subdir="raw_images",
    labels_subdir="reference_masks",
) -> None:
    """Create ``root/{raw_images,reference_masks}/<split>/<seq>/frame_NNN.png``."""
    for sp in splits:
        for seq in seqs:
            for i in range(n_frames):
                fname = f"frame_{i:03d}.png"
                _touch(root / images_subdir / sp / seq / fname, "img")
                _touch(root / labels_subdir / sp / seq / fname, "msk")


def _write_tip_csv(path: Path, rows: list[dict]) -> None:
    """Tiny CSV writer that matches the merged_v2 schema we read at runtime."""
    cols = [
        "annotation_id", "split", "seq_name", "frame_name",
        "image_relpath", "reference_mask_relpath",
        "tip_index", "tip_x", "tip_y", "num_tips_in_frame",
        "visibility", "quality", "annotator_notes", "source_dataset",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


def _build_minimal_project(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create a fully consistent (yaml + datasets + tip csv) sandbox.

    Returns (config_path, ds_a_root, ds_b_root, tip_csv_path).
    """
    ds_a = tmp_path / "ds_a"
    ds_b = tmp_path / "ds_b"
    _make_split_then_kind_dataset(
        ds_a, splits=("train", "val"), seqs=("seq_a", "seq_b"), n_frames=3,
    )
    _make_kind_then_split_dataset(
        ds_b, splits=("train",), seqs=("seq_x",), n_frames=2,
        images_subdir="raw_images", labels_subdir="reference_masks",
    )

    tip_csv = tmp_path / "tip_merged.csv"
    _write_tip_csv(tip_csv, [
        {"split": "train", "seq_name": "seq_a", "frame_name": "frame_000.png",
         "tip_x": "10.0", "tip_y": "20.0", "tip_index": 0,
         "visibility": "single_tip", "quality": "ok",
         "source_dataset": "ds_a"},
        {"split": "train", "seq_name": "seq_x", "frame_name": "frame_001.png",
         "tip_x": "5.5", "tip_y": "6.5", "tip_index": 0,
         "visibility": "single_tip", "quality": "ok",
         "source_dataset": "ds_b"},
        # An entry that points at a frame not on disk (negative case)
        {"split": "train", "seq_name": "seq_a", "frame_name": "frame_999.png",
         "tip_x": "1.0", "tip_y": "1.0", "tip_index": 0,
         "visibility": "single_tip", "quality": "ok",
         "source_dataset": "ds_a"},
    ])

    cfg = {
        "datasets": {
            "ds_a": {
                "root": str(ds_a),
                "splits": ["train", "val"],
                "images_subdir": "images",
                "labels_subdir": "labels",
                "layout": "split_then_kind",
            },
            "ds_b": {
                "root": str(ds_b),
                "splits": ["train"],
                "images_subdir": "raw_images",
                "labels_subdir": "reference_masks",
                "layout": "kind_then_split",
            },
        },
        "annotations": {
            "tip_csv": str(tip_csv),
            "tip_source_datasets": {"ds_a": "ds_a", "ds_b": "ds_b"},
        },
    }
    config_path = tmp_path / "data_paths.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    return config_path, ds_a, ds_b, tip_csv


# ---------------------------------------------------------------------------
# Synthetic-tree tests
# ---------------------------------------------------------------------------


def test_loads_minimal_project(tmp_path):
    cfg_path, _, _, _ = _build_minimal_project(tmp_path)
    paths = DataPaths.from_yaml(cfg_path)
    assert set(paths.datasets) == {"ds_a", "ds_b"}
    assert paths.datasets["ds_a"].layout == "split_then_kind"
    assert paths.datasets["ds_b"].layout == "kind_then_split"
    assert paths.tip_source_datasets == {"ds_a": "ds_a", "ds_b": "ds_b"}


def test_list_sequences_split_then_kind(tmp_path):
    cfg_path, _, _, _ = _build_minimal_project(tmp_path)
    paths = DataPaths.from_yaml(cfg_path)
    assert paths.list_sequences("ds_a", "train") == ["seq_a", "seq_b"]
    assert paths.list_sequences("ds_a", "val") == ["seq_a", "seq_b"]


def test_list_sequences_kind_then_split(tmp_path):
    cfg_path, _, _, _ = _build_minimal_project(tmp_path)
    paths = DataPaths.from_yaml(cfg_path)
    assert paths.list_sequences("ds_b", "train") == ["seq_x"]


def test_list_sequences_drops_label_less_seq(tmp_path):
    """A sequence with only images and no labels must be filtered out."""
    cfg_path, ds_a, _, _ = _build_minimal_project(tmp_path)
    # 偷偷加一个只有图、没 label 的序列
    _touch(ds_a / "train" / "images" / "seq_orphan" / "frame_000.png", "img")
    paths = DataPaths.from_yaml(cfg_path)
    assert "seq_orphan" not in paths.list_sequences("ds_a", "train")


def test_list_frames_returns_intersection(tmp_path):
    """Frames present only in images or only in labels must be excluded."""
    cfg_path, ds_a, _, _ = _build_minimal_project(tmp_path)
    # 给 seq_a 多塞一个只有 image 没 label 的帧
    _touch(ds_a / "train" / "images" / "seq_a" / "ghost.png", "img")
    paths = DataPaths.from_yaml(cfg_path)
    frames = paths.list_frames("ds_a", "train", "seq_a")
    assert "ghost.png" not in frames
    assert frames == ["frame_000.png", "frame_001.png", "frame_002.png"]


def test_sequence_dirs_resolves_per_layout(tmp_path):
    cfg_path, ds_a, ds_b, _ = _build_minimal_project(tmp_path)
    paths = DataPaths.from_yaml(cfg_path)

    img_a, lab_a = paths.sequence_dirs("ds_a", "train", "seq_a")
    assert img_a == ds_a / "train" / "images" / "seq_a"
    assert lab_a == ds_a / "train" / "labels" / "seq_a"

    img_b, lab_b = paths.sequence_dirs("ds_b", "train", "seq_x")
    assert img_b == ds_b / "raw_images" / "train" / "seq_x"
    assert lab_b == ds_b / "reference_masks" / "train" / "seq_x"


def test_missing_dataset_root_raises(tmp_path):
    cfg = {
        "datasets": {"phantom": {
            "root": str(tmp_path / "does-not-exist"),
            "splits": ["train"],
        }},
        "annotations": {"tip_csv": str(tmp_path / "fake.csv")},
    }
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(FileNotFoundError):
        DataPaths.from_yaml(cfg_path)


def test_invalid_layout_string_caught(tmp_path):
    """An unknown layout value must surface clearly, not silently break."""
    cfg_path, ds_a, _, _ = _build_minimal_project(tmp_path)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["datasets"]["ds_a"]["layout"] = "bogus_layout"
    cfg_path.write_text(yaml.safe_dump(cfg))
    # 加载阶段会调用 split_kind_dir 校验，触发 ValueError
    with pytest.raises(ValueError, match="unknown layout"):
        DataPaths.from_yaml(cfg_path)


def test_tip_source_pointing_at_unknown_dataset_rejected(tmp_path):
    cfg_path, _, _, _ = _build_minimal_project(tmp_path)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["annotations"]["tip_source_datasets"]["ghost"] = "ds_missing"
    cfg_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="no such dataset"):
        DataPaths.from_yaml(cfg_path)


def test_load_tip_csv_and_resolve(tmp_path):
    cfg_path, _, _, _ = _build_minimal_project(tmp_path)
    paths = DataPaths.from_yaml(cfg_path)

    rows = paths.load_tip_csv()
    assert len(rows) == 3
    first = rows[0]
    assert first.seq_name == "seq_a"
    assert first.tip_x == 10.0 and first.tip_y == 20.0

    # 第一行点到的图像存在 -> resolve 成功
    img_path = paths.resolve_tip_image(first)
    assert img_path.is_file()

    # 第三行点向 frame_999 不存在 -> 应该 FileNotFoundError
    ghost = rows[2]
    with pytest.raises(FileNotFoundError):
        paths.resolve_tip_image(ghost)


def test_verify_tip_alignment_reports_per_source(tmp_path):
    cfg_path, _, _, _ = _build_minimal_project(tmp_path)
    paths = DataPaths.from_yaml(cfg_path)
    rep = paths.verify_tip_alignment()
    assert rep["rows_total"] == 3
    assert rep["rows_matched"] == 2     # frame_999 不在盘上
    assert rep["rows_missing"] == 1
    by_src = rep["by_source"]
    assert by_src["ds_a"]["matched"] == 1
    assert by_src["ds_a"]["missing"] == 1
    assert by_src["ds_b"]["matched"] == 1


def test_tip_index_by_seq_groups_correctly(tmp_path):
    cfg_path, _, _, _ = _build_minimal_project(tmp_path)
    paths = DataPaths.from_yaml(cfg_path)
    idx = paths.tip_index_by_seq()
    assert set(idx.keys()) == {"seq_a", "seq_x"}
    assert "frame_000.png" in idx["seq_a"]
    assert idx["seq_x"]["frame_001.png"].tip_x == 5.5


# ---------------------------------------------------------------------------
# Integration tests against the real configs/data_paths.yaml (auto-skip)
# ---------------------------------------------------------------------------


def _real_config_or_skip() -> DataPaths:
    """Helper: load the real config; skip cleanly if datasets aren't mounted."""
    try:
        return DataPaths.from_default_config()
    except FileNotFoundError as exc:
        pytest.skip(f"real data not mounted: {exc}")


def test_real_dense_v1_cmu_seq_007_has_107_frames():
    paths = _real_config_or_skip()
    if "dense_v1" not in paths.datasets:
        pytest.skip("dense_v1 not in real config")
    frames = paths.list_frames("dense_v1", "train", "cmu_seq_007")
    assert len(frames) == 107


def test_real_tip_alignment_full_match():
    """Every row in the merged tip CSV must resolve to a real file."""
    paths = _real_config_or_skip()
    if not paths.tip_source_datasets:
        pytest.skip("tip_source_datasets not configured")
    rep = paths.verify_tip_alignment()
    assert rep["rows_missing"] == 0, (
        f"some tip rows missing on disk: {rep}"
    )
    assert rep["rows_total"] == rep["rows_matched"] > 0
