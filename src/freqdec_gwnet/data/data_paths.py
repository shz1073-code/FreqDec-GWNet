"""Single source of truth for FreqDec-GWNet dataset paths.

Loads :file:`configs/data_paths.yaml`, validates every advertised path actually
exists, and exposes a small query API for the rest of the project::

    from freqdec_gwnet.data.data_paths import DataPaths

    paths = DataPaths.from_default_config()
    seqs = paths.list_sequences("clean", "train")
    img_dir, label_dir = paths.sequence_dirs("clean", "train", "cmu_seq_007")
    tip_df = paths.load_tip_csv()

The resolver intentionally fails loudly at startup (``FileNotFoundError`` on
any missing path) — much better than partway through training. Per-split
sub-CSVs are optional; if they are missing we silently skip them, since the
canonical merged CSV is enough for most flows.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# 数据集元信息
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset entry from data_paths.yaml.

    Attributes:
        name: dataset key (``"clean"`` / ``"aug"`` / ``"ori"`` / ``"dense_v1"``).
        root: absolute root path.
        splits: subset of ``{"train", "val", "test"}`` actually present.
        images_subdir / labels_subdir: subdir names holding frames vs masks.
        layout: directory ordering relative to ``root``:
            * ``"split_then_kind"``: ``<root>/<split>/<kind>/<seq>/<frame>``
              (used by clean / aug / ori).
            * ``"kind_then_split"``: ``<root>/<kind>/<split>/<seq>/<frame>``
              (used by tip_control_dense_v1).
        manifest_csv: optional auxiliary manifest (e.g. aug profiles).
    """

    name: str
    root: Path
    splits: Tuple[str, ...]
    images_subdir: str = "images"
    labels_subdir: str = "labels"
    layout: str = "split_then_kind"
    manifest_csv: Optional[Path] = None

    def split_kind_dir(self, split: str, kind: str) -> Path:
        """Resolve the directory holding all sequences of ``kind`` in ``split``.

        Encapsulates the layout difference so callers don't branch on it.
        """
        if self.layout == "split_then_kind":
            return self.root / split / kind
        if self.layout == "kind_then_split":
            return self.root / kind / split
        raise ValueError(
            f"unknown layout '{self.layout}' for dataset '{self.name}'"
        )


@dataclass
class TipAnnotation:
    """One row from ``tip_control_merged_v2.csv``.

    Coordinates are in the original 512×512 image frame (top-left origin,
    +x = right, +y = down).
    """

    seq_name: str
    frame_name: str
    tip_x: float
    tip_y: float
    split: str = "train"
    tip_index: int = 0
    visibility: str = "single_tip"
    quality: str = "ok"
    source_dataset: str = "unknown"


# ---------------------------------------------------------------------------
# 主资源解析器
# ---------------------------------------------------------------------------


@dataclass
class DataPaths:
    """Resolved + validated data paths."""

    config_path: Path
    datasets: Dict[str, DatasetSpec]
    tip_csv: Path
    tip_csv_split: Dict[str, Path] = field(default_factory=dict)
    # CSV ``source_dataset`` value -> dataset key in ``self.datasets``.
    tip_source_datasets: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, config_path: os.PathLike) -> "DataPaths":
        """Load ``configs/data_paths.yaml`` and validate everything exists."""
        config_path = Path(config_path).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"data_paths config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as fh:
            cfg: Dict[str, Any] = yaml.safe_load(fh)

        # ---- datasets ----
        datasets: Dict[str, DatasetSpec] = {}
        for name, spec in (cfg.get("datasets") or {}).items():
            root = Path(spec["root"])
            if not root.is_dir():
                raise FileNotFoundError(
                    f"dataset '{name}' root does not exist: {root}"
                )
            splits = tuple(spec.get("splits", []))
            images_subdir = spec.get("images_subdir", "images")
            labels_subdir = spec.get("labels_subdir", "labels")
            layout = spec.get("layout", "split_then_kind")
            ds = DatasetSpec(
                name=name,
                root=root,
                splits=splits,
                images_subdir=images_subdir,
                labels_subdir=labels_subdir,
                layout=layout,
                manifest_csv=None,
            )
            # 通过 split_kind_dir 校验，确保 layout 正确
            for sp in splits:
                for sub in (images_subdir, labels_subdir):
                    p = ds.split_kind_dir(sp, sub)
                    if not p.is_dir():
                        raise FileNotFoundError(
                            f"dataset '{name}' split '{sp}' missing dir: {p}"
                        )
            manifest = spec.get("manifest_csv")
            manifest_p = Path(manifest) if manifest else None
            if manifest_p is not None and not manifest_p.is_file():
                raise FileNotFoundError(
                    f"dataset '{name}' manifest_csv missing: {manifest_p}"
                )
            # 用最终的 manifest_p 重建一次（dataclass frozen 的小代价）
            datasets[name] = DatasetSpec(
                name=name,
                root=root,
                splits=splits,
                images_subdir=images_subdir,
                labels_subdir=labels_subdir,
                layout=layout,
                manifest_csv=manifest_p,
            )

        # ---- annotations ----
        ann = cfg.get("annotations") or {}
        tip_csv = Path(ann.get("tip_csv", ""))
        if not tip_csv.is_file():
            raise FileNotFoundError(f"tip_csv not found: {tip_csv}")

        tip_split: Dict[str, Path] = {}
        for split in ("train", "val", "test"):
            key = f"tip_csv_{split}"
            if key in ann:
                p = Path(ann[key])
                # split-level CSV 是可选的；缺失只警告（通过缺席）
                if p.is_file():
                    tip_split[split] = p

        tip_sources_raw = ann.get("tip_source_datasets") or {}
        tip_sources: Dict[str, str] = {}
        for src, ds_name in tip_sources_raw.items():
            if ds_name not in datasets:
                raise ValueError(
                    f"annotations.tip_source_datasets['{src}']='{ds_name}' "
                    f"but no such dataset; available: {list(datasets)}"
                )
            tip_sources[src] = ds_name

        return cls(
            config_path=config_path,
            datasets=datasets,
            tip_csv=tip_csv,
            tip_csv_split=tip_split,
            tip_source_datasets=tip_sources,
        )

    @classmethod
    def from_default_config(cls) -> "DataPaths":
        """Find ``configs/data_paths.yaml`` by walking up from this file.

        Lets unit tests and scripts call ``DataPaths.from_default_config()``
        without hard-coding the project root.
        """
        here = Path(__file__).resolve()
        # 向上找到 FreqDec-GWNet 根目录（包含 configs/data_paths.yaml）
        for ancestor in here.parents:
            candidate = ancestor / "configs" / "data_paths.yaml"
            if candidate.is_file():
                return cls.from_yaml(candidate)
        raise FileNotFoundError(
            "configs/data_paths.yaml not found in any ancestor of "
            f"{here}"
        )

    # ------------------------------------------------------------------
    # 序列查询
    # ------------------------------------------------------------------

    def get_dataset(self, name: str) -> DatasetSpec:
        if name not in self.datasets:
            raise KeyError(
                f"unknown dataset '{name}'; available: {list(self.datasets)}"
            )
        return self.datasets[name]

    def list_sequences(self, dataset: str, split: str) -> List[str]:
        """List sequence directory names under ``<root>/<split>/<images_subdir>``.

        The returned list is alphabetically sorted; sequences without a
        matching label directory are silently skipped so that the caller never
        picks up a label-less sequence by accident.
        """
        ds = self.get_dataset(dataset)
        if split not in ds.splits:
            raise KeyError(
                f"dataset '{dataset}' has no split '{split}'; "
                f"available: {list(ds.splits)}"
            )
        images_dir = ds.split_kind_dir(split, ds.images_subdir)
        labels_dir = ds.split_kind_dir(split, ds.labels_subdir)
        seqs = sorted(
            p.name for p in images_dir.iterdir()
            if p.is_dir() and (labels_dir / p.name).is_dir()
        )
        return seqs

    def sequence_dirs(
        self, dataset: str, split: str, seq_name: str,
    ) -> Tuple[Path, Path]:
        """Return ``(images_dir, labels_dir)`` for a single sequence."""
        ds = self.get_dataset(dataset)
        if split not in ds.splits:
            raise KeyError(
                f"dataset '{dataset}' has no split '{split}'"
            )
        images_dir = ds.split_kind_dir(split, ds.images_subdir) / seq_name
        labels_dir = ds.split_kind_dir(split, ds.labels_subdir) / seq_name
        if not images_dir.is_dir():
            raise FileNotFoundError(f"missing image dir: {images_dir}")
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"missing label dir: {labels_dir}")
        return images_dir, labels_dir

    def list_frames(
        self, dataset: str, split: str, seq_name: str,
    ) -> List[str]:
        """Sorted frame filenames present in *both* images and labels dirs.

        Mismatch (e.g. an image without a label) is silently dropped — keeps
        the caller from accidentally training on label-less frames.
        """
        images_dir, labels_dir = self.sequence_dirs(dataset, split, seq_name)
        img_set = {p.name for p in images_dir.iterdir() if p.is_file()}
        lbl_set = {p.name for p in labels_dir.iterdir() if p.is_file()}
        return sorted(img_set & lbl_set)

    # ------------------------------------------------------------------
    # Tip 标注查询
    # ------------------------------------------------------------------

    def load_tip_csv(
        self,
        split: Optional[str] = None,
    ) -> List[TipAnnotation]:
        """Load tip annotations.

        Args:
            split: if ``None`` returns the merged CSV; otherwise the
                pre-split CSV. If a split-level file is absent we fall back
                to filtering the merged CSV by its ``split`` column.
        """
        if split is None:
            return _read_tip_csv(self.tip_csv)
        if split in self.tip_csv_split:
            return _read_tip_csv(self.tip_csv_split[split])
        return [a for a in _read_tip_csv(self.tip_csv) if a.split == split]

    def resolve_tip_image(self, ann: TipAnnotation) -> Path:
        """Resolve the actual image file path for one ``TipAnnotation``.

        Looks up the dataset associated with ``ann.source_dataset`` and joins
        the on-disk path. Raises ``KeyError`` if the source is unknown,
        ``FileNotFoundError`` if the resolved path does not exist.
        """
        if ann.source_dataset not in self.tip_source_datasets:
            raise KeyError(
                f"tip CSV source_dataset='{ann.source_dataset}' not mapped "
                f"to any dataset; configure annotations.tip_source_datasets"
            )
        ds_name = self.tip_source_datasets[ann.source_dataset]
        ds = self.get_dataset(ds_name)
        path = ds.split_kind_dir(ann.split, ds.images_subdir) \
            / ann.seq_name / ann.frame_name
        if not path.is_file():
            raise FileNotFoundError(f"tip image missing on disk: {path}")
        return path

    def verify_tip_alignment(self) -> Dict[str, Any]:
        """Cross-check tip CSV rows against the configured image sources.

        Iterates every CSV row, looks up the matching dataset via
        ``tip_source_datasets``, and reports per-source totals plus which
        rows resolve to a real file on disk. A non-zero ``unresolved`` count
        means the tip CSV and the configured datasets have drifted apart
        and the YAML needs fixing.
        """
        if not self.tip_source_datasets:
            raise ValueError(
                "annotations.tip_source_datasets is not set in YAML"
            )

        # 把每个相关数据集的 (split, seq, frame) 一次性收成集合
        on_disk: Dict[str, set] = {}
        for ds_name in set(self.tip_source_datasets.values()):
            ds = self.get_dataset(ds_name)
            seen: set = set()
            for split in ds.splits:
                split_root = ds.split_kind_dir(split, ds.images_subdir)
                if not split_root.is_dir():
                    continue
                for seq_dir in split_root.iterdir():
                    if not seq_dir.is_dir():
                        continue
                    for frame in seq_dir.iterdir():
                        if frame.is_file():
                            seen.add((split, seq_dir.name, frame.name))
            on_disk[ds_name] = seen

        rows = _read_tip_csv(self.tip_csv)
        per_source: Dict[str, Dict[str, int]] = {}
        unresolved = 0
        for r in rows:
            ds_name = self.tip_source_datasets.get(r.source_dataset)
            entry = per_source.setdefault(
                r.source_dataset, {"total": 0, "matched": 0, "missing": 0},
            )
            entry["total"] += 1
            if ds_name is None:
                unresolved += 1
                entry["missing"] += 1
                continue
            if (r.split, r.seq_name, r.frame_name) in on_disk[ds_name]:
                entry["matched"] += 1
            else:
                entry["missing"] += 1
        return {
            "rows_total": len(rows),
            "rows_matched": sum(p["matched"] for p in per_source.values()),
            "rows_missing": sum(p["missing"] for p in per_source.values()),
            "rows_unresolved_source": unresolved,
            "by_source": per_source,
        }

    def tip_index_by_seq(
        self,
        split: Optional[str] = None,
    ) -> Dict[str, Dict[str, TipAnnotation]]:
        """Return ``{seq_name: {frame_name: TipAnnotation}}`` mapping.

        Convenient when iterating frames in order and asking "does this frame
        have a tip annotation?". O(1) lookup.
        """
        out: Dict[str, Dict[str, TipAnnotation]] = {}
        for ann in self.load_tip_csv(split):
            out.setdefault(ann.seq_name, {})[ann.frame_name] = ann
        return out


# ---------------------------------------------------------------------------
# CSV 读取（不引入 pandas，保持轻量）
# ---------------------------------------------------------------------------


def _read_tip_csv(csv_path: Path) -> List[TipAnnotation]:
    """Parse one tip-annotation CSV into ``TipAnnotation`` rows."""
    rows: List[TipAnnotation] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            try:
                rows.append(TipAnnotation(
                    seq_name=raw["seq_name"],
                    frame_name=raw["frame_name"],
                    tip_x=float(raw["tip_x"]),
                    tip_y=float(raw["tip_y"]),
                    split=raw.get("split", "train") or "train",
                    tip_index=int(raw.get("tip_index", 0) or 0),
                    visibility=raw.get("visibility", "single_tip") or "single_tip",
                    quality=raw.get("quality", "ok") or "ok",
                    source_dataset=raw.get("source_dataset", "unknown") or "unknown",
                ))
            except (KeyError, ValueError) as exc:
                # 单行损坏不应该让整次加载挂掉
                raise ValueError(
                    f"malformed tip CSV row in {csv_path}: {raw}"
                ) from exc
    return rows
