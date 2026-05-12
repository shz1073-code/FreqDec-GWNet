"""Per-sequence loader for FreqDec-GWNet.

The R3 main path operates on whole bursts (cumulative summation along time),
so the natural unit of evaluation is *one sequence at a time*. This loader
takes a ``(dataset, split, seq_name)`` tuple and returns:

    * the raw frames stacked as ``[T, 1, H, W]`` float32 in ``[0, 1]``;
    * the wire masks aligned to the same frame names, binary float;
    * the optional tip landmark trajectory ``[T, 2]`` with ``NaN`` for frames
      that have no entry in the merged tip CSV.

A pytorch ``Dataset`` wrapper that yields *windows* over these sequences will
live on top of this — it is a separate concern and not needed for drift
evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from .data_paths import DataPaths, TipAnnotation


@dataclass
class FluoroSequence:
    """A single fluoroscopy burst loaded into memory.

    Attributes:
        dataset: dataset key (``"clean"``, ``"dense_v1"``, ...).
        split: ``"train"`` / ``"val"`` / ``"test"``.
        seq_name: e.g. ``"cmu_seq_007"``.
        frame_names: the ordered frame filenames; ``frames[i]`` corresponds
            to ``frame_names[i]``.
        frames: ``[T, 1, H, W]`` float32 tensor in ``[0, 1]``.
        masks: ``[T, 1, H, W]`` float32 binary mask (1 = wire).
        tip_xy: ``[T, 2]`` float32; columns are ``(x, y)`` in pixel coords.
            Frames without a tip annotation contain ``NaN``.
        tip_present: ``[T]`` boolean — true iff the row had a tip CSV entry.
    """

    dataset: str
    split: str
    seq_name: str
    frame_names: List[str]
    frames: torch.Tensor
    masks: torch.Tensor
    tip_xy: torch.Tensor
    tip_present: torch.Tensor

    @property
    def num_frames(self) -> int:
        return self.frames.shape[0]

    @property
    def shape(self) -> tuple:
        return tuple(self.frames.shape)


# ---------------------------------------------------------------------------
# helpers — image / mask -> tensor
# ---------------------------------------------------------------------------


def _load_grayscale_png(path: Path) -> torch.Tensor:
    """Load a PNG as ``[1, H, W]`` float32 in ``[0, 1]``."""
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    # uint8 [0, 255] -> float32 [0, 1]
    t = torch.from_numpy(arr.astype(np.float32) / 255.0)
    return t.unsqueeze(0)              # [1, H, W]


def _load_binary_mask(path: Path, threshold: float = 0.5) -> torch.Tensor:
    """Load a PNG mask as ``[1, H, W]`` float32 binary.

    Threshold defaults to 0.5 (post-normalization), so mask files saved as
    either 0/255 or 0/1 both load correctly.
    """
    t = _load_grayscale_png(path)
    return (t > threshold).to(torch.float32)


# ---------------------------------------------------------------------------
# main loader
# ---------------------------------------------------------------------------


class FluoroSequenceLoader:
    """Loads whole sequences for FreqDec-GWNet evaluation pipelines.

    The loader is stateless beyond the :class:`DataPaths` it caches; it does
    *no* augmentation, no random sampling, no windowing — those are upstream
    concerns. It guarantees:

        * ``frames`` and ``masks`` have identical ``frame_names`` ordering;
        * frames missing a label are dropped (already enforced by
          :meth:`DataPaths.list_frames`);
        * tip annotations are looked up by ``(seq_name, frame_name)`` and
          inserted into ``tip_xy`` at the matching time index, leaving
          ``NaN`` for frames that have no row in the CSV.
    """

    def __init__(self, paths: DataPaths):
        self.paths = paths

    def load_sequence(
        self,
        dataset: str,
        split: str,
        seq_name: str,
        max_frames: Optional[int] = None,
    ) -> FluoroSequence:
        """Load one sequence into memory.

        Args:
            dataset: dataset key registered in data_paths.yaml.
            split: ``"train"`` / ``"val"`` / ``"test"``.
            seq_name: sequence directory name (e.g. ``"cmu_seq_007"``).
            max_frames: optional cap on T, useful for quick smoke tests.
        """
        images_dir, labels_dir = self.paths.sequence_dirs(
            dataset, split, seq_name,
        )
        frame_names = self.paths.list_frames(dataset, split, seq_name)
        if max_frames is not None:
            frame_names = frame_names[:max_frames]
        if not frame_names:
            raise RuntimeError(
                f"sequence {dataset}/{split}/{seq_name} has no usable frames"
            )

        frames = torch.stack(
            [_load_grayscale_png(images_dir / fn) for fn in frame_names],
            dim=0,
        )                                               # [T, 1, H, W]
        masks = torch.stack(
            [_load_binary_mask(labels_dir / fn) for fn in frame_names],
            dim=0,
        )                                               # [T, 1, H, W]

        tip_xy, tip_present = self._load_tip_trajectory(
            split, seq_name, frame_names,
        )

        return FluoroSequence(
            dataset=dataset,
            split=split,
            seq_name=seq_name,
            frame_names=frame_names,
            frames=frames,
            masks=masks,
            tip_xy=tip_xy,
            tip_present=tip_present,
        )

    # ------------------------------------------------------------------
    # internal: attach tip annotations matching this sequence
    # ------------------------------------------------------------------

    def _load_tip_trajectory(
        self,
        split: str,
        seq_name: str,
        frame_names: List[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(tip_xy [T, 2], tip_present [T])`` for ``frame_names``."""
        T = len(frame_names)
        tip_xy = torch.full((T, 2), float("nan"))
        tip_present = torch.zeros(T, dtype=torch.bool)

        rows = self.paths.load_tip_csv(split)
        # 只保留这一序列的行，再按 frame_name 建索引（per-call cost 1 次扫描）
        per_frame: dict[str, TipAnnotation] = {}
        for r in rows:
            if r.seq_name == seq_name:
                per_frame[r.frame_name] = r

        for i, fname in enumerate(frame_names):
            ann = per_frame.get(fname)
            if ann is None:
                continue
            tip_xy[i, 0] = ann.tip_x
            tip_xy[i, 1] = ann.tip_y
            tip_present[i] = True
        return tip_xy, tip_present
