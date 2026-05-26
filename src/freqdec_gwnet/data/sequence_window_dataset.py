"""Window sampler over full fluoroscopy bursts (used by stage 2 / stage 3).

The legacy :class:`RealGuidewireVideoDataset` slices each burst into
overlapping 5-frame windows because FAST-LiteNet only needed short-term
context. Stage 2 / Stage 3 of FreqDec-GWNet need *long* windows
(``T_window`` ≥ 64) to expose at least one full respiratory cycle to the
state branch. This dataset:

    * keeps the burst as a single unit;
    * samples non-redundant windows by stride;
    * loads the matching :class:`StateLabel` ``.npz`` produced by Phase B;
    * (optionally) loads the wire-mask ``labels/`` PNGs as the active
      exclusion mask consumed by R4's :class:`MotionLoss`.

Returned sample (dict):

    images       : float32 [T, 1, H, W] in [0, 1]
    wire_mask    : float32 [T, 1, H, W] in {0, 1}
    cos_phi      : float32 [T]
    sin_phi      : float32 [T]
    amplitude    : float32 [T]
    valid_mask   : bool    [T]
    amp_scale    : float32 scalar
    reference_index : int  (from the StateLabel)
    seq_name     : str
    window_start : int

Splits and per-sequence state-label files are decoupled so the same dataset
can host different label-generation runs (e.g. before vs. after stage 1
wire-mask injection).
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

# OpenCV multi-thread stability under DataLoader workers
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from .state_labels.pipeline import StateLabel


# ---------------------------------------------------------------------------
# Sample index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _WindowIndex:
    """Pointer into a single window: which sequence, which start frame."""
    seq_name: str
    seq_dir_images: Path
    seq_dir_labels: Optional[Path]
    state_label_path: Path
    frame_names: Tuple[str, ...]                  # all frames in this seq
    window_start: int                             # 0 ≤ start ≤ T_seq − T_window


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FluoroSequenceWindowDataset(Dataset):
    """Stride-windowed fluoroscopy burst dataset with state labels."""

    def __init__(
        self,
        data_root: str,
        state_labels_dir: str,
        split: str = "train",
        *,
        T_window: int = 64,
        stride: Optional[int] = None,             # default: T_window // 2
        img_size: Tuple[int, int] = (512, 512),
        require_wire_mask: bool = True,
        require_state_label: bool = True,
        min_burst_frames: Optional[int] = None,    # default: T_window
        excluded_sequences: Optional[List[str]] = None,
        included_sequences: Optional[List[str]] = None,
        state_labels_subdir_per_split: bool = True,
        augment: bool = False,
        augment_seed: int = 0,
        cardiac_labels_dir: Optional[str] = None,  # 启用双周期解耦时设置
    ):
        self.data_root = Path(data_root)
        # Phase B 的 generate_state_labels.py 默认按 {split}/{seq}.npz 落盘，
        # 所以 state_labels_dir 通常指向 .../clean/  而非 .../clean/train/。
        # 这里按 split 自动追加子目录；如果用户已经指向具体 split 目录则关掉。
        sl_root = Path(state_labels_dir)
        if state_labels_subdir_per_split and (sl_root / split).is_dir():
            self.state_labels_dir = sl_root / split
        else:
            self.state_labels_dir = sl_root
        self.split = split
        self.T_window = int(T_window)
        self.stride = int(stride) if stride is not None else self.T_window // 2
        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.require_wire_mask = bool(require_wire_mask)
        self.require_state_label = bool(require_state_label)
        self.min_burst_frames = (
            int(min_burst_frames)
            if min_burst_frames is not None else self.T_window
        )
        self.excluded_sequences = set(excluded_sequences or [])
        # If given, ONLY these sequences are kept (applied after exclusion).
        # Used by k-fold cross-validation to carve a fold out of a pooled
        # sequence directory.
        self.included_sequences = (
            set(included_sequences) if included_sequences is not None else None
        )
        self.augment = bool(augment)
        self.augment_seed = int(augment_seed)
        # Cardiac labels: if dir given, mirror state_labels_dir layout.
        if cardiac_labels_dir is not None:
            card_root = Path(cardiac_labels_dir)
            if state_labels_subdir_per_split and (card_root / split).is_dir():
                self.cardiac_labels_dir = card_root / split
            else:
                self.cardiac_labels_dir = card_root
        else:
            self.cardiac_labels_dir = None

        if self.T_window < 16:
            raise ValueError(
                f"T_window={self.T_window} too small; need ≥16 frames for "
                "stable Hilbert / band-pass on the state branch path"
            )
        if self.stride < 1:
            raise ValueError(f"stride={self.stride} must be >= 1")

        # 假设 split-then-kind 布局（clean / aug / ori 都是这样）
        self.images_dir = self.data_root / split / "images"
        self.labels_dir = self.data_root / split / "labels"
        if not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"images dir not found: {self.images_dir}; "
                "this dataset assumes split-then-kind layout (clean/aug/ori)"
            )

        # 缓存状态标签的存在性（路径用 {state_labels_dir}/{seq_name}.npz）
        self.windows: List[_WindowIndex] = []
        self._scan_sequences()

        if not self.windows:
            raise RuntimeError(
                f"no usable windows discovered under {self.images_dir} "
                f"(state_labels at {self.state_labels_dir}). "
                f"Either generate state labels first or relax "
                f"require_state_label."
            )

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    def _scan_sequences(self) -> None:
        skipped: Dict[str, int] = {
            "excluded": 0,
            "not_included": 0,
            "no_label_dir": 0,
            "no_state_label": 0,
            "too_short": 0,
        }
        for seq_name in sorted(os.listdir(self.images_dir)):
            seq_img = self.images_dir / seq_name
            if not seq_img.is_dir():
                continue
            if seq_name in self.excluded_sequences:
                skipped["excluded"] += 1
                continue
            if (self.included_sequences is not None
                    and seq_name not in self.included_sequences):
                skipped["not_included"] += 1
                continue

            seq_lbl = self.labels_dir / seq_name
            if self.require_wire_mask and not seq_lbl.is_dir():
                skipped["no_label_dir"] += 1
                continue

            state_path = self.state_labels_dir / f"{seq_name}.npz"
            if self.require_state_label and not state_path.is_file():
                skipped["no_state_label"] += 1
                continue

            frames = tuple(sorted(os.listdir(seq_img)))
            if len(frames) < self.min_burst_frames:
                skipped["too_short"] += 1
                continue

            # Phase-B can produce labels shorter than the image folder
            # (warmup-edge trimming or quality-driven truncation). The
            # window sampler must respect the label's length too — otherwise
            # __getitem__ raises mid-training.
            try:
                sl_peek = StateLabel.load_npz(state_path)
                effective_T = min(len(frames), int(sl_peek.num_frames))
            except Exception:                                    # corrupt npz
                skipped["no_state_label"] += 1
                continue
            if effective_T < self.min_burst_frames:
                skipped["too_short"] += 1
                continue

            # 滑窗位置：起点从 0 到 (T_seq − T_window)
            T_seq = effective_T
            last_start = T_seq - self.T_window
            for start in range(0, last_start + 1, self.stride):
                self.windows.append(_WindowIndex(
                    seq_name=seq_name,
                    seq_dir_images=seq_img,
                    seq_dir_labels=seq_lbl if seq_lbl.is_dir() else None,
                    state_label_path=state_path,
                    frame_names=frames,
                    window_start=start,
                ))
            # 末窗保底（若 stride 没刚好覆盖最后位置，补一个）
            if self.windows and self.windows[-1].window_start != last_start:
                self.windows.append(_WindowIndex(
                    seq_name=seq_name,
                    seq_dir_images=seq_img,
                    seq_dir_labels=seq_lbl if seq_lbl.is_dir() else None,
                    state_label_path=state_path,
                    frame_names=frames,
                    window_start=last_start,
                ))

        # 缓存 state label（一条序列只读一次）
        self._state_label_cache: Dict[str, StateLabel] = {}

        n_seqs = len({w.seq_name for w in self.windows})
        print(
            f"[FluoroSequenceWindowDataset {self.split}] "
            f"sequences={n_seqs} windows={len(self.windows)} "
            f"T_window={self.T_window} stride={self.stride}"
        )
        for k, v in skipped.items():
            if v:
                print(f"  skipped[{k}]={v}")

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.windows)

    # ------------------------------------------------------------------
    # Real-data augmentation (k-fold CV training)
    # ------------------------------------------------------------------

    def _augment_window(
        self, images: np.ndarray, wire: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply ONE random augmentation, identical across all T frames.

        A burst-consistent transform leaves frame-to-frame respiratory
        motion untouched, so the state labels stay valid. Two families:

        * geometric — small affine (rotation/scale/translation) applied
          to image + wire-mask with the SAME matrix;
        * photometric — brightness/contrast/gamma/noise on the image only.

        Stochastic per call (entropy-seeded) so each epoch sees fresh
        variants of every window.
        """
        rng = np.random.default_rng()
        T, _, H, W = images.shape

        # --- geometric: one affine matrix for the whole window ---
        angle = float(rng.uniform(-5.0, 5.0))
        scale = float(rng.uniform(0.95, 1.05))
        tx = float(rng.uniform(-0.04, 0.04)) * W
        ty = float(rng.uniform(-0.04, 0.04)) * H
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        do_geom = angle != 0.0 or scale != 1.0 or tx != 0.0 or ty != 0.0

        # --- photometric params (image only) ---
        brightness = float(rng.uniform(-0.10, 0.10))
        contrast = float(rng.uniform(0.85, 1.15))
        gamma = float(rng.uniform(0.80, 1.25))
        noise_sigma = float(rng.uniform(0.0, 0.02))
        flip = bool(rng.random() < 0.5)

        out_img = np.empty_like(images)
        out_wire = np.empty_like(wire)
        for k in range(T):
            im = images[k, 0]
            wm = wire[k, 0]
            if do_geom:
                im = cv2.warpAffine(
                    im, M, (W, H), flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                wm = cv2.warpAffine(
                    wm, M, (W, H), flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
            if flip:
                im = im[:, ::-1]
                wm = wm[:, ::-1]
            # photometric (image only)
            im = (im - 0.5) * contrast + 0.5 + brightness
            im = np.clip(im, 0.0, 1.0)
            im = np.power(im, gamma)
            if noise_sigma > 0:
                im = im + rng.normal(0.0, noise_sigma, im.shape).astype(np.float32)
            out_img[k, 0] = np.clip(im, 0.0, 1.0).astype(np.float32)
            out_wire[k, 0] = (wm > 0.5).astype(np.float32)
        return out_img, out_wire

    def _load_state_label(self, seq_name: str, path: Path) -> StateLabel:
        if seq_name not in self._state_label_cache:
            self._state_label_cache[seq_name] = StateLabel.load_npz(path)
        return self._state_label_cache[seq_name]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        w = self.windows[idx]
        T = self.T_window
        start = w.window_start
        H, W = self.img_size

        # 1) 加载 T 帧图像
        images = np.zeros((T, 1, H, W), dtype=np.float32)
        wire = np.zeros((T, 1, H, W), dtype=np.float32)
        for k in range(T):
            fname = w.frame_names[start + k]
            img_path = w.seq_dir_images / fname
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"could not read {img_path}")
            if (img.shape[0], img.shape[1]) != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            images[k, 0] = img.astype(np.float32) / 255.0

            if w.seq_dir_labels is not None:
                mask_path = w.seq_dir_labels / fname
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    if (mask.shape[0], mask.shape[1]) != (H, W):
                        mask = cv2.resize(
                            mask, (W, H), interpolation=cv2.INTER_NEAREST,
                        )
                    wire[k, 0] = (mask.astype(np.float32) / 255.0 > 0.5).astype(np.float32)

        # 1b) 真实数据增强（仅 train fold）。所有变换在 T 帧上一致施加，
        # 因此呼吸的相对运动不变 —— cos/sin/amplitude 标签依然有效。
        if self.augment:
            images, wire = self._augment_window(images, wire)

        # 2) 切窗 state label
        sl = self._load_state_label(w.seq_name, w.state_label_path)
        if sl.num_frames < start + T:
            raise RuntimeError(
                f"state label for {w.seq_name} has only {sl.num_frames} "
                f"frames; window asks [{start}, {start+T})"
            )
        sl_slice = slice(start, start + T)

        sample = {
            "images": torch.from_numpy(images),
            "wire_mask": torch.from_numpy(wire),
            "cos_phi": torch.from_numpy(sl.cos_phi[sl_slice].copy()),
            "sin_phi": torch.from_numpy(sl.sin_phi[sl_slice].copy()),
            "amplitude": torch.from_numpy(sl.amplitude[sl_slice].copy()),
            "valid_mask": torch.from_numpy(sl.valid_mask[sl_slice].copy()),
            "amp_scale": torch.tensor(sl.amp_scale, dtype=torch.float32),
            "reference_index": torch.tensor(
                max(0, min(T - 1, sl.reference_index - start)),
                dtype=torch.long,
            ),
            "seq_name": w.seq_name,
            "window_start": torch.tensor(start, dtype=torch.long),
        }

        # Optional: cardiac labels (dual-cycle training).
        # Phase-B for the cardiac band may produce a DIFFERENT length than
        # respiratory (band-pass + Hilbert have different effective lengths
        # under the higher passband). We always produce a T-length window
        # via safe overlap: positions outside the cardiac array's range get
        # zero with valid=0.
        if self.cardiac_labels_dir is not None:
            card_path = self.cardiac_labels_dir / f"{w.seq_name}.npz"
            if card_path.is_file():
                card = self._load_state_label(
                    w.seq_name + "__cardiac", card_path,
                )
                cos_c = np.zeros(T, dtype=np.float32)
                sin_c = np.zeros(T, dtype=np.float32)
                amp_c = np.zeros(T, dtype=np.float32)
                val_c = np.zeros(T, dtype=card.valid_mask.dtype)
                src_lo = max(0, min(start, card.num_frames))
                src_hi = max(0, min(start + T, card.num_frames))
                dst_lo = src_lo - start
                dst_hi = src_hi - start
                if src_hi > src_lo and dst_hi > dst_lo:
                    cos_c[dst_lo:dst_hi] = card.cos_phi[src_lo:src_hi]
                    sin_c[dst_lo:dst_hi] = card.sin_phi[src_lo:src_hi]
                    amp_c[dst_lo:dst_hi] = card.amplitude[src_lo:src_hi]
                    val_c[dst_lo:dst_hi] = card.valid_mask[src_lo:src_hi]
                sample["cardiac_cos_phi"] = torch.from_numpy(cos_c)
                sample["cardiac_sin_phi"] = torch.from_numpy(sin_c)
                sample["cardiac_amplitude"] = torch.from_numpy(amp_c)
                sample["cardiac_valid_mask"] = torch.from_numpy(val_c)
                sample["cardiac_amp_scale"] = torch.tensor(card.amp_scale,
                                                            dtype=torch.float32)
        return sample


# ===========================================================================
# ChronologicalSampler — paper §8 main ablation #2 (state training protocol)
# ===========================================================================


class ChronologicalSampler(Sampler[int]):
    """Yield window indices grouped by sequence and ordered by ``window_start``.

    Used by the ``chronological_carry`` training protocol (paper §8.1.2):

        * Within each sequence, windows are visited in strict chronological
          order so that hidden state from window ``k`` can be carried into
          window ``k+1`` of the same sequence.
        * Sequences themselves are shuffled epoch-to-epoch (controlled by
          ``shuffle_sequences``) to avoid memorizing per-batch ordering.
        * The trainer is responsible for resetting ``h0`` whenever the
          ``seq_name`` of the next batch differs from the previous one.
          A boundary detection helper :func:`is_sequence_boundary` is
          provided for that purpose.

    Args:
        dataset: a :class:`FluoroSequenceWindowDataset`. Its ``windows``
            list is read once at construction time.
        shuffle_sequences: shuffle the sequence order each epoch
            (deterministic w.r.t. ``seed`` + ``set_epoch``). Default True.
        seed: base RNG seed; combined with the per-epoch counter via
            :meth:`set_epoch` to produce reproducible shuffles.
        drop_last_partial_seq: if True, sequences that have fewer windows
            than ``min_windows_per_seq`` are dropped (so every retained
            sequence has at least one full carry chain). Default False.
        min_windows_per_seq: lower bound when ``drop_last_partial_seq``
            is on. Default 1 (no drop).

    Notes — DDP: this sampler is single-rank only. For DDP+chronological
        training, give each rank a *disjoint* set of sequences upstream
        and instantiate one ChronologicalSampler per rank.
    """

    def __init__(
        self,
        dataset: "FluoroSequenceWindowDataset",
        *,
        shuffle_sequences: bool = True,
        seed: int = 0,
        drop_last_partial_seq: bool = False,
        min_windows_per_seq: int = 1,
    ):
        self.dataset = dataset
        self.shuffle_sequences = bool(shuffle_sequences)
        self.seed = int(seed)
        self._epoch = 0

        # Group window indices by seq_name; sort each group by window_start
        groups: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for idx, w in enumerate(dataset.windows):
            groups[w.seq_name].append((int(w.window_start), idx))
        for k in list(groups.keys()):
            groups[k].sort()
            if drop_last_partial_seq and len(groups[k]) < min_windows_per_seq:
                del groups[k]
        self._groups: Dict[str, List[int]] = {
            k: [i for _, i in pairs] for k, pairs in groups.items()
        }
        self._seq_order_cache: List[str] = sorted(self._groups.keys())

    # ---------------------------------------------------------------------
    # DistributedSampler-like API (so trainers can call set_epoch uniformly)
    # ---------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    # ---------------------------------------------------------------------
    # Sampler protocol
    # ---------------------------------------------------------------------

    def __iter__(self) -> Iterator[int]:
        seqs = list(self._seq_order_cache)
        if self.shuffle_sequences:
            rng = np.random.default_rng(self.seed + self._epoch)
            rng.shuffle(seqs)
        for seq in seqs:
            for idx in self._groups[seq]:
                yield idx

    def __len__(self) -> int:
        return sum(len(v) for v in self._groups.values())

    # ---------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------

    @property
    def num_sequences(self) -> int:
        return len(self._groups)

    def windows_per_sequence(self) -> Dict[str, int]:
        """For logging / sanity checks."""
        return {k: len(v) for k, v in self._groups.items()}


def is_sequence_boundary(
    prev_seq_names: Optional[List[str]],
    curr_seq_names: List[str],
) -> List[bool]:
    """Per-batch-element flags: True where the sequence changed vs. last batch.

    Used by the ``chronological_carry`` training loop to decide which
    elements of the carried ``h0`` need to be reset to zero.

    Args:
        prev_seq_names: list of seq_name strings from the previous batch,
            or ``None`` if this is the first batch of the epoch.
        curr_seq_names: list of seq_name strings from the current batch.

    Returns:
        ``[B]`` boolean list. ``True`` means "this element starts a new
        sequence — reset its h0."
    """
    if prev_seq_names is None:
        return [True] * len(curr_seq_names)
    if len(prev_seq_names) != len(curr_seq_names):
        # 上一个 batch 长度不同（最后一批不满）→ 全部 reset 是最稳的兜底
        return [True] * len(curr_seq_names)
    return [a != b for a, b in zip(prev_seq_names, curr_seq_names)]
