"""Smart reference frame selection (PROJECT_CONSTRAINTS_19.6.md §2 + tech route §1).

Picks the reference frame ``r`` that anchors the cumulative displacement
formula ``U_t = sum_{i=r+1}^{t} Δu_i`` of R3. Default-to-first-frame is
explicitly forbidden by the constraints document, so this module implements
the three-tier priority:

    1. clinical roadmap / mask reference frame, when available;
    2. best frame within an initial warm-up window, scored as

           score_t = sharpness_t + anchor_quality_t - state_change_t

       (each term per-window min-max normalized so that the three quantities
       compose meaningfully despite very different raw scales);
    3. first frame as a final fallback.

The selector always reports the chosen index *and* the strategy so that
training / evaluation logs satisfy the "must report reference strategy"
clause of the constraints.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sharpness metric — classical Laplacian variance
# ---------------------------------------------------------------------------

# 3x3 Laplacian kernel：经典的二阶导数算子
# 卷积后每个像素 ≈ 周围 4 邻域的差异；图越锐利，结果方差越大
_LAPLACIAN_KERNEL = torch.tensor(
    [[0.0, 1.0, 0.0],
     [1.0, -4.0, 1.0],
     [0.0, 1.0, 0.0]]
).view(1, 1, 3, 3)


def laplacian_variance(images: torch.Tensor) -> torch.Tensor:
    """Per-frame Laplacian variance (a classical sharpness metric).

    A blurred frame has weak gradients, so its Laplacian response has small
    variance; a crisp frame has the opposite. Used here as the
    ``image_sharpness`` term in the warm-up score.

    Args:
        images: ``[T, H, W]`` or ``[T, 1, H, W]`` grayscale frames.

    Returns:
        ``[T]`` sharpness score per frame (higher = sharper).
    """
    if images.dim() == 3:
        x = images.unsqueeze(1)        # [T, 1, H, W]
    elif images.dim() == 4 and images.shape[1] == 1:
        x = images
    else:
        raise ValueError(
            f"images must be [T, H, W] or [T, 1, H, W], got {tuple(images.shape)}"
        )

    kernel = _LAPLACIAN_KERNEL.to(device=x.device, dtype=x.dtype)
    # padding=1 保持空间尺寸，边界用零填充
    lap = F.conv2d(x, kernel, padding=1)        # [T, 1, H, W]
    # 沿空间维求方差，得到每帧一个标量
    return lap.flatten(start_dim=1).var(dim=1)  # [T]


# ---------------------------------------------------------------------------
# State change magnitude — ‖Δs‖
# ---------------------------------------------------------------------------


def state_change_norm(state_seq: torch.Tensor) -> torch.Tensor:
    """Per-frame state change magnitude ``‖s_t - s_{t-1}‖`` with t=0 set to 0.

    Args:
        state_seq: ``[T, D]`` low-frequency physiological state per frame.

    Returns:
        ``[T]`` non-negative magnitudes; index 0 is 0 by convention.
    """
    if state_seq.dim() != 2:
        raise ValueError(
            f"state_seq must be [T, D], got {tuple(state_seq.shape)}"
        )
    out = torch.zeros(state_seq.shape[0],
                      device=state_seq.device, dtype=state_seq.dtype)
    if state_seq.shape[0] >= 2:
        diff = state_seq[1:] - state_seq[:-1]      # [T-1, D]
        out[1:] = diff.norm(dim=-1)                # L2 over D
    return out


# ---------------------------------------------------------------------------
# Min-max normalization with safe handling of constant signals
# ---------------------------------------------------------------------------


def _minmax01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-vector min-max normalize to ``[0, 1]``; constant input -> all zeros.

    Used so the three score components share a common scale before being
    summed. A constant signal contains no usable ranking information, so it
    is collapsed to zero (i.e. it contributes nothing to argmax).
    """
    x_min = x.min()
    x_max = x.max()
    rng = (x_max - x_min)
    if rng.item() < eps:
        return torch.zeros_like(x)
    return (x - x_min) / rng


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ReferenceSelection:
    """Outcome of a reference-frame selection call.

    Attributes:
        reference_index: chosen frame index ``r``.
        strategy: which tier produced the choice (always reported, per
            PROJECT_CONSTRAINTS_19.6.md §2).
        score_components: per-frame breakdown of the warm-up scoring (only
            populated when the warm-up tier is actually evaluated).
        warmup_window: ``(start, end)`` half-open range that was scored.
    """

    reference_index: int
    # "clinical" | "warmup_score" | "first_frame" | "random"
    strategy: str
    score_components: Dict[str, torch.Tensor] = field(default_factory=dict)
    warmup_window: Optional[tuple] = None


# ---------------------------------------------------------------------------
# Main selector
# ---------------------------------------------------------------------------


class ReferenceSelector:
    """Three-tier reference frame selector for a single fluoroscopy burst.

    Args:
        warmup_min: minimum number of warm-up frames to score.
        warmup_max: maximum number of warm-up frames to score. The actual
            window length is ``min(warmup_max, T)``, but never below
            ``warmup_min`` (or ``T`` if shorter).
        weight_sharpness / weight_anchor / weight_state_change: per-term
            multipliers applied **after** min-max normalization. Defaults
            give all three equal weight, matching the technical-route formula
            ``score = sharpness + anchor_quality − state_change``.
    """

    def __init__(
        self,
        warmup_min: int = 5,
        warmup_max: int = 10,
        weight_sharpness: float = 1.0,
        weight_anchor: float = 1.0,
        weight_state_change: float = 1.0,
        force_strategy: Optional[str] = None,
        random_seed: int = 0,
    ):
        if warmup_min < 1:
            raise ValueError("warmup_min must be >= 1")
        if warmup_max < warmup_min:
            raise ValueError("warmup_max must be >= warmup_min")
        if force_strategy is not None and force_strategy not in {
            "auto", "clinical", "warmup_score", "first_frame", "random",
        }:
            raise ValueError(
                f"force_strategy must be one of "
                "{None, 'auto', 'clinical', 'warmup_score', 'first_frame', "
                f"'random'}}, got '{force_strategy}'"
            )
        self.warmup_min = warmup_min
        self.warmup_max = warmup_max
        self.w_sharp = weight_sharpness
        self.w_anchor = weight_anchor
        self.w_state = weight_state_change
        # 'auto' 与 None 等价：保持 3-tier 优先级；其它值则强制走对应分支。
        self.force_strategy = (
            None if force_strategy in (None, "auto") else force_strategy
        )
        self.random_seed = int(random_seed)

    def select(
        self,
        images: Optional[torch.Tensor] = None,        # [T, H, W] or [T, 1, H, W]
        state_seq: Optional[torch.Tensor] = None,     # [T, D]
        anchor_quality: Optional[torch.Tensor] = None,  # [T]
        clinical_roadmap_index: Optional[int] = None,
        total_frames: Optional[int] = None,
    ) -> ReferenceSelection:
        """Run the three-tier selection.

        Args:
            images: warm-up frames for sharpness scoring. May contain only the
                first ``warmup_max`` frames or the full sequence — the selector
                only looks at the configured warm-up window.
            state_seq: low-frequency physiological state, used for the
                ``state_change`` term. Optional.
            anchor_quality: per-frame anchor quality from the state-label
                pipeline. Optional.
            clinical_roadmap_index: index of an externally provided clinical
                roadmap frame; if not None, this short-circuits to tier 1.
            total_frames: total number of frames in the burst. If omitted the
                selector infers it from ``images``/``state_seq``/
                ``anchor_quality``.

        Returns:
            :class:`ReferenceSelection`. ``strategy`` is always populated.
        """
        T = self._infer_T(images, state_seq, anchor_quality, total_frames)
        if T <= 0:
            raise ValueError("cannot select a reference from an empty burst")

        # ---- 决定 warm-up 窗口长度（即便强制 first_frame/random 也用得到）----
        N = max(self.warmup_min, min(self.warmup_max, T))
        N = min(N, T)
        window = (0, N)

        # ---- §6.1 ablation 强制策略短路 ----
        # force_strategy 优先级最高：用户已经知道自己在做哪个 ablation
        if self.force_strategy == "first_frame":
            return ReferenceSelection(
                reference_index=0,
                strategy="first_frame",
                warmup_window=window,
            )
        if self.force_strategy == "random":
            # 在 warm-up 窗口内均匀采样：作为 reference selection 的随机 baseline
            # （比 first_frame 噪声更大、比 warmup_score 决策更弱）
            g = torch.Generator().manual_seed(self.random_seed)
            r = int(torch.randint(0, N, (1,), generator=g).item())
            return ReferenceSelection(
                reference_index=r,
                strategy="random",
                warmup_window=window,
            )
        if self.force_strategy == "clinical":
            if clinical_roadmap_index is None:
                raise ValueError(
                    "force_strategy='clinical' requires "
                    "clinical_roadmap_index to be provided"
                )
            if not (0 <= clinical_roadmap_index < T):
                raise ValueError(
                    f"clinical_roadmap_index={clinical_roadmap_index} out of "
                    f"range [0, {T})"
                )
            return ReferenceSelection(
                reference_index=int(clinical_roadmap_index),
                strategy="clinical",
            )
        # force_strategy == "warmup_score" 走下面的常规打分分支

        # ---- 第 1 级：临床路图帧（最高优先级，仅在非强制模式下生效）----
        if (self.force_strategy is None and clinical_roadmap_index is not None):
            if not (0 <= clinical_roadmap_index < T):
                raise ValueError(
                    f"clinical_roadmap_index={clinical_roadmap_index} out of "
                    f"range [0, {T})"
                )
            return ReferenceSelection(
                reference_index=int(clinical_roadmap_index),
                strategy="clinical",
            )

        # ---- 没有任何打分依据 → 第 3 级 fallback ----
        if images is None and state_seq is None and anchor_quality is None:
            return ReferenceSelection(
                reference_index=0,
                strategy="first_frame",
                warmup_window=window,
            )

        # ---- 第 2 级：warm-up 评分 ----
        # 三项各自归一化到 [0, 1]，然后按权重加和
        sharp_n = self._sharpness_norm(images, N)              # [N]
        anchor_n = self._anchor_norm(anchor_quality, N)         # [N]
        change_n = self._state_change_norm(state_seq, N)        # [N]

        score = (
            self.w_sharp * sharp_n
            + self.w_anchor * anchor_n
            - self.w_state * change_n
        )

        r = int(torch.argmax(score).item())
        return ReferenceSelection(
            reference_index=r,
            strategy="warmup_score",
            score_components={
                "sharpness_norm": sharp_n.detach().cpu(),
                "anchor_norm": anchor_n.detach().cpu(),
                "state_change_norm": change_n.detach().cpu(),
                "score": score.detach().cpu(),
            },
            warmup_window=window,
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_T(
        images: Optional[torch.Tensor],
        state_seq: Optional[torch.Tensor],
        anchor_quality: Optional[torch.Tensor],
        total_frames: Optional[int],
    ) -> int:
        if total_frames is not None:
            return int(total_frames)
        for x in (images, state_seq, anchor_quality):
            if x is not None:
                return int(x.shape[0])
        return 0

    def _sharpness_norm(
        self, images: Optional[torch.Tensor], N: int
    ) -> torch.Tensor:
        if images is None:
            return torch.zeros(N)
        # 只在 warm-up 窗口内打分
        sharp = laplacian_variance(images[:N])
        return _minmax01(sharp)

    def _anchor_norm(
        self, anchor_quality: Optional[torch.Tensor], N: int
    ) -> torch.Tensor:
        if anchor_quality is None:
            return torch.zeros(N)
        # 锚点质量已经语义上是"越大越好"，只需归一化即可
        return _minmax01(anchor_quality[:N].float())

    def _state_change_norm(
        self, state_seq: Optional[torch.Tensor], N: int
    ) -> torch.Tensor:
        if state_seq is None:
            return torch.zeros(N)
        # 注意：state_change_norm 是"越小越好"，所以下游公式里用减号
        # 这里仍按 min-max 归一化到 [0, 1]，让减号在归一化空间生效
        change = state_change_norm(state_seq[:N])
        return _minmax01(change)


# 函数式快捷接口（懒得 new 一个 selector 时用）
def select_reference(
    images: Optional[torch.Tensor] = None,
    state_seq: Optional[torch.Tensor] = None,
    anchor_quality: Optional[torch.Tensor] = None,
    clinical_roadmap_index: Optional[int] = None,
    total_frames: Optional[int] = None,
    warmup_min: int = 5,
    warmup_max: int = 10,
) -> ReferenceSelection:
    """Functional shortcut for :class:`ReferenceSelector`."""
    sel = ReferenceSelector(warmup_min=warmup_min, warmup_max=warmup_max)
    return sel.select(
        images=images,
        state_seq=state_seq,
        anchor_quality=anchor_quality,
        clinical_roadmap_index=clinical_roadmap_index,
        total_frames=total_frames,
    )
