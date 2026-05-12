"""Phase B.5 — produce wire masks via stage-1 inference for state-label gen.

Phase B.4 generated state labels with ``wire_masks=None`` (treats the whole
frame as background) because stage 1 was not yet trained. Now that stage 1
is trained, the same labels can be regenerated with real wire masks
injected into B.1 background-motion estimation, which:

    1. excludes the active wire push from the bg_motion signal,
    2. tightens the multi-anchor consensus → higher anchor_quality,
    3. raises the valid_mask ratio (typically 71% → 90%+).

Usage:
    from freqdec_gwnet.data.state_labels.wire_mask_inference import (
        WireMaskInferencer,
    )
    inferer = WireMaskInferencer(stage1_ckpt_path, device="cuda")
    masks = inferer.infer_burst(images_uint8)   # [T, H, W] -> [T, H, W] uint8
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class WireMaskInferencer:
    """Wrap a stage-1 FAST-LiteNet ckpt and run frame-by-frame inference.

    Frame-by-frame is the FAST-LiteNet contract (stage-1 trained that way),
    so we honor it here even though batched inference would be faster — the
    feature_bank carry-over is what gives stage-1 its temporal smoothing.

    Args:
        ckpt_path: path to a stage-1 ``.pth`` (resume-format or weights-only).
        device: ``"cuda"`` / ``"cpu"`` / explicit ``torch.device``.
        threshold: sigmoid threshold for binarizing the seg logits.
        width_mult: must match the ckpt's width_mult (1.0 / 1.5 / 2.0).
        freq_mode / local_window_size / etc.: must match training config —
            these are passed straight through to FAST_LiteNet.__init__.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        *,
        threshold: float = 0.5,
        width_mult: float = 1.0,
        freq_mode: str = "global",
        local_window_size: int = 4,
        local_window_sizes: Tuple[int, ...] = (4, 8),
        enable_conf_gate: bool = False,
        gate_fuse_threshold: float = 0.5,
        gate_freeze_threshold: float = 0.45,
        gate_reinit_threshold: float = 0.20,
        bank_momentum: float = 0.7,
        freq_gate_type: str = "none",
        freq_gate_reduction: int = 4,
        multi_task: bool = False,
        edge_aux: bool = False,
        use_refine_head: bool = False,
        refine_kernel_size: int = 7,
    ):
        # Lazy import — avoids dragging the whole models tree into the
        # state_labels package eagerly.
        from ...models.fast_litenet import FAST_LiteNet
        self.device = (
            torch.device(device) if isinstance(device, str) else device
        )
        self.threshold = float(threshold)

        self.model = FAST_LiteNet(
            in_channels=1, num_classes=1, width_mult=width_mult,
            freq_mode=freq_mode,
            local_window_size=local_window_size,
            local_window_sizes=local_window_sizes,
            enable_conf_gate=enable_conf_gate,
            gate_fuse_threshold=gate_fuse_threshold,
            gate_freeze_threshold=gate_freeze_threshold,
            gate_reinit_threshold=gate_reinit_threshold,
            bank_momentum=bank_momentum,
            freq_gate_type=freq_gate_type,
            freq_gate_reduction=freq_gate_reduction,
            multi_task=multi_task,
            edge_aux=edge_aux,
            use_refine_head=use_refine_head,
            refine_kernel_size=refine_kernel_size,
        ).to(self.device)
        self._load_ckpt(Path(ckpt_path))
        self.model.eval()

    # ------------------------------------------------------------------

    def _load_ckpt(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"stage-1 ckpt not found: {path}")
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt
        # 兼容 DataParallel/DDP 保存的 module.* 前缀
        sd = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in sd.items()
        }
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(
                f"[WireMaskInferencer] partial load: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer_burst(
        self,
        images: np.ndarray,
        *,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Run frame-by-frame inference with feature_bank carry-over.

        Args:
            images: ``[T, H, W]`` greyscale frames (uint8 or float in [0,1]).
            target_size: optional (H, W) to resize to before inference;
                masks are then re-projected back to the input resolution.

        Returns:
            ``[T, H, W]`` uint8 binary wire masks (1 inside wire).
        """
        if images.ndim != 3:
            raise ValueError(
                f"images must be [T, H, W]; got {images.shape}"
            )
        T, H_in, W_in = images.shape
        if images.dtype == np.uint8:
            imgs_f = images.astype(np.float32) / 255.0
        else:
            imgs_f = images.astype(np.float32, copy=False)

        # Optional resize before forward
        if target_size is not None and target_size != (H_in, W_in):
            H_t, W_t = target_size
            t = torch.from_numpy(imgs_f).unsqueeze(1)        # [T, 1, H, W]
            t = F.interpolate(t, size=(H_t, W_t), mode="bilinear",
                              align_corners=False)
            imgs_t_full = t.to(self.device)
        else:
            H_t, W_t = H_in, W_in
            imgs_t_full = torch.from_numpy(imgs_f).unsqueeze(1).to(self.device)

        feature_bank = None
        masks = np.zeros((T, H_in, W_in), dtype=np.uint8)
        for t in range(T):
            x = imgs_t_full[t:t+1]                            # [1, 1, H_t, W_t]
            out, feature_bank = self.model(
                x, feature_bank=feature_bank, reset_flag=(t == 0),
            )
            logits = out["seg"] if isinstance(out, dict) else out
            prob = torch.sigmoid(logits)
            pred = (prob > self.threshold).float()
            # Resize back to input resolution
            if (H_t, W_t) != (H_in, W_in):
                pred = F.interpolate(
                    pred, size=(H_in, W_in), mode="nearest",
                )
            masks[t] = pred[0, 0].cpu().numpy().astype(np.uint8)
        return masks
