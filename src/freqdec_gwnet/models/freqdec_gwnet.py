"""End-to-end glue model for FreqDec-GWNet 19.6.

Composes the three protected red-line modules into a single ``nn.Module``:

    R1 (frequency-guided guidewire observer)  -> :class:`FAST_LiteNet`
    R2 (full-sequence physiological state)     -> :class:`GlobalStateBranch`
    R3 (sequence-relative motion increments)   -> :class:`RelativeMotionField`

Four forward modes pin which sub-modules are exercised on each call:

    * ``stage1_seg``    — frame-by-frame R1 forward with ``feature_bank``
                          carry-over. Identical to FAST_LiteNet's training
                          interface so existing stage-1 scripts keep working.
    * ``stage2_state``  — window forward; encoder + state branch only.
                          Used to train R2 with ``StateLoss`` while the
                          segmentation head is frozen.
    * ``stage3_joint``  — window forward; encoder + state branch +
                          motion field. Used to train R3 with ``MotionLoss``
                          on top of the converged state branch.
    * ``inference``     — window forward; produces seg + state + motion in
                          a single pass for evaluation / deployment.

The state-dim contract is pinned here: :class:`GlobalStateBranch` emits a
4-D state ``(cos_phi, sin_phi, amplitude, confidence)`` but the motion
field intentionally consumes only the first three. The slice happens
inside :meth:`_forward_stage3` so downstream callers cannot accidentally
widen the input. ``tests/test_state_branch.py`` and the new
``tests/test_freqdec_gwnet.py`` together protect this invariant.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .absolute_motion_field import AbsoluteMotionField
from .fast_litenet import FAST_LiteNet
from .global_state_branch import GlobalStateBranch
from .relative_motion_field import RelativeMotionField


# ---------------------------------------------------------------------------
# Argument groups for the three sub-modules
# ---------------------------------------------------------------------------


# 这些参数会原样转发到 FAST_LiteNet __init__（保持 R1 旧接口不变）
_R1_KWARGS = {
    "in_channels", "num_classes", "width_mult",
    "multi_task", "edge_aux", "use_refine_head", "refine_kernel_size",
    "freq_mode", "local_window_size", "local_window_sizes",
    "enable_conf_gate", "gate_fuse_threshold",
    "gate_freeze_threshold", "gate_reinit_threshold", "bank_momentum",
    "freq_gate_type", "freq_gate_reduction",
}

_R2_KWARGS = {
    "state_proj_dim", "state_hidden_dim", "state_lp_kernel",
    "state_core_type", "state_d_state",
    "state_detach_encoder", "state_learnable_lp",
}

_R3_KWARGS = {
    "motion_state_dim", "motion_hidden_dim", "motion_output_dim",
    "motion_field_type",
}


_VALID_MODES = {"stage1_seg", "stage2_state", "stage3_joint", "inference"}


# ---------------------------------------------------------------------------
# Main glue class
# ---------------------------------------------------------------------------


class FreqDecGWNet(nn.Module):
    """Compose R1 + R2 + R3 with stage-aware forward modes."""

    def __init__(
        self,
        # ---- R1: forwarded to FAST_LiteNet ----
        in_channels: int = 1,
        num_classes: int = 1,
        width_mult: float = 1.0,
        multi_task: bool = False,
        edge_aux: bool = False,
        use_refine_head: bool = False,
        refine_kernel_size: int = 7,
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
        # ---- R2: forwarded to GlobalStateBranch ----
        state_proj_dim: int = 128,
        state_hidden_dim: int = 64,
        state_lp_kernel: int = 5,
        state_core_type: str = "gru",
        state_d_state: int = 16,
        state_detach_encoder: bool = True,
        state_learnable_lp: bool = True,
        # ---- R3: forwarded to RelativeMotionField (or its baseline) ----
        motion_state_dim: int = 3,
        motion_hidden_dim: int = 64,
        motion_output_dim: int = 2,
        motion_field_type: str = "relative",   # "relative" (default) | "absolute"
        motion_input_mode: str = "delta",      # "delta" (6D/4D) | "with_prev" (9D)
    ):
        super().__init__()

        # 1) R1 — frequency-guided segmentation backbone (FAST-LiteNet)
        self.r1 = FAST_LiteNet(
            in_channels=in_channels,
            num_classes=num_classes,
            width_mult=width_mult,
            multi_task=multi_task,
            edge_aux=edge_aux,
            use_refine_head=use_refine_head,
            refine_kernel_size=refine_kernel_size,
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
        )

        # GhostEncoder.out_channels = (c_1/2, c_1/4, c_1/8, c_1/16, c_1/32)
        c_18 = int(self.r1.encoder.out_channels[2])

        # 2) R2 — global state branch on 1/8 encoder features
        self.state_branch = GlobalStateBranch(
            in_channels=c_18,
            proj_dim=state_proj_dim,
            hidden_dim=state_hidden_dim,
            lp_kernel=state_lp_kernel,
            state_core_type=state_core_type,
            d_state=state_d_state,
            detach_encoder=state_detach_encoder,
            learnable_lp=state_learnable_lp,
        )

        # 3) R3 — motion field (default = sequence-relative; "absolute"
        # is the §6.1 ablation #1 baseline that PROJECT_CONSTRAINTS_19.6 §1
        # forbids as the main method but explicitly allows as ablation.)
        self.motion_field_type = motion_field_type
        if motion_field_type == "relative":
            self.motion_field = RelativeMotionField(
                state_dim_for_motion=motion_state_dim,
                hidden_dim=motion_hidden_dim,
                output_dim=motion_output_dim,
                input_mode=motion_input_mode,
            )
        elif motion_field_type == "absolute":
            # Absolute baseline 不消费 Δs_t / s_{t-1}，input_mode 在此无意义
            self.motion_field = AbsoluteMotionField(
                state_dim_for_motion=motion_state_dim,
                hidden_dim=motion_hidden_dim,
                output_dim=motion_output_dim,
            )
        else:
            raise ValueError(
                f"unknown motion_field_type='{motion_field_type}'; "
                "expected 'relative' (default) or 'absolute'"
            )

        # 缓存：state branch 期望的 motion state_dim 对得上 [cos, sin, amp]
        # 当 GlobalStateBranch 未来扩展输出维度时，单测会立即报警。
        self._state_branch_output_dim = 4  # [cos, sin, amp, conf]
        self._motion_state_dim = int(motion_state_dim)
        if self._motion_state_dim > self._state_branch_output_dim - 1:
            raise ValueError(
                f"motion_state_dim={self._motion_state_dim} but state branch "
                f"only emits {self._state_branch_output_dim - 1} usable "
                "channels (last channel is confidence)"
            )

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        mode: str = "stage3_joint",
        **mode_kwargs: Any,
    ) -> Dict[str, Any]:
        """Dispatch to the per-stage forward method.

        Args:
            images: shape depends on ``mode``:
                * ``stage1_seg``: ``[B, 1, H, W]`` single frame.
                * other modes: ``[B, T, 1, H, W]`` window of T frames.
            mode: one of ``stage1_seg``, ``stage2_state``, ``stage3_joint``,
                ``inference``.
            **mode_kwargs: forwarded to the chosen ``_forward_*`` method.

        Returns:
            Dict whose keys depend on the mode (see each method docstring).
        """
        if mode not in _VALID_MODES:
            raise ValueError(
                f"unknown mode '{mode}'; expected one of {sorted(_VALID_MODES)}"
            )
        if mode == "stage1_seg":
            return self._forward_stage1(images, **mode_kwargs)
        if mode == "stage2_state":
            return self._forward_stage2(images, **mode_kwargs)
        if mode == "stage3_joint":
            return self._forward_stage3(images, **mode_kwargs)
        return self._forward_inference(images, **mode_kwargs)

    # ------------------------------------------------------------------
    # Stage-1: pass-through to FAST_LiteNet (frame-by-frame, feature_bank)
    # ------------------------------------------------------------------

    def _forward_stage1(
        self,
        x_curr: torch.Tensor,
        feature_bank: Optional[torch.Tensor] = None,
        reset_flag: bool = False,
    ) -> Dict[str, Any]:
        """Frame-by-frame R1 forward; matches FAST_LiteNet's training API.

        Returns:
            ``{"seg": [B, num_classes, H, W], "feature_bank": [B, C, 16, 16]}``,
            optionally including ``centerline``/``tip``/``edge`` heads when
            the underlying FAST-LiteNet was built with them.
        """
        out, next_bank = self.r1(
            x_curr, feature_bank=feature_bank, reset_flag=reset_flag,
        )
        if isinstance(out, dict):
            return {"feature_bank": next_bank, **out}
        return {"seg": out, "feature_bank": next_bank}

    # ------------------------------------------------------------------
    # Stage-2: window forward; encoder + state branch only
    # ------------------------------------------------------------------

    def _forward_stage2(
        self,
        images: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encoder + GlobalStateBranch forward over a T-frame window.

        Args:
            images: ``[B, T, 1, H, W]``.
            h0: optional initial hidden state for the state core.

        Returns:
            ``{"state": [B, T, 4], "feat_18_seq": [B, T, C_18, H/8, W/8]}``.
        """
        if images.dim() != 5:
            raise ValueError(
                f"stage2 expects [B, T, 1, H, W]; got {tuple(images.shape)}"
            )
        B, T, C_in, H, W = images.shape

        # 展平 T 维度，一次过完 encoder（节省 Python 循环开销）
        flat = images.reshape(B * T, C_in, H, W)
        feats = self.r1.encoder(flat)              # 5-tuple
        feat_18 = feats[2]                         # [B*T, C_18, H/8, W/8]
        feat_18_seq = feat_18.reshape(B, T, *feat_18.shape[1:])

        state = self.state_branch.forward_window(feat_18_seq, h0=h0)
        return {
            "state": state,                        # [B, T, 4]
            "feat_18_seq": feat_18_seq,
        }

    # ------------------------------------------------------------------
    # Stage-3: encoder + state branch + motion field (with R3 contract)
    # ------------------------------------------------------------------

    def _forward_stage3(
        self,
        images: torch.Tensor,
        reference_index=None,            # int | LongTensor[B] | None
        return_seg: bool = False,
        h0: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Stage 2 + R3 motion field on top.

        The R2 → R3 state-dim contract is realised here: the 4-D state from
        :class:`GlobalStateBranch` is sliced to its first ``motion_state_dim``
        channels (default 3, dropping ``confidence``) before feeding the
        motion field. ``tests/test_freqdec_gwnet.py`` pins this slice.

        Args:
            images: ``[B, T, 1, H, W]``.
            reference_index: forwarded to :meth:`RelativeMotionField.forward`
                (``None`` → per-sample default = 0).
            return_seg: also run the segmentation pathway and include
                ``seg`` in the output.
            h0: initial state-core hidden state.

        Returns:
            Dict with ``state``, ``feat_18_seq``, ``delta_s``, ``delta_u``,
            ``U`` always present; ``seg`` present iff ``return_seg``.
        """
        out = self._forward_stage2(images, h0=h0)
        state = out["state"]                       # [B, T, 4]

        # State-dim adapter (pinned by tests)
        s_for_motion = state[..., : self._motion_state_dim]
        # RelativeMotionField 默认 reference_index=0；None 在这里就显式回退到 0，
        # 避免把 None 透传到 cumulative_displacement 引发类型错误。
        if reference_index is None:
            motion = self.motion_field(s_for_motion)
        else:
            motion = self.motion_field(
                s_for_motion, reference_index=reference_index,
            )

        result: Dict[str, torch.Tensor] = {
            "state": state,
            "feat_18_seq": out["feat_18_seq"],
            "delta_s": motion["delta_s"],
            "delta_u": motion["delta_u"],
            "U": motion["U"],
        }
        if return_seg:
            result["seg"] = self._forward_seg_window(images)
        return result

    # ------------------------------------------------------------------
    # Inference: stage-3 + segmentation per frame
    # ------------------------------------------------------------------

    def _forward_inference(
        self,
        images: torch.Tensor,
        reference_index=None,
    ) -> Dict[str, torch.Tensor]:
        """Full pipeline forward (R1 + R2 + R3) for evaluation."""
        return self._forward_stage3(
            images,
            reference_index=reference_index,
            return_seg=True,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _forward_seg_window(
        self, images: torch.Tensor,
    ) -> torch.Tensor:
        """Frame-by-frame segmentation forward to preserve feature_bank.

        ``feature_bank`` is reset at t=0 (matching FAST-LiteNet's training
        protocol) and carried through the rest of the window.

        Returns:
            ``[B, T, num_classes, H, W]`` segmentation logits.
        """
        B, T = images.shape[:2]
        feature_bank = None
        seg_list = []
        for t in range(T):
            out, feature_bank = self.r1(
                images[:, t], feature_bank=feature_bank, reset_flag=(t == 0),
            )
            seg_t = out["seg"] if isinstance(out, dict) else out
            seg_list.append(seg_t)
        return torch.stack(seg_list, dim=1)        # [B, T, num_classes, H, W]

    # ------------------------------------------------------------------
    # Stage-1 checkpoint partial-loading
    # ------------------------------------------------------------------

    def load_stage1_checkpoint(
        self,
        path,
        strict: bool = False,
    ) -> Tuple[list, list]:
        """Partial-load a FAST-LiteNet stage-1 checkpoint into ``self.r1``.

        Stage-1 checkpoints contain only FAST-LiteNet weights. R2 / R3
        sub-modules stay at their initial state (zero-init for
        ``RelativeMotionField`` and Glorot for ``GlobalStateBranch``).

        Args:
            path: filesystem path to a state-dict or training checkpoint
                (the latter is unwrapped via the ``model_state_dict`` key).
            strict: whether to require all R1 keys to match. Default False.

        Returns:
            ``(missing_keys, unexpected_keys)`` — same convention as
            ``nn.Module.load_state_dict``.
        """
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        # 兼容 DataParallel/DDP 保存的 module.* 前缀
        state_dict = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }

        return self.r1.load_state_dict(state_dict, strict=strict)
