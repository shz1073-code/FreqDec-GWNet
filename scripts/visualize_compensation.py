#!/usr/bin/env python
"""Generate a 4-panel compensation comparison animation (paper figure).

For each frame ``t`` of a chosen sequence we render side-by-side:

    Panel 1   Original frame I_t (with overlaid wire mask outline).
    Panel 2   Reference frame I_r (the chosen reference, fixed).
    Panel 3   Compensated frame: warp(I_t, -U_t) (current → reference coords).
              Background should "freeze" at the reference position while
              the wire still moves (active push) — this is the paper claim
              made visible.
    Panel 4   Tip-residual plot: ‖lm[t] - lm[r] - U[t]‖ as a running line
              over time, with the current frame marked.

Outputs:
    ``{output_dir}/{seq}.gif`` — paper figure for §V.A
    ``{output_dir}/{seq}_static.png`` — single-frame static for §III
                                          (uses the frame at the middle
                                           of the burst by default)

Usage::

    python scripts/visualize_compensation.py \\
        --dataset dense_v1 --split train --seq cmu_seq_015 \\
        --ckpt experiments/checkpoints/stage3_main_v3_full_best.pth \\
        --state-core-type mamba \\
        --output-dir reports/figures/compensation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freqdec_gwnet.data import DataPaths, FluoroSequenceLoader        # noqa: E402
from freqdec_gwnet.models.freqdec_gwnet import FreqDecGWNet           # noqa: E402
from freqdec_gwnet.models.relative_motion_field import (              # noqa: E402
    apply_compensation,
)


# ---------------------------------------------------------------------------
# Frame composition
# ---------------------------------------------------------------------------


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Coerce [H, W] float in [0, 1] → uint8 [H, W] with safe clipping."""
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def _composite_with_mask_outline(
    image: np.ndarray, mask: Optional[np.ndarray],
    color_rgb_01: Tuple[float, float, float] = (1.0, 0.05, 0.05),
    extra_mask: Optional[np.ndarray] = None,
    extra_color_rgb_01: Tuple[float, float, float] = (0.05, 0.6, 1.0),
    fill_alpha: float = 0.7,
    dilate_px: int = 5,
) -> np.ndarray:
    """[H, W] greyscale in [0, 1] -> [H, W, 3] RGB float in [0, 1].

    Wires are 1–2 px wide; pure outline drawing produces near-invisible
    lines on a high-contrast X-ray background. Instead we render each
    mask as a **semi-transparent filled overlay** (alpha blend) after a
    small morphological dilation so the wire is clearly visible:

        mask    → red fill   (the wire location to highlight)
        extra_mask → blue fill (the reference target overlay used in
                                panel 3 to show what compensation aims at)

    Output remains float32 in [0, 1] for matplotlib imshow.
    """
    import cv2
    img = image.astype(np.float32, copy=False)
    rgb = np.repeat(img[..., None], 3, axis=2).copy()

    k = max(1, 2 * dilate_px + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def _blend(rgb_in, mask_in, color):
        if mask_in is None:
            return rgb_in
        m = (mask_in > 0).astype(np.uint8) * 255
        if dilate_px > 0:
            m = cv2.dilate(m, kernel, iterations=1)
        m_f = (m > 0)[..., None].astype(np.float32)         # [H, W, 1]
        color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
        # alpha-blend
        rgb_in = rgb_in * (1.0 - fill_alpha * m_f) + color_arr * (fill_alpha * m_f)
        return rgb_in

    # extra (reference target) first → red foreground sits on top
    rgb = _blend(rgb, extra_mask, extra_color_rgb_01)
    rgb = _blend(rgb, mask, color_rgb_01)
    return np.clip(rgb, 0.0, 1.0)


def _render_frame(
    t: int, T: int,
    raw_uint8: np.ndarray,                # [T, H, W]
    ref_uint8: np.ndarray,                # [H, W]
    comp_uint8: np.ndarray,               # [T, H, W]
    wire_masks: Optional[np.ndarray],     # [T, H, W] or None (original-coord masks)
    residual_curve: np.ndarray,           # [T]
    *,
    seq_name: str,
    fig_size: Tuple[int, int] = (16, 5),
    dpi: int = 150,
    wire_masks_compensated: Optional[np.ndarray] = None,  # [T, H, W]
    wire_mask_ref: Optional[np.ndarray] = None,            # [H, W]
    crop_to_wire_bbox: bool = True,
) -> np.ndarray:
    """Render one composite RGB frame at time t and return uint8 array.

    Panel layout:
        (1) Original I_t        — red outline = current wire position
        (2) Reference I_r        — blue outline = reference wire position
        (3) Compensated warp(I_t, -U_t)
              — red outline = wire mask warped by SAME -U_t (so it tracks
                the wire in the compensated image — was the bug we fixed)
              — blue outline = reference-frame wire position (paper claim:
                if compensation is perfect AND wire is static, the red
                outline should land on the blue one; the gap is the
                active push residual)
        (4) Tip residual after compensation
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=fig_size, dpi=dpi)
    fig.suptitle(
        f"{seq_name}    frame {t+1}/{T}",
        fontsize=11, y=0.99,
    )

    wm_t_orig = wire_masks[t] if wire_masks is not None else None
    wm_t_comp = (
        wire_masks_compensated[t]
        if wire_masks_compensated is not None else None
    )
    panel1 = _composite_with_mask_outline(
        raw_uint8[t].astype(np.float32) / 255.0, wm_t_orig,
    )
    panel2 = _composite_with_mask_outline(
        ref_uint8.astype(np.float32) / 255.0, wire_mask_ref,
        color_rgb_01=(0.0, 0.7, 1.0),                  # cyan-blue for ref
    )
    panel3 = _composite_with_mask_outline(
        comp_uint8[t].astype(np.float32) / 255.0,
        mask=wm_t_comp,                                 # red: warped current wire
        extra_mask=wire_mask_ref,                       # blue: reference target
    )

    # Tight bounding box around the union of wire regions across all 3 panels
    # so the wire and overlays fill most of each tile.
    if crop_to_wire_bbox:
        union = np.zeros(raw_uint8.shape[1:], dtype=np.uint8)
        for m in (wm_t_orig, wm_t_comp, wire_mask_ref):
            if m is not None:
                union |= (m > 0).astype(np.uint8)
        ys, xs = np.where(union > 0)
        if ys.size:
            pad = 40
            H_full, W_full = union.shape
            y0 = max(0, ys.min() - pad)
            y1 = min(H_full, ys.max() + pad)
            x0 = max(0, xs.min() - pad)
            x1 = min(W_full, xs.max() + pad)
            # 长宽比保持 1:1 让三联面板尺寸一致
            side = max(y1 - y0, x1 - x0)
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            y0 = max(0, cy - side // 2)
            y1 = min(H_full, y0 + side)
            x0 = max(0, cx - side // 2)
            x1 = min(W_full, x0 + side)
            panel1 = panel1[y0:y1, x0:x1]
            panel2 = panel2[y0:y1, x0:x1]
            panel3 = panel3[y0:y1, x0:x1]

    for ax, img, title in zip(
        axes[:3],
        [panel1, panel2, panel3],
        ["(1) Original I_t  [red=current wire]",
         "(2) Reference I_r  [blue=ref wire]",
         "(3) Compensated  [red=warped wire, blue=ref target]"],
    ):
        # Float-in-[0,1] images can lose saturated overlay colors under
        # matplotlib's normalization pipeline; uint8 preserves them.
        img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        ax.imshow(img_u8, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axes[3]
    ts = np.arange(T)
    ax.plot(ts, residual_curve, color="C0", lw=0.9, label="‖lm−lm_r−U‖")
    ax.axvline(t, color="C3", lw=1.1, ls="--", label=f"t={t}")
    ax.set_title("(4) Tip residual after compensation", fontsize=9)
    ax.set_xlabel("frame index", fontsize=8)
    ax.set_ylabel("px", fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.canvas.draw()
    # Convert matplotlib canvas to RGB numpy (matplotlib >= 3.8 uses
    # buffer_rgba; old API tostring_rgb removed).
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return buf[..., :3]


# ---------------------------------------------------------------------------
# Main rendering pipeline
# ---------------------------------------------------------------------------


def render_compensation_animation(
    seq, model, *, device: str, output_dir: Path,
    fps: int = 6, max_frames: Optional[int] = None,
    mode: str = "current_to_reference",
    fmt: str = "mp4",
    dpi: int = 150,
) -> dict:
    """Produce the GIF + static PNG for one sequence.

    Returns a dict with output paths and basic stats.
    """
    import imageio.v3 as iio

    T = seq.num_frames
    if max_frames is not None:
        T = min(T, int(max_frames))
    raw = seq.frames[:T]                                        # [T, 1, H, W]
    images_uint8 = (raw[:, 0].cpu().numpy() * 255.0).astype(np.uint8)

    # ---- model forward ----
    model.eval()
    with torch.no_grad():
        out = model(
            raw.unsqueeze(0).to(device),
            mode="stage3_joint", reference_index=0,
        )
    U = out["U"][0].cpu()                                        # [T, 2]

    # ---- compensated frames ----
    # apply_compensation warps the image stack by -U (current → reference).
    raw_for_warp = raw.to(device)                                # [T, 1, H, W]
    comp = apply_compensation(
        raw_for_warp.unsqueeze(0),                               # [1, T, 1, H, W]
        U.unsqueeze(0).to(device),
        mode=mode,
    )[0].cpu()                                                   # [T, 1, H, W]
    comp_uint8 = (comp[:, 0].numpy() * 255.0).clip(0, 255).astype(np.uint8)

    # Reference frame (chosen as frame 0)
    ref_uint8 = images_uint8[0]

    # Wire masks (if seg labels are available).
    # IMPORTANT: panel 3 shows the compensated image, where the wire has been
    # translated by the same -U_t as the rest of the image. So the mask
    # outline drawn on panel 3 must also be warped by the same -U_t —
    # otherwise the outline appears to drift away from the actual wire
    # location in the compensated frame (the bug a previous render had).
    wire_masks = None
    wire_masks_compensated = None
    wire_mask_ref = None
    if seq.masks is not None:
        masks = seq.masks[:T, 0].cpu().numpy()
        wire_masks = (masks > 0.5).astype(np.uint8)
        # Warp the wire mask stack with the same compensation as the image
        masks_for_warp = seq.masks[:T].to(device).float()         # [T, 1, H, W]
        masks_comp = apply_compensation(
            masks_for_warp.unsqueeze(0),                          # [1, T, 1, H, W]
            U.unsqueeze(0).to(device),
            mode=mode,
        )[0].cpu()                                                # [T, 1, H, W]
        wire_masks_compensated = (
            masks_comp[:, 0].numpy() > 0.5
        ).astype(np.uint8)
        # Reference wire mask (the static target the compensated wire should
        # ideally land on if compensation were perfect AND the wire didn't
        # move actively)
        wire_mask_ref = wire_masks[0]

    # Tip residual curve (uses landmarks if present)
    residual_curve = np.full(T, np.nan, dtype=np.float32)
    if seq.tip_xy is not None and seq.tip_present is not None:
        lm = seq.tip_xy[:T].cpu().numpy()
        present = seq.tip_present[:T].cpu().numpy().astype(bool)
        if present.any() and present[0]:
            lm_r = lm[0]
            U_np = U.cpu().numpy()
            for t in range(T):
                if not present[t]:
                    continue
                residual = lm[t] - lm_r - U_np[t]
                residual_curve[t] = float(np.linalg.norm(residual))

    # ---- render frame buffer (higher DPI for video clarity) ----
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in range(T):
        frame = _render_frame(
            t, T, images_uint8, ref_uint8, comp_uint8,
            wire_masks, residual_curve,
            seq_name=f"{seq.dataset}/{seq.split}/{seq.seq_name}",
            dpi=dpi,
            wire_masks_compensated=wire_masks_compensated,
            wire_mask_ref=wire_mask_ref,
        )
        frames.append(frame)

    # ---- export video / animation ----
    written: dict = {}
    if fmt in ("mp4", "both"):
        mp4_path = output_dir / f"{seq.seq_name}.mp4"
        # H.264 在 16-pixel 整数倍上编码最稳；自动 pad
        h, w, _ = frames[0].shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h or pad_w:
            frames_pad = []
            for fr in frames:
                p = np.pad(
                    fr, ((0, pad_h), (0, pad_w), (0, 0)),
                    constant_values=255,
                )
                frames_pad.append(p)
        else:
            frames_pad = frames
        iio.imwrite(
            mp4_path, frames_pad,
            plugin="pyav", fps=fps, codec="h264",
            # 高质量编码：crf 越低越清晰（推荐 18-23），pix_fmt yuv420p 兼容性好
            out_pixel_format="yuv420p",
        )
        written["mp4"] = mp4_path
    if fmt in ("gif", "both"):
        gif_path = output_dir / f"{seq.seq_name}.gif"
        iio.imwrite(gif_path, frames, plugin="pillow", fps=fps, loop=0)
        written["gif"] = gif_path

    # ---- static figure: pick a frame near the middle, save large PNG ----
    static_path = output_dir / f"{seq.seq_name}_static.png"
    mid = T // 2
    static_frame = _render_frame(
        mid, T, images_uint8, ref_uint8, comp_uint8,
        wire_masks, residual_curve,
        seq_name=f"{seq.dataset}/{seq.split}/{seq.seq_name}",
        fig_size=(14, 5),
        dpi=200,                              # 静态图更高清
        wire_masks_compensated=wire_masks_compensated,
        wire_mask_ref=wire_mask_ref,
    )
    iio.imwrite(static_path, static_frame, plugin="pillow")
    written["static"] = static_path

    # ---- summary stats for the user ----
    U_norm = U.norm(dim=-1).cpu().numpy()
    mean_res = (
        float(np.nanmean(residual_curve))
        if np.isfinite(residual_curve).any() else float("nan")
    )
    return {
        **{f"{k}_path": v for k, v in written.items()},
        "U_norm_max": float(U_norm.max()),
        "U_norm_mean": float(U_norm.mean()),
        "mean_residual": mean_res,
        "n_frames": T,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Render compensation comparison GIF + static figure",
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", required=True,
                   choices=("train", "val", "test"))
    p.add_argument("--seq", nargs="+", required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "reports" / "figures" / "compensation")
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--state-core-type",
                   choices=("gru", "mamba"), default="mamba")
    p.add_argument("--motion-state-dim", type=int, default=3)
    p.add_argument("--motion-input-mode",
                   choices=("delta", "with_prev"), default="delta")
    p.add_argument("--motion-field-type",
                   choices=("relative", "absolute"), default="relative")
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--mode", default="current_to_reference",
                   choices=("current_to_reference", "roadmap_to_current"))
    p.add_argument("--format", default="mp4",
                   choices=("mp4", "gif", "both"),
                   help="output animation format. mp4 is recommended "
                        "(H.264, much sharper than GIF's 256-color palette)")
    p.add_argument("--dpi", type=int, default=150,
                   help="matplotlib DPI for video frames; 200 = print-ready")
    p.add_argument("--device", default="cuda",
                   choices=("cuda", "cpu"))

    args = p.parse_args(list(argv) if argv is not None else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = (
        DataPaths.from_yaml(args.config) if args.config
        else DataPaths.from_default_config()
    )
    loader = FluoroSequenceLoader(paths)

    device = (
        args.device if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    print(f"[viz_compensation] device={device} ckpt={args.ckpt}")
    model = FreqDecGWNet(
        width_mult=args.width_mult,
        state_core_type=args.state_core_type,
        motion_state_dim=args.motion_state_dim,
        motion_input_mode=args.motion_input_mode,
        motion_field_type=args.motion_field_type,
    ).to(device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    sd = {(k[len("module."):] if k.startswith("module.") else k): v
          for k, v in sd.items()}
    miss, unex = model.load_state_dict(sd, strict=False)
    print(f"  loaded missing={len(miss)} unexpected={len(unex)}")

    for sname in args.seq:
        seq = loader.load_sequence(args.dataset, args.split, sname)
        result = render_compensation_animation(
            seq, model, device=device,
            output_dir=args.output_dir,
            fps=args.fps,
            max_frames=args.max_frames,
            mode=args.mode,
            fmt=args.format,
            dpi=args.dpi,
        )
        lines = [
            f"  {sname:15s}  T={result['n_frames']}  "
            f"U_max={result['U_norm_max']:.1f}  "
            f"U_mean={result['U_norm_mean']:.1f}  "
            f"residual_mean={result['mean_residual']:.2f}"
        ]
        for k in ("mp4", "gif", "static"):
            if f"{k}_path" in result:
                lines.append(f"    {k}: {result[f'{k}_path']}")
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
