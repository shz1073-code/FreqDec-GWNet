#!/usr/bin/env python
"""Paper-grade visualization (paper §V figure 5).

Replaces ``visualize_compensation.py``'s misleading red-vs-blue wire
overlay. Per GPT's review, the paper's compensation claim is **about
background/anatomy alignment, not about putting the wire back to its
reference shape** (the wire actively moves under operator control).

Panel layout (4 panels stacked left-to-right):

    (1) Reference frame I_r with background patch anchors drawn as
        small yellow crosses (only on background, far from wire mask).
    (2) Current frame I_t with the same anchors *tracked* (Lucas-Kanade)
        — visualizes raw respiratory drift of the anchors.
    (3) Compensated frame warp(I_t, -U_t) with anchors *re-projected*
        by subtracting U_t. Yellow points overlaid on the reference
        anchor positions — anchor crosses should sit close to where
        they started in (1) if compensation works.
    (4) Difference image |I_r − warp(I_t, -U_t)| with the wire mask
        masked OUT (white background = perfect alignment, residual
        intensity = remaining respiratory drift).

Also outputs a sidebar with Metric A + Metric B + reset_every for the
sequence, so each MP4 is self-describing.

Usage:
    python scripts/visualize_paper_v6.py \\
        --dataset dense_v1 --split test --seq cmu_seq_094 \\
        --ckpt experiments/checkpoints/stage3_rel_zm_v5_best.pth \\
        --reset-every 16 \\
        --output-dir reports/paper_v6_viz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freqdec_gwnet.data import DataPaths, FluoroSequenceLoader         # noqa: E402
from freqdec_gwnet.models.freqdec_gwnet import FreqDecGWNet            # noqa: E402
from freqdec_gwnet.models.relative_motion_field import (               # noqa: E402
    apply_compensation,
)
from freqdec_gwnet.utils.background_patch_tracking import (            # noqa: E402
    detect_background_features, track_features_across_burst,
)
from freqdec_gwnet.utils.reference_selector import ReferenceSelector   # noqa: E402


def _apply_reset(U: torch.Tensor, reset: int) -> torch.Tensor:
    if reset is None or reset <= 0:
        return U
    T = U.shape[0]
    delta_u = torch.zeros_like(U)
    delta_u[1:] = U[1:] - U[:-1]
    out = torch.zeros_like(U)
    for s in range(0, T, reset):
        e = min(T, s + reset)
        chunk = delta_u[s:e].clone()
        chunk[0] = 0.0
        out[s:e] = torch.cumsum(chunk, dim=0)
    return out


def _draw_anchors(
    image_uint8: np.ndarray, anchors_xy: np.ndarray,
    color_bgr: Tuple[int, int, int] = (0, 220, 220),
    marker_size: int = 10,
) -> np.ndarray:
    """Draw a + cross at each anchor; image becomes a 3-channel BGR uint8."""
    if image_uint8.ndim == 2:
        out = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    else:
        out = image_uint8.copy()
    for x, y in anchors_xy:
        if np.isnan(x) or np.isnan(y):
            continue
        ix, iy = int(round(x)), int(round(y))
        cv2.drawMarker(out, (ix, iy), color_bgr,
                       markerType=cv2.MARKER_CROSS,
                       markerSize=marker_size, thickness=2)
    return out


def _diff_image(ref_uint8: np.ndarray, comp_uint8: np.ndarray,
                wire_dilated: np.ndarray) -> np.ndarray:
    diff = np.abs(ref_uint8.astype(np.int16) - comp_uint8.astype(np.int16))
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    # Mask out wire region (we don't compensate the wire, so wire pixels in
    # the diff are meaningless residuals from the active push).
    diff[wire_dilated > 0] = 0
    return diff


def render_frame(
    t: int, T: int,
    ref_uint8: np.ndarray,
    cur_uint8: np.ndarray,
    comp_uint8: np.ndarray,
    anchors_ref: np.ndarray,
    anchors_cur: np.ndarray,
    anchors_comp: np.ndarray,
    wire_mask_t_dilated: np.ndarray,
    sidebar_text: str,
    dpi: int = 200,
) -> np.ndarray:
    """Render one composite frame at time t."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(15, 4), dpi=dpi)
    fig.suptitle(f"frame {t+1}/{T}    {sidebar_text}", fontsize=10, y=0.99)

    panel1 = _draw_anchors(ref_uint8, anchors_ref, color_bgr=(0, 220, 220))
    panel2 = _draw_anchors(cur_uint8, anchors_cur, color_bgr=(0, 220, 220))
    panel3 = _draw_anchors(comp_uint8, anchors_comp, color_bgr=(0, 220, 220))
    panel3 = _draw_anchors(panel3, anchors_ref, color_bgr=(60, 60, 240),
                           marker_size=8)
    diff = _diff_image(ref_uint8, comp_uint8, wire_mask_t_dilated)

    titles = [
        "(1) Reference I_r  [yellow=bg anchors]",
        f"(2) Current I_t (frame {t+1})  [yellow=LK-tracked anchors]",
        "(3) Compensated warp(I_t, -U_t)  [yellow=current, red=ref target]",
        "(4) |I_r - warp(I_t, -U_t)|  (wire masked out)",
    ]
    panels = [panel1, panel2, panel3, diff]
    cmaps = [None, None, None, "hot"]

    for ax, img, title, cm in zip(axes, panels, titles, cmaps):
        if cm is not None:
            ax.imshow(img, cmap=cm, vmin=0, vmax=80)
        elif img.ndim == 3:
            ax.imshow(img[..., ::-1])  # BGR -> RGB
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    rgb = buf[..., :3].copy()
    plt.close(fig)
    return rgb


def render_sequence(
    paths: DataPaths,
    dataset: str, split: str, seq_name: str,
    *,
    model: Optional[FreqDecGWNet],
    motion_field_type: str,
    reset_every: int,
    output_dir: Path,
    fps: int = 10,
    dpi: int = 200,
    max_frames: Optional[int] = None,
    device: str = "cpu",
) -> dict:
    import imageio.v3 as iio
    loader = FluoroSequenceLoader(paths)
    seq = loader.load_sequence(dataset, split, seq_name, max_frames=max_frames)
    T = seq.num_frames
    images_uint8 = (seq.frames[:, 0].cpu().numpy() * 255).astype(np.uint8)
    wire_mask_uint8 = None
    if seq.masks is not None:
        wire_mask_uint8 = (seq.masks[:, 0].cpu().numpy() > 0.5).astype(np.uint8)
    H, W = images_uint8.shape[1:]

    selector = ReferenceSelector(warmup_min=5, warmup_max=10)
    ref = selector.select(images=seq.frames[:10, 0], total_frames=T)
    ref_idx = ref.reference_index

    # Inference
    if model is None:
        U = torch.zeros(T, 2)
        method_label = "no-compensation baseline"
    else:
        with torch.no_grad():
            x = seq.frames.unsqueeze(0).to(device)
            out = model(x, mode="stage3_joint", reference_index=ref_idx)
            U = out["U"][0].cpu()
        method_label = f"{motion_field_type} + reset={reset_every}"
    U = _apply_reset(U, reset_every)

    # Compensated images
    with torch.no_grad():
        comp = apply_compensation(
            seq.frames.unsqueeze(0).to(device),
            U.unsqueeze(0).to(device),
            mode="current_to_reference",
        )[0].cpu()
    comp_uint8 = (comp[:, 0].numpy() * 255).clip(0, 255).astype(np.uint8)

    # Anchors
    wire_ref = (wire_mask_uint8[ref_idx] if wire_mask_uint8 is not None
                else np.zeros((H, W), dtype=np.uint8))
    anchors_ref = detect_background_features(
        images_uint8[ref_idx], wire_ref, max_features=12, wire_dilation_px=32,
    )
    track = (
        track_features_across_burst(images_uint8, anchors_ref, reference_index=ref_idx)
        if anchors_ref.shape[0] > 0 else None
    )

    # Dilated wire mask per frame for diff masking
    if wire_mask_uint8 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))
        wire_dilated = np.stack(
            [cv2.dilate(w, kernel) for w in wire_mask_uint8], axis=0,
        )
    else:
        wire_dilated = np.zeros_like(images_uint8)

    sidebar_base = f"{seq.dataset}/{seq.split}/{seq.seq_name}  {method_label}"

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    U_np = U.cpu().numpy()
    for t in range(T):
        ar = anchors_ref
        ac = (track.feature_xy_per_frame[t] if track is not None
              else np.full_like(anchors_ref, np.nan))
        # Compensated anchor position: tracked - U_t (so should land on ref)
        acmp = ac - U_np[t][None, :] if ac.size else ac
        frame = render_frame(
            t, T, images_uint8[ref_idx], images_uint8[t], comp_uint8[t],
            anchors_ref=ar, anchors_cur=ac, anchors_comp=acmp,
            wire_mask_t_dilated=wire_dilated[t],
            sidebar_text=sidebar_base, dpi=dpi,
        )
        frames.append(frame)

    mp4_path = output_dir / f"{seq.seq_name}_paper_viz.mp4"
    h, w, _ = frames[0].shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h or pad_w:
        frames_pad = [np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)),
                             constant_values=255) for f in frames]
    else:
        frames_pad = frames
    iio.imwrite(mp4_path, frames_pad,
                plugin="pyav", fps=fps, codec="h264",
                out_pixel_format="yuv420p")

    # Static thumbnail at mid-sequence
    mid = min(T - 1, T // 2)
    static_path = output_dir / f"{seq.seq_name}_paper_viz_static.png"
    iio.imwrite(static_path, frames[mid])

    return {
        "mp4": mp4_path, "static": static_path,
        "T": T, "reference_index": ref_idx,
        "n_anchors": int(anchors_ref.shape[0]),
        "U_max": float(np.linalg.norm(U_np, axis=-1).max()),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", required=True, choices=("train", "val", "test"))
    p.add_argument("--seq", nargs="+", required=True)
    p.add_argument("--ckpt", type=Path, default=None,
                   help="if omitted, renders no-compensation baseline")
    p.add_argument("--state-core-type", default="mamba", choices=("gru", "mamba"))
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--motion-field-type", default="relative",
                   choices=("relative", "absolute"))
    p.add_argument("--reset-every", type=int, default=16)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args(list(argv) if argv is not None else None)

    paths = (
        DataPaths.from_yaml(args.config) if args.config
        else DataPaths.from_default_config()
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = None
    if args.ckpt is not None:
        model = FreqDecGWNet(
            width_mult=args.width_mult,
            state_core_type=args.state_core_type,
            motion_field_type=args.motion_field_type,
        ).to(device)
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck)
        sd = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in sd.items()
        }
        model.load_state_dict(sd, strict=False)
        model.eval()
        print(f"[viz] ckpt={args.ckpt.name}  motion={args.motion_field_type}  "
              f"reset={args.reset_every}")
    else:
        print(f"[viz] baseline (no compensation)")

    for sname in args.seq:
        out = render_sequence(
            paths,
            dataset=args.dataset, split=args.split, seq_name=sname,
            model=model, motion_field_type=args.motion_field_type,
            reset_every=args.reset_every,
            output_dir=args.output_dir,
            fps=args.fps, dpi=args.dpi,
            max_frames=args.max_frames, device=device,
        )
        print(f"  {sname:18s} T={out['T']:3d}  ref={out['reference_index']:3d}  "
              f"anchors={out['n_anchors']:2d}  U_max={out['U_max']:.1f}  "
              f"-> {out['mp4'].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
