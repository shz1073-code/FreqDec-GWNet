"""Diagnostic: is the GRU val failure a per-sequence phase OFFSET, not a
learning failure?

For each val sequence we forward the state branch, then compute:
  raw_r      — Pearson(cos_pred, cos_gt)              [what eval reports]
  best_delta — phase offset that best aligns pred to gt (degrees)
  aligned_r  — Pearson(cos(phi_pred - delta), cos_gt) after that offset
  aligned_mae— circular MAE after offset (degrees)

If aligned_r is high (>0.8) on the sequences where raw_r is negative,
the model DOES track respiration — it is only off by a constant phase
reference. That is a label-consistency problem, not a model-capacity one.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freqdec_gwnet.data import FluoroSequenceWindowDataset           # noqa: E402
from freqdec_gwnet.data.state_labels import StateLabel               # noqa: E402
from freqdec_gwnet.models.freqdec_gwnet import FreqDecGWNet          # noqa: E402

import cv2

DATA_ROOT = Path("/workspace/shz/clean_data")
SL_DIR = ROOT / "reports/state_labels/clean_v2"
CKPT = ROOT / "experiments/checkpoints/stage2_gru_chrono_v5_best.pth"
BURN_IN = 16


def forward_seq(model, ds, seq_name, device):
    matching = [w for w in ds.windows if w.seq_name == seq_name]
    if not matching:
        return None
    sl = StateLabel.load_npz(matching[0].state_label_path)
    T = sl.num_frames
    H, W = ds.img_size
    imgs = np.zeros((T, 1, H, W), dtype=np.float32)
    for k, fn in enumerate(matching[0].frame_names[:T]):
        img = cv2.imread(str(matching[0].seq_dir_images / fn), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if (img.shape[0], img.shape[1]) != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        imgs[k, 0] = img.astype(np.float32) / 255.0
    imgs_t = torch.from_numpy(imgs)
    inner = model
    feats = []
    with torch.no_grad():
        for s in range(0, T, 32):
            e = min(T, s + 32)
            f = inner.r1.encoder(imgs_t[s:e].to(device))[2].cpu()
            feats.append(f)
    feat = torch.cat(feats, 0).unsqueeze(0).to(device)
    with torch.no_grad():
        st = inner.state_branch.forward_window(feat)[0].cpu()
    cos_p, sin_p = st[:, 0], st[:, 1]
    n = torch.sqrt(cos_p ** 2 + sin_p ** 2).clamp(min=1e-6)
    cos_p, sin_p = cos_p / n, sin_p / n
    valid = torch.from_numpy(sl.valid_mask).clone()
    valid[:BURN_IN] = False
    return (cos_p, sin_p,
            torch.from_numpy(sl.cos_phi), torch.from_numpy(sl.sin_phi),
            valid)


def pearson(a, b, m):
    a, b = a[m].float(), b[m].float()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b / (a.norm() * b.norm()).clamp(min=1e-8)).item())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FreqDecGWNet(width_mult=1.0, state_core_type="gru").to(device)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    ds = FluoroSequenceWindowDataset(
        data_root=DATA_ROOT, state_labels_dir=SL_DIR, split="val",
        T_window=64, img_size=(512, 512),
    )
    seqs = sorted({w.seq_name for w in ds.windows})
    print(f"{'seq':14s} {'raw_r':>7s} {'best_dphi':>10s} "
          f"{'aligned_r':>10s} {'aligned_MAE':>12s}")
    print("-" * 58)
    for s in seqs:
        out = forward_seq(model, ds, s, device)
        if out is None:
            continue
        cos_p, sin_p, cos_g, sin_g, m = out
        raw_r = pearson(cos_p, cos_g, m)
        # optimal phase offset: angle of sum(z_pred * conj(z_gt))
        zp = cos_p + 1j * sin_p
        zg = cos_g + 1j * sin_g
        cross = (zp[m] * torch.conj(zg[m])).sum()
        delta = torch.atan2(cross.imag, cross.real)          # phi_pred - phi_gt
        cos_pa = cos_p * math.cos(-delta) - sin_p * math.sin(-delta)
        sin_pa = sin_p * math.cos(-delta) + cos_p * math.sin(-delta)
        aligned_r = pearson(cos_pa, cos_g, m)
        cd = (cos_pa * cos_g + sin_pa * sin_g).clamp(-1, 1)
        aligned_mae = float(torch.acos(cd[m]).mean().item()) * 180 / math.pi
        print(f"{s:14s} {raw_r:+7.3f} {float(delta)*180/math.pi:+10.1f} "
              f"{aligned_r:+10.3f} {aligned_mae:12.1f}")


if __name__ == "__main__":
    main()
