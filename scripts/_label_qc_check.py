"""Ad-hoc QC: assess Phase-B label quality across val sequences."""
import numpy as np
from pathlib import Path

fs = 15.0
val_dir = Path("reports/state_labels/clean_v2/val")
# good vs bad as ranked by state-branch Pearson
order = ["cmu_seq_007", "zhu_seq_001", "cmu_seq_001",
         "cmu_seq_006", "cmu_seq_005", "zhu_seq_003"]

hdr = f"{'seq':14s} {'T':>4s} {'valid%':>7s} {'anchorQ':>8s} {'f_dom':>7s} {'inband%':>8s} {'ampCV':>7s}"
print(hdr)
print("-" * len(hdr))
for s in order:
    d = np.load(val_dir / f"{s}.npz", allow_pickle=True)
    cos, sin, amp = d["cos_phi"], d["sin_phi"], d["amplitude"]
    vm, aq = d["valid_mask"], d["anchor_quality"]
    z = cos + 1j * sin
    P = np.abs(np.fft.fft(z)) ** 2
    fr = np.fft.fftfreq(len(z), d=1.0 / fs)
    inband = (np.abs(fr) >= 0.15) & (np.abs(fr) <= 0.5)
    frac = P[inband].sum() / P.sum()
    pos = fr > 0
    f_dom = fr[pos][np.argmax(P[pos])]
    ampcv = amp.std() / (abs(amp.mean()) + 1e-6)
    print(f"{s:14s} {len(z):4d} {100*vm.mean():7.1f} {aq.mean():8.3f} "
          f"{f_dom:7.3f} {100*frac:8.1f} {ampcv:7.2f}")
