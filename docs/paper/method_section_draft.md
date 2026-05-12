# FreqDec-GWNet — Method Section Draft (JBHI target)

> Status: draft v1 (2026-05-11). Intended target: IEEE JBHI.
> Mamba kept as the default state core per career-development rationale;
> the technical case is "we use a contemporary state-space model whose
> selective-SSM structure is well-suited to long, irregularly-sampled
> physiological time series." We do **not** make Mamba's superiority over
> GRU a paper claim — the comparison is provided as an ablation only.

---

## III. Method

We formulate ECG-free physiological compensation for fluoroscopic
roadmapping as three coupled sub-problems:

1. **frequency-guided guidewire observation** — segment the active wire
   and provide a per-frame foreground mask used to gate downstream
   modules (§III-A);
2. **full-sequence physiological state estimation** — recover a
   low-dimensional, band-limited respiratory state from the
   background dynamics (§III-B);
3. **sequence-relative incremental motion compensation** — predict and
   accumulate a per-frame translation that warps the background to the
   reference frame's coordinate system, while leaving the wire's active
   motion untouched (§III-C).

The three modules share a single GhostNet-derived encoder; subsequent
heads operate at the 1/8-resolution bottleneck. Figure 1 shows the
overall architecture.

### III-A. Frequency-Guided Local Wire Observer

We adopt the FAST-LiteNet-derived frequency-guided segmenter. Each
frame I_t passes through a 5-stage Ghost encoder producing feature maps
F_t^{(2)} (1/2 res), F_t^{(4)}, F_t^{(8)}, F_t^{(16)}, F_t^{(32)}. At
the 1/32 bottleneck we apply a spatio-temporal attention block (STA)
that fuses:

* **frequency residual** — a learnable per-channel mask applied in the
  2-D rFFT domain, plus an inverse-FFT residual added back to the
  spatial feature, optionally gated by a channel-wise sigmoid;
* **temporal cross-attention** — a feature bank f_{t-1}^{(32)} from the
  previous frame is keyed by f_t^{(32)} and aggregated through a 1-D
  channel attention, modulated by the cosine similarity between the
  flattened features (so abrupt scene changes do not poison the bank).

The decoder is four bilinear-upsample + skip-concat blocks back to full
resolution. The wire segmentation head emits per-pixel logits trained
under a Dice + clDice topology loss; optional centerline / edge heads
sharpen the boundaries. The wire mask M_t \\in \\{0,1\\}^{H \\times W} is used by
both the weak-supervision pipeline (§III-B) and the motion loss (§III-C)
to exclude active wire pixels from any background-related estimation.

### III-B. Full-Sequence Physiological State with Multi-Anchor Weak Supervision

The state branch ingests the 1/8 feature map F_t^{(8)}, applies a
learnable depthwise low-pass filter (Gaussian-initialised) to suppress
wire high-frequency response, projects to a 128-channel space, performs
global-average pooling, and feeds the resulting z_t \\in R^{128}
sequence into a temporal core. We use a Mamba selective-SSM core as
default (d_state = 16, expand = 2); a GRU core is available as ablation.
A 4-D output head emits

```
s_t = (cos\\phi_t, \\sin\\phi_t, A_t, c_t)
```

where \\phi_t and A_t are the instantaneous respiratory phase and
amplitude, and c_t is a learned confidence. The first three channels
are consumed by the motion field (§III-C); confidence drives a future
reference-reset extension.

**Weak supervision generation.** Because patient-level ECG signals are
not available in our retrospective dataset, we generate per-burst
state labels offline through a four-stage pipeline (Algorithm 1):

1. **Background motion estimation** — for each consecutive pair
   (I_{t-1}, I_t) we apply 2-D Hann-windowed phase correlation to a
   joint background mask built by dilating M_{t-1} \\cup M_t by 16 px,
   yielding the per-step translation \\Delta b_t \\in R^2.
2. **Principal-axis projection** — cumulative positions
   b_t = \\sum_{s\\le t} \\Delta b_s are projected onto the first PCA
   axis. In our data this axis is consistently within 4° of the
   image-vertical direction, matching AP respiratory anatomy.
3. **Band-limited filtering and Hilbert envelope** — the 1-D
   projection is passed through a zero-phase Butterworth band-pass
   (0.15–0.50 Hz, order 4) and an analytic Hilbert transform produces
   the instantaneous phase and envelope.
4. **Multi-anchor consensus** — Steps 1–3 are repeated independently
   over a 2×2 spatial grid of overlapping anchor regions. The
   per-frame validity mask v_t \\in \\{0,1\\} is set to 1 iff (i) at
   least 3 of the 4 anchors agreed on the phase (phase circular
   variance < 0.5) and (ii) the underlying phase-correlation
   confidence was above its joint background pixel-fraction threshold.

The resulting labels (cos \\phi_t^*, sin \\phi_t^*, A_t^*, v_t, A_{scale})
are persisted as a per-burst .npz file with A_{scale} = P_{95}(A^*)
over valid frames, used downstream to normalise the loss.

**State loss.** We supervise the state branch using a circular phase
loss combined with a scale-normalised amplitude loss and a burn-in
weighting that downweights the first frames of each window:

```
\\mathcal{L}_{state} = \\frac{1}{Z} \\sum_t w_t \\big[
        1 - (\\widehat{e}_t \\cdot e_t^*)
        + \\lambda_{amp} \\cdot \\mathrm{SmoothL1}\\big(\\widehat{A}_t/A_{scale},
                                                       A_t^*/A_{scale}\\big)
    \\big]
```

where `\\widehat{e}_t = (\\cos\\widehat{\\phi}_t, \\sin\\widehat{\\phi}_t)` is
unit-normalised before the dot product (preventing the model from
gaming the loss via magnitude). w_t combines the burn-in mask with v_t,
and Z is the total weight. The circular form avoids the wrap-around
discontinuity of MSE-on-angle.

### III-C. Sequence-Relative Incremental Motion Field

The compensation field is built sequence-relative rather than
globally-absolute: rather than predicting an absolute displacement
u_t = G_{motion}(s_t), the model predicts an **increment**
\\Delta u_t = G_{\\delta}(s_t, \\Delta s_t) and accumulates them
relative to a chosen reference frame r:

```
\\Delta s_t = s_t - s_{t-1},                \\Delta s_0 := 0
\\psi_t   = [s_t, \\Delta s_t]   \\in R^{2D}     (D = 3 by default)
\\Delta u_t = G_{\\delta}(\\psi_t)            \\in R^2  (pixel translation)
\\Delta u_0 := 0
U_t = \\sum_{i=r+1}^{t} \\Delta u_i              for t \\ge r
U_t = 0                                       for t \\le r
```

G_{\\delta} is a small 3-layer MLP with a zero-initialised final layer
so that an untrained model produces U_t \\equiv 0 — preventing any
inadvertent "wild compensation" at the start of training.

**Reference frame selection.** The reference r is selected with a
three-tier policy. (1) If the clinician provides an explicit roadmap
frame index it is used directly. (2) Otherwise we score the first
N = 5–10 warm-up frames by

```
score_t = \\sigma\\big(\\mathrm{Sharpness}(I_t)\\big)
        + \\sigma\\big(\\text{anchor\\_quality}(t)\\big)
        - \\sigma\\big(\\|s_t - s_{t-1}\\|\\big)
```

(all terms min–max normalised over the warm-up window) and take the
argmax. (3) The first frame is the fall-back only when neither of the
above is available. We always report which tier produced r.

**Zero-mean regularisation.** A naive cumulative formulation admits a
shortcut: G_{\\delta} can drift to a small positive constant, which
minimises the local photometric loss while letting U_t grow without
bound. We discovered this empirically (Section IV.A); without
regularisation, U_t reached 200–400 px on bursts whose true
respiratory amplitude was 15–25 px. Because respiratory motion is
bounded and approximately zero-mean over a sufficiently long window,
we add an explicit zero-mean penalty:

```
\\mathcal{L}_{zm} = \\Big\\| \\frac{1}{T-1}
                  \\sum_{t=1}^{T-1} \\Delta u_t \\Big\\|_2^2
```

This regulariser is **the key engineering contribution** of this work:
it is invariant to true oscillation (a perfectly periodic signal has
zero mean) but directly punishes the monotonic-bias failure mode.

**Motion loss.** Compensation is supervised by warping the previous
frame's 1/8 feature map forward by +\\Delta u_t and comparing against the
current frame within the dilated background mask:

```
\\mathcal{L}_{photo}
= \\frac{1}{\\sum_t \\sum_{\\Omega} M^{bg}_t}
  \\sum_t \\sum_{\\Omega} M^{bg}_t \\cdot
        \\big\\| \\mathrm{warp}(F_{t-1}^{(8)}, +\\Delta u_t)
                - F_t^{(8)} \\big\\|_2^2
```

where M^{bg}_t = 1 - dilate(M_t, k = 3) is the background-only mask
on the current frame. The feature maps are detached on both sides
(detach\\_prev = detach\\_curr = True) so that \\mathcal{L}_{photo} only
optimises G_{\\delta} and the state branch's input pathway, not the
segmentation encoder/decoder representations. \\Delta u_t is scaled by
H^{(8)} / H so the warp operates in the feature-map pixel grid.

The full stage-3 objective is

```
\\mathcal{L} = \\lambda_{state} \\mathcal{L}_{state}
            + \\lambda_{motion} \\big(
                \\mathcal{L}_{photo}
                + \\lambda_{zm} \\mathcal{L}_{zm}
              \\big)
```

with default weights (\\lambda_{state}, \\lambda_{motion}, \\lambda_{zm})
= (1.0, 0.5, 1.0). All gradients flow only into the state branch and
G_{\\delta}; segmentation and encoder parameters are frozen during
stage 3, having been trained in stage 1.

### III-D. Training Protocol

Training proceeds in three stages.

1. **Stage 1 — segmentation pre-training.** Image-and-label pairs from
   the open clean_data partition train the segmentation backbone with
   the FAST-LiteNet loss (Dice + clDice + boundary).
2. **Stage 2 — state branch.** With R1 frozen, the state branch is
   trained against the weakly-supervised labels (§III-B). We use the
   chronological-carry protocol: windows from the same burst are
   processed in temporal order and the hidden state at the end of
   window k seeds window k+1, allowing the SSM to model multi-cycle
   coherence.
3. **Stage 3 — joint motion.** Stage-2 weights initialise the
   FreqDec-GWNet; we further train the motion field G_{\\delta} jointly
   with the state branch under
   \\mathcal{L}_{state} + 0.5 \\cdot (\\mathcal{L}_{photo} + \\mathcal{L}_{zm}).

All stages use AdamW with cosine annealing, base lr = 1e-3 (stage 2)
and 5e-4 (stage 3), weight decay 1e-4, mixed precision off (the FFT
branch is unstable under AMP on our CUDA stack), and batch size 1 with
window length T = 64.

---

## IV. Experiments (skeleton — fill in once stage3_main_v3_full done)

### IV-A. Datasets, Splits, Metrics

* **clean_data** — public single-burst fluoroscopy with frame-level
  wire segmentation labels. 47 train / 10 val sequences; we exclude
  18 bursts shorter than 64 frames so the state pipeline can resolve
  at least one respiratory cycle.
* **dense_v1** — subset with frame-level guidewire-tip 2-D landmarks
  (3 train / 3 val sequences). Used exclusively for drift / landmark
  evaluation.

Metrics: Dice / clDice for segmentation; circular phase error and
amplitude RMSE for state estimation; **drift_max**, **improvement_mean
on tip landmarks**, and **E_cycle** for compensation; phase correlation
and compensation error for the dropped-frame robustness study.

### IV-B. Segmentation

\\[Quote stage1_v1_best.pth — Dice ≈ 0.79 on clean_data/val\\]

### IV-C. State Estimation

\\[Quote stage2_mamba_chrono_v2 phase MAE + amplitude RMSE on val\\]

### IV-D. Compensation — Main Result

Headline number: improvement_mean and drift_max on dense_v1 test
sequences from stage3_main_v3_full (with zero-mean regulariser).
We compare to:

| variant | drift_max ↓ | improvement_mean ↑ |
| --- | --- | --- |
| absolute baseline (G_{motion}(s_t)) | TBD | TBD |
| relative w/o zero-mean (= v2 reproduction) | 264–419 px | −99 to −159 px |
| **relative + zero-mean (ours, v3)** | **44–54 px** | **TBD** |
| relative w/o detach prev feature | 285–502 px | −137 px |

### IV-E. Inter-Burst Continuity (Mamba-relevant sandbox)

We split one 300-frame burst into K=3 chunks separated by 30-frame gaps
and compare four state-core protocols: naive (h_0=0 per chunk), carry
(h_final carried), carry+Δt (with Δt modulation), and oracle (unchunked).
Boundary phase jumps quantify whether the SSM retains physiological
state across the simulated burst gap.

### IV-F. Dropped-Frame Robustness

Single 4090, random drop rate {0.1, 0.3, 0.5} and fps downsampling
factor 2. Phase correlation, compensation error, and drift ratio
reported.

### IV-G. Ablation Study

Five rows: zero-mean on/off; relative vs absolute; detach policy;
reference selection (auto / first / random); GRU vs Mamba.

### IV-H. Qualitative

Figure 4: 4-panel compensation animation showing one sequence with
the wire moving against a now-static background. Figure 5: tip-residual
curve before/after compensation.

---

## V. Discussion

* Limitations: single-centre data, K=3 patient cohort for landmark
  evaluation, no cross-modality verification.
* Future work: explicit Δt embedding and clinical multi-burst
  evaluation, reference-reset triggered by anchor quality, integration
  with intra-procedural roadmap update.
