# FreqDec-GWNet — Method §III (v2 draft, B+ framing)

> Status: draft v2 (2026-05-12). Direction = **A-lite + B+ hybrid** per GPT
> critique (2026-05-12). Headline claim drifts away from "relative >
> absolute on tip-projected residual" toward **state estimation +
> stabilization framework** with cumulative-drift mitigation as a key
> engineering contribution. Final decision on the headline is deferred
> until the v4 pipeline finishes (see `background_pipeline_v4.sh`).

---

## Possible paper titles (pick after v4 results)

* **(if relative_v4 reset32 beats absolute on Metric A)**
  "Frequency-Guided Cumulative Motion Decoupling with Zero-Mean
  Regularization for ECG-Free Fluoroscopic Roadmapping"

* **(if relative still loses on Metric A but inter-burst Mamba wins)**  
  "ECG-Free Physiological State Estimation for Fluoroscopic Roadmap
  Stabilization: A Multi-Anchor Weak-Supervision Framework"

* **(safe fallback)**
  "Mamba-Based Physiological State Recurrence and Drift-Aware Motion
  Compensation in Fluoroscopic Sequences"

---

## III. Method (B+ framing)

We frame ECG-free fluoroscopic roadmap stabilization as three coupled
sub-problems and present a unified framework that addresses each with
contributions that *do not depend on the cumulative-motion endpoint
being superior to a direct-regression baseline*:

* (a) **Frequency-guided guidewire observation** (§III-A) — segments the
  active instrument so it can be excluded from background-motion
  estimation downstream.

* (b) **Multi-anchor band-limited weak supervision for physiological
  state** (§III-B) — **primary novelty**: extracts an ECG-free
  respiratory-state estimate (cos φ, sin φ, amplitude) from the
  background dynamics through phase correlation, Butterworth band-pass,
  Hilbert envelope, and 2×2 spatial multi-anchor consensus. The
  per-burst output is a valid_mask + amp_scale that the state branch
  consumes during stage-2 supervision.

* (c) **Drift-aware sequence-relative motion compensation** (§III-C) —
  parametrizes the per-frame physiological motion as an increment
  Δu_t = G_δ(s_t, Δs_t) accumulated relative to a chosen reference
  frame r. **Engineering contribution**: a zero-mean regularizer that
  addresses a previously-unreported failure mode of cumulative
  formulations (monotonic-bias drift) plus an inference-time reference
  reset that bounds the cumulative window.

### III-A. Frequency-Guided Local Wire Observer

Standard FAST-LiteNet-derived segmentation backbone with a
spatio-temporal attention block at the 1/32 bottleneck (frequency
residual via 2D rFFT + temporal cross-attention with a feature bank).
The frame-level wire mask M_t feeds both §III-B (used as exclusion
region in background-motion estimation) and §III-C (used to mask the
photometric residual). This section is technique-borrowing rather than
the paper's primary novelty.

### III-B. Multi-Anchor Band-Limited Weak Supervision

This is the **primary methodological contribution**. We never assume
ECG signals are available; instead we recover a respiratory state
purely from how the visible background of each fluoroscopic burst
moves across frames. The four-stage offline pipeline (Algorithm 1) is:

1. **Background motion via phase correlation** — Hann-windowed 2-D
   phase correlation on consecutive frames after dilating the wire
   mask by 16 px and zero-meaning the joint background ROI. Returns
   per-frame Δb_t ∈ R².

2. **PCA principal-axis projection** — cumulative b_t = Σ Δb_t is
   projected onto its first PCA axis. On our data this axis is
   consistently within 4° of the image-vertical (AP) direction.

3. **Butterworth band-pass + Hilbert envelope** — the 1-D projection
   is passed through an order-4 zero-phase band-pass at 0.15–0.50 Hz
   (covering 9–30 BPM) and an analytic Hilbert transform produces
   instantaneous phase φ_t and envelope A_t. **Zero-phase is
   essential**: phase-shifted output would break the multi-anchor
   alignment downstream.

4. **Multi-anchor consensus** — steps 1–3 are repeated on each of 4
   spatial anchor ROIs (2×2 grid with 10% overlap). Per-frame
   validity v_t requires ≥3 of the 4 anchors to agree on the
   instantaneous phase (circular variance < 0.5).

The resulting per-burst label `(cos φ_t*, sin φ_t*, A_t*, v_t,
A_scale = P95(A_t*|v_t))` is stored as a .npz and consumed by the
state branch (§III-D) under a circular phase loss + amp-scale-
normalized SmoothL1.

**Why this matters**: prior fluoroscopy compensation work either
requires intra-procedural ECG signals (which interventional
suites often lack), or trains the state branch from synthetic
respiratory traces (which generalize poorly). Our pipeline is the
first, to our knowledge, to derive a full-burst ECG-free state
label directly from the X-ray dynamics via multi-anchor band-
limited extraction.

### III-C. Drift-Aware Sequence-Relative Motion Field

We predict an incremental displacement Δu_t and accumulate it
relative to a reference frame r:

```
Δs_t = s_t - s_{t-1},  Δs_0 := 0
ψ_t = [s_t, Δs_t] ∈ R^{2D} (default D=3)
Δu_t = G_δ(ψ_t) ∈ R²   (pixel translation)
Δu_0 := 0
U_t = Σ_{i=r+1..t} Δu_i       for t ≥ r
U_t = 0                        for t < r
```

G_δ is a 3-layer MLP with zero-initialised final layer.

**The cumulative-bias failure mode and our zero-mean fix** (engineering
contribution to be reported transparently in §IV.A). Initial training
under a pure photometric L_motion produced a pathology we did not
expect: G_δ converged to a small positive bias (~+1.5 px per frame on
our data), and cumulative summation accumulated this bias into a
monotonic drift of U_t > 200 px over 100-frame bursts, even though
the actual respiratory amplitude was ~20 px. The pairwise photometric
residual was locally satisfied (small mean Δu) but the cumulative
endpoint was wildly wrong.

We address this with two complementary mechanisms:

* **Zero-mean regularizer (training)**:

    ```
    L_zm = ‖(1/T) Σ_t Δu_t‖²
    ```

  Because respiratory motion is a bounded periodic signal whose
  long-window mean approaches zero, L_zm cleanly penalizes the
  bias-shortcut without punishing legitimate oscillation.

* **Reference reset (inference)**: at deployment time we periodically
  zero U_t every N frames (default N=32). This is operationally
  realistic — clinicians re-anchor when they reach a new vessel
  segment — and bounds the cumulative window the model has to
  remain accurate over.

We report drift_max reductions of 5–8× on real fluoroscopy bursts
when zero-mean is enabled vs an otherwise-identical baseline, and a
further ~1.5–2× reduction when reference reset is applied at eval.

### III-D. State Branch and Training Protocol

Mamba selective-SSM state core (d_state=16, expand=2) on the 1/8
encoder feature map after spatial low-pass + GAP. Output head emits
the 4-D `(cos, sin, amp, conf)` vector consumed by §III-C.

Stage 1 trains R1 with Dice + clDice on clean_data. Stage 2 trains
the state branch (encoder frozen) under the chronological-carry
protocol: windows from the same burst are processed in temporal
order so the SSM hidden state carries from window k into window
k+1 (the SSM ablation §IV.E shows this gives a 3.3× reduction in
the inter-burst phase boundary jump). Stage 3 freezes R1 again,
trains state-branch + motion-field jointly under
λ_state · L_state + λ_motion · (L_photo + λ_zm · L_zm).

---

## IV. Experiments (skeleton — fill once v4 finishes)

The paper's experimental claim is a layered one:

* **§IV-A** Datasets and metrics. Two **field-standard** metrics that
  separate respiratory motion from active push:
    * **Metric A** — respiratory-projected residual on tip landmarks,
      following Shechter et al. 2004 TMI (Algorithm 2)
    * **Metric B** — background-patch alignment via LK on Shi-Tomasi
      corners outside the dilated wire region, following Ablitt et
      al. 2004 TMI / Ambrosini et al. 2015 TMI

* **§IV-B** Segmentation backbone — Dice and clDice on clean_data/val
  (~0.79–0.82 expected; this is technique-borrowing, not the
  contribution).

* **§IV-C** State estimation quality — phase MAE, amplitude RMSE on
  the held-out clean_data/val sequences (this section is a clean
  positive endpoint regardless of motion-field outcome).

* **§IV-D** Compensation — **the headline**. Three rows:
  1. **No compensation (baseline)** — Metric A/B on raw landmarks
  2. **Absolute (G_motion(s_t))** — direct-regression head
  3. **Ours (sequence-relative + zero-mean + reset)** — the full method

* **§IV-E** Ablations — zero-mean on/off, reference-reset N sweep,
  GRU vs Mamba on phase Pearson and inter-burst boundary jump.

* **§IV-F** Inter-burst continuity — the §III-D chronological-carry
  protocol's clear positive endpoint.

The paper's headline claim **DOES NOT REQUIRE relative to dominate
absolute on every metric**. The defensible positions are:

* (i) State estimation is high-quality on real fluoroscopy without ECG
  (Metric: phase MAE + amplitude RMSE on val sequences).
* (ii) The cumulative formulation has a drift failure mode we
  characterize and fix (drift_max reduction 5–8× via zero-mean).
* (iii) Mamba state recurrence carries respiratory phase across burst
  gaps better than GRU (Metric: inter-burst boundary jump ratio).

If (i)–(iii) all hold cleanly, the paper is publishable in JBHI/TBME
even when relative ties or marginally loses to absolute on tip-level
compensation — the paper is then positioned as a *framework for
ECG-free physiological state estimation with drift-aware compensation
support*, which is a cleaner contribution than a single-metric horse
race.

---

## V. Limitations (will be expanded after v4 results)

* Single-centre data; no cross-hospital validation.
* Tip-landmark cohort (dense_v1) is only 3 train + 3 val sequences;
  the inter-burst sandbox uses 3 long sequences from clean_data
  for the §III-D positive endpoint.
* No prospective patient evaluation.
* The zero-mean regularizer is a *bounded-periodic-signal* heuristic;
  it would fail on sequences with sustained genuine bulk drift (e.g.
  C-arm panning during the burst) — see §IV.D, cmu_seq_017 outlier
  discussion.
