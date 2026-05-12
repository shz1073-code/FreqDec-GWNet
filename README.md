# FreqDec-GWNet

Frequency-guided sequence-relative physiological motion decoupling for guidewire observation and ECG-free fluoroscopic roadmapping.

## Project Positioning

FreqDec-GWNet is not a generic Mamba segmentation network. The local guidewire observation front-end is derived from FAST-LiteNet, while GRU/Mamba is used only as a temporal state core for low-frequency physiological state modeling.

The current protected story is defined in `PROJECT_CONSTRAINTS_19.6.md`. Read that file before changing model design, training protocol, or motion compensation logic.

## Directory Layout

```text
docs/                  Paper outlines, technical routes, constraints, references
src/freqdec_gwnet/     Package code
scripts/               Training, evaluation, and data-preparation entry points
configs/               Stage-wise experiment configs
tests/                 Unit tests and convention checks
data/                  Local data placeholders, ignored by git
experiments/           Experiment outputs, ignored by git
checkpoints/           Model checkpoints, ignored by git
reports/               Evaluation reports and figures
```

## Current Implementation Status

- `src/freqdec_gwnet/models/`: FAST-LiteNet-derived local front-end and global state branch.
- `src/freqdec_gwnet/losses/`: segmentation losses and physiological state losses.
- `scripts/train_stage1_seg.py`: copied baseline training entry point for Stage 1 segmentation.
- `scripts/evaluate_segmentation.py`: copied segmentation evaluation entry point.
- `scripts/evaluate_stability.py`: copied localization/stability evaluation entry point.

## Protected Core

1. Frequency-guided local guidewire observation.
2. Full-sequence anchor-supervised low-frequency physiological state learning.
3. Sequence-relative incremental physiological motion modeling.
4. Active guidewire motion and physiological background motion decoupling.

