---
name: Implementation Status
description: 代码骨架现状：已完成模块和19.6待实现的三个防守点
type: project
---

**已完成（约97%）：**
- models/: fast_litenet.py, frequency.py, ghost_blocks.py, ghost_encoder.py, global_state_branch.py(549行), sta_module_v2.py, temporal.py, unet_baseline.py
- losses/: segmentation_losses.py, state_losses.py
- utils/: localization.py, motion_decoupling.py, tta.py, vis_frequency.py
- data/: physics_aug.py, real_dataset.py, video_dataset.py
- scripts/: train_stage1_seg.py(980行), evaluate_segmentation.py(572行), evaluate_stability.py(162行)

**待实现（19.6防守三件事）：**

第一批（优先）：
1. `utils/reference_selector.py` — ReferenceSelector（smart reference frame selection）
2. `scripts/evaluate_drift.py` — drift/cycle-consistency评估脚本
3. `models/relative_motion_field.py` — RelativeMotionField 6D版本
4. `tests/test_motion_convention.py` — 运动方向约定单元测试

第二批：
5. dropped-frame evaluator（在evaluate_drift.py中扩展）
6. delta_t输入选项（global_state_branch.py扩展）
7. cycle-consistency report

第三批：
8. reference reset ablation
9. Mamba vs GRU robustness final table

**Why:** 技术路线19.6要求补齐reference选择、drift量化、Mamba公平压力测试，其余主模型已基本完整。

**How to apply:** 下一步从第一批开始实现；每次实现前核对constraints.md中的R1-R4。
