---
name: PROJECT_CONSTRAINTS_19.6 红线
description: 不可破坏的四条红线R1-R4及其他规则，每次实现前必须核对
type: project
---

**R1（频域前端）：** 必须保留FAST-LiteNet-derived local guidewire observer；禁止把Mamba改成segmentation backbone；禁止删频域模块。

**R2（整序列弱监督）：** 必须保留full-sequence state label、multi-anchor quality fusion、band-limited filtering、quality_report/valid_mask；禁止window-wise Hilbert或global mean/PCA作为主标签。

**R3（序列相对增量）：** 主方法必须是 `delta_u_phys(t) = G_delta(s_t, s_t - s_{t-1})` + `cumulative_sum`；禁止 `u_phys(t) = G_motion(s_t)` 作为主方法。

**R4（主动/背景解耦）：** motion loss只作用于background；wire dilated exclusion mask；只补偿生理分量；禁止全局光流替代；禁止补偿导丝主动运动。

**Reference Frame规则：** 优先级：clinical roadmap帧 > warm-up高分帧 > first frame；评分=image_sharpness + anchor_quality - ||s_t - s_{t-1}||

**Mamba规则：** Mamba只用于temporal state branch；必须先做固定帧率公平对比；只有实验显著优于GRU才在标题写Mamba。

**L_motion规则：** 主配置detach_prev_feature=true, detach_curr_feature=true, detach_state_input=false；写作"optimizes state-conditioned motion mapping"，不写"directly improves encoder feature"。

**Why:** 这是19.6最终防守版的约束，保证论文主线不被审稿人攻击。

**How to apply:** 每个新任务先检查是否破坏R1-R4；reference frame是否有策略；drift是否可量化；compensation direction是否有单元测试。
