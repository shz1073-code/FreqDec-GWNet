# PROJECT_CONSTRAINTS_19.6.md
# FreqDec-GWNet 最终防守边界条件

## 0. 总原则

19.6 是 19.x 的最终防守版。后续 Codex / Claude Code 实现必须以本文件为准。

最终论文主张：

> Frequency-guided sequence-relative physiological motion decoupling for guidewire observation and ECG-free fluoroscopic roadmapping.

默认模型名：

`FreqDec-GWNet`

默认题目：

**Frequency-Guided Sequence-Relative Physiological Motion Decoupling for Guidewire Observation and ECG-Free Fluoroscopic Roadmapping**

## 1. 不可动红线

### R1：频域导丝观测前端

必须保留 FAST-LiteNet-derived local guidewire observer。

禁止：

- 把 Mamba 改成 segmentation backbone。
- 删除频域模块后继续声称 frequency-guided。

### R2：整序列锚点弱监督

必须保留：

- full-sequence state label。
- multi-anchor quality fusion。
- band-limited filtering。
- quality_report / valid_mask。

禁止：

- window-wise Hilbert 主标签。
- global mean / PCA 主标签。

### R3：序列相对增量运动

主方法必须是：

```text
delta_u_phys(t) = G_delta(s_t, s_t - s_{t-1})
U_t = cumulative_sum(delta_u_phys, reference=r)
```

禁止：

```text
u_phys(t) = G_motion(s_t)
```

作为主方法。

### R4：主动/背景运动解耦

必须：

- motion loss only on background。
- wire dilated exclusion mask。
- compensation only physiological component。

禁止：

- 全局光流替代主方法。
- 补偿导丝主动运动。

## 2. Reference Frame 规则

不得默认只用第一帧。

优先级：

1. clinical roadmap/mask frame。
2. warm-up frames 中 score 最高的帧。
3. first frame baseline。

warm-up score：

```text
score_t = image_sharpness_t + anchor_quality_t - ||s_t - s_{t-1}||
```

必须报告 reference strategy。

## 3. Drift 评估规则

必须报告：

- compensation error vs time。
- cumulative drift curve。
- landmark residual before/after。

如果存在 near-cycle：

```text
E_cycle = ||U_K - U_ref||
```

若无完整周期，报告 not available，不伪造 cycle。

TMI 版本建议做 reference reset ablation。

## 4. Mamba vs GRU 规则

Mamba 是可选 state core，不是论文根基。

必须先做固定帧率公平对比。

建议增加：

- dropped-frame stress test。
- fps downsampling test。
- irregular interval test。

若做 irregular interval，建议输入 `delta_t` 或 time-gap embedding。

禁止写：

- Mamba 必然优于 GRU。
- Mamba 天然适合所有序列。

只有在实验显著优于 GRU 时，标题或贡献才写 Mamba。

## 5. L_motion 表述规则

主配置：

```yaml
detach_prev_feature: true
detach_curr_feature: true
detach_state_input: false
```

论文中应写：

> L_motion optimizes the state-conditioned motion mapping.

不要写成：

> L_motion directly improves encoder feature representation.

feature-adaptive 配置只能作为消融。

## 6. 消融优先级

主文：

1. absolute vs relative。
2. state training protocol。
3. motion loss detach policy。
4. reference selection。
5. GRU vs Mamba fixed/dropped-frame。

附录：

1. 9D vs 6D vs 4D。
2. frequency ablation。
3. max_shift sensitivity。
4. motion feature source。
5. LP input ablation。

频域消融保留，但不抢主线。

## 7. 实现前检查

每个新任务先确认：

- 是否破坏 R1-R4。
- reference frame 是否有策略。
- drift 是否可量化。
- Mamba 是否仍只在 temporal state branch。
- compensation direction 是否有单元测试。
- patient/sequence split 是否无泄漏。

## 8. 一句话

> 后续实现不再改故事，只围绕 reference selection、drift quantification 和 fair Mamba/GRU robustness testing 做防守增强；主线永远是频域导丝观测、整序列低频状态、序列相对运动增量和主动/背景运动解耦。
