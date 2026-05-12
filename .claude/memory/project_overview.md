---
name: FreqDec-GWNet Project Overview
description: 论文项目总体情况：目标、主张、三阶段训练、投稿目标
type: project
---

论文题目：Frequency-Guided Sequence-Relative Physiological Motion Decoupling for Guidewire Observation and ECG-Free Fluoroscopic Roadmapping

模型名：FreqDec-GWNet

**核心主张：** 频域导丝观测 + 整序列低频状态学习 + 序列相对生理运动增量建模 + 主动/背景运动解耦，实现无ECG roadmapping。

**三阶段训练：**
1. Stage1 (configs/stage1_seg.yaml)：分割前端训练，FAST-LiteNet
2. Stage2 (configs/stage2_state.yaml)：全局生理状态分支训练
3. Stage3 (configs/stage3_joint.yaml)：联合训练+运动场

**投稿目标：** JBHI（最低要求），TMI（需要更多ablation和cycle-consistency）

**Why:** 用于透视荧光导丝手术中的背景生理运动补偿，无需ECG信号。

**How to apply:** 每次修改前检查是否违反R1-R4约束；代码目录在/home/kuka7/Desktop/FreqDec-GWNet；实验在docker容器shz中运行。
