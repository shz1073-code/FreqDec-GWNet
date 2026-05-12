# PROJECT_CONSTRAINTS_19.2.md
# PhysDec-GWNet 创新保护线、实现守则与 AI 协作规则

---

## 0. 目标

本文件约束 19.2 之后的所有代码修改、论文修改和 AI 对话。
每次新建 Claude Code / Codex 对话，第一步必须让 AI 读取本文件。

目标期刊：首选 TMI，保底 JBHI。

---

## 1. 动了就死的三条红线

### R1：频域任务分解不能被删除

必须保留：
- 局部分支面向导丝高频细长结构（frequency residual enhancement 或等价 frequency-aware 模块）
- 全局分支面向低频生理状态
- 全局分支输入必须有显式低通（LP）或等价低频约束
- 两个分支的功能分工必须能在论文中清晰解释

禁止：
- 把两个分支都改成普通卷积，删除所有频域操作
- 用普通 channel attention 替代全部频域任务分解
- 删除 frequency enhancement 后仍声称频域是核心创新

### R2：整序列锚点弱监督不能退化

必须保留：
- 整条序列离线构建 phase/state label（不能在训练窗口内估计）
- 多锚点质量评分和融合（等权归一化方式）
- 带限滤波（0.15–0.4 Hz 呼吸频带）
- peak-valley 或稳健相位恢复（Hilbert 只作对照）
- quality_report.json 和 waveform/spectrum 图
- 低质量 state label 降权（valid_mask 机制）

禁止：
- 训练窗口内独立 Hilbert 作为主相位来源
- 全图均值或全局 PCA 作为主 state label
- 单锚点无质量控制无 fallback

### R3：导丝主动运动与生理背景运动必须显式解耦

必须保留：
- `u_obs = u_wire + u_phys + epsilon` 的一阶解耦假设
- 只用 `u_phys` 做 roadmapping/background compensation
- motion loss 只在背景区域计算（dilated exclusion mask）
- compensation 方向（roadmap warp vs current-frame warp）在代码中显式定义且有单元测试

禁止：
- 直接补偿全部观测运动
- 用全局光流替代 state-conditioned motion decoupling 主方法
- 在导丝区域施加 motion consistency loss
- 不定义补偿坐标方向就上报 compensation 指标

---

## 2. 必须默认开启的稳定训练保护

### P1：Global branch stop-gradient（默认开启）

```yaml
global_state:
  detach_encoder: true   # 默认
```

原因：L_seg/L_clDice 要求 encoder 保留导丝高频边缘；
      L_state 倾向低频背景，二者梯度方向冲突。
      detach 让 encoder 只被分割损失优化。

允许：detach=false 只作为消融实验，不作为主配置。

### P2：Amplitude normalization（必须）

必须：
- amplitude label 和 prediction 除以 amp_scale 后再计算 SmoothL1
- amp_scale = P95(|u_filtered|) per sequence，保存在 state label 文件中

禁止：raw pixel amplitude 直接进入 L_amp（振幅量级 5-30px 会压制圆周损失量级 0-2）

### P3：State burn-in（必须）

窗口前 max(3, T//5) 帧的 state loss 权重设为 0.1（非 0，保留弱监督）。

原因：GRU/Mamba 从零状态冷启动，前几帧预测天然不稳定；
      若把这些帧纳入全权损失，会引入噪声梯度，
      还会使 Mamba 看起来比 GRU 差（但实际是初始化问题）。

### P4：State carry-over detach（必须）

若跨窗口传递隐状态：
```python
h_prev = h_prev.detach()  # 必须，否则 BPTT 链跨窗口爆显存
```
第一版建议：不做跨窗口 carry-over，只用 burn-in mask 即可。

### P5：soft-clDice（必须）

训练损失中的 clDice 必须是可微 soft-clDice（Shit et al., CVPR 2021）。
实现：iterative max-pooling soft skeleton，纯 torch ops，不含 numpy/skimage。

禁止在训练损失中使用：
- skimage.skeletonize
- OpenCV thinning
- Zhang-Suen thinning
- 任何 numpy skeletonization

---

## 3. 可动但必须消融的部分

### A1：Mamba vs GRU

Mamba 是条件创新，不是论文根基。
必须跑 GRU baseline，参数量尽量匹配，输入/burn-in/detach/loss 完全相同。

Mamba 的用法注意：
- 本项目 Mamba 用于时间维度（跨帧生理状态递推），不是空间维度（图像内部 patch）
- 与 VM-UNet / SegMamba 等工作有本质区别，在相关工作中必须主动说明
- 如果 Mamba 不显著优于 GRU：题目不写 Mamba，贡献点不主推 Mamba

### A2：运动模型自由度

默认第一版：translation（tx, ty，2自由度）
升级条件：translation 改善不足时再做 affine（5自由度）

### A3：Backbone

GhostEncoder 可替换，但必须仍输出多尺度特征且保留 1/8 尺度供全局分支。

### A4：局部时序融合

允许 gated temporal fusion / feature bank / short-window concat，
不能替代全局低频状态分支的功能。

---

## 4. 实验前必须通过的四个检查

### C1：数据划分检查
```
scripts/check_split_leakage.py
```
检查 patient_id / sequence_id / intervention 无交集，aug_data source 不属于 val/test。
没有通过，不得汇报正式结果。

### C2：motion convention 检查
```
scripts/test_motion_convention.py
```
用已知平移量测试：补偿后 residual 下降，反方向补偿 residual 上升。

### C3：state label 质量检查
必须生成 waveform plot / spectrum plot / quality_report.json / valid_mask。
低质量 label 不强行进入全权 L_state。

### C4：clDice 实现检查
确认可微，纯 torch ops，训练速度可接受（< 3x overhead）。

---

## 5. 关键物理约束与公式约定

运动分解一阶近似（论文中必须声明边界条件）：
```
u_obs(t, r) = u_wire(t, r) + u_phys(t, r) + epsilon_t
```
适用条件：生理位移相对视野较小；不追求真实稠密投影运动恢复。

圆周相位损失（不用 MSE，原因：0 和 2π 等价，MSE 在 wrap 点产生巨大错误梯度）：
```
L_circular = 1 - (e_pred · e_gt) = 1 - cos(phi_pred - phi_gt)
e_t = [cos(phi_t), sin(phi_t)]，F.normalize 归一化后再点积
```

全局分支输入：
```
B_t = LP_spatial(stopgrad(F_t^{1/8}))
z_t^g = GAP(proj(B_t))
h_t = StateCore(z_t^g, h_{t-1})
```

运动场：
```
u_phys(t) = G_motion(s_t) = MLP([cos,sin,amp] -> [tx,ty])
```

---

## 6. 代码结构约定

```
phys_dec_gwnet/
  models/
    ghost_encoder.py        # FROM fast-litenet，保留不动
    local_wire_branch.py    # FROM fast-litenet，轻微适配接口
    global_state_branch.py  # NEW，第一个要实现的模块
    motion_field.py         # NEW
    phys_dec_gwnet.py       # NEW，主模型
  losses/
    seg_losses.py           # FROM fast-litenet
    state_losses.py         # NEW
    motion_losses.py        # NEW
  scripts/
    build_anchor_state_labels.py  # NEW
    check_split_leakage.py        # NEW
    test_motion_convention.py     # NEW
    train_stage1.py / train_stage2.py / train_stage3.py
    evaluate.py
  configs/
    base.yaml
    stage1_seg.yaml
    stage2_state.yaml
    stage3_joint.yaml
```

网络名：PhysDec-GWNet（不是 FreqMamba-GWNet，不是 RSA-GWNet）

---

## 7. 论文写作禁区

禁止写：
- "FFT 本身是核心创新"
- "Mamba 天然优于 GRU/Transformer"
- "完整临床导航系统已部署"
- "本文恢复真实稠密生理运动场"
- "全局光流即可代表生理运动"

建议写：
- "frequency-guided task decomposition"
- "low-frequency physiological state"
- "state-conditioned physiological motion compensation"
- "first-order low-degree motion approximation"
- "ECG-free application validation"

---

## 8. AI 协作规则（Claude Code / Codex 必须遵守）

### 规则 1：每次对话先读本文件
开场白模板：
"请先读取 PROJECT_CONSTRAINTS_19_2.md，然后再开始。"

### 规则 2：任务必须切小，每次只做一件事
错误方式："帮我实现整个 PhysDec-GWNet"
正确方式："帮我实现 GlobalStateBranch，输入 [B,C,H,W] 的 1/8 特征，
          要求：(1) detach_encoder 开关，(2) 5x5 blur LP，
          (3) GRU 作为默认 state core，(4) 支持 burn-in mask"

### 规则 3：每个模块写完立刻写单元测试
必须测试：
- detach 模式下梯度是否真的到不了 encoder（检查 .grad 是否为 None）
- 输出是否满足数学约束（如 cos²+sin²≈1）
- burn-in mask 权重是否正确

### 规则 4：坐标系和补偿方向必须用伪数据验证
在任何 grid_sample 代码写完后，立刻运行单点平移测试：
施加已知平移 +dx，补偿后 residual 必须下降，反方向必须上升。

### 规则 5：不要一次性重构整个项目
新增文件，旧文件保留（fast-litenet 的代码用 alias 过渡）。
等新模块测试通过后，再逐步替换旧文件。

### 规则 6：每次对话结束输出改动摘要
格式：
- 修改/新增了哪些文件，每个文件改了什么
- 新增了哪些函数/类
- 未解决问题或需要手动确认的事项

### 规则 7：禁止的操作（即使用户要求）
- 删除 detach_encoder 机制
- 在训练损失中用 skimage.skeletonize
- 把 aug_data 加入 val/test
- 用窗口内 Hilbert 替代整序列相位标签
- 在导丝区域计算 motion consistency loss
- 不做单元测试直接上报实验结果

---

## 9. 核心创新点（正式投稿表述）

1. 频域引导的双时间尺度任务分解：高频局部分支观测导丝，低频全局分支恢复生理状态
2. 整序列锚点弱监督的 ECG-free 生理状态构建：多锚点、带限、quality control、非窗口 Hilbert
3. 导丝主动运动与生理背景运动的显式解耦：只补偿 u_phys，不抵消医生操作
4. 真实临床透视数据上的多层级验证：segmentation / localization / dynamic stability / roadmap compensation

Mamba 只有实验成立时作为第 5 条：
5. Mamba/SSM 在长窗口低频状态建模中优于 GRU/TCN，改善 jitter 与 compensation error

---

## 10. 一句话守则

守住频域任务分解、整序列锚点弱监督、主动/生理运动解耦三条红线；
默认 stop-gradient、幅值归一化、state burn-in、soft-clDice 和背景 motion loss；
Mamba 是加分项，不是没有实验也必须硬撑的故事根基。
