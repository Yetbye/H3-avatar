# minWM 五阶段 AR + DMD 流水线

> 材料来源：A100 `/mnt/data/yetbye/minWM`（只读参考资产，不属于本项目工作区）。
> 依据文件：`Wan21/scripts/training/` 下 5 个 launch 脚本、`docs/Wan_Action2V_Reproduction/` 下 5 篇 `ANALYSIS_*.md` 与 `CONSENSUS_Wan_Action2V.md`、`learning_docs/`。
> 状态日期：2026-09-25

---

## 0. 全局定位

minWM（远端 git remote 指向 `Yetbye/maxwm`）是在 **Wan2.1-T2V-1.3B** 上做的训练流水线，目标是把一个双向视频扩散模型改造成因果自回归 + 少步蒸馏的模型。它**不是** H3 项目的现有实现，而是本项目在 H3 之前的 Wan 路线历史资产，价值在于：它把「双向 → AR → 分布初始化 → 步数蒸馏」这条链完整写成了可运行的代码，且每一步的配置、初始化来源、损失函数都可查。

## 1. 五阶段总表

| # | 名称 | launch 脚本 | config | Trainer | `causal` | 关键机制 | 权重初始化来源 |
|---|---|---|---|---|---|---|---|
| 0 | Phase 1 Bidirectional SFT | `run_stage0_bidirectional_camera.sh` | `bidirectional_camera.yaml` | `camera_bidirectional_diffusion` | `false` | 双向全序列 Flow Matching | 预训练 Wan2.1 |
| 1 | Stage 1 Teacher Forcing AR | `run_stage1_ar_camera.sh` | `ar_camera_tf.yaml` | `camera_diffusion` | `true` | `teacher_forcing: true` | Phase 1 `bidirectional/model.pt` |
| 2a | Stage 2a ODE Regression | `run_stage2_causal_ode_camera.sh` | `causal_ode_camera.yaml` | `camera_ode` | `true` | 预计算 ODE 轨迹回归 | Stage 1 `ar_diffusion_tf/model.pt` |
| 2b | Stage 2b Consistency Distillation | `run_stage2_causal_cd_camera.sh` | `causal_cd_camera.yaml` | `camera_consistency_distillation` | `true` | Teacher + Consistency | Stage 1 `ar_diffusion_tf/model.pt` |
| 3 | Stage 3 DMD | `run_stage3_causal_dmd_camera.sh` | `causal_forcing_dmd_camera.yaml` | `camera_score_distillation` | Generator 因果 / Score 双向 | DMD + Self Rollout | Generator ← Stage 2a `causal_ode/model.pt`；Real/Fake Score ← Phase 1 `bidirectional/model.pt` |

全部通过统一入口 `Wan21/wan_train.py --config_path <yaml> --logdir <dir> --sp_size 4` 启动，由 `config.trainer` 字段分发到对应 Trainer。

**两个容易看错的地方**（值得单独记住）：

1. **2a 与 2b 是并行分支，不是串联。** 两者的 `generator_ckpt` 都指向 Stage 1 的 `ar_diffusion_tf/model.pt`。Stage 3 取的是 **2a 的输出**（`causal_ode/model.pt`）。
2. **Stage 3 的三个网络不是同一个结构。** `generator` 是 `is_causal=True`（因为要实际自回归生成），`real_score` 和 `fake_score` 都是 `is_causal=False`——双向模型。因为 score 网络只需要评估一段完整视频，不需要 AR。这是 DMD 非对称性的实现要点。

## 2. 逐阶段细节

### 阶段 0 — Phase 1 Bidirectional SFT

```yaml
trainer: camera_bidirectional_diffusion
causal: false
use_camera: true                 # PRoPE 相机条件
model_kwargs: {timestep_shift: 5.0, use_camera: True}
image_or_video_shape: [1, 20, 16, 60, 104]   # [B, F, C, H, W]
num_frame_per_block: 4
batch_size: 1
total_batch_size: 8              # 8 GPU 全局 batch
lr: 2.0e-06
```

- 模型用 `WanModel`（双向），全局注意力（`local_attn_size: -1`）。
- `uniform_timestep: true`。
- 数据：`dataset/Wan21/Action2V/data`（LMDB）。
- 作用：把基座适配到目标域（这里是 Action2V + 相机控制）。这一步**保留双向性**，是后面所有因果化的起点。

### 阶段 1 — Stage 1 Teacher Forcing AR

与阶段 0 的差异（原文档给的对照表）：

| 特性 | Phase 1 | Stage 1 |
|---|---|---|
| 模型 | `CameraBidirectionalDiffusion` | `CameraCausalDiffusion` |
| `causal` | `false` | `true` |
| `teacher_forcing` | `false` | `true` |
| `uniform_timestep` | `true` | `false` |
| `local_attn_size` | `-1`（全局） | `20`（局部窗口） |
| 初始化 | 预训练 Wan2.1 | Phase 1 checkpoint |

**Teacher Forcing 在扩散语境下的含义**：生成当前 block 时，模型看到的「之前 block 的 clean latent」是**真实的**，不是自己逐步去噪出来的。

代码路径：`generator_loss` 里对 clean latent 做小噪声增强（`noise_augmentation_max_timestep`，例如 50），然后把 `clean_x=clean_latent_aug` 与 `aug_t=timestep_clean_aug` 传进模型。`CausalWanModel.forward` 对每个 block：

```python
if block_idx == 0:
    context = None
else:
    if clean_x is not None and self.teacher_forcing:
        context = clean_x[:, :block_idx * block_size]   # 真实 latent
        if aug_t is not None:
            context = self.add_noise(context, noise, aug_t)
    else:
        context = self.generated_latents[:, :block_idx * block_size]  # 自己生成的
```

→ 这就是标准 TF 的暴露偏差来源：训练看真值，推理看自己。阶段 2a / 2b 都是在处理这个问题。

### 阶段 2a — Causal ODE Regression

```yaml
trainer: camera_ode
generator_ckpt: ./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt
denoising_step_list: [1000, 750, 500, 250]
warp_denoising_step: true
data_path: ./dataset/Wan21/Action2V/ode_lmdb     # 预计算 ODE 轨迹
lr: 2.0e-06
beta1: 0.9                                        # 注意：与 Stage 1 不同
ema_weight: 0.99
ema_start_step: 200
```

- 动机：Stage 1 的 TF 有暴露偏差，推理时上下文是模型自己的输出、有误差累积。
- 做法：用 Stage 1 的 Teacher 模型对每个样本跑完整去噪过程，**预计算**从噪声到 clean 的 ODE 轨迹，训练模型直接回归轨迹中的中间状态，让它学会「单步预测」。
- 代价：需要一整套预计算数据集（`ode_lmdb`），数据准备成本高；且只覆盖固定几个 timestep（1000/750/500/250），泛化受限。

### 阶段 2b — Causal Consistency Distillation

```yaml
trainer: camera_consistency_distillation
generator_ckpt: ./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt
discrete_cd_N: 50
guidance_scale: 3.0
lr: 2.0e-06
lr_critic: 4.0e-07
ema_weight: 0.99
ema_start_step: 200
data_path: ./dataset/Wan21/Action2V/data          # 普通 latent 数据，不需预计算
teacher_forcing: True
causal: True
```

- 三网络：`student`（因果，可训）+ `teacher`（因果，`requires_grad_(False)`，提供 CFG + Euler 步进的参考轨迹）+ EMA。
- 动机：绕开 2a 的预计算成本——不依赖预计算轨迹，实时生成训练数据。
- 目标：学习一致性函数，使任意 timestep 的噪声状态都能直接预测 clean latent，从而支持单步生成。

### 阶段 3 — Asymmetric DMD with Self Rollout

```yaml
trainer: camera_score_distillation
distribution_loss: dmd
generator_ckpt: ./ckpts/Wan21/Action2V/causal_ode/model.pt     # ← 来自 2a
real_ckpt:      ./ckpts/Wan21/Action2V/bidirectional/model.pt  # ← 来自 Phase 1
fake_ckpt:      ./ckpts/Wan21/Action2V/bidirectional/model.pt  # ← 来自 Phase 1
denoising_step_list: [1000, 750, 500, 250]
dfake_gen_update_ratio: 5          # Generator 更新 5 次，Critic 更新 1 次
lr: 2.0e-06
lr_critic: 4.0e-07
```

三网络职责：

| 网络 | 结构 | 梯度 | 作用 |
|---|---|---|---|
| Generator | 因果 | 可训 | 实际生成假样本 |
| Real Score | 双向 | 冻结 | 评估真实数据分布 |
| Fake Score | 双向 | 可训 | 评估生成数据分布 |

**Self Rollout**（`_run_generator`）：随机采样 `num_generated_blocks`，从 `initial_latent`（第一帧）出发自回归 rollout 出整段假样本 `pred_image: [B, F, C, H, W]`，再由 DMD loss 提供梯度。

**Generator loss 的本质**：DMD 的 KL 梯度等价于把 `(real_score 预测 − fake_score 预测)` 当作梯度信号直接作用在 `x_gen` 上。代码里的写法是先构造 `target = x_gen.detach() - grad`，再算 `0.5 * ‖x_gen − target‖²`，于是 `∇_θL = grad · ∇_θ x_gen`。梯度只通过 `x_gen` 传播。

**Critic loss**：`torch.no_grad()` 下 rollout 出假样本 → 加噪 → `fake_score` 预测 → 用 Flow Matching loss 训练 `fake_score`，让它学会区分真实与生成样本。

**交替训练**：

```python
while True:
    batch = next(self.dataloader)
    for _ in range(self.config.dfake_gen_update_ratio):   # 默认 5
        self.fwdbwd_one_step(batch, train_generator=True)
        self.generator_optimizer.step()
    self.fwdbwd_one_step(batch, train_generator=False)
    self.critic_optimizer.step()
```

## 3. 权重交接图

```
预训练 Wan2.1-T2V-1.3B
      │
      ▼
[阶段0] bidirectional ──────────────┬──────────────────┐
      │                             │                  │
      ▼                             │                  │
[阶段1] ar_diffusion_tf             │                  │
      │                             │                  │
      ├──────────────┐              │                  │
      ▼              ▼              │                  │
[阶段2a] causal_ode  [阶段2b] causal_cd                │
      │                                                │
      ▼                                                ▼
[阶段3] Generator ← causal_ode          Real/Fake Score ← bidirectional
```

## 4. 本机实际状态（重要限定）

磁盘上 `/mnt/data/yetbye/minWM/ckpts/Wan21/` 只有：

```
Wan2.1-T2V-1.3B/     （基座）
Action2V/bidirectional/
Action2V/causal_cd/
```

`ar_diffusion_tf/` 与 `causal_ode/` 只在配置里被引用，**不在本机 ckpts 目录下**。

所以上面的交接图是**代码与配置定义的链**，不是本机已完整复现的链。引用 minWM 的经验时，必须说清哪些阶段是在文档/配置层面确认的，哪些是有本地权重产物佐证的。

（另：本项目 `workflow/STATUS.md` 记载 minWM 历史训练到 step 1400、测试 14/14 通过；本次核查只覆盖 `ckpts/` 目录列表，未复核该记录。）

## 5. 对本项目的可复用性边界

| 可复用 | 不可直接复用 |
|---|---|
| 五阶段的**顺序与目的**，以及每阶段「为什么要存在」 | 具体代码（Wan2.1 的 chunk 语义、PRoPE 相机条件、LMDB 数据管线） |
| Teacher Forcing 在扩散中的实现方式（`clean_x` + `aug_t` + block-wise context） | `local_attn_size: 20`、`num_frame_per_block: 4` 等超参 |
| DMD 三网络非对称结构（Generator 因果 / Score 双向） | `dfake_gen_update_ratio`、学习率等标定值 |
| 「2a/2b 二选一即可」这一经验——SolarWM 给出了把它们合并的答案 | ODE 轨迹预计算数据集的做法（H3 规模下代价过高） |

对应关系与合并方案见 `SOLARWM_H3_VS_MINWM.md`。
