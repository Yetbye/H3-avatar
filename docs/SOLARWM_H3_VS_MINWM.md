# SolarWM-H3 与 minWM 对照，及 guide 四步的映射

> 材料来源：
> - SolarWM 源码与文档：A100 `/mnt/data/yetbye/h3-interactive/third_party/SolarWM`（revision `ce1da4e`）——`README.md`、`docs/backends/minimax-h3.md`、`docs/architecture.md`、`environments/README.md`、`configs/examples/minimax_h3/*.yaml`，以及 `src/solarwm/backends/minimax_h3/` 下的配置校验与 rollout 代码。
> - minWM：见 `MINWM_STAGE_PIPELINE.md`。
> 状态日期：2026-09-25

---

## 1. 一句话结论

两者目标相同——把一个双向视频扩散模型变成少步因果 AR 模型；差别在于 **SolarWM 把 minWM 的「2a 因果 ODE + 2b 一致性蒸馏」两步合并进了 Stage1 的损失函数里**，因此少一个阶段、少一整套预计算数据集。

SolarWM `README.md` 原文：

> The staged workflow is **Stage0.5 FM → Stage1 TF-AnyFlow → Stage2 DMD via SGF**, turning a bidirectional video model into a camera-controlled few-step autoregressive model.

> （Stage1）removes the need for a separate ODE or consistency-distillation initialization before Stage2/DMD.

## 2. 阶段对照

| 目的 | minWM（Wan2.1 Action2V） | SolarWM-H3 |
|---|---|---|
| 双向后训练 | Phase 1 Bidirectional SFT | **Stage0.5 FM**，`causal_mode: bidirectional`，`objective: flow_matching` |
| 因果化 + AR | Stage 1 TF-AR，`teacher_forcing: true` | **Stage1 TF-AnyFlow**，`causal_mode: teacher_forcing`，`objective: anyflow_forward_map` |
| 分布初始化 | Stage 2a Causal ODE **或** Stage 2b Causal CD（两条并行分支） | **无独立阶段**——由 Stage1 的 AnyFlow 项承担 |
| 步数蒸馏 | Stage 3 DMD + Self Rollout | **Stage2 DMD via SGF**，`causal_mode: self_gradient_forcing`，`objective: flow_matching` |

阶段数：minWM 5 个（含并行分支），SolarWM-H3 3 个。

## 3. 配置层面对照

### 3.1 SolarWM-H3 三阶段关键字段（从配置文件直接读出）

| 字段 | Stage0.5 | Stage1 | Stage2 |
|---|---|---|---|
| `train.stage` | `stage0p5` | `stage1` | `stage2` |
| `train.causal_mode` | `bidirectional` | `teacher_forcing` | `self_gradient_forcing` |
| `train.objective` | `flow_matching` | `anyflow_forward_map` | `flow_matching` |
| `train.max_steps` | 30000 | 30000 | 4000 |
| `train.learning_rate` | — | 3.0e-05 | 2.0e-06 |
| `train.global_batch_size` | 128 | 128 | 64 |
| `train.video_timestep_shift` | 12.0 | 12.0 | 12.0 |
| `train.audio_timestep_shift` | 3.0 | 3.0 | 3.0 |
| `train.keyframe_noise_augmentation` | 0.999 | 0.999 | 0.999 |
| `train.audio_loss_weight` | **0.0** | **0.0** | **0.0** |
| `model.num_frames_per_block` | 5 | 5 | 5 |
| `model.max_prior_clean_chunks` | 5 | 5 | 5 |
| `distributed.world_size` | 256 | 256 | 256 |
| `distributed.sequence_parallel_size` | 2 | 2 | 4 |

Stage1 独有（AnyFlow 相关）：`objective_variant: v1_5`、`anyflow_gate: 0.25`、`deltatime_type: r`、`finite_difference_epsilon: 5.0`、`diffusion_ratio: 0.5`、`consistency_ratio: 0.25`。
Stage2 独有：`critic_updates_per_student: 5`、`num_denoising_steps: 4`、`student_rope_mode: sliding_local`、`score_rope_mode: native_absolute`、`audio_condition_policy: fixed_noised_silence_per_rollout`、`score_min_sigma: 0.02`、`score_max_sigma: 0.98`、`cache_mode: exit`、`per_rank_exit_step: true`。

### 3.2 权重交接（SolarWM-H3）

| 阶段 | student 初始化 | teacher | critic |
|---|---|---|---|
| Stage0.5 | 基座 `SolarWM-h3-33B-base` | — | — |
| Stage1 | `SolarWM-h3-33B-bid-stage0p5-158f`（EMA，step 10500） | — | — |
| Stage2 | `SolarWM-h3-33B-tf-stage1-158f`（EMA，step 3000） | `SolarWM-h3-33B-bid-stage0p5-158f` | `SolarWM-h3-33B-bid-stage0p5-158f` |

三个包全部 `weight_source: ema`。推荐推理权重是 `SolarWM-h3-33B-sgf-stage2-158f-fix`（step 3900 EMA），官方说明它取代了原先的 step 1200 版本，改善了细节质量。EMA 配置：`decay: 0.999`，`update_every_steps: 1`，`dtype: float32`，`sharded: true`。

对照 minWM：minWM 的 DMD 采用 Real/Fake 两个 score 网络都从**双向** Phase 1 初始化；SolarWM-H3 的 teacher 与 critic **共用同一个 Stage0.5 包**。结构上等价于「把 minWM 的 real_score 与 teacher 合成一个角色」，因为 Stage0.5 就是双向模型。

### 3.3 参数量与训练范围对照

| | minWM | SolarWM-H3 |
|---|---|---|
| 训练方式 | 全参数（`requires_grad_(True)` 直接作用于整个模型） | **LoRA**：`type: lora, target: block_qkvo_ffn, rank: 384, alpha: 384` |
| 可训参数 | 全模型 | `expected_trainable_parameters: 2075394048`，`expected_target_linear_modules: 312` |
| 相对规模 | — | 约 2.08B / 33B ≈ **6.3%** |

**这一条需要修正一个常见说法。** guide 里写「LoRA 只微调自注意力层，0.5%–6.5%」。但 SolarWM 实际发布的 H3 配置目标是 `block_qkvo_ffn`——**Q/K/V/Out + FFN，不是只有自注意力层**。6.3% 正好落在 guide 所述区间的**上界**。所以：
- 「0.5%–6.5%」这个区间与 SolarWM 的实际取值吻合，区间本身没问题；
- 但「只微调自注意力层」与官方配置不符。如果本项目要严格照 guide 只训 self-attention，需要自己把 target 收窄并重新标定，这会产生一个**与官方权重不可直接对比**的分支。

（`block_qkvo_ffn` 是否包含 cross-attention 分支，本次未逐层核实，仅从命名与 `312` 个目标线性层推断为 block 内的 qkvo + ffn。）

## 4. 为什么 H3 主线选 SolarWM 而不是移植 minWM

1. **H3 已经官方支持。** SolarWM README 的 backend 矩阵里 MiniMax-H3 是 `✓✓✓`（三个阶段全支持），`docs/backends/minimax-h3.md` 给出完整的三阶段训练、推理、checkpoint 下载与 preencode 命令。而 minWM 只有 Wan2.1/Wan2.2 路线，H3 需要自己移植。
2. **省掉一整套预计算数据集。** minWM 的 Stage2a 需要 `ode_lmdb`（用 Teacher 跑完整去噪轨迹预计算）。在 H3 的规模下（158 帧 / 768×1344 / 33B）这个成本极高。SolarWM 的 Stage1 AnyFlow 用有限差分（`deltatime_type: r`，`finite_difference_epsilon: 5.0`）在线构造一致性项，不需要预计算。
3. **有独立佐证。** LiveAct（`LIVEACT_MECHANISM.md` §5）从另一条技术路线得出同样结论：Self Forcing 需要 ODE init 训练 + 1000 步蒸馏；Neighbor Forcing 不需要，只需 300 步。两个项目互相独立地指向「不必单独做 ODE 初始化」。

## 5. 采用 SolarWM 路线的代价与风险

必须与上面的收益一起记录，否则会在执行时才发现：

1. **规模不匹配。** 官方配置 `world_size: 256`（32 节点 × 8 卡），Stage0.5/1 需要 `--set distributed.world_size=16 --set train.global_batch_size=8` 这样的降配才能跑两节点。本项目只有**单张 A100 80GB**。官方文档给出的最小降配示例仍是 2 节点，单卡能否跑通尚未验证。
2. **数据依赖。** 需要 `SolarWM-Data` releases-v1 的 latent-wds 预编码数据，以及两个 support 文件 `h3_silence_153_158_170.safetensors` 与 `encoder_contract.json`。原始 raw-WDS 需走 Dataset Access Form；另有独立测试集 v1（1,300 例 / 77.4 GB）。**这些数据本项目都还没有。**
3. **环境互斥。** SolarWM 的 H3 环境是 Python 3.10 / PyTorch 2.6.0 / CUDA 12.4 / FlashAttention 2.8.3 / PEFT 0.20.0 / Transformers 5.12.1 / Diffusers 0.40.0。本项目现有 `talkverse` profile 是 PyTorch 2.7.1+cu128 / Transformers 4.51.3 / Diffusers 0.39.0 / FlashAttention 2.8.0.post2。`environments/README.md` 明确写 **"Do not install these three stacks into one environment."**——这与本项目「唯一 Conda 环境 + 可切换 profile」的策略存在张力，切换时必须重新验证。
4. **音频未被训练（见下节）。**

## 6. 与本项目最相关的三个未解问题

### 6.1 音频条件在这条路线里是空的

证据链（三份配置 + 后端代码）：

- `train.audio_loss_weight: 0.0`——**三个阶段全部为 0**；
- Stage2 `audio_condition_policy: fixed_noised_silence_per_rollout`（`config.py:273` 的默认值）；
- 全量推理 `audio_condition_policy: fixed_noised_encoded_158f_silence_per_rollout`（`full_inference.py:578`）；
- `H3SGFInputs` 有 `audio_rows` 与 `audio_timestep` 字段——**接口位置存在**，但现有配置全部填静音。

也就是说，公开的 SolarWM-H3 权重是「视频 AR 化 + 4 步蒸馏」的产物，**它的音频分支既没有被训练，也没有被外部音频驱动**。H3 的原生音画联合生成能力在这条路线里被保留成了「无条件生成音频」或「静音条件」。

这正面回答了 guide 里的 open question 的**现状部分**：不是我们没找到方案，而是当前公开的 H3 AR 化工作都绕开了这个问题。guide 的 W5 实验 A（audio latent clamp）与实验 B（音频条件 adapter）在这个背景下需要重新定位起点——先要确认「在 silence 条件上 AR 化的 H3」还剩余多少音频条件能力，再谈注入。

### 6.2 长时一致性缺一个记忆机制

SolarWM-H3 用 `student_rope_mode: sliding_local` + `max_prior_clean_chunks: 5` 做**滑窗截断**——窗口外的历史直接丢弃。而 LiveAct 的 ConvKV Memory 提供的是「把窗口外的历史压成定长记忆」。两者是同一个问题（KV cache 无界增长）的两条不同路线。见 `LIVEACT_MECHANISM.md` §7.1。

### 6.3 单卡可行性未知

官方给出的降配示例仍以 2 节点为最小单位。单 A100 上 H3 33B + LoRA 384 的 FSDP 配置（`FULL_SHARD` / `HYBRID_SHARD`、`activation_checkpointing: true`）能否装下，目前只是「未验证」，不是「不行」。

---

## 7. guide 四步 ↔ 两框架阶段映射表

| guide 的步骤 | 这一步要解决什么 | minWM 对应 | SolarWM-H3 对应 | 差异要点 |
|---|---|---|---|---|
| **① 双向后训练** | 在目标域/条件上先做双向 Flow Matching，把基座适配过来 | Phase 1 Bidirectional SFT<br>`bidirectional_camera.yaml`<br>`causal: false` | Stage0.5 FM<br>`stage0p5-158f-lora384-sp2.yaml`<br>`causal_mode: bidirectional` | minWM 全参训练 + PRoPE 相机条件；SolarWM 是 LoRA rank 384 + 原生相机路径 |
| **② 加 causal mask 蒸馏为 AR** | 换成块状因果注意力，用 teacher forcing 学会自回归 | Stage 1 TF-AR<br>`ar_camera_tf.yaml`<br>`causal: true, teacher_forcing: true` | Stage1 TF-AnyFlow<br>`causal_mode: teacher_forcing`<br>`objective: anyflow_forward_map` | **SolarWM 在同一阶段叠加了 AnyFlow 一致性损失**，这是两框架最大的结构差异 |
| **③ 分布初始化** | 弥合训练/推理分布差（暴露偏差），让模型能在自己输出上继续跑 | 两条并行分支：<br>Stage2a Causal ODE（预计算轨迹回归）<br>Stage2b Causal CD（teacher + consistency，`discrete_cd_N: 50`） | **无独立阶段**<br>由 Stage1 的 AnyFlow 项承担：`objective_variant: v1_5`、`anyflow_gate: 0.25`、`diffusion_ratio: 0.5`、`consistency_ratio: 0.25`、`deltatime_type: r` | minWM 需预计算 `ode_lmdb` 数据集；SolarWM 用有限差分在线构造 |
| **④ 步数蒸馏** | 用分布匹配把多步压到少步 | Stage 3 DMD + Self Rollout<br>Generator(因果) + Real/Fake Score(双向)<br>`dfake_gen_update_ratio: 5`<br>lr 2e-6 / lr_critic 4e-7 | Stage2 DMD via SGF<br>`causal_mode: self_gradient_forcing`<br>`num_denoising_steps: 4`<br>`critic_updates_per_student: 5`<br>lr 2e-6 / critic 4e-7 | 更新比例（5）与两档学习率（2e-6 / 4e-7）**完全一致**，说明 DMD 这部分的超参在两个骨干上是稳定的 |

**映射表的读法**：guide 的四步与 SolarWM 的三阶段不是一一对应——②③在 SolarWM 里被合并成了一个阶段。如果按 guide 的字面步骤去找 SolarWM 的对应物，会找不到「分布初始化」这一步；正确理解是「step ③ 的职能被吸收进 Stage1」。

## 8. 尚待核实

以下条目本次**没有**验证，列出以免被当成已确认结论：

- `block_qkvo_ffn` 的具体层组成（是否含 cross-attention），仅从命名与目标模块数推断。
- SolarWM-H3 权重包的实际大小与下载可行性（HF 仓库 `junchaoh-cs/SolarWM-H3-33B` 需接受访问条款；本机与 A100 目前都无法直连 huggingface.co）。
- `SolarWM-Data` releases-v1 的获取难度与体积。
- 单 A100 上能否加载 H3 33B（哪怕只做 LoRA 推理）。
- minWM 的 `learning_docs/` 与 `tools/` 目录内容（本次只读到 `learning_docs/` 的文件名列表）。
