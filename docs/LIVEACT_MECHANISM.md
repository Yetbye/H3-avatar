# LiveAct 机制拆解与对 MiniMax H3 的映射

> 材料来源：本地 `LiveAct.pdf`（arXiv:2603.11746v2，14 页）全文提取后阅读。
> 本文件只陈述论文中可核对的内容，以及明确标注为「推断」的映射结论。
> 状态日期：2026-09-25

---

## 1. 论文要解决的问题

AR 扩散模型分歧点不在于「是否自回归」，而在于**沿 AR 链传递什么表示**。论文 Table 1 把已有方法按这个维度排列：

| 方法 | ARPP（沿链传递的表示） | 训练时 AR 分解 | KV 可复用 |
|---|---|---|---|
| Teacher Forcing | ground-truth samples | `p(x̂ᶠₜ \| x¹:ᶠ⁻¹)` | ✗ |
| Rolling / Resampling / Diffusion Forcing | noisy 或 resampled 的 ground-truth samples | `p(x̂ᶠₜ \| x̃¹:ᶠ⁻¹ₜ′)` | ✗ |
| Self Forcing | self-generated last-step samples | `p(x̂ᶠₜ \| x̂¹:ᶠ⁻¹ₜ′)` | ✓ |
| **Neighbor Forcing（本文）** | **same-step neighbor reference states** | `p(x̂ᶠₜ \| x̂¹:ᶠ⁻¹ₜ)` | ✓ |

唯一的结构差异是下标：前三种是 `t′`（条件帧与目标帧处在**不同**扩散步），Neighbor Forcing 是 `t`（**同一步**）。

论文 Figure 1 的关键观察：把 causal mask 直接加到一个预训练的非 AR 扩散模型上，Diffusion Forcing 与 Self Forcing 在**零样本**下产不出时间一致的视频；而用「上一帧在同一步的 latent」作条件，同一个非 AR 模型零样本就能生成主体一致、时间稳定的视频——即使在原非 AR 模型上有明显的 chunk 级抖动。结论：**AR 与非 AR backbone 并非天然不兼容，起决定作用的是条件表示的选择。**

## 2. 理论依据（Appendix A）

- **Proposition 1（时间邻居即 latent 邻居）**：在流形低维 + 编码器局部 Lipschitz 假设下，`‖z₀ᶠ⁺¹ − z₀ᶠ‖ ≤ L_E(L_g Δu + 2ε_r)`。即相邻帧的 clean latent 距离有界。
- **Proposition 2（同一步加噪保持邻域结构）**：`E‖zₜᶠ⁺¹ − zₜᶠ‖² = αₜ²‖z₀ᶠ⁺¹ − z₀ᶠ‖² + 2σₜ²d`。

意义：同一步的邻居 latent 差异 = 「真实差异的 αₜ 倍」+「与步相关的噪声地板」。而跨步条件会引入无法对齐的噪声语义，迫使模型去学一个本不该学的跨步对齐任务。

## 3. 训练目标与实现细节

- 条件：`cₙ = {x_ref, x¹:ⁿ⁻¹ₜ, c_audio, c_text}`，所有块共享同一个 `t`。
- 损失（Eq.4）：`L(θ) = E_{t∼U(0,1)} ‖(ε − x) − G_θ(xₜ, t, Mask)‖²`。
- 块状因果 mask：`Mask_{i,j} = 1 iff ⌊j/m⌋ ≤ ⌊i/m⌋`。
- 块大小消融（Table 5，实时约束为平均每帧 < 50 ms @ 20 FPS）：

| memory-block | current-block | cost/frame | 达标 |
|---|---|---|---|
| 6 | 6 | 52 ms | ✗ |
| 8 | 8 | 60 ms | ✗ |
| **6** | **8** | **49 ms** | **✓** |

## 4. ConvKV Memory

**动机**：Neighbor Forcing 的步一致条件带来一个重要系统后果——上一帧的 latent 在步 `t` 构造好之后，预测后续帧时**不需要重算**它的表示，KV 可以直接复用。但朴素 KV cache 随时间线性增长，无法支撑小时级生成。

**机制**：
- 用 1D 卷积压缩，压缩比 λ=5（kernel = stride = 5），每 5 个 chunk 的 KV 压成 1 个 chunk：`M = (Conv_θ(k), Conv_θ(v))`。
- 压缩后做 **RoPE reset**，把位置编码对齐到起始位置 `s`：`M = (RoPE(Conv_θ(k), f_reps), RoPE(Conv_θ(v), f_reps))`。
- **训练时**原始 KV 不能改动，所以把压缩记忆 `M` **append 到 K/V 末尾**，并同步修改 attention mask。
- **推理时**前两个 block 不压缩；从第三次迭代开始 KV 分四部分：参考图 2 chunk + 长时记忆 2 chunk + 短时记忆 2 chunk（上一迭代最后两个 chunk）+ 当前块，合计 6 chunk。
- 理论支撑：Neighbor Forcing 带来的局部平滑与步一致条件，使历史 KV **高度可压缩**——这是 ConvKV 能work的前提，不是任意 AR 模型都能挂。
- 开销：推理时间 +1.9%。

## 5. 两阶段训练配方

| 阶段 | 做什么 |
|---|---|
| Stage 1 | Neighbor Forcing AR 训练，用 300 小时视频+音频+情绪/动作标注数据，**优化 audio cross-attention**，对齐唇动、手势、表情 |
| Stage 2 | 把 ConvKV Memory 接入 DMD 的 rollout，联合优化，3 步推理设定，训练 400 步 |

**初始化来源**（这是判断能否迁移的关键）：
- self-attention 与 text&image cross-attention ← **Wan2.1**
- **audio cross-attention ← InfiniteTalk**

与 Self Forcing 的训练成本对比（Table 4）：

| 方法 | ODE Init. 训练 | DMD 蒸馏训练 |
|---|---|---|
| Self Forcing | 需要 | 1000 步 |
| Neighbor Forcing | **不需要** | **300 步** |

## 6. 效率数字

| 模型 | 吞吐 | 延迟 | GPU | TFLOPs/frame |
|---|---|---|---|---|
| InfiniteTalk† | 25 FPS | 3.20 s | 8 | 50.2 |
| Live-Avatar | 20 FPS | 2.89 s | 5 | 39.1 |
| **LiveAct** | **20 FPS** | **0.94 s** | **2** | **27.2** |

† 使用了 LightX2V 的 LoRA 加权 4 步蒸馏。分辨率 512×512 或 720×416。
另有自报口径：Bidirectional baseline 50.2 TFLOPs/frame，Live-Avatar 39.1，本文 27.2。
FP8 端到端精度 + 序列并行 + 算子融合。

---

## 7. 对 MiniMax H3 的映射

### 7.1 可直接迁移

| LiveAct 的机制 | 在 H3 / SolarWM-H3 上的对应物 | 迁移判断 |
|---|---|---|
| 步一致的邻居条件（Neighbor Forcing） | SolarWM Stage1 的 `causal_mode: teacher_forcing` 与 Stage2 的 `causal_mode: self_gradient_forcing` | 同一思想谱系。SolarWM 的 SGF 更进一步：rollout 阶段用 detached raw-KV，再并行梯度 replay |
| KV 复用 | H3 后端已有 `H3RawKVCache`，全量推理配置使用 `kv_cache="raw_before_rope_and_prope"` | 已存在，无需移植 |
| **ConvKV 定长记忆** | SolarWM 目前只用 `student_rope_mode: sliding_local` + `max_prior_clean_chunks: 5` 做**滑窗截断** | **这是 SolarWM-H3 缺的那一块**：滑窗外的历史被直接丢弃，而 ConvKV 是把它压成定长记忆保留。可作为长时一致性的明确改进项（推断，未验证） |
| 每帧延迟预算驱动的超参标定方法（< 50 ms @ 20 FPS） | 项目实时指标尚未定义 | 方法学可用，数字不可用 |
| 「先条件对齐 → 再蒸馏」两段式 | guide 的「双向后训练 → 因果蒸馏 → 分布初始化 → 步数蒸馏」 | 同构 |
| 「邻居条件免去 ODE init」这一结论 | 与 SolarWM README 的说法互相独立佐证 | 见 `SOLARWM_H3_VS_MINWM.md` §4 |

### 7.2 不能直接迁移（必须改接口）

1. **音频注入范式根本不同。**
   LiveAct 的音频是**一路独立 cross-attention 分支**（从 InfiniteTalk 初始化），主干 self-attention 和 text&image cross-attention 来自 Wan2.1。也就是说它的「音画同步」是靠一个外挂条件分支实现的，Stage 1 只训这一支。
   H3 是**原生联合生成 video + audio latent**的单流 Omni Transformer，不存在一个独立的「audio cross-attention 模块」可以单独训。
   → LiveAct 的 Stage 1 在 H3 上没有对应模块，这一步必须重新设计。这正是 guide 里「原生音画同步如何交互」这个 open question 的具体落点。

2. **块语义与压缩比需要重新标定。**
   λ=5、以及「2+2+2 = 6 chunk」这套配比是在 Wan2.1 的 chunk 语义下标定的。H3 侧 `num_frames_per_block: 5`、`max_prior_clean_chunks: 5` 是不同的划分粒度，λ 与窗口配比不能照抄。

3. **分辨率与帧数不在同一量级。**
   LiveAct 在 512×512 / 720×416；SolarWM 的 H3 公开配置是 768×1344、158 像素帧 / 47 latent 帧。27.2 TFLOPs/frame 与 0.94 s 延迟不可直接换算到 H3。

4. **领域不同。**
   LiveAct 面向「人在说话」这一窄域（HDTF 面部动态 + EMTD 全身动作），H3 是通用视频生成。LiveAct 的质量指标（Sync-C/Sync-D 唇同步）对通用视频不适用。

### 7.3 一个必须记录的交叉发现

SolarWM 的三份 H3 配置里，`audio_loss_weight` **全部为 0.0**；Stage2 的 `audio_condition_policy: fixed_noised_silence_per_rollout`，全量推理用 `fixed_noised_encoded_158f_silence_per_rollout`。

含义：SolarWM-H3 的公开配方在自 rollout 阶段喂给模型的是**固定加噪的静音 latent**，且完全不训练音频损失。它做的是「把 H3 的视频能力 AR 化 + 步数蒸馏」，音频分支既未被训练，也未被外部音频驱动。

这直接解释了为什么 guide 把「原生音画同步如何交互」列为 open question——不是我们没找到答案，而是**当前公开的 H3 AR 化工作全都没有回答这个问题**。`H3SGFInputs` 里的 `audio_rows` / `audio_timestep` 字段是注入真实音频的接口位置，但现有配置全部填静音。

（对本项目的直接后果：guide 的 W5 实验 A「audio latent clamp」与实验 B「音频条件 adapter」，起点不是「改写一个已有的音频条件机制」，而是「先确认 Silence 条件下 AR 化的 H3 还剩下多少音频条件能力」。）
