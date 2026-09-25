# MiniMax H3 实时交互音画项目指南

> 文档日期：2026-09-23  
> 项目阶段：技术选型与基线准备  
> 原始需求来源：[minimax-guide.md](./minimax-guide.md)  
> 核心参考论文：[LiveAct.pdf](./LiveAct.pdf)

## 1. 项目定义

本项目的目标是把 MiniMax H3 从离线、固定时长的原生音画生成模型，改造成可持续生成、可被用户实时控制的交互式数字人/视频系统。

目标不是简单部署 H3，也不是继续复现 Wan Action2V，而是完成三个层次的改造：

1. **模型层**：保留 H3 原生联合音视频建模能力，引入块级因果生成、交互条件和少步蒸馏。
2. **流式层**：让模型分块持续生成，控制首帧延迟、块间连续性、长期身份一致性和 KV 显存增长。
3. **系统层**：接入现有实时视频系统和 Agent 系统，使文本、动作或实时音频能够在运行中改变后续生成内容。

研究目标可概括为：

> 用 MiniMax H3 的原生音画联合生成能力，构建一个可交互、可持续、低延迟的实时数字人生成系统，并形成可发表的模型与 HCI 系统贡献。

## 2. 为什么选择 MiniMax H3

[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3) 是 33B 参数的单流 Omni Transformer。文本、视觉与音频通过各自编码器或 VAE 转成统一序列，Transformer 联合预测视频和立体声音频 latent。其公开权重支持微调，开放模型原生输出为短边 768p、4-15 秒；官方完整工作流中的 Context-IR 和 2K Regenerate 尚未完全本地开放。

它与 Wan 音频驱动视频的根本差别是：

- Wan 数字人路线通常把音频作为额外条件，通过音频 cross-attention 驱动视频生成。
- H3 在一个 Transformer 中联合生成音频与视频 latent，音画同步是基础模型能力的一部分。
- 因此，不能直接把 Wan 的音频接口平移到 H3。首先要定义“交互输入如何进入联合音画序列，以及哪些 audio latent 是条件、哪些是预测目标”。

这也是项目的核心研究问题，而不只是工程适配问题。

## 3. 相关工作如何分工

| 项目 | 可直接借鉴的内容 | 不应直接照搬的部分 |
| --- | --- | --- |
| [SolarWM-H3](https://github.com/Junchao-cs/SolarWM) | H3 的双向适配、Teacher Forcing AR、DMD、预编码、训练与推理主干 | 当前重点是相机控制世界模型，不等于音频驱动数字人 |
| [TaoMate-H3](https://github.com/TaoLiveAIGC/TaoMate-H3) | H3 的 3-step LoRA、分块发布、KV 管理与低延迟运行时 | 公开接口主要是每 5 秒 prompt 控制，不能直接证明支持实时语音驱动 |
| [H3-World](https://github.com/Danzer1xxxxChan/H3-World) | 把离散动作转为逐 latent 文本指令，并用定向 attention routing 绑定动作与未来 latent | 训练数据和控制目标是游戏相机运动，不是人脸口型、情绪和手势 |
| [SoulX-LiveAct](https://github.com/Soul-AILab/SoulX-LiveAct) | Neighbor Forcing、音频 cross-attention、DMD、ConvKV Memory、长时流式数字人评测 | 基座为 Wan2.1，并非 H3 的联合音画 Transformer |
| `/mnt/data/yetbye/minWM` | 已验证的单卡 LoRA、Causal CD、DMD 和 checkpoint 评估经验 | 旧实现针对 Wan Action2V，不应作为 H3 主代码库 |
| [FastH3/FastVideo](https://github.com/hao-ai-lab/FastVideo) | 4-step 蒸馏、VSA、推理配置和 H3 adapter 加载机制 | Adapter 不只是普通 LoRA，还包含 norm/bias delta 与 VSA gate 权重 |
| [VDN-H3](https://github.com/OpenVDN/vdn-minimax-h3) | 线性/Softmax 混合 attention、训练代码、长序列加速 | 主要解决吞吐，不自动解决 AR 交互或音频条件设计 |
| [Sol-H3](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/) | 稀疏 attention、算子融合、多卡部署性能上限 | 面向 8×B300 数据中心，不适合作为当前单 A100 起步方案 |
| [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) | 1.3B 实时数字人和流式 UI，可作为系统联调模型 | 不是 H3，也不能验证 H3 联合音画改造是否正确 |

结论：**训练主线优先采用 SolarWM-H3，而不是从 minWM 重新移植；LiveAct 用于指导音频驱动、Neighbor Forcing 和长期记忆模块。**

## 4. LiveAct 给本项目的直接启示

LiveAct 的训练分两阶段：

1. Neighbor Forcing：在同一个 diffusion step 下，用时间相邻的历史 latent 条件化当前块，避免不同噪声状态之间的语义错配。
2. ConvKV Memory + Step Distillation：在 DMD 蒸馏中联合训练固定长度的长时 KV 压缩，使流式推理的显存和延迟不随视频长度线性增长。

论文结果表明：

- Neighbor Forcing 不需要 Self Forcing 的 ODE 初始化，论文实验中蒸馏由 1000 steps 降到 300 steps。
- ConvKV 使用 1D convolution 将每 5 个旧 KV chunk 压缩为 1 个，并重置 RoPE 位置。
- 参考图像、长时记忆和短时记忆共同构成固定长度上下文。
- 论文系统在两张 H100/H200 上达到 20 FPS；单卡消费级模型的公开实现则约为 6 FPS。

对本项目的含义：

- SolarWM-H3 已提供 Teacher Forcing + DMD 的可运行主线，第一版应先复现它。
- 当基础 AR 链路稳定后，Neighbor Forcing 可作为减少训练成本和改善同 step KV 复用的研究分支。
- ConvKV 不应在第一阶段就引入；先证明短窗口流式生成正确，再解决无限长度显存问题。

## 5. 原生音画如何变成“可交互”

这是尚未解决的关键设计点。建议把交互能力分为三个递进等级。

### 5.1 文本/动作交互

每个生成块接收一条文本或离散动作指令，只影响当前及未来块。

实现参考：

- TaoMate-H3 的分块 prompt。
- H3-World 的逐 latent 指令和 directed attention routing。

这是最容易验证的 MVP，因为不需要改变音频的输入/输出角色。

### 5.2 预录音频驱动

给定一段音频，让 H3 生成与音频时间对齐的视频，同时保留或直接复用输入音轨。

优先验证两种方案：

1. **Audio latent clamp**：用 H3 AudioVAE 编码输入音频，将对应 audio latent 作为已知 token，只预测视频 latent，并屏蔽 audio reconstruction loss。
2. **Audio condition adapter**：冻结 H3 主干，在 self-attention 前后增加轻量音频条件适配器；仅当 clamp 方案不能稳定对齐时采用。

必须检查 attention mask，确保当前视频块能看到相应音频，但不能泄漏不应访问的未来视频状态。

### 5.3 实时麦克风交互

将连续音频切成带少量 look-ahead 的块，并和视频 latent block 建立固定时间映射。系统需要同时处理：

- 音频采集、VAD 和分块；
- AudioVAE 增量编码或短窗口重编码；
- 视频 AR rollout；
- 输出音轨复用、缓冲与音画时钟同步；
- 用户中断、提示词更新和状态重置。

在预录音频驱动通过之前，不进入此阶段。

## 6. 推荐模型架构

第一版研究模型建议保持最小改动：

```text
参考图像/历史视频 ── VisualVAE ─┐
实时或预录音频 ─── AudioVAE ────┼─> packed multimodal tokens
文本/动作指令 ─── H3 Encoder ────┘
                                  │
                      block-causal attention
                                  │
                     H3 Omni Transformer
                                  │
                   video latent / optional audio latent
                                  │
                        增量 VAE 解码与发布
```

训练参数默认只开放 self-attention 的 `qkv_proj` 和 `out_proj` LoRA。H3-World 的公开配置采用 rank-32，共 65.6M 参数，占 33B 主干约 0.199%；因此原始需求中“0.5%-6.5%”不应作为硬约束，应以显存、收敛和质量实验决定 rank 与目标层。

暂不训练 FFN、AdaLN、VAE 和文本编码器。只有在音频条件无法进入主干或同步质量不足时，再扩大可训练模块。

## 7. 数据方案

### 7.1 主数据

[TalkVerse](https://zhenzhiwang.github.io/talkverse/) 提供约 6300 小时、230 万个高分辨率音画同步单人视频片段，并包含姿态、视觉和音频风格标注，适合作为音频驱动数字人的主要候选数据源。

第一轮不应下载和处理全量数据。建议先构建三个固定规模的数据层：

- 100-500 条：格式、AudioVAE、VisualVAE 和时序对齐冒烟测试。
- 5k-10k 条：验证 LoRA 是否能学到口型与身份保持。
- 规模化数据：仅在小规模实验的训练曲线和生成质量证明方向有效后启用。

### 7.2 数据契约

每条样本至少包含：

- 唯一样本 ID 和来源许可信息；
- 视频、原始音频、帧率、采样率和精确时长；
- 参考图像或身份帧；
- 音频/视频 latent 的起止时间映射；
- 人物、场景、情绪、动作和音频风格描述；
- 人脸可见度、单人约束、音画同步得分和质量过滤结果。

所有训练前预编码必须记录编码器版本与参数，避免模型更新后 latent 不兼容。

## 8. 环境与工程约束

不同参考项目的依赖版本不能同时混装：

- H3-World 已验证 Python 3.10、CUDA 12.8、PyTorch 2.10.0+cu128。
- VDN-H3 使用另一套更新环境。
- FlashHead 使用 PyTorch 2.7.1+cu128。
- SolarWM 为不同 backbone 提供独立环境定义。

项目只维护一个 Conda 环境 `/mnt/data/yetbye/envs/h3-interactive`，启动名为 `h3-interactive`。按 `talkverse`、`h3-solarwm`、`flashhead` 阶段切换依赖 profile；切换前后导出独立锁文件并重跑上一阶段必须保留的 smoke。该策略减少环境数量，但不承诺互相冲突的依赖可同时存在。每个 profile 固定：

- NVIDIA 驱动与 CUDA runtime；
- Python、PyTorch、Triton；
- FlashAttention/SageAttention 或其他编译扩展的 commit 与 wheel；
- 模型、数据和编译缓存目录。

安装 FlashAttention/SageAttention 时优先使用与 Python、PyTorch、CUDA 和 CXX11 ABI 完全匹配的固定 wheel。若无兼容 wheel，再使用 `--no-build-isolation` 从固定 Git commit 构建，而不是盲目升级 pip 或依赖。

模型和数据优先使用 ModelScope 或可断点续传的镜像，但必须记录原始仓库、revision 和文件校验值。

## 9. 分阶段实现规划

### Phase 0：资产与基线冻结

目标：获得一个可复现、可比较的技术起点。

- 克隆并固定 MiniMax-H3、SolarWM、TaoMate-H3、H3-World 和 LiveAct revision。
- 阅读并接受 MiniMax H3 Community License；该许可证明确排除美国、欧盟、英国和韩国，并约束衍生模型与托管服务。
- 在项目目录内维护唯一 Conda 环境，不修改现有 `minWM` 环境。
- 下载所需的 H3 FL2VA 权重和一个已公开 adapter。
- 在可用硬件上完成一条官方短视频推理。

验收：同一配置和 seed 可重复生成；记录峰值显存、首帧延迟、总耗时和产物哈希。

### Phase 1：系统层 MVP

目标：先验证现有实时视频系统的输入、流式输出和 Agent 控制链路。

- 在用户电脑上的既有实时系统中接入 FlashHead 1.3B。
- 定义统一的模型适配接口：`start_session`、`push_audio`、`update_control`、`pull_video_chunk`、`reset`。
- 基线接口固定使用异步 session 生命周期；音频边界为 16 kHz 单声道 PCM s16le、20 ms/块，视频块基准节拍为 40 ms，并显式暴露 PTS、背压、reset epoch 和幂等关闭语义。
- FlashHead 适配分为稳定 backend 合同与上游 GPU runtime 两层；上游按约 1 秒级生成块推理，runtime 必须负责音频聚合、motion context、逐帧拆分和有界队列，不能把 Gradio 的约 3 秒 MP4 分段误称为逐帧实时接口。
- 把模型服务、音视频传输和 Agent 控制解耦。
- 旧直播项目保留在本地且不纳入 H3 仓库；服务器项目通过私有 HTTP chunk API 提供模型会话，本地系统继续负责 WebRTC/播放。服务默认只绑定 loopback 或受控私网，未经 L4 安全检查不得公开暴露。

验收：FlashHead 能通过统一接口持续输出，Agent 可在不中断会话的情况下更新控制指令。

### Phase 2：H3 分块交互基线

目标：证明 H3 可以在块边界接受新指令并保持连续。

- 先运行 SolarWM-H3 已发布 Stage2 权重，不训练。
- 对比 TaoMate-H3、SolarWM-H3 和基础 H3 的质量、延迟与显存。
- 实现文本/动作逐块控制，不做实时音频。

验收：至少连续生成 3 个块；块间身份、背景和动作连续；后续 prompt 不影响已生成内容。

### Phase 3：预录音频驱动 PoC

目标：验证 H3 原生 audio latent 能否作为时间对齐条件驱动视频。

- 实现 AudioVAE 编码与音视频 latent 时间映射测试。
- 先做 audio latent clamp，仅训练 self-attention LoRA。
- 与“外部音频 cross-attention adapter”做最小对照实验。
- 使用 TalkVerse 小样本集训练和评估。

验收：口型同步优于无音频条件基线；输入音频内容和时长不被模型擅自改写；身份与画面质量可接受。

### Phase 4：AR 与少步蒸馏

目标：把验证过的音频驱动模型转成低步数流式模型。

- 先复用 SolarWM-H3 的 Stage0.5 → Stage1 → Stage2/DMD 流程。
- 将音频条件加入每个阶段的数据和 attention contract。
- 与 LiveAct Neighbor Forcing 做对照；只有指标或训练成本明显更好时才替换 SolarWM 默认路线。
- 蒸馏目标先定为 4-step，再尝试 3-step。

验收：固定数据集上 4-step 质量达到教师模型可接受阈值；连续多块生成无明显发散。

### Phase 5：实时与长时记忆

目标：降低首块延迟、稳定每块耗时，并阻止 KV cache 无限增长。

- 引入同 step KV reuse。
- 先保留固定短窗口，再实现 ConvKV Memory。
- 对比 dense、VSA、VDN 和可用的稀疏 attention。
- 增加 FP8/BF16 混合精度、算子融合和增量 VAE 解码。

验收：运行 30 分钟显存无单调增长；块延迟稳定；身份、服装和背景长期一致。

### Phase 6：论文与产品验证

目标：形成清晰的研究贡献，而不是多个开源项目的拼装。

候选贡献：

- 原生联合音画 Transformer 的实时外部音频条件机制；
- 支持中途控制更新的音画 AR attention contract；
- 面向音画联合 token 的 Neighbor Forcing 或 KV 压缩；
- 新一代虚拟交互范式与用户研究。

验收：完成消融、性能/质量基准、长期运行测试和 HCI 用户研究方案。

## 10. 评估指标

模型质量：

- 口型同步：Sync-C、Sync-D。
- 视频质量：FVD、FID、VBench、VBench-2.0 Human Fidelity。
- 身份保持：人脸相似度和长时漂移曲线。
- 音频：输入音轨保持误差、音画事件对齐、是否出现额外语音或声音。
- 交互：指令生效延迟、控制成功率、切换后恢复稳定所需块数。

系统性能：

- 首个可播放块延迟，而不只统计 DiT 时间。
- 稳态 FPS、每块 P50/P95 延迟。
- 峰值显存、主存、KV cache 大小和长时增长率。
- 音视频播放时钟偏移与卡顿率。

## 11. 当前硬件判断

现有远程服务器只有一张 A100 80 GB PCIe。它适合：

- 阅读和修改代码；
- 数据预处理和小样本检查；
- FlashHead 级别模型的系统联调；
- 通过 offload、量化或分阶段加载进行 H3 推理冒烟测试。

它不适合直接复现公开 H3 实时指标或默认训练配置：

- SolarWM-H3 示例训练和 Stage2 推理使用 8 GPU。
- H3-World 默认 LoRA 训练使用 4 GPU。
- TaoMate-H3 公开运行时要求 4 或 8 GPU，性能数据来自 8×H20。
- LiveAct 的 20 FPS 结果来自 2×H100/H200。

因此，第一阶段应把单 A100 当作开发和验证节点；正式 H3 训练/实时性能实验需要额外多卡资源。

## 12. 暂不做的事情

- 不在目标尚未验证前下载 TalkVerse 全量数据。
- 不把旧 minWM 代码直接改造成 H3 训练框架。
- 不一开始就引入 ConvKV、稀疏 attention、量化和端到端 Agent 功能。
- 不把“能生成得比播放快”当作“可实时交互”；还必须验证首块延迟、中途控制和稳定输出。
- 不在没有许可证审查和数据来源记录的情况下对外发布模型或服务。

## 13. 最近的可执行里程碑

第一个里程碑应是：

> 在隔离环境中运行一个公开 H3/SolarWM-H3 推理样例，同时用 FlashHead 打通现有实时系统，并产出统一的性能与质量基线表。

这一步完成后，才能基于实测结果决定是优先做 H3 分块文本交互，还是直接投入预录音频驱动 PoC。
