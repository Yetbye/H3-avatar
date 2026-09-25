# MiniMax H3 实时交互项目完整开发流程

## 1. 最终目标

在不修改服务器上其他项目的前提下，建立一套可复现、可评估的 MiniMax H3 实时交互系统：

- 保留 H3 原生联合音画生成能力；
- 支持文本、动作和外部音频改变后续生成；
- 支持分块连续生成、低步数推理和长时记忆；
- 通过统一服务接口接入现有实时视频系统和 Agent；
- 所有结论可追溯到代码、数据、命令、日志、指标和产物。

首个 MVP 默认是“预录音频驱动的单人数字人视频”，但必须在 W0 阶段由项目负责人确认。如果产品目标改为世界模型或通用互动视频，应重新冻结接口和评估指标。

## 2. 不变量

1. 唯一可写远程根目录是 `/mnt/data/yetbye/h3-interactive`。
2. `minWM` 只作为 Wan AR/DMD 经验参考，不复制其环境、日志或 checkpoint 到本项目路径之外。
3. 外部仓库固定 commit；本地适配通过 `src/`、配置覆盖或可审计 patch 实现。
4. 只维护工作区内一个 Conda 环境；依赖按阶段切换，每个可工作状态必须保存锁文件并完成回归 smoke。
5. 未通过模型基线和系统基线前，不开始音频条件训练。
6. 未通过小数据 PoC 前，不处理 TalkVerse 全量数据。
7. 未通过预录音频驱动前，不进入实时麦克风链路。
8. 单 A100 不默认承担公开方案的完整多卡训练。

## 3. 开发状态机

每个工作项只能处于：

```text
DRAFT -> READY -> RUNNING -> PASS | PARTIAL_PASS | FAIL | BLOCKED
```

只有预注册的必要 Gate 全部通过才可标记 `PASS`。修复失败实验必须使用新 `run_id`，不得覆盖原始证据。

## 4. 阶段规划

### W0 — 项目治理和需求冻结

目标：让团队对产品、许可证和资源边界达成一致。

任务：

- 确认 MVP 输入、输出、目标用户和最大可接受延迟。
- 归档 MiniMax H3 Community License 的适用性结论。
- 确认现有实时视频系统与 Agent 的代码位置、协议和负责人。
- 确认可用 GPU：单 A100 用于开发，是否存在 4-8 卡训练节点。
- 固定仓库命名、实验状态、指标定义和发布审批人。

产物：

- `docs/MVP_CONTRACT.md`
- `docs/LICENSE_REVIEW.md`
- `docs/RESOURCE_BUDGET.md`

退出条件：产品定义、许可和资源约束不再阻塞第一个基线。

### W1 — 可复现环境与第三方源码

目标：建立互不污染的基础环境。

任务：

- 将 MiniMax H3、SolarWM-H3、FlashHead 克隆到 `third_party/`。
- 记录每个仓库的 remote、commit、license 和补丁状态。
- 建立 `.conda/envs/h3-interactive` 单一环境，并为 TalkVerse、H3/SolarWM-H3、FlashHead 维护阶段锁文件。
- 固定 Python、PyTorch、CUDA、FlashAttention 和系统依赖。
- 将 Hugging Face/ModelScope 缓存显式指向本工作区内 `.cache/`。

产物：

- `third_party/MANIFEST.md`
- `configs/environments/*.lock`
- `scripts/bootstrap_*.sh`

退出条件：单一环境的基础状态可复现，首个 TalkVerse 阶段配置完成 import smoke，且所有缓存与输出都留在工作区。

### W2 — H3/SolarWM-H3 模型基线

目标：用固定输入得到可重复的 H3 音画输出。

任务：

- 下载被许可使用的 H3 权重并记录 revision/hash。
- 运行官方最小推理样例。
- 运行 SolarWM-H3 已发布低步数/AR 权重。
- 记录首帧时间、总耗时、峰值显存、分辨率、帧率、采样率和输出时长。
- 判断单 A100 需要量化、CPU offload，还是只能转移到多卡节点。

对照：基础 H3、SolarWM-H3、可选的 TaoMate-H3/FastH3 只在同一输入和硬件口径下比较。

退出条件：至少一个 H3 系模型在固定 revision 下可重复生成可解码的音视频，并有完整 manifest。

### W3 — FlashHead 实时系统基线

目标：先验证与大模型无关的实时系统接口。

任务：

- 在单一 Conda 环境切换到 FlashHead 阶段配置后运行 FlashHead 1.3B。
- 定义统一请求：会话、人物参考、音频块、文本/动作控制和重置。
- 定义统一响应：视频帧、音频帧、时间戳、队列深度和错误状态。
- 测量采集、排队、推理、编码、传输的分段延迟。
- 验证取消、重置、背压、断连恢复和 Agent 指令更新。

产物：

- `docs/SERVICE_CONTRACT.md`
- FlashHead adapter 和端到端 smoke 测试。

退出条件：连续会话可运行，服务接口不依赖 FlashHead 的内部张量格式。

### W4 — TalkVerse 小样本数据管线

目标：建立可靠的音画时间轴和最小训练集。

任务：

- 先选择 20-100 个合规片段，不下载全量数据。
- 生成人物/视频级互斥的 train/dev/test manifest。
- 固定 FPS、音频采样率、裁剪规则、时长和坏样本处理。
- 验证视频帧、音频 sample 与 H3 audio/video latent 的时间映射。
- 缓存预编码结果，同时保留原始文件 hash 和编码器 revision。

退出条件：随机抽样和自动检查均证明无人物泄漏、无明显音画错位、无静默坏样本。

### W5 — 外部音频条件 PoC

目标：证明外部音频对视频生成有真实因果作用。

实验 A：Audio latent clamp。

- 用 H3 AudioVAE 编码输入音频。
- 将对应 audio latent 作为已知 token。
- 只预测视频 latent，并屏蔽 audio reconstruction loss。
- 审计 attention mask，禁止未来视频信息泄漏。

实验 B：音频条件 adapter。

- 仅当实验 A 无法获得稳定对齐时启用。
- 冻结主干，训练轻量音频条件模块。

共同对照：正常音频、静音、shuffle、时间反转和无音频条件。

退出条件：口型或音画事件对齐优于无音频基线，输入音轨内容和时长未被擅自改写，身份与画质处于预注册护栏内。

### W6 — AR 与少步 DMD

目标：将通过 W5 的条件机制迁移到流式少步生成。

任务：

- 复用 SolarWM-H3 的双向适配、Teacher Forcing AR 和 DMD 阶段。
- 在所有阶段保持一致的音频条件、时间块和 attention contract。
- 首轮 LoRA 仅作用于 self-attention Q/K/V/Out。
- 对比双向 teacher、Teacher Forcing AR、DMD student。
- 审计生成器/critic checkpoint、初始化 hash 和可训练参数比例。

退出条件：少步模型获得明确速度收益，同时通过质量、同步、连续性和身份护栏。

### W7 — 实时音频、记忆与 Agent

目标：从预录音频扩展为连续交互。

任务：

- 加入麦克风采集、VAD、重采样、音频分块和有限 look-ahead。
- 建立 audio/video block 的单调时间戳映射。
- 增加抖动缓冲、背压、取消和会话重置。
- 比较滑动窗口与 ConvKV/压缩记忆。
- 接入 Agent 文本、动作和情绪控制，并规定冲突优先级。
- 进行 1、5、15、30 分钟长时稳定性测试。

退出条件：没有不可接受的累计延迟、显存增长、身份漂移、背景跳变或音频断裂。

### W8 — 冻结、论文与发布

目标：形成可复现结果和可审计发布物。

任务：

- 冻结代码 commit、环境锁、权重 revision、数据 manifest 和评测集。
- 完成消融：条件方式、LoRA 范围、AR 策略、蒸馏步数和记忆机制。
- 完成盲测、失败案例分类和资源成本表。
- 复核许可证、数据隐私、肖像权和内容安全。
- 生成模型卡、数据卡、部署指南和限制说明。

退出条件：论文、演示和服务中的每个数字都能回溯到不可覆盖的实验目录。

## 5. 标准实验流程

每个 L2-L4 实验遵循：

```text
提出问题
  -> 复制 EXPERIMENT_TEMPLATE.md
  -> 冻结单一变量、数据、模型与 PASS 阈值
  -> workspace/env/data/model preflight
  -> 最小样本 smoke
  -> 正式运行
  -> 指标与产物审计
  -> Builder review
  -> Red-team review
  -> PASS/PARTIAL_PASS/FAIL/BLOCKED
```

推荐目录：

```text
experiments/<campaign>/<run_id>/
├── experiment.md
├── command.sh
├── manifest.json
├── logs/
├── metrics.json
├── validation.md
├── summary.md
├── artifacts/
└── run.done
```

`run.done` 只有在退出码、必需产物和 Gate 审计通过后才能创建。

## 6. 统一评估

质量：画面质量、身份、背景、动作自然度、提示遵循。  
音画：口型、事件对齐、输入音轨保持误差、额外声音。  
连续性：块边界闪烁、身份漂移、背景跳变、长期误差累计。  
性能：模型加载、首帧、首块、稳态块延迟、FPS、峰值显存。  
系统：队列深度、丢帧、背压、取消、重置、断连恢复。  
成本：GPU 型号/数量、GPU 小时、存储、预处理时间和训练总成本。

所有比较必须固定输入、seed、分辨率、时长、采样参数和硬件；无法固定的项必须显式披露。

## 7. 近期执行顺序

1. 完成 W0 的 MVP、许可证和资源确认。
2. 完成 W1 的三个隔离环境与源码 manifest。
3. 并行完成 W2 模型基线与 W3 系统基线。
4. 两个基线均通过后进入 W4 小数据管线。
5. W5 先验证 Audio latent clamp，再决定是否开发 adapter。
6. 只有 W5 通过后进入 W6-W7。

当前不应启动：TalkVerse 全量处理、H3 大规模训练、实时麦克风开发或外部发布。
