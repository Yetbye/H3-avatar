# MiniMax H3 实时交互项目执行状态

> 更新日期：2026-09-25  
> 权威背景与完整盘点：[`../PROJECT_STATUS.md`](../PROJECT_STATUS.md)  
> 训练路线调研：远程 `docs/LIVEACT_MECHANISM.md`、`docs/MINWM_STAGE_PIPELINE.md`、`docs/SOLARWM_H3_VS_MINWM.md`（本地 `docs/` 同副本）  
> 稳定执行协议：[`h3_workflow.md`](h3_workflow.md)

## 当前阶段

**S0 — 项目初始化与预录音频基线准备。**

独立远程工作区已经固定四个参考仓库和唯一 Conda 环境；TalkVerse 依赖 profile 已完成可重复安装、核心模块导入和 CLI 烟雾测试。Wan2.2-TI2V-5B、TalkVerse LoRA 与 Wav2Vec2 已完成下载、revision 固定、SHA-256、独立加载检查和静态联合兼容性预检；项目内已有一组 6 秒 smoke fixture，但尚无联合模型加载或生成结果。`/mnt/data/yetbye/minWM` 是 Wan 路线的历史参考资产，不是 H3 项目的现有实现。

## 当前目标

下一个里程碑是让两个互相独立的基线都达到 `PASS`：

1. **模型基线**：在隔离环境中可重复运行 MiniMax H3 或 SolarWM-H3 官方推理样例。
2. **系统基线**：在现有实时视频系统中接入 FlashHead，验证统一的输入、连续输出、会话重置和 Agent 控制接口。

完成这两个基线前，不启动 H3 音频条件训练、AR 改造或 DMD 蒸馏。

## 已确认事实

- 本地工作区：`D:\venus\mm`。
- 独立远程工作区：A100 服务器 `/mnt/data/yetbye/h3-interactive`。
- 远程资源：1× NVIDIA A100 80 GB PCIe。
- H3 训练主线：优先 SolarWM-H3，避免从 minWM 重复移植。
- 实时系统联调模型：优先 FlashHead 1.3B。
- 参数高效训练首选：self-attention 的 Q/K/V/Out LoRA。
- 音频条件首个假设：AudioVAE latent clamp；失败后再评估音频 adapter。
- 数据候选：TalkVerse；条件机制验证前仅使用小样本。
- 首个 MVP 已冻结为：单人参考图 + 5–10 秒预录 WAV + 文本提示，输出保留原音频的 MP4。
- 环境策略：项目只维护 `/mnt/data/yetbye/envs/h3-interactive`（启动名 `h3-interactive`）；按阶段切换依赖 profile，并为每个 profile 保留锁文件与回归测试。
- 2026-09-25 环境已从项目内旧前缀迁移到上述公共环境目录；短名称激活、核心依赖导入和 12/12 CPU 回归通过，旧前缀已由 Conda 移除。
- 项目已增加 `pyproject.toml` 和根 `requirements.txt`，并以 editable 方式安装到唯一环境；可从任意目录使用 `h3-interactive-server` 启动。安装后 12/12 CPU 回归与临时 `/healthz` smoke 通过。
- FlashHead 上游接口审计和无 GPU backend 合同已完成：20 ms PCM、sequence/PTS、40 ms 帧、timeout、reset、close 与显式未映射控制均有 fake-runtime 测试；完整 CPU 回归 15/15 通过。
- FlashHead 真实 runtime 已实现（远程 `src/h3_interactive/backends/flashhead_runtime.py`）：单 GPU worker 线程串行化上游 chunk 状态（lite 24 帧/0.96 s 音频、pro 28 帧/1.12 s）、8 s 滚动音频窗口、运动帧 latent 携带、EOS 静音补齐、reset 清窗口且 PTS 归零、JPEG 帧输出；无 GPU 时干净失败且错误经 pull 上抛。`h3-interactive-server --backend flashhead` 已通过 `FLASHHEAD_*` 环境变量接入真实工厂。
- FlashHead headless 基线实验已就绪（远程 `experiments/system-interface/flashhead-headless-001/`）：合同（预注册口径：8 s 静音冷启动、chunk 突发到达、播放调度侧 p95、lite 专属 RTF 门限）、runner（双重 GPU 护栏、实时节奏推送、播放排程仿真、MP4+stills+metrics.json+SHA-256）与 CPU 侧单测（chunk 布局、EOS 补齐、无 GPU 干净失败、close 幂等）。GPU 运行待 A100 空闲。
- FlashHead 权重（Model_Lite/Model_Pro/VAE_LTX/VAE_Wan）与 wav2vec2-base-960h 已下载完成；TalkVerse 元数据下载完成；H3 Ref2VA 权重已下载完成（82/82 文件、约 135 GiB，HF 镜像，中断后断点续传 + SHA-256 校验）。
- 已固定代码 revision：MiniMax-H3 `d21241f`、SolarWM `ce1da4e`、TalkVerse `3607ff2`、SoulX-FlashHead `9bc03de`。
- 当前 `talkverse` profile：Python 3.10.21、PyTorch 2.7.1+cu128、Transformers 4.51.3、Diffusers 0.39.0、FlashAttention 2.8.0.post2；CUDA 可见且 TalkVerse `generate.py --help` 通过。
- Wan2.2-TI2V-5B 基础权重已从官方 ModelScope 下载完成：22 个上游文件，目录约 32 GiB，safetensors 分片齐全并生成 SHA-256 清单；尚未进行模型加载或推理。
- TalkVerse LoRA 已从匿名镜像固定到 revision `3a58ee5`，safetensors 头读取成功（640 tensors）；Wav2Vec2 固定到 `569a623`，Processor 与 315M 参数模型本地加载成功。
- TalkVerse 静态预检为 `PASS_WITH_WARNINGS`：基础模型索引含 825 个张量、3 个分片齐全；LoRA 的 240 个目标权重全部存在，rank 为 128。上游单条推理分支漏传 `lora_ckpt`，首次生成固定使用 batch 分支。
- 项目自有 smoke fixture 位于远程 `data/raw/talkverse-smoke/`：512×512 参考图、6 秒/16 kHz/单声道 WAV 和单条 batch 清单。
- 2026-09-25 检查时 A100 正由其他用户的 4 个进程占用；安全运行脚本已验证会在存在任意 GPU 计算进程时退出（退出码 2），未初始化模型或增加显存占用。
- 工作区已包含开发计划、架构约束、实验合同模板、边界检查脚本和独立 Git 仓库；2026-09-25 已创建首个提交并推送到 GitHub（Yetbye/H3-avatar）。
- SolarWM-H3 训练配方已核到配置与代码层：三阶段为 Stage0.5 FM（`causal_mode: bidirectional`）→ Stage1 TF-AnyFlow（`causal_mode: teacher_forcing`, `objective: anyflow_forward_map`）→ Stage2 DMD via SGF（`causal_mode: self_gradient_forcing`, `num_denoising_steps: 4`）；`max_steps` 30000/30000/4000；官方配置 `world_size: 256`。
- SolarWM-H3 的 LoRA 目标是 `block_qkvo_ffn`（qkvo + FFN，含 FFN），rank=alpha=384，312 个目标线性层，2,075,394,048 可训参数，约占 33B 的 6.3%。**这与 guide 的「只微调自注意力层」不符**，需在正式启动训练前明确取舍。
- SolarWM-H3 三阶段的 `audio_loss_weight` 全部为 0.0；Stage2 的 `audio_condition_policy` 默认为 `fixed_noised_silence_per_rollout`，全量推理用 `fixed_noised_encoded_158f_silence_per_rollout`。即公开的 H3 AR 化权重**未训练音频分支、且以静音为音频条件**。`H3SGFInputs` 的 `audio_rows`/`audio_timestep` 是注入真实音频的接口位置。
- SolarWM 官方 H3 环境（Py3.10 / PyTorch 2.6.0 / CUDA 12.4 / FA 2.8.3 / PEFT 0.20.0 / Transformers 5.12.1 / Diffusers 0.40.0）与当前 `talkverse` profile（PyTorch 2.7.1+cu128 / Transformers 4.51.3 / Diffusers 0.39.0）冲突，`environments/README.md` 明确要求不要装进同一环境。
- minWM 本机 `ckpts/Wan21/Action2V/` 只存在 `bidirectional/` 与 `causal_cd/`；配置中引用的 `ar_diffusion_tf/` 与 `causal_ode/` 无本地产物。minWM 的 2a（ODE）与 2b（CD）是从 Stage1 分叉的两条并行分支，Stage3 的 Generator 取 2a 输出。

## 当前未完成

- [x] 确认 H3 Community License 的适用区域（服务器/开发人员/服务区域均在中国大陆，2026-09-25 负责人确认）。
- [x] 取得并归档 H3 Community License 原文，逐条核验条款（远程 `docs/licenses/` 与 `docs/LICENSE_REVIEW.md`）。
- [x] TalkVerse HF 门控已由用户接受（2026-09-25），元数据下载可进行。
- [x] 定位并完成现有实时视频系统的只读接口审计；真实运行基线和模型 bridge 仍待完成。
- [x] 在服务器主工作区实现模型无关的异步接口、mock backend 与 session manager，并通过 CPU 生命周期测试。
- [x] 实现 LiveTalking bridge，并用 fake legacy runtime 验证 PCM 转换、控制映射、视频 PTS、reset/close 和超时语义。
- [x] 在服务器项目中实现私有 HTTP 模型服务 API，并用临时 loopback + MockBackend 完成 8/8 CPU 回归测试。
- [x] 实现异步 Python 客户端 SDK；自动管理音频 sequence/PTS、控制 revision 和 reset，同完整回归 10/10 通过。
- [x] 实现安全服务启动器并完成本地 SDK 经 SSH 隧道访问服务器 MockBackend 的跨主机 smoke；清理后无残留端口或进程。
- [x] 建立项目唯一 Conda 环境并固定 TalkVerse profile 依赖锁文件。
- [x] 克隆并固定 MiniMax-H3、SolarWM、TalkVerse 和 SoulX-FlashHead 代码 revision。
- [x] 下载 Wan2.2-TI2V-5B 基础权重并生成本地 SHA-256 清单。
- [x] 下载 TalkVerse LoRA 与 Wav2Vec2 权重，固定 revision、校验值并完成独立加载检查。
- [ ] 运行 H3/SolarWM-H3 推理 smoke test。
- [ ] 运行 FlashHead 实时服务 smoke test。
- [ ] 建立小规模 TalkVerse 数据契约与样本集。
- [ ] 设计并运行预录音频驱动对照实验。

## 当前阻塞项

| 阻塞项 | 影响 | 解除条件 |
| --- | --- | --- |
| 现有实时系统尚未完成真实模型基线 | 统一协议、bridge、API、SDK 和 SSH 隧道均已通过，runtime 与 headless runner 已实现并通过 CPU 侧验证，但尚无 FlashHead 实际出帧证据 | GPU 空闲后运行 `experiments/system-interface/flashhead-headless-001/run.sh`（双重 GPU 护栏已内置） |
| TalkVerse/Wan 联合加载尚未验证 | 静态权重布局已兼容，但还不能宣称模型可生成 | 等 A100 无其他计算进程后运行受保护的真实样本 smoke |
| A100 正被其他用户进程占用 | 当前启动模型会争抢显存和算力 | GPU 空闲后按队列运行：FlashHead headless 基线 → TalkVerse smoke |
| 只有单张 A100 | 无法照搬公开多卡训练配置 | 先用公开权重做推理，训练前确认多卡资源或单卡方案 |
| H3 许可区域未确认 | ~~阻塞 H3 与 SolarWM-H3 权重下载、推理与衍生训练~~ | 2026-09-25 已确认：区域均在中国大陆，负责人已接受协议；H3 权重下载已启动 |
| H3 §V.3 隔离要求未落实 | 用 H3 产物改进其他模型违约；本项目同时持有 Wan 基线 | 把两条线的隔离措施写进 `workflow/h3_workflow.md` |
| H3 §IV.2 / §V.5 / A.12 未落实 | 影响未来对外服务与直播产品合规 | 设计阶段纳入 UI 署名、机器生成披露、举报与累犯处置 |

## 最近执行队列

| 优先级 | 工作项 | 当前状态 | PASS 证据 |
| --- | --- | --- | --- |
| P0 | 冻结 MVP 输入输出契约 | `PASS` | 远程 `docs/MVP_CONTRACT.md` |
| P0 | 现有实时系统接口审计 | `PASS` | `workflow/LIVESTREAM_INTERFACE_AUDIT.md`；待独立运行基线 |
| P0 | 统一接口与 CPU mock 生命周期 | `PASS` | 远程 `experiments/system-interface/mock-lifecycle-001/`，3/3 tests PASS |
| P0 | LiveTalking bridge 合同 | `PASS` | 远程 `experiments/system-interface/livetalking-bridge-001/`，合计 6/6 tests PASS |
| P0 | 私有 HTTP 模型服务 API | `PASS` | 远程 `experiments/system-interface/http-api-001/`，合计 8/8 tests PASS |
| P0 | Python 客户端 SDK | `PASS` | 远程 `experiments/system-interface/client-sdk-001/`，合计 10/10 tests PASS |
| P0 | SSH 隧道跨主机 Mock smoke | `PASS` | 远程 `experiments/system-interface/ssh-tunnel-smoke-001/`，12/12 tests + 跨主机生命周期 |
| P1 | H3/SolarWM-H3 基线 | `DRAFT` | manifest、完整日志、输出音画、显存与耗时 |
| P1 | FlashHead 系统基线 | `PARTIAL` | backend 合同与 CLI 安全拒绝已通过；真实 runtime + headless 实验已实现并通过 CPU 侧验证（`flashhead-headless-001/`）；待 GPU 空闲后运行出帧（连续会话录像、接口日志、延迟统计） |
| P2 | TalkVerse 环境与 CLI 基线 | `PASS` | profile 锁文件、核心导入、`generate.py --help` |
| P2 | Wan2.2-TI2V-5B 基础权重 | `PASS` | 22 个文件、分片索引检查、SHA-256 清单 |
| P2 | TalkVerse LoRA 与 Wav2Vec2 | `PASS` | 固定 revision、SHA-256、safetensors/本地加载检查 |
| P2 | TalkVerse 小样本与离线生成 | `PARTIAL` | fixture 与静态预检已通过；待输出 MP4、运行日志、显存与耗时 |
| P3 | Audio latent clamp PoC | `DRAFT` | 与无音频条件基线的预注册对照结果 |
| P3 | 训练路线文档（LiveAct / minWM / SolarWM-H3 对照） | `PASS` | 远程 `docs/` 三份文档，两侧 SHA-256 一致（`d65d9519…`、`65923bcb…`、`6bfd51c5…`） |

## 状态更新规则

- 本文件只保存当前快照，不保存逐日流水账。
- 只有实际命令、日志和产物共同验证后，才可把工作项标成 `PASS`。
- 外部仓库“已经公开”不等于当前服务器“已经部署”。
- 推理跑通不等于实时达标；离线音画同步不等于长时交互稳定。
- 状态变化后同步更新本文件；架构或阶段规划变化时同步更新 `../PROJECT_GUIDE.md` 与 `../PROJECT_STATUS.md`。
