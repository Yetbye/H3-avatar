# MiniMax H3 实时交互项目状态

> 状态日期：2026-09-25  
> 远程工作区：A100 `/mnt/data/yetbye/h3-interactive`  
> 状态口径：区分“已有研究资产”“当前项目已实现”“外部项目已公开”

## 1. 总体状态

项目当前处于 **Stage 0：需求澄清、技术调研和基线准备**。

远程服务器已固定 MiniMax-H3、SolarWM、TalkVerse 和 SoulX-FlashHead 源码，并建立项目唯一 Conda 环境。TalkVerse profile 已通过依赖安装、核心模块导入和 CLI 烟雾测试；MiniMax-H3 Ref2VA、FlashHead、Wan2.2-TI2V-5B、TalkVerse LoRA 与 Wav2Vec2 权重已完成下载与校验，数据集和真实生成结果仍不存在。

因此，不能把现有 `minWM` 的训练成果当作 MiniMax H3 项目已经实现；它只能作为前置经验与参考资产。

## 2. 已确认的项目背景

- 目标基座是 MiniMax H3，而不是 Wan。
- 目标是从离线原生音画生成扩展到流式、可交互和长时稳定的视频系统。
- 训练原则是参数高效微调，优先只训练 self-attention LoRA。
- 数据候选是 TalkVerse。
- AR 与蒸馏方向包括双向适配、因果 attention、Teacher/Neighbor Forcing 和 DMD 步数蒸馏。
- 模型训练与实时服务基础设施解耦。
- 现有实时系统和 Agent 基础版本据称已存在于用户电脑，但当前工作区和远程服务器中未找到，尚未现场验证。

## 3. 当前已有资产

### 3.1 本地工作区

| 资产 | 状态 |
| --- | --- |
| `minimax-guide.md` | 已读取，作为原始需求来源 |
| `LiveAct.pdf` | 已完整阅读；模型图、训练流程、ConvKV 和记忆/性能表已核查 |
| `PROJECT_GUIDE.md` | 已生成，包含背景、架构判断和分阶段路线 |
| 现有实时视频系统 | 当前工作区未发现 |

### 3.2 A100 服务器

| 项目 | 状态 |
| --- | --- |
| `/mnt/data/yetbye/minWM` | 存在，Wan2.1 Action2V 路线，Git 工作树干净 |
| `/mnt/data/yetbye/h3-interactive` | 独立 Git 工作区；源码、开发文档、依赖锁、Wan2.2-TI2V-5B 权重和 TalkVerse smoke fixture 已建立；单一 Conda 环境位于 `/mnt/data/yetbye/envs/h3-interactive` |
| minWM DMD LoRA | 历史训练到 step 1400，测试 14/14 通过 |
| MiniMax H3 官方代码/权重 | 代码已固定到 `d21241f`；Ref2VA 任务族权重已下载（82/82 文件，约 135 GiB，断点续传 + SHA-256 校验）；FL2VA 未下载 |
| SolarWM / SolarWM-H3 | 代码已固定到 `ce1da4e`；权重未下载 |
| TaoMate-H3 | 未发现 |
| H3-World | 未发现 |
| SoulX-LiveAct / FlashHead | FlashHead 代码已固定到 `9bc03de`；权重已下载（Model_Lite/Pro + VAE，约 14 GiB） |
| TalkVerse 数据 | 元数据 parquet 已下载（约 937 MiB）；源视频未获取 |
| 项目 Conda 环境 | 唯一环境位于 `/mnt/data/yetbye/envs/h3-interactive`，启动名 `h3-interactive`；当前为 `talkverse` profile，已导出锁文件并完成项目 editable 安装 |
| Wan2.2-TI2V-5B | 官方 ModelScope 22 个文件下载完成，约 32 GiB；分片检查与 SHA-256 生成完成，尚未推理 |
| TalkVerse LoRA | revision `3a58ee5`，约 2.0 GiB；SHA-256 与 safetensors 头检查通过 |
| Wav2Vec2 XLSR-53 English | revision `569a623`，约 1.2 GiB；Processor 和模型本地加载通过 |

### 3.3 服务器资源

- GPU：1× NVIDIA A100 80 GB PCIe。
- 驱动：580.173.02。
- `/mnt/data`：约 3.6 TB，总剩余约 2.0 TB。
- `/mnt/data/yetbye`：约占 601 GB。
- 检查时没有属于本项目的训练或推理进程。
- 2026-09-25 最近一次检查时，A100 正由其他用户的 4 个计算进程满载使用；本项目的受保护 runner 已按预期拒绝启动。

## 4. 外部生态最新状态

以下项目已经公开；其中 MiniMax-H3、SolarWM、TalkVerse 和 FlashHead 源码已部署到当前服务器，H3 Ref2VA、FlashHead、Wav2Vec2、Wan2.2-TI2V-5B 与 TalkVerse LoRA 权重已部署；H3 FL2VA、SolarWM-H3 权重与 TalkVerse 源数据尚未部署：

| 外部项目 | 截至 2026-09-23 的公开状态 | 与本项目的关系 |
| --- | --- | --- |
| [MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3) | H3-Base 权重和推理代码已公开，完整 2K/Context-IR 工作流仍有托管部分 | 目标基座 |
| [SolarWM-H3](https://github.com/Junchao-cs/SolarWM) | Stage0.5、Stage1、Stage2 的训练、推理、权重和预编码数据已公开 | 首选训练主线 |
| [TaoMate-H3](https://github.com/TaoLiveAIGC/TaoMate-H3) | 3-step LoRA 与流式运行时已公开 | H3 分块低延迟参考 |
| [H3-World](https://github.com/Danzer1xxxxChan/H3-World) | LoRA 训练、动作注入 patch、权重已公开 | 交互控制参考 |
| [VDN-H3](https://github.com/OpenVDN/vdn-minimax-h3) | 训练、推理、权重已公开 | 长序列 attention 加速参考 |
| [FastH3](https://github.com/hao-ai-lab/FastVideo) | 4-step sparse-distilled adapter 与推理支持已公开 | 蒸馏与 VSA 参考 |
| [SoulX-LiveAct](https://github.com/Soul-AILab/SoulX-LiveAct) | 推理代码和模型已公开 | 音频驱动、Neighbor Forcing、ConvKV 参考 |
| [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) | 1.3B 模型、推理与流式 demo 已公开 | 系统接口验证模型 |

重要变化：SolarWM-H3 已经提供原需求中计划自行搭建的大部分 AR + DMD 基础能力，因此项目规划已调整为“基于 SolarWM-H3 做音频交互扩展”，而不是“从 minWM 移植 H3”。

## 5. 已完成、未完成与未验证

### 已完成

- [x] 读取原始 MiniMax H3 项目需求。
- [x] 阅读 LiveAct 论文并核对关键模型图和实验表。
- [x] 盘点 A100 服务器的项目、环境、GPU、磁盘和进程。
- [x] 识别 minWM 可复用经验与不可直接复用的代码边界。
- [x] 核查 MiniMax H3、SolarWM-H3、TaoMate-H3、H3-World、FastH3、VDN-H3 和 FlashHead 的公开状态。
- [x] 形成第一版技术路线和阶段验收标准。
- [x] 冻结首个预录音频驱动 MVP 输入输出契约。
- [x] 克隆并固定 MiniMax-H3、SolarWM、TalkVerse 和 SoulX-FlashHead revision。
- [x] 建立唯一 Conda 环境并通过 TalkVerse profile 核心导入与 CLI 烟雾测试。
- [x] 下载 Wan2.2-TI2V-5B 基础权重，检查分片并生成 SHA-256 清单。
- [x] 下载 TalkVerse LoRA 与 Wav2Vec2，固定 revision、SHA-256 并完成独立加载检查。
- [x] 完成 TalkVerse/Wan 静态兼容性预检：3 个基础分片齐全，LoRA 的 240 个目标权重全部匹配，rank 128。
- [x] 建立项目内 6 秒预录音频 smoke fixture 和 GPU 占用保护 runner。

### 未完成

- [x] 已接受并归档 MiniMax H3 Community License（2026-08-02 版），区域与负责人确认完成（`docs/LICENSE_REVIEW.md`）。
- [x] 已下载 MiniMax H3 Ref2VA 任务族权重（82/82 文件，约 135 GiB）；FL2VA 与 SolarWM-H3 权重待训练阶段前下载。
- [ ] 尚未运行 H3、SolarWM-H3 或 TaoMate-H3 基线。
- [ ] 尚未下载 TalkVerse 样本或建立数据契约。
- [x] 已定位并完成现有实时视频系统 `D:\venus\digital-human-livestream-main` 的只读接口审计；尚未完成真实模型运行验证。
- [x] 已完成现有实时系统接口审计，并在服务器实现统一异步接口、CPU mock backend 和 session manager；3/3 生命周期测试通过。
- [x] 已实现不导入旧项目的 LiveTalking bridge；fake legacy runtime 与既有回归合计 6/6 测试通过。
- [x] 已在服务器项目实现私有 HTTP 模型服务 API；临时 loopback 与 MockBackend 合计 8/8 CPU 测试通过，无残留服务进程。
- [x] 已实现服务器项目统一维护的异步 Python 客户端 SDK；完整 CPU 回归 10/10 通过。
- [x] 已完成安全 loopback 服务启动器与 SSH 隧道跨主机 Mock smoke；12/12 回归通过，清理后无残留服务或端口。
- [ ] FlashHead 真实 runtime 已实现并通过 CPU 侧 8/8 单测，权重已下载；headless 基线实验就绪，真实 GPU 出帧待 A100 空闲。
- [ ] 尚未实现 H3 音频条件、AR 流式或步数蒸馏改造。

### 无法从当前材料验证

- 用户电脑上实时系统的当前可运行状态；代码位置已确认，技术栈初步确认为 aiohttp、WebRTC/RTC push、wav2lip/MuseTalk、LLM/TTS 管线。
- Agent 基础版本的接口和部署状态。
- 团队可使用的多 GPU 训练资源。
- 更长期的产品形态仍可演进；首个验证范围已冻结为单人参考图、5–10 秒预录 WAV 和文本提示生成保留原音频的 MP4。

## 6. 当前关键判断

1. **单 A100 是开发节点，不是目标性能节点。** 公开 H3 方案通常使用 4-8 GPU；单卡可做代码、数据、量化/offload 冒烟测试和 FlashHead 联调。
2. **SolarWM-H3 是训练起点。** 它已经公开完整 H3 AR + DMD 阶段，重复移植 minWM 的收益很低。
3. **音频交互仍是最大未知项。** H3 联合生成 audio/video latent，不等于它天然支持实时外部音频驱动。
4. **先系统、后大模型。** 用 FlashHead 验证实时协议，可以避免模型训练完成后才发现服务接口不合适。
5. **先小数据实验。** TalkVerse 全量约 6300 小时，不应在条件设计未经验证时投入全量预处理。
6. **许可证是进入条件。** H3 社区许可证对地域、衍生模型和托管服务有明确限制，任何公开发布前都要单独审查。

## 7. 当前阻塞项

| 阻塞项 | 影响 | 解除条件 |
| --- | --- | --- |
| 现有实时系统适配边界尚未冻结 | 路径已定位，但还不能开始可靠的 FlashHead/H3 接入 | 完成启动基线、接口和时钟/队列语义审计 |
| 只有单 A100 | 无法按公开默认配置训练 H3 | 获得 4-8 卡节点，或先仅使用已发布权重 |
| TalkVerse/Wan 联合加载尚未验证 | 权重布局审计已通过，但当前不能宣称预录音频生成可运行 | GPU 空闲后完成受保护的真实样本推理 |
| A100 当前由其他用户任务占用 | 立即测试会争抢共享 GPU 资源 | 等现有计算进程结束后运行项目 runner |
| ~~H3 许可证尚未归档确认~~ | ~~影响下载、训练和对外服务~~ | 2026-09-25 已归档并逐条核验；对外服务前仍需单独合规复核 |

## 8. 下一里程碑

建议将下一个里程碑定义为“两个独立基线同时通过”：

### 模型基线

- 在隔离环境中运行一个 H3 或 SolarWM-H3 官方样例。
- 记录模型 revision、环境锁文件、seed、输入、输出、峰值显存和耗时。
- 确认单 A100 是可以 offload/量化运行，还是必须转移到多卡节点。

### 系统基线

- 找到现有实时视频系统。
- 用 FlashHead 1.3B 替换或新增模型后端。
- 验证音频输入、连续视频输出、会话重置和 Agent 控制接口。

只有这两个基线都通过后，才进入 H3 分块交互和预录音频驱动实验。

## 9. 成功标准

短期成功不是“下载了模型”，而是：

- 一个可重复运行的 H3 基线；
- 一个与模型无关的实时系统接口；
- 一份固定的小规模音画数据集；
- 一条被实验支持的 H3 外部音频条件方案；
- 一套同时覆盖质量、音画同步、交互延迟和长期稳定性的评估脚本。

达到以上条件，项目才从调研阶段进入可持续实现阶段。
