# H3-Avatar

基于 MiniMax H3 的**持续、可控、实时**音视频生成系统：把原生联合音画 Transformer 改造成可交互的数字人，并在统一服务协议上完成实时系统与 Agent 集成。

> 研究项目：当前阶段为基线准备（S0）。执行状态见 [`workflow/STATUS.md`](workflow/STATUS.md)，完整资产盘点见 [`PROJECT_STATUS.md`](PROJECT_STATUS.md)，路线图见 [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)。

## 1. 项目目标

MiniMax H3 原生输出为 4–15 秒的离线音画生成。本项目在其上完成三层改造：

| 层 | 改造内容 |
| --- | --- |
| 模型层 | 保留 H3 原生联合音视频建模，引入块级因果生成、交互条件与少步蒸馏（主线：[SolarWM-H3](https://github.com/Junchao-cs/SolarWM) 的 Bidirectional FM → TF-AnyFlow → DMD via SGF） |
| 流式层 | 分块持续生成：控制首帧延迟、块间连续性与身份一致性，防止 KV 显存随时长线性增长（参考：[SoulX-LiveAct](https://github.com/Soul-AILab/SoulX-LiveAct) 的 Neighbor Forcing 与 ConvKV Memory） |
| 系统层 | 统一会话协议对接实时视频系统与 Agent：文本/动作/实时音频可在运行中改变后续生成内容 |

核心研究问题：**交互输入如何进入联合音画序列——哪些 audio latent 是条件、哪些是预测目标**，而非简单的工程适配。

## 2. 当前状态（2026-09-25）

- **系统层链路已全通（mock 全链路）**：统一协议、session 管理、私有 HTTP API、异步 SDK、SSH 隧道跨主机冒烟，12+ 项 CPU 回归全部 PASS。
- **FlashHead 真实 runtime 已实现**：单 GPU worker、上游 chunk 状态机、8 s 音频窗口、EOS 补齐、reset/PTS 语义；headless 基线实验（60 s 连续 + 25 FPS + reset <2 s 门槛）就绪，待 A100 计算空闲运行。
- **许可证审查完成**：H3 Community License 区域（服务器/开发/服务均在中国大陆）与负责人确认两项已通过；H3 Ref2VA 权重下载完成（82/82 文件）。
- 两条基线（H3/SolarWM 模型基线、FlashHead 系统基线）PASS 前，不启动音频条件训练、AR 改造或 DMD 蒸馏。

## 3. 系统架构

```text
前端 / 直播系统 / Agent
    │  HTTP API（loopback 或 SSH 隧道） / Python SDK
    ▼
h3_interactive 会话与服务层
    │  start / push_audio / update_control / pull_video_chunk / reset / close
    │  20 ms PCM s16le 输入 · 40 ms 视频帧节拍 · 显式 sequence/PTS/epoch
    ▼
统一 InteractiveBackend
    ├── MockBackend            （协议验证，无模型）
    ├── LiveTalkingBridge      （旧直播系统只读桥接）
    └── FlashHeadBackend ──── SoulXFlashHeadRuntime（真实 GPU runtime）
                                  │
                                  ▼
                    third_party/SoulX-FlashHead（1.3B，未修改）
```

## 4. 仓库结构

```text
src/h3_interactive/    统一协议、会话管理、HTTP API、SDK、backend 适配与 GPU runtime
experiments/           实验合同（experiment.md）+ 可复现 runner + 本地证据（logs/artifacts 不入库）
scripts/               环境安装、预检、安全启动器、冒烟入口
configs/               依赖 profile 与锁文件
docs/                  训练路线调研（LiveAct / minWM / SolarWM-H3）、许可证审查与协议原文
tests/                 服务层回归测试
workflow/              执行状态（STATUS.md）与稳定执行协议（h3_workflow.md）
PROJECT_GUIDE.md       架构与路线图
PROJECT_STATUS.md      已验证资产/状态快照
```

**不入库**（体积或凭据原因，本地/服务器保留）：`checkpoints/`（模型权重）、`data/`（数据集，除冒烟 fixture）、`logs/`、`outputs/`、`tmp/`、`third_party/`（上游仓库，见 §6 克隆命令）、`.conda/` 等环境目录。

## 5. 快速开始

```bash
# 环境：唯一 Conda 环境（A100：/mnt/data/yetbye/envs/h3-interactive，启动名 h3-interactive）
# 依赖按阶段 profile 切换，见 configs/；当前 talkverse profile：
#   Python 3.10 / PyTorch 2.7.1+cu128 / FlashAttention 2.8.0.post2 / transformers 4.51.3 / diffusers 0.39.0

# 1) 安装服务层（editable）
pip install -e .

# 2) 启动服务（默认 mock 后端，仅 loopback）
h3-interactive-server --backend mock --port 8765

# 3) FlashHead 后端（真实 GPU runtime，需先完成 §7 资源下载）
export FLASHHEAD_CKPT_DIR=$PWD/checkpoints/SoulX-FlashHead-1_3B
export FLASHHEAD_WAV2VEC_DIR=$PWD/checkpoints/wav2vec2-base-960h
export FLASHHEAD_REPO_DIR=$PWD/third_party/SoulX-FlashHead
export FLASHHEAD_COND_IMAGE=$PWD/data/raw/talkverse-smoke/reference.png
h3-interactive-server --backend flashhead

# 4) 系统基线 headless 实验（GPU 空闲时；自动护栏拒绝与他人的作业争抢）
bash experiments/system-interface/flashhead-headless-001/run.sh
```

## 6. 上游参考仓库（固定 revision）

```bash
# 置于 third_party/ 下，revision 与 PROJECT_STATUS.md 记录保持一致
git clone https://github.com/MiniMax-AI/MiniMax-H3.git      && git -C MiniMax-H3 checkout d21241f
git clone https://github.com/Junchao-cs/SolarWM.git         && git -C SolarWM checkout ce1da4e
git clone https://hf-mirror.com/zhenzhiwang/talkverse.git TalkVerse && git -C TalkVerse checkout 3607ff2
git clone https://github.com/Soul-AILab/SoulX-FlashHead.git && git -C SoulX-FlashHead checkout 9bc03de
```

上游代码只读，不直接修改；适配与 runtime 全部放在 `src/h3_interactive/`。

## 7. 资源下载清单

HF 下载约定：token 写入服务器 `~/.hf_token`（模式 600，**绝不入库**）；大陆网络使用 `HF_ENDPOINT=https://hf-mirror.com`。

| 资源 | 用途 | 大小 | 许可 | 是否必须 |
| --- | --- | --- | --- | --- |
| `Soul-AILab/SoulX-FlashHead-1_3B`（Model_Lite/Pro + VAE_LTX/Wan） | 系统基线实时模型 | 14.3 GB | Apache-2.0（非门控） | 必需 |
| `facebook/wav2vec2-base-960h` | FlashHead 音频编码器 | 1.1 GB | Apache-2.0 | 必需 |
| `zhenzhiwang/talkverse`（元数据 parquet） | 数据契约与样本规划 | 0.98 GB | Snap 非商业（gated） | 必需 |
| `MiniMaxAI/MiniMax-H3` Ref2VA 任务族 | 模型基线推理（全参考模式：文本 + 参考图/视频/音频） | ~135 GB | H3 Community License（非门控） | 必需 |
| `MiniMaxAI/MiniMax-H3` FL2VA 任务族 | 首末帧模式（0/1/2 张输入图，t2va 变体） | ~144 GB | 同上 | 可选 |
| `Junchao-cs/SolarWM-H3-33B`（base/bid/tf/sgf 四包） | 训练主线权重（gated） | ~290 GB | 受 H3 许可约束 | 必需（训练阶段） |
| TalkVerse 源视频（OpenHumanVid 审批 + Panda70M 抓取 + UMT5 预计算） | 训练数据 | 未定 | 逐源审查 | 必需（训练阶段） |
| `Wan-AI/Wan2.2-TI2V-5B`（ModelScope 镜像） | Wan 路线对照基线 | 32 GiB | Apache-2.0 | 可选 |
| TalkVerse LoRA（`3a58ee5`） | Wan 路线对照基线 | 2.0 GB | 非商业 | 可选 |

下载约定：在服务器上通过 `snapshot_download` 拉取（token 从 `~/.hf_token` 读取、不入库，日志写 `logs/`），每条资源下载后固定 revision 并生成 SHA-256 清单（记录见 [`PROJECT_STATUS.md`](PROJECT_STATUS.md)）。

## 8. 阶段路线图

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| P0 资产与基线冻结 | 上游固定、许可证、唯一环境、官方推理样例 | 同 seed 可复现 + 显存/延迟/哈希记录 |
| P1 系统层 MVP | FlashHead 接入统一接口，Agent 运行中控制 | 连续输出 + 控制生效（进行中） |
| P2 H3 分块交互基线 | SolarWM-H3 Stage2 权重零训练推理，逐块文本/动作控制 | 连续 ≥3 块且块间一致 |
| P3 预录音频驱动 PoC | AudioVAE latent clamp（首个假设）+ 自注意力 LoRA + 对照实验 | 口型同步优于无音频基线 |
| P4 AR 与少步蒸馏 | SolarWM Stage0.5→1→2 全流程 + 音频条件 + 4-step | 4-step 质量达可接受阈值 |
| P5 实时与长时记忆 | 同 step KV reuse、固定短窗口、ConvKV、混合精度 | 30 分钟无显存增长、延迟稳定 |
| P6 论文与产品验证 | 消融、基准、长时运行、HCI 用户研究 | 形成独立研究贡献 |

## 9. 许可证与合规

- **MiniMax H3 Community License（2026-08-02）**：适用区域不含欧盟/英国/韩国/美国；LoRA、蒸馏、基于中间表示的训练均属 Model Derivatives；§V.3 禁止用 H3 产物改进其他模型（与 Wan 路线严格隔离）；§IV.2 商业产品须在 UI 展示 "MiniMax H3"；§V.5 + Exhibit A.12 要求机器生成披露与举报机制。逐条核验见 [`docs/LICENSE_REVIEW.md`](docs/LICENSE_REVIEW.md)，协议原文见 `docs/licenses/`。
- **TalkVerse**：Snap Inc. Non-Commercial License，仅非商业研究；数据需通过 HF 门控接受。
- **SoulX-FlashHead**：源码 Apache-2.0；权重按模型卡复核。
- 本项目当前为**非商业研究**用途；对外发布、演示或服务前须完成单独合规复核。

## 10. 安全约定

- 凭据（HF token、服务器口令）只存于服务器家目录或会话环境变量，**不写入仓库、日志、命令文件或实验 manifest**。
- 服务默认只绑定 loopback；跨主机访问走 SSH 隧道（已验证）；未经安全检查不得公开暴露。
- GPU 实验带双重护栏：存在他人计算进程即拒绝启动（退出码 2），不与共享 A100 上的其他作业争抢。
