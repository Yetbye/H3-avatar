# Experiment: flashhead-headless-001

Status: `DRAFT`（已预注册；CPU 侧验证通过，GPU 运行待 A100 空闲）

## Question and hypothesis

系统基线问题：上游 SoulX-FlashHead 推理能否封装为 `FlashHeadRuntime` 真实实现，经 `FlashHeadBackend` 适配器在真实权重上完成「20 ms PCM 连续输入 → 25 FPS 连续出帧 → 会话重置 → 干净关闭」，并达到实时档 PASS 门槛（连续 ≥60 s、播放侧帧间隔 p95 ≤40 ms、重置 <2 s）？

通过条件：G1–G4 全部满足（见下）。本实验不改动上游仓库、不启动对外服务、只在 GPU 空闲时运行。

## Fixed behavior

- 真实 runtime（`src/h3_interactive/backends/flashhead_runtime.py`）继承上游流式推理的全部机制：33 帧生成 chunk（lite 每 chunk 输出 24 帧 / 0.96 s 音频，pro 28 帧 / 1.12 s）、8 s 滚动音频窗口、运动帧 latent 跨 chunk 携带、EOS 静音补齐。
- 所有 GPU 调用串行化在单一 worker 线程；事件循环只接触 CPU 队列。
- 帧 payload 为 JPEG（quality 92），帧率语义 40 ms/帧，PTS 按 chunk 连续编号；reset 后 PTS 从 0 重新开始。
- 运行前双重 GPU 护栏（`run.sh` 与 runner 各自检查 `nvidia-smi` 计算进程），发现他人进程即以退出码 2 拒绝启动。

### 预注册口径（运行前固定，避免事后解释）

1. **前 ~8 s 帧来自静音冷启动缓存**：上游 `generate_video.py` 以零填充 8 s 音频窗口，本实验忠实继承。连续性/延迟指标不受影响；嘴型定性看 t≥10 s 的 stills。
2. **帧按 chunk 突发到达**（lite 每 ~0.96 s 一组 24 帧）是上游生成机制决定的；平滑 40 ms 播放是消费端抖动缓冲的职责。因此「帧间隔 p95 ≤40 ms」按**播放调度口径**衡量：以首帧到达为播放起点、每帧 40 ms 排程，p95 ≤40 ms 等价于零饥饿。原始到达间隔另行报告，不入门。
3. **RTF ≥1.0 只适用于 lite**：上游数据 Model_Pro 为 10.8 FPS@4090，已知非实时；pro 运行只报告 RTF，不参与 G2。
4. **60 s 连续音频由 6 s fixture 循环拼接**（`data/raw/talkverse-smoke/drive_6s.wav` ×10）：循环拼接对连续性与延迟指标有效；嘴型定性在 t≥10 s 的 stills 上进行。

## Inputs

| 输入 | 路径 | 说明 |
| --- | --- | --- |
| 权重 | `checkpoints/SoulX-FlashHead-1_3B` | Model_Lite 6.1 GB（本实验用 lite）+ VAE_LTX；已 SHA-256 校验 |
| wav2vec2 | `checkpoints/wav2vec2-base-960h` | 上游音频编码器 |
| 上游源码 | `third_party/SoulX-FlashHead`（revision `9bc03de`） | 只读；infer_params.yaml 需要 cwd=repo 根 |
| 参考图 | `data/raw/talkverse-smoke/reference.png` | 512×512 |
| 音频 | `data/raw/talkverse-smoke/drive_6s.wav` | 6 s / 16 kHz / 单声道 |
| 环境 | `/mnt/data/yetbye/envs/h3-interactive` | torch 2.7.1+cu128、FA 2.8.0.post2（符合 FlashHead 要求） |

## PASS gates

| Gate | 判据 | 来源 |
| --- | --- | --- |
| G1 | epoch 0 连续 ≥60 s（1500 帧、PTS 无缺口）且 reset 后 epoch 1 连续 5 s（125 帧、PTS 无缺口） | `metrics.json` |
| G2 | 播放侧帧间隔 p95 ≤40 ms 且零饥饿，steady-state RTF ≥1.0（仅 lite） | `metrics.json` |
| G3 | 会话 reset <2 s | `metrics.json` |
| G4 | 无 worker 崩溃、`close` 后状态为 closed、MP4/6 张 stills/日志齐全 | `metrics.json` + 产物 |

辅助记录（不入门）：首帧延迟（含模型加载）、峰值显存（`torch.cuda.max_memory_allocated` + `gpu-after.txt`）、原始到达间隔分布、关闭耗时、pending 队列采样。

## Procedure

```bash
cd /mnt/data/yetbye/h3-interactive
bash experiments/system-interface/flashhead-headless-001/run.sh   # GPU 空闲时执行
```

runner 流程：GPU 护栏 → 构造 60 s + 5 s 音频 → `FlashHeadBackend(make_runtime_factory(...))` → `start` → 实时节奏推送 20 ms PCM（epoch 0，60 s）→ 并发拉帧（播放排程仿真 + 写 MP4 + 抽 stills）→ `reset`（计时）→ epoch 1 推送 5 s → 拉 125 帧 → `close`（计时）→ ffmpeg 混流 → `metrics.json` + 门限判定 → 退出码 0/1/2/3。

## CPU 侧预验证（已完成于 GPU 运行前）

- `compileall` 全部通过（runtime、runner、test）。
- `test_flashhead_runtime_cpu.py` 以 `CUDA_VISIBLE_DEVICES=''` 运行：chunk 布局算术（lite 24/15360、pro 28/17920）、音频累积与 EOS 补齐、无 GPU 时 worker 干净失败且错误经 `pull_video_frame` 上抛、reset-before-start 空操作、close 幂等。

## Evidence

| 产物 | 路径 |
| --- | --- |
| 运行日志 | `logs/run.log` |
| 指标与门限 | `logs/metrics.json` |
| 退出码 | `logs/exit-code.txt` |
| GPU 前后快照 | `logs/gpu-before.txt`、`logs/gpu-after.txt` |
| 进程残留检查 | `logs/python-processes-after.txt` |
| 校验和 | `logs/SHA256SUMS` |
| 连续录像 | `outputs/session.mp4`（含音轨）、`outputs/session_audio.wav` |
| 嘴型定性 | `outputs/stills/t{10..60}s.png` |

## Result

（待 GPU 空闲后运行填写）
