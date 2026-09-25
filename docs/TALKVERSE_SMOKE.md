# TalkVerse 首次生成 Smoke

## 已完成的安全检查

- `scripts/preflight_talkverse.py` 只读取索引和 safetensors 头，不加载模型张量。
- 基础模型索引：825 个张量，3 个分片，无缺失。
- TalkVerse checkpoint：640 个张量，其中 480 个 LoRA A/B 张量，对应 240 个基础权重；目标全部存在，rank 为 128。
- 上游 `generate.py` 的 batch 分支正确传入 LoRA；单条分支漏传 `lora_ckpt`，本 smoke 只使用 batch 分支且不修改第三方代码。
- `scripts/run_talkverse_smoke.sh` 在发现任意 GPU 计算进程时退出，避免影响共享服务器上的其他任务。

## 输入

`scripts/prepare_talkverse_smoke.py` 已在 `data/raw/talkverse-smoke/` 生成：

- `reference.png`：512×512 参考图；
- `drive_6s.wav`：6 秒、16 kHz、单声道 PCM；
- `batch.json`：单条 batch 输入。

这些文件来自工作区内已有的 FlashHead 示例，仅用于本地 smoke，不是 TalkVerse 数据集。

## GPU 空闲后的执行命令

```bash
cd /mnt/data/yetbye/h3-interactive
scripts/run_talkverse_smoke.sh
```

runner 固定使用项目唯一 Conda 环境、batch 推理分支、`fasttalk-480`、1 个 120 帧 clip、CPU T5 与模型 offload。输出和日志分别写入：

```text
experiments/talkverse-smoke-001/outputs/smoke_001.mp4
experiments/talkverse-smoke-001/logs/run.log
```

只有 MP4、完整日志、峰值显存和耗时均核验后，才把离线生成标记为 `PASS`。
