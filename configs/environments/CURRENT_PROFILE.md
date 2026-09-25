# 当前环境 Profile

> 验证日期：2026-09-25

- 环境：`/mnt/data/yetbye/envs/h3-interactive`（启动名：`h3-interactive`）
- Profile：`talkverse`
- 状态：`PASS`（仅代码与依赖基线，不代表模型推理通过）
- Python：3.10.21
- PyTorch：2.7.1+cu128
- Transformers：4.51.3
- Diffusers：0.39.0
- FlashAttention：2.8.0.post2
- GPU 探测：CUDA 可用，1 张设备可见

已运行检查：

- TalkVerse 安装脚本重复运行成功；
- PyTorch、TorchVision、TorchAudio、Transformers、Diffusers、FlashAttention、Librosa、Decord 导入成功；
- `TalkVerse/generate.py --help` 返回码为 0；
- 工作区边界检查通过。

尚未运行：

- 已下载 Wan2.2-TI2V-5B、TalkVerse LoRA 和 Wav2Vec2，但尚未下载 H3、SolarWM-H3 或 FlashHead 权重；
- 未下载 TalkVerse 数据；
- 未执行 GPU 模型推理或生成 MP4。

模型资产详见 `../models/MANIFEST.md`。

已知警告：PyPI 发布的 `decord 0.6.0` wheel 带有 CPython 3.6 标签，`pip check` 在 Python 3.10 中报告平台不支持；实际模块及其动态库导入成功。安装脚本只允许这一条精确匹配的警告。
