# 既有数字人直播系统接口审计

权威审计记录保存在本地项目文档 `workflow/LIVESTREAM_INTERFACE_AUDIT.md`。结论如下：

- 源系统：`D:\venus\digital-human-livestream-main`，保持只读。
- 主入口：`app.py`；核心抽象：`BaseReal`；传输：`HumanPlayer`/aiortc。
- 固定节拍：16 kHz 单声道音频、20 ms/320 samples 音频块、25 FPS 视频，每帧对应两个音频块。
- 已有能力：session、音频文件/帧输入、文本/LLM/TTS、中断、WebRTC 音视频输出、录制和说话状态。
- 主要缺口：无统一 session 生命周期、无输入背压、reset 语义不完整、输出依赖私有队列、全局状态不适合多会话。
- 新接口和 FlashHead/H3 backend 在本服务器项目的 `src/` 与 `tests/` 中实现，不修改源系统。

下一步先用 CPU mock backend 验证统一生命周期，再接 LiveTalking bridge 和真实模型。
