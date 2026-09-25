# 既有数字人直播系统接口审计

> 审计日期：2026-09-25  
> 源目录：`D:\venus\digital-human-livestream-main`  
> 审计方式：只读静态检查；未启动服务、未加载模型、未占用 GPU。

## 结论

该目录是可复用的 LiveTalking 派生系统骨架，适合作为系统层基线。它已有音频输入、渲染后端、会话、WebRTC/RTC push、LLM/TTS、录制和中断能力，但当前接口直接绑定全局状态、具体渲染类和 aiortc 队列，不能直接作为 FlashHead/H3 的稳定服务契约。

新接口与模型适配代码以服务器 `/mnt/data/yetbye/h3-interactive` 为主工作区并由 GitHub 维护；本地 `D:\venus\mm` 只保留规划、状态与必要同步文档。旧目录保持只读。

## 已确认架构

```text
HTTP / WebRTC 客户端
        │
        ├─ /offer ────────────── 创建 session + PeerConnection
        ├─ /human ────────────── 文本直驱或 LLM → TTS
        ├─ /humanaudio ───────── 上传音频文件
        ├─ /interrupt_talk ───── 清空 TTS/ASR 输入
        └─ /is_speaking ──────── 查询说话状态
                              │
                         BaseReal
                              │
              ┌───────────────┼───────────────┐
            LipReal          MuseReal        LightReal
              │
       res_frame_queue
              │
       HumanPlayer / aiortc
              │
        audio/video tracks
```

入口与职责：

- `app.py`：服务入口、session 字典、HTTP API、模型加载与 warm-up。
- `basereal.py`：统一音频输入、TTS/ASR 清空、帧合成与音视频投递。
- `webrtc.py`：aiortc track、时间戳、25 FPS 视频和 20 ms 音频节拍。
- `lipreal.py`、`musereal.py`、`lightreal.py`：具体渲染后端。
- `pipeline.py`：弹幕、过滤、LLM 与渲染业务编排。

## 音视频契约

| 项目 | 当前语义 |
| --- | --- |
| 音频采样率 | 16 kHz |
| 音频声道 | 输入多声道时只取第一声道 |
| 音频块 | 320 samples，即 20 ms，float32 输入 |
| 视频节拍 | 25 FPS，即 40 ms/帧 |
| 音画映射 | 每个视频帧对应两个 20 ms 音频帧 |
| WebRTC 音频时基 | `1/16000` |
| WebRTC 视频时基 | `1/90000` |
| WebRTC track 队列 | 每条 track 最大 100 项 |
| 模型结果队列 | 通常为 `batch_size * 2` |
| ASR 输入队列 | 无界队列，当前没有背压协议 |

音频文件会被完整读入内存、转为 float32、必要时重采样到 16 kHz，再切成 320-sample 块；末尾不足 320 samples 的部分当前会被丢弃。

## 统一接口映射

| 目标接口 | 可复用能力 | 仍需补齐 |
| --- | --- | --- |
| `start_session` | `/offer`、`build_nerfreal`、`HumanPlayer` | 去除全局 `opt.sessionid` 写入；明确容量和失败语义 |
| `push_audio` | `put_audio_file`、`put_audio_frame` | 背压、序号、时间戳、流结束标记 |
| `update_control` | `/human`、`set_audiotype` | 统一文本、动作、Agent 指令结构与生效块边界 |
| `pull_video_chunk` | `PlayerStreamTrack.video` | 从 aiortc 私有队列解耦；定义 chunk、PTS 和超时 |
| `reset` | `flush_talk` | 同时清空模型状态、输出队列和时钟；定义 reset 完成点 |
| `close_session` | PeerConnection close 回调 | 幂等关闭、线程 join、队列释放和异常清理 |
| `get_status` | `is_speaking` | 队列深度、模型状态、错误、延迟和资源指标 |

## 已发现风险

1. session 上限检查被注释，`nerfreals` 是进程内全局字典。
2. `build_nerfreal` 会修改全局 `opt.sessionid`，不适合作为可靠多会话边界。
3. `flush_talk` 只清空 TTS/ASR 输入，不能证明模型缓存、输出队列和 WebRTC 时间轴已重置。
4. ASR 输入队列无界，慢模型下可能持续积压。
5. 输出接口直接依赖 `PlayerStreamTrack._queue` 私有成员。
6. 主程序在监听端口前加载并 warm-up GPU 模型，无法独立验证纯服务控制面。
7. 目录不是 Git 仓库，无法从 commit 追溯当前版本和本地改动。
8. 当前只确认 `models/wav2lip.pth` 和四组 wav2lip avatar；MuseTalk/UltraLight 的完整权重资产尚未确认。
9. `D:\envs\live` 可导入 CUDA PyTorch、aiohttp、aiortc、OpenCV 和 soundfile，但未安装 pytest，尚未运行仓库测试。
10. README 中存在疑似真实 API 凭据；必须轮换，且不得迁移到新仓库。

## 下一步验收

下一步在服务器项目中先实现与具体模型无关的协议类型、mock backend 和 session manager，并以 CPU 测试验证：

```text
start_session
  → push_audio(连续多个带时间戳的块)
  → pull_video_chunk
  → update_control
  → reset
  → close_session
```

该步骤不接入真实 FlashHead/H3，不启动网络服务，不使用 GPU。通过后再实现 LiveTalking HTTP/WebRTC bridge 和 FlashHead backend。
