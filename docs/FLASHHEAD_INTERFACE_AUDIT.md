# SoulX-FlashHead 接口审计

> 审计日期：2026-09-25  
> 上游路径：`third_party/SoulX-FlashHead`  
> 固定 revision：`9bc03de`

## 结论

上游已经提供单卡 Lite/Pro 推理、离线视频生成和 Gradio streaming demo，但其 streaming 语义是“连续生成模型块并分段封装 MP4”，不是直接面向网络的逐帧会话服务。当前项目可以先稳定自身的 20 ms PCM 输入、40 ms/25 FPS 输出和生命周期合同；真实 GPU runtime 仍需单独实现并验证。

## 已确认的上游接口

- `flash_head.inference.get_pipeline(...)`：加载 FlashHead 和 Wav2Vec2 权重。
- `get_base_data(...)`：用参考图、seed、分辨率和 motion context 初始化 pipeline。
- `get_audio_embedding(...)`：把 16 kHz 音频窗口编码为条件特征。
- `run_pipeline(...)`：生成一个视频块并复用 pipeline 内部状态。
- `get_infer_params()`：读取 25 FPS、生成块长度、motion-frame 数量和缓存音频时长。
- `gradio_app_streaming.py`：工作线程按块推理，三个生成块合并为一个浏览器可播放 MP4 片段。

上游源码许可证为 Apache-2.0。模型权重许可证仍需在下载前单独核验，不能由源码许可证代替。

## 当前合同映射

| 项目合同 | FlashHead runtime 责任 |
| --- | --- |
| 16 kHz、单声道、PCM s16le | 转为 float32，并加入连续音频缓存 |
| 20 ms 音频块、严格 sequence/PTS | 聚合到上游一次推理所需的音频跨度 |
| 40 ms 视频帧 | 把生成块拆为带连续 PTS 的独立编码帧 |
| `reset` | 清空音频、视频和 motion context，重新初始化参考图状态 |
| `close` | 停止 worker，释放队列和 GPU pipeline |
| 背压与 timeout | 使用有界输入/输出队列，不能无限缓存 |

`src/h3_interactive/backends/flashhead.py` 已定义上述稳定边界，并用 fake runtime 验证 CPU 生命周期。它没有导入或修改第三方源码。

## 尚未实现或验证

- FlashHead 1.3B 与 `facebook/wav2vec2-base-960h` 权重尚未下载、校验或加载。
- 当前 TalkVerse profile 的 Transformers 4.51.3 与 FlashHead 要求的 4.57.3 冲突，切换 profile 前必须保存锁文件并执行回归。
- 参考图目前是上游 pipeline 初始化参数，尚未加入公开 HTTP 会话合同。
- 上游没有项目所需的文本/动作控制映射；适配层会明确拒绝，而不是静默忽略。
- 真实块长度、首帧延迟、稳定 FPS、显存峰值、reset 后状态和长时连续性均未测试。
- 视频 payload 的最终编码格式尚未冻结。

因此 `h3-interactive-server --backend flashhead` 目前会在监听端口前明确报告 runtime/权重未配置，不能作为真实模型已接入的证据。
