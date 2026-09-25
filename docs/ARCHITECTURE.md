# 目标架构

## 1. 逻辑数据流

```text
麦克风/预录音频 ──> 音频接入层 ──> AudioVAE/条件适配 ─┐
文本/动作/Agent ──> 控制接入层 ──────────────────────┤
人物/场景参考 ────> 视觉条件层 ──────────────────────┤
                                                     ▼
                                      H3 多模态 token 时间轴
                                                     │
                               AR block + 少步 DMD + 状态记忆
                                                     │
                                 ┌───────────────────┴──────────────┐
                                 ▼                                  ▼
                           video latent                        audio latent
                                 │                                  │
                              VideoVAE                           AudioVAE
                                 └───────────────────┬──────────────┘
                                                     ▼
                                    时间戳对齐、编码、流式传输
                                                     │
                                                     ▼
                                            实时客户端/Agent
```

## 2. 代码所有权

```text
third_party/  外部源码，固定 revision，默认只读
src/
  adapters/   H3/SolarWM/FlashHead 的窄适配层
  audio/      VAD、重采样、AudioVAE、条件机制
  control/    文本、动作、Agent 指令规范化
  streaming/  分块、时间轴、缓存、背压和会话状态
  evaluation/ 离线质量、同步、连续性和性能评估
services/     与模型无关的 API、worker 和协议
configs/      本项目配置覆盖，不改第三方默认配置
```

第三方模型内部 patch 必须以独立 patch 文件或明确 fork commit 保存，并在 manifest 中记录；不得在无法追踪的状态下直接编辑 `third_party/`。

## 3. 服务边界

模型服务不得直接暴露第三方内部张量。统一接口至少包含：

- `session_id`、模型 revision 和人物参考；
- 音频块、采样率、起止时间戳和是否为最终块；
- 文本/动作/情绪控制及生效时间；
- 视频帧或编码包、音频包和输出时间戳；
- 队列深度、处理延迟、丢帧和错误状态；
- `cancel`、`reset`、`close`。

FlashHead 与 H3 应实现同一接口，以便先验证系统、后替换模型。

## 4. 时间轴约束

- 所有音频 sample、视频 frame、audio latent 和 video latent 使用同一单调时钟。
- 每个 block 保存输入时间范围、可见上下文范围和输出时间范围。
- attention mask 必须显式区分已知音频、历史视频、当前预测和不可见未来。
- look-ahead 是接口的一部分，不能隐藏在实现中。
- 重置会话必须同时清空模型 KV、压缩记忆、音频缓冲和编码队列。

## 5. 部署原则

- 单机开发时，API、模型 worker 和编码器可分进程运行。
- 模型 worker 是唯一持有 GPU 模型的进程。
- 请求队列必须有上限；积压时采用明确的背压或丢帧策略。
- 性能报告拆分采集、预处理、排队、模型、VAE、编码和传输延迟。
- 实时系统失败时必须能够取消当前生成并释放会话状态。
