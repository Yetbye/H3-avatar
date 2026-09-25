# 资源预算

状态：`READY`  
日期：2026-09-25

## 已确认资源

- 远程根目录：`/mnt/data/yetbye/h3-interactive`。
- GPU：1x NVIDIA A100 80GB PCIe。
- `/mnt/data`：约 3.6TB，总剩余约 2.2TB。
- 当前 GPU 被其他用户任务占用，利用率约 99%；环境和文档工作可继续，推理必须等待空闲窗口。
- Conda：`/usr/local/miniconda3/bin/conda`，版本 26.1.1。

## 单环境策略

只维护一个 Conda 环境：

```text
/mnt/data/yetbye/envs/h3-interactive
```

Conda 包缓存和 pip 缓存也必须位于本工作区：

```text
.conda/pkgs/
.cache/pip/
.cache/huggingface/
```

由于模型依赖冲突，该环境按阶段切换依赖配置，而不是承诺四个模型同时可运行。每个通过 smoke 的状态必须导出 Conda/pip 锁文件。

## 已发现依赖冲突

| 项目 | 官方关键版本 |
| --- | --- |
| SolarWM-H3 | Python 3.10、PyTorch 2.6/CUDA 12.4、Transformers 5.12.1、FlashAttention 2.8.3 |
| TalkVerse | Python >=3.10、Transformers 4.49-4.51.3、NumPy <2 |
| FlashHead | Python 3.10、PyTorch 2.7.1/CUDA 12.8、Transformers 4.57.3、FlashAttention 2.8.0.post2 |
| MiniMax H3 | PyTorch >=2.4、Diffusers >=0.32.2、Transformers >=4.45 |

因此环境顺序为：`base -> talkverse -> h3/solarwm -> flashhead`。切换阶段前保存锁文件，切换后重新运行该阶段和必要的回归 smoke。

## 当前限制

- 不下载 TalkVerse 全量视频。
- 不照搬需要 4-8 GPU 的训练配置。
- 未确认前不进行超过 100GB 的单次下载。
- 未预注册实验前不启动长时训练。
- GPU 繁忙时不与其他用户抢占计算资源。

## 扩容触发条件

当出现以下任一情况时再申请多卡节点：

- H3 官方推理在 80GB 显存和合理 offload 下无法完成。
- SolarWM-H3 Stage 训练无法缩小为有效 smoke。
- 单卡预估训练时间不具备实验迭代价值。
- 实时目标需要模型并行才能满足延迟。
