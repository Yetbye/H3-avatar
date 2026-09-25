# 实验合同：<campaign>/<run_id>

状态：`DRAFT`  
负责人：  
日期：  

## 1. 问题与假设

- 本轮只回答：
- 机制假设：
- 反驳条件：
- 明确不回答：

## 2. 单一变量与对照

- 基线：
- 候选：
- 唯一主要变量：
- 固定项：seed、输入、时长、分辨率、采样步数、CFG、硬件。

## 3. 来源与环境

- 代码 commit/diff：
- 第三方 revision：
- 权重 revision/hash：
- 环境锁文件：
- GPU/驱动/CUDA：

## 4. 数据契约

- manifest/hash：
- train/dev/test：
- FPS/采样率/时长：
- 音画时间轴：
- 排除和坏样本规则：

## 5. 训练或推理配置

- 可训练参数：
- checkpoint 起点：
- loss/学习率/batch/步数：
- attention mask 与缓存语义：
- 资源预算和预计时间：
- 中断恢复规则：

## 6. 评估与 PASS 阈值

- 主指标及阈值：
- 护栏指标及阈值：
- 性能指标：
- 人工评测样本数与盲测方式：
- 证据不足条件：

## 7. 执行记录

- 完整命令：`command.sh`
- 日志：`logs/`
- 指标：`metrics.json`
- 产物：`artifacts/`
- 退出码：

## 8. Gate 审计

| Gate | PASS/FAIL/NA | 证据 |
| --- | --- | --- |
| Provenance |  |  |
| Environment & Model Load |  |  |
| Offline Generation |  |  |
| Chunk Continuity |  |  |
| Control Causality |  |  |
| Training Health |  |  |
| Realtime System |  |  |
| Long-Horizon & Conclusion |  |  |

## 9. 结论

- 最终状态：
- 事实与测量：
- 推断与未验证项：
- 失败案例：
- 下一步：
