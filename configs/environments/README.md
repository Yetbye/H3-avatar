# 单一 Conda 环境策略

环境路径：

```text
/mnt/data/yetbye/envs/h3-interactive
```

项目只维护这一个环境，但依赖按阶段演进：

1. `base`：Python 3.10、pip 和审计工具。
2. `talkverse`：首个预录音频驱动基线。
3. `h3-solarwm`：MiniMax H3 与 SolarWM-H3 基线及训练。
4. `flashhead`：实时服务基线。

每次阶段切换前后都必须导出：

```text
configs/environments/<stage>-conda-explicit.lock
configs/environments/<stage>-pip-freeze.txt
```

环境切换不是简单增量安装。遇到版本降级或卸载时，必须重新执行上一阶段要求保留的 smoke；无法同时满足的基线以各自锁文件恢复。

创建基础环境：

```bash
bash scripts/create_conda_env.sh
```

激活环境：

```bash
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate h3-interactive
```

当前 profile 与验证结果记录在 `CURRENT_PROFILE.md`。TalkVerse 的 PyPI `decord 0.6.0` wheel 元数据错误地声明为 CPython 3.6，因此新版 pip 会产生平台警告；安装脚本只豁免这一条已知警告，并继续验证 `decord` 的实际导入。其他依赖错误仍会令脚本失败。
