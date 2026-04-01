# mytest

一个最小可运行的 PPO 调度强化学习模板。

## 目录结构

- `configs/`：环境与训练配置
- `src/envs/`：自定义调度环境
- `src/utils/`：配置加载与随机种子工具
- `src/train.py`：训练入口
- `src/eval.py`：评估入口
- `scripts/`：PowerShell 启动脚本

## 运行步骤

### 1. 创建环境
```powershell
conda create -n Mytest python=3.10 -y
conda activate Mytest
