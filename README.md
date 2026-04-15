# mytest

一个面向多机器人智能体的异构算力调度实验项目，当前包含三类强化学习/图强化学习策略：

- `PPO-Baseline`
- `PPO-GAT-Naive`
- `PPO-GAT-Scoring`

其中 `PPO-GAT-Scoring` 是当前的主融合方案，采用“节点打分式 PPO+GAT”结构，用于在多机器人任务源、异构计算节点和拓扑约束下进行算力调度。

## 项目定位

本项目当前分成两层：

- 算法实验层  
  用于训练、对比和分析 `PPO`、朴素 `PPO+GAT`、节点打分式 `PPO+GAT`

- 轻量系统验证层  
  用于在“多机器人任务持续产生 -> 中央调度器决策 -> 计算节点执行”的流程下，导出事件日志和系统级摘要结果

## 当前环境说明

当前环境已经升级为“多机器人智能体调度环境”，核心特征包括：

- 多机器人异构画像：不同机器人具有不同的本地算力、任务到达率、任务规模偏好、deadline 偏好
- 任务来源绑定：每个任务都带有 `source_robot_id`
- 任务类型差异：支持不同类型任务的规模、优先级、传输需求差异
- 异构计算节点：本地 / 边缘 / 云节点具有不同 CPU、时延、能耗因子
- 拓扑约束：机器人到节点的距离会影响传输时延与传输能耗
- 邻接拥塞传播：节点负载会向相邻节点扩散，形成更真实的资源竞争关系

核心环境文件：

- `configs/env.yaml`
- `src/envs/multi_robot_scheduler_env.py`
- `src/envs/task.py`
- `src/envs/robot.py`
- `src/envs/node.py`

## 目录结构

```text
configs/
  env.yaml
  train_plain_ppo.yaml
  train_naive_gat.yaml
  train_scoring_gat.yaml

scripts/
  train_ppo_baseline.ps1
  train_ppo_naive.ps1
  train_ppo_scoring.ps1
  compare_baselines.ps1
  compare_ppo_models.ps1
  plot_results.ps1
  run_baseline_experiment.ps1
  run_gat_experiment.ps1
  run_tensorboard.ps1
  run_lightweight_multi_robot_validation.ps1

src/
  train_plain_ppo.py
  train_naive_gat.py
  train_scoring_gat.py
  compare_baselines.py
  compare_ppo_models.py
  plot_results.py
  lightweight_multi_robot_validation.py
  envs/
  models/
  baselines/
  utils/
```

## 环境准备

建议使用 Python 3.10。

```powershell
conda create -n Mytest python=3.10 -y
conda activate Mytest
pip install -r requirements.txt
```

## 训练脚本

### 1. 单 PPO

```powershell
.\scripts\train_ppo_baseline.ps1
```

### 2. 朴素 PPO+GAT

```powershell
.\scripts\train_ppo_naive.ps1
```

### 3. 节点打分式 PPO+GAT

```powershell
.\scripts\train_ppo_scoring.ps1
```

## 对比脚本

### 1. 基线策略对比

用于比较：

- `PPO-Baseline`
- `Random-Policy`
- `RoundRobin-Policy`
- `GreedyCPU-Policy`

```powershell
.\scripts\compare_baselines.ps1
```

输出：

- `outputs/results/policy_baseline_comparison.csv`
- `outputs/results/policy_baseline_comparison_per_seed.csv`

### 2. PPO / GAT 融合策略对比

用于比较：

- `PPO-Baseline`
- `PPO-GAT-Naive`
- `PPO-GAT-Scoring`

```powershell
.\scripts\compare_ppo_models.ps1
```

输出：

- `outputs/results/ppo_gat_comparison.csv`
- `outputs/results/ppo_gat_comparison_per_seed.csv`

说明：

- 当前对比脚本采用固定多 seed 评估
- 汇总表输出均值与标准差
- `*_per_seed.csv` 保存每个 seed 的原始结果，方便论文进一步分析

## 绘图脚本

```powershell
.\scripts\plot_results.ps1
```

输出目录：

- `outputs/figures/`

当前图表支持：

- 平均值柱状图
- 标准差误差线
- baseline 对比图
- PPO / GAT 融合对比图

## 一键实验脚本

### 1. 基线实验流程

```powershell
.\scripts\run_baseline_experiment.ps1
```

流程：

- 训练 `PPO-Baseline`
- 运行 baseline 对比
- 生成图表

### 2. GAT 融合实验流程

```powershell
.\scripts\run_gat_experiment.ps1
```

流程：

- 训练 `PPO-Baseline`
- 训练 `PPO-GAT-Naive`
- 训练 `PPO-GAT-Scoring`
- 运行 PPO/GAT 对比
- 生成图表

## TensorBoard

```powershell
.\scripts\run_tensorboard.ps1
```

该脚本会同时加载三套日志目录：

- `outputs/tensorboard_baseline`
- `outputs/tensorboard_naive`
- `outputs/tensorboard_scoring`

默认地址：

- `http://localhost:6006`

## 轻量系统仿真验证

```powershell
.\scripts\run_lightweight_multi_robot_validation.ps1
```

该脚本用于做一个轻量级“多机器人任务流 + 中央调度 + 节点执行”的系统验证，不依赖 Gazebo。

输出：

- `outputs/results/lightweight_multi_robot_validation_events.csv`
- `outputs/results/lightweight_multi_robot_validation_summary.csv`

用途：

- 展示多机器人任务连续产生与调度过程
- 导出事件日志，便于论文中做系统级验证说明
- 在尚未接入 Gazebo/ROS 时，先完成轻量系统验证闭环

## 结果目录

常用输出目录如下：

- `outputs/checkpoints/`  
  训练完成的模型文件

- `outputs/results/`  
  对比 CSV、逐 seed 结果、轻量验证结果

- `outputs/figures/`  
  图表输出

- `outputs/tensorboard_*`  
  TensorBoard 日志

## 当前建议使用顺序

如果你要完整跑一轮当前实验，建议顺序如下：

```powershell
.\scripts\train_ppo_baseline.ps1
.\scripts\train_ppo_naive.ps1
.\scripts\train_ppo_scoring.ps1
.\scripts\compare_baselines.ps1
.\scripts\compare_ppo_models.ps1
.\scripts\plot_results.ps1
.\scripts\run_lightweight_multi_robot_validation.ps1
```

## 说明

当前项目的论文主线建议为：

- 单 `PPO` 作为强化学习基线
- `PPO-GAT-Naive` 作为朴素融合对照
- `PPO-GAT-Scoring` 作为主创新方法
- 多 seed 算法实验作为主体实验
- 轻量系统仿真验证作为“面向多机器人智能体调度”的系统级支撑

