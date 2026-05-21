# PPO-GAT 20 Robots / 10 Nodes Scheduler

当前项目聚焦于 20 个机器人、10 个异构计算节点的多机器人任务调度实验。

保留的主线模型：

- `PPO-Baseline`
- `PPO-GAT-Naive`
- `PPO-GAT-Scoring`

其中 `PPO-GAT-Scoring` 是当前主方法，用节点打分式 GAT 特征提取器辅助 PPO 选择调度节点。

## 当前配置

```text
configs/
  env_20r_10n.yaml
  train_plain_ppo_20r_10n.yaml
  train_naive_gat_20r_10n.yaml
  train_scoring_gat_20r_10n.yaml
```

## 主要脚本

```text
scripts/
  train_ppo_baseline.ps1
  train_ppo_naive.ps1
  train_ppo_scoring.ps1
  compare_baselines.ps1
  compare_ppo_models.ps1
  run_gat_experiment.ps1
  run_lightweight_multi_robot_validation.ps1
  run_tensorboard.ps1
  plot_metric_panel.py
```

## 环境准备

建议使用 Python 3.10。

```powershell
conda create -n Mytest python=3.10 -y
conda activate Mytest
pip install -r requirements.txt
```

## 一键完整实验

```powershell
.\scripts\run_gat_experiment.ps1
```

该脚本会依次执行：

- 训练 `PPO-Baseline`
- 训练 `PPO-GAT-Naive`
- 训练 `PPO-GAT-Scoring`
- 对比 final checkpoint
- 对比 best model
- 运行轻量系统验证

主要输出：

```text
outputs/results/ppo_gat_comparison_20r_10n.csv
outputs/results/ppo_gat_comparison_20r_10n_best.csv
outputs/results/ppo_gat_comparison_20r_10n_per_seed.csv
outputs/results/ppo_gat_comparison_20r_10n_best_per_seed.csv
outputs/results/lightweight_multi_robot_validation_20r_10n_summary.csv
outputs/results/lightweight_multi_robot_validation_20r_10n_events.csv
```

## 单独训练

这三个训练入口默认使用 20机器人/10节点配置。

```powershell
.\scripts\train_ppo_baseline.ps1
.\scripts\train_ppo_naive.ps1
.\scripts\train_ppo_scoring.ps1
```

## 单独对比

PPO 与传统基准算法对比：

```powershell
.\scripts\compare_baselines.ps1
```

该脚本比较：

- `PPO-Baseline`
- `Random-Policy`
- `RoundRobin-Policy`
- `GreedyCPU-Policy`

输出：

```text
outputs/results/policy_baseline_comparison_20r_10n.csv
outputs/results/policy_baseline_comparison_20r_10n_per_seed.csv
```

final checkpoint 对比：

```powershell
.\scripts\compare_ppo_models.ps1
```

best model 对比：

```powershell
.\scripts\compare_ppo_models.ps1 --output-prefix ppo_gat_comparison_20r_10n_best --use-best-model
```

## 绘制指标面板图

```powershell
python scripts\plot_metric_panel.py
```

输出只保留两张面板图：

```text
outputs/figures/metric_panel.png
outputs/figures/metric_panel_best.png
```

## TensorBoard

```powershell
.\scripts\run_tensorboard.ps1
```

加载的日志目录：

```text
outputs/tensorboard_baseline_20r_10n
outputs/tensorboard_naive_20r_10n
outputs/tensorboard_scoring_20r_10n
```

默认地址：

```text
http://localhost:6006
```

## 当前论文实验主线

- `PPO-Baseline` 作为强化学习基线
- `Random-Policy`、`RoundRobin-Policy`、`GreedyCPU-Policy` 作为传统基准算法
- `PPO-GAT-Naive` 作为朴素 GAT 融合对照
- `PPO-GAT-Scoring` 作为主创新方法
- 多 seed 结果用于算法对比
- 轻量系统验证用于展示多机器人任务流、中央调度和异构节点执行闭环
