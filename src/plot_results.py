from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_bar_chart(df, x_col, y_col, title, ylabel, save_path):
    plt.figure(figsize=(8, 5))
    plt.bar(df[x_col], df[y_col])
    plt.title(title)
    plt.xlabel("Policy")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    csv_path = Path("outputs/results/baseline_comparison_v2.csv")
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"结果文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)

    print(">>> 已读取结果文件:")
    print(df.to_string(index=False))

    # 1. 平均奖励
    plot_bar_chart(
        df=df,
        x_col="policy",
        y_col="avg_reward",
        title="Average Reward Comparison",
        ylabel="Average Reward",
        save_path=output_dir / "avg_reward_comparison.png",
    )

    # 2. 平均总时延
    plot_bar_chart(
        df=df,
        x_col="policy",
        y_col="avg_total_time",
        title="Average Total Time Comparison",
        ylabel="Average Total Time",
        save_path=output_dir / "avg_total_time_comparison.png",
    )

    # 3. 平均能耗
    plot_bar_chart(
        df=df,
        x_col="policy",
        y_col="avg_energy_cost",
        title="Average Energy Cost Comparison",
        ylabel="Average Energy Cost",
        save_path=output_dir / "avg_energy_cost_comparison.png",
    )

    # 4. 平均过载惩罚
    plot_bar_chart(
        df=df,
        x_col="policy",
        y_col="avg_overload_penalty",
        title="Average Overload Penalty Comparison",
        ylabel="Average Overload Penalty",
        save_path=output_dir / "avg_overload_penalty_comparison.png",
    )

    print("\n>>> 图表已生成到 outputs/figures/")
    for file in output_dir.glob("*.png"):
        print(file)


if __name__ == "__main__":
    main()