from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_bar_chart(df, x_col, y_col, title, ylabel, save_path, xlabel="Method"):
    plt.figure(figsize=(10, 5))
    plt.bar(df[x_col], df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_baseline_results(output_dir: Path) -> None:
    csv_path = Path("outputs/results/policy_baseline_comparison.csv")
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    print(">>> loaded baseline comparison results")
    print(df.to_string(index=False))

    plot_bar_chart(
        df=df,
        x_col="policy",
        y_col="avg_reward",
        title="Baseline Average Reward Comparison",
        ylabel="Average Reward",
        save_path=output_dir / "baseline_avg_reward_comparison.png",
    )
    plot_bar_chart(
        df=df,
        x_col="policy",
        y_col="avg_total_time",
        title="Baseline Average Total Time Comparison",
        ylabel="Average Total Time",
        save_path=output_dir / "baseline_avg_total_time_comparison.png",
    )
    plot_bar_chart(
        df=df,
        x_col="policy",
        y_col="avg_energy_cost",
        title="Baseline Average Energy Cost Comparison",
        ylabel="Average Energy Cost",
        save_path=output_dir / "baseline_avg_energy_cost_comparison.png",
    )
    plot_bar_chart(
        df=df,
        x_col="policy",
        y_col="avg_overload_penalty",
        title="Baseline Average Overload Penalty Comparison",
        ylabel="Average Overload Penalty",
        save_path=output_dir / "baseline_avg_overload_penalty_comparison.png",
    )


def plot_ppo_gat_results(output_dir: Path) -> None:
    csv_path = Path("outputs/results/ppo_gat_comparison.csv")
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    print(">>> loaded PPO and GAT comparison results")
    print(df.to_string(index=False))

    plot_bar_chart(
        df=df,
        x_col="model",
        y_col="avg_reward",
        title="PPO and GAT Average Reward Comparison",
        ylabel="Average Reward",
        save_path=output_dir / "ppo_gat_avg_reward_comparison.png",
    )
    plot_bar_chart(
        df=df,
        x_col="model",
        y_col="avg_total_time",
        title="PPO and GAT Average Total Time Comparison",
        ylabel="Average Total Time",
        save_path=output_dir / "ppo_gat_avg_total_time_comparison.png",
    )
    plot_bar_chart(
        df=df,
        x_col="model",
        y_col="avg_energy_cost",
        title="PPO and GAT Average Energy Cost Comparison",
        ylabel="Average Energy Cost",
        save_path=output_dir / "ppo_gat_avg_energy_cost_comparison.png",
    )
    plot_bar_chart(
        df=df,
        x_col="model",
        y_col="avg_overload_penalty",
        title="PPO and GAT Average Overload Penalty Comparison",
        ylabel="Average Overload Penalty",
        save_path=output_dir / "ppo_gat_avg_overload_penalty_comparison.png",
    )


def main():
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_baseline_results(output_dir)
    plot_ppo_gat_results(output_dir)

    if not any(output_dir.glob("*.png")):
        raise FileNotFoundError("No result CSV files found. Run the comparison scripts first.")

    print("\n>>> charts generated under outputs/figures/")
    for file in output_dir.glob("*.png"):
        print(file)


if __name__ == "__main__":
    main()
