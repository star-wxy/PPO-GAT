from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_bar_chart(
    df,
    x_col,
    y_col,
    title,
    ylabel,
    save_path,
    xlabel="Method",
    std_col=None,
):
    plt.figure(figsize=(10, 5))

    yerr = None
    if std_col is not None and std_col in df.columns:
        yerr = df[std_col]

    plt.bar(
        df[x_col],
        df[y_col],
        yerr=yerr,
        capsize=5 if yerr is not None else 0,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.8,
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metric_group(df, label_col, prefix, output_dir: Path) -> None:
    metric_specs = [
        ("avg_reward", "Average Reward", f"{prefix} Average Reward Comparison"),
        ("avg_total_time", "Average Total Time", f"{prefix} Average Total Time Comparison"),
        ("avg_queue_delay", "Average Queue Delay", f"{prefix} Average Queue Delay Comparison"),
        ("avg_energy_cost", "Average Energy Cost", f"{prefix} Average Energy Cost Comparison"),
        (
            "avg_deadline_penalty",
            "Average Deadline Penalty",
            f"{prefix} Average Deadline Penalty Comparison",
        ),
        (
            "avg_overload_penalty",
            "Average Overload Penalty",
            f"{prefix} Average Overload Penalty Comparison",
        ),
    ]

    file_prefix = prefix.lower().replace(" ", "_").replace("-", "_")
    for metric, ylabel, title in metric_specs:
        std_col = f"{metric}_std"
        plot_bar_chart(
            df=df,
            x_col=label_col,
            y_col=metric,
            std_col=std_col,
            title=title,
            ylabel=ylabel,
            save_path=output_dir / f"{file_prefix}_{metric}_comparison.png",
        )


def plot_baseline_results(output_dir: Path) -> None:
    csv_path = Path("outputs/results/policy_baseline_comparison.csv")
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    if "avg_reward" in df.columns:
        df = df.sort_values(by="avg_reward", ascending=False)

    print(">>> loaded baseline comparison results")
    print(df.to_string(index=False))

    plot_metric_group(df=df, label_col="policy", prefix="Baseline", output_dir=output_dir)


def plot_ppo_gat_results(output_dir: Path) -> None:
    csv_path = Path("outputs/results/ppo_gat_comparison.csv")
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    if "avg_reward" in df.columns:
        df = df.sort_values(by="avg_reward", ascending=False)

    print(">>> loaded PPO and GAT comparison results")
    print(df.to_string(index=False))

    plot_metric_group(df=df, label_col="model", prefix="PPO GAT", output_dir=output_dir)


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
