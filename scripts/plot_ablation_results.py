"""
Plot ablation results from the mixed-context experiment.

Usage:
    python scripts/plot_ablation_results.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


SUMMARY_CSV = Path("outputs/results/ablation_mixed_context.csv")
OUTPUT_DIR = Path("outputs/figures")

MODEL_ORDER = [
    "Full-Scoring",
    "No-RobotState",
    "No-Congestion",
    "Fixed-Reward",
    "No-Charging",
    "No-GAT",
    "No-HeuristicGate",
    "No-NodeScoring",
]

MODEL_LABELS = {
    "Full-Scoring": "Full",
    "No-RobotState": "No\nRobot",
    "No-Congestion": "No\nCong.",
    "Fixed-Reward": "Fixed\nReward",
    "No-Charging": "No\nCharge",
    "No-GAT": "No GAT",
    "No-HeuristicGate": "No\nHeur.",
    "No-NodeScoring": "No\nScore",
}

MODEL_COLORS = {
    "Full-Scoring": "#1b9e77",
    "No-RobotState": "#7570b3",
    "No-Congestion": "#66a61e",
    "Fixed-Reward": "#e6ab02",
    "No-Charging": "#e7298a",
    "No-GAT": "#d95f02",
    "No-HeuristicGate": "#a6761d",
    "No-NodeScoring": "#666666",
}

METRICS = [
    ("avg_reward", "Average Reward", "higher"),
    ("avg_total_time", "Total Time", "lower"),
    ("avg_queue_delay", "Queue Delay", "lower"),
    ("avg_network_latency", "Network Latency", "lower"),
    ("avg_deadline_penalty", "Deadline Penalty", "lower"),
    ("avg_overload_penalty", "Overload Penalty", "lower"),
]

NODE_COLUMNS = [
    ("local_node_ratio", "Local", "#4c78a8"),
    ("edge_node_ratio", "Edge", "#f58518"),
    ("regional_node_ratio", "Regional", "#54a24b"),
    ("cloud_node_ratio", "Cloud", "#b279a2"),
]


def load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [m for m in MODEL_ORDER if m not in set(df["model"])]
    if missing:
        raise ValueError(f"Missing models in {csv_path}: {missing}")

    df = df[df["model"].isin(MODEL_ORDER)].copy()
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df.sort_values("model").reset_index(drop=True)


def annotate_bars(ax: plt.Axes, bars, values: np.ndarray, metric: str) -> None:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    span = vmax - vmin if vmax != vmin else max(abs(vmax), 1.0)

    for bar, value in zip(bars, values):
        if metric == "avg_reward":
            y = value - span * 0.04
            va = "top"
            text = f"{value:.1f}"
        elif value < 1:
            y = value + span * 0.04
            va = "bottom"
            text = f"{value:.3f}"
        else:
            y = value + span * 0.04
            va = "bottom"
            text = f"{value:.2f}"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            text,
            ha="center",
            va=va,
            fontsize=8,
            rotation=0,
        )


def set_metric_ylim(ax: plt.Axes, values: np.ndarray, metric: str) -> None:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    span = vmax - vmin if vmax != vmin else max(abs(vmax), 1.0)

    if metric == "avg_reward":
        ax.set_ylim(vmin - span * 0.18, vmax + span * 0.18)
    else:
        ax.set_ylim(0, vmax + span * 0.28 + 0.03)


def plot_metric_panel(df: pd.DataFrame, output_path: Path, tag: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    x = np.arange(len(df))
    labels = [MODEL_LABELS[m] for m in df["model"].astype(str)]
    colors = [MODEL_COLORS[m] for m in df["model"].astype(str)]

    for ax, (metric, title, direction) in zip(axes, METRICS):
        values = df[metric].astype(float).to_numpy()
        std_col = f"{metric}_std"
        errors = df[std_col].astype(float).to_numpy() if std_col in df.columns else None

        bars = ax.bar(
            x,
            values,
            yerr=errors,
            capsize=3,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            width=0.68,
            error_kw={"linewidth": 1.0, "capthick": 1.0},
        )

        ax.set_title(f"{title} ({direction} is better)", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        set_metric_ylim(ax, values, metric)
        annotate_bars(ax, bars, values, metric)

        if metric == "avg_reward":
            ax.axhline(0, color="#333333", linewidth=0.8, alpha=0.5)

    fig.suptitle(
        f"Ablation Performance on Mixed-Context Scenario ({tag} training steps)",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_node_distribution(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.2))

    x = np.arange(len(df))
    labels = [MODEL_LABELS[m] for m in df["model"].astype(str)]
    bottom = np.zeros(len(df), dtype=float)

    for col, label, color in NODE_COLUMNS:
        values = df[col].astype(float).to_numpy() * 100
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            width=0.68,
        )

        for i, value in enumerate(values):
            if value >= 8:
                ax.text(
                    x[i],
                    bottom[i] + value / 2,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
        bottom += values

    ax.set_title(
        "Node Selection Distribution by Ablation Group",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    ax.set_ylabel("Selection Ratio (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--tag", default="30k")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_summary(args.csv)

    metric_path = args.out_dir / f"ablation_{args.tag}_metric_panel.png"
    node_path = args.out_dir / f"ablation_{args.tag}_node_distribution.png"

    plot_metric_panel(df, metric_path, args.tag)
    plot_node_distribution(df, node_path)

    print(metric_path)
    print(node_path)


if __name__ == "__main__":
    main()
