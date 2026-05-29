"""
Export training curves and thesis-ready result tables for ablation experiments.

Usage:
    python scripts/export_ablation_training_curves_and_tables.py --tag 100k
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
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


MODEL_DIRS = {
    "Full-Scoring": "full_scoring",
    "No-Charging": "no_charging",
    "No-Congestion": "no_congestion",
    "Fixed-Reward": "fixed_reward",
    "No-RobotState": "no_robot_state",
    "No-HeuristicGate": "no_heuristic_gate",
    "No-GAT": "no_gat",
    "No-NodeScoring": "no_node_scoring",
}

MODEL_ORDER = [
    "No-Charging",
    "Full-Scoring",
    "No-Congestion",
    "Fixed-Reward",
    "No-RobotState",
    "No-HeuristicGate",
    "No-GAT",
    "No-NodeScoring",
]

MODEL_COLORS = {
    "Full-Scoring": "#1b9e77",
    "No-Charging": "#e7298a",
    "No-Congestion": "#66a61e",
    "Fixed-Reward": "#e6ab02",
    "No-RobotState": "#7570b3",
    "No-HeuristicGate": "#a6761d",
    "No-GAT": "#d95f02",
    "No-NodeScoring": "#666666",
}

COMPARABLE_CURVE_MODELS = [
    "Full-Scoring",
    "No-RobotState",
    "No-HeuristicGate",
    "No-GAT",
    "No-NodeScoring",
]

SUMMARY_COLUMNS = [
    ("model", "Model"),
    ("avg_reward", "Avg Reward"),
    ("avg_reward_std", "Reward Std"),
    ("avg_total_time", "Total Time"),
    ("avg_queue_delay", "Queue Delay"),
    ("avg_network_latency", "Network Latency"),
    ("avg_energy_cost", "Energy Cost"),
    ("avg_deadline_penalty", "Deadline Penalty"),
    ("avg_overload_penalty", "Overload Penalty"),
]

NODE_COLUMNS = [
    ("model", "Model"),
    ("local_node_ratio", "Local"),
    ("edge_node_ratio", "Edge"),
    ("regional_node_ratio", "Regional"),
    ("cloud_node_ratio", "Cloud"),
]


def order_models(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df.sort_values("model").reset_index(drop=True)


def load_training_curves(eval_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for model, dirname in MODEL_DIRS.items():
        path = eval_root / dirname / "evaluations.npz"
        if not path.exists():
            continue

        data = np.load(path)
        timesteps = data["timesteps"].astype(int)
        rewards = data["results"].astype(float)
        lengths = data["ep_lengths"].astype(float)
        step_rewards = rewards / np.maximum(lengths, 1.0)

        for i, timestep in enumerate(timesteps):
            rows.append(
                {
                    "model": model,
                    "timestep": int(timestep),
                    "avg_episode_reward": float(rewards[i].mean()),
                    "episode_reward_std": float(rewards[i].std(ddof=0)),
                    "avg_step_reward": float(step_rewards[i].mean()),
                    "step_reward_std": float(step_rewards[i].std(ddof=0)),
                    "avg_episode_length": float(lengths[i].mean()),
                }
            )

    if not rows:
        raise FileNotFoundError(f"No evaluations.npz files found under {eval_root}")
    df = pd.DataFrame(rows)
    df["model"] = pd.Categorical(df["model"], categories=list(MODEL_DIRS), ordered=True)
    return df.sort_values(["model", "timestep"]).reset_index(drop=True)


def plot_training_curve(curves: pd.DataFrame, output_path: Path, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.5))

    for model in MODEL_DIRS:
        sub = curves[curves["model"].astype(str) == model].copy()
        if sub.empty:
            continue
        x = sub["timestep"].to_numpy(dtype=float)
        y = sub["avg_step_reward"].to_numpy(dtype=float)
        color = MODEL_COLORS[model]

        ax.plot(
            x,
            y,
            label=model,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3.8,
        )

    ax.set_title(f"Ablation Training Curves ({tag} training steps)", fontweight="bold", pad=10)
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Evaluation Avg Reward per Step")
    ax.grid(alpha=0.25)
    ax.legend(ncols=2, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_comparable_training_curve(curves: pd.DataFrame, output_path: Path, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.2))

    for model in COMPARABLE_CURVE_MODELS:
        sub = curves[curves["model"].astype(str) == model].copy()
        if sub.empty:
            continue
        x = sub["timestep"].to_numpy(dtype=float)
        y = sub["avg_step_reward"].to_numpy(dtype=float)
        color = MODEL_COLORS[model]

        ax.plot(
            x,
            y,
            label=model,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=4.0,
        )

    ax.set_title(
        f"Comparable Model-Component Training Curves ({tag} training steps)",
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Evaluation Avg Reward per Step")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def make_summary_table(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    df = order_models(df)

    out = pd.DataFrame()
    for col, label in SUMMARY_COLUMNS:
        if col == "model":
            out[label] = df[col].astype(str)
        else:
            out[label] = df[col].astype(float).map(lambda x: f"{x:.3f}")
    return out


def make_node_table(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    df = order_models(df)

    out = pd.DataFrame()
    for col, label in NODE_COLUMNS:
        if col == "model":
            out[label] = df[col].astype(str)
        else:
            out[label] = (df[col].astype(float) * 100.0).map(lambda x: f"{x:.1f}%")
    return out


def save_table_image(table: pd.DataFrame, output_path: Path, title: str) -> None:
    fig_height = 0.55 * len(table) + 1.2
    fig_width = max(10.5, 1.15 * len(table.columns))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontweight="bold", pad=12)

    mpl_table = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(8.5)
    mpl_table.scale(1, 1.35)

    for (row, _), cell in mpl_table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(weight="bold")
        cell.set_edgecolor("#d0d0d0")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def to_markdown_table(table: pd.DataFrame) -> str:
    headers = [str(col) for col in table.columns]
    rows = [[str(value) for value in row] for row in table.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="100k")
    parser.add_argument("--eval-root", type=Path, default=Path("outputs/ablation_eval/mixed_context"))
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/results/ablation_mixed_context.csv"))
    parser.add_argument("--figure-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--table-dir", type=Path, default=Path("outputs/tables"))
    args = parser.parse_args()

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    curves = load_training_curves(args.eval_root)
    curve_csv = Path("outputs/results") / f"ablation_{args.tag}_training_curve.csv"
    curves.to_csv(curve_csv, index=False, encoding="utf-8-sig")

    curve_png = args.figure_dir / f"ablation_{args.tag}_training_curve.png"
    comparable_curve_png = args.figure_dir / f"ablation_{args.tag}_training_curve_model_components.png"
    plot_training_curve(curves, curve_png, args.tag)
    plot_comparable_training_curve(curves, comparable_curve_png, args.tag)

    summary_table = make_summary_table(args.summary_csv)
    node_table = make_node_table(args.summary_csv)

    summary_csv = args.table_dir / f"ablation_{args.tag}_summary_table.csv"
    summary_md = args.table_dir / f"ablation_{args.tag}_summary_table.md"
    summary_png = args.figure_dir / f"ablation_{args.tag}_summary_table.png"

    node_csv = args.table_dir / f"ablation_{args.tag}_node_distribution_table.csv"
    node_md = args.table_dir / f"ablation_{args.tag}_node_distribution_table.md"
    node_png = args.figure_dir / f"ablation_{args.tag}_node_distribution_table.png"

    summary_table.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_md.write_text(to_markdown_table(summary_table), encoding="utf-8")
    save_table_image(summary_table, summary_png, f"Ablation Result Table ({args.tag})")

    node_table.to_csv(node_csv, index=False, encoding="utf-8-sig")
    node_md.write_text(to_markdown_table(node_table), encoding="utf-8")
    save_table_image(node_table, node_png, f"Node Selection Distribution Table ({args.tag})")

    for path in [
        curve_csv,
        curve_png,
        comparable_curve_png,
        summary_csv,
        summary_md,
        summary_png,
        node_csv,
        node_md,
        node_png,
    ]:
        print(path)


if __name__ == "__main__":
    main()
