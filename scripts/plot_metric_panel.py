"""
PPO-GAT 指标对比图 — 一张大图 2×3 小图，每个指标一个小图。
针对值跨度大的指标优化纵坐标，避免小柱子被压扁。

用法:
    pip install matplotlib pandas numpy
    python scripts/plot_metric_panel.py
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── 风格 ──────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

COLORS = {
    "PPO-GAT-Scoring": "#2E86AB",
    "PPO-GAT-Naive":   "#A23B72",
    "PPO-Baseline":    "#F18F01",
}
SORT_ORDER = ["PPO-GAT-Scoring", "PPO-GAT-Naive", "PPO-Baseline"]

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 6 个指标 ──────────────────────────────────────────
# (列名, 显示名, 越小越好?, 纵轴策略)
#   ymode: "linear0"=从0开始线性, "linear_cluster"=从最小值附近开始, "log"=对数
METRICS = [
    ("avg_reward",          "Avg Reward",          False, "linear_free"),
    ("avg_total_time",      "Total Time (s)",       True,  "linear0"),
    ("avg_queue_delay",     "Queue Delay (s)",      True,  "linear0"),
    ("avg_deadline_penalty","Deadline Penalty",     True,  "linear0"),
    ("avg_overload_penalty","Overload Penalty",     True,  "linear0"),
    ("avg_energy_cost",     "Energy Cost",          True,  "linear_cluster"),
]

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["model"].isin(SORT_ORDER)].copy()
    df["model"] = pd.Categorical(df["model"], categories=SORT_ORDER, ordered=True)
    df = df.sort_values("model").reset_index(drop=True)
    return df


def plot_panel(df: pd.DataFrame, output_name: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, (col, label, lower_better, ymode) in enumerate(METRICS):
        ax = axes[idx]
        vals = df[col].values.astype(float)
        std_col = f"{col}_std"
        stds = df[std_col].values.astype(float) if std_col in df.columns else None

        models_short = [m.replace("PPO-", "") for m in SORT_ORDER]
        bar_colors = [COLORS[m] for m in SORT_ORDER]

        # ── 画柱子 ──
        bars = ax.bar(
            models_short, vals,
            yerr=stds, capsize=4,
            color=bar_colors,
            edgecolor="white", linewidth=0.7,
            width=0.5,
            error_kw={"linewidth": 1.2, "capthick": 1.2},
        )

        # ── 纵轴策略 ──
        vmin, vmax = vals.min(), vals.max()
        val_range = vmax - vmin if vmax != vmin else 1.0
        label_suffix = ""

        if ymode == "linear0":
            ax.set_ylim(bottom=0)
            top_pad = vmax * 0.35
            ax.set_ylim(top=vmax + top_pad)

        elif ymode == "linear_free":
            # avg_reward: 自动，带 padding
            pad = val_range * 0.2
            ax.set_ylim(bottom=vmin - pad, top=vmax + pad)

        elif ymode == "linear_cluster":
            # 值很接近时从最小值附近开始，放大差异
            bottom = max(0, vmin - val_range * 0.5)
            top = vmax + val_range * 0.4
            ax.set_ylim(bottom=bottom, top=top)

        # ── 柱上标数值 ──
        for bar, v in zip(bars, vals):
            if ymode == "linear0" or ymode == "linear_cluster":
                offset = max(vmax * 0.03, 0.02)
                va = "bottom"
                y_pos = bar.get_height() + offset
            else:
                offset = val_range * 0.03 + 0.01
                va = "bottom" if v >= 0 else "top"
                y_pos = bar.get_height() + offset if v >= 0 else bar.get_height() - offset

            fmt = ".4f" if "penalty" in col else ".3f"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{v:{fmt}}",
                ha="center", va=va, fontsize=10, fontweight="bold",
            )

        # ── 额外注解：当 Scoring 接近 0 时加标注 ──
        if col in ("avg_deadline_penalty", "avg_queue_delay", "avg_overload_penalty"):
            scoring_val = vals[0]
            if scoring_val < vmax * 0.05:
                ax.annotate(
                    f"≈ {scoring_val:.4f}",
                    xy=(0, scoring_val),
                    xytext=(0, scoring_val + vmax * 0.15),
                    ha="center", va="bottom",
                    fontsize=9, color=COLORS[SORT_ORDER[0]],
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=COLORS[SORT_ORDER[0]], lw=0.8),
                )

        ax.set_title(f"{label}{label_suffix}", fontsize=13, fontweight="bold", pad=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "PPO-GAT Comparison — All Metrics (20 Robots, 10 Nodes)",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / output_name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {output_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/results/ppo_gat_comparison_20r_10n.csv")
    parser.add_argument("--best-csv", default="outputs/results/ppo_gat_comparison_20r_10n_best.csv")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"[错误] 找不到 {args.csv}")
        return
    df = load_data(args.csv)
    print(f">>> 读取 {Path(args.csv).name}")
    plot_panel(df, "metric_panel.png")

    if Path(args.best_csv).exists():
        df_best = load_data(args.best_csv)
        print(f">>> 读取 {Path(args.best_csv).name}")
        plot_panel(df_best, "metric_panel_best.png")

    print(f"\n✓ 完成，图片保存在 {OUTPUT_DIR}/")
    for p in sorted(OUTPUT_DIR.glob("metric_panel*.png")):
        print(f"   {p}")


if __name__ == "__main__":
    main()
