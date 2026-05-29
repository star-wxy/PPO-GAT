"""
Generate thesis overview diagrams:

1. Experimental system overview.
2. Method framework diagram.

Usage:
    python scripts/plot_thesis_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path("outputs/figures")

FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]

matplotlib.rcParams.update(
    {
        "font.sans-serif": FONT_CANDIDATES,
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


COLORS = {
    "green": "#1b9e77",
    "blue": "#4c78a8",
    "orange": "#f58518",
    "purple": "#7570b3",
    "red": "#d95f02",
    "gray": "#f5f5f5",
    "dark": "#2f2f2f",
    "line": "#9aa0a6",
}


def add_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    facecolor: str,
    edgecolor: str | None = None,
    title_color: str = "white",
    body_color: str = "#222222",
    title_size: int = 13,
    body_size: int = 10,
    radius: float = 0.03,
) -> None:
    x, y = xy
    edgecolor = edgecolor or facecolor
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=1.2,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)

    ax.text(
        x + width / 2,
        y + height * 0.72,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        x + width / 2,
        y + height * 0.36,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=body_color,
        linespacing=1.45,
    )


def add_light_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    accent: str,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.4,
        facecolor="white",
        edgecolor=accent,
    )
    ax.add_patch(patch)
    ax.plot([x + 0.02, x + width - 0.02], [y + height * 0.67, y + height * 0.67], color=accent, lw=2.0)
    ax.text(
        x + width / 2,
        y + height * 0.79,
        title,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=accent,
    )
    ax.text(
        x + width / 2,
        y + height * 0.35,
        body,
        ha="center",
        va="center",
        fontsize=9.5,
        color="#222222",
        linespacing=1.45,
    )


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["line"],
    lw: float = 1.8,
    rad: float = 0.0,
    mutation_scale: float = 14,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def plot_experiment_overview(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.8, 8.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.94,
        "实验体系总览：从基准验证到模块贡献分析",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=COLORS["dark"],
    )
    ax.text(
        0.5,
        0.895,
        "四套实验共同验证本文多机器人算力互联网调度方法的有效性、适应性与可解释性",
        ha="center",
        va="center",
        fontsize=12,
        color="#555555",
    )

    boxes = [
        (
            "实验一\n基准算法对比",
            "PPO vs Random / RoundRobin / GreedyCPU\n验证强化学习调度优于传统策略",
            COLORS["blue"],
        ),
        (
            "实验二\n模型结构对比",
            "PPO → PPO-GAT → PPO-GAT-Scoring\n验证图建模与节点评分的增益",
            COLORS["green"],
        ),
        (
            "实验三\n多场景适应性",
            "低电量 / 高负载 / 紧急任务\n高时延 / 混合场景",
            COLORS["orange"],
        ),
        (
            "实验四\n消融实验",
            "No-GAT / No-Scoring / No-Heuristic\nFixed-Reward / No-State\nNo-Congestion / No-Charging",
            COLORS["purple"],
        ),
    ]

    x0 = 0.055
    gap = 0.026
    w = (0.89 - 3 * gap) / 4
    y = 0.58
    h = 0.23
    for i, (title, body, color) in enumerate(boxes):
        x = x0 + i * (w + gap)
        add_box(
            ax,
            (x, y),
            w,
            h,
            title,
            body,
            facecolor=color,
            title_size=12,
            body_size=9.1,
            body_color="white",
        )
        if i < 3:
            add_arrow(ax, (x + w + 0.005, y + h / 2), (x + w + gap - 0.005, y + h / 2), color="#7f8c8d")

    evidence = [
        ("算法有效性", "PPO 获得更高奖励\n降低时延与过载"),
        ("结构贡献", "GAT 与 Scoring\n提升节点选择质量"),
        ("场景鲁棒性", "动态奖励适配不同\n任务与网络状态"),
        ("模块必要性", "去除关键模块后\n性能明显退化"),
    ]

    ey = 0.29
    eh = 0.16
    for i, (title, body) in enumerate(evidence):
        x = x0 + i * (w + gap)
        add_light_box(ax, (x, ey), w, eh, title, body, accent=boxes[i][2])
        add_arrow(ax, (x + w / 2, y - 0.015), (x + w / 2, ey + eh + 0.015), color=boxes[i][2], lw=1.5)

    add_box(
        ax,
        (0.245, 0.065),
        0.51,
        0.115,
        "论文实验结论",
        "本文方法能够在多机器人任务调度中综合优化任务完成时间、队列等待、能耗、截止期违约与节点负载均衡",
        facecolor="#263238",
        title_size=13,
        body_size=10.2,
        body_color="white",
    )

    for i in range(4):
        x = x0 + i * (w + gap) + w / 2
        add_arrow(ax, (x, ey - 0.015), (0.5, 0.19), color=COLORS["line"], lw=1.2, rad=0.08 * (1.5 - i))

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_method_framework(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15.2, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.94,
        "PPO-GAT-Scoring 多机器人算力调度方法框架",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=COLORS["dark"],
    )
    ax.text(
        0.5,
        0.895,
        "状态感知 → 图特征提取 → 节点评分 → 策略决策 → 调度执行 → 动态奖励反馈",
        ha="center",
        va="center",
        fontsize=12,
        color="#555555",
    )

    top_y = 0.64
    box_h = 0.18
    box_w = 0.142
    gap = 0.016
    start_x = 0.035

    pipeline = [
        (
            "多源状态输入",
            "任务类型 / 大小 / 截止期\n机器人电量 / 队列 / 本地算力\n节点负载 / 时延 / 拓扑",
            COLORS["blue"],
        ),
        (
            "图结构建模",
            "机器人-节点-网络拓扑\n构建调度图表示",
            COLORS["green"],
        ),
        (
            "GAT 特征提取",
            "聚合邻居节点信息\n获得拓扑感知表示",
            COLORS["green"],
        ),
        (
            "节点评分模块",
            "学习评分 + 启发式评分\n门控融合得到节点偏好",
            COLORS["orange"],
        ),
        (
            "PPO 策略网络",
            "根据状态特征输出\n目标计算节点动作",
            COLORS["purple"],
        ),
        (
            "调度执行",
            "选择 local / edge /\nregional / cloud 节点",
            COLORS["red"],
        ),
    ]

    centers = []
    for i, (title, body, color) in enumerate(pipeline):
        x = start_x + i * (box_w + gap)
        centers.append((x + box_w / 2, top_y + box_h / 2))
        add_box(
            ax,
            (x, top_y),
            box_w,
            box_h,
            title,
            body,
            facecolor=color,
            title_size=11,
            body_size=8.8,
            body_color="white",
        )
        if i < len(pipeline) - 1:
            add_arrow(ax, (x + box_w + 0.004, top_y + box_h / 2), (x + box_w + gap - 0.004, top_y + box_h / 2))

    # Lower layer: environment metrics and reward.
    add_light_box(
        ax,
        (0.08, 0.31),
        0.25,
        0.17,
        "环境状态更新",
        "任务完成时间、队列等待\n网络时延、能耗、过载状态\n机器人电量与充电状态",
        accent=COLORS["blue"],
    )
    add_light_box(
        ax,
        (0.375, 0.31),
        0.25,
        0.17,
        "多场景上下文识别",
        "低电量风险 / 高负载风险\n紧急程度 / 通信风险\n动态调整奖励权重",
        accent=COLORS["orange"],
    )
    add_light_box(
        ax,
        (0.67, 0.31),
        0.25,
        0.17,
        "动态奖励反馈",
        "综合奖励：时延、能耗\n截止期惩罚、过载惩罚\n负载均衡与松弛奖励",
        accent=COLORS["purple"],
    )

    add_arrow(ax, (0.87, top_y - 0.01), (0.795, 0.49), color=COLORS["red"], lw=1.8, rad=-0.1)
    add_arrow(ax, (0.33, 0.395), (0.375, 0.395), color=COLORS["line"], lw=1.7)
    add_arrow(ax, (0.625, 0.395), (0.67, 0.395), color=COLORS["line"], lw=1.7)
    add_arrow(ax, (0.67, 0.31), (0.47, top_y - 0.015), color=COLORS["purple"], lw=1.8, rad=-0.28)
    ax.text(
        0.52,
        0.505,
        "reward",
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["purple"],
        fontweight="bold",
    )

    add_box(
        ax,
        (0.21, 0.085),
        0.58,
        0.12,
        "优化目标",
        "在满足任务实时性与资源约束的同时，降低系统总时延、能耗、队列积压和节点过载，提升多机器人协同调度效率",
        facecolor="#263238",
        title_size=13,
        body_size=10.2,
        body_color="white",
    )

    add_arrow(ax, (0.205, 0.31), (0.37, 0.205), color=COLORS["line"], lw=1.1, rad=0.08)
    add_arrow(ax, (0.5, 0.31), (0.5, 0.205), color=COLORS["line"], lw=1.1)
    add_arrow(ax, (0.795, 0.31), (0.63, 0.205), color=COLORS["line"], lw=1.1, rad=-0.08)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    experiment_path = OUTPUT_DIR / "thesis_experiment_system_overview.png"
    method_path = OUTPUT_DIR / "thesis_method_framework.png"
    plot_experiment_overview(experiment_path)
    plot_method_framework(method_path)
    print(experiment_path)
    print(method_path)


if __name__ == "__main__":
    main()
