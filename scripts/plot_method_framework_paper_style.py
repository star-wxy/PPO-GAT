"""
Generate a paper-style method framework diagram.

The layout follows a black-and-white academic schematic style with two stages:
training and scheduling/testing. It highlights the technical contributions:
dynamic graph state modeling, GAT feature extraction, node scoring with heuristic
gate fusion, and scenario-aware dynamic reward.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


OUTPUT_DIR = Path("outputs/figures")

matplotlib.rcParams.update(
    {
        "font.sans-serif": [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


BLACK = "#111111"
GRAY = "#666666"
LIGHT = "#f7f7f7"
SOFT = "#fbfbfb"
RED = "#c62828"
BLUE = "#2f5f9f"


def box(ax, x, y, w, h, title, body="", *, dashed=False, lw=1.2, fs=10, title_fs=10.5, fill="white"):
    rect = Rectangle(
        (x, y),
        w,
        h,
        linewidth=lw,
        edgecolor=BLACK,
        facecolor=fill,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=title_fs, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center", fontsize=fs, linespacing=1.35)
    return rect


def light_box(ax, x, y, w, h, title, body="", *, fs=9.2, title_fs=10, fill=LIGHT):
    rect = Rectangle((x, y), w, h, linewidth=1.0, edgecolor=BLACK, facecolor=fill)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", fontsize=title_fs, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h * 0.35, body, ha="center", va="center", fontsize=fs, linespacing=1.3)
    return rect


def arrow(ax, start, end, *, color=BLACK, lw=1.2, rad=0.0, scale=12):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=scale,
        color=color,
        linewidth=lw,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)


def doc_icon(ax, x, y, w=0.035, h=0.075):
    ax.add_patch(Rectangle((x, y), w, h, linewidth=1.2, edgecolor=BLACK, facecolor="white"))
    ax.plot([x + w * 0.15, x + w * 0.82], [y + h * 0.75, y + h * 0.75], color=BLACK, lw=1)
    ax.plot([x + w * 0.15, x + w * 0.82], [y + h * 0.55, y + h * 0.55], color=BLACK, lw=1)
    ax.plot([x + w * 0.15, x + w * 0.62], [y + h * 0.35, y + h * 0.35], color=BLACK, lw=1)


def robot_icon(ax, cx, cy, s):
    ax.add_patch(Rectangle((cx - s * 0.33, cy - s * 0.20), s * 0.66, s * 0.42, ec=BLACK, fc="white", lw=1.0))
    ax.add_patch(Rectangle((cx - s * 0.24, cy + s * 0.23), s * 0.48, s * 0.26, ec=BLACK, fc=LIGHT, lw=0.9))
    ax.plot([cx, cx], [cy + s * 0.49, cy + s * 0.66], color=BLACK, lw=0.9)
    ax.add_patch(plt.Circle((cx, cy + s * 0.70), s * 0.045, ec=BLACK, fc=RED, lw=0.8))
    ax.add_patch(plt.Circle((cx - s * 0.12, cy + s * 0.36), s * 0.035, ec=BLACK, fc=BLACK, lw=0.8))
    ax.add_patch(plt.Circle((cx + s * 0.12, cy + s * 0.36), s * 0.035, ec=BLACK, fc=BLACK, lw=0.8))
    ax.plot([cx - s * 0.20, cx + s * 0.20], [cy - s * 0.01, cy - s * 0.01], color=BLACK, lw=0.8)
    ax.add_patch(plt.Circle((cx - s * 0.23, cy - s * 0.28), s * 0.07, ec=BLACK, fc=LIGHT, lw=0.8))
    ax.add_patch(plt.Circle((cx + s * 0.23, cy - s * 0.28), s * 0.07, ec=BLACK, fc=LIGHT, lw=0.8))


def compute_node_icon(ax, cx, cy, s, label, *, selected=False):
    edge = RED if selected else BLACK
    face = "#fff7f7" if selected else "white"
    ax.add_patch(plt.Circle((cx, cy), s * 0.30, ec=edge, fc=face, lw=1.15))
    ax.text(cx, cy + s * 0.03, label, ha="center", va="center", fontsize=7.0, color=edge if selected else BLACK, fontweight="bold")
    for i, height in enumerate([0.12, 0.20, 0.28]):
        bx = cx - s * 0.19 + i * s * 0.13
        ax.add_patch(Rectangle((bx, cy - s * 0.43), s * 0.07, height * s, ec=BLACK, fc=LIGHT, lw=0.55))


def mini_graph(ax, x, y, w, h):
    robots = [(x + w * 0.15, y + h * 0.70), (x + w * 0.15, y + h * 0.34)]
    nodes = [
        (x + w * 0.48, y + h * 0.80, "L", False),
        (x + w * 0.54, y + h * 0.48, "E", True),
        (x + w * 0.77, y + h * 0.70, "R", False),
        (x + w * 0.79, y + h * 0.30, "C", False),
    ]
    for i, (nx, ny, _, _) in enumerate(nodes):
        for nx2, ny2, _, _ in nodes[i + 1 :]:
            ax.plot([nx, nx2], [ny, ny2], color="#c8c8c8", lw=0.5, linestyle=":")
    for rx, ry in robots:
        for nx, ny, _, selected in nodes:
            ax.plot([rx + w * 0.065, nx - w * 0.025], [ry, ny], color=RED if selected else "#969696", lw=1.15 if selected else 0.58, alpha=0.95)
    for rx, ry in robots:
        robot_icon(ax, rx, ry, min(w, h) * 0.30)
    for nx, ny, label, selected in nodes:
        compute_node_icon(ax, nx, ny, min(w, h) * 0.34, label, selected=selected)
    ax.text(x + w * 0.47, y + h * 0.06, "robot-node graph", ha="center", fontsize=6.3, color=GRAY)


def mini_gat(ax, x, y, w, h):
    src = [
        (x + w * 0.12, y + h * 0.76),
        (x + w * 0.12, y + h * 0.54),
        (x + w * 0.12, y + h * 0.32),
    ]
    center = (x + w * 0.43, y + h * 0.54)
    weights = [0.75, 1.65, 1.05]
    for (sx, sy), weight in zip(src, weights):
        ax.plot([sx, center[0]], [sy, center[1]], color=RED if weight > 1.3 else "#8c8c8c", lw=weight)
        ax.add_patch(plt.Circle((sx, sy), min(w, h) * 0.045, ec=BLACK, fc="white", lw=0.9))
    ax.add_patch(plt.Circle(center, min(w, h) * 0.055, ec=RED, fc="#fff6f6", lw=1.2))
    ax.text(center[0], center[1], "α", ha="center", va="center", fontsize=7.0, color=RED, fontweight="bold")

    arrow(ax, (x + w * 0.51, y + h * 0.54), (x + w * 0.66, y + h * 0.54), color=RED, lw=1.0, scale=7)
    for i, height in enumerate([0.26, 0.52, 0.40, 0.70]):
        bx = x + w * (0.70 + i * 0.055)
        ax.add_patch(Rectangle((bx, y + h * 0.26), w * 0.032, h * height, ec=BLACK, fc=LIGHT, lw=0.8))
    ax.text(x + w * 0.39, y + h * 0.86, "attention aggregation", ha="center", fontsize=7.0, color=RED)
    ax.text(x + w * 0.77, y + h * 0.12, "node embedding", ha="center", fontsize=6.4, color=GRAY)


def mini_gat_clean(ax, x, y, w, h):
    src = [
        (x + w * 0.12, y + h * 0.76),
        (x + w * 0.12, y + h * 0.54),
        (x + w * 0.12, y + h * 0.32),
    ]
    center = (x + w * 0.43, y + h * 0.54)
    weights = [0.75, 1.45, 1.05]
    for (sx, sy), weight in zip(src, weights):
        ax.plot([sx, center[0] - w * 0.025], [sy, center[1]], color=RED if weight > 1.25 else "#8c8c8c", lw=weight)
        ax.add_patch(plt.Circle((sx, sy), min(w, h) * 0.045, ec=BLACK, fc="white", lw=0.9))
    ax.add_patch(plt.Circle(center, min(w, h) * 0.055, ec=RED, fc="#fff6f6", lw=1.2))
    ax.text(center[0], center[1], "a", ha="center", va="center", fontsize=7.0, color=RED, fontweight="bold")
    arrow(ax, (x + w * 0.52, y + h * 0.54), (x + w * 0.65, y + h * 0.54), color=RED, lw=0.9, scale=7)
    for i, height in enumerate([0.26, 0.52, 0.40, 0.70]):
        bx = x + w * (0.70 + i * 0.055)
        ax.add_patch(Rectangle((bx, y + h * 0.24), w * 0.032, h * height, ec=BLACK, fc=LIGHT, lw=0.8))
    ax.text(x + w * 0.37, y + h * 0.87, "attention weights", ha="center", fontsize=6.8, color=RED)
    ax.text(x + w * 0.77, y + h * 0.12, "node embedding", ha="center", fontsize=6.4, color=GRAY)


def mini_scoring(ax, x, y, w, h):
    labels = ["S_l", "S_h", "g", "S"]
    xs = [0.16, 0.16, 0.50, 0.82]
    ys = [0.68, 0.30, 0.50, 0.50]
    for label, rx, ry in zip(labels, xs, ys):
        ax.add_patch(Rectangle((x + w * rx - w * 0.055, y + h * ry - h * 0.09), w * 0.11, h * 0.18, lw=1, ec=BLACK, fc="white"))
        ax.text(x + w * rx, y + h * ry, label, ha="center", va="center", fontsize=8.5)
    arrow(ax, (x + w * 0.22, y + h * 0.67), (x + w * 0.44, y + h * 0.53), scale=8)
    arrow(ax, (x + w * 0.22, y + h * 0.31), (x + w * 0.44, y + h * 0.47), scale=8)
    arrow(ax, (x + w * 0.56, y + h * 0.50), (x + w * 0.74, y + h * 0.50), color=RED, scale=8)
    ax.text(x + w * 0.50, y + h * 0.14, "门控融合", ha="center", fontsize=8, color=RED)


def mini_scoring_rich(ax, x, y, w, h):
    ax.add_patch(Rectangle((x + w * 0.05, y + h * 0.60), w * 0.22, h * 0.18, ec=BLACK, fc="white", lw=0.9))
    ax.add_patch(Rectangle((x + w * 0.05, y + h * 0.28), w * 0.22, h * 0.18, ec=BLACK, fc="white", lw=0.9))
    ax.text(x + w * 0.16, y + h * 0.69, "S_learn", ha="center", va="center", fontsize=6.4)
    ax.text(x + w * 0.16, y + h * 0.37, "S_rule", ha="center", va="center", fontsize=6.4)
    ax.add_patch(plt.Circle((x + w * 0.48, y + h * 0.53), min(w, h) * 0.09, ec=RED, fc="#fff6f6", lw=1.1))
    ax.text(x + w * 0.48, y + h * 0.53, "gate\nsigmoid", ha="center", va="center", fontsize=5.7, color=RED)
    arrow(ax, (x + w * 0.27, y + h * 0.69), (x + w * 0.41, y + h * 0.57), scale=7)
    arrow(ax, (x + w * 0.27, y + h * 0.37), (x + w * 0.41, y + h * 0.49), scale=7)
    ax.add_patch(Rectangle((x + w * 0.66, y + h * 0.43), w * 0.16, h * 0.20, ec=BLACK, fc=LIGHT, lw=0.9))
    ax.text(x + w * 0.74, y + h * 0.53, "S_node", ha="center", va="center", fontsize=6.5)
    arrow(ax, (x + w * 0.57, y + h * 0.53), (x + w * 0.66, y + h * 0.53), color=RED, scale=7)
    ax.text(x + w * 0.47, y + h * 0.15, "S = g*S_learn + (1-g)*S_rule", ha="center", fontsize=6.0, color=RED)


def mini_scoring_clean(ax, x, y, w, h):
    ax.add_patch(Rectangle((x + w * 0.06, y + h * 0.64), w * 0.18, h * 0.16, ec=BLACK, fc="white", lw=0.9))
    ax.add_patch(Rectangle((x + w * 0.06, y + h * 0.30), w * 0.18, h * 0.16, ec=BLACK, fc="white", lw=0.9))
    ax.text(x + w * 0.15, y + h * 0.72, "S_l", ha="center", va="center", fontsize=7.2)
    ax.text(x + w * 0.15, y + h * 0.38, "S_h", ha="center", va="center", fontsize=7.2)

    gate_center = (x + w * 0.49, y + h * 0.54)
    ax.add_patch(Rectangle((gate_center[0] - w * 0.07, gate_center[1] - h * 0.085), w * 0.14, h * 0.17, ec=RED, fc="#fff7f7", lw=1.0))
    ax.text(gate_center[0], gate_center[1], "G", ha="center", va="center", fontsize=7.4, color=RED, fontweight="bold")

    arrow(ax, (x + w * 0.24, y + h * 0.72), (x + w * 0.42, y + h * 0.58), scale=6.5, lw=0.9)
    arrow(ax, (x + w * 0.24, y + h * 0.38), (x + w * 0.42, y + h * 0.50), scale=6.5, lw=0.9)

    ax.add_patch(Rectangle((x + w * 0.70, y + h * 0.44), w * 0.16, h * 0.20, ec=BLACK, fc=LIGHT, lw=0.9))
    ax.text(x + w * 0.78, y + h * 0.54, "S", ha="center", va="center", fontsize=7.4)
    arrow(ax, (x + w * 0.56, y + h * 0.54), (x + w * 0.70, y + h * 0.54), color=RED, scale=6.5, lw=0.9)
    ax.text(x + w * 0.48, y + h * 0.18, "gate fusion", ha="center", fontsize=6.4, color=RED)


def mini_reward(ax, x, y, w, h):
    ax.add_patch(Rectangle((x + w * 0.08, y + h * 0.18), w * 0.22, h * 0.55, ec=BLACK, fc="white", lw=1))
    ax.add_patch(Rectangle((x + w * 0.39, y + h * 0.18), w * 0.22, h * 0.55, ec=BLACK, fc="white", lw=1))
    ax.add_patch(Rectangle((x + w * 0.70, y + h * 0.18), w * 0.22, h * 0.55, ec=BLACK, fc="white", lw=1))
    ax.text(x + w * 0.19, y + h * 0.46, "场景\n识别", ha="center", va="center", fontsize=8)
    ax.text(x + w * 0.50, y + h * 0.46, "权重\n调整", ha="center", va="center", fontsize=8)
    ax.text(x + w * 0.81, y + h * 0.46, "奖励\n反馈", ha="center", va="center", fontsize=8)
    arrow(ax, (x + w * 0.30, y + h * 0.46), (x + w * 0.39, y + h * 0.46), scale=8)
    arrow(ax, (x + w * 0.61, y + h * 0.46), (x + w * 0.70, y + h * 0.46), color=RED, scale=8)


def mini_state_encoder(ax, x, y, w, h):
    labels = ["Task", "Robot", "Node", "Topo"]
    for i, label in enumerate(labels):
        rx = x + w * 0.05
        ry = y + h * (0.74 - i * 0.20)
        ax.add_patch(Rectangle((rx, ry), w * 0.24, h * 0.13, ec=BLACK, fc="white", lw=0.9))
        ax.text(rx + w * 0.12, ry + h * 0.065, label, ha="center", va="center", fontsize=6.8)
        arrow(ax, (rx + w * 0.25, ry + h * 0.065), (x + w * 0.54, y + h * 0.50), lw=0.8, scale=6)

    for i in range(7):
        ax.add_patch(Rectangle((x + w * (0.58 + i * 0.045), y + h * 0.41), w * 0.028, h * 0.18, ec=BLACK, fc=LIGHT, lw=0.8))
    ax.text(x + w * 0.72, y + h * 0.25, "state vector", ha="center", fontsize=6.8, color=GRAY)


def mini_topology_panel(ax, x, y, w, h):
    robot_nodes = [(x + w * 0.18, y + h * 0.70), (x + w * 0.18, y + h * 0.35)]
    compute_nodes = [
        (x + w * 0.50, y + h * 0.80),
        (x + w * 0.50, y + h * 0.52),
        (x + w * 0.50, y + h * 0.24),
        (x + w * 0.82, y + h * 0.60),
    ]
    for rn in robot_nodes:
        for cn in compute_nodes[:3]:
            ax.plot([rn[0], cn[0]], [rn[1], cn[1]], color="#a0a0a0", lw=0.65)
    for i in range(3):
        ax.plot([compute_nodes[i][0], compute_nodes[3][0]], [compute_nodes[i][1], compute_nodes[3][1]], color="#a0a0a0", lw=0.65)
    for rn in robot_nodes:
        ax.add_patch(plt.Circle(rn, min(w, h) * 0.055, ec=BLACK, fc="white", lw=1.0))
        ax.text(rn[0], rn[1], "R", ha="center", va="center", fontsize=6.5)
    node_labels = ["L", "E", "E", "C"]
    for cn, label in zip(compute_nodes, node_labels):
        face = LIGHT if label != "C" else "#eeeeee"
        ax.add_patch(Rectangle((cn[0] - w * 0.035, cn[1] - h * 0.045), w * 0.07, h * 0.09, ec=BLACK, fc=face, lw=1.0))
        ax.text(cn[0], cn[1], label, ha="center", va="center", fontsize=6.6)
    ax.text(x + w * 0.50, y + h * 0.05, "dynamic graph", ha="center", fontsize=6.8, color=GRAY)


def mini_actor_critic(ax, x, y, w, h):
    # Shared encoder.
    for i in range(4):
        ax.add_patch(plt.Circle((x + w * 0.12, y + h * (0.25 + i * 0.16)), min(w, h) * 0.028, ec=BLACK, fc="white", lw=0.8))
    for i in range(3):
        ax.add_patch(plt.Circle((x + w * 0.38, y + h * (0.33 + i * 0.16)), min(w, h) * 0.028, ec=BLACK, fc=LIGHT, lw=0.8))
    for i in range(4):
        y1 = y + h * (0.25 + i * 0.16)
        for j in range(3):
            y2 = y + h * (0.33 + j * 0.16)
            ax.plot([x + w * 0.12, x + w * 0.38], [y1, y2], color="#aaaaaa", lw=0.45)

    ax.add_patch(Rectangle((x + w * 0.62, y + h * 0.56), w * 0.25, h * 0.18, ec=BLACK, fc="white", lw=0.9))
    ax.add_patch(Rectangle((x + w * 0.62, y + h * 0.22), w * 0.25, h * 0.18, ec=BLACK, fc="white", lw=0.9))
    ax.text(x + w * 0.745, y + h * 0.65, "Actor", ha="center", va="center", fontsize=6.8)
    ax.text(x + w * 0.745, y + h * 0.31, "Critic", ha="center", va="center", fontsize=6.8)
    arrow(ax, (x + w * 0.43, y + h * 0.52), (x + w * 0.62, y + h * 0.65), lw=0.8, scale=6)
    arrow(ax, (x + w * 0.43, y + h * 0.45), (x + w * 0.62, y + h * 0.31), lw=0.8, scale=6)
    ax.text(x + w * 0.50, y + h * 0.08, "policy / value", ha="center", fontsize=6.8, color=GRAY)


def mini_action_selector(ax, x, y, w, h):
    labels = ["L", "E", "R", "C"]
    heights = [0.38, 0.66, 0.52, 0.30]
    selected_idx = 1
    selected_center = None
    selected_top = None
    for i, (label, height) in enumerate(zip(labels, heights)):
        bx = x + w * (0.10 + i * 0.21)
        by = y + h * 0.18
        bw = w * 0.115
        bh = h * height
        is_selected = i == selected_idx
        ax.add_patch(
            Rectangle(
                (bx, by),
                bw,
                bh,
                ec=RED if is_selected else BLACK,
                fc="#fff5f5" if is_selected else LIGHT,
                lw=1.25 if is_selected else 0.85,
            )
        )
        ax.text(bx + bw / 2, y + h * 0.055, label, ha="center", va="center", fontsize=6.2)
        if is_selected:
            selected_center = bx + bw / 2
            selected_top = by + bh

    if selected_center is None or selected_top is None:
        return

    ax.annotate(
        "select",
        xy=(selected_center, selected_top + h * 0.015),
        xytext=(x + w * 0.68, y + h * 0.96),
        ha="center",
        va="center",
        fontsize=6.4,
        color=RED,
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="->",
            color=RED,
            lw=0.65,
            shrinkA=1,
            shrinkB=1,
            mutation_scale=9,
            connectionstyle="arc3,rad=0.0",
        ),
    )


def mini_reward_weights(ax, x, y, w, h):
    names = ["delay", "energy", "load", "deadline"]
    vals = [0.55, 0.42, 0.72, 0.62]
    for i, (name, value) in enumerate(zip(names, vals)):
        yy = y + h * (0.74 - i * 0.19)
        ax.text(x + w * 0.04, yy, name, ha="left", va="center", fontsize=6.4)
        ax.add_patch(Rectangle((x + w * 0.33, yy - h * 0.035), w * 0.50, h * 0.055, ec=BLACK, fc="white", lw=0.6))
        ax.add_patch(Rectangle((x + w * 0.33, yy - h * 0.035), w * 0.50 * value, h * 0.055, ec="none", fc="#d9d9d9"))
    ax.text(x + w * 0.54, y + h * 0.06, "adaptive weights", ha="center", fontsize=6.6, color=GRAY)


def plot(output_path: Path):
    fig, ax = plt.subplots(figsize=(16.2, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.968, "PPO-GAT-Scoring 多机器人算力调度方法框架", ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(0.5, 0.938, "动态图建模、图注意力特征提取、节点评分门控与多场景动态奖励联合优化", ha="center", va="center", fontsize=10.5, color=GRAY)

    # Stage separators and labels.
    ax.plot([0.075, 0.075], [0.53, 0.915], color=BLACK, lw=1.2, linestyle=(0, (6, 5)))
    ax.plot([0.925, 0.925], [0.53, 0.915], color=BLACK, lw=1.2, linestyle=(0, (6, 5)))
    ax.text(0.050, 0.875, "输入", ha="center", fontsize=10, fontweight="bold")
    ax.text(0.965, 0.895, "输出", ha="center", fontsize=10, fontweight="bold")
    doc_icon(ax, 0.032, 0.765)

    box(ax, 0.095, 0.75, 0.14, 0.12, "步骤一：多源状态编码", "", fs=8.5, title_fs=9.5, fill=SOFT)
    box(ax, 0.095, 0.585, 0.14, 0.12, "步骤二：动态图构建", "", fs=8.5, title_fs=9.5, fill=SOFT)
    mini_state_encoder(ax, 0.108, 0.755, 0.115, 0.072)
    mini_topology_panel(ax, 0.108, 0.592, 0.115, 0.082)
    arrow(ax, (0.060, 0.807), (0.095, 0.81))
    arrow(ax, (0.165, 0.75), (0.165, 0.705))

    # Innovation modules.
    ax.add_patch(
        Rectangle(
            (0.255, 0.545),
            0.47,
            0.35,
            linewidth=1.2,
            edgecolor=BLACK,
            facecolor="none",
            linestyle="--",
        )
    )
    ax.text(0.49, 0.883, "核心网络结构与技术创新", ha="center", va="center", fontsize=10.6, fontweight="bold")
    light_box(ax, 0.275, 0.715, 0.13, 0.135, "动态图状态建模", "节点负载、时延、能耗\n随调度过程动态更新", fs=8.0, title_fs=9.0)
    mini_graph(ax, 0.292, 0.585, 0.095, 0.095)
    ax.text(0.34, 0.565, "机器人-节点拓扑", ha="center", fontsize=8)

    light_box(ax, 0.425, 0.715, 0.13, 0.135, "GAT 特征提取", "注意力聚合邻居信息\n学习拓扑感知表示", fs=8.0, title_fs=9.0)
    mini_gat_clean(ax, 0.432, 0.585, 0.115, 0.095)

    light_box(ax, 0.575, 0.715, 0.13, 0.135, "节点评分机制", "学习评分 + 启发式评分\n门控融合选择节点", fs=8.0, title_fs=9.0)
    mini_scoring_clean(ax, 0.582, 0.585, 0.115, 0.095)

    arrow(ax, (0.405, 0.782), (0.425, 0.782))
    arrow(ax, (0.555, 0.782), (0.575, 0.782))

    box(ax, 0.745, 0.75, 0.13, 0.12, "步骤三：PPO策略学习", "", fs=8.4, title_fs=9.5, fill=SOFT)
    mini_actor_critic(ax, 0.753, 0.755, 0.113, 0.070)
    arrow(ax, (0.725, 0.782), (0.745, 0.81))
    box(ax, 0.775, 0.56, 0.09, 0.12, "节点动作", "", fs=8.2, title_fs=9.2, fill=SOFT)
    mini_action_selector(ax, 0.779, 0.565, 0.082, 0.066)
    arrow(ax, (0.81, 0.75), (0.82, 0.68))
    arrow(ax, (0.865, 0.62), (0.925, 0.62))
    ax.text(0.957, 0.635, "调度决策\n目标算力节点", ha="center", va="center", fontsize=9)

    # Reward and feedback block.
    ax.add_patch(
        Rectangle(
            (0.255, 0.305),
            0.61,
            0.17,
            linewidth=1.2,
            edgecolor=BLACK,
            facecolor="none",
            linestyle="--",
        )
    )
    ax.text(0.56, 0.458, "步骤四：多场景动态奖励反馈", ha="center", va="center", fontsize=10.2, fontweight="bold")
    light_box(ax, 0.275, 0.34, 0.15, 0.082, "场景上下文识别", "低电量 / 高负载 / 紧急任务 / 高时延", fs=7.9, title_fs=8.8)
    light_box(ax, 0.445, 0.34, 0.15, 0.082, "动态权重调整", "", fs=7.9, title_fs=8.8)
    light_box(ax, 0.615, 0.34, 0.15, 0.082, "综合奖励计算", "时延 + 能耗 + 过载 + 负载均衡 + 松弛奖励", fs=7.9, title_fs=8.8)
    mini_reward(ax, 0.775, 0.33, 0.075, 0.10)
    mini_reward_weights(ax, 0.452, 0.342, 0.135, 0.058)
    arrow(ax, (0.425, 0.381), (0.445, 0.381))
    arrow(ax, (0.595, 0.381), (0.615, 0.381))
    arrow(ax, (0.765, 0.381), (0.775, 0.381))
    arrow(ax, (0.815, 0.475), (0.815, 0.56), color=RED, lw=1.3)
    ax.text(0.83, 0.512, "reward", color=RED, fontsize=8.5, fontweight="bold")

    # Testing/deployment phase.
    ax.plot([0.075, 0.075], [0.06, 0.24], color=BLACK, lw=1.2, linestyle=(0, (6, 5)))
    ax.plot([0.925, 0.925], [0.06, 0.24], color=BLACK, lw=1.2, linestyle=(0, (6, 5)))
    ax.text(0.050, 0.213, "输入", ha="center", fontsize=10, fontweight="bold")
    ax.text(0.965, 0.213, "输出", ha="center", fontsize=10, fontweight="bold")
    doc_icon(ax, 0.033, 0.130, w=0.03, h=0.06)

    y = 0.105
    h = 0.10
    w = 0.13
    x_positions = [0.095, 0.255, 0.415, 0.575, 0.735]
    titles = [
        "实时状态输入",
        "图表示更新",
        "GAT + Scoring",
        "PPO策略推理",
        "调度结果输出",
    ]
    bodies = [
        "任务与资源状态",
        "更新节点负载\n与拓扑特征",
        "计算节点优先级\n形成动作偏好",
        "选择目标节点\n执行调度动作",
        "节点类型与编号\n性能反馈",
    ]
    for i, (x, t, b) in enumerate(zip(x_positions, titles, bodies)):
        box(ax, x, y, w, h, f"步骤{i + 1}：{t}", b, fs=7.8, title_fs=8.6, fill=SOFT)
        if i < len(x_positions) - 1:
            arrow(ax, (x + w, y + h / 2), (x_positions[i + 1], y + h / 2))
    arrow(ax, (0.865, y + h / 2), (0.925, y + h / 2))
    ax.text(0.5, 0.255, "(a) 训练阶段", ha="center", fontsize=10)
    ax.text(0.5, 0.028, "(b) 调度/测试阶段", ha="center", fontsize=10)

    # Red highlight callouts for innovations.
    ax.add_patch(Rectangle((0.568, 0.555), 0.142, 0.305, linewidth=1.0, edgecolor=RED, facecolor="none"))
    ax.text(0.64, 0.905, "创新点1：显式节点评分与启发式门控", ha="center", fontsize=8.5, color=RED)
    ax.add_patch(Rectangle((0.255, 0.305), 0.61, 0.17, linewidth=1.0, edgecolor=RED, facecolor="none"))
    ax.text(0.56, 0.285, "创新点2：面向低电量、高负载、紧急任务和高时延的动态奖励机制", ha="center", fontsize=8.5, color=RED)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "thesis_method_framework_paper_style.png"
    plot(path)
    print(path)


if __name__ == "__main__":
    main()
