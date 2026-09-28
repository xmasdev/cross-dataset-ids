import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = r"F:\BTP\reports\figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9})


def pipeline():
    stages = [
        ("Raw datasets\nCIC-IDS-2017 | UNSW-NB15 | TON_IoT", "#cfe2f3"),
        ("Phase 1: Cleaning\nNaN/Inf fix, label normalisation, drop identity features", "#d9ead3"),
        ("Phase 2: Feature harmonisation\ncommon semantic subspace (~15-25 dims)", "#fff2cc"),
        ("Phase 3: Baselines\nsame-dataset vs cross-dataset (3x3 matrix)", "#fce5cd"),
        ("Phase 4: Metaheuristic FS\nGA / PSO / DE, fitness = cross-dataset F1", "#f4cccc"),
        ("Phase 5: Baseline FS comparison\nMI, RF-importance, PCA, RFE, LASSO", "#e6d0f0"),
        ("Phase 6: Analysis\ntransferable features & attack classes", "#d0e0f0"),
        ("Phase 7: Guidelines\nactionable recommendations for generalisable IDS", "#d9d2e9"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, len(stages) * 1.15 + 0.3)
    ax.axis("off")
    y = len(stages) * 1.15
    boxes = []
    for text, color in stages:
        y -= 1.05
        box = FancyBboxPatch((1.2, y), 7.6, 0.85, boxstyle="round,pad=0.08,rounding_size=0.12",
                             linewidth=1.1, edgecolor="#444", facecolor=color)
        ax.add_patch(box)
        ax.text(5.0, y + 0.42, text, ha="center", va="center", fontsize=8.6)
        boxes.append(y)
    for i in range(len(boxes) - 1):
        ax.add_patch(FancyArrowPatch((5.0, boxes[i] - 0.02), (5.0, boxes[i + 1] + 0.87),
                                     arrowstyle="-|>", mutation_scale=11, color="#333", linewidth=1.2))
    # feedback arrow (FS -> baseline)
    ax.add_patch(FancyArrowPatch((8.9, boxes[5] + 0.42), (8.9, boxes[3] + 0.42),
                                 connectionstyle="arc3,rad=-0.5", arrowstyle="-|>",
                                 mutation_scale=10, color="#b45f06", linewidth=1.1, linestyle="--"))
    ax.text(9.55, (boxes[3] + boxes[5]) / 2 + 0.42, "evaluate\nsame protocol", rotation=90,
            ha="center", va="center", fontsize=7, color="#b45f06")
    ax.set_title("Proposed end-to-end methodology", fontweight="bold", fontsize=11)
    fig.savefig(os.path.join(FIG, "fig_pipeline.png"))
    plt.close(fig)


def gantt():
    tasks = [
        ("Study of project & objective", 1, 2, "#cfe2f3"),
        ("Literature survey", 2, 3, "#d9ead3"),
        ("Dataset analysis & cleaning", 3, 3, "#fff2cc"),
        ("Feature harmonisation", 5, 2, "#fce5cd"),
        ("Cross-dataset baselines", 6, 2, "#f4cccc"),
        ("Metaheuristic feature selection", 8, 4, "#e6d0f0"),
        ("Baseline FS comparison", 11, 2, "#d0e0f0"),
        ("Analysis & ablations", 12, 2, "#d9d2e9"),
        ("Report & guidelines", 14, 1, "#efefef"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    for i, (name, start, dur, color) in enumerate(tasks):
        ax.barh(i, dur, left=start, color=color, edgecolor="#444", height=0.6)
        ax.text(start + dur / 2, i, f"W{start}-{start+dur-1}", ha="center", va="center", fontsize=7)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t[0] for t in tasks], fontsize=8)
    ax.invert_yaxis()
    ax.set_xticks(range(1, 16))
    ax.set_xlabel("Week")
    ax.set_xlim(0.5, 15.5)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_title("Project Gantt chart (one semester)", fontweight="bold", fontsize=11)
    font_kw = dict(fontsize=7, color="#b00000", fontweight="bold")
    ax.axvline(6.5, color="#b00000", linestyle="--", linewidth=1)
    ax.text(6.6, -0.9, "WE ARE HERE (mid-semester)", **font_kw)
    fig.savefig(os.path.join(FIG, "fig_gantt.png"))
    plt.close(fig)


def shared_subspace():
    ds = ["CIC-IDS-2017", "UNSW-NB15", "TON_IoT"]
    total = [78, 42, 42]
    shared = [6, 6, 6]
    uniq = [t - s for t, s in zip(total, shared)]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = range(len(ds))
    ax.bar(x, shared, color="#2ca02c", label="Semantically shared (~6 raw / ~15-25 encoded)")
    ax.bar(x, uniq, bottom=shared, color="#d62728", label="Dataset-specific")
    for i, (s, u) in enumerate(zip(shared, uniq)):
        ax.text(i, s / 2, str(s), ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        ax.text(i, s + u / 2, str(u), ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        ax.text(i, s + u + 1.5, f"total {s+u}", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(ds)
    ax.set_ylabel("Feature columns")
    ax.set_title("Shared vs dataset-specific features (schema mismatch)", fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_ylim(0, 92)
    fig.savefig(os.path.join(FIG, "fig_shared_subspace.png"))
    plt.close(fig)


if __name__ == "__main__":
    pipeline(); gantt(); shared_subspace()
    print("extra figures written")
