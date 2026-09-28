"""Generate figures for the baseline cross-dataset results (full-data run)."""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9})

DS = ["CIC-IDS-2017", "UNSW-NB15", "TON_IoT"]
METRICS = ["accuracy", "balanced_accuracy", "macro_f1"]


def load(metric):
    out = {}
    for path in glob.glob(os.path.join(RESULTS, f"baseline_*_{metric}.csv")):
        base = os.path.basename(path)
        # avoid matching balanced_accuracy when metric == accuracy
        if metric == "accuracy" and base.endswith("_balanced_accuracy.csv"):
            continue
        model = base[len("baseline_"):-len(f"_{metric}.csv")]
        out[model] = pd.read_csv(path, index_col=0).reindex(index=DS, columns=DS)
    return out


def fig_same_vs_cross():
    acc, bal = load("accuracy"), load("balanced_accuracy")
    models = sorted(acc.keys())
    same = [np.mean([acc[m].loc[d, d] for d in DS]) for m in models]
    cross = [np.mean([acc[m].loc[s, t] for s in DS for t in DS if s != t]) for m in models]
    same_b = [np.mean([bal[m].loc[d, d] for d in DS]) for m in models]
    cross_b = [np.mean([bal[m].loc[s, t] for s in DS for t in DS if s != t]) for m in models]

    x = np.arange(len(models))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (a, b, title) in zip(axes, [(same, cross, "Accuracy"),
                                        (same_b, cross_b, "Balanced accuracy")]):
        ax.bar(x - w / 2, a, w, label="Same-dataset", color="#2ca02c")
        ax.bar(x + w / 2, b, w, label="Cross-dataset (mean)", color="#d62728")
        ax.axhline(0.5, color="#888", linestyle="--", linewidth=1)
        ax.text(len(models) - 0.5, 0.505, "random\n(0.5)", fontsize=6, color="#555", ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        for xi, v in zip(x - w / 2, a):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=6)
        for xi, v in zip(x + w / 2, b):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=6)
    fig.suptitle("Same-dataset vs cross-dataset performance (full datasets)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_baseline_same_vs_cross.png"))
    plt.close(fig)


def fig_heatmaps():
    acc = load("accuracy")
    models = sorted(acc.keys())
    n = len(models)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.3 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, m in zip(axes, models):
        sns.heatmap(acc[m].astype(float), annot=True, fmt=".2f", cmap="RdYlGn",
                    vmin=0, vmax=1, cbar=False, ax=ax, square=True,
                    xticklabels=["CIC", "UNSW", "TON"], yticklabels=["CIC", "UNSW", "TON"])
        ax.set_title(f"{m.upper()} — accuracy", fontweight="bold", fontsize=9)
        ax.set_xlabel("tested on")
        ax.set_ylabel("trained on")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Cross-dataset accuracy matrices (rows = trained on, cols = tested on)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_baseline_heatmaps.png"))
    plt.close(fig)


def fig_xgb_balanced():
    bal = load("balanced_accuracy")
    if "xgb" not in bal:
        return
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    sns.heatmap(bal["xgb"].astype(float), annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0, vmax=1, cbar=True, ax=ax, square=True,
                xticklabels=["CIC", "UNSW", "TON"], yticklabels=["CIC", "UNSW", "TON"])
    ax.set_title("XGBoost — balanced accuracy", fontweight="bold")
    ax.set_xlabel("tested on")
    ax.set_ylabel("trained on")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_baseline_xgb_balanced.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_same_vs_cross()
    fig_heatmaps()
    fig_xgb_balanced()
    print("baseline figures written to", FIG)
