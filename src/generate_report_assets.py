import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"F:\BTP\datasets"
FIG = r"F:\BTP\reports\figures"
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
})

CIC_COLORS = {"BENIGN": "#2ca02c"}

def cicids_class_dist():
    d = os.path.join(BASE, "CIC-IDS-2017", "MachineLearningCVE")
    agg = {}
    for f in sorted(os.listdir(d)):
        df = pd.read_csv(os.path.join(d, f), usecols=lambda c: c.strip() == "Label", low_memory=False)
        df.columns = ["Label"]
        for k, v in df["Label"].value_counts().items():
            agg[k] = agg.get(k, 0) + v
    return agg

def unsw_attack_dist():
    d = os.path.join(BASE, "UNSW-NB15")
    agg = {}
    for f in ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]:
        df = pd.read_csv(os.path.join(d, f), usecols=["attack_cat"], low_memory=False)
        for k, v in df["attack_cat"].value_counts().items():
            agg[k] = agg.get(k, 0) + v
    return agg

def ton_type_dist():
    p = os.path.join(BASE, "TON_IOT network train-test", "Train_Test_Network_dataset", "train_test_network.csv")
    df = pd.read_csv(p, usecols=["type"], low_memory=False)
    return df["type"].value_counts().to_dict()

def barh_plot(data, title, xlabel, path, color_map=None):
    labels = list(data.keys())
    vals = list(data.values())
    order = np.argsort(vals)
    labels = [labels[i] for i in order]
    vals = [vals[i] for i in order]
    colors = [color_map.get(l, "#1f77b4") for l in labels] if color_map else "#1f77b4"
    fig, ax = plt.subplots(figsize=(7, max(3, 0.34 * len(labels) + 1)))
    ax.barh(labels, vals, color=colors)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

if __name__ == "__main__":
    cic = cicids_class_dist()
    unsw = unsw_attack_dist()
    ton = ton_type_dist()

    # charts
    barh_plot(cic, "CIC-IDS-2017: Label distribution (all 8 days, log scale)",
              "Number of flows (log scale)", os.path.join(FIG, "fig_cicids_class_dist.png"))
    barh_plot(unsw, "UNSW-NB15: attack_cat distribution (train + test, log scale)",
              "Number of records (log scale)", os.path.join(FIG, "fig_unsw_class_dist.png"))
    barh_plot(ton, "TON_IoT: attack type distribution (log scale)",
              "Number of records (log scale)", os.path.join(FIG, "fig_ton_class_dist.png"))

    # feature count comparison
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ds = ["CIC-IDS-2017", "UNSW-NB15", "TON_IoT"]
    feats = [78, 42, 42]
    cols = ["#d62728", "#ff7f0e", "#9467bd"]
    bars = ax.bar(ds, feats, color=cols)
    ax.set_ylabel("Number of feature columns")
    ax.set_title("Feature-space size per dataset")
    for b, v in zip(bars, feats):
        ax.text(b.get_x() + b.get_width()/2, v + 1, str(v), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 90)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_feature_counts.png"))
    plt.close(fig)

    # save corrected stats json (class dists + feature counts)
    stats = {
        "cicids2017": {"total_rows": 2830743, "label_dist": cic, "feature_columns": 78,
                        "unique_features": 77, "nan_cells": 1358, "inf_cells": 4376},
        "unsw_nb15": {"total_rows": 257673, "attack_cat": unsw,
                       "feature_columns": 42, "nan_cells": 0, "inf_cells": 0},
        "ton_iot": {"total_rows": 211043, "type": ton, "feature_columns": 42,
                     "nan_cells": 0, "inf_cells": 0},
    }
    with open(r"F:\BTP\src\dataset_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print("charts + stats written to", FIG)
    print(json.dumps(stats, indent=2))
