"""All end-semester figures, drawn from the CSV files in results/v2/ into reports/figures/v2_*.png.

    python experiments/10_figures.py
"""
import os, sys, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from xids.config import OUT, ROOT

FIG = os.path.join(ROOT, "reports", "figures"); os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white", "axes.facecolor": "white"})
SH = {"CIC-IDS-2017": "CIC-IDS-2017", "UNSW-NB15": "UNSW-NB15", "TON_IoT": "TON_IoT", "CSE-CIC-IDS2018": "CSE-CIC-IDS2018"}
T3 = ["CIC-IDS-2017", "UNSW-NB15", "TON_IoT"]
rd = lambda f: pd.read_csv(os.path.join(OUT, f))
have = lambda f: os.path.exists(os.path.join(OUT, f))


def save(name):
    plt.savefig(os.path.join(FIG, name)); plt.close(); print("wrote", name)


# ---------------------------------------------------------------- 1. same- vs cross-dataset, every supervised model
def fig_supervised():
    S = rd("supervised_benchmark.csv"); S = S[(S.rep == "raw") & (S.model != "majority")]
    same = S[S.protocol == "same"].groupby("model").auc.mean(); cross = S[S.protocol == "pair"].groupby("model").auc.mean()
    names = {"lr": "Logistic regression", "linsvm": "Linear SVM", "nb": "Naive Bayes", "dt": "Decision tree", "rf": "Random forest",
             "et": "Extra trees", "hgb": "Hist. gradient boosting", "xgb": "XGBoost", "lgbm": "LightGBM", "knn": "k-nearest neighbours",
             "mlp": "MLP", "ftt": "FT-Transformer", "dann": "DANN (domain-adversarial)", "dcoral": "Deep CORAL"}
    order = cross.sort_values().index.tolist(); fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for i, m in enumerate(order):
        if m in same.index:
            ax.plot([cross[m], same[m]], [i, i], color=AXIS, lw=1.5, zorder=1)
            ax.scatter(same[m], i, s=42, color=BLUE, zorder=3, edgecolor="white", linewidth=1)
        ax.scatter(cross[m], i, s=42, color=ORANGE, zorder=3, edgecolor="white", linewidth=1)
    ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 3))); ax.text(0.505, len(order) - 0.35, "chance", color=MUTED, fontsize=8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names.get(m, m) for m in order], color=INK2); ax.set_xlim(0.35, 1.02)
    ax.set_xlabel("ROC AUC (mean over datasets / dataset pairs, 3 seeds)"); ax.grid(axis="y", visible=False)
    ax.scatter([], [], s=42, color=BLUE, label="tested on the dataset it was trained on (5-fold CV)")
    ax.scatter([], [], s=42, color=ORANGE, label="tested on a different dataset")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=1, fontsize=8.5)
    save("v2_supervised_same_vs_cross.png")


# ---------------------------------------------------------------- 2. direction of every feature in every dataset
def fig_direction():
    A = rd("diag_direction.csv"); cols = ["CIC17", "UNSW", "TON"]
    A = A.sort_values(["flips", "family"], ascending=[False, True]).reset_index(drop=True)
    cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a987", "#c4481a"])
    fig, ax = plt.subplots(figsize=(4.6, 7.2)); im = ax.imshow(A[cols].to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(A)):
        for j, c in enumerate(cols):
            v = A.loc[i, c]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if abs(v - 0.5) > 0.33 else INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(["CIC-IDS-2017", "UNSW-NB15", "TON_IoT"], color=INK2); ax.xaxis.tick_top()
    ax.set_yticks(range(len(A))); ax.set_yticklabels([("▸ " if f else "") + n for n, f in zip(A.feature, A.flips)], color=INK2, fontsize=8)
    ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]; ax.tick_params(length=0)
    nf = int(A.flips.sum()); ax.axhline(nf - 0.5, color=INK, lw=1.2)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.set_label("single-feature AUC, attack vs benign\n(<0.5: attacks have smaller values, >0.5: larger)", fontsize=8); cb.outline.set_visible(False)
    ax.set_xlabel(f"▸ = direction flips between two datasets ({nf} of {len(A)} features)", fontsize=8.5)
    save("v2_feature_direction.png")


# ---------------------------------------------------------------- 3. the mid-semester selection, with and without the leak
def fig_leak():
    S = rd("supfs_summary.csv"); S = S[S.metric == "auc"].groupby("rep")[["all_features", "oracle_selected", "leakfree_selected"]].mean()
    S = S.loc[["raw", "dom_quantile", "benign_quantile"]]; lab = ["Raw features", "Per-dataset\nquantile alignment", "Benign-anchored\nquantiles"]
    fig, ax = plt.subplots(figsize=(6.2, 3.3)); w = 0.25; x = np.arange(3)
    for k, (c, col, name) in enumerate([("all_features", MUTED, "all 12 features"), ("oracle_selected", ORANGE, "best subset chosen on the test pairs (mid-semester protocol)"),
                                         ("leakfree_selected", BLUE, "best subset chosen without the target dataset")]):
        b = ax.bar(x + (k - 1) * (w + 0.02), S[c], w, color=col, label=name)
        for xi, v in zip(x + (k - 1) * (w + 0.02), S[c]):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3))); ax.set_xticks(x); ax.set_xticklabels(lab, color=INK2); ax.set_ylim(0, 0.9)
    ax.set_ylabel("mean cross-dataset AUC"); ax.grid(axis="x", visible=False); ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), fontsize=8)
    save("v2_leak_audit.png")


# ---------------------------------------------------------------- 4. headline: what works on each target
def fig_main():
    D = rd("final_methods.csv"); D = D[(D.protocol == "random") & (D.n_cal == 6000) & (D.prev == "natural")]
    S = rd("supervised_benchmark.csv"); best = S[(S.protocol.isin(["pair", "lodo"])) & (S.model != "majority")].groupby(["target", "protocol", "rep", "model"]).auc.mean().groupby("target").max()
    meth = [("xfer", "best supervised transfer from other datasets (oracle choice of model)", MUTED), ("knn_all", "kNN to local benign flows, all features", AQUA),
            ("knn_ga", "kNN + label-free GA selection", YELLOW), ("pu", "benign-vs-unlabelled classifier", VIOLET), ("brace", "BRACE (fusion)", BLUE),
            ("sup_target", "supervised with target labels (upper reference)", ORANGE)]
    fig, ax = plt.subplots(figsize=(7.4, 3.6)); w = 0.13; x = np.arange(3)
    for k, (m, name, col) in enumerate(meth):
        v = [best[t] if m == "xfer" else D[(D.target == t) & (D.method == m)].auc.mean() for t in T3]
        e = [0 if m == "xfer" else D[(D.target == t) & (D.method == m)].auc.std() for t in T3]
        ax.bar(x + (k - 2.5) * (w + 0.012), v, w, color=col, label=name, yerr=e, error_kw=dict(ecolor=INK2, lw=0.8))
        for xi, vi in zip(x + (k - 2.5) * (w + 0.012), v):
            ax.text(xi, vi + 0.03, f"{vi:.3f}", ha="center", fontsize=6.5, color=INK2, rotation=90)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3))); ax.set_xticks(x); ax.set_xticklabels(T3, color=INK2); ax.set_ylim(0, 1.19); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("ROC AUC on the target (5 seeds)"); ax.grid(axis="x", visible=False); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.8)
    save("v2_main_result.png")


# ---------------------------------------------------------------- 5. attack prevalence of the unlabelled stream
def fig_prevalence():
    D = rd("final_methods.csv"); D = D[D.protocol == "random"]; prevs = ["natural", "0.1", "0.02", "0.005"]; D["prev"] = D.prev.astype(str)
    meth = [("knn_all", "kNN, all features", AQUA, "o"), ("pu", "benign-vs-unlabelled classifier", VIOLET, "s"), ("brace", "BRACE (fusion)", BLUE, "D")]
    fig, axs = plt.subplots(2, 3, figsize=(7.6, 4.6), sharey=True, sharex=True)
    for r, ncal in enumerate([6000, 20000]):
        for c, t in enumerate(T3):
            ax = axs[r, c]
            for m, name, col, mk in meth:
                g = D[(D.target == t) & (D.n_cal == ncal) & (D.method == m)].groupby("prev").auc.mean().reindex(prevs)
                ax.plot(range(4), g.values, color=col, marker=mk, ms=5, lw=2, label=name, markeredgecolor="white", markeredgewidth=0.8)
            ax.set_xticks(range(4)); ax.set_xticklabels(["natural", "10%", "2%", "0.5%"]); ax.set_ylim(0.68, 1.01)
            if r == 0: ax.set_title(t)
            if c == 0: ax.set_ylabel(f"AUC ({ncal // 1000}k calibration flows)", fontsize=8)
            if r == 1 and c == 1: ax.set_xlabel("attack share of the unlabelled stream")
    h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), fontsize=8.5)
    plt.tight_layout(); save("v2_prevalence.png")


# ---------------------------------------------------------------- 6. what the GA buys: detection at 1% false alarms
def fig_ga():
    D = rd("final_methods.csv"); D = D[(D.protocol == "random") & (D.n_cal == 6000)]; D["prev"] = D.prev.astype(str); prevs = ["natural", "0.1", "0.02", "0.005"]
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.7), sharey=True)
    for c, t in enumerate(T3):
        ax = axs[c]; w = 0.36; x = np.arange(4)
        for k, (m, name, col) in enumerate([("knn_all", "kNN, all 30 features", AQUA), ("knn_ga", "kNN, GA-selected subset (label-free fitness)", YELLOW)]):
            g = D[(D.target == t) & (D.method == m)].groupby("prev")["tpr@0.01"].agg(["mean", "std"]).reindex(prevs)
            ax.bar(x + (k - 0.5) * (w + 0.03), g["mean"], w, color=col, label=name, yerr=g["std"], error_kw=dict(ecolor=INK2, lw=0.8))
        ax.set_xticks(x); ax.set_xticklabels(["natural", "10%", "2%", "0.5%"]); ax.set_title(t); ax.grid(axis="x", visible=False); ax.set_ylim(0, 1)
        if c == 1: ax.set_xlabel("attack share of the unlabelled stream")
        if c == 0: ax.set_ylabel("detection rate at 1% false alarms")
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.1), fontsize=8.5)
    plt.tight_layout(); save("v2_ga_gain.png")


# ---------------------------------------------------------------- 7. optimisers on the label-free fitness
def fig_optimisers():
    R = rd("optimisers_large.csv")
    names = {"rs": "Random search", "sfs": "Greedy forward", "sa": "Simulated annealing", "ga": "Genetic algorithm", "pso": "Binary PSO", "de": "Binary DE",
             "gwo": "Grey wolf", "woa": "Whale", "nsga2": "NSGA-II"}
    ts = [t for t in T3 if (R.target == t).any()]; fig, axs = plt.subplots(1, len(ts), figsize=(2.6 * len(ts), 3.0), squeeze=False, sharey=True)
    order = R.assign(r=R.groupby("target").best_fit.rank()).groupby("algo").r.mean().sort_values().index.tolist()
    for c, t in enumerate(ts):
        ax = axs[0, c]; G = R[R.target == t]
        for i, algo in enumerate(order):
            v = G[G.algo == algo].best_fit.to_numpy()
            ax.scatter(v, np.full(len(v), i), s=14, color=BLUE, alpha=0.35, zorder=2, linewidth=0)
            ax.scatter(v.mean(), i, s=46, color=BLUE, zorder=3, edgecolor="white", linewidth=1)
        fa = G.fit_all.iloc[0]; ax.axvline(fa, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.text(fa, len(order) - 0.45, " all 30 features", color=MUTED, fontsize=7.5, ha="left")
        ax.set_title(t); ax.set_xlabel("label-free fitness reached\n(300 evaluations)"); ax.grid(axis="y", visible=False); ax.set_ylim(-0.6, len(order) - 0.1); ax.locator_params(axis="x", nbins=4)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([names[a_] for a_ in order], color=INK2)
    plt.tight_layout(); save("v2_optimisers.png")


# ---------------------------------------------------------------- 8. which attack families are caught
def fig_families():
    F = rd("final_families.csv"); F = F[(F.protocol == "random") & (F.n_cal == 6000) & (F.prev.astype(str) == "natural") & (F.alpha == 0.05) & (F.n >= 30)]
    fig, axs = plt.subplots(1, 3, figsize=(7.8, 3.4))
    for c, t in enumerate(T3):
        ax = axs[c]; P = F[F.target == t].groupby(["family", "method"]).tpr.mean().unstack("method")[["knn_all", "pu", "brace"]].sort_values("brace")
        y = np.arange(len(P))
        for k, (m, name, col) in enumerate([("knn_all", "kNN, all features", AQUA), ("pu", "benign-vs-unlabelled", VIOLET), ("brace", "BRACE", BLUE)]):
            ax.barh(y + (k - 1) * 0.27, P[m], 0.24, color=col, label=name)
        ax.set_yticks(y); ax.set_yticklabels(P.index, color=INK2); ax.set_xlim(0, 1.02); ax.set_title(t); ax.grid(axis="y", visible=False)
        if c == 1: ax.set_xlabel("detection rate at 5% false alarms")
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h[::-1], l[::-1], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), fontsize=8.5)
    plt.tight_layout(); save("v2_families.png")


# ---------------------------------------------------------------- 9. random split vs calibrate-on-another-day; external dataset
def fig_temporal():
    parts = [rd(f) for f in ("final_methods.csv", "final_methods_temporal.csv", "final_methods_external.csv") if have(f)]
    D = pd.concat(parts); D = D[(D.n_cal == 20000) & (D.prev.astype(str) == "natural")]
    cfg = [("CIC-IDS-2017", "random", "CIC-IDS-2017\nsame days"), ("CIC-IDS-2017", "temporal", "CIC-IDS-2017\nother days"),
           ("CSE-CIC-IDS2018", "random", "CIC-IDS2018\nsame days"), ("CSE-CIC-IDS2018", "temporal", "CIC-IDS2018\nother days")]
    cfg = [c for c in cfg if ((D.target == c[0]) & (D.protocol == c[1])).any()]
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.0)); w = 0.2; x = np.arange(len(cfg))
    for ax, met, yl in ((axs[0], "auc", "ROC AUC"), (axs[1], "fpr@0.05", "false-alarm rate at a 5% target")):
        for k, (m, name, col) in enumerate([("knn_all", "kNN, all features", AQUA), ("knn_ga", "kNN + GA", YELLOW), ("pu", "benign-vs-unlabelled", VIOLET), ("brace", "BRACE", BLUE)]):
            v = [D[(D.target == t) & (D.protocol == p) & (D.method == m)][met].mean() for t, p, _ in cfg]
            ax.bar(x + (k - 1.5) * (w + 0.01), v, w, color=col, label=name)
        ax.set_xticks(x); ax.set_xticklabels([c[2] for c in cfg], fontsize=7, color=INK2); ax.set_ylabel(yl); ax.grid(axis="x", visible=False)
        if met == "auc": ax.set_ylim(0, 1.02)
        else: ax.axhline(0.05, color=MUTED, lw=1, ls=(0, (3, 3)))
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08), fontsize=8.5)
    plt.tight_layout(); save("v2_temporal_external.png")


# ---------------------------------------------------------------- 10. benign-only detectors
def fig_detectors():
    B = rd("benign_detectors.csv"); B = B[(B.pool == "v2_30") & (B.rep == "quantile")]
    P = B.groupby(["detector", "target"]).auc.mean().unstack("target"); P = P[[t for t in T3 if t in P.columns]]; P = P.loc[P.mean(1).sort_values().index]
    names = {"knn1": "kNN (k=1)", "knn5": "kNN (k=5)", "knn10": "kNN (k=10)", "knn20": "kNN (k=20)", "gmm": "Gaussian mixture", "lof": "LOF",
             "autoencoder": "Autoencoder", "mahalanobis": "Mahalanobis", "iforest": "Isolation forest", "pca": "PCA reconstruction", "hbos": "HBOS",
             "ocsvm": "One-class SVM", "ecod": "ECOD"}
    fig, ax = plt.subplots(figsize=(6.4, 3.9)); y = np.arange(len(P))
    for i in y:
        ax.plot([P.iloc[i].min(), P.iloc[i].max()], [i, i], color=AXIS, lw=1.2, zorder=1)
    for t, col, mk in zip(P.columns, [BLUE, ORANGE, AQUA], ["o", "s", "D"]):
        ax.scatter(P[t], y, s=34, color=col, marker=mk, label=t, zorder=3, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y); ax.set_yticklabels([names.get(d_, d_) for d_ in P.index], color=INK2); ax.set_xlim(0.65, 1.0)
    ax.set_xlabel("ROC AUC, detector fitted on 4,200 benign flows of the target only"); ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=8.5); save("v2_detectors.png")


# ---------------------------------------------------------------- 11. pipeline diagram
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.8, 3.5)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 46)

    def box(x, y, w, h, text, fc="#f4f6fb", ec=BLUE, fs=8, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, weight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1, shrinkA=0, shrinkB=0))

    box(0.5, 31, 21, 10, "Benign calibration\nflows of the target\n(known-clean window)", fc="#eaf6f1", ec=AQUA, fs=7.5)
    box(0.5, 6, 21, 10, "Unlabelled live stream\nof the target (benign +\nunknown attacks)", fc="#fdf3e4", ec=YELLOW, fs=7.5)
    box(26, 18.5, 14, 10, "Benign-\nanchored\nquantiles", fs=7.5)
    box(44.5, 31, 22, 10, "kNN distance to benign\nreference, on the subset\nchosen by the GA", fs=7.5)
    box(44.5, 6, 22, 10, "Benign-vs-unlabelled\nclassifier\n(5 bagged boosters)", fs=7.5)
    box(44.5, 19.5, 22, 8, "GA fitness: tail alarm rate\nof the unlabelled stream\n(needs no attack label)", fc="#fff8e6", ec=YELLOW, fs=6.8)
    box(70.5, 18.5, 12, 10, "Conformal\np-values +\nFisher fusion", fs=7.5)
    box(86.5, 18.5, 13, 10, "Alarm at a\nchosen false-\nalarm rate", fc="#eaf1fb", bold=True, fs=7.5)
    arrow(22.1, 36, 26, 26.5); arrow(22.1, 11, 26, 20.5); arrow(40.6, 26, 44.5, 34); arrow(40.6, 21, 44.5, 12)
    arrow(55.5, 27.8, 55.5, 30.6); arrow(67.1, 36, 70.5, 26.5); arrow(67.1, 11, 70.5, 20.5); arrow(83.1, 23.5, 86.5, 23.5)
    ax.text(50, 1.5, "No source dataset and no attack label is used anywhere on the target.", ha="center", fontsize=8.5, color=INK2, style="italic")
    save("v2_pipeline.png")


if __name__ == "__main__":
    only = sys.argv[1:]
    for name, fn, need in [("supervised", fig_supervised, "supervised_benchmark.csv"), ("direction", fig_direction, "diag_direction.csv"),
                           ("leak", fig_leak, "supfs_summary.csv"), ("main", fig_main, "final_methods.csv"), ("prevalence", fig_prevalence, "final_methods.csv"),
                           ("ga", fig_ga, "final_methods.csv"), ("optimisers", fig_optimisers, "optimisers_large.csv"), ("families", fig_families, "final_families.csv"),
                           ("temporal", fig_temporal, "final_methods.csv"), ("detectors", fig_detectors, "benign_detectors.csv"), ("pipeline", fig_pipeline, None)]:
        if (not only or name in only) and (need is None or have(need)):
            try:
                fn()
            except Exception as e:
                print("FAILED", name, repr(e))
