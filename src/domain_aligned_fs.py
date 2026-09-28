"""
DA-MFS: Domain-Aligned Metaheuristic Feature Selection for cross-dataset IDS.

Motivation
----------
Plain feature selection on raw features barely helps cross-dataset transfer because
the *marginal distributions* of the shared features differ wildly between datasets
(different collection tools / networks). We first align each dataset's feature
marginals to a common distribution (per-dataset quantile / rank transform -> normal,
in the spirit of Knothe-Rosenblatt / quantile domain adaptation), and then let a
bio-inspired optimiser (GA / binary PSO) search for the subset of aligned features
that maximises CROSS-DATASET performance.

This combination -- domain alignment + metaheuristic feature selection with a
cross-dataset fitness -- is, to our knowledge, new for network intrusion detection.

Pipeline
--------
  1. Load the harmonised full datasets.
  2. Align each dataset independently with a QuantileTransformer (-> standard normal).
  3. Optimise a feature mask with a GA and a binary PSO; fitness = mean cross-dataset
     score (0.5*balanced-accuracy + 0.5*macro-F1) over the 6 train->test pairs
     (fast Random Forest on a stratified sample).
  4. Re-evaluate on the FULL aligned datasets and compare:
        (a) raw all features      (b) aligned all features     (c) aligned + optimised subset

Outputs (results/):
    da_mfs_results.csv, da_mfs_summary.md, da_mfs_search.json
  (reports/figures/): fig_damfs_summary.png, fig_damfs_convergence.png
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, recall_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_cross_dataset as b
from metaheuristic_fs import genetic_algorithm, binary_pso

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = b.ROOT
OUT = b.OUT
FIG = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIG, exist_ok=True)
FEATURES = b.FEATURES
DATASETS = ["CIC-IDS-2017", "UNSW-NB15", "TON_IoT"]


# ---------------------------------------------------------------------------
def harmonized_cache(name, tag="all"):
    return os.path.join(OUT, f"harmonized_{name.replace('/', '_')}_n{tag}.csv")


def load_full(name, seed):
    p = harmonized_cache(name)
    return pd.read_csv(p) if os.path.exists(p) else b.get_dataset(name, None, seed, False)


def align_fit_transform(X):
    """Per-dataset quantile transform to a standard normal marginal."""
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             subsample=200000, random_state=0)
    return qt.fit_transform(X).astype(np.float32)


def sample_xy(X, y, n, seed):
    if n is None or n <= 0 or len(y) <= n:
        return X, y
    idx = []
    rng = np.random.default_rng(seed)
    for lab in np.unique(y):
        ids = np.where(y == lab)[0]
        k = max(1, int(round(n * len(ids) / len(y))))
        idx.append(rng.choice(ids, size=min(k, len(ids)), replace=False))
    idx = np.concatenate(idx)
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
def metrics_row(src, tgt, yt, pred):
    return {"source": src, "target": tgt,
            "accuracy": accuracy_score(yt, pred),
            "balanced_accuracy": balanced_accuracy_score(yt, pred),
            "macro_f1": f1_score(yt, pred, average="macro", zero_division=0),
            "attack_recall": recall_score(yt, pred, pos_label=1, zero_division=0)}


def evaluate_full(data, mask, seed, n_estimators=200):
    idx = [i for i, v in enumerate(mask) if v]
    rows = []
    for src in DATASETS:
        Xs, ys = data[src]
        clf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=5,
                                     class_weight="balanced", n_jobs=-1,
                                     random_state=seed).fit(Xs[:, idx], ys)
        for tgt in DATASETS:
            Xt, yt = data[tgt]
            rows.append(metrics_row(src, tgt, yt, clf.predict(Xt[:, idx])))
    return pd.DataFrame(rows)


def make_fitness(data_xy, seed, cache):
    def make_clf():
        return RandomForestClassifier(n_estimators=40, min_samples_leaf=5,
                                      n_jobs=-1, class_weight="balanced",
                                      random_state=seed)

    def fitness(mask):
        key = tuple(int(v) for v in mask)
        if key in cache:
            return cache[key]
        idx = [i for i, v in enumerate(key) if v]
        if not idx:
            cache[key] = 0.0
            return 0.0
        scores = []
        for src in DATASETS:
            Xs, ys = data_xy[src]
            clf = make_clf().fit(Xs[:, idx], ys)
            for tgt in DATASETS:
                if tgt == src:
                    continue
                Xt, yt = data_xy[tgt]
                pred = clf.predict(Xt[:, idx])
                scores.append(0.5 * balanced_accuracy_score(yt, pred)
                              + 0.5 * f1_score(yt, pred, average="macro", zero_division=0))
        val = float(np.mean(scores))
        cache[key] = val
        return val

    return fitness


def cross_mean(df):
    return df[df.source != df.target]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gen", type=int, default=15)
    ap.add_argument("--particles", type=int, default=20)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    n_features = len(FEATURES)

    print("Loading FULL datasets and aligning (quantile -> normal) ...")
    raw_data, aligned_data = {}, {}
    for name in DATASETS:
        df = load_full(name, args.seed)
        X = df[FEATURES].to_numpy(np.float32)
        y = df["label"].to_numpy()
        raw_data[name] = (X, y)
        Xa = align_fit_transform(X)
        aligned_data[name] = (Xa, y)
        print(f"  {name:14s} rows={len(y):>9,}")

    # sample for the search
    aligned_sample = {n: sample_xy(*(aligned_data[n]), args.sample, args.seed)
                      for n in DATASETS}
    cache = {}
    fitness = make_fitness(aligned_sample, args.seed, cache)

    all_mask = np.ones(n_features, dtype=int)
    base_fit = fitness(all_mask)
    print(f"\nAligned all-features sample fitness = {base_fit:.4f}")

    print("\nRunning Genetic Algorithm (DA-MFS) ...")
    t0 = time.time()
    ga_mask, ga_fit, ga_hist = genetic_algorithm(fitness, n_features,
                                                 args.pop, args.gen, args.seed)
    print(f"  GA fitness={ga_fit:.4f} ({time.strftime('%M:%S', time.gmtime(time.time()-t0))}) "
          f"feats={[FEATURES[i] for i, v in enumerate(ga_mask) if v]}")

    print("\nRunning Binary PSO ...")
    t0 = time.time()
    pso_mask, pso_fit, pso_hist = binary_pso(fitness, n_features,
                                             args.particles, args.iters, args.seed)
    print(f"  PSO fitness={pso_fit:.4f} ({time.strftime('%M:%S', time.gmtime(time.time()-t0))}) "
          f"feats={[FEATURES[i] for i, v in enumerate(pso_mask) if v]}")

    if pso_fit > ga_fit:
        best_mask, best_fit, best_name = pso_mask, pso_fit, "PSO"
    else:
        best_mask, best_fit, best_name = ga_mask, ga_fit, "GA"
    print(f"\nBest optimiser: {best_name} (fitness {best_fit:.4f}); "
          f"{int(best_mask.sum())} features selected")

    # full-data evaluation of the three configurations
    print("\nFull-data evaluation: raw | aligned | aligned+optimised ...")
    t0 = time.time()
    res_raw = evaluate_full(raw_data, all_mask, args.seed)
    res_aligned = evaluate_full(aligned_data, all_mask, args.seed)
    res_opt = evaluate_full(aligned_data, best_mask, args.seed)
    print(f"  done ({time.strftime('%M:%S', time.gmtime(time.time()-t0))})")

    res_raw["config"] = "raw_all"
    res_aligned["config"] = "aligned_all"
    res_opt["config"] = "aligned_optimised"
    allres = pd.concat([res_raw, res_aligned, res_opt], ignore_index=True)
    allres.to_csv(os.path.join(OUT, "da_mfs_results.csv"), index=False)

    with open(os.path.join(OUT, "da_mfs_search.json"), "w") as fh:
        json.dump({"features": FEATURES,
                   "ga_mask": ga_mask.tolist(), "ga_fit": ga_fit, "ga_hist": ga_hist,
                   "pso_mask": pso_mask.tolist(), "pso_fit": pso_fit, "pso_hist": pso_hist,
                   "best_optimiser": best_name, "best_mask": best_mask.tolist(),
                   "aligned_all_fit": base_fit}, fh, indent=2)

    def cm(df):
        c = cross_mean(df)
        return {m: c[m].mean() for m in ["accuracy", "balanced_accuracy", "macro_f1", "attack_recall"]}

    r0, r1, r2 = cm(res_raw), cm(res_aligned), cm(res_opt)
    lines = []
    lines.append("# DA-MFS — Domain-Aligned Metaheuristic Feature Selection\n")
    lines.append(f"- Optimiser: **{best_name}** (GA + binary PSO both run)")
    lines.append(f"- Best subset ({int(best_mask.sum())}/{n_features} features): "
                 f"{', '.join(FEATURES[i] for i, v in enumerate(best_mask) if v)}")
    lines.append("- Alignment: per-dataset quantile transform -> standard normal")
    lines.append("- Fitness: mean cross-dataset 0.5*balanced-accuracy + 0.5*macro-F1\n")
    lines.append("## Cross-dataset results (mean over 6 train→test pairs, FULL data)\n")
    lines.append("| Configuration | Accuracy | Balanced acc | Macro-F1 | Attack recall |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| (a) Raw, all features | {r0['accuracy']:.4f} | {r0['balanced_accuracy']:.4f} | {r0['macro_f1']:.4f} | {r0['attack_recall']:.4f} |")
    lines.append(f"| (b) Aligned, all features | {r1['accuracy']:.4f} | {r1['balanced_accuracy']:.4f} | {r1['macro_f1']:.4f} | {r1['attack_recall']:.4f} |")
    lines.append(f"| (c) Aligned + optimised subset | {r2['accuracy']:.4f} | {r2['balanced_accuracy']:.4f} | {r2['macro_f1']:.4f} | {r2['attack_recall']:.4f} |")
    lines.append("")
    lines.append("## Improvement over raw baseline\n")
    lines.append("| Metric | Raw | DA-MFS | Change |")
    lines.append("|---|---|---|---|")
    for m, lab in [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced acc"),
                   ("macro_f1", "Macro-F1"), ("attack_recall", "Attack recall")]:
        lines.append(f"| {lab} | {r0[m]:.4f} | {r2[m]:.4f} | {r2[m]-r0[m]:+.4f} |")
    summary = "\n".join(lines)
    with open(os.path.join(OUT, "da_mfs_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")
    print("\n" + summary + "\n")

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(range(len(ga_hist)), ga_hist, marker="o", label="GA")
    ax.plot(range(len(pso_hist)), pso_hist, marker="s", label="PSO")
    ax.axhline(base_fit, color="#888", linestyle="--", label="aligned all-features")
    ax.set_xlabel("Generation / iteration")
    ax.set_ylabel("Mean cross-dataset fitness (sample)")
    ax.set_title("DA-MFS convergence (domain-aligned features)", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_damfs_convergence.png"))
    plt.close(fig)

    labels = ["Accuracy", "Balanced acc", "Macro-F1", "Attack recall"]
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "attack_recall"]
    x = np.arange(len(keys)); w = 0.27
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.bar(x - w, [r0[k] for k in keys], w, label="Raw, all features", color="#d62728")
    ax.bar(x, [r1[k] for k in keys], w, label="Aligned, all features", color="#ff7f0e")
    ax.bar(x + w, [r2[k] for k in keys], w, label="Aligned + optimised (DA-MFS)", color="#2ca02c")
    ax.axhline(0.5, color="#888", linestyle="--", linewidth=1)
    ax.text(3.4, 0.505, "random (0.5)", fontsize=6, color="#555", ha="right")
    for xi, k in zip(x, keys):
        for off, rr in [(-w, r0), (0, r1), (w, r2)]:
            ax.text(xi + off, rr[k] + 0.008, f"{rr[k]:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Mean over 6 cross-dataset pairs")
    ax.set_title("Cross-dataset performance: raw vs aligned vs DA-MFS", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_damfs_summary.png"))
    plt.close(fig)
    print("Figures -> reports/figures/fig_damfs_convergence.png, fig_damfs_summary.png")
    print("Summary ->", os.path.join(OUT, "da_mfs_summary.md"))


if __name__ == "__main__":
    main()
