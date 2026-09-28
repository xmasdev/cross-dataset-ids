"""
Metaheuristic (bio-inspired) feature selection with a CROSS-DATASET fitness.

This is the project's core contribution. Unlike prior work that selects features
to maximise SAME-dataset accuracy, we search for a feature subset that maximises
performance when a model is trained on one dataset and tested on another.

Two optimisers are implemented from scratch (no extra dependencies):
    * Genetic Algorithm (GA)   — selection, crossover, mutation
    * Binary Particle Swarm Optimization (PSO)

Workflow
--------
1. Load a stratified SAMPLE of each harmonised dataset (fast fitness evaluation).
2. Search for the shared feature mask that maximises mean cross-dataset balanced
   accuracy over all 6 source->target pairs.
3. Evaluate the winning mask on the FULL datasets with a Random Forest and compare
   against the all-features baseline.

Outputs (results/):
    metaheuristic_results.csv     per-pair full-data metrics for the best mask
    metaheuristic_summary.md      before/after comparison + best features
    metaheuristic_convergence.png (in reports/figures)
    metaheuristic_before_after.png

Usage:
    python src/metaheuristic_fs.py
    python src/metaheuristic_fs.py --sample 20000 --pop 16 --gen 10
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, recall_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_cross_dataset as b

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
# Data loading
# ---------------------------------------------------------------------------
def harmonized_cache(name, tag="all"):
    return os.path.join(OUT, f"harmonized_{name.replace('/', '_')}_n{tag}.csv")


def load_sample(name, n, seed):
    p = harmonized_cache(name)
    if os.path.exists(p):
        df = pd.read_csv(p)
    else:
        df = b.get_dataset(name, None, seed, False)
    return b.stratified_sample(df, n, seed)


def make_xy(df):
    return (df[FEATURES].to_numpy(dtype=np.float32), df["label"].to_numpy())


# ---------------------------------------------------------------------------
# Fitness: mean cross-dataset balanced accuracy over all ordered pairs
# ---------------------------------------------------------------------------
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
                scores.append(balanced_accuracy_score(yt, pred))
        val = float(np.mean(scores))
        cache[key] = val
        return val

    return fitness


# ---------------------------------------------------------------------------
# Optimisers (binary search space)
# ---------------------------------------------------------------------------
def genetic_algorithm(fitness, n_features, pop_size, generations, seed):
    rng = np.random.default_rng(seed)

    def random_ind():
        ind = rng.integers(0, 2, size=n_features)
        if ind.sum() == 0:
            ind[rng.integers(n_features)] = 1
        return ind

    def tournament(pop, fits, k=3):
        idx = rng.integers(0, len(pop), size=k)
        return pop[idx[fits[idx].argmax()]].copy()

    pop = np.array([random_ind() for _ in range(pop_size)])
    fits = np.array([fitness(ind) for ind in pop])
    best_i = int(fits.argmax())
    best, best_fit = pop[best_i].copy(), fits[best_i]
    history = [best_fit]

    p_cross, p_mut = 0.85, 1.0 / n_features
    for _ in range(generations):
        newpop = [best.copy()]  # elitism
        while len(newpop) < pop_size:
            if rng.random() < p_cross:
                p1, p2 = tournament(pop, fits), tournament(pop, fits)
                mask = rng.random(n_features) < 0.5
                child = np.where(mask, p1, p2)
            else:
                child = tournament(pop, fits)
            child = child.copy()
            for j in range(n_features):
                if rng.random() < p_mut:
                    child[j] = 1 - child[j]
            if child.sum() == 0:
                child[rng.integers(n_features)] = 1
            newpop.append(child)
        pop = np.array(newpop)
        fits = np.array([fitness(ind) for ind in pop])
        g_best = int(fits.argmax())
        if fits[g_best] > best_fit:
            best, best_fit = pop[g_best].copy(), fits[g_best]
        history.append(best_fit)
    return best, best_fit, history


def binary_pso(fitness, n_features, n_particles, iters, seed,
               w=0.72, c1=1.49, c2=1.49):
    rng = np.random.default_rng(seed)
    X = rng.random((n_particles, n_features))
    V = rng.uniform(-0.1, 0.1, size=(n_particles, n_features))

    def binarize(pos):
        prob = 1.0 / (1.0 + np.exp(-pos))
        b = (rng.random(pos.shape) < prob).astype(int)
        for i in range(len(b)):
            if b[i].sum() == 0:
                b[i, rng.integers(n_features)] = 1
        return b

    B = binarize(X)
    pb_fit = np.array([fitness(bi) for bi in B])
    pb = B.copy()
    gi = int(pb_fit.argmax())
    gb, gb_fit = pb[gi].copy(), pb_fit[gi]
    history = [gb_fit]

    for _ in range(iters):
        r1, r2 = rng.random((n_particles, n_features)), rng.random((n_particles, n_features))
        V = w * V + c1 * r1 * (pb - X) + c2 * r2 * (gb - X)
        X = np.clip(X + V, 0.0, 1.0)
        B = binarize(X)
        fits = np.array([fitness(bi) for bi in B])
        improved = fits > pb_fit
        pb[improved], pb_fit[improved] = B[improved], fits[improved]
        gi = int(pb_fit.argmax())
        if pb_fit[gi] > gb_fit:
            gb, gb_fit = pb[gi].copy(), pb_fit[gi]
        history.append(gb_fit)
    return gb, gb_fit, history


# ---------------------------------------------------------------------------
# Full-data evaluation of a mask
# ---------------------------------------------------------------------------
def evaluate_mask_full(mask, full_xy, seed):
    idx = [i for i, v in enumerate(mask) if v]
    rows = []
    for src in DATASETS:
        Xs, ys = full_xy[src]
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                     n_jobs=-1, class_weight="balanced",
                                     random_state=seed)
        clf.fit(Xs[:, idx], ys)
        for tgt in DATASETS:
            Xt, yt = full_xy[tgt]
            pred = clf.predict(Xt[:, idx])
            rows.append({
                "source": src, "target": tgt,
                "accuracy": accuracy_score(yt, pred),
                "balanced_accuracy": balanced_accuracy_score(yt, pred),
                "macro_f1": f1_score(yt, pred, average="macro", zero_division=0),
                "recall_attack": recall_score(yt, pred, pos_label=1, zero_division=0),
            })
    return pd.DataFrame(rows)


def cross_mean(df, same=True):
    """Mean over cross-dataset (off-diagonal) or same-dataset (diagonal) cells."""
    if same:
        vals = [df[(df.source == d) & (df.target == d)] for d in DATASETS]
    else:
        vals = [df[(df.source == s) & (df.target == t)]
                for s in DATASETS for t in DATASETS if s != t]
    return pd.concat(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--gen", type=int, default=10)
    ap.add_argument("--particles", type=int, default=16)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    n_features = len(FEATURES)
    print(f"Shared subspace: {n_features} features -> {FEATURES}\n")

    print("Loading stratified samples for fitness evaluation ...")
    sample_xy = {}
    for name in DATASETS:
        df = load_sample(name, args.sample, args.seed)
        sample_xy[name] = make_xy(df)
        print(f"  {name:14s} sample rows={len(df):>7,}")

    cache = {}
    fitness = make_fitness(sample_xy, args.seed, cache)

    print("\nRunning Genetic Algorithm ...")
    best_all = np.ones(n_features, dtype=int)
    base_fit = fitness(best_all)
    print(f"  all-features sample fitness (mean cross-dataset bal-acc) = {base_fit:.4f}")
    t0 = time.time()
    ga_mask, ga_fit, ga_hist = genetic_algorithm(fitness, n_features, args.pop, args.gen, args.seed)
    print(f"  GA best fitness = {ga_fit:.4f}  ({time.strftime('%M:%S', time.gmtime(time.time()-t0))})")
    print(f"  GA selected features: {[FEATURES[i] for i,v in enumerate(ga_mask) if v]}")

    print("\nRunning Binary Particle Swarm Optimization ...")
    t0 = time.time()
    pso_mask, pso_fit, pso_hist = binary_pso(fitness, n_features, args.particles, args.iters, args.seed)
    print(f"  PSO best fitness = {pso_fit:.4f}  ({time.strftime('%M:%S', time.gmtime(time.time()-t0))})")
    print(f"  PSO selected features: {[FEATURES[i] for i,v in enumerate(pso_mask) if v]}")

    # choose the best optimiser result
    if pso_fit > ga_fit:
        best_mask, best_fit, best_name, best_hist = pso_mask, pso_fit, "PSO", pso_hist
    else:
        best_mask, best_fit, best_name, best_hist = ga_mask, ga_fit, "GA", ga_hist
    print(f"\nBest optimiser: {best_name} (sample fitness {best_fit:.4f})")

    # ---- full-data evaluation ----
    print("\nLoading FULL datasets for final evaluation ...")
    full_xy = {}
    for name in DATASETS:
        p = harmonized_cache(name)
        df = pd.read_csv(p) if os.path.exists(p) else b.get_dataset(name, None, args.seed, False)
        full_xy[name] = make_xy(df)
        print(f"  {name:14s} rows={len(df):>9,}")

    print(f"\nEvaluating the optimised subset on FULL data ...")
    t0 = time.time()
    after = evaluate_mask_full(best_mask, full_xy, args.seed)
    print(f"  done ({time.strftime('%M:%S', time.gmtime(time.time()-t0))})")

    all_mask = np.ones(n_features, dtype=int)
    print("Evaluating all-features baseline on FULL data (same pipeline) ...")
    t0 = time.time()
    before = evaluate_mask_full(all_mask, full_xy, args.seed)
    print(f"  done ({time.strftime('%M:%S', time.gmtime(time.time()-t0))})")

    # ---- results ----
    after.to_csv(os.path.join(OUT, "metaheuristic_results.csv"), index=False)
    before.to_csv(os.path.join(OUT, "metaheuristic_baseline_results.csv"), index=False)
    import json
    with open(os.path.join(OUT, "metaheuristic_search.json"), "w") as fh:
        json.dump({
            "features": FEATURES,
            "ga_mask": ga_mask.tolist(), "ga_fit": ga_fit, "ga_hist": ga_hist,
            "pso_mask": pso_mask.tolist(), "pso_fit": pso_fit, "pso_hist": pso_hist,
            "best_optimiser": best_name, "best_mask": best_mask.tolist(),
            "base_fit": base_fit,
        }, fh, indent=2)

    b_cross = cross_mean(before, same=False)
    a_cross = cross_mean(after, same=False)
    b_same = cross_mean(before, same=True)
    a_same = cross_mean(after, same=True)

    def m(df, col):
        return df[col].mean()

    summary_lines = []
    summary_lines.append("# Metaheuristic Feature Selection — Results\n")
    summary_lines.append(f"- Optimiser: **{best_name}** (also ran GA and binary PSO)")
    summary_lines.append(f"- Search space: {n_features} shared features")
    summary_lines.append(f"- Best subset ({int(best_mask.sum())} features): "
                         f"{', '.join(FEATURES[i] for i, v in enumerate(best_mask) if v)}\n")
    summary_lines.append("## Cross-dataset (mean over 6 source→target pairs)\n")
    summary_lines.append("| Metric | Baseline (all features) | Optimised subset | Change |")
    summary_lines.append("|---|---|---|---|")
    for col, label in [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
                       ("macro_f1", "Macro-F1"), ("recall_attack", "Attack recall")]:
        bv, av = m(b_cross, col), m(a_cross, col)
        summary_lines.append(f"| {label} | {bv:.4f} | {av:.4f} | {av-bv:+.4f} |")
    summary_lines.append("\n## Same-dataset (mean over 3 datasets)\n")
    summary_lines.append("| Metric | Baseline | Optimised | Change |")
    summary_lines.append("|---|---|---|---|")
    for col, label in [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
                       ("macro_f1", "Macro-F1")]:
        bv, av = m(b_same, col), m(a_same, col)
        summary_lines.append(f"| {label} | {bv:.4f} | {av:.4f} | {av-bv:+.4f} |")

    summary = "\n".join(summary_lines)
    with open(os.path.join(OUT, "metaheuristic_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")
    print("\n" + summary + "\n")

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(range(len(ga_hist)), ga_hist, marker="o", label="GA")
    ax.plot(range(len(pso_hist)), pso_hist, marker="s", label="PSO")
    ax.axhline(base_fit, color="#888", linestyle="--", label="all-features fitness")
    ax.set_xlabel("Iteration / generation")
    ax.set_ylabel("Mean cross-dataset balanced accuracy (sample)")
    ax.set_title("Metaheuristic convergence", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_meta_convergence.png"))
    plt.close(fig)

    metrics = ["accuracy", "balanced_accuracy", "macro_f1"]
    labels = ["Accuracy", "Balanced acc.", "Macro-F1"]
    before_v = [m(b_cross, c) for c in metrics]
    after_v = [m(a_cross, c) for c in metrics]
    x = np.arange(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(x - w/2, before_v, w, label="Baseline (all features)", color="#d62728")
    ax.bar(x + w/2, after_v, w, label="Optimised subset", color="#2ca02c")
    ax.axhline(0.5, color="#888", linestyle="--", linewidth=1)
    for xi, v in zip(x - w/2, before_v):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(x + w/2, after_v):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Mean over 6 cross-dataset pairs")
    ax.set_title("Cross-dataset performance: baseline vs optimised", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_meta_before_after.png"))
    plt.close(fig)

    print("Figures -> reports/figures/fig_meta_convergence.png, fig_meta_before_after.png")
    print("Summary ->", os.path.join(OUT, "metaheuristic_summary.md"))


if __name__ == "__main__":
    main()
