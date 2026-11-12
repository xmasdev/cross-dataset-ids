"""Exhaustive fitness landscape on the 12-feature pool (all 4,095 subsets).

For every target, attack prevalence of the unlabelled stream and data seed, compute for EVERY subset
    - the label-free fitness values (pu_auc, alarm@5%, pu_pauc), which use no attack label
    - the true test AUC (labels used only here)
This answers, without any optimiser in the loop:
    * how well does each label-free fitness rank subsets by their true quality?
    * how good is the subset it would pick, against all features and against the true optimum?
    * do subset rankings agree ACROSS datasets (can a subset chosen elsewhere be reused)?
The tables are also the ground truth for the optimiser comparison (07_optimisers.py).

    python experiments/06_landscape.py --seeds 2
Writes results/v2/landscape_tables.npz, landscape_summary.csv, landscape_cross_dataset.csv
"""
import os, sys, time, argparse, itertools, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from xids.config import DATASETS, OUT, SHORT
from xids.features import POOL_V1
from xids.data import load
from xids.core import TargetContext, all_masks, fitness_from_scores

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--prev", default="natural,0.1,0.02,0.005"); ap.add_argument("--nadapt", type=int, default=15000)
a = ap.parse_args()
M = all_masks(len(POOL_V1)); ALL = len(M) - 1
KINDS = ["pu_auc", "alarm", "pu_pauc"]
tabs, rows = {}, []; t0 = time.time()
npz = os.path.join(OUT, "landscape_tables.npz")
if os.path.exists(npz):
    tabs = dict(np.load(npz))
for name in DATASETS:
    data = load(name, POOL_V1)
    for prev_s in a.prev.split(","):
        for seed in range(a.seeds):
            key = f"{name}|{prev_s}|{seed}"
            if key + "|true" not in tabs:
                ctx = TargetContext(name, seed, POOL_V1, n_cal=6000, n_adapt=a.nadapt, data=data, test_cap=20000,
                                    prevalence=None if prev_s == "natural" else float(prev_s))
                true = np.zeros(len(M)); fit = np.zeros((len(KINDS), len(M)))
                for i, m in enumerate(M):
                    su, sb = ctx.scores(m, "adapt"), ctx.scores(m, "sel")
                    fit[:, i] = [fitness_from_scores(su, sb, k) for k in KINDS]
                    true[i] = ctx.test_auc(m)
                tabs[key + "|true"] = true; tabs[key + "|fit"] = fit; tabs[key + "|att"] = np.array([ctx.y_adapt.mean()])
                np.savez_compressed(npz, **tabs)
                print(f"{key} done ({time.time()-t0:.0f}s)", flush=True)
            true, fit = tabs[key + "|true"], tabs[key + "|fit"]
            for j, k in enumerate(KINDS):
                b = fit[j].argmax()
                rows.append(dict(target=name, prev=prev_s, seed=seed, att_frac=float(tabs[key + "|att"][0]), fitness=k,
                                 auc_all=true[ALL], auc_selected=true[b], auc_oracle=true.max(), auc_median=np.median(true),
                                 spearman=spearmanr(fit[j], true).statistic, percentile=(true <= true[b]).mean() * 100, k=int(M[b].sum())))
S = pd.DataFrame(rows); S.to_csv(os.path.join(OUT, "landscape_summary.csv"), index=False)
pd.set_option("display.width", 220)
print(S.groupby(["target", "prev", "fitness"], sort=False)[["att_frac", "auc_all", "auc_selected", "auc_oracle", "auc_median", "spearman", "percentile", "k"]].mean().round(3).to_string())

# do subset rankings transfer across datasets? (true AUC tables; a subset chosen with labels elsewhere)
rows = []
for seed in range(a.seeds):
    T = {n: tabs[f"{n}|natural|{seed}|true"] for n in DATASETS}
    for x, y in itertools.combinations(DATASETS, 2):
        rows.append(dict(kind="spearman", seed=seed, a=SHORT[x], b=SHORT[y], value=spearmanr(T[x], T[y]).statistic))
    for t in DATASETS:
        src = np.mean([T[s] for s in DATASETS if s != t], axis=0); b = src.argmax()
        rows.append(dict(kind="lodo_selected_auc", seed=seed, a=SHORT[t], b="", value=T[t][b]))
        rows.append(dict(kind="lodo_percentile", seed=seed, a=SHORT[t], b="", value=(T[t] <= T[t][b]).mean() * 100))
        rows.append(dict(kind="all_features_auc", seed=seed, a=SHORT[t], b="", value=T[t][ALL]))
        rows.append(dict(kind="oracle_auc", seed=seed, a=SHORT[t], b="", value=T[t].max()))
C = pd.DataFrame(rows); C.to_csv(os.path.join(OUT, "landscape_cross_dataset.csv"), index=False)
print("\n=== can a subset chosen (with labels) on the other two datasets be reused? ===")
print(C.groupby(["kind", "a", "b"])["value"].mean().round(3).to_string())
