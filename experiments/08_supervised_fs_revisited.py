"""The mid-semester idea, re-examined without leakage: can ANY feature subset fix supervised transfer?

For all 4,095 subsets of the 12-feature pool, train a gradient-boosting classifier on each source and
test on the other two datasets. With the full table in hand, three questions have exact answers:

  1. Oracle (mid-semester protocol): pick the subset with the best mean over the six pairs and report
     that same mean. This is selection on the test labels and is an upper bound for every optimiser.
  2. Leak-free: for a target C pick the subset that transfers best BETWEEN the other two datasets
     (A->B and B->A) and only then test A->C and B->C.
  3. Standard feature selection computed on the source alone (mutual information, random-forest
     importance, RFE, L1 logistic regression, correlation filter), at every subset size.

    python experiments/08_supervised_fs_revisited.py --seeds 3
Writes results/v2/supfs_tables.npz, supfs_summary.csv, supfs_standard.csv
"""
import os, sys, time, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from xids.config import DATASETS, OUT, SHORT
from xids.features import POOL_V1
from xids.data import load, stratified_cap, make_target_split
from xids.anchor import make_anchor
from xids.core import all_masks
from xids.metrics import auc

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--jobs", type=int, default=6); ap.add_argument("--analyse-only", action="store_true")
a = ap.parse_args()
d = len(POOL_V1); M = all_masks(d); ALL = len(M) - 1; W = 1 << np.arange(d)
REPS = ["raw", "dom_quantile", "benign_quantile"]
PAIRS = [(s, t) for s in range(3) for t in range(3) if s != t]


def one_mask(m, TR, TE):
    cols = np.flatnonzero(m); out = np.zeros((3, 3, 2))
    for s in range(3):
        clf = LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.1, class_weight="balanced",
                             n_jobs=1, verbose=-1, random_state=0).fit(TR[s][0][:, cols], TR[s][1])
        for t in range(3):
            p = clf.predict_proba(TE[t][0][:, cols])[:, 1]
            out[s, t] = auc(TE[t][1], p), balanced_accuracy_score(TE[t][1], p >= 0.5)
    return out


npz = os.path.join(OUT, "supfs_tables.npz")
tabs = dict(np.load(npz)) if os.path.exists(npz) else {}
D = {n: load(n, POOL_V1) for n in DATASETS}
t0 = time.time(); std_rows = []
for seed in range(a.seeds):
    rng = np.random.default_rng(seed); tr, sp = {}, {}
    for n in DATASETS:
        X, y, _ = D[n]; sp[n] = make_target_split(y, seed, n_cal=6000, n_adapt=30000, test_cap=40000)
        free = np.setdiff1d(np.arange(len(y)), np.r_[sp[n]["cal"], sp[n]["test"]])
        tr[n] = free[stratified_cap(y[free], 20000, rng)]
    for rep in REPS:
        key = f"{rep}|{seed}"
        TR, TE = [], []
        for n in DATASETS:
            X, y, _ = D[n]
            if rep == "raw":
                TR.append((X[tr[n]], y[tr[n]])); TE.append((X[sp[n]["test"]], y[sp[n]["test"]]))
            elif rep == "dom_quantile":
                TR.append((make_anchor("quantile").fit(X[tr[n]]).transform(X[tr[n]]), y[tr[n]]))
                TE.append((make_anchor("quantile").fit(X[sp[n]["adapt"]]).transform(X[sp[n]["test"]]), y[sp[n]["test"]]))
            else:
                TR.append((make_anchor("quantile").fit(X[tr[n]][y[tr[n]] == 0]).transform(X[tr[n]]), y[tr[n]]))
                TE.append((make_anchor("quantile").fit(X[sp[n]["cal"]]).transform(X[sp[n]["test"]]), y[sp[n]["test"]]))
        if key not in tabs and not a.analyse_only:
            res = Parallel(n_jobs=a.jobs, batch_size=16)(delayed(one_mask)(m, TR, TE) for m in M)
            tabs[key] = np.array(res); np.savez_compressed(npz, **tabs)
            print(f"{key} table done ({time.time()-t0:.0f}s)", flush=True)
        # standard feature-selection rankings, computed on each SOURCE only
        if rep == "raw" and key in tabs:
            for s, n in enumerate(DATASETS):
                Xs, ys = TR[s]; Xl = np.log1p(np.abs(Xs)) * np.sign(Xs); Xl = (Xl - Xl.mean(0)) / (Xl.std(0) + 1e-9)
                rank = {"mutual_info": np.argsort(-mutual_info_classif(Xl, ys, random_state=seed)),
                        "rf_importance": np.argsort(-RandomForestClassifier(200, min_samples_leaf=5, n_jobs=6, random_state=seed).fit(Xs, ys).feature_importances_),
                        "l1_logistic": np.argsort(-np.abs(LogisticRegression(penalty="l1", C=0.05, solver="liblinear").fit(Xl, ys).coef_[0])),
                        "rfe": np.argsort(RFE(LogisticRegression(max_iter=200), n_features_to_select=1).fit(Xl, ys).ranking_)}
                c = np.abs(np.corrcoef(Xl.T)); order = list(rank["mutual_info"]); keep = []
                for j in order:                                    # correlation filter: drop features with |r| > 0.9 to one already kept
                    if all(c[j, k] <= 0.9 for k in keep):
                        keep.append(j)
                rank["corr_filter+MI"] = np.array(keep + [j for j in order if j not in keep])
                for meth, r in rank.items():
                    for k in range(1, d + 1):
                        std_rows.append(dict(seed=seed, source=SHORT[n], method=meth, k=k, mask_index=int(W[r[:k]].sum()) - 1))

# ------------------------------------------------------------------ analysis of the tables
rows = []
for key, T in tabs.items():
    rep, seed = key.split("|")
    for mi, metric in enumerate(["auc", "balanced_accuracy"]):
        P = T[:, :, :, mi]
        cross = np.array([P[:, s, t] for s, t in PAIRS]).mean(0)             # mean over the six pairs, per subset
        b = cross.argmax()
        lodo, lodo_all, lodo_best = [], [], []
        for c in range(3):
            a_, b_ = [i for i in range(3) if i != c]
            sel = (P[:, a_, b_] + P[:, b_, a_]).argmax()                     # chosen without touching dataset c
            tgt = (P[:, a_, c] + P[:, b_, c]) / 2
            lodo.append(tgt[sel]); lodo_all.append(tgt[ALL]); lodo_best.append(tgt.max())
        rows.append(dict(rep=rep, seed=int(seed), metric=metric, all_features=cross[ALL], oracle_selected=cross[b],
                         oracle_k=int(M[b].sum()), leakfree_selected=np.mean(lodo), leakfree_all=np.mean(lodo_all),
                         per_target_oracle=np.mean(lodo_best), median_subset=np.median(cross), same_dataset=np.mean([P[ALL, i, i] for i in range(3)])))
S = pd.DataFrame(rows); S.to_csv(os.path.join(OUT, "supfs_summary.csv"), index=False)
pd.set_option("display.width", 220)
print(S.groupby(["metric", "rep"])[["all_features", "oracle_selected", "leakfree_selected", "leakfree_all", "per_target_oracle", "median_subset"]].mean().round(3).to_string())
if std_rows:
    R = pd.DataFrame(std_rows); T = tabs
    out = []
    for _, r in R.iterrows():
        P = T[f"raw|{r.seed}"][:, :, :, 0]; s = [SHORT[n] for n in DATASETS].index(r.source)
        out.append(np.mean([P[r.mask_index, s, t] for t in range(3) if t != s]))
    R["cross_auc"] = out
    R.to_csv(os.path.join(OUT, "supfs_standard.csv"), index=False)
    print("\nstandard feature selection on the source, mean cross-dataset AUC by subset size:")
    print(R.groupby(["method", "k"])["cross_auc"].mean().unstack("k").round(3).to_string())
