"""Why does a source-trained classifier fail? Three measurements.

(a) direction   For every feature and dataset: AUC of the raw feature value for attack-vs-benign.
                > 0.5 means attacks have larger values, < 0.5 smaller. A feature whose direction
                flips between datasets tells a source-trained model the opposite of the truth.
(b) domain      How easily can a classifier tell which dataset a BENIGN flow comes from, before and
                after benign anchoring? (marginal shift, and how much anchoring removes)
(c) family      Train "benign vs attack family F" on one dataset, test on the same family elsewhere
                (benign-anchored features). Removes the label-set mismatch and isolates whether one
                attack family looks the same on two networks.

    python experiments/05_shift_diagnosis.py
Writes results/v2/diag_direction.csv, diag_domain.csv, diag_family.csv
"""
import os, sys, itertools, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from xids.config import DATASETS, OUT, SHORT
from xids.features import POOL_V2, FAMILY_OF
from xids.data import load
from xids.anchor import make_anchor
from xids.metrics import auc

rng = np.random.default_rng(0)
D = {n: load(n, POOL_V2) for n in DATASETS}
lgb = lambda: LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.1, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, class_weight="balanced", n_jobs=6, verbose=-1, random_state=0)

# ------------------------------------------------------------------ (a) direction of every feature
rows = []
for n in DATASETS:
    X, y, _ = D[n]; i = rng.choice(len(y), min(len(y), 300000), replace=False)
    for j, f in enumerate(POOL_V2):
        rows.append(dict(feature=f, family=FAMILY_OF[f], dataset=SHORT[n], auc=auc(y[i], X[i, j])))
A = pd.DataFrame(rows).pivot(index=["family", "feature"], columns="dataset", values="auc")[[SHORT[n] for n in DATASETS]]
strong = (A - 0.5).abs() > 0.1
A["flips"] = [int(any((r[a] - 0.5) * (r[b] - 0.5) < 0 and abs(r[a] - 0.5) > 0.1 and abs(r[b] - 0.5) > 0.1
                      for a, b in itertools.combinations(A.columns[:3], 2))) for _, r in A.iterrows()]
A.to_csv(os.path.join(OUT, "diag_direction.csv"))
pd.set_option("display.width", 200)
print("=== (a) single-feature attack-vs-benign AUC (0.5 = uninformative) ===")
print(A.round(3).to_string()); print("features with a strong direction flip between two datasets:", int(A.flips.sum()), "of", len(A))

# ------------------------------------------------------------------ (b) can benign flows be told apart by dataset?
rows = []
for rep in ("raw", "benign_quantile"):
    Xs, ds = [], []
    for k, n in enumerate(DATASETS):
        X, y, _ = D[n]; ben = np.flatnonzero(y == 0); idx = rng.choice(ben, 30000, replace=False)
        if rep == "raw":
            Z = X[idx]
        else:                                   # anchor on a DISJOINT benign sample of the same dataset
            ref = rng.choice(np.setdiff1d(ben, idx), 6000, replace=False); Z = make_anchor("quantile").fit(X[ref]).transform(X[idx])
        Xs.append(Z); ds.append(np.full(len(idx), k))
    Xa, da = np.vstack(Xs), np.concatenate(ds)
    for a_, b_ in itertools.combinations(range(3), 2):
        m = np.isin(da, [a_, b_]); yy = (da[m] == b_).astype(int)
        p = cross_val_predict(lgb(), Xa[m], yy, cv=StratifiedKFold(3, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        rows.append(dict(rep=rep, pair=f"{SHORT[DATASETS[a_]]} vs {SHORT[DATASETS[b_]]}", domain_auc=auc(yy, p)))
        per = [abs(auc(yy, Xa[m][:, j]) - 0.5) + 0.5 for j in range(Xa.shape[1])]
        rows[-1]["mean_single_feature_auc"] = float(np.mean(per))
B = pd.DataFrame(rows); B.to_csv(os.path.join(OUT, "diag_domain.csv"), index=False)
print("\n=== (b) telling datasets apart from BENIGN flows only (0.5 = indistinguishable) ===")
print(B.round(3).to_string(index=False))

# ------------------------------------------------------------------ (c) family-matched transfer
FAMS = ["dos", "recon", "bruteforce", "web", "backdoor"]
Z = {}
for n in DATASETS:
    X, y, meta = D[n]; ben = np.flatnonzero(y == 0); ref = rng.choice(ben, 6000, replace=False)
    Z[n] = (make_anchor("quantile").fit(X[ref]).transform(X), y, meta["family"].to_numpy(), np.setdiff1d(ben, ref))
rows = []
for fam in FAMS:
    have = [n for n in DATASETS if (Z[n][2] == fam).sum() >= 500]
    for s in have:
        Zs, ys, fs, bs = Z[s]
        ia = np.flatnonzero(fs == fam); ia = rng.choice(ia, min(len(ia), 40000), replace=False); ib = rng.choice(bs, min(len(bs), 40000), replace=False)
        idx = np.r_[ia, ib]; yy = np.r_[np.ones(len(ia)), np.zeros(len(ib))]
        p_in = cross_val_predict(lgb(), Zs[idx], yy, cv=StratifiedKFold(3, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        rows.append(dict(family=fam, source=SHORT[s], target=SHORT[s], auc=auc(yy, p_in), n_attack=len(ia)))
        clf = lgb().fit(Zs[idx], yy)
        for t in have:
            if t == s:
                continue
            Zt, yt, ft, bt = Z[t]
            ja = np.flatnonzero(ft == fam); ja = rng.choice(ja, min(len(ja), 40000), replace=False); jb = rng.choice(bt, min(len(bt), 40000), replace=False)
            jj = np.r_[ja, jb]; yt2 = np.r_[np.ones(len(ja)), np.zeros(len(jb))]
            rows.append(dict(family=fam, source=SHORT[s], target=SHORT[t], auc=auc(yt2, clf.predict_proba(Zt[jj])[:, 1]), n_attack=len(ja)))
C = pd.DataFrame(rows); C.to_csv(os.path.join(OUT, "diag_family.csv"), index=False)
print("\n=== (c) benign-vs-one-family, trained on row, tested on column (benign-anchored features) ===")
for fam in FAMS:
    sub = C[C.family == fam]
    if len(sub):
        print(f"\n{fam}"); print(sub.pivot(index="source", columns="target", values="auc").round(3).to_string())
