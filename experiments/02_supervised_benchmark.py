"""Leak-free supervised benchmark: does a classifier trained on other networks work on a new one?

Protocols
    same   stratified 5-fold cross-validation inside one dataset (held-out folds, not training accuracy)
    pair   train on one dataset, test on another (6 ordered pairs)
    lodo   train on two datasets, test on the third (leave-one-dataset-out)

Representations (what target-side information the preprocessing may use)
    raw             L0  log1p + standardise with SOURCE statistics only
    dom_std         L1  standardise every domain with its own statistics (target: unlabelled stream)
    dom_quantile    L1  per-domain quantile transform (the mid-semester DA-MFS alignment)
    coral           L1  CORAL: re-colour source features with the target covariance
    benign_quantile L2  quantile transform anchored on each domain's benign traffic
                        (target: benign calibration flows, which are excluded from the test rows)
Models: see xids/models.py. `dann` and `dcoral` are deep domain-adaptation baselines (L1).

    python experiments/02_supervised_benchmark.py --seeds 3
Writes results/v2/supervised_benchmark.csv (resumable).
"""
import os, sys, time, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import linalg
from sklearn.model_selection import StratifiedKFold
from xids.config import DATASETS, OUT, SHORT
from xids.features import POOL_V2
from xids.data import load, stratified_cap, make_target_split
from xids.anchor import make_anchor
from xids.models import fit_predict, SK_MODELS
from xids.metrics import score_metrics, threshold_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--ntrain", type=int, default=100000)
ap.add_argument("--ntest", type=int, default=100000); ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default="supervised_benchmark.csv")
a = ap.parse_args()

ZOO = SK_MODELS + ["knn", "mlp", "ftt"]
LADDER_MODELS = ["lr", "rf", "lgbm", "mlp"]
LADDER_REPS = ["dom_std", "dom_quantile", "coral", "benign_quantile"]
DA_MODELS = ["dann", "dcoral"]
if a.quick:
    ZOO = ["majority", "lr", "lgbm", "knn", "mlp", "ftt"]; LADDER_MODELS = ["lr", "lgbm"]; a.ntrain = 20000; a.ntest = 20000; a.seeds = 1

path = os.path.join(OUT, a.out)
done = set()
if os.path.exists(path):
    prev = pd.read_csv(path)
    done = set(zip(prev.protocol, prev.rep, prev.model, prev.source, prev.target, prev.seed, prev.fold))
rows = []


def record(protocol, rep, model, src, tgt, seed, fold, y, s, secs):
    r = dict(protocol=protocol, rep=rep, model=model, source=src, target=tgt, seed=seed, fold=fold, secs=round(secs, 1))
    r.update(score_metrics(y, s)); r.update(threshold_metrics(y, (s >= 0.5).astype(int)))
    rows.append(r)


def flush():
    global rows
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False); rows = []


def coral(Xs, Xt):
    cs = np.cov(Xs.T) + np.eye(Xs.shape[1]); ct = np.cov(Xt.T) + np.eye(Xt.shape[1])
    A = linalg.fractional_matrix_power(cs, -0.5) @ linalg.fractional_matrix_power(ct, 0.5)
    return np.real((Xs - Xs.mean(0)) @ A + Xt.mean(0)).astype(np.float32)


DATA = {n: load(n, POOL_V2) for n in DATASETS}
t_all = time.time()
for seed in range(a.seeds):
    rng = np.random.default_rng(1000 + seed)
    tr, sp = {}, {}
    for n in DATASETS:
        X, y, meta = DATA[n]
        sp[n] = make_target_split(y, seed, n_cal=6000, n_adapt=30000, test_cap=a.ntest)
        free = np.setdiff1d(np.arange(len(y)), sp[n]["cal"])
        tr[n] = free[stratified_cap(y[free], a.ntrain, rng, strata=meta["attack_cat"].to_numpy()[free])]
    trh = {n: np.random.default_rng(seed).permutation(tr[n])[:a.ntrain // 2] for n in DATASETS}   # half-size sets for pooled training

    # ------------------------------------------------------------------ same-dataset 5-fold CV (raw)
    for n in DATASETS:
        X, y, _ = DATA[n]; Xs, ys = X[tr[n]], y[tr[n]]
        for fold, (i_tr, i_te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, ys)):
            an = make_anchor("log_std").fit(Xs[i_tr]); A, B = an.transform(Xs[i_tr]), an.transform(Xs[i_te])
            for m in ZOO:
                if ("same", "raw", m, n, n, seed, fold) in done:
                    continue
                t0 = time.time(); s = fit_predict(m, A, ys[i_tr], B, seed)
                record("same", "raw", m, n, n, seed, fold, ys[i_te], s, time.time() - t0)
            flush()
        print(f"[seed {seed}] same {n} done ({time.time()-t_all:.0f}s)", flush=True)

    # ------------------------------------------------------------------ representations
    def src_matrix(n, rep, idx):
        X, y, _ = DATA[n]
        if rep in ("raw", "dom_std", "coral"):
            return make_anchor("log_std").fit(X[idx])
        if rep == "dom_quantile":
            return make_anchor("quantile").fit(X[idx])
        if rep == "benign_quantile":
            return make_anchor("quantile").fit(X[idx][y[idx] == 0])

    def tgt_anchor(n, rep, src_anchor):
        X, y, _ = DATA[n]
        if rep in ("raw", "coral"):
            return src_anchor
        if rep == "dom_std":
            return make_anchor("log_std").fit(X[sp[n]["adapt"]])
        if rep == "dom_quantile":
            return make_anchor("quantile").fit(X[sp[n]["adapt"]])
        if rep == "benign_quantile":
            return make_anchor("quantile").fit(X[sp[n]["cal"]])

    # ------------------------------------------------------------------ pair + lodo
    settings = [("pair", [s], t) for s in DATASETS for t in DATASETS if s != t] + \
               [("lodo", [s for s in DATASETS if s != t], t) for t in DATASETS]
    for protocol, srcs, tgt in settings:
        sname = "+".join(SHORT[s] for s in srcs)
        Xt_raw, yt, _ = DATA[tgt]; te = sp[tgt]["test"]; y_te = yt[te]
        for rep in ["raw"] + LADDER_REPS:
            models = (ZOO + DA_MODELS) if rep == "raw" else LADDER_MODELS
            todo = [m for m in models if (protocol, rep, m, sname, tgt, seed, -1) not in done]
            if not todo:
                continue
            parts, ys_ = [], []
            anchors = []
            for s in srcs:
                idx = trh[s] if len(srcs) > 1 else tr[s]
                an = src_matrix(s, rep, idx); anchors.append(an)
                parts.append((DATA[s][0][idx], an)); ys_.append(DATA[s][1][idx])
            if rep in ("raw", "coral") and len(srcs) > 1:            # one shared source scaler for pooled sources
                pooled = np.vstack([p[0] for p in parts]); an = make_anchor("log_std").fit(pooled)
                A = an.transform(pooled); anchors = [an]
            else:
                A = np.vstack([an.transform(Xp) for Xp, an in parts])
            ytr = np.concatenate(ys_)
            ta = tgt_anchor(tgt, rep, anchors[0])
            B = ta.transform(Xt_raw[te]); U = ta.transform(Xt_raw[sp[tgt]["adapt"]])
            if rep == "coral":
                A = coral(A, U)
            for m in todo:
                t0 = time.time()
                s = fit_predict(m, A, ytr, B, seed, Xtgt_unlab=U if m in DA_MODELS else None)
                record(protocol, rep, m, sname, tgt, seed, -1, y_te, s, time.time() - t0)
            flush()
        print(f"[seed {seed}] {protocol} {sname}->{SHORT[tgt]} done ({time.time()-t_all:.0f}s)", flush=True)
print("finished", time.time() - t_all)
