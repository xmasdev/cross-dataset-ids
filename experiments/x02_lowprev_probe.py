"""Development probe (NOT a reported experiment): what limits the benign-vs-unlabelled classifier
when attacks are rare in the unlabelled stream, and which variant is robust?

Uses development seeds >= 100 so that no design decision is taken on the seeds (0-4) that are reported.
    python experiments/x02_lowprev_probe.py
"""
import os, sys, time, warnings, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xids.config import DATASETS, OUT
from xids.features import POOL_V2
from xids.data import load
from xids.core import TargetContext
from xids.knn import knn_score
from xids.metrics import auc

d = len(POOL_V2); ALL = np.ones(d, bool)


def pvals(s, s_thr):
    srt = np.sort(s_thr); n = len(srt)
    return (1 + n - np.searchsorted(srt, s, "left")) / (n + 1.0)


def lgb(seed, trees, leaves, child, lr=0.05, col=0.8):
    return LGBMClassifier(n_estimators=trees, num_leaves=leaves, learning_rate=lr, min_child_samples=child, subsample=0.8,
                          subsample_freq=1, colsample_bytree=col, reg_lambda=5.0, class_weight="balanced", n_jobs=8,
                          random_state=seed, verbose=-1)


def fit_bag(make, Zb, Zu, K, seed, es=False, wu=None):
    """K models, each trained without one fold of benign and one fold of unlabelled flows."""
    rng = np.random.default_rng(seed); fb = rng.integers(0, K, len(Zb)); fu = rng.integers(0, K, len(Zu)); ms = []
    for k in range(K):
        tb, tu = fb != k, fu != k
        X = np.vstack([Zb[tb], Zu[tu]]); y = np.r_[np.zeros(tb.sum()), np.ones(tu.sum())]
        w = None if wu is None else np.r_[np.ones(tb.sum()), wu[tu]]
        m = make(seed + k)
        if es:
            Xv = np.vstack([Zb[~tb], Zu[~tu]]); yv = np.r_[np.zeros((~tb).sum()), np.ones((~tu).sum())]
            m.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], eval_metric="binary_logloss", callbacks=[early_stopping(30, verbose=False), log_evaluation(0)])
        else:
            m.fit(X, y, sample_weight=w)
        ms.append(m)
    return ms


pred = lambda ms, Z: np.mean([m.predict_proba(Z)[:, 1] for m in ms], axis=0)
rows = []
for name, prev, ncal, nad, seed in itertools.product(["CIC-IDS-2017", "UNSW-NB15", "TON_IoT"], [0.02, 0.005, None], [6000, 20000], [30000, 100000], [100]):
    if name != "CIC-IDS-2017" and nad > 30000:
        continue
    t0 = time.time()
    ctx = TargetContext(name, seed, POOL_V2, n_cal=ncal, n_adapt=nad, prevalence=prev, test_cap=100000)
    y = ctx.y_test
    Zb = torch.cat([ctx.Z["ref"], ctx.Z["sel"]]).cpu().numpy(); Zu = ctx.Z["adapt"].cpu().numpy()
    Zthr, Zte = ctx.Z["thr"].cpu().numpy(), ctx.Z["test"].cpu().numpy()
    kn = lambda part: ctx.scores(ALL, part).cpu().numpy()
    sk, skt, sku = kn("test"), kn("thr"), kn("adapt")
    base = dict(target=name, prev=prev if prev else "natural", n_cal=ncal, n_adapt=len(Zu), n_att=int(ctx.y_adapt.sum()))

    def rec(method, s, st):
        th = np.quantile(st, 0.99); fs = -np.log(pvals(sk, skt)) - np.log(pvals(s, st))
        rows.append(dict(base, method=method, auc=auc(y, s), tpr1=float((s > th)[y == 1].mean()), fpr1=float((s > th)[y == 0].mean()), fisher=auc(y, fs)))

    rec("knn", sk, skt)
    V = {"lgb_es(cur)": (lambda s: lgb(s, 600, 15, 40), 1, True),
         "lgb_es_bag5": (lambda s: lgb(s, 600, 15, 40), 5, True),
         "lgb100_l7": (lambda s: lgb(s, 100, 7, 100), 5, False),
         "lgb300_l7": (lambda s: lgb(s, 300, 7, 100), 5, False),
         "lgb100_l15": (lambda s: lgb(s, 100, 15, 40), 5, False),
         "lgb300_l31": (lambda s: lgb(s, 300, 31, 20), 5, False),
         "rf_leaf20": (lambda s: RandomForestClassifier(200, min_samples_leaf=20, max_features=0.4, class_weight="balanced_subsample", n_jobs=8, random_state=s), 1, False),
         "rf_leaf5": (lambda s: RandomForestClassifier(200, min_samples_leaf=5, max_features=0.4, class_weight="balanced_subsample", n_jobs=8, random_state=s), 1, False),
         "et_leaf10": (lambda s: ExtraTreesClassifier(300, min_samples_leaf=10, max_features=0.5, class_weight="balanced_subsample", n_jobs=8, random_state=s), 1, False),
         "lr": (lambda s: LogisticRegression(C=1.0, class_weight="balanced", max_iter=300), 1, False)}
    for meth, (make, K, es) in V.items():
        if K == 1 and not es:
            m = [make(seed).fit(np.vstack([Zb, Zu]), np.r_[np.zeros(len(Zb)), np.ones(len(Zu))])]
        elif K == 1:
            m = fit_bag(make, Zb, Zu, 5, seed, es=True)[:1]
        else:
            m = fit_bag(make, Zb, Zu, K, seed, es=es)
        rec(meth, pred(m, Zte), pred(m, Zthr))
    # kNN-guided enrichment: positives = the unlabelled flows the benign-only detector finds most unusual
    for q in (0.05, 0.2):
        top = sku >= np.quantile(sku, 1 - q)
        m = fit_bag(lambda s: lgb(s, 300, 7, 100), Zb, Zu[top], 5, seed)
        rec(f"enrich{q}_lgb300_l7", pred(m, Zte), pred(m, Zthr))
    # kNN-weighted unlabelled flows (weight = 1 / conformal p-value, capped)
    wu = np.minimum(1.0 / pvals(sku, skt), 50.0)
    m = fit_bag(lambda s: lgb(s, 300, 7, 100), Zb, Zu, 5, seed, wu=wu)
    rec("knnweight_lgb300_l7", pred(m, Zte), pred(m, Zthr))
    print(f"{name} prev={prev} ncal={ncal} nadapt={len(Zu)} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "x02_lowprev_probe.csv"), index=False)
R = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
R["cfg"] = R.target.str[:4] + "|" + R.prev.astype(str) + "|" + (R.n_cal // 1000).astype(str) + "k|" + (R.n_adapt // 1000).astype(str) + "k"
print(R.pivot(index="method", columns="cfg", values="auc").round(3).to_string())
print(R.pivot(index="method", columns="cfg", values="fisher").round(3).to_string())
