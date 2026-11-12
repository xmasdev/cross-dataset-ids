"""Benign-calibrated detection with NO source data and NO attack labels.

For each dataset as the deployment target: fit a detector on benign calibration flows of that
network only, score the disjoint test stream. Compares 13 detectors x 3 representations x 2 feature
pools, and sweeps the calibration size for the kNN detector.

    python experiments/03_benign_detectors.py --seeds 5
Writes results/v2/benign_detectors.csv and results/v2/calibration_size.csv
"""
import os, sys, time, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from xids.config import DATASETS, OUT
from xids.features import POOL_V1, POOL_V2
from xids.data import load, make_target_split
from xids.anchor import make_anchor
from xids.detectors import detect, DETECTORS
from xids.metrics import auc, average_precision_score

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--datasets", default=",".join(DATASETS)); ap.add_argument("--tag", default="")
a = ap.parse_args()
names = a.datasets.split(",")


def score_row(y, s, s_thr):
    r = {"auc": auc(y, s), "auprc": float(average_precision_score(y, s))}
    for al in (0.01, 0.05):
        th = np.quantile(s_thr, 1 - al); p = s > th
        r[f"tpr@{al}"] = float(p[y == 1].mean()); r[f"fpr@{al}"] = float(p[y == 0].mean())
    return r


rows, rows2 = [], []; t0 = time.time()
for name in names:
    for pool_name, pool in (("v2_30", POOL_V2), ("v1_12", POOL_V1)):
        X, y, _ = load(name, pool)
        for seed in range(a.seeds):
            sp = make_target_split(y, seed, n_cal=6000, n_adapt=30000, test_cap=100000)
            ref, thr, te = sp["cal"][:4200], sp["cal"][4200:], sp["test"]
            for rep in ("quantile", "robust_z", "log_std"):
                an = make_anchor(rep).fit(X[ref]); Zr, Zt, Zq = an.transform(X[ref]), an.transform(X[thr]), an.transform(X[te])
                for det in DETECTORS:
                    t1 = time.time(); s = detect(det, Zr, np.vstack([Zt, Zq]), seed)
                    r = dict(target=name, pool=pool_name, rep=rep, detector=det, seed=seed, secs=round(time.time() - t1, 2))
                    r.update(score_row(y[te], s[len(Zt):], s[:len(Zt)])); rows.append(r)
            # calibration-size sweep, kNN-5, quantile anchoring
            max_ben = int((y == 0).sum()) // 3
            for ncal in (50, 100, 300, 1000, 3000, 10000, 30000):
                if ncal > max_ben:
                    continue
                sp2 = make_target_split(y, seed, n_cal=ncal, n_adapt=1000, test_cap=100000)
                c = sp2["cal"]; nr = int(0.7 * len(c)); ref2, thr2, te2 = c[:nr], c[nr:], sp2["test"]
                an = make_anchor("quantile").fit(X[ref2])
                s = detect("knn5", an.transform(X[ref2]), np.vstack([an.transform(X[thr2]), an.transform(X[te2])]), seed)
                r = dict(target=name, pool=pool_name, n_cal=ncal, seed=seed); r.update(score_row(y[te2], s[len(thr2):], s[:len(thr2)])); rows2.append(r)
            print(f"{name} {pool_name} seed {seed} ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, f"benign_detectors{a.tag}.csv"), index=False)
            pd.DataFrame(rows2).to_csv(os.path.join(OUT, f"calibration_size{a.tag}.csv"), index=False)
D = pd.DataFrame(rows); pd.set_option("display.width", 220)
print(D.groupby(["pool", "rep", "detector"])["auc"].mean().unstack("rep").round(3).to_string())
print(D[(D.rep == "quantile")].groupby(["pool", "detector", "target"])["auc"].mean().unstack("target").round(3).to_string())
print(pd.DataFrame(rows2).groupby(["pool", "target", "n_cal"])[["auc", "tpr@0.05", "fpr@0.05"]].mean().round(3).to_string())
