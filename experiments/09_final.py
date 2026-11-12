"""Final method (frozen) and its ablations, on every target, attack prevalence and seed.

    knn_all   kNN distance to the benign reference set, all 30 features          (benign calibration only)
    knn_ga    the same detector on the subset chosen by a GA whose fitness is the label-free
              tail alarm rate (pu_pauc) on the unlabelled stream
    pu        benign-vs-unlabelled classifier (5 small bagged boosters)
    brace     Fisher combination of the conformal p-values of knn_ga and pu       (THE METHOD)
    fisher_all  the same combination with knn_all instead of knn_ga               (ablation: no selection)
    sup_target  LightGBM trained on the unlabelled stream WITH its true labels    (upper reference, not label-free)

Protocols
    random    calibration, unlabelled and test streams are random disjoint parts of the dataset
    temporal  calibration flows come from one capture day, the streams from the other days
              (CIC-IDS-2017: Monday; CSE-CIC-IDS2018: the first day)

    python experiments/09_final.py --seeds 5
Writes results/v2/final_methods.csv and final_families.csv (resumable)
"""
import os, sys, time, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from xids.config import DATASETS, OUT
from xids.features import POOL_V2
from xids.data import load, make_target_split
from xids.core import TargetContext
from xids.optimisers import optimise
from xids.pu import pu_bag_fit, pu_bag_predict
from xids.metrics import auc

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--budget", type=int, default=300)
ap.add_argument("--prev", default="natural,0.1,0.02,0.005"); ap.add_argument("--ncal", default="6000,20000")
ap.add_argument("--nadapt", type=int, default=30000); ap.add_argument("--datasets", default=",".join(DATASETS))
ap.add_argument("--protocol", default="random"); ap.add_argument("--tag", default="")
a = ap.parse_args()
path = os.path.join(OUT, f"final_methods{a.tag}.csv"); fpath = os.path.join(OUT, f"final_families{a.tag}.csv")
done = set()
if os.path.exists(path):
    p_ = pd.read_csv(path); done = set(zip(p_.target, p_.protocol, p_.prev.astype(str), p_.seed, p_.n_cal))
d = len(POOL_V2); ALL = np.ones(d, bool)
CAL_DAY = {"CIC-IDS-2017": "Monday", "CSE-CIC-IDS2018": "Wednesday-14-02-2018"}


def pvals(s, s_thr):
    srt = np.sort(s_thr); n = len(srt)
    return (1 + n - np.searchsorted(srt, s, "left")) / (n + 1.0)


t0 = time.time()
for name in a.datasets.split(","):
    data = load(name, POOL_V2); X, yall, meta = data
    fam_all = meta["family"].to_numpy(); grp = meta["grp"].to_numpy()
    for ncal in [int(v) for v in a.ncal.split(",")]:
        for prev_s in a.prev.split(","):
            prev = None if prev_s == "natural" else float(prev_s)
            for seed in range(a.seeds):
                if (name, a.protocol, prev_s, seed, ncal) in done:
                    continue
                if a.protocol == "temporal":
                    sp = make_target_split(yall, seed, ncal, a.nadapt, prev, 100000, cal_idx=np.flatnonzero(grp == CAL_DAY[name]),
                                           pool_idx=np.flatnonzero(grp != CAL_DAY[name]))
                else:
                    sp = make_target_split(yall, seed, ncal, a.nadapt, prev, 100000)
                ctx = TargetContext(name, seed, POOL_V2, split=sp, data=data)
                y = ctx.y_test; fam = fam_all[ctx.i_test]; rows, frows = [], []
                base = dict(target=name, protocol=a.protocol, prev=prev_s, seed=seed, n_cal=ncal, n_adapt=len(ctx.i_adapt),
                            att_frac=float(ctx.y_adapt.mean()), n_att=int(ctx.y_adapt.sum()))

                def add(method, s, s_thr, extra=None):
                    r = dict(base, method=method, auc=auc(y, s), auprc=float(average_precision_score(y, s)))
                    for al in (0.01, 0.05):
                        al_ = s > np.quantile(s_thr, 1 - al)
                        r[f"tpr@{al}"] = float(al_[y == 1].mean()); r[f"fpr@{al}"] = float(al_[y == 0].mean())
                        for f in np.unique(fam[y == 1]):
                            frows.append(dict(base, method=method, family=f, alpha=al, n=int((fam == f).sum()), tpr=float(al_[fam == f].mean())))
                    r.update(extra or {}); rows.append(r)

                sc = lambda m, part: ctx.scores(m, part).cpu().numpy()
                sk, skt = sc(ALL, "test"), sc(ALL, "thr"); add("knn_all", sk, skt, dict(k=d))
                t1 = time.time(); f_all = ctx.fitness(ALL, "pu_pauc")
                res = optimise("ga", lambda m: ctx.fitness(m, "pu_pauc"), d, a.budget, seed); gm = res["best_mask"]
                sg, sgt = sc(gm, "test"), sc(gm, "thr")
                add("knn_ga", sg, sgt, dict(k=int(gm.sum()), features=",".join(np.array(POOL_V2)[gm]), fit_sel=res["best_fit"], fit_all=f_all, secs=time.time() - t1))
                Zb = torch.cat([ctx.Z["ref"], ctx.Z["sel"]]).cpu().numpy(); Zu = ctx.Z["adapt"].cpu().numpy()
                Zthr, Zte = ctx.Z["thr"].cpu().numpy(), ctx.Z["test"].cpu().numpy()
                t1 = time.time(); ms = pu_bag_fit(Zb, Zu, seed); sp_, spt = pu_bag_predict(ms, Zte), pu_bag_predict(ms, Zthr)
                add("pu", sp_, spt, dict(secs=time.time() - t1))
                fz = lambda s1, t1_, s2, t2_: -np.log(pvals(s1, t1_)) - np.log(pvals(s2, t2_))
                add("brace", fz(sg, sgt, sp_, spt), fz(sgt, sgt, spt, spt))
                add("fisher_all", fz(sk, skt, sp_, spt), fz(skt, skt, spt, spt))
                if ctx.y_adapt.sum() >= 20:
                    m = LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.1, class_weight="balanced", n_jobs=8, verbose=-1,
                                       random_state=seed).fit(Zu, ctx.y_adapt)
                    add("sup_target", m.predict_proba(Zte)[:, 1], m.predict_proba(Zthr)[:, 1])
                pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
                pd.DataFrame(frows).to_csv(fpath, mode="a", header=not os.path.exists(fpath), index=False)
                print(f"{name} {a.protocol} ncal={ncal} prev={prev_s} seed={seed} ({time.time()-t0:.0f}s) " +
                      " ".join(f"{r['method']}={r['auc']:.3f}/{r['tpr@0.01']:.2f}" for r in rows), flush=True)
D = pd.read_csv(path); pd.set_option("display.width", 250)
print(D.groupby(["protocol", "n_cal", "prev", "method", "target"], sort=False)["auc"].mean().unstack("target").round(3).to_string())
