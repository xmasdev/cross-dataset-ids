"""Target-side methods that use NO attack labels: benign calibration flows + the unlabelled stream.

For each dataset as the deployment target, each seed and each attack prevalence of the unlabelled
adaptation stream, evaluate on a disjoint test stream:

    knn_all        kNN distance to benign reference, all 30 features (needs benign calibration only)
    knn_v1         same, on the 12 mid-semester features
    knn_ga_*       kNN + GA feature selection with a label-free fitness (pu_auc / alarm / pu_pauc)
    knn_sfs_*      kNN + greedy forward selection with the same fitness
    pu             benign-vs-unlabelled gradient boosting (semi-supervised novelty detection)
    pu_knn         the same classifier with the kNN score as an additional input
    fisher         Fisher combination of the conformal p-values of knn_all and pu
    knn_oracle     GA on the TRUE test AUC: an upper bound, not a usable method

    python experiments/04_target_side_methods.py --seeds 5
Writes results/v2/target_side_methods.csv (resumable).
"""
import os, sys, time, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from xids.config import DATASETS, OUT
from xids.features import POOL_V1, POOL_V2
from xids.data import load
from xids.core import TargetContext
from xids.knn import knn_score
from xids.optimisers import optimise
from xids.pu import pu_fit, pu_predict
from xids.metrics import auc

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--budget", type=int, default=1200)
ap.add_argument("--prev", default="natural,0.1,0.02,0.005"); ap.add_argument("--ncal", type=int, default=6000)
ap.add_argument("--nadapt", type=int, default=30000); ap.add_argument("--datasets", default=",".join(DATASETS))
ap.add_argument("--out", default="target_side_methods.csv"); ap.add_argument("--no-oracle", action="store_true")
ap.add_argument("--fitness", default="pu_auc,alarm,pu_pauc")
a = ap.parse_args()
path = os.path.join(OUT, a.out)
done = set()
if os.path.exists(path):
    p_ = pd.read_csv(path); done = set(zip(p_.target, p_.prev.astype(str), p_.seed, p_.n_cal, p_.n_adapt))
V1 = np.array([f in POOL_V1 for f in POOL_V2]); d = len(POOL_V2); ALL = np.ones(d, bool)


def eval_scores(y, s, s_thr):
    r = {"auc": auc(y, s)}
    for al in (0.01, 0.05):
        th = np.quantile(s_thr, 1 - al); p = s > th
        r[f"tpr@{al}"] = float(p[y == 1].mean()); r[f"fpr@{al}"] = float(p[y == 0].mean())
    return r


def pvals(s, s_thr):
    srt = np.sort(s_thr); n = len(srt)
    return (1 + n - np.searchsorted(srt, s, "left")) / (n + 1.0)


t0 = time.time()
for name in a.datasets.split(","):
    data = load(name, POOL_V2)
    for prev_s in a.prev.split(","):
        prev = None if prev_s == "natural" else float(prev_s)
        for seed in range(a.seeds):
            if (name, prev_s, seed, a.ncal, a.nadapt) in done:
                continue
            ctx = TargetContext(name, seed, POOL_V2, n_cal=a.ncal, n_adapt=a.nadapt, prevalence=prev, data=data, test_cap=100000)
            y = ctx.y_test; rows = []
            base = dict(target=name, prev=prev_s, seed=seed, n_cal=a.ncal, n_adapt=len(ctx.i_adapt), att_frac=float(ctx.y_adapt.mean()))

            def add(method, res, mask=None, extra=None):
                r = dict(base, method=method, k=int(mask.sum()) if mask is not None else d,
                         features=",".join(np.array(POOL_V2)[mask]) if mask is not None else "")
                r.update(res); r.update(extra or {}); rows.append(r)

            def knn_eval(mask):
                return eval_scores(y, ctx.scores(mask, "test").cpu().numpy(), ctx.scores(mask, "thr").cpu().numpy())

            add("knn_all", knn_eval(ALL), ALL); add("knn_v1", knn_eval(V1), V1)
            for kind in a.fitness.split(","):
                fit = lambda m, kind=kind: ctx.fitness(m, kind)
                f_all = fit(ALL)
                for algo in (("ga", "sfs") if kind == "pu_auc" else ("ga",)):
                    t1 = time.time(); res = optimise(algo, fit, d, a.budget, seed)
                    add(f"knn_{algo}_{kind}", knn_eval(res["best_mask"]), res["best_mask"],
                        dict(fit_sel=res["best_fit"], fit_all=f_all, secs=time.time() - t1))
            if not a.no_oracle:
                res = optimise("ga", lambda m: ctx.test_auc(m), d, a.budget, seed)
                add("knn_oracle", knn_eval(res["best_mask"]), res["best_mask"])
            # benign-vs-unlabelled classifier
            Zb = torch.cat([ctx.Z["ref"], ctx.Z["sel"]]).cpu().numpy(); Zu = ctx.Z["adapt"].cpu().numpy()
            Zthr, Zte = ctx.Z["thr"].cpu().numpy(), ctx.Z["test"].cpu().numpy()
            m = pu_fit(Zb, Zu, seed); s_pu, s_pu_thr = pu_predict(m, Zte), pu_predict(m, Zthr)
            add("pu", eval_scores(y, s_pu, s_pu_thr), None, dict(n_trees=m.best_iteration_))
            kn = lambda part: ctx.scores(ALL, part).cpu().numpy()[:, None]
            k_ref = knn_score(ctx.Z["ref"], ctx.Z["ref"], k=ctx.k, loo=True).cpu().numpy()[:, None]   # leave-one-out for reference rows
            Kb = np.vstack([k_ref, kn("sel")])
            m2 = pu_fit(np.hstack([Zb, Kb]), np.hstack([Zu, kn("adapt")]), seed)
            add("pu_knn", eval_scores(y, pu_predict(m2, np.hstack([Zte, kn("test")])), pu_predict(m2, np.hstack([Zthr, kn("thr")]))))
            s_k, s_k_thr = kn("test")[:, 0], kn("thr")[:, 0]
            fz = lambda sk, sp, skt, spt: -np.log(pvals(sk, skt)) - np.log(pvals(sp, spt))
            add("fisher", eval_scores(y, fz(s_k, s_pu, s_k_thr, s_pu_thr), fz(s_k_thr, s_pu_thr, s_k_thr, s_pu_thr)))
            pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            print(f"{name} prev={prev_s} seed={seed} ({time.time()-t0:.0f}s) " +
                  " ".join(f"{r['method']}={r['auc']:.3f}" for r in rows), flush=True)
D = pd.read_csv(path); pd.set_option("display.width", 250)
print(D.groupby(["prev", "method", "target"], sort=False)["auc"].mean().unstack("target").round(3).to_string())
