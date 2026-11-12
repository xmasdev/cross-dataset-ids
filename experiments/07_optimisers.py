"""Comparison of the metaheuristic optimisers on the label-free selection fitness.

Part A  12-feature pool, where the exhaustive tables of 06_landscape.py give the exact optimum:
        every optimiser x 50 runs per landscape, budget 400 unique evaluations (about 10% of the space).
        Reported: how often the global optimum is found, mean regret, evaluations needed.
Part B  30-feature pool (2^30 subsets, no ground truth): every optimiser x R runs, real fitness,
        convergence curve, quality of the selected subset on the held-out test stream.

    python experiments/07_optimisers.py --part A
    python experiments/07_optimisers.py --part B --runs 8 --budget 800
Writes results/v2/optimisers_small.csv, optimisers_large.csv, optimisers_large_hist.npz
"""
import os, sys, time, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from xids.config import DATASETS, OUT
from xids.features import POOL_V1, POOL_V2
from xids.optimisers import optimise, ALGOS

ap = argparse.ArgumentParser(); ap.add_argument("--part", default="A"); ap.add_argument("--runs", type=int, default=8)
ap.add_argument("--budget", type=int, default=800); ap.add_argument("--fitness", default="pu_auc")
ap.add_argument("--prev", default="natural"); ap.add_argument("--datasets", default=",".join(DATASETS))
a = ap.parse_args()
KINDS = ["pu_auc", "alarm", "pu_pauc"]

if a.part == "A":
    tabs = np.load(os.path.join(OUT, "landscape_tables.npz")); d = len(POOL_V1)
    w = 1 << np.arange(d)
    rows = []
    keys = sorted({k.rsplit("|", 1)[0] for k in tabs.files})
    for key in keys:
        name, prev, seed = key.split("|")
        fit = tabs[key + "|fit"][KINDS.index(a.fitness)]; true = tabs[key + "|true"]
        opt = fit.max(); f = lambda m: fit[int((m * w).sum()) - 1]
        for algo in ALGOS:
            for run in range(50):
                r = optimise(algo, f, d, 400, run)
                h = r["history"]; hit = np.flatnonzero(h >= opt - 1e-12)
                idx = int((r["best_mask"] * w).sum()) - 1
                rows.append(dict(target=name, prev=prev, data_seed=int(seed), algo=algo, run=run, found=len(hit) > 0,
                                 evals_to_opt=int(hit[0]) + 1 if len(hit) else np.nan,
                                 regret50=opt - h[49], regret100=opt - h[99], regret200=opt - h[199], regret400=opt - h[399],
                                 true_auc_selected=true[idx], true_auc_opt_fitness=true[fit.argmax()], true_auc_oracle=true.max()))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "optimisers_small.csv"), index=False)
    pd.set_option("display.width", 200)
    g = R[R.prev == "natural"].groupby("algo")
    print(pd.DataFrame({"found_optimum_%": g.found.mean() * 100, "median_evals_to_opt": g.evals_to_opt.median(),
                        "regret@50": g.regret50.mean(), "regret@100": g.regret100.mean(), "regret@400": g.regret400.mean()}).round(4).to_string())
else:
    from xids.data import load
    from xids.core import TargetContext
    d = len(POOL_V2); rows = []; hist = {}; t0 = time.time()
    out_csv = os.path.join(OUT, "optimisers_large.csv")
    for name in a.datasets.split(","):
        ctx = TargetContext(name, 0, POOL_V2, n_cal=6000, n_adapt=30000, test_cap=50000,
                            prevalence=None if a.prev == "natural" else float(a.prev))
        cache = {}

        def f(m):
            k = m.tobytes()
            if k not in cache:
                cache[k] = ctx.fitness(m, a.fitness)
            return cache[k]
        f_all = f(np.ones(d, bool)); auc_all = ctx.test_auc(np.ones(d, bool))
        for algo in ALGOS:
            for run in range(1 if algo == "sfs" else a.runs):
                r = optimise(algo, f, d, a.budget, run)
                hist[f"{name}|{algo}|{run}"] = r["history"]
                rows.append(dict(target=name, algo=algo, run=run, best_fit=r["best_fit"], fit_all=f_all, k=int(r["best_mask"].sum()),
                                 test_auc=ctx.test_auc(r["best_mask"]), test_auc_all=auc_all, evals=r["evals"],
                                 features=",".join(np.array(POOL_V2)[r["best_mask"]])))
            pd.DataFrame(rows).to_csv(out_csv, index=False); np.savez_compressed(os.path.join(OUT, "optimisers_large_hist.npz"), **hist)
            print(f"{name} {algo} done ({time.time()-t0:.0f}s, cache {len(cache)})", flush=True)
    R = pd.DataFrame(rows); pd.set_option("display.width", 200)
    print(R.groupby(["target", "algo"])[["best_fit", "test_auc", "k"]].agg(["mean", "std"]).round(4).to_string())
