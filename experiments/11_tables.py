"""Every table of the final report, as Markdown, from the CSV files in results/v2/  ->  results/v2/TABLES.md

    python experiments/11_tables.py
"""
import os, sys, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from xids.config import OUT

rd = lambda f: pd.read_csv(os.path.join(OUT, f)); have = lambda f: os.path.exists(os.path.join(OUT, f))
T3 = ["CIC-IDS-2017", "UNSW-NB15", "TON_IoT"]; out = []


def md(df, title, floatfmt=".3f"):
    out.append(f"\n### {title}\n\n" + df.to_markdown(floatfmt=floatfmt) + "\n")


def pm(g):
    return f"{g.mean():.3f} ± {g.std():.3f}"


if have("harmonised_summary.csv"):
    md(rd("harmonised_summary.csv").set_index("dataset") if "dataset" in rd("harmonised_summary.csv") else rd("harmonised_summary.csv"), "T1 harmonised datasets")

if have("supervised_benchmark.csv"):
    S = rd("supervised_benchmark.csv")
    R = S[S.rep == "raw"].groupby(["model", "protocol"])[["auc", "balanced_accuracy", "macro_f1"]].mean().unstack("protocol")
    R.columns = [f"{m} ({p})" for m, p in R.columns]; cols = [c for p in ("same", "pair", "lodo") for c in R.columns if f"({p})" in c and ("auc" in c or "balanced" in c)]
    md(R[cols].sort_values("auc (pair)", ascending=False), "T2 supervised models, raw features: same-dataset 5-fold CV vs one-source (pair) vs two-source (lodo) transfer")
    A = S[(S.protocol == "pair")].groupby(["rep", "model"]).auc.mean().unstack("model")
    md(A[[c for c in ["lr", "rf", "lgbm", "mlp", "dann", "dcoral"] if c in A.columns]], "T3a alignment / domain adaptation, one source -> one target, AUC")
    A = S[(S.protocol == "lodo")].groupby(["rep", "model"]).auc.mean().unstack("model")
    md(A[[c for c in ["lr", "rf", "lgbm", "mlp", "dann", "dcoral"] if c in A.columns]], "T3b alignment / domain adaptation, two sources -> held-out target, AUC")
    P = S[(S.protocol == "pair") & (S.rep == "raw") & (S.model == "lgbm")].groupby(["source", "target"]).auc.mean().unstack("target")
    md(P, "T3c LightGBM raw, source (row) -> target (column) AUC")
    md(S[S.protocol.isin(["pair", "lodo"]) & (S.model != "majority")].groupby(["target", "protocol", "rep", "model"]).auc.mean().groupby("target").agg(["max", "median"]),
       "T3d best and median supervised-transfer AUC per target over every model x alignment x protocol (best = oracle choice)")
    md(S[(S.protocol == "pair") & (S.rep == "raw")].groupby("model")[["accuracy", "balanced_accuracy", "macro_f1", "mcc", "tpr", "fpr"]].mean(), "T3e cross-dataset threshold metrics at 0.5, raw")

if have("supfs_summary.csv"):
    F = rd("supfs_summary.csv")
    md(F.groupby(["metric", "rep"])[["all_features", "oracle_selected", "oracle_k", "leakfree_selected", "per_target_oracle", "median_subset", "same_dataset"]].mean(),
       f"T4 mid-semester selection re-examined on all 4,095 subsets (seeds per rep: {F.groupby('rep').seed.nunique().to_dict()})")
if have("supfs_standard.csv"):
    R = rd("supfs_standard.csv"); md(R.groupby(["method", "k"]).cross_auc.mean().unstack("k"), "T4b standard feature selection on the source: mean cross-dataset AUC by subset size", ".2f")

if have("diag_domain.csv"):
    md(rd("diag_domain.csv").set_index(["rep", "pair"]), "T5a telling datasets apart from benign flows only")
    C = rd("diag_family.csv"); md(C.pivot_table(index=["family", "source"], columns="target", values="auc"), "T5b benign-vs-one-family: trained on row, tested on column")

if have("final_methods.csv") and rd("final_methods.csv").target.nunique() == 3:
    D = rd("final_methods.csv"); D["prev"] = D.prev.astype(str); order = ["knn_all", "knn_ga", "pu", "fisher_all", "brace", "sup_target"]
    n = D.groupby(["target", "n_cal", "prev"]).seed.nunique()
    out.append(f"\nconfigs done: {len(n)} of 24; seeds per config min {n.min()} max {n.max()}\n")
    for ncal in sorted(D.n_cal.unique()):
        M = D[(D.n_cal == ncal) & (D.prev == "natural")]
        for met in ["auc", "auprc", "tpr@0.01", "tpr@0.05", "fpr@0.01", "fpr@0.05"]:
            md(M.groupby(["method", "target"])[met].apply(pm).unstack("target").reindex(order)[T3], f"T6 natural prevalence, n_cal={ncal}: {met}")
        P = D[D.n_cal == ncal].groupby(["target", "prev", "method"]).auc.mean().unstack("method")[order]
        md(P.reindex([(t, p) for t in T3 for p in ["natural", "0.1", "0.02", "0.005"]]), f"T7 AUC by attack share of the unlabelled stream, n_cal={ncal}")
        P = D[D.n_cal == ncal].groupby(["target", "prev", "method"])["tpr@0.01"].mean().unstack("method")[order]
        md(P.reindex([(t, p) for t in T3 for p in ["natural", "0.1", "0.02", "0.005"]]), f"T7b detection rate at 1% false alarms, n_cal={ncal}")
    G = D[D.method == "knn_ga"]; md(G.groupby(["target", "prev"])[["k", "fit_sel", "fit_all", "secs"]].mean(), "T8 GA selection: subset size, fitness of selected vs all features, seconds")
    if "features" in G:
        freq = G[G.prev == "natural"].assign(f=G.features.str.split(",")).explode("f").groupby(["f", "target"]).size().unstack("target").fillna(0)
        freq = (freq / G[G.prev == "natural"].groupby("target").size()).round(2); md(freq.loc[freq.mean(1).sort_values(ascending=False).index], "T8b selection frequency of each feature (natural prevalence, both n_cal)", ".2f")
    W = D.pivot_table(index=["target", "n_cal", "prev", "seed"], columns="method", values=["auc", "tpr@0.01"])
    rows = []
    for met in ["auc", "tpr@0.01"]:
        for a_, b_ in [("brace", "knn_all"), ("brace", "pu"), ("brace", "fisher_all"), ("knn_ga", "knn_all"), ("pu", "knn_all")]:
            x, y = W[met][a_], W[met][b_]; ok = x.notna() & y.notna(); dlt = (x - y)[ok]
            rows.append(dict(metric=met, comparison=f"{a_} vs {b_}", n=int(ok.sum()), mean_diff=dlt.mean(), wins=int((dlt > 0).sum()), losses=int((dlt < 0).sum()),
                             worst=dlt.min(), p_wilcoxon=wilcoxon(dlt).pvalue if (dlt != 0).any() else 1.0))
    md(pd.DataFrame(rows).set_index(["metric", "comparison"]), "T9 paired comparisons over every (target, calibration size, prevalence, seed)", ".4g")
    md(D.groupby(["method"])[["fpr@0.01", "fpr@0.05"]].agg(["mean", "max"]), "T9b realised false-alarm rate over all configurations")

if have("final_families.csv"):
    F = rd("final_families.csv"); F = F[(F.n_cal == 6000) & (F.prev.astype(str) == "natural") & (F.n >= 30)]
    for al in (0.05, 0.01):
        md(F[F.alpha == al].pivot_table(index=["target", "family"], columns="method", values="tpr")[["knn_all", "knn_ga", "pu", "brace", "sup_target"]], f"T10 detection rate per attack family at {al:.0%} false alarms (natural prevalence, n_cal=6000)")

for f, title in (("final_methods_temporal.csv", "T11 temporal protocol"), ("final_methods_external.csv", "T12 external dataset CSE-CIC-IDS2018")):
    if have(f):
        D = rd(f); D["prev"] = D.prev.astype(str)
        for met in ["auc", "tpr@0.01", "tpr@0.05", "fpr@0.01", "fpr@0.05"]:
            md(D.groupby(["target", "protocol", "n_cal", "prev", "method"])[met].mean().unstack("method")[["knn_all", "knn_ga", "pu", "fisher_all", "brace", "sup_target"]], f"{title}: {met}")
        out.append(f"\nseeds per config: {D.groupby(['protocol', 'n_cal', 'prev']).seed.nunique().to_dict()}\n")

if have("optimisers_large.csv"):
    R = rd("optimisers_large.csv")
    md(R.groupby(["algo", "target"])[["best_fit", "test_auc", "k"]].mean().unstack("target"), "T13 optimisers on the label-free fitness (30 features, budget 300, 3 runs)", ".4f")
    md(R.groupby("target")[["fit_all", "test_auc_all"]].first(), "T13b all-feature reference", ".4f")

if have("benign_detectors.csv"):
    B = rd("benign_detectors.csv")
    md(B[(B.pool == "v2_30") & (B.rep == "quantile")].groupby(["detector", "target"]).auc.mean().unstack("target"), f"T14 benign-only detectors, 30 features, quantile anchoring (seeds {B.seed.nunique()})")
    md(B[B.detector == "knn5"].groupby(["pool", "rep", "target"]).auc.mean().unstack("target"), "T14b kNN-5 by representation and pool")
    md(B[(B.pool == "v2_30")].groupby(["rep", "detector"]).auc.mean().unstack("rep"), "T14c detector x representation, mean over targets")
if have("calibration_size.csv"):
    C = rd("calibration_size.csv"); md(C[C.pool == "v2_30"].groupby(["n_cal", "target"])[["auc", "fpr@0.05"]].mean().unstack("target"), "T15 calibration-set size, kNN-5")

open(os.path.join(OUT, "TABLES.md"), "w").write("# Tables generated by experiments/11_tables.py\n" + "\n".join(out))
print("wrote TABLES.md,", len(out), "blocks")
