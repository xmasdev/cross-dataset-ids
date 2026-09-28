"""
Baseline cross-dataset evaluation for ML-based intrusion detection.

WHAT THIS DOES
--------------
Trains a baseline classifier on each of three datasets and evaluates it on all
three, producing a 3x3 "trained on -> tested on" table of accuracy, balanced
accuracy and macro-F1.

Because the three datasets use different feature extractors, we project them
onto a small SHARED SEMANTIC SUBSPACE of features that are computable in all
three. This is the only way to train on one dataset and test on another.

This script is the *baseline only*. Metaheuristic / transfer-oriented feature
selection is intentionally left for future (end-semester) work.

AVAILABLE MODELS (--models)
---------------------------
    rf     Random Forest (default; strong, robust)
    et     Extra Trees (extremely randomised trees)
    hgb    Histogram Gradient Boosting (sklearn; fast on large data)
    xgb    XGBoost (gradient boosting; requires `pip install xgboost`)
    lgbm   LightGBM (requires `pip install lightgbm`)
    vote   Soft-voting ensemble of rf + et + hgb (+ xgb if installed)

OUTPUTS (written to  results/ )
-------------------------------
    harmonized_<dataset>_n<sample>.csv   cached cleaned shared-subspace data
    baseline_long.csv                    tidy table: model, source, target, metrics
    baseline_<model>_accuracy.csv        3x3 accuracy matrix per model
    baseline_<model>_balanced_acc.csv    3x3 balanced-accuracy matrix per model
    baseline_<model>_macro_f1.csv        3x3 macro-F1 matrix per model

USAGE
-----
    # entire datasets, all ensemble models
    python src/baseline_cross_dataset.py --models rf,et,hgb,xgb,vote

    # a single model
    python src/baseline_cross_dataset.py --models xgb

    # subsample for a quick smoke test (50k rows per dataset)
    python src/baseline_cross_dataset.py --sample 50000 --models rf

    # ignore cached harmonised CSVs and re-read raw data
    python src/baseline_cross_dataset.py --force-reload

DEPENDENCIES
------------
    pip install pandas numpy scikit-learn xgboost
    pip install lightgbm        # only if you use --models lgbm
"""

import os
import time
import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "datasets")
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

CICIDS_DIR = os.path.join(DATA, "CIC-IDS-2017", "MachineLearningCVE")
UNSW_TRAIN = os.path.join(DATA, "UNSW-NB15", "UNSW_NB15_training-set.csv")
UNSW_TEST = os.path.join(DATA, "UNSW-NB15", "UNSW_NB15_testing-set.csv")
TON_FILE = os.path.join(DATA, "TON_IOT network train-test",
                        "Train_Test_Network_dataset", "train_test_network.csv")

# ----------------------------------------------------------------------------
# Shared semantic feature subspace (raw + derived)
# ----------------------------------------------------------------------------
FEATURES = [
    "duration",              # seconds (CICIDS microseconds converted)
    "src_bytes",
    "dst_bytes",
    "src_pkts",
    "dst_pkts",
    "total_bytes",           # derived
    "total_pkts",            # derived
    "bytes_per_pkt_src",     # derived
    "bytes_per_pkt_dst",     # derived
    "total_bytes_per_pkt",   # derived
    "src_dst_byte_ratio",    # derived
    "src_dst_pkt_ratio",     # derived
]

EPS = 1e-9


# ----------------------------------------------------------------------------
# Harmonisation helpers
# ----------------------------------------------------------------------------
def _derive(df):
    """Given raw columns duration/src_bytes/dst_bytes/src_pkts/dst_pkts, add
    derived features and return the shared-subspace frame with `label`."""
    src_bytes = pd.to_numeric(df["src_bytes"], errors="coerce")
    dst_bytes = pd.to_numeric(df["dst_bytes"], errors="coerce")
    src_pkts = pd.to_numeric(df["src_pkts"], errors="coerce")
    dst_pkts = pd.to_numeric(df["dst_pkts"], errors="coerce")
    duration = pd.to_numeric(df["duration"], errors="coerce")

    total_bytes = src_bytes + dst_bytes
    total_pkts = src_pkts + dst_pkts

    out = pd.DataFrame({
        "duration": duration,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "src_pkts": src_pkts,
        "dst_pkts": dst_pkts,
        "total_bytes": total_bytes,
        "total_pkts": total_pkts,
        "bytes_per_pkt_src": src_bytes / (src_pkts + EPS),
        "bytes_per_pkt_dst": dst_bytes / (dst_pkts + EPS),
        "total_bytes_per_pkt": total_bytes / (total_pkts + EPS),
        "src_dst_byte_ratio": src_bytes / (dst_bytes + 1.0),
        "src_dst_pkt_ratio": src_pkts / (dst_pkts + 1.0),
    })
    out["label"] = df["label"].astype(int)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def load_cicids():
    """CIC-IDS-2017: 8 per-day CSVs. Label = BENIGN -> 0 else 1."""
    wanted = {
        "Flow Duration", "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Total Fwd Packets", "Total Backward Packets", "Label",
    }
    frames = []
    for f in sorted(os.listdir(CICIDS_DIR)):
        if not f.lower().endswith(".csv"):
            continue
        path = os.path.join(CICIDS_DIR, f)
        df = pd.read_csv(path, usecols=lambda c: c.strip() in wanted, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        raw = pd.DataFrame({
            "duration": pd.to_numeric(df["Flow Duration"], errors="coerce") / 1e6,  # us -> s
            "src_bytes": df["Total Length of Fwd Packets"],
            "dst_bytes": df["Total Length of Bwd Packets"],
            "src_pkts": df["Total Fwd Packets"],
            "dst_pkts": df["Total Backward Packets"],
            "label": (df["Label"].astype(str).str.strip().str.upper() != "BENIGN").astype(int),
        })
        frames.append(raw)
    return _derive(pd.concat(frames, ignore_index=True))


def load_unsw():
    """UNSW-NB15: combine the provided train + test splits."""
    frames = []
    for path in (UNSW_TRAIN, UNSW_TEST):
        df = pd.read_csv(path, low_memory=False)
        raw = pd.DataFrame({
            "duration": df["dur"],
            "src_bytes": df["sbytes"],
            "dst_bytes": df["dbytes"],
            "src_pkts": df["spkts"],
            "dst_pkts": df["dpkts"],
            "label": df["label"].astype(int),
        })
        frames.append(raw)
    return _derive(pd.concat(frames, ignore_index=True))


def load_ton():
    """TON_IoT network train/test file."""
    df = pd.read_csv(TON_FILE, low_memory=False)
    raw = pd.DataFrame({
        "duration": df["duration"],
        "src_bytes": df["src_bytes"],
        "dst_bytes": df["dst_bytes"],
        "src_pkts": df["src_pkts"],
        "dst_pkts": df["dst_pkts"],
        "label": df["label"].astype(int),
    })
    return _derive(raw)


LOADERS = {"CIC-IDS-2017": load_cicids, "UNSW-NB15": load_unsw, "TON_IoT": load_ton}


# ----------------------------------------------------------------------------
# Sampling + caching
# ----------------------------------------------------------------------------
def stratified_sample(df, n, seed):
    if n is None or n <= 0 or len(df) <= n:
        return df.reset_index(drop=True)
    parts = []
    for _, g in df.groupby("label"):
        k = max(1, int(round(n * len(g) / len(df))))
        k = min(k, len(g))
        parts.append(g.sample(k, random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def get_dataset(name, sample, seed, force_reload):
    tag = "all" if (sample is None or sample <= 0) else str(sample)
    cache = os.path.join(OUT, f"harmonized_{name.replace('/', '_')}_n{tag}.csv")
    if os.path.exists(cache) and not force_reload:
        print(f"  [cache] {name}: loading {cache}")
        return pd.read_csv(cache)
    t0 = time.time()
    print(f"  [load ] {name}: reading + harmonising raw data ...")
    df = LOADERS[name]()
    df = stratified_sample(df, sample, seed)
    df.to_csv(cache, index=False)
    print(f"  [save ] {name}: {len(df):,} rows -> {cache}  ({time.time()-t0:.0f}s)")
    return df


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def get_model(name, seed):
    name = name.lower()
    if name in ("rf", "randomforest"):
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, max_depth=None, n_jobs=-1,
            class_weight="balanced", random_state=seed,
        )
    if name in ("et", "extratrees"):
        return ExtraTreesClassifier(
            n_estimators=200, min_samples_leaf=5, max_depth=None, n_jobs=-1,
            class_weight="balanced", random_state=seed,
        )
    if name in ("hgb", "hist"):
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
            early_stopping=False, random_state=seed,
        )
    if name in ("xgb", "xgboost"):
        if not HAVE_XGB:
            raise RuntimeError("xgboost is not installed. Run: pip install xgboost")
        return XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            eval_metric="logloss", n_jobs=-1, random_state=seed,
        )
    if name in ("lgbm", "lightgbm"):
        if not HAVE_LGBM:
            raise RuntimeError("lightgbm is not installed. Run: pip install lightgbm")
        return LGBMClassifier(
            n_estimators=400, num_leaves=31, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
            n_jobs=-1, random_state=seed,
        )
    if name in ("vote", "voting"):
        estimators = [
            ("rf", get_model("rf", seed)),
            ("et", get_model("et", seed)),
            ("hgb", get_model("hgb", seed)),
        ]
        if HAVE_XGB:
            estimators.append(("xgb", get_model("xgb", seed)))
        return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    raise ValueError(f"Unknown model '{name}'. Use one of: rf, et, hgb, xgb, lgbm, vote.")


def fit_model(model, name, X, y):
    """Fit, applying class-balance tweaks where the estimator needs them."""
    if name.lower() in ("xgb", "xgboost"):
        npos = int((y == 1).sum())
        nneg = int((y == 0).sum())
        if npos > 0:
            model.set_params(scale_pos_weight=nneg / npos)
    return model.fit(X, y)


# ----------------------------------------------------------------------------
# Main evaluation
# ----------------------------------------------------------------------------
def evaluate(models, sample, seed, force_reload):
    names = list(LOADERS.keys())
    tag = "ALL rows" if (sample is None or sample <= 0) else f"<= {sample:,} rows"
    print(f"Preparing datasets (shared semantic subspace; {tag}) ...")
    data = {n: get_dataset(n, sample, seed, force_reload) for n in names}
    for n in names:
        d = data[n]
        print(f"    {n:14s} rows={len(d):>9,}  attack-rate={d['label'].mean():.3f}  "
              f"features={len(FEATURES)}")

    # pre-extract float32 arrays once
    X = {n: data[n][FEATURES].to_numpy(dtype=np.float32) for n in names}
    y = {n: data[n]["label"].to_numpy() for n in names}

    all_rows = []
    for mname in models:
        print(f"\n===== Model: {mname} =====")
        results = []
        for src in names:
            t0 = time.time()
            model = get_model(mname, seed)
            print(f"  training on {src} ...", end=" ", flush=True)
            fit_model(model, mname, X[src], y[src])
            print(f"({time.time()-t0:.0f}s)")
            for tgt in names:
                pred = model.predict(X[tgt])
                results.append({
                    "model": mname,
                    "source": src,
                    "target": tgt,
                    "accuracy": accuracy_score(y[tgt], pred),
                    "balanced_accuracy": balanced_accuracy_score(y[tgt], pred),
                    "macro_f1": f1_score(y[tgt], pred, average="macro", zero_division=0),
                    "n_train": len(y[src]),
                    "n_test": len(y[tgt]),
                })
        rdf = pd.DataFrame(results)
        all_rows.append(rdf)

        for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
            mat = rdf.pivot(index="source", columns="target", values=metric)
            mat = mat.reindex(index=names, columns=names)
            out_csv = os.path.join(OUT, f"baseline_{mname}_{metric}.csv")
            mat.to_csv(out_csv)
            print(f"\n  {metric.upper()}  (rows = trained on, cols = tested on)")
            print(mat.round(4).to_string())

    long = pd.concat(all_rows, ignore_index=True)
    long.to_csv(os.path.join(OUT, "baseline_long.csv"), index=False)
    print(f"\nSaved tidy results -> {os.path.join(OUT, 'baseline_long.csv')}")
    return long


def main():
    ap = argparse.ArgumentParser(description="Baseline cross-dataset IDS evaluation")
    ap.add_argument("--models", default="rf",
                    help="comma-separated from: rf, et, hgb, xgb, lgbm, vote (default: rf)")
    ap.add_argument("--sample", type=int, default=0,
                    help="max rows per dataset, stratified. 0 = use ALL rows (default).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-reload", action="store_true",
                    help="ignore cached harmonised CSVs and re-read raw data")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sample = None if args.sample <= 0 else args.sample
    evaluate(models, sample, args.seed, args.force_reload)


if __name__ == "__main__":
    main()
