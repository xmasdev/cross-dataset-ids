import os
import json
import numpy as np
import pandas as pd

BASE = r"F:\BTP\datasets"

def summarize_cicids():
    d = os.path.join(BASE, "CIC-IDS-2017", "MachineLearningCVE")
    files = sorted(os.listdir(d))
    out = {"files": [], "columns": None}
    total_rows = 0
    total_nan = 0
    total_inf = 0
    for f in files:
        p = os.path.join(d, f)
        hdr = pd.read_csv(p, nrows=0)
        cols = [c.strip() for c in hdr.columns]
        if out["columns"] is None:
            out["columns"] = cols
        df = pd.read_csv(p, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        label_col = "Label"
        dist = df[label_col].value_counts(dropna=False).to_dict()
        n = len(df)
        total_rows += n
        # numeric-only NaN/Inf
        num = df.select_dtypes(include=[np.number])
        nan_c = int(num.isna().sum().sum())
        inf_c = int(np.isinf(num.replace([np.inf, -np.inf], np.nan)).sum().sum()) if len(num.columns) else 0
        total_nan += nan_c
        total_inf += inf_c
        out["files"].append({
            "file": f, "rows": n, "label_dist": dist,
            "nan_cells": nan_c, "inf_cells": inf_c,
            "n_numeric_cols": int(len(num.columns)),
        })
    out["total_rows"] = total_rows
    out["total_nan_cells"] = total_nan
    out["total_inf_cells"] = total_inf
    return out

def summarize_unsw():
    d = os.path.join(BASE, "UNSW-NB15")
    out = {}
    for name, f in [("train", "UNSW_NB15_training-set.csv"), ("test", "UNSW_NB15_testing-set.csv")]:
        p = os.path.join(d, f)
        df = pd.read_csv(p, low_memory=False)
        out[name] = {
            "rows": len(df),
            "columns": list(df.columns),
            "attack_cat": df["attack_cat"].value_counts(dropna=False).to_dict(),
            "label": df["label"].value_counts(dropna=False).to_dict(),
            "nan_cells": int(df.isna().sum().sum()),
        }
    return out

def summarize_ton():
    p = os.path.join(BASE, "TON_IOT network train-test", "Train_Test_Network_dataset", "train_test_network.csv")
    df = pd.read_csv(p, low_memory=False)
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "type": df["type"].value_counts(dropna=False).to_dict(),
        "label": df["label"].value_counts(dropna=False).to_dict(),
        "nan_cells": int(df.isna().sum().sum()),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }

if __name__ == "__main__":
    result = {
        "cicids2017": summarize_cicids(),
        "unsw_nb15": summarize_unsw(),
        "ton_iot": summarize_ton(),
    }
    out_p = r"F:\BTP\src\dataset_stats.json"
    with open(out_p, "w") as fh:
        json.dump(result, fh, indent=2)
    print("saved", out_p)
    print("CICIDS total rows:", result["cicids2017"]["total_rows"])
    print("CICIDS nan cells:", result["cicids2017"]["total_nan_cells"], "inf cells:", result["cicids2017"]["total_inf_cells"])
    print("UNSW train rows:", result["unsw_nb15"]["train"]["rows"], "test rows:", result["unsw_nb15"]["test"]["rows"])
    print("TON rows:", result["ton_iot"]["rows"])
