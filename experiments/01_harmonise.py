"""Build the harmonised (v2) base tables for every dataset and print a sanity report.

    python experiments/01_harmonise.py [--force] [--external]
"""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, pandas as pd
from xids.config import DATASETS, EXTERNAL, OUT
from xids.harmonise import get_base
from xids.features import derive, POOL_V2

ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true"); ap.add_argument("--external", action="store_true")
a = ap.parse_args()
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
rows = []
for name in DATASETS + (EXTERNAL if a.external else []):
    b = get_base(name, force=a.force)
    X = derive(b)
    ben = b["label"] == 0
    rows.append(dict(dataset=name, rows=len(b), attack_frac=round(b["label"].mean(), 4), n_attack_types=b.loc[~ben, "attack_cat"].nunique(),
                     tcp=round(b["is_tcp"].mean(), 3), udp=round(b["is_udp"].mean(), 3), no_reply=round((b["dst_pkts"] == 0).mean(), 3),
                     med_src_bpp=round(X["src_bytes_per_pkt"].median(), 1), dur0=round((b["duration"] == 0).mean(), 3)))
    print(f"\n=== {name}: {len(b):,} rows, {X.shape[1]} features, non-finite cells = {int((~np.isfinite(X.to_numpy())).sum())}")
    print("families:", b["family"].value_counts().to_dict())
    print("service :", b["svc"].value_counts().to_dict())
    print("groups  :", b["grp"].value_counts().to_dict())
S = pd.DataFrame(rows); print("\n", S.to_string(index=False))
S.to_csv(os.path.join(OUT, "harmonised_summary.csv"), index=False)
