"""The shared feature pool derived from the harmonised base columns.

Everything here is computable in all datasets from five flow counters (duration,
packets and bytes in each direction) plus protocol and a coarse service class.

Features are grouped into FAMILIES so results can be reported per family.
POOL_V1 reproduces the 12 mid-semester features (payload-style definitions on the
wire-byte counters) and is small enough to enumerate exhaustively (4,095 subsets);
POOL_V2 is the full pool the optimisers search (2^30 subsets).
"""
import numpy as np
import pandas as pd

from .harmonise import SERVICES

EPS_T = 1e-6        # floor for duration (seconds) when forming rates


def derive(base, payload=False):
    """Feature frame from a harmonised base table.

    payload=True uses the payload-only byte counts (mid-semester definition); only valid
    for datasets that have them (CIC, TON). Default is the header-inclusive wire bytes.
    """
    sb = base["src_payload"] if payload else base["src_bytes"]
    db = base["dst_payload"] if payload else base["dst_bytes"]
    sb = sb.to_numpy(np.float64); db = db.to_numpy(np.float64)
    dur = base["duration"].to_numpy(np.float64)
    sp = base["src_pkts"].to_numpy(np.float64); dp = base["dst_pkts"].to_numpy(np.float64)
    tb, tp = sb + db, sp + dp
    t = np.maximum(dur, EPS_T)
    f = {
        # volume
        "duration": dur, "src_pkts": sp, "dst_pkts": dp, "src_bytes": sb, "dst_bytes": db,
        "total_pkts": tp, "total_bytes": tb,
        # packet size
        "src_bytes_per_pkt": sb / np.maximum(sp, 1), "dst_bytes_per_pkt": db / np.maximum(dp, 1),
        "bytes_per_pkt": tb / np.maximum(tp, 1),
        "pkt_size_asym": sb / np.maximum(sp, 1) - db / np.maximum(dp, 1),
        # direction ratios
        "pkt_ratio": sp / (dp + 1.0), "byte_ratio": sb / (db + 1.0),
        # rates
        "pkts_per_s": tp / t, "bytes_per_s": tb / t,
        "src_pkts_per_s": sp / t, "dst_pkts_per_s": dp / t,
        "src_bytes_per_s": sb / t, "dst_bytes_per_s": db / t,
        "iat_mean": dur / np.maximum(tp - 1, 1),
        # connection shape
        "no_reply": (dp == 0).astype(np.float64),
        # protocol
        "is_tcp": base["is_tcp"].to_numpy(np.float64), "is_udp": base["is_udp"].to_numpy(np.float64),
    }
    svc = base["svc"].astype(str).to_numpy()
    for s in SERVICES:
        f[f"svc_{s}"] = (svc == s).astype(np.float64)
    f["svc_other"] = (~np.isin(svc, SERVICES)).astype(np.float64)
    out = pd.DataFrame(f)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


FAMILIES = {
    "volume": ["duration", "src_pkts", "dst_pkts", "src_bytes", "dst_bytes", "total_pkts", "total_bytes"],
    "size": ["src_bytes_per_pkt", "dst_bytes_per_pkt", "bytes_per_pkt", "pkt_size_asym"],
    "ratio": ["pkt_ratio", "byte_ratio"],
    "rate": ["pkts_per_s", "bytes_per_s", "src_pkts_per_s", "dst_pkts_per_s", "src_bytes_per_s",
             "dst_bytes_per_s", "iat_mean"],
    "shape": ["no_reply"],
    "proto": ["is_tcp", "is_udp"],
    "service": [f"svc_{s}" for s in SERVICES] + ["svc_other"],
}
POOL_V2 = [f for fam in FAMILIES.values() for f in fam]
FAMILY_OF = {f: fam for fam, fs in FAMILIES.items() for f in fs}

# the 12 mid-semester features under their new names (old name -> new name)
V1_RENAME = {"duration": "duration", "src_bytes": "src_bytes", "dst_bytes": "dst_bytes",
             "src_pkts": "src_pkts", "dst_pkts": "dst_pkts", "total_bytes": "total_bytes",
             "total_pkts": "total_pkts", "bytes_per_pkt_src": "src_bytes_per_pkt",
             "bytes_per_pkt_dst": "dst_bytes_per_pkt", "total_bytes_per_pkt": "bytes_per_pkt",
             "src_dst_byte_ratio": "byte_ratio", "src_dst_pkt_ratio": "pkt_ratio"}
POOL_V1 = list(V1_RENAME.values())
