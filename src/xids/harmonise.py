"""Feature harmonisation, version 2.

Maps every dataset onto one schema of *base* columns, from which the shared
feature pool is derived (see features.py).

What changed against the mid-semester harmonisation (src/baseline_cross_dataset.py)
------------------------------------------------------------------------------------
1. Byte counts are now "wire" bytes (IP-level, headers included) in every dataset.
   The old mapping mixed payload bytes (CIC-IDS-2017, TON_IoT) with header-inclusive
   bytes (UNSW-NB15), so a zero-payload TCP packet counted as 0 bytes in two datasets
   and about 40 bytes in the third.
       CIC-IDS-2017 / CSE-CIC-IDS2018 : payload length + reported header length
       UNSW-NB15                      : sbytes / dbytes   (Argus, header-inclusive)
       TON_IoT                        : src_ip_bytes / dst_ip_bytes (Zeek, IP-level)
   The payload-only counts are kept where they exist (src_payload / dst_payload)
   so the old definition can be reproduced for the ablation.
2. Protocol (TCP / UDP) and a coarse service class are added.
3. The original attack name and a coarse attack family are kept for every row, so
   results can be broken down per attack type.
4. Connection state is deliberately NOT harmonised: the TCP-flag counters of
   CIC-IDS-2017 are unreliable (SYN count is 0 for about 94% of TCP flows).
"""
import os
import numpy as np
import pandas as pd

from .config import DATA, CACHE

BASE_NUM = ["duration", "src_pkts", "dst_pkts", "src_bytes", "dst_bytes",
            "src_payload", "dst_payload", "is_tcp", "is_udp"]
SERVICES = ["dns", "http", "ssl", "ftp", "ssh", "mail"]      # everything else -> "other"

PORT2SVC = {53: "dns", 5353: "dns", 80: "http", 8080: "http", 8000: "http", 443: "ssl", 8443: "ssl",
            20: "ftp", 21: "ftp", 22: "ssh", 25: "mail", 110: "mail", 143: "mail", 465: "mail",
            587: "mail", 993: "mail", 995: "mail"}


def _svc_from_port(port):
    return pd.to_numeric(port, errors="coerce").map(PORT2SVC).fillna("other")


# --------------------------------------------------------------------------- families
def _family_cic(label):
    s = label.str.strip().str.lower()
    out = pd.Series("other", index=s.index)
    out[s == "benign"] = "benign"
    out[s.str.contains("dos") | s.str.contains("ddos")] = "dos"
    out[s.str.contains("portscan")] = "recon"
    out[s.str.contains("patator") | s.str.contains("bruteforce") | (s.str.contains("brute") & ~s.str.contains("web"))] = "bruteforce"
    out[s.str.contains("bot")] = "botnet"
    out[s.str.contains("web") | s.str.contains("xss") | s.str.contains("sql")] = "web"
    out[s.str.contains("infilt")] = "infiltration"
    return out


_UNSW_FAM = {"Normal": "benign", "DoS": "dos", "Reconnaissance": "recon", "Analysis": "recon",
             "Backdoor": "backdoor", "Exploits": "exploit", "Fuzzers": "fuzzers", "Generic": "generic",
             "Shellcode": "shellcode", "Worms": "worm"}
_TON_FAM = {"normal": "benign", "dos": "dos", "ddos": "dos", "scanning": "recon", "password": "bruteforce",
            "backdoor": "backdoor", "injection": "web", "xss": "web", "ransomware": "ransomware", "mitm": "mitm"}


# --------------------------------------------------------------------------- loaders
def _cic_wire(payload, hdr, pkts):
    """payload bytes + reported header bytes; repair the few overflowed header values."""
    hdr = pd.to_numeric(hdr, errors="coerce").astype("float64")
    pkts = pkts.astype("float64")
    bad = hdr.isna() | (hdr < 0) | (hdr > 60 * pkts)
    hdr = hdr.where(~bad, 32.0 * pkts)
    return payload.astype("float64") + hdr


def load_cic2017():
    d = os.path.join(DATA, "CIC-IDS-2017", "MachineLearningCVE")
    want = {"Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
            "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Header Length",
            "Bwd Header Length", "Init_Win_bytes_forward", "Label"}
    frames = []
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(d, f), usecols=lambda c: c.strip() in want,
                         low_memory=False, encoding="latin-1")
        df.columns = [c.strip() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        tcp = (pd.to_numeric(df["Init_Win_bytes_forward"], errors="coerce") >= 0)
        label = df["Label"].astype(str).str.replace("�", "-").str.replace("\x96", "-").str.strip()
        frames.append(pd.DataFrame({
            "duration": pd.to_numeric(df["Flow Duration"], errors="coerce").clip(lower=0) / 1e6,
            "src_pkts": df["Total Fwd Packets"], "dst_pkts": df["Total Backward Packets"],
            "src_bytes": _cic_wire(df["Total Length of Fwd Packets"], df["Fwd Header Length"], df["Total Fwd Packets"]),
            "dst_bytes": _cic_wire(df["Total Length of Bwd Packets"], df["Bwd Header Length"], df["Total Backward Packets"]),
            "src_payload": df["Total Length of Fwd Packets"], "dst_payload": df["Total Length of Bwd Packets"],
            "is_tcp": tcp.astype(int), "is_udp": (~tcp).astype(int),   # CICFlowMeter only emits TCP and UDP flows
            "svc": _svc_from_port(df["Destination Port"]),
            "attack_cat": label, "family": _family_cic(label),
            "label": (label.str.upper() != "BENIGN").astype(int),
            "grp": f.split("-")[0],                                     # capture day
        }))
    return pd.concat(frames, ignore_index=True)


def load_unsw():
    frames = []
    for split, f in (("train", "UNSW_NB15_training-set.csv"), ("test", "UNSW_NB15_testing-set.csv")):
        df = pd.read_csv(os.path.join(DATA, "UNSW-NB15", f), low_memory=False, encoding="utf-8-sig")
        svc = df["service"].astype(str).replace({"ftp-data": "ftp", "smtp": "mail", "pop3": "mail"})
        svc = svc.where(svc.isin(SERVICES), "other")
        cat = df["attack_cat"].astype(str).str.strip()
        frames.append(pd.DataFrame({
            "duration": df["dur"], "src_pkts": df["spkts"], "dst_pkts": df["dpkts"],
            "src_bytes": df["sbytes"], "dst_bytes": df["dbytes"],
            "src_payload": np.nan, "dst_payload": np.nan,                # not available in UNSW-NB15
            "is_tcp": (df["proto"] == "tcp").astype(int), "is_udp": (df["proto"] == "udp").astype(int),
            "svc": svc, "attack_cat": cat, "family": cat.map(_UNSW_FAM).fillna("other"),
            "label": df["label"].astype(int), "grp": split,
        }))
    return pd.concat(frames, ignore_index=True)


def load_ton():
    p = os.path.join(DATA, "TON_IOT network train-test", "Train_Test_Network_dataset", "train_test_network.csv")
    df = pd.read_csv(p, low_memory=False, encoding="utf-8-sig")
    zeek = df["service"].astype(str)
    svc = _svc_from_port(df["dst_port"])
    for name in ("dns", "http", "ssl", "ftp"):                           # Zeek's own detection wins when present
        svc = svc.where(~zeek.str.contains(name), name)
    cat = df["type"].astype(str).str.strip()
    return pd.DataFrame({
        "duration": df["duration"], "src_pkts": df["src_pkts"], "dst_pkts": df["dst_pkts"],
        "src_bytes": df["src_ip_bytes"], "dst_bytes": df["dst_ip_bytes"],
        "src_payload": df["src_bytes"], "dst_payload": df["dst_bytes"],
        "is_tcp": (df["proto"] == "tcp").astype(int), "is_udp": (df["proto"] == "udp").astype(int),
        "svc": svc, "attack_cat": cat, "family": cat.map(_TON_FAM).fillna("other"),
        "label": df["label"].astype(int), "grp": "all",
    })


def load_cic2018():
    """External validation set. Same extractor family as CIC-IDS-2017, different network (AWS), year and attacks."""
    d = os.path.join(DATA, "CSE-CIC-IDS2018")
    want = ["Dst Port", "Protocol", "Timestamp", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
            "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Header Len", "Bwd Header Len", "Label"]
    frames = []
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(d, f), usecols=want, low_memory=False)
        df = df[df["Label"] != "Label"]                                  # some files repeat the header row
        num = {c: pd.to_numeric(df[c], errors="coerce") for c in want if c not in ("Timestamp", "Label")}
        label = df["Label"].astype(str).str.strip()
        proto = num["Protocol"]
        frames.append(pd.DataFrame({
            "duration": num["Flow Duration"].clip(lower=0) / 1e6,
            "src_pkts": num["Tot Fwd Pkts"], "dst_pkts": num["Tot Bwd Pkts"],
            "src_bytes": _cic_wire(num["TotLen Fwd Pkts"].fillna(0), num["Fwd Header Len"], num["Tot Fwd Pkts"].fillna(0)),
            "dst_bytes": _cic_wire(num["TotLen Bwd Pkts"].fillna(0), num["Bwd Header Len"], num["Tot Bwd Pkts"].fillna(0)),
            "src_payload": num["TotLen Fwd Pkts"], "dst_payload": num["TotLen Bwd Pkts"],
            "is_tcp": (proto == 6).astype(int), "is_udp": (proto == 17).astype(int),
            "svc": _svc_from_port(num["Dst Port"]),
            "attack_cat": label, "family": _family_cic(label),
            "label": (label.str.upper() != "BENIGN").astype(int),
            "grp": f.split("_")[0],                                      # capture day, e.g. Friday-02-03-2018
        }))
    return pd.concat(frames, ignore_index=True)


LOADERS = {"CIC-IDS-2017": load_cic2017, "UNSW-NB15": load_unsw, "TON_IoT": load_ton,
           "CSE-CIC-IDS2018": load_cic2018}


def _clean(df):
    for c in BASE_NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    core = ["duration", "src_pkts", "dst_pkts", "src_bytes", "dst_bytes"]
    df[core] = df[core].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=core).reset_index(drop=True)
    df[core] = df[core].clip(lower=0)
    for c in ("svc", "attack_cat", "family", "grp"):
        df[c] = df[c].astype("category")
    return df


def get_base(name, force=False):
    """Harmonised base table for one dataset (cached as parquet)."""
    path = os.path.join(CACHE, f"base_{name}.parquet")
    if os.path.exists(path) and not force:
        return pd.read_parquet(path)
    df = _clean(LOADERS[name]())
    df.to_parquet(path, index=False)
    return df
