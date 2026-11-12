"""Paths and constants shared by every experiment."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "datasets")
CACHE = os.path.join(ROOT, "results", "cache")      # harmonised parquet files (git-ignored)
OUT = os.path.join(ROOT, "results", "v2")           # end-semester result tables
FIG = os.path.join(ROOT, "reports", "figures")
for _d in (CACHE, OUT, FIG):
    os.makedirs(_d, exist_ok=True)

# The three development datasets of the project, and the external dataset that is
# only ever used for the final confirmation run (never for any design decision).
DATASETS = ["CIC-IDS-2017", "UNSW-NB15", "TON_IoT"]
EXTERNAL = ["CSE-CIC-IDS2018"]
SHORT = {"CIC-IDS-2017": "CIC17", "UNSW-NB15": "UNSW", "TON_IoT": "TON", "CSE-CIC-IDS2018": "CIC18"}

SEED = 42
