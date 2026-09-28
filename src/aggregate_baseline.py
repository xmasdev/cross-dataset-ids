"""Aggregate the per-model baseline matrices into a tidy table and a summary
(same-dataset vs cross-dataset performance) for the report.

Run AFTER baseline_cross_dataset.py:
    python src/aggregate_baseline.py
"""
import os
import glob
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
FEAT_DS = ["CIC-IDS-2017", "UNSW-NB15", "TON_IoT"]
METRICS = ["accuracy", "balanced_accuracy", "macro_f1"]


def load_matrices():
    """Return {metric: {model: 3x3 DataFrame}}."""
    data = {m: {} for m in METRICS}
    suffixes = [("_balanced_accuracy.csv", "balanced_accuracy"),
                ("_macro_f1.csv", "macro_f1"),
                ("_accuracy.csv", "accuracy")]
    for path in glob.glob(os.path.join(OUT, "baseline_*.csv")):
        base = os.path.basename(path)
        for suffix, metric in suffixes:
            if base.endswith(suffix):
                model = base[len("baseline_"):-len(suffix)]
                df = pd.read_csv(path, index_col=0)
                data[metric][model] = df.reindex(index=FEAT_DS, columns=FEAT_DS)
                break
    return data


def main():
    data = load_matrices()
    models = sorted(data["accuracy"].keys())
    print("Models found:", models, "\n")

    # 1) tidy long table
    rows = []
    for model in models:
        for metric in METRICS:
            mat = data[metric][model]
            for src in FEAT_DS:
                for tgt in FEAT_DS:
                    rows.append({"model": model, "metric": metric,
                                 "source": src, "target": tgt,
                                 "value": float(mat.loc[src, tgt]),
                                 "same_dataset": src == tgt})
    long = pd.DataFrame(rows)
    long.to_csv(os.path.join(OUT, "baseline_long_all.csv"), index=False)

    # 2) summary: same-dataset (diag) vs cross-dataset (off-diag) means
    summary = []
    for model in models:
        row = {"model": model}
        for metric in METRICS:
            mat = data[metric][model]
            diag = [mat.loc[d, d] for d in FEAT_DS]
            off = [mat.loc[s, t] for s in FEAT_DS for t in FEAT_DS if s != t]
            row[f"same_{metric}"] = sum(diag) / len(diag)
            row[f"cross_{metric}"] = sum(off) / len(off)
        summary.append(row)
    sdf = pd.DataFrame(summary).set_index("model")
    sdf.to_csv(os.path.join(OUT, "baseline_summary.csv"))
    print("SUMMARY: same-dataset vs cross-dataset (mean over datasets/pairs)\n")
    print(sdf.round(4).to_string())

    # 3) markdown tables for direct paste into the report
    md = []
    md.append("### Same-dataset vs cross-dataset performance (full datasets)\n")
    md.append("| Model | Same-dataset Acc | Cross-dataset Acc | Same-dataset Bal-Acc | Cross-dataset Bal-Acc | Same-dataset F1 | Cross-dataset F1 |")
    md.append("|---|---|---|---|---|---|---|")
    for model in models:
        r = sdf.loc[model]
        md.append(f"| {model} | {r['same_accuracy']:.4f} | {r['cross_accuracy']:.4f} | "
                  f"{r['same_balanced_accuracy']:.4f} | {r['cross_balanced_accuracy']:.4f} | "
                  f"{r['same_macro_f1']:.4f} | {r['cross_macro_f1']:.4f} |")
    md.append("")
    for model in models:
        mat = data["accuracy"][model]
        md.append(f"\n**{model} — accuracy (rows = trained on, cols = tested on)**\n")
        md.append("| trained \\ tested | " + " | ".join(FEAT_DS) + " |")
        md.append("|---|" + "---|" * len(FEAT_DS))
        for src in FEAT_DS:
            md.append(f"| {src} | " + " | ".join(f"{mat.loc[src, t]:.4f}" for t in FEAT_DS) + " |")
    with open(os.path.join(OUT, "baseline_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print(f"\nSaved -> {os.path.join(OUT, 'baseline_long_all.csv')}")
    print(f"Saved -> {os.path.join(OUT, 'baseline_summary.csv')}")
    print(f"Saved -> {os.path.join(OUT, 'baseline_summary.md')}")


if __name__ == "__main__":
    main()
