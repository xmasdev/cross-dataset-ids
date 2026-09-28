"""Render 3x3 source->target matrices (before/after) from da_mfs_results.csv."""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
DS = ["CIC-IDS-2017", "UNSW-NB15", "TON_IoT"]
SHORT = {"CIC-IDS-2017": "CIC", "UNSW-NB15": "UNSW", "TON_IoT": "TON"}
CONFIG_LABEL = {
    "raw_all": "Raw, all features (baseline / before)",
    "aligned_all": "Aligned, all features",
    "aligned_optimised": "DA-MFS (aligned + optimised subset / after)",
}
METRICS = [("accuracy", "Accuracy"),
           ("balanced_accuracy", "Balanced accuracy"),
           ("macro_f1", "Macro-F1"),
           ("attack_recall", "Attack recall")]


def main():
    df = pd.read_csv(os.path.join(OUT, "da_mfs_results.csv"))
    lines = ["# DA-MFS — 3x3 cross-dataset matrices (rows = trained on, cols = tested on)\n"]
    for metric, label in METRICS:
        lines.append(f"\n## {label}\n")
        for cfg in ["raw_all", "aligned_all", "aligned_optimised"]:
            sub = df[df.config == cfg]
            mat = sub.pivot(index="source", columns="target", values=metric)
            mat = mat.reindex(index=DS, columns=DS)
            lines.append(f"\n**{CONFIG_LABEL[cfg]}**\n")
            header = "| trained \\ tested | " + " | ".join(SHORT[d] for d in DS) + " |"
            lines.append(header)
            lines.append("|---|" + "---|" * len(DS))
            for src in DS:
                cells = []
                for tgt in DS:
                    v = mat.loc[src, tgt]
                    cell = f"**{v:.3f}**" if src == tgt else f"{v:.3f}"
                    cells.append(cell)
                lines.append(f"| {SHORT[src]} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "da_mfs_matrices.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print("Saved ->", os.path.join(OUT, "da_mfs_matrices.md"))


if __name__ == "__main__":
    main()
