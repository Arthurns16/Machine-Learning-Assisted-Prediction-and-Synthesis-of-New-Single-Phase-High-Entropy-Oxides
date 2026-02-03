#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make a 2x2 figure with normalized confusion matrices from CSV files
(Gradient Boosting, Random Forest, MLP, ELM), using the purple style
and translating phase labels (PT -> EN).

Expected: the CSV files are in the same directory where you run this script.
Outputs:
  - confusion_matrices_4models_purple_600dpi.png  (600 DPI)
  - confusion_matrices_4models_purple_600dpi.pdf  (vector)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# -------------------- Inputs --------------------
files = [
    ("Gradient Boosting", "confusion_norm_GradientBoosting.csv"),
    ("Random Forest", "confusion_norm_RandomForest.csv"),
    ("MLP", "confusion_norm_MLP.csv"),
    ("ELM", "confusion_norm_ELM.csv"),
]

pt_to_en = {
    "Fluorita": "Fluorite",
    "Perovskita": "Perovskite",
    "Pirocloro": "Pyrochlore",
    "RockSalt": "Rock-Salt",
    "Rocksalt": "Rock-Salt",
    "Rock Salt": "Rock-Salt",
    "Spinel": "Spinel",
    "Mixed": "Mixed",
    "Mista": "Mixed",
}

# -------------------- Purple style (your standard) --------------------
rcParams["savefig.dpi"] = 600
rcParams["figure.dpi"]  = 600

rcParams["text.usetex"]      = False
rcParams["font.family"]      = "serif"
rcParams["mathtext.fontset"] = "cm"
rcParams["text.color"]       = "black"

rcParams["font.size"]        = 22
rcParams["axes.labelsize"]   = 24
rcParams["axes.titlesize"]   = 24
rcParams["xtick.labelsize"]  = 18
rcParams["ytick.labelsize"]  = 18
rcParams["legend.fontsize"]  = 20

rcParams["axes.facecolor"]   = "#ffffff"
rcParams["figure.facecolor"] = "#ffffff"
rcParams["axes.edgecolor"]   = "#4a148c"
rcParams["axes.labelcolor"]  = "black"
rcParams["xtick.color"]      = "black"
rcParams["ytick.color"]      = "black"
rcParams["axes.linewidth"]   = 0.5

rcParams["axes.grid"]        = False
rcParams["grid.color"]       = "#6a1b9a"
rcParams["grid.linestyle"]   = ":"
rcParams["grid.alpha"]       = 0.25
rcParams["grid.linewidth"]   = 2.5

rcParams["legend.frameon"]    = True
rcParams["legend.fancybox"]   = False
rcParams["legend.edgecolor"]  = "#4a148c"
rcParams["legend.framealpha"] = 1.0

rcParams["lines.linewidth"]   = 9


# -------------------- Helpers --------------------
def translate_label(x: str) -> str:
    x = str(x).strip()
    return pt_to_en.get(x, x)

def read_confusion_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Handle common "Unnamed: 0" index column
    if df.columns[0].lower() in {"unnamed: 0", "index", ""}:
        df = pd.read_csv(csv_path, index_col=0)
    # Translate row/col labels
    df.index = [translate_label(i) for i in df.index]
    df.columns = [translate_label(c) for c in df.columns]
    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def draw_cm(ax, df: pd.DataFrame, title: str, panel_letter: str,
            cmap: str = "Purples", vmin: float = 0.0, vmax: float = 1.0) -> None:
    cm = df.values
    ylabels = df.index.tolist()
    xlabels = df.columns.tolist()

    im = ax.imshow(cm, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(f"{panel_letter}  {title}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)

    # Grid aligned to cell boundaries (purple style)
    ax.set_xticks(np.arange(-0.5, cm.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, cm.shape[0], 1), minor=True)
    ax.grid(
        which="minor",
        color=rcParams["grid.color"],
        linestyle=rcParams["grid.linestyle"],
        linewidth=rcParams["grid.linewidth"],
        alpha=rcParams["grid.alpha"],
    )
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotations
    # Use white text on high values for contrast
    thresh = 0.5 * vmax
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if np.isnan(val):
                txt = "nan"
                color = "black"
            else:
                txt = f"{val:.2f}"
                color = "white" if val > thresh else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=16, color=color)

    # Make spines purple (already via rcParams, but enforce)
    for spine in ax.spines.values():
        spine.set_edgecolor("#4a148c")
        spine.set_linewidth(0.8)

    return im


def main() -> None:
    base_dir = Path(".").resolve()

    dfs = []
    for model_name, fname in files:
        p = base_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        dfs.append((model_name, read_confusion_csv(p)))

    # Common color scale across all panels (recommended for comparison)
    # If your matrices are normalized, keep 0..1.
    vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)

    panel_letters = ["(a)", "(b)", "(c)", "(d)"]
    ims = []
    for ax, (panel, (model_name, df)) in zip(axes.flatten(), zip(panel_letters, dfs)):
        im = draw_cm(
            ax=ax,
            df=df,
            title=model_name,
            panel_letter=panel,
            cmap="Purples",   # purple palette
            vmin=vmin,
            vmax=vmax,
        )
        ims.append(im)

    # One shared colorbar for all subplots
    cbar = fig.colorbar(ims[0], ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("Proportion")

    # Save (600 DPI PNG + vector PDF)
    out_png = base_dir / "confusion_matrices_4models_purple_600dpi.png"
    out_pdf = base_dir / "confusion_matrices_4models_purple_600dpi.pdf"

    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {out_png}")
    print(f"[OK] Saved: {out_pdf}")


if __name__ == "__main__":
    main()