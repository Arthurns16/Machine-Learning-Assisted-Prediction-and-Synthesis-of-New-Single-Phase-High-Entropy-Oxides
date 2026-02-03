import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

# --- minimal purple style, tuned for speed and 600 DPI output ---
rcParams["savefig.dpi"] = 600
rcParams["figure.dpi"]  = 600
rcParams["text.usetex"] = False
rcParams["font.family"] = "serif"
rcParams["mathtext.fontset"] = "cm"

rcParams["font.size"] = 22
rcParams["axes.labelsize"] = 24
rcParams["axes.titlesize"] = 25
rcParams["xtick.labelsize"] = 22
rcParams["ytick.labelsize"] = 22

rcParams["axes.edgecolor"] = "#4a148c"
rcParams["axes.linewidth"] = 0.5

rcParams["grid.color"] = "#6a1b9a"
rcParams["grid.linestyle"] = ":"
rcParams["grid.alpha"] = 0.25
rcParams["grid.linewidth"] = 2.5

pt_to_en = {
    "Fluorita": "Fluorite",
    "Perovskita": "Perovskite",
    "Pirocloro": "Pyrochlore",
    "RockSalt": "Rock Salt",
    "Rocksalt": "Rock Salt",
    "Rock Salt": "Rock Salt",
    "Spinel": "Spinel",
    "Mixed": "Mixed",
    "Mista": "Mixed",
}
def tr(x): return pt_to_en.get(str(x).strip(), str(x).strip())

# Load
csv_path = Path("confusion_norm_Ensemble.csv")
df = pd.read_csv(csv_path)
if df.columns[0].lower() in {"unnamed: 0", "index", ""}:
    df = pd.read_csv(csv_path, index_col=0)

df.index = [tr(i) for i in df.index]
df.columns = [tr(c) for c in df.columns]
df = df.apply(pd.to_numeric, errors="coerce")

cm = df.values
ylabels = df.index.tolist()
xlabels = df.columns.tolist()

# Plot
fig, ax = plt.subplots(figsize=(8.5, 7.5))
im = ax.imshow(cm, cmap="Purples", vmin=0.0, vmax=1.0, aspect="equal")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Proportion")

ax.set_title("Normalized Confusion Matrix (Ensemble)", pad=18.00)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")

ax.set_xticks(np.arange(len(xlabels)))
ax.set_yticks(np.arange(len(ylabels)))
ax.set_xticklabels(xlabels, rotation=45, ha="right")
ax.set_yticklabels(ylabels)

# Grid on cell boundaries
ax.set_xticks(np.arange(-0.5, cm.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, cm.shape[0], 1), minor=True)
ax.grid(which="minor",
        color=rcParams["grid.color"],
        linestyle=rcParams["grid.linestyle"],
        linewidth=rcParams["grid.linewidth"],
        alpha=rcParams["grid.alpha"])
ax.tick_params(which="minor", bottom=False, left=False)

# Values
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = cm[i, j]
        txt = "nan" if np.isnan(val) else f"{val:.2f}"
        ax.text(j, i, txt, ha="center", va="center",
                color="white" if (not np.isnan(val) and val > 0.5) else "black")

fig.tight_layout()

out_png = Path("confusion_matrix_Ensemble_EN_600dpi.png")
fig.savefig(out_png, dpi=600)
plt.close(fig)
