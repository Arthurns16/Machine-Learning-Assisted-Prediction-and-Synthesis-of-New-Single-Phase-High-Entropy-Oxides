# Pearson |correlation| matrix (absolute values), per-cell annotations, 600 DPI
# English feature names: only last token translated:
# soma->sum, maximo->max, minimo->min, desvio->std, media/média->avg
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
from matplotlib.font_manager import FontProperties

# -----------------------------
# Inputs
# -----------------------------
xlsx_path = Path("Input_Total.xlsx")  # adjust if needed
sheet_name = 0

features_pt = [
    "atomic_ea_minimo",
    "atomic_ea_maximo",
    "atomic_ea_soma",
    "atomic_en_allen _soma",
    "atomic_en_allen _desvio",
    "atomic_en_allredroch_minimo",
    "atomic_hatm_minimo",
    "atomic_spacegroupnum_maximo",
    "atomic_spacegroupnum_desvio",
    "electrical_resist_maximo",
    "mineral_hardness_maximo",
    "van_der_waals_rad_minimo",
    "vel_of_sound_minimo",
]

out_png = Path("pearson_abs_corr_matrix_annot_600dpi.png")
out_csv = Path("pearson_abs_corr_matrix.csv")

# -----------------------------
# Helpers
# -----------------------------
def normalize_colname(s: str) -> str:
    return " ".join(str(s).split())

def strip_accents(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )

_last_token_map = {
    "soma": "sum",
    "maximo": "max",
    "minimo": "min",
    "desvio": "std",
    "media": "avg",  # covers "média" after strip_accents
}

def last_token_to_english(feat: str) -> str:
    feat = normalize_colname(feat)
    parts = feat.split("_")
    if not parts:
        return feat
    last_raw = parts[-1]
    last_key = strip_accents(last_raw.lower())
    parts[-1] = _last_token_map.get(last_key, last_raw)
    return "_".join(parts)

def get_title_size_points(fig) -> float:
    """
    Return the default title fontsize in *points* as a float, even if rcParams uses strings like 'large'.
    """
    rc = plt.rcParams.get("axes.titlesize", "medium")
    # If numeric already
    if isinstance(rc, (int, float)):
        return float(rc)
    # If string, let Matplotlib resolve it
    fp = FontProperties(size=rc)
    return fp.get_size_in_points()

# -----------------------------
# Load data
# -----------------------------
df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
df = df.rename(columns={c: normalize_colname(c) for c in df.columns})

features_pt_norm = [normalize_colname(c) for c in features_pt]

missing = [c for c in features_pt_norm if c not in df.columns]
if missing:
    raise KeyError(
        "Missing requested columns (after normalizing spaces):\n"
        + "\n".join(missing)
        + "\n\nAvailable columns (first 80):\n"
        + "\n".join(map(str, df.columns[:80]))
    )

X = df[features_pt_norm].apply(pd.to_numeric, errors="coerce")

# -----------------------------
# Pearson correlation (absolute)
# -----------------------------
corr_abs = X.corr(method="pearson").abs()

labels_en = [last_token_to_english(c) for c in features_pt_norm]
corr_abs.index = labels_en
corr_abs.columns = labels_en

corr_abs.to_csv(out_csv, index=True)

# -----------------------------
# Plot (annotated)
# -----------------------------
n = len(labels_en)
fig_w = max(9, 0.62 * n)
fig_h = max(8, 0.62 * n)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
im = ax.imshow(corr_abs.values, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")

# title fontsize (+50%) robustly
base_title_fs = get_title_size_points(fig)
title_fs = base_title_fs * 1.5
ax.set_title("Pearson absolute correlation matrix", pad=12, fontsize=title_fs)

ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(labels_en, rotation=45, ha="right")
ax.set_yticklabels(labels_en)

ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
ax.grid(which="minor", linestyle="-", linewidth=0.5)
ax.tick_params(which="minor", bottom=False, left=False)

# annotation fontsize (+50%)
base_fontsize = 8 if n <= 14 else 7
fontsize = int(round(base_fontsize * 1.5))

for i in range(n):
    for j in range(n):
        val = corr_abs.values[i, j]
        txt_color = "white" if val >= 0.60 else "black"
        ax.text(
            j, i, f"{val:.2f}",
            ha="center", va="center",
            color=txt_color, fontsize=fontsize
        )

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("|Pearson r|")

plt.tight_layout()
plt.savefig(out_png, dpi=600, bbox_inches="tight")
plt.show()

print(f"Saved figure: {out_png.resolve()}")
print(f"Saved matrix (CSV): {out_csv.resolve()}")