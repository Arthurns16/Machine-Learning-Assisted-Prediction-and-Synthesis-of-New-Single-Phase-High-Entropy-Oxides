#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Config
# -------------------------
INPUT_XLSX = Path("Input_Total.xlsx")   # ajuste se necessário
EXCLUDE = {"O", "F"}

OUT_PNG_600DPI = Path("pareto_atoms_600dpi.png")  # saída


# -------------------------
# Helpers
# -------------------------
def _norm(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.strip().lower()

def parse_atoms_cell(x):
    """Parse cells like "['Ni','Fe','O']" into list[str]."""
    if pd.isna(x):
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    s = str(x).strip()
    if not s:
        return None
    try:
        return list(ast.literal_eval(s))
    except Exception:
        s2 = s.strip("[]").replace("'", "").replace('"', "")
        parts = [p.strip() for p in s2.split(",") if p.strip()]
        return parts if parts else None


# -------------------------
# Load + find "Átomos" column
# -------------------------
df = pd.read_excel(INPUT_XLSX)

col_map = {c: _norm(c) for c in df.columns}
atom_col = None

for c, n in col_map.items():
    if n == "atomos" or n.replace("_", " ") == "atomos":
        atom_col = c
        break

if atom_col is None:
    for c, n in col_map.items():
        if "atom" in n:
            atom_col = c
            break

if atom_col is None:
    raise ValueError(f"Could not find atoms column. Columns = {list(df.columns)}")


# -------------------------
# Count atoms (excluding O and F)
# -------------------------
counts = {}
for atoms in df[atom_col].apply(parse_atoms_cell).dropna():
    for a in atoms:
        a = str(a).strip()
        if (not a) or (a in EXCLUDE):
            continue
        counts[a] = counts.get(a, 0) + 1

if not counts:
    raise ValueError("No atoms were counted. Check the 'Átomos' column formatting.")

count_df = (
    pd.DataFrame({"Element": list(counts.keys()), "Count": list(counts.values())})
    .sort_values("Count", ascending=False)
    .reset_index(drop=True)
)
count_df["Cumulative"] = count_df["Count"].cumsum()
total = count_df["Count"].sum()
count_df["CumulativePct"] = 100.0 * count_df["Cumulative"] / total


# -------------------------
# Plot style (creative purple)
# -------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 13,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.linewidth": 2.2,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "grid.linewidth": 1.0,
})

PURPLE_BAR = "#6A1B9A"
PURPLE_LINE = "#B23AEE"
PURPLE_ACCENT = "#2E0D41"


# -------------------------
# Pareto plot
# -------------------------
fig, ax = plt.subplots(figsize=(9.5, 6.5))
x = np.arange(len(count_df))

ax.bar(
    x,
    count_df["Count"].values,
    color=PURPLE_BAR,
    edgecolor=PURPLE_ACCENT,
    linewidth=1.6,
    alpha=0.95,
)

ax.set_title("Pareto Chart of Atom Frequency")
ax.set_xlabel("Element")
ax.set_ylabel("Count")

ax.set_xticks(x)
ax.set_xticklabels(count_df["Element"].tolist(), rotation=45, ha="right")

# X labels reduced by 10% three times => 13 * 0.9^3
x_label_size = 13.0 * (0.9 ** 3)  # 9.477 pt
ax.tick_params(axis="x", labelsize=x_label_size)

ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

ax2 = ax.twinx()
ax2.plot(
    x,
    count_df["CumulativePct"].values,
    color=PURPLE_LINE,
    linewidth=9,
    marker="o",
    markersize=6,
)
ax2.set_ylabel("Cumulative (%)")
ax2.set_ylim(0, 105)

# 80% reference line
ax2.axhline(80, color=PURPLE_ACCENT, linewidth=2.5, alpha=0.7, linestyle="--")
ax2.text(
    len(x) - 1, 81.5, "80%",
    ha="right", va="bottom",
    color=PURPLE_ACCENT,
    fontsize=12, fontweight="bold"
)

fig.tight_layout()

# Save ONLY PNG at 600 DPI
fig.savefig(OUT_PNG_600DPI, bbox_inches="tight", dpi=600)
plt.show()

print(f"Saved: {OUT_PNG_600DPI.resolve()}")
