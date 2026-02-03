#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2x2 ROC (OvR) curves for 4 models, purple style, AUC-focused.
Assumes you run this script in a directory that contains:

- roc_ovr_auc_table.csv   (optional, used as fallback to infer class order/names)
- folds/outer_fold_01/, folds/outer_fold_02/, ... each containing:
    - y_test.npy
    - proba_GradientBoosting.npy
    - proba_RandomForest.npy
    - proba_MLP.npy
    - proba_Ensemble.npy

Outputs:
- roc_ovr_4models_purple_600dpi.png
- roc_ovr_4models_purple.pdf
- roc_ovr_auc_table_recomputed.csv
"""

from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


# -----------------------------
# USER SETTINGS
# -----------------------------
ARTIFACTS_DIR = Path(".")  # <-- folds/ is in the execution directory

MODELS = [
    ("Gradient Boosting", "GradientBoosting"),
    ("Random Forest", "RandomForest"),
    ("Multi-Layer Perceptron", "MLP"),
    ("Ensemble", "Ensemble"),
]

AUC_TABLE_NAME = "roc_ovr_auc_table.csv"

# PT -> EN phase names
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


# -----------------------------
# PURPLE STYLE
# -----------------------------
rcParams["savefig.dpi"] = 1200
rcParams["figure.dpi"]  = 1200

rcParams["text.usetex"]      = False
rcParams["font.family"]      = "serif"
rcParams["mathtext.fontset"] = "cm"
rcParams["text.color"]       = "black"

rcParams["font.size"]        = 20
rcParams["axes.labelsize"]   = 22
rcParams["axes.titlesize"]   = 24
rcParams["xtick.labelsize"]  = 18
rcParams["ytick.labelsize"]  = 18
rcParams["legend.fontsize"]  = 14  # per-panel legend

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

# 33% thinner than 9 => 6.0
rcParams["lines.linewidth"]   = 6.0


# -----------------------------
# Helpers
# -----------------------------
def tr(x: str) -> str:
    x = str(x).strip()
    return pt_to_en.get(x, x)

def sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", s)

def find_fold_dirs(root: Path) -> list[Path]:
    """
    Expects: root/folds/outer_fold_01, outer_fold_02, ...
    Also supports: root/fold_0, fold_1, ...
    """
    # Preferred layout
    outer = sorted([p for p in (root / "folds").glob("outer_fold_*") if p.is_dir()])
    if outer:
        return outer

    # Alternate layout
    direct = sorted([p for p in root.glob("fold_*") if p.is_dir()])
    if direct:
        return direct

    raise FileNotFoundError(
        f"Could not find fold directories. Expected '{root}/folds/outer_fold_*' "
        f"or '{root}/fold_*'. Root resolved: {root.resolve()}"
    )

def rebuild_oof_probas(root: Path, model_keys: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    fold_dirs = find_fold_dirs(root)
    y_all = []
    P_all = {m: [] for m in model_keys}

    for fp in fold_dirs:
        y_path = fp / "y_test.npy"
        if not y_path.exists():
            raise FileNotFoundError(f"Missing {y_path}")
        y = np.load(y_path).astype(int)
        y_all.append(y)

        for m in model_keys:
            p_path = fp / f"proba_{sanitize_name(m)}.npy"
            if not p_path.exists():
                raise FileNotFoundError(f"Missing {p_path}")
            P = np.load(p_path).astype(float)
            if P.shape[0] != y.shape[0]:
                raise ValueError(f"{fp.name} {m}: proba rows {P.shape[0]} != y rows {y.shape[0]}")
            P_all[m].append(P)

    y_all = np.concatenate(y_all)
    for m in model_keys:
        P_all[m] = np.concatenate(P_all[m])

    return y_all, P_all

def try_load_label_encoder_classes(root: Path) -> list[str] | None:
    """
    Best source of class names/order: label_encoder_classes.npy
    Tries a few common locations relative to execution dir.
    """
    candidates = [
        root / "label_encoder_classes.npy",
        root / "final_bundle" / "label_encoder_classes.npy",
        root / "bundle" / "label_encoder_classes.npy",
        root / "outputs" / "label_encoder_classes.npy",
        root / ".." / "final_bundle" / "label_encoder_classes.npy",
        root / ".." / "label_encoder_classes.npy",
    ]
    for p in candidates:
        if p.exists():
            classes = np.load(p, allow_pickle=True).tolist()
            return [tr(c) for c in classes]
    return None

def load_class_order_from_auc_table(path: Path) -> list[str] | None:
    """
    Fallback: use roc_ovr_auc_table.csv to infer class names/order.
    """
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if not {"class_name", "model", "auc_ovr"}.issubset(df.columns):
        return None

    if (df["model"] == "Ensemble").any():
        order = df.loc[df["model"] == "Ensemble", "class_name"].tolist()
    else:
        order = df["class_name"].tolist()

    seen, out = set(), []
    for c in order:
        c_en = tr(c)
        if c_en not in seen:
            out.append(c_en)
            seen.add(c_en)
    return out if out else None

def compute_auc_rows(y_true: np.ndarray, P: np.ndarray, class_names: list[str], model_key: str) -> list[dict]:
    n_classes = len(class_names)
    Y = label_binarize(y_true, classes=np.arange(n_classes))

    rows = []
    for k, cls in enumerate(class_names):
        if Y[:, k].sum() == 0 or Y[:, k].sum() == Y.shape[0]:
            rows.append({"model": model_key, "class_name": cls, "auc_ovr": np.nan})
            continue
        fpr, tpr, _ = roc_curve(Y[:, k], P[:, k])
        rows.append({"model": model_key, "class_name": cls, "auc_ovr": float(auc(fpr, tpr))})
    return rows

def plot_roc_panel(ax, y_true: np.ndarray, P: np.ndarray, class_names: list[str], title: str):
    n_classes = len(class_names)
    Y = label_binarize(y_true, classes=np.arange(n_classes))

    for k, cls in enumerate(class_names):
        if Y[:, k].sum() == 0 or Y[:, k].sum() == Y.shape[0]:
            continue
        fpr, tpr, _ = roc_curve(Y[:, k], P[:, k])
        ak = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{cls} (AUC={ak:.2f})")

    # baseline (no legend)
    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=3)

    ax.set_title(title, pad=12)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True)

    # breathing room (avoid 0/1 glued to borders)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)

    leg = ax.legend(loc="lower right", fontsize=14, frameon=True)
    leg.get_frame().set_edgecolor("#4a148c")
    leg.get_frame().set_linewidth(1.0)


def main():
    # 1) Load OOF probas + labels from folds
    model_keys = [k for _, k in MODELS]
    y_true, P_all = rebuild_oof_probas(ARTIFACTS_DIR, model_keys)

    # 2) Determine class names/order (best: label_encoder_classes.npy)
    class_names = try_load_label_encoder_classes(ARTIFACTS_DIR)
    if class_names is None:
        class_names = load_class_order_from_auc_table(Path(AUC_TABLE_NAME))
        if class_names is None:
            n_classes = P_all[model_keys[0]].shape[1]
            class_names = [f"Class {i}" for i in range(n_classes)]
            print("[WARN] Could not infer class names from label_encoder_classes.npy or roc_ovr_auc_table.csv.")

    n_classes = len(class_names)

    # 3) Validate shapes
    for mk in model_keys:
        if P_all[mk].shape[1] != n_classes:
            raise ValueError(
                f"{mk}: proba has {P_all[mk].shape[1]} columns but class_names has {n_classes}. "
                "Make sure label_encoder_classes.npy corresponds to these folds, "
                "or remove it so the script falls back to roc_ovr_auc_table.csv."
            )

    # 4) Recompute and save per-class AUC table (OOF)
    auc_rows = []
    for _, key in MODELS:
        auc_rows.extend(compute_auc_rows(y_true, P_all[key], class_names, key))

    pd.DataFrame(auc_rows).to_csv("roc_ovr_auc_table_recomputed.csv", index=False)
    print("[OK] Saved: roc_ovr_auc_table_recomputed.csv")

    # 5) Plot 2x2 ROC curves (AUC-focused)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=False)
    axes = axes.flatten()

    for i, (pretty, key) in enumerate(MODELS):
        plot_roc_panel(axes[i], y_true, P_all[key], class_names, pretty)

    fig.subplots_adjust(top=0.92, wspace=0.25, hspace=0.30)

    out_png = Path("roc_ovr_4models_purple_600dpi.png")
    out_pdf = Path("roc_ovr_4models_purple.pdf")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {out_png}")
    print(f"[OK] Saved: {out_pdf}")
    print(f"[INFO] Classes (EN): {class_names}")
    print(f"[INFO] ARTIFACTS_DIR resolved to: {ARTIFACTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
