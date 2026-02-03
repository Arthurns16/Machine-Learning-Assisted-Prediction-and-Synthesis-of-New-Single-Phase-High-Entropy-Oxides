#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings

# Silence common numerical warnings from ELM tanh implementation on extreme values
warnings.filterwarnings('ignore', message='overflow encountered in exp', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='invalid value encountered in divide', category=RuntimeWarning)
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---------------------------
# Label normalization helpers
# ---------------------------
def normalize_label_output(label: str) -> str:
    """Normalize labels for user-facing outputs (e.g., Excel)."""
    if label is None:
        return label
    s = str(label).strip()
    # Known mismatch: Portuguese vs English
    if s == "Fluorita":
        return "Fluorite"
    return s

def normalize_label_to_classes(label: str, classes: list[str]) -> str:
    """Normalize labels so they match the trained class names (bundle classes)."""
    if label is None:
        return label
    s = str(label).strip()
    cls = set(classes or [])
    # If bundle uses Portuguese label but data uses English, map English -> Portuguese
    if "Fluorita" in cls and s == "Fluorite":
        return "Fluorita"
    # If bundle uses English label but predictions/data use Portuguese, map Portuguese -> English
    if "Fluorite" in cls and s == "Fluorita":
        return "Fluorite"
    return s



# =====================================================================================
# IMPORTANT FOR ELM LOADING
# =====================================================================================
# Your saved ELM object was pickled as an instance of a class named "ELMWrapper" in
# the __main__ module (because it was created inside nestedcv_finalize.py).
# When unpickling on Windows, Python expects to find a symbol:
#   __main__.ELMWrapper
# in the *current script*.
#
# So we define a compatible top-level class with that name. The unpickled object
# already contains its internal state (attributes), so this class only needs to
# implement predict().


class ELMWrapper:
    """Compatibility class for unpickling elm_final.joblib."""

    def predict(self, X):
        X = np.asarray(X)

        # In the bundle, the object typically stores the trained ELM instance in self.model_
        model = getattr(self, "model_", None)
        if model is None:
            raise RuntimeError(
                "ELMWrapper loaded, but attribute 'model_' is missing. "
                "This means the saved object does not contain the trained ELM model state."
            )

        y_pred = model.predict(X)
        y_pred = np.asarray(y_pred)

        # Basic guard: avoid silent shape mismatches
        if y_pred.shape[0] != X.shape[0]:
            raise RuntimeError(
                f"ELM predict returned {y_pred.shape[0]} outputs for X with {X.shape[0]} rows. "
                "Your elm.py predict may be ignoring the input X and using internal data."
            )

        return y_pred.astype(int)


def _ensure_elm_module(bundle_dir: Path) -> None:
    """Ensure bundle_dir/elm.py is importable as module name 'elm' before joblib.load."""
    bundle_dir = Path(bundle_dir)
    if str(bundle_dir) not in sys.path:
        sys.path.insert(0, str(bundle_dir))

    elm_path = bundle_dir / "elm.py"
    if not elm_path.exists():
        # Not fatal if your ELM doesn't rely on this module name during unpickle,
        # but in most cases it is required.
        return

    spec = importlib.util.spec_from_file_location("elm", str(elm_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {elm_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["elm"] = mod
    spec.loader.exec_module(mod)  # type: ignore


def load_bundle(bundle_dir: Path, disable_calibration: bool) -> Dict[str, Any]:
    bundle_dir = Path(bundle_dir)

    _ensure_elm_module(bundle_dir)

    classes = np.load(bundle_dir / "label_encoder_classes.npy", allow_pickle=True).tolist()
    pre = json.loads((bundle_dir / "final_preprocessing.json").read_text(encoding="utf-8"))

    rf = joblib.load(bundle_dir / "rf_final.joblib")
    gb = joblib.load(bundle_dir / "gb_final.joblib")
    mlp = joblib.load(bundle_dir / "mlp_final.joblib")
    elm = joblib.load(bundle_dir / "elm_final.joblib")

    rf_cal_path = bundle_dir / "rf_calibrated.joblib"
    gb_cal_path = bundle_dir / "gb_calibrated.joblib"
    mlp_cal_path = bundle_dir / "mlp_calibrated.joblib"

    cal_available = rf_cal_path.exists() and gb_cal_path.exists() and mlp_cal_path.exists()
    use_cal = (cal_available and (not disable_calibration))

    rf_cal = gb_cal = mlp_cal = None
    if use_cal:
        rf_cal = joblib.load(rf_cal_path)
        gb_cal = joblib.load(gb_cal_path)
        mlp_cal = joblib.load(mlp_cal_path)

    return {
        "bundle_dir": bundle_dir,
        "classes": classes,
        "pre": pre,
        "rf": rf,
        "gb": gb,
        "mlp": mlp,
        "elm": elm,
        "cal_available": bool(cal_available),
        "use_cal": bool(use_cal),
        "rf_cal": rf_cal,
        "gb_cal": gb_cal,
        "mlp_cal": mlp_cal,
    }


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    suf = path.suffix.lower()
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suf in [".csv"]:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input file extension: {suf}")


def _coerce_numeric_with_stats(X: pd.DataFrame) -> Tuple[pd.DataFrame, int, int, int]:
    """Convert non-numeric columns using to_numeric(errors='coerce') and track NaN changes."""
    X0 = X.copy()
    nan_before = int(X0.isna().to_numpy().sum())

    for c in X0.columns:
        if not pd.api.types.is_numeric_dtype(X0[c]):
            X0[c] = pd.to_numeric(X0[c], errors="coerce")

    nan_after = int(X0.isna().to_numpy().sum())
    nan_added = int(max(0, nan_after - nan_before))
    return X0, nan_before, nan_after, nan_added


def preprocess_with_diagnostics(df: pd.DataFrame, pre: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply the same preprocessing as inference_ensemble.py, but also return diagnostics."""

    # Match training-time filtering
    X = df.copy()
    X = X.filter(regex=r"^(?!.*Keq)")
    for c in ["Classe", "Composto", "Átomos"]:
        if c in X.columns:
            X = X.drop(columns=[c])

    # Numeric coercion (may introduce NaNs)
    X, nan_before, nan_after, nan_added_by_coerce = _coerce_numeric_with_stats(X)

    med: Dict[str, float] = pre["medians"]
    med_keys = list(med.keys())

    missing_cols = [c for c in med_keys if c not in X.columns]
    extra_cols = [c for c in X.columns if c not in med_keys]

    # Add missing columns as NaN so we can impute (one shot; avoids fragmentation)
    if len(missing_cols) > 0:
        X = X.reindex(columns=list(X.columns) + list(missing_cols))
    # Order like training median dict keys
    X = X[med_keys]

    n_missing_cells_before_impute = int(X.isna().to_numpy().sum())

    # Median impute
    X = X.fillna(pd.Series(med))

    # Constant filter and final feature subset
    kept_after_const = pre.get("kept_after_constant_filter", None)
    if kept_after_const is not None:
        X = X[kept_after_const]

    final_features: List[str] = list(pre["final_features"])
    X = X[final_features]

    # Scale (mean/scale were fit on final_features)
    mu = np.array(pre["scaler_mean"], dtype=float)
    sg = np.array(pre["scaler_scale"], dtype=float)
    X_scaled = (X.values - mu) / sg

    diagnostics: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_input_columns": int(df.shape[1]),
        "n_required_median_columns": int(len(med_keys)),
        "n_missing_required_columns": int(len(missing_cols)),
        "missing_required_columns": missing_cols,
        "n_extra_columns_ignored": int(len(extra_cols)),
        "extra_columns_ignored": extra_cols,
        "nan_cells_before_numeric_coerce": nan_before,
        "nan_cells_after_numeric_coerce": nan_after,
        "nan_cells_added_by_numeric_coerce": nan_added_by_coerce,
        "nan_cells_before_imputation": n_missing_cells_before_impute,
        "imputed_cells": n_missing_cells_before_impute,
        "features_used": final_features,
        "n_features_used": int(len(final_features)),
    }

    return X.values, X_scaled, diagnostics


def _idx_to_labels(idx: np.ndarray, classes: List[str]) -> List[str]:
    return [classes[int(i)] for i in idx]


def predict_all(df_in: pd.DataFrame, bundle: Dict[str, Any]) -> Dict[str, Any]:
    classes = bundle["classes"]

    X_raw, X_scaled, diag = preprocess_with_diagnostics(df_in, bundle["pre"])

    use_cal = bundle["use_cal"]

    # Probabilistic models
    if use_cal:
        P_rf = bundle["rf_cal"].predict_proba(X_raw)
        P_gb = bundle["gb_cal"].predict_proba(X_raw)
        P_mlp = bundle["mlp_cal"].predict_proba(X_scaled)
    else:
        P_rf = bundle["rf"].predict_proba(X_raw)
        P_gb = bundle["gb"].predict_proba(X_raw)
        P_mlp = bundle["mlp"].predict_proba(X_scaled)

    y_rf = np.argmax(P_rf, axis=1)
    y_gb = np.argmax(P_gb, axis=1)
    y_mlp = np.argmax(P_mlp, axis=1)

    # ELM (labels only)
    y_elm = np.asarray(bundle["elm"].predict(X_scaled)).astype(int)
    P_elm = np.zeros_like(P_rf)
    P_elm[np.arange(len(y_elm)), y_elm] = 1.0

    # Ensemble rule
    P_ens = (P_rf + P_gb + P_mlp + P_elm) / 4.0
    y_ens = np.argmax(P_ens, axis=1)

    results = {
        "diagnostics": diag,
        "RandomForest": {"y_idx": y_rf, "y_label": _idx_to_labels(y_rf, classes), "proba": P_rf},
        "GradientBoosting": {"y_idx": y_gb, "y_label": _idx_to_labels(y_gb, classes), "proba": P_gb},
        "MLP": {"y_idx": y_mlp, "y_label": _idx_to_labels(y_mlp, classes), "proba": P_mlp},
        "ELM": {"y_idx": y_elm, "y_label": _idx_to_labels(y_elm, classes), "proba": P_elm},
        "Ensemble": {"y_idx": y_ens, "y_label": _idx_to_labels(y_ens, classes), "proba": P_ens},
    }

    return results



def normalize_label(label: str) -> str:
    """Normalize label strings to ensure consistent matching between predictions and ground truth."""
    if label is None:
        return label
    s = str(label).strip()
    # Handle known naming mismatch (Portuguese vs English)
    if s == "Fluorita":
        return "Fluorite"
    return s


def compute_metrics(
    y_true_label: List[str],
    y_pred_label: List[str],
    classes: List[str],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Compute accuracy, weighted P/R/F1, and class-wise P/R/F1/support."""
    labels = list(classes)

    acc = float(accuracy_score(y_true_label, y_pred_label))

    # Weighted metrics (keep as-is)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true_label, y_pred_label, labels=labels, average="weighted", zero_division=0
    )

    # Class-wise metrics (keep as-is)
    p_c, r_c, f1_c, sup_c = precision_recall_fscore_support(
        y_true_label, y_pred_label, labels=labels, average=None, zero_division=0
    )

    summary: Dict[str, Any] = {
        "accuracy": acc,
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f1_w),
    }

    classwise = pd.DataFrame(
        {
            "class": labels,
            "precision": p_c.astype(float),
            "recall": r_c.astype(float),
            "f1": f1_c.astype(float),
            "support": sup_c.astype(int),
        }
    )

    return summary, classwise



# =====================================================================================
# ROC (OvR) for Ensemble on external validation (purple style)
# =====================================================================================
def plot_roc_ovr_ensemble_external(
    y_true_labels: List[str],
    P: np.ndarray,
    classes: List[str],
    out_prefix: str = "roc_ovr_ensemble_purple",
) -> Optional[pd.DataFrame]:
    """
    Plots one-vs-rest ROC curves for the Ensemble using the provided probabilities.
    Saves:
      - {out_prefix}_600dpi.png
      - {out_prefix}.pdf
      - {out_prefix}_auc_table.csv

    Returns: DataFrame with per-class AUC (translated to EN names for readability).
    """
    import re as _re
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    # PT -> EN phase names (for plot labels only)
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

    def tr(x: str) -> str:
        x = str(x).strip()
        return pt_to_en.get(x, x)

    # -----------------------------
    # PURPLE STYLE (match your script)
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
    rcParams["legend.fontsize"]  = 14

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
    # Prepare y_true indices + binarized matrix
    # -----------------------------
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[x] for x in y_true_labels], dtype=int)

    n_classes = len(classes)
    if P.shape[1] != n_classes:
        raise ValueError(f"Ensemble proba has {P.shape[1]} columns, expected {n_classes}.")

    Y = label_binarize(y_idx, classes=np.arange(n_classes))

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=False)

    rows = []
    any_curve = False
    for k, cls in enumerate(classes):
        pos = int(Y[:, k].sum())
        if pos == 0 or pos == Y.shape[0]:
            rows.append({"class_name": tr(cls), "auc_ovr": np.nan, "support_pos": pos})
            continue

        fpr, tpr, _ = roc_curve(Y[:, k], P[:, k])
        ak = float(auc(fpr, tpr))
        ax.plot(fpr, tpr, label=f"{tr(cls)} (AUC={ak:.2f})")
        rows.append({"class_name": tr(cls), "auc_ovr": ak, "support_pos": pos})
        any_curve = True

    # Baseline
    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=3)

    ax.set_title("Ensemble ROC (Akrami et al. dataset)", pad=12)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)

    if any_curve:
        leg = ax.legend(loc="lower right", fontsize=24, frameon=True)
        leg.get_frame().set_edgecolor("#4a148c")
        leg.get_frame().set_linewidth(1.0)

    out_png = Path(f"{out_prefix}_600dpi.png")
    out_pdf = Path(f"{out_prefix}.pdf")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    df_auc = pd.DataFrame(rows)
    df_auc.to_csv(f"{out_prefix}_auc_table.csv", index=False)

    return df_auc



def evaluate_ensemble_if_possible(
    df_in: pd.DataFrame,
    results: Dict[str, Any],
    classes: List[str],
    label_col: str = "Crystal Structure",
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame], Dict[str, Any]]:
    """Evaluate ensemble predictions if a true-label column exists."""
    meta: Dict[str, Any] = {"label_col": label_col}

    if label_col not in df_in.columns:
        meta["status"] = "skipped"
        meta["reason"] = f"Label column '{label_col}' not found."
        return None, None, meta

    s = df_in[label_col]
    mask = s.notna()
    meta["n_rows_total"] = int(len(df_in))
    meta["n_rows_with_label"] = int(mask.sum())
    meta["n_rows_missing_label"] = int((~mask).sum())

    if int(mask.sum()) == 0:
        meta["status"] = "skipped"
        meta["reason"] = f"Column '{label_col}' exists, but all values are NaN."
        return None, None, meta
    y_true = s[mask].astype(str).str.strip().map(lambda x: normalize_label_to_classes(x, classes))

    y_pred = pd.Series(results["Ensemble"]["y_label"], index=df_in.index)[mask].astype(str).str.strip().map(lambda x: normalize_label_to_classes(x, classes))

    # Drop unknown true labels (not in the trained class set)
    known = y_true.isin(classes)
    n_unknown = int((~known).sum())
    meta["n_rows_unknown_label"] = n_unknown

    # Row indices used for evaluation (to slice probas for ROC)
    mask_idx = np.where(mask.to_numpy())[0]
    eval_idx = mask_idx[known.to_numpy()]
    meta['eval_row_indices'] = eval_idx.tolist()


    if n_unknown > 0:
        unknown_vals = sorted(set(y_true[~known].tolist()))
        meta["unknown_labels_preview"] = "; ".join(map(str, unknown_vals[:20])) + (" ..." if len(unknown_vals) > 20 else "")

        y_true = y_true[known]
        y_pred = y_pred[known]

    meta['y_true_labels_eval'] = y_true.tolist()
    meta["n_rows_evaluated"] = int(len(y_true))

    if len(y_true) == 0:
        meta["status"] = "skipped"
        meta["reason"] = "No evaluable rows after filtering unknown labels."
        return None, None, meta

    summary, classwise = compute_metrics(
        y_true_label=y_true.tolist(),
        y_pred_label=y_pred.tolist(),
        classes=classes,
    )

    meta["status"] = "ok"
    return summary, classwise, meta



def _add_model_outputs(
    out: pd.DataFrame,
    model_name: str,
    classes: List[str],
    y_idx: np.ndarray,
    y_label: List[str],
    proba: np.ndarray,
    write_full_proba: bool,
) -> None:
    prefix = model_name
    out[f"pred_{prefix}"] = [normalize_label_output(x) for x in y_label]
    out[f"pred_{prefix}_idx"] = y_idx.astype(int)
    out[f"pred_{prefix}_pmax"] = np.max(proba, axis=1).astype(float)

    if write_full_proba:
        for k, cname in enumerate(classes):
            out[f"proba_{prefix}_{cname}"] = proba[:, k].astype(float)


def write_excel(
    df_in: pd.DataFrame,
    bundle: Dict[str, Any],
    results: Dict[str, Any],
    output_path: Path,
    write_full_proba: bool,
) -> None:
    classes = bundle["classes"]

    out = df_in.copy()

    for model_name in ["RandomForest", "GradientBoosting", "MLP", "ELM", "Ensemble"]:
        r = results[model_name]
        _add_model_outputs(
            out,
            model_name=model_name,
            classes=classes,
            y_idx=r["y_idx"],
            y_label=r["y_label"],
            proba=r["proba"],
            write_full_proba=write_full_proba,
        )

    diag = results["diagnostics"].copy()
    diag.update(
        {
            "bundle_dir": str(bundle["bundle_dir"].resolve()),
            "calibration_available": bundle["cal_available"],
            "calibration_used": bundle["use_cal"],
            "note": "If missing_required_columns>0, those columns were created and imputed with training medians.",
        }
    )

    info_rows = []
    for k, v in diag.items():
        if isinstance(v, list):
            # write lists as a single joined string (Excel-friendly)
            info_rows.append({"key": k, "value": "; ".join(map(str, v))})
        else:
            info_rows.append({"key": k, "value": v})
    # info sheet disabled by user request
    # features_used sheet disabled by user request

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="predictions", index=False)

        # Optional: evaluation metrics (if label column exists)
        if results.get("metrics_summary") is not None:
            pd.DataFrame([results["metrics_summary"]]).to_excel(writer, sheet_name="metrics_summary", index=False)

        if results.get("metrics_classwise") is not None:
            results["metrics_classwise"].to_excel(writer, sheet_name="metrics_classwise", index=False)

        if results.get("metrics_meta") is not None:
            pd.DataFrame([results["metrics_meta"]]).to_excel(writer, sheet_name="metrics_meta", index=False)

        if results.get("roc_auc_table") is not None:
            results["roc_auc_table"].to_excel(writer, sheet_name="roc_auc_table", index=False)


def _print_diagnostics(bundle: Dict[str, Any], diagnostics: Dict[str, Any]) -> None:
    # Reduced terminal output by user request (no preprocessing diagnostics block).
    print("\n===== Inference settings =====")
    print(f"Bundle dir: {bundle['bundle_dir'].resolve()}")
    print(f"Calibration available: {bundle['cal_available']}")
    print(f"Calibration used:      {bundle['use_cal']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bundle_dir",
        type=str,
        default=".",
        help="Path to final_bundle directory (default: current directory).",
    )
    ap.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input spreadsheet to run inference on (.xlsx/.xls/.csv).",
    )
    ap.add_argument(
        "--label_col",
        type=str,
        default="Crystal Structure",
        help="Name of the column containing the true labels (for external validation metrics). Default: 'Crystal Structure'.",
    )

    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output Excel file path (default: <bundle_dir>/predictions_ensemble.xlsx).",
    )
    ap.add_argument(
        "--disable_calibration",
        action="store_true",
        help="Force use of uncalibrated RF/GB/MLP even if calibrated models exist in the bundle.",
    )
    ap.add_argument(
        "--write_full_proba",
        action="store_true",
        help="If set, write per-class probability columns for each model (can be many columns).",
    )
    ap.add_argument(
        "--suppress_version_warnings",
        action="store_true",
        help="Suppress sklearn InconsistentVersionWarning messages.",
    )
    args = ap.parse_args()

    if args.suppress_version_warnings:
        warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

    bundle_dir = Path(args.bundle_dir)
    if not bundle_dir.exists():
        raise FileNotFoundError(bundle_dir)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_path = Path(args.output) if args.output is not None else (bundle_dir / "predictions_ensemble.xlsx")

    df_in = _read_table(input_path)

    bundle = load_bundle(bundle_dir=bundle_dir, disable_calibration=args.disable_calibration)

    results = predict_all(df_in, bundle)
    # =========================
    # External validation metrics (if true labels exist)
    # =========================
    metrics_summary, metrics_classwise, metrics_meta = evaluate_ensemble_if_possible(
        df_in=df_in,
        results=results,
        classes=bundle["classes"],
        label_col=args.label_col,
    )

    results["metrics_summary"] = metrics_summary
    results["metrics_classwise"] = metrics_classwise
    results["metrics_meta"] = metrics_meta


    # ROC curve (Ensemble) on external validation (saved as PNG/PDF + AUC table)
    results["roc_auc_table"] = None
    try:
        if metrics_meta.get("status") == "ok":
            eval_idx = metrics_meta.get("eval_row_indices", [])
            y_true_labels_eval = metrics_meta.get("y_true_labels_eval", [])
            if len(eval_idx) > 0 and len(y_true_labels_eval) == len(eval_idx):
                P_eval = np.asarray(results["Ensemble"]["proba"], dtype=float)[eval_idx, :]
                results["roc_auc_table"] = plot_roc_ovr_ensemble_external(
                    y_true_labels=y_true_labels_eval,
                    P=P_eval,
                    classes=bundle["classes"],
                    out_prefix="roc_ovr_ensemble_purple",
                )
    except Exception:
        # Keep silent: small external validation may lack enough positives for ROC in some classes
        results["roc_auc_table"] = None

    if metrics_summary is not None:
        print("\n===== Ensemble metrics (external validation) =====")
        for k, v in metrics_summary.items():
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

        print("\n===== Ensemble class-wise (precision/recall/f1/support) =====")
        assert metrics_classwise is not None
        print(metrics_classwise.to_string(index=False))

        if metrics_meta.get("n_rows_unknown_label", 0) > 0:
            print("\n[WARN] Some true labels were not in the trained class set and were excluded from evaluation.")
            prev = metrics_meta.get("unknown_labels_preview", "")
            if prev:
                print(f"       Unknown labels (preview): {prev}")
    else:
        reason = (metrics_meta or {}).get("reason", "Unknown reason")
        print(f"\n[INFO] Metrics skipped: {reason}")



    _print_diagnostics(bundle, results["diagnostics"])

    write_excel(
        df_in=df_in,
        bundle=bundle,
        results=results,
        output_path=output_path,
        write_full_proba=args.write_full_proba,
    )

    print(f"\nOK: {output_path.resolve()}")


if __name__ == "__main__":
    main()