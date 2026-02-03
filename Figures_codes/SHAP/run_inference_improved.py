#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


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

    # Add missing columns as NaN so we can impute
    for c in missing_cols:
        X[c] = np.nan

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
    out[f"pred_{prefix}"] = y_label
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
    df_info = pd.DataFrame(info_rows)

    # Also provide a dedicated features sheet
    df_features = pd.DataFrame({"feature": results["diagnostics"]["features_used"]})

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="predictions", index=False)
        df_info.to_excel(writer, sheet_name="info", index=False)
        df_features.to_excel(writer, sheet_name="features_used", index=False)


def _print_diagnostics(bundle: Dict[str, Any], diagnostics: Dict[str, Any]) -> None:
    print("\n===== Inference settings =====")
    print(f"Bundle dir: {bundle['bundle_dir'].resolve()}")
    print(f"Calibration available: {bundle['cal_available']}")
    print(f"Calibration used:      {bundle['use_cal']}")

    print("\n===== Preprocessing diagnostics =====")
    print(f"Rows in input: {diagnostics['n_rows']}")
    print(f"Features used: {diagnostics['n_features_used']}")

    if diagnostics["n_missing_required_columns"] > 0:
        print(f"Missing required columns: {diagnostics['n_missing_required_columns']}")
        print("  " + ", ".join(diagnostics["missing_required_columns"][:50]) + (" ..." if len(diagnostics["missing_required_columns"]) > 50 else ""))
    else:
        print("Missing required columns: 0")

    print(f"NaN cells before imputation: {diagnostics['nan_cells_before_imputation']}")
    if diagnostics["nan_cells_before_imputation"] == 0:
        print("Imputation: none needed (input already complete for required features).")
    else:
        print(f"Imputation: filled {diagnostics['imputed_cells']} cells using training medians.")

    if diagnostics["nan_cells_added_by_numeric_coerce"] > 0:
        print(f"Warning: numeric coercion introduced {diagnostics['nan_cells_added_by_numeric_coerce']} NaNs (non-numeric entries).")

    # Print a short feature preview
    feats = diagnostics["features_used"]
    preview = feats[:25]
    print("\nFirst 25 features used:")
    for f in preview:
        print("  -", f)
    if len(feats) > 25:
        print(f"  ... (+{len(feats) - 25} more)")


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
        "--output",
        type=str,
        default=None,
        help="Output Excel file path (default: <bundle_dir>/predicoes_ensemble.xlsx).",
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

    output_path = Path(args.output) if args.output is not None else (bundle_dir / "predicoes_ensemble.xlsx")

    df_in = _read_table(input_path)

    bundle = load_bundle(bundle_dir=bundle_dir, disable_calibration=args.disable_calibration)

    results = predict_all(df_in, bundle)

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
