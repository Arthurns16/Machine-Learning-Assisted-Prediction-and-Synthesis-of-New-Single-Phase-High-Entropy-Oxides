#!/usr/bin/env python3
# =========================
# Finalize nested CV results:
# - Aggregate metrics across folds (mean ± std) INCLUDING class-wise mean ± std
# - Confusion matrices (counts + normalized) for all models (OOF)
# - One-vs-rest ROC curves and per-class AUC (OOF) for models with probabilities (RF/GB/MLP/Ensemble)
# - Train FINAL models on ALL data using hyperparameters aggregated from folds (NO hardcoded fallback)
# - Write an inference bundle (final_bundle/) and write ALL CV metric tables ONLY in final_models/
# =========================

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize


# ---------------- Utilities ----------------

def sanitize_name(s: str) -> str:
    # Keep consistent with nestedcv_run_fold.py behavior (no separators)
    return re.sub(r"[^A-Za-z0-9]+", "", s)

def json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (tuple,)):
        return list(o)
    return str(o)

def safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

def atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


# ---------------- ELM Wrapper ----------------

class ELMWrapper(BaseEstimator, ClassifierMixin):
    """
    Pickle-safe wrapper around the user's ELM implementation in elm.py.

    Supports two APIs:
    1) sklearn-like: elm.py defines ELMClassifier with fit/predict.
    2) core API (used in nestedcv_run_fold.py): elm.py defines class `elm`,
       constructed with x/y then trained via .fit(treinador).
    """

    def __init__(self, random_state: int = 42, **kwargs):
        self.random_state = int(random_state)
        self.kwargs = dict(kwargs)
        self.model_ = None
        self._mode = None  # "sklearn" or "core"

    def _load_elm_module(self):
        import importlib.util

        elm_path = Path(self.kwargs.get("_elm_path", "elm.py"))
        if not elm_path.exists():
            raise FileNotFoundError(f"ELM file not found: {elm_path}")

        # Ensure folder is importable
        if str(elm_path.parent) not in sys.path:
            sys.path.insert(0, str(elm_path.parent))

        # Import as module name "elm" (critical for joblib pickling)
        spec = importlib.util.spec_from_file_location("elm", str(elm_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import elm module from {elm_path}")

        mod = importlib.util.module_from_spec(spec)
        sys.modules["elm"] = mod
        spec.loader.exec_module(mod)  # type: ignore
        return mod

    def fit(self, X, y):
        mod = self._load_elm_module()

        kwargs = dict(self.kwargs)
        kwargs.pop("_elm_path", None)
        kwargs.setdefault("random_state", self.random_state)

        if hasattr(mod, "ELMClassifier"):
            self._mode = "sklearn"
            ELMClassifier = getattr(mod, "ELMClassifier")
            self.model_ = ELMClassifier(**kwargs)
            self.model_.fit(X, y)
            return self

        if hasattr(mod, "elm"):
            self._mode = "core"
            ELMCore = getattr(mod, "elm")

            hidden_units = int(kwargs.get("hidden_units", 50))
            activation_function = str(kwargs.get("activation_function", "relu"))
            C = float(kwargs.get("C", 1))
            random_type = str(kwargs.get("random_type", "normal"))
            treinador = str(kwargs.get("treinador", "no_re"))

            np.random.seed(int(kwargs.get("random_state", self.random_state)))
            self.model_ = ELMCore(
                hidden_units=hidden_units,
                activation_function=activation_function,
                x=np.asarray(X),
                y=np.asarray(y).astype(int),
                C=C,
                elm_type="clf",
                one_hot=True,
                random_type=random_type,
            )
            self.model_.fit(treinador)
            return self

        raise AttributeError(
            "elm.py must define either ELMClassifier (sklearn-like) or elm (core API)."
        )

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("ELMWrapper not fitted.")
        X = np.asarray(X)
        y_pred = self.model_.predict(X)

        # Guard against implementations that ignore X and return wrong-length output
        if hasattr(y_pred, "shape") and y_pred.shape[0] != X.shape[0]:
            raise RuntimeError(
                f"ELM predict returned {y_pred.shape[0]} preds for X with {X.shape[0]} rows."
            )
        return np.asarray(y_pred).astype(int)


# ---------------- Checkpoint Manager ----------------

@dataclass
class CheckpointManager:
    artifacts_dir: Path
    outer_splits: int

    @property
    def done_path(self) -> Path:
        return self.artifacts_dir / "done_folds.json"

    def fold_path(self, fold: int) -> Path:
        # Canonical layout written by nestedcv_run_fold.py:
        #   <artifacts>/folds/outer_fold_XX/
        candidates = [
            self.artifacts_dir / "folds" / f"outer_fold_{fold:02d}",
            self.artifacts_dir / "folds" / f"outer_fold_{fold}",
            self.artifacts_dir / f"fold_{fold}",
            self.artifacts_dir / f"outer_fold_{fold:02d}",
        ]
        for p in candidates:
            if p.exists():
                return p
        return candidates[0]

    def load_done_folds(self) -> List[int]:
        if not self.done_path.exists():
            return []
        txt = self.done_path.read_text(encoding="utf-8").strip()
        if not txt:
            return []
        data = json.loads(txt)
        return sorted(int(x) for x in data)

    def rebuild_oof(self, model_labels, model_proba):
        done = self.load_done_folds()
        if not done:
            raise RuntimeError("No folds completed yet (done_folds.json is empty).")

        oof_labels = {m: {"y_true": [], "y_pred": []} for m in model_labels}
        oof_proba = {m: [] for m in model_proba}

        for fold in done:
            fp = self.fold_path(fold)

            y_test_path = fp / "y_test.npy"
            if not y_test_path.exists():
                raise FileNotFoundError(f"Missing {y_test_path}")

            y_test = np.load(y_test_path).astype(int)

            for m in model_labels:
                p = fp / f"y_pred_{sanitize_name(m)}.npy"
                if not p.exists():
                    raise FileNotFoundError(f"Missing {p}")
                yp = np.load(p).astype(int)

                if len(yp) != len(y_test):
                    raise ValueError(
                        f"Fold {fold} model {m}: len(y_test)={len(y_test)} != len(y_pred)={len(yp)}"
                    )

                oof_labels[m]["y_true"].append(y_test)
                oof_labels[m]["y_pred"].append(yp)

            for m in model_proba:
                p = fp / f"proba_{sanitize_name(m)}.npy"
                if not p.exists():
                    raise FileNotFoundError(f"Missing {p}")
                P = np.load(p).astype(float)

                if P.shape[0] != len(y_test):
                    raise ValueError(
                        f"Fold {fold} model {m}: proba rows {P.shape[0]} != len(y_test)={len(y_test)}"
                    )

                oof_proba[m].append(P)

        # concatenate
        for m in model_labels:
            oof_labels[m]["y_true"] = np.concatenate(oof_labels[m]["y_true"]).astype(int)
            oof_labels[m]["y_pred"] = np.concatenate(oof_labels[m]["y_pred"]).astype(int)

        for m in model_proba:
            oof_proba[m] = np.concatenate(oof_proba[m], axis=0).astype(float)

        return oof_labels, oof_proba, done


# ---------------- Data utilities ----------------

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def median_impute(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    med = df.median(numeric_only=True).to_dict()
    return df.fillna(pd.Series(med)), med

def drop_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    nun = df.apply(pd.Series.nunique)
    keep = nun[nun != 1].index.tolist()
    dropped = [c for c in df.columns if c not in keep]
    return df[keep], dropped

def drop_high_corr_columns(X: pd.DataFrame, thresh: float) -> Tuple[pd.DataFrame, List[str]]:
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if (upper[col] > thresh).any()]
    return X.drop(columns=to_drop), to_drop


# ---------------- Postprocess: confusion + ROC ----------------

def postprocess_all(ckpt: CheckpointManager, class_names, n_classes, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    MODEL_LABELS = ["RandomForest", "GradientBoosting", "MLP", "ELM", "Ensemble"]
    MODEL_PROBA  = ["RandomForest", "GradientBoosting", "MLP", "Ensemble"]
    oof, oofP, done = ckpt.rebuild_oof(MODEL_LABELS, MODEL_PROBA)

    # Confusion matrices (OOF)
    for m in MODEL_LABELS:
        y_true, y_pred = oof[m]["y_true"], oof[m]["y_pred"]
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
        cmn = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes), normalize="true")

        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
            out_dir / f"confusion_counts_{sanitize_name(m)}.csv"
        )
        pd.DataFrame(cmn, index=class_names, columns=class_names).to_csv(
            out_dir / f"confusion_norm_{sanitize_name(m)}.csv"
        )

        fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
            ax=ax, cmap=None, colorbar=True, values_format="d"
        )
        ax.set_title(f"Confusion Matrix (Counts) - {m} (OOF)")
        plt.tight_layout()
        plt.savefig(out_dir / f"confusion_counts_{sanitize_name(m)}.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
        ConfusionMatrixDisplay(cmn, display_labels=class_names).plot(
            ax=ax, cmap=None, colorbar=True, values_format=".2f"
        )
        ax.set_title(f"Confusion Matrix (Normalized) - {m} (OOF)")
        plt.tight_layout()
        plt.savefig(out_dir / f"confusion_norm_{sanitize_name(m)}.png", dpi=300)
        plt.close(fig)

    # ROC OvR (OOF) for probability models
    def roc_ovr(model_name: str):
        y_true, proba = oof[model_name]["y_true"], oofP[model_name]
        Y = label_binarize(y_true, classes=np.arange(n_classes))
        rows = []

        plt.figure(figsize=(7, 6), dpi=200)
        for k in range(n_classes):
            if Y[:, k].sum() == 0 or Y[:, k].sum() == Y.shape[0]:
                rows.append({"model": model_name, "class_name": class_names[k], "auc_ovr": np.nan})
                continue
            fpr, tpr, _ = roc_curve(Y[:, k], proba[:, k])
            ak = auc(fpr, tpr)
            rows.append({"model": model_name, "class_name": class_names[k], "auc_ovr": float(ak)})
            plt.plot(fpr, tpr, linewidth=2, label=f"{class_names[k]} (AUC={ak:.2f})")

        plt.plot([0, 1], [0, 1], linestyle=":", linewidth=2)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"One-vs-Rest ROC Curves (OOF) - {model_name}")
        plt.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        plt.savefig(out_dir / f"roc_ovr_{sanitize_name(model_name)}.png", dpi=300)
        plt.close()
        return rows

    roc_rows = []
    for m in MODEL_PROBA:
        roc_rows.extend(roc_ovr(m))
    pd.DataFrame(roc_rows).to_csv(out_dir / "roc_ovr_auc_table.csv", index=False)

    safe_write_text(
        out_dir / "POSTPROCESS_SUMMARY.json",
        json.dumps({"done_folds": done, "n_folds": len(done)}, indent=2),
    )


# ---------------- Hyperparameter aggregation from folds ----------------

def strip_prefix(d: Dict[str, Any], prefix: str = "model__") -> Dict[str, Any]:
    out = {}
    for k, v in (d or {}).items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out

def _mode_with_tiebreak(values: List[Any], folds_here: List[int], outer_acc: Dict[int, float]) -> Any:
    """Choose most frequent value; tie-break using best outer fold accuracy among tied values."""

    def key(v):
        return json.dumps(v, default=json_default, sort_keys=True)

    counts: Dict[str, int] = {}
    for v in values:
        counts[key(v)] = counts.get(key(v), 0) + 1

    maxc = max(counts.values())
    tied_keys = [k for k, c in counts.items() if c == maxc]

    if len(tied_keys) == 1:
        tgt = tied_keys[0]
        for v in values:
            if key(v) == tgt:
                return v

    best_fold = max(folds_here, key=lambda f: outer_acc.get(f, -1e9))
    for v, f in zip(values, folds_here):
        if f == best_fold and key(v) in tied_keys:
            return v

    tgt = tied_keys[0]
    for v in values:
        if key(v) == tgt:
            return v
    return values[0]

def select_final_params_from_folds(ckpt: CheckpointManager, done_folds: List[int], model_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Read best_params.json from each fold and aggregate hyperparams by mode (tie-break by outer accuracy).
    Returns params WITHOUT pipeline prefix (model__).
    """

    outer_acc: Dict[str, Dict[int, float]] = {m: {} for m in model_names}
    for fold in done_folds:
        fp = ckpt.fold_path(fold)
        yt = np.load(fp / "y_test.npy").astype(int)
        for m in model_names:
            yp = np.load(fp / f"y_pred_{sanitize_name(m)}.npy").astype(int)
            outer_acc[m][fold] = float(accuracy_score(yt, yp))

    best_params_by_fold: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for fold in done_folds:
        bp_path = ckpt.fold_path(fold) / "best_params.json"
        if not bp_path.exists():
            raise FileNotFoundError(f"Missing {bp_path}. Cannot build final hyperparameters without it.")
        best_params_by_fold[fold] = json.loads(bp_path.read_text(encoding="utf-8"))

    final_params: Dict[str, Dict[str, Any]] = {}
    for model in model_names:
        keys = sorted({k for f in done_folds for k in (best_params_by_fold[f].get(model, {}) or {}).keys()})
        out: Dict[str, Any] = {}
        for k in keys:
            vals: List[Any] = []
            folds_here: List[int] = []
            for f in done_folds:
                d = best_params_by_fold[f].get(model, {}) or {}
                if k in d:
                    vals.append(d[k])
                    folds_here.append(f)
            if vals:
                out[k] = _mode_with_tiebreak(vals, folds_here, outer_acc[model])
        final_params[model] = strip_prefix(out, prefix="model__")

    return final_params


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--elm", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--outer_splits", type=int, default=4)
    ap.add_argument("--pearson_thresh", type=float, default=0.59)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--final_feature_policy", choices=["all_data", "stable_50pct"], default="all_data")
    ap.add_argument("--calibrate_final", action="store_true")
    ap.add_argument("--cal_method", type=str, default="sigmoid")
    ap.add_argument("--cal_cv", type=int, default=3)
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    elm_path = Path(args.elm)
    artifacts_dir = Path(args.artifacts)

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    if not elm_path.exists():
        raise FileNotFoundError(elm_path)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ckpt = CheckpointManager(artifacts_dir=artifacts_dir, outer_splits=args.outer_splits)
    done_folds = ckpt.load_done_folds()
    if len(done_folds) == 0:
        raise RuntimeError("No folds completed yet. Run the array jobs first.")

    # ---------- Load dataset ----------
    df = pd.read_excel(dataset_path) if dataset_path.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(dataset_path)

    # Basic cleaning/filters used in runs
    df = df.filter(regex=r"^(?!.*Keq)")
    for c in ["Composto", "Átomos"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    if "Classe" not in df.columns:
        raise ValueError("Dataset must contain 'Classe' column.")

    y_raw = df["Classe"].astype(str).values
    X = df.drop(columns=["Classe"]).copy()
    X = coerce_numeric(X)

    # Median impute
    X, medians = median_impute(X)

    # Drop constants
    X, dropped_const = drop_constant_columns(X)

    # Pearson drop on ALL data (for bundle)
    X_all_corr, dropped_corr_all = drop_high_corr_columns(X, args.pearson_thresh)

    # Choose final feature set policy
    if args.final_feature_policy == "all_data":
        final_features = X_all_corr.columns.tolist()
    else:
        fold_feats = []
        for f in done_folds:
            p = ckpt.fold_path(f) / "selected_features.csv"
            if p.exists():
                fold_feats.append(pd.read_csv(p)["feature"].astype(str).tolist())
        if not fold_feats:
            final_features = X_all_corr.columns.tolist()
        else:
            from collections import Counter
            cnt = Counter([c for lst in fold_feats for c in lst])
            min_count = max(1, int(np.ceil(len(fold_feats) * 0.5)))
            final_features = sorted([k for k, v in cnt.items() if v >= min_count])

    # Final matrices for training
    X_final = X_all_corr[final_features].copy()

    # Label encoding
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(int)
    classes = le.classes_.tolist()
    n_classes = len(classes)

    # Standard scaler for MLP/ELM paths
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final.values)
    mu = scaler.mean_.tolist()
    sigma = scaler.scale_.tolist()

    # Output dirs
    final_bundle_dir = artifacts_dir / "final_bundle"
    final_bundle_dir.mkdir(parents=True, exist_ok=True)

    final_models_dir = artifacts_dir / "final_models"
    final_models_dir.mkdir(parents=True, exist_ok=True)

    np.save(final_bundle_dir / "label_encoder_classes.npy", np.array(classes, dtype=object), allow_pickle=True)

    preproc = {
        "medians": medians,
        "dropped_by_constant_filter": dropped_const,
        "kept_after_constant_filter": X.columns.tolist(),
        "final_features": final_features,
        "dropped_by_pearson_all_data": dropped_corr_all,
        "scaler_mean": mu,
        "scaler_scale": sigma,
    }
    safe_write_text(final_bundle_dir / "final_preprocessing.json", json.dumps(preproc, indent=2, default=json_default))
    pd.DataFrame({"final_features": final_features}).to_csv(final_bundle_dir / "final_features.csv", index=False)

    # ---------- Aggregate hyperparameters from folds (NO hardcoded fallback) ----------
    model_names = ["RandomForest", "GradientBoosting", "MLP", "ELM"]
    final_params = select_final_params_from_folds(ckpt, done_folds, model_names)
    safe_write_text(final_models_dir / "final_hyperparameters_from_folds.json", json.dumps(final_params, indent=2, default=json_default))

    # ---------- Train final models on ALL data ----------
    TOTAL_CPUS = os.cpu_count() or 1

    rf_params = dict(final_params.get("RandomForest", {}))
    rf_params.setdefault("n_jobs", TOTAL_CPUS)
    rf = RandomForestClassifier(random_state=args.random_state, **rf_params)

    gb = GradientBoostingClassifier(random_state=args.random_state, **dict(final_params.get("GradientBoosting", {})))
    mlp = MLPClassifier(random_state=args.random_state, **dict(final_params.get("MLP", {})))

    elm_params = dict(final_params.get("ELM", {}))
    elm_params["_elm_path"] = str(elm_path)
    elm = ELMWrapper(random_state=args.random_state, **elm_params)

    rf.fit(X_final.values, y)
    gb.fit(X_final.values, y)
    mlp.fit(X_scaled, y)
    elm.fit(X_scaled, y)

    joblib.dump(rf, final_bundle_dir / "rf_final.joblib")
    joblib.dump(gb, final_bundle_dir / "gb_final.joblib")
    joblib.dump(mlp, final_bundle_dir / "mlp_final.joblib")
    joblib.dump(elm, final_bundle_dir / "elm_final.joblib")

    # Optional calibration for inference bundle
    min_count = int(pd.Series(y).value_counts().min())
    can_cal = (min_count >= int(args.cal_cv))
    cal_info = {
        "calibrate_final": bool(args.calibrate_final),
        "method": args.cal_method if args.calibrate_final else None,
        "cv": int(args.cal_cv) if args.calibrate_final else None,
        "min_class_count": min_count,
        "calibration_applied": False,
        "note": "",
    }

    if args.calibrate_final and can_cal:
        rf_cal = CalibratedClassifierCV(rf, method=args.cal_method, cv=args.cal_cv)
        gb_cal = CalibratedClassifierCV(gb, method=args.cal_method, cv=args.cal_cv)
        mlp_cal = CalibratedClassifierCV(mlp, method=args.cal_method, cv=args.cal_cv)

        rf_cal.fit(X_final.values, y)
        gb_cal.fit(X_final.values, y)
        mlp_cal.fit(X_scaled, y)

        joblib.dump(rf_cal, final_bundle_dir / "rf_calibrated.joblib")
        joblib.dump(gb_cal, final_bundle_dir / "gb_calibrated.joblib")
        joblib.dump(mlp_cal, final_bundle_dir / "mlp_calibrated.joblib")

        cal_info["calibration_applied"] = True
        cal_info["note"] = "Calibrated on ALL data with internal CV."
    else:
        cal_info["note"] = "Calibration disabled or skipped (insufficient samples per class or flag off)."

    safe_write_text(final_bundle_dir / "calibration_info.json", json.dumps(cal_info, indent=2, default=json_default))

    # Save ensemble config
    ensemble_cfg = {
        "rule": "P_ens = (P_rf + P_gb + P_mlp + onehot(argmax_elm))/4",
        "use_calibrated_probas": bool(cal_info["calibration_applied"]),
    }
    safe_write_text(final_bundle_dir / "ensemble_config.json", json.dumps(ensemble_cfg, indent=2))

    # ---------- OOF metrics (whole OOF) ----------
    oof_labels, oof_proba, _ = ckpt.rebuild_oof(
        ["RandomForest", "GradientBoosting", "MLP", "ELM", "Ensemble"],
        ["RandomForest", "GradientBoosting", "MLP", "Ensemble"],
    )

    # Save OOF metrics (overall + class-wise) in final_models/
    rows_summary, rows_classwise = [], []
    for m in ["RandomForest", "GradientBoosting", "MLP", "ELM", "Ensemble"]:
        yt, yp = oof_labels[m]["y_true"], oof_labels[m]["y_pred"]
        acc = accuracy_score(yt, yp)

        prec, rec, f1, sup = precision_recall_fscore_support(
            yt, yp, labels=np.arange(n_classes), zero_division=0
        )
        pm, rm, f1m, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
        pw, rw, f1w, _ = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)

        rows_summary.append({
            "model": m,
            "oof_accuracy": float(acc),
            "precision_macro": float(pm),
            "recall_macro": float(rm),
            "f1_macro": float(f1m),
            "precision_weighted": float(pw),
            "recall_weighted": float(rw),
            "f1_weighted": float(f1w),
            "n_samples_oof": int(len(yt)),
        })

        for k in range(n_classes):
            rows_classwise.append({
                "model": m,
                "class_name": classes[k],
                "precision": float(prec[k]),
                "recall": float(rec[k]),
                "f1": float(f1[k]),
                "support": int(sup[k]),
            })

    pd.DataFrame(rows_summary).to_csv(final_models_dir / "OOF_metrics_summary.csv", index=False)
    pd.DataFrame(rows_classwise).to_csv(final_models_dir / "OOF_metrics_classwise.csv", index=False)

    # ---------- Option 2: per-fold metrics -> mean±std (overall + class-wise) ----------
    MODEL_LABELS = ["RandomForest", "GradientBoosting", "MLP", "ELM", "Ensemble"]
    rows_fold_summary: List[dict] = []
    rows_fold_classwise: List[dict] = []

    for fold in done_folds:
        fp = ckpt.fold_path(fold)
        yt = np.load(fp / "y_test.npy").astype(int)
        n_test = int(len(yt))

        for m in MODEL_LABELS:
            yp = np.load(fp / f"y_pred_{sanitize_name(m)}.npy").astype(int)
            if len(yp) != len(yt):
                raise ValueError(
                    f"Fold {fold} model {m}: len(y_test)={len(yt)} != len(y_pred)={len(yp)}"
                )

            acc = float(accuracy_score(yt, yp))

            # class-wise
            prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
                yt, yp, labels=np.arange(n_classes), zero_division=0
            )
            for k in range(n_classes):
                rows_fold_classwise.append({
                    "fold": int(fold),
                    "model": m,
                    "class_name": classes[k],
                    "precision": float(prec_c[k]),
                    "recall": float(rec_c[k]),
                    "f1": float(f1_c[k]),
                    "support": int(sup_c[k]),
                })

            # macro/weighted
            pm, rm, f1m, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
            pw, rw, f1w, _ = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)

            rows_fold_summary.append({
                "fold": int(fold),
                "model": m,
                "accuracy": acc,
                "precision_macro": float(pm),
                "recall_macro": float(rm),
                "f1_macro": float(f1m),
                "precision_weighted": float(pw),
                "recall_weighted": float(rw),
                "f1_weighted": float(f1w),
                "n_test": n_test,
            })

    df_fold_summary = pd.DataFrame(rows_fold_summary)
    df_fold_classwise = pd.DataFrame(rows_fold_classwise)

    # Save per-fold tables (ONLY in final_models/)
    df_fold_summary.to_csv(final_models_dir / "fold_metrics.csv", index=False)
    df_fold_classwise.to_csv(final_models_dir / "fold_metrics_classwise.csv", index=False)

    def _flatten_agg(df: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
        g = df.groupby(group_cols, dropna=False)[value_cols].agg(["mean", "std"])
        g.columns = [f"{c}_{stat}" for c, stat in g.columns]
        return g.reset_index()

    metric_cols = [
        "accuracy",
        "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted",
    ]

    # Summary mean/std across folds per model
    df_ms = _flatten_agg(df_fold_summary, group_cols=["model"], value_cols=metric_cols)

    # Add n_folds and total samples
    n_folds_series = df_fold_summary.groupby("model", dropna=False)["fold"].nunique()
    n_samples_series = df_fold_summary.groupby("model", dropna=False)["n_test"].sum()
    df_ms["n_folds"] = df_ms["model"].map(n_folds_series).astype(int)
    df_ms["n_samples_total"] = df_ms["model"].map(n_samples_series).astype(int)

    # Pretty mean±std columns
    for c in metric_cols:
        mcol = f"{c}_mean"
        scol = f"{c}_std"
        if (mcol in df_ms.columns) and (scol in df_ms.columns):
            df_ms[f"{c}_mean_pm_std"] = df_ms.apply(lambda r: f"{r[mcol]:.4f} ± {r[scol]:.4f}", axis=1)

    # Save mean/std (ONLY in final_models/)
    df_ms.to_csv(final_models_dir / "fold_metrics_mean_std.csv", index=False)
    df_ms.to_csv(final_models_dir / "CV_metrics_summary_mean_std.csv", index=False)

    # Class-wise mean/std across folds per model+class
    df_cms = _flatten_agg(
        df_fold_classwise,
        group_cols=["model", "class_name"],
        value_cols=["precision", "recall", "f1", "support"],
    )

    sup_total = (
        df_fold_classwise.groupby(["model", "class_name"], dropna=False)["support"]
        .sum()
        .reset_index()
        .rename(columns={"support": "support_total"})
    )
    df_cms = df_cms.merge(sup_total, on=["model", "class_name"], how="left")

    for c in ["precision", "recall", "f1"]:
        mcol = f"{c}_mean"
        scol = f"{c}_std"
        if (mcol in df_cms.columns) and (scol in df_cms.columns):
            df_cms[f"{c}_mean_pm_std"] = df_cms.apply(lambda r: f"{r[mcol]:.4f} ± {r[scol]:.4f}", axis=1)

    # Save class-wise mean/std (ONLY in final_models/)
    df_cms.to_csv(final_models_dir / "fold_metrics_classwise_mean_std.csv", index=False)
    df_cms.to_csv(final_models_dir / "CV_metrics_classwise_mean_std.csv", index=False)

    # ---------- Postprocess figures (OOF) ----------
    outputs_dir = artifacts_dir / "outputs"
    postprocess_all(ckpt, classes, n_classes, outputs_dir)

    # ---------- Inference helper in bundle ----------
    inference_py = f'''\
import json
import sys
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

def _ensure_elm_module(bundle_dir: Path) -> None:
    """Ensure bundled elm.py is importable as module name 'elm' (needed for joblib.load)."""
    if str(bundle_dir) not in sys.path:
        sys.path.insert(0, str(bundle_dir))
    elm_path = bundle_dir / "elm.py"
    if elm_path.exists():
        spec = importlib.util.spec_from_file_location("elm", str(elm_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for {{elm_path}}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["elm"] = mod
        spec.loader.exec_module(mod)  # type: ignore

def load_bundle(bundle_dir):
    bundle_dir = Path(bundle_dir)
    _ensure_elm_module(bundle_dir)

    classes = np.load(bundle_dir / "label_encoder_classes.npy", allow_pickle=True).tolist()
    pre = json.loads((bundle_dir / "final_preprocessing.json").read_text(encoding="utf-8"))

    rf = joblib.load(bundle_dir / "rf_final.joblib")
    gb = joblib.load(bundle_dir / "gb_final.joblib")
    mlp = joblib.load(bundle_dir / "mlp_final.joblib")
    elm = joblib.load(bundle_dir / "elm_final.joblib")

    use_cal = (bundle_dir / "rf_calibrated.joblib").exists()
    rf_cal = gb_cal = mlp_cal = None
    if use_cal:
        rf_cal = joblib.load(bundle_dir / "rf_calibrated.joblib")
        gb_cal = joblib.load(bundle_dir / "gb_calibrated.joblib")
        mlp_cal = joblib.load(bundle_dir / "mlp_calibrated.joblib")

    return {{
        "classes": classes, "pre": pre,
        "rf": rf, "gb": gb, "mlp": mlp, "elm": elm,
        "use_cal": use_cal, "rf_cal": rf_cal, "gb_cal": gb_cal, "mlp_cal": mlp_cal
    }}

def preprocess(df, pre):
    X = df.copy()
    X = X.filter(regex=r"^(?!.*Keq)")
    for c in ["Classe","Composto","Átomos"]:
        if c in X.columns:
            X = X.drop(columns=[c])

    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    med = pre["medians"]
    for c in med.keys():
        if c not in X.columns:
            X[c] = np.nan

    X = X[list(med.keys())].fillna(pd.Series(med))
    X = X[pre["kept_after_constant_filter"]]
    X = X[pre["final_features"]]

    mu = np.array(pre["scaler_mean"], float)
    sg = np.array(pre["scaler_scale"], float)
    Xs = (X.values - mu) / sg
    return X.values, Xs

def predict_ensemble(df, bundle):
    pre = bundle["pre"]
    Xu, Xs = preprocess(df, pre)

    if bundle["use_cal"]:
        P_rf = bundle["rf_cal"].predict_proba(Xu)
        P_gb = bundle["gb_cal"].predict_proba(Xu)
        P_mlp = bundle["mlp_cal"].predict_proba(Xs)
    else:
        P_rf = bundle["rf"].predict_proba(Xu)
        P_gb = bundle["gb"].predict_proba(Xu)
        P_mlp = bundle["mlp"].predict_proba(Xs)

    y_elm = bundle["elm"].predict(Xs).astype(int)
    P_elm = np.zeros_like(P_rf)
    P_elm[np.arange(len(y_elm)), y_elm] = 1.0

    P = (P_rf + P_gb + P_mlp + P_elm) / 4.0
    y = np.argmax(P, axis=1)
    labels = [bundle["classes"][i] for i in y]
    return y, labels, P
'''
    safe_write_text(final_bundle_dir / "inference_ensemble.py", inference_py)

    # Copy elm.py into bundle for joblib load convenience
    atomic_copy(elm_path, final_bundle_dir / "elm.py")

    print("[OK] Final bundle:", final_bundle_dir)
    print("[OK] Final models tables (metrics):", final_models_dir)
    print("[OK] Outputs (ROC/confusion):", outputs_dir)
    print(f"[INFO] TOTAL_CPUS used for RF n_jobs: {TOTAL_CPUS}")


if __name__ == "__main__":
    main()

