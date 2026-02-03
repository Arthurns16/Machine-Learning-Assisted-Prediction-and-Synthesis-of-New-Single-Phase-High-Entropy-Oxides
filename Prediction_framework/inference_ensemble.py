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
            raise ImportError(f"Could not create import spec for {elm_path}")
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

    return {
        "classes": classes, "pre": pre,
        "rf": rf, "gb": gb, "mlp": mlp, "elm": elm,
        "use_cal": use_cal, "rf_cal": rf_cal, "gb_cal": gb_cal, "mlp_cal": mlp_cal
    }

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
