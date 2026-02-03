#!/usr/bin/env python3
# =========================
# SLURM array-safe: run ONE outer fold with inner tuning
# - RF/GB/MLP: GridSearchCV (parallel via n_jobs)
# - ELM: manual inner-CV tuning (robust to bad elm.py predict length)
# Saves checkpoints + status.json per fold + done_folds.json (locked)
#
# FIXED: race-condition on *.tmp files (unique tmp + atomic os.replace)
# FIXED: ELM inconsistent prediction length -> guarded + manual tuning
# FIXED: joblib.loads/dumps misuse -> replaced with sklearn.base.clone
# DEFAULT: inner_splits=3
# FIXED: ELM joblib pickle -> load elm.py as module name "elm" + register in sys.modules
# ASSUMPTION: dataset and elm.py are in same directory as this script
# =========================

import os, json, argparse, traceback, time, uuid, sys
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # kept for future use

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

try:
    import joblib
except Exception:
    joblib = None

try:
    from joblib import Parallel, delayed
    print("joblib ok")
except Exception:
    Parallel = None
    delayed = None

# ---------- CPU / thread hygiene ----------
def detect_allocated_cpus() -> int:
    for k in ["SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"]:
        v = os.environ.get(k)
        if v:
            import re
            m = re.search(r"\d+", v)
            if m:
                return max(1, int(m.group(0)))
    return max(1, os.cpu_count() or 1)

def set_thread_env():
    # Prevent oversubscription when GridSearchCV uses many processes
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

TOTAL_CPUS = detect_allocated_cpus()
set_thread_env()


# ---------- helpers ----------
def sanitize_name(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)

def _unique_tmp_path(final_path: Path) -> Path:
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    return final_path.parent / f".{final_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"

def safe_write_text(path: Path, text: str) -> None:
    path = Path(path)
    tmp = _unique_tmp_path(path)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

def safe_save_npy(path: Path, arr: np.ndarray) -> None:
    path = Path(path)
    tmp = _unique_tmp_path(path)
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)

def json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.ndarray,)): return o.tolist()
    if isinstance(o, (tuple,)): return list(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


# ---------- Transformers ----------
class DFMedianImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.feature_names_in_ = list(X.columns)
        self.medians_ = X.median(numeric_only=True).fillna(0.0)
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X = X.reindex(columns=self.feature_names_in_, fill_value=np.nan)
        return X.fillna(self.medians_)

class DFRemoveConstant(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.feature_names_in_ = list(X.columns)
        nunique = X.nunique(dropna=False)
        self.kept_features_ = nunique[nunique > 1].index.tolist()
        self.dropped_features_ = nunique[nunique <= 1].index.tolist()
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X = X.reindex(columns=self.feature_names_in_, fill_value=np.nan)
        return X[self.kept_features_]

class DFCorrFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.59):
        self.threshold = float(threshold)
    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.feature_names_in_ = list(X.columns)
        if X.shape[1] <= 1:
            self.kept_features_ = list(X.columns)
            self.dropped_features_ = []
            return self
        corr = X.corr(method="pearson").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if (upper[col] > self.threshold).any()]
        self.dropped_features_ = sorted(to_drop)
        self.kept_features_ = [c for c in X.columns if c not in to_drop]
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X = X.reindex(columns=self.feature_names_in_, fill_value=np.nan)
        return X[self.kept_features_]

class DFStandardScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.feature_names_in_ = list(X.columns)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X.values)
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X = X.reindex(columns=self.feature_names_in_, fill_value=0.0)
        Z = self.scaler_.transform(X.values)
        return pd.DataFrame(Z, columns=self.feature_names_in_, index=X.index)


# ---------- ELM import + wrapper ----------
def import_module_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, str(Path(path).resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {path}")
    mod = importlib.util.module_from_spec(spec)

    # CRITICAL: register so pickle/joblib can resolve module path on load
    sys.modules[module_name] = mod

    spec.loader.exec_module(mod)  # type: ignore
    return mod

class ELMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, ELMCore=None, hidden_units=50, activation_function="relu", C=1,
                 random_type="normal", treinador="no_re", random_state=42):
        self.ELMCore = ELMCore
        self.hidden_units = hidden_units
        self.activation_function = activation_function
        self.C = C
        self.random_type = random_type
        self.treinador = treinador
        self.random_state = random_state

    def fit(self, X, y):
        if self.ELMCore is None:
            raise RuntimeError("ELMCore not provided.")
        X = np.asarray(X)
        y = np.asarray(y)
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        np.random.seed(self.random_state)
        self.model_ = self.ELMCore(
            hidden_units=int(self.hidden_units),
            activation_function=str(self.activation_function),
            x=X,
            y=y_enc,
            C=float(self.C),
            elm_type="clf",
            one_hot=True,
            random_type=str(self.random_type),
        )
        self.model_.fit(str(self.treinador))
        return self

    def predict(self, X):
        X = np.asarray(X)
        pred_enc = self.model_.predict(X).astype(int)

        # CRITICAL GUARD: if elm.py ignores X and returns wrong length, fail fast
        if pred_enc.shape[0] != X.shape[0]:
            raise RuntimeError(
                f"ELM predict returned {pred_enc.shape[0]} preds for X with {X.shape[0]} rows. "
                "Your elm.py predict likely ignores the input X and uses internal training data."
            )
        return self.le_.inverse_transform(pred_enc)


# ---------- Pipeline helpers ----------
def make_pipeline(model, use_scaler=False, pearson_thresh=0.59):
    steps = [
        ("imputer", DFMedianImputer()),
        ("rm_const", DFRemoveConstant()),
        ("corr", DFCorrFilter(threshold=pearson_thresh)),
    ]
    if use_scaler:
        steps.append(("scaler", DFStandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)

def fit_best(pipe, grid, X_train, y_train, inner_cv, n_jobs):
    gs = GridSearchCV(
        pipe, grid,
        scoring="accuracy",
        cv=inner_cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
        error_score=np.nan
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_, float(gs.best_score_)

def get_pipeline_feature_info(fitted_pipeline: Pipeline) -> dict:
    info = {}
    rc = fitted_pipeline.named_steps.get("rm_const", None)
    cf = fitted_pipeline.named_steps.get("corr", None)
    if rc is not None:
        info["rm_const_kept_features"] = getattr(rc, "kept_features_", None)
        info["rm_const_dropped_features"] = getattr(rc, "dropped_features_", None)
    if cf is not None:
        info["pearson_kept_features"] = getattr(cf, "kept_features_", None)
        info["pearson_dropped_features"] = getattr(cf, "dropped_features_", None)
    return info

def get_estimator_classes(est):
    if hasattr(est, "classes_"):
        return np.asarray(est.classes_)
    if hasattr(est, "named_steps") and "model" in est.named_steps:
        if hasattr(est.named_steps["model"], "classes_"):
            return np.asarray(est.named_steps["model"].classes_)
    raise AttributeError("Could not retrieve estimator classes_.")

def align_proba(proba, classes, n_classes):
    out = np.zeros((proba.shape[0], n_classes), dtype=float)
    for j, c in enumerate(classes):
        out[:, int(c)] = proba[:, j]
    return out

def predict_proba_safe(best_est, X_train, y_train, X_test, n_classes, calibrate, cal_method, cal_cv):
    if calibrate:
        cal = CalibratedClassifierCV(best_est, method=cal_method, cv=cal_cv)
        cal.fit(X_train, y_train)
        proba = cal.predict_proba(X_test)
        classes = get_estimator_classes(cal)
        return align_proba(proba, classes, n_classes), cal
    proba = best_est.predict_proba(X_test)
    classes = get_estimator_classes(best_est)
    return align_proba(proba, classes, n_classes), None


# ---------- Manual ELM tuning (robust) ----------
def iter_param_dicts(param_grid_elm):
    # param_grid_elm is a list of dicts (like sklearn param_grid)
    for grid in param_grid_elm:
        keys = list(grid.keys())
        vals = [grid[k] for k in keys]
        import itertools
        for combo in itertools.product(*vals):
            yield {k: v for k, v in zip(keys, combo)}



def tune_elm_manual(
    pipe_elm_template: Pipeline,
    param_grid_elm,
    X_train,
    y_train,
    inner_cv,
    random_state: int,
    fold_dir: Path,
    status_hook=None,
    n_jobs_elm: int = 1,
    batch_size: int | None = None,
    parallel_backend: str = "loky",   # "loky" (processos) ÃƒÂ© o padrÃƒÂ£o do joblib
):
    """
    Manual inner-CV tuning do ELM, agora PARALLELIZADO por TRIAL (combinaÃƒÂ§ÃƒÂ£o de hiperparÃƒÂ¢metros).

    - Cada trial executa o inner-CV (INNER_SPLITS) sequencialmente dentro do worker.
    - VÃƒÂ¡rios trials rodam em paralelo usando n_jobs_elm.
    - Robustez mantida: trials que falham sÃƒÂ£o ignorados.

    ObservaÃƒÂ§ÃƒÂ£o: isto usa mÃƒÂºltiplos CPUs por paralelismo entre trials; cada fit do ELM tende a ser single-thread.
    """
    # fallback para modo sequencial se joblib Parallel nÃƒÂ£o estiver disponÃƒÂ­vel
    can_parallel = (Parallel is not None) and (delayed is not None) and (n_jobs_elm is not None) and (n_jobs_elm > 1)

    params_list = list(iter_param_dicts(param_grid_elm))
    n_trials = len(params_list)

    best_score = -np.inf
    best_params = None
    best_est = None

    total = 0
    ok = 0
    bad = 0

    log_path = fold_dir / "elm_tuning_log.tsv"
    if not log_path.exists():
        safe_write_text(log_path, "trial\tmean_acc\tn_ok_folds\tn_fail_folds\tparams_json\n")

    # batch_size: por default, um lote proporcional ao nÃƒÂºmero de CPUs (reduz overhead do joblib)
    if batch_size is None:
        batch_size = max(25, int(10 * (n_jobs_elm if n_jobs_elm else 1)))

    # FunÃƒÂ§ÃƒÂ£o que avalia 1 trial
    def _eval_one(trial_idx: int, params: dict):
        # trial_idx ÃƒÂ© 1-based (sÃƒÂ³ para log)
        est = clone(pipe_elm_template)
        est.set_params(**params)

        fold_scores = []
        n_ok_local = 0
        n_fail_local = 0

        for _, (tr_i, va_i) in enumerate(inner_cv.split(X_train, y_train), start=1):
            X_tr = X_train.iloc[tr_i]
            y_tr = y_train[tr_i]
            X_va = X_train.iloc[va_i]
            y_va = y_train[va_i]
            try:
                est_i = clone(est)
                est_i.fit(X_tr, y_tr)
                y_hat = est_i.predict(X_va)  # ELMWrapper pode levantar erro se tamanho inconsistente
                acc = accuracy_score(y_va, y_hat)
                fold_scores.append(acc)
                n_ok_local += 1
            except Exception:
                n_fail_local += 1
                continue

        if n_ok_local == 0:
            mean_acc = np.nan
        else:
            mean_acc = float(np.mean(fold_scores))

        return (trial_idx, mean_acc, n_ok_local, n_fail_local, params)

    # Executa em lotes (para poder atualizar status e log continuamente)
    trial = 1
    while trial <= n_trials:
        end = min(n_trials, trial + batch_size - 1)
        batch = [(i, params_list[i - 1]) for i in range(trial, end + 1)]

        if can_parallel:
            results = Parallel(n_jobs=n_jobs_elm, backend=parallel_backend)(
                delayed(_eval_one)(i, p) for (i, p) in batch
            )
        else:
            results = [_eval_one(i, p) for (i, p) in batch]

        # processa resultados do lote na thread principal (log/status/best)
        for (trial_idx, mean_acc, n_ok_local, n_fail_local, params) in results:
            total += 1
            if n_ok_local == 0:
                bad += 1
            else:
                ok += 1

            # append no log
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{trial_idx}\t{mean_acc}\t{n_ok_local}\t{n_fail_local}\t{json.dumps(params, default=json_default)}\n"
                )

            # atualiza best
            if np.isfinite(mean_acc) and mean_acc > best_score:
                best_score = float(mean_acc)
                best_params = params
                # guarda uma cÃƒÂ³pia "limpa" configurada (vai ser re-fit ao final)
                best_est = clone(pipe_elm_template)
                best_est.set_params(**params)

            # status hook (a cada 25 trials, como antes)
            if status_hook and (trial_idx % 25 == 0):
                status_hook({
                    "elm_trials_done": int(trial_idx),
                    "elm_total_seen": int(total),
                    "elm_ok_trials": int(ok),
                    "elm_bad_trials": int(bad),
                    "elm_current_best": float(best_score) if np.isfinite(best_score) else None,
                    "elm_n_jobs": int(n_jobs_elm),
                    "elm_parallel": bool(can_parallel),
                    "elm_backend": parallel_backend if can_parallel else None,
                })

        trial = end + 1

    if best_est is None:
        raise RuntimeError(
            "ELM manual tuning failed: no configuration produced valid predictions. "
            "Your elm.py predict is likely incompatible with sklearn-style usage."
        )

    # fit do melhor no outer-train completo
    best_est.fit(X_train, y_train)
    return best_est, best_params, float(best_score)





# ---------- CheckpointManager with file-lock for done_folds.json ----------
class CheckpointManager:
    def __init__(self, root_dir: Path):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fold_dir = self.root / "folds"
        self.fold_dir.mkdir(parents=True, exist_ok=True)
        self.done_path = self.root / "done_folds.json"

    def fold_path(self, fold: int) -> Path:
        p = self.fold_dir / f"outer_fold_{fold:02d}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _locked_read_done(self) -> set:
        if not self.done_path.exists():
            return set()
        import fcntl
        with open(self.done_path, "r", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                data = f.read().strip()
                return set(json.loads(data)) if data else set()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def load_done_folds(self):
        return self._locked_read_done()

    def mark_done(self, fold: int):
        import fcntl
        self.done_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.done_path, "a+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                data = f.read().strip()
                done = set(json.loads(data)) if data else set()
                done.add(int(fold))
                f.seek(0)
                f.truncate()
                f.write(json.dumps(sorted(done), indent=2))
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def save_fold_arrays(self, fold, train_idx, test_idx, y_test, y_pred_by_model, proba_by_model):
        fp = self.fold_path(fold)
        safe_save_npy(fp / "train_idx.npy", np.asarray(train_idx))
        safe_save_npy(fp / "test_idx.npy", np.asarray(test_idx))
        safe_save_npy(fp / "y_test.npy", np.asarray(y_test))
        for model, y_pred in y_pred_by_model.items():
            safe_save_npy(fp / f"y_pred_{sanitize_name(model)}.npy", np.asarray(y_pred))
        for model, P in proba_by_model.items():
            safe_save_npy(fp / f"proba_{sanitize_name(model)}.npy", np.asarray(P))

    def save_fold_json(self, fold, name, obj: dict):
        fp = self.fold_path(fold)
        safe_write_text(fp / name, json.dumps(obj, indent=2, default=json_default))

    def try_save_joblib(self, fold, model_name, obj, suffix, enabled):
        if not enabled or joblib is None:
            return
        fp = self.fold_path(fold) / "models"
        fp.mkdir(parents=True, exist_ok=True)
        out = fp / f"{sanitize_name(model_name)}_{suffix}.joblib"
        try:
            joblib.dump(obj, out)
        except Exception as e:
            safe_write_text(fp / f"{sanitize_name(model_name)}_{suffix}_SAVE_ERROR.txt", str(e))

def write_status(fold_dir: Path, payload: dict):
    safe_write_text(fold_dir / "status.json", json.dumps(payload, indent=2, default=json_default))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--elm", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--fold", type=int, required=True, help="Outer fold id (1..OUTER_SPLITS)")
    ap.add_argument("--outer_splits", type=int, default=5)
    ap.add_argument("--inner_splits", type=int, default=5)  # <-- INNER DEFAULT = 3
    ap.add_argument("--pearson_thresh", type=float, default=0.59)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--calibrate_probas", action="store_true")
    ap.add_argument("--cal_method", type=str, default="sigmoid")
    ap.add_argument("--cal_cv", type=int, default=3)
    ap.add_argument("--save_models", action="store_true")
    ap.add_argument("--n_jobs", type=int, default=0, help="0=auto(all allocated cpus)")
    args = ap.parse_args()

    fold = int(args.fold)
    OUTER_SPLITS = int(args.outer_splits)
    INNER_SPLITS = int(args.inner_splits)
    PEARSON_THRESH = float(args.pearson_thresh)
    RANDOM_STATE = int(args.random_state)

    if not (1 <= fold <= OUTER_SPLITS):
        raise ValueError(f"--fold must be in [1,{OUTER_SPLITS}]")

    n_jobs = args.n_jobs if args.n_jobs and args.n_jobs > 0 else TOTAL_CPUS

    # If user passes relative paths, treat them relative to this script dir.
    script_dir = Path(__file__).resolve().parent
    dataset_path = Path(args.dataset)
    elm_path = Path(args.elm)
    if not dataset_path.is_absolute():
        dataset_path = (script_dir / dataset_path).resolve()
    if not elm_path.is_absolute():
        elm_path = (script_dir / elm_path).resolve()

    art_root = Path(args.artifacts)
    if not art_root.is_absolute():
        art_root = (script_dir / art_root).resolve()

    ckpt = CheckpointManager(art_root)
    fold_dir = ckpt.fold_path(fold)

    slurm = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
    }

    t0 = time.time()
    write_status(fold_dir, {
        "fold": fold,
        "state": "RUNNING",
        "started_at_unix": t0,
        "slurm": slurm,
        "n_jobs_gridsearch": n_jobs,
        "thread_env": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]},
        "paths": {
            "script_dir": str(script_dir),
            "dataset_path": str(dataset_path),
            "elm_path": str(elm_path),
            "artifacts_root": str(art_root),
        }
    })

    try:
        # --- CRITICAL FOR ELM PICKLING ---
        # Ensure elm.py directory is importable and load module with name "elm"
        elm_dir = str(elm_path.parent.resolve())
        if elm_dir not in sys.path:
            sys.path.insert(0, elm_dir)

        # Import ELM as module name "elm" so joblib can pickle/unpickle it
        elm_mod = import_module_from_path("elm", str(elm_path))
        if not hasattr(elm_mod, "elm"):
            raise AttributeError("elm.py must define class 'elm'.")
        ELMCore = elm_mod.elm

        # Load dataset (fixed cleaning)
        dataset = pd.read_excel(dataset_path)
        if "Classe" not in dataset.columns:
            raise ValueError("Target column 'Classe' not found.")

        y_raw = dataset["Classe"].copy()
        X = dataset.copy()

        X = X.filter(regex=r"^(?!.*Keq)")
        cols_to_drop = ["Classe", "Composto", "ÃƒÂtomos"]
        X = X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors="ignore")

        for c in X.columns:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce")

        all_nan_cols = X.columns[X.isna().all()].tolist()
        if all_nan_cols:
            X = X.drop(columns=all_nan_cols)

        le_global = LabelEncoder()
        y = le_global.fit_transform(y_raw.values).astype(int)
        class_names = list(le_global.classes_)
        n_classes = len(class_names)

        # OPTIONAL: auto-shrink INNER splits if needed
        min_class_count = int(pd.Series(y).value_counts().min())
        if INNER_SPLITS > min_class_count:
            INNER_SPLITS = max(2, min_class_count)
            payload = json.loads((fold_dir / "status.json").read_text(encoding="utf-8"))
            payload["inner_splits_adjusted_to"] = INNER_SPLITS
            payload["note_inner_adjustment"] = f"Adjusted inner_splits due to min class count = {min_class_count}"
            write_status(fold_dir, payload)

        # Save dataset info once (race-safe)
        info_path = art_root / "dataset_info.json"
        if not info_path.exists():
            safe_write_text(info_path, json.dumps({
                "dataset_path": str(dataset_path),
                "n_samples": int(X.shape[0]),
                "n_features_after_fixed_cleaning": int(X.shape[1]),
                "feature_names_after_fixed_cleaning": list(X.columns),
                "class_names": class_names,
                "class_counts": pd.Series(y).value_counts().sort_index().to_dict(),
                "pearson_thresh": PEARSON_THRESH,
                "outer_splits": OUTER_SPLITS,
                "inner_splits": INNER_SPLITS,
                "random_state": RANDOM_STATE,
            }, indent=2, default=json_default))

        # Define models + grids
        pipe_rf  = make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE), use_scaler=False, pearson_thresh=PEARSON_THRESH)
        pipe_gb  = make_pipeline(GradientBoostingClassifier(random_state=RANDOM_STATE), use_scaler=False, pearson_thresh=PEARSON_THRESH)
        pipe_mlp = make_pipeline(MLPClassifier(random_state=RANDOM_STATE), use_scaler=True,  pearson_thresh=PEARSON_THRESH)

        # ELM pipeline template (NO GridSearchCV)
        pipe_elm_template = make_pipeline(
            ELMWrapper(ELMCore=ELMCore, random_state=RANDOM_STATE),
            use_scaler=True,
            pearson_thresh=PEARSON_THRESH
        )

        param_grid_rf = {
            "model__n_estimators": [50, 100, 200, 500],
            "model__max_depth": [5,6,7,8,9,10,11,12,13,14,15],
            "model__max_features": ["sqrt", "log2"],
            "model__max_leaf_nodes": [7, 9, 12, 14],
            "model__criterion": ["gini", "entropy", "log_loss"],
        }
        param_grid_gb = {
            "model__loss": ["log_loss"],
            "model__criterion": ["friedman_mse", "squared_error"],
            "model__max_features": ["sqrt", "log2"],
            "model__max_leaf_nodes": [7, 9, 12, 14],
            "model__n_estimators": [50, 100, 200, 500],
            "model__max_depth": [3, 5, 7],
        }
        param_grid_mlp = {
            "model__hidden_layer_sizes": [(30,), (36,), (40,), (30,30), (36,36), (40,40)],
            "model__activation": ["identity", "logistic", "tanh", "relu"],
            "model__solver": ["lbfgs", "sgd", "adam"],
            "model__alpha": [0.0001, 1e-16],
            "model__max_iter": [20000],
            "model__learning_rate_init": [0.001, 0.000001],
            "model__learning_rate": ["constant", "invscaling", "adaptive"],
        }
        param_grid_elm = [
            {
                "model__hidden_units": [3,6,9,12,15,18,20,22,25,32,36,40,50,75,100,150,200,400],
                "model__activation_function": ["sigmoid", "relu", "sin", "tanh", "leaky_relu"],
                "model__C": [0,1,2,3,4,5,6,7,8,9,10,13,17,25],
                "model__random_type": ["normal","uniform"],
                "model__treinador": ["no_re"],
            },
            {
                "model__hidden_units": [3,6,9,12,15,18,20,22,25,32,36,40,50,75,100,150,200,400],
                "model__activation_function": ["sigmoid", "relu", "sin", "tanh", "leaky_relu"],
                "model__C": [1,2,3,4,5,6,7,8,9,10,13,17,25],
                "model__random_type": ["normal","uniform"],
                "model__treinador": ["solution1","solution2"],
            }
        ]

        BASE_MODELS = {
            "RandomForest": (pipe_rf, param_grid_rf, True),
            "GradientBoosting": (pipe_gb, param_grid_gb, True),
            "MLP": (pipe_mlp, param_grid_mlp, True),
            # ELM handled separately
        }
        MODEL_LABELS = ["RandomForest", "GradientBoosting", "MLP", "ELM", "Ensemble"]
        MODEL_PROBA  = ["RandomForest", "GradientBoosting", "MLP", "Ensemble"]

        outer_cv = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

        # Pick this fold deterministically
        splits = list(outer_cv.split(X, y))
        train_idx, test_idx = splits[fold - 1]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Resume guard
        done = ckpt.load_done_folds()
        if fold in done:
            write_status(fold_dir, {
                "fold": fold, "state": "SKIPPED_ALREADY_DONE",
                "slurm": slurm, "n_jobs_gridsearch": n_jobs,
            })
            print(f"[RESUME] fold={fold} already in done_folds.json. Exiting.")
            return

        print(f"=== Running OUTER fold {fold}/{OUTER_SPLITS} | GridSearch n_jobs={n_jobs} | CPUs={TOTAL_CPUS} | inner_splits={INNER_SPLITS} ===")

        fitted = {}
        best_params = {}
        best_inner_score = {}
        feature_info = {}

        # RF/GB/MLP via GridSearchCV
        for name, (pipe, grid, has_proba) in BASE_MODELS.items():
            print(f"  -> Tuning {name} (GridSearchCV) ...")
            best_est, bp, bis = fit_best(pipe, grid, X_train, y_train, inner_cv, n_jobs=n_jobs)
            fitted[name] = best_est
            best_params[name] = bp
            best_inner_score[name] = bis
            feature_info[name] = get_pipeline_feature_info(best_est)

            if args.save_models and joblib is not None:
                ckpt.try_save_joblib(fold, name, best_est, "best_estimator", enabled=True)

        # ELM via manual tuning
        print("  -> Tuning ELM (manual inner CV, robust) ...")

        def _status_hook(extra):
            payload = json.loads((fold_dir / "status.json").read_text(encoding="utf-8"))
            payload.update(extra)
            write_status(fold_dir, payload)

        best_elm, bp_elm, bis_elm = tune_elm_manual(
            pipe_elm_template=pipe_elm_template,
            param_grid_elm=param_grid_elm,
            X_train=X_train,
            y_train=y_train,
            inner_cv=inner_cv,
            random_state=RANDOM_STATE,
            fold_dir=fold_dir,
            status_hook=_status_hook,
            n_jobs_elm=n_jobs,          # <-- usa todos CPUs alocados
            parallel_backend="loky"     # processos
        )

        fitted["ELM"] = best_elm
        best_params["ELM"] = bp_elm
        best_inner_score["ELM"] = bis_elm
        feature_info["ELM"] = get_pipeline_feature_info(best_elm)

        if args.save_models and joblib is not None:
            ckpt.try_save_joblib(fold, "ELM", best_elm, "best_estimator", enabled=True)

        preds, probas = {}, {}

        # Predict outer-test for all models
        for name in ["RandomForest", "GradientBoosting", "MLP", "ELM"]:
            y_pred = fitted[name].predict(X_test)
            preds[name] = np.asarray(y_pred, dtype=int)

            if name != "ELM":
                P, cal_obj = predict_proba_safe(
                    fitted[name], X_train, y_train, X_test, n_classes,
                    calibrate=bool(args.calibrate_probas),
                    cal_method=args.cal_method, cal_cv=args.cal_cv
                )
                probas[name] = P
                if args.save_models and args.calibrate_probas and (cal_obj is not None) and joblib is not None:
                    ckpt.try_save_joblib(fold, name, cal_obj, "calibrated_wrapper", enabled=True)

            acc = accuracy_score(y_test, preds[name])
            print(f"    {name}: acc={acc:.4f} | inner_best={best_inner_score[name]:.4f}")

        # Ensemble: (RF+GB+MLP+onehot(ELM))/4
        P_sum = probas["RandomForest"] + probas["GradientBoosting"] + probas["MLP"]
        elm_pred = preds["ELM"].astype(int)
        P_elm = np.zeros_like(P_sum)
        P_elm[np.arange(len(elm_pred)), elm_pred] = 1.0
        P_ens = (P_sum + P_elm) / 4.0
        ens_pred = np.argmax(P_ens, axis=1).astype(int)

        preds["Ensemble"] = ens_pred
        probas["Ensemble"] = P_ens

        print(f"    Ensemble: acc={accuracy_score(y_test, ens_pred):.4f}")

        y_pred_by_model = {m: preds[m] for m in MODEL_LABELS}
        proba_by_model  = {m: probas[m] for m in MODEL_PROBA}

        ckpt.save_fold_arrays(fold, train_idx, test_idx, y_test, y_pred_by_model, proba_by_model)
        ckpt.save_fold_json(fold, "best_params.json", best_params)
        ckpt.save_fold_json(fold, "feature_info.json", feature_info)
        ckpt.save_fold_json(fold, "fold_meta.json", {
            "fold": fold,
            "best_inner_score": best_inner_score,
            "ensemble_rule": "P_ens=(RF+GB+MLP+onehot(ELM))/4, equal weights",
            "calibrate_probas": bool(args.calibrate_probas),
            "cal_method": args.cal_method if args.calibrate_probas else None,
            "cal_cv": args.cal_cv if args.calibrate_probas else None,
            "gridsearch_n_jobs": n_jobs,
            "elm_tuning": "manual_inner_cv_with_skip_on_errors",
            "inner_splits_used": INNER_SPLITS,
        })

        ckpt.mark_done(fold)

        t1 = time.time()
        write_status(fold_dir, {
            "fold": fold,
            "state": "DONE",
            "started_at_unix": t0,
            "finished_at_unix": t1,
            "elapsed_sec": float(t1 - t0),
            "slurm": slurm,
            "n_jobs_gridsearch": n_jobs,
        })
        print(f"[OK] fold={fold} DONE. Artifacts: {fold_dir}")

    except Exception as e:
        tb = traceback.format_exc()
        t1 = time.time()
        write_status(fold_dir, {
            "fold": fold,
            "state": "ERROR",
            "started_at_unix": t0,
            "failed_at_unix": t1,
            "elapsed_sec": float(t1 - t0),
            "slurm": slurm,
            "n_jobs_gridsearch": n_jobs,
            "error": str(e),
            "traceback": tb,
        })
        print(tb)
        raise


if __name__ == "__main__":
    main()
