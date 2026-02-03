#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# =========================
# MAX resources knobs
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = ""
NCPU = str(os.cpu_count() or 1)
os.environ["OMP_NUM_THREADS"] = NCPU
os.environ["OPENBLAS_NUM_THREADS"] = NCPU
os.environ["MKL_NUM_THREADS"] = NCPU
os.environ["VECLIB_MAXIMUM_THREADS"] = NCPU
os.environ["NUMEXPR_NUM_THREADS"] = NCPU


# =========================
# LABEL TRANSLATION (plots only)
# =========================
_SUFFIX_MAP = {
    "minimo": "minimum",
    "maximo": "maximum",
    "soma": "sum",
    "desvio": "std",
    "media": "mean",
    "mediana": "median",
    "variancia": "variance",
}

_CLASS_MAP = {
    "fluorita": "Fluorite",
    "perovskita": "Perovskite",
    "pirocloro": "Pyrochlore",
    "rocksalt": "RockSalt",
    "spinel": "Spinel",
    "mixed": "Mixed",
}


def translate_feature_label(name: str) -> str:
    s = str(name)
    s = re.sub(r"\s*_\s*", "_", s).strip()  # ex: "atomic_en_allen _soma" -> "atomic_en_allen_soma"
    m = re.search(
        r"^(.*)_(minimo|maximo|soma|desvio|media|mediana|variancia)$",
        s,
        flags=re.IGNORECASE,
    )
    if not m:
        return s
    base, suf = m.group(1), m.group(2).lower()
    return f"{base}_{_SUFFIX_MAP.get(suf, suf)}"


def translate_class_label(name: str) -> str:
    s = str(name).strip()
    return _CLASS_MAP.get(s.lower(), s)


def pick_class_index(class_names_raw: list[str], preferred: str) -> int:
    p = (preferred or "").strip().lower()
    if not p:
        return 0
    # tenta direto (PT/EN)
    for i, c in enumerate(class_names_raw):
        if str(c).strip().lower() == p:
            return i
    # tenta via tradução
    p2 = translate_class_label(preferred).lower()
    for i, c in enumerate(class_names_raw):
        if translate_class_label(c).lower() == p2:
            return i
    return 0


def _json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_dir", type=str, default=".", help="Diretório do bundle (default: .)")
    ap.add_argument("--input", type=str, default="Input_Total.xlsx", help="Entrada (default: Input_Total.xlsx)")
    ap.add_argument("--output_dir", type=str, default=".", help="Saída (default: .)")

    ap.add_argument(
        "--model",
        type=str,
        default="Ensemble",
        choices=["Ensemble", "RandomForest", "GradientBoosting", "MLP", "ELM"],
        help="Qual modelo explicar (default: Ensemble).",
    )
    ap.add_argument(
        "--class_name",
        type=str,
        default="Fluorite",
        help="Classe do layered_violin (aceita PT/EN; default: Fluorite).",
    )
    ap.add_argument(
        "--disable_calibration",
        action="store_true",
        help="Se setado, força usar RF/GB/MLP não calibrados (default: calibração ON se existir no bundle).",
    )
    ap.add_argument("--max_display", type=int, default=30, help="max_display nos plots (default: 30)")
    ap.add_argument("--dpi", type=int, default=300, help="DPI dos PNGs (default: 300)")

    # NOVO: salvar artefatos para replot sem recalcular SHAP
    ap.add_argument(
        "--save_artifacts",
        action="store_true",
        help="Se setado, salva X, probabilidades, shap_values, expected_value e metadados (replot no futuro sem recalcular).",
    )
    ap.add_argument(
        "--artifacts_format",
        type=str,
        default="npz",
        choices=["npz", "npy"],
        help="Formato para salvar SHAP values: npz (compacto) ou npy (um arquivo por classe).",
    )

    args = ap.parse_args()

    bundle_dir = Path(args.bundle_dir)
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bundle_dir.exists():
        raise FileNotFoundError(bundle_dir)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    import run_inference_improved as inf

    # Necessário para unpickle do elm_final.joblib (foi salvo como __main__.ELMWrapper)
    sys.modules["__main__"].ELMWrapper = inf.ELMWrapper  # type: ignore[attr-defined]

    bundle = inf.load_bundle(bundle_dir=bundle_dir, disable_calibration=args.disable_calibration)
    use_cal = bool(bundle["use_cal"])

    # Classes (raw) e classes traduzidas só para exibição
    class_names_raw = [str(c) for c in bundle["classes"]]
    class_names_plot = [translate_class_label(c) for c in class_names_raw]

    # Preprocess idêntico ao pipeline de inferência
    df = pd.read_excel(input_path)
    X_raw, X_scaled, diag = inf.preprocess_with_diagnostics(df, bundle["pre"])
    feature_names_raw = list(diag["features_used"])

    # X para SHAP (raw, na ordem final)
    X = pd.DataFrame(X_raw, columns=feature_names_raw)

    # Apenas para plot: traduz sufixos e limpa underscores
    feature_names_plot = [translate_feature_label(f) for f in feature_names_raw]
    X_plot = X.copy()
    X_plot.columns = feature_names_plot

    # Classe do violin (aceita PT/EN)
    class_idx = pick_class_index(class_names_raw, args.class_name)
    class_label_plot = class_names_plot[class_idx] if class_names_plot else str(class_idx)

    # -------- Probabilidades (replicando predict_all do run_inference_improved) --------
    def proba_rf(X_in: np.ndarray) -> np.ndarray:
        Xn = np.asarray(X_in, dtype=float)
        return bundle["rf_cal"].predict_proba(Xn) if use_cal else bundle["rf"].predict_proba(Xn)

    def proba_gb(X_in: np.ndarray) -> np.ndarray:
        Xn = np.asarray(X_in, dtype=float)
        return bundle["gb_cal"].predict_proba(Xn) if use_cal else bundle["gb"].predict_proba(Xn)

    def proba_mlp(X_in: np.ndarray) -> np.ndarray:
        Xn = np.asarray(X_in, dtype=float)
        mu = np.array(bundle["pre"]["scaler_mean"], dtype=float)
        sg = np.array(bundle["pre"]["scaler_scale"], dtype=float)
        Xs = (Xn - mu) / sg
        return bundle["mlp_cal"].predict_proba(Xs) if use_cal else bundle["mlp"].predict_proba(Xs)

    def proba_elm(X_in: np.ndarray) -> np.ndarray:
        Xn = np.asarray(X_in, dtype=float)
        mu = np.array(bundle["pre"]["scaler_mean"], dtype=float)
        sg = np.array(bundle["pre"]["scaler_scale"], dtype=float)
        Xs = (Xn - mu) / sg
        y = np.asarray(bundle["elm"].predict(Xs)).astype(int)
        k = len(class_names_raw)
        P = np.zeros((len(y), k), dtype=float)
        P[np.arange(len(y)), y] = 1.0  # one-hot (ELM labels only)
        return P

    def proba_ensemble(X_in: np.ndarray) -> np.ndarray:
        # média RF + GB + MLP + ELM(one-hot)
        return (proba_rf(X_in) + proba_gb(X_in) + proba_mlp(X_in) + proba_elm(X_in)) / 4.0

    if args.model == "Ensemble":
        f = proba_ensemble
    elif args.model == "RandomForest":
        f = proba_rf
    elif args.model == "GradientBoosting":
        f = proba_gb
    elif args.model == "MLP":
        f = proba_mlp
    elif args.model == "ELM":
        f = proba_elm
    else:
        raise ValueError(args.model)

    # =========================
    # SHAP: KernelExplainer para TODOS (calibração ON)
    # Background = dataset inteiro (MAX)
    # =========================
    background = X.values
    explainer = shap.KernelExplainer(f, background)
    shap_values = explainer.shap_values(X.values, nsamples="auto")

    # garante formato list[classes] com (n, f)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = [shap_values[:, :, k] for k in range(shap_values.shape[2])]
    if not isinstance(shap_values, list):
        raise RuntimeError(f"Unexpected shap_values type/shape: {type(shap_values)}")

    # =========================
    # SAVE ARTIFACTS (para replot no futuro sem recalcular)
    # =========================
    prefix = f"shap_kernel_{args.model}"
    if args.save_artifacts:
        art_dir = out_dir / f"{prefix}_artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)

        # 1) Metadados completos (para reconstruir qualquer plot)
        meta = {
            "model": args.model,
            "use_calibration": use_cal,
            "disable_calibration_flag": bool(args.disable_calibration),
            "input_file": str(input_path),
            "bundle_dir": str(bundle_dir),
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "class_names_raw": class_names_raw,
            "class_names_plot": class_names_plot,
            "feature_names_raw": feature_names_raw,
            "feature_names_plot": feature_names_plot,
            "class_idx_for_violin": int(class_idx),
            "class_name_arg": args.class_name,
            "class_label_plot": class_label_plot,
            "max_display_used": int(args.max_display),
            "kernel_nsamples": "auto",
            "background": "ALL_DATASET",
            "notes": (
                "Artifacts include X_raw/X_plot, predict_proba outputs, SHAP values per class, "
                "expected_value, and metadata. You can replot later (top-5, new colors, etc.) without recomputing SHAP."
            ),
        }
        _json_dump(meta, art_dir / "meta.json")

        # 2) Dados usados (raw + plot labels) em formatos úteis
        #    (Parquet é rápido e compacto; CSV é universal)
        X.to_parquet(art_dir / "X_raw.parquet", index=False)
        X_plot.to_parquet(art_dir / "X_plot.parquet", index=False)
        X.to_csv(art_dir / "X_raw.csv", index=False)
        X_plot.to_csv(art_dir / "X_plot.csv", index=False)

        # 3) Probabilidades do modelo/ensemble no dataset (para qualquer plot custom depois)
        P = f(X.values)  # (n_samples, n_classes)
        np.save(art_dir / "proba.npy", P)

        # 4) expected_value do KernelExplainer (baseline)
        #    (pode ser escalar, lista ou array por classe; salvamos como object)
        ev = explainer.expected_value
        np.save(art_dir / "expected_value.npy", np.asarray(ev, dtype=object), allow_pickle=True)

        # 5) SHAP values (principal artefato)
        #    Opção A (default): um .npz compacto com todas as classes
        #    Opção B: um .npy por classe
        if args.artifacts_format == "npz":
            to_save = {f"class_{k}": sv for k, sv in enumerate(shap_values)}
            np.savez_compressed(art_dir / "shap_values_by_class.npz", **to_save)
        else:
            for k, sv in enumerate(shap_values):
                np.save(art_dir / f"shap_values_class_{k}.npy", sv)

        # 6) Também salvamos um “ranking global” pronto (mean |SHAP|) por classe e macro-média
        #    Útil para replot top-5 sem recalcular
        mean_abs_by_class = []
        for k, sv in enumerate(shap_values):
            mean_abs = np.mean(np.abs(sv), axis=0)  # (n_features,)
            mean_abs_by_class.append(mean_abs)

        mean_abs_by_class = np.vstack(mean_abs_by_class)  # (n_classes, n_features)
        macro_mean_abs = mean_abs_by_class.mean(axis=0)    # (n_features,)

        np.save(art_dir / "mean_abs_shap_by_class.npy", mean_abs_by_class)
        np.save(art_dir / "macro_mean_abs_shap.npy", macro_mean_abs)

        # ranking (índices das features ordenadas por importância macro)
        rank_idx = np.argsort(-macro_mean_abs)
        np.save(art_dir / "feature_rank_idx_macro.npy", rank_idx)

    # =========================
    # Plots (bar + layered_violin + combinado)
    # =========================
    dpi = int(args.dpi)
    bar_path = out_dir / f"{prefix}_bar.png"
    violin_path = out_dir / f"{prefix}_violin_{class_label_plot}.png"
    combined_path = out_dir / f"{prefix}_combined_{class_label_plot}.png"

    # (1) Bar global
    plt.figure(figsize=(12, 6))
    shap.summary_plot(
        shap_values,
        X_plot,
        plot_type="bar",
        class_names=class_names_plot,
        show=False,
        max_display=int(args.max_display),
    )
    ax = plt.gca()
    ax.set_xlabel("Mean |SHAP value| (impact on model output)", fontsize=14)
    plt.tight_layout()
    plt.gcf().savefig(bar_path, dpi=dpi, bbox_inches="tight")
    plt.clf()

    # (2) Violin para classe escolhida
    plt.figure(figsize=(8, 10))
    sv_cls = shap_values[class_idx] if class_idx < len(shap_values) else shap_values[0]
    shap.summary_plot(
        sv_cls,
        X_plot,
        plot_type="layered_violin",
        show=False,
        max_display=int(args.max_display),
    )
    plt.tight_layout()
    plt.gcf().savefig(violin_path, dpi=dpi, bbox_inches="tight")
    plt.clf()

    # (3) Combina PNGs
    bar_img = mpimg.imread(bar_path)
    violin_img = mpimg.imread(violin_path)
    h_bar, w_bar, _ = bar_img.shape
    h_violin, w_violin, _ = violin_img.shape

    w_px = max(w_bar, w_violin)
    h_px = h_bar + h_violin

    fig_w = w_px / dpi
    fig_h = h_px / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.suptitle(
        f"Analysis of Features Using the SHAP Framework ({args.model} | class={class_label_plot})",
        fontsize=14,
        y=1.03,
    )

    ax1 = fig.add_axes([0, h_violin / h_px, 1, h_bar / h_px])
    ax1.imshow(bar_img)
    ax1.axis("off")
    ax1.text(
        0.01, 0.98, "a)",
        transform=ax1.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )

    ax2 = fig.add_axes([0, 0, 1, h_violin / h_px])
    ax2.imshow(violin_img)
    ax2.axis("off")
    ax2.text(
        0.01, 0.98, "b)",
        transform=ax2.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )

    fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print("\n=== SHAP (KernelExplainer, calibration ON if available) ===")
    print("Model:    ", args.model)
    print("Class:    ", class_label_plot)
    print("Bar:      ", bar_path.resolve())
    print("Violin:   ", violin_path.resolve())
    print("Combined: ", combined_path.resolve())
    if args.save_artifacts:
        print("Artifacts:", (out_dir / f"{prefix}_artifacts").resolve())
        print("  - meta.json")
        print("  - X_raw/X_plot (parquet + csv)")
        print("  - proba.npy")
        print("  - expected_value.npy")
        if args.artifacts_format == "npz":
            print("  - shap_values_by_class.npz")
        else:
            print("  - shap_values_class_*.npy")
        print("  - mean_abs_shap_by_class.npy")
        print("  - macro_mean_abs_shap.npy")
        print("  - feature_rank_idx_macro.npy")
    print("Done.")


if __name__ == "__main__":
    main()