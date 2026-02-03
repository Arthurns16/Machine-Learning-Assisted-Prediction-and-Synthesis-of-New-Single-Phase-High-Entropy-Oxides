#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def _load_meta(art_dir: Path) -> dict:
    meta_path = art_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_shap_values(art_dir: Path) -> list[np.ndarray]:
    npz_path = art_dir / "shap_values_by_class.npz"
    if npz_path.exists():
        z = np.load(npz_path)
        # garante ordem class_0, class_1, ...
        keys = sorted(z.files, key=lambda s: int(s.split("_")[1]))
        return [z[k] for k in keys]

    # fallback: múltiplos npy
    vals = []
    k = 0
    while True:
        p = art_dir / f"shap_values_class_{k}.npy"
        if not p.exists():
            break
        vals.append(np.load(p))
        k += 1
    if not vals:
        raise FileNotFoundError(f"No shap values found in {art_dir}")
    return vals


def _select_topk(shap_values: list[np.ndarray], X_plot: pd.DataFrame, topk: int) -> tuple[list[np.ndarray], pd.DataFrame]:
    """
    (LEGACY) Seleciona top-k features usando macro-mean(|SHAP|).
    Mantido por compatibilidade, mas NÃO é mais usado no fluxo principal.
    """
    if topk <= 0 or topk >= X_plot.shape[1]:
        return shap_values, X_plot

    macro_path = art_dir_global / "macro_mean_abs_shap.npy"
    if macro_path.exists():
        macro = np.load(macro_path)
        idx = np.argsort(-macro)[:topk]
    else:
        mean_abs_by_class = [np.mean(np.abs(sv), axis=0) for sv in shap_values]
        macro = np.mean(np.vstack(mean_abs_by_class), axis=0)
        idx = np.argsort(-macro)[:topk]

    Xk = X_plot.iloc[:, idx].copy()
    svk = [sv[:, idx] for sv in shap_values]
    return svk, Xk


def _norm_label(s: str) -> str:
    """Normaliza rótulos para comparação (remove espaços e põe em minúsculo)."""
    return re.sub(r"\s+", "", str(s).strip().lower())


def _pretty_class_name(name: str) -> str:
    """Padroniza nomes de classes para exibição."""
    s = str(name).strip()
    # Trata variações: RockSalt / Rocksalt / Rock Salt / Rocksalt  ...
    if _norm_label(s) == "rocksalt":
        return "Rock Salt"
    return s


def _pretty_feature_name(name: str) -> str:
    """Abrevia sufixos de features: *_maximum -> *_max; *_minimum -> *_min."""
    s = str(name).strip()
    # cobre typo comum "maximun" também
    s = re.sub(r"_(maximum|maximun)$", "_max", s, flags=re.IGNORECASE)
    s = re.sub(r"_(minimum)$", "_min", s, flags=re.IGNORECASE)
    return s


def _pick_class_index(class_names_raw: list[str], class_names_plot: list[str], preferred: str) -> int:
    p = _norm_label(preferred or "")
    if not p:
        return 0
    for i, c in enumerate(class_names_raw):
        if _norm_label(c) == p:
            return i
    for i, c in enumerate(class_names_plot):
        if _norm_label(c) == p:
            return i
    return 0


def _apply_style(style: str) -> None:
    """
    Você pode estender isso pra customizações visuais futuras.
    (Não mexo em cores aqui para não “forçar” estilo.)
    """
    if style == "default":
        plt.rcdefaults()
        return
    if style == "clean":
        plt.rcdefaults()
        plt.rcParams.update({
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })
        return
    if style == "dark":
        plt.style.use("dark_background")
        return
    raise ValueError(f"Unknown style: {style}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Ensemble",
                    choices=["Ensemble", "RandomForest", "GradientBoosting", "MLP", "ELM"],
                    help="Qual modelo (default: Ensemble).")
    ap.add_argument("--output_dir", type=str, default=".", help="Saída (default: .)")
    ap.add_argument("--class_name", type=str, default="", help="Classe do violin (PT/EN). Default: usa a salva no meta.")
    ap.add_argument("--topk", type=int, default=0, help="(DEPRECATED) Alias para --max_display quando --max_display não é fornecido.")
    ap.add_argument("--max_display", type=int, default=None,
                    help="max_display nos plots (default: 5; se --topk>0 e --max_display não for fornecido, usa topk)")
    ap.add_argument("--dpi", type=int, default=300, help="DPI dos PNGs (default: 300)")
    ap.add_argument("--style", type=str, default="default", choices=["default", "clean", "dark"],
                    help="Estilo matplotlib (default/clean/dark).")
    args = ap.parse_args()

    # --topk é DEPRECATED: use --max_display. Aqui tratamos --topk como alias de --max_display
    user_provided_max_display = args.max_display is not None
    if args.max_display is None:
        args.max_display = int(args.topk) if (args.topk and args.topk > 0) else 5
    elif args.topk and args.topk > 0:
        print("[WARN] --topk é DEPRECATED e será ignorado porque --max_display foi fornecido.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global art_dir_global
    art_dir_global = Path(f"shap_kernel_{args.model}_artifacts")
    if not art_dir_global.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {art_dir_global}")

    print("\n=== RE-PLOT from saved artifacts (no SHAP recompute) ===")
    print("Artifacts:", art_dir_global.resolve())

    meta = _load_meta(art_dir_global)

    # carrega X_plot
    xplot_parquet = art_dir_global / "X_plot.parquet"
    xplot_csv = art_dir_global / "X_plot.csv"
    if xplot_parquet.exists():
        X_plot = pd.read_parquet(xplot_parquet)
    elif xplot_csv.exists():
        X_plot = pd.read_csv(xplot_csv)
    else:
        raise FileNotFoundError("X_plot not found (parquet/csv).")

    shap_values = _load_shap_values(art_dir_global)

    # Padroniza nomes de features para exibição
    X_plot = X_plot.copy()
    X_plot.columns = [_pretty_feature_name(c) for c in X_plot.columns]

    # Padroniza nomes de classes para exibição
    class_names_raw = [str(c).strip() for c in meta["class_names_raw"]]
    class_names_plot = [_pretty_class_name(c) for c in meta["class_names_plot"]]

    # define classe do violin
    if args.class_name.strip():
        class_idx = _pick_class_index(class_names_raw, class_names_plot, args.class_name)
    else:
        class_idx = int(meta.get("class_idx_for_violin", 0))

    if class_names_plot:
        class_idx = max(0, min(class_idx, len(class_names_plot) - 1))
        class_label_plot = class_names_plot[class_idx]
    else:
        class_label_plot = str(class_idx)

    # Sem pré-filtragem por top-k: usamos apenas max_display para limitar o que aparece nos plots
    sv_use, X_use = shap_values, X_plot

    _apply_style(args.style)

    dpi = int(args.dpi)
    prefix = f"replot_shap_kernel_{args.model}"
    suffix = ""
    bar_path = out_dir / f"{prefix}_bar{suffix}.png"
    violin_path = out_dir / f"{prefix}_violin_{class_label_plot}{suffix}.png"
    combined_path = out_dir / f"{prefix}_combined_{class_label_plot}{suffix}.png"

    # (1) Bar global
    plt.figure(figsize=(12, 6))
    shap.summary_plot(
        sv_use,
        X_use,
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

    # (2) Layered violin
    plt.figure(figsize=(8, 10))
    sv_cls = sv_use[class_idx] if class_idx < len(sv_use) else sv_use[0]
    shap.summary_plot(
        sv_cls,
        X_use,
        plot_type="layered_violin",
        show=False,
        max_display=int(args.max_display),
    )
    plt.tight_layout()
    plt.gcf().savefig(violin_path, dpi=dpi, bbox_inches="tight")
    plt.clf()

    # (3) Combined
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
    ax1.text(0.01, 0.98, "a)", transform=ax1.transAxes, fontsize=16, fontweight="bold", va="top")

    ax2 = fig.add_axes([0, 0, 1, h_violin / h_px])
    ax2.imshow(violin_img)
    ax2.axis("off")
    ax2.text(0.01, 0.98, "b)", transform=ax2.transAxes, fontsize=16, fontweight="bold", va="top")

    fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print("Model:    ", args.model)
    print("Class:    ", class_label_plot)
    print("max_display:", int(args.max_display))
    print("Bar:      ", bar_path.resolve())
    print("Violin:   ", violin_path.resolve())
    print("Combined: ", combined_path.resolve())
    print("Done.\n")


if __name__ == "__main__":
    main()
