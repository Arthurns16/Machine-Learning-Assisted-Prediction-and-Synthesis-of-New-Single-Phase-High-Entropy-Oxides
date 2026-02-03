#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count most frequent selected features across OUTER folds, reading the artifacts
created by nestedcv_run_fold.py (feature_info.json per fold).

It reports:
- folds_present: in how many folds the feature appears (union over selected models per fold)
- occurrences: total appearances across fold × model (counts duplicates across models)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
import csv


DEFAULT_MODELS = ["RandomForest", "GradientBoosting", "MLP", "ELM"]


def iter_fold_dirs(artifacts_root: Path) -> list[Path]:
    folds_root = artifacts_root / "folds"
    if not folds_root.exists():
        raise FileNotFoundError(f"Could not find folds directory: {folds_root}")

    fold_dirs = sorted([p for p in folds_root.glob("outer_fold_*") if p.is_dir()])
    return fold_dirs


def load_feature_info(fold_dir: Path) -> dict:
    p = fold_dir / "feature_info.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    # sometimes it can come as tuple, etc.
    try:
        return list(x)
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--artifacts",
        type=str,
        default="./nestedcv_artifacts",
        help="Artifacts root used in nestedcv_run_fold.py (contains folds/outer_fold_*/feature_info.json).",
    )
    ap.add_argument(
        "--stage",
        type=str,
        default="pearson_kept_features",
        choices=[
            "pearson_kept_features",
            "pearson_dropped_features",
            "rm_const_kept_features",
            "rm_const_dropped_features",
        ],
        help="Which feature list to count from feature_info.json.",
    )
    ap.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model names to include (e.g., RandomForest,MLP).",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=50,
        help="How many top features to print.",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="",
        help="Optional output CSV path. Default: <artifacts>/feature_freq_<stage>.csv",
    )
    args = ap.parse_args()

    artifacts_root = Path(args.artifacts).resolve()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    stage = args.stage

    fold_dirs = iter_fold_dirs(artifacts_root)
    if not fold_dirs:
        raise RuntimeError(f"No fold directories found under: {artifacts_root / 'folds'}")

    # Count 1) presence per fold (union over models), and 2) occurrences across fold×model
    fold_presence = Counter()
    occurrences = Counter()
    models_present = defaultdict(set)

    n_folds_with_info = 0
    skipped = 0

    for fd in fold_dirs:
        info = load_feature_info(fd)
        if not info:
            skipped += 1
            continue

        n_folds_with_info += 1

        # union set per fold
        fold_union = set()

        for model in models:
            if model not in info:
                continue
            feats = safe_list(info[model].get(stage))
            if not feats:
                continue

            # occurrences: count per fold×model
            for feat in feats:
                occurrences[feat] += 1
                models_present[feat].add(model)

            # presence: union per fold
            fold_union.update(feats)

        for feat in fold_union:
            fold_presence[feat] += 1

    if n_folds_with_info == 0:
        raise RuntimeError(
            f"Found folds, but no feature_info.json could be read. "
            f"Expected: {artifacts_root}/folds/outer_fold_*/feature_info.json"
        )

    # Build rows
    rows = []
    for feat in set(fold_presence.keys()) | set(occurrences.keys()):
        fp = int(fold_presence.get(feat, 0))
        oc = int(occurrences.get(feat, 0))
        frac = fp / float(n_folds_with_info)
        mp = ",".join(sorted(models_present.get(feat, set())))
        rows.append(
            {
                "feature": feat,
                "folds_present": fp,
                "folds_fraction": f"{frac:.3f}",
                "occurrences": oc,
                "models_present": mp,
            }
        )

    # Sort: primary by folds_present desc, secondary by occurrences desc, tertiary by feature name
    rows_sorted = sorted(
        rows,
        key=lambda r: (-r["folds_present"], -r["occurrences"], r["feature"]),
    )

    # Output CSV
    out_csv = Path(args.out_csv) if args.out_csv else (artifacts_root / f"feature_freq_{stage}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["feature", "folds_present", "folds_fraction", "occurrences", "models_present"],
        )
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)

    # Print summary
    print(f"[OK] Artifacts root: {artifacts_root}")
    print(f"[OK] Stage: {stage}")
    print(f"[OK] Models included: {models}")
    print(f"[OK] Folds scanned: {len(fold_dirs)} | with feature_info.json: {n_folds_with_info} | skipped: {skipped}")
    print(f"[OK] CSV saved to: {out_csv}")
    print()
    print(f"Top-{args.top} features by folds_present (then occurrences):")
    for r in rows_sorted[: args.top]:
        print(
            f"- {r['feature']}: folds_present={r['folds_present']}/{n_folds_with_info} "
            f"({r['folds_fraction']}), occurrences={r['occurrences']}, models=[{r['models_present']}]"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)