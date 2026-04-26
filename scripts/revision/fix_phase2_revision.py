from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
RHO = 0.60
PI = 0.15


def compute_dece(scores: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(scores)
    if total == 0:
        return float("nan")

    dece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (scores >= lo) & (scores < hi)
        else:
            mask = (scores >= lo) & (scores <= hi)

        if mask.sum() == 0:
            continue

        bin_conf = scores[mask].mean()
        bin_acc = correct[mask].mean()
        dece += (mask.sum() / total) * abs(bin_conf - bin_acc)
    return float(dece)


def bootstrap_metrics(df: pd.DataFrame, gt_total: int, n_boot: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)

    scores = df["score"].to_numpy(dtype=float)
    correct = df["correct"].to_numpy(dtype=int)
    n = len(df)

    f1s = []
    deces = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = correct[idx]

        tp = int(sample.sum())
        fp = int((sample == 0).sum())
        fn = max(0, gt_total - tp)

        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)

        dece = compute_dece(scores[idx], sample.astype(float), n_bins=10)

        f1s.append(f1)
        deces.append(dece)

    f1_lo, f1_hi = np.percentile(f1s, [2.5, 97.5])
    dece_lo, dece_hi = np.percentile(deces, [2.5, 97.5])

    return float(f1_lo), float(f1_hi), float(dece_lo), float(dece_hi)


def build_model_outputs(model_name, matched_csv, gt_csv, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matched = pd.read_csv(matched_csv)
    gt = pd.read_csv(gt_csv)

    if "nominal_occlusion" not in gt.columns:
        raise ValueError(f"{gt_csv} must contain nominal_occlusion")

    gt_small = gt[["image_key", "nominal_occlusion"]].drop_duplicates().copy()
    gt_small["nominal_occlusion"] = gt_small["nominal_occlusion"].astype(float)

    df = matched.merge(gt_small, on="image_key", how="left")
    if df["nominal_occlusion"].isna().any():
        missing = df["nominal_occlusion"].isna().sum()
        raise ValueError(f"{model_name}: {missing} matched rows missing nominal_occlusion after merge")

    # ---------- Bootstrap CIs ----------
    gt_counts_full = gt.groupby("nominal_occlusion").size().to_dict()
    ci_rows = []

    for occ in LEVELS:
        sub = df[df["nominal_occlusion"] == occ].copy()
        gt_total = int(gt_counts_full.get(occ, 0))
        if len(sub) == 0 or gt_total == 0:
            continue

        f1_lo, f1_hi, dece_lo, dece_hi = bootstrap_metrics(sub, gt_total)
        ci_rows.append({
            "occlusion_level": occ,
            "f1_lo": f1_lo,
            "f1_hi": f1_hi,
            "dece_lo": dece_lo,
            "dece_hi": dece_hi,
            "n_predictions": len(sub),
            "num_ground_truth": gt_total,
        })

    ci_df = pd.DataFrame(ci_rows)
    ci_path = out_dir / f"{model_name}_bootstrap_cis_fixed.csv"
    ci_df.to_csv(ci_path, index=False)

    # ---------- Held-out DZ ----------
    # Split by GT image_key, not matched rows
    keys = sorted(gt_small["image_key"].unique().tolist())
    rng = np.random.default_rng(42)
    rng.shuffle(keys)
    mid = len(keys) // 2
    calib_keys = set(keys[:mid])
    eval_keys = set(keys[mid:])

    gt_small["split_id"] = gt_small["image_key"].map(lambda x: "calib" if x in calib_keys else "eval")
    df = df.merge(gt_small[["image_key", "split_id"]], on="image_key", how="left", suffixes=("", "_dup"))
    if "split_id_dup" in df.columns:
        df = df.drop(columns=["split_id_dup"])

    # Per-image calibration gap on clean split only
    clean_calib = df[(df["split_id"] == "calib") & (df["nominal_occlusion"] == 0.0)].copy()
    per_image_gap = (
        clean_calib.groupby("image_key")
        .apply(lambda x: abs(x["score"].mean() - x["correct"].mean()))
        .reset_index(name="image_gap")
    )

    if len(per_image_gap) == 0:
        raise ValueError(f"{model_name}: no clean calibration images found")

    # Build eval rows once
    gt_eval = gt_small[gt_small["split_id"] == "eval"].copy()
    gt_eval_counts = gt_eval.groupby("nominal_occlusion").size().to_dict()

    eval_rows = []
    eval_df = df[df["split_id"] == "eval"].copy()

    for occ in LEVELS:
        sub = eval_df[eval_df["nominal_occlusion"] == occ].copy()
        gt_total = int(gt_eval_counts.get(occ, 0))

        if len(sub) == 0 or gt_total == 0:
            continue

        tp = int(sub["correct"].sum())
        fp = int((sub["correct"] == 0).sum())
        fn = max(0, gt_total - tp)

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        dece = compute_dece(
            sub["score"].to_numpy(dtype=float),
            sub["correct"].to_numpy(dtype=float),
            n_bins=10,
        )

        eval_rows.append({
            "model": model_name,
            "occlusion_level": occ,
            "precision": precision,
            "recall": recall,
            "dece": dece,
            "n_predictions": len(sub),
            "num_ground_truth": gt_total,
        })

    eval_rows_df = pd.DataFrame(eval_rows)

    # Sensitivity over percentiles
    sens_rows = []
    detailed_rows = []

    for pct in [50, 60, 70, 75, 80, 85, 90]:
        delta_m = float(np.percentile(per_image_gap["image_gap"].to_numpy(), pct))

        flagged = []
        for _, row in eval_rows_df.iterrows():
            dz = (
                (row["recall"] >= RHO) and
                (row["precision"] >= PI) and
                (row["dece"] >= delta_m)
            )
            detailed_rows.append({
                "model": model_name,
                "percentile": pct,
                "delta_m": delta_m,
                **row.to_dict(),
                "dangerous_zone": dz,
            })
            if dz:
                flagged.append(row["occlusion_level"])

        sens_rows.append({
            "model": model_name,
            "percentile": pct,
            "delta_m": delta_m,
            "dz_levels": flagged,
        })

    sens_df = pd.DataFrame(sens_rows)
    detail_df = pd.DataFrame(detailed_rows)

    sens_path = out_dir / f"{model_name}_heldout_dz_sensitivity_fixed.csv"
    detail_path = out_dir / f"{model_name}_heldout_dz_eval_rows_fixed.csv"

    sens_df.to_csv(sens_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    return ci_path, sens_path, detail_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "model_name": "YOLOv8m",
            "matched_csv": "/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_overlap_scale2000/yolo_overlap_matched_thr040.csv",
            "gt_csv": "/home/karthikadari/occdet-calib/data/processed/occdet_v1_scale2000/overlap_gt_eval.csv",
        },
        {
            "model_name": "FCOS-R50",
            "matched_csv": "/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_overlap_scale1000/fcos_overlap_matched_thr020.csv",
            "gt_csv": "/home/karthikadari/occdet-calib/data/processed/occdet_v1_scale1000/overlap_gt_eval.csv",
        },
        {
            "model_name": "Deformable-DETR-R50",
            "matched_csv": "/home/karthikadari/occdet-calib/outputs/evals/occdet_v1_overlap_scale1000/deformable_detr_overlap_matched_thr030.csv",
            "gt_csv": "/home/karthikadari/occdet-calib/data/processed/occdet_v1_scale1000/overlap_gt_eval.csv",
        },
    ]

    summary = []
    for cfg in configs:
        ci_path, sens_path, detail_path = build_model_outputs(
            cfg["model_name"], cfg["matched_csv"], cfg["gt_csv"], out_dir
        )
        summary.append({
            "model": cfg["model_name"],
            "bootstrap_ci_csv": str(ci_path),
            "heldout_sensitivity_csv": str(sens_path),
            "heldout_eval_rows_csv": str(detail_path),
        })

    pd.DataFrame(summary).to_csv(out_dir / "phase2_fixed_summary.csv", index=False)
    print("Saved fixed Phase 2 outputs to:", out_dir)


if __name__ == "__main__":
    main()
