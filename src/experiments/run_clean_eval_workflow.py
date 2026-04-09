from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.metrics.matching import (
    add_image_key_from_path,
    match_predictions_to_ground_truth,
)
from src.metrics.map_eval import (
    summarize_detection_metrics,
    summarize_per_class_detection_metrics,
    summarize_per_image_detection_metrics,
)
from src.metrics.reliability import summarize_reliability
from src.metrics.monotonicity import compute_monotonicity_from_dataframe


def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)

    raise ValueError(f"Unsupported file type: {p.suffix}")


def save_table(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.suffix.lower() == ".csv":
        df.to_csv(p, index=False)
    elif p.suffix.lower() == ".parquet":
        df.to_parquet(p, index=False)
    else:
        raise ValueError(f"Unsupported output file type: {p.suffix}")


def normalize_class_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clean evaluation workflow.")
    parser.add_argument("--pred_path", type=str, required=True, help="Predictions CSV/Parquet")
    parser.add_argument("--gt_path", type=str, required=True, help="Ground-truth CSV/Parquet")
    parser.add_argument("--matched_out", type=str, required=True, help="Matched predictions output")
    parser.add_argument("--per_class_out", type=str, required=True, help="Per-class summary output")
    parser.add_argument("--per_image_out", type=str, required=True, help="Per-image summary output")
    parser.add_argument("--reliability_bins_out", type=str, required=True, help="Reliability bins output")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--n_bins", type=int, default=15)
    parser.add_argument("--pred_image_path_col", type=str, default="image_path")
    parser.add_argument("--gt_image_key_col", type=str, default="image_key")
    parser.add_argument("--classwise", action="store_true", help="Use classwise matching")
    args = parser.parse_args()

    pred_df = load_table(args.pred_path)
    gt_df = load_table(args.gt_path)

    pred_df = add_image_key_from_path(
        pred_df,
        image_path_col=args.pred_image_path_col,
        output_col="image_key",
    )

    if args.gt_image_key_col != "image_key":
        gt_df = gt_df.rename(columns={args.gt_image_key_col: "image_key"})

    # Use normalized class names for cross-model / COCO consistency.
    if "class_name" in pred_df.columns and "class_name" in gt_df.columns:
        pred_df["eval_class"] = normalize_class_series(pred_df["class_name"])
        gt_df["eval_class"] = normalize_class_series(gt_df["class_name"])
        class_col = "eval_class"
    else:
        class_col = "class_id"

    matched_pred_df = match_predictions_to_ground_truth(
        pred_df=pred_df,
        gt_df=gt_df,
        image_key_col="image_key",
        class_id_col=class_col,
        score_col="score",
        iou_threshold=args.iou_threshold,
        classwise=args.classwise,
    )

    overall = summarize_detection_metrics(matched_pred_df, gt_df)
    per_class = summarize_per_class_detection_metrics(
        matched_pred_df,
        gt_df,
        class_id_col=class_col,
    )
    per_image = summarize_per_image_detection_metrics(matched_pred_df, gt_df)

    reliability_summary = summarize_reliability(
        matched_pred_df,
        confidence_col="score",
        correctness_col="correct",
        n_bins=args.n_bins,
    )
    bins_df = reliability_summary["bins_df"]

    monotonicity_summary = compute_monotonicity_from_dataframe(
        matched_pred_df,
        confidence_col="score",
        correctness_col="correct",
        n_bins=args.n_bins,
    )

    save_table(matched_pred_df, args.matched_out)
    save_table(per_class, args.per_class_out)
    save_table(per_image, args.per_image_out)
    save_table(bins_df, args.reliability_bins_out)

    print("OVERALL SUMMARY")
    for k, v in overall.items():
        print(f"{k}: {v}")

    print("\nCALIBRATION SUMMARY")
    print(f"dece: {reliability_summary['dece']}")
    print(f"total_count: {reliability_summary['total_count']}")

    print("\nMONOTONICITY SUMMARY")
    print(f"monotonic: {monotonicity_summary['monotonic']}")
    print(f"inversion_count: {monotonicity_summary['inversion_count']}")
    print(f"inversion_pairs: {monotonicity_summary['inversion_pairs']}")


if __name__ == "__main__":
    main()
