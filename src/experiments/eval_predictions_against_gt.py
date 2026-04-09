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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detection predictions against GT.")
    parser.add_argument("--pred_path", type=str, required=True, help="Prediction CSV/Parquet")
    parser.add_argument("--gt_path", type=str, required=True, help="Ground-truth CSV/Parquet")
    parser.add_argument("--matched_out", type=str, required=True, help="Matched predictions output")
    parser.add_argument("--per_class_out", type=str, required=True, help="Per-class summary output")
    parser.add_argument("--per_image_out", type=str, required=True, help="Per-image summary output")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--pred_image_path_col", type=str, default="image_path")
    parser.add_argument("--gt_image_key_col", type=str, default="image_key")
    parser.add_argument("--classwise", action="store_true", help="Classwise matching")
    args = parser.parse_args()

    pred_df = load_table(args.pred_path)
    gt_df = load_table(args.gt_path)

    # predictions usually have full paths; convert to image filename key
    pred_df = add_image_key_from_path(
        pred_df,
        image_path_col=args.pred_image_path_col,
        output_col="image_key",
    )

    # GT is expected to already have image_key
    if args.gt_image_key_col != "image_key":
        gt_df = gt_df.rename(columns={args.gt_image_key_col: "image_key"})

    matched_pred_df = match_predictions_to_ground_truth(
        pred_df=pred_df,
        gt_df=gt_df,
        image_key_col="image_key",
        class_id_col="class_id",
        score_col="score",
        iou_threshold=args.iou_threshold,
        classwise=args.classwise,
    )

    overall = summarize_detection_metrics(matched_pred_df, gt_df)
    per_class = summarize_per_class_detection_metrics(matched_pred_df, gt_df)
    per_image = summarize_per_image_detection_metrics(matched_pred_df, gt_df)

    save_table(matched_pred_df, args.matched_out)
    save_table(per_class, args.per_class_out)
    save_table(per_image, args.per_image_out)

    print("OVERALL SUMMARY")
    for k, v in overall.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
