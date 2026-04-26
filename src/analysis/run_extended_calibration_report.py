from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.metrics.calibration_metrics_extended import (
    classwise_calibration_table,
    laece_from_classwise,
    summarize_calibration_metrics,
)


def maybe_add_occlusion(df: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    if "image_key" not in df.columns or "image_key" not in gt.columns:
        return df

    keep_cols = [c for c in ["image_key", "nominal_occlusion", "estimated_occlusion"] if c in gt.columns]
    if not keep_cols:
        return df

    gt_small = gt[keep_cols].drop_duplicates("image_key")
    return df.merge(gt_small, on="image_key", how="left")


def run_for_score_column(
    df: pd.DataFrame,
    score_col: str,
    out_dir: Path,
    prefix: str,
    class_col: str = "class_name",
):
    metrics = summarize_calibration_metrics(df, confidence_col=score_col, correctness_col="correct", n_bins=15)
    overall_df = pd.DataFrame([metrics])

    classwise_df = classwise_calibration_table(
        df,
        class_col=class_col,
        confidence_col=score_col,
        correctness_col="correct",
        n_bins=15,
        min_support=25,
    )

    laece = laece_from_classwise(classwise_df)
    overall_df["laece"] = laece
    overall_df["score_column"] = score_col

    overall_df.to_csv(out_dir / f"{prefix}_overall_metrics.csv", index=False)
    classwise_df.to_csv(out_dir / f"{prefix}_classwise_metrics.csv", index=False)

    if "nominal_occlusion" in df.columns:
        rows = []
        for occ, sub in df.groupby("nominal_occlusion"):
            m = summarize_calibration_metrics(sub, confidence_col=score_col, correctness_col="correct", n_bins=15)
            cw = classwise_calibration_table(
                sub,
                class_col=class_col,
                confidence_col=score_col,
                correctness_col="correct",
                n_bins=15,
                min_support=10,
            )
            m["laece"] = laece_from_classwise(cw)
            m["nominal_occlusion"] = occ
            m["score_column"] = score_col
            rows.append(m)
        pd.DataFrame(rows).sort_values("nominal_occlusion").to_csv(
            out_dir / f"{prefix}_by_occlusion_metrics.csv", index=False
        )

    print("\nOVERALL")
    print(overall_df)
    if len(classwise_df):
        print("\nCLASSWISE HEAD")
        print(classwise_df.head(10))


def main():
    parser = argparse.ArgumentParser(description="Run extended calibration metric report.")
    parser.add_argument("--matched_pred_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gt_path", type=str, default=None)
    parser.add_argument("--class_col", type=str, default="class_name")
    args = parser.parse_args()

    df = pd.read_csv(args.matched_pred_path)
    if args.gt_path:
        gt = pd.read_csv(args.gt_path)
        df = maybe_add_occlusion(df, gt)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_columns = ["score"]
    for col in ["score_global_ts", "score_oc_ts"]:
        if col in df.columns:
            score_columns.append(col)

    for col in score_columns:
        prefix = col
        run_for_score_column(df, col, out_dir, prefix, class_col=args.class_col)


if __name__ == "__main__":
    main()
