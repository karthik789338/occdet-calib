from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def bucket_from_occlusion(x: float) -> str:
    if x <= 0.2:
        return "high_visibility"
    if x <= 0.4:
        return "medium_visibility"
    if x <= 0.6:
        return "low_visibility"
    return "very_low_visibility"


def main():
    parser = argparse.ArgumentParser(description="Build visibility-proxy training table from matched predictions.")
    parser.add_argument("--matched_pred_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    pred = pd.read_csv(args.matched_pred_path)
    gt = pd.read_csv(args.gt_path)

    gt_small = gt[["image_key", "estimated_occlusion", "nominal_occlusion"]].drop_duplicates("image_key").copy()
    gt_small["visibility_bucket"] = gt_small["estimated_occlusion"].map(bucket_from_occlusion)

    df = pred.merge(gt_small, on="image_key", how="left")
    if df["visibility_bucket"].isna().any():
        raise RuntimeError("Some rows missing visibility bucket after merge.")

    # basic geometric features
    df["box_w"] = (df["x2"] - df["x1"]).clip(lower=1e-6)
    df["box_h"] = (df["y2"] - df["y1"]).clip(lower=1e-6)
    df["box_area"] = df["box_w"] * df["box_h"]
    df["aspect_ratio"] = df["box_w"] / df["box_h"]

    keep = [
        "image_key", "model_name", "class_id", "class_name", "score",
        "x1", "y1", "x2", "y2", "box_w", "box_h", "box_area", "aspect_ratio",
        "correct", "estimated_occlusion", "nominal_occlusion", "visibility_bucket"
    ]
    out = df[keep].copy()

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(out.head())
    print("Rows written:", len(out))
    print(out["visibility_bucket"].value_counts())


if __name__ == "__main__":
    main()
