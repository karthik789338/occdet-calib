from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


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
    parser = argparse.ArgumentParser(description="Prepare GT table for overlap variants.")
    parser.add_argument("--seed_object_table", type=str, required=True)
    parser.add_argument("--overlap_metadata_csv", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    seed_df = pd.read_csv(args.seed_object_table)
    overlap_df = pd.read_csv(args.overlap_metadata_csv)

    required_seed_cols = [
        "image_id", "image_key", "ann_id", "class_id", "class_name",
        "x1", "y1", "x2", "y2"
    ]
    for col in required_seed_cols:
        if col not in seed_df.columns:
            raise KeyError(f"Seed object table missing required column: {col}")

    required_overlap_cols = [
        "image_id", "image_key", "ann_id", "class_name",
        "nominal_occlusion", "estimated_occlusion", "variant_image_path"
    ]
    for col in required_overlap_cols:
        if col not in overlap_df.columns:
            raise KeyError(f"Overlap metadata missing required column: {col}")

    seed_keep = seed_df[
        ["image_id", "image_key", "ann_id", "class_id", "class_name", "x1", "y1", "x2", "y2"]
    ].copy()

    merged = overlap_df.merge(
        seed_keep,
        on=["image_id", "image_key", "ann_id", "class_name"],
        how="left",
        validate="many_to_one",
    )

    if merged["class_id"].isna().any():
        missing = merged[merged["class_id"].isna()].head(10)
        raise RuntimeError(
            "Some overlap rows could not be matched back to seed objects.\n"
            f"Sample:\n{missing}"
        )

    merged["variant_image_key"] = merged["variant_image_path"].map(lambda p: Path(str(p)).name)

    gt_df = merged[
        [
            "image_id",
            "image_key",
            "variant_image_key",
            "variant_image_path",
            "ann_id",
            "class_id",
            "class_name",
            "x1",
            "y1",
            "x2",
            "y2",
            "nominal_occlusion",
            "estimated_occlusion",
            "occluder_class_name",
            "variant_type",
        ]
    ].copy()

    save_table(gt_df, args.output_path)

    print(f"Saved overlap GT table to: {args.output_path}")
    print(f"Rows written: {len(gt_df)}")
    if len(gt_df):
        print(gt_df.groupby('nominal_occlusion').size())


if __name__ == "__main__":
    main()
