from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.data.load_coco_annotations import load_coco_detection_gt


def load_target_classes(classes_yaml_path: str) -> list[str]:
    with open(classes_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    classes = cfg.get("target_classes", None)
    if classes is None:
        raise KeyError("Expected key 'target_classes' in classes YAML.")

    return list(classes)


def load_image_keys(image_list_path: str) -> list[str]:
    with open(image_list_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return [line for line in lines if line]


def save_table(df: pd.DataFrame, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() == ".csv":
        df.to_csv(output, index=False)
    elif output.suffix.lower() == ".parquet":
        df.to_parquet(output, index=False)
    else:
        raise ValueError("Output must end with .csv or .parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare COCO GT table for evaluation.")
    parser.add_argument("--ann_json", type=str, required=True, help="Path to COCO instances JSON")
    parser.add_argument("--classes_yaml", type=str, required=True, help="Path to configs/classes.yaml")
    parser.add_argument("--output_path", type=str, required=True, help="Output CSV or Parquet")
    parser.add_argument("--image_list", type=str, default=None, help="Optional text file with image filenames")
    parser.add_argument("--drop_crowd", action="store_true", help="Drop crowd annotations")
    args = parser.parse_args()

    target_classes = load_target_classes(args.classes_yaml)

    image_keys = None
    if args.image_list is not None:
        image_keys = load_image_keys(args.image_list)

    gt_df = load_coco_detection_gt(
        annotation_json_path=args.ann_json,
        include_category_names=target_classes,
        include_image_keys=image_keys,
        drop_crowd=args.drop_crowd,
    )

    save_table(gt_df, args.output_path)

    print(f"Saved GT table to: {args.output_path}")
    print(f"Rows written: {len(gt_df)}")
    print(f"Classes included: {sorted(gt_df['class_name'].unique().tolist()) if len(gt_df) else []}")


if __name__ == "__main__":
    main()
