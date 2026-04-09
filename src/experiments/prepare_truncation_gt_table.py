from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GT table for truncation variants.")
    parser.add_argument("--truncation_metadata_csv", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.truncation_metadata_csv)

    required = [
        "image_id", "ann_id", "class_id", "class_name",
        "variant_image_path", "visible_x1", "visible_y1", "visible_x2", "visible_y2",
        "nominal_occlusion", "estimated_occlusion", "variant_type", "truncation_side"
    ]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")

    gt = df.copy()
    gt["image_key"] = gt["variant_image_path"].map(lambda p: Path(str(p)).name)
    gt["x1"] = gt["visible_x1"]
    gt["y1"] = gt["visible_y1"]
    gt["x2"] = gt["visible_x2"]
    gt["y2"] = gt["visible_y2"]

    gt = gt[
        [
            "image_id",
            "image_key",
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
            "variant_type",
            "truncation_side",
        ]
    ].copy()

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    gt.to_csv(out, index=False)

    print(f"Saved truncation GT to: {out}")
    print(f"Rows written: {len(gt)}")
    if len(gt):
        print(gt.groupby('nominal_occlusion').size())


if __name__ == "__main__":
    main()
