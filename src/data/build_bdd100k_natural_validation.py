from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BDD_TO_COCO_NAME = {
    "person": "person",
    "bike": "bicycle",
    "motor": "motorcycle",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
}

# Keep class_id as string for this validation path so preds/gt can be normalized to names.
def natural_group(occluded: bool, truncated: bool) -> str:
    if occluded and truncated:
        return "occluded_and_truncated"
    if occluded:
        return "occluded_only"
    if truncated:
        return "truncated_only"
    return "clear"


def main():
    parser = argparse.ArgumentParser(description="Build BDD100K natural-occlusion validation manifest and GT.")
    parser.add_argument("--labels_json", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument("--gt_out", required=True)
    parser.add_argument("--min_box_area", type=float, default=300.0)
    parser.add_argument("--min_short_side", type=float, default=12.0)
    args = parser.parse_args()

    with open(args.labels_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_rows = []
    manifest_rows = []
    seen_images = set()
    ann_counter = 0

    for frame in data:
        image_key = frame.get("name")
        if not image_key:
            continue

        image_path = str(Path(args.image_root) / image_key)
        labels = frame.get("labels", [])
        kept_any = False

        for obj in labels:
            category = obj.get("category")
            if category not in BDD_TO_COCO_NAME:
                continue

            box = obj.get("box2d")
            if not box:
                continue

            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            area = w * h

            if area < args.min_box_area or min(w, h) < args.min_short_side:
                continue

            attrs = obj.get("attributes", {})
            occluded = bool(attrs.get("occluded", False))
            truncated = bool(attrs.get("truncated", False))

            class_name = BDD_TO_COCO_NAME[category]
            group = natural_group(occluded, truncated)

            gt_rows.append(
                {
                    "image_id": image_key,
                    "image_key": image_key,
                    "image_path": image_path,
                    "ann_id": ann_counter,
                    "class_id": class_name,
                    "class_name": class_name,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "natural_group": group,
                    "bdd_category": category,
                    "bdd_occluded": occluded,
                    "bdd_truncated": truncated,
                }
            )
            ann_counter += 1
            kept_any = True

        if kept_any and image_key not in seen_images:
            manifest_rows.append(
                {
                    "image_key": image_key,
                    "image_path": image_path,
                }
            )
            seen_images.add(image_key)

    gt_df = pd.DataFrame(gt_rows)
    manifest_df = pd.DataFrame(manifest_rows)

    Path(args.gt_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)

    gt_df.to_csv(args.gt_out, index=False)
    manifest_df.to_csv(args.manifest_out, index=False)

    print("Saved manifest:", args.manifest_out, "rows:", len(manifest_df))
    print("Saved gt:", args.gt_out, "rows:", len(gt_df))
    if len(gt_df):
        print("\nGroup counts:")
        print(gt_df["natural_group"].value_counts())
        print("\nClass counts:")
        print(gt_df["class_name"].value_counts())


if __name__ == "__main__":
    main()
