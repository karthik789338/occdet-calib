from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from pycocotools.coco import COCO


def load_target_classes(classes_yaml_path: str) -> list[str]:
    with open(classes_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [str(x) for x in cfg["target_classes"]]


def load_occlusion_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_name(name: str) -> str:
    return str(name).strip().lower()


def bbox_xywh_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return float(x), float(y), float(x + w), float(y + h)


def compute_original_occlusion_ratio(coco: COCO, anns: list[dict], target_ann: dict) -> float:
    target_mask = coco.annToMask(target_ann).astype(bool)
    target_area = target_mask.sum()
    if target_area == 0:
        return 1.0

    other_union = np.zeros_like(target_mask, dtype=bool)
    for ann in anns:
        if ann["id"] == target_ann["id"]:
            continue
        if "segmentation" not in ann:
            continue
        try:
            other_mask = coco.annToMask(ann).astype(bool)
        except Exception:
            continue
        other_union |= other_mask

    overlap = np.logical_and(target_mask, other_union).sum()
    return float(overlap / target_area)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select seed COCO images for OccDet-Calib.")
    parser.add_argument("--ann_json", type=str, required=True)
    parser.add_argument("--classes_yaml", type=str, required=True)
    parser.add_argument("--occlusion_yaml", type=str, required=True)
    parser.add_argument("--output_image_list", type=str, required=True)
    parser.add_argument("--output_object_table", type=str, required=True)
    args = parser.parse_args()

    target_classes = load_target_classes(args.classes_yaml)
    target_class_set = {normalize_name(x) for x in target_classes}

    occ_cfg = load_occlusion_config(args.occlusion_yaml)

    target_num_images = int(occ_cfg["target_num_images"])
    min_box_area = float(occ_cfg["min_box_area"])
    max_original_occlusion = float(occ_cfg["max_original_occlusion"])
    min_target_instances_per_image = int(occ_cfg["min_target_instances_per_image"])

    coco = COCO(args.ann_json)

    all_cats = coco.loadCats(coco.getCatIds())
    matched_cats = [cat for cat in all_cats if normalize_name(cat["name"]) in target_class_set]

    cat_ids = [int(cat["id"]) for cat in matched_cats]
    cat_id_to_name = {int(cat["id"]): str(cat["name"]) for cat in matched_cats}

    print("Requested target classes:", sorted(target_class_set))
    print("Matched COCO categories:", sorted(cat_id_to_name.values()))
    print("Matched category ids:", cat_ids)

    if not cat_ids:
        raise RuntimeError(
            "No COCO categories matched the requested target classes. "
            "Check class-name normalization and the contents of configs/classes.yaml."
        )

    # IMPORTANT: union over classes, not a single multi-class call.
    image_id_set = set()
    for cat_id in cat_ids:
        image_id_set.update(coco.getImgIds(catIds=[cat_id]))

    image_ids = sorted(image_id_set)
    image_infos = coco.loadImgs(image_ids)

    selected_images: List[str] = []
    object_rows: List[Dict[str, object]] = []
    image_scores: List[tuple] = []

    for img in image_infos:
        img_id = int(img["id"])
        file_name = str(img["file_name"])

        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        valid_target_rows = []

        for ann in anns:
            if int(ann.get("category_id")) not in cat_ids:
                continue
            if int(ann.get("iscrowd", 0)) == 1:
                continue
            if "bbox" not in ann or "segmentation" not in ann:
                continue

            bbox = ann["bbox"]
            box_area = float(bbox[2] * bbox[3])
            if box_area < min_box_area:
                continue

            try:
                mask = coco.annToMask(ann)
            except Exception:
                continue

            mask_area = int(mask.sum())
            if mask_area <= 0:
                continue

            occ_ratio = compute_original_occlusion_ratio(coco, anns, ann)
            if occ_ratio > max_original_occlusion:
                continue

            x1, y1, x2, y2 = bbox_xywh_to_xyxy(bbox)

            valid_target_rows.append(
                {
                    "image_id": img_id,
                    "image_key": file_name,
                    "ann_id": int(ann["id"]),
                    "class_id": int(ann["category_id"]),
                    "class_name": cat_id_to_name[int(ann["category_id"])],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "bbox_area": box_area,
                    "mask_area": mask_area,
                    "original_occlusion_ratio": occ_ratio,
                }
            )

        if len(valid_target_rows) < min_target_instances_per_image:
            continue

        mean_occ = float(np.mean([r["original_occlusion_ratio"] for r in valid_target_rows]))
        num_targets = len(valid_target_rows)

        image_scores.append((mean_occ, -num_targets, file_name, valid_target_rows))

    image_scores = sorted(image_scores, key=lambda x: (x[0], x[1], x[2]))
    chosen = image_scores[:target_num_images]

    for _, _, file_name, rows in chosen:
        selected_images.append(file_name)
        object_rows.extend(rows)

    out_image_path = Path(args.output_image_list)
    out_image_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_image_path, "w", encoding="utf-8") as f:
        for name in selected_images:
            f.write(name + "\n")

    object_df = pd.DataFrame(object_rows)
    out_obj_path = Path(args.output_object_table)
    out_obj_path.parent.mkdir(parents=True, exist_ok=True)

    if out_obj_path.suffix.lower() == ".csv":
        object_df.to_csv(out_obj_path, index=False)
    elif out_obj_path.suffix.lower() == ".parquet":
        object_df.to_parquet(out_obj_path, index=False)
    else:
        raise ValueError("output_object_table must end with .csv or .parquet")

    print(f"Candidate images examined: {len(image_infos)}")
    print(f"Selected images: {len(selected_images)}")
    print(f"Selected objects: {len(object_df)}")
    if len(object_df):
        print(object_df[['class_name', 'original_occlusion_ratio']].groupby('class_name').agg(['count', 'mean']))
    print(f"Image list saved to: {args.output_image_list}")
    print(f"Object table saved to: {args.output_object_table}")


if __name__ == "__main__":
    main()
