from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def coco_bbox_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    """
    COCO bbox format: [x, y, width, height]
    Returns: (x1, y1, x2, y2)
    """
    if len(bbox) != 4:
        raise ValueError(f"Invalid COCO bbox: {bbox}")

    x, y, w, h = bbox
    x1 = float(x)
    y1 = float(y)
    x2 = float(x + w)
    y2 = float(y + h)
    return x1, y1, x2, y2


def load_coco_detection_gt_from_dict(
    coco_data: dict,
    *,
    include_category_names: Optional[Iterable[str]] = None,
    include_image_keys: Optional[Iterable[str]] = None,
    drop_crowd: bool = False,
) -> pd.DataFrame:
    """
    Convert COCO detection annotations into a flat GT table.

    Output columns:
        image_id, image_key, ann_id,
        class_id, class_name,
        x1, y1, x2, y2,
        area, iscrowd
    """
    categories = coco_data.get("categories", [])
    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])

    cat_id_to_name = {int(cat["id"]): str(cat["name"]) for cat in categories}
    image_id_to_name = {int(img["id"]): str(img["file_name"]) for img in images}

    include_category_names = (
        set(include_category_names) if include_category_names is not None else None
    )
    include_image_keys = (
        set(include_image_keys) if include_image_keys is not None else None
    )

    rows = []

    for ann in annotations:
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])
        class_name = cat_id_to_name.get(category_id, str(category_id))
        image_key = image_id_to_name.get(image_id, str(image_id))

        if include_category_names is not None and class_name not in include_category_names:
            continue

        if include_image_keys is not None and image_key not in include_image_keys:
            continue

        iscrowd = int(ann.get("iscrowd", 0))
        if drop_crowd and iscrowd == 1:
            continue

        bbox = ann.get("bbox", None)
        if bbox is None:
            continue

        x1, y1, x2, y2 = coco_bbox_to_xyxy(bbox)

        rows.append(
            {
                "image_id": image_id,
                "image_key": image_key,
                "ann_id": int(ann["id"]),
                "class_id": category_id,
                "class_name": class_name,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "area": float(ann.get("area", 0.0)),
                "iscrowd": iscrowd,
            }
        )

    return pd.DataFrame(rows)


def load_coco_detection_gt(
    annotation_json_path: str,
    *,
    include_category_names: Optional[Iterable[str]] = None,
    include_image_keys: Optional[Iterable[str]] = None,
    drop_crowd: bool = False,
) -> pd.DataFrame:
    annotation_path = Path(annotation_json_path)
    if not annotation_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {annotation_json_path}")

    with open(annotation_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    return load_coco_detection_gt_from_dict(
        coco_data=coco_data,
        include_category_names=include_category_names,
        include_image_keys=include_image_keys,
        drop_crowd=drop_crowd,
    )
