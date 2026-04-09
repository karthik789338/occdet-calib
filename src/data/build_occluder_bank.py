from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from pycocotools.coco import COCO


def load_target_classes(classes_yaml_path: str) -> list[str]:
    with open(classes_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [str(x) for x in cfg["target_classes"]]


def load_occlusion_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an occluder bank from COCO masks.")
    parser.add_argument("--ann_json", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--classes_yaml", type=str, required=True)
    parser.add_argument("--occlusion_yaml", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_target_classes_only", action="store_true")
    args = parser.parse_args()

    target_classes = load_target_classes(args.classes_yaml)
    occ_cfg = load_occlusion_config(args.occlusion_yaml)

    min_mask_area = int(occ_cfg["min_occluder_mask_area"])
    max_mask_area = int(occ_cfg["max_occluder_mask_area"])
    max_per_class = int(occ_cfg["max_occluders_per_class"])

    coco = COCO(args.ann_json)

    if args.use_target_classes_only:
        cat_ids = coco.getCatIds(catNms=target_classes)
    else:
        cat_ids = coco.getCatIds()

    cats = coco.loadCats(cat_ids)
    cat_id_to_name = {cat["id"]: cat["name"] for cat in cats}

    output_dir = Path(args.output_dir)
    patch_dir = output_dir / "patches"
    mask_dir = output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    per_class_counts: Dict[str, int] = {}

    for cat_id in cat_ids:
        class_name = cat_id_to_name[cat_id]
        per_class_counts.setdefault(class_name, 0)

        ann_ids = coco.getAnnIds(catIds=[cat_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            if per_class_counts[class_name] >= max_per_class:
                break

            if "segmentation" not in ann or "bbox" not in ann:
                continue

            img_info = coco.loadImgs([ann["image_id"]])[0]
            img_path = Path(args.image_root) / img_info["file_name"]
            if not img_path.exists():
                continue

            try:
                mask = coco.annToMask(ann).astype(np.uint8)
            except Exception:
                continue

            mask_area = int(mask.sum())
            if mask_area < min_mask_area or mask_area > max_mask_area:
                continue

            x, y, w, h = ann["bbox"]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            image = Image.open(img_path).convert("RGBA")
            image_np = np.array(image)

            crop_img = image_np[y1:y2, x1:x2].copy()
            crop_mask = mask[y1:y2, x1:x2].copy()

            if crop_mask.sum() == 0:
                continue

            crop_img[..., 3] = crop_mask * 255

            patch_name = f"{class_name.replace(' ', '_')}_img{ann['image_id']}_ann{ann['id']}.png"
            mask_name = f"{class_name.replace(' ', '_')}_img{ann['image_id']}_ann{ann['id']}_mask.png"

            patch_path = patch_dir / patch_name
            mask_path = mask_dir / mask_name

            Image.fromarray(crop_img).save(patch_path)
            Image.fromarray((crop_mask * 255).astype(np.uint8)).save(mask_path)

            rows.append(
                {
                    "ann_id": int(ann["id"]),
                    "image_id": int(ann["image_id"]),
                    "image_key": str(img_info["file_name"]),
                    "class_id": int(cat_id),
                    "class_name": class_name,
                    "mask_area": mask_area,
                    "bbox_x": float(x),
                    "bbox_y": float(y),
                    "bbox_w": float(w),
                    "bbox_h": float(h),
                    "patch_path": str(patch_path),
                    "mask_path": str(mask_path),
                }
            )

            per_class_counts[class_name] += 1

    meta_df = pd.DataFrame(rows)
    meta_df.to_csv(output_dir / "occluder_bank_metadata.csv", index=False)

    print(f"Saved occluder bank to: {output_dir}")
    print(f"Total occluders: {len(meta_df)}")
    if len(meta_df):
        print(meta_df["class_name"].value_counts())

if __name__ == "__main__":
    main()
