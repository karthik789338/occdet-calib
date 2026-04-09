from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from PIL import Image


def load_occlusion_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_rgba(image: Image.Image) -> Image.Image:
    if image.mode != "RGBA":
        return image.convert("RGBA")
    return image


def compute_mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def paste_occluder(
    base_rgba: Image.Image,
    occ_rgba: Image.Image,
    x: int,
    y: int,
) -> Image.Image:
    canvas = base_rgba.copy()
    canvas.alpha_composite(occ_rgba, (x, y))
    return canvas


def estimate_occlusion_ratio_from_alpha(
    target_box: Tuple[int, int, int, int],
    occ_rgba: Image.Image,
    paste_x: int,
    paste_y: int,
) -> float:
    x1, y1, x2, y2 = target_box
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    target_area = box_w * box_h

    occ_np = np.array(occ_rgba)
    alpha = occ_np[..., 3] > 0

    ys, xs = np.where(alpha)
    if len(xs) == 0:
        return 0.0

    occ_x1 = paste_x + xs.min()
    occ_y1 = paste_y + ys.min()
    occ_x2 = paste_x + xs.max() + 1
    occ_y2 = paste_y + ys.max() + 1

    inter_x1 = max(x1, occ_x1)
    inter_y1 = max(y1, occ_y1)
    inter_x2 = min(x2, occ_x2)
    inter_y2 = min(y2, occ_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    overlap_area = inter_w * inter_h

    return float(overlap_area / target_area)


def choose_occluder_scale_for_target(
    target_box: Tuple[float, float, float, float],
    nominal_occ: float,
) -> float:
    """
    Simple heuristic scale:
    bigger occlusion target => larger scale.
    """
    x1, y1, x2, y2 = target_box
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    ref = min(box_w, box_h)

    if nominal_occ <= 0.0:
        return 0.0
    if nominal_occ <= 0.2:
        return 0.35
    if nominal_occ <= 0.4:
        return 0.55
    if nominal_occ <= 0.6:
        return 0.75
    return 0.95


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate overlap variants for OccDet-Calib.")
    parser.add_argument("--seed_object_table", type=str, required=True)
    parser.add_argument("--occluder_bank_csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--occlusion_yaml", type=str, required=True)
    parser.add_argument("--output_image_dir", type=str, required=True)
    parser.add_argument("--output_metadata_csv", type=str, required=True)
    parser.add_argument("--max_seed_objects", type=int, default=200)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    occ_cfg = load_occlusion_config(args.occlusion_yaml)
    nominal_levels = list(occ_cfg["nominal_occlusion_levels"])

    seed_df = pd.read_csv(args.seed_object_table)
    occ_bank_df = pd.read_csv(args.occluder_bank_csv)

    seed_df = seed_df.head(args.max_seed_objects).copy()

    output_image_dir = Path(args.output_image_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []

    for _, seed_row in seed_df.iterrows():
        image_key = str(seed_row["image_key"])
        image_id = int(seed_row["image_id"])
        ann_id = int(seed_row["ann_id"])
        class_name = str(seed_row["class_name"])

        x1 = float(seed_row["x1"])
        y1 = float(seed_row["y1"])
        x2 = float(seed_row["x2"])
        y2 = float(seed_row["y2"])
        target_box = (x1, y1, x2, y2)

        src_path = Path(args.image_root) / image_key
        if not src_path.exists():
            continue

        base_img = ensure_rgba(Image.open(src_path))

        target_w = max(1, int(x2 - x1))
        target_h = max(1, int(y2 - y1))

        # Exclude same annotation if the image/object somehow appears in bank semantics later.
        valid_occluders = occ_bank_df.copy()
        if len(valid_occluders) == 0:
            continue

        for nominal_occ in nominal_levels:
            variant_img = base_img.copy()

            if nominal_occ == 0.0:
                variant_name = f"{Path(image_key).stem}_ann{ann_id}_occ0.png"
                variant_path = output_image_dir / variant_name
                variant_img.save(variant_path)

                rows.append(
                    {
                        "image_id": image_id,
                        "image_key": image_key,
                        "ann_id": ann_id,
                        "class_name": class_name,
                        "variant_type": "overlap",
                        "nominal_occlusion": nominal_occ,
                        "estimated_occlusion": 0.0,
                        "occluder_class_name": None,
                        "variant_image_path": str(variant_path),
                    }
                )
                continue

            occ_row = valid_occluders.sample(n=1, random_state=random.randint(0, 10**9)).iloc[0]
            occ_patch_path = str(occ_row["patch_path"])
            occ_class_name = str(occ_row["class_name"])

            occ_patch = ensure_rgba(Image.open(occ_patch_path))
            scale = choose_occluder_scale_for_target(target_box, nominal_occ)

            new_w = max(1, int(target_w * scale))
            new_h = max(1, int(target_h * scale))
            occ_patch = occ_patch.resize((new_w, new_h), Image.Resampling.BILINEAR)

            # Place near box center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            paste_x = max(0, center_x - new_w // 2)
            paste_y = max(0, center_y - new_h // 2)

            variant_img = paste_occluder(variant_img, occ_patch, paste_x, paste_y)
            est_occ = estimate_occlusion_ratio_from_alpha(target_box, occ_patch, paste_x, paste_y)

            variant_name = f"{Path(image_key).stem}_ann{ann_id}_occ{int(nominal_occ*100)}.png"
            variant_path = output_image_dir / variant_name
            variant_img.save(variant_path)

            rows.append(
                {
                    "image_id": image_id,
                    "image_key": image_key,
                    "ann_id": ann_id,
                    "class_name": class_name,
                    "variant_type": "overlap",
                    "nominal_occlusion": nominal_occ,
                    "estimated_occlusion": est_occ,
                    "occluder_class_name": occ_class_name,
                    "variant_image_path": str(variant_path),
                }
            )

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output_metadata_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Generated variants: {len(out_df)}")
    if len(out_df):
        print(out_df.groupby('nominal_occlusion').size())
    print(f"Variant metadata saved to: {args.output_metadata_csv}")


if __name__ == "__main__":
    main()
