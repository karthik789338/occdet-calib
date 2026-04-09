from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from PIL import Image


def load_occlusion_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_rgba(image: Image.Image) -> Image.Image:
    return image.convert("RGBA") if image.mode != "RGBA" else image


def paste_occluder(base_rgba: Image.Image, occ_rgba: Image.Image, x: int, y: int) -> Image.Image:
    canvas = base_rgba.copy()
    canvas.alpha_composite(occ_rgba, (x, y))
    return canvas


def overlap_ratio(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    ix1 = max(xa1, xb1)
    iy1 = max(ya1, yb1)
    ix2 = min(xa2, xb2)
    iy2 = min(ya2, yb2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    if area_a <= 0:
        return 0.0
    return float(inter / area_a)


def choose_scale(target_w: int, target_h: int, nominal_occ: float) -> Tuple[int, int]:
    if nominal_occ <= 0.0:
        return 0, 0

    if nominal_occ <= 0.2:
        scale = 0.35
    elif nominal_occ <= 0.4:
        scale = 0.55
    elif nominal_occ <= 0.6:
        scale = 0.75
    else:
        scale = 0.95

    return max(1, int(target_w * scale)), max(1, int(target_h * scale))


def candidate_positions(
    img_w: int,
    img_h: int,
    target_box: Tuple[float, float, float, float],
    occ_w: int,
    occ_h: int,
    margin: int = 10,
) -> List[Tuple[int, int]]:
    x1, y1, x2, y2 = map(int, target_box)
    candidates = []

    # left
    candidates.append((max(0, x1 - occ_w - margin), max(0, y1)))
    # right
    candidates.append((min(img_w - occ_w, x2 + margin), max(0, y1)))
    # top
    candidates.append((max(0, x1), max(0, y1 - occ_h - margin)))
    # bottom
    candidates.append((max(0, x1), min(img_h - occ_h, y2 + margin)))

    valid = []
    for px, py in candidates:
        if 0 <= px <= img_w - occ_w and 0 <= py <= img_h - occ_h:
            valid.append((px, py))
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate distractor-control variants.")
    parser.add_argument("--seed_object_table", type=str, required=True)
    parser.add_argument("--occluder_bank_csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--occlusion_yaml", type=str, required=True)
    parser.add_argument("--output_image_dir", type=str, required=True)
    parser.add_argument("--output_metadata_csv", type=str, required=True)
    parser.add_argument("--max_seed_objects", type=int, default=500)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    occ_cfg = load_occlusion_config(args.occlusion_yaml)
    nominal_levels = list(occ_cfg["nominal_occlusion_levels"])

    seed_df = pd.read_csv(args.seed_object_table).head(args.max_seed_objects).copy()
    occ_bank_df = pd.read_csv(args.occluder_bank_csv)

    output_image_dir = Path(args.output_image_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []

    for _, seed_row in seed_df.iterrows():
        image_id = int(seed_row["image_id"])
        image_key = str(seed_row["image_key"])
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
        img_w, img_h = base_img.size

        target_w = max(1, int(x2 - x1))
        target_h = max(1, int(y2 - y1))

        for nominal_occ in nominal_levels:
            variant_img = base_img.copy()

            if nominal_occ == 0.0:
                variant_name = f"{Path(image_key).stem}_ann{ann_id}_dist0.png"
                variant_path = output_image_dir / variant_name
                variant_img.save(variant_path)

                rows.append(
                    {
                        "image_id": image_id,
                        "image_key": image_key,
                        "ann_id": ann_id,
                        "class_name": class_name,
                        "variant_type": "distractor",
                        "nominal_occlusion": nominal_occ,
                        "estimated_occlusion": 0.0,
                        "occluder_class_name": None,
                        "variant_image_path": str(variant_path),
                    }
                )
                continue

            occ_row = occ_bank_df.sample(n=1, random_state=random.randint(0, 10**9)).iloc[0]
            occ_patch = ensure_rgba(Image.open(str(occ_row["patch_path"])))
            occ_class_name = str(occ_row["class_name"])

            new_w, new_h = choose_scale(target_w, target_h, nominal_occ)
            occ_patch = occ_patch.resize((new_w, new_h), Image.Resampling.BILINEAR)

            positions = candidate_positions(img_w, img_h, target_box, new_w, new_h, margin=10)
            if not positions:
                continue

            placed = False
            for px, py in positions:
                occ_box = (px, py, px + new_w, py + new_h)
                occ_ratio = overlap_ratio(target_box, occ_box)

                # distractor control: almost no overlap with target
                if occ_ratio <= 0.01:
                    variant_img = paste_occluder(variant_img, occ_patch, px, py)
                    variant_name = f"{Path(image_key).stem}_ann{ann_id}_dist{int(nominal_occ*100)}.png"
                    variant_path = output_image_dir / variant_name
                    variant_img.save(variant_path)

                    rows.append(
                        {
                            "image_id": image_id,
                            "image_key": image_key,
                            "ann_id": ann_id,
                            "class_name": class_name,
                            "variant_type": "distractor",
                            "nominal_occlusion": nominal_occ,
                            "estimated_occlusion": occ_ratio,
                            "occluder_class_name": occ_class_name,
                            "variant_image_path": str(variant_path),
                        }
                    )
                    placed = True
                    break

            if not placed:
                continue

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output_metadata_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Generated distractor variants: {len(out_df)}")
    if len(out_df):
        print(out_df.groupby('nominal_occlusion').size())
    print(f"Saved metadata to: {args.output_metadata_csv}")


if __name__ == "__main__":
    main()
