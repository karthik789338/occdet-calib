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


def clip_box(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    return (
        max(0.0, min(float(img_w), x1)),
        max(0.0, min(float(img_h), y1)),
        max(0.0, min(float(img_w), x2)),
        max(0.0, min(float(img_h), y2)),
    )


def box_area(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate truncation variants.")
    parser.add_argument("--seed_object_table", type=str, required=True)
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

    output_image_dir = Path(args.output_image_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    sides = ["left", "right", "top", "bottom"]

    for _, seed_row in seed_df.iterrows():
        image_id = int(seed_row["image_id"])
        image_key = str(seed_row["image_key"])
        ann_id = int(seed_row["ann_id"])
        class_id = int(seed_row["class_id"])
        class_name = str(seed_row["class_name"])

        x1 = float(seed_row["x1"])
        y1 = float(seed_row["y1"])
        x2 = float(seed_row["x2"])
        y2 = float(seed_row["y2"])

        src_path = Path(args.image_root) / image_key
        if not src_path.exists():
            continue

        base_img = ensure_rgba(Image.open(src_path))
        img_w, img_h = base_img.size
        obj_w = max(1.0, x2 - x1)
        obj_h = max(1.0, y2 - y1)
        orig_area = obj_w * obj_h

        local_rng = random.Random(args.random_seed + ann_id)
        chosen_side = local_rng.choice(sides)

        for nominal_occ in nominal_levels:
            variant_img = Image.new("RGBA", (img_w, img_h), (114, 114, 114, 255))
            dx, dy = 0, 0

            if nominal_occ > 0.0:
                if chosen_side in ("left", "right"):
                    cut_px = int(round(nominal_occ * obj_w))
                    cut_px = min(cut_px, int(obj_w) - 1)
                    if chosen_side == "left":
                        dx = -int(round(x1 + cut_px))
                    else:
                        dx = int(round(img_w - x2 + cut_px))
                else:
                    cut_px = int(round(nominal_occ * obj_h))
                    cut_px = min(cut_px, int(obj_h) - 1)
                    if chosen_side == "top":
                        dy = -int(round(y1 + cut_px))
                    else:
                        dy = int(round(img_h - y2 + cut_px))

            variant_img.paste(base_img, (dx, dy))

            vx1, vy1, vx2, vy2 = clip_box(x1 + dx, y1 + dy, x2 + dx, y2 + dy, img_w, img_h)
            visible_box = (vx1, vy1, vx2, vy2)
            visible_area = box_area(visible_box)

            if visible_area <= 1.0:
                continue

            actual_trunc = 1.0 - (visible_area / orig_area)

            side_tag = chosen_side[0].upper()
            trunc_tag = int(round(nominal_occ * 100))
            variant_name = f"{Path(image_key).stem}_ann{ann_id}_trunc{side_tag}{trunc_tag}.png"
            variant_path = output_image_dir / variant_name
            variant_img.save(variant_path)

            rows.append(
                {
                    "image_id": image_id,
                    "image_key": image_key,
                    "ann_id": ann_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "variant_type": "truncation",
                    "nominal_occlusion": nominal_occ,
                    "estimated_occlusion": actual_trunc,
                    "truncation_side": chosen_side,
                    "variant_image_path": str(variant_path),
                    "visible_x1": vx1,
                    "visible_y1": vy1,
                    "visible_x2": vx2,
                    "visible_y2": vy2,
                }
            )

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output_metadata_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Generated truncation variants: {len(out_df)}")
    if len(out_df):
        print(out_df.groupby("nominal_occlusion").size())
        print(out_df["truncation_side"].value_counts())
    print(f"Saved metadata to: {args.output_metadata_csv}")


if __name__ == "__main__":
    main()
