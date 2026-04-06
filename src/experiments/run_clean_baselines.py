from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import yaml

from src.detectors.yolo_wrapper import YOLOWrapper


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_images(image_dir: str, limit: int | None = None) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [
        str(p) for p in Path(image_dir).rglob("*") if p.suffix.lower() in exts
    ]
    image_paths = sorted(image_paths)
    if limit is not None:
        image_paths = image_paths[:limit]
    return image_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clean baseline inference.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory of images")
    parser.add_argument("--output_csv", type=str, required=True, help="Where to save predictions CSV")
    parser.add_argument("--model_config", type=str, required=True, help="Path to YOLO model config YAML")
    parser.add_argument("--limit", type=int, default=50, help="Max number of images")
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)

    detector = YOLOWrapper(
        weights=model_cfg["weights"],
        device=model_cfg.get("device", "cpu"),
        conf_threshold=model_cfg.get("conf_threshold", 0.001),
        iou_threshold=model_cfg.get("iou_threshold", 0.7),
        imgsz=model_cfg.get("imgsz", 640),
        max_det=model_cfg.get("max_det", 300),
        model_name=model_cfg.get("name", "yolo_v8m"),
    )

    image_paths = collect_images(args.image_dir, args.limit)
    if not image_paths:
        raise RuntimeError(f"No images found in: {args.image_dir}")

    rows = []
    for image_path in image_paths:
        preds = detector.predict(image_path)
        if not preds:
            rows.append(
                {
                    "image_path": image_path,
                    "model_name": detector.model_name,
                    "class_id": None,
                    "class_name": None,
                    "score": None,
                    "x1": None,
                    "y1": None,
                    "x2": None,
                    "y2": None,
                }
            )
            continue

        for det in preds:
            rows.append(det.to_dict())

    df = pd.DataFrame(rows)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved predictions to: {output_path}")
    print(f"Images processed: {len(image_paths)}")
    print(f"Rows written: {len(df)}")


if __name__ == "__main__":
    main()