from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import yaml

from src.detectors.yolo_wrapper import YOLOWrapper
from src.detectors.fcos_wrapper import FCOSWrapper
from src.detectors.deformable_detr_wrapper import DeformableDETRWrapper


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


def build_detector(model_cfg):
    family = model_cfg.get("family")
    architecture = model_cfg.get("architecture")

    if family == "yolo":
        return YOLOWrapper(
            weights=model_cfg["weights"],
            device=model_cfg.get("device", "cpu"),
            conf_threshold=model_cfg.get("conf_threshold", 0.001),
            iou_threshold=model_cfg.get("iou_threshold", 0.7),
            imgsz=model_cfg.get("imgsz", 640),
            max_det=model_cfg.get("max_det", 300),
            model_name=model_cfg.get("name", "yolo_v8m"),
        )

    if family == "mmdet" and architecture == "fcos":
        return FCOSWrapper(
            config_path=model_cfg["config_path"],
            checkpoint_path=model_cfg["checkpoint_path"],
            device=model_cfg.get("device", "cuda:0"),
            score_thr=model_cfg.get("score_thr", 0.001),
            max_per_img=model_cfg.get("max_per_img", 300),
            model_name=model_cfg.get("name", "fcos_r50"),
        )

    if family == "mmdet" and architecture == "deformable_detr":
        return DeformableDETRWrapper(
            config_path=model_cfg["config_path"],
            checkpoint_path=model_cfg["checkpoint_path"],
            device=model_cfg.get("device", "cuda:0"),
            score_thr=model_cfg.get("score_thr", 0.001),
            max_per_img=model_cfg.get("max_per_img", 300),
            model_name=model_cfg.get("name", "deformable_detr_r50"),
        )

    raise ValueError(f"Unsupported model config: {model_cfg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clean baseline inference.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory of images")
    parser.add_argument("--output_csv", type=str, required=True, help="Where to save predictions CSV")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--limit", type=int, default=50, help="Max number of images")
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)
    detector = build_detector(model_cfg)

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
