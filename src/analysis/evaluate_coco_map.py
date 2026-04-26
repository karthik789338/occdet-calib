from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    parser = argparse.ArgumentParser(description="Evaluate COCO mAP from prediction CSV.")
    parser.add_argument("--pred_csv", type=str, required=True)
    parser.add_argument("--ann_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--score_threshold", type=float, default=0.0)
    parser.add_argument("--image_root_hint", type=str, default=None)
    args = parser.parse_args()

    pred = pd.read_csv(args.pred_csv).copy()
    pred = pred[pred["score"] >= args.score_threshold].copy()

    coco_gt = COCO(args.ann_json)

    image_name_to_id = {}
    for img_id, info in coco_gt.imgs.items():
        file_name = info["file_name"]
        image_name_to_id[file_name] = img_id

    results = []
    for _, row in pred.iterrows():
        image_path = str(row["image_path"])
        image_name = Path(image_path).name
        if image_name not in image_name_to_id:
            continue

        x1 = float(row["x1"])
        y1 = float(row["y1"])
        x2 = float(row["x2"])
        y2 = float(row["y2"])

        results.append(
            {
                "image_id": int(image_name_to_id[image_name]),
                "category_id": int(row["class_id"]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(row["score"]),
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_json = out_dir / "coco_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f)

    coco_dt = coco_gt.loadRes(str(results_json))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats_names = [
        "mAP_50_95", "mAP_50", "mAP_75", "mAP_small", "mAP_medium", "mAP_large",
        "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"
    ]
    summary = pd.DataFrame([dict(zip(stats_names, coco_eval.stats.tolist()))])
    summary["num_predictions"] = len(results)
    summary["score_threshold"] = args.score_threshold
    summary.to_csv(out_dir / "coco_map_summary.csv", index=False)

    print(summary)


if __name__ == "__main__":
    main()
