from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd

from src.metrics.matching import add_image_key_from_path, match_predictions_to_ground_truth
from src.metrics.map_eval import (
    summarize_detection_metrics,
    summarize_per_class_detection_metrics,
    summarize_per_image_detection_metrics,
)
from src.metrics.reliability import summarize_reliability
from src.metrics.monotonicity import compute_monotonicity_from_dataframe


pred_df = pd.DataFrame(
    [
        {
            "image_path": "/tmp/img1.jpg",
            "model_name": "toy_model",
            "class_id": 1,
            "class_name": "person",
            "score": 0.95,
            "x1": 10,
            "y1": 10,
            "x2": 50,
            "y2": 50,
            "raw_label": None,
            "extra": None,
        },
        {
            "image_path": "/tmp/img1.jpg",
            "model_name": "toy_model",
            "class_id": 1,
            "class_name": "person",
            "score": 0.60,
            "x1": 12,
            "y1": 12,
            "x2": 52,
            "y2": 52,
            "raw_label": None,
            "extra": None,
        },
        {
            "image_path": "/tmp/img1.jpg",
            "model_name": "toy_model",
            "class_id": 3,
            "class_name": "car",
            "score": 0.80,
            "x1": 100,
            "y1": 100,
            "x2": 140,
            "y2": 140,
            "raw_label": None,
            "extra": None,
        },
    ]
)

gt_df = pd.DataFrame(
    [
        {
            "image_key": "img1.jpg",
            "ann_id": 101,
            "class_id": 1,
            "class_name": "person",
            "x1": 11,
            "y1": 11,
            "x2": 51,
            "y2": 51,
            "area": 1600.0,
            "iscrowd": 0,
        },
        {
            "image_key": "img1.jpg",
            "ann_id": 102,
            "class_id": 17,
            "class_name": "cat",
            "x1": 200,
            "y1": 200,
            "x2": 240,
            "y2": 240,
            "area": 1600.0,
            "iscrowd": 0,
        },
    ]
)

pred_df = add_image_key_from_path(pred_df, image_path_col="image_path", output_col="image_key")

matched = match_predictions_to_ground_truth(
    pred_df=pred_df,
    gt_df=gt_df,
    image_key_col="image_key",
    class_id_col="class_id",
    score_col="score",
    iou_threshold=0.5,
    classwise=True,
)

overall = summarize_detection_metrics(matched, gt_df)
per_class = summarize_per_class_detection_metrics(matched, gt_df)
per_image = summarize_per_image_detection_metrics(matched, gt_df)
reliability = summarize_reliability(matched, confidence_col="score", correctness_col="correct", n_bins=3)
monotonicity = compute_monotonicity_from_dataframe(matched, confidence_col="score", correctness_col="correct", n_bins=3)

print("MATCHED")
print(matched)
print("\nOVERALL")
print(overall)
print("\nPER CLASS")
print(per_class)
print("\nPER IMAGE")
print(per_image)
print("\nRELIABILITY")
print(reliability["bins_df"])
print("DECE:", reliability["dece"])
print("\nMONOTONICITY")
print(monotonicity["monotonic"], monotonicity["inversion_count"], monotonicity["inversion_pairs"])
