from __future__ import annotations

import pandas as pd

from src.metrics.matching import match_predictions_to_ground_truth, summarize_matches


pred_df = pd.DataFrame(
    [
        {
            "image_key": "img1.jpg",
            "class_id": 15,
            "score": 0.95,
            "x1": 10,
            "y1": 10,
            "x2": 50,
            "y2": 50,
        },
        {
            "image_key": "img1.jpg",
            "class_id": 15,
            "score": 0.60,
            "x1": 12,
            "y1": 12,
            "x2": 52,
            "y2": 52,
        },
        {
            "image_key": "img1.jpg",
            "class_id": 16,
            "score": 0.80,
            "x1": 100,
            "y1": 100,
            "x2": 140,
            "y2": 140,
        },
    ]
)

gt_df = pd.DataFrame(
    [
        {
            "image_key": "img1.jpg",
            "class_id": 15,
            "x1": 11,
            "y1": 11,
            "x2": 51,
            "y2": 51,
        }
    ]
)

matched = match_predictions_to_ground_truth(
    pred_df=pred_df,
    gt_df=gt_df,
    image_key_col="image_key",
    class_id_col="class_id",
    score_col="score",
    iou_threshold=0.5,
    classwise=True,
)

print(matched)
print(summarize_matches(matched))
