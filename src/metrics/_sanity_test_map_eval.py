from __future__ import annotations

import pandas as pd

from src.metrics.map_eval import (
    summarize_detection_metrics,
    summarize_per_class_detection_metrics,
    summarize_per_image_detection_metrics,
)


matched_pred_df = pd.DataFrame(
    [
        {"image_key": "img1.jpg", "class_id": 15, "correct": 1},
        {"image_key": "img1.jpg", "class_id": 15, "correct": 0},
        {"image_key": "img1.jpg", "class_id": 16, "correct": 0},
    ]
)

gt_df = pd.DataFrame(
    [
        {"image_key": "img1.jpg", "class_id": 15},
        {"image_key": "img1.jpg", "class_id": 17},
    ]
)

overall = summarize_detection_metrics(matched_pred_df, gt_df)
per_class = summarize_per_class_detection_metrics(matched_pred_df, gt_df)
per_image = summarize_per_image_detection_metrics(matched_pred_df, gt_df)

print("OVERALL")
print(overall)
print("\nPER CLASS")
print(per_class)
print("\nPER IMAGE")
print(per_image)
