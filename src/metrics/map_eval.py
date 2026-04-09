from __future__ import annotations

from typing import Dict, List

import pandas as pd


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def summarize_detection_metrics(
    matched_pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    *,
    correct_col: str = "correct",
) -> Dict[str, float]:
    """
    Basic detection summary after one-to-one matching has already been done.

    This is not COCO mAP.
    It is a clean evaluation scaffold for calibration work.
    """
    if correct_col not in matched_pred_df.columns:
        raise KeyError(f"Missing column: {correct_col}")

    tp = int((matched_pred_df[correct_col] == 1).sum())
    fp = int((matched_pred_df[correct_col] == 0).sum())
    total_gt = int(len(gt_df))
    fn = max(0, total_gt - tp)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, total_gt)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "num_predictions": int(len(matched_pred_df)),
        "num_ground_truth": total_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def summarize_per_class_detection_metrics(
    matched_pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    *,
    class_id_col: str = "class_id",
    correct_col: str = "correct",
) -> pd.DataFrame:
    if class_id_col not in matched_pred_df.columns:
        raise KeyError(f"Missing prediction class column: {class_id_col}")
    if class_id_col not in gt_df.columns:
        raise KeyError(f"Missing GT class column: {class_id_col}")
    if correct_col not in matched_pred_df.columns:
        raise KeyError(f"Missing prediction correctness column: {correct_col}")

    pred_classes = set(matched_pred_df[class_id_col].dropna().tolist())
    gt_classes = set(gt_df[class_id_col].dropna().tolist())
    all_classes = sorted(pred_classes | gt_classes, key=lambda x: str(x))

    rows: List[Dict[str, float]] = []

    for class_value in all_classes:
        pred_subset = matched_pred_df[matched_pred_df[class_id_col] == class_value]
        gt_subset = gt_df[gt_df[class_id_col] == class_value]

        tp = int((pred_subset[correct_col] == 1).sum())
        fp = int((pred_subset[correct_col] == 0).sum())
        total_gt = int(len(gt_subset))
        fn = max(0, total_gt - tp)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, total_gt)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        rows.append(
            {
                "class_id": class_value,
                "num_predictions": int(len(pred_subset)),
                "num_ground_truth": total_gt,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return pd.DataFrame(rows)


def summarize_per_image_detection_metrics(
    matched_pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    *,
    image_key_col: str = "image_key",
    correct_col: str = "correct",
) -> pd.DataFrame:
    if image_key_col not in matched_pred_df.columns:
        raise KeyError(f"Missing prediction image key column: {image_key_col}")
    if image_key_col not in gt_df.columns:
        raise KeyError(f"Missing GT image key column: {image_key_col}")
    if correct_col not in matched_pred_df.columns:
        raise KeyError(f"Missing prediction correctness column: {correct_col}")

    pred_images = set(matched_pred_df[image_key_col].dropna().astype(str).tolist())
    gt_images = set(gt_df[image_key_col].dropna().astype(str).tolist())
    all_images = sorted(pred_images | gt_images)

    rows: List[Dict[str, float]] = []

    for image_key in all_images:
        pred_subset = matched_pred_df[matched_pred_df[image_key_col] == image_key]
        gt_subset = gt_df[gt_df[image_key_col] == image_key]

        tp = int((pred_subset[correct_col] == 1).sum())
        fp = int((pred_subset[correct_col] == 0).sum())
        total_gt = int(len(gt_subset))
        fn = max(0, total_gt - tp)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, total_gt)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        rows.append(
            {
                "image_key": image_key,
                "num_predictions": int(len(pred_subset)),
                "num_ground_truth": total_gt,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return pd.DataFrame(rows)
