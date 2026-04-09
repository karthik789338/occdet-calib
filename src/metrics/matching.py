from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REQUIRED_PRED_COLS = {"score", "x1", "y1", "x2", "y2"}
REQUIRED_GT_COLS = {"x1", "y1", "x2", "y2"}


def _validate_columns(df: pd.DataFrame, required: set[str], df_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{df_name} is missing required columns: {sorted(missing)}")


def add_image_key_from_path(
    df: pd.DataFrame,
    image_path_col: str = "image_path",
    output_col: str = "image_key",
) -> pd.DataFrame:
    if image_path_col not in df.columns:
        raise KeyError(f"Missing column: {image_path_col}")

    out = df.copy()
    out[output_col] = out[image_path_col].astype(str).map(lambda p: Path(p).name)
    return out


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0

    return float(inter_area / union)


def _best_gt_match(
    pred_box: np.ndarray,
    gt_subset: pd.DataFrame,
    used_gt_indices: set[int],
) -> Tuple[Optional[int], float]:
    best_gt_idx = None
    best_iou = 0.0

    for gt_idx, gt_row in gt_subset.iterrows():
        if gt_idx in used_gt_indices:
            continue

        gt_box = gt_row[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        iou = compute_iou(pred_box, gt_box)

        if iou > best_iou:
            best_iou = iou
            best_gt_idx = gt_idx

    return best_gt_idx, best_iou


def match_predictions_to_ground_truth(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    *,
    image_key_col: str = "image_key",
    class_id_col: str = "class_id",
    score_col: str = "score",
    iou_threshold: float = 0.5,
    classwise: bool = True,
) -> pd.DataFrame:
    _validate_columns(pred_df, REQUIRED_PRED_COLS, "pred_df")
    _validate_columns(gt_df, REQUIRED_GT_COLS, "gt_df")

    if image_key_col not in pred_df.columns:
        raise KeyError(f"pred_df missing image key column: {image_key_col}")
    if image_key_col not in gt_df.columns:
        raise KeyError(f"gt_df missing image key column: {image_key_col}")
    if class_id_col not in pred_df.columns or class_id_col not in gt_df.columns:
        raise KeyError(f"Both pred_df and gt_df must contain class column: {class_id_col}")
    if score_col not in pred_df.columns:
        raise KeyError(f"pred_df missing score column: {score_col}")

    out = pred_df.copy()
    out["correct"] = 0
    out["matched_iou"] = 0.0
    out["matched_gt_index"] = pd.NA
    out["tp"] = 0
    out["fp"] = 1

    for image_key, pred_img_df in out.groupby(image_key_col):
        gt_img_df = gt_df[gt_df[image_key_col] == image_key]

        if gt_img_df.empty:
            continue

        if classwise:
            pred_classes = pred_img_df[class_id_col].dropna().tolist()
            gt_classes = gt_img_df[class_id_col].dropna().tolist()
            class_values = sorted(set(pred_classes + gt_classes), key=lambda x: str(x))
        else:
            class_values = [None]

        for class_value in class_values:
            if classwise:
                pred_subset = pred_img_df[pred_img_df[class_id_col] == class_value]
                gt_subset = gt_img_df[gt_img_df[class_id_col] == class_value]
            else:
                pred_subset = pred_img_df
                gt_subset = gt_img_df

            if pred_subset.empty:
                continue

            used_gt_indices: set[int] = set()
            pred_subset = pred_subset.sort_values(score_col, ascending=False)

            for pred_idx, pred_row in pred_subset.iterrows():
                pred_box = pred_row[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
                best_gt_idx, best_iou = _best_gt_match(
                    pred_box=pred_box,
                    gt_subset=gt_subset,
                    used_gt_indices=used_gt_indices,
                )

                out.at[pred_idx, "matched_iou"] = float(best_iou)

                if best_gt_idx is not None and best_iou >= iou_threshold:
                    used_gt_indices.add(best_gt_idx)
                    out.at[pred_idx, "correct"] = 1
                    out.at[pred_idx, "matched_gt_index"] = int(best_gt_idx)
                    out.at[pred_idx, "tp"] = 1
                    out.at[pred_idx, "fp"] = 0

    return out


def summarize_matches(
    matched_pred_df: pd.DataFrame,
    correct_col: str = "correct",
) -> Dict[str, int]:
    if correct_col not in matched_pred_df.columns:
        raise KeyError(f"Missing column: {correct_col}")

    tp = int((matched_pred_df[correct_col] == 1).sum())
    fp = int((matched_pred_df[correct_col] == 0).sum())

    return {
        "num_predictions": int(len(matched_pred_df)),
        "true_positives": tp,
        "false_positives": fp,
    }
