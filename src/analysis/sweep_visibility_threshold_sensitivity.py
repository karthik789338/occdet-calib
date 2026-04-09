from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.metrics.reliability import summarize_reliability


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else float(a / b)


def summarize_subset(pred_subset: pd.DataFrame, gt_subset: pd.DataFrame) -> dict:
    tp = int((pred_subset["correct"] == 1).sum())
    fp = int((pred_subset["correct"] == 0).sum())
    total_gt = int(len(gt_subset))
    fn = max(0, total_gt - tp)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, total_gt)
    f1 = safe_div(2 * precision * recall, precision + recall)

    dece = None
    if len(pred_subset) > 0:
        dece = float(
            summarize_reliability(
                pred_subset,
                confidence_col="score",
                correctness_col="correct",
                n_bins=15,
            )["dece"]
        )

    return {
        "num_predictions": len(pred_subset),
        "num_ground_truth": total_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dece": dece,
        "avg_score": float(pred_subset["score"].mean()) if len(pred_subset) else None,
        "avg_correct": float(pred_subset["correct"].mean()) if len(pred_subset) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep visibility thresholds on overlap variants.")
    parser.add_argument("--matched_pred_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    args = parser.parse_args()

    matched = pd.read_csv(args.matched_pred_path)
    gt = pd.read_csv(args.gt_path)

    if "estimated_occlusion" not in gt.columns:
        raise KeyError("GT must contain estimated_occlusion")

    rows = []

    for thr in args.thresholds:
        visible_gt = gt[gt["estimated_occlusion"] <= thr].copy()
        occluded_gt = gt[gt["estimated_occlusion"] > thr].copy()

        visible_keys = set(visible_gt["image_key"].astype(str).tolist())
        occluded_keys = set(occluded_gt["image_key"].astype(str).tolist())

        visible_pred = matched[matched["image_key"].astype(str).isin(visible_keys)].copy()
        occluded_pred = matched[matched["image_key"].astype(str).isin(occluded_keys)].copy()

        vis_stats = summarize_subset(visible_pred, visible_gt)
        occ_stats = summarize_subset(occluded_pred, occluded_gt)

        rows.append({"threshold": thr, "subset": "visible_leq_thr", **vis_stats})
        rows.append({"threshold": thr, "subset": "occluded_gt_thr", **occ_stats})

    out = pd.DataFrame(rows)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(out)


if __name__ == "__main__":
    main()
