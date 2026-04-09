from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VEHICLE_SET = {"bicycle", "motorcycle", "car", "bus", "truck"}
PERSON_SET = {"person"}
SIGN_SET = {"stop sign"}


def normalize_name(x: str) -> str:
    return str(x).strip().lower()


def semantic_group(name: str) -> str:
    name = normalize_name(name)
    if name in PERSON_SET:
        return "person"
    if name in VEHICLE_SET:
        return "vehicle"
    if name in SIGN_SET:
        return "traffic_sign"
    return "other"


def relation_type(target_class: str, occluder_class: str) -> str:
    t = normalize_name(target_class)
    o = normalize_name(occluder_class)

    if pd.isna(occluder_class):
        return "none"
    if t == o:
        return "same_class"

    tg = semantic_group(t)
    og = semantic_group(o)

    if tg == og:
        return "same_group"

    return "different_group"


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else float(a / b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize matched predictions by occluder semantics.")
    parser.add_argument("--matched_pred_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    matched = pd.read_csv(args.matched_pred_path)
    gt = pd.read_csv(args.gt_path)

    if "occluder_class_name" not in gt.columns:
        raise KeyError("GT table must contain occluder_class_name")
    if "class_name" not in gt.columns:
        raise KeyError("GT table must contain class_name")

    gt = gt.copy()
    gt["semantic_relation"] = gt.apply(
        lambda r: relation_type(r["class_name"], r["occluder_class_name"]),
        axis=1,
    )

    rows = []

    for rel, gt_subset in gt.groupby("semantic_relation"):
        image_keys = set(gt_subset["image_key"].astype(str).tolist())
        pred_subset = matched[matched["image_key"].astype(str).isin(image_keys)].copy()

        tp = int((pred_subset["correct"] == 1).sum())
        fp = int((pred_subset["correct"] == 0).sum())
        total_gt = int(len(gt_subset))
        fn = max(0, total_gt - tp)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, total_gt)
        f1 = safe_div(2 * precision * recall, precision + recall)

        rows.append(
            {
                "semantic_relation": rel,
                "num_predictions": len(pred_subset),
                "num_ground_truth": total_gt,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    out_df = pd.DataFrame(rows).sort_values("semantic_relation")
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(out_df)


if __name__ == "__main__":
    main()
