from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.metrics.reliability import summarize_reliability
from src.metrics.monotonicity import compute_monotonicity_from_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DECE by occlusion level.")
    parser.add_argument("--matched_pred_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--occlusion_col", type=str, default="nominal_occlusion")
    parser.add_argument("--n_bins", type=int, default=15)
    args = parser.parse_args()

    matched = pd.read_csv(args.matched_pred_path)
    gt = pd.read_csv(args.gt_path)

    rows = []

    for occ_value, gt_subset in gt.groupby(args.occlusion_col):
        image_keys = set(gt_subset["image_key"].astype(str).tolist())
        pred_subset = matched[matched["image_key"].astype(str).isin(image_keys)].copy()

        if len(pred_subset) == 0:
            rows.append(
                {
                    "occlusion_level": occ_value,
                    "num_predictions": 0,
                    "dece": None,
                    "monotonic": None,
                    "inversion_count": None,
                    "avg_score": None,
                    "avg_correct": None,
                }
            )
            continue

        reliability = summarize_reliability(
            pred_subset,
            confidence_col="score",
            correctness_col="correct",
            n_bins=args.n_bins,
        )
        monotonicity = compute_monotonicity_from_dataframe(
            pred_subset,
            confidence_col="score",
            correctness_col="correct",
            n_bins=args.n_bins,
        )

        rows.append(
            {
                "occlusion_level": occ_value,
                "num_predictions": len(pred_subset),
                "dece": reliability["dece"],
                "monotonic": monotonicity["monotonic"],
                "inversion_count": monotonicity["inversion_count"],
                "avg_score": float(pred_subset["score"].mean()),
                "avg_correct": float(pred_subset["correct"].mean()),
            }
        )

    out_df = pd.DataFrame(rows).sort_values("occlusion_level")
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(out_df)


if __name__ == "__main__":
    main()
