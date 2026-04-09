from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.metrics.map_eval import summarize_detection_metrics
from src.metrics.reliability import summarize_reliability
from src.metrics.monotonicity import compute_monotonicity_from_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize overall metrics for thresholded matched predictions.")
    parser.add_argument("--matched_pred_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--n_bins", type=int, default=15)
    args = parser.parse_args()

    matched = pd.read_csv(args.matched_pred_path)
    gt = pd.read_csv(args.gt_path)

    overall = summarize_detection_metrics(matched, gt)

    if len(matched) > 0:
        rel = summarize_reliability(
            matched,
            confidence_col="score",
            correctness_col="correct",
            n_bins=args.n_bins,
        )
        mono = compute_monotonicity_from_dataframe(
            matched,
            confidence_col="score",
            correctness_col="correct",
            n_bins=args.n_bins,
        )
        dece = rel["dece"]
        monotonic = mono["monotonic"]
        inversion_count = mono["inversion_count"]
        avg_score = float(matched["score"].mean())
        avg_correct = float(matched["correct"].mean())
    else:
        dece = None
        monotonic = None
        inversion_count = None
        avg_score = None
        avg_correct = None

    out = pd.DataFrame(
        [
            {
                **overall,
                "dece": dece,
                "monotonic": monotonic,
                "inversion_count": inversion_count,
                "avg_score": avg_score,
                "avg_correct": avg_correct,
            }
        ]
    )

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(out)


if __name__ == "__main__":
    main()
