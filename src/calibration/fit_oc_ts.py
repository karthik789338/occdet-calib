from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics.reliability import summarize_reliability


EPS = 1e-6


def bucket_from_occlusion(x: float) -> str:
    if x <= 0.2:
        return "high_visibility"
    if x <= 0.4:
        return "medium_visibility"
    if x <= 0.6:
        return "low_visibility"
    return "very_low_visibility"


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def bce_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.clip(probs, EPS, 1.0 - EPS)
    return float(-(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)).mean())


def fit_temperature(scores: np.ndarray, labels: np.ndarray) -> float:
    logits = to_logit(scores)

    grid = np.concatenate([
        np.linspace(0.25, 1.5, 26),
        np.linspace(1.6, 5.0, 18),
        np.linspace(6.0, 10.0, 9),
    ])

    best_t = 1.0
    best_loss = float("inf")

    for t in grid:
        probs = sigmoid(logits / t)
        loss = bce_loss(probs, labels)
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)

    return best_t


def compute_dece(df: pd.DataFrame, confidence_col: str, correctness_col: str = "correct", n_bins: int = 15) -> float:
    return float(
        summarize_reliability(
            df,
            confidence_col=confidence_col,
            correctness_col=correctness_col,
            n_bins=n_bins,
        )["dece"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit global TS and OC-TS using visibility buckets.")
    parser.add_argument("--matched_pred_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--min_bucket_size", type=int, default=200)
    args = parser.parse_args()

    matched = pd.read_csv(args.matched_pred_path)
    gt = pd.read_csv(args.gt_path)

    gt_small = gt[["image_key", "estimated_occlusion"]].drop_duplicates("image_key").copy()
    gt_small["visibility_bucket"] = gt_small["estimated_occlusion"].map(bucket_from_occlusion)

    df = matched.merge(gt_small, on="image_key", how="left")
    if df["visibility_bucket"].isna().any():
        raise RuntimeError("Some predictions could not be assigned a visibility bucket.")

    df["score"] = df["score"].clip(EPS, 1.0 - EPS)
    labels = df["correct"].astype(float).to_numpy()
    scores = df["score"].astype(float).to_numpy()

    global_t = fit_temperature(scores, labels)
    df["score_global_ts"] = sigmoid(to_logit(scores) / global_t)

    bucket_temps = {}
    for bucket, sub in df.groupby("visibility_bucket"):
        if len(sub) < args.min_bucket_size:
            bucket_temps[bucket] = global_t
            continue
        bucket_temps[bucket] = fit_temperature(
            sub["score"].astype(float).to_numpy(),
            sub["correct"].astype(float).to_numpy(),
        )

    df["bucket_temp"] = df["visibility_bucket"].map(bucket_temps).astype(float)
    df["score_oc_ts"] = sigmoid(to_logit(df["score"].to_numpy()) / df["bucket_temp"].to_numpy())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params_rows = [{"scope": "global", "bucket": "all", "temperature": global_t}]
    params_rows += [{"scope": "bucket", "bucket": k, "temperature": v} for k, v in bucket_temps.items()]
    pd.DataFrame(params_rows).to_csv(out_dir / "temperatures.csv", index=False)

    df.to_csv(out_dir / "matched_with_oc_ts.csv", index=False)

    report_rows = []

    for mode, col in [
        ("raw", "score"),
        ("global_ts", "score_global_ts"),
        ("oc_ts", "score_oc_ts"),
    ]:
        report_rows.append(
            {
                "scope": "overall",
                "bucket": "all",
                "mode": mode,
                "num_predictions": len(df),
                "dece": compute_dece(df, col),
                "avg_score": float(df[col].mean()),
                "avg_correct": float(df["correct"].mean()),
            }
        )

        for bucket, sub in df.groupby("visibility_bucket"):
            report_rows.append(
                {
                    "scope": "bucket",
                    "bucket": bucket,
                    "mode": mode,
                    "num_predictions": len(sub),
                    "dece": compute_dece(sub, col),
                    "avg_score": float(sub[col].mean()),
                    "avg_correct": float(sub["correct"].mean()),
                }
            )

    report = pd.DataFrame(report_rows)
    report.to_csv(out_dir / "dece_report.csv", index=False)

    print("Saved:", out_dir / "temperatures.csv")
    print("Saved:", out_dir / "matched_with_oc_ts.csv")
    print("Saved:", out_dir / "dece_report.csv")
    print(report)


if __name__ == "__main__":
    main()
