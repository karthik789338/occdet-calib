from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _validate_inputs(confidences: np.ndarray, correctness: np.ndarray) -> None:
    if confidences.ndim != 1 or correctness.ndim != 1:
        raise ValueError("confidences and correctness must be 1D arrays.")
    if len(confidences) != len(correctness):
        raise ValueError("confidences and correctness must have the same length.")
    if len(confidences) == 0:
        raise ValueError("Empty inputs are not allowed.")
    if np.any(confidences < 0.0) or np.any(confidences > 1.0):
        raise ValueError("confidences must be in [0, 1].")
    if not np.all(np.isin(correctness, [0, 1])):
        raise ValueError("correctness must contain only 0/1 values.")


def compute_monotonicity_summary(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, object]:
    """
    Measures whether empirical accuracy tends to increase with confidence.

    Returns:
        - monotonic: whether non-empty bins are non-decreasing in accuracy
        - inversion_count: number of downward steps between consecutive non-empty bins
        - inversion_pairs: list of bin-index pairs where accuracy decreases
        - bins_df: per-bin summary
    """
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=int)
    _validate_inputs(confidences, correctness)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidences, bin_edges[1:-1], right=False)

    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())

        lower = float(bin_edges[b])
        upper = float(bin_edges[b + 1])

        if count == 0:
            rows.append(
                {
                    "bin_index": b,
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "avg_confidence": np.nan,
                    "avg_accuracy": np.nan,
                }
            )
            continue

        rows.append(
            {
                "bin_index": b,
                "lower": lower,
                "upper": upper,
                "count": count,
                "avg_confidence": float(confidences[mask].mean()),
                "avg_accuracy": float(correctness[mask].mean()),
            }
        )

    bins_df = pd.DataFrame(rows)
    non_empty = bins_df.dropna(subset=["avg_accuracy"]).copy()

    inversion_pairs: List[tuple[int, int]] = []
    prev_acc = None
    prev_bin = None

    for _, row in non_empty.iterrows():
        curr_acc = float(row["avg_accuracy"])
        curr_bin = int(row["bin_index"])

        if prev_acc is not None and curr_acc < prev_acc:
            inversion_pairs.append((prev_bin, curr_bin))

        prev_acc = curr_acc
        prev_bin = curr_bin

    inversion_count = len(inversion_pairs)
    monotonic = inversion_count == 0

    return {
        "monotonic": monotonic,
        "inversion_count": inversion_count,
        "inversion_pairs": inversion_pairs,
        "bins_df": bins_df,
    }


def compute_monotonicity_from_dataframe(
    df: pd.DataFrame,
    confidence_col: str = "score",
    correctness_col: str = "correct",
    n_bins: int = 15,
) -> Dict[str, object]:
    if confidence_col not in df.columns:
        raise KeyError(f"Missing column: {confidence_col}")
    if correctness_col not in df.columns:
        raise KeyError(f"Missing column: {correctness_col}")

    confidences = df[confidence_col].to_numpy(dtype=float)
    correctness = df[correctness_col].to_numpy(dtype=int)

    return compute_monotonicity_summary(
        confidences=confidences,
        correctness=correctness,
        n_bins=n_bins,
    )
