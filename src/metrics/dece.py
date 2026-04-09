from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class CalibrationBin:
    bin_index: int
    lower: float
    upper: float
    count: int
    avg_confidence: float
    avg_accuracy: float
    abs_gap: float


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


def compute_dece(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, object]:
    """
    Detection Expected Calibration Error (binning-based).

    Parameters
    ----------
    confidences : np.ndarray
        1D array of detector confidence scores in [0, 1].
    correctness : np.ndarray
        1D binary array where 1 means a correct detection and 0 means incorrect.
    n_bins : int
        Number of equal-width bins over [0, 1].

    Returns
    -------
    dict with:
        - dece
        - bins_df
        - total_count
    """
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=int)

    _validate_inputs(confidences, correctness)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # include 1.0 in the last bin
    bin_ids = np.digitize(confidences, bin_edges[1:-1], right=False)

    bin_rows = []
    total_count = len(confidences)
    dece = 0.0

    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())

        lower = float(bin_edges[b])
        upper = float(bin_edges[b + 1])

        if count == 0:
            bin_rows.append(
                CalibrationBin(
                    bin_index=b,
                    lower=lower,
                    upper=upper,
                    count=0,
                    avg_confidence=np.nan,
                    avg_accuracy=np.nan,
                    abs_gap=np.nan,
                )
            )
            continue

        avg_conf = float(confidences[mask].mean())
        avg_acc = float(correctness[mask].mean())
        abs_gap = abs(avg_acc - avg_conf)

        dece += (count / total_count) * abs_gap

        bin_rows.append(
            CalibrationBin(
                bin_index=b,
                lower=lower,
                upper=upper,
                count=count,
                avg_confidence=avg_conf,
                avg_accuracy=avg_acc,
                abs_gap=abs_gap,
            )
        )

    bins_df = pd.DataFrame([row.__dict__ for row in bin_rows])

    return {
        "dece": float(dece),
        "bins_df": bins_df,
        "total_count": int(total_count),
    }


def compute_dece_from_dataframe(
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

    return compute_dece(
        confidences=confidences,
        correctness=correctness,
        n_bins=n_bins,
    )
