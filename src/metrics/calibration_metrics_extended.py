from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass
class CalibrationStats:
    bin_index: int
    lower: float
    upper: float
    count: int
    avg_confidence: Optional[float]
    avg_accuracy: Optional[float]
    abs_gap: Optional[float]


def _safe_mean(x: np.ndarray) -> Optional[float]:
    if len(x) == 0:
        return None
    return float(np.mean(x))


def _stats_from_edges(scores: np.ndarray, labels: np.ndarray, edges: np.ndarray) -> pd.DataFrame:
    rows = []
    n_bins = len(edges) - 1

    for i in range(n_bins):
        lower = float(edges[i])
        upper = float(edges[i + 1])

        if i == n_bins - 1:
            mask = (scores >= lower) & (scores <= upper)
        else:
            mask = (scores >= lower) & (scores < upper)

        sub_scores = scores[mask]
        sub_labels = labels[mask]

        avg_conf = _safe_mean(sub_scores)
        avg_acc = _safe_mean(sub_labels)
        abs_gap = None if avg_conf is None or avg_acc is None else abs(avg_conf - avg_acc)

        rows.append(
            {
                "bin_index": i,
                "lower": lower,
                "upper": upper,
                "count": int(mask.sum()),
                "avg_confidence": avg_conf,
                "avg_accuracy": avg_acc,
                "abs_gap": abs_gap,
            }
        )

    return pd.DataFrame(rows)


def uniform_bin_stats(
    df: pd.DataFrame,
    confidence_col: str = "score",
    correctness_col: str = "correct",
    n_bins: int = 15,
) -> pd.DataFrame:
    scores = df[confidence_col].astype(float).to_numpy()
    labels = df[correctness_col].astype(float).to_numpy()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return _stats_from_edges(scores, labels, edges)


def adaptive_bin_stats(
    df: pd.DataFrame,
    confidence_col: str = "score",
    correctness_col: str = "correct",
    n_bins: int = 15,
) -> pd.DataFrame:
    scores = df[confidence_col].astype(float).to_numpy()
    labels = df[correctness_col].astype(float).to_numpy()

    if len(scores) == 0:
        return pd.DataFrame(
            columns=["bin_index", "lower", "upper", "count", "avg_confidence", "avg_accuracy", "abs_gap"]
        )

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(scores, quantiles)
    edges[0] = 0.0
    edges[-1] = 1.0

    # enforce strictly non-decreasing edges
    edges = np.maximum.accumulate(edges)

    # if all scores collapse, create one effective bin
    if np.allclose(edges, edges[0]):
        edges = np.array([0.0, 1.0], dtype=float)

    return _stats_from_edges(scores, labels, edges)


def ece_from_stats(stats: pd.DataFrame) -> float:
    if len(stats) == 0:
        return 0.0
    total = stats["count"].sum()
    if total == 0:
        return 0.0
    s = 0.0
    for _, row in stats.iterrows():
        if pd.isna(row["abs_gap"]):
            continue
        s += (row["count"] / total) * row["abs_gap"]
    return float(s)


def mce_from_stats(stats: pd.DataFrame) -> float:
    if len(stats) == 0:
        return 0.0
    vals = stats["abs_gap"].dropna().tolist()
    if not vals:
        return 0.0
    return float(max(vals))


def brier_score(
    df: pd.DataFrame,
    confidence_col: str = "score",
    correctness_col: str = "correct",
) -> float:
    s = df[confidence_col].astype(float).to_numpy()
    y = df[correctness_col].astype(float).to_numpy()
    if len(s) == 0:
        return 0.0
    return float(np.mean((s - y) ** 2))


def nll_score(
    df: pd.DataFrame,
    confidence_col: str = "score",
    correctness_col: str = "correct",
) -> float:
    s = np.clip(df[confidence_col].astype(float).to_numpy(), EPS, 1.0 - EPS)
    y = df[correctness_col].astype(float).to_numpy()
    if len(s) == 0:
        return 0.0
    return float(-(y * np.log(s) + (1.0 - y) * np.log(1.0 - s)).mean())


def summarize_calibration_metrics(
    df: pd.DataFrame,
    confidence_col: str = "score",
    correctness_col: str = "correct",
    n_bins: int = 15,
) -> dict:
    uniform = uniform_bin_stats(df, confidence_col, correctness_col, n_bins=n_bins)
    adaptive = adaptive_bin_stats(df, confidence_col, correctness_col, n_bins=n_bins)

    return {
        "num_predictions": int(len(df)),
        "d_ece": ece_from_stats(uniform),
        "ace": ece_from_stats(adaptive),
        "mce": mce_from_stats(uniform),
        "brier": brier_score(df, confidence_col, correctness_col),
        "nll": nll_score(df, confidence_col, correctness_col),
    }


def classwise_calibration_table(
    df: pd.DataFrame,
    class_col: str = "class_name",
    confidence_col: str = "score",
    correctness_col: str = "correct",
    n_bins: int = 15,
    min_support: int = 25,
) -> pd.DataFrame:
    rows = []
    for class_value, sub in df.groupby(class_col):
        support = len(sub)
        if support < min_support:
            continue

        uniform = uniform_bin_stats(sub, confidence_col, correctness_col, n_bins=n_bins)
        adaptive = adaptive_bin_stats(sub, confidence_col, correctness_col, n_bins=n_bins)

        rows.append(
            {
                "class_value": class_value,
                "support": int(support),
                "d_ece": ece_from_stats(uniform),
                "ace": ece_from_stats(adaptive),
                "mce": mce_from_stats(uniform),
                "brier": brier_score(sub, confidence_col, correctness_col),
                "nll": nll_score(sub, confidence_col, correctness_col),
                "avg_score": float(sub[confidence_col].mean()),
                "avg_correct": float(sub[correctness_col].mean()),
            }
        )

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["support", "class_value"], ascending=[False, True]).reset_index(drop=True)
    return out


def laece_from_classwise(classwise_df: pd.DataFrame) -> float:
    if len(classwise_df) == 0:
        return 0.0
    total = classwise_df["support"].sum()
    if total == 0:
        return 0.0
    return float((classwise_df["support"] * classwise_df["d_ece"]).sum() / total)
