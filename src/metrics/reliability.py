from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.metrics.dece import compute_dece_from_dataframe


def build_reliability_dataframe(
    df: pd.DataFrame,
    confidence_col: str = "score",
    correctness_col: str = "correct",
    n_bins: int = 15,
) -> pd.DataFrame:
    result = compute_dece_from_dataframe(
        df=df,
        confidence_col=confidence_col,
        correctness_col=correctness_col,
        n_bins=n_bins,
    )
    return result["bins_df"]


def plot_reliability_diagram(
    bins_df: pd.DataFrame,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None,
) -> None:
    plot_df = bins_df.dropna(subset=["avg_confidence", "avg_accuracy"]).copy()

    if plot_df.empty:
        raise ValueError("No populated bins to plot.")

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.scatter(plot_df["avg_confidence"], plot_df["avg_accuracy"])
    plt.plot(plot_df["avg_confidence"], plot_df["avg_accuracy"])

    plt.xlabel("Average confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def summarize_reliability(
    df: pd.DataFrame,
    confidence_col: str = "score",
    correctness_col: str = "correct",
    n_bins: int = 15,
) -> Dict[str, object]:
    result = compute_dece_from_dataframe(
        df=df,
        confidence_col=confidence_col,
        correctness_col=correctness_col,
        n_bins=n_bins,
    )
    return {
        "dece": result["dece"],
        "bins_df": result["bins_df"],
        "total_count": result["total_count"],
    }
