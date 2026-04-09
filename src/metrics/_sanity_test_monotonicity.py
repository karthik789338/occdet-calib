from __future__ import annotations

import pandas as pd

from src.metrics.monotonicity import compute_monotonicity_from_dataframe


df = pd.DataFrame(
    {
        "score": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        "correct": [0, 0, 1, 1, 1, 0],
    }
)

result = compute_monotonicity_from_dataframe(df, n_bins=3)
print("Monotonic:", result["monotonic"])
print("Inversion count:", result["inversion_count"])
print("Inversion pairs:", result["inversion_pairs"])
print(result["bins_df"])
