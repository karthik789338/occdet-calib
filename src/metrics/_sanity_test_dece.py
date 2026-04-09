from __future__ import annotations

import pandas as pd

from src.metrics.reliability import summarize_reliability


df = pd.DataFrame(
    {
        "score": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        "correct": [0, 0, 1, 1, 1, 0],
    }
)

result = summarize_reliability(df, n_bins=3)
print("DECE:", result["dece"])
print(result["bins_df"])
