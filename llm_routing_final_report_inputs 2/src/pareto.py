from __future__ import annotations

import pandas as pd


def pareto_frontier(df, cost_col="avg_cost", quality_col="avg_quality"):
    """Return rows not dominated by another row with lower cost and higher quality."""
    if df.empty:
        return df.copy()
    keep = []
    records = df.to_dict("records")
    for row in records:
        dominated = False
        for other in records:
            no_worse = other[cost_col] <= row[cost_col] and other[quality_col] >= row[quality_col]
            strictly_better = (
                other[cost_col] < row[cost_col] or other[quality_col] > row[quality_col]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(row)
    return pd.DataFrame(keep).sort_values([cost_col, quality_col], ascending=[True, False])
