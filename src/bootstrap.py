import numpy as np, pandas as pd
from numpy.random import default_rng


#Bootstrap CHANGE B, is 10 (low) for convenience (runs faster)
def bootstrap_ci(df, stat_fn, group_col='exchange_id', B=10, alpha=0.05, rng = None):
    """
    Bootstrap a statistic with optional grouping. If group_col exists and has
    non-null values, resample groups (to respect exchange structure).
    stat_fn: function that takes a DataFrame sample and returns a float.
    """
    
    vals = []
    if group_col in df.columns and df[group_col].notna().any():
        groups = df[group_col].dropna().unique()
        for _ in range(B):
            samp_groups = rng.choice(groups, size=len(groups), replace=True)
            samp = pd.concat([df[df[group_col] == g] for g in samp_groups], ignore_index=True)
            vals.append(float(stat_fn(samp)))
    else:
        idx = np.arange(len(df))
        for _ in range(B):
            samp = df.iloc[rng.choice(idx, size=len(idx), replace=True)]
            vals.append(float(stat_fn(samp)))
    vals = np.array(vals, dtype=float)
    lo, hi = np.quantile(vals, [alpha/2, 1 - alpha/2])
    return float(vals.mean()), float(lo), float(hi)