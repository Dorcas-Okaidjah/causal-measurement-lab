"""Validating a proxy metric (leading indicator).

Teams often judge a campaign on a fast, frequent metric (website visits)
because the outcome they care about (spend) is slow, rare or noisy. That only
works if a campaign's effect on the proxy predicts its effect on the outcome.

A proxy can be correlated with the outcome across customers (people who visit
spend more) and still be a poor guide to effects. So we test what matters:
across many segments, do the effects move together?

Noise trap: in one sample, effects on visit and spend share the same random
noise and look correlated even if the true effects are not. We estimate the
visit effect on one random half of customers and the spend effect on the other
half, so their noise is independent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C


def segment_effects(df: pd.DataFrame, treat: str, metrics: list[str],
                    segments: list[str], min_n: int = 200) -> pd.DataFrame:
    """Treatment effect on each metric within each segment."""
    rows = []
    for key, g in df.groupby(segments, observed=True):
        t = g[treat].astype(bool)
        if t.sum() < min_n or (~t).sum() < min_n:
            continue
        row = dict(zip(segments, key if isinstance(key, tuple) else (key,)))
        row["n"] = len(g)
        for m in metrics:
            row[f"effect_{m}"] = g.loc[t, m].mean() - g.loc[~t, m].mean()
        rows.append(row)
    cols = [*segments, "n", *(f"effect_{m}" for m in metrics)]
    return pd.DataFrame(rows, columns=cols)


def proxy_validation(df: pd.DataFrame, treat: str, proxy: str, outcome: str,
                     segments: list[str], seed=C.RANDOM_STATE, n_splits: int = 50) -> dict:
    """Split-sample check of whether effects on `proxy` track effects on `outcome`.

    Repeats the random half-split n_splits times and reports the median
    correlation, so the answer doesn't hinge on one lucky split.
    """
    rng = np.random.default_rng(seed)
    corrs, slopes = [], []
    last = None
    for _ in range(n_splits):
        half = rng.random(len(df)) < 0.5
        a = segment_effects(df[half], treat, [proxy], segments)
        b = segment_effects(df[~half], treat, [outcome], segments)
        both = a.merge(b, on=segments, suffixes=("_a", "_b"))
        if len(both) < 3:            # too few segments to say anything
            continue
        x, y = both[f"effect_{proxy}"], both[f"effect_{outcome}"]
        corrs.append(np.corrcoef(x, y)[0, 1])
        slopes.append(np.polyfit(x, y, 1)[0])
        last = both
    full = segment_effects(df, treat, [proxy, outcome], segments)
    if not corrs:
        raise ValueError("Too few segments with enough customers to validate the proxy.")
    return {"split_sample_corr": float(np.median(corrs)),
            "corr_range": (float(np.percentile(corrs, 5)), float(np.percentile(corrs, 95))),
            "slope": float(np.median(slopes)),
            "n_segments": int(len(last)),
            "individual_corr": float(df[[proxy, outcome]].corr().iloc[0, 1]),
            "segments_table": full}
