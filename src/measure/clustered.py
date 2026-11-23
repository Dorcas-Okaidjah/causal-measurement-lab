"""Experiments where whole groups (clusters) are randomized: stores, regions,
agencies, markets.

People in the same cluster resemble each other, so 1,000 customers in 40
markets carry less information than 1,000 independent customers. Ignoring
this makes confidence intervals too narrow and finds 'effects' that are noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .. import config as C


def ols_effect(df: pd.DataFrame, outcome: str, treat: str, controls: list[str] = (),
               cluster: str | None = None, fixed_effects: str | None = None,
               alpha=C.ALPHA) -> dict:
    """OLS effect of `treat`, with optional controls, fixed effects and
    cluster-robust standard errors."""
    import statsmodels.api as sm

    X = df[[treat, *controls]].astype(float)
    if fixed_effects:
        # one 0/1 column per cluster (minus one): compares people within clusters
        X = pd.concat([X, pd.get_dummies(df[fixed_effects], prefix="fe",
                                         drop_first=True).astype(float)], axis=1)
    model = sm.OLS(df[outcome].astype(float), sm.add_constant(X))
    if cluster:
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster].to_numpy()})
    else:
        fit = model.fit(cov_type="HC1")
    b, se = fit.params[treat], fit.bse[treat]
    z = stats.norm.ppf(1 - alpha / 2)
    return {"effect": float(b), "se": float(se), "ci_low": float(b - z * se),
            "ci_high": float(b + z * se), "p_value": float(fit.pvalues[treat])}


def icc(y, groups) -> float:
    """Intraclass correlation: the share of outcome variance that lies between
    clusters (one-way ANOVA estimator). 0 = clusters don't matter."""
    d = pd.DataFrame({"y": np.asarray(y, float), "g": np.asarray(groups)})
    k = d["g"].nunique()
    n = len(d)
    sizes = d.groupby("g").size()
    m0 = (n - (sizes ** 2).sum() / n) / (k - 1)              # average cluster size, adjusted
    grand = d["y"].mean()
    msb = (sizes * (d.groupby("g")["y"].mean() - grand) ** 2).sum() / (k - 1)
    msw = ((d["y"] - d.groupby("g")["y"].transform("mean")) ** 2).sum() / (n - k)
    var_between = max((msb - msw) / m0, 0.0)
    return float(var_between / (var_between + msw))


def design_effect(icc_value: float, mean_cluster_size: float) -> float:
    """How much a clustered design inflates the variance: 1 + (m - 1) * ICC.
    Effective sample size = actual sample size / design effect."""
    return float(1 + (mean_cluster_size - 1) * icc_value)


def clusters_needed(n_per_arm_individual: int, icc_value: float, cluster_size: float) -> int:
    """Clusters per arm a cluster-randomized test needs to match the power of an
    individually randomized test with n_per_arm_individual people per arm."""
    deff = design_effect(icc_value, cluster_size)
    return int(np.ceil(n_per_arm_individual * deff / cluster_size))


def power_by_design(df: pd.DataFrame, outcome: str, cluster: str, effect: float,
                    n_sims: int = C.N_POWER_SIMS, alpha=C.ALPHA, seed=C.RANDOM_STATE) -> dict:
    """Simulated power of two designs on the same real people:

    * individual: randomize people, compare means (Welch test)
    * cluster:    randomize whole clusters, compare cluster means (t-test on
                  cluster means, the simplest valid analysis)
    * cluster, wrong analysis: randomize clusters but analyse as if people
                  were randomized. Reported under effect = 0 too, to show how
                  often it falsely declares an effect.

    A known `effect` is added to the treated group's outcome in each simulation.
    """
    rng = np.random.default_rng(seed)
    y0 = df[outcome].to_numpy(float)
    g = df[cluster].to_numpy()
    clusters = np.unique(g)
    hits = {"individual": 0, "cluster": 0, "cluster_wrong_analysis": 0}
    false_pos = 0
    for _ in range(n_sims):
        # individual randomization
        t = rng.random(len(y0)) < 0.5
        y = y0 + effect * t
        if stats.ttest_ind(y[t], y[~t], equal_var=False).pvalue < alpha:
            hits["individual"] += 1
        # cluster randomization
        treated_clusters = rng.choice(clusters, len(clusters) // 2, replace=False)
        tc = np.isin(g, treated_clusters)
        y = y0 + effect * tc
        means = pd.Series(y).groupby(g).mean()
        is_t = means.index.isin(treated_clusters)
        if stats.ttest_ind(means[is_t], means[~is_t], equal_var=False).pvalue < alpha:
            hits["cluster"] += 1
        if stats.ttest_ind(y[tc], y[~tc], equal_var=False).pvalue < alpha:
            hits["cluster_wrong_analysis"] += 1
        # same cluster design, no true effect, wrong analysis
        if stats.ttest_ind(y0[tc], y0[~tc], equal_var=False).pvalue < alpha:
            false_pos += 1
    out = {k: v / n_sims for k, v in hits.items()}
    out["false_positive_rate_wrong_analysis"] = false_pos / n_sims
    return out
