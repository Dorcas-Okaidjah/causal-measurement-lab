"""Analysing a randomized experiment (A/B test).

Each function does one job and returns plain numbers, so the same functions
can analyse any test: an e-mail campaign, a website change, a pricing pilot.

Order of work for any readout:
1. srm_test        -> did randomization work? (sample ratio mismatch)
2. balance_table   -> do the groups look alike before treatment?
3. diff_in_means   -> the effect, with a confidence interval
4. cuped / regression_adjusted -> the same effect, measured more precisely
5. holm            -> correct p-values when testing several things at once
6. mde / required_n -> how big a test the next campaign needs
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .. import config as C


def _z(alpha=C.ALPHA):
    return stats.norm.ppf(1 - alpha / 2)       # 1.96 for a 95% interval


# --------------------------------------------------------------------------- #
# 1. Randomization checks
# --------------------------------------------------------------------------- #
def srm_test(counts: dict, expected_shares: dict | None = None) -> dict:
    """Sample ratio mismatch: are group sizes consistent with the planned split?

    A tiny p-value (below 0.001 is the usual alarm) means assignment or logging
    is broken, and no effect estimate from the test should be trusted.
    """
    groups = list(counts)
    observed = np.array([counts[g] for g in groups], dtype=float)
    shares = (np.array([expected_shares[g] for g in groups]) if expected_shares
              else np.full(len(groups), 1 / len(groups)))
    expected = shares * observed.sum()
    chi2, p = stats.chisquare(observed, expected)
    return {"chi2": float(chi2), "p_value": float(p), "ok": bool(p > 0.001),
            "observed": dict(zip(groups, observed.astype(int).tolist()))}


def smd(x_t, x_c, w_t=None, w_c=None) -> float:
    """Standardized mean difference: gap in means / pooled standard deviation.

    Below 0.1 in absolute value is the usual standard for 'balanced'.
    Optional weights give the balance after propensity weighting.
    """
    x_t, x_c = np.asarray(x_t, float), np.asarray(x_c, float)
    w_t = np.ones_like(x_t) if w_t is None else np.asarray(w_t, float)
    w_c = np.ones_like(x_c) if w_c is None else np.asarray(w_c, float)
    m_t, m_c = np.average(x_t, weights=w_t), np.average(x_c, weights=w_c)
    v_t = np.average((x_t - m_t) ** 2, weights=w_t)
    v_c = np.average((x_c - m_c) ** 2, weights=w_c)
    pooled = np.sqrt((v_t + v_c) / 2)
    return float((m_t - m_c) / pooled) if pooled > 0 else 0.0


def balance_table(X: pd.DataFrame, treated) -> pd.DataFrame:
    """Mean of each pre-period covariate by group, and the SMD between them."""
    t = np.asarray(treated, bool)
    rows = []
    for col in X.columns:
        x = X[col].to_numpy(float)
        rows.append({"covariate": col, "mean_treated": x[t].mean(),
                     "mean_control": x[~t].mean(), "smd": smd(x[t], x[~t])})
    return pd.DataFrame(rows).set_index("covariate")


# --------------------------------------------------------------------------- #
# 2. The effect
# --------------------------------------------------------------------------- #
def diff_in_means(y_t, y_c, alpha=C.ALPHA) -> dict:
    """Difference in means with a Welch (unequal-variance) confidence interval.

    Also reports the relative lift (treated / control - 1) with a delta-method
    interval, because business partners usually ask 'by what percent?'.
    """
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    n_t, n_c = len(y_t), len(y_c)
    m_t, m_c = y_t.mean(), y_c.mean()
    v_t, v_c = y_t.var(ddof=1) / n_t, y_c.var(ddof=1) / n_c
    est, se = m_t - m_c, np.sqrt(v_t + v_c)
    z = _z(alpha)
    p = 2 * stats.norm.sf(abs(est / se))
    # relative lift: ratio of two independent means, delta method on the log scale
    if m_t > 0 and m_c > 0:
        rel = m_t / m_c - 1
        se_log = np.sqrt(v_t / m_t ** 2 + v_c / m_c ** 2)
        rel_ci = (np.exp(np.log(m_t / m_c) - z * se_log) - 1,
                  np.exp(np.log(m_t / m_c) + z * se_log) - 1)
    else:  # a percentage lift is undefined when a mean is zero or negative
        rel, rel_ci = np.nan, (np.nan, np.nan)
    return {"effect": est, "se": se, "ci_low": est - z * se, "ci_high": est + z * se,
            "p_value": p, "mean_treated": m_t, "mean_control": m_c,
            "n_treated": n_t, "n_control": n_c, "rel_lift": rel,
            "rel_ci_low": rel_ci[0], "rel_ci_high": rel_ci[1]}


def cuped(y, x, treated, alpha=C.ALPHA) -> dict:
    """CUPED: subtract the part of the outcome that a pre-period metric predicts.

    y_adj = y - theta * (x - mean(x)),  theta = cov(y, x) / var(x)

    x is measured before treatment, so it cannot be affected by it: the
    adjustment removes noise without introducing bias. The effect estimate
    stays the same on average, but its confidence interval shrinks by a factor
    of sqrt(1 - corr(y, x)^2).
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    t = np.asarray(treated, bool)
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)   # pooled over both arms
    y_adj = y - theta * (x - x.mean())
    raw = diff_in_means(y[t], y[~t], alpha)
    adj = diff_in_means(y_adj[t], y_adj[~t], alpha)
    adj["theta"] = theta
    adj["variance_reduction"] = 1 - (adj["se"] / raw["se"]) ** 2
    # the relative lift is still measured against the raw control mean
    adj["rel_lift"] = adj["effect"] / raw["mean_control"]
    adj["rel_ci_low"] = adj["ci_low"] / raw["mean_control"]
    adj["rel_ci_high"] = adj["ci_high"] / raw["mean_control"]
    return adj


def regression_adjusted(df: pd.DataFrame, outcome: str, treat: str,
                        covariates: list[str], alpha=C.ALPHA) -> dict:
    """OLS of the outcome on treatment + pre-period covariates (Lin, 2013).

    Covariates are centred and interacted with treatment, which keeps the
    estimate unbiased for the average effect even if the linear model is
    wrong. Standard errors are heteroskedasticity-robust (HC2).
    """
    import statsmodels.api as sm

    X = df[covariates].astype(float)
    Xc = X - X.mean()
    t = df[treat].astype(float).to_numpy()
    design = pd.concat([pd.Series(t, index=df.index, name="treat"), Xc,
                        Xc.mul(t, axis=0).add_suffix(":treat")], axis=1)
    fit = sm.OLS(df[outcome].astype(float), sm.add_constant(design)).fit(cov_type="HC2")
    est, se = fit.params["treat"], fit.bse["treat"]
    z = _z(alpha)
    return {"effect": float(est), "se": float(se), "ci_low": float(est - z * se),
            "ci_high": float(est + z * se), "p_value": float(fit.pvalues["treat"])}


def logistic_glm(df: pd.DataFrame, outcome: str, treat: str,
                 covariates: list[str], alpha=C.ALPHA) -> dict:
    """Binomial GLM (logistic regression) for a yes/no outcome such as a visit.

    Returns the odds ratio of treatment and the average marginal effect: the
    change in probability, averaged over customers, which is the number a
    business partner can use ('+7.6 points of visit rate').
    """
    import statsmodels.api as sm

    X = sm.add_constant(pd.concat([df[[treat]], df[covariates]], axis=1).astype(float))
    fit = sm.GLM(df[outcome].astype(float), X, family=sm.families.Binomial()).fit(cov_type="HC1")
    b, se = fit.params[treat], fit.bse[treat]
    z = _z(alpha)
    X1, X0 = X.copy(), X.copy()
    X1[treat], X0[treat] = 1.0, 0.0
    ame = float((fit.predict(X1) - fit.predict(X0)).mean())
    return {"odds_ratio": float(np.exp(b)), "or_ci_low": float(np.exp(b - z * se)),
            "or_ci_high": float(np.exp(b + z * se)), "p_value": float(fit.pvalues[treat]),
            "avg_marginal_effect": ame}


# --------------------------------------------------------------------------- #
# 3. Several tests at once
# --------------------------------------------------------------------------- #
def holm(p_values: dict) -> dict:
    """Holm-Bonferroni adjustment: keeps the chance of any false positive among
    all the tests at alpha. Less conservative than plain Bonferroni."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


# --------------------------------------------------------------------------- #
# 4. Planning the next test
# --------------------------------------------------------------------------- #
def mde(sd: float, n_per_arm: int, alpha=C.ALPHA, power=C.POWER_TARGET) -> float:
    """Minimum detectable effect: the smallest true difference in means that a
    test with n_per_arm customers per group detects with the given power."""
    return float((_z(alpha) + stats.norm.ppf(power)) * sd * np.sqrt(2 / n_per_arm))


def required_n(effect: float, sd: float, alpha=C.ALPHA, power=C.POWER_TARGET) -> int:
    """Customers needed per group to detect `effect` with the given power."""
    return int(np.ceil(2 * ((_z(alpha) + stats.norm.ppf(power)) * sd / effect) ** 2))
