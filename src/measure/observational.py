"""Estimating an effect when treatment was NOT randomized.

When a marketer chooses who gets a campaign, treated and untreated customers
differ before the campaign starts, so a simple comparison mixes the campaign's
effect with those differences (confounding). These estimators adjust for the
pre-period differences we can measure. None of them can fix confounding by
something we did not measure; that limitation must be stated to stakeholders.

Estimands (which average effect we want):
* ATE: average effect over everyone in the sample
* ATT: average effect on the people who were treated
* ATC: average effect on the people who were not treated
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import config as C
from .experiments import smd


def _ci(est, se, alpha=C.ALPHA):
    z = stats.norm.ppf(1 - alpha / 2)
    return {"effect": float(est), "se": float(se), "ci_low": float(est - z * se),
            "ci_high": float(est + z * se), "p_value": float(2 * stats.norm.sf(abs(est / se)))}


def design_matrix(df: pd.DataFrame, numeric: list[str], categorical: list[str] = ()) -> pd.DataFrame:
    """Numbers as they are, labels as 0/1 columns (one level dropped)."""
    parts = [df[numeric].astype(float)]
    if categorical:
        parts.append(pd.get_dummies(df[list(categorical)], drop_first=True).astype(float))
    return pd.concat(parts, axis=1)


# --------------------------------------------------------------------------- #
# Propensity score: probability of being treated, given pre-period covariates
# --------------------------------------------------------------------------- #
def propensity(X: pd.DataFrame, t, clip=C.PROPENSITY_CLIP) -> np.ndarray:
    """Logistic regression of treatment on the covariates (standardized)."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    model.fit(X, t)
    return np.clip(model.predict_proba(X)[:, 1], *clip)


def overlap_summary(ps, t) -> dict:
    """How far the treated and untreated propensity distributions overlap.
    Little overlap = some treated people have no comparable untreated person,
    and every method then leans on extrapolation."""
    t = np.asarray(t, bool)
    lo, hi = max(ps[t].min(), ps[~t].min()), min(ps[t].max(), ps[~t].max())
    inside = (ps >= lo) & (ps <= hi)
    return {"common_support": (float(lo), float(hi)),
            "share_treated_inside": float(inside[t].mean()),
            "share_control_inside": float(inside[~t].mean())}


def ipw_weights(ps, t, estimand):
    t = np.asarray(t, bool)
    if estimand == "ATE":
        return np.where(t, 1 / ps, 1 / (1 - ps))
    if estimand == "ATT":
        return np.where(t, 1.0, ps / (1 - ps))
    if estimand == "ATC":
        return np.where(t, (1 - ps) / ps, 1.0)
    raise ValueError(estimand)


def weighted_balance(X: pd.DataFrame, t, w=None) -> pd.Series:
    t = np.asarray(t, bool)
    w = np.ones(len(t)) if w is None else np.asarray(w)
    return pd.Series({c: smd(X[c].to_numpy()[t], X[c].to_numpy()[~t], w[t], w[~t])
                      for c in X.columns})


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def naive(y, t) -> dict:
    y, t = np.asarray(y, float), np.asarray(t, bool)
    est = y[t].mean() - y[~t].mean()
    se = np.sqrt(y[t].var(ddof=1) / t.sum() + y[~t].var(ddof=1) / (~t).sum())
    return _ci(est, se)


def ols_adjusted(X: pd.DataFrame, t, y) -> dict:
    """Outcome regression: y on treatment + covariates (robust SEs)."""
    import statsmodels.api as sm

    D = sm.add_constant(pd.concat([pd.Series(np.asarray(t, float), index=X.index, name="t"), X],
                                  axis=1))
    fit = sm.OLS(np.asarray(y, float), D).fit(cov_type="HC1")
    return _ci(fit.params["t"], fit.bse["t"])


def ipw(ps, t, y, estimand="ATE") -> dict:
    """Inverse probability weighting (normalized / Hajek). Each person is
    weighted so the two groups look alike on the covariates. Standard errors
    from weighted least squares with robust variance, treating weights as fixed."""
    import statsmodels.api as sm

    t = np.asarray(t, float)
    w = ipw_weights(ps, t, estimand)
    fit = sm.WLS(np.asarray(y, float), sm.add_constant(t), weights=w).fit(cov_type="HC1")
    out = _ci(fit.params[1], fit.bse[1])
    out["max_weight_share"] = float(w.max() / w.sum())   # one person's share of all weight
    return out


def psm(ps, t, y, estimand="ATT", caliper=C.PSM_CALIPER) -> dict:
    """1-nearest-neighbour matching on the logit of the propensity score, with
    replacement and a caliper (matches further than caliper x sd are dropped).

    ATT matches each treated person to the closest untreated one; ATC the
    reverse. The standard error treats the matched pairs as fixed, so it is
    approximate (it ignores the uncertainty in choosing the matches).
    """
    t, y = np.asarray(t, bool), np.asarray(y, float)
    logit = np.log(ps / (1 - ps))
    width = caliper * logit.std()
    src, tgt = (t, ~t) if estimand == "ATT" else (~t, t)
    nn = NearestNeighbors(n_neighbors=1).fit(logit[tgt].reshape(-1, 1))
    dist, idx = nn.kneighbors(logit[src].reshape(-1, 1))
    keep = dist[:, 0] <= width
    y_src = y[src][keep]
    y_match = y[tgt][idx[keep, 0]]
    diff = (y_src - y_match) if estimand == "ATT" else (y_match - y_src)
    out = _ci(diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff)))
    out["share_matched"] = float(keep.mean())
    out["n_distinct_matches"] = int(len(np.unique(idx[keep, 0])))
    return out


def _crossfit(X, t, y, n_folds, seed, outcome_model):
    """Out-of-fold predictions of e(x)=P(T=1|x), m1(x)=E[y|T=1,x], m0(x)=E[y|T=0,x]
    and m(x)=E[y|x]. Each prediction comes from models that never saw that person,
    which stops overfitting from leaking into the effect estimate."""
    X = np.asarray(X, float)
    t, y = np.asarray(t, int), np.asarray(y, float)
    e, m1, m0, m = (np.zeros(len(y)) for _ in range(4))
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        ps_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        e[te] = ps_model.fit(X[tr], t[tr]).predict_proba(X[te])[:, 1]
        for arm, store in ((1, m1), (0, m0)):
            sel = tr[t[tr] == arm]
            store[te] = outcome_model().fit(X[sel], y[sel]).predict(X[te])
        m[te] = outcome_model().fit(X[tr], y[tr]).predict(X[te])
    return np.clip(e, *C.AIPW_CLIP), m1, m0, m


def _gbm_regressor():
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                         min_samples_leaf=100, random_state=C.RANDOM_STATE)


def aipw(X, t, y, estimand="ATE", n_folds=C.N_FOLDS, seed=C.RANDOM_STATE,
         outcome_model=_gbm_regressor) -> dict:
    """Augmented IPW, the doubly robust estimator, with cross-fitting.

    It combines an outcome model and a propensity model, and stays consistent
    if EITHER one is right. Standard errors come from its influence function.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    e, m1, m0, _ = _crossfit(X, t, y, n_folds, seed, outcome_model)
    n = len(y)
    if estimand == "ATE":
        psi = m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
        est = psi.mean()
        phi = psi - est
    elif estimand == "ATT":
        p1 = t.mean()
        a = t * (y - m0) - (1 - t) * e / (1 - e) * (y - m0)
        est = a.sum() / t.sum()
        phi = (a - t * est) / p1
    elif estimand == "ATC":
        p0 = 1 - t.mean()
        b = (1 - t) * (m1 - y) + t * (1 - e) / e * (y - m1)
        est = b.sum() / (1 - t).sum()
        phi = (b - (1 - t) * est) / p0
    else:
        raise ValueError(estimand)
    return _ci(est, phi.std(ddof=1) / np.sqrt(n))


def dml_plr(X, t, y, n_folds=C.N_FOLDS, seed=C.RANDOM_STATE,
            outcome_model=_gbm_regressor) -> dict:
    """Double machine learning, partially linear model (Chernozhukov et al., 2018).

    1. Predict y from X and T from X with flexible models, out of fold.
    2. Regress the part of y that X cannot explain on the part of T that X
       cannot explain. The slope is the effect.

    If the effect differs across people, this is a variance-weighted average
    effect rather than exactly the ATE, which is worth saying when reporting it.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    e, _, _, m = _crossfit(X, t, y, n_folds, seed, outcome_model)
    rt, ry = t - e, y - m
    theta = (rt * ry).sum() / (rt ** 2).sum()
    phi = (ry - theta * rt) * rt / (rt ** 2).mean()
    return _ci(theta, phi.std(ddof=1) / np.sqrt(len(y)))
