"""Part 3: how should the next campaign test be designed?

Questions answered:
1. How many customers does it need?          MDE and sample size, spend vs visit
2. Can 'visit' stand in for spend?           Split-sample proxy validation
3. Can CUPED make it cheaper?                What the available pre-period data can remove
4. What if we must randomize whole markets?  ICC, design effect, clusters needed,
                                             simulated power, and the cost of the wrong analysis
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..measure import clustered as CL, experiments as E, surrogates

# Segments for the proxy check: purchase-history groups that marketers target on,
# fixed before looking at results.
PROXY_SEGMENTS = ["history_segment", "mens", "womens", "newbie"]


def markets(df: pd.DataFrame) -> pd.Series:
    """Label each customer with a 'market': location type x channel x spending tier."""
    return df[C.MARKET_KEYS].astype(str).agg(" | ".join, axis=1)


def _r2(df: pd.DataFrame, outcome: str) -> float:
    """Share of an outcome's variance that all pre-period data together predicts
    (the most that CUPED-style adjustment could remove)."""
    import statsmodels.api as sm

    X = pd.concat([df[C.PRE_COVARIATES].astype(float),
                   pd.get_dummies(df[C.PRE_CATEGORICAL], drop_first=True).astype(float)], axis=1)
    return float(sm.OLS(df[outcome].astype(float), sm.add_constant(X)).fit().rsquared)


def run(h: pd.DataFrame, effects: pd.DataFrame) -> dict:
    res = {}
    ctrl = h[h["arm"] == C.CONTROL].reset_index(drop=True)

    # 1. sample size: spend is rare and skewed, visit is common
    sd = float(ctrl["spend"].std())
    n_arm = int(h["arm"].value_counts().min())
    base = float(ctrl["spend"].mean())
    res["power"] = {
        "spend_sd": sd, "n_per_arm": n_arm, "control_mean": base,
        "mde_spend": E.mde(sd, n_arm),
        "mde_spend_pct": E.mde(sd, n_arm) / base,
        "n_for_10pct_lift": E.required_n(0.10 * base, sd),
        "n_for_10pct_lift_visit": E.required_n(
            0.10 * ctrl["visit"].mean(), ctrl["visit"].std()),
    }

    # 2. can visit stand in for spend?
    hp = h.assign(any_email=(h["arm"] != C.CONTROL).astype(int))
    res["proxy"] = surrogates.proxy_validation(
        hp, "any_email", "visit", "spend", PROXY_SEGMENTS)
    # campaign level: extra spend per extra visit for each e-mail
    eff = effects.set_index(["arm", "metric"])["effect"]
    res["proxy"]["spend_per_extra_visit"] = {
        arm: float(eff[(arm, "spend")] / eff[(arm, "visit")]) for arm in C.TREATMENTS}

    # 3. CUPED: the single best covariate, and everything we have combined
    res["cuped"] = {m: {"corr_history": float(np.corrcoef(ctrl[m], ctrl[C.CUPED_COVARIATE])[0, 1]),
                        "r2_all_pre_period": _r2(ctrl, m)} for m in ["visit", "spend"]}

    # 4. randomizing whole markets
    g = markets(ctrl)
    icc = CL.icc(ctrl["visit"], g)
    m = float(g.value_counts().mean())
    lift = C.PLANNED_LIFT * ctrl["visit"].mean()          # absolute visit-rate effect to detect
    n_ind = E.required_n(lift, ctrl["visit"].std())
    deff = CL.design_effect(icc, m)
    # one example market-randomized test on the control customers, analysed two ways
    rng = np.random.default_rng(C.RANDOM_STATE)
    groups = g.unique()
    treated = rng.choice(groups, len(groups) // 2, replace=False)
    ex = ctrl.assign(market=g, t=g.isin(treated).astype(int))
    res["cluster"] = {
        "n_markets": int(len(groups)), "mean_market_size": m, "icc_visit": icc,
        "design_effect": deff, "planned_lift": C.PLANNED_LIFT, "planned_effect": float(lift),
        "n_per_arm_individual": n_ind, "n_per_arm_cluster": int(np.ceil(n_ind * deff)),
        "markets_per_arm_needed": CL.clusters_needed(n_ind, icc, m),
        "example_naive": CL.ols_effect(ex, "visit", "t"),
        "example_cluster_robust": CL.ols_effect(ex, "visit", "t", cluster="market"),
        "power": CL.power_by_design(ctrl.assign(market=g), "visit", "market", effect=lift),
    }
    return res
