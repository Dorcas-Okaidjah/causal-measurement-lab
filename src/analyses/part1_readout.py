"""Part 1: did the campaign work? Read out the randomized test.

Questions answered, in order:
1. Was the test run properly?          SRM check, covariate balance
2. Did each e-mail work, and how much? Effects on visit, conversion, spend
3. Can we measure it more precisely?   CUPED, regression adjustment, GLM
4. How sure should leadership be?      Bayesian probability of benefit, expected loss
5. Who responds?                       Pre-registered subgroups, Holm-corrected
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..measure import bayes, experiments as E

SUBGROUPS = {"bought_mens": "mens", "bought_womens": "womens", "new_customer": "newbie"}


def pair(h: pd.DataFrame, arm: str) -> pd.DataFrame:
    d = h[h["arm"].isin([arm, C.CONTROL])].copy()
    d["t"] = (d["arm"] == arm).astype(int)
    return d


def covariates(d: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(d[C.PRE_CATEGORICAL], drop_first=True).astype(float)
    return pd.concat([d[C.PRE_COVARIATES].astype(float), dummies], axis=1)


def run(h: pd.DataFrame) -> dict:
    res = {"n_customers": len(h)}

    # 1. randomization checks
    res["srm"] = E.srm_test(h["arm"].value_counts().to_dict())
    res["balance"] = {}
    for arm in C.TREATMENTS:
        d = pair(h, arm)
        res["balance"][arm] = E.balance_table(covariates(d), d["t"])

    # 2-3. effects, three ways
    rows, pvals = [], {}
    for arm in C.TREATMENTS:
        d = pair(h, arm)
        X = covariates(d)
        dx = pd.concat([d[["t"] + C.METRICS], X], axis=1)
        for m in C.METRICS:
            raw = E.diff_in_means(d.loc[d.t == 1, m], d.loc[d.t == 0, m])
            cup = E.cuped(d[m], d[C.CUPED_COVARIATE], d["t"])
            reg = E.regression_adjusted(dx, m, "t", list(X.columns))
            rows.append({"arm": arm, "metric": m, **{k: raw[k] for k in
                         ["mean_control", "mean_treated", "effect", "se", "ci_low", "ci_high",
                          "p_value", "rel_lift", "rel_ci_low", "rel_ci_high"]},
                         "cuped_se": cup["se"], "cuped_variance_reduction": cup["variance_reduction"],
                         "reg_effect": reg["effect"], "reg_se": reg["se"]})
            pvals[f"{arm}|{m}"] = raw["p_value"]
    effects = pd.DataFrame(rows)
    adj = E.holm(pvals)
    effects["p_holm"] = [adj[f"{a}|{m}"] for a, m in zip(effects.arm, effects.metric)]
    res["effects"] = effects

    # GLM for the binary visit outcome
    res["glm_visit"] = {}
    for arm in C.TREATMENTS:
        d = pair(h, arm)
        X = covariates(d)
        res["glm_visit"][arm] = E.logistic_glm(pd.concat([d[["t", "visit"]], X], axis=1),
                                               "visit", "t", list(X.columns))

    # CUPED: why it helps so little here
    res["cuped_corr"] = {m: float(np.corrcoef(h[m], h[C.CUPED_COVARIATE])[0, 1]) for m in C.METRICS}

    # 4. Bayesian view
    res["bayes"] = {}
    ctrl = h[h.arm == C.CONTROL]
    for arm in C.TREATMENTS:
        tr = h[h.arm == arm]
        res["bayes"][arm] = {
            m: bayes.beta_binomial(int(tr[m].sum()), len(tr), int(ctrl[m].sum()), len(ctrl))
            for m in ["visit", "conversion"]}
        res["bayes"][arm]["spend"] = bayes.bayesian_bootstrap_mean_diff(tr["spend"], ctrl["spend"])

    # 5. subgroups (decided before looking at results)
    sub_rows, sub_p = [], {}
    for arm in C.TREATMENTS:
        d = pair(h, arm)
        for label, col in SUBGROUPS.items():
            for val in (0, 1):
                g = d[d[col] == val]
                for m in ["visit", "spend"]:
                    r = E.diff_in_means(g.loc[g.t == 1, m], g.loc[g.t == 0, m])
                    key = f"{arm}|{label}={val}|{m}"
                    sub_rows.append({"arm": arm, "subgroup": f"{label}={val}", "metric": m,
                                     "n": len(g), "effect": r["effect"], "ci_low": r["ci_low"],
                                     "ci_high": r["ci_high"], "p_value": r["p_value"]})
                    sub_p[key] = r["p_value"]
    subs = pd.DataFrame(sub_rows)
    adj = E.holm(sub_p)
    subs["p_holm"] = [adj[f"{a}|{s}|{m}"] for a, s, m in zip(subs.arm, subs.subgroup, subs.metric)]
    res["subgroups"] = subs
    return res
