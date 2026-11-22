"""Study 2: can observational methods recover a known experimental answer?

Two benchmarks where the true effect is known from a randomized experiment:

A. Hillstrom, deliberately biased. Start from the randomized Women's-email
   test, then drop treated customers with a probability that rises with past
   spend and recent activity, mimicking a marketer who targets engaged
   customers. Every control is kept, so the control group is still a random
   sample of all customers. That makes the effect on the controls (ATC) equal
   to the population effect the randomized test measured: the truth.

B. LaLonde (1986). The NSW job-training programme was randomized; its effect
   on 1978 earnings is known. Replace the experimental controls with people
   from a national survey (CPS) and ask each method to recover the effect on
   the trainees (ATT). This is the classic test of whether non-experimental
   methods can be trusted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..measure import experiments as E, observational as O


def _all_methods(X, t, y, estimand: str) -> dict:
    ps = O.propensity(X, t)
    w = O.ipw_weights(ps, t, estimand)
    out = {
        "Naive difference": O.naive(y, t),
        "Regression adjustment": O.ols_adjusted(X, t, y),
        "Propensity matching": O.psm(ps, t, y, estimand),
        "Inverse probability weighting": O.ipw(ps, t, y, estimand),
        "Doubly robust (AIPW)": O.aipw(X, t, y, estimand),
        "Double ML": O.dml_plr(X, t, y),
    }
    diag = {"overlap": O.overlap_summary(ps, t),
            "balance_before": O.weighted_balance(X, t),
            "balance_after": O.weighted_balance(X, t, w),
            "propensity": ps, "treated": np.asarray(t, bool)}
    return out, diag


# --------------------------------------------------------------------------- #
# A. Hillstrom with targeting bias
# --------------------------------------------------------------------------- #
def make_targeted_sample(h: pd.DataFrame, seed=C.RANDOM_STATE) -> pd.DataFrame:
    d = h[h["arm"].isin([C.OBS_TREATMENT, C.CONTROL])].reset_index(drop=True)
    t = (d["arm"] == C.OBS_TREATMENT).to_numpy()

    def z(x):
        return (x - x.mean()) / x.std()

    # the marketer's (hidden) targeting rule: engaged, recent, web customers
    logit = (-0.4 + 1.2 * z(np.log1p(d["history"])) - 0.8 * z(d["recency"])
             + 0.7 * (d["channel"] == "Web") + 0.5 * d["womens"])
    keep_prob = 1 / (1 + np.exp(-logit))
    rng = np.random.default_rng(seed)
    keep = ~t | (rng.random(len(d)) < keep_prob)
    out = d[keep].reset_index(drop=True)
    out["t"] = (out["arm"] == C.OBS_TREATMENT).astype(int)
    return out


def hillstrom_features(d: pd.DataFrame, drop: list[str] = ()) -> pd.DataFrame:
    X = O.design_matrix(d, C.PRE_COVARIATES, C.PRE_CATEGORICAL)
    X["log_history"] = np.log1p(d["history"])
    return X.drop(columns=[c for c in X.columns if any(c.startswith(p) for p in drop)])


def run_hillstrom(h: pd.DataFrame) -> dict:
    full = h[h["arm"].isin([C.OBS_TREATMENT, C.CONTROL])]
    t_full = full["arm"] == C.OBS_TREATMENT
    y = C.OBS_OUTCOME
    truth = E.diff_in_means(full.loc[t_full, y], full.loc[~t_full, y])
    d = make_targeted_sample(h)
    X = hillstrom_features(d)
    est, diag = _all_methods(X, d["t"].to_numpy(), d[y].to_numpy(), "ATC")
    # limitation demo: the same analysis if past spend had not been recorded
    X_blind = hillstrom_features(d, drop=["history", "log_history"])
    blind = {"Doubly robust (AIPW), past spend unobserved":
             O.aipw(X_blind, d["t"].to_numpy(), d[y].to_numpy(), "ATC"),
             "Inverse probability weighting, past spend unobserved":
             O.ipw(O.propensity(X_blind, d["t"]), d["t"].to_numpy(), d[y].to_numpy(), "ATC")}
    return {"truth": truth, "estimates": est, "unmeasured": blind, "diagnostics": diag,
            "n_treated_kept": int(d["t"].sum()), "n_treated_full": int(t_full.sum()),
            "n_control": int((d["t"] == 0).sum()), "outcome": y}


# --------------------------------------------------------------------------- #
# B. LaLonde: NSW trainees vs CPS comparison group
# --------------------------------------------------------------------------- #
def lalonde_features(d: pd.DataFrame) -> pd.DataFrame:
    """Dehejia-Wahba (1999) propensity specification."""
    X = d[C.LALONDE_COVARIATES].copy()
    X["age2"], X["age3"] = X["age"] ** 2, X["age"] ** 3
    X["educ2"] = X["educ"] ** 2
    X["u74"] = (X["re74"] == 0).astype(float)
    X["u75"] = (X["re75"] == 0).astype(float)
    X["educ_re74"] = X["educ"] * X["re74"]
    return X


def run_lalonde(l: pd.DataFrame) -> dict:
    nsw = l[l["sample"] == "NSW experiment"]
    truth = E.diff_in_means(nsw.loc[nsw.treat == 1, "re78"], nsw.loc[nsw.treat == 0, "re78"])
    d = pd.concat([nsw[nsw.treat == 1], l[l["sample"] == "CPS comparison"]]).reset_index(drop=True)
    X = lalonde_features(d)
    est, diag = _all_methods(X, d["treat"].to_numpy().astype(int), d["re78"].to_numpy(), "ATT")
    return {"truth": truth, "estimates": est, "diagnostics": diag,
            "n_treated": int(d["treat"].sum()), "n_comparison": int((d["treat"] == 0).sum())}
