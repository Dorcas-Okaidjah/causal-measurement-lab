"""Unit tests: every estimator must recover an answer we planted in simulated data.

Run with:  python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src.measure import bayes, clustered as CL, experiments as E, observational as O


@pytest.fixture
def rng():
    return np.random.default_rng(123)


# ---------------------------------------------------------------- experiments
def test_srm_passes_balanced_and_flags_broken_split():
    assert E.srm_test({"a": 10_000, "b": 10_050})["ok"]
    assert not E.srm_test({"a": 10_000, "b": 10_600})["ok"]


def test_diff_in_means_recovers_effect(rng):
    y_c = rng.normal(10, 2, 20_000)
    y_t = rng.normal(10.5, 2, 20_000)
    r = E.diff_in_means(y_t, y_c)
    assert r["ci_low"] < 0.5 < r["ci_high"]
    assert abs(r["rel_lift"] - 0.05) < 0.01


def test_cuped_keeps_effect_and_removes_corr_squared_of_variance(rng):
    n = 40_000
    x = rng.normal(0, 1, n)
    t = rng.integers(0, 2, n)
    y = 0.7 * x + np.sqrt(1 - 0.49) * rng.normal(0, 1, n) + 0.1 * t   # corr(y, x) ~ 0.7
    r = E.cuped(y, x, t)
    assert abs(r["effect"] - 0.1) < 0.02
    assert abs(r["variance_reduction"] - 0.49) < 0.03


def test_regression_adjustment_matches_truth(rng):
    n = 20_000
    x = rng.normal(0, 1, n)
    t = rng.integers(0, 2, n)
    df = pd.DataFrame({"x": x, "t": t, "y": 2 * x + 0.3 * t + rng.normal(0, 1, n)})
    r = E.regression_adjusted(df, "y", "t", ["x"])
    assert r["ci_low"] < 0.3 < r["ci_high"]


def test_holm_is_monotone_and_at_least_raw():
    adj = E.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)


def test_mde_and_required_n_are_consistent():
    n = E.required_n(effect=0.1, sd=1.0)
    assert E.mde(1.0, n) == pytest.approx(0.1, rel=0.01)


# ---------------------------------------------------------------- Bayesian
def test_beta_binomial_identical_arms_is_a_coin_flip():
    r = bayes.beta_binomial(500, 10_000, 500, 10_000, draws=50_000)
    assert 0.4 < r["prob_better"] < 0.6


def test_bayesian_bootstrap_detects_real_difference(rng):
    r = bayes.bayesian_bootstrap_mean_diff(rng.normal(1.2, 1, 5000), rng.normal(1.0, 1, 5000),
                                           draws=1000)
    assert r["prob_better"] > 0.99
    assert r["cred_low"] < 0.2 < r["cred_high"]


# ---------------------------------------------------------------- observational
def _confounded(rng, n=20_000, effect=2.0):
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    e = 1 / (1 + np.exp(-(0.8 * x1 - 0.5 * x2)))
    t = rng.binomial(1, e)
    y = 3 * x1 + 2 * x2 + np.sin(x1) + effect * t + rng.normal(0, 1, n)
    return pd.DataFrame({"x1": x1, "x2": x2}), t, y


def test_naive_is_biased_under_confounding(rng):
    X, t, y = _confounded(rng)
    assert O.naive(y, t)["effect"] > 3.0          # truth is 2.0


@pytest.mark.parametrize("estimand", ["ATE", "ATT", "ATC"])
def test_aipw_recovers_effect(rng, estimand):
    X, t, y = _confounded(rng)
    r = O.aipw(X, t, y, estimand)
    assert abs(r["effect"] - 2.0) < 0.15


def test_dml_recovers_effect(rng):
    X, t, y = _confounded(rng)
    assert abs(O.dml_plr(X, t, y)["effect"] - 2.0) < 0.15


def test_ipw_and_matching_recover_effect(rng):
    X, t, y = _confounded(rng)
    ps = O.propensity(X, t)
    assert abs(O.ipw(ps, t, y, "ATE")["effect"] - 2.0) < 0.25
    assert abs(O.psm(ps, t, y, "ATT")["effect"] - 2.0) < 0.3


def test_weighting_improves_balance(rng):
    X, t, _ = _confounded(rng)
    ps = O.propensity(X, t)
    before = O.weighted_balance(X, t).abs().max()
    after = O.weighted_balance(X, t, O.ipw_weights(ps, t, "ATE")).abs().max()
    assert before > 0.3 and after < 0.05


# ---------------------------------------------------------------- clustered
def _clustered(rng, k=60, m=40, icc=0.2):
    g = np.repeat(np.arange(k), m)
    u = rng.normal(0, np.sqrt(icc), k)[g]
    y = u + rng.normal(0, np.sqrt(1 - icc), k * m)
    return pd.DataFrame({"y": y, "g": g})


def test_icc_recovers_planted_value(rng):
    d = _clustered(rng)
    assert abs(CL.icc(d["y"], d["g"]) - 0.2) < 0.07


def test_design_effect_formula():
    assert CL.design_effect(0.1, 11) == pytest.approx(2.0)


def test_cluster_robust_se_is_wider_when_treatment_is_clustered(rng):
    d = _clustered(rng)
    d["t"] = (d["g"] % 2 == 0).astype(int)           # whole clusters treated
    naive = CL.ols_effect(d, "y", "t")
    robust = CL.ols_effect(d, "y", "t", cluster="g")
    assert robust["se"] > 2 * naive["se"]
