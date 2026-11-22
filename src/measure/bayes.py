"""Bayesian A/B analysis.

Instead of a p-value, these functions answer the questions decision-makers
actually ask: 'How likely is it that the e-mail beats no e-mail?' and 'If we
roll it out and we're wrong, how much do we expect to lose?'
"""
from __future__ import annotations

import numpy as np

from .. import config as C


def _summarize(diff: np.ndarray, base: np.ndarray) -> dict:
    lift = diff / base
    return {"effect_mean": float(diff.mean()),
            "cred_low": float(np.percentile(diff, 2.5)),
            "cred_high": float(np.percentile(diff, 97.5)),
            "prob_better": float((diff > 0).mean()),
            # average shortfall if we ship the treatment and it is actually worse
            "expected_loss": float(np.maximum(-diff, 0).mean()),
            "lift_mean": float(lift.mean()),
            "lift_low": float(np.percentile(lift, 2.5)),
            "lift_high": float(np.percentile(lift, 97.5))}


def beta_binomial(successes_t: int, n_t: int, successes_c: int, n_c: int,
                  prior=C.BETA_PRIOR, draws=C.N_POSTERIOR_DRAWS, seed=C.RANDOM_STATE) -> dict:
    """Posterior for two rates (visit rate, conversion rate).

    With a Beta(a, b) prior and k successes out of n, the posterior is
    Beta(a + k, b + n - k). We draw from both posteriors and compare.
    """
    rng = np.random.default_rng(seed)
    a, b = prior
    p_t = rng.beta(a + successes_t, b + n_t - successes_t, draws)
    p_c = rng.beta(a + successes_c, b + n_c - successes_c, draws)
    out = _summarize(p_t - p_c, p_c)
    out["rate_treated"] = float(p_t.mean())
    out["rate_control"] = float(p_c.mean())
    return out


def bayesian_bootstrap_mean_diff(y_t, y_c, draws: int = 4000, chunk: int = 250,
                                 seed=C.RANDOM_STATE) -> dict:
    """Posterior for a difference in means of a skewed metric (spend).

    The Bayesian bootstrap (Rubin, 1981) gives each customer a random
    Dirichlet weight and recomputes the mean; repeating this gives a
    posterior for the mean without assuming spend is normally distributed.
    Draws are made in chunks to keep memory small.
    """
    rng = np.random.default_rng(seed)
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)

    def means(y):
        out = []
        for start in range(0, draws, chunk):
            k = min(chunk, draws - start)
            w = rng.exponential(1.0, (k, len(y)))       # Dirichlet = normalized exponentials
            out.append((w @ y) / w.sum(axis=1))
        return np.concatenate(out)

    m_t, m_c = means(y_t), means(y_c)
    return _summarize(m_t - m_c, m_c)
