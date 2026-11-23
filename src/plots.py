"""Charts for the README and report (matplotlib, PNG)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config as C  # noqa: E402

COL = C.COLORS


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=COL["ink"], pad=10)
    ax.set_xlabel(xlabel, color=COL["muted"])
    ax.set_ylabel(ylabel, color=COL["muted"])
    ax.grid(True, color=COL["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(COL["grid"])
    ax.tick_params(colors=COL["muted"])


def _save(fig, name):
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.REPORT_DIR / name, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def _forest(ax, labels, est, lo, hi, color, truth=None, truth_ci=None):
    y = np.arange(len(labels))[::-1]
    if truth_ci is not None:
        ax.axvspan(*truth_ci, color=COL["grid"], alpha=0.7, lw=0)
    if truth is not None:
        ax.axvline(truth, color=COL["ink"], lw=1.5)
    ax.errorbar(est, y, xerr=[np.array(est) - np.array(lo), np.array(hi) - np.array(est)],
                fmt="o", color=color, ecolor=color, elinewidth=2, capsize=0, ms=7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)


def lift_forest(effects):
    """Relative lift of each e-mail on each metric, with 95% intervals."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    d = effects.copy()
    d["label"] = d["arm"].str.replace(" E-Mail", " e-mail") + "  ·  " + d["metric"]
    y = np.arange(len(d))[::-1]
    for arm, color in zip(C.TREATMENTS, [COL["a"], COL["b"]]):
        m = (d["arm"] == arm).to_numpy()
        ax.errorbar(d.loc[m, "rel_lift"], y[m],
                    xerr=[d.loc[m, "rel_lift"] - d.loc[m, "rel_ci_low"],
                          d.loc[m, "rel_ci_high"] - d.loc[m, "rel_lift"]],
                    fmt="o", color=color, elinewidth=2, ms=7, label=arm)
    ax.axvline(0, color=COL["ink"], lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _style(ax, "Both e-mails lift every metric; the men's e-mail lifts more",
           "Lift vs no e-mail (95% interval)", "")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "experiment_lift.png")


def cuped_curve(corr: float):
    """CUPED's variance reduction is the squared correlation with the covariate."""
    r = np.linspace(0, 0.9, 100)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(r, r ** 2, color=COL["a"], lw=2)
    ax.plot([corr], [corr ** 2], "o", color=COL["b"], ms=9)
    ax.annotate(f"This test: corr {corr:.2f}\n→ {corr ** 2:.2%} less variance",
                (corr, corr ** 2), xytext=(0.12, 0.35), color=COL["ink"], fontsize=9,
                arrowprops=dict(arrowstyle="-", color=COL["muted"]))
    ax.annotate("corr 0.7 → 49% less variance\n(about half the sample needed)",
                (0.7, 0.49), xytext=(0.32, 0.62), color=COL["ink"], fontsize=9,
                arrowprops=dict(arrowstyle="-", color=COL["muted"]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _style(ax, "CUPED only helps when the pre-period metric predicts the outcome",
           "Correlation of pre-period covariate with outcome", "Variance removed")
    _save(fig, "cuped_variance_reduction.png")


def proxy_scatter(table):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(table["effect_visit"], table["effect_spend"], s=table["n"] / 40,
               color=COL["a"], alpha=0.8, edgecolor="white", linewidth=1)
    ax.axhline(0, color=COL["muted"], lw=0.8)
    _style(ax, "Visit lifts only loosely track spend lifts across segments",
           "Effect of an e-mail on visit rate (segment)", "Effect on spend per customer ($)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _save(fig, "proxy_validation.png")


def benchmark_forest(obs):
    """Each method's estimate vs the randomized answer (grey band)."""
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    est = {**obs["estimates"], **obs["unmeasured"]}
    names = list(est)
    t = obs["truth"]
    _forest(ax, names, [est[n]["effect"] for n in names], [est[n]["ci_low"] for n in names],
            [est[n]["ci_high"] for n in names], COL["b"], truth=t["effect"],
            truth_ci=(t["ci_low"], t["ci_high"]))
    _style(ax, "Adjustment removes the targeting bias",
           "Effect on visit rate (grey band = randomized test)", "")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    _save(fig, "without_a_test.png")


def overlap_hist(ps, t, name, title):
    t = np.asarray(t, bool)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(ps[~t], bins=bins, density=True, color=COL["a"], alpha=0.75, label="Comparison group")
    ax.hist(ps[t], bins=bins, density=True, color=COL["b"], alpha=0.75, label="Treated")
    _style(ax, title, "Propensity score (estimated chance of treatment)", "Density")
    ax.legend(frameon=False)
    _save(fig, name)


def market_chart(cl):
    """Power of customer vs market randomization, and false positives from the wrong analysis."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    p = cl["power"]
    labels = ["Customers\nrandomized", "Markets\nrandomized", "Markets randomized,\nwrong analysis,\nno real effect"]
    vals = [p["individual"], p["cluster"], p["false_positive_rate_wrong_analysis"]]
    bars = ax.bar(labels, vals, color=[COL["a"], COL["a"], COL["b"]], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center",
                color=COL["ink"], fontsize=10)
    ax.axhline(C.ALPHA, color=COL["muted"], lw=0.8, ls="--")
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _style(ax, f"Share of simulated tests significant (true lift {cl['planned_lift']:.0%})", "",
           "Share of simulated tests")
    ax.grid(axis="x", visible=False)
    _save(fig, "market_randomization.png")


def bayes_chart(bayes):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    rows = [(arm, m, bayes[arm][m]) for arm in C.TREATMENTS for m in ["visit", "conversion", "spend"]]
    labels = [f"{a.replace(' E-Mail', '')} · {m}" for a, m, _ in rows]
    vals = [r["prob_better"] for _, _, r in rows]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color=[COL["a"] if a == C.TREATMENTS[0] else COL["b"] for a, _, _ in rows],
            height=0.6)
    for yy, v in zip(y, vals):
        label = "> 99.99%" if v > 0.9999 else f"{v:.2%}"
        ax.text(min(v, 1) - 0.01, yy, label, va="center", ha="right", color="white", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _style(ax, "Probability each e-mail beats no e-mail", "Posterior probability", "")
    ax.grid(axis="y", visible=False)
    _save(fig, "bayes_probability.png")
