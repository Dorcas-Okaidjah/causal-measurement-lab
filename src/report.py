"""Write reports/summary.md: the readout a measurement team would hand to leadership."""
from __future__ import annotations

from . import config as C


def _p(x):
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


def _prob(x):
    return "> 99.99%" if x > 0.9999 else f"{x:.2%}"


def write_summary(r: dict) -> None:
    e, o, n = r["part1"], r["part2"], r["part3"]
    L = []
    w = L.append
    if r.get("synthetic"):
        w("> **Synthetic data run** (smoke test). Numbers below are not meaningful.\n")
    w("# E-mail campaign measurement readout\n")

    # ---------------- part 1 ----------------
    w("## Part 1. Did the campaign work? (randomized test, 64,000 customers)\n")
    srm = e["srm"]
    w(f"**Test health.** Group sizes {', '.join(f'{k}: {v:,}' for k, v in srm['observed'].items())}; "
      f"sample-ratio check p = {srm['p_value']:.2f} ({'pass' if srm['ok'] else 'FAIL'}). "
      f"Largest pre-period imbalance |SMD| = "
      f"{max(b['smd'].abs().max() for b in e['balance'].values()):.3f} (threshold 0.1).\n")
    w("| E-mail | Metric | Control | Treated | Effect (95% CI) | Lift (95% CI) | Holm p |")
    w("|---|---|---|---|---|---|---|")
    for _, x in e["effects"].iterrows():
        f = "{:.2f}" if x.metric == "spend" else "{:.2%}"
        w(f"| {x.arm} | {x.metric} | {f.format(x.mean_control)} | {f.format(x.mean_treated)} | "
          f"{f.format(x.effect)} ({f.format(x.ci_low)} to {f.format(x.ci_high)}) | "
          f"{x.rel_lift:+.0%} ({x.rel_ci_low:+.0%} to {x.rel_ci_high:+.0%}) | {_p(x.p_holm)} |")
    g = e["glm_visit"]
    w(f"\nLogistic GLM (visit, with pre-period covariates): odds ratio "
      + "; ".join(f"{a} {v['odds_ratio']:.2f} ({v['or_ci_low']:.2f}-{v['or_ci_high']:.2f}), "
                  f"+{v['avg_marginal_effect']:.1%} points" for a, v in g.items()) + ".\n")
    sp = e["effects"][e["effects"].metric == "spend"]
    w(f"**CUPED.** Using prior-year spend removed only {sp['cuped_variance_reduction'].max():.2%} "
      f"of the variance in spend, because it correlates just {e['cuped_corr']['spend']:.2f} "
      f"with two-week spend.\n")
    b = e["bayes"]
    w("**Bayesian view.** Probability each e-mail increases spend: "
      + "; ".join(f"{a} {_prob(b[a]['spend']['prob_better'])} (expected loss if wrong "
                  f"${b[a]['spend']['expected_loss']:.4f} per customer)" for a in C.TREATMENTS) + ".\n")
    s = e["subgroups"]
    nz = s[(s.metric == "spend") & (s.p_holm > C.ALPHA)]
    w("**Subgroups.** No measurable spend effect (Holm p > 0.05) in: "
      + "; ".join(f"{x.arm} / {x.subgroup}" for _, x in nz.iterrows()) + ".\n")

    # ---------------- part 2 ----------------
    w("## Part 2. Could we have measured it without a test?\n")
    t = o["truth"]
    w(f"Treated customers were thinned by a targeting rule favouring engaged customers "
      f"({o['n_treated_kept']:,} of {o['n_treated_full']:,} kept). Randomized truth for the "
      f"Women's e-mail effect on visit rate: {t['effect']:.2%} ({t['ci_low']:.2%} to {t['ci_high']:.2%}).\n")
    w("| Method | Estimate | 95% CI | Gap to truth |")
    w("|---|---|---|---|")
    for k, v in {**o["estimates"], **o["unmeasured"]}.items():
        w(f"| {k} | {v['effect']:.2%} | {v['ci_low']:.2%} to {v['ci_high']:.2%} | "
          f"{(v['effect'] - t['effect']) * 100:+.2f} pts |")
    d = o["diagnostics"]
    w(f"\nBalance: largest |SMD| {d['balance_before'].abs().max():.2f} before weighting, "
      f"{d['balance_after'].abs().max():.2f} after. Overlap: "
      f"{d['overlap']['share_treated_inside']:.0%} of treated and "
      f"{d['overlap']['share_control_inside']:.0%} of untreated customers inside the common range.\n")

    # ---------------- part 3 ----------------
    w("## Part 3. How should the next test be designed?\n")
    pw = n["power"]
    w(f"**Sample size.** With {pw['n_per_arm']:,} customers per group the smallest detectable spend "
      f"effect is ${pw['mde_spend']:.2f} ({pw['mde_spend_pct']:.0%} of the control mean). Detecting a "
      f"10% spend lift needs {pw['n_for_10pct_lift']:,} per group; a 10% visit lift needs "
      f"{pw['n_for_10pct_lift_visit']:,}.\n")
    pr = n["proxy"]
    w(f"**Can visit stand in for spend?** Visitors spend more (customer-level correlation "
      f"{pr['individual_corr']:.2f}), but across {pr['n_segments']} segments the visit effect "
      f"predicts the spend effect only weakly (split-sample correlation {pr['split_sample_corr']:.2f}; "
      f"90% of splits between {pr['corr_range'][0]:.2f} and {pr['corr_range'][1]:.2f}). "
      f"At campaign level both e-mails earn about "
      + " and ".join(f"${v:.0f}" for v in pr["spend_per_extra_visit"].values())
      + " per extra visit.\n")
    cu = n["cuped"]
    w(f"**CUPED.** All pre-period data together predicts {cu['visit']['r2_all_pre_period']:.1%} of "
      f"the variance in visits and {cu['spend']['r2_all_pre_period']:.2%} in spend, so CUPED "
      f"would save little here. A metric measured just before the test would do better.\n")
    c = n["cluster"]
    w(f"**Randomizing whole markets.** {c['n_markets']} markets (location type x channel x "
      f"spending tier), about {c['mean_market_size']:.0f} customers each in the control group. "
      f"Visit-rate ICC {c['icc_visit']:.3f}, design effect {c['design_effect']:.1f}. To detect a "
      f"{c['planned_lift']:.0%} visit lift, customer randomization needs "
      f"{c['n_per_arm_individual']:,} customers per arm; market randomization needs "
      f"{c['markets_per_arm_needed']} markets per arm, more than exist. In one example market "
      f"test, the naive standard error was {c['example_naive']['se']:.4f} and the cluster-robust one "
      f"{c['example_cluster_robust']['se']:.4f}.\n")
    p = c["power"]
    w(f"Simulated power: customers randomized {p['individual']:.0%}, markets randomized "
      f"{p['cluster']:.0%}. Randomizing markets but analysing customers as independent declared "
      f"an effect {p['false_positive_rate_wrong_analysis']:.0%} of the time when there was none "
      f"(should be {C.ALPHA:.0%}).\n")
    (C.REPORT_DIR / "summary.md").write_text("\n".join(L) + "\n")
