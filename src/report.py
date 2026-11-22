"""Write reports/summary.md: the readout a measurement team would hand to leadership."""
from __future__ import annotations

from . import config as C


def _p(x):
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


def _prob(x):
    return "> 99.99%" if x > 0.9999 else f"{x:.2%}"


def write_summary(r: dict) -> None:
    e, o, l, s = r["experiment"], r["hillstrom_obs"], r["lalonde"], r["star"]
    L = []
    w = L.append
    if r.get("synthetic"):
        w("> **Synthetic data run** (smoke test). Numbers below are not meaningful.\n")
    w("# Measurement readout\n")

    # ---------------- experiment ----------------
    w("## 1. E-mail campaign test (Hillstrom, randomized, 64,000 customers)\n")
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
      f"of the variance in spend, because it correlates just "
      f"{e['cuped_corr']['spend']:.2f} with two-week spend. Variance reduction equals the squared "
      f"correlation, so CUPED pays off only with a strongly predictive pre-period metric.\n")
    b = e["bayes"]
    w("**Bayesian view.** Probability each e-mail increases spend: "
      + "; ".join(f"{a} {_prob(b[a]['spend']['prob_better'])} (expected loss if wrong "
                  f"${b[a]['spend']['expected_loss']:.4f} per customer)" for a in C.TREATMENTS) + ".\n")
    pr = e["proxy"]
    w(f"**Is visit a valid proxy for spend?** Visitors spend more (customer-level correlation "
      f"{pr['individual_corr']:.2f}), but across {pr['n_segments']} segments the visit effect "
      f"predicts the spend effect only weakly (split-sample correlation {pr['split_sample_corr']:.2f}; "
      f"90% of splits between {pr['corr_range'][0]:.2f} and {pr['corr_range'][1]:.2f}). "
      f"At campaign level both e-mails earn about "
      + " and ".join(f"${v:.0f}" for v in pr["spend_per_extra_visit"].values())
      + " per extra visit. Use visit as an early read on a whole campaign, not to choose segments.\n")
    pw = e["power"]
    w(f"**Next test.** With {pw['n_per_arm']:,} customers per group the smallest detectable spend "
      f"effect is ${pw['mde_spend']:.2f} ({pw['mde_spend_pct']:.0%} of the control mean). Detecting a "
      f"10% spend lift needs {pw['n_for_10pct_lift']:,} per group; a 10% visit lift needs "
      f"{pw['n_for_10pct_lift_visit']:,}.\n")

    # ---------------- observational ----------------
    w("## 2. Observational methods vs known answers\n")
    for name, res, fmt, gap in [
        ("Hillstrom with targeting bias (effect on visit rate, ATC)", o, "{:.2%}",
         lambda v: f"{v * 100:+.2f} pts"),
        ("LaLonde: NSW trainees vs CPS comparison group (effect on 1978 earnings, ATT)", l,
         "${:,.0f}", lambda v: f"{'+' if v >= 0 else '-'}${abs(v):,.0f}")]:
        t = res["truth"]
        w(f"**{name}.** Experimental truth {fmt.format(t['effect'])} "
          f"({fmt.format(t['ci_low'])} to {fmt.format(t['ci_high'])}).\n")
        w("| Method | Estimate | 95% CI | Gap to truth |")
        w("|---|---|---|---|")
        for k, v in {**res["estimates"], **res.get("unmeasured", {})}.items():
            w(f"| {k} | {fmt.format(v['effect'])} | {fmt.format(v['ci_low'])} to "
              f"{fmt.format(v['ci_high'])} | {gap(v['effect'] - t['effect'])} |")
        d = res["diagnostics"]
        w(f"\nBalance: largest |SMD| {d['balance_before'].abs().max():.2f} before weighting, "
          f"{d['balance_after'].abs().max():.2f} after. Comparison-group share inside the "
          f"treated propensity range: {d['overlap']['share_control_inside']:.0%}.\n")

    # ---------------- STAR ----------------
    w("## 3. Cluster randomization (Tennessee STAR, kindergarten)\n")
    w(f"{s['n_students']:,} students in {s['n_schools']} schools, small vs regular classes, "
      f"randomized within schools. School-level ICC {s['icc']:.2f}, average "
      f"{s['mean_cluster_size']:.0f} students per school. If whole schools were randomized "
      f"instead, the design effect would be {s['design_effect']:.1f}: these students would carry "
      f"the information of about {s['effective_n']:,.0f} independent students.\n")
    w("| Analysis | Effect (sd) | SE | 95% CI |")
    w("|---|---|---|---|")
    for k, v in s["models"].items():
        w(f"| {k} | {v['effect']:.3f} | {v['se']:.3f} | {v['ci_low']:.3f} to {v['ci_high']:.3f} |")
    p = s["power"]
    w(f"\nPower to detect {s['power_effect']} sd: randomizing students {p['individual']:.0%}; "
      f"randomizing schools {p['cluster']:.0%}. Randomizing schools but analysing students as "
      f"independent declared a significant effect {p['false_positive_rate_wrong_analysis']:.0%} of "
      f"the time when there was none (should be {C.ALPHA:.0%}).\n")
    (C.REPORT_DIR / "summary.md").write_text("\n".join(L) + "\n")
