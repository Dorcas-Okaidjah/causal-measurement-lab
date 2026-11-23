"""End-to-end check: the whole pipeline runs on synthetic data and writes every output."""
from pathlib import Path

import run_all


def test_pipeline_runs_on_synthetic_data(tmp_path: Path):
    res = run_all.main(synthetic=True, out_dir=tmp_path)
    rep = tmp_path / "reports"
    for f in ["summary.md", "metrics.json", "experiment_lift.png", "bayes_probability.png",
              "without_a_test.png", "targeting_overlap.png", "proxy_validation.png",
              "cuped_variance_reduction.png", "market_randomization.png", "experiment_effects.csv"]:
        assert (rep / f).exists(), f"missing {f}"
    assert res["part1"]["srm"]["ok"]
    # synthetic visits have a known positive effect: part 2's doubly robust estimate must see it
    assert res["part2"]["estimates"]["Doubly robust (AIPW)"]["ci_low"] > 0
