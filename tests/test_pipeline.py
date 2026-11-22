"""End-to-end check: the whole pipeline runs on synthetic data and writes every output."""
from pathlib import Path

import run_all


def test_pipeline_runs_on_synthetic_data(tmp_path: Path):
    res = run_all.main(synthetic=True, out_dir=tmp_path)
    rep = tmp_path / "reports"
    for f in ["summary.md", "metrics.json", "experiment_lift.png", "observational_benchmark.png",
              "star_cluster.png", "cuped_variance_reduction.png", "proxy_validation.png",
              "experiment_effects.csv"]:
        assert (rep / f).exists(), f"missing {f}"
    assert res["experiment"]["srm"]["ok"]
    # the synthetic LaLonde effect is 1,800: doubly robust should land near it
    aipw = res["lalonde"]["estimates"]["Doubly robust (AIPW)"]["effect"]
    assert 800 < aipw < 2800
