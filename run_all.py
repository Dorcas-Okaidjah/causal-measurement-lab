"""Run all three studies end to end.

    python run_all.py               # real data (download first: python -m src.data --download)
    python run_all.py --synthetic   # fake data with known answers, no download (smoke test)

Outputs land in reports/: charts, tables, metrics.json and summary.md.
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from src import config as C, plots, report
from src import data as D
from src.analyses import email_experiment, observational_benchmark, star_cluster


def _jsonable(obj):
    """Turn results (with tables and arrays inside) into plain JSON values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()
                if k not in ("propensity", "treated", "segments_table")}
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return _jsonable(obj.reset_index().to_dict(orient="records")
                         if isinstance(obj, pd.DataFrame) else obj.to_dict())
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def main(synthetic: bool = False, out_dir=None) -> dict:
    t0 = time.time()
    if out_dir is not None:
        C.REPORT_DIR = out_dir / "reports"

    print("1/5 loading data")
    if synthetic:
        h, l, s = (D.make_synthetic_experiment(), D.make_synthetic_lalonde(),
                   D.make_synthetic_star())
    else:
        h, l, s = D.load_hillstrom(), D.load_lalonde(), D.load_star()

    print("2/5 study 1: e-mail experiment readout")
    exp = email_experiment.run(h)
    print("3/5 study 2: observational methods vs known answers")
    obs_h = observational_benchmark.run_hillstrom(h)
    obs_l = observational_benchmark.run_lalonde(l)
    print("4/5 study 3: cluster randomization (STAR)")
    star = star_cluster.run(s)

    print("5/5 writing charts, tables and the summary")
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plots.lift_forest(exp["effects"])
    plots.bayes_chart(exp["bayes"])
    plots.cuped_curve(exp["cuped_corr"]["spend"])
    plots.proxy_scatter(exp["proxy"]["segments_table"])
    plots.benchmark_forest(obs_h, obs_l)
    plots.overlap_hist(obs_l["diagnostics"]["propensity"],
                       obs_l["diagnostics"]["treated"],
                       "lalonde_overlap.png",
                       "LaLonde: many comparison people look nothing like trainees")
    plots.star_chart(star)
    exp["effects"].to_csv(C.REPORT_DIR / "experiment_effects.csv", index=False)
    exp["subgroups"].to_csv(C.REPORT_DIR / "experiment_subgroups.csv", index=False)
    res = {"synthetic": synthetic, "experiment": exp, "hillstrom_obs": obs_h,
           "lalonde": obs_l, "star": star}
    report.write_summary(res)
    res["seconds"] = round(time.time() - t0, 1)
    (C.REPORT_DIR / "metrics.json").write_text(json.dumps(_jsonable(res), indent=2))
    print(f"done in {res['seconds']}s -> {C.REPORT_DIR / 'summary.md'}")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    a = p.parse_args()
    main(a.synthetic)
