"""Run the whole e-mail campaign measurement project.

    python run_all.py               # real data (download first: python -m src.data --download)
    python run_all.py --synthetic   # fake data with known answers, no download (smoke test)

Part 1: did the campaign work?             (randomized test readout)
Part 2: could we have measured it without a test?   (observational methods vs the truth)
Part 3: how should the next test be designed?       (power, proxy, CUPED, market randomization)

Outputs land in reports/: charts, tables, metrics.json and summary.md.
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from src import config as C, plots, report
from src import data as D
from src.analyses import part1_readout, part2_without_a_test, part3_next_test


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
    h = D.make_synthetic_experiment() if synthetic else D.load_hillstrom()

    print("2/5 part 1: did the campaign work?")
    p1 = part1_readout.run(h)
    print("3/5 part 2: could we have measured it without a test?")
    p2 = part2_without_a_test.run(h)
    print("4/5 part 3: how should the next test be designed?")
    p3 = part3_next_test.run(h, p1["effects"])

    print("5/5 writing charts, tables and the summary")
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plots.lift_forest(p1["effects"])
    plots.bayes_chart(p1["bayes"])
    plots.benchmark_forest(p2)
    plots.overlap_hist(p2["diagnostics"]["propensity"], p2["diagnostics"]["treated"],
                       "targeting_overlap.png",
                       "Targeted customers still overlap with untargeted ones")
    plots.proxy_scatter(p3["proxy"]["segments_table"])
    plots.cuped_curve(p1["cuped_corr"]["spend"])
    plots.market_chart(p3["cluster"])
    p1["effects"].to_csv(C.REPORT_DIR / "experiment_effects.csv", index=False)
    p1["subgroups"].to_csv(C.REPORT_DIR / "experiment_subgroups.csv", index=False)
    res = {"synthetic": synthetic, "part1": p1, "part2": p2, "part3": p3}
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
