"""Study 3: cluster randomization, using Tennessee STAR (Krueger, 1999).

In 1985-86, kindergarten students and teachers in 79 Tennessee schools were
randomly assigned, within each school, to small classes (13-17 students),
regular classes, or regular classes with a teacher's aide. We compare small
with regular classes.

Students in the same school share teachers, neighbourhoods and resources, so
their scores are correlated. The analysis must respect that, and so must the
design of any future test: randomizing whole schools costs a lot of power.
"""
from __future__ import annotations

from .. import config as C
from ..measure import clustered as CL


def run(s) -> dict:
    d = s[s["class_type"].isin(["small", "regular"])].reset_index(drop=True)
    y = C.STAR_OUTCOME
    controls = ["girl", "free_lunch"]
    res = {"n_students": len(d), "n_schools": int(d["school"].nunique())}
    res["models"] = {
        "Naive (students independent)": CL.ols_effect(d, y, "small"),
        "Cluster-robust by school": CL.ols_effect(d, y, "small", cluster="school"),
        "School fixed effects + cluster-robust": CL.ols_effect(
            d, y, "small", fixed_effects="school", cluster="school"),
        "+ student covariates": CL.ols_effect(
            d, y, "small", controls, fixed_effects="school", cluster="school"),
    }
    res["icc"] = CL.icc(d[y], d["school"])
    res["mean_cluster_size"] = float(d.groupby("school").size().mean())
    res["design_effect"] = CL.design_effect(res["icc"], res["mean_cluster_size"])
    res["effective_n"] = res["n_students"] / res["design_effect"]
    # power of a future test on these students: individual vs school randomization
    reg = s[s["class_type"] == "regular"].reset_index(drop=True)
    res["power_effect"] = 0.15
    res["power"] = CL.power_by_design(reg, y, "school", effect=res["power_effect"])
    return res
