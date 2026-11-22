"""Download and prepare the three datasets.

Usage:
    python -m src.data --download     # fetch all three (public, no login)
    python -m src.data --inspect      # print a short summary of each
"""
from __future__ import annotations

import argparse
import hashlib
import tempfile
import urllib.request

import numpy as np
import pandas as pd

from . import config as C

HILLSTROM_FILE = C.DATA_DIR / "hillstrom.csv"
STAR_FILE = C.DATA_DIR / "star.pkl"
LALONDE_FILE = C.DATA_DIR / "lalonde.pkl"


def _fetch(url: str, timeout: int = 120) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download_hillstrom() -> None:
    for url in C.HILLSTROM_URLS:
        try:
            raw = _fetch(url)
        except Exception as e:
            print(f"  could not reach {url.split('/')[2]} ({e}); trying next source")
            continue
        if hashlib.sha256(raw).hexdigest() != C.HILLSTROM_SHA256:
            print(f"  file from {url.split('/')[2]} does not match the original; skipped")
            continue
        HILLSTROM_FILE.write_bytes(raw)
        print(f"  Hillstrom: saved from {url.split('/')[2]} (checksum verified)")
        return
    raise RuntimeError("Hillstrom data could not be downloaded from any source.")


def download_star() -> None:
    import pyreadr

    raw = _fetch(C.STAR_URL)
    with tempfile.NamedTemporaryFile(suffix=".rda") as tmp:
        tmp.write(raw)
        tmp.flush()
        star = next(iter(pyreadr.read_r(tmp.name).values()))
    star.to_pickle(STAR_FILE)
    print(f"  STAR: saved {len(star):,} students")


def download_lalonde() -> None:
    from causaldata import cps_mixtape, nsw_mixtape

    nsw = nsw_mixtape.load_pandas().data.assign(sample="NSW experiment")
    cps = cps_mixtape.load_pandas().data.assign(sample="CPS comparison")
    pd.concat([nsw, cps], ignore_index=True).to_pickle(LALONDE_FILE)
    print(f"  LaLonde: saved {len(nsw)} NSW and {len(cps):,} CPS records")


def download() -> None:
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_hillstrom()
    download_star()
    download_lalonde()


# --------------------------------------------------------------------------- #
# Load and prepare
# --------------------------------------------------------------------------- #
def _require(path):
    if not path.exists():
        raise FileNotFoundError("No data yet. Run:  python -m src.data --download")


def load_hillstrom() -> pd.DataFrame:
    """One row per customer: arm, pre-period covariates and three outcomes."""
    _require(HILLSTROM_FILE)
    df = pd.read_csv(HILLSTROM_FILE)
    df = df.rename(columns={"segment": "arm"})
    df["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})  # typo in source
    return df


def load_star() -> pd.DataFrame:
    """Kindergarten students in STAR with a test score and a class type.

    score = average of standardized reading and maths scores (mean 0, sd 1 in
    regular classes), so the effect reads as 'standard deviations of
    achievement', the convention in education research.
    """
    _require(STAR_FILE)
    s = pd.read_pickle(STAR_FILE)
    k = s[["stark", "readk", "mathk", "schoolidk", "gender", "ethnicity",
           "lunchk", "experiencek"]].dropna(subset=["stark", "readk", "mathk", "schoolidk"])
    k = k.rename(columns={"stark": "class_type", "schoolidk": "school",
                          "lunchk": "free_lunch", "experiencek": "teacher_exp"})
    k["class_type"] = k["class_type"].astype(str)
    k["school"] = k["school"].astype(int)
    for col in ["readk", "mathk", "teacher_exp"]:
        k[col] = pd.to_numeric(k[col], errors="coerce").astype(float)
    reg = k["class_type"] == "regular"
    for col in ["readk", "mathk"]:
        k[col] = (k[col] - k.loc[reg, col].mean()) / k.loc[reg, col].std()
    k[C.STAR_OUTCOME] = (k["readk"] + k["mathk"]) / 2
    k["small"] = (k["class_type"] == "small").astype(int)
    k["free_lunch"] = (k["free_lunch"].astype(str) == "free").astype(int)
    k["girl"] = (k["gender"].astype(str) == "female").astype(int)
    return k.reset_index(drop=True)


def load_lalonde() -> pd.DataFrame:
    _require(LALONDE_FILE)
    df = pd.read_pickle(LALONDE_FILE)
    for c in C.LALONDE_COVARIATES + ["treat", "re78"]:
        df[c] = df[c].astype(float)
    return df


# --------------------------------------------------------------------------- #
# Synthetic data with a known answer (tests and offline demos)
# --------------------------------------------------------------------------- #
def make_synthetic_experiment(n: int = 60000, effect: float = 0.5, seed: int = 0):
    """Fake e-mail test in the Hillstrom schema. The true spend effect of each
    e-mail is `effect` dollars per customer."""
    rng = np.random.default_rng(seed)
    history = rng.gamma(1.5, 160, n)
    recency = rng.integers(1, 13, n)
    df = pd.DataFrame({
        "recency": recency, "history": history.round(2),
        "mens": rng.integers(0, 2, n), "womens": rng.integers(0, 2, n),
        "newbie": rng.integers(0, 2, n),
        "zip_code": rng.choice(["Urban", "Suburban", "Rural"], n),
        "channel": rng.choice(["Web", "Phone", "Multichannel"], n),
        "arm": rng.choice([C.CONTROL] + C.TREATMENTS, n),
    })
    df["history_segment"] = pd.cut(history, [0, 100, 200, 350, 500, 750, 1000, np.inf],
                                   labels=["1) $0 - $100", "2) $100 - $200", "3) $200 - $350",
                                           "4) $350 - $500", "5) $500 - $750", "6) $750 - $1,000",
                                           "7) $1,000 +"]).astype(str)
    treated = (df["arm"] != C.CONTROL).astype(float)
    p_visit = 1 / (1 + np.exp(-(-2.0 + 0.002 * history - 0.05 * recency + 0.4 * treated)))
    df["visit"] = rng.binomial(1, p_visit)
    base = 0.6 + 0.004 * history
    df["spend"] = np.maximum(0, base + effect * treated + rng.normal(0, 4, n))
    df["conversion"] = (df["spend"] > np.quantile(df["spend"], 0.97)).astype(int)
    return df


def make_synthetic_lalonde(n_treated: int = 200, n_comparison: int = 4000, effect: float = 1800,
                           seed: int = 0) -> pd.DataFrame:
    """Fake job-training data: trainees are younger, poorer and less educated
    than the comparison group, and the true effect on trainees is `effect`."""
    rng = np.random.default_rng(seed)

    def people(n, trainee):
        age = rng.normal(26 if trainee else 30, 7, n).clip(17, 55).round()
        educ = rng.normal(10.5 if trainee else 11.5, 2, n).clip(3, 18).round()
        re74 = np.where(rng.random(n) < (0.5 if trainee else 0.25), 0,
                        rng.gamma(2, 4000 if trainee else 6000, n))
        re75 = np.where(rng.random(n) < (0.5 if trainee else 0.25), 0, re74 * 0.8 + rng.gamma(2, 1000, n))
        return pd.DataFrame({"age": age, "educ": educ, "black": rng.binomial(1, 0.6 if trainee else 0.3, n),
                             "hisp": rng.binomial(1, 0.1, n), "marr": rng.binomial(1, 0.3 if trainee else 0.5, n),
                             "nodegree": (educ < 12).astype(int), "re74": re74, "re75": re75})

    def outcome(d, treat):
        return (1000 + 0.6 * d["re75"] + 150 * d["educ"] + rng.normal(0, 4000, len(d))
                + effect * treat).clip(0)

    tr, ex_c, cps = people(n_treated, True), people(n_treated, True), people(n_comparison, False)
    tr["treat"], ex_c["treat"], cps["treat"] = 1.0, 0.0, 0.0
    tr["re78"], ex_c["re78"], cps["re78"] = outcome(tr, 1), outcome(ex_c, 0), outcome(cps, 0)
    tr["sample"] = ex_c["sample"] = "NSW experiment"
    cps["sample"] = "CPS comparison"
    return pd.concat([tr, ex_c, cps], ignore_index=True)


def make_synthetic_star(n_schools: int = 40, per_school: int = 60, effect: float = 0.2,
                        seed: int = 0) -> pd.DataFrame:
    """Fake STAR: students nested in schools, school effects create clustering."""
    rng = np.random.default_rng(seed)
    school = np.repeat(np.arange(n_schools), per_school)
    school_effect = rng.normal(0, 0.5, n_schools)[school]
    class_type = rng.choice(["small", "regular", "regular+aide"], len(school))
    small = (class_type == "small").astype(int)
    return pd.DataFrame({"class_type": class_type, "school": school, "small": small,
                         "girl": rng.integers(0, 2, len(school)),
                         "free_lunch": rng.integers(0, 2, len(school)),
                         C.STAR_OUTCOME: school_effect + effect * small + rng.normal(0, 1, len(school))})


# --------------------------------------------------------------------------- #
def inspect() -> None:
    h = load_hillstrom()
    print(f"\nHillstrom: {len(h):,} customers")
    print(h.groupby("arm")[C.METRICS].mean().round(4).to_string())
    l = load_lalonde()
    print(f"\nLaLonde: {len(l):,} people")
    print(l.groupby(["sample", "treat"])["re78"].agg(["size", "mean"]).round(0).to_string())
    s = load_star()
    print(f"\nSTAR kindergarten: {len(s):,} students in {s.school.nunique()} schools")
    print(s.groupby("class_type")[C.STAR_OUTCOME].agg(["size", "mean"]).round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    p.add_argument("--inspect", action="store_true")
    a = p.parse_args()
    if a.download:
        download()
    if a.inspect:
        inspect()
