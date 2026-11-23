"""Download and prepare the Hillstrom e-mail campaign data.

Usage:
    python -m src.data --download     # fetch the data (public, no login)
    python -m src.data --inspect      # print a short summary
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.request

import numpy as np
import pandas as pd

from . import config as C

HILLSTROM_FILE = C.DATA_DIR / "hillstrom.csv"


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


def download() -> None:
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_hillstrom()


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


# --------------------------------------------------------------------------- #
def inspect() -> None:
    h = load_hillstrom()
    print(f"\n{len(h):,} customers")
    print(h.groupby("arm")[C.METRICS].mean().round(4).to_string())
    print("\npre-period covariates:")
    print(h[C.PRE_COVARIATES].describe().T.round(2).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    p.add_argument("--inspect", action="store_true")
    a = p.parse_args()
    if a.download:
        download()
    if a.inspect:
        inspect()
