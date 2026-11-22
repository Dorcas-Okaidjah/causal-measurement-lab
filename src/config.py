"""Central configuration. Change settings here, not inside the modules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42
ALPHA = 0.05            # significance level for every test and interval (95%)

# --- Data sources (all public, no login needed) -----------------------------
# Hillstrom MineThatData e-mail experiment (64,000 customers, randomized).
HILLSTROM_URLS = [
    "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
    # mirror used when the original site is unreachable (same file, checked below)
    "https://raw.githubusercontent.com/Carlos-Orellana-Heredia/hillstrom-email-experiment/main/data/raw/hillstrom_email_experiment.csv",
]
# SHA-256 of the original file, as published by the libuplift package. A file
# that does not match is rejected, so a mirror can never feed us altered data.
HILLSTROM_SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"

# Tennessee STAR class-size experiment, from the R package AER (via GitHub).
STAR_URL = "https://raw.githubusercontent.com/cran/AER/master/data/STAR.rda"
# LaLonde / NSW job-training data: loaded from the `causaldata` Python package.

# --- Experiment analysis ------------------------------------------------------
CONTROL = "No E-Mail"
TREATMENTS = ["Mens E-Mail", "Womens E-Mail"]
METRICS = ["visit", "conversion", "spend"]   # leading indicator -> outcome
# Pre-period covariates: measured before the e-mail, so they cannot be
# affected by it. `history` (spend in the past year) is the CUPED covariate.
PRE_COVARIATES = ["recency", "history", "mens", "womens", "newbie"]
PRE_CATEGORICAL = ["zip_code", "channel"]
CUPED_COVARIATE = "history"

# Bayesian A/B: weak priors that let the data speak.
BETA_PRIOR = (1.0, 1.0)          # uniform prior on a conversion or visit rate
N_POSTERIOR_DRAWS = 200_000

# Power analysis for the next campaign
POWER_TARGET = 0.80

# --- Observational (non-randomized) analysis ----------------------------------
# To test observational methods we need a known right answer. We take the
# randomized Women's-email test and remove treated customers in a way that
# depends on their pre-period behaviour, mimicking a campaign that a marketer
# targeted at engaged customers. The randomized estimate is the truth that
# each method should recover.
OBS_TREATMENT = "Womens E-Mail"
OBS_OUTCOME = "visit"
N_FOLDS = 5                      # cross-fitting folds for AIPW and double ML
PROPENSITY_CLIP = (1e-6, 1 - 1e-6)  # only guards against division by zero
AIPW_CLIP = (0.01, 0.99)            # doubly robust / DML: limit extreme weights
PSM_CALIPER = 0.2                # in standard deviations of the logit propensity

# --- LaLonde ------------------------------------------------------------------
LALONDE_COVARIATES = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]
LALONDE_BENCHMARK_NOTE = "NSW randomized experiment"

# --- STAR cluster analysis ---------------------------------------------------
STAR_OUTCOME = "score"          # average of kindergarten reading and maths, standardized
N_POWER_SIMS = 500

# Chart colours (validated colour-blind-safe set + neutral ink)
COLORS = {"a": "#2a78d6", "b": "#eb6834", "c": "#1baa7a", "d": "#8a63d2",
          "ink": "#0b0b0b", "grid": "#e4e3df", "muted": "#52514e"}
