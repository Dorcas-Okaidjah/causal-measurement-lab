"""Central configuration for the e-mail campaign measurement project.
Change settings here, not inside the modules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42
ALPHA = 0.05            # significance level for every test and interval (95%)

# --- Data source (public, no login needed) -----------------------------------
# Hillstrom MineThatData e-mail experiment (64,000 customers, randomized).
HILLSTROM_URLS = [
    "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
    # mirror used when the original site is unreachable (same file, checked below)
    "https://raw.githubusercontent.com/Carlos-Orellana-Heredia/hillstrom-email-experiment/main/data/raw/hillstrom_email_experiment.csv",
]
# SHA-256 of the original file, as published by the libuplift package. A file
# that does not match is rejected, so a mirror can never feed us altered data.
HILLSTROM_SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"


# --- Part 1: reading out the randomized test --------------------------------
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

# Power analysis (Part 3)
POWER_TARGET = 0.80

# --- Part 2: measuring without a test -----------------------------------------
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

# --- Part 3: designing the next test --------------------------------------------
# Some campaigns can't be randomized customer by customer (a store promotion, a
# regional TV spot), so whole markets are randomized instead. The data has no
# store or region, so we form 'markets' from customers who share a location
# type, purchase channel and spending tier: 57 real groups with real
# differences between them.
MARKET_KEYS = ["zip_code", "channel", "history_segment"]
PLANNED_LIFT = 0.15        # visit lift the next campaign should be able to detect (relative)
N_POWER_SIMS = 500         # simulated tests per design

# Chart colours (validated colour-blind-safe set + neutral ink)
COLORS = {"a": "#2a78d6", "b": "#eb6834", "c": "#1baa7a", "d": "#8a63d2",
          "ink": "#0b0b0b", "grid": "#e4e3df", "muted": "#52514e"}
