# config.py

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"

PRICE_DATA_PATH = RAW_DATA_DIR / "commodity_prices.csv"
FEATURES_PATH = PROCESSED_DATA_DIR / "features.csv"
SCORES_PATH = PROCESSED_DATA_DIR / "scores.csv"
WEIGHTS_PATH = PROCESSED_DATA_DIR / "target_weights.csv"
BACKTEST_RESULTS_PATH = RESULTS_DIR / "backtest_results.csv"


# ============================================================
# UNIVERSE
# ============================================================

UNIVERSE = {
    "GLD": {
        "name": "Gold",
        "group": "precious_metals",
        "description": "Gold ETF proxy",
    },
    "SLV": {
        "name": "Silver",
        "group": "precious_metals",
        "description": "Silver ETF proxy",
    },
    "USO": {
        "name": "Oil",
        "group": "energy",
        "description": "Crude oil ETF proxy",
    },
    "UNG": {
        "name": "Natural Gas",
        "group": "energy",
        "description": "Natural gas ETF proxy",
    },
    "CPER": {
        "name": "Copper",
        "group": "industrial_metals",
        "description": "Copper ETF proxy",
    },
    "DBA": {
        "name": "Agriculture",
        "group": "agriculture",
        "description": "Agriculture ETF proxy",
    },
}

TICKERS = list(UNIVERSE.keys())


# ============================================================
# BACKTEST SETTINGS
# ============================================================

START_DATE = "2015-01-01"
END_DATE = None

CASH_ANNUAL_YIELD = 0.04

START_DATE = "2015-01-01"
END_DATE = None

REBALANCE_FREQUENCY = "W-FRI"  # legacy / optional

BACKTEST_REBALANCE_MODE = "monthly"  # "daily", "weekly", or "monthly"

INITIAL_CAPITAL = 10_000

TRADING_DAYS_PER_YEAR = 252


# ============================================================
# FEATURE WINDOWS
# ============================================================

MOMENTUM_WINDOWS = [20, 60, 120]

TREND_FAST_WINDOW = 50
TREND_SLOW_WINDOW = 200

VOLATILITY_WINDOWS = [20, 60]

DRAWDOWN_WINDOW = 60

CORRELATION_WINDOW = 60

VOLUME_WINDOW = 20


# ============================================================
# SCORE WEIGHTS
# ============================================================

SCORE_WEIGHTS = {
    "momentum_score": 0.21,
    "relative_strength_score": 0.16,
    "trend_score": 0.04,
    "trend_persistence_score": 0.10,
    "volatility_score": 0.20,
    "risk_score": 0.29,
}
MIN_SCORE_TO_HOLD = 0.65
MAX_ASSET_WEIGHT = 0.32

# ============================================================
# PORTFOLIO CONSTRAINTS
# ============================================================

ALLOW_CASH = True

MIN_ASSET_WEIGHT = 0.00


MAX_GROUP_WEIGHT = {
    "precious_metals": 0.50,
    "energy": 0.40,
    "industrial_metals": 0.30,
    "agriculture": 0.30,
}

MAX_TOTAL_RISK_ASSET_EXPOSURE = 1.00

TARGET_PORTFOLIO_VOL = 0.12  # 12% annualised volatility target


VOL_TARGETING_ENABLED = True

DRAWDOWN_DE_RISKING_ENABLED = True
DRAWDOWN_DE_RISK_THRESHOLD = -0.10  # reduce exposure if portfolio drawdown worse than -10%


# ============================================================
# TRANSACTION COSTS
# ============================================================

TRANSACTION_COST_BPS = 5  # 5 basis points per trade

SLIPPAGE_BPS = 5

TOTAL_COST_BPS = TRANSACTION_COST_BPS + SLIPPAGE_BPS


# ============================================================
# DATA CLEANING SETTINGS
# ============================================================

MIN_PRICE_HISTORY_DAYS = 252

FORWARD_FILL_LIMIT = 5

MIN_AVERAGE_DAILY_VOLUME = 100_000


# ============================================================
# OUTPUT SETTINGS
# ============================================================

SAVE_INTERMEDIATE_FILES = True

ROUND_WEIGHTS_TO_DP = 4
ROUND_SCORES_TO_DP = 4