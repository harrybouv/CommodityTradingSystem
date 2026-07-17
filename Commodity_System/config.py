# config.py

import os
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env", override=False)

STRATEGY_REPORT_THEME = "light"

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
# MACRO DATA SETTINGS
# ============================================================

MACRO_UNIVERSE = {
    "UUP": {
        "name": "US Dollar Bullish ETF",
        "macro_role": "usd",
        "description": "Tradable USD strength proxy",
    },
    "^TNX": {
        "name": "10-Year Treasury Yield",
        "macro_role": "rates",
        "description": "US 10-year yield proxy",
    },
    "TIP": {
        "name": "TIPS ETF",
        "macro_role": "inflation",
        "description": "Inflation-protected Treasuries proxy",
    },
    "IEF": {
        "name": "7-10 Year Treasury ETF",
        "macro_role": "nominal_rates",
        "description": "Nominal Treasury duration proxy used with TIP",
    },
    "SPY": {
        "name": "S&P 500 ETF",
        "macro_role": "growth_risk",
        "description": "Equity risk appetite proxy",
    },
    "^VIX": {
        "name": "VIX Index",
        "macro_role": "stress",
        "description": "Equity volatility / stress proxy",
    },
    "DBC": {
        "name": "Broad Commodity ETF",
        "macro_role": "broad_commodities",
        "description": "Broad commodity trend proxy",
    },
}

MACRO_TICKERS = list(MACRO_UNIVERSE.keys())

MACRO_PRICE_DATA_PATH = RAW_DATA_DIR / "macro_prices.csv"
MACRO_DATA_QUALITY_REPORT_PATH = RAW_DATA_DIR / "macro_data_quality_report.csv"

MIN_MACRO_PRICE_HISTORY_DAYS = 252

# ============================================================
# BACKTEST SETTINGS
# ============================================================

START_DATE = "2015-01-01"
END_DATE = None

CASH_ANNUAL_YIELD = 0.04

REBALANCE_FREQUENCY = "W-FRI"  # legacy / optional

BACKTEST_REBALANCE_MODE = "monthly"  # "daily", "weekly", or "monthly"

INITIAL_CAPITAL = 10_000

TRADING_DAYS_PER_YEAR = 252

# ============================================================
# BACKTEST V2 REALISM SETTINGS
# ============================================================

BACKTEST_V2_ENABLED = True

# Main realism switches
BACKTEST_V2_USE_DETAILED_TRANSACTION_COSTS = True
BACKTEST_V2_USE_COMMISSION_COSTS = True
BACKTEST_V2_USE_BID_ASK_SPREAD_COSTS = True
BACKTEST_V2_USE_SLIPPAGE_COSTS = True
BACKTEST_V2_USE_LEGACY_FLAT_COSTS = False
BACKTEST_V2_USE_TURNOVER_CONTROLS = True
BACKTEST_V2_USE_NO_TRADE_BAND = True
BACKTEST_V2_NO_TRADE_BAND = 0.005          # 0.5% portfolio-weight change ignored

BACKTEST_V2_USE_MAX_REBALANCE_TURNOVER = True
BACKTEST_V2_MAX_REBALANCE_TURNOVER = 0.50  # max 50% of portfolio traded per rebalance
BACKTEST_V2_PARTIAL_REBALANCE_FRACTION = 1.00

BACKTEST_V2_USE_LIQUIDITY_CAPS = True
BACKTEST_V2_ADV_LOOKBACK_DAYS = 20
BACKTEST_V2_MAX_ADV_PARTICIPATION = 0.01   # max 1% of rolling ADV per trade

# If None, liquidity caps use actual simulated equity.
# Set to e.g. 100_000, 1_000_000, 10_000_000 for capacity testing.
BACKTEST_V2_LIQUIDITY_TEST_CAPITAL = None

BACKTEST_V2_USE_EXECUTION_DELAY = True
BACKTEST_V2_EXECUTION_DELAY_DAYS = 1

BACKTEST_V2_USE_PORTFOLIO_WEIGHT_DRIFT = True

# Cost scenario used for the main run
BACKTEST_V2_COST_SCENARIO = "base"  # "base", "conservative", "stress"

# These are assumptions, not facts. Refine later if you collect actual spread data.
BACKTEST_V2_TRANSACTION_COST_ASSUMPTIONS = {
    "base": {
        "GLD":  {"commission_bps": 0.00, "full_spread_bps": 1.0,  "slippage_bps": 0.5},
        "SLV":  {"commission_bps": 0.00, "full_spread_bps": 2.0,  "slippage_bps": 0.75},
        "USO":  {"commission_bps": 0.00, "full_spread_bps": 3.0,  "slippage_bps": 1.5},
        "UNG":  {"commission_bps": 0.00, "full_spread_bps": 5.0,  "slippage_bps": 2.5},
        "CPER": {"commission_bps": 0.00, "full_spread_bps": 10.0, "slippage_bps": 5.0},
        "DBA":  {"commission_bps": 0.00, "full_spread_bps": 8.0,  "slippage_bps": 4.0},
    },
    "conservative": {
        "GLD":  {"commission_bps": 0.00, "full_spread_bps": 2.0,  "slippage_bps": 1.0},
        "SLV":  {"commission_bps": 0.00, "full_spread_bps": 4.0,  "slippage_bps": 1.5},
        "USO":  {"commission_bps": 0.00, "full_spread_bps": 6.0,  "slippage_bps": 3.0},
        "UNG":  {"commission_bps": 0.00, "full_spread_bps": 10.0, "slippage_bps": 5.0},
        "CPER": {"commission_bps": 0.00, "full_spread_bps": 20.0, "slippage_bps": 10.0},
        "DBA":  {"commission_bps": 0.00, "full_spread_bps": 16.0, "slippage_bps": 8.0},
    },
    "stress": {
        "GLD":  {"commission_bps": 0.00, "full_spread_bps": 4.0,  "slippage_bps": 2.0},
        "SLV":  {"commission_bps": 0.00, "full_spread_bps": 8.0,  "slippage_bps": 4.0},
        "USO":  {"commission_bps": 0.00, "full_spread_bps": 12.0, "slippage_bps": 6.0},
        "UNG":  {"commission_bps": 0.00, "full_spread_bps": 20.0, "slippage_bps": 10.0},
        "CPER": {"commission_bps": 0.00, "full_spread_bps": 35.0, "slippage_bps": 18.0},
        "DBA":  {"commission_bps": 0.00, "full_spread_bps": 30.0, "slippage_bps": 15.0},
    },
}

BACKTEST_V2_RUN_SCENARIO_TESTS = True

BACKTEST_V2_SCENARIOS = [
    {
        "name": "no_costs_delay_1d",
        "cost_scenario": "base",
        "execution_delay_days": 1,
        "use_detailed_transaction_costs": False,
        "use_legacy_flat_costs": False,
        "use_commission_costs": False,
        "use_bid_ask_spread_costs": False,
        "use_slippage_costs": False,
        "use_liquidity_caps": False,
        "use_turnover_controls": False,
    },

    {
        "name": "base_costs_delay_1d",
        "cost_scenario": "base",
        "execution_delay_days": 1,
    },
    {
        "name": "conservative_costs_delay_1d",
        "cost_scenario": "conservative",
        "execution_delay_days": 1,
    },
    {
        "name": "stress_costs_delay_1d",
        "cost_scenario": "stress",
        "execution_delay_days": 1,
    },
    {
        "name": "base_costs_delay_2d",
        "cost_scenario": "base",
        "execution_delay_days": 2,
    },
    {
        "name": "base_costs_delay_5d",
        "cost_scenario": "base",
        "execution_delay_days": 5,
    },
]

# ============================================================
# PORTFOLIO CONSTRUCTION EXPERIMENTS
# ============================================================

USE_REBALANCE_THRESHOLD = True
REBALANCE_THRESHOLD = 0.17

USE_TURNOVER_PENALTY = True
TURNOVER_PENALTY = 0.2

# ============================================================
# EMERGENCY REBALANCING
# ============================================================

USE_EMERGENCY_REBALANCE = False
USE_ASSET_LEVEL_EMERGENCY = True
USE_PORTFOLIO_LEVEL_EMERGENCY = False

EMERGENCY_MIN_POSITION_WEIGHT = 0.10
EMERGENCY_ASSET_DROP_THRESHOLD = 0.15
EMERGENCY_HARD_EXIT_TARGET_WEIGHT = 0.02

EMERGENCY_MIN_PORTFOLIO_EXPOSURE = 0.50
EMERGENCY_EXPOSURE_DROP_THRESHOLD = 0.30

EMERGENCY_CUT_SPEED = 1.00

EMERGENCY_ASSET_COOLDOWN_DAYS = 10
EMERGENCY_PORTFOLIO_COOLDOWN_DAYS = 15

# ============================================================
# DYNAMIC CASH ALLOCATION
# ============================================================

USE_DYNAMIC_CASH_ALLOCATION = False

DYNAMIC_CASH_MAX_CUT = 0.20

DYNAMIC_CASH_MIN_MULTIPLIER = 0.80
DYNAMIC_CASH_STRESS_START = 0.45
DYNAMIC_CASH_QUALITY_FLOOR = 0.60

DYNAMIC_CASH_STRESS_WEIGHT = 0.65
DYNAMIC_CASH_WEAK_QUALITY_WEIGHT = 0.35


# ============================================================
# BEAR-ONLY SHORT OVERLAY
# ============================================================
# Research/production toggle.

BEAR_SHORT_OVERLAY_ENABLED = True
BEAR_SHORT_SCORE_THRESHOLD = 0.35

# Short sizing caps.
BEAR_SHORT_MAX_TOTAL_SHORT = 0.25       # max total gross short exposure
BEAR_SHORT_MAX_SINGLE_SHORT = 0.08      # max short per ETF

# Portfolio construction rules.
BEAR_SHORT_OVERRIDE_LONGS = True        # if candidate is shorted, remove its long weight first
BEAR_SHORT_SCALE_LONGS_TO_MAKE_ROOM = True
BEAR_SHORT_ALLOWED_REGIMES = ["bear"]

# Regime classifier: price/breadth based, not final-score based.
BEAR_SHORT_REGIME_MIN_BEAR_VOTES = 3

BEAR_SHORT_BASKET_MA_WINDOW = 200
BEAR_SHORT_BASKET_RETURN_3M_THRESHOLD = -0.03
BEAR_SHORT_BASKET_RETURN_6M_THRESHOLD = -0.05
BEAR_SHORT_BASKET_RETURN_12M_THRESHOLD = -0.08
BEAR_SHORT_BASKET_DRAWDOWN_THRESHOLD = -0.10

BEAR_SHORT_BREADTH_ABOVE_MA_THRESHOLD = 0.35
BEAR_SHORT_BREADTH_POS_3M_THRESHOLD = 0.35
BEAR_SHORT_BREADTH_POS_6M_THRESHOLD = 0.35

# Backtest engine support.
# This must be True whenever negative target weights are allowed.
ALLOW_SHORT_WEIGHTS = BEAR_SHORT_OVERLAY_ENABLED
MAX_TOTAL_PORTFOLIO_GROSS_EXPOSURE = 1.00

# ============================================================
# VOLATILITY TARGETING
# ============================================================

VOL_TARGETING_ENABLED = False

TARGET_PORTFOLIO_VOL = 0.12
VOL_TARGET_LOOKBACK_DAYS = 60

VOL_TARGET_VOL_BUFFER = 1.10

VOL_TARGET_MIN_SCALE = 0.70
VOL_TARGET_MAX_SCALE = 1.00

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
# GOLD-SPECIFIC OVERLAY SETTINGS
# ============================================================

GOLD_OVERLAY_ENABLED = True
GOLD_OVERLAY_BLEND_WEIGHT = 0.10
GOLD_OVERLAY_REQUIRE_FEATURES = True
GOLD_FEATURE_ASOF_TOLERANCE_DAYS = 10

# Core production toggles for first ablation set
GOLD_USE_REAL_YIELD = True
GOLD_USE_USD = True
GOLD_USE_STRESS = False

# Built for later, off until tested
GOLD_USE_POLICY_RATE_REGIME = False
GOLD_USE_CENTRAL_BANK_DEMAND = False
GOLD_USE_POSITIONING_CROWDING = False

# Weights are normalised across enabled components inside GLD_scoring.py.
GOLD_COMPONENT_WEIGHTS = {
    "gold_real_yield_score": 0.5,
    "gold_usd_score": 0.5,
    "gold_stress_score": 0.00,
    "gold_policy_rate_score": 0.00,
    "gold_central_bank_score": 0.00,
    "gold_positioning_score": 0.00,
}

# ============================================================
# SILVER-SPECIFIC OVERLAY SETTINGS
# ============================================================

SILVER_OVERLAY_ENABLED = True
SILVER_OVERLAY_BLEND_WEIGHT = 0.1
SILVER_OVERLAY_REQUIRE_FEATURES = True
SILVER_FEATURE_ASOF_TOLERANCE_DAYS = 10

SILVER_USE_GOLD_RATIO = False
SILVER_USE_COPPER_RATIO = False
SILVER_USE_GOLD_CONFIRMATION = True
SILVER_USE_USD = False
SILVER_USE_REAL_YIELD = True

SILVER_COMPONENT_WEIGHTS = {
    "silver_gold_ratio_score": 0.00,
    "silver_copper_ratio_score": 0.0,
    "silver_gold_confirmation_score": 0.05,
    "silver_usd_score": 0.00,
    "silver_real_yield_score": 0.95,
}

# ============================================================
# COPPER-SPECIFIC OVERLAY SETTINGS
# ============================================================

COPPER_OVERLAY_ENABLED = True
COPPER_OVERLAY_BLEND_WEIGHT = 0.03
COPPER_OVERLAY_REQUIRE_FEATURES = True

# Monthly China/economic data needs wider as-of tolerance than daily market data.
COPPER_FEATURE_ASOF_TOLERANCE_DAYS = 35

COPPER_USE_CHINA_ELECTRICITY = False
COPPER_USE_CHINA_CLI = True
COPPER_USE_USD = False
COPPER_USE_BROAD_COMMODITY_TREND = False
COPPER_USE_OIL_PRICE = False
COPPER_USE_GLOBAL_GROWTH = False

# Release-lag assumptions. These prevent accidental lookahead bias.
COPPER_CHINA_ELECTRICITY_RELEASE_LAG_DAYS = 20
COPPER_CHINA_CLI_RELEASE_LAG_DAYS = 15

COPPER_COMPONENT_WEIGHTS = {
    "copper_china_electricity_score": 0.0,
    "copper_china_cli_score": 1.0,
    "copper_usd_score": 0.0,
    "copper_broad_commodity_trend_score": 0.0,
    "copper_oil_price_score": 0.0,
    "copper_global_growth_score": 0.0,
}

# ============================================================
# USO / OIL OVERLAY CONFIG
# ============================================================

USO_OVERLAY_ENABLED = True
USO_OVERLAY_BLEND_WEIGHT = 0.25
USO_OVERLAY_REQUIRE_FEATURES = True
USO_FEATURE_ASOF_TOLERANCE_DAYS = 10

USO_USE_INVENTORY_TIGHTNESS = False
USO_USE_CUSHING_TIGHTNESS = False
USO_USE_CURVE_ROLL = True
USO_USE_SUPPLY_REFINERY = True
USO_USE_GLOBAL_DEMAND = False
USO_USE_USD = True

USO_COMPONENT_WEIGHTS = {
    "oil_inventory_tightness_score": 0.0,
    "oil_cushing_tightness_score": 0.0,
    "oil_curve_roll_score": 0.0,
    "oil_supply_refinery_score": 0.5,
    "oil_global_demand_score": 0.0,
    "oil_usd_score": 0.5,
}

# ============================================================
# NATURAL GAS / UNG OVERLAY CONFIG
# ============================================================

UNG_OVERLAY_ENABLED = False
UNG_OVERLAY_BLEND_WEIGHT = 0.075
UNG_OVERLAY_REQUIRE_FEATURES = True
UNG_FEATURE_ASOF_TOLERANCE_DAYS = 10

UNG_USE_WEATHER_DEMAND = False
UNG_USE_STORAGE_TIGHTNESS = True
UNG_USE_STORAGE_MOMENTUM = True
UNG_USE_CURVE_ROLL = True
UNG_USE_SUPPLY_PRESSURE = False
UNG_USE_LNG_EXPORT_DEMAND = False
UNG_USE_OIL_RELATIVE_VALUE = False
UNG_USE_ENERGY_CONFIRMATION = False

UNG_COMPONENT_WEIGHTS = {
    "gas_weather_demand_score": 0.00,
    "gas_storage_tightness_score": 0.40,
    "gas_storage_momentum_score": 0.40,
    "gas_curve_roll_score": 0.20,
    "gas_supply_pressure_score": 0.00,
    "gas_lng_export_demand_score": 0.00,
    "gas_oil_relative_value_score": 0.00,
    "gas_energy_confirmation_score": 0.00,
}

# ============================================================
# AGRICULTURE / DBA OVERLAY CONFIG
# ============================================================

DBA_OVERLAY_ENABLED = True
DBA_OVERLAY_BLEND_WEIGHT = 0.15
DBA_OVERLAY_REQUIRE_FEATURES = True
DBA_FEATURE_ASOF_TOLERANCE_DAYS = 10

# Core V1 agriculture overlay toggles.
# Exports are intentionally off until ablated. They are noisy and easy to overfit.
DBA_USE_USD = False
DBA_USE_RATES = False
DBA_USE_CROP_MOMENTUM = False
DBA_USE_CROP_RELATIVE_STRENGTH = True
DBA_USE_BROAD_COMMODITY_CONFIRMATION = False
DBA_USE_SEASONALITY = False
DBA_USE_EXPORT_DEMAND = False

# Weights are normalised across enabled components inside DBA_scoring.py.
DBA_COMPONENT_WEIGHTS = {
    "agri_usd_score": 0.0,
    "agri_rates_score": 0.00,
    "agri_crop_momentum_score": 0.00,
    "agri_crop_relative_strength_score": 1.0,
    "agri_broad_commodity_confirmation": 0.00,
    "agri_seasonality_score": 0.0,
    "agri_export_demand_score": 0.00,
}

# ============================================================
# AGRICULTURE DATA DOWNLOAD SETTINGS
# ============================================================

DBA_DOWNLOAD_ESR_EXPORTS = False
DBA_USE_CACHED_ESR_EXPORTS = True

# ============================================================
# SCORE WEIGHTS
# ============================================================

SCORE_WEIGHTS = {
    "momentum_score": 0.2300,
    "relative_strength_score": 0.1600,
    "trend_score": 0.0300,
    "trend_persistence_score": 0.1000,
    "volatility_score": 0.1900,
    "risk_score": 0.2600,
    "macro_score": 0.0300,
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

DRAWDOWN_DE_RISKING_ENABLED = False
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

MIN_AVERAGE_DAILY_VOLUME = 0


# ============================================================
# OUTPUT SETTINGS
# ============================================================

SAVE_INTERMEDIATE_FILES = True

ROUND_WEIGHTS_TO_DP = 4
ROUND_SCORES_TO_DP = 4