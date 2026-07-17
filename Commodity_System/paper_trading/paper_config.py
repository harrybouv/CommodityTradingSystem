# Commodity_System/paper_trading/paper_config.py

from __future__ import annotations

from pathlib import Path

from config import (
    BASE_DIR,
    INITIAL_CAPITAL,
    CASH_ANNUAL_YIELD,
    TRADING_DAYS_PER_YEAR,
    PRICE_DATA_PATH,
    PROCESSED_DATA_DIR,
    WEIGHTS_PATH,
    TICKERS,
)

PAPER_TICKERS = TICKERS

try:
    from config import TOTAL_COST_BPS
except ImportError:
    TOTAL_COST_BPS = 0.0


# ============================================================
# PAPER TRADING PATHS
# ============================================================

PAPER_TRADING_DIR = BASE_DIR / "paper_trading"

PAPER_STATE_DIR = PAPER_TRADING_DIR / "state"
PAPER_SNAPSHOTS_DIR = PAPER_TRADING_DIR / "snapshots"
PAPER_REPORTS_DIR = PAPER_TRADING_DIR / "reports"

PAPER_TARGET_WEIGHT_SNAPSHOTS_DIR = PAPER_SNAPSHOTS_DIR / "target_weights"
PAPER_FINAL_SCORE_SNAPSHOTS_DIR = PAPER_SNAPSHOTS_DIR / "final_scores"
PAPER_DECISION_SNAPSHOTS_DIR = PAPER_SNAPSHOTS_DIR / "decisions"

PAPER_POSITIONS_PATH = PAPER_STATE_DIR / "paper_positions.csv"
PAPER_TRADES_PATH = PAPER_STATE_DIR / "paper_trades.csv"
PAPER_EQUITY_CURVE_PATH = PAPER_STATE_DIR / "paper_equity_curve.csv"
PAPER_STATE_PATH = PAPER_STATE_DIR / "paper_state.csv"

LATEST_ORDERS_PATH = PAPER_REPORTS_DIR / "latest_orders.csv"
LATEST_PORTFOLIO_PATH = PAPER_REPORTS_DIR / "latest_portfolio.csv"
LATEST_SUMMARY_PATH = PAPER_REPORTS_DIR / "latest_summary.txt"


# ============================================================
# INPUT PATHS FROM EXISTING SYSTEM
# ============================================================

TARGET_WEIGHTS_PATH = WEIGHTS_PATH
FINAL_SCORES_PATH = PROCESSED_DATA_DIR / "final_scores.csv"
PRICE_PATH = PRICE_DATA_PATH


# ============================================================
# PAPER ACCOUNT SETTINGS
# ============================================================

PAPER_ACCOUNT_NAME = "commodity_system_paper"

PAPER_INITIAL_CAPITAL = float(INITIAL_CAPITAL)
PAPER_CASH_ANNUAL_YIELD = float(CASH_ANNUAL_YIELD)
PAPER_TRADING_DAYS_PER_YEAR = int(TRADING_DAYS_PER_YEAR)

# Start simple and close to the backtest.
ALLOW_FRACTIONAL_SHARES = True

# Keep V1 paper fills clean. Add realistic costs later as a validation layer.
PAPER_TOTAL_COST_BPS = float(TOTAL_COST_BPS)
PAPER_COMMISSION_PER_TRADE = 0.00
PAPER_SLIPPAGE_BPS = 0.00

# Ignore irrelevant tiny rebalances.
MIN_TRADE_NOTIONAL = 1.00

# Monthly system. The first run opens the paper portfolio.
# After that, it rebalances only once per new signal month.
PAPER_REBALANCE_MODE = "monthly"

# Manual override for debugging. Keep False for real paper tracking.
FORCE_REBALANCE = False

# If buy / cover orders exceed cash because of costs/slippage, scale them down.
# This avoids accidental negative cash while keeping the script robust.
ALLOW_NEGATIVE_CASH = False

# ============================================================
# PAPER SHORTING SETTINGS
# ============================================================
# Paper mode now supports signed ETF weights:
#   positive target_weight = long
#   negative target_weight = short
# This is synthetic paper shorting only. It does not submit broker orders.
PAPER_ALLOW_SHORTS = True
PAPER_SYNTHETIC_SHORTS_ONLY = True

# Keep all tickers enabled for synthetic paper tracking.
# For a real broker/live layer, set thin or unavailable names to False.
SHORTING_ALLOWED_BY_TICKER = {
    "GLD": True,
    "SLV": True,
    "USO": True,
    "UNG": True,
    "CPER": True,
    "DBA": True,
}

# Must stay aligned with the production/backtest overlay caps.
PAPER_MAX_GROSS_EXPOSURE = 1.00
PAPER_MAX_TOTAL_SHORT = 0.25
PAPER_MAX_SINGLE_SHORT = 0.08


# ============================================================
# CSV SCHEMAS
# ============================================================

POSITIONS_COLUMNS = [
    "ticker",
    "shares",
    "avg_cost",
    "position_type",
    "cost_basis",
    "price_date",
    "price",
    "market_value",
    "abs_market_value",
    "current_weight",
    "abs_current_weight",
    "unrealised_pnl",
    "unrealised_return",
]

TRADES_COLUMNS = [
    "run_timestamp",
    "run_date",
    "signal_date",
    "ticker",
    "side",
    "shares",
    "market_price",
    "fill_price",
    "fill_notional",
    "commission",
    "slippage_cost",
    "cash_flow",
    "realised_pnl",
    "target_weight",
    "current_weight_before",
    "target_shares",
    "reason",
    "status",
]

EQUITY_COLUMNS = [
    "date",
    "signal_date",
    "equity",
    "cash",
    "invested_value",
    "total_exposure",
    "net_exposure",
    "gross_exposure",
    "long_exposure",
    "short_exposure",
    "cash_weight",
    "cash_buffer_weight",
    "daily_pnl",
    "cumulative_pnl",
    "cash_interest_accrued",
    "rebalanced",
    "trades_count",
    "latest_price_date",
]

STATE_COLUMNS = [
    "key",
    "value",
]

DECISION_COLUMNS = [
    "date",
    "signal_date",
    "ticker",
    "final_score",
    "rank",
    "target_weight",
    "current_weight_before",
    "abs_current_weight_before",
    "current_weight_after",
    "target_shares",
    "current_shares_before",
    "current_shares_after",
    "trade_shares",
    "side",
    "market_price",
    "reason",
]

# ============================================================
# COMPATIBILITY ALIASES FOR paper_runner.py
# ============================================================

# Source paths expected by paper_runner.py
TARGET_WEIGHTS_SOURCE_PATH = TARGET_WEIGHTS_PATH
FINAL_SCORES_SOURCE_PATH = FINAL_SCORES_PATH
PRICE_SOURCE_PATH = PRICE_PATH

# Snapshot directory names expected by paper_runner.py
PAPER_TARGET_WEIGHT_SNAPSHOT_DIR = PAPER_TARGET_WEIGHT_SNAPSHOTS_DIR
PAPER_FINAL_SCORE_SNAPSHOT_DIR = PAPER_FINAL_SCORE_SNAPSHOTS_DIR
PAPER_DECISION_SNAPSHOT_DIR = PAPER_DECISION_SNAPSHOTS_DIR

# Rebalance policy expected by paper_runner.py
PAPER_REBALANCE_POLICY = "new_signal_month"

# Ignore tiny weight changes
MIN_WEIGHT_CHANGE_TO_TRADE = 0.0025

# Safe repeated-run behaviour
OVERWRITE_SAME_SIGNAL_DATE_EQUITY_ROW = True

# Output formatting expected by paper_runner.py
DATE_FORMAT = "%Y-%m-%d"
SNAPSHOT_DATE_FORMAT = "%Y%m%d"
FLOAT_FORMAT = "%.10f"

def ensure_paper_directories() -> None:
    """
    Create every paper-trading folder needed by paper_runner.py.

    Kept here so the runner can stay focused on trading/ledger logic.
    """
    dirs = [
        PAPER_TRADING_DIR,
        PAPER_STATE_DIR,
        PAPER_SNAPSHOTS_DIR,
        PAPER_REPORTS_DIR,
        PAPER_TARGET_WEIGHT_SNAPSHOTS_DIR,
        PAPER_FINAL_SCORE_SNAPSHOTS_DIR,
        PAPER_DECISION_SNAPSHOTS_DIR,
    ]

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)