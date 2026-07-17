from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

THIS_FILE = Path(__file__).resolve()
RESEARCH_DIR = THIS_FILE.parent
COMMODITY_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[2] if len(THIS_FILE.parents) > 2 else COMMODITY_ROOT.parent

for path in [PROJECT_ROOT, COMMODITY_ROOT, RESEARCH_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ============================================================
# IMPORTS
# ============================================================

try:
    from Commodity_System import config as CFG
except ImportError:
    import config as CFG

try:
    from Commodity_System.commodity_strategy import build_production_strategy_weight_matrix
except ImportError:
    from commodity_strategy import build_production_strategy_weight_matrix

from analytics import (
    calculate_drawdown_series,
    calculate_full_summary,
    calculate_alpha_beta,
)


# ============================================================
# SMALL CONFIG HELPERS
# ============================================================

def cfg(name: str, default: Any) -> Any:
    return getattr(CFG, name, default)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


# ============================================================
# CORE SETTINGS
# ============================================================

PRICE_DATA_PATH = cfg("PRICE_DATA_PATH", COMMODITY_ROOT / "data/raw/commodity_prices.csv")
RESULTS_DIR = cfg("RESULTS_DIR", COMMODITY_ROOT / "results")

INITIAL_CAPITAL = float(cfg("INITIAL_CAPITAL", 10_000))
TRADING_DAYS_PER_YEAR = int(cfg("TRADING_DAYS_PER_YEAR", 252))
BACKTEST_REBALANCE_MODE = cfg("BACKTEST_REBALANCE_MODE", "monthly")
CASH_ANNUAL_YIELD = float(cfg("CASH_ANNUAL_YIELD", 0.04))
TOTAL_COST_BPS = float(cfg("TOTAL_COST_BPS", 0.0))
ALLOW_SHORT_WEIGHTS = bool(
    cfg("ALLOW_SHORT_WEIGHTS", cfg("BEAR_SHORT_OVERLAY_ENABLED", False))
)
MAX_TOTAL_PORTFOLIO_GROSS_EXPOSURE = float(
    cfg("MAX_TOTAL_PORTFOLIO_GROSS_EXPOSURE", 1.0)
)
VOL_TARGETING_ENABLED = bool(cfg("VOL_TARGETING_ENABLED", False))
TARGET_PORTFOLIO_VOL = float(cfg("TARGET_PORTFOLIO_VOL", 0.12))
VOL_TARGET_LOOKBACK_DAYS = int(cfg("VOL_TARGET_LOOKBACK_DAYS", 60))
VOL_TARGET_VOL_BUFFER = float(cfg("VOL_TARGET_VOL_BUFFER", 1.10))
VOL_TARGET_MIN_SCALE = float(cfg("VOL_TARGET_MIN_SCALE", 0.70))
VOL_TARGET_MAX_SCALE = float(cfg("VOL_TARGET_MAX_SCALE", 1.00))

OUTPUT_DIR = RESULTS_DIR / "backtest_V2"


# ============================================================
# DEFAULT V2 SETTINGS
# ============================================================

DEFAULT_TRANSACTION_COST_ASSUMPTIONS = {
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

DEFAULT_SCENARIOS = [
    {
        "name": "no_costs_delay_1d",
        "cost_scenario": "base",
        "execution_delay_days": 1,
        "use_detailed_transaction_costs": False,
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


def build_base_settings(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = {
        "cost_scenario": cfg("BACKTEST_V2_COST_SCENARIO", "base"),

        "use_detailed_transaction_costs": bool(
            cfg("BACKTEST_V2_USE_DETAILED_TRANSACTION_COSTS", True)
        ),
        "use_legacy_flat_costs": bool(
            cfg("BACKTEST_V2_USE_LEGACY_FLAT_COSTS", False)
        ),
        "use_commission_costs": bool(
            cfg("BACKTEST_V2_USE_COMMISSION_COSTS", True)
        ),
        "use_bid_ask_spread_costs": bool(
            cfg("BACKTEST_V2_USE_BID_ASK_SPREAD_COSTS", True)
        ),
        "use_slippage_costs": bool(
            cfg("BACKTEST_V2_USE_SLIPPAGE_COSTS", True)
        ),

        "use_turnover_controls": bool(
            cfg("BACKTEST_V2_USE_TURNOVER_CONTROLS", True)
        ),
        "use_no_trade_band": bool(
            cfg("BACKTEST_V2_USE_NO_TRADE_BAND", True)
        ),
        "no_trade_band": float(
            cfg("BACKTEST_V2_NO_TRADE_BAND", 0.005)
        ),
        "use_max_rebalance_turnover": bool(
            cfg("BACKTEST_V2_USE_MAX_REBALANCE_TURNOVER", True)
        ),
        "max_rebalance_turnover": float(
            cfg("BACKTEST_V2_MAX_REBALANCE_TURNOVER", 0.50)
        ),
        "partial_rebalance_fraction": float(
            cfg("BACKTEST_V2_PARTIAL_REBALANCE_FRACTION", 1.00)
        ),

        "use_liquidity_caps": bool(
            cfg("BACKTEST_V2_USE_LIQUIDITY_CAPS", True)
        ),
        "adv_lookback_days": int(
            cfg("BACKTEST_V2_ADV_LOOKBACK_DAYS", 20)
        ),
        "max_adv_participation": float(
            cfg("BACKTEST_V2_MAX_ADV_PARTICIPATION", 0.01)
        ),
        "liquidity_test_capital": cfg(
            "BACKTEST_V2_LIQUIDITY_TEST_CAPITAL", None
        ),

        "use_execution_delay": bool(
            cfg("BACKTEST_V2_USE_EXECUTION_DELAY", True)
        ),
        "execution_delay_days": int(
            cfg("BACKTEST_V2_EXECUTION_DELAY_DAYS", 1)
        ),

        "use_portfolio_weight_drift": bool(
            cfg("BACKTEST_V2_USE_PORTFOLIO_WEIGHT_DRIFT", True)
        ),
    }

    if overrides:
        settings.update(overrides)

    settings["no_trade_band"] = max(0.0, float(settings["no_trade_band"]))
    settings["max_rebalance_turnover"] = max(
        0.0,
        float(settings["max_rebalance_turnover"]),
    )
    settings["partial_rebalance_fraction"] = min(
        max(float(settings["partial_rebalance_fraction"]), 0.0),
        1.0,
    )
    settings["max_adv_participation"] = max(
        0.0,
        float(settings["max_adv_participation"]),
    )
    settings["execution_delay_days"] = max(
        0,
        int(settings["execution_delay_days"]),
    )

    if not settings["use_execution_delay"]:
        settings["execution_delay_days"] = 0

    return settings


# ============================================================
# DATA LOADING
# ============================================================

def load_market_data(settings: dict[str, Any]) -> dict[str, pd.DataFrame]:
    prices = pd.read_csv(PRICE_DATA_PATH)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()

    required = ["date", "ticker", "adj_close"]
    missing = [col for col in required if col not in prices.columns]

    if missing:
        raise ValueError(f"Price data missing required columns: {missing}")

    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    prices = prices.dropna(subset=["adj_close"]).copy()

    prices["daily_return"] = (
        prices.groupby("ticker")["adj_close"]
        .pct_change()
    )

    if "volume" in prices.columns:
        prices["volume"] = pd.to_numeric(prices["volume"], errors="coerce").fillna(0.0)
    else:
        prices["volume"] = np.nan

    prices["dollar_volume"] = prices["adj_close"] * prices["volume"]

    adv_lookback = int(settings["adv_lookback_days"])

    prices["rolling_adv_dollar"] = (
        prices.groupby("ticker")["dollar_volume"]
        .transform(lambda x: x.rolling(adv_lookback, min_periods=5).mean())
    )

    returns = (
        prices.pivot(index="date", columns="ticker", values="daily_return")
        .sort_index()
        .fillna(0.0)
    )

    adv = (
        prices.pivot(index="date", columns="ticker", values="rolling_adv_dollar")
        .sort_index()
    )

    dollar_volume = (
        prices.pivot(index="date", columns="ticker", values="dollar_volume")
        .sort_index()
    )

    close = (
        prices.pivot(index="date", columns="ticker", values="adj_close")
        .sort_index()
    )

    return {
        "prices": prices,
        "returns": returns,
        "adv": adv,
        "dollar_volume": dollar_volume,
        "close": close,
    }


def load_target_weights() -> pd.DataFrame:
    weights = build_production_strategy_weight_matrix()
    weights.index = pd.to_datetime(weights.index)
    return weights.sort_index().fillna(0.0)


# ============================================================
# TARGET WEIGHT PREPARATION
# ============================================================

def normalise_asset_weights(weights: pd.Series) -> pd.Series:
    out = weights.copy().astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if not ALLOW_SHORT_WEIGHTS:
        out = out.clip(lower=0.0)

        total = out.sum()

        if total > 1.0:
            out = out / total

        return out

    gross = float(out.abs().sum())

    if gross > MAX_TOTAL_PORTFOLIO_GROSS_EXPOSURE and gross > 0:
        out = out * (MAX_TOTAL_PORTFOLIO_GROSS_EXPOSURE / gross)

    return out

def calculate_cash_weight_from_positions(weights: pd.Series) -> float:
    if ALLOW_SHORT_WEIGHTS:
        gross_exposure = float(weights.abs().sum())
        return max(0.0, 1.0 - gross_exposure)

    net_exposure = float(weights.sum())
    return max(0.0, 1.0 - net_exposure)


def cap_position_gross_exposure(weights: pd.Series) -> pd.Series:
    out = weights.copy().astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if not ALLOW_SHORT_WEIGHTS:
        out = out.clip(lower=0.0)

        if out.sum() > 1.0:
            out = out / out.sum()

        return out

    gross = float(out.abs().sum())

    if gross > MAX_TOTAL_PORTFOLIO_GROSS_EXPOSURE and gross > 0:
        out = out * (MAX_TOTAL_PORTFOLIO_GROSS_EXPOSURE / gross)

    return out

def build_rebalance_signal_weights(
    raw_weights: pd.DataFrame,
    market_dates: pd.Index,
    mode: str,
) -> pd.DataFrame:
    raw_weights = raw_weights.sort_index().fillna(0.0)

    daily_weights = (
        raw_weights
        .reindex(market_dates)
        .ffill()
        .fillna(0.0)
    )

    daily_weights = daily_weights.apply(normalise_asset_weights, axis=1)

    if mode == "daily":
        return daily_weights

    if mode == "weekly":
        periods = daily_weights.index.to_period("W-FRI")
    elif mode == "monthly":
        periods = daily_weights.index.to_period("M")
    else:
        raise ValueError("BACKTEST_REBALANCE_MODE must be 'daily', 'weekly', or 'monthly'.")

    signal_dates = (
        pd.Series(daily_weights.index, index=daily_weights.index)
        .groupby(periods)
        .last()
        .tolist()
    )

    return daily_weights.loc[signal_dates].copy()


def apply_volatility_targeting(
    daily_weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    if not VOL_TARGETING_ENABLED:
        return daily_weights.fillna(0.0)

    weights = daily_weights.sort_index().fillna(0.0)
    returns = returns.sort_index().fillna(0.0)

    common_dates = weights.index.intersection(returns.index)
    common_tickers = weights.columns.intersection(returns.columns)

    if len(common_dates) == 0 or len(common_tickers) == 0:
        return weights

    aligned_weights = weights.loc[common_dates, common_tickers]
    aligned_returns = returns.loc[common_dates, common_tickers]

    pre_target_returns = (
        aligned_weights.shift(1).fillna(0.0)
        * aligned_returns
    ).sum(axis=1)

    realised_vol = (
        pre_target_returns
        .rolling(VOL_TARGET_LOOKBACK_DAYS)
        .std()
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    trigger_vol = TARGET_PORTFOLIO_VOL * VOL_TARGET_VOL_BUFFER
    raw_scale = TARGET_PORTFOLIO_VOL / realised_vol

    scale = pd.Series(1.0, index=common_dates)

    scale = scale.where(
        realised_vol <= trigger_vol,
        raw_scale,
    )

    scale = (
        scale
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=VOL_TARGET_MIN_SCALE, upper=VOL_TARGET_MAX_SCALE)
        .shift(1)
        .fillna(1.0)
    )

    out = weights.copy()
    out.loc[common_dates, common_tickers] = aligned_weights.multiply(scale, axis=0)

    return out.fillna(0.0)


def prepare_signal_weights(
    raw_weights: pd.DataFrame,
    returns: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    market_dates = returns.index

    scheduled = build_rebalance_signal_weights(
        raw_weights=raw_weights,
        market_dates=market_dates,
        mode=mode,
    )

    daily_scheduled = (
        scheduled
        .reindex(market_dates)
        .ffill()
        .fillna(0.0)
    )

    daily_scheduled = apply_volatility_targeting(
        daily_weights=daily_scheduled,
        returns=returns,
    )

    signal_weights = daily_scheduled.loc[scheduled.index].copy()
    signal_weights = signal_weights.apply(normalise_asset_weights, axis=1)

    return signal_weights


def build_execution_plan(
    signal_weights: pd.DataFrame,
    market_dates: pd.Index,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    delay = int(settings["execution_delay_days"])

    date_positions = pd.Series(
        range(len(market_dates)),
        index=market_dates,
    )

    execution_rows = []
    execution_dates = []
    signal_dates = []

    for signal_date, row in signal_weights.iterrows():
        if signal_date not in date_positions.index:
            continue

        signal_pos = int(date_positions.loc[signal_date])
        execution_pos = signal_pos + delay

        if execution_pos >= len(market_dates):
            continue

        execution_date = market_dates[execution_pos]

        execution_rows.append(row)
        execution_dates.append(execution_date)
        signal_dates.append(signal_date)

    if not execution_rows:
        empty_plan = pd.DataFrame(columns=signal_weights.columns)
        empty_signal_dates = pd.Series(dtype="datetime64[ns]")
        return empty_plan, empty_signal_dates

    execution_plan = pd.DataFrame(
        execution_rows,
        index=pd.to_datetime(execution_dates),
        columns=signal_weights.columns,
    ).sort_index()

    signal_date_by_execution = pd.Series(
        pd.to_datetime(signal_dates),
        index=pd.to_datetime(execution_dates),
        name="signal_date",
    ).sort_index()

    if execution_plan.index.has_duplicates:
        execution_plan = execution_plan.groupby(level=0).last()
        signal_date_by_execution = signal_date_by_execution.groupby(level=0).last()

    execution_plan = execution_plan.apply(normalise_asset_weights, axis=1)

    return execution_plan, signal_date_by_execution


# ============================================================
# COST ASSUMPTIONS
# ============================================================

def get_transaction_cost_assumptions(
    tickers: list[str],
    settings: dict[str, Any],
) -> pd.DataFrame:
    assumptions = cfg(
        "BACKTEST_V2_TRANSACTION_COST_ASSUMPTIONS",
        DEFAULT_TRANSACTION_COST_ASSUMPTIONS,
    )

    scenario = settings["cost_scenario"]

    if scenario not in assumptions:
        print(f"Warning: unknown cost scenario '{scenario}'. Falling back to 'base'.")
        scenario = "base"

    scenario_assumptions = assumptions.get(scenario, {})

    rows = []

    for ticker in tickers:
        values = scenario_assumptions.get(
            ticker,
            {"commission_bps": 0.0, "full_spread_bps": 5.0, "slippage_bps": 2.0},
        )

        rows.append(
            {
                "ticker": ticker,
                "commission_bps": safe_float(values.get("commission_bps", 0.0)),
                "full_spread_bps": safe_float(values.get("full_spread_bps", 5.0)),
                "half_spread_bps": safe_float(values.get("full_spread_bps", 5.0)) / 2.0,
                "slippage_bps": safe_float(values.get("slippage_bps", 2.0)),
            }
        )

    return pd.DataFrame(rows).set_index("ticker")


def calculate_trade_costs(
    ticker: str,
    executed_trade_weight: float,
    equity_before_costs: float,
    cost_table: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, float]:
    trade_notional = abs(executed_trade_weight) * equity_before_costs

    if trade_notional <= 0 or equity_before_costs <= 0:
        return {
            "trade_notional": 0.0,
            "commission_cost": 0.0,
            "spread_cost": 0.0,
            "slippage_cost": 0.0,
            "legacy_flat_cost": 0.0,
            "total_trade_cost": 0.0,
        }

    if not settings["use_detailed_transaction_costs"]:
        if settings.get("use_legacy_flat_costs", False):
            legacy_flat_cost = trade_notional * (TOTAL_COST_BPS / 10_000)
        else:
            legacy_flat_cost = 0.0

        return {
            "trade_notional": trade_notional,
            "commission_cost": 0.0,
            "spread_cost": 0.0,
            "slippage_cost": 0.0,
            "legacy_flat_cost": legacy_flat_cost,
            "total_trade_cost": legacy_flat_cost,
        }

    if ticker not in cost_table.index:
        commission_bps = 0.0
        half_spread_bps = 2.5
        slippage_bps = 2.0
    else:
        row = cost_table.loc[ticker]
        commission_bps = safe_float(row["commission_bps"])
        half_spread_bps = safe_float(row["half_spread_bps"])
        slippage_bps = safe_float(row["slippage_bps"])

    if not settings["use_commission_costs"]:
        commission_bps = 0.0

    if not settings["use_bid_ask_spread_costs"]:
        half_spread_bps = 0.0

    if not settings["use_slippage_costs"]:
        slippage_bps = 0.0

    commission_cost = trade_notional * (commission_bps / 10_000)
    spread_cost = trade_notional * (half_spread_bps / 10_000)
    slippage_cost = trade_notional * (slippage_bps / 10_000)

    total_trade_cost = commission_cost + spread_cost + slippage_cost

    return {
        "trade_notional": trade_notional,
        "commission_cost": commission_cost,
        "spread_cost": spread_cost,
        "slippage_cost": slippage_cost,
        "legacy_flat_cost": 0.0,
        "total_trade_cost": total_trade_cost,
    }


# ============================================================
# TURNOVER AND LIQUIDITY CONTROLS
# ============================================================

def apply_turnover_controls(
    desired_trades: pd.Series,
    settings: dict[str, Any],
) -> tuple[pd.Series, dict[str, Any]]:
    desired = desired_trades.copy().astype(float).fillna(0.0)

    raw_desired_turnover = float(desired.abs().sum())
    trades = desired.copy()

    no_trade_band_count = 0
    turnover_capped = False
    turnover_scale = 1.0

    if settings["use_turnover_controls"] and settings["use_no_trade_band"]:
        band = float(settings["no_trade_band"])
        tiny = trades.abs() < band
        no_trade_band_count = int(tiny.sum())
        trades.loc[tiny] = 0.0

    turnover_after_band = float(trades.abs().sum())

    if (
        settings["use_turnover_controls"]
        and settings["use_max_rebalance_turnover"]
        and turnover_after_band > 0
    ):
        max_turnover = float(settings["max_rebalance_turnover"])

        if max_turnover > 0 and turnover_after_band > max_turnover:
            turnover_scale = max_turnover / turnover_after_band
            trades = trades * turnover_scale
            turnover_capped = True

    partial_fraction = 1.0

    if settings["use_turnover_controls"]:
        partial_fraction = float(settings["partial_rebalance_fraction"])
        trades = trades * partial_fraction

    executed_turnover_before_liquidity = float(trades.abs().sum())

    diagnostics = {
        "raw_desired_turnover": raw_desired_turnover,
        "turnover_after_no_trade_band": turnover_after_band,
        "executed_turnover_before_liquidity": executed_turnover_before_liquidity,
        "no_trade_band_count": no_trade_band_count,
        "turnover_capped": turnover_capped,
        "turnover_scale": turnover_scale,
        "partial_rebalance_fraction": partial_fraction,
    }

    return trades, diagnostics


def apply_liquidity_caps(
    trades: pd.Series,
    date: pd.Timestamp,
    equity: float,
    initial_capital: float,
    adv: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.Series, dict[str, dict[str, Any]]]:
    trades = trades.copy().astype(float).fillna(0.0)

    if not settings["use_liquidity_caps"]:
        diagnostics = {}

        for ticker, trade_weight in trades.items():
            diagnostics[ticker] = {
                "adv_dollar": np.nan,
                "liquidity_test_trade_notional": abs(trade_weight) * equity,
                "max_trade_notional": np.nan,
                "fill_ratio": 1.0,
                "unfilled_trade_weight": 0.0,
                "liquidity_capped": False,
                "missing_adv": True,
                "participation_rate": np.nan,
            }

        return trades, diagnostics

    if settings["liquidity_test_capital"] is None:
        liquidity_equity = equity
    else:
        base_capital = float(settings["liquidity_test_capital"])
        liquidity_equity = base_capital * (equity / initial_capital)

    max_adv_participation = float(settings["max_adv_participation"])

    if date in adv.index:
        adv_row = adv.loc[date]
    else:
        adv_row = pd.Series(np.nan, index=trades.index)

    executed = trades.copy()
    diagnostics = {}

    for ticker, trade_weight in trades.items():
        abs_trade_weight = abs(float(trade_weight))

        liquidity_test_trade_notional = abs_trade_weight * liquidity_equity
        adv_dollar = safe_float(adv_row.get(ticker, np.nan), default=np.nan)

        missing_adv = pd.isna(adv_dollar)
        liquidity_capped = False
        fill_ratio = 1.0
        max_trade_notional = np.nan
        unfilled_trade_weight = 0.0
        participation_rate = np.nan

        if abs_trade_weight == 0:
            diagnostics[ticker] = {
                "adv_dollar": adv_dollar,
                "liquidity_test_trade_notional": 0.0,
                "max_trade_notional": np.nan,
                "fill_ratio": 1.0,
                "unfilled_trade_weight": 0.0,
                "liquidity_capped": False,
                "missing_adv": missing_adv,
                "participation_rate": np.nan,
            }
            continue

        if not missing_adv and adv_dollar > 0 and max_adv_participation > 0:
            max_trade_notional = adv_dollar * max_adv_participation
            participation_rate = liquidity_test_trade_notional / adv_dollar

            if liquidity_test_trade_notional > max_trade_notional:
                fill_ratio = max_trade_notional / liquidity_test_trade_notional
                fill_ratio = min(max(fill_ratio, 0.0), 1.0)
                executed.loc[ticker] = trade_weight * fill_ratio
                liquidity_capped = True

                unfilled_trade_weight = trade_weight - executed.loc[ticker]

        elif not missing_adv and adv_dollar <= 0:
            executed.loc[ticker] = 0.0
            fill_ratio = 0.0
            liquidity_capped = True
            unfilled_trade_weight = trade_weight
            max_trade_notional = 0.0
            participation_rate = np.nan

        diagnostics[ticker] = {
            "adv_dollar": adv_dollar,
            "liquidity_test_trade_notional": liquidity_test_trade_notional,
            "max_trade_notional": max_trade_notional,
            "fill_ratio": fill_ratio,
            "unfilled_trade_weight": unfilled_trade_weight,
            "liquidity_capped": liquidity_capped,
            "missing_adv": missing_adv,
            "participation_rate": participation_rate,
        }

    return executed, diagnostics


# ============================================================
# BENCHMARK WEIGHTS
# ============================================================

def make_equal_weight(index: pd.Index, tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        1.0 / len(tickers),
        index=index,
        columns=tickers,
    )


def make_gold_only(index: pd.Index, tickers: list[str]) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=index, columns=tickers)

    if "GLD" in weights.columns:
        weights["GLD"] = 1.0

    return weights


def make_cash(index: pd.Index, tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=index, columns=tickers)


# ============================================================
# REALISTIC EXECUTION ENGINE
# ============================================================

def simulate_strategy_v2(
    name: str,
    raw_target_weights: pd.DataFrame,
    market_data: dict[str, pd.DataFrame],
    settings: dict[str, Any],
    initial_capital: float = INITIAL_CAPITAL,
) -> dict[str, pd.DataFrame]:
    returns = market_data["returns"].sort_index().fillna(0.0)
    adv = market_data["adv"].sort_index()

    tickers = list(returns.columns)
    market_dates = returns.index

    raw_target_weights = (
        raw_target_weights
        .sort_index()
        .reindex(columns=tickers)
        .fillna(0.0)
    )

    signal_weights = prepare_signal_weights(
        raw_weights=raw_target_weights,
        returns=returns,
        mode=BACKTEST_REBALANCE_MODE,
    )

    execution_plan, signal_date_by_execution = build_execution_plan(
        signal_weights=signal_weights,
        market_dates=market_dates,
        settings=settings,
    )

    cost_table = get_transaction_cost_assumptions(
        tickers=tickers,
        settings=settings,
    )

    cash_daily_return = (
        (1.0 + CASH_ANNUAL_YIELD) ** (1.0 / TRADING_DAYS_PER_YEAR)
        - 1.0
    )

    weights = pd.Series(0.0, index=tickers, dtype=float)
    equity = float(initial_capital)

    curve_rows = []
    trade_rows = []
    executed_weight_rows = []

    for date in market_dates:
        start_equity = equity
        starting_weights = weights.copy()

        asset_returns = returns.loc[date].reindex(tickers).fillna(0.0)

        starting_net_exposure = float(weights.sum())
        starting_gross_exposure = float(weights.abs().sum())
        starting_short_exposure = float(weights[weights < 0].abs().sum())

        starting_exposure = (
            starting_gross_exposure if ALLOW_SHORT_WEIGHTS else starting_net_exposure
        )
        starting_cash_weight = calculate_cash_weight_from_positions(weights)

        gross_return = float((weights * asset_returns).sum())
        cash_return = float(starting_cash_weight * cash_daily_return)
        pre_cost_return = gross_return + cash_return

        equity_before_costs = equity * (1.0 + pre_cost_return)

        if settings["use_portfolio_weight_drift"]:
            denominator = 1.0 + pre_cost_return

            if denominator > 0:
                drifted_weights = weights * (1.0 + asset_returns) / denominator
                weights = drifted_weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                weights = cap_position_gross_exposure(weights)

        equity = equity_before_costs

        commission_cost = 0.0
        spread_cost = 0.0
        slippage_cost = 0.0
        legacy_flat_cost = 0.0
        total_trade_cost = 0.0

        raw_desired_turnover = 0.0
        turnover_after_no_trade_band = 0.0
        executed_turnover_before_liquidity = 0.0
        executed_turnover = 0.0
        no_trade_band_count = 0
        turnover_capped = False
        turnover_scale = 1.0
        partial_rebalance_fraction = 1.0
        execution_event = False
        signal_date = pd.NaT
        tracking_error_to_target = np.nan
        liquidity_capped_trade_count = 0

        if date in execution_plan.index:
            execution_event = True
            signal_date = signal_date_by_execution.loc[date]

            target_weights = execution_plan.loc[date].reindex(tickers).fillna(0.0)
            target_weights = normalise_asset_weights(target_weights)

            previous_weights_before_trade = weights.copy()

            desired_trades = target_weights - weights

            controlled_trades, turnover_diagnostics = apply_turnover_controls(
                desired_trades=desired_trades,
                settings=settings,
            )

            raw_desired_turnover = turnover_diagnostics["raw_desired_turnover"]
            turnover_after_no_trade_band = turnover_diagnostics["turnover_after_no_trade_band"]
            executed_turnover_before_liquidity = turnover_diagnostics[
                "executed_turnover_before_liquidity"
            ]
            no_trade_band_count = turnover_diagnostics["no_trade_band_count"]
            turnover_capped = turnover_diagnostics["turnover_capped"]
            turnover_scale = turnover_diagnostics["turnover_scale"]
            partial_rebalance_fraction = turnover_diagnostics[
                "partial_rebalance_fraction"
            ]

            executed_trades, liquidity_diagnostics = apply_liquidity_caps(
                trades=controlled_trades,
                date=date,
                equity=equity,
                initial_capital=initial_capital,
                adv=adv,
                settings=settings,
            )

            executed_turnover = float(executed_trades.abs().sum())

            for ticker in tickers:
                executed_trade_weight = float(executed_trades.get(ticker, 0.0))

                if abs(executed_trade_weight) < 1e-12:
                    continue

                cost_info = calculate_trade_costs(
                    ticker=ticker,
                    executed_trade_weight=executed_trade_weight,
                    equity_before_costs=equity,
                    cost_table=cost_table,
                    settings=settings,
                )

                liq = liquidity_diagnostics.get(ticker, {})

                commission_cost += cost_info["commission_cost"]
                spread_cost += cost_info["spread_cost"]
                slippage_cost += cost_info["slippage_cost"]
                legacy_flat_cost += cost_info["legacy_flat_cost"]
                total_trade_cost += cost_info["total_trade_cost"]

                if bool(liq.get("liquidity_capped", False)):
                    liquidity_capped_trade_count += 1

                trade_rows.append(
                    {
                        "strategy": name,
                        "date": date,
                        "signal_date": signal_date,
                        "execution_date": date,
                        "execution_delay_days": settings["execution_delay_days"],
                        "cost_scenario": settings["cost_scenario"],
                        "ticker": ticker,

                        "previous_weight": float(previous_weights_before_trade.get(ticker, 0.0)),
                        "target_weight": float(target_weights.get(ticker, 0.0)),
                        "desired_trade_weight": float(desired_trades.get(ticker, 0.0)),
                        "controlled_trade_weight": float(controlled_trades.get(ticker, 0.0)),
                        "executed_trade_weight": executed_trade_weight,
                        "unfilled_trade_weight": float(liq.get("unfilled_trade_weight", 0.0)),

                        "trade_notional": cost_info["trade_notional"],
                        "liquidity_test_trade_notional": liq.get(
                            "liquidity_test_trade_notional",
                            np.nan,
                        ),
                        "adv_dollar": liq.get("adv_dollar", np.nan),
                        "max_trade_notional": liq.get("max_trade_notional", np.nan),
                        "participation_rate": liq.get("participation_rate", np.nan),
                        "fill_ratio": liq.get("fill_ratio", np.nan),
                        "liquidity_capped": bool(liq.get("liquidity_capped", False)),
                        "missing_adv": bool(liq.get("missing_adv", False)),

                        "commission_cost": cost_info["commission_cost"],
                        "spread_cost": cost_info["spread_cost"],
                        "slippage_cost": cost_info["slippage_cost"],
                        "legacy_flat_cost": cost_info["legacy_flat_cost"],
                        "total_trade_cost": cost_info["total_trade_cost"],
                    }
                )

            if total_trade_cost > 0:
                equity = max(0.0, equity - total_trade_cost)

            weights = weights + executed_trades
            weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            weights = cap_position_gross_exposure(weights)

            tracking_error_to_target = float((target_weights - weights).abs().sum())

        if start_equity > 0:
            net_return = (equity / start_equity) - 1.0
            commission_cost_drag = commission_cost / start_equity
            spread_cost_drag = spread_cost / start_equity
            slippage_cost_drag = slippage_cost / start_equity
            legacy_flat_cost_drag = legacy_flat_cost / start_equity
            total_transaction_cost_drag = total_trade_cost / start_equity
        else:
            net_return = 0.0
            commission_cost_drag = 0.0
            spread_cost_drag = 0.0
            slippage_cost_drag = 0.0
            legacy_flat_cost_drag = 0.0
            total_transaction_cost_drag = 0.0

        ending_net_exposure = float(weights.sum())
        ending_gross_exposure = float(weights.abs().sum())
        ending_short_exposure = float(weights[weights < 0].abs().sum())

        ending_exposure = (
            ending_gross_exposure if ALLOW_SHORT_WEIGHTS else ending_net_exposure
        )
        ending_cash_weight = calculate_cash_weight_from_positions(weights)

        curve_rows.append(
            {
                "date": date,
                "strategy": name,
                "signal_date": signal_date,
                "execution_date": date if execution_event else pd.NaT,
                "execution_event": execution_event,
                "execution_delay_days": settings["execution_delay_days"],
                "cost_scenario": settings["cost_scenario"],

                "gross_return": gross_return,
                "cash_return": cash_return,
                "pre_cost_return": pre_cost_return,

                "commission_cost_drag": commission_cost_drag,
                "spread_cost_drag": spread_cost_drag,
                "slippage_cost_drag": slippage_cost_drag,
                "legacy_flat_cost_drag": legacy_flat_cost_drag,
                "total_transaction_cost_drag": total_transaction_cost_drag,

                "net_return": net_return,
                "equity": equity,

                "raw_desired_turnover": raw_desired_turnover,
                "turnover_after_no_trade_band": turnover_after_no_trade_band,
                "executed_turnover_before_liquidity": executed_turnover_before_liquidity,
                "turnover": executed_turnover,

                "no_trade_band_count": no_trade_band_count,
                "turnover_capped": turnover_capped,
                "turnover_scale": turnover_scale,
                "partial_rebalance_fraction": partial_rebalance_fraction,

                "liquidity_capped_trade_count": liquidity_capped_trade_count,
                "tracking_error_to_target": tracking_error_to_target,

                "starting_exposure": starting_exposure,
                "starting_net_exposure": starting_net_exposure,
                "starting_gross_exposure": starting_gross_exposure,
                "starting_short_exposure": starting_short_exposure,

                "exposure": ending_exposure,
                "net_exposure": ending_net_exposure,
                "gross_exposure": ending_gross_exposure,
                "short_exposure": ending_short_exposure,
                "cash_weight": ending_cash_weight,
            }
        )

        executed_weight_row = {"date": date, "strategy": name}
        executed_weight_row.update(weights.to_dict())
        executed_weight_rows.append(executed_weight_row)

    curve = pd.DataFrame(curve_rows)
    curve["date"] = pd.to_datetime(curve["date"])
    curve = curve.set_index("date").sort_index()

    curve["drawdown"] = calculate_drawdown_series(curve["net_return"])

    trade_log = pd.DataFrame(trade_rows)

    executed_weights = pd.DataFrame(executed_weight_rows)
    executed_weights["date"] = pd.to_datetime(executed_weights["date"])
    executed_weights = executed_weights.set_index("date").sort_index()

    return {
        "curve": curve,
        "trade_log": trade_log,
        "executed_weights": executed_weights,
        "signal_weights": signal_weights,
        "execution_plan": execution_plan,
    }


# ============================================================
# SUMMARY TABLES
# ============================================================

def build_performance_summary(
    results: dict[str, dict[str, pd.DataFrame]],
    benchmark_name: str | None = "equal_weight",
) -> pd.DataFrame:
    rows = []

    benchmark_returns = None

    if benchmark_name is not None and benchmark_name in results:
        benchmark_returns = results[benchmark_name]["curve"]["net_return"]

    for name, result in results.items():
        curve = result["curve"]

        summary = calculate_full_summary(
            returns=curve["net_return"],
            equity=curve["equity"],
            turnover=curve["turnover"],
            transaction_cost=curve["total_transaction_cost_drag"],
            exposure=curve["exposure"],
            benchmark_returns=benchmark_returns,
            strategy_name=name,
            benchmark_name=benchmark_name,
            initial_capital=INITIAL_CAPITAL,
            risk_free_rate_annual=0.0,
            periods_per_year=TRADING_DAYS_PER_YEAR,
        )

        rows.append(summary)

    out = pd.DataFrame(rows)

    if not out.empty:
        sort_cols = [col for col in ["sharpe", "calmar", "cagr"] if col in out.columns]
        out = out.sort_values(sort_cols, ascending=False)

    return out


def build_alpha_beta_summary(
    results: dict[str, dict[str, pd.DataFrame]],
    benchmark_names: list[str],
) -> pd.DataFrame:
    rows = []

    for strategy_name, strategy_result in results.items():
        strategy_returns = strategy_result["curve"]["net_return"]

        for benchmark_name in benchmark_names:
            if strategy_name == benchmark_name:
                continue

            if benchmark_name not in results:
                continue

            benchmark_returns = results[benchmark_name]["curve"]["net_return"]

            stats = calculate_alpha_beta(
                strategy_returns=strategy_returns,
                benchmark_returns=benchmark_returns,
                risk_free_rate_annual=0.0,
                periods_per_year=TRADING_DAYS_PER_YEAR,
            )

            rows.append(
                {
                    "strategy": strategy_name,
                    "benchmark": benchmark_name,
                    **stats,
                }
            )

    return pd.DataFrame(rows)


def build_cost_summary(
    strategy_name: str,
    curve: pd.DataFrame,
    trade_log: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, Any]:
    execution_rows = curve[curve["execution_event"]].copy()

    total_gross_plus_cash_return = (
        (1.0 + curve["gross_return"] + curve["cash_return"]).prod() - 1.0
        if not curve.empty
        else np.nan
    )

    total_cost_drag = curve["total_transaction_cost_drag"].sum()

    if pd.notna(total_gross_plus_cash_return) and abs(total_gross_plus_cash_return) > 1e-12:
        cost_drag_as_pct_of_gross_return = total_cost_drag / abs(total_gross_plus_cash_return)
    else:
        cost_drag_as_pct_of_gross_return = np.nan

    if trade_log.empty:
        liquidity_capped_trades = 0
        average_fill_ratio = np.nan
        missing_adv_trades = 0
        number_of_trades = 0
    else:
        liquidity_capped_trades = int(trade_log["liquidity_capped"].sum())
        average_fill_ratio = trade_log["fill_ratio"].replace([np.inf, -np.inf], np.nan).mean()
        missing_adv_trades = int(trade_log["missing_adv"].sum())
        number_of_trades = len(trade_log)

    return {
        "strategy": strategy_name,
        "cost_scenario": settings["cost_scenario"],
        "execution_delay_days": settings["execution_delay_days"],

        "number_of_rebalances": int(curve["execution_event"].sum()),
        "number_of_trades": number_of_trades,

        "total_turnover": curve["turnover"].sum(),
        "annualised_turnover": curve["turnover"].mean() * TRADING_DAYS_PER_YEAR,
        "average_daily_turnover": curve["turnover"].mean(),
        "average_rebalance_turnover": execution_rows["turnover"].mean()
        if not execution_rows.empty
        else np.nan,
        "max_rebalance_turnover": execution_rows["turnover"].max()
        if not execution_rows.empty
        else np.nan,

        "total_commission_drag": curve["commission_cost_drag"].sum(),
        "total_spread_drag": curve["spread_cost_drag"].sum(),
        "total_slippage_drag": curve["slippage_cost_drag"].sum(),
        "total_legacy_flat_cost_drag": curve["legacy_flat_cost_drag"].sum(),
        "total_cost_drag": total_cost_drag,
        "annualised_cost_drag": curve["total_transaction_cost_drag"].mean() * TRADING_DAYS_PER_YEAR,
        "cost_drag_as_pct_of_gross_return": cost_drag_as_pct_of_gross_return,

        "liquidity_capped_trades": liquidity_capped_trades,
        "missing_adv_trades": missing_adv_trades,
        "average_fill_ratio": average_fill_ratio,

        "average_exposure": curve["exposure"].mean(),
        "average_cash": curve["cash_weight"].mean(),
        "average_tracking_error_to_target": execution_rows["tracking_error_to_target"].mean()
        if not execution_rows.empty
        else np.nan,
    }


def build_cost_summary_table(
    results: dict[str, dict[str, pd.DataFrame]],
    settings: dict[str, Any],
) -> pd.DataFrame:
    rows = []

    for name, result in results.items():
        rows.append(
            build_cost_summary(
                strategy_name=name,
                curve=result["curve"],
                trade_log=result["trade_log"],
                settings=settings,
            )
        )

    return pd.DataFrame(rows)


# ============================================================
# SCENARIO TESTING
# ============================================================

def run_model_scenarios(
    model_weights: pd.DataFrame,
    market_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    scenarios = cfg("BACKTEST_V2_SCENARIOS", DEFAULT_SCENARIOS)
    rows = []

    for scenario in scenarios:
        scenario = dict(scenario)
        scenario_name = scenario.pop("name", "unnamed_scenario")

        settings = build_base_settings(overrides=scenario)

        result = simulate_strategy_v2(
            name=scenario_name,
            raw_target_weights=model_weights,
            market_data=market_data,
            settings=settings,
            initial_capital=INITIAL_CAPITAL,
        )

        curve = result["curve"]
        trade_log = result["trade_log"]

        perf = calculate_full_summary(
            returns=curve["net_return"],
            equity=curve["equity"],
            turnover=curve["turnover"],
            transaction_cost=curve["total_transaction_cost_drag"],
            exposure=curve["exposure"],
            benchmark_returns=None,
            strategy_name=scenario_name,
            benchmark_name=None,
            initial_capital=INITIAL_CAPITAL,
            risk_free_rate_annual=0.0,
            periods_per_year=TRADING_DAYS_PER_YEAR,
        )

        costs = build_cost_summary(
            strategy_name=scenario_name,
            curve=curve,
            trade_log=trade_log,
            settings=settings,
        )

        row = {
            "scenario": scenario_name,
            "cost_scenario": settings["cost_scenario"],
            "execution_delay_days": settings["execution_delay_days"],

            "final_equity": perf.get("final_equity"),
            "cagr": perf.get("cagr"),
            "annualised_volatility": perf.get("annualised_volatility"),
            "sharpe": perf.get("sharpe"),
            "sortino": perf.get("sortino"),
            "calmar": perf.get("calmar"),
            "max_drawdown": perf.get("max_drawdown"),

            "total_turnover": costs.get("total_turnover"),
            "annualised_turnover": costs.get("annualised_turnover"),
            "total_cost_drag": costs.get("total_cost_drag"),
            "annualised_cost_drag": costs.get("annualised_cost_drag"),
            "total_commission_drag": costs.get("total_commission_drag"),
            "total_spread_drag": costs.get("total_spread_drag"),
            "total_slippage_drag": costs.get("total_slippage_drag"),
            "liquidity_capped_trades": costs.get("liquidity_capped_trades"),
            "average_fill_ratio": costs.get("average_fill_ratio"),
            "average_exposure": costs.get("average_exposure"),
            "average_cash": costs.get("average_cash"),
            "average_tracking_error_to_target": costs.get(
                "average_tracking_error_to_target"
            ),
        }

        rows.append(row)

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            ["sharpe", "calmar", "cagr"],
            ascending=False,
        )

    return out


# ============================================================
# OUTPUT SAVING
# ============================================================

def save_outputs(
    results: dict[str, dict[str, pd.DataFrame]],
    performance_summary: pd.DataFrame,
    alpha_beta_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    scenario_summary: pd.DataFrame | None,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_curves = []
    all_trade_logs = []
    all_executed_weights = []

    for name, result in results.items():
        curve = result["curve"].reset_index()
        all_curves.append(curve)

        trade_log = result["trade_log"]
        if not trade_log.empty:
            all_trade_logs.append(trade_log)

        executed_weights = result["executed_weights"].reset_index()
        all_executed_weights.append(executed_weights)

        if name == "model":
            result["curve"].reset_index().to_csv(
                OUTPUT_DIR / "model_curve_V2.csv",
                index=False,
            )
            result["trade_log"].to_csv(
                OUTPUT_DIR / "model_trade_log_V2.csv",
                index=False,
            )
            result["executed_weights"].reset_index().to_csv(
                OUTPUT_DIR / "executed_model_weights_V2.csv",
                index=False,
            )
            result["signal_weights"].reset_index().to_csv(
                OUTPUT_DIR / "target_signal_weights_V2.csv",
                index=False,
            )
            result["execution_plan"].reset_index().rename(
                columns={"index": "execution_date"}
            ).to_csv(
                OUTPUT_DIR / "execution_plan_V2.csv",
                index=False,
            )

    if all_curves:
        pd.concat(all_curves, ignore_index=True).to_csv(
            OUTPUT_DIR / "all_curves_V2.csv",
            index=False,
        )

    if all_trade_logs:
        pd.concat(all_trade_logs, ignore_index=True).to_csv(
            OUTPUT_DIR / "all_trade_logs_V2.csv",
            index=False,
        )
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / "all_trade_logs_V2.csv", index=False)

    if all_executed_weights:
        pd.concat(all_executed_weights, ignore_index=True).to_csv(
            OUTPUT_DIR / "all_executed_weights_V2.csv",
            index=False,
        )

    performance_summary.to_csv(
        OUTPUT_DIR / "performance_summary_V2.csv",
        index=False,
    )

    alpha_beta_summary.to_csv(
        OUTPUT_DIR / "alpha_beta_summary_V2.csv",
        index=False,
    )

    cost_summary.to_csv(
        OUTPUT_DIR / "cost_summary_V2.csv",
        index=False,
    )

    if scenario_summary is not None:
        scenario_summary.to_csv(
            OUTPUT_DIR / "scenario_summary_V2.csv",
            index=False,
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n========== BACKTEST V2: REALISTIC EXECUTION MODEL ==========")

    settings = build_base_settings()

    print("\nMain settings:")
    for key, value in settings.items():
        print(f"  {key}: {value}")

    market_data = load_market_data(settings=settings)
    returns = market_data["returns"]

    model_weights = load_target_weights()
    tickers = list(model_weights.columns)

    model_weights = model_weights.reindex(columns=tickers).fillna(0.0)

    equal_weight = make_equal_weight(
        index=model_weights.index,
        tickers=tickers,
    )

    gold_only = make_gold_only(
        index=model_weights.index,
        tickers=tickers,
    )

    cash = make_cash(
        index=model_weights.index,
        tickers=tickers,
    )

    strategies = {
        "model": model_weights,
        "equal_weight": equal_weight,
        "gold_only": gold_only,
        "cash": cash,
    }

    results = {}

    for name, weights in strategies.items():
        print(f"\nRunning V2 strategy: {name}")

        result = simulate_strategy_v2(
            name=name,
            raw_target_weights=weights,
            market_data=market_data,
            settings=settings,
            initial_capital=INITIAL_CAPITAL,
        )

        results[name] = result

    performance_summary = build_performance_summary(
        results=results,
        benchmark_name="equal_weight",
    )

    alpha_beta_summary = build_alpha_beta_summary(
        results=results,
        benchmark_names=["equal_weight", "gold_only"],
    )

    cost_summary = build_cost_summary_table(
        results=results,
        settings=settings,
    )

    scenario_summary = None

    if bool(cfg("BACKTEST_V2_RUN_SCENARIO_TESTS", True)):
        print("\nRunning model scenario tests...")
        scenario_summary = run_model_scenarios(
            model_weights=model_weights,
            market_data=market_data,
        )

    save_outputs(
        results=results,
        performance_summary=performance_summary,
        alpha_beta_summary=alpha_beta_summary,
        cost_summary=cost_summary,
        scenario_summary=scenario_summary,
    )

    cols = [
        "strategy",
        "benchmark",
        "final_equity",
        "cagr",
        "annualised_volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "average_daily_turnover",
        "annualised_turnover",
        "total_transaction_cost_drag",
        "annualised_transaction_cost_drag",
        "average_exposure",
        "average_cash",
        "alpha_annualised",
        "beta",
        "information_ratio",
    ]

    cols = [col for col in cols if col in performance_summary.columns]

    print("\nBacktest V2 complete.")
    print(f"Saved outputs to: {OUTPUT_DIR}")

    print("\nPerformance summary:")
    print(performance_summary[cols].to_string(index=False))

    print("\nCost summary:")
    print(cost_summary.to_string(index=False))

    if scenario_summary is not None and not scenario_summary.empty:
        scenario_cols = [
            "scenario",
            "cost_scenario",
            "execution_delay_days",
            "final_equity",
            "cagr",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "total_cost_drag",
            "annualised_turnover",
            "liquidity_capped_trades",
            "average_fill_ratio",
            "average_tracking_error_to_target",
        ]

        scenario_cols = [col for col in scenario_cols if col in scenario_summary.columns]

        print("\nScenario summary:")
        print(scenario_summary[scenario_cols].to_string(index=False))


if __name__ == "__main__":
    main()