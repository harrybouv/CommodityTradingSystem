from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
    WEIGHTS_PATH,
    PRICE_DATA_PATH,
    UNIVERSE,
)

try:
    import paper_trading.paper_config as pc
except Exception:
    pc = None


SLEEVE_NAME = "commodities"
MODEL_VERSION = "commodity_system_v1"

EXPORT_DIR = BASE_DIR / "exports" / "portfolio_manager"

TARGET_WEIGHTS_PATH = WEIGHTS_PATH
FINAL_SCORES_PATH = PROCESSED_DATA_DIR / "final_scores.csv"
PRICE_PATH = PRICE_DATA_PATH

if pc is not None:
    PAPER_POSITIONS_PATH = pc.PAPER_POSITIONS_PATH
    PAPER_TRADES_PATH = pc.PAPER_TRADES_PATH
    PAPER_EQUITY_CURVE_PATH = pc.PAPER_EQUITY_CURVE_PATH
    PAPER_STATE_PATH = pc.PAPER_STATE_PATH
    LATEST_ORDERS_PATH = pc.LATEST_ORDERS_PATH
    LATEST_PORTFOLIO_PATH = pc.LATEST_PORTFOLIO_PATH
    LATEST_SUMMARY_PATH = pc.LATEST_SUMMARY_PATH
else:
    PAPER_DIR = BASE_DIR / "paper_trading"
    PAPER_POSITIONS_PATH = PAPER_DIR / "state" / "paper_positions.csv"
    PAPER_TRADES_PATH = PAPER_DIR / "state" / "paper_trades.csv"
    PAPER_EQUITY_CURVE_PATH = PAPER_DIR / "state" / "paper_equity_curve.csv"
    PAPER_STATE_PATH = PAPER_DIR / "state" / "paper_state.csv"
    LATEST_ORDERS_PATH = PAPER_DIR / "reports" / "latest_orders.csv"
    LATEST_PORTFOLIO_PATH = PAPER_DIR / "reports" / "latest_portfolio.csv"
    LATEST_SUMMARY_PATH = PAPER_DIR / "reports" / "latest_summary.txt"


OUT_SIGNAL = EXPORT_DIR / "commodity_signal.csv"
OUT_POSITIONS = EXPORT_DIR / "commodity_positions.csv"
OUT_EQUITY_CURVE = EXPORT_DIR / "commodity_equity_curve.csv"
OUT_EQUAL_WEIGHT_BENCHMARK = EXPORT_DIR / "commodity_equal_weight_benchmark.csv"
OUT_METRICS = EXPORT_DIR / "commodity_metrics.csv"
OUT_ORDERS = EXPORT_DIR / "commodity_orders.csv"
OUT_MANIFEST = EXPORT_DIR / "commodity_manifest.json"
OUT_MEMO_SEED = EXPORT_DIR / "commodity_memo_seed.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        out = float(value)
        if np.isnan(out) or np.isinf(out):
            return default
        return out
    except Exception:
        return default


def clean_ticker(value: Any) -> str:
    return str(value).upper().strip()


def normalise_date_column(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce")
        out = out.dropna(subset=[col]).copy()
    return out


def latest_rows_by_date(df: pd.DataFrame, date_col: str = "date") -> tuple[pd.DataFrame, pd.Timestamp | None]:
    if df.empty or date_col not in df.columns:
        return pd.DataFrame(), None

    out = normalise_date_column(df, date_col)
    if out.empty:
        return pd.DataFrame(), None

    latest_date = pd.Timestamp(out[date_col].max()).normalize()
    latest = out[out[date_col].dt.normalize() == latest_date].copy()

    return latest.reset_index(drop=True), latest_date


def asset_name(ticker: str) -> str:
    if ticker == "CASH":
        return "Cash"
    return UNIVERSE.get(ticker, {}).get("name", ticker)


def asset_group(ticker: str) -> str:
    if ticker == "CASH":
        return "cash"
    return UNIVERSE.get(ticker, {}).get("group", "unknown")


def add_asset_metadata(df: pd.DataFrame, ticker_col: str = "ticker") -> pd.DataFrame:
    out = df.copy()
    if ticker_col not in out.columns:
        return out

    out[ticker_col] = out[ticker_col].map(clean_ticker)
    out["asset_name"] = out[ticker_col].map(asset_name)
    out["asset_group"] = out[ticker_col].map(asset_group)
    return out


def load_latest_state() -> dict[str, str]:
    state = read_csv_if_exists(PAPER_STATE_PATH)

    if state.empty or not {"key", "value"}.issubset(state.columns):
        return {}

    return {
        str(row["key"]): str(row["value"])
        for _, row in state.iterrows()
    }


def build_equity_curve_export() -> pd.DataFrame:
    equity = read_csv_if_exists(PAPER_EQUITY_CURVE_PATH)

    if equity.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "sleeve",
                "equity",
                "daily_return",
                "drawdown",
            ]
        )

    equity = normalise_date_column(equity, "date").sort_values("date").reset_index(drop=True)

    for col in [
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
        "since_rebalance_pnl",
        "unrealised_pnl",
        "trade_count",
    ]:
        if col in equity.columns:
            equity[col] = pd.to_numeric(equity[col], errors="coerce")

    if "equity" in equity.columns:
        equity["daily_return"] = equity["equity"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        equity["running_peak"] = equity["equity"].cummax()
        equity["drawdown"] = equity["equity"] / equity["running_peak"] - 1.0
    else:
        equity["daily_return"] = 0.0
        equity["drawdown"] = 0.0

    equity.insert(1, "sleeve", SLEEVE_NAME)

    keep = [
        "date",
        "sleeve",
        "equity",
        "daily_return",
        "drawdown",
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
        "since_rebalance_pnl",
        "unrealised_pnl",
        "trade_count",
        "signal_date",
        "latest_price_date",
        "action",
        "rebalance_due",
    ]

    for col in keep:
        if col not in equity.columns:
            equity[col] = np.nan

    return equity[keep].copy()



def _build_price_matrix_for_equal_weight_benchmark() -> pd.DataFrame:
    """
    Build a clean date x ticker adjusted-close matrix for the equal-weight
    commodity benchmark.

    This version deliberately avoids renaming the whole DataFrame before
    selecting columns. Some vendor/yfinance-style files can contain both
    close-like and adjusted-close-like columns, which can accidentally create
    duplicate column names such as two ``adj_close`` columns. In pandas,
    ``data["adj_close"]`` then returns a DataFrame instead of a Series, which
    breaks ``pd.to_numeric``. Selecting columns by position first avoids that.
    """
    prices = read_csv_if_exists(PRICE_PATH)

    if prices.empty:
        return pd.DataFrame()

    universe_tickers = [ticker for ticker in UNIVERSE.keys() if ticker != "CASH"]
    data = prices.copy()

    def normalise_col_name(value: Any) -> str:
        return str(value).strip().lower().replace(" ", "_").replace("-", "_")

    columns = list(data.columns)
    normalised_columns = [normalise_col_name(col) for col in columns]

    def first_column_index(candidates: set[str]) -> int | None:
        for idx, normalised in enumerate(normalised_columns):
            if normalised in candidates:
                return idx
        return None

    def first_ticker_column_index(ticker: str) -> int | None:
        ticker_upper = ticker.upper()
        ticker_lower = ticker.lower()

        for idx, col in enumerate(columns):
            raw = str(col).strip()
            normalised = normalised_columns[idx]

            if raw.upper() == ticker_upper:
                return idx

            if normalised == ticker_lower:
                return idx

        return None

    date_idx = first_column_index({"date", "datetime", "timestamp"})
    ticker_idx = first_column_index({"ticker", "symbol"})

    adjusted_price_idx = first_column_index(
        {
            "adj_close",
            "adjusted_close",
            "adjclose",
            "adjusted",
        }
    )

    fallback_price_idx = first_column_index(
        {
            "close",
            "price",
            "last",
        }
    )

    price_idx = adjusted_price_idx if adjusted_price_idx is not None else fallback_price_idx

    if date_idx is None:
        return pd.DataFrame()

    # Long format: date / ticker / adj_close or close.
    if ticker_idx is not None and price_idx is not None:
        long_data = data.iloc[:, [date_idx, ticker_idx, price_idx]].copy()
        long_data.columns = ["date", "ticker", "adj_close"]

        long_data = normalise_date_column(long_data, "date")

        if long_data.empty:
            return pd.DataFrame()

        long_data["ticker"] = long_data["ticker"].map(clean_ticker)
        long_data = long_data[long_data["ticker"].isin(universe_tickers)].copy()
        long_data["adj_close"] = pd.to_numeric(long_data["adj_close"], errors="coerce")

        matrix = (
            long_data.dropna(subset=["adj_close"])
            .drop_duplicates(["date", "ticker"], keep="last")
            .pivot(index="date", columns="ticker", values="adj_close")
            .sort_index()
        )

    # Wide format: date plus one column per ticker, e.g. date, GLD, SLV, USO...
    else:
        ticker_indices: list[int] = []
        ticker_names: list[str] = []

        for ticker in universe_tickers:
            idx = first_ticker_column_index(ticker)

            if idx is not None:
                ticker_indices.append(idx)
                ticker_names.append(ticker)

        if not ticker_indices:
            return pd.DataFrame()

        wide_data = data.iloc[:, [date_idx] + ticker_indices].copy()
        wide_data.columns = ["date"] + ticker_names
        wide_data = normalise_date_column(wide_data, "date")

        if wide_data.empty:
            return pd.DataFrame()

        matrix = wide_data.drop_duplicates("date", keep="last")
        matrix = matrix.set_index("date").sort_index()
        matrix = matrix.apply(pd.to_numeric, errors="coerce")

    matrix = matrix.reindex(columns=[ticker for ticker in universe_tickers if ticker in matrix.columns])
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    matrix = matrix.ffill()
    matrix = matrix.dropna(how="all")

    # Require at least two priced assets on a day. A one-asset equal-weight
    # benchmark is misleading and usually signals a bad data slice.
    matrix = matrix[matrix.notna().sum(axis=1) >= 2]

    return matrix


def build_equal_weight_benchmark_export(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """
    Export a monthly rebalanced equal-weight commodity benchmark.

    This is the right first benchmark for the commodity sleeve: same asset
    universe, same commodity-only opportunity set, no CASH, no shorts, and no
    transaction costs. PortfolioManager can then compare the active sleeve with
    a passive commodity basket instead of an irrelevant 60/40 portfolio.
    """
    price_matrix = _build_price_matrix_for_equal_weight_benchmark()

    columns = [
        "date",
        "benchmark",
        "benchmark_name",
        "equity",
        "daily_return",
        "drawdown",
        "asset_count",
        "rebalance_flag",
    ]

    if price_matrix.empty:
        return pd.DataFrame(columns=columns)

    starting_equity = np.nan

    if not equity_curve.empty and "equity" in equity_curve.columns:
        starting_equity = safe_float(equity_curve.sort_values("date").iloc[0].get("equity"))

    if pd.isna(starting_equity) or starting_equity <= 0:
        starting_equity = 10_000.0

    holdings = pd.Series(0.0, index=price_matrix.columns, dtype="float64")
    previous_equity = float(starting_equity)
    current_month: pd.Period | None = None
    rows: list[dict[str, Any]] = []

    for date, prices in price_matrix.iterrows():
        valid_prices = prices.dropna()
        valid_prices = valid_prices[valid_prices > 0]

        if valid_prices.empty:
            continue

        month = pd.Timestamp(date).to_period("M")
        rebalance_flag = current_month is None or month != current_month

        if current_month is None:
            equity = previous_equity
        else:
            priced_holdings = holdings.reindex(valid_prices.index).fillna(0.0)
            equity = float((priced_holdings * valid_prices).sum())

            if equity <= 0 or np.isnan(equity) or np.isinf(equity):
                equity = previous_equity

        daily_return = 0.0 if not rows or previous_equity <= 0 else equity / previous_equity - 1.0

        if rebalance_flag:
            weight = 1.0 / len(valid_prices)
            holdings = pd.Series(0.0, index=price_matrix.columns, dtype="float64")
            holdings.loc[valid_prices.index] = (equity * weight) / valid_prices
            current_month = month

        rows.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "benchmark": "equal_weight_commodities",
                "benchmark_name": "Equal Weight Commodities",
                "equity": equity,
                "daily_return": daily_return,
                "asset_count": int(len(valid_prices)),
                "rebalance_flag": bool(rebalance_flag),
            }
        )

        previous_equity = equity

    if not rows:
        return pd.DataFrame(columns=columns)

    benchmark = pd.DataFrame(rows)
    benchmark["equity"] = pd.to_numeric(benchmark["equity"], errors="coerce")
    benchmark["daily_return"] = pd.to_numeric(benchmark["daily_return"], errors="coerce").fillna(0.0)
    benchmark["running_peak"] = benchmark["equity"].cummax()
    benchmark["drawdown"] = benchmark["equity"] / benchmark["running_peak"] - 1.0
    benchmark["drawdown"] = benchmark["drawdown"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return benchmark[columns].copy()

def build_positions_export(equity_curve: pd.DataFrame) -> pd.DataFrame:
    positions = read_csv_if_exists(LATEST_PORTFOLIO_PATH)

    if positions.empty:
        positions = read_csv_if_exists(PAPER_POSITIONS_PATH)

    if positions.empty:
        positions = pd.DataFrame(
            columns=[
                "ticker",
                "shares",
                "position_type",
                "market_value",
                "current_weight",
            ]
        )

    positions = positions.copy()

    if "ticker" not in positions.columns:
        positions["ticker"] = ""

    positions["ticker"] = positions["ticker"].map(clean_ticker)
    positions = add_asset_metadata(positions, "ticker")

    for col in [
        "shares",
        "avg_cost",
        "cost_basis",
        "price",
        "market_value",
        "abs_market_value",
        "current_weight",
        "abs_current_weight",
        "unrealised_pnl",
        "unrealised_return",
    ]:
        if col not in positions.columns:
            positions[col] = np.nan
        positions[col] = pd.to_numeric(positions[col], errors="coerce")

    if "position_type" not in positions.columns:
        positions["position_type"] = np.where(
            positions["shares"] > 0,
            "LONG",
            np.where(positions["shares"] < 0, "SHORT", "FLAT"),
        )

    latest_date = None
    cash_value = np.nan
    cash_buffer_weight = np.nan

    if not equity_curve.empty:
        latest_eq = equity_curve.sort_values("date").iloc[-1]
        latest_date = latest_eq.get("date")
        cash_value = safe_float(latest_eq.get("cash"))
        cash_buffer_weight = safe_float(latest_eq.get("cash_buffer_weight"))

    if pd.isna(cash_buffer_weight):
        invested_abs = positions["current_weight"].abs().sum(skipna=True)
        cash_buffer_weight = max(0.0, 1.0 - float(invested_abs))

    cash_row = {
        "ticker": "CASH",
        "asset_name": "Cash",
        "asset_group": "cash",
        "shares": np.nan,
        "avg_cost": np.nan,
        "position_type": "CASH",
        "cost_basis": np.nan,
        "price_date": latest_date,
        "price": 1.0,
        "market_value": cash_value,
        "abs_market_value": abs(cash_value) if not pd.isna(cash_value) else np.nan,
        "current_weight": cash_buffer_weight,
        "abs_current_weight": cash_buffer_weight,
        "unrealised_pnl": 0.0,
        "unrealised_return": 0.0,
    }

    positions = pd.concat([positions, pd.DataFrame([cash_row])], ignore_index=True)
    positions.insert(0, "sleeve", SLEEVE_NAME)

    keep = [
        "sleeve",
        "ticker",
        "asset_name",
        "asset_group",
        "position_type",
        "shares",
        "avg_cost",
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

    for col in keep:
        if col not in positions.columns:
            positions[col] = np.nan

    return positions[keep].copy()


def recommendation_from_weights(target: float, current: float) -> str:
    diff = target - current

    if abs(diff) < 0.0025:
        return "HOLD"

    if target > 0.001 and current <= 0.001:
        return "OPEN_LONG"

    if target < -0.001 and current >= -0.001:
        return "OPEN_SHORT"

    if abs(target) <= 0.001 and abs(current) > 0.001:
        return "EXIT"

    if diff > 0:
        return "INCREASE"

    return "REDUCE"


def build_signal_export(positions: pd.DataFrame, equity_curve: pd.DataFrame) -> pd.DataFrame:
    weights_raw = read_csv_if_exists(TARGET_WEIGHTS_PATH)

    if weights_raw.empty:
        raise FileNotFoundError(
            f"No target weights found at {TARGET_WEIGHTS_PATH}. "
            "Run commodity_strategy.py first."
        )

    if not {"date", "ticker", "target_weight"}.issubset(weights_raw.columns):
        raise ValueError("target_weights.csv must contain date, ticker, target_weight.")

    latest_weights, signal_date = latest_rows_by_date(weights_raw, "date")

    if latest_weights.empty or signal_date is None:
        raise ValueError("No latest target weights available.")

    latest_weights["ticker"] = latest_weights["ticker"].map(clean_ticker)
    latest_weights["target_weight"] = pd.to_numeric(latest_weights["target_weight"], errors="coerce").fillna(0.0)

    scores_raw = read_csv_if_exists(FINAL_SCORES_PATH)
    latest_scores = pd.DataFrame()

    if not scores_raw.empty and {"date", "ticker"}.issubset(scores_raw.columns):
        scores_raw = normalise_date_column(scores_raw, "date")
        scores_raw["ticker"] = scores_raw["ticker"].map(clean_ticker)
        scores_raw = scores_raw[scores_raw["date"].dt.normalize() <= signal_date].copy()

        if not scores_raw.empty:
            score_date = pd.Timestamp(scores_raw["date"].max()).normalize()
            latest_scores = scores_raw[scores_raw["date"].dt.normalize() == score_date].copy()

    signal = latest_weights.copy()

    if not latest_scores.empty:
        scores_idx = latest_scores.drop_duplicates("ticker").set_index("ticker")

        for col in latest_scores.columns:
            if col in ["date", "ticker"]:
                continue

            if col not in signal.columns:
                signal[col] = signal["ticker"].map(scores_idx[col])

    positions_no_cash = positions[positions["ticker"] != "CASH"].copy()

    if not positions_no_cash.empty:
        pos_idx = positions_no_cash.drop_duplicates("ticker").set_index("ticker")

        for col in [
            "current_weight",
            "position_type",
            "market_value",
            "unrealised_pnl",
            "unrealised_return",
        ]:
            if col in pos_idx.columns:
                signal[col] = signal["ticker"].map(pos_idx[col])

    if "current_weight" not in signal.columns:
        signal["current_weight"] = 0.0

    signal["current_weight"] = pd.to_numeric(signal["current_weight"], errors="coerce").fillna(0.0)
    signal["weight_drift"] = signal["target_weight"] - signal["current_weight"]
    signal["recommendation"] = [
        recommendation_from_weights(t, c)
        for t, c in zip(signal["target_weight"], signal["current_weight"])
    ]

    signal = add_asset_metadata(signal, "ticker")

    target_cash_buffer = max(0.0, 1.0 - float(signal["target_weight"].abs().sum()))

    current_cash_buffer = np.nan
    if not equity_curve.empty and "cash_buffer_weight" in equity_curve.columns:
        current_cash_buffer = safe_float(equity_curve.sort_values("date").iloc[-1].get("cash_buffer_weight"))

    if pd.isna(current_cash_buffer):
        current_cash_buffer = max(0.0, 1.0 - float(signal["current_weight"].abs().sum()))

    cash_row = {
        "date": signal_date,
        "ticker": "CASH",
        "target_weight": target_cash_buffer,
        "current_weight": current_cash_buffer,
        "weight_drift": target_cash_buffer - current_cash_buffer,
        "recommendation": recommendation_from_weights(target_cash_buffer, current_cash_buffer),
        "asset_name": "Cash",
        "asset_group": "cash",
        "final_score": np.nan,
        "rank": np.nan,
        "position_type": "CASH",
    }

    signal = pd.concat([signal, pd.DataFrame([cash_row])], ignore_index=True)

    signal.insert(0, "sleeve", SLEEVE_NAME)
    signal["as_of_date"] = pd.Timestamp(signal_date).date().isoformat()
    signal["source_signal_date"] = pd.Timestamp(signal_date).date().isoformat()
    signal["model_version"] = MODEL_VERSION
    signal["exported_at_utc"] = utc_now()

    preferred = [
        "as_of_date",
        "sleeve",
        "ticker",
        "asset_name",
        "asset_group",
        "target_weight",
        "current_weight",
        "weight_drift",
        "recommendation",
        "position_type",
        "final_score",
        "rank",
        "signal_quality",
        "momentum_score",
        "relative_strength_score",
        "trend_score",
        "trend_persistence_score",
        "volatility_score",
        "risk_score",
        "macro_score",
        "realised_vol_60d",
        "macro_group",
        "macro_regime",
        "commodity_regime",
        "bear_regime_flag",
        "bear_short_candidate",
        "bear_short_weight",
        "cash_weight",
        "market_value",
        "unrealised_pnl",
        "unrealised_return",
        "source_signal_date",
        "model_version",
        "exported_at_utc",
    ]

    extra_cols = [col for col in signal.columns if col not in preferred]
    return signal[[col for col in preferred if col in signal.columns] + extra_cols].copy()


def build_orders_export() -> pd.DataFrame:
    orders = read_csv_if_exists(LATEST_ORDERS_PATH)

    if orders.empty:
        return pd.DataFrame(
            columns=[
                "sleeve",
                "ticker",
                "side",
                "filled_shares",
                "filled_notional",
                "target_weight",
                "current_weight_before_trade",
                "reason",
            ]
        )

    orders = orders.copy()

    if "ticker" in orders.columns:
        orders["ticker"] = orders["ticker"].map(clean_ticker)
        orders = add_asset_metadata(orders, "ticker")

    orders.insert(0, "sleeve", SLEEVE_NAME)
    orders["exported_at_utc"] = utc_now()

    return orders


def build_metrics_export(equity_curve: pd.DataFrame, signal: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    if equity_curve.empty:
        latest_equity = {}
        returns = pd.Series(dtype=float)
    else:
        equity_curve = equity_curve.sort_values("date").copy()
        latest_equity = equity_curve.iloc[-1].to_dict()
        returns = pd.to_numeric(equity_curve["daily_return"], errors="coerce").dropna()

    current_drawdown = safe_float(latest_equity.get("drawdown"), 0.0)
    max_drawdown = safe_float(equity_curve["drawdown"].min() if not equity_curve.empty and "drawdown" in equity_curve.columns else np.nan)

    vol_20d = returns.tail(20).std(ddof=0) * np.sqrt(252) if len(returns) >= 5 else np.nan
    vol_60d = returns.tail(60).std(ddof=0) * np.sqrt(252) if len(returns) >= 10 else np.nan
    var_95 = returns.quantile(0.05) if len(returns) >= 20 else np.nan
    cvar_95 = returns[returns <= var_95].mean() if len(returns) >= 20 and not pd.isna(var_95) else np.nan
    win_rate = float((returns > 0).mean()) if len(returns) > 0 else np.nan

    active_signal = signal[(signal["ticker"] != "CASH") & (signal["target_weight"].abs() > 0.001)].copy()

    top_target_asset = None
    if not active_signal.empty:
        top_target_asset = active_signal.sort_values("target_weight", key=lambda x: x.abs(), ascending=False).iloc[0]["ticker"]

    row = {
        "as_of_date": latest_equity.get("date", pd.NaT),
        "sleeve": SLEEVE_NAME,
        "model_version": MODEL_VERSION,
        "equity": safe_float(latest_equity.get("equity")),
        "daily_return": safe_float(latest_equity.get("daily_return")),
        "daily_pnl": safe_float(latest_equity.get("daily_pnl")),
        "cumulative_pnl": safe_float(latest_equity.get("cumulative_pnl")),
        "cash": safe_float(latest_equity.get("cash")),
        "cash_weight": safe_float(latest_equity.get("cash_weight")),
        "cash_buffer_weight": safe_float(latest_equity.get("cash_buffer_weight")),
        "gross_exposure": safe_float(latest_equity.get("gross_exposure")),
        "net_exposure": safe_float(latest_equity.get("net_exposure")),
        "long_exposure": safe_float(latest_equity.get("long_exposure")),
        "short_exposure": safe_float(latest_equity.get("short_exposure")),
        "current_drawdown": current_drawdown,
        "max_drawdown": max_drawdown,
        "realised_vol_20d": safe_float(vol_20d),
        "realised_vol_60d": safe_float(vol_60d),
        "var_95_1d": safe_float(var_95),
        "cvar_95_1d": safe_float(cvar_95),
        "win_rate": safe_float(win_rate),
        "active_positions": int((positions["current_weight"].abs() > 0.001).sum()) if "current_weight" in positions.columns else 0,
        "top_target_asset": top_target_asset,
        "signal_date": latest_equity.get("signal_date"),
        "latest_price_date": latest_equity.get("latest_price_date"),
        "action": latest_equity.get("action"),
        "rebalance_due": latest_equity.get("rebalance_due"),
        "exported_at_utc": utc_now(),
    }

    return pd.DataFrame([row])


def build_memo_seed(metrics: pd.DataFrame, signal: pd.DataFrame, positions: pd.DataFrame) -> dict[str, Any]:
    metric_row = metrics.iloc[0].to_dict() if not metrics.empty else {}

    active_signal = signal[(signal["ticker"] != "CASH") & (signal["target_weight"].abs() > 0.001)].copy()

    top_targets = (
        active_signal.sort_values("target_weight", key=lambda x: x.abs(), ascending=False)
        .head(3)[["ticker", "asset_name", "target_weight", "final_score", "recommendation"]]
        .to_dict(orient="records")
        if not active_signal.empty
        else []
    )

    largest_drifts = (
        signal[signal["ticker"] != "CASH"]
        .assign(abs_drift=lambda x: x["weight_drift"].abs())
        .sort_values("abs_drift", ascending=False)
        .head(3)[["ticker", "asset_name", "target_weight", "current_weight", "weight_drift", "recommendation"]]
        .to_dict(orient="records")
        if not signal.empty and "weight_drift" in signal.columns
        else []
    )

    return {
        "sleeve": SLEEVE_NAME,
        "model_version": MODEL_VERSION,
        "generated_at_utc": utc_now(),
        "summary_metrics": metric_row,
        "top_targets": top_targets,
        "largest_drifts": largest_drifts,
        "instruction": (
            "Use this structured data to write a factual portfolio memo. "
            "Do not invent market views. Explain only what changed in the system outputs."
        ),
    }


def json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float) and np.isnan(value):
        return None

    return str(value)


def write_manifest(outputs: dict[str, Path]) -> None:
    files = {}

    for name, path in outputs.items():
        row_count = None

        if path.suffix.lower() == ".csv" and path.exists():
            try:
                row_count = len(pd.read_csv(path))
            except Exception:
                row_count = None

        files[name] = {
            "path": str(path),
            "exists": path.exists(),
            "row_count": row_count,
            "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
            if path.exists()
            else None,
        }

    manifest = {
        "sleeve": SLEEVE_NAME,
        "model_version": MODEL_VERSION,
        "exported_at_utc": utc_now(),
        "commodity_root": str(BASE_DIR),
        "export_dir": str(EXPORT_DIR),
        "files": files,
    }

    OUT_MANIFEST.write_text(
        json.dumps(manifest, indent=2, default=json_default),
        encoding="utf-8",
    )


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    equity_curve = build_equity_curve_export()
    positions = build_positions_export(equity_curve)
    benchmark = build_equal_weight_benchmark_export(equity_curve)
    signal = build_signal_export(positions, equity_curve)
    orders = build_orders_export()
    metrics = build_metrics_export(equity_curve, signal, positions)
    memo_seed = build_memo_seed(metrics, signal, positions)

    signal.to_csv(OUT_SIGNAL, index=False)
    positions.to_csv(OUT_POSITIONS, index=False)
    equity_curve.to_csv(OUT_EQUITY_CURVE, index=False)
    benchmark.to_csv(OUT_EQUAL_WEIGHT_BENCHMARK, index=False)
    metrics.to_csv(OUT_METRICS, index=False)
    orders.to_csv(OUT_ORDERS, index=False)

    OUT_MEMO_SEED.write_text(
        json.dumps(memo_seed, indent=2, default=json_default),
        encoding="utf-8",
    )

    write_manifest(
        {
            "signal": OUT_SIGNAL,
            "positions": OUT_POSITIONS,
            "equity_curve": OUT_EQUITY_CURVE,
            "equal_weight_benchmark": OUT_EQUAL_WEIGHT_BENCHMARK,
            "metrics": OUT_METRICS,
            "orders": OUT_ORDERS,
            "memo_seed": OUT_MEMO_SEED,
        }
    )

    print("\n" + "=" * 80)
    print("COMMODITY EXPORT FOR PORTFOLIO MANAGER COMPLETE")
    print("=" * 80)
    print(f"Export directory: {EXPORT_DIR}")
    print(f"- {OUT_SIGNAL}")
    print(f"- {OUT_POSITIONS}")
    print(f"- {OUT_EQUITY_CURVE}")
    print(f"- {OUT_EQUAL_WEIGHT_BENCHMARK}")
    print(f"- {OUT_METRICS}")
    print(f"- {OUT_ORDERS}")
    print(f"- {OUT_MANIFEST}")
    print(f"- {OUT_MEMO_SEED}")
    print("=" * 80)


if __name__ == "__main__":
    main()