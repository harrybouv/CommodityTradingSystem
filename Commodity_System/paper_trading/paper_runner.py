# Commodity_System/paper_trading/paper_runner.py

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
PAPER_TRADING_DIR = THIS_FILE.parent
COMMODITY_ROOT = THIS_FILE.parents[1]

for path in [COMMODITY_ROOT, PAPER_TRADING_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# ============================================================
# IMPORTS
# ============================================================

from paper_config import (
    PAPER_TICKERS,
    PAPER_INITIAL_CAPITAL,
    PAPER_CASH_ANNUAL_YIELD,
    ALLOW_FRACTIONAL_SHARES,
    PAPER_COMMISSION_PER_TRADE,
    PAPER_SLIPPAGE_BPS,
    MIN_TRADE_NOTIONAL,
    MIN_WEIGHT_CHANGE_TO_TRADE,
    PAPER_REBALANCE_POLICY,
    FORCE_REBALANCE,
    OVERWRITE_SAME_SIGNAL_DATE_EQUITY_ROW,
    TARGET_WEIGHTS_SOURCE_PATH,
    FINAL_SCORES_SOURCE_PATH,
    PRICE_SOURCE_PATH,
    PAPER_STATE_DIR,
    PAPER_REPORTS_DIR,
    PAPER_TARGET_WEIGHT_SNAPSHOT_DIR,
    PAPER_FINAL_SCORE_SNAPSHOT_DIR,
    PAPER_DECISION_SNAPSHOT_DIR,
    PAPER_POSITIONS_PATH,
    PAPER_TRADES_PATH,
    PAPER_EQUITY_CURVE_PATH,
    PAPER_STATE_PATH,
    LATEST_ORDERS_PATH,
    LATEST_PORTFOLIO_PATH,
    LATEST_SUMMARY_PATH,
    DATE_FORMAT,
    SNAPSHOT_DATE_FORMAT,
    FLOAT_FORMAT,
)

try:
    from paper_config import (
        PAPER_ALLOW_SHORTS,
        PAPER_SYNTHETIC_SHORTS_ONLY,
        SHORTING_ALLOWED_BY_TICKER,
        PAPER_MAX_GROSS_EXPOSURE,
        PAPER_MAX_TOTAL_SHORT,
        PAPER_MAX_SINGLE_SHORT,
        ALLOW_NEGATIVE_CASH,
    )
except ImportError:
    PAPER_ALLOW_SHORTS = False
    PAPER_SYNTHETIC_SHORTS_ONLY = True
    SHORTING_ALLOWED_BY_TICKER = {ticker: False for ticker in PAPER_TICKERS}
    PAPER_MAX_GROSS_EXPOSURE = 1.0
    PAPER_MAX_TOTAL_SHORT = 0.0
    PAPER_MAX_SINGLE_SHORT = 0.0
    ALLOW_NEGATIVE_CASH = False


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_directories() -> None:
    for path in [
        PAPER_STATE_DIR,
        PAPER_REPORTS_DIR,
        PAPER_TARGET_WEIGHT_SNAPSHOT_DIR,
        PAPER_FINAL_SCORE_SNAPSHOT_DIR,
        PAPER_DECISION_SNAPSHOT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _require_columns(df: pd.DataFrame, required_cols: list[str], name: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _clean_ticker(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _to_datetime_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _safe_numeric(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=FLOAT_FORMAT)


def _append_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        out = pd.concat([existing, df], ignore_index=True)
    else:
        out = df.copy()
    out.to_csv(path, index=False, float_format=FLOAT_FORMAT)


def _empty_orders_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "run_timestamp_utc",
            "accounting_date",
            "signal_date",
            "ticker",
            "side",
            "requested_shares",
            "filled_shares",
            "raw_price",
            "fill_price",
            "raw_notional",
            "filled_notional",
            "commission",
            "slippage_cost",
            "cash_flow",
            "realised_pnl",
            "target_weight",
            "current_weight_before_trade",
            "reason",
            "scaled_buy_order",
        ]
    )


def _date_str(date: pd.Timestamp) -> str:
    return pd.Timestamp(date).normalize().strftime(DATE_FORMAT)


def _period_str(date: pd.Timestamp) -> str:
    return pd.Timestamp(date).normalize().strftime("%Y-%m")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if np.isnan(out) or np.isinf(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _position_type_from_shares(shares: float) -> str:
    if shares > 1e-10:
        return "LONG"
    if shares < -1e-10:
        return "SHORT"
    return "FLAT"


# ============================================================
# STATE MANAGEMENT
# ============================================================

def default_state() -> dict[str, str]:
    return {
        "initial_capital": f"{PAPER_INITIAL_CAPITAL:.10f}",
        "cash": f"{PAPER_INITIAL_CAPITAL:.10f}",
        "last_run_date": "",
        "last_signal_date": "",
        "last_rebalance_period": "",
        "last_equity": f"{PAPER_INITIAL_CAPITAL:.10f}",
        "last_price_date": "",
    }


def load_state() -> dict[str, str]:
    if not PAPER_STATE_PATH.exists():
        return default_state()
    raw = pd.read_csv(PAPER_STATE_PATH)
    _require_columns(raw, ["key", "value"], "paper_state.csv")
    state = {str(row["key"]): str(row["value"]) for _, row in raw.iterrows()}
    merged = default_state()
    merged.update(state)
    return merged


def save_state(state: dict[str, Any]) -> None:
    _write_csv(pd.DataFrame([{"key": key, "value": value} for key, value in state.items()]), PAPER_STATE_PATH)


def _state_float(state: dict[str, str], key: str, default: float) -> float:
    try:
        return float(state.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _state_date(state: dict[str, str], key: str) -> pd.Timestamp | None:
    value = state.get(key, "")
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


# ============================================================
# LOAD EXISTING STRATEGY OUTPUTS
# ============================================================

def load_latest_target_weights() -> tuple[pd.DataFrame, pd.Timestamp]:
    if not TARGET_WEIGHTS_SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Target weights not found: {TARGET_WEIGHTS_SOURCE_PATH}\n"
            "Run commodity_strategy.py or backtest_runner.py first."
        )

    weights = pd.read_csv(TARGET_WEIGHTS_SOURCE_PATH)
    _require_columns(weights, ["date", "ticker", "target_weight"], "target_weights.csv")

    weights = weights.copy()
    weights["date"] = _to_datetime_series(weights["date"])
    weights["ticker"] = _clean_ticker(weights["ticker"])
    weights["target_weight"] = _safe_numeric(weights["target_weight"], 0.0)
    weights = weights.dropna(subset=["date"]).copy()
    weights = weights[weights["ticker"].isin(PAPER_TICKERS)].copy()

    if weights.empty:
        raise ValueError("No usable target weights found for PAPER_TICKERS.")

    latest_signal_date = pd.Timestamp(weights["date"].max()).normalize()
    latest = weights[weights["date"] == latest_signal_date].copy()

    duplicate_count = int(latest.duplicated("ticker").sum())
    if duplicate_count > 0:
        raise ValueError(f"Latest target weights contain {duplicate_count} duplicate ticker rows.")

    missing = sorted(set(PAPER_TICKERS).difference(set(latest["ticker"])))
    if missing:
        raise ValueError(
            f"Latest target weights are missing tickers: {missing}. "
            "The strategy output should contain every ticker in the universe."
        )

    if not bool(PAPER_ALLOW_SHORTS) and (latest["target_weight"] < -1e-12).any():
        bad = latest.loc[latest["target_weight"] < -1e-12, ["ticker", "target_weight"]]
        raise ValueError(
            "Negative target weights detected but PAPER_ALLOW_SHORTS=False.\n"
            f"Negative targets:\n{bad.to_string(index=False)}"
        )

    if bool(PAPER_ALLOW_SHORTS):
        blocked = []
        for idx, row in latest.iterrows():
            ticker = str(row["ticker"])
            target = float(row["target_weight"])
            if target < -1e-12 and not bool(SHORTING_ALLOWED_BY_TICKER.get(ticker, False)):
                blocked.append((ticker, target))
                latest.at[idx, "target_weight"] = 0.0
        if blocked:
            print("Warning: blocked unavailable paper shorts and set them to zero:")
            for ticker, target in blocked:
                print(f"  {ticker}: {target:.2%}")
    else:
        latest["target_weight"] = latest["target_weight"].clip(lower=0.0)

    gross_weight = float(latest["target_weight"].abs().sum())
    net_weight = float(latest["target_weight"].sum())
    short_weight = float(latest.loc[latest["target_weight"] < 0, "target_weight"].abs().sum())
    max_single_short = float(latest.loc[latest["target_weight"] < 0, "target_weight"].abs().max() or 0.0)

    if gross_weight > float(PAPER_MAX_GROSS_EXPOSURE) + 1e-6:
        raise ValueError(
            f"Latest target weights gross exposure is {gross_weight:.6f}, above "
            f"PAPER_MAX_GROSS_EXPOSURE={PAPER_MAX_GROSS_EXPOSURE:.6f}."
        )

    if short_weight > float(PAPER_MAX_TOTAL_SHORT) + 1e-6:
        raise ValueError(
            f"Latest total short exposure is {short_weight:.6f}, above "
            f"PAPER_MAX_TOTAL_SHORT={PAPER_MAX_TOTAL_SHORT:.6f}."
        )

    if max_single_short > float(PAPER_MAX_SINGLE_SHORT) + 1e-6:
        raise ValueError(
            f"Latest single-name short exposure is {max_single_short:.6f}, above "
            f"PAPER_MAX_SINGLE_SHORT={PAPER_MAX_SINGLE_SHORT:.6f}."
        )

    if not bool(PAPER_ALLOW_SHORTS) and net_weight > 1.000001:
        raise ValueError(f"Latest target weights sum to {net_weight:.6f}, above 1.0.")

    return latest.sort_values("ticker").reset_index(drop=True), latest_signal_date


def load_latest_final_scores(signal_date: pd.Timestamp) -> pd.DataFrame:
    if not FINAL_SCORES_SOURCE_PATH.exists():
        return pd.DataFrame()
    scores = pd.read_csv(FINAL_SCORES_SOURCE_PATH)
    if scores.empty:
        return pd.DataFrame()
    _require_columns(scores, ["date", "ticker", "final_score", "rank"], "final_scores.csv")
    scores = scores.copy()
    scores["date"] = _to_datetime_series(scores["date"])
    scores["ticker"] = _clean_ticker(scores["ticker"])
    scores = scores[(scores["date"] == signal_date) & (scores["ticker"].isin(PAPER_TICKERS))].copy()
    return scores.sort_values("ticker").reset_index(drop=True)


def load_latest_prices_asof(asof_date: pd.Timestamp) -> pd.DataFrame:
    if not PRICE_SOURCE_PATH.exists():
        raise FileNotFoundError(f"Price file not found: {PRICE_SOURCE_PATH}\nRun data.py / backtest_runner.py first.")
    prices = pd.read_csv(PRICE_SOURCE_PATH)
    _require_columns(prices, ["date", "ticker", "adj_close"], "commodity_prices.csv")
    prices = prices.copy()
    prices["date"] = _to_datetime_series(prices["date"])
    prices["ticker"] = _clean_ticker(prices["ticker"])
    prices["adj_close"] = _safe_numeric(prices["adj_close"], np.nan)
    prices = prices.dropna(subset=["date", "ticker", "adj_close"]).copy()
    prices = prices[prices["ticker"].isin(PAPER_TICKERS)].copy()
    prices = prices[prices["date"] <= pd.Timestamp(asof_date).normalize()].copy()
    if prices.empty:
        raise ValueError(f"No prices available on or before accounting date {asof_date.date()}.")
    latest = prices.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1).copy()
    latest = latest.rename(columns={"date": "price_date", "adj_close": "price"})
    latest = latest[["ticker", "price_date", "price"]].copy()
    missing = sorted(set(PAPER_TICKERS).difference(set(latest["ticker"])))
    if missing:
        raise ValueError(f"Missing latest prices for tickers: {missing}.")
    latest["price"] = _safe_numeric(latest["price"], np.nan)
    bad_prices = latest[latest["price"].isna() | (latest["price"] <= 0)]
    if not bad_prices.empty:
        raise ValueError(f"Invalid latest prices for tickers: {bad_prices['ticker'].tolist()}")
    return latest.sort_values("ticker").reset_index(drop=True)


# ============================================================
# POSITIONS
# ============================================================

def empty_positions() -> pd.DataFrame:
    return pd.DataFrame({"ticker": PAPER_TICKERS, "shares": 0.0, "avg_cost": 0.0})


def load_positions() -> pd.DataFrame:
    if not PAPER_POSITIONS_PATH.exists():
        return empty_positions()
    positions = pd.read_csv(PAPER_POSITIONS_PATH)
    _require_columns(positions, ["ticker", "shares", "avg_cost"], "paper_positions.csv")
    positions = positions.copy()
    positions["ticker"] = _clean_ticker(positions["ticker"])
    positions["shares"] = _safe_numeric(positions["shares"], 0.0)
    positions["avg_cost"] = _safe_numeric(positions["avg_cost"], 0.0)
    positions = empty_positions()[["ticker"]].merge(
        positions[["ticker", "shares", "avg_cost"]], on="ticker", how="left"
    ).fillna({"shares": 0.0, "avg_cost": 0.0})
    return positions.sort_values("ticker").reset_index(drop=True)


def mark_positions_to_market(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    cash: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out = positions.copy()
    out = out.merge(prices, on="ticker", how="left", validate="one_to_one")
    _require_columns(out, ["ticker", "shares", "avg_cost", "price", "price_date"], "marked positions")

    out["cost_basis"] = out["shares"] * out["avg_cost"]
    out["market_value"] = out["shares"] * out["price"]
    out["abs_market_value"] = out["market_value"].abs()
    out["position_type"] = out["shares"].map(_position_type_from_shares)

    out["unrealised_pnl"] = np.where(
        out["shares"].abs() > 1e-12,
        (out["price"] - out["avg_cost"]) * out["shares"],
        0.0,
    )
    out["unrealised_return"] = np.where(
        out["cost_basis"].abs() > 1e-12,
        out["unrealised_pnl"] / out["cost_basis"].abs(),
        0.0,
    )

    net_invested_value = float(out["market_value"].sum())
    gross_invested_value = float(out["abs_market_value"].sum())
    long_value = float(out.loc[out["market_value"] > 0, "market_value"].sum())
    short_value = float(out.loc[out["market_value"] < 0, "market_value"].abs().sum())

    equity = float(cash + net_invested_value)
    if equity <= 0:
        raise ValueError(f"Paper account equity is non-positive: {equity:.2f}. State file may be corrupted.")

    out["current_weight"] = out["market_value"] / equity
    out["abs_current_weight"] = out["current_weight"].abs()

    stats = {
        "cash": float(cash),
        "invested_value": net_invested_value,
        "gross_invested_value": gross_invested_value,
        "long_value": long_value,
        "short_value": short_value,
        "equity": equity,
        "net_exposure": net_invested_value / equity,
        "gross_exposure": gross_invested_value / equity,
        "long_exposure": long_value / equity,
        "short_exposure": short_value / equity,
        # Backward-compatible alias now means risk/gross exposure, not net long-only exposure.
        "total_exposure": gross_invested_value / equity,
        # Actual cash balance can exceed 100% of equity when shorts are open.
        "cash_weight": float(cash / equity),
        # This is the clean portfolio cash buffer comparable to the backtest target cash.
        "cash_buffer_weight": max(0.0, 1.0 - gross_invested_value / equity),
        "unrealised_pnl": float(out["unrealised_pnl"].sum()),
        "cost_basis": float(out["cost_basis"].sum()),
        "abs_cost_basis": float(out["cost_basis"].abs().sum()),
    }
    return out.sort_values("ticker").reset_index(drop=True), stats


def save_positions(marked_positions: pd.DataFrame) -> None:
    cols = [
        "ticker", "shares", "avg_cost", "position_type", "cost_basis", "price_date", "price",
        "market_value", "abs_market_value", "current_weight", "abs_current_weight",
        "unrealised_pnl", "unrealised_return",
    ]
    out = marked_positions.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    _write_csv(out[cols], PAPER_POSITIONS_PATH)
    _write_csv(out[cols], LATEST_PORTFOLIO_PATH)


# ============================================================
# CASH YIELD
# ============================================================

def apply_cash_yield(cash: float, state: dict[str, str], accounting_date: pd.Timestamp) -> tuple[float, float, int]:
    last_run_date = _state_date(state, "last_run_date")
    if last_run_date is None:
        return cash, 0.0, 0
    accounting_date = pd.Timestamp(accounting_date).normalize()
    days_elapsed = int((accounting_date - last_run_date).days)
    if days_elapsed <= 0:
        return cash, 0.0, 0
    interest = cash * ((1.0 + PAPER_CASH_ANNUAL_YIELD) ** (days_elapsed / 365.0) - 1.0)
    return cash + float(interest), float(interest), days_elapsed


# ============================================================
# REBALANCE DECISION
# ============================================================

def signal_period(signal_date: pd.Timestamp) -> str:
    if PAPER_REBALANCE_POLICY != "new_signal_month":
        raise ValueError("Only PAPER_REBALANCE_POLICY='new_signal_month' is currently supported.")
    return _period_str(signal_date)


def is_rebalance_due(signal_date: pd.Timestamp, accounting_date: pd.Timestamp, state: dict[str, str]) -> tuple[bool, str]:
    period = _period_str(accounting_date)
    if FORCE_REBALANCE:
        return True, period
    last_rebalance_period = state.get("last_rebalance_period", "")
    if period == last_rebalance_period:
        return False, period
    return True, period


# ============================================================
# ORDER GENERATION
# ============================================================

def _order_side(current_shares: float, target_shares: float) -> list[tuple[str, float]]:
    """Return order legs as (side, signed_requested_shares)."""
    eps = 1e-10
    if abs(target_shares - current_shares) <= eps:
        return []

    # Same side or one side flat.
    if current_shares >= -eps and target_shares >= -eps:
        delta = target_shares - current_shares
        return [("BUY" if delta > 0 else "SELL", delta)]

    if current_shares <= eps and target_shares <= eps:
        delta = target_shares - current_shares
        return [("COVER" if delta > 0 else "SHORT", delta)]

    # Cross through zero: split so realised P&L and avg cost are clean.
    if current_shares > eps and target_shares < -eps:
        return [("SELL", -current_shares), ("SHORT", target_shares)]

    if current_shares < -eps and target_shares > eps:
        return [("COVER", -current_shares), ("BUY", target_shares)]

    return []


def generate_target_orders(
    latest_weights: pd.DataFrame,
    marked_positions: pd.DataFrame,
    account_stats: dict[str, float],
    signal_date: pd.Timestamp,
    accounting_date: pd.Timestamp,
    run_timestamp_utc: str,
) -> pd.DataFrame:
    equity = float(account_stats["equity"])
    data = latest_weights[["ticker", "target_weight"]].copy()
    data = data.merge(
        marked_positions[["ticker", "shares", "price", "current_weight"]],
        on="ticker",
        how="left",
        validate="one_to_one",
    )
    _require_columns(data, ["ticker", "target_weight", "shares", "price", "current_weight"], "order generation input")

    data["target_value"] = data["target_weight"] * equity
    data["target_shares"] = data["target_value"] / data["price"]
    if not ALLOW_FRACTIONAL_SHARES:
        # Preserve sign for shorts.
        data["target_shares"] = np.where(
            data["target_shares"] >= 0,
            np.floor(data["target_shares"]),
            np.ceil(data["target_shares"]),
        )

    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        ticker = str(row["ticker"])
        price = float(row["price"])
        target_weight = float(row["target_weight"])
        current_weight = float(row["current_weight"])
        current_shares = float(row["shares"])
        target_shares = float(row["target_shares"])
        weight_change = abs(target_weight - current_weight)

        for side, requested_shares in _order_side(current_shares, target_shares):
            raw_notional = abs(requested_shares) * price
            if raw_notional < MIN_TRADE_NOTIONAL or weight_change < MIN_WEIGHT_CHANGE_TO_TRADE:
                continue
            rows.append(
                {
                    "run_timestamp_utc": run_timestamp_utc,
                    "accounting_date": _date_str(accounting_date),
                    "signal_date": _date_str(signal_date),
                    "ticker": ticker,
                    "side": side,
                    "requested_shares": requested_shares,
                    "raw_price": price,
                    "raw_notional": raw_notional,
                    "target_weight": target_weight,
                    "current_weight_before_trade": current_weight,
                    "reason": "monthly_rebalance",
                }
            )

    if not rows:
        return _empty_orders_frame()

    side_order = {"SELL": 0, "SHORT": 1, "COVER": 2, "BUY": 3}
    orders = pd.DataFrame(rows)
    orders["_side_order"] = orders["side"].map(side_order).fillna(9)
    orders = orders.sort_values(["_side_order", "ticker"]).drop(columns=["_side_order"])
    return orders.reset_index(drop=True)


# ============================================================
# ORDER EXECUTION
# ============================================================

def execute_orders(
    orders: pd.DataFrame,
    positions: pd.DataFrame,
    cash: float,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    if orders.empty:
        return positions.copy(), float(cash), _empty_orders_frame()

    pos = positions.copy().set_index("ticker")
    executed_rows: list[dict[str, Any]] = []
    slippage = float(PAPER_SLIPPAGE_BPS) / 10_000.0

    def append_execution(order: pd.Series, filled_shares: float, fill_price: float, cash_flow: float, realised_pnl: float, scaled: bool = False) -> None:
        executed_rows.append(
            {
                **order.to_dict(),
                "filled_shares": filled_shares,
                "fill_price": fill_price,
                "filled_notional": abs(filled_shares) * fill_price,
                "commission": float(PAPER_COMMISSION_PER_TRADE),
                "slippage_cost": abs(filled_shares) * float(order["raw_price"]) * slippage,
                "cash_flow": cash_flow,
                "realised_pnl": realised_pnl,
                "scaled_buy_order": scaled,
            }
        )

    def execute_cash_generating(order: pd.Series) -> None:
        nonlocal cash
        ticker = str(order["ticker"])
        side = str(order["side"])
        requested_abs = abs(float(order["requested_shares"]))
        raw_price = float(order["raw_price"])
        fill_price = raw_price * (1.0 - slippage)
        commission = float(PAPER_COMMISSION_PER_TRADE)

        current_shares = float(pos.loc[ticker, "shares"])
        avg_cost = float(pos.loc[ticker, "avg_cost"])

        if side == "SELL":
            filled_abs = min(requested_abs, max(current_shares, 0.0))
            if filled_abs <= 0:
                return
            realised_pnl = (fill_price - avg_cost) * filled_abs
            cash_flow = filled_abs * fill_price - commission
            cash += cash_flow
            new_shares = current_shares - filled_abs
            if abs(new_shares) <= 1e-10:
                new_shares = 0.0
                new_avg_cost = 0.0
            else:
                new_avg_cost = avg_cost
            pos.loc[ticker, "shares"] = new_shares
            pos.loc[ticker, "avg_cost"] = new_avg_cost
            append_execution(order, filled_abs, fill_price, cash_flow, realised_pnl)
            return

        if side == "SHORT":
            if not bool(PAPER_ALLOW_SHORTS):
                return
            if not bool(SHORTING_ALLOWED_BY_TICKER.get(ticker, False)):
                return
            filled_abs = requested_abs
            if filled_abs <= 0:
                return
            existing_short_abs = abs(min(current_shares, 0.0))
            new_short_abs = existing_short_abs + filled_abs
            new_avg_cost = (
                (existing_short_abs * avg_cost + filled_abs * fill_price) / new_short_abs
                if new_short_abs > 0 else 0.0
            )
            cash_flow = filled_abs * fill_price - commission
            cash += cash_flow
            pos.loc[ticker, "shares"] = current_shares - filled_abs
            pos.loc[ticker, "avg_cost"] = new_avg_cost
            append_execution(order, -filled_abs, fill_price, cash_flow, 0.0)
            return

    def raw_cash_needed(order: pd.Series) -> float:
        side = str(order["side"])
        requested_abs = abs(float(order["requested_shares"]))
        raw_price = float(order["raw_price"])
        fill_price = raw_price * (1.0 + slippage)
        return requested_abs * fill_price + float(PAPER_COMMISSION_PER_TRADE)

    def execute_cash_consuming(order: pd.Series, scale: float) -> None:
        nonlocal cash
        ticker = str(order["ticker"])
        side = str(order["side"])
        requested_abs = abs(float(order["requested_shares"])) * scale
        raw_price = float(order["raw_price"])
        fill_price = raw_price * (1.0 + slippage)
        commission = float(PAPER_COMMISSION_PER_TRADE)
        if requested_abs <= 0:
            return

        current_shares = float(pos.loc[ticker, "shares"])
        avg_cost = float(pos.loc[ticker, "avg_cost"])

        if side == "COVER":
            filled_abs = min(requested_abs, abs(min(current_shares, 0.0)))
            total_cost = filled_abs * fill_price + commission
            if not bool(ALLOW_NEGATIVE_CASH) and total_cost > cash + 1e-8:
                return
            realised_pnl = (avg_cost - fill_price) * filled_abs
            cash -= total_cost
            new_shares = current_shares + filled_abs
            if abs(new_shares) <= 1e-10:
                new_shares = 0.0
                new_avg_cost = 0.0
            else:
                new_avg_cost = avg_cost
            pos.loc[ticker, "shares"] = new_shares
            pos.loc[ticker, "avg_cost"] = new_avg_cost
            append_execution(order, filled_abs, fill_price, -total_cost, realised_pnl, scale < 0.999999)
            return

        if side == "BUY":
            filled_abs = requested_abs
            total_cost = filled_abs * fill_price + commission
            if not bool(ALLOW_NEGATIVE_CASH) and total_cost > cash + 1e-8:
                return
            new_shares = current_shares + filled_abs
            new_avg_cost = (
                fill_price if current_shares <= 0
                else (current_shares * avg_cost + filled_abs * fill_price) / new_shares
            )
            cash -= total_cost
            pos.loc[ticker, "shares"] = new_shares
            pos.loc[ticker, "avg_cost"] = new_avg_cost
            append_execution(order, filled_abs, fill_price, -total_cost, 0.0, scale < 0.999999)
            return

    # Cash-generating trades first: sell longs and open/increase shorts.
    for _, order in orders[orders["side"].isin(["SELL", "SHORT"])].iterrows():
        execute_cash_generating(order)

    consuming = orders[orders["side"].isin(["COVER", "BUY"])].copy()
    buy_scale = 1.0
    if not consuming.empty and not bool(ALLOW_NEGATIVE_CASH):
        raw_cost = sum(raw_cash_needed(order) for _, order in consuming.iterrows())
        if raw_cost > cash and raw_cost > 0:
            buy_scale = max(min(cash / raw_cost, 1.0), 0.0)

    for _, order in consuming.iterrows():
        execute_cash_consuming(order, buy_scale)

    new_positions = pos.reset_index().sort_values("ticker").reset_index(drop=True)
    executed = pd.DataFrame(executed_rows)
    expected_cols = list(_empty_orders_frame().columns)
    if executed.empty:
        executed = _empty_orders_frame()
    else:
        for col in expected_cols:
            if col not in executed.columns:
                executed[col] = np.nan
        executed = executed[expected_cols]
    return new_positions, float(cash), executed


# ============================================================
# SNAPSHOTS AND REPORTS
# ============================================================

def build_decision_snapshot(
    latest_weights: pd.DataFrame,
    final_scores: pd.DataFrame,
    marked_before: pd.DataFrame,
    orders: pd.DataFrame,
    signal_date: pd.Timestamp,
    accounting_date: pd.Timestamp,
    rebalance_due: bool,
) -> pd.DataFrame:
    decision = latest_weights.copy()
    position_cols = [
        "ticker", "shares", "avg_cost", "position_type", "cost_basis", "price_date", "price",
        "market_value", "abs_market_value", "current_weight", "abs_current_weight",
        "unrealised_pnl", "unrealised_return",
    ]
    decision = decision.merge(
        marked_before[[col for col in position_cols if col in marked_before.columns]],
        on="ticker",
        how="left",
        validate="one_to_one",
    )

    score_like_cols = [
        "final_score", "rank", "momentum_score", "relative_strength_score", "trend_score",
        "trend_persistence_score", "volatility_score", "risk_score", "macro_score",
        "signal_quality", "combined_stress_score", "commodity_model", "commodity_model_score",
        "commodity_model_version",
    ]
    for col in score_like_cols:
        if col in decision.columns and col in final_scores.columns:
            decision = decision.drop(columns=[col])
    if not final_scores.empty:
        score_cols = [col for col in ["ticker", *score_like_cols] if col in final_scores.columns]
        decision = decision.merge(final_scores[score_cols], on="ticker", how="left", validate="one_to_one")
    if "final_score" not in decision.columns:
        decision["final_score"] = np.nan
    if "rank" not in decision.columns:
        decision["rank"] = np.nan

    if orders.empty:
        trade_map = pd.DataFrame({"ticker": decision["ticker"], "trade_required": False, "side": "", "requested_shares": 0.0, "raw_notional": 0.0})
    else:
        trade_map = orders[["ticker", "side", "requested_shares", "raw_notional"]].copy()
        trade_map["trade_required"] = True
        trade_map = trade_map.groupby("ticker", as_index=False).agg(
            side=("side", lambda x: "/".join([str(v) for v in x if str(v)])),
            requested_shares=("requested_shares", "sum"),
            raw_notional=("raw_notional", "sum"),
            trade_required=("trade_required", "max"),
        )

    decision = decision.merge(trade_map, on="ticker", how="left")
    decision["trade_required"] = decision["trade_required"].fillna(False)
    decision["side"] = decision["side"].fillna("")
    decision["requested_shares"] = _safe_numeric(decision.get("requested_shares", pd.Series(0.0, index=decision.index)), 0.0)
    decision["raw_notional"] = _safe_numeric(decision.get("raw_notional", pd.Series(0.0, index=decision.index)), 0.0)
    decision["accounting_date"] = _date_str(accounting_date)
    decision["signal_date"] = _date_str(signal_date)
    decision["rebalance_due"] = bool(rebalance_due)
    decision["current_weight"] = _safe_numeric(decision.get("current_weight", pd.Series(0.0, index=decision.index)), 0.0)
    decision["target_weight"] = _safe_numeric(decision.get("target_weight", pd.Series(0.0, index=decision.index)), 0.0)
    decision["weight_gap"] = decision["target_weight"] - decision["current_weight"]
    decision["abs_target_weight"] = decision["target_weight"].abs()
    decision["abs_current_weight"] = decision["current_weight"].abs()
    decision["target_position_type"] = np.where(decision["target_weight"] > 0.001, "LONG", np.where(decision["target_weight"] < -0.001, "SHORT", "FLAT"))
    decision["status"] = np.where(
        decision["target_weight"] > 0.001,
        "LONG",
        np.where(decision["target_weight"] < -0.001, "SHORT", np.where(decision["final_score"].fillna(0.0) >= 0, "NO HOLD", "NO SCORE")),
    )
    return decision.sort_values(["abs_target_weight", "rank", "final_score"], ascending=[False, True, False]).reset_index(drop=True)


def save_snapshots(latest_weights: pd.DataFrame, final_scores: pd.DataFrame, decision: pd.DataFrame, signal_date: pd.Timestamp) -> None:
    stamp = signal_date.strftime(SNAPSHOT_DATE_FORMAT)
    _write_csv(latest_weights, PAPER_TARGET_WEIGHT_SNAPSHOT_DIR / f"target_weights_{stamp}.csv")
    if not final_scores.empty:
        _write_csv(final_scores, PAPER_FINAL_SCORE_SNAPSHOT_DIR / f"final_scores_{stamp}.csv")
    _write_csv(decision, PAPER_DECISION_SNAPSHOT_DIR / f"decision_{stamp}.csv")


def save_equity_row(row: dict[str, Any]) -> None:
    new_row = pd.DataFrame([row])
    existing = _read_csv_if_exists(PAPER_EQUITY_CURVE_PATH)
    if existing.empty:
        out = new_row
    else:
        if OVERWRITE_SAME_SIGNAL_DATE_EQUITY_ROW and "date" in existing.columns:
            existing = existing[existing["date"] != row["date"]].copy()
        out = pd.concat([existing, new_row], ignore_index=True)
    out = out.sort_values("date").reset_index(drop=True)
    _write_csv(out, PAPER_EQUITY_CURVE_PATH)


def save_latest_orders(executed_orders: pd.DataFrame) -> None:
    _write_csv(executed_orders, LATEST_ORDERS_PATH)


def previous_equity_from_file(accounting_date: pd.Timestamp) -> float | None:
    existing = _read_csv_if_exists(PAPER_EQUITY_CURVE_PATH)
    if existing.empty or "equity" not in existing.columns or "date" not in existing.columns:
        return None
    existing = existing.copy()
    existing["date"] = _to_datetime_series(existing["date"])
    existing["equity"] = _safe_numeric(existing["equity"], np.nan)
    existing = existing.dropna(subset=["date", "equity"]).copy()
    existing = existing[existing["date"] < pd.Timestamp(accounting_date).normalize()].copy()
    if existing.empty:
        return None
    existing = existing.sort_values("date")
    return float(existing.iloc[-1]["equity"])


def latest_rebalance_equity_from_file() -> float | None:
    existing = _read_csv_if_exists(PAPER_EQUITY_CURVE_PATH)
    if existing.empty or "equity" not in existing.columns:
        return None
    existing = existing.copy()
    existing["equity"] = _safe_numeric(existing["equity"], np.nan)
    if "action" in existing.columns:
        rb = existing[existing["action"].astype(str).str.contains("rebalance", case=False, na=False)].copy()
    elif "rebalance_due" in existing.columns:
        rb = existing[existing["rebalance_due"].astype(str).str.lower().isin(["true", "1"])].copy()
    else:
        rb = pd.DataFrame()
    if rb.empty:
        return None
    if "date" in rb.columns:
        rb["date"] = _to_datetime_series(rb["date"])
        rb = rb.dropna(subset=["date"]).sort_values("date")
    return float(rb.iloc[-1]["equity"])


def write_summary(
    *,
    accounting_date: pd.Timestamp,
    signal_date: pd.Timestamp,
    price_date_min: pd.Timestamp,
    price_date_max: pd.Timestamp,
    rebalance_due: bool,
    action: str,
    account_before: dict[str, float],
    account_after: dict[str, float],
    interest_earned: float,
    days_interest: int,
    executed_orders: pd.DataFrame,
    decision: pd.DataFrame,
    daily_pnl: float,
    cumulative_pnl: float,
    since_rebalance_pnl: float,
) -> None:
    lines: list[str] = []
    lines.append("COMMODITY SYSTEM PAPER TRADING SUMMARY")
    lines.append("=" * 52)
    lines.append("")
    lines.append(f"Accounting date:      {_date_str(accounting_date)}")
    lines.append(f"Signal date:          {_date_str(signal_date)}")
    lines.append(f"Price dates used:     {_date_str(price_date_min)} to {_date_str(price_date_max)}")
    lines.append(f"Rebalance due:        {rebalance_due}")
    lines.append(f"Action:               {action}")
    lines.append("")
    lines.append("ACCOUNT")
    lines.append("-" * 52)
    lines.append(f"Equity before:        ${account_before['equity']:,.2f}")
    lines.append(f"Equity after:         ${account_after['equity']:,.2f}")
    lines.append(f"Cash after:           ${account_after['cash']:,.2f} ({account_after['cash_weight']:.2%} actual cash/equity)")
    lines.append(f"Net invested after:   ${account_after['invested_value']:,.2f}")
    lines.append(f"Gross invested after: ${account_after['gross_invested_value']:,.2f}")
    lines.append(f"Long exposure after:  {account_after['long_exposure']:.2%}")
    lines.append(f"Short exposure after: {account_after['short_exposure']:.2%}")
    lines.append(f"Net exposure after:   {account_after['net_exposure']:.2%}")
    lines.append(f"Gross exposure after: {account_after['gross_exposure']:.2%}")
    lines.append(f"Cash buffer after:    {account_after['cash_buffer_weight']:.2%}")
    lines.append(f"Daily/check P&L:      ${daily_pnl:,.2f}")
    lines.append(f"Since rebalance P&L:  ${since_rebalance_pnl:,.2f}")
    lines.append(f"Cumulative P&L:       ${cumulative_pnl:,.2f}")
    lines.append(f"Interest earned:      ${interest_earned:,.4f} over {days_interest} day(s)")
    lines.append("")
    lines.append("TARGET ALLOCATION")
    lines.append("-" * 52)
    allocation_cols = ["ticker", "target_position_type", "target_weight", "current_weight", "weight_gap", "final_score", "rank", "side", "raw_notional"]
    visible_cols = [col for col in allocation_cols if col in decision.columns]
    allocation = decision[visible_cols].copy()
    for _, row in allocation.iterrows():
        ticker = row.get("ticker", "")
        target_type = str(row.get("target_position_type", ""))
        target = _safe_float(row.get("target_weight", 0.0))
        current = _safe_float(row.get("current_weight", 0.0))
        gap = _safe_float(row.get("weight_gap", 0.0))
        side = str(row.get("side", ""))
        notional = _safe_float(row.get("raw_notional", 0.0))
        score_text = f", score {_safe_float(row['final_score']):.3f}" if "final_score" in row and pd.notna(row["final_score"]) else ""
        rank_text = f", rank {_safe_float(row['rank']):.0f}" if "rank" in row and pd.notna(row["rank"]) else ""
        trade_text = f", {side} ${notional:,.2f}" if side else ""
        lines.append(f"{ticker:>4}: {target_type:>5} target {target:>+7.2%}, current {current:>+7.2%}, gap {gap:>+7.2%}{score_text}{rank_text}{trade_text}")
    lines.append("")
    lines.append("EXECUTED PAPER ORDERS")
    lines.append("-" * 52)
    if executed_orders.empty:
        lines.append("No orders executed.")
    else:
        for _, row in executed_orders.iterrows():
            lines.append(
                f"{row['side']:>5} {row['ticker']:>4} "
                f"{float(row['filled_shares']):>+,.6f} shares "
                f"@ ${float(row['fill_price']):,.4f} "
                f"(${float(row['filled_notional']):,.2f}), "
                f"cash flow ${float(row['cash_flow']):+,.2f}, "
                f"realised P&L ${float(row['realised_pnl']):+,.2f}"
            )
    lines.append("")
    LATEST_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# MAIN RUNNER
# ============================================================

def main() -> None:
    ensure_directories()
    run_timestamp_utc = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    accounting_date = pd.Timestamp.today().normalize()

    latest_weights, signal_date = load_latest_target_weights()
    final_scores = load_latest_final_scores(signal_date)
    latest_prices = load_latest_prices_asof(accounting_date)
    price_date_min = pd.Timestamp(latest_prices["price_date"].min()).normalize()
    price_date_max = pd.Timestamp(latest_prices["price_date"].max()).normalize()

    state = load_state()
    cash = _state_float(state, "cash", PAPER_INITIAL_CAPITAL)
    cash, interest_earned, days_interest = apply_cash_yield(cash=cash, state=state, accounting_date=accounting_date)

    positions_before = load_positions()
    marked_before, account_before = mark_positions_to_market(positions=positions_before, prices=latest_prices, cash=cash)
    previous_equity = previous_equity_from_file(accounting_date)
    initial_capital = _state_float(state, "initial_capital", PAPER_INITIAL_CAPITAL)
    rebalance_due, current_period = is_rebalance_due(signal_date=signal_date, accounting_date=accounting_date, state=state)

    if rebalance_due:
        requested_orders = generate_target_orders(
            latest_weights=latest_weights,
            marked_positions=marked_before,
            account_stats=account_before,
            signal_date=signal_date,
            accounting_date=accounting_date,
            run_timestamp_utc=run_timestamp_utc,
        )
        positions_after, cash_after, executed_orders = execute_orders(orders=requested_orders, positions=positions_before, cash=cash)
        action = "monthly_rebalance_executed" if not executed_orders.empty else "monthly_rebalance_due_no_orders"
        last_rebalance_period = current_period
    else:
        requested_orders = _empty_orders_frame()
        positions_after = positions_before.copy()
        cash_after = cash
        executed_orders = _empty_orders_frame()
        action = "mark_to_market_only"
        last_rebalance_period = state.get("last_rebalance_period", "")

    marked_after, account_after = mark_positions_to_market(positions=positions_after, prices=latest_prices, cash=cash_after)

    daily_pnl = account_after["equity"] - (initial_capital if previous_equity is None else previous_equity)
    cumulative_pnl = account_after["equity"] - initial_capital
    rebalance_anchor = latest_rebalance_equity_from_file()
    if rebalance_due or rebalance_anchor is None:
        since_rebalance_pnl = 0.0
        rebalance_anchor_equity = account_after["equity"]
    else:
        since_rebalance_pnl = account_after["equity"] - rebalance_anchor
        rebalance_anchor_equity = rebalance_anchor

    decision = build_decision_snapshot(
        latest_weights=latest_weights,
        final_scores=final_scores,
        marked_before=marked_before,
        orders=requested_orders,
        signal_date=signal_date,
        accounting_date=accounting_date,
        rebalance_due=rebalance_due,
    )

    save_positions(marked_after)
    save_latest_orders(executed_orders)
    save_snapshots(latest_weights=latest_weights, final_scores=final_scores, decision=decision, signal_date=signal_date)
    _append_csv(executed_orders, PAPER_TRADES_PATH)

    equity_row = {
        "run_timestamp_utc": run_timestamp_utc,
        "date": _date_str(accounting_date),
        "signal_date": _date_str(signal_date),
        "latest_price_date": _date_str(price_date_max),
        "equity": account_after["equity"],
        "cash": account_after["cash"],
        "invested_value": account_after["invested_value"],
        "gross_invested_value": account_after["gross_invested_value"],
        "long_value": account_after["long_value"],
        "short_value": account_after["short_value"],
        "total_exposure": account_after["total_exposure"],
        "net_exposure": account_after["net_exposure"],
        "gross_exposure": account_after["gross_exposure"],
        "long_exposure": account_after["long_exposure"],
        "short_exposure": account_after["short_exposure"],
        "cash_weight": account_after["cash_weight"],
        "cash_buffer_weight": account_after["cash_buffer_weight"],
        "daily_pnl": daily_pnl,
        "cumulative_pnl": cumulative_pnl,
        "since_rebalance_pnl": since_rebalance_pnl,
        "rebalance_anchor_equity": rebalance_anchor_equity,
        "unrealised_pnl": account_after.get("unrealised_pnl", 0.0),
        "cost_basis": account_after.get("cost_basis", 0.0),
        "abs_cost_basis": account_after.get("abs_cost_basis", 0.0),
        "interest_earned": interest_earned,
        "days_interest": days_interest,
        "trade_count": int(len(executed_orders)),
        "rebalance_due": bool(rebalance_due),
        "action": action,
    }
    save_equity_row(equity_row)

    state.update(
        {
            "cash": f"{account_after['cash']:.10f}",
            "last_run_date": _date_str(accounting_date),
            "last_signal_date": _date_str(signal_date),
            "last_rebalance_period": last_rebalance_period,
            "last_equity": f"{account_after['equity']:.10f}",
            "last_price_date": _date_str(price_date_max),
        }
    )
    save_state(state)

    write_summary(
        accounting_date=accounting_date,
        signal_date=signal_date,
        price_date_min=price_date_min,
        price_date_max=price_date_max,
        rebalance_due=rebalance_due,
        action=action,
        account_before=account_before,
        account_after=account_after,
        interest_earned=interest_earned,
        days_interest=days_interest,
        executed_orders=executed_orders,
        decision=decision,
        daily_pnl=daily_pnl,
        cumulative_pnl=cumulative_pnl,
        since_rebalance_pnl=since_rebalance_pnl,
    )

    print("\n" + "=" * 80)
    print("PAPER TRADING RUN COMPLETE")
    print("=" * 80)
    print(f"Accounting date:   {_date_str(accounting_date)}")
    print(f"Signal date:       {_date_str(signal_date)}")
    print(f"Latest price date: {_date_str(price_date_max)}")
    print(f"Action:            {action}")
    print(f"Rebalance due:     {rebalance_due}")
    print(f"Equity:            ${account_after['equity']:,.2f}")
    print(f"Cash:              ${account_after['cash']:,.2f}")
    print(f"Gross exposure:    {account_after['gross_exposure']:.2%}")
    print(f"Net exposure:      {account_after['net_exposure']:.2%}")
    print(f"Short exposure:    {account_after['short_exposure']:.2%}")
    print(f"Daily/check P&L:   ${daily_pnl:,.2f}")
    print(f"Cumulative P&L:    ${cumulative_pnl:,.2f}")
    print(f"Orders executed:   {len(executed_orders)}")
    print("")
    print("Outputs:")
    print(f"- {PAPER_POSITIONS_PATH}")
    print(f"- {PAPER_TRADES_PATH}")
    print(f"- {PAPER_EQUITY_CURVE_PATH}")
    print(f"- {LATEST_SUMMARY_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
