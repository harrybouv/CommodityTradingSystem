from __future__ import annotations

"""
benchmark_decomposition.py

Purpose
-------
A compact benchmark, alpha/beta and return-source attribution layer for the
commodity allocation system.

Place this file in:
    Commodity_System/research/Backtesting/benchmark_decomposition.py

Run from that folder or from the project root:
    python benchmark_decomposition.py

Design choices
--------------
- Reuses backtest_V2.py's execution engine.
- Does NOT change production strategy logic.
- Keeps the benchmark set deliberately small.
- Uses cash-adjusted alpha/beta, not the old 0% risk-free assumption.
- Saves a few useful CSVs and charts, not a giant diagnostics dump.

Main outputs
------------
results/benchmark_decomposition/
    benchmark_performance_summary.csv
    alpha_beta_matrix.csv
    return_decomposition_summary.csv
    asset_contribution_decomposition.csv
    benchmark_decomposition_notes.txt
    charts/
        equity_vs_benchmarks.png
        drawdown_vs_benchmarks.png
        benchmark_cagr_bar.png
        model_return_decomposition.png
        model_asset_contribution.png
"""

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# ============================================================
# PATH SETUP
# ============================================================

THIS_FILE = Path(__file__).resolve()
BACKTESTING_DIR = THIS_FILE.parent
RESEARCH_DIR = BACKTESTING_DIR.parent
COMMODITY_ROOT = RESEARCH_DIR.parent
PROJECT_ROOT = COMMODITY_ROOT.parent

for path in [PROJECT_ROOT, COMMODITY_ROOT, RESEARCH_DIR, BACKTESTING_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ============================================================
# IMPORT EXISTING ENGINE
# ============================================================

try:
    import backtest_V2 as V2
except ImportError as exc:
    raise ImportError(
        "Could not import backtest_V2.py. Put benchmark_decomposition.py next to "
        "backtest_V2.py inside Commodity_System/research/Backtesting."
    ) from exc


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = V2.RESULTS_DIR / "benchmark_decomposition"
CHART_DIR = OUTPUT_DIR / "charts"

PERIODS_PER_YEAR = int(getattr(V2, "TRADING_DAYS_PER_YEAR", 252))
INITIAL_CAPITAL = float(getattr(V2, "INITIAL_CAPITAL", 10_000))
CASH_RATE = float(getattr(V2, "CASH_ANNUAL_YIELD", 0.04))

# Keep the benchmark set deliberately focused.
RUN_NO_GOLD_DIAGNOSTIC = True

INVERSE_VOL_LOOKBACK = 60
TSMOM_TREND_WINDOW = 200
TSMOM_VOL_LOOKBACK = 60

CORE_EQUITY_CHART_STRATEGIES = [
    "model",
    "equal_weight",
    "inverse_vol",
    "trend_following",
    "gold_only",
]

ALPHA_BETA_BENCHMARKS = [
    "equal_weight",
    "inverse_vol",
    "trend_following",
    "gold_only",
    "cash",
    "no_short_model",
    "zero_cash_yield_model",
]

if RUN_NO_GOLD_DIAGNOSTIC:
    ALPHA_BETA_BENCHMARKS.append("no_gold_model")


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)


def clean_series(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out.astype(float)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def annual_to_daily_rate(annual_rate: float, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    return (1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0


def total_return(returns: pd.Series) -> float:
    r = clean_series(returns)
    if r.empty:
        return np.nan
    return float((1.0 + r).prod() - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    r = clean_series(returns)
    if r.empty:
        return np.nan

    tr = total_return(r)
    years = len(r) / periods_per_year

    if years <= 0 or (1.0 + tr) <= 0:
        return np.nan

    return float((1.0 + tr) ** (1.0 / years) - 1.0)


def ann_vol(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    r = clean_series(returns)
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * math.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    r = clean_series(returns)
    if r.empty:
        return np.nan
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    r = clean_series(returns)
    if r.empty:
        return pd.Series(dtype=float)
    equity = (1.0 + r).cumprod()
    return equity / equity.cummax() - 1.0


def sharpe(returns: pd.Series, risk_free_rate: float = CASH_RATE) -> float:
    r = clean_series(returns)
    if len(r) < 2:
        return np.nan

    rf_daily = annual_to_daily_rate(risk_free_rate)
    excess = r - rf_daily
    vol = excess.std(ddof=1)

    if vol <= 1e-12:
        # Cash-like series: if it earns the risk-free rate, Sharpe is not meaningful.
        return 0.0 if abs(excess.mean()) < 1e-12 else np.nan

    return float(excess.mean() / vol * math.sqrt(PERIODS_PER_YEAR))


def sortino(returns: pd.Series, risk_free_rate: float = CASH_RATE) -> float:
    r = clean_series(returns)
    if len(r) < 2:
        return np.nan

    rf_daily = annual_to_daily_rate(risk_free_rate)
    excess = r - rf_daily
    downside = excess[excess < 0]

    if len(downside) < 2:
        return np.nan

    downside_vol = downside.std(ddof=1)
    if downside_vol <= 1e-12:
        return np.nan

    return float(excess.mean() / downside_vol * math.sqrt(PERIODS_PER_YEAR))


def calmar(returns: pd.Series) -> float:
    ann = cagr(returns)
    dd = max_drawdown(returns)
    if pd.isna(ann) or pd.isna(dd) or dd >= 0:
        return np.nan
    return float(ann / abs(dd))


def safe_divide(a: float, b: float) -> float:
    if b is None or pd.isna(b) or abs(b) <= 1e-12:
        return np.nan
    return float(a / b)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_indexed_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def get_curve(result: dict[str, pd.DataFrame]) -> pd.DataFrame:
    curve = result["curve"].copy()
    curve.index = pd.to_datetime(curve.index)
    return curve.sort_index()


def get_returns(result: dict[str, pd.DataFrame]) -> pd.Series:
    curve = get_curve(result)
    out = curve["net_return"].copy()
    out.name = "net_return"
    return clean_series(out)


def get_weight_matrix(result: dict[str, pd.DataFrame]) -> pd.DataFrame:
    weights = result["executed_weights"].copy()
    weights.index = pd.to_datetime(weights.index)

    if "strategy" in weights.columns:
        weights = weights.drop(columns=["strategy"])

    return clean_frame(weights.sort_index()).fillna(0.0)


def align_returns_matrix(asset_returns: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    out = asset_returns.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out.reindex(index=weights.index, columns=weights.columns).fillna(0.0)
    return clean_frame(out)


# ============================================================
# BENCHMARK WEIGHT BUILDERS
# ============================================================

def normalise_rows_to_unit_gross(weights: pd.DataFrame) -> pd.DataFrame:
    """
    Normalises each row so absolute weights sum to 1.
    Zero rows remain zero.
    """
    out = weights.copy().astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = out.abs().sum(axis=1)
    non_zero = gross > 0
    out.loc[non_zero] = out.loc[non_zero].div(gross.loc[non_zero], axis=0)
    out.loc[~non_zero] = 0.0
    return out.fillna(0.0)


def make_inverse_vol_weights(
    returns: pd.DataFrame,
    tickers: list[str],
    lookback: int = INVERSE_VOL_LOOKBACK,
) -> pd.DataFrame:
    """
    Long-only inverse-vol commodity basket.

    This is a stronger naive benchmark than equal weight because it controls
    basic volatility concentration without using your strategy signals.
    """
    r = returns.reindex(columns=tickers).fillna(0.0)
    vol = r.rolling(lookback, min_periods=max(20, lookback // 2)).std()
    inv_vol = 1.0 / vol.replace(0.0, np.nan)

    # Shift by one day so the signal does not use same-day return information.
    raw = inv_vol.shift(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = normalise_rows_to_unit_gross(raw)

    # Before vol history exists, fall back to equal weight.
    early = weights.abs().sum(axis=1) == 0
    if early.any():
        weights.loc[early, tickers] = 1.0 / len(tickers)

    return weights.sort_index().fillna(0.0)


def make_trend_following_weights(
    close: pd.DataFrame,
    returns: pd.DataFrame,
    tickers: list[str],
    trend_window: int = TSMOM_TREND_WINDOW,
    vol_lookback: int = TSMOM_VOL_LOOKBACK,
) -> pd.DataFrame:
    """
    Simple CTA-style time-series momentum baseline.

    Rule:
      - Long asset if prior close is above prior 200d moving average.
      - Active assets are inverse-vol weighted.
      - If no assets are active, hold cash.

    This is intentionally simple. The point is not to create a new strategy;
    it is to test whether your production system beats a basic trend-following
    commodity benchmark.
    """
    px = close.reindex(columns=tickers).sort_index().ffill()
    r = returns.reindex(index=px.index, columns=tickers).fillna(0.0)

    ma = px.rolling(trend_window, min_periods=max(60, trend_window // 2)).mean()
    active = (px > ma).astype(float).shift(1).fillna(0.0)

    vol = r.rolling(vol_lookback, min_periods=max(20, vol_lookback // 2)).std()
    inv_vol = (1.0 / vol.replace(0.0, np.nan)).shift(1)

    raw = active * inv_vol
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = normalise_rows_to_unit_gross(raw)

    return weights.sort_index().fillna(0.0)


def make_no_short_weights(model_weights: pd.DataFrame) -> pd.DataFrame:
    """
    Removes short positions without redistributing their gross notional.

    This is the fairest diagnostic because it tests whether the short sleeve adds
    value without giving the long-only variant extra capital that the production
    model did not actually allocate to longs.
    """
    return model_weights.clip(lower=0.0).fillna(0.0)


def make_no_gold_weights(model_weights: pd.DataFrame) -> pd.DataFrame:
    """
    Removes GLD without redistributing the freed allocation.

    This tests gold dependency conservatively. The removed allocation becomes
    cash through the existing backtest engine.
    """
    out = model_weights.copy().fillna(0.0)
    if "GLD" in out.columns:
        out["GLD"] = 0.0
    return out


# ============================================================
# SIMULATION WRAPPERS
# ============================================================

def simulate_with_cash_rate(
    *,
    name: str,
    raw_target_weights: pd.DataFrame,
    market_data: dict[str, pd.DataFrame],
    settings: dict[str, Any],
    cash_rate: float,
) -> dict[str, pd.DataFrame]:
    """
    Runs V2's execution engine while temporarily overriding the module-level
    cash yield used inside simulate_strategy_v2().
    """
    old_cash_rate = V2.CASH_ANNUAL_YIELD

    try:
        V2.CASH_ANNUAL_YIELD = float(cash_rate)
        result = V2.simulate_strategy_v2(
            name=name,
            raw_target_weights=raw_target_weights,
            market_data=market_data,
            settings=settings,
            initial_capital=INITIAL_CAPITAL,
        )
    finally:
        V2.CASH_ANNUAL_YIELD = old_cash_rate

    return result


def run_strategy_suite() -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, pd.DataFrame], pd.DataFrame]:
    settings = V2.build_base_settings()
    market_data = V2.load_market_data(settings=settings)
    returns = market_data["returns"].sort_index().fillna(0.0)
    close = market_data["close"].sort_index().ffill()

    model_weights = V2.load_target_weights()
    model_weights.index = pd.to_datetime(model_weights.index)
    model_weights = model_weights.sort_index().fillna(0.0)

    tickers = list(model_weights.columns)
    model_weights = model_weights.reindex(columns=tickers).fillna(0.0)

    equal_weight = V2.make_equal_weight(index=model_weights.index, tickers=tickers)
    gold_only = V2.make_gold_only(index=model_weights.index, tickers=tickers)
    cash = V2.make_cash(index=model_weights.index, tickers=tickers)

    inverse_vol = make_inverse_vol_weights(
        returns=returns,
        tickers=tickers,
        lookback=INVERSE_VOL_LOOKBACK,
    )

    trend_following = make_trend_following_weights(
        close=close,
        returns=returns,
        tickers=tickers,
        trend_window=TSMOM_TREND_WINDOW,
        vol_lookback=TSMOM_VOL_LOOKBACK,
    )

    no_short_model = make_no_short_weights(model_weights)

    strategies: dict[str, pd.DataFrame] = {
        "model": model_weights,
        "equal_weight": equal_weight,
        "inverse_vol": inverse_vol,
        "trend_following": trend_following,
        "gold_only": gold_only,
        "cash": cash,
        "no_short_model": no_short_model,
        "zero_cash_yield_model": model_weights,
    }

    if RUN_NO_GOLD_DIAGNOSTIC:
        strategies["no_gold_model"] = make_no_gold_weights(model_weights)

    results: dict[str, dict[str, pd.DataFrame]] = {}

    print("\n========== BENCHMARK + DECOMPOSITION RUN ==========")
    print(f"Cash/risk-free rate used for reporting: {CASH_RATE:.2%}")
    print(f"Output folder: {OUTPUT_DIR}")

    for name, weights in strategies.items():
        print(f"\nRunning strategy/benchmark: {name}")

        cash_rate = 0.0 if name == "zero_cash_yield_model" else CASH_RATE

        result = simulate_with_cash_rate(
            name=name,
            raw_target_weights=weights,
            market_data=market_data,
            settings=settings,
            cash_rate=cash_rate,
        )

        results[name] = result

    return results, market_data, model_weights


# ============================================================
# PERFORMANCE + ALPHA/BETA
# ============================================================

def strategy_summary_row(name: str, result: dict[str, pd.DataFrame]) -> dict[str, Any]:
    curve = get_curve(result)
    returns = clean_series(curve["net_return"])
    years = len(returns) / PERIODS_PER_YEAR if len(returns) else np.nan

    total_cost_drag = float(curve.get("total_transaction_cost_drag", pd.Series(0.0, index=curve.index)).sum())
    annualised_cost_drag = safe_divide(total_cost_drag, years)

    avg_cash = float(curve.get("cash_weight", pd.Series(np.nan, index=curve.index)).mean())
    avg_gross = float(curve.get("gross_exposure", pd.Series(np.nan, index=curve.index)).mean())
    avg_net = float(curve.get("net_exposure", pd.Series(np.nan, index=curve.index)).mean())
    avg_short = float(curve.get("short_exposure", pd.Series(0.0, index=curve.index)).mean())
    avg_turnover = float(curve.get("turnover", pd.Series(0.0, index=curve.index)).mean())
    annualised_turnover = avg_turnover * PERIODS_PER_YEAR

    dd = max_drawdown(returns)

    return {
        "strategy": name,
        "start_date": returns.index.min().date() if len(returns) else None,
        "end_date": returns.index.max().date() if len(returns) else None,
        "observations": len(returns),
        "total_return": total_return(returns),
        "cagr": cagr(returns),
        "annualised_volatility": ann_vol(returns),
        "sharpe_cash_adjusted": sharpe(returns, risk_free_rate=CASH_RATE),
        "sortino_cash_adjusted": sortino(returns, risk_free_rate=CASH_RATE),
        "max_drawdown": dd,
        "calmar": calmar(returns),
        "average_cash": avg_cash,
        "average_gross_exposure": avg_gross,
        "average_net_exposure": avg_net,
        "average_short_exposure": avg_short,
        "annualised_turnover": annualised_turnover,
        "total_cost_drag": total_cost_drag,
        "annualised_cost_drag": annualised_cost_drag,
    }


def build_performance_summary(results: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = [strategy_summary_row(name, result) for name, result in results.items()]
    out = pd.DataFrame(rows)

    if "model" in out["strategy"].values:
        model = out.set_index("strategy").loc["model"]
        for metric in ["cagr", "sharpe_cash_adjusted", "max_drawdown", "annualised_volatility"]:
            out[f"delta_{metric}_vs_model"] = out[metric] - model[metric]

    sort_order = [
        "model",
        "equal_weight",
        "inverse_vol",
        "trend_following",
        "gold_only",
        "cash",
        "no_short_model",
        "zero_cash_yield_model",
        "no_gold_model",
    ]
    rank = {name: i for i, name in enumerate(sort_order)}
    out["_rank"] = out["strategy"].map(rank).fillna(999)
    out = out.sort_values("_rank").drop(columns=["_rank"])

    return out


def alpha_beta_stats(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = CASH_RATE,
) -> dict[str, float]:
    df = pd.concat(
        [clean_series(strategy_returns).rename("strategy"), clean_series(benchmark_returns).rename("benchmark")],
        axis=1,
    ).dropna()

    if len(df) < 3:
        return {
            "alpha_daily": np.nan,
            "alpha_ann_arithmetic": np.nan,
            "alpha_ann_compounded": np.nan,
            "beta": np.nan,
            "r_squared": np.nan,
            "correlation": np.nan,
            "tracking_error": np.nan,
            "information_ratio": np.nan,
            "alpha_t_stat": np.nan,
            "beta_t_stat": np.nan,
        }

    rf_daily = annual_to_daily_rate(risk_free_rate)
    y = df["strategy"] - rf_daily
    x = df["benchmark"] - rf_daily

    if x.var(ddof=1) <= 1e-16:
        # A cash benchmark has near-zero excess-return variance. Regression beta/alpha
        # is undefined. Excess CAGR is still reported elsewhere.
        return {
            "alpha_daily": np.nan,
            "alpha_ann_arithmetic": np.nan,
            "alpha_ann_compounded": np.nan,
            "beta": np.nan,
            "r_squared": np.nan,
            "correlation": np.nan,
            "tracking_error": np.nan,
            "information_ratio": np.nan,
            "alpha_t_stat": np.nan,
            "beta_t_stat": np.nan,
        }

    beta = float(y.cov(x) / x.var(ddof=1))
    alpha_daily = float(y.mean() - beta * x.mean())
    alpha_ann_arithmetic = alpha_daily * PERIODS_PER_YEAR
    alpha_ann_compounded = (1.0 + alpha_daily) ** PERIODS_PER_YEAR - 1.0

    y_hat = alpha_daily + beta * x
    residual = y - y_hat

    sse = float((residual ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - sse / sst if abs(sst) > 1e-16 else np.nan
    correlation = float(y.corr(x))

    tracking_error = float(residual.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR))
    information_ratio = safe_divide(alpha_ann_arithmetic, tracking_error)

    n = len(df)
    x_centered_sum_sq = float(((x - x.mean()) ** 2).sum())

    if n > 2 and x_centered_sum_sq > 1e-16:
        residual_variance = sse / (n - 2)
        se_beta = math.sqrt(residual_variance / x_centered_sum_sq)
        se_alpha = math.sqrt(
            residual_variance * (1.0 / n + (x.mean() ** 2) / x_centered_sum_sq)
        )
        beta_t_stat = safe_divide(beta, se_beta)
        alpha_t_stat = safe_divide(alpha_daily, se_alpha)
    else:
        alpha_t_stat = np.nan
        beta_t_stat = np.nan

    return {
        "alpha_daily": alpha_daily,
        "alpha_ann_arithmetic": alpha_ann_arithmetic,
        "alpha_ann_compounded": alpha_ann_compounded,
        "beta": beta,
        "r_squared": r_squared,
        "correlation": correlation,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "alpha_t_stat": alpha_t_stat,
        "beta_t_stat": beta_t_stat,
    }


def build_alpha_beta_matrix(
    results: dict[str, dict[str, pd.DataFrame]],
    performance_summary: pd.DataFrame,
) -> pd.DataFrame:
    if "model" not in results:
        return pd.DataFrame()

    perf = performance_summary.set_index("strategy")
    model_returns = get_returns(results["model"])
    rows = []

    for benchmark in ALPHA_BETA_BENCHMARKS:
        if benchmark not in results or benchmark == "model":
            continue

        benchmark_returns = get_returns(results[benchmark])
        stats = alpha_beta_stats(
            strategy_returns=model_returns,
            benchmark_returns=benchmark_returns,
            risk_free_rate=CASH_RATE,
        )

        strategy_cagr = float(perf.loc["model", "cagr"])
        benchmark_cagr = float(perf.loc[benchmark, "cagr"])

        rows.append(
            {
                "strategy": "model",
                "benchmark": benchmark,
                "risk_free_rate_used": CASH_RATE,
                "strategy_cagr": strategy_cagr,
                "benchmark_cagr": benchmark_cagr,
                "excess_cagr": strategy_cagr - benchmark_cagr,
                "strategy_sharpe_cash_adjusted": float(perf.loc["model", "sharpe_cash_adjusted"]),
                "benchmark_sharpe_cash_adjusted": float(perf.loc[benchmark, "sharpe_cash_adjusted"]),
                "sharpe_difference": float(perf.loc["model", "sharpe_cash_adjusted"] - perf.loc[benchmark, "sharpe_cash_adjusted"]),
                "strategy_max_drawdown": float(perf.loc["model", "max_drawdown"]),
                "benchmark_max_drawdown": float(perf.loc[benchmark, "max_drawdown"]),
                "max_drawdown_improvement": float(perf.loc[benchmark, "max_drawdown"] - perf.loc["model", "max_drawdown"]),
                **stats,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# RETURN DECOMPOSITION
# ============================================================

def decompose_strategy_returns(
    name: str,
    result: dict[str, pd.DataFrame],
    asset_returns: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Decomposes realised daily returns into approximate arithmetic contribution.

    Important:
    - This is an additive attribution of daily returns, not compounded wealth attribution.
    - The engine's exact net return is still taken from curve['net_return'].
    - Long/short asset contribution uses prior end-of-day executed weights, matching how
      the engine carries weights into the next trading day.
    """
    curve = get_curve(result)
    weights = get_weight_matrix(result)
    held_weights = weights.shift(1).fillna(0.0)
    aligned_asset_returns = align_returns_matrix(asset_returns, held_weights)

    common_cols = held_weights.columns.intersection(aligned_asset_returns.columns)
    held_weights = held_weights[common_cols]
    aligned_asset_returns = aligned_asset_returns[common_cols]

    long_weights = held_weights.clip(lower=0.0)
    short_weights = held_weights.clip(upper=0.0)

    long_daily = (long_weights * aligned_asset_returns).sum(axis=1)
    short_daily = (short_weights * aligned_asset_returns).sum(axis=1)
    asset_daily = long_daily + short_daily

    cash_daily = curve["cash_return"].reindex(asset_daily.index).fillna(0.0)
    cost_daily = curve["total_transaction_cost_drag"].reindex(asset_daily.index).fillna(0.0)
    net_daily = curve["net_return"].reindex(asset_daily.index).fillna(0.0)

    explained_daily = asset_daily + cash_daily - cost_daily
    residual_daily = net_daily - explained_daily

    years = len(net_daily) / PERIODS_PER_YEAR if len(net_daily) else np.nan

    summary = {
        "strategy": name,
        "total_return_compounded": total_return(net_daily),
        "net_return_arithmetic_sum": float(net_daily.sum()),
        "long_asset_contribution_sum": float(long_daily.sum()),
        "short_asset_contribution_sum": float(short_daily.sum()),
        "gross_asset_contribution_sum": float(asset_daily.sum()),
        "cash_contribution_sum": float(cash_daily.sum()),
        "transaction_cost_drag_sum": float(cost_daily.sum()),
        "residual_sum": float(residual_daily.sum()),
        "long_asset_contribution_ann": safe_divide(float(long_daily.sum()), years),
        "short_asset_contribution_ann": safe_divide(float(short_daily.sum()), years),
        "cash_contribution_ann": safe_divide(float(cash_daily.sum()), years),
        "transaction_cost_drag_ann": safe_divide(float(cost_daily.sum()), years),
        "residual_ann": safe_divide(float(residual_daily.sum()), years),
    }

    asset_rows = []

    for ticker in common_cols:
        lw = long_weights[ticker]
        sw = short_weights[ticker]
        ar = aligned_asset_returns[ticker]

        long_contrib = float((lw * ar).sum())
        short_contrib = float((sw * ar).sum())
        net_contrib = long_contrib + short_contrib

        asset_rows.append(
            {
                "strategy": name,
                "ticker": ticker,
                "long_contribution_sum": long_contrib,
                "short_contribution_sum": short_contrib,
                "net_asset_contribution_sum": net_contrib,
                "annualised_net_asset_contribution": safe_divide(net_contrib, years),
                "mean_weight": float(held_weights[ticker].mean()),
                "mean_abs_weight": float(held_weights[ticker].abs().mean()),
                "mean_long_weight": float(lw.mean()),
                "mean_short_weight": float(sw.mean()),
                "max_long_weight": float(lw.max()),
                "max_short_weight": float(sw.min()),
            }
        )

    asset_table = pd.DataFrame(asset_rows)

    if not asset_table.empty:
        total_abs = asset_table["net_asset_contribution_sum"].abs().sum()
        if total_abs > 1e-12:
            asset_table["share_of_abs_net_asset_contribution"] = (
                asset_table["net_asset_contribution_sum"].abs() / total_abs
            )
        else:
            asset_table["share_of_abs_net_asset_contribution"] = np.nan

        asset_table = asset_table.sort_values("net_asset_contribution_sum", ascending=False)

    return summary, asset_table


def build_decomposition_tables(
    results: dict[str, dict[str, pd.DataFrame]],
    market_data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    asset_returns = market_data["returns"].sort_index().fillna(0.0)

    summary_rows = []
    model_asset_table = pd.DataFrame()

    # Decompose every strategy at summary level, but only save asset-level detail for model.
    for name, result in results.items():
        summary, asset_table = decompose_strategy_returns(
            name=name,
            result=result,
            asset_returns=asset_returns,
        )
        summary_rows.append(summary)

        if name == "model":
            model_asset_table = asset_table.copy()

    return pd.DataFrame(summary_rows), model_asset_table


# ============================================================
# CHARTS
# ============================================================

def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_equity_vs_benchmarks(results: dict[str, dict[str, pd.DataFrame]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    for name in CORE_EQUITY_CHART_STRATEGIES:
        if name not in results:
            continue
        curve = get_curve(results[name])
        ax.plot(curve.index, curve["equity"], label=name)

    ax.set_title("Equity curve vs core benchmarks")
    ax.set_ylabel("Portfolio value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, CHART_DIR / "equity_vs_benchmarks.png")


def plot_drawdown_vs_benchmarks(results: dict[str, dict[str, pd.DataFrame]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    for name in CORE_EQUITY_CHART_STRATEGIES:
        if name not in results:
            continue
        r = get_returns(results[name])
        dd = drawdown_series(r)
        ax.plot(dd.index, dd, label=name)

    ax.set_title("Drawdown vs core benchmarks")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, CHART_DIR / "drawdown_vs_benchmarks.png")


def plot_cagr_bar(performance_summary: pd.DataFrame) -> None:
    plot_df = performance_summary.copy()
    plot_df = plot_df.sort_values("cagr", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["strategy"], plot_df["cagr"])
    ax.set_title("CAGR by strategy / benchmark")
    ax.set_xlabel("CAGR")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis="x", alpha=0.3)
    save_figure(fig, CHART_DIR / "benchmark_cagr_bar.png")


def plot_model_return_decomposition(decomposition_summary: pd.DataFrame) -> None:
    if decomposition_summary.empty:
        return

    model = decomposition_summary[decomposition_summary["strategy"] == "model"]
    if model.empty:
        return

    row = model.iloc[0]
    pieces = pd.Series(
        {
            "Long assets": row["long_asset_contribution_ann"],
            "Short assets": row["short_asset_contribution_ann"],
            "Cash": row["cash_contribution_ann"],
            "Costs": -row["transaction_cost_drag_ann"],
            "Residual": row["residual_ann"],
        }
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(pieces.index, pieces.values)
    ax.set_title("Model annualised arithmetic return decomposition")
    ax.set_ylabel("Annualised contribution")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, CHART_DIR / "model_return_decomposition.png")


def plot_model_asset_contribution(asset_contribution: pd.DataFrame) -> None:
    if asset_contribution.empty:
        return

    plot_df = asset_contribution.sort_values("net_asset_contribution_sum", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot_df["ticker"], plot_df["annualised_net_asset_contribution"])
    ax.set_title("Model asset contribution")
    ax.set_xlabel("Annualised arithmetic contribution")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis="x", alpha=0.3)
    save_figure(fig, CHART_DIR / "model_asset_contribution.png")


def generate_charts(
    results: dict[str, dict[str, pd.DataFrame]],
    performance_summary: pd.DataFrame,
    decomposition_summary: pd.DataFrame,
    asset_contribution: pd.DataFrame,
) -> None:
    plot_equity_vs_benchmarks(results)
    plot_drawdown_vs_benchmarks(results)
    plot_cagr_bar(performance_summary)
    plot_model_return_decomposition(decomposition_summary)
    plot_model_asset_contribution(asset_contribution)


# ============================================================
# NOTES / CONSOLE SUMMARY
# ============================================================

def format_pct(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:.2%}"


def format_num(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:.2f}"


def build_notes(
    performance_summary: pd.DataFrame,
    alpha_beta_matrix: pd.DataFrame,
    decomposition_summary: pd.DataFrame,
    asset_contribution: pd.DataFrame,
) -> str:
    perf = performance_summary.set_index("strategy")
    lines = []

    lines.append("BENCHMARK + DECOMPOSITION NOTES")
    lines.append("=" * 38)
    lines.append("")
    lines.append(f"Risk-free/cash rate used in Sharpe and alpha/beta: {CASH_RATE:.2%}")
    lines.append("")

    if "model" in perf.index:
        model = perf.loc["model"]
        lines.append("Model headline:")
        lines.append(f"  CAGR: {format_pct(model['cagr'])}")
        lines.append(f"  Cash-adjusted Sharpe: {format_num(model['sharpe_cash_adjusted'])}")
        lines.append(f"  Max drawdown: {format_pct(model['max_drawdown'])}")
        lines.append(f"  Average gross exposure: {format_pct(model['average_gross_exposure'])}")
        lines.append(f"  Average short exposure: {format_pct(model['average_short_exposure'])}")
        lines.append(f"  Average cash: {format_pct(model['average_cash'])}")
        lines.append("")

    if not alpha_beta_matrix.empty:
        lines.append("Alpha/beta interpretation:")
        lines.append("  Alpha is cash-adjusted regression intercept against the named benchmark.")
        lines.append("  It is NOT standalone 'pure alpha'. Treat it as benchmark-relative residual return.")
        lines.append("")

        display_cols = [
            "benchmark",
            "excess_cagr",
            "alpha_ann_arithmetic",
            "beta",
            "r_squared",
            "information_ratio",
        ]
        lines.append("Key alpha/beta rows:")
        for _, row in alpha_beta_matrix[display_cols].iterrows():
            lines.append(
                "  vs "
                f"{row['benchmark']}: "
                f"excess CAGR {format_pct(row['excess_cagr'])}, "
                f"alpha {format_pct(row['alpha_ann_arithmetic'])}, "
                f"beta {format_num(row['beta'])}, "
                f"R2 {format_num(row['r_squared'])}, "
                f"IR {format_num(row['information_ratio'])}"
            )
        lines.append("")

    if not decomposition_summary.empty:
        model_decomp = decomposition_summary[decomposition_summary["strategy"] == "model"]
        if not model_decomp.empty:
            row = model_decomp.iloc[0]
            lines.append("Model return-source decomposition, annualised arithmetic contribution:")
            lines.append(f"  Long assets: {format_pct(row['long_asset_contribution_ann'])}")
            lines.append(f"  Short assets: {format_pct(row['short_asset_contribution_ann'])}")
            lines.append(f"  Cash: {format_pct(row['cash_contribution_ann'])}")
            lines.append(f"  Costs: -{format_pct(row['transaction_cost_drag_ann'])}")
            lines.append(f"  Residual: {format_pct(row['residual_ann'])}")
            lines.append("")

    if not asset_contribution.empty:
        top_asset = asset_contribution.iloc[0]
        lines.append("Largest model asset contribution:")
        lines.append(
            f"  {top_asset['ticker']}: "
            f"{format_pct(top_asset['annualised_net_asset_contribution'])} annualised arithmetic contribution"
        )
        lines.append("")

    lines.append("Files saved:")
    lines.append("  benchmark_performance_summary.csv")
    lines.append("  alpha_beta_matrix.csv")
    lines.append("  return_decomposition_summary.csv")
    lines.append("  asset_contribution_decomposition.csv")
    lines.append("  charts/*.png")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_dirs()

    results, market_data, _model_weights = run_strategy_suite()

    performance_summary = build_performance_summary(results)
    alpha_beta_matrix = build_alpha_beta_matrix(results, performance_summary)
    decomposition_summary, asset_contribution = build_decomposition_tables(results, market_data)

    save_csv(performance_summary, OUTPUT_DIR / "benchmark_performance_summary.csv")
    save_csv(alpha_beta_matrix, OUTPUT_DIR / "alpha_beta_matrix.csv")
    save_csv(decomposition_summary, OUTPUT_DIR / "return_decomposition_summary.csv")
    save_csv(asset_contribution, OUTPUT_DIR / "asset_contribution_decomposition.csv")

    generate_charts(
        results=results,
        performance_summary=performance_summary,
        decomposition_summary=decomposition_summary,
        asset_contribution=asset_contribution,
    )

    notes = build_notes(
        performance_summary=performance_summary,
        alpha_beta_matrix=alpha_beta_matrix,
        decomposition_summary=decomposition_summary,
        asset_contribution=asset_contribution,
    )

    notes_path = OUTPUT_DIR / "benchmark_decomposition_notes.txt"
    notes_path.write_text(notes, encoding="utf-8")

    print("\n" + notes)
    print(f"\nSaved benchmark/decomposition package to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
