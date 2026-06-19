from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


THIS_FILE = Path(__file__).resolve()
RESEARCH_DIR = THIS_FILE.parent
COMMODITY_DIR = THIS_FILE.parents[1]

PATHS_TO_ADD = [
    RESEARCH_DIR,
    COMMODITY_DIR,
]

if len(THIS_FILE.parents) >= 3:
    PATHS_TO_ADD.append(THIS_FILE.parents[2])

for path in PATHS_TO_ADD:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


try:
    from Commodity_System.config import (
        RESULTS_DIR,
        INITIAL_CAPITAL,
        TRADING_DAYS_PER_YEAR,
        TOTAL_COST_BPS,
        BACKTEST_REBALANCE_MODE,
    )
except ModuleNotFoundError:
    from config import (
        RESULTS_DIR,
        INITIAL_CAPITAL,
        TRADING_DAYS_PER_YEAR,
        TOTAL_COST_BPS,
        BACKTEST_REBALANCE_MODE,
    )

from analytics import (
    calculate_full_summary,
    calculate_drawdown_series,
    calculate_asset_contribution,
)

from backtester import (
    load_return_matrix,
    load_target_weights,
    apply_rebalance,
    make_equal_weight,
    make_gold_only,
    simulate_strategy,
)


OUTPUT_DIR = RESULTS_DIR / "stress_tests"
CHARTS_DIR = OUTPUT_DIR / "charts"

RISK_FREE_RATE_ANNUAL = 0.0
MODEL_NAME = "full_v0"

COST_BPS_TESTS = [
    TOTAL_COST_BPS,
    25,
    50,
    100,
]

EXECUTION_DELAY_DAYS = [
    0,
    1,
    2,
    5,
    10,
]

ROLLING_WINDOWS = {
    "1_month": 21,
    "3_month": 63,
    "6_month": 126,
    "12_month": 252,
}

HISTORICAL_WINDOWS = [
    {
        "scenario": "2020_covid_crash",
        "start": "2020-02-19",
        "end": "2020-04-30",
    },
    {
        "scenario": "2020_covid_full_year",
        "start": "2020-01-01",
        "end": "2020-12-31",
    },
    {
        "scenario": "2021_reflation_commodity_strength",
        "start": "2021-01-01",
        "end": "2021-12-31",
    },
    {
        "scenario": "2022_rates_inflation_shock",
        "start": "2022-01-01",
        "end": "2022-12-31",
    },
    {
        "scenario": "2023_post_shock_chop",
        "start": "2023-01-01",
        "end": "2023-12-31",
    },
    {
        "scenario": "2024_gold_strength_risk_on",
        "start": "2024-01-01",
        "end": "2024-12-31",
    },
    {
        "scenario": "2025_policy_dollar_gold_regime",
        "start": "2025-01-01",
        "end": "2025-12-31",
    },
    {
        "scenario": "2026_ytd",
        "start": "2026-01-01",
        "end": "2026-12-31",
    },
]

ASSET_SHOCKS = {
    "gold_minus_10": {
        "GLD": -0.10,
    },
    "silver_minus_15": {
        "SLV": -0.15,
    },
    "precious_metals_reversal": {
        "GLD": -0.10,
        "SLV": -0.15,
    },
    "oil_minus_20": {
        "USO": -0.20,
    },
    "natural_gas_minus_25": {
        "UNG": -0.25,
    },
    "energy_selloff": {
        "USO": -0.20,
        "UNG": -0.25,
    },
    "copper_minus_15": {
        "CPER": -0.15,
    },
    "agriculture_minus_10": {
        "DBA": -0.10,
    },
    "broad_commodity_selloff": {
        "GLD": -0.10,
        "SLV": -0.12,
        "USO": -0.15,
        "UNG": -0.20,
        "CPER": -0.12,
        "DBA": -0.08,
    },
    "inflation_shock_commodities_up": {
        "GLD": 0.08,
        "SLV": 0.10,
        "USO": 0.15,
        "UNG": 0.20,
        "CPER": 0.08,
        "DBA": 0.08,
    },
}


def prepare_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    returns = load_return_matrix()
    raw_weights = load_target_weights()

    returns.index = pd.to_datetime(returns.index)
    raw_weights.index = pd.to_datetime(raw_weights.index)

    returns = returns.sort_index().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raw_weights = raw_weights.sort_index().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    common_tickers = sorted(raw_weights.columns.intersection(returns.columns))

    if not common_tickers:
        raise ValueError("No overlapping tickers between model weights and returns.")

    returns = returns[common_tickers]
    raw_weights = raw_weights[common_tickers]

    return returns, raw_weights, common_tickers


def build_base_curves(
    returns: pd.DataFrame,
    raw_weights: pd.DataFrame,
    tickers: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    model_weights = apply_rebalance(
        raw_weights,
        BACKTEST_REBALANCE_MODE,
    )

    equal_weight = make_equal_weight(
        model_weights.index,
        tickers,
    )

    gold_only = make_gold_only(
        model_weights.index,
        tickers,
    )

    strategies = {
        MODEL_NAME: model_weights,
        "equal_weight": equal_weight,
        "gold_only": gold_only,
    }

    curves = {}

    for name, weights in strategies.items():
        curve, _ = simulate_strategy(
            name=name,
            target_weights=weights,
            returns=returns,
            initial_capital=INITIAL_CAPITAL,
            total_cost_bps=TOTAL_COST_BPS,
        )

        curves[name] = curve

    return curves, model_weights


def summarise_curve(
    curve: pd.DataFrame,
    strategy_name: str,
    benchmark_curve: pd.DataFrame | None = None,
    benchmark_name: str | None = None,
) -> dict:
    benchmark_returns = None

    if benchmark_curve is not None:
        benchmark_returns = benchmark_curve["net_return"]

    return calculate_full_summary(
        returns=curve["net_return"],
        equity=curve["equity"],
        turnover=curve["turnover"],
        transaction_cost=curve["transaction_cost"],
        exposure=curve["exposure"],
        benchmark_returns=benchmark_returns,
        strategy_name=strategy_name,
        benchmark_name=benchmark_name,
        initial_capital=INITIAL_CAPITAL,
        risk_free_rate_annual=RISK_FREE_RATE_ANNUAL,
        periods_per_year=TRADING_DAYS_PER_YEAR,
    )


def build_baseline_summary(curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    benchmark_curve = curves.get("equal_weight")

    for name, curve in curves.items():
        summary = summarise_curve(
            curve=curve,
            strategy_name=name,
            benchmark_curve=benchmark_curve,
            benchmark_name="equal_weight",
        )

        rows.append(summary)

    return pd.DataFrame(rows)


def run_historical_window_tests(
    curves: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for window in HISTORICAL_WINDOWS:
        start = pd.to_datetime(window["start"])
        end = pd.to_datetime(window["end"])

        for strategy_name, curve in curves.items():
            sub = curve.loc[start:end].copy()

            if sub.empty:
                continue

            summary = summarise_curve(
                curve=sub,
                strategy_name=strategy_name,
                benchmark_curve=None,
                benchmark_name=None,
            )

            summary["scenario"] = window["scenario"]
            summary["scenario_start"] = sub.index.min().date()
            summary["scenario_end"] = sub.index.max().date()

            rows.append(summary)

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(["scenario", "strategy"])

    return out


def run_cost_sensitivity_tests(
    returns: pd.DataFrame,
    model_weights: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for cost_bps in COST_BPS_TESTS:
        curve, _ = simulate_strategy(
            name=f"{MODEL_NAME}_cost_{cost_bps}_bps",
            target_weights=model_weights,
            returns=returns,
            initial_capital=INITIAL_CAPITAL,
            total_cost_bps=cost_bps,
        )

        summary = summarise_curve(
            curve=curve,
            strategy_name=f"{MODEL_NAME}_cost_{cost_bps}_bps",
        )

        summary["total_cost_bps"] = cost_bps

        rows.append(summary)

    return pd.DataFrame(rows).sort_values("total_cost_bps")


def run_execution_delay_tests(
    returns: pd.DataFrame,
    model_weights: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for delay_days in EXECUTION_DELAY_DAYS:
        delayed_weights = model_weights.shift(delay_days).fillna(0.0)

        curve, _ = simulate_strategy(
            name=f"{MODEL_NAME}_delay_{delay_days}_days",
            target_weights=delayed_weights,
            returns=returns,
            initial_capital=INITIAL_CAPITAL,
            total_cost_bps=TOTAL_COST_BPS,
        )

        summary = summarise_curve(
            curve=curve,
            strategy_name=f"{MODEL_NAME}_delay_{delay_days}_days",
        )

        summary["execution_delay_days"] = delay_days

        rows.append(summary)

    return pd.DataFrame(rows).sort_values("execution_delay_days")


def calculate_worst_rolling_windows(
    curve: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    returns = curve["net_return"].dropna()

    for label, window in ROLLING_WINDOWS.items():
        if len(returns) < window:
            continue

        rolling_return = (
            (1 + returns)
            .rolling(window)
            .apply(np.prod, raw=True)
            - 1
        )

        worst_end = rolling_return.idxmin()
        worst_return = rolling_return.loc[worst_end]

        end_position = returns.index.get_loc(worst_end)
        start_position = max(0, end_position - window + 1)
        worst_start = returns.index[start_position]

        period_returns = returns.loc[worst_start:worst_end]
        period_drawdown = calculate_drawdown_series(period_returns)

        rows.append(
            {
                "window": label,
                "trading_days": window,
                "start_date": worst_start.date(),
                "end_date": worst_end.date(),
                "worst_return": worst_return,
                "worst_return_pct": worst_return * 100,
                "max_drawdown_inside_window": period_drawdown.min(),
                "max_drawdown_inside_window_pct": period_drawdown.min() * 100,
            }
        )

    return pd.DataFrame(rows)


def run_asset_shock_tests(
    latest_weights: pd.Series,
    latest_equity: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    detail_rows = []

    latest_weights = latest_weights.fillna(0.0)

    for scenario_name, shock_map in ASSET_SHOCKS.items():
        shock_returns = pd.Series(0.0, index=latest_weights.index)

        for ticker, shock_return in shock_map.items():
            if ticker in shock_returns.index:
                shock_returns.loc[ticker] = shock_return

        contribution = latest_weights * shock_returns
        portfolio_return = contribution.sum()
        equity_impact = latest_equity * portfolio_return
        equity_after = latest_equity + equity_impact

        rows.append(
            {
                "scenario": scenario_name,
                "portfolio_return": portfolio_return,
                "portfolio_return_pct": portfolio_return * 100,
                "latest_equity": latest_equity,
                "equity_impact": equity_impact,
                "equity_after": equity_after,
            }
        )

        for ticker in latest_weights.index:
            detail_rows.append(
                {
                    "scenario": scenario_name,
                    "ticker": ticker,
                    "latest_weight": latest_weights.loc[ticker],
                    "shock_return": shock_returns.loc[ticker],
                    "return_contribution": contribution.loc[ticker],
                }
            )

    summary = pd.DataFrame(rows).sort_values("portfolio_return")
    details = pd.DataFrame(detail_rows)

    return summary, details


def build_asset_weight_summary(
    model_weights: pd.DataFrame,
) -> pd.DataFrame:
    weights = model_weights.copy().fillna(0.0)

    latest_date = weights.index.max()
    latest_weights = weights.loc[latest_date]

    out = pd.DataFrame(
        {
            "ticker": weights.columns,
            "average_weight": weights.mean().values,
            "median_weight": weights.median().values,
            "max_weight": weights.max().values,
            "min_weight": weights.min().values,
            "latest_weight": latest_weights.reindex(weights.columns).values,
        }
    )

    out["latest_date"] = latest_date.date()
    out = out.sort_values("average_weight", ascending=False)

    return out


def build_asset_contribution_summary(
    model_weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    common_dates = model_weights.index.intersection(returns.index)
    common_tickers = model_weights.columns.intersection(returns.columns)

    weights = model_weights.loc[common_dates, common_tickers].fillna(0.0)
    asset_returns = returns.loc[common_dates, common_tickers].fillna(0.0)

    used_weights = weights.shift(1).fillna(0.0)

    contribution = calculate_asset_contribution(
        used_weights=used_weights,
        asset_returns=asset_returns,
    )

    return contribution


def save_chart(fig, filename: str) -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cost_sensitivity(cost_df: pd.DataFrame) -> None:
    if cost_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        cost_df["total_cost_bps"],
        cost_df["cagr"] * 100,
        marker="o",
        label="CAGR",
    )

    ax.plot(
        cost_df["total_cost_bps"],
        cost_df["sharpe"],
        marker="o",
        label="Sharpe",
    )

    ax.set_title("Cost sensitivity")
    ax.set_xlabel("Total cost per trade, bps")
    ax.set_ylabel("Metric")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_chart(fig, "01_cost_sensitivity.png")


def plot_delay_sensitivity(delay_df: pd.DataFrame) -> None:
    if delay_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        delay_df["execution_delay_days"],
        delay_df["cagr"] * 100,
        marker="o",
        label="CAGR",
    )

    ax.plot(
        delay_df["execution_delay_days"],
        delay_df["sharpe"],
        marker="o",
        label="Sharpe",
    )

    ax.set_title("Execution delay sensitivity")
    ax.set_xlabel("Extra execution delay, trading days")
    ax.set_ylabel("Metric")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_chart(fig, "02_execution_delay_sensitivity.png")


def plot_historical_window_drawdowns(historical_df: pd.DataFrame) -> None:
    if historical_df.empty:
        return

    data = historical_df[historical_df["strategy"] == MODEL_NAME].copy()

    if data.empty:
        return

    data["max_drawdown_pct"] = data["max_drawdown"] * 100

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        data["scenario"],
        data["max_drawdown_pct"],
    )

    ax.set_title("Full V0 drawdown by historical stress window")
    ax.set_xlabel("Stress window")
    ax.set_ylabel("Max drawdown (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)

    save_chart(fig, "03_historical_window_drawdowns.png")


def plot_asset_shocks(shock_df: pd.DataFrame) -> None:
    if shock_df.empty:
        return

    data = shock_df.copy().sort_values("portfolio_return_pct")

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.barh(
        data["scenario"],
        data["portfolio_return_pct"],
    )

    ax.set_title("Latest portfolio impact under asset shock scenarios")
    ax.set_xlabel("Portfolio return impact (%)")
    ax.set_ylabel("Shock scenario")
    ax.grid(True, axis="x", alpha=0.3)

    save_chart(fig, "04_asset_shock_impacts.png")


def plot_asset_contribution(contribution_df: pd.DataFrame) -> None:
    if contribution_df.empty:
        return

    data = contribution_df.copy()
    data = data.sort_values("total_return_contribution")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        data["ticker"],
        data["total_return_contribution"] * 100,
    )

    ax.set_title("Full V0 asset return contribution")
    ax.set_xlabel("Cumulative return contribution (%)")
    ax.set_ylabel("Ticker")
    ax.grid(True, axis="x", alpha=0.3)

    save_chart(fig, "05_asset_contribution.png")


def save_charts(
    cost_df: pd.DataFrame,
    delay_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    shock_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
) -> None:
    plot_cost_sensitivity(cost_df)
    plot_delay_sensitivity(delay_df)
    plot_historical_window_drawdowns(historical_df)
    plot_asset_shocks(shock_df)
    plot_asset_contribution(contribution_df)


def save_outputs(
    baseline_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    delay_df: pd.DataFrame,
    worst_df: pd.DataFrame,
    shock_df: pd.DataFrame,
    shock_details_df: pd.DataFrame,
    weight_summary_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
    curves: dict[str, pd.DataFrame],
) -> None:
    baseline_df.to_csv(OUTPUT_DIR / "stress_baseline_summary.csv", index=False)
    historical_df.to_csv(OUTPUT_DIR / "stress_historical_windows.csv", index=False)
    cost_df.to_csv(OUTPUT_DIR / "stress_cost_sensitivity.csv", index=False)
    delay_df.to_csv(OUTPUT_DIR / "stress_execution_delay.csv", index=False)
    worst_df.to_csv(OUTPUT_DIR / "stress_worst_rolling_windows.csv", index=False)
    shock_df.to_csv(OUTPUT_DIR / "stress_asset_shocks.csv", index=False)
    shock_details_df.to_csv(OUTPUT_DIR / "stress_asset_shock_details.csv", index=False)
    weight_summary_df.to_csv(OUTPUT_DIR / "stress_asset_weight_summary.csv", index=False)
    contribution_df.to_csv(OUTPUT_DIR / "stress_asset_contribution.csv", index=False)

    curve_df = pd.concat(
        [
            curve.reset_index().assign(strategy=name)
            for name, curve in curves.items()
        ],
        ignore_index=True,
    )

    curve_df.to_csv(OUTPUT_DIR / "stress_base_curves.csv", index=False)


def run_stress_tests() -> dict[str, pd.DataFrame]:
    prepare_dirs()

    returns, raw_weights, tickers = load_model_inputs()

    curves, model_weights = build_base_curves(
        returns=returns,
        raw_weights=raw_weights,
        tickers=tickers,
    )

    baseline_df = build_baseline_summary(curves)

    historical_df = run_historical_window_tests(curves)

    cost_df = run_cost_sensitivity_tests(
        returns=returns,
        model_weights=model_weights,
    )

    delay_df = run_execution_delay_tests(
        returns=returns,
        model_weights=model_weights,
    )

    worst_df = calculate_worst_rolling_windows(
        curves[MODEL_NAME],
    )

    latest_equity = curves[MODEL_NAME]["equity"].iloc[-1]
    latest_weights = model_weights.iloc[-1]

    shock_df, shock_details_df = run_asset_shock_tests(
        latest_weights=latest_weights,
        latest_equity=latest_equity,
    )

    weight_summary_df = build_asset_weight_summary(
        model_weights=model_weights,
    )

    contribution_df = build_asset_contribution_summary(
        model_weights=model_weights,
        returns=returns,
    )

    save_outputs(
        baseline_df=baseline_df,
        historical_df=historical_df,
        cost_df=cost_df,
        delay_df=delay_df,
        worst_df=worst_df,
        shock_df=shock_df,
        shock_details_df=shock_details_df,
        weight_summary_df=weight_summary_df,
        contribution_df=contribution_df,
        curves=curves,
    )

    save_charts(
        cost_df=cost_df,
        delay_df=delay_df,
        historical_df=historical_df,
        shock_df=shock_df,
        contribution_df=contribution_df,
    )

    print("\nStress testing complete.")
    print(f"Saved outputs to: {OUTPUT_DIR}")
    print(f"Saved charts to: {CHARTS_DIR}")

    display_cols = [
        "strategy",
        "cagr",
        "annualised_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "final_equity",
        "average_exposure",
    ]

    display_cols = [col for col in display_cols if col in baseline_df.columns]

    print("\nBaseline:")
    print(baseline_df[display_cols].to_string(index=False))

    print("\nWorst rolling windows:")
    print(worst_df.to_string(index=False))

    print("\nAsset shock impact:")
    print(shock_df.to_string(index=False))

    return {
        "baseline": baseline_df,
        "historical_windows": historical_df,
        "cost_sensitivity": cost_df,
        "execution_delay": delay_df,
        "worst_rolling_windows": worst_df,
        "asset_shocks": shock_df,
        "asset_shock_details": shock_details_df,
        "asset_weight_summary": weight_summary_df,
        "asset_contribution": contribution_df,
    }


if __name__ == "__main__":
    run_stress_tests()
