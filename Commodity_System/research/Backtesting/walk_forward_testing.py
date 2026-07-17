from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATH SETUP
# ============================================================

THIS_FILE = Path(__file__).resolve()
BACKTESTING_DIR = THIS_FILE.parent
RESEARCH_DIR = BACKTESTING_DIR.parent
COMMODITY_ROOT = RESEARCH_DIR.parent
PROJECT_ROOT = COMMODITY_ROOT.parent

for path in [PROJECT_ROOT, COMMODITY_ROOT, RESEARCH_DIR, BACKTESTING_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# ============================================================
# IMPORTS
# ============================================================

try:
    from Commodity_System.config import (
        INITIAL_CAPITAL,
        TRADING_DAYS_PER_YEAR,
        TOTAL_COST_BPS,
        SCORE_WEIGHTS,
        RESULTS_DIR,
        BACKTEST_REBALANCE_MODE,
    )

    from Commodity_System.data import load_price_data, make_return_matrix

    from Commodity_System.commodity_strategy import (
        load_score_inputs,
        build_strategy_weights_from_spec,
        build_production_strategy_weight_matrix,
        weights_long_to_matrix,
    )

except ImportError:
    from config import (
        INITIAL_CAPITAL,
        TRADING_DAYS_PER_YEAR,
        TOTAL_COST_BPS,
        SCORE_WEIGHTS,
        RESULTS_DIR,
        BACKTEST_REBALANCE_MODE,
    )

    from data import load_price_data, make_return_matrix

    from commodity_strategy import (
        load_score_inputs,
        build_strategy_weights_from_spec,
        build_production_strategy_weight_matrix,
        weights_long_to_matrix,
    )

from analytics import (
    calculate_full_summary,
    build_equity_curve,
    calculate_turnover,
    calculate_cost_drag,
    calculate_exposure,
    calculate_drawdown_series,
)


TRAIN_YEARS = 4
TEST_YEARS = 1

BENCHMARK_NAME = "equal_weight_commodities"

OUTPUT_DIR = RESULTS_DIR / "walk_forward"

STRATEGY_SPECS = {
    "production_model": {
        "custom": "production_commodity_models",
    },
}

REGIME_DEFINITIONS = [
    {
        "name": "2019_late_cycle",
        "start": "2019-01-01",
        "end": "2019-12-31",
    },
    {
        "name": "2020_covid_shock_recovery",
        "start": "2020-01-01",
        "end": "2020-12-31",
    },
    {
        "name": "2021_reflation_commodity_strength",
        "start": "2021-01-01",
        "end": "2021-12-31",
    },
    {
        "name": "2022_rates_inflation_shock",
        "start": "2022-01-01",
        "end": "2022-12-31",
    },
    {
        "name": "2023_post_shock_chop",
        "start": "2023-01-01",
        "end": "2023-12-31",
    },
    {
        "name": "2024_gold_strength_risk_on",
        "start": "2024-01-01",
        "end": "2024-12-31",
    },
    {
        "name": "2025_policy_dollar_gold_regime",
        "start": "2025-01-01",
        "end": "2025-12-31",
    },
    {
        "name": "2026_ytd",
        "start": "2026-01-01",
        "end": "2026-12-31",
    },
]

CHARTS_DIR = OUTPUT_DIR / "charts"

def prepare_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_return_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    out = returns.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = load_score_inputs()
    scores["date"] = pd.to_datetime(scores["date"])

    prices = load_price_data()
    returns = make_return_matrix(prices)
    returns = clean_return_matrix(returns)

    return scores, returns


def get_common_available_dates(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DatetimeIndex:
    score_dates = pd.DatetimeIndex(scores["date"].dropna().unique())
    return_dates = pd.DatetimeIndex(returns.index.dropna().unique())

    common_dates = score_dates.intersection(return_dates)
    common_dates = common_dates.sort_values()

    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between scores and returns.")

    return common_dates


def generate_windows(
    available_dates: pd.DatetimeIndex,
    train_years: int = TRAIN_YEARS,
    test_years: int = TEST_YEARS,
) -> list[dict]:
    available_dates = pd.DatetimeIndex(available_dates).sort_values()

    first_year = available_dates.min().year
    final_date = available_dates.max()

    windows = []

    for train_start_year in range(first_year, final_date.year + 1):
        train_start = pd.Timestamp(
            year=train_start_year,
            month=1,
            day=1,
        )

        train_end = pd.Timestamp(
            year=train_start_year + train_years,
            month=1,
            day=1,
        ) - pd.Timedelta(days=1)

        test_start = train_end + pd.Timedelta(days=1)

        test_end = pd.Timestamp(
            year=train_start_year + train_years + test_years,
            month=1,
            day=1,
        ) - pd.Timedelta(days=1)

        if test_start > final_date:
            continue

        test_end = min(test_end, final_date)

        has_test_data = (
            (available_dates >= test_start)
            & (available_dates <= test_end)
        ).any()

        if not has_test_data:
            continue

        windows.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )

    if not windows:
        raise ValueError("No valid walk-forward windows generated.")

    return windows


def get_rebalance_dates(
    index: pd.DatetimeIndex,
    rebalance_mode: str,
) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(index).sort_values()

    if len(index) == 0:
        return index

    mode = str(rebalance_mode).lower()

    if mode == "daily":
        return index

    date_series = pd.Series(index=index, data=index)

    if mode == "weekly":
        return pd.DatetimeIndex(
            date_series.groupby(index.to_period("W-FRI")).max().values
        )

    if mode == "monthly":
        return pd.DatetimeIndex(
            date_series.groupby(index.to_period("M")).max().values
        )

    raise ValueError(
        f"Unsupported rebalance mode: {rebalance_mode}. Use daily, weekly, or monthly."
    )


def prepare_rebalanced_weights(
    target_weight_matrix: pd.DataFrame,
    returns_index: pd.DatetimeIndex,
    rebalance_mode: str = BACKTEST_REBALANCE_MODE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns_index = pd.DatetimeIndex(returns_index).sort_values()

    aligned_targets = (
        target_weight_matrix
        .reindex(returns_index)
        .ffill()
        .fillna(0.0)
    )

    rebalance_dates = get_rebalance_dates(
        returns_index,
        rebalance_mode=rebalance_mode,
    )

    rebalanced_targets = pd.DataFrame(
        np.nan,
        index=returns_index,
        columns=aligned_targets.columns,
    )

    rebalanced_targets.loc[rebalance_dates] = aligned_targets.loc[rebalance_dates]
    rebalanced_targets = rebalanced_targets.ffill().fillna(0.0)

    used_weights = rebalanced_targets.shift(1).fillna(0.0)

    return used_weights, rebalanced_targets


def build_equal_weight_target_matrix(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    valid_assets = returns.notna()
    asset_count = valid_assets.sum(axis=1).replace(0, np.nan)

    equal_weights = valid_assets.div(asset_count, axis=0)
    equal_weights = equal_weights.fillna(0.0)

    return equal_weights


def run_portfolio_window(
    target_weight_matrix: pd.DataFrame,
    returns: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    rebalance_mode: str = BACKTEST_REBALANCE_MODE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_cols = returns.columns.intersection(target_weight_matrix.columns)

    if len(common_cols) == 0:
        raise ValueError("No overlapping tickers between returns and weights.")

    returns = returns[common_cols].copy()
    target_weight_matrix = target_weight_matrix[common_cols].copy()

    full_returns = returns.loc[returns.index <= test_end].copy()

    used_weights_full, rebalanced_targets_full = prepare_rebalanced_weights(
        target_weight_matrix=target_weight_matrix,
        returns_index=full_returns.index,
        rebalance_mode=rebalance_mode,
    )

    test_asset_returns = full_returns.loc[test_start:test_end].copy()
    used_weights = used_weights_full.loc[test_start:test_end].copy()
    rebalanced_targets = rebalanced_targets_full.loc[test_start:test_end].copy()

    test_asset_returns = test_asset_returns.reindex(
        index=used_weights.index,
        columns=used_weights.columns,
    ).fillna(0.0)

    gross_returns = (used_weights * test_asset_returns).sum(axis=1)

    turnover_full = calculate_turnover(used_weights_full)
    turnover = turnover_full.loc[test_start:test_end]

    transaction_cost = calculate_cost_drag(
        turnover=turnover,
        total_cost_bps=TOTAL_COST_BPS,
    )

    net_returns = gross_returns - transaction_cost
    exposure = calculate_exposure(used_weights)

    result = pd.DataFrame(
        {
            "gross_return": gross_returns,
            "return": net_returns,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "exposure": exposure,
        }
    )

    target_long = (
        rebalanced_targets
        .stack()
        .rename("target_weight")
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "ticker"})
    )

    used_long = (
        used_weights
        .stack()
        .rename("used_weight")
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "ticker"})
    )

    weight_detail = target_long.merge(
        used_long,
        on=["date", "ticker"],
        how="left",
    )

    return result, weight_detail


def run_benchmark_window(
    returns: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    rebalance_mode: str = BACKTEST_REBALANCE_MODE,
) -> pd.Series:
    equal_weight_targets = build_equal_weight_target_matrix(returns)

    benchmark_result, _ = run_portfolio_window(
        target_weight_matrix=equal_weight_targets,
        returns=returns,
        test_start=test_start,
        test_end=test_end,
        rebalance_mode=rebalance_mode,
    )

    return benchmark_result["return"]


def build_strategy_weights_for_window(
    scores: pd.DataFrame,
    spec: dict,
    test_end: pd.Timestamp,
) -> pd.DataFrame:
    score_subset = scores[scores["date"] <= test_end].copy()

    if score_subset.empty:
        raise ValueError(f"No score data available up to {test_end.date()}.")

    if spec.get("custom") == "production_commodity_models":
        return build_production_strategy_weight_matrix(
            scores=score_subset,
        )

    weights_long = build_strategy_weights_from_spec(
        scores=score_subset,
        spec=spec,
    )

    weights_matrix = weights_long_to_matrix(weights_long)

    return weights_matrix


def summarise_window(
    result: pd.DataFrame,
    benchmark_returns: pd.Series,
    strategy_name: str,
    window: dict,
) -> dict:
    returns = result["return"].dropna()
    equity = build_equity_curve(
        returns,
        initial_capital=INITIAL_CAPITAL,
    )

    summary = calculate_full_summary(
        returns=returns,
        equity=equity,
        turnover=result["turnover"],
        transaction_cost=result["transaction_cost"],
        exposure=result["exposure"],
        benchmark_returns=benchmark_returns,
        strategy_name=strategy_name,
        benchmark_name=BENCHMARK_NAME,
        initial_capital=INITIAL_CAPITAL,
        risk_free_rate_annual=0.0,
        periods_per_year=TRADING_DAYS_PER_YEAR,
    )

    summary["train_start"] = window["train_start"].date()
    summary["train_end"] = window["train_end"].date()
    summary["test_start"] = window["test_start"].date()
    summary["test_end"] = window["test_end"].date()

    return summary


def summarise_full_stitched_result(
    stitched: pd.DataFrame,
    strategy_name: str,
) -> dict:
    data = stitched.copy()
    data = data.sort_values("date")
    data = data.drop_duplicates(subset=["date"], keep="last")
    data = data.set_index("date")

    returns = data["return"].dropna()

    equity = build_equity_curve(
        returns,
        initial_capital=INITIAL_CAPITAL,
    )

    benchmark_returns = data["benchmark_return"].dropna()

    summary = calculate_full_summary(
        returns=returns,
        equity=equity,
        turnover=data["turnover"],
        transaction_cost=data["transaction_cost"],
        exposure=data["exposure"],
        benchmark_returns=benchmark_returns,
        strategy_name=strategy_name,
        benchmark_name=BENCHMARK_NAME,
        initial_capital=INITIAL_CAPITAL,
        risk_free_rate_annual=0.0,
        periods_per_year=TRADING_DAYS_PER_YEAR,
    )

    return summary


def build_equity_output(
    stitched: pd.DataFrame,
    strategy_name: str,
) -> pd.DataFrame:
    data = stitched.copy()
    data = data.sort_values("date")
    data = data.drop_duplicates(subset=["date"], keep="last")
    data = data.set_index("date")

    returns = data["return"].dropna()
    benchmark_returns = data["benchmark_return"].reindex(returns.index).fillna(0.0)

    equity = build_equity_curve(
        returns,
        initial_capital=INITIAL_CAPITAL,
    )

    benchmark_equity = build_equity_curve(
        benchmark_returns,
        initial_capital=INITIAL_CAPITAL,
    )

    drawdown = calculate_drawdown_series(returns)

    out = pd.DataFrame(
        {
            "date": returns.index,
            "strategy": strategy_name,
            "equity": equity.values,
            "benchmark_equity": benchmark_equity.values,
            "drawdown": drawdown.reindex(returns.index).values,
        }
    )

    return out

def get_regime_label(date) -> str:
    date = pd.to_datetime(date)

    for regime in REGIME_DEFINITIONS:
        start = pd.to_datetime(regime["start"])
        end = pd.to_datetime(regime["end"])

        if start <= date <= end:
            return regime["name"]

    return "unclassified"


def add_regime_columns(
    returns_df: pd.DataFrame,
    periods_df: pd.DataFrame,
    equity_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns_out = returns_df.copy()
    periods_out = periods_df.copy()
    equity_out = equity_df.copy()

    returns_out["date"] = pd.to_datetime(returns_out["date"])
    equity_out["date"] = pd.to_datetime(equity_out["date"])

    periods_out["test_start"] = pd.to_datetime(periods_out["test_start"])
    periods_out["test_end"] = pd.to_datetime(periods_out["test_end"])

    returns_out["regime"] = returns_out["date"].apply(get_regime_label)
    equity_out["regime"] = equity_out["date"].apply(get_regime_label)
    periods_out["regime"] = periods_out["test_start"].apply(get_regime_label)
    periods_out["test_year"] = periods_out["test_start"].dt.year

    return returns_out, periods_out, equity_out


def calculate_regime_summary(returns_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    data = returns_df.copy()
    data["date"] = pd.to_datetime(data["date"])

    for (strategy, regime), group in data.groupby(["strategy", "regime"]):
        group = group.sort_values("date").set_index("date")

        returns = group["return"].dropna()

        if returns.empty:
            continue

        benchmark_returns = group["benchmark_return"].reindex(returns.index)

        equity = build_equity_curve(
            returns,
            initial_capital=INITIAL_CAPITAL,
        )

        summary = calculate_full_summary(
            returns=returns,
            equity=equity,
            turnover=group["turnover"],
            transaction_cost=group["transaction_cost"],
            exposure=group["exposure"],
            benchmark_returns=benchmark_returns,
            strategy_name=strategy,
            benchmark_name=BENCHMARK_NAME,
            initial_capital=INITIAL_CAPITAL,
            risk_free_rate_annual=0.0,
            periods_per_year=TRADING_DAYS_PER_YEAR,
        )

        summary["regime"] = regime
        summary["regime_start"] = returns.index.min().date()
        summary["regime_end"] = returns.index.max().date()

        rows.append(summary)

    return pd.DataFrame(rows)


def save_chart(fig, filename: str) -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_walk_forward_equity(equity_df: pd.DataFrame) -> None:
    data = equity_df.copy()
    data["date"] = pd.to_datetime(data["date"])

    pivot = data.pivot(
        index="date",
        columns="strategy",
        values="equity",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(ax=ax)

    ax.set_title("Walk-forward equity curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)

    save_chart(fig, "walk_forward_equity_curve.png")


def plot_walk_forward_drawdown(equity_df: pd.DataFrame) -> None:
    data = equity_df.copy()
    data["date"] = pd.to_datetime(data["date"])

    pivot = data.pivot(
        index="date",
        columns="strategy",
        values="drawdown",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(ax=ax)

    ax.set_title("Walk-forward drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)

    save_chart(fig, "walk_forward_drawdown.png")


def plot_period_cagr(periods_df: pd.DataFrame) -> None:
    data = periods_df.copy()
    data["cagr_pct"] = data["cagr"] * 100

    pivot = data.pivot(
        index="test_year",
        columns="strategy",
        values="cagr_pct",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax)

    ax.set_title("Walk-forward CAGR by test year")
    ax.set_xlabel("Test year")
    ax.set_ylabel("CAGR (%)")
    ax.grid(True, axis="y", alpha=0.3)

    save_chart(fig, "walk_forward_cagr_by_year.png")


def plot_period_sharpe(periods_df: pd.DataFrame) -> None:
    data = periods_df.copy()

    pivot = data.pivot(
        index="test_year",
        columns="strategy",
        values="sharpe",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax)

    ax.set_title("Walk-forward Sharpe by test year")
    ax.set_xlabel("Test year")
    ax.set_ylabel("Sharpe")
    ax.grid(True, axis="y", alpha=0.3)

    save_chart(fig, "walk_forward_sharpe_by_year.png")


def plot_period_exposure(periods_df: pd.DataFrame) -> None:
    data = periods_df.copy()
    data["average_exposure_pct"] = data["average_exposure"] * 100

    pivot = data.pivot(
        index="test_year",
        columns="strategy",
        values="average_exposure_pct",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax)

    ax.set_title("Average exposure by test year")
    ax.set_xlabel("Test year")
    ax.set_ylabel("Average exposure (%)")
    ax.grid(True, axis="y", alpha=0.3)

    save_chart(fig, "walk_forward_exposure_by_year.png")


def plot_regime_cagr(regime_df: pd.DataFrame) -> None:
    data = regime_df.copy()
    data["cagr_pct"] = data["cagr"] * 100

    regime_order = [
        regime["name"]
        for regime in REGIME_DEFINITIONS
    ]

    data["regime"] = pd.Categorical(
        data["regime"],
        categories=regime_order,
        ordered=True,
    )

    pivot = data.pivot(
        index="regime",
        columns="strategy",
        values="cagr_pct",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(14, 6))
    pivot.plot(kind="bar", ax=ax)

    ax.set_title("CAGR by manual regime")
    ax.set_xlabel("Regime")
    ax.set_ylabel("CAGR (%)")
    ax.grid(True, axis="y", alpha=0.3)

    save_chart(fig, "walk_forward_cagr_by_regime.png")


def create_walk_forward_charts(
    periods_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> None:
    plot_walk_forward_equity(equity_df)
    plot_walk_forward_drawdown(equity_df)
    plot_period_cagr(periods_df)
    plot_period_sharpe(periods_df)
    plot_period_exposure(periods_df)
    plot_regime_cagr(regime_df)

    print(f"Saved charts to: {CHARTS_DIR}")

def run_walk_forward_test() -> dict[str, pd.DataFrame]:
    prepare_output_dir()

    scores, returns = load_inputs()
    available_dates = get_common_available_dates(scores, returns)

    windows = generate_windows(
        available_dates=available_dates,
        train_years=TRAIN_YEARS,
        test_years=TEST_YEARS,
    )

    all_period_summaries = []
    all_daily_results = []
    all_equity_outputs = []
    all_weight_outputs = []

    for strategy_name, spec in STRATEGY_SPECS.items():
        print(f"\nRunning strategy: {strategy_name}")

        strategy_daily_results = []

        for window in windows:
            train_start = window["train_start"]
            train_end = window["train_end"]
            test_start = window["test_start"]
            test_end = window["test_end"]

            print(
                f"Train {train_start.date()} to {train_end.date()} | "
                f"Test {test_start.date()} to {test_end.date()}"
            )

            weights_matrix = build_strategy_weights_for_window(
                scores=scores,
                spec=spec,
                test_end=test_end,
            )

            result, weight_detail = run_portfolio_window(
                target_weight_matrix=weights_matrix,
                returns=returns,
                test_start=test_start,
                test_end=test_end,
                rebalance_mode=BACKTEST_REBALANCE_MODE,
            )

            benchmark_returns = run_benchmark_window(
                returns=returns,
                test_start=test_start,
                test_end=test_end,
                rebalance_mode=BACKTEST_REBALANCE_MODE,
            )

            result = result.copy()
            result["benchmark_return"] = benchmark_returns.reindex(result.index)
            result["strategy"] = strategy_name
            result["train_start"] = train_start.date()
            result["train_end"] = train_end.date()
            result["test_start"] = test_start.date()
            result["test_end"] = test_end.date()
            result = result.reset_index().rename(columns={"index": "date"})

            weight_detail = weight_detail.copy()
            weight_detail["strategy"] = strategy_name
            weight_detail["train_start"] = train_start.date()
            weight_detail["train_end"] = train_end.date()
            weight_detail["test_start"] = test_start.date()
            weight_detail["test_end"] = test_end.date()

            period_summary = summarise_window(
                result=result.set_index("date"),
                benchmark_returns=benchmark_returns,
                strategy_name=strategy_name,
                window=window,
            )

            all_period_summaries.append(period_summary)
            all_daily_results.append(result)
            all_weight_outputs.append(weight_detail)
            strategy_daily_results.append(result)

        stitched = pd.concat(strategy_daily_results, ignore_index=True)

        full_summary = summarise_full_stitched_result(
            stitched=stitched,
            strategy_name=strategy_name,
        )

        all_equity_outputs.append(
            build_equity_output(
                stitched=stitched,
                strategy_name=strategy_name,
            )
        )

        full_summary["train_years"] = TRAIN_YEARS
        full_summary["test_years"] = TEST_YEARS
        full_summary["rebalance_mode"] = BACKTEST_REBALANCE_MODE

        if "full_summaries" not in locals():
            full_summaries = []

        full_summaries.append(full_summary)

    summary_df = pd.DataFrame(full_summaries)
    periods_df = pd.DataFrame(all_period_summaries)
    returns_df = pd.concat(all_daily_results, ignore_index=True)
    equity_df = pd.concat(all_equity_outputs, ignore_index=True)
    weights_df = pd.concat(all_weight_outputs, ignore_index=True)

    returns_df, periods_df, equity_df = add_regime_columns(
        returns_df=returns_df,
        periods_df=periods_df,
        equity_df=equity_df,
    )

    regime_df = calculate_regime_summary(returns_df)

    summary_df.to_csv(OUTPUT_DIR / "walk_forward_summary.csv", index=False)
    periods_df.to_csv(OUTPUT_DIR / "walk_forward_periods.csv", index=False)
    returns_df.to_csv(OUTPUT_DIR / "walk_forward_returns.csv", index=False)
    equity_df.to_csv(OUTPUT_DIR / "walk_forward_equity.csv", index=False)
    weights_df.to_csv(OUTPUT_DIR / "walk_forward_weights.csv", index=False)
    regime_df.to_csv(OUTPUT_DIR / "walk_forward_regime_summary.csv", index=False)

# create_walk_forward_charts(
#     periods_df=periods_df,
#     returns_df=returns_df,
#     equity_df=equity_df,
#     regime_df=regime_df,
# )
    print("\nWalk-forward validation complete.")
    print(f"Saved results to: {OUTPUT_DIR}")

    display_cols = [
        "strategy",
        "cagr",
        "annualised_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "final_equity",
        "alpha_annualised",
        "beta",
        "average_exposure",
    ]

    available_display_cols = [
        col for col in display_cols if col in summary_df.columns
    ]

    print("\nSummary:")
    print(summary_df[available_display_cols].to_string(index=False))

    return {
        "summary": summary_df,
        "periods": periods_df,
        "returns": returns_df,
        "equity": equity_df,
        "weights": weights_df,
        "regimes": regime_df,
    }


if __name__ == "__main__":
    run_walk_forward_test()