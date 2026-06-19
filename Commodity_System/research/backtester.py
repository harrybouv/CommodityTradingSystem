import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Commodity_System.config import (
    PRICE_DATA_PATH,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    INITIAL_CAPITAL,
    TRADING_DAYS_PER_YEAR,
    TOTAL_COST_BPS,
    BACKTEST_REBALANCE_MODE,
    CASH_ANNUAL_YIELD,
)


from analytics import (
    build_equity_curve,
    calculate_drawdown_series,
    calculate_turnover,
    calculate_cost_drag,
    calculate_exposure,
    calculate_full_summary,
    calculate_monthly_returns_pivot,
    calculate_rolling_sharpe,
    calculate_rolling_volatility,
    calculate_rolling_beta,
    calculate_asset_contribution,
    calculate_alpha_beta,
)

from commodity_strategy import build_production_strategy_weight_matrix


# ============================================================
# SETTINGS
# ============================================================


REBALANCE_MODE = BACKTEST_REBALANCE_MODE
RISK_FREE_RATE_ANNUAL = 0.0

OUTPUT_DIR = RESULTS_DIR / "backtest"
CHARTS_DIR = OUTPUT_DIR / "charts"


# ============================================================
# DATA LOADING
# ============================================================

def load_return_matrix() -> pd.DataFrame:
    prices = pd.read_csv(PRICE_DATA_PATH)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])

    prices["daily_return"] = (
        prices.groupby("ticker")["adj_close"]
        .pct_change()
    )

    returns = (
        prices.pivot(index="date", columns="ticker", values="daily_return")
        .sort_index()
        .fillna(0)
    )

    return returns


def load_target_weights() -> pd.DataFrame:
    return build_production_strategy_weight_matrix()

# ============================================================
# REBALANCING
# ============================================================

def apply_rebalance(
    raw_weights: pd.DataFrame,
    mode: str = "monthly",
) -> pd.DataFrame:
    raw_weights = raw_weights.sort_index().fillna(0)

    if mode == "daily":
        return raw_weights.copy()

    if mode == "weekly":
        periods = raw_weights.index.to_period("W-FRI")
    elif mode == "monthly":
        periods = raw_weights.index.to_period("M")
    else:
        raise ValueError("mode must be 'daily', 'weekly', or 'monthly'.")

    rebalance_dates = (
        pd.Series(raw_weights.index, index=raw_weights.index)
        .groupby(periods)
        .last()
        .tolist()
    )

    out = raw_weights.loc[rebalance_dates]
    out = out.reindex(raw_weights.index).ffill().fillna(0)

    return out


# ============================================================
# BENCHMARK WEIGHTS
# ============================================================

def make_equal_weight(
    index: pd.Index,
    tickers: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        1 / len(tickers),
        index=index,
        columns=tickers,
    )


def make_gold_only(
    index: pd.Index,
    tickers: list[str],
) -> pd.DataFrame:
    weights = pd.DataFrame(
        0.0,
        index=index,
        columns=tickers,
    )

    if "GLD" in weights.columns:
        weights["GLD"] = 1.0

    return weights


def make_cash(
    index: pd.Index,
    tickers: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        0.0,
        index=index,
        columns=tickers,
    )


# ============================================================
# BACKTEST ENGINE
# ============================================================

def simulate_strategy(
    name: str,
    target_weights: pd.DataFrame,
    returns: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    total_cost_bps: float = TOTAL_COST_BPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_weights = target_weights.sort_index().fillna(0)
    returns = returns.sort_index().fillna(0)

    common_dates = target_weights.index.intersection(returns.index)
    common_tickers = target_weights.columns.intersection(returns.columns)

    if len(common_dates) == 0:
        raise ValueError(f"No overlapping dates for {name}.")

    if len(common_tickers) == 0:
        raise ValueError(f"No overlapping tickers for {name}.")

    weights = target_weights.loc[common_dates, common_tickers].fillna(0)
    asset_returns = returns.loc[common_dates, common_tickers].fillna(0)

    # Critical: avoids lookahead.
    used_weights = weights.shift(1).fillna(0)

    # Commodity return from positions actually held today
    gross_return = (used_weights * asset_returns).sum(axis=1)

    # Exposure actually held today, after the one-day lag
    exposure = used_weights.sum(axis=1).clip(lower=0.0, upper=1.0)

    # Residual uninvested capital earns cash yield
    cash_weight = (1.0 - exposure).clip(lower=0.0, upper=1.0)

    cash_daily_return = (
            (1.0 + CASH_ANNUAL_YIELD) ** (1.0 / TRADING_DAYS_PER_YEAR)
            - 1.0
    )

    cash_return = cash_weight * cash_daily_return

    turnover = weights.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = weights.iloc[0].abs().sum()

    transaction_cost = turnover * (total_cost_bps / 10_000)

    net_return = gross_return + cash_return - transaction_cost

    equity = build_equity_curve(
        net_return,
        initial_capital=initial_capital,
    )

    drawdown = calculate_drawdown_series(net_return)

    curve = pd.DataFrame(
        {
            "strategy": name,
            "gross_return": gross_return,
            "cash_return": cash_return,
            "transaction_cost": transaction_cost,
            "net_return": net_return,
            "equity": equity,
            "drawdown": drawdown,
            "turnover": turnover,
            "exposure": exposure,
            "cash_weight": cash_weight,
        }
    )

    curve.index.name = "date"

    asset_contribution = calculate_asset_contribution(
        used_weights=used_weights,
        asset_returns=asset_returns,
    )

    asset_contribution.insert(0, "strategy", name)

    return curve, asset_contribution


# ============================================================
# SUMMARY TABLES
# ============================================================

def build_performance_summary(
    curves: dict[str, pd.DataFrame],
    benchmark_name: str | None = None,
) -> pd.DataFrame:
    rows = []

    benchmark_returns = None

    if benchmark_name is not None and benchmark_name in curves:
        benchmark_returns = curves[benchmark_name]["net_return"]

    for name, curve in curves.items():
        summary = calculate_full_summary(
            returns=curve["net_return"],
            equity=curve["equity"],
            turnover=curve["turnover"],
            transaction_cost=curve["transaction_cost"],
            exposure=curve["exposure"],
            benchmark_returns=benchmark_returns,
            strategy_name=name,
            benchmark_name=benchmark_name,
            initial_capital=INITIAL_CAPITAL,
            risk_free_rate_annual=RISK_FREE_RATE_ANNUAL,
            periods_per_year=TRADING_DAYS_PER_YEAR,
        )

        rows.append(summary)

    summary_df = pd.DataFrame(rows)

    return summary_df.sort_values(
        ["sharpe", "calmar", "cagr"],
        ascending=False,
    )


def build_alpha_beta_summary(
    curves: dict[str, pd.DataFrame],
    benchmark_names: list[str],
) -> pd.DataFrame:
    rows = []

    for strategy_name, strategy_curve in curves.items():
        for benchmark_name in benchmark_names:
            if strategy_name == benchmark_name:
                continue

            if benchmark_name not in curves:
                continue

            stats = calculate_alpha_beta(
                strategy_returns=strategy_curve["net_return"],
                benchmark_returns=curves[benchmark_name]["net_return"],
                risk_free_rate_annual=RISK_FREE_RATE_ANNUAL,
                periods_per_year=TRADING_DAYS_PER_YEAR,
            )

            row = {
                "strategy": strategy_name,
                "benchmark": benchmark_name,
                **stats,
            }

            rows.append(row)

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            ["benchmark", "alpha_annualised", "information_ratio"],
            ascending=[True, False, False],
        )

    return out


def build_rolling_metrics(
    model_curve: pd.DataFrame,
    benchmark_curves: dict[str, pd.DataFrame],
    window: int = 252,
) -> pd.DataFrame:
    out = pd.DataFrame(index=model_curve.index)

    out["rolling_sharpe_252d"] = calculate_rolling_sharpe(
        model_curve["net_return"],
        window=window,
        risk_free_rate_annual=RISK_FREE_RATE_ANNUAL,
        periods_per_year=TRADING_DAYS_PER_YEAR,
    )

    out["rolling_volatility_252d"] = calculate_rolling_volatility(
        model_curve["net_return"],
        window=window,
        periods_per_year=TRADING_DAYS_PER_YEAR,
    )

    for name, bench_curve in benchmark_curves.items():
        out[f"rolling_beta_vs_{name}_252d"] = calculate_rolling_beta(
            model_curve["net_return"],
            bench_curve["net_return"],
            window=window,
        )

    out.index.name = "date"
    return out


# ============================================================
# CHARTING
# ============================================================

def save_equity_curve_chart(curves: dict[str, pd.DataFrame]) -> None:
    plt.figure(figsize=(11, 6))

    for name, curve in curves.items():
        plt.plot(curve.index, curve["equity"], label=name)

    plt.title("Equity Curve vs Benchmarks")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "01_equity_curve_vs_benchmarks.png", dpi=160)
    plt.close()


def save_drawdown_chart(curves: dict[str, pd.DataFrame]) -> None:
    plt.figure(figsize=(11, 6))

    for name, curve in curves.items():
        if name != "cash":
            plt.plot(curve.index, curve["drawdown"], label=name)

    plt.title("Drawdown vs Benchmarks")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "02_drawdown_vs_benchmarks.png", dpi=160)
    plt.close()


def save_allocation_chart(weights: pd.DataFrame) -> None:
    plot_weights = weights.resample("ME").last().fillna(0)
    plot_weights["Cash"] = 1 - plot_weights.sum(axis=1)
    plot_weights["Cash"] = plot_weights["Cash"].clip(lower=0)

    plt.figure(figsize=(12, 6))
    plt.stackplot(
        plot_weights.index,
        [plot_weights[col] for col in plot_weights.columns],
        labels=plot_weights.columns,
    )

    plt.title("Model Allocation Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Weight")
    plt.legend(loc="upper left", ncol=4)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "03_allocation_over_time.png", dpi=160)
    plt.close()


def save_monthly_returns_heatmap(model_curve: pd.DataFrame) -> None:
    monthly = calculate_monthly_returns_pivot(model_curve["net_return"])

    if monthly.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.imshow(monthly, aspect="auto")
    plt.colorbar(label="Monthly Return")

    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    plt.xticks(
        range(len(monthly.columns)),
        [month_labels[int(m) - 1] for m in monthly.columns],
        rotation=45,
    )

    plt.yticks(range(len(monthly.index)), monthly.index)

    for i in range(len(monthly.index)):
        for j in range(len(monthly.columns)):
            value = monthly.iloc[i, j]
            if pd.notna(value):
                plt.text(
                    j,
                    i,
                    f"{value * 100:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    plt.title("Model Monthly Returns")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "04_monthly_returns_heatmap.png", dpi=160)
    plt.close()


def save_rolling_metrics_chart(rolling_metrics: pd.DataFrame) -> None:
    plt.figure(figsize=(11, 6))

    plt.plot(
        rolling_metrics.index,
        rolling_metrics["rolling_sharpe_252d"],
        label="Rolling Sharpe 252d",
    )

    plt.plot(
        rolling_metrics.index,
        rolling_metrics["rolling_volatility_252d"],
        label="Rolling Volatility 252d",
    )

    plt.title("Rolling Sharpe and Volatility")
    plt.xlabel("Date")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "05_rolling_sharpe_volatility.png", dpi=160)
    plt.close()


def save_asset_contribution_chart(asset_contribution: pd.DataFrame) -> None:
    model_contrib = asset_contribution[
        asset_contribution["strategy"] == "model"
    ].copy()

    if model_contrib.empty:
        return

    model_contrib = model_contrib.sort_values("total_return_contribution")

    plt.figure(figsize=(9, 5))
    plt.barh(
        model_contrib["ticker"],
        model_contrib["total_return_contribution"],
    )

    plt.title("Model Asset Return Contribution")
    plt.xlabel("Cumulative Return Contribution")
    plt.grid(axis="x")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "06_asset_contribution.png", dpi=160)
    plt.close()


def save_charts(
    curves: dict[str, pd.DataFrame],
    model_weights: pd.DataFrame,
    rolling_metrics: pd.DataFrame,
    asset_contribution: pd.DataFrame,
) -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    save_equity_curve_chart(curves)
    save_drawdown_chart(curves)
    save_allocation_chart(model_weights)
    save_monthly_returns_heatmap(curves["model"])
    save_rolling_metrics_chart(rolling_metrics)
    save_asset_contribution_chart(asset_contribution)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    returns = load_return_matrix()
    raw_model_weights = load_target_weights()

    tickers = list(raw_model_weights.columns)

    model_weights = apply_rebalance(raw_model_weights, REBALANCE_MODE)

    equal_weight = make_equal_weight(model_weights.index, tickers)
    gold_only = make_gold_only(model_weights.index, tickers)
    cash = make_cash(model_weights.index, tickers)

    strategies = {
        "model": model_weights,
        "equal_weight": equal_weight,
        "gold_only": gold_only,
        "cash": cash,
    }

    curves = {}
    contribution_frames = []

    for name, weights in strategies.items():
        curve, asset_contribution = simulate_strategy(
            name=name,
            target_weights=weights,
            returns=returns,
            initial_capital=INITIAL_CAPITAL,
            total_cost_bps=TOTAL_COST_BPS,
        )

        curves[name] = curve
        contribution_frames.append(asset_contribution)

    asset_contribution_df = pd.concat(contribution_frames, ignore_index=True)

    performance_summary = build_performance_summary(
        curves,
        benchmark_name="equal_weight",
    )

    alpha_beta_summary = build_alpha_beta_summary(
        curves,
        benchmark_names=["equal_weight", "gold_only"],
    )

    benchmark_curves = {
        "equal_weight": curves["equal_weight"],
        "gold_only": curves["gold_only"],
    }

    rolling_metrics = build_rolling_metrics(
        model_curve=curves["model"],
        benchmark_curves=benchmark_curves,
        window=252,
    )

    monthly_returns = calculate_monthly_returns_pivot(
        curves["model"]["net_return"]
    )

    all_curves = (
        pd.concat(
            [
                curve.reset_index()
                for curve in curves.values()
            ],
            ignore_index=True,
        )
    )

    model_curve = curves["model"].reset_index()

    all_curves.to_csv(OUTPUT_DIR / "all_curves.csv", index=False)
    model_curve.to_csv(OUTPUT_DIR / "model_curve.csv", index=False)
    performance_summary.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    alpha_beta_summary.to_csv(OUTPUT_DIR / "alpha_beta_summary.csv", index=False)
    monthly_returns.to_csv(OUTPUT_DIR / "monthly_returns.csv")
    asset_contribution_df.to_csv(OUTPUT_DIR / "asset_contribution.csv", index=False)
    rolling_metrics.to_csv(OUTPUT_DIR / "rolling_metrics.csv")

    save_charts(
        curves=curves,
        model_weights=model_weights,
        rolling_metrics=rolling_metrics,
        asset_contribution=asset_contribution_df,
    )

    cols = [
        "strategy",
        "benchmark",
        "cagr",
        "annualised_volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "hit_rate",
        "average_daily_turnover",
        "average_exposure",
        "average_cash",
        "alpha_annualised",
        "beta",
        "r_squared",
        "information_ratio",
    ]

    cols = [col for col in cols if col in performance_summary.columns]

    print("\nBacktest complete.")
    print(f"Saved outputs to: {OUTPUT_DIR}")
    print(f"Saved charts to: {CHARTS_DIR}")

    print("\nPerformance summary:")
    print(performance_summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()