# research/parameter_testing.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Commodity_System.config import (
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    INITIAL_CAPITAL,
    TRADING_DAYS_PER_YEAR,
    TOTAL_COST_BPS,
    MAX_GROUP_WEIGHT,
    MAX_TOTAL_RISK_ASSET_EXPOSURE,
    UNIVERSE,
    CASH_ANNUAL_YIELD,
    BACKTEST_REBALANCE_MODE,
)

from analytics import calculate_full_summary
from backtester import load_return_matrix, apply_rebalance, simulate_strategy


# ============================================================
# SETTINGS
# ============================================================

N_RUNS = 1000
TOP_N_FULL_ANALYTICS = 50
RANDOM_SEED = 42

REBALANCE_MODE = BACKTEST_REBALANCE_MODE
MIN_VOL_FOR_SIZING = 0.05

OUTPUT_DIR = RESULTS_DIR / "parameter_testing"
CHARTS_DIR = OUTPUT_DIR / "charts"

np.random.seed(RANDOM_SEED)


# ============================================================
# FILTERS
# ============================================================

MAX_ALLOWED_DRAWDOWN = -0.22
MAX_ALLOWED_VOLATILITY = 0.16
MIN_ALLOWED_SHARPE = 0.75
MAX_ALLOWED_AVERAGE_EXPOSURE = 1.00
MAX_ALLOWED_ANNUALISED_TURNOVER = 10.0


# ============================================================
# LOAD DATA
# ============================================================

def load_score_data() -> pd.DataFrame:
    momentum = pd.read_csv(PROCESSED_DATA_DIR / "momentum_scores.csv")
    relative_strength = pd.read_csv(PROCESSED_DATA_DIR / "relative_strength_scores.csv")
    trend = pd.read_csv(PROCESSED_DATA_DIR / "trend_scores.csv")
    volatility = pd.read_csv(PROCESSED_DATA_DIR / "volatility_scores.csv")
    risk = pd.read_csv(PROCESSED_DATA_DIR / "risk_scores.csv")
    trend_persistence = pd.read_csv(PROCESSED_DATA_DIR / "trend_persistence_scores.csv")

    for df in [momentum, relative_strength, trend, trend_persistence, volatility, risk]:
        df["date"] = pd.to_datetime(df["date"])

    scores = momentum[["date", "ticker", "adj_close", "momentum_score"]].copy()

    scores = scores.merge(
        relative_strength[["date", "ticker", "relative_strength_score"]],
        on=["date", "ticker"],
        how="inner",
    )

    scores = scores.merge(
        trend[["date", "ticker", "trend_score"]],
        on=["date", "ticker"],
        how="inner",
    )
    scores = scores.merge(
        trend_persistence[["date", "ticker", "trend_persistence_score"]],
        on=["date", "ticker"],
        how="left",
    )

    scores["trend_persistence_score"] = (
        scores["trend_persistence_score"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.50)
        .clip(0, 1)
    )

    scores = scores.merge(
        volatility[
            [
                "date",
                "ticker",
                "volatility_score",
                "realised_vol_60d",
            ]
        ],
        on=["date", "ticker"],
        how="inner",
    )

    scores = scores.merge(
        risk[["date", "ticker", "risk_score"]],
        on=["date", "ticker"],
        how="inner",
    )

    scores = scores.dropna()
    scores = scores.sort_values(["date", "ticker"]).reset_index(drop=True)

    if scores.empty:
        raise ValueError("Merged score data is empty.")

    return scores

def prepare_matrices(scores: pd.DataFrame, returns: pd.DataFrame) -> dict:
    dates = sorted(scores["date"].unique())
    tickers = sorted(scores["ticker"].unique())

    score_mats = {}

    for col in [
        "momentum_score",
        "relative_strength_score",
        "trend_score",
        "trend_persistence_score",
        "volatility_score",
        "risk_score",
        "realised_vol_60d",
    ]:
        score_mats[col] = (
            scores.pivot(index="date", columns="ticker", values=col)
            .reindex(index=dates, columns=tickers)
            .fillna(0.0)
        )

    returns_matrix = (
        returns.reindex(index=dates, columns=tickers)
        .fillna(0.0)
    )

    group_map = {
        ticker: UNIVERSE[ticker]["group"]
        for ticker in tickers
        if ticker in UNIVERSE
    }

    groups = sorted(set(group_map.values()))

    return {
        "dates": dates,
        "tickers": tickers,
        "groups": groups,
        "group_map": group_map,
        "score_mats": score_mats,
        "returns": returns_matrix,
    }


# ============================================================
# FAST WEIGHT ENGINE
# ============================================================

def sample_params() -> dict:
    """
    Local search around the best trend-persistence candidate.

    Base:
    momentum 0.20
    relative strength 0.15
    trend 0.05
    trend persistence 0.10
    volatility 0.20
    risk 0.30
    """

    base = np.array([0.20, 0.15, 0.05, 0.10, 0.20, 0.30])

    noise = np.random.normal(
        loc=0.0,
        scale=np.array([0.05, 0.04, 0.025, 0.04, 0.05, 0.06]),
    )

    weights = base + noise
    weights = np.clip(weights, 0.01, None)
    weights = weights / weights.sum()

    return {
        "momentum_weight": weights[0],
        "relative_strength_weight": weights[1],
        "trend_weight": weights[2],
        "trend_persistence_weight": weights[3],
        "volatility_weight": weights[4],
        "risk_weight": weights[5],
        "min_score_to_hold": np.random.uniform(0.55, 0.72),
        "max_asset_weight": np.random.uniform(0.28, 0.38),
    }

def build_fast_weights(mats: dict, params: dict) -> pd.DataFrame:
    sm = mats["score_mats"]

    final_score = (
            params["momentum_weight"] * sm["momentum_score"]
            + params["relative_strength_weight"] * sm["relative_strength_score"]
            + params["trend_weight"] * sm["trend_score"]
            + params["trend_persistence_weight"] * sm["trend_persistence_score"]
            + params["volatility_weight"] * sm["volatility_score"]
            + params["risk_weight"] * sm["risk_score"]
    ).clip(0, 1)

    vol_for_sizing = sm["realised_vol_60d"].clip(lower=MIN_VOL_FOR_SIZING)

    raw_signal = final_score / vol_for_sizing
    raw_signal = raw_signal.where(final_score >= params["min_score_to_hold"], 0.0)

    signal_sum = raw_signal.sum(axis=1)
    weights = raw_signal.div(signal_sum.replace(0, np.nan), axis=0).fillna(0.0)

    weights = weights.clip(lower=0.0, upper=params["max_asset_weight"])

    # Group caps
    for group, cap in MAX_GROUP_WEIGHT.items():
        group_tickers = [
            ticker for ticker, ticker_group in mats["group_map"].items()
            if ticker_group == group and ticker in weights.columns
        ]

        if not group_tickers:
            continue

        group_weight = weights[group_tickers].sum(axis=1)
        scale = (cap / group_weight).where(group_weight > cap, 1.0)
        scale = scale.replace([np.inf, -np.inf], 1.0).fillna(1.0)

        weights[group_tickers] = weights[group_tickers].mul(scale, axis=0)

    # Total exposure cap
    total_weight = weights.sum(axis=1)
    total_scale = (
        MAX_TOTAL_RISK_ASSET_EXPOSURE / total_weight
    ).where(total_weight > MAX_TOTAL_RISK_ASSET_EXPOSURE, 1.0)

    total_scale = total_scale.replace([np.inf, -np.inf], 1.0).fillna(1.0)
    weights = weights.mul(total_scale, axis=0)

    return weights.fillna(0.0)

def fast_backtest(weights: pd.DataFrame, returns: pd.DataFrame) -> dict:
    weights = apply_rebalance(weights, REBALANCE_MODE)

    common_index = weights.index.intersection(returns.index)
    weights = weights.loc[common_index]
    rets = returns.loc[common_index, weights.columns]

    used_weights = weights.shift(1).fillna(0.0)

    gross_return = (used_weights * rets).sum(axis=1)

    exposure = used_weights.sum(axis=1).clip(lower=0.0, upper=1.0)
    cash_weight = (1.0 - exposure).clip(lower=0.0, upper=1.0)

    cash_daily_return = (
        (1.0 + CASH_ANNUAL_YIELD) ** (1.0 / TRADING_DAYS_PER_YEAR)
        - 1.0
    )

    cash_return = cash_weight * cash_daily_return

    turnover = weights.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = weights.iloc[0].abs().sum()

    transaction_cost = turnover * (TOTAL_COST_BPS / 10_000)

    net_return = gross_return + cash_return - transaction_cost

    equity = (1 + net_return).cumprod()
    drawdown = equity / equity.cummax() - 1

    if len(net_return) < 100 or net_return.std() == 0:
        return {}

    years = len(net_return) / TRADING_DAYS_PER_YEAR

    total_return = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1
    vol = net_return.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = net_return.mean() / net_return.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    average_exposure = exposure.mean()
    average_cash = cash_weight.mean()

    annualised_turnover = turnover.mean() * TRADING_DAYS_PER_YEAR

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualised_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "average_exposure": average_exposure,
        "average_cash": average_cash,
        "annualised_turnover": annualised_turnover,
        "average_daily_turnover": turnover.mean(),
        "average_daily_cash_return": cash_return.mean(),
        "final_equity_fast": INITIAL_CAPITAL * equity.iloc[-1],
    }



# ============================================================
# OBJECTIVE
# ============================================================

def passes_filters(row: dict) -> bool:
    if not np.isfinite(row.get("cagr", np.nan)):
        return False

    return (
        row["max_drawdown"] >= MAX_ALLOWED_DRAWDOWN
        and row["annualised_volatility"] <= MAX_ALLOWED_VOLATILITY
        and row["sharpe"] >= MIN_ALLOWED_SHARPE
        and row["average_exposure"] <= MAX_ALLOWED_AVERAGE_EXPOSURE
        and row["annualised_turnover"] <= MAX_ALLOWED_ANNUALISED_TURNOVER
    )


def robust_score(row: dict) -> float:
    turnover_penalty = max(0.0, row["annualised_turnover"] - 4.0) / 6.0
    drawdown_penalty = max(0.0, abs(row["max_drawdown"]) - 0.15) / 0.10

    return (
        0.45 * row["cagr"]
        + 0.25 * row["sharpe"]
        + 0.20 * row["calmar"]
        - 0.05 * turnover_penalty
        - 0.05 * drawdown_penalty
    )


# ============================================================
# FULL ANALYTICS ON TOP CANDIDATES
# ============================================================

def run_full_analytics_for_top(
    mats: dict,
    fast_results: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    top = fast_results.head(TOP_N_FULL_ANALYTICS).copy()

    for idx, row in top.iterrows():
        params = {
            "momentum_weight": row["momentum_weight"],
            "relative_strength_weight": row["relative_strength_weight"],
            "trend_weight": row["trend_weight"],
            "trend_persistence_weight": row["trend_persistence_weight"],
            "volatility_weight": row["volatility_weight"],
            "risk_weight": row["risk_weight"],
            "min_score_to_hold": row["min_score_to_hold"],
            "max_asset_weight": row["max_asset_weight"],
        }

        raw_weights = build_fast_weights(mats, params)
        weights = apply_rebalance(raw_weights, REBALANCE_MODE)

        curve, _ = simulate_strategy(
            name=f"candidate_{idx}",
            target_weights=weights,
            returns=returns,
            initial_capital=INITIAL_CAPITAL,
            total_cost_bps=TOTAL_COST_BPS,
        )

        summary = calculate_full_summary(
            returns=curve["net_return"],
            equity=curve["equity"],
            turnover=curve["turnover"],
            transaction_cost=curve["transaction_cost"],
            exposure=curve["exposure"],
            strategy_name=f"candidate_{idx}",
            initial_capital=INITIAL_CAPITAL,
            periods_per_year=TRADING_DAYS_PER_YEAR,
        )

        rows.append({**params, **summary})

    out = pd.DataFrame(rows)

    if not out.empty:
        out["full_robust_score"] = out.apply(robust_score, axis=1)
        out = out.sort_values(
            ["full_robust_score", "cagr", "sharpe"],
            ascending=False,
        )

    return out


# ============================================================
# CHARTS
# ============================================================

def save_charts(results: pd.DataFrame) -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    plt.scatter(results["cagr"], results["max_drawdown"], alpha=0.35)
    plt.xlabel("CAGR")
    plt.ylabel("Max Drawdown")
    plt.title("Parameter Search: CAGR vs Max Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "01_cagr_vs_max_drawdown.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.scatter(
        results["momentum_weight"],
        results["risk_weight"],
        c=results["score"],
        alpha=0.45,
    )
    plt.colorbar(label="Score")
    plt.xlabel("Momentum Weight")
    plt.ylabel("Risk Weight")
    plt.title("Momentum vs Risk Weight")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "02_momentum_vs_risk.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.scatter(
        results["volatility_weight"],
        results["trend_weight"],
        c=results["score"],
        alpha=0.45,
    )
    plt.colorbar(label="Score")
    plt.xlabel("Volatility Weight")
    plt.ylabel("Trend Weight")
    plt.title("Volatility vs Trend Weight")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "03_volatility_vs_trend.png", dpi=160)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scores = load_score_data()
    returns = load_return_matrix()
    mats = prepare_matrices(scores, returns)

    rows = []

    for i in range(N_RUNS):
        params = sample_params()

        weights = build_fast_weights(mats, params)
        metrics = fast_backtest(weights, mats["returns"])

        if not metrics:
            continue

        row = {**params, **metrics}
        row["passed_filters"] = passes_filters(row)
        row["score"] = robust_score(row)

        rows.append(row)

        if (i + 1) % 1000 == 0:
            print(f"Completed {i + 1:,} / {N_RUNS:,} runs")

    results = pd.DataFrame(rows)

    results = results.sort_values(
        ["score", "cagr", "sharpe"],
        ascending=False,
    )

    filtered = results[results["passed_filters"]].copy()
    filtered = filtered.sort_values(
        ["score", "cagr", "sharpe"],
        ascending=False,
    )

    results.to_csv(OUTPUT_DIR / "parameter_search_all_fast_results.csv", index=False)
    filtered.to_csv(OUTPUT_DIR / "parameter_search_filtered_fast_results.csv", index=False)

    if filtered.empty:
        print("\nNo candidates passed filters. Loosen filters.")
        print(f"Saved all results to: {OUTPUT_DIR}")
        return

    full = run_full_analytics_for_top(
        mats=mats,
        fast_results=filtered,
        returns=returns,
    )

    full.to_csv(OUTPUT_DIR / "parameter_search_top_full_analytics.csv", index=False)

    top_100 = filtered.head(100).copy()
    top_100.to_csv(OUTPUT_DIR / "parameter_search_top_100.csv", index=False)

    weight_cols = [
        "momentum_weight",
        "relative_strength_weight",
        "trend_weight",
        "trend_persistence_weight",
        "volatility_weight",
        "risk_weight",
        "min_score_to_hold",
        "max_asset_weight",
    ]

    region = pd.DataFrame(
        {
            "mean_top_100": top_100[weight_cols].mean(),
            "median_top_100": top_100[weight_cols].median(),
            "std_top_100": top_100[weight_cols].std(),
            "min_top_100": top_100[weight_cols].min(),
            "max_top_100": top_100[weight_cols].max(),
        }
    )

    region.to_csv(OUTPUT_DIR / "parameter_search_top_100_region.csv")

    save_charts(filtered)

    print("\nTop 20 fast-filtered candidates:")
    display_cols = [
        "momentum_weight",
        "relative_strength_weight",
        "trend_weight",
        "volatility_weight",
        "risk_weight",
        "min_score_to_hold",
        "max_asset_weight",
        "cagr",
        "annualised_volatility",
        "sharpe",
        "calmar",
        "max_drawdown",
        "average_exposure",
        "annualised_turnover",
        "score",
    ]

    print(filtered[display_cols].head(20).to_string(index=False))

    print("\nTop full-analytics candidates:")
    full_cols = [
        "momentum_weight",
        "relative_strength_weight",
        "trend_weight",
        "volatility_weight",
        "risk_weight",
        "min_score_to_hold",
        "max_asset_weight",
        "cagr",
        "annualised_volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "average_exposure",
        "annualised_turnover",
        "full_robust_score",
    ]

    full_cols = [col for col in full_cols if col in full.columns]
    print(full[full_cols].head(20).to_string(index=False))

    print("\nTop 100 parameter region:")
    print(region.to_string())

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()