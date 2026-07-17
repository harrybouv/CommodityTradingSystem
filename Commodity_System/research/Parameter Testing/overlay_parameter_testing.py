# Commodity_System/research/overlay_parameter_testing.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    MIN_SCORE_TO_HOLD,
    MAX_ASSET_WEIGHT,
)

from analytics import calculate_full_summary
from Commodity_System.research.Backtesting.backtester import load_return_matrix, apply_rebalance, simulate_strategy


# ============================================================
# SETTINGS
# ============================================================

N_RUNS = 10000
TOP_N_FULL_ANALYTICS = 50
RANDOM_SEED = 42

REBALANCE_MODE = BACKTEST_REBALANCE_MODE
MIN_VOL_FOR_SIZING = 0.05

OUTPUT_DIR = RESULTS_DIR / "overlay_parameter_testing"

np.random.seed(RANDOM_SEED)


# ============================================================
# BASELINE V1 SETTINGS
# ============================================================

# Freeze the alpha engine.
# We are NOT re-optimising the whole strategy here.
V1_SCORE_WEIGHTS = {
    "momentum_score": 0.20,
    "relative_strength_score": 0.15,
    "trend_score": 0.05,
    "volatility_score": 0.22,
    "risk_score": 0.38,
}

BASE_MIN_SCORE_TO_HOLD = MIN_SCORE_TO_HOLD
BASE_MAX_ASSET_WEIGHT = MAX_ASSET_WEIGHT


# ============================================================
# FILTERS
# ============================================================

MIN_ALLOWED_CAGR = 0.118
MIN_ALLOWED_SHARPE = 1.15
MAX_ALLOWED_DRAWDOWN = -0.145
MAX_ALLOWED_VOLATILITY = 0.13
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

    for df in [momentum, relative_strength, trend, volatility, risk]:
        df["date"] = pd.to_datetime(df["date"])

    required_vol_cols = [
        "date",
        "ticker",
        "volatility_score",
        "realised_vol_60d",
        "vol_stress_score",
        "vol_allocation_score",
    ]

    required_risk_cols = [
        "date",
        "ticker",
        "risk_score",
        "risk_stress_score",
        "risk_allocation_score",
    ]

    missing_vol = [c for c in required_vol_cols if c not in volatility.columns]
    missing_risk = [c for c in required_risk_cols if c not in risk.columns]

    if missing_vol:
        raise ValueError(
            f"volatility_scores.csv is missing columns: {missing_vol}. "
            "Rerun Commodity_System/scoring/volatility.py."
        )

    if missing_risk:
        raise ValueError(
            f"risk_scores.csv is missing columns: {missing_risk}. "
            "Rerun Commodity_System/scoring/risk.py."
        )

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
        volatility[required_vol_cols],
        on=["date", "ticker"],
        how="inner",
    )

    scores = scores.merge(
        risk[required_risk_cols],
        on=["date", "ticker"],
        how="inner",
    )

    scores = scores.dropna(
        subset=[
            "momentum_score",
            "relative_strength_score",
            "trend_score",
            "volatility_score",
            "risk_score",
            "realised_vol_60d",
        ]
    )

    scores = scores.sort_values(["date", "ticker"]).reset_index(drop=True)

    if scores.empty:
        raise ValueError("Merged score data is empty.")

    return scores


def prepare_matrices(scores: pd.DataFrame, returns: pd.DataFrame) -> dict:
    dates = sorted(scores["date"].unique())
    tickers = sorted(scores["ticker"].unique())

    score_mats = {}

    cols = [
        "momentum_score",
        "relative_strength_score",
        "trend_score",
        "volatility_score",
        "risk_score",
        "realised_vol_60d",
        "vol_stress_score",
        "vol_allocation_score",
        "risk_stress_score",
        "risk_allocation_score",
    ]

    for col in cols:
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

    return {
        "dates": dates,
        "tickers": tickers,
        "group_map": group_map,
        "score_mats": score_mats,
        "returns": returns_matrix,
    }


# ============================================================
# PARAMETER SAMPLING
# ============================================================

def sample_overlay_params() -> dict:
    mode = np.random.choice(
        ["asset_only", "asset_and_portfolio"],
        p=[0.60, 0.40],
    )

    return {
        "overlay_mode": mode,

        # Asset-level overlay.
        # These are intentionally mild. The earlier overlay was probably too harsh.
        "risk_weak_penalty": np.random.uniform(0.02, 0.22),
        "risk_strong_penalty": np.random.uniform(0.00, 0.05),
        "vol_weak_penalty": np.random.uniform(0.02, 0.18),
        "vol_strong_penalty": np.random.uniform(0.00, 0.04),
        "min_asset_multiplier": np.random.uniform(0.85, 1.00),

        # Portfolio-level overlay.
        # Even milder, because cutting total exposure can easily kill CAGR.
        "portfolio_weak_penalty": np.random.uniform(0.00, 0.12),
        "portfolio_strong_penalty": np.random.uniform(0.00, 0.035),
        "min_portfolio_multiplier": np.random.uniform(0.90, 1.00),
    }


# ============================================================
# WEIGHT ENGINE
# ============================================================

def build_base_final_score(mats: dict) -> pd.DataFrame:
    sm = mats["score_mats"]

    final_score = (
        V1_SCORE_WEIGHTS["momentum_score"] * sm["momentum_score"]
        + V1_SCORE_WEIGHTS["relative_strength_score"] * sm["relative_strength_score"]
        + V1_SCORE_WEIGHTS["trend_score"] * sm["trend_score"]
        + V1_SCORE_WEIGHTS["volatility_score"] * sm["volatility_score"]
        + V1_SCORE_WEIGHTS["risk_score"] * sm["risk_score"]
    ).clip(0, 1)

    return final_score


def apply_group_and_exposure_caps(weights: pd.DataFrame, mats: dict) -> pd.DataFrame:
    weights = weights.copy()

    # Asset cap.
    weights = weights.clip(lower=0.0, upper=BASE_MAX_ASSET_WEIGHT)

    # Group caps.
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

    # Total exposure cap.
    total_weight = weights.sum(axis=1)

    total_scale = (
        MAX_TOTAL_RISK_ASSET_EXPOSURE / total_weight
    ).where(total_weight > MAX_TOTAL_RISK_ASSET_EXPOSURE, 1.0)

    total_scale = total_scale.replace([np.inf, -np.inf], 1.0).fillna(1.0)

    weights = weights.mul(total_scale, axis=0)

    return weights.fillna(0.0)


def build_overlay_weights(mats: dict, params: dict) -> pd.DataFrame:
    sm = mats["score_mats"]

    final_score = build_base_final_score(mats)

    signal_quality = (
        0.35 * sm["momentum_score"]
        + 0.25 * sm["relative_strength_score"]
        + 0.20 * sm["trend_score"]
        + 0.20 * final_score
    ).clip(0, 1)

    weak_signal = (1.0 - signal_quality).clip(0, 1)

    risk_stress = sm["risk_stress_score"].clip(0, 1)
    vol_stress = sm["vol_stress_score"].clip(0, 1)

    # Asset overlay.
    risk_penalty = risk_stress * (
        params["risk_weak_penalty"] * weak_signal
        + params["risk_strong_penalty"] * signal_quality
    )

    vol_penalty = vol_stress * (
        params["vol_weak_penalty"] * weak_signal
        + params["vol_strong_penalty"] * signal_quality
    )

    risk_multiplier = (1.0 - risk_penalty).clip(lower=0.70, upper=1.0)
    vol_multiplier = (1.0 - vol_penalty).clip(lower=0.75, upper=1.0)

    asset_multiplier = (
        risk_multiplier * vol_multiplier
    ).clip(lower=params["min_asset_multiplier"], upper=1.0)

    vol_for_sizing = sm["realised_vol_60d"].clip(lower=MIN_VOL_FOR_SIZING)

    raw_signal = final_score / vol_for_sizing
    raw_signal = raw_signal.where(final_score >= BASE_MIN_SCORE_TO_HOLD, 0.0)

    raw_signal = raw_signal * asset_multiplier

    signal_sum = raw_signal.sum(axis=1)
    weights = raw_signal.div(signal_sum.replace(0, np.nan), axis=0).fillna(0.0)

    weights = apply_group_and_exposure_caps(weights, mats)

    # Optional portfolio-level overlay.
    if params["overlay_mode"] == "asset_and_portfolio":
        combined_stress = (
            0.55 * risk_stress
            + 0.45 * vol_stress
        ).clip(0, 1)

        total_weight = weights.sum(axis=1)

        weighted_stress = (weights * combined_stress).sum(axis=1)
        weighted_quality = (weights * signal_quality).sum(axis=1)

        portfolio_stress = (
            weighted_stress / total_weight.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 1)

        portfolio_quality = (
            weighted_quality / total_weight.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 1)

        portfolio_penalty = portfolio_stress * (
            params["portfolio_weak_penalty"] * (1.0 - portfolio_quality)
            + params["portfolio_strong_penalty"] * portfolio_quality
        )

        portfolio_multiplier = (
            1.0 - portfolio_penalty
        ).clip(lower=params["min_portfolio_multiplier"], upper=1.0)

        weights = weights.mul(portfolio_multiplier, axis=0)

    return weights.fillna(0.0)


# ============================================================
# BACKTEST
# ============================================================

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

    equity = (1.0 + net_return).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    if len(net_return) < 100 or net_return.std() == 0:
        return {}

    years = len(net_return) / TRADING_DAYS_PER_YEAR

    total_return = equity.iloc[-1] - 1.0
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0
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
        "final_equity_fast": INITIAL_CAPITAL * equity.iloc[-1],
    }


def passes_filters(row: dict) -> bool:
    if not np.isfinite(row.get("cagr", np.nan)):
        return False

    return (
        row["cagr"] >= MIN_ALLOWED_CAGR
        and row["sharpe"] >= MIN_ALLOWED_SHARPE
        and row["max_drawdown"] >= MAX_ALLOWED_DRAWDOWN
        and row["annualised_volatility"] <= MAX_ALLOWED_VOLATILITY
        and row["annualised_turnover"] <= MAX_ALLOWED_ANNUALISED_TURNOVER
    )


def robust_score(row: dict) -> float:
    drawdown_penalty = max(0.0, abs(row["max_drawdown"]) - 0.125) / 0.05
    cagr_shortfall_penalty = max(0.0, 0.124 - row["cagr"]) / 0.03
    turnover_penalty = max(0.0, row["annualised_turnover"] - 4.0) / 6.0

    return (
        0.40 * row["cagr"]
        + 0.30 * row["sharpe"]
        + 0.20 * row["calmar"]
        - 0.05 * drawdown_penalty
        - 0.03 * cagr_shortfall_penalty
        - 0.02 * turnover_penalty
    )


# ============================================================
# FULL ANALYTICS FOR TOP CANDIDATES
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
            "overlay_mode": row["overlay_mode"],
            "risk_weak_penalty": row["risk_weak_penalty"],
            "risk_strong_penalty": row["risk_strong_penalty"],
            "vol_weak_penalty": row["vol_weak_penalty"],
            "vol_strong_penalty": row["vol_strong_penalty"],
            "min_asset_multiplier": row["min_asset_multiplier"],
            "portfolio_weak_penalty": row["portfolio_weak_penalty"],
            "portfolio_strong_penalty": row["portfolio_strong_penalty"],
            "min_portfolio_multiplier": row["min_portfolio_multiplier"],
        }

        raw_weights = build_overlay_weights(mats, params)
        weights = apply_rebalance(raw_weights, REBALANCE_MODE)

        curve, _ = simulate_strategy(
            name=f"overlay_candidate_{idx}",
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
            strategy_name=f"overlay_candidate_{idx}",
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
# BASELINE
# ============================================================

def build_baseline_weights(mats: dict) -> pd.DataFrame:
    sm = mats["score_mats"]

    final_score = build_base_final_score(mats)

    vol_for_sizing = sm["realised_vol_60d"].clip(lower=MIN_VOL_FOR_SIZING)

    raw_signal = final_score / vol_for_sizing
    raw_signal = raw_signal.where(final_score >= BASE_MIN_SCORE_TO_HOLD, 0.0)

    signal_sum = raw_signal.sum(axis=1)
    weights = raw_signal.div(signal_sum.replace(0, np.nan), axis=0).fillna(0.0)

    weights = apply_group_and_exposure_caps(weights, mats)

    return weights.fillna(0.0)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading score data...")
    scores = load_score_data()

    print("Loading returns...")
    returns = load_return_matrix()

    print("Preparing matrices...")
    mats = prepare_matrices(scores, returns)

    print("\nRunning baseline V1-style fast backtest...")
    baseline_weights = build_baseline_weights(mats)
    baseline_metrics = fast_backtest(baseline_weights, mats["returns"])

    print("Baseline:")
    print(
        f"  CAGR:   {baseline_metrics['cagr']:.4%}\n"
        f"  Sharpe: {baseline_metrics['sharpe']:.3f}\n"
        f"  Max DD: {baseline_metrics['max_drawdown']:.2%}\n"
        f"  Equity: {baseline_metrics['final_equity_fast']:.0f}"
    )

    rows = []

    print(f"\nRunning {N_RUNS:,} overlay parameter trials...")

    for i in range(N_RUNS):
        params = sample_overlay_params()

        weights = build_overlay_weights(mats, params)
        metrics = fast_backtest(weights, mats["returns"])

        if not metrics:
            continue

        row = {**params, **metrics}
        row["passed_filters"] = passes_filters(row)
        row["score"] = robust_score(row)

        # Improvement versus baseline.
        row["cagr_vs_baseline"] = row["cagr"] - baseline_metrics["cagr"]
        row["sharpe_vs_baseline"] = row["sharpe"] - baseline_metrics["sharpe"]
        row["drawdown_vs_baseline"] = row["max_drawdown"] - baseline_metrics["max_drawdown"]
        row["equity_vs_baseline"] = row["final_equity_fast"] - baseline_metrics["final_equity_fast"]

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

    results.to_csv(OUTPUT_DIR / "overlay_parameter_search_all_fast_results.csv", index=False)
    filtered.to_csv(OUTPUT_DIR / "overlay_parameter_search_filtered_fast_results.csv", index=False)

    if filtered.empty:
        print("\nNo candidates passed filters.")
        print(f"Saved all results to: {OUTPUT_DIR}")
        return

    print("\nRunning full analytics for top candidates...")
    full = run_full_analytics_for_top(
        mats=mats,
        fast_results=filtered,
        returns=returns,
    )

    full.to_csv(OUTPUT_DIR / "overlay_parameter_search_top_full_analytics.csv", index=False)

    top_100 = filtered.head(100).copy()
    top_100.to_csv(OUTPUT_DIR / "overlay_parameter_search_top_100.csv", index=False)

    param_cols = [
        "risk_weak_penalty",
        "risk_strong_penalty",
        "vol_weak_penalty",
        "vol_strong_penalty",
        "min_asset_multiplier",
        "portfolio_weak_penalty",
        "portfolio_strong_penalty",
        "min_portfolio_multiplier",
    ]

    region = pd.DataFrame(
        {
            "mean_top_100": top_100[param_cols].mean(),
            "median_top_100": top_100[param_cols].median(),
            "std_top_100": top_100[param_cols].std(),
            "min_top_100": top_100[param_cols].min(),
            "max_top_100": top_100[param_cols].max(),
        }
    )

    region.to_csv(OUTPUT_DIR / "overlay_parameter_search_top_100_region.csv")

    display_cols = [
        "overlay_mode",
        "risk_weak_penalty",
        "risk_strong_penalty",
        "vol_weak_penalty",
        "vol_strong_penalty",
        "min_asset_multiplier",
        "portfolio_weak_penalty",
        "portfolio_strong_penalty",
        "min_portfolio_multiplier",
        "cagr",
        "sharpe",
        "calmar",
        "max_drawdown",
        "average_exposure",
        "annualised_turnover",
        "cagr_vs_baseline",
        "sharpe_vs_baseline",
        "drawdown_vs_baseline",
        "score",
    ]

    print("\nTop 20 fast overlay candidates:")
    print(filtered[display_cols].head(20).to_string(index=False))

    full_cols = [
        "overlay_mode",
        "risk_weak_penalty",
        "risk_strong_penalty",
        "vol_weak_penalty",
        "vol_strong_penalty",
        "min_asset_multiplier",
        "portfolio_weak_penalty",
        "portfolio_strong_penalty",
        "min_portfolio_multiplier",
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

    full_cols = [c for c in full_cols if c in full.columns]

    print("\nTop full-analytics candidates:")
    print(full[full_cols].head(20).to_string(index=False))

    print("\nTop 100 overlay parameter region:")
    print(region.to_string())

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()