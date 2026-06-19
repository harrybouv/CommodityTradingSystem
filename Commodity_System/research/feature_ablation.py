# research/feature_ablation.py

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
    MAX_ASSET_WEIGHT,
    MIN_SCORE_TO_HOLD,
    BACKTEST_REBALANCE_MODE,
)

from analytics import (
    calculate_full_summary,
    calculate_alpha_beta,
)

from backtester import (
    load_return_matrix,
    apply_rebalance,
    make_equal_weight,
    make_gold_only,
    make_cash,
    simulate_strategy,
)

from commodity_strategy import (
    load_score_inputs,
    build_strategy_weights_from_spec,
    weights_long_to_matrix,
)

REBALANCE_MODES = [BACKTEST_REBALANCE_MODE]
OUTPUT_DIR = RESULTS_DIR / "ablation"

MIN_VOL_FOR_SIZING = 0.05
USE_INVERSE_VOL_SIZING = True

RISK_FREE_RATE_ANNUAL = 0.0

ABLATIONS = {
    "momentum_only": {
        "weights": {
            "momentum_score": 1.0,
        },
        "normalise": True,
    },

    "trend_only": {
        "weights": {
            "trend_score": 1.0,
        },
        "normalise": True,
    },

    "volatility_only": {
        "weights": {
            "volatility_score": 1.0,
        },
        "normalise": True,
    },

    "risk_only": {
        "weights": {
            "risk_score": 1.0,
        },
        "normalise": True,
    },

    "momentum_trend": {
        "weights": {
            "momentum_score": 0.60,
            "trend_score": 0.40,
        },
        "normalise": True,
    },

    "momentum_trend_risk": {
        "weights": {
            "momentum_score": 0.45,
            "trend_score": 0.30,
            "risk_score": 0.25,
        },
        "normalise": True,
    },

    "momentum_trend_volatility": {
        "weights": {
            "momentum_score": 0.45,
            "trend_score": 0.30,
            "volatility_score": 0.25,
        },
        "normalise": True,
    },

    "full_v0_old_raw_weights": {
        "weights": {
            "momentum_score": 0.30,
            "trend_score": 0.05,
            "volatility_score": 0.45,
            "risk_score": 0.30,
        },
        "normalise": False,
    },

    "full_v0_normalised": {
        "weights": {
            "momentum_score": 0.15,
            "trend_score": 0.03,
            "volatility_score": 0.25,
            "risk_score": 0.57,
        },
        "normalise": True,
    },

    "signal_x_risk_modifier": {
        "custom": "signal_x_risk_modifier",
    },

    "momentum_x_risk": {
        "custom": "momentum_x_risk",
    },

    "momentum_x_volatility": {
        "custom": "momentum_x_volatility",
    },
}


def build_summary_row(
    strategy_name: str,
    curve: pd.DataFrame,
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


def build_alpha_beta_table(
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

            rows.append(
                {
                    "strategy": strategy_name,
                    "benchmark": benchmark_name,
                    **stats,
                }
            )

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            ["benchmark", "alpha_annualised", "information_ratio"],
            ascending=[True, False, False],
        )

    return out


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scores = load_score_inputs()
    returns = load_return_matrix()

    tickers = sorted(scores["ticker"].unique())
    base_index = pd.Index(sorted(scores["date"].unique()))

    curves = {}
    contribution_frames = []
    summary_rows = []

    benchmark_weights = {
        "equal_weight": make_equal_weight(base_index, tickers),
        "gold_only": make_gold_only(base_index, tickers),
        "cash": make_cash(base_index, tickers),
    }

    # -------------------------
    # Benchmarks
    # -------------------------

    for benchmark_name, weights in benchmark_weights.items():
        for mode in REBALANCE_MODES:
            rebalanced = apply_rebalance(weights, mode)
            name = f"{benchmark_name}_{mode}"

            curve, contribution = simulate_strategy(
                name=name,
                target_weights=rebalanced,
                returns=returns,
                initial_capital=INITIAL_CAPITAL,
                total_cost_bps=TOTAL_COST_BPS,
            )

            curves[name] = curve
            contribution_frames.append(contribution)

            summary = build_summary_row(
                strategy_name=name,
                curve=curve,
                benchmark_curve=None,
                benchmark_name=None,
            )

            summary["ablation_type"] = "benchmark"
            summary["rebalance"] = mode

            summary_rows.append(summary)

    # -------------------------
    # Model ablations
    # -------------------------

    primary_benchmark_name = f"equal_weight_{REBALANCE_MODES[0]}"
    primary_benchmark_curve = curves.get(primary_benchmark_name)

    for ablation_name, spec in ABLATIONS.items():
        print(f"Running ablation: {ablation_name}", flush=True)

        weights_long = build_strategy_weights_from_spec(scores, spec)
        raw_weights = weights_long_to_matrix(weights_long)

        for mode in REBALANCE_MODES:
            print(f"  Rebalance: {mode}", flush=True)

            rebalanced = apply_rebalance(raw_weights, mode)
            name = f"{ablation_name}_{mode}"

            curve, contribution = simulate_strategy(
                name=name,
                target_weights=rebalanced,
                returns=returns,
                initial_capital=INITIAL_CAPITAL,
                total_cost_bps=TOTAL_COST_BPS,
            )

            curves[name] = curve
            contribution_frames.append(contribution)

            summary = build_summary_row(
                strategy_name=name,
                curve=curve,
                benchmark_curve=primary_benchmark_curve,
                benchmark_name=primary_benchmark_name,
            )

            summary["ablation_type"] = "model"
            summary["rebalance"] = mode

            summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        ["sharpe", "calmar", "cagr"],
        ascending=False,
    )

    curves_df = pd.concat(
        [
            curve.reset_index()
            for curve in curves.values()
        ],
        ignore_index=True,
    )

    contributions_df = pd.concat(
        contribution_frames,
        ignore_index=True,
    )

    alpha_beta_df = build_alpha_beta_table(
        curves=curves,
        benchmark_names=[
            f"equal_weight_{REBALANCE_MODES[0]}",
            f"gold_only_{REBALANCE_MODES[0]}",
        ],
    )

    summary_df.to_csv(OUTPUT_DIR / "ablation_summary.csv", index=False)
    curves_df.to_csv(OUTPUT_DIR / "ablation_curves.csv", index=False)
    contributions_df.to_csv(OUTPUT_DIR / "ablation_asset_contribution.csv", index=False)
    alpha_beta_df.to_csv(OUTPUT_DIR / "ablation_alpha_beta.csv", index=False)

    print("\nAblation complete.")
    print(f"Saved outputs to: {OUTPUT_DIR}")

    cols = [
        "strategy",
        "ablation_type",
        "cagr",
        "annualised_volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "hit_rate",
        "average_exposure",
        "average_cash",
        "average_daily_turnover",
        "alpha_annualised",
        "beta",
        "r_squared",
        "information_ratio",
    ]

    cols = [col for col in cols if col in summary_df.columns]

    print("\nTop ablation results:")
    print(summary_df[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()