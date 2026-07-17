from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

THIS_FILE = Path(__file__).resolve()
COMMODITY_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]

for path in [COMMODITY_ROOT, REPO_ROOT, THIS_FILE.parent]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ============================================================
# IMPORTS
# ============================================================

import config
import scoring.commodity_models.Gold.GLD_scoring as gld_scoring

from config import RESULTS_DIR
from commodity_strategy import load_score_inputs, build_production_strategy_weight_matrix

from Commodity_System.research.Backtesting.backtester import (
    load_return_matrix,
    apply_rebalance,
    apply_volatility_targeting,
    simulate_strategy,
    make_equal_weight,
    make_gold_only,
    make_cash,
    build_performance_summary,
    REBALANCE_MODE,
    INITIAL_CAPITAL,
    TOTAL_COST_BPS,
)


# ============================================================
# TEST GRID
# ============================================================

BLEND_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

COMPONENT_MIXES = [
    {
        "name": "usd_only",
        "real": 0.00,
        "usd": 1.00,
        "stress": 0.00,
    },
    {
        "name": "real_only",
        "real": 1.00,
        "usd": 0.00,
        "stress": 0.00,
    },
    {
        "name": "stress_only",
        "real": 0.00,
        "usd": 0.00,
        "stress": 1.00,
    },
    {
        "name": "theory_core",
        "real": 0.45,
        "usd": 0.35,
        "stress": 0.20,
    },
    {
        "name": "balanced",
        "real": 0.40,
        "usd": 0.40,
        "stress": 0.20,
    },
    {
        "name": "usd_heavy",
        "real": 0.20,
        "usd": 0.65,
        "stress": 0.15,
    },
    {
        "name": "stress_light",
        "real": 0.40,
        "usd": 0.50,
        "stress": 0.10,
    },
    {
        "name": "no_stress",
        "real": 0.50,
        "usd": 0.50,
        "stress": 0.00,
    },
    {
        "name": "real_heavy",
        "real": 0.60,
        "usd": 0.30,
        "stress": 0.10,
    },
]


OUTPUT_DIR = RESULTS_DIR / "gold_overlay_parameter_tests"


# ============================================================
# PARAMETER PATCHING
# ============================================================

def set_gold_overlay_params(
    enabled: bool,
    blend_weight: float,
    real_weight: float,
    usd_weight: float,
    stress_weight: float,
) -> None:
    """
    Patch both config and GLD_scoring module variables.

    This matters because GLD_scoring imports config values into module-level
    constants at import time. Editing config alone during a loop would not be
    enough.
    """

    component_weights = {
        "gold_real_yield_score": float(real_weight),
        "gold_usd_score": float(usd_weight),
        "gold_stress_score": float(stress_weight),
        "gold_policy_rate_score": 0.0,
        "gold_central_bank_score": 0.0,
        "gold_positioning_score": 0.0,
    }

    config.GOLD_OVERLAY_ENABLED = enabled
    config.GOLD_OVERLAY_BLEND_WEIGHT = float(blend_weight)
    config.GOLD_USE_REAL_YIELD = real_weight > 0
    config.GOLD_USE_USD = usd_weight > 0
    config.GOLD_USE_STRESS = stress_weight > 0
    config.GOLD_USE_POLICY_RATE_REGIME = False
    config.GOLD_USE_CENTRAL_BANK_DEMAND = False
    config.GOLD_USE_POSITIONING_CROWDING = False
    config.GOLD_COMPONENT_WEIGHTS = component_weights

    gld_scoring.GOLD_OVERLAY_ENABLED = enabled
    gld_scoring.GOLD_OVERLAY_BLEND_WEIGHT = float(blend_weight)
    gld_scoring.GOLD_USE_REAL_YIELD = real_weight > 0
    gld_scoring.GOLD_USE_USD = usd_weight > 0
    gld_scoring.GOLD_USE_STRESS = stress_weight > 0
    gld_scoring.GOLD_USE_POLICY_RATE_REGIME = False
    gld_scoring.GOLD_USE_CENTRAL_BANK_DEMAND = False
    gld_scoring.GOLD_USE_POSITIONING_CROWDING = False
    gld_scoring.GOLD_COMPONENT_WEIGHTS = component_weights


# ============================================================
# ONE BACKTEST
# ============================================================

def run_one_test(
    name: str,
    score_inputs: pd.DataFrame,
    returns: pd.DataFrame,
    blend_weight: float,
    real_weight: float,
    usd_weight: float,
    stress_weight: float,
    overlay_enabled: bool = True,
) -> dict:
    set_gold_overlay_params(
        enabled=overlay_enabled,
        blend_weight=blend_weight,
        real_weight=real_weight,
        usd_weight=usd_weight,
        stress_weight=stress_weight,
    )

    raw_weights = build_production_strategy_weight_matrix(scores=score_inputs)

    model_weights = apply_rebalance(
        raw_weights=raw_weights,
        mode=REBALANCE_MODE,
    )

    model_weights, _ = apply_volatility_targeting(
        weights=model_weights,
        returns=returns,
    )

    tickers = list(model_weights.columns)

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
    contributions = []

    for strategy_name, weights in strategies.items():
        curve, asset_contribution = simulate_strategy(
            name=strategy_name,
            target_weights=weights,
            returns=returns,
            initial_capital=INITIAL_CAPITAL,
            total_cost_bps=TOTAL_COST_BPS,
        )

        curves[strategy_name] = curve
        contributions.append(asset_contribution)

    performance = build_performance_summary(
        curves=curves,
        benchmark_name="equal_weight",
    )

    model_row = performance[performance["strategy"] == "model"].iloc[0].to_dict()

    asset_contribution = pd.concat(contributions, ignore_index=True)
    model_contrib = asset_contribution[asset_contribution["strategy"] == "model"].copy()

    gld_contribution = np.nan
    gld_contribution_share_abs = np.nan

    if not model_contrib.empty and "GLD" in model_contrib["ticker"].values:
        gld_row = model_contrib[model_contrib["ticker"] == "GLD"].iloc[0]
        gld_contribution = gld_row.get("total_return_contribution", np.nan)
        gld_contribution_share_abs = gld_row.get("contribution_share_abs", np.nan)

    gld_avg_weight = (
        model_weights["GLD"].mean()
        if "GLD" in model_weights.columns
        else np.nan
    )

    gld_max_weight = (
        model_weights["GLD"].max()
        if "GLD" in model_weights.columns
        else np.nan
    )

    gld_months_held = (
        (model_weights["GLD"].resample("ME").last() > 0).sum()
        if "GLD" in model_weights.columns
        else np.nan
    )

    return {
        "test_name": name,
        "overlay_enabled": overlay_enabled,
        "blend_weight": blend_weight,
        "real_yield_weight": real_weight,
        "usd_weight": usd_weight,
        "stress_weight": stress_weight,

        "final_equity": model_row.get("final_equity", np.nan),
        "cagr": model_row.get("cagr", np.nan),
        "annualised_volatility": model_row.get("annualised_volatility", np.nan),
        "sharpe": model_row.get("sharpe", np.nan),
        "sortino": model_row.get("sortino", np.nan),
        "calmar": model_row.get("calmar", np.nan),
        "max_drawdown": model_row.get("max_drawdown", np.nan),
        "hit_rate": model_row.get("hit_rate", np.nan),
        "average_daily_turnover": model_row.get("average_daily_turnover", np.nan),
        "annualised_turnover": model_row.get("annualised_turnover", np.nan),
        "total_transaction_cost_drag": model_row.get("total_transaction_cost_drag", np.nan),
        "average_exposure": model_row.get("average_exposure", np.nan),
        "average_cash": model_row.get("average_cash", np.nan),
        "alpha_annualised": model_row.get("alpha_annualised", np.nan),
        "beta": model_row.get("beta", np.nan),
        "information_ratio": model_row.get("information_ratio", np.nan),

        "gld_total_return_contribution": gld_contribution,
        "gld_contribution_share_abs": gld_contribution_share_abs,
        "gld_avg_weight": gld_avg_weight,
        "gld_max_weight": gld_max_weight,
        "gld_months_held": gld_months_held,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading existing score inputs and returns once...")
    print("This avoids rerunning data/scoring pipelines for every parameter set.")

    score_inputs = load_score_inputs()
    returns = load_return_matrix()

    rows = []

    print("\nRunning baseline with gold overlay disabled...")

    rows.append(
        run_one_test(
            name="baseline_no_overlay",
            score_inputs=score_inputs,
            returns=returns,
            blend_weight=0.0,
            real_weight=0.0,
            usd_weight=0.0,
            stress_weight=0.0,
            overlay_enabled=False,
        )
    )

    total_tests = len(BLEND_WEIGHTS) * len(COMPONENT_MIXES)
    done = 0

    print(f"\nRunning {total_tests} gold overlay parameter tests...")

    for blend_weight in BLEND_WEIGHTS:
        for mix in COMPONENT_MIXES:
            done += 1

            name = f"{mix['name']}_blend_{blend_weight:.2f}"

            print(
                f"[{done:02d}/{total_tests}] {name} "
                f"(real={mix['real']:.2f}, usd={mix['usd']:.2f}, "
                f"stress={mix['stress']:.2f})"
            )

            row = run_one_test(
                name=name,
                score_inputs=score_inputs,
                returns=returns,
                blend_weight=blend_weight,
                real_weight=mix["real"],
                usd_weight=mix["usd"],
                stress_weight=mix["stress"],
                overlay_enabled=True,
            )

            rows.append(row)

    results = pd.DataFrame(rows)

    # Ranking: avoid pure CAGR-chasing. This rewards risk-adjusted quality.
    results["rank_score"] = (
        results["sharpe"].rank(ascending=False, pct=True)
        + results["sortino"].rank(ascending=False, pct=True)
        + results["calmar"].rank(ascending=False, pct=True)
        + results["max_drawdown"].rank(ascending=False, pct=True)  # less negative is better
        + results["cagr"].rank(ascending=False, pct=True)
    ) / 5.0

    results = results.sort_values(
        ["rank_score", "sharpe", "calmar", "cagr"],
        ascending=False,
    ).reset_index(drop=True)

    output_path = OUTPUT_DIR / "gold_overlay_parameter_summary.csv"
    results.to_csv(output_path, index=False)

    print("\nSaved results to:")
    print(output_path)

    display_cols = [
        "test_name",
        "blend_weight",
        "real_yield_weight",
        "usd_weight",
        "stress_weight",
        "final_equity",
        "cagr",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "average_exposure",
        "average_cash",
        "gld_total_return_contribution",
        "gld_avg_weight",
        "gld_max_weight",
        "rank_score",
    ]

    print("\nTop 12 candidates:")
    print(results[display_cols].head(12).to_string(index=False))

    print("\nBaseline row:")
    print(
        results[results["test_name"] == "baseline_no_overlay"][display_cols]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()