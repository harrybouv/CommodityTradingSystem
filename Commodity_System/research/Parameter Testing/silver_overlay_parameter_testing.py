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
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# ============================================================
# IMPORTS
# ============================================================

import config
import scoring.commodity_models.Silver.SLV_scoring as slv_scoring

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

# Keep this controlled. If the first grid is promising, we can expand later.
BLEND_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25]


COMPONENT_MIXES = [
    # --------------------------------------------------------
    # Single-factor tests
    # --------------------------------------------------------
    {
        "name": "gold_ratio_only",
        "gold_ratio": 1.00,
        "copper_ratio": 0.00,
        "gold_confirmation": 0.00,
        "usd": 0.00,
        "real_yield": 0.00,
    },
    {
        "name": "copper_ratio_only",
        "gold_ratio": 0.00,
        "copper_ratio": 1.00,
        "gold_confirmation": 0.00,
        "usd": 0.00,
        "real_yield": 0.00,
    },
    {
        "name": "gold_confirmation_only",
        "gold_ratio": 0.00,
        "copper_ratio": 0.00,
        "gold_confirmation": 1.00,
        "usd": 0.00,
        "real_yield": 0.00,
    },
    {
        "name": "usd_only",
        "gold_ratio": 0.00,
        "copper_ratio": 0.00,
        "gold_confirmation": 0.00,
        "usd": 1.00,
        "real_yield": 0.00,
    },
    {
        "name": "real_yield_only",
        "gold_ratio": 0.00,
        "copper_ratio": 0.00,
        "gold_confirmation": 0.00,
        "usd": 0.00,
        "real_yield": 1.00,
    },

    # --------------------------------------------------------
    # Main candidates based on early results
    # --------------------------------------------------------
    {
        "name": "real_goldconf_50_50",
        "gold_ratio": 0.00,
        "copper_ratio": 0.00,
        "gold_confirmation": 0.50,
        "usd": 0.00,
        "real_yield": 0.50,
    },
    {
        "name": "real_heavy_goldconf",
        "gold_ratio": 0.00,
        "copper_ratio": 0.00,
        "gold_confirmation": 0.35,
        "usd": 0.00,
        "real_yield": 0.65,
    },
    {
        "name": "goldconf_heavy_real",
        "gold_ratio": 0.00,
        "copper_ratio": 0.00,
        "gold_confirmation": 0.65,
        "usd": 0.00,
        "real_yield": 0.35,
    },
    {
        "name": "real_goldconf_copper",
        "gold_ratio": 0.00,
        "copper_ratio": 0.15,
        "gold_confirmation": 0.35,
        "usd": 0.00,
        "real_yield": 0.50,
    },
    {
        "name": "real_heavy_copper_light",
        "gold_ratio": 0.00,
        "copper_ratio": 0.10,
        "gold_confirmation": 0.25,
        "usd": 0.00,
        "real_yield": 0.65,
    },

    # --------------------------------------------------------
    # Theory candidates
    # --------------------------------------------------------
    {
        "name": "theory_core",
        "gold_ratio": 0.30,
        "copper_ratio": 0.25,
        "gold_confirmation": 0.20,
        "usd": 0.15,
        "real_yield": 0.10,
    },
    {
        "name": "macro_light",
        "gold_ratio": 0.10,
        "copper_ratio": 0.20,
        "gold_confirmation": 0.35,
        "usd": 0.05,
        "real_yield": 0.30,
    },
    {
        "name": "macro_heavy",
        "gold_ratio": 0.00,
        "copper_ratio": 0.10,
        "gold_confirmation": 0.25,
        "usd": 0.20,
        "real_yield": 0.45,
    },
    {
        "name": "no_usd_no_goldratio",
        "gold_ratio": 0.00,
        "copper_ratio": 0.20,
        "gold_confirmation": 0.35,
        "usd": 0.00,
        "real_yield": 0.45,
    },
    {
        "name": "small_goldratio_test",
        "gold_ratio": 0.10,
        "copper_ratio": 0.10,
        "gold_confirmation": 0.30,
        "usd": 0.00,
        "real_yield": 0.50,
    },
]


OUTPUT_DIR = RESULTS_DIR / "silver_overlay_parameter_tests"


# ============================================================
# PARAMETER PATCHING
# ============================================================

def set_silver_overlay_params(
    enabled: bool,
    blend_weight: float,
    gold_ratio_weight: float,
    copper_ratio_weight: float,
    gold_confirmation_weight: float,
    usd_weight: float,
    real_yield_weight: float,
) -> None:
    """
    Patch both config and SLV_scoring module variables.

    This matters because SLV_scoring imports config values into module-level
    constants at import time. Editing config alone inside a loop is not enough.
    """

    component_weights = {
        "silver_gold_ratio_score": float(gold_ratio_weight),
        "silver_copper_ratio_score": float(copper_ratio_weight),
        "silver_gold_confirmation_score": float(gold_confirmation_weight),
        "silver_usd_score": float(usd_weight),
        "silver_real_yield_score": float(real_yield_weight),
    }

    config.SILVER_OVERLAY_ENABLED = enabled
    config.SILVER_OVERLAY_BLEND_WEIGHT = float(blend_weight)
    config.SILVER_USE_GOLD_RATIO = gold_ratio_weight > 0
    config.SILVER_USE_COPPER_RATIO = copper_ratio_weight > 0
    config.SILVER_USE_GOLD_CONFIRMATION = gold_confirmation_weight > 0
    config.SILVER_USE_USD = usd_weight > 0
    config.SILVER_USE_REAL_YIELD = real_yield_weight > 0
    config.SILVER_COMPONENT_WEIGHTS = component_weights

    slv_scoring.SILVER_OVERLAY_ENABLED = enabled
    slv_scoring.SILVER_OVERLAY_BLEND_WEIGHT = float(blend_weight)
    slv_scoring.SILVER_USE_GOLD_RATIO = gold_ratio_weight > 0
    slv_scoring.SILVER_USE_COPPER_RATIO = copper_ratio_weight > 0
    slv_scoring.SILVER_USE_GOLD_CONFIRMATION = gold_confirmation_weight > 0
    slv_scoring.SILVER_USE_USD = usd_weight > 0
    slv_scoring.SILVER_USE_REAL_YIELD = real_yield_weight > 0
    slv_scoring.SILVER_COMPONENT_WEIGHTS = component_weights


# ============================================================
# BENCHMARK HELPERS
# ============================================================

def make_single_asset(
    index: pd.Index,
    tickers: list[str],
    asset: str,
) -> pd.DataFrame:
    weights = pd.DataFrame(
        0.0,
        index=index,
        columns=tickers,
    )

    if asset in weights.columns:
        weights[asset] = 1.0

    return weights


# ============================================================
# ONE BACKTEST
# ============================================================

def run_one_test(
    name: str,
    score_inputs: pd.DataFrame,
    returns: pd.DataFrame,
    blend_weight: float,
    gold_ratio_weight: float,
    copper_ratio_weight: float,
    gold_confirmation_weight: float,
    usd_weight: float,
    real_yield_weight: float,
    overlay_enabled: bool = True,
) -> dict:
    set_silver_overlay_params(
        enabled=overlay_enabled,
        blend_weight=blend_weight,
        gold_ratio_weight=gold_ratio_weight,
        copper_ratio_weight=copper_ratio_weight,
        gold_confirmation_weight=gold_confirmation_weight,
        usd_weight=usd_weight,
        real_yield_weight=real_yield_weight,
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
    silver_only = make_single_asset(model_weights.index, tickers, "SLV")
    cash = make_cash(model_weights.index, tickers)

    strategies = {
        "model": model_weights,
        "equal_weight": equal_weight,
        "gold_only": gold_only,
        "silver_only": silver_only,
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

    model_row = performance[
        performance["strategy"] == "model"
    ].iloc[0].to_dict()

    asset_contribution = pd.concat(contributions, ignore_index=True)
    model_contrib = asset_contribution[
        asset_contribution["strategy"] == "model"
    ].copy()

    slv_contribution = np.nan
    slv_contribution_share_abs = np.nan

    if not model_contrib.empty and "SLV" in model_contrib["ticker"].values:
        slv_row = model_contrib[model_contrib["ticker"] == "SLV"].iloc[0]
        slv_contribution = slv_row.get("total_return_contribution", np.nan)
        slv_contribution_share_abs = slv_row.get("contribution_share_abs", np.nan)

    slv_avg_weight = (
        model_weights["SLV"].mean()
        if "SLV" in model_weights.columns
        else np.nan
    )

    slv_max_weight = (
        model_weights["SLV"].max()
        if "SLV" in model_weights.columns
        else np.nan
    )

    slv_months_held = (
        (model_weights["SLV"].resample("ME").last() > 0).sum()
        if "SLV" in model_weights.columns
        else np.nan
    )

    return {
        "test_name": name,
        "overlay_enabled": overlay_enabled,
        "blend_weight": blend_weight,

        "gold_ratio_weight": gold_ratio_weight,
        "copper_ratio_weight": copper_ratio_weight,
        "gold_confirmation_weight": gold_confirmation_weight,
        "usd_weight": usd_weight,
        "real_yield_weight": real_yield_weight,

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

        "slv_total_return_contribution": slv_contribution,
        "slv_contribution_share_abs": slv_contribution_share_abs,
        "slv_avg_weight": slv_avg_weight,
        "slv_max_weight": slv_max_weight,
        "slv_months_held": slv_months_held,
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

    print("\nRunning baseline with silver overlay disabled...")

    rows.append(
        run_one_test(
            name="baseline_no_silver_overlay",
            score_inputs=score_inputs,
            returns=returns,
            blend_weight=0.0,
            gold_ratio_weight=0.0,
            copper_ratio_weight=0.0,
            gold_confirmation_weight=0.0,
            usd_weight=0.0,
            real_yield_weight=0.0,
            overlay_enabled=False,
        )
    )

    total_tests = len(BLEND_WEIGHTS) * len(COMPONENT_MIXES)
    done = 0

    print(f"\nRunning {total_tests} silver overlay parameter tests...")

    for blend_weight in BLEND_WEIGHTS:
        for mix in COMPONENT_MIXES:
            done += 1

            name = f"{mix['name']}_blend_{blend_weight:.2f}"

            print(
                f"[{done:03d}/{total_tests}] {name} "
                f"(gold_ratio={mix['gold_ratio']:.2f}, "
                f"copper={mix['copper_ratio']:.2f}, "
                f"gold_conf={mix['gold_confirmation']:.2f}, "
                f"usd={mix['usd']:.2f}, "
                f"real={mix['real_yield']:.2f})"
            )

            row = run_one_test(
                name=name,
                score_inputs=score_inputs,
                returns=returns,
                blend_weight=blend_weight,
                gold_ratio_weight=mix["gold_ratio"],
                copper_ratio_weight=mix["copper_ratio"],
                gold_confirmation_weight=mix["gold_confirmation"],
                usd_weight=mix["usd"],
                real_yield_weight=mix["real_yield"],
                overlay_enabled=True,
            )

            rows.append(row)

    results = pd.DataFrame(rows)

    baseline = results[
        results["test_name"] == "baseline_no_silver_overlay"
    ].iloc[0]

    results["delta_cagr_vs_baseline"] = results["cagr"] - baseline["cagr"]
    results["delta_sharpe_vs_baseline"] = results["sharpe"] - baseline["sharpe"]
    results["delta_sortino_vs_baseline"] = results["sortino"] - baseline["sortino"]
    results["delta_calmar_vs_baseline"] = results["calmar"] - baseline["calmar"]
    results["delta_maxdd_vs_baseline"] = results["max_drawdown"] - baseline["max_drawdown"]
    results["delta_final_equity_vs_baseline"] = results["final_equity"] - baseline["final_equity"]

    # Ranking: higher is better for all included columns.
    # max_drawdown is better when less negative, i.e. numerically higher.
    results["rank_score"] = (
        results["sharpe"].rank(ascending=True, pct=True)
        + results["sortino"].rank(ascending=True, pct=True)
        + results["calmar"].rank(ascending=True, pct=True)
        + results["max_drawdown"].rank(ascending=True, pct=True)
        + results["cagr"].rank(ascending=True, pct=True)
    ) / 5.0

    # Slight penalty for solutions that get better only by massively increasing SLV exposure.
    if "slv_avg_weight" in results.columns:
        baseline_slv_weight = baseline["slv_avg_weight"]
        results["delta_slv_avg_weight_vs_baseline"] = (
            results["slv_avg_weight"] - baseline_slv_weight
        )

        results["rank_score_adjusted"] = (
            results["rank_score"]
            - 0.10 * results["delta_slv_avg_weight_vs_baseline"].clip(lower=0.0)
        )
    else:
        results["rank_score_adjusted"] = results["rank_score"]

    results = results.sort_values(
        ["rank_score_adjusted", "sharpe", "calmar", "cagr"],
        ascending=False,
    ).reset_index(drop=True)

    output_path = OUTPUT_DIR / "silver_overlay_parameter_summary.csv"
    results.to_csv(output_path, index=False)

    print("\nSaved results to:")
    print(output_path)

    display_cols = [
        "test_name",
        "blend_weight",
        "gold_ratio_weight",
        "copper_ratio_weight",
        "gold_confirmation_weight",
        "usd_weight",
        "real_yield_weight",
        "final_equity",
        "cagr",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "average_exposure",
        "average_cash",
        "slv_total_return_contribution",
        "slv_avg_weight",
        "slv_max_weight",
        "delta_cagr_vs_baseline",
        "delta_sharpe_vs_baseline",
        "delta_maxdd_vs_baseline",
        "rank_score_adjusted",
    ]

    available_display_cols = [
        col for col in display_cols
        if col in results.columns
    ]

    print("\nTop 15 candidates:")
    print(results[available_display_cols].head(15).to_string(index=False))

    print("\nBaseline row:")
    print(
        results[
            results["test_name"] == "baseline_no_silver_overlay"
        ][available_display_cols].to_string(index=False)
    )


if __name__ == "__main__":
    main()