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
import scoring.commodity_models.registry as commodity_registry

from config import RESULTS_DIR

from commodity_strategy import (
    load_score_inputs,
    build_production_strategy_weight_matrix,
)

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
# RESOLVE ACTIVE USO SCORING MODULE
# ============================================================

def get_active_uso_scoring_module():
    """
    Finds the actual module used by the registry for USO.

    This avoids fragile assumptions about whether USO_scoring.py lives at:
    - scoring/commodity_models/USO_scoring.py
    - scoring/commodity_models/Oil/USO_scoring.py

    The registry decides what is actually used in production. We patch that.
    """
    scorer = commodity_registry.COMMODITY_SCORERS.get("USO")

    if scorer is None:
        raise ValueError("Registry has no USO scorer.")

    module_name = scorer.__module__

    if module_name not in sys.modules:
        __import__(module_name)

    return sys.modules[module_name]


uso_scoring = get_active_uso_scoring_module()


# ============================================================
# TEST GRID
# ============================================================

# Wider than gold/silver/copper because the individual USO tests were strong.
# We deliberately test aggressive overlay weights up to 50/50 base-vs-overlay.
BLEND_WEIGHTS = [
    0.03,
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.50,
]


COMPONENT_MIXES = [
    # --------------------------------------------------------
    # Single-factor anchors
    # These confirm the individual tests inside one framework.
    # --------------------------------------------------------
    {
        "name": "inventory_only",
        "inventory": 1.00,
        "cushing": 0.00,
        "curve": 0.00,
        "refinery": 0.00,
        "demand": 0.00,
        "usd": 0.00,
    },
    {
        "name": "cushing_only",
        "inventory": 0.00,
        "cushing": 1.00,
        "curve": 0.00,
        "refinery": 0.00,
        "demand": 0.00,
        "usd": 0.00,
    },
    {
        "name": "curve_only",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 1.00,
        "refinery": 0.00,
        "demand": 0.00,
        "usd": 0.00,
    },
    {
        "name": "refinery_only",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.00,
        "refinery": 1.00,
        "demand": 0.00,
        "usd": 0.00,
    },
    {
        "name": "demand_only",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.00,
        "refinery": 0.00,
        "demand": 1.00,
        "usd": 0.00,
    },
    {
        "name": "usd_only",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.00,
        "refinery": 0.00,
        "demand": 0.00,
        "usd": 1.00,
    },

    # --------------------------------------------------------
    # Original theory mixes
    # --------------------------------------------------------
    {
        "name": "theory_original",
        "inventory": 0.30,
        "cushing": 0.20,
        "curve": 0.25,
        "refinery": 0.15,
        "demand": 0.07,
        "usd": 0.03,
    },
    {
        "name": "theory_but_usd_heavy",
        "inventory": 0.10,
        "cushing": 0.05,
        "curve": 0.30,
        "refinery": 0.25,
        "demand": 0.10,
        "usd": 0.20,
    },
    {
        "name": "theory_but_market_led",
        "inventory": 0.05,
        "cushing": 0.05,
        "curve": 0.30,
        "refinery": 0.30,
        "demand": 0.10,
        "usd": 0.20,
    },
    {
        "name": "balanced_all",
        "inventory": 1 / 6,
        "cushing": 1 / 6,
        "curve": 1 / 6,
        "refinery": 1 / 6,
        "demand": 1 / 6,
        "usd": 1 / 6,
    },

    # --------------------------------------------------------
    # Result-led core candidates
    # Individual tests suggested: USD, refinery, curve strongest;
    # demand useful; inventory/cushing weak as standalone predictors.
    # --------------------------------------------------------
    {
        "name": "curve_refinery_usd_equal",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.3333,
        "refinery": 0.3333,
        "demand": 0.00,
        "usd": 0.3334,
    },
    {
        "name": "curve_refinery_usd_demand",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.30,
        "refinery": 0.30,
        "demand": 0.10,
        "usd": 0.30,
    },
    {
        "name": "usd_refinery_curve_demand",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.25,
        "refinery": 0.30,
        "demand": 0.10,
        "usd": 0.35,
    },
    {
        "name": "usd_heavy_core",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.20,
        "refinery": 0.25,
        "demand": 0.10,
        "usd": 0.45,
    },
    {
        "name": "very_usd_heavy",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.15,
        "refinery": 0.20,
        "demand": 0.05,
        "usd": 0.60,
    },
    {
        "name": "refinery_heavy_core",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.25,
        "refinery": 0.45,
        "demand": 0.10,
        "usd": 0.20,
    },
    {
        "name": "curve_heavy_core",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.45,
        "refinery": 0.25,
        "demand": 0.10,
        "usd": 0.20,
    },
    {
        "name": "curve_refinery_only",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.50,
        "refinery": 0.50,
        "demand": 0.00,
        "usd": 0.00,
    },
    {
        "name": "curve_usd_only",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.50,
        "refinery": 0.00,
        "demand": 0.00,
        "usd": 0.50,
    },
    {
        "name": "refinery_usd_only",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.00,
        "refinery": 0.50,
        "demand": 0.00,
        "usd": 0.50,
    },

    # --------------------------------------------------------
    # Demand as supporting confirmation
    # --------------------------------------------------------
    {
        "name": "demand_usd",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.00,
        "refinery": 0.00,
        "demand": 0.35,
        "usd": 0.65,
    },
    {
        "name": "demand_curve_usd",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.30,
        "refinery": 0.00,
        "demand": 0.20,
        "usd": 0.50,
    },
    {
        "name": "demand_refinery_usd",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.00,
        "refinery": 0.35,
        "demand": 0.20,
        "usd": 0.45,
    },
    {
        "name": "demand_balanced_market",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.25,
        "refinery": 0.25,
        "demand": 0.25,
        "usd": 0.25,
    },

    # --------------------------------------------------------
    # Small fundamental guardrail tests
    # These test whether inventory/cushing help when kept small.
    # --------------------------------------------------------
    {
        "name": "small_inventory_no_cushing",
        "inventory": 0.05,
        "cushing": 0.00,
        "curve": 0.30,
        "refinery": 0.30,
        "demand": 0.10,
        "usd": 0.25,
    },
    {
        "name": "small_cushing_no_inventory",
        "inventory": 0.00,
        "cushing": 0.05,
        "curve": 0.30,
        "refinery": 0.30,
        "demand": 0.10,
        "usd": 0.25,
    },
    {
        "name": "small_inventory_cushing",
        "inventory": 0.05,
        "cushing": 0.05,
        "curve": 0.30,
        "refinery": 0.25,
        "demand": 0.10,
        "usd": 0.25,
    },
    {
        "name": "inventory_light_market_core",
        "inventory": 0.10,
        "cushing": 0.00,
        "curve": 0.30,
        "refinery": 0.25,
        "demand": 0.10,
        "usd": 0.25,
    },
    {
        "name": "cushing_light_market_core",
        "inventory": 0.00,
        "cushing": 0.10,
        "curve": 0.30,
        "refinery": 0.25,
        "demand": 0.10,
        "usd": 0.25,
    },
    {
        "name": "fundamentals_light_market_core",
        "inventory": 0.10,
        "cushing": 0.05,
        "curve": 0.30,
        "refinery": 0.25,
        "demand": 0.10,
        "usd": 0.20,
    },

    # --------------------------------------------------------
    # Fundamental block tests
    # Mainly to prove whether inventory/cushing deserve production weight.
    # --------------------------------------------------------
    {
        "name": "inventory_cushing_equal",
        "inventory": 0.50,
        "cushing": 0.50,
        "curve": 0.00,
        "refinery": 0.00,
        "demand": 0.00,
        "usd": 0.00,
    },
    {
        "name": "inventory_cushing_curve",
        "inventory": 0.25,
        "cushing": 0.25,
        "curve": 0.50,
        "refinery": 0.00,
        "demand": 0.00,
        "usd": 0.00,
    },
    {
        "name": "inventory_cushing_refinery",
        "inventory": 0.25,
        "cushing": 0.25,
        "curve": 0.00,
        "refinery": 0.50,
        "demand": 0.00,
        "usd": 0.00,
    },
    {
        "name": "inventory_cushing_usd",
        "inventory": 0.25,
        "cushing": 0.25,
        "curve": 0.00,
        "refinery": 0.00,
        "demand": 0.00,
        "usd": 0.50,
    },

    # --------------------------------------------------------
    # Aggressive high-conviction candidates
    # --------------------------------------------------------
    {
        "name": "best_individuals_no_demand",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.30,
        "refinery": 0.35,
        "demand": 0.00,
        "usd": 0.35,
    },
    {
        "name": "best_individuals_with_demand",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.275,
        "refinery": 0.325,
        "demand": 0.075,
        "usd": 0.325,
    },
    {
        "name": "usd_refinery_dominant",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.15,
        "refinery": 0.35,
        "demand": 0.05,
        "usd": 0.45,
    },
    {
        "name": "curve_refinery_dominant",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.40,
        "refinery": 0.40,
        "demand": 0.05,
        "usd": 0.15,
    },
    {
        "name": "usd_curve_dominant",
        "inventory": 0.00,
        "cushing": 0.00,
        "curve": 0.35,
        "refinery": 0.15,
        "demand": 0.05,
        "usd": 0.45,
    },
]


OUTPUT_DIR = RESULTS_DIR / "oil_overlay_parameter_tests"


# ============================================================
# PARAMETER PATCHING
# ============================================================

def set_oil_overlay_params(
    enabled: bool,
    blend_weight: float,
    inventory_weight: float,
    cushing_weight: float,
    curve_weight: float,
    refinery_weight: float,
    demand_weight: float,
    usd_weight: float,
) -> None:
    """
    Patch both config and the active USO_scoring module.

    This matters because USO_scoring imports config values into module-level
    constants at import time. Editing config alone inside a loop is not enough.
    """

    component_weights = {
        "oil_inventory_tightness_score": float(inventory_weight),
        "oil_cushing_tightness_score": float(cushing_weight),
        "oil_curve_roll_score": float(curve_weight),
        "oil_supply_refinery_score": float(refinery_weight),
        "oil_global_demand_score": float(demand_weight),
        "oil_usd_score": float(usd_weight),
    }

    config.USO_OVERLAY_ENABLED = bool(enabled)
    config.USO_OVERLAY_BLEND_WEIGHT = float(blend_weight)

    config.USO_USE_INVENTORY_TIGHTNESS = inventory_weight > 0
    config.USO_USE_CUSHING_TIGHTNESS = cushing_weight > 0
    config.USO_USE_CURVE_ROLL = curve_weight > 0
    config.USO_USE_SUPPLY_REFINERY = refinery_weight > 0
    config.USO_USE_GLOBAL_DEMAND = demand_weight > 0
    config.USO_USE_USD = usd_weight > 0

    config.USO_COMPONENT_WEIGHTS = component_weights

    uso_scoring.USO_OVERLAY_ENABLED = bool(enabled)
    uso_scoring.USO_OVERLAY_BLEND_WEIGHT = float(blend_weight)

    uso_scoring.USO_USE_INVENTORY_TIGHTNESS = inventory_weight > 0
    uso_scoring.USO_USE_CUSHING_TIGHTNESS = cushing_weight > 0
    uso_scoring.USO_USE_CURVE_ROLL = curve_weight > 0
    uso_scoring.USO_USE_SUPPLY_REFINERY = refinery_weight > 0
    uso_scoring.USO_USE_GLOBAL_DEMAND = demand_weight > 0
    uso_scoring.USO_USE_USD = usd_weight > 0

    uso_scoring.USO_COMPONENT_WEIGHTS = component_weights


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
    inventory_weight: float,
    cushing_weight: float,
    curve_weight: float,
    refinery_weight: float,
    demand_weight: float,
    usd_weight: float,
    overlay_enabled: bool = True,
) -> dict:
    set_oil_overlay_params(
        enabled=overlay_enabled,
        blend_weight=blend_weight,
        inventory_weight=inventory_weight,
        cushing_weight=cushing_weight,
        curve_weight=curve_weight,
        refinery_weight=refinery_weight,
        demand_weight=demand_weight,
        usd_weight=usd_weight,
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
    oil_only = make_single_asset(model_weights.index, tickers, "USO")
    cash = make_cash(model_weights.index, tickers)

    strategies = {
        "model": model_weights,
        "equal_weight": equal_weight,
        "gold_only": gold_only,
        "oil_only": oil_only,
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

    uso_contribution = np.nan
    uso_contribution_share_abs = np.nan

    if not model_contrib.empty and "USO" in model_contrib["ticker"].values:
        uso_row = model_contrib[model_contrib["ticker"] == "USO"].iloc[0]
        uso_contribution = uso_row.get("total_return_contribution", np.nan)
        uso_contribution_share_abs = uso_row.get("contribution_share_abs", np.nan)

    uso_avg_weight = (
        model_weights["USO"].mean()
        if "USO" in model_weights.columns
        else np.nan
    )

    uso_max_weight = (
        model_weights["USO"].max()
        if "USO" in model_weights.columns
        else np.nan
    )

    uso_months_held = (
        int((model_weights["USO"].resample("ME").last() > 0).sum())
        if "USO" in model_weights.columns
        else np.nan
    )

    return {
        "test_name": name,
        "overlay_enabled": overlay_enabled,
        "blend_weight": blend_weight,

        "inventory_weight": inventory_weight,
        "cushing_weight": cushing_weight,
        "curve_weight": curve_weight,
        "refinery_weight": refinery_weight,
        "demand_weight": demand_weight,
        "usd_weight": usd_weight,

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

        "uso_total_return_contribution": uso_contribution,
        "uso_contribution_share_abs": uso_contribution_share_abs,
        "uso_avg_weight": uso_avg_weight,
        "uso_max_weight": uso_max_weight,
        "uso_months_held": uso_months_held,
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

    print("\nRunning baseline with oil overlay disabled...")

    rows.append(
        run_one_test(
            name="baseline_no_oil_overlay",
            score_inputs=score_inputs,
            returns=returns,
            blend_weight=0.0,
            inventory_weight=0.0,
            cushing_weight=0.0,
            curve_weight=0.0,
            refinery_weight=0.0,
            demand_weight=0.0,
            usd_weight=0.0,
            overlay_enabled=False,
        )
    )

    total_tests = len(BLEND_WEIGHTS) * len(COMPONENT_MIXES)
    done = 0

    print(f"\nRunning {total_tests} oil overlay parameter tests...")

    for blend_weight in BLEND_WEIGHTS:
        for mix in COMPONENT_MIXES:
            done += 1

            name = f"{mix['name']}_blend_{blend_weight:.3f}"

            print(
                f"[{done:03d}/{total_tests}] {name} "
                f"(inv={mix['inventory']:.3f}, cush={mix['cushing']:.3f}, "
                f"curve={mix['curve']:.3f}, ref={mix['refinery']:.3f}, "
                f"demand={mix['demand']:.3f}, usd={mix['usd']:.3f})"
            )

            row = run_one_test(
                name=name,
                score_inputs=score_inputs,
                returns=returns,
                blend_weight=blend_weight,
                inventory_weight=mix["inventory"],
                cushing_weight=mix["cushing"],
                curve_weight=mix["curve"],
                refinery_weight=mix["refinery"],
                demand_weight=mix["demand"],
                usd_weight=mix["usd"],
                overlay_enabled=True,
            )

            rows.append(row)

    results = pd.DataFrame(rows)

    baseline = results[
        results["test_name"] == "baseline_no_oil_overlay"
    ].iloc[0]

    results["delta_cagr_vs_baseline"] = results["cagr"] - baseline["cagr"]
    results["delta_sharpe_vs_baseline"] = results["sharpe"] - baseline["sharpe"]
    results["delta_sortino_vs_baseline"] = results["sortino"] - baseline["sortino"]
    results["delta_calmar_vs_baseline"] = results["calmar"] - baseline["calmar"]
    results["delta_maxdd_vs_baseline"] = results["max_drawdown"] - baseline["max_drawdown"]
    results["delta_final_equity_vs_baseline"] = (
        results["final_equity"] - baseline["final_equity"]
    )

    baseline_uso_weight = baseline["uso_avg_weight"]

    results["delta_uso_avg_weight_vs_baseline"] = (
        results["uso_avg_weight"] - baseline_uso_weight
    )

    results["delta_average_exposure_vs_baseline"] = (
        results["average_exposure"] - baseline["average_exposure"]
    )

    results["delta_turnover_vs_baseline"] = (
        results["annualised_turnover"] - baseline["annualised_turnover"]
    )

    # --------------------------------------------------------
    # Acceptance flags
    # --------------------------------------------------------
    # max_drawdown is better when numerically higher, e.g. -0.08 beats -0.10.
    # These are not final production rules. They are a first filter before
    # walk-forward and stress testing.
    results["passes_cagr_rule"] = (
        (results["delta_cagr_vs_baseline"] >= 0.0025)
        & (results["delta_sharpe_vs_baseline"] >= -0.010)
        & (results["delta_maxdd_vs_baseline"] >= -0.0030)
    )

    results["passes_sharpe_rule"] = (
        (results["delta_sharpe_vs_baseline"] >= 0.030)
        & (results["delta_cagr_vs_baseline"] >= -0.0010)
        & (results["delta_maxdd_vs_baseline"] >= -0.0030)
    )

    results["passes_high_return_rule"] = (
        (results["delta_cagr_vs_baseline"] >= 0.0045)
        & (results["delta_sharpe_vs_baseline"] >= 0.010)
        & (results["delta_maxdd_vs_baseline"] >= -0.0040)
    )

    results["passes_clean_win_rule"] = (
        (results["delta_cagr_vs_baseline"] > 0.0)
        & (results["delta_sharpe_vs_baseline"] > 0.0)
        & (results["delta_sortino_vs_baseline"] > 0.0)
        & (results["delta_maxdd_vs_baseline"] >= -0.0025)
    )

    results["passes_any_rule"] = (
        results["passes_cagr_rule"]
        | results["passes_sharpe_rule"]
        | results["passes_high_return_rule"]
        | results["passes_clean_win_rule"]
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------
    # Higher is better for all included columns.
    # max_drawdown is better when less negative, i.e. numerically higher.
    results["rank_score"] = (
        0.30 * results["cagr"].rank(ascending=True, pct=True)
        + 0.25 * results["sharpe"].rank(ascending=True, pct=True)
        + 0.20 * results["sortino"].rank(ascending=True, pct=True)
        + 0.15 * results["calmar"].rank(ascending=True, pct=True)
        + 0.10 * results["max_drawdown"].rank(ascending=True, pct=True)
    )

    # Do not over-penalise exposure because oil was genuinely helpful in the
    # individual tests. But still penalise cases that only win by loading much
    # more USO or by massively increasing turnover.
    results["rank_score_adjusted"] = (
        results["rank_score"]
        - 0.08 * results["delta_uso_avg_weight_vs_baseline"].clip(lower=0.0)
        - 0.03 * results["delta_average_exposure_vs_baseline"].clip(lower=0.0)
        - 0.02 * results["delta_turnover_vs_baseline"].clip(lower=0.0)
    )

    results = results.sort_values(
        [
            "passes_any_rule",
            "rank_score_adjusted",
            "cagr",
            "sharpe",
            "calmar",
        ],
        ascending=False,
    ).reset_index(drop=True)

    output_path = OUTPUT_DIR / "oil_overlay_parameter_summary.csv"
    results.to_csv(output_path, index=False)

    best_by_blend = (
        results
        .sort_values(["blend_weight", "rank_score_adjusted"], ascending=[True, False])
        .groupby("blend_weight", as_index=False)
        .head(1)
        .sort_values("blend_weight")
        .reset_index(drop=True)
    )

    best_by_blend_path = OUTPUT_DIR / "oil_overlay_best_by_blend.csv"
    best_by_blend.to_csv(best_by_blend_path, index=False)

    passing = results[results["passes_any_rule"]].copy()
    passing_path = OUTPUT_DIR / "oil_overlay_passing_candidates.csv"
    passing.to_csv(passing_path, index=False)

    print("\nSaved full results to:")
    print(output_path)

    print("\nSaved best-by-blend results to:")
    print(best_by_blend_path)

    print("\nSaved passing candidates to:")
    print(passing_path)

    display_cols = [
        "test_name",
        "blend_weight",
        "inventory_weight",
        "cushing_weight",
        "curve_weight",
        "refinery_weight",
        "demand_weight",
        "usd_weight",
        "final_equity",
        "cagr",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "average_exposure",
        "average_cash",
        "annualised_turnover",
        "uso_total_return_contribution",
        "uso_avg_weight",
        "uso_max_weight",
        "delta_final_equity_vs_baseline",
        "delta_cagr_vs_baseline",
        "delta_sharpe_vs_baseline",
        "delta_sortino_vs_baseline",
        "delta_maxdd_vs_baseline",
        "passes_any_rule",
        "rank_score_adjusted",
    ]

    available_display_cols = [
        col for col in display_cols
        if col in results.columns
    ]

    print("\nTop 25 candidates:")
    print(results[available_display_cols].head(25).to_string(index=False))

    print("\nBest candidate by blend weight:")
    print(best_by_blend[available_display_cols].to_string(index=False))

    print("\nBaseline row:")
    print(
        results[
            results["test_name"] == "baseline_no_oil_overlay"
        ][available_display_cols].to_string(index=False)
    )

    print("\nPassing candidates:")
    if passing.empty:
        print("None. Do not force the oil overlay into production.")
    else:
        print(passing[available_display_cols].head(25).to_string(index=False))

    print("\nInterpretation reminder:")
    print("- The winner here is still in-sample.")
    print("- Do not lock production until stress tests and walk-forward confirm it.")
    print("- If high blend weights win, they are allowed, but must survive validation.")


if __name__ == "__main__":
    main()