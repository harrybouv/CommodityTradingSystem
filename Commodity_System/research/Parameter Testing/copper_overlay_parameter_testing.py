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
# RESOLVE ACTIVE CPER SCORING MODULE
# ============================================================

def get_active_cper_scoring_module():
    """
    Finds the actual module used by the registry for CPER.

    This is deliberately robust because CPER_scoring.py may live either at:
    - scoring/commodity_models/CPER_scoring.py
    - scoring/commodity_models/Copper/CPER_scoring.py

    The registry decides what is actually used in production. We patch that.
    """
    scorer = commodity_registry.COMMODITY_SCORERS.get("CPER")

    if scorer is None:
        raise ValueError("Registry has no CPER scorer.")

    module_name = scorer.__module__

    if module_name not in sys.modules:
        __import__(module_name)

    return sys.modules[module_name]


cper_scoring = get_active_cper_scoring_module()


# ============================================================
# TEST GRID
# ============================================================

# Do not go crazy yet. First prove whether any stable region exists.
BLEND_WEIGHTS = [0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]


COMPONENT_MIXES = [
    # --------------------------------------------------------
    # Single-factor anchors
    # --------------------------------------------------------
    {
        "name": "electricity_only",
        "electricity": 1.00,
        "cli": 0.00,
        "usd": 0.00,
        "broad": 0.00,
        "oil": 0.00,
        "growth": 0.00,
    },
    {
        "name": "cli_only",
        "electricity": 0.00,
        "cli": 1.00,
        "usd": 0.00,
        "broad": 0.00,
        "oil": 0.00,
        "growth": 0.00,
    },
    {
        "name": "usd_only",
        "electricity": 0.00,
        "cli": 0.00,
        "usd": 1.00,
        "broad": 0.00,
        "oil": 0.00,
        "growth": 0.00,
    },
    {
        "name": "broad_only",
        "electricity": 0.00,
        "cli": 0.00,
        "usd": 0.00,
        "broad": 1.00,
        "oil": 0.00,
        "growth": 0.00,
    },
    {
        "name": "oil_only",
        "electricity": 0.00,
        "cli": 0.00,
        "usd": 0.00,
        "broad": 0.00,
        "oil": 1.00,
        "growth": 0.00,
    },
    {
        "name": "growth_only",
        "electricity": 0.00,
        "cli": 0.00,
        "usd": 0.00,
        "broad": 0.00,
        "oil": 0.00,
        "growth": 1.00,
    },

    # --------------------------------------------------------
    # China block tests
    # --------------------------------------------------------
    {
        "name": "china_equal",
        "electricity": 0.50,
        "cli": 0.50,
        "usd": 0.00,
        "broad": 0.00,
        "oil": 0.00,
        "growth": 0.00,
    },
    {
        "name": "china_cli_heavy",
        "electricity": 0.25,
        "cli": 0.75,
        "usd": 0.00,
        "broad": 0.00,
        "oil": 0.00,
        "growth": 0.00,
    },
    {
        "name": "china_electricity_heavy",
        "electricity": 0.75,
        "cli": 0.25,
        "usd": 0.00,
        "broad": 0.00,
        "oil": 0.00,
        "growth": 0.00,
    },

    # --------------------------------------------------------
    # The realistic candidates
    # --------------------------------------------------------
    {
        "name": "cli_growth",
        "electricity": 0.00,
        "cli": 0.70,
        "usd": 0.00,
        "broad": 0.00,
        "oil": 0.00,
        "growth": 0.30,
    },
    {
        "name": "cli_broad",
        "electricity": 0.00,
        "cli": 0.70,
        "usd": 0.00,
        "broad": 0.30,
        "oil": 0.00,
        "growth": 0.00,
    },
    {
        "name": "cli_growth_broad",
        "electricity": 0.00,
        "cli": 0.55,
        "usd": 0.00,
        "broad": 0.20,
        "oil": 0.00,
        "growth": 0.25,
    },
    {
        "name": "cli_growth_oil",
        "electricity": 0.00,
        "cli": 0.55,
        "usd": 0.00,
        "broad": 0.00,
        "oil": 0.20,
        "growth": 0.25,
    },

    # --------------------------------------------------------
    # Electricity rescue tests
    # These answer: can electricity work when not overweighted?
    # --------------------------------------------------------
    {
        "name": "electricity_light_cli_growth",
        "electricity": 0.15,
        "cli": 0.60,
        "usd": 0.00,
        "broad": 0.00,
        "oil": 0.00,
        "growth": 0.25,
    },
    {
        "name": "electricity_light_cli_broad",
        "electricity": 0.15,
        "cli": 0.60,
        "usd": 0.00,
        "broad": 0.25,
        "oil": 0.00,
        "growth": 0.00,
    },
    {
        "name": "electricity_light_cli_growth_broad",
        "electricity": 0.10,
        "cli": 0.55,
        "usd": 0.00,
        "broad": 0.15,
        "oil": 0.00,
        "growth": 0.20,
    },

    # --------------------------------------------------------
    # Original theory-style mixes
    # --------------------------------------------------------
    {
        "name": "theory_core_original",
        "electricity": 0.30,
        "cli": 0.20,
        "usd": 0.15,
        "broad": 0.15,
        "oil": 0.10,
        "growth": 0.10,
    },
    {
        "name": "no_electricity_no_usd",
        "electricity": 0.00,
        "cli": 0.45,
        "usd": 0.00,
        "broad": 0.25,
        "oil": 0.10,
        "growth": 0.20,
    },
    {
        "name": "china_plus_market_confirmation",
        "electricity": 0.15,
        "cli": 0.45,
        "usd": 0.00,
        "broad": 0.20,
        "oil": 0.05,
        "growth": 0.15,
    },
    {
        "name": "macro_full_but_usd_light",
        "electricity": 0.10,
        "cli": 0.45,
        "usd": 0.05,
        "broad": 0.20,
        "oil": 0.05,
        "growth": 0.15,
    },
]


OUTPUT_DIR = RESULTS_DIR / "copper_overlay_parameter_tests"


# ============================================================
# PARAMETER PATCHING
# ============================================================

def set_copper_overlay_params(
    enabled: bool,
    blend_weight: float,
    electricity_weight: float,
    cli_weight: float,
    usd_weight: float,
    broad_weight: float,
    oil_weight: float,
    growth_weight: float,
) -> None:
    """
    Patch both config and the active CPER_scoring module.

    This matters because CPER_scoring imports config values into module-level
    constants at import time. Editing config alone inside a loop is not enough.
    """

    component_weights = {
        "copper_china_electricity_score": float(electricity_weight),
        "copper_china_cli_score": float(cli_weight),
        "copper_usd_score": float(usd_weight),
        "copper_broad_commodity_trend_score": float(broad_weight),
        "copper_oil_price_score": float(oil_weight),
        "copper_global_growth_score": float(growth_weight),
    }

    config.COPPER_OVERLAY_ENABLED = enabled
    config.COPPER_OVERLAY_BLEND_WEIGHT = float(blend_weight)

    config.COPPER_USE_CHINA_ELECTRICITY = electricity_weight > 0
    config.COPPER_USE_CHINA_CLI = cli_weight > 0
    config.COPPER_USE_USD = usd_weight > 0
    config.COPPER_USE_BROAD_COMMODITY_TREND = broad_weight > 0
    config.COPPER_USE_OIL_PRICE = oil_weight > 0
    config.COPPER_USE_GLOBAL_GROWTH = growth_weight > 0

    config.COPPER_COMPONENT_WEIGHTS = component_weights

    cper_scoring.COPPER_OVERLAY_ENABLED = enabled
    cper_scoring.COPPER_OVERLAY_BLEND_WEIGHT = float(blend_weight)

    cper_scoring.COPPER_USE_CHINA_ELECTRICITY = electricity_weight > 0
    cper_scoring.COPPER_USE_CHINA_CLI = cli_weight > 0
    cper_scoring.COPPER_USE_USD = usd_weight > 0
    cper_scoring.COPPER_USE_BROAD_COMMODITY_TREND = broad_weight > 0
    cper_scoring.COPPER_USE_OIL_PRICE = oil_weight > 0
    cper_scoring.COPPER_USE_GLOBAL_GROWTH = growth_weight > 0

    cper_scoring.COPPER_COMPONENT_WEIGHTS = component_weights


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
    electricity_weight: float,
    cli_weight: float,
    usd_weight: float,
    broad_weight: float,
    oil_weight: float,
    growth_weight: float,
    overlay_enabled: bool = True,
) -> dict:
    set_copper_overlay_params(
        enabled=overlay_enabled,
        blend_weight=blend_weight,
        electricity_weight=electricity_weight,
        cli_weight=cli_weight,
        usd_weight=usd_weight,
        broad_weight=broad_weight,
        oil_weight=oil_weight,
        growth_weight=growth_weight,
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
    copper_only = make_single_asset(model_weights.index, tickers, "CPER")
    cash = make_cash(model_weights.index, tickers)

    strategies = {
        "model": model_weights,
        "equal_weight": equal_weight,
        "gold_only": gold_only,
        "copper_only": copper_only,
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

    cper_contribution = np.nan
    cper_contribution_share_abs = np.nan

    if not model_contrib.empty and "CPER" in model_contrib["ticker"].values:
        cper_row = model_contrib[model_contrib["ticker"] == "CPER"].iloc[0]
        cper_contribution = cper_row.get("total_return_contribution", np.nan)
        cper_contribution_share_abs = cper_row.get("contribution_share_abs", np.nan)

    cper_avg_weight = (
        model_weights["CPER"].mean()
        if "CPER" in model_weights.columns
        else np.nan
    )

    cper_max_weight = (
        model_weights["CPER"].max()
        if "CPER" in model_weights.columns
        else np.nan
    )

    cper_months_held = (
        int((model_weights["CPER"].resample("ME").last() > 0).sum())
        if "CPER" in model_weights.columns
        else np.nan
    )

    return {
        "test_name": name,
        "overlay_enabled": overlay_enabled,
        "blend_weight": blend_weight,

        "electricity_weight": electricity_weight,
        "cli_weight": cli_weight,
        "usd_weight": usd_weight,
        "broad_weight": broad_weight,
        "oil_weight": oil_weight,
        "growth_weight": growth_weight,

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

        "cper_total_return_contribution": cper_contribution,
        "cper_contribution_share_abs": cper_contribution_share_abs,
        "cper_avg_weight": cper_avg_weight,
        "cper_max_weight": cper_max_weight,
        "cper_months_held": cper_months_held,
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

    print("\nRunning baseline with copper overlay disabled...")

    rows.append(
        run_one_test(
            name="baseline_no_copper_overlay",
            score_inputs=score_inputs,
            returns=returns,
            blend_weight=0.0,
            electricity_weight=0.0,
            cli_weight=0.0,
            usd_weight=0.0,
            broad_weight=0.0,
            oil_weight=0.0,
            growth_weight=0.0,
            overlay_enabled=False,
        )
    )

    total_tests = len(BLEND_WEIGHTS) * len(COMPONENT_MIXES)
    done = 0

    print(f"\nRunning {total_tests} copper overlay parameter tests...")

    for blend_weight in BLEND_WEIGHTS:
        for mix in COMPONENT_MIXES:
            done += 1

            name = f"{mix['name']}_blend_{blend_weight:.3f}"

            print(
                f"[{done:03d}/{total_tests}] {name} "
                f"(elec={mix['electricity']:.2f}, cli={mix['cli']:.2f}, "
                f"usd={mix['usd']:.2f}, broad={mix['broad']:.2f}, "
                f"oil={mix['oil']:.2f}, growth={mix['growth']:.2f})"
            )

            row = run_one_test(
                name=name,
                score_inputs=score_inputs,
                returns=returns,
                blend_weight=blend_weight,
                electricity_weight=mix["electricity"],
                cli_weight=mix["cli"],
                usd_weight=mix["usd"],
                broad_weight=mix["broad"],
                oil_weight=mix["oil"],
                growth_weight=mix["growth"],
                overlay_enabled=True,
            )

            rows.append(row)

    results = pd.DataFrame(rows)

    baseline = results[
        results["test_name"] == "baseline_no_copper_overlay"
    ].iloc[0]

    results["delta_cagr_vs_baseline"] = results["cagr"] - baseline["cagr"]
    results["delta_sharpe_vs_baseline"] = results["sharpe"] - baseline["sharpe"]
    results["delta_sortino_vs_baseline"] = results["sortino"] - baseline["sortino"]
    results["delta_calmar_vs_baseline"] = results["calmar"] - baseline["calmar"]
    results["delta_maxdd_vs_baseline"] = results["max_drawdown"] - baseline["max_drawdown"]
    results["delta_final_equity_vs_baseline"] = (
        results["final_equity"] - baseline["final_equity"]
    )

    baseline_cper_weight = baseline["cper_avg_weight"]
    results["delta_cper_avg_weight_vs_baseline"] = (
        results["cper_avg_weight"] - baseline_cper_weight
    )

    # --------------------------------------------------------
    # Acceptance flags
    # --------------------------------------------------------
    # max_drawdown is better when numerically higher, e.g. -0.09 beats -0.12.
    results["passes_sharpe_rule"] = (
        (results["delta_sharpe_vs_baseline"] >= 0.03)
        & (results["delta_cagr_vs_baseline"] >= -0.003)
    )

    results["passes_drawdown_rule"] = (
        (results["delta_maxdd_vs_baseline"] >= 0.005)
        & (results["delta_cagr_vs_baseline"] >= -0.005)
    )

    results["passes_cagr_rule"] = (
        (results["delta_cagr_vs_baseline"] > 0.0)
        & (results["delta_sharpe_vs_baseline"] >= -0.01)
        & (results["delta_maxdd_vs_baseline"] >= -0.005)
    )

    results["passes_any_rule"] = (
        results["passes_sharpe_rule"]
        | results["passes_drawdown_rule"]
        | results["passes_cagr_rule"]
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------
    # Higher is better for all included columns.
    # max_drawdown is better when less negative, i.e. numerically higher.
    results["rank_score"] = (
        results["sharpe"].rank(ascending=True, pct=True)
        + results["sortino"].rank(ascending=True, pct=True)
        + results["calmar"].rank(ascending=True, pct=True)
        + results["max_drawdown"].rank(ascending=True, pct=True)
        + results["cagr"].rank(ascending=True, pct=True)
    ) / 5.0

    # Small penalty if a candidate only "works" by hugely increasing CPER exposure.
    results["rank_score_adjusted"] = (
        results["rank_score"]
        - 0.10 * results["delta_cper_avg_weight_vs_baseline"].clip(lower=0.0)
    )

    results = results.sort_values(
        [
            "passes_any_rule",
            "rank_score_adjusted",
            "sharpe",
            "calmar",
            "cagr",
        ],
        ascending=False,
    ).reset_index(drop=True)

    output_path = OUTPUT_DIR / "copper_overlay_parameter_summary.csv"
    results.to_csv(output_path, index=False)

    print("\nSaved results to:")
    print(output_path)

    display_cols = [
        "test_name",
        "blend_weight",
        "electricity_weight",
        "cli_weight",
        "usd_weight",
        "broad_weight",
        "oil_weight",
        "growth_weight",
        "final_equity",
        "cagr",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "average_exposure",
        "average_cash",
        "cper_total_return_contribution",
        "cper_avg_weight",
        "cper_max_weight",
        "delta_cagr_vs_baseline",
        "delta_sharpe_vs_baseline",
        "delta_maxdd_vs_baseline",
        "passes_any_rule",
        "rank_score_adjusted",
    ]

    available_display_cols = [
        col for col in display_cols
        if col in results.columns
    ]

    print("\nTop 20 candidates:")
    print(results[available_display_cols].head(20).to_string(index=False))

    print("\nBaseline row:")
    print(
        results[
            results["test_name"] == "baseline_no_copper_overlay"
        ][available_display_cols].to_string(index=False)
    )

    passing = results[results["passes_any_rule"]].copy()

    print("\nPassing candidates:")
    if passing.empty:
        print("None. Do not force the overlay into production.")
    else:
        print(passing[available_display_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()