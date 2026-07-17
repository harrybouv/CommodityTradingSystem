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
# RESOLVE ACTIVE DBA SCORING MODULE
# ============================================================

def get_active_dba_scoring_module():
    """
    Finds the actual module used by the registry for DBA.

    This avoids fragile assumptions about whether DBA_scoring.py lives at:
    - scoring/commodity_models/DBA_scoring.py
    - scoring/commodity_models/Agriculture/DBA_scoring.py

    The registry decides what is actually used in production. We patch that.
    """

    scorer = commodity_registry.COMMODITY_SCORERS.get("DBA")

    if scorer is None:
        raise ValueError("Registry has no DBA scorer.")

    module_name = scorer.__module__

    if module_name not in sys.modules:
        __import__(module_name)

    return sys.modules[module_name]


dba_scoring = get_active_dba_scoring_module()


# ============================================================
# TEST GRID
# ============================================================

# This is deliberately focused.
# Individual tests showed:
# - crop relative strength was the only clear positive standalone component
# - USD and seasonality were close / plausible supports
# - rates was only marginal
# - crop momentum and broad commodity confirmation were weaker
# - exports remain optional/noisy and are off by default
#
# So this grid tests the plausible region instead of spraying the search space.

BLEND_WEIGHTS = [
    0.025,
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.20,
    0.25,
    0.30,
]

INCLUDE_EXPORT_TESTS = False


COMPONENT_MIXES = [
    # --------------------------------------------------------
    # Single-factor anchors
    # These re-check the individual tests inside the same framework.
    # --------------------------------------------------------
    {
        "name": "relative_strength_only",
        "usd": 0.00,
        "rates": 0.00,
        "crop_momentum": 0.00,
        "relative_strength": 1.00,
        "broad": 0.00,
        "seasonality": 0.00,
        "exports": 0.00,
    },
    {
        "name": "usd_only",
        "usd": 1.00,
        "rates": 0.00,
        "crop_momentum": 0.00,
        "relative_strength": 0.00,
        "broad": 0.00,
        "seasonality": 0.00,
        "exports": 0.00,
    },
    {
        "name": "seasonality_only",
        "usd": 0.00,
        "rates": 0.00,
        "crop_momentum": 0.00,
        "relative_strength": 0.00,
        "broad": 0.00,
        "seasonality": 1.00,
        "exports": 0.00,
    },
    {
        "name": "rates_only",
        "usd": 0.00,
        "rates": 1.00,
        "crop_momentum": 0.00,
        "relative_strength": 0.00,
        "broad": 0.00,
        "seasonality": 0.00,
        "exports": 0.00,
    },

    # --------------------------------------------------------
    # Relative-strength-led candidates
    # Main viable region. If agriculture overlay is worth keeping,
    # it is probably somewhere in this block.
    # --------------------------------------------------------
    {
        "name": "rs_usd_80_20",
        "usd": 0.20,
        "rates": 0.00,
        "crop_momentum": 0.00,
        "relative_strength": 0.80,
        "broad": 0.00,
        "seasonality": 0.00,
        "exports": 0.00,
    },
    {
        "name": "rs_usd_70_30",
        "usd": 0.30,
        "rates": 0.00,
        "crop_momentum": 0.00,
        "relative_strength": 0.70,
        "broad": 0.00,
        "seasonality": 0.00,
        "exports": 0.00,
    },
    {
        "name": "rs_usd_60_40",
        "usd": 0.40,
        "rates": 0.00,
        "crop_momentum": 0.00,
        "relative_strength": 0.60,
        "broad": 0.00,
        "seasonality": 0.00,
        "exports": 0.00,
    },
    {
        "name": "rs_seasonality_90_10",
        "usd": 0.00,
        "rates": 0.00,
        "crop_momentum": 0.00,
        "relative_strength": 0.90,
        "broad": 0.00,
        "seasonality": 0.10,
        "exports": 0.00,
    },
    {
        "name": "rs_seasonality_80_20",
        "usd": 0.00,
        "rates": 0.00,
        "crop_momentum": 0.00,
        "relative_strength": 0.80,
        "broad": 0.00,
        "seasonality": 0.20,
        "exports": 0.00,
    },
    {
        "name": "rs_seasonality_70_30",
        "usd": 0.00,
        "rates": 0.00,
        "crop_momentum": 0.00,
        "relative_strength": 0.70,
        "broad": 0.00,
        "seasonality": 0.30,
        "exports": 0.00,
    },

    # --------------------------------------------------------
    # Three-factor candidate region
    # Previous 55/25/20 style blend weakened. These keep RS dominant.
    # --------------------------------------------------------
    {
        "name": "rs_usd_seasonality_90_05_05",
        "usd": 0.05,
        "rates": 0.00,
        "crop_momentum": 0.00,
        "relative_strength": 0.90,
        "broad": 0.00,
        "seasonality": 0.05,
        "exports": 0.00,
    },
    {
        "name": "rs_usd_seasonality_80_10_10",
        "usd": 0.10,
        "rates": 0.00,
        "crop_momentum": 0.00,
        "relative_strength": 0.80,
        "broad": 0.00,
        "seasonality": 0.10,
        "exports": 0.00,
    },
    {
        "name": "rs_usd_seasonality_70_20_10",
        "usd": 0.20,
        "rates": 0.00,
        "crop_momentum": 0.00,
        "relative_strength": 0.70,
        "broad": 0.00,
        "seasonality": 0.10,
        "exports": 0.00,
    },
    {
        "name": "rs_usd_seasonality_70_10_20",
        "usd": 0.10,
        "rates": 0.00,
        "crop_momentum": 0.00,
        "relative_strength": 0.70,
        "broad": 0.00,
        "seasonality": 0.20,
        "exports": 0.00,
    },
    {
        "name": "rs_usd_seasonality_60_25_15",
        "usd": 0.25,
        "rates": 0.00,
        "crop_momentum": 0.00,
        "relative_strength": 0.60,
        "broad": 0.00,
        "seasonality": 0.15,
        "exports": 0.00,
    },

    # --------------------------------------------------------
    # Low-rate-support tests
    # Rates were not awful, but should not be allowed to dominate.
    # --------------------------------------------------------
    {
        "name": "rs_usd_rates_70_20_10",
        "usd": 0.20,
        "rates": 0.10,
        "crop_momentum": 0.00,
        "relative_strength": 0.70,
        "broad": 0.00,
        "seasonality": 0.00,
        "exports": 0.00,
    },
    {
        "name": "rs_usd_seasonality_rates_65_20_10_05",
        "usd": 0.20,
        "rates": 0.05,
        "crop_momentum": 0.00,
        "relative_strength": 0.65,
        "broad": 0.00,
        "seasonality": 0.10,
        "exports": 0.00,
    },

    # --------------------------------------------------------
    # Weak-component rescue tests
    # These are intentionally limited. If light weights do not help,
    # crop momentum and broad confirmation should be dropped.
    # --------------------------------------------------------
    {
        "name": "rs_crop_momentum_light_85_15",
        "usd": 0.00,
        "rates": 0.00,
        "crop_momentum": 0.15,
        "relative_strength": 0.85,
        "broad": 0.00,
        "seasonality": 0.00,
        "exports": 0.00,
    },
    {
        "name": "rs_broad_light_85_15",
        "usd": 0.00,
        "rates": 0.00,
        "crop_momentum": 0.00,
        "relative_strength": 0.85,
        "broad": 0.15,
        "seasonality": 0.00,
        "exports": 0.00,
    },
    {
        "name": "rs_usd_crop_momentum_light",
        "usd": 0.20,
        "rates": 0.00,
        "crop_momentum": 0.10,
        "relative_strength": 0.70,
        "broad": 0.00,
        "seasonality": 0.00,
        "exports": 0.00,
    },
    {
        "name": "rs_usd_broad_light",
        "usd": 0.20,
        "rates": 0.00,
        "crop_momentum": 0.00,
        "relative_strength": 0.70,
        "broad": 0.10,
        "seasonality": 0.00,
        "exports": 0.00,
    },

    # --------------------------------------------------------
    # Theory/original mix
    # Included as a sanity check, not because it looked good.
    # --------------------------------------------------------
    {
        "name": "theory_original",
        "usd": 0.20,
        "rates": 0.07,
        "crop_momentum": 0.25,
        "relative_strength": 0.25,
        "broad": 0.10,
        "seasonality": 0.13,
        "exports": 0.00,
    },
]


if INCLUDE_EXPORT_TESTS:
    COMPONENT_MIXES += [
        {
            "name": "rs_export_light_95_05",
            "usd": 0.00,
            "rates": 0.00,
            "crop_momentum": 0.00,
            "relative_strength": 0.95,
            "broad": 0.00,
            "seasonality": 0.00,
            "exports": 0.05,
        },
        {
            "name": "rs_usd_seasonality_export_light",
            "usd": 0.20,
            "rates": 0.00,
            "crop_momentum": 0.00,
            "relative_strength": 0.65,
            "broad": 0.00,
            "seasonality": 0.10,
            "exports": 0.05,
        },
    ]


OUTPUT_DIR = RESULTS_DIR / "agriculture_overlay_parameter_tests"


# ============================================================
# PARAMETER PATCHING
# ============================================================

def set_agriculture_overlay_params(
    enabled: bool,
    blend_weight: float,
    usd_weight: float,
    rates_weight: float,
    crop_momentum_weight: float,
    relative_strength_weight: float,
    broad_weight: float,
    seasonality_weight: float,
    export_weight: float,
) -> None:
    """
    Patch both config and the active DBA_scoring module.

    This matters because DBA_scoring imports config values into module-level
    constants at import time. Editing config alone inside a loop is not enough.
    """

    component_weights = {
        "agri_usd_score": float(usd_weight),
        "agri_rates_score": float(rates_weight),
        "agri_crop_momentum_score": float(crop_momentum_weight),
        "agri_crop_relative_strength_score": float(relative_strength_weight),
        "agri_broad_commodity_confirmation": float(broad_weight),
        "agri_seasonality_score": float(seasonality_weight),
        "agri_export_demand_score": float(export_weight),
    }

    config.DBA_OVERLAY_ENABLED = bool(enabled)
    config.DBA_OVERLAY_BLEND_WEIGHT = float(blend_weight)

    config.DBA_USE_USD = usd_weight > 0
    config.DBA_USE_RATES = rates_weight > 0
    config.DBA_USE_CROP_MOMENTUM = crop_momentum_weight > 0
    config.DBA_USE_CROP_RELATIVE_STRENGTH = relative_strength_weight > 0
    config.DBA_USE_BROAD_COMMODITY_CONFIRMATION = broad_weight > 0
    config.DBA_USE_SEASONALITY = seasonality_weight > 0
    config.DBA_USE_EXPORT_DEMAND = export_weight > 0

    config.DBA_COMPONENT_WEIGHTS = component_weights

    dba_scoring.DBA_OVERLAY_ENABLED = bool(enabled)
    dba_scoring.DBA_OVERLAY_BLEND_WEIGHT = float(blend_weight)

    dba_scoring.DBA_USE_USD = usd_weight > 0
    dba_scoring.DBA_USE_RATES = rates_weight > 0
    dba_scoring.DBA_USE_CROP_MOMENTUM = crop_momentum_weight > 0
    dba_scoring.DBA_USE_CROP_RELATIVE_STRENGTH = relative_strength_weight > 0
    dba_scoring.DBA_USE_BROAD_COMMODITY_CONFIRMATION = broad_weight > 0
    dba_scoring.DBA_USE_SEASONALITY = seasonality_weight > 0
    dba_scoring.DBA_USE_EXPORT_DEMAND = export_weight > 0

    dba_scoring.DBA_COMPONENT_WEIGHTS = component_weights


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
    mix_name: str,
    score_inputs: pd.DataFrame,
    returns: pd.DataFrame,
    blend_weight: float,
    usd_weight: float,
    rates_weight: float,
    crop_momentum_weight: float,
    relative_strength_weight: float,
    broad_weight: float,
    seasonality_weight: float,
    export_weight: float,
    overlay_enabled: bool = True,
) -> dict:
    set_agriculture_overlay_params(
        enabled=overlay_enabled,
        blend_weight=blend_weight,
        usd_weight=usd_weight,
        rates_weight=rates_weight,
        crop_momentum_weight=crop_momentum_weight,
        relative_strength_weight=relative_strength_weight,
        broad_weight=broad_weight,
        seasonality_weight=seasonality_weight,
        export_weight=export_weight,
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
    agriculture_only = make_single_asset(model_weights.index, tickers, "DBA")
    cash = make_cash(model_weights.index, tickers)

    strategies = {
        "model": model_weights,
        "equal_weight": equal_weight,
        "gold_only": gold_only,
        "agriculture_only": agriculture_only,
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

    dba_contribution = np.nan
    dba_contribution_share_abs = np.nan

    if not model_contrib.empty and "DBA" in model_contrib["ticker"].values:
        dba_row = model_contrib[model_contrib["ticker"] == "DBA"].iloc[0]
        dba_contribution = dba_row.get("total_return_contribution", np.nan)
        dba_contribution_share_abs = dba_row.get("contribution_share_abs", np.nan)

    dba_avg_weight = (
        model_weights["DBA"].mean()
        if "DBA" in model_weights.columns
        else np.nan
    )

    dba_max_weight = (
        model_weights["DBA"].max()
        if "DBA" in model_weights.columns
        else np.nan
    )

    dba_months_held = (
        int((model_weights["DBA"].resample("ME").last() > 0).sum())
        if "DBA" in model_weights.columns
        else np.nan
    )

    return {
        "test_name": name,
        "mix_name": mix_name,
        "overlay_enabled": overlay_enabled,
        "blend_weight": blend_weight,

        "usd_weight": usd_weight,
        "rates_weight": rates_weight,
        "crop_momentum_weight": crop_momentum_weight,
        "relative_strength_weight": relative_strength_weight,
        "broad_weight": broad_weight,
        "seasonality_weight": seasonality_weight,
        "export_weight": export_weight,

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

        "dba_total_return_contribution": dba_contribution,
        "dba_contribution_share_abs": dba_contribution_share_abs,
        "dba_avg_weight": dba_avg_weight,
        "dba_max_weight": dba_max_weight,
        "dba_months_held": dba_months_held,
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

    print("\nRunning baseline with agriculture overlay disabled...")

    rows.append(
        run_one_test(
            name="baseline_no_agriculture_overlay",
            mix_name="baseline",
            score_inputs=score_inputs,
            returns=returns,
            blend_weight=0.0,
            usd_weight=0.0,
            rates_weight=0.0,
            crop_momentum_weight=0.0,
            relative_strength_weight=0.0,
            broad_weight=0.0,
            seasonality_weight=0.0,
            export_weight=0.0,
            overlay_enabled=False,
        )
    )

    total_tests = len(BLEND_WEIGHTS) * len(COMPONENT_MIXES)
    done = 0

    print(f"\nRunning {total_tests} agriculture overlay parameter tests...")

    for blend_weight in BLEND_WEIGHTS:
        for mix in COMPONENT_MIXES:
            done += 1

            name = f"{mix['name']}_blend_{blend_weight:.3f}"

            print(
                f"[{done:03d}/{total_tests}] {name} "
                f"(rs={mix['relative_strength']:.2f}, usd={mix['usd']:.2f}, "
                f"season={mix['seasonality']:.2f}, rates={mix['rates']:.2f}, "
                f"mom={mix['crop_momentum']:.2f}, broad={mix['broad']:.2f}, "
                f"exports={mix['exports']:.2f})"
            )

            row = run_one_test(
                name=name,
                mix_name=mix["name"],
                score_inputs=score_inputs,
                returns=returns,
                blend_weight=blend_weight,
                usd_weight=mix["usd"],
                rates_weight=mix["rates"],
                crop_momentum_weight=mix["crop_momentum"],
                relative_strength_weight=mix["relative_strength"],
                broad_weight=mix["broad"],
                seasonality_weight=mix["seasonality"],
                export_weight=mix["exports"],
                overlay_enabled=True,
            )

            rows.append(row)

    results = pd.DataFrame(rows)

    baseline = results[
        results["test_name"] == "baseline_no_agriculture_overlay"
    ].iloc[0]

    results["delta_final_equity_vs_baseline"] = (
        results["final_equity"] - baseline["final_equity"]
    )
    results["delta_cagr_vs_baseline"] = results["cagr"] - baseline["cagr"]
    results["delta_sharpe_vs_baseline"] = results["sharpe"] - baseline["sharpe"]
    results["delta_sortino_vs_baseline"] = results["sortino"] - baseline["sortino"]
    results["delta_calmar_vs_baseline"] = results["calmar"] - baseline["calmar"]
    results["delta_maxdd_vs_baseline"] = results["max_drawdown"] - baseline["max_drawdown"]
    results["delta_turnover_vs_baseline"] = (
        results["annualised_turnover"] - baseline["annualised_turnover"]
    )
    results["delta_exposure_vs_baseline"] = (
        results["average_exposure"] - baseline["average_exposure"]
    )

    # Positive delta_maxdd means less negative drawdown, i.e. better.
    results["beats_baseline_core"] = (
        (results["cagr"] > baseline["cagr"])
        & (results["sharpe"] >= baseline["sharpe"])
        & (results["max_drawdown"] >= baseline["max_drawdown"] - 0.0025)
    )

    results["quality_improvement"] = (
        (results["delta_sharpe_vs_baseline"] > 0)
        & (results["delta_sortino_vs_baseline"] > 0)
        & (results["delta_calmar_vs_baseline"] > 0)
    )

    results["worth_investigating"] = (
        results["beats_baseline_core"]
        | (
            (results["delta_cagr_vs_baseline"] >= -0.0010)
            & (results["delta_sharpe_vs_baseline"] > 0)
            & (results["delta_maxdd_vs_baseline"] >= -0.0025)
        )
    )

    # Ranking: avoid pure CAGR-chasing. This rewards risk-adjusted quality.
    results["rank_score"] = (
        0.25 * results["sharpe"].rank(ascending=False, pct=True)
        + 0.20 * results["sortino"].rank(ascending=False, pct=True)
        + 0.20 * results["calmar"].rank(ascending=False, pct=True)
        + 0.20 * results["cagr"].rank(ascending=False, pct=True)
        + 0.15 * results["max_drawdown"].rank(ascending=False, pct=True)
    )

    # Mild penalty for increasing turnover/exposure without clear quality gain.
    turnover_penalty = (
        results["delta_turnover_vs_baseline"]
        .clip(lower=0.0)
        .fillna(0.0)
        / max(float(abs(baseline["annualised_turnover"])), 1e-9)
    ).clip(0.0, 0.05)

    exposure_penalty = (
        results["delta_exposure_vs_baseline"]
        .clip(lower=0.0)
        .fillna(0.0)
    ).clip(0.0, 0.05)

    results["rank_score_penalised"] = (
        results["rank_score"]
        - turnover_penalty
        - exposure_penalty
    )

    results = results.sort_values(
        [
            "rank_score_penalised",
            "sharpe",
            "sortino",
            "calmar",
            "cagr",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    all_results_path = OUTPUT_DIR / "agriculture_overlay_parameter_results.csv"
    top_results_path = OUTPUT_DIR / "agriculture_overlay_top_25.csv"
    viable_results_path = OUTPUT_DIR / "agriculture_overlay_viable_candidates.csv"

    best_by_mix_path = OUTPUT_DIR / "agriculture_overlay_best_by_mix.csv"
    best_by_blend_path = OUTPUT_DIR / "agriculture_overlay_best_by_blend.csv"

    results.to_csv(all_results_path, index=False)
    results.head(25).to_csv(top_results_path, index=False)

    viable = results[
        results["worth_investigating"]
        & (results["test_name"] != "baseline_no_agriculture_overlay")
    ].copy()

    viable.to_csv(viable_results_path, index=False)

    best_by_mix = (
        results[results["test_name"] != "baseline_no_agriculture_overlay"]
        .sort_values("rank_score_penalised", ascending=False)
        .groupby("mix_name", as_index=False)
        .head(1)
        .sort_values("rank_score_penalised", ascending=False)
    )

    best_by_mix.to_csv(best_by_mix_path, index=False)

    best_by_blend = (
        results[results["test_name"] != "baseline_no_agriculture_overlay"]
        .sort_values("rank_score_penalised", ascending=False)
        .groupby("blend_weight", as_index=False)
        .head(1)
        .sort_values("blend_weight")
    )

    best_by_blend.to_csv(best_by_blend_path, index=False)

    print("\nSaved agriculture overlay parameter results:")
    print(f"All results:       {all_results_path}")
    print(f"Top 25:            {top_results_path}")
    print(f"Viable candidates: {viable_results_path}")
    print(f"Best by mix:       {best_by_mix_path}")
    print(f"Best by blend:     {best_by_blend_path}")

    print("\nBaseline:")
    baseline_display_cols = [
        "test_name",
        "final_equity",
        "cagr",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "average_exposure",
        "annualised_turnover",
    ]
    print(baseline[baseline_display_cols].to_string())

    print("\nTop 15 agriculture overlay tests:")
    display_cols = [
        "test_name",
        "blend_weight",
        "relative_strength_weight",
        "usd_weight",
        "seasonality_weight",
        "rates_weight",
        "crop_momentum_weight",
        "broad_weight",
        "export_weight",
        "final_equity",
        "cagr",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "delta_cagr_vs_baseline",
        "delta_sharpe_vs_baseline",
        "delta_maxdd_vs_baseline",
        "rank_score_penalised",
        "worth_investigating",
        "dba_avg_weight",
        "dba_months_held",
    ]

    display_cols = [
        col for col in display_cols
        if col in results.columns
    ]

    print(results.head(15)[display_cols].to_string(index=False))

    print("\nBest result by mix:")
    print(best_by_mix.head(15)[display_cols].to_string(index=False))

    print("\nRun complete.")


if __name__ == "__main__":
    main()
