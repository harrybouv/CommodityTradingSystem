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
# RESOLVE ACTIVE UNG SCORING MODULE
# ============================================================

def get_active_ung_scoring_module():
    scorer = commodity_registry.COMMODITY_SCORERS.get("UNG")

    if scorer is None:
        raise ValueError("Registry has no UNG scorer.")

    module_name = scorer.__module__

    if module_name not in sys.modules:
        __import__(module_name)

    return sys.modules[module_name]


ung_scoring = get_active_ung_scoring_module()


# ============================================================
# TEST GRID
# ============================================================

# Keep this deliberately tight. The individual tests were weak,
# so this is a sanity check, not a broad optimiser.
BLEND_WEIGHTS = [0.03, 0.05, 0.075, 0.10, 0.125, 0.15]


COMPONENT_MIXES = [
    # Best individual candidates / anchors
    {
        "name": "storage_tightness_only",
        "weather": 0.00,
        "storage_tightness": 1.00,
        "storage_momentum": 0.00,
        "curve": 0.00,
        "supply": 0.00,
        "lng": 0.00,
        "oil_relative": 0.00,
        "energy_confirmation": 0.00,
    },
    {
        "name": "storage_momentum_only",
        "weather": 0.00,
        "storage_tightness": 0.00,
        "storage_momentum": 1.00,
        "curve": 0.00,
        "supply": 0.00,
        "lng": 0.00,
        "oil_relative": 0.00,
        "energy_confirmation": 0.00,
    },
    {
        "name": "energy_confirmation_only",
        "weather": 0.00,
        "storage_tightness": 0.00,
        "storage_momentum": 0.00,
        "curve": 0.00,
        "supply": 0.00,
        "lng": 0.00,
        "oil_relative": 0.00,
        "energy_confirmation": 1.00,
    },

    # Storage-led serious candidates
    {
        "name": "storage_equal",
        "weather": 0.00,
        "storage_tightness": 0.50,
        "storage_momentum": 0.50,
        "curve": 0.00,
        "supply": 0.00,
        "lng": 0.00,
        "oil_relative": 0.00,
        "energy_confirmation": 0.00,
    },
    {
        "name": "storage_tightness_heavy",
        "weather": 0.00,
        "storage_tightness": 0.65,
        "storage_momentum": 0.35,
        "curve": 0.00,
        "supply": 0.00,
        "lng": 0.00,
        "oil_relative": 0.00,
        "energy_confirmation": 0.00,
    },
    {
        "name": "storage_momentum_heavy",
        "weather": 0.00,
        "storage_tightness": 0.35,
        "storage_momentum": 0.65,
        "curve": 0.00,
        "supply": 0.00,
        "lng": 0.00,
        "oil_relative": 0.00,
        "energy_confirmation": 0.00,
    },

    # Storage + curve, because UNG is a futures ETF
    {
        "name": "storage_curve_light",
        "weather": 0.00,
        "storage_tightness": 0.40,
        "storage_momentum": 0.40,
        "curve": 0.20,
        "supply": 0.00,
        "lng": 0.00,
        "oil_relative": 0.00,
        "energy_confirmation": 0.00,
    },
    {
        "name": "storage_curve_equal",
        "weather": 0.00,
        "storage_tightness": 0.3333,
        "storage_momentum": 0.3333,
        "curve": 0.3334,
        "supply": 0.00,
        "lng": 0.00,
        "oil_relative": 0.00,
        "energy_confirmation": 0.00,
    },

    # Weather only as light confirmation, not as a dominant signal
    {
        "name": "weather_light_storage",
        "weather": 0.15,
        "storage_tightness": 0.425,
        "storage_momentum": 0.425,
        "curve": 0.00,
        "supply": 0.00,
        "lng": 0.00,
        "oil_relative": 0.00,
        "energy_confirmation": 0.00,
    },
    {
        "name": "weather_storage_curve",
        "weather": 0.15,
        "storage_tightness": 0.35,
        "storage_momentum": 0.35,
        "curve": 0.15,
        "supply": 0.00,
        "lng": 0.00,
        "oil_relative": 0.00,
        "energy_confirmation": 0.00,
    },

    # Energy confirmation looked closest on final equity but worsened drawdown.
    # Test it only as a small supporting feature.
    {
        "name": "storage_energy_light",
        "weather": 0.00,
        "storage_tightness": 0.425,
        "storage_momentum": 0.425,
        "curve": 0.00,
        "supply": 0.00,
        "lng": 0.00,
        "oil_relative": 0.00,
        "energy_confirmation": 0.15,
    },
    {
        "name": "storage_curve_energy_light",
        "weather": 0.00,
        "storage_tightness": 0.35,
        "storage_momentum": 0.35,
        "curve": 0.15,
        "supply": 0.00,
        "lng": 0.00,
        "oil_relative": 0.00,
        "energy_confirmation": 0.15,
    },

    # Theory versions, included to prove whether they fail.
    {
        "name": "theory_core",
        "weather": 0.25,
        "storage_tightness": 0.25,
        "storage_momentum": 0.20,
        "curve": 0.15,
        "supply": 0.10,
        "lng": 0.00,
        "oil_relative": 0.05,
        "energy_confirmation": 0.00,
    },
    {
        "name": "expanded_theory",
        "weather": 0.20,
        "storage_tightness": 0.22,
        "storage_momentum": 0.18,
        "curve": 0.14,
        "supply": 0.10,
        "lng": 0.06,
        "oil_relative": 0.05,
        "energy_confirmation": 0.05,
    },
]


OUTPUT_DIR = RESULTS_DIR / "gas_overlay_parameter_tests"


# ============================================================
# PARAMETER PATCHING
# ============================================================

def set_gas_overlay_params(
    enabled: bool,
    blend_weight: float,
    weather_weight: float,
    storage_tightness_weight: float,
    storage_momentum_weight: float,
    curve_weight: float,
    supply_weight: float,
    lng_weight: float,
    oil_relative_weight: float,
    energy_confirmation_weight: float,
) -> None:
    component_weights = {
        "gas_weather_demand_score": float(weather_weight),
        "gas_storage_tightness_score": float(storage_tightness_weight),
        "gas_storage_momentum_score": float(storage_momentum_weight),
        "gas_curve_roll_score": float(curve_weight),
        "gas_supply_pressure_score": float(supply_weight),
        "gas_lng_export_demand_score": float(lng_weight),
        "gas_oil_relative_value_score": float(oil_relative_weight),
        "gas_energy_confirmation_score": float(energy_confirmation_weight),
    }

    config.UNG_OVERLAY_ENABLED = bool(enabled)
    config.UNG_OVERLAY_BLEND_WEIGHT = float(blend_weight)

    config.UNG_USE_WEATHER_DEMAND = weather_weight > 0
    config.UNG_USE_STORAGE_TIGHTNESS = storage_tightness_weight > 0
    config.UNG_USE_STORAGE_MOMENTUM = storage_momentum_weight > 0
    config.UNG_USE_CURVE_ROLL = curve_weight > 0
    config.UNG_USE_SUPPLY_PRESSURE = supply_weight > 0
    config.UNG_USE_LNG_EXPORT_DEMAND = lng_weight > 0
    config.UNG_USE_OIL_RELATIVE_VALUE = oil_relative_weight > 0
    config.UNG_USE_ENERGY_CONFIRMATION = energy_confirmation_weight > 0
    config.UNG_COMPONENT_WEIGHTS = component_weights

    ung_scoring.UNG_OVERLAY_ENABLED = bool(enabled)
    ung_scoring.UNG_OVERLAY_BLEND_WEIGHT = float(blend_weight)

    ung_scoring.UNG_USE_WEATHER_DEMAND = weather_weight > 0
    ung_scoring.UNG_USE_STORAGE_TIGHTNESS = storage_tightness_weight > 0
    ung_scoring.UNG_USE_STORAGE_MOMENTUM = storage_momentum_weight > 0
    ung_scoring.UNG_USE_CURVE_ROLL = curve_weight > 0
    ung_scoring.UNG_USE_SUPPLY_PRESSURE = supply_weight > 0
    ung_scoring.UNG_USE_LNG_EXPORT_DEMAND = lng_weight > 0
    ung_scoring.UNG_USE_OIL_RELATIVE_VALUE = oil_relative_weight > 0
    ung_scoring.UNG_USE_ENERGY_CONFIRMATION = energy_confirmation_weight > 0
    ung_scoring.UNG_COMPONENT_WEIGHTS = component_weights


# ============================================================
# BENCHMARK HELPERS
# ============================================================

def make_single_asset(
    index: pd.Index,
    tickers: list[str],
    asset: str,
) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=index, columns=tickers)

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
    weather_weight: float,
    storage_tightness_weight: float,
    storage_momentum_weight: float,
    curve_weight: float,
    supply_weight: float,
    lng_weight: float,
    oil_relative_weight: float,
    energy_confirmation_weight: float,
    overlay_enabled: bool = True,
) -> dict:
    set_gas_overlay_params(
        enabled=overlay_enabled,
        blend_weight=blend_weight,
        weather_weight=weather_weight,
        storage_tightness_weight=storage_tightness_weight,
        storage_momentum_weight=storage_momentum_weight,
        curve_weight=curve_weight,
        supply_weight=supply_weight,
        lng_weight=lng_weight,
        oil_relative_weight=oil_relative_weight,
        energy_confirmation_weight=energy_confirmation_weight,
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
    gas_only = make_single_asset(model_weights.index, tickers, "UNG")
    cash = make_cash(model_weights.index, tickers)

    strategies = {
        "model": model_weights,
        "equal_weight": equal_weight,
        "gold_only": gold_only,
        "gas_only": gas_only,
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

    ung_contribution = np.nan
    ung_contribution_share_abs = np.nan

    if not model_contrib.empty and "UNG" in model_contrib["ticker"].values:
        ung_row = model_contrib[model_contrib["ticker"] == "UNG"].iloc[0]
        ung_contribution = ung_row.get("total_return_contribution", np.nan)
        ung_contribution_share_abs = ung_row.get("contribution_share_abs", np.nan)

    ung_avg_weight = (
        model_weights["UNG"].mean()
        if "UNG" in model_weights.columns
        else np.nan
    )

    ung_max_weight = (
        model_weights["UNG"].max()
        if "UNG" in model_weights.columns
        else np.nan
    )

    ung_months_held = (
        int((model_weights["UNG"].resample("ME").last() > 0).sum())
        if "UNG" in model_weights.columns
        else np.nan
    )

    return {
        "test_name": name,
        "overlay_enabled": overlay_enabled,
        "blend_weight": blend_weight,

        "weather_weight": weather_weight,
        "storage_tightness_weight": storage_tightness_weight,
        "storage_momentum_weight": storage_momentum_weight,
        "curve_weight": curve_weight,
        "supply_weight": supply_weight,
        "lng_weight": lng_weight,
        "oil_relative_weight": oil_relative_weight,
        "energy_confirmation_weight": energy_confirmation_weight,

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

        "ung_total_return_contribution": ung_contribution,
        "ung_contribution_share_abs": ung_contribution_share_abs,
        "ung_avg_weight": ung_avg_weight,
        "ung_max_weight": ung_max_weight,
        "ung_months_held": ung_months_held,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading existing score inputs and returns once.")
    print("This avoids rerunning data/scoring pipelines for every parameter set.")

    score_inputs = load_score_inputs()
    returns = load_return_matrix()

    rows = []

    print("\nRunning baseline with gas overlay disabled.")

    rows.append(
        run_one_test(
            name="baseline_no_gas_overlay",
            score_inputs=score_inputs,
            returns=returns,
            blend_weight=0.0,
            weather_weight=0.0,
            storage_tightness_weight=0.0,
            storage_momentum_weight=0.0,
            curve_weight=0.0,
            supply_weight=0.0,
            lng_weight=0.0,
            oil_relative_weight=0.0,
            energy_confirmation_weight=0.0,
            overlay_enabled=False,
        )
    )

    total_tests = len(BLEND_WEIGHTS) * len(COMPONENT_MIXES)
    done = 0

    print(f"\nRunning {total_tests} gas overlay parameter tests.")

    for blend_weight in BLEND_WEIGHTS:
        for mix in COMPONENT_MIXES:
            done += 1

            name = f"{mix['name']}_blend_{blend_weight:.3f}"

            print(
                f"[{done:03d}/{total_tests}] {name} "
                f"(weather={mix['weather']:.2f}, "
                f"tight={mix['storage_tightness']:.2f}, "
                f"mom={mix['storage_momentum']:.2f}, "
                f"curve={mix['curve']:.2f}, "
                f"supply={mix['supply']:.2f}, "
                f"lng={mix['lng']:.2f}, "
                f"oil_rel={mix['oil_relative']:.2f}, "
                f"energy={mix['energy_confirmation']:.2f})"
            )

            row = run_one_test(
                name=name,
                score_inputs=score_inputs,
                returns=returns,
                blend_weight=blend_weight,
                weather_weight=mix["weather"],
                storage_tightness_weight=mix["storage_tightness"],
                storage_momentum_weight=mix["storage_momentum"],
                curve_weight=mix["curve"],
                supply_weight=mix["supply"],
                lng_weight=mix["lng"],
                oil_relative_weight=mix["oil_relative"],
                energy_confirmation_weight=mix["energy_confirmation"],
                overlay_enabled=True,
            )

            rows.append(row)

    results = pd.DataFrame(rows)

    baseline = results[
        results["test_name"] == "baseline_no_gas_overlay"
    ].iloc[0]

    results["delta_cagr_vs_baseline"] = results["cagr"] - baseline["cagr"]
    results["delta_sharpe_vs_baseline"] = results["sharpe"] - baseline["sharpe"]
    results["delta_sortino_vs_baseline"] = results["sortino"] - baseline["sortino"]
    results["delta_calmar_vs_baseline"] = results["calmar"] - baseline["calmar"]
    results["delta_maxdd_vs_baseline"] = results["max_drawdown"] - baseline["max_drawdown"]
    results["delta_final_equity_vs_baseline"] = results["final_equity"] - baseline["final_equity"]
    results["delta_ung_avg_weight_vs_baseline"] = results["ung_avg_weight"] - baseline["ung_avg_weight"]

    # Hard pass: must improve risk-adjusted quality, not just CAGR.
    results["beats_baseline_cleanly"] = (
        (results["delta_cagr_vs_baseline"] > 0.0010)
        & (results["delta_sharpe_vs_baseline"] > 0.0000)
        & (results["delta_sortino_vs_baseline"] > 0.0000)
        & (results["delta_maxdd_vs_baseline"] >= -0.0030)
    )

    # Watchlist: not production, but maybe worth stress/walk-forward testing.
    results["watchlist_candidate"] = (
        (results["delta_cagr_vs_baseline"] > -0.0010)
        & (results["delta_sharpe_vs_baseline"] > -0.0200)
        & (results["delta_maxdd_vs_baseline"] >= -0.0050)
    )

    # Ranking: risk-adjusted, not CAGR-chasing.
    results["rank_score"] = (
        results["sharpe"].rank(ascending=False, pct=True)
        + results["sortino"].rank(ascending=False, pct=True)
        + results["calmar"].rank(ascending=False, pct=True)
        + results["max_drawdown"].rank(ascending=False, pct=True)
        + results["cagr"].rank(ascending=False, pct=True)
    ) / 5.0

    results = results.sort_values(
        ["beats_baseline_cleanly", "watchlist_candidate", "rank_score", "sharpe", "calmar", "cagr"],
        ascending=False,
    ).reset_index(drop=True)

    output_path = OUTPUT_DIR / "gas_overlay_parameter_summary.csv"
    results.to_csv(output_path, index=False)

    display_cols = [
        "test_name",
        "blend_weight",
        "weather_weight",
        "storage_tightness_weight",
        "storage_momentum_weight",
        "curve_weight",
        "supply_weight",
        "lng_weight",
        "oil_relative_weight",
        "energy_confirmation_weight",
        "final_equity",
        "cagr",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "average_exposure",
        "average_cash",
        "ung_total_return_contribution",
        "ung_avg_weight",
        "ung_max_weight",
        "delta_cagr_vs_baseline",
        "delta_sharpe_vs_baseline",
        "delta_maxdd_vs_baseline",
        "beats_baseline_cleanly",
        "watchlist_candidate",
        "rank_score",
    ]

    print("\nSaved results to:")
    print(output_path)

    print("\nTop 15 candidates:")
    print(results[display_cols].head(15).to_string(index=False))

    print("\nClean baseline row:")
    print(
        results[results["test_name"] == "baseline_no_gas_overlay"][display_cols]
        .to_string(index=False)
    )

    print("\nClean pass candidates:")
    clean = results[results["beats_baseline_cleanly"]]
    if clean.empty:
        print("None.")
    else:
        print(clean[display_cols].to_string(index=False))

    print("\nWatchlist candidates:")
    watchlist = results[
        (results["watchlist_candidate"])
        & (results["test_name"] != "baseline_no_gas_overlay")
    ]
    if watchlist.empty:
        print("None.")
    else:
        print(watchlist[display_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()