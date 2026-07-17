from __future__ import annotations
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

"""
parameter_sensitivity_maps.py

Robustness-testing suite for the commodity allocation system.

Outputs:
    - baseline_metrics.csv
    - one_way_sensitivity_results.csv
    - two_way_heatmap_results.csv
    - random_perturbation_results.csv
    - parameter_importance.csv
    - robustness_summary.csv
    - charts/*.png

Purpose:
    Demonstrate whether the frozen strategy is robust to plausible local
    parameter changes. This is NOT an optimiser. Do not promote the best random
    trial to production just because it has the highest backtest Sharpe.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATH SETUP
# ============================================================

THIS_FILE = Path(__file__).resolve()
PARAMETER_DIR = THIS_FILE.parent
RESEARCH_DIR = PARAMETER_DIR.parent
COMMODITY_ROOT = RESEARCH_DIR.parent
PROJECT_ROOT = COMMODITY_ROOT.parent
BACKTESTING_DIR = RESEARCH_DIR / "Backtesting"

for path in [PROJECT_ROOT, COMMODITY_ROOT, RESEARCH_DIR, BACKTESTING_DIR, PARAMETER_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# ============================================================
# IMPORTS
# ============================================================

try:
    from Commodity_System import config as CFG

    from Commodity_System.commodity_strategy import (
        load_score_inputs,
        build_strategy_weights_from_spec,
        build_production_strategy_weight_matrix,
        weights_long_to_matrix,
    )

    from Commodity_System.research.Backtesting.backtester import (
        load_return_matrix,
        apply_rebalance,
        apply_volatility_targeting,
        simulate_strategy,
        make_equal_weight,
    )

except ImportError:
    import config as CFG

    from commodity_strategy import (
        load_score_inputs,
        build_strategy_weights_from_spec,
        build_production_strategy_weight_matrix,
        weights_long_to_matrix,
    )

    try:
        from Backtesting.backtester import (
            load_return_matrix,
            apply_rebalance,
            apply_volatility_targeting,
            simulate_strategy,
            make_equal_weight,
        )
    except ImportError:
        from backtester import (
            load_return_matrix,
            apply_rebalance,
            apply_volatility_targeting,
            simulate_strategy,
            make_equal_weight,
        )

try:
    from analytics import calculate_full_summary
except ImportError:
    from Commodity_System.research.analytics import calculate_full_summary


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42
N_RANDOM_TRIALS = 1000

OUTPUT_DIR = CFG.RESULTS_DIR / "parameter_sensitivity"
CHARTS_DIR = OUTPUT_DIR / "charts"

INITIAL_CAPITAL = float(CFG.INITIAL_CAPITAL)
TRADING_DAYS_PER_YEAR = int(CFG.TRADING_DAYS_PER_YEAR)
TOTAL_COST_BPS = float(CFG.TOTAL_COST_BPS)
REBALANCE_MODE = CFG.BACKTEST_REBALANCE_MODE
RISK_FREE_RATE_ANNUAL = 0.0

BASE_SCORE_WEIGHTS = dict(CFG.SCORE_WEIGHTS)
BASE_MIN_SCORE_TO_HOLD = float(CFG.MIN_SCORE_TO_HOLD)
BASE_MAX_ASSET_WEIGHT = float(CFG.MAX_ASSET_WEIGHT)
BASE_MAX_GROUP_WEIGHT = dict(CFG.MAX_GROUP_WEIGHT)
BASE_MAX_TOTAL_EXPOSURE = float(CFG.MAX_TOTAL_RISK_ASSET_EXPOSURE)

SCORE_WEIGHT_COLS = list(BASE_SCORE_WEIGHTS.keys())

CORE_PARAM_COLS = (
    SCORE_WEIGHT_COLS
    + [
        "min_score_to_hold",
        "max_asset_weight",
    ]
)

REPORT_METRICS = [
    "final_equity",
    "total_return",
    "cagr",
    "annualised_volatility",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "hit_rate",
    "var_95",
    "cvar_95",
    "annualised_turnover",
    "average_daily_turnover",
    "annualised_transaction_cost_drag",
    "average_exposure",
    "average_cash",
    "alpha_annualised",
    "beta",
    "r_squared",
    "information_ratio_active",
]

ONE_WAY_MULTIPLIERS = [0.50, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50]

MIN_SCORE_GRID = [0.55, 0.60, 0.625, 0.65, 0.675, 0.70, 0.75]
MAX_ASSET_WEIGHT_GRID = [0.24, 0.28, 0.30, 0.32, 0.34, 0.36, 0.40]

RISK_WEIGHT_GRID = [0.13, 0.18, 0.22, 0.26, 0.30, 0.34, 0.39]
VOL_WEIGHT_GRID = [0.095, 0.13, 0.16, 0.19, 0.22, 0.25, 0.285]

MOMENTUM_WEIGHT_GRID = [0.115, 0.16, 0.20, 0.23, 0.26, 0.30, 0.345]
RELATIVE_STRENGTH_WEIGHT_GRID = [0.08, 0.11, 0.14, 0.16, 0.18, 0.21, 0.24]

NO_TRADE_BAND_GRID = [0.000, 0.0025, 0.0050, 0.0075, 0.0100, 0.0150, 0.0200]
MAX_REBALANCE_TURNOVER_GRID = [0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.00]

ROBUSTNESS_FILTERS = {
    "min_cagr": 0.08,
    "min_sharpe": 0.85,
    "max_drawdown_floor": -0.22,
    "max_annualised_volatility": 0.16,
    "max_annualised_turnover": 10.0,
}


# ============================================================
# BASIC HELPERS
# ============================================================

def prepare_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_filename(text: str) -> str:
    out = str(text).lower().strip()
    out = out.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return out


def clean_metric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in REPORT_METRICS + CORE_PARAM_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out.replace([np.inf, -np.inf], np.nan)


def normalise_score_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {
        key: max(0.0, float(value))
        for key, value in weights.items()
    }

    total = sum(cleaned.values())

    if total <= 0:
        raise ValueError("Score weights sum to zero.")

    return {
        key: value / total
        for key, value in cleaned.items()
    }


def base_params() -> dict[str, float]:
    weights = normalise_score_weights(BASE_SCORE_WEIGHTS)

    return {
        **weights,
        "min_score_to_hold": BASE_MIN_SCORE_TO_HOLD,
        "max_asset_weight": BASE_MAX_ASSET_WEIGHT,
    }


def params_to_score_weights(params: dict[str, Any]) -> dict[str, float]:
    return normalise_score_weights({
        col: safe_float(params[col])
        for col in SCORE_WEIGHT_COLS
    })


def make_params(
    score_weights: dict[str, float] | None = None,
    min_score_to_hold: float | None = None,
    max_asset_weight: float | None = None,
) -> dict[str, float]:
    if score_weights is None:
        score_weights = BASE_SCORE_WEIGHTS

    weights = normalise_score_weights(score_weights)

    return {
        **weights,
        "min_score_to_hold": (
            BASE_MIN_SCORE_TO_HOLD
            if min_score_to_hold is None
            else float(min_score_to_hold)
        ),
        "max_asset_weight": (
            BASE_MAX_ASSET_WEIGHT
            if max_asset_weight is None
            else float(max_asset_weight)
        ),
    }


# ============================================================
# PARAMETER TRANSFORMS
# ============================================================

def adjust_one_score_weight(
    base: dict[str, float],
    selected_col: str,
    multiplier: float,
) -> dict[str, float]:
    base = normalise_score_weights(base)

    selected_new = base[selected_col] * float(multiplier)
    selected_new = max(0.0, selected_new)

    remaining_cols = [col for col in base if col != selected_col]
    remaining_old_total = sum(base[col] for col in remaining_cols)
    remaining_new_total = 1.0 - selected_new

    if remaining_new_total <= 0:
        raise ValueError(
            f"{selected_col} multiplier {multiplier} leaves no room for other weights."
        )

    out = {selected_col: selected_new}

    for col in remaining_cols:
        out[col] = base[col] / remaining_old_total * remaining_new_total

    return normalise_score_weights(out)


def set_two_score_weights(
    base: dict[str, float],
    first_col: str,
    first_value: float,
    second_col: str,
    second_value: float,
) -> dict[str, float]:
    base = normalise_score_weights(base)

    if first_col == second_col:
        raise ValueError("Two-way weight test requires two different columns.")

    first_value = max(0.0, float(first_value))
    second_value = max(0.0, float(second_value))

    fixed_total = first_value + second_value

    if fixed_total >= 0.98:
        raise ValueError(
            f"{first_col} + {second_col} too high: {fixed_total:.3f}"
        )

    remaining_cols = [
        col for col in base
        if col not in [first_col, second_col]
    ]

    remaining_old_total = sum(base[col] for col in remaining_cols)
    remaining_new_total = 1.0 - fixed_total

    out = {
        first_col: first_value,
        second_col: second_value,
    }

    for col in remaining_cols:
        out[col] = base[col] / remaining_old_total * remaining_new_total

    return normalise_score_weights(out)


# ============================================================
# DATA LOADING
# ============================================================

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading score inputs.")
    scores = load_score_inputs()
    scores["date"] = pd.to_datetime(scores["date"])

    missing = [col for col in SCORE_WEIGHT_COLS if col not in scores.columns]
    if missing:
        raise ValueError(f"Missing score columns: {missing}")

    print("Loading return matrix.")
    returns = load_return_matrix()
    returns.index = pd.to_datetime(returns.index)

    returns = (
        returns
        .sort_index()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    return scores, returns


# ============================================================
# STRATEGY CONSTRUCTION
# ============================================================

def build_core_raw_weights(
    scores: pd.DataFrame,
    params: dict[str, Any],
) -> pd.DataFrame:
    score_weights = params_to_score_weights(params)

    spec = {
        "weights": score_weights,
        "normalise": True,
    }

    weights_long = build_strategy_weights_from_spec(
        scores=scores,
        spec=spec,
        min_score_to_hold=float(params["min_score_to_hold"]),
        max_asset_weight=float(params["max_asset_weight"]),
        max_group_weight=BASE_MAX_GROUP_WEIGHT,
        max_total_exposure=BASE_MAX_TOTAL_EXPOSURE,
    )

    weights_matrix = weights_long_to_matrix(weights_long)
    weights_matrix.index = pd.to_datetime(weights_matrix.index)

    return weights_matrix.sort_index().fillna(0.0)


def build_exact_production_raw_weights(scores: pd.DataFrame) -> pd.DataFrame:
    weights = build_production_strategy_weight_matrix(scores=scores)
    weights.index = pd.to_datetime(weights.index)

    return weights.sort_index().fillna(0.0)


# ============================================================
# EXECUTION-CONTROL ENGINE
# ============================================================

def get_scheduled_rebalance_dates(
    index: pd.DatetimeIndex,
    mode: str,
) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(index).sort_values()

    if len(index) == 0:
        return index

    mode = str(mode).lower()

    if mode == "daily":
        return index

    date_series = pd.Series(index=index, data=index)

    if mode == "weekly":
        return pd.DatetimeIndex(
            date_series.groupby(index.to_period("W-FRI")).max().values
        )

    if mode == "monthly":
        return pd.DatetimeIndex(
            date_series.groupby(index.to_period("M")).max().values
        )

    raise ValueError(f"Unsupported rebalance mode: {mode}")


def apply_rebalance_with_controls(
    raw_weights: pd.DataFrame,
    mode: str = REBALANCE_MODE,
    no_trade_band: float = 0.0,
    max_rebalance_turnover: float | None = None,
    cap_first_rebalance: bool = False,
) -> pd.DataFrame:
    """
    Local execution-control engine for sensitivity testing.

    This avoids mutating config.py. It approximates no-trade-band and
    max-rebalance-turnover effects to test implementation robustness.
    """
    raw = raw_weights.copy()
    raw.index = pd.to_datetime(raw.index)

    raw = (
        raw
        .sort_index()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
    )

    scheduled_dates = get_scheduled_rebalance_dates(raw.index, mode)
    scheduled = raw.loc[scheduled_dates].copy()

    previous = None
    rows = []

    no_trade_band = max(0.0, float(no_trade_band))

    if max_rebalance_turnover is not None:
        max_rebalance_turnover = max(0.0, float(max_rebalance_turnover))

    for _, desired in scheduled.iterrows():
        desired = desired.fillna(0.0).clip(lower=0.0)

        if previous is None:
            if cap_first_rebalance and max_rebalance_turnover is not None:
                zero = pd.Series(0.0, index=desired.index)
                trade = desired - zero
                turnover = float(trade.abs().sum())

                if turnover > max_rebalance_turnover and turnover > 0:
                    desired = zero + trade * (max_rebalance_turnover / turnover)

            executed = desired.copy()

        else:
            trade = desired - previous

            if no_trade_band > 0:
                trade = trade.where(trade.abs() >= no_trade_band, 0.0)

            turnover = float(trade.abs().sum())

            if (
                max_rebalance_turnover is not None
                and turnover > max_rebalance_turnover
                and turnover > 0
            ):
                trade = trade * (max_rebalance_turnover / turnover)

            executed = previous + trade

        executed = executed.fillna(0.0).clip(lower=0.0)

        total = float(executed.sum())
        if total > 1.0:
            executed = executed / total

        rows.append(executed)
        previous = executed.copy()

    sparse = pd.DataFrame(
        rows,
        index=scheduled.index,
        columns=raw.columns,
    )

    daily = (
        sparse
        .reindex(raw.index)
        .ffill()
        .fillna(0.0)
    )

    return daily


# ============================================================
# BACKTEST WRAPPERS
# ============================================================

def build_benchmark_curve(
    raw_index: pd.DatetimeIndex,
    tickers: list[str],
    returns: pd.DataFrame,
) -> pd.DataFrame:
    equal_weight = make_equal_weight(raw_index, tickers)
    equal_weight = apply_rebalance(equal_weight, REBALANCE_MODE)
    equal_weight, _ = apply_volatility_targeting(equal_weight, returns)

    curve, _ = simulate_strategy(
        name="equal_weight",
        target_weights=equal_weight,
        returns=returns,
        initial_capital=INITIAL_CAPITAL,
        total_cost_bps=TOTAL_COST_BPS,
    )

    return curve


def summarise_curve(
    name: str,
    curve: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, Any]:
    return calculate_full_summary(
        returns=curve["net_return"],
        equity=curve["equity"],
        turnover=curve["turnover"],
        transaction_cost=curve["transaction_cost"],
        exposure=curve["exposure"],
        benchmark_returns=benchmark_returns,
        strategy_name=name,
        benchmark_name="equal_weight" if benchmark_returns is not None else None,
        initial_capital=INITIAL_CAPITAL,
        risk_free_rate_annual=RISK_FREE_RATE_ANNUAL,
        periods_per_year=TRADING_DAYS_PER_YEAR,
    )


def run_backtest_from_raw_weights(
    name: str,
    raw_weights: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    execution_controls: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    raw_weights = raw_weights.copy()
    raw_weights.index = pd.to_datetime(raw_weights.index)
    raw_weights = raw_weights.sort_index().fillna(0.0)

    if execution_controls:
        executed_weights = apply_rebalance_with_controls(
            raw_weights=raw_weights,
            mode=execution_controls.get("rebalance_mode", REBALANCE_MODE),
            no_trade_band=execution_controls.get("no_trade_band", 0.0),
            max_rebalance_turnover=execution_controls.get("max_rebalance_turnover"),
            cap_first_rebalance=execution_controls.get("cap_first_rebalance", False),
        )
    else:
        executed_weights = apply_rebalance(raw_weights, REBALANCE_MODE)

    executed_weights, _ = apply_volatility_targeting(
        executed_weights,
        returns,
    )

    curve, _ = simulate_strategy(
        name=name,
        target_weights=executed_weights,
        returns=returns,
        initial_capital=INITIAL_CAPITAL,
        total_cost_bps=TOTAL_COST_BPS,
    )

    summary = summarise_curve(
        name=name,
        curve=curve,
        benchmark_returns=benchmark_returns,
    )

    return summary, curve, executed_weights


def run_param_case(
    name: str,
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark_returns: pd.Series | None,
    params: dict[str, Any],
) -> dict[str, Any]:
    raw_weights = build_core_raw_weights(scores, params)

    summary, _, _ = run_backtest_from_raw_weights(
        name=name,
        raw_weights=raw_weights,
        returns=returns,
        benchmark_returns=benchmark_returns,
    )

    return {
        "case": name,
        **params,
        **summary,
    }


# ============================================================
# BASELINES
# ============================================================

def run_baselines(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    print("\nRunning baselines.")

    production_raw = build_exact_production_raw_weights(scores)
    tickers = list(production_raw.columns)

    benchmark_curve = build_benchmark_curve(
        raw_index=production_raw.index,
        tickers=tickers,
        returns=returns,
    )

    benchmark_returns = benchmark_curve["net_return"]

    production_summary, production_curve, production_weights = run_backtest_from_raw_weights(
        name="production_exact",
        raw_weights=production_raw,
        returns=returns,
        benchmark_returns=benchmark_returns,
    )

    core_raw = build_core_raw_weights(scores, base_params())

    core_summary, core_curve, core_weights = run_backtest_from_raw_weights(
        name="core_score_baseline",
        raw_weights=core_raw,
        returns=returns,
        benchmark_returns=benchmark_returns,
    )

    equal_weight_summary = summarise_curve(
        name="equal_weight",
        curve=benchmark_curve,
        benchmark_returns=None,
    )

    out = clean_metric_frame(pd.DataFrame(
        [
            {"baseline_type": "production_exact", **production_summary},
            {"baseline_type": "core_score_baseline", **core_summary},
            {"baseline_type": "equal_weight", **equal_weight_summary},
        ]
    ))

    out.to_csv(OUTPUT_DIR / "baseline_metrics.csv", index=False)

    production_curve.reset_index().to_csv(
        OUTPUT_DIR / "production_exact_curve.csv",
        index=False,
    )
    core_curve.reset_index().to_csv(
        OUTPUT_DIR / "core_score_baseline_curve.csv",
        index=False,
    )
    benchmark_curve.reset_index().to_csv(
        OUTPUT_DIR / "equal_weight_curve.csv",
        index=False,
    )

    production_weights.to_csv(OUTPUT_DIR / "production_exact_executed_weights.csv")
    core_weights.to_csv(OUTPUT_DIR / "core_score_baseline_executed_weights.csv")

    return out, production_raw, benchmark_returns


# ============================================================
# ONE-WAY SENSITIVITY
# ============================================================

def run_one_way_sensitivity(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark_returns: pd.Series | None,
) -> pd.DataFrame:
    print("\nRunning one-way sensitivity.")

    rows = []

    for param in SCORE_WEIGHT_COLS:
        for multiplier in ONE_WAY_MULTIPLIERS:
            adjusted = adjust_one_score_weight(
                BASE_SCORE_WEIGHTS,
                selected_col=param,
                multiplier=multiplier,
            )

            params = make_params(score_weights=adjusted)

            row = run_param_case(
                name=f"one_way_{param}_{multiplier:.2f}x",
                scores=scores,
                returns=returns,
                benchmark_returns=benchmark_returns,
                params=params,
            )

            row.update(
                {
                    "test_family": "one_way",
                    "parameter": param,
                    "parameter_type": "score_weight",
                    "multiplier": multiplier,
                    "parameter_value": params[param],
                }
            )

            rows.append(row)
            print(f"  one-way completed: {len(rows)} cases", flush=True)

    for value in MIN_SCORE_GRID:
        params = make_params(min_score_to_hold=value)

        row = run_param_case(
            name=f"one_way_min_score_to_hold_{value:.4f}",
            scores=scores,
            returns=returns,
            benchmark_returns=benchmark_returns,
            params=params,
        )

        row.update(
            {
                "test_family": "one_way",
                "parameter": "min_score_to_hold",
                "parameter_type": "portfolio_construction",
                "multiplier": value / BASE_MIN_SCORE_TO_HOLD,
                "parameter_value": value,
            }
        )

        rows.append(row)

    for value in MAX_ASSET_WEIGHT_GRID:
        params = make_params(max_asset_weight=value)

        row = run_param_case(
            name=f"one_way_max_asset_weight_{value:.4f}",
            scores=scores,
            returns=returns,
            benchmark_returns=benchmark_returns,
            params=params,
        )

        row.update(
            {
                "test_family": "one_way",
                "parameter": "max_asset_weight",
                "parameter_type": "portfolio_construction",
                "multiplier": value / BASE_MAX_ASSET_WEIGHT,
                "parameter_value": value,
            }
        )

        rows.append(row)

    out = clean_metric_frame(pd.DataFrame(rows))
    out.to_csv(OUTPUT_DIR / "one_way_sensitivity_results.csv", index=False)

    return out


# ============================================================
# TWO-WAY HEATMAPS
# ============================================================

def add_heatmap_metadata(
    row: dict[str, Any],
    heatmap_name: str,
    x_param: str,
    x_value: float,
    y_param: str,
    y_value: float,
) -> dict[str, Any]:
    row.update(
        {
            "test_family": "two_way_heatmap",
            "heatmap_name": heatmap_name,
            "x_param": x_param,
            "x_value": x_value,
            "y_param": y_param,
            "y_value": y_value,
        }
    )

    return row


def run_two_way_heatmaps(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark_returns: pd.Series | None,
    production_raw_weights: pd.DataFrame,
) -> pd.DataFrame:
    print("\nRunning two-way heatmaps.")

    rows = []

    heatmap = "min_score_to_hold_x_max_asset_weight"

    for min_score in MIN_SCORE_GRID:
        for max_weight in MAX_ASSET_WEIGHT_GRID:
            params = make_params(
                min_score_to_hold=min_score,
                max_asset_weight=max_weight,
            )

            row = run_param_case(
                name=f"{heatmap}_{min_score:.4f}_{max_weight:.4f}",
                scores=scores,
                returns=returns,
                benchmark_returns=benchmark_returns,
                params=params,
            )

            rows.append(add_heatmap_metadata(
                row=row,
                heatmap_name=heatmap,
                x_param="min_score_to_hold",
                x_value=min_score,
                y_param="max_asset_weight",
                y_value=max_weight,
            ))

    heatmap = "risk_weight_x_volatility_weight"

    for risk_weight in RISK_WEIGHT_GRID:
        for vol_weight in VOL_WEIGHT_GRID:
            adjusted = set_two_score_weights(
                base=BASE_SCORE_WEIGHTS,
                first_col="risk_score",
                first_value=risk_weight,
                second_col="volatility_score",
                second_value=vol_weight,
            )

            params = make_params(score_weights=adjusted)

            row = run_param_case(
                name=f"{heatmap}_{risk_weight:.4f}_{vol_weight:.4f}",
                scores=scores,
                returns=returns,
                benchmark_returns=benchmark_returns,
                params=params,
            )

            rows.append(add_heatmap_metadata(
                row=row,
                heatmap_name=heatmap,
                x_param="risk_score",
                x_value=risk_weight,
                y_param="volatility_score",
                y_value=vol_weight,
            ))
    print(f"  heatmaps completed: {len(rows)} cases", flush=True)
    heatmap = "momentum_weight_x_relative_strength_weight"

    for momentum_weight in MOMENTUM_WEIGHT_GRID:
        for rs_weight in RELATIVE_STRENGTH_WEIGHT_GRID:
            adjusted = set_two_score_weights(
                base=BASE_SCORE_WEIGHTS,
                first_col="momentum_score",
                first_value=momentum_weight,
                second_col="relative_strength_score",
                second_value=rs_weight,
            )

            params = make_params(score_weights=adjusted)

            row = run_param_case(
                name=f"{heatmap}_{momentum_weight:.4f}_{rs_weight:.4f}",
                scores=scores,
                returns=returns,
                benchmark_returns=benchmark_returns,
                params=params,
            )

            rows.append(add_heatmap_metadata(
                row=row,
                heatmap_name=heatmap,
                x_param="momentum_score",
                x_value=momentum_weight,
                y_param="relative_strength_score",
                y_value=rs_weight,
            ))

    heatmap = "no_trade_band_x_max_rebalance_turnover"

    for band in NO_TRADE_BAND_GRID:
        for max_turnover in MAX_REBALANCE_TURNOVER_GRID:
            execution_controls = {
                "rebalance_mode": REBALANCE_MODE,
                "no_trade_band": band,
                "max_rebalance_turnover": max_turnover,
                "cap_first_rebalance": False,
            }

            summary, _, _ = run_backtest_from_raw_weights(
                name=f"{heatmap}_{band:.4f}_{max_turnover:.4f}",
                raw_weights=production_raw_weights,
                returns=returns,
                benchmark_returns=benchmark_returns,
                execution_controls=execution_controls,
            )

            row = {
                "case": f"{heatmap}_{band:.4f}_{max_turnover:.4f}",
                **base_params(),
                "no_trade_band": band,
                "max_rebalance_turnover": max_turnover,
                **summary,
            }

            rows.append(add_heatmap_metadata(
                row=row,
                heatmap_name=heatmap,
                x_param="no_trade_band",
                x_value=band,
                y_param="max_rebalance_turnover",
                y_value=max_turnover,
            ))

    out = clean_metric_frame(pd.DataFrame(rows))
    out.to_csv(OUTPUT_DIR / "two_way_heatmap_results.csv", index=False)

    return out


# ============================================================
# RANDOM PERTURBATION CLOUD
# ============================================================

def sample_random_params(rng: np.random.Generator) -> dict[str, float]:
    base = normalise_score_weights(BASE_SCORE_WEIGHTS)

    base_vector = np.array(
        [base[col] for col in SCORE_WEIGHT_COLS],
        dtype=float,
    )

    noise = rng.lognormal(
        mean=0.0,
        sigma=0.25,
        size=len(base_vector),
    )

    sampled = base_vector * noise
    sampled = np.clip(sampled, 1e-6, None)
    sampled = sampled / sampled.sum()

    score_weights = {
        col: sampled[i]
        for i, col in enumerate(SCORE_WEIGHT_COLS)
    }

    min_score = float(np.clip(
        BASE_MIN_SCORE_TO_HOLD + rng.normal(0.0, 0.04),
        0.55,
        0.75,
    ))

    max_asset = float(np.clip(
        BASE_MAX_ASSET_WEIGHT + rng.normal(0.0, 0.035),
        0.24,
        0.40,
    ))

    return make_params(
        score_weights=score_weights,
        min_score_to_hold=min_score,
        max_asset_weight=max_asset,
    )


def passes_robustness_filters(row: pd.Series) -> bool:
    return bool(
        safe_float(row.get("cagr")) >= ROBUSTNESS_FILTERS["min_cagr"]
        and safe_float(row.get("sharpe")) >= ROBUSTNESS_FILTERS["min_sharpe"]
        and safe_float(row.get("max_drawdown")) >= ROBUSTNESS_FILTERS["max_drawdown_floor"]
        and safe_float(row.get("annualised_volatility")) <= ROBUSTNESS_FILTERS["max_annualised_volatility"]
        and safe_float(row.get("annualised_turnover")) <= ROBUSTNESS_FILTERS["max_annualised_turnover"]
    )


def run_random_perturbation_cloud(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark_returns: pd.Series | None,
) -> pd.DataFrame:
    print(f"\nRunning random perturbation cloud: {N_RANDOM_TRIALS:,} trials.")

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for trial in range(1, N_RANDOM_TRIALS + 1):
        params = sample_random_params(rng)

        row = run_param_case(
            name=f"random_perturbation_{trial:04d}",
            scores=scores,
            returns=returns,
            benchmark_returns=benchmark_returns,
            params=params,
        )

        row.update(
            {
                "test_family": "random_perturbation",
                "trial": trial,
            }
        )

        rows.append(row)

        if trial % 100 == 0:
            print(f"  completed {trial:,}/{N_RANDOM_TRIALS:,}")

    out = clean_metric_frame(pd.DataFrame(rows))

    out["passes_robustness_filters"] = out.apply(
        passes_robustness_filters,
        axis=1,
    )

    out.to_csv(OUTPUT_DIR / "random_perturbation_results.csv", index=False)

    return out


# ============================================================
# PARAMETER IMPORTANCE
# ============================================================

def ridge_coefficients(
    df: pd.DataFrame,
    param_cols: list[str],
    metric: str,
    alpha: float = 1.0,
) -> pd.Series:
    data = df[param_cols + [metric]].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(data) < max(30, len(param_cols) + 5):
        return pd.Series(np.nan, index=param_cols)

    x = data[param_cols].astype(float)
    y = data[metric].astype(float)

    x_std = x.std(ddof=0).replace(0.0, np.nan)
    y_std = y.std(ddof=0)

    if not np.isfinite(y_std) or y_std == 0:
        return pd.Series(np.nan, index=param_cols)

    xz = ((x - x.mean()) / x_std).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)

    yz = (y - y.mean()) / y_std

    x_mat = xz.to_numpy(dtype=float)
    y_vec = yz.to_numpy(dtype=float)

    identity = np.eye(x_mat.shape[1])
    beta = np.linalg.pinv(x_mat.T @ x_mat + alpha * identity) @ x_mat.T @ y_vec

    return pd.Series(beta, index=param_cols)


def build_parameter_importance(random_df: pd.DataFrame) -> pd.DataFrame:
    print("\nBuilding parameter importance.")

    if random_df.empty:
        return pd.DataFrame()

    df = clean_metric_frame(random_df)

    param_cols = [col for col in CORE_PARAM_COLS if col in df.columns]

    metrics = [
        "cagr",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "annualised_volatility",
        "annualised_turnover",
        "average_exposure",
        "average_cash",
    ]

    metrics = [metric for metric in metrics if metric in df.columns]

    rows = []

    for metric in metrics:
        ridge = ridge_coefficients(df, param_cols, metric, alpha=1.0)

        for param in param_cols:
            pair = df[[param, metric]].replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()

            if len(pair) >= 10:
                spearman = pair[param].rank().corr(pair[metric].rank(), method="pearson")
                pearson = pair[param].corr(pair[metric], method="pearson")
            else:
                spearman = np.nan
                pearson = np.nan

            beta = ridge.get(param, np.nan)

            rows.append(
                {
                    "metric": metric,
                    "parameter": param,
                    "spearman_rank_correlation": spearman,
                    "pearson_correlation": pearson,
                    "ridge_standardised_beta": beta,
                    "abs_ridge_standardised_beta": abs(beta),
                }
            )

    out = pd.DataFrame(rows)

    out = out.sort_values(
        ["metric", "abs_ridge_standardised_beta"],
        ascending=[True, False],
    )

    out.to_csv(OUTPUT_DIR / "parameter_importance.csv", index=False)

    return out


# ============================================================
# ROBUSTNESS SUMMARY
# ============================================================

def percentile_of_baseline(
    values: pd.Series,
    baseline_value: float,
    higher_is_better: bool = True,
) -> float:
    values = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if values.empty or not np.isfinite(baseline_value):
        return np.nan

    if higher_is_better:
        return float((values <= baseline_value).mean())

    return float((values >= baseline_value).mean())


def build_robustness_summary(
    baseline_df: pd.DataFrame,
    random_df: pd.DataFrame,
) -> pd.DataFrame:
    print("\nBuilding robustness summary.")

    if random_df.empty:
        return pd.DataFrame()

    baseline_row = baseline_df[
        baseline_df["baseline_type"] == "core_score_baseline"
    ]

    if baseline_row.empty:
        baseline_row = baseline_df.head(1)

    baseline = baseline_row.iloc[0]
    df = clean_metric_frame(random_df)

    rows = []

    metric_direction = {
        "cagr": True,
        "sharpe": True,
        "sortino": True,
        "calmar": True,
        "max_drawdown": True,
        "annualised_volatility": False,
        "annualised_turnover": False,
        "average_exposure": False,
    }

    for metric, higher_is_better in metric_direction.items():
        if metric not in df.columns or metric not in baseline.index:
            continue

        series = pd.to_numeric(df[metric], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        baseline_value = safe_float(baseline[metric])

        rows.extend(
            [
                {"statistic": f"{metric}_baseline", "value": baseline_value},
                {"statistic": f"{metric}_random_median", "value": series.median()},
                {"statistic": f"{metric}_random_p10", "value": series.quantile(0.10)},
                {"statistic": f"{metric}_random_p90", "value": series.quantile(0.90)},
                {
                    "statistic": f"{metric}_baseline_percentile_vs_random",
                    "value": percentile_of_baseline(
                        values=series,
                        baseline_value=baseline_value,
                        higher_is_better=higher_is_better,
                    ),
                },
            ]
        )

    rows.extend(
        [
            {"statistic": "random_trial_count", "value": len(df)},
            {
                "statistic": "share_random_trials_sharpe_above_threshold",
                "value": (df["sharpe"] >= ROBUSTNESS_FILTERS["min_sharpe"]).mean(),
            },
            {
                "statistic": "share_random_trials_cagr_above_threshold",
                "value": (df["cagr"] >= ROBUSTNESS_FILTERS["min_cagr"]).mean(),
            },
            {
                "statistic": "share_random_trials_drawdown_above_floor",
                "value": (df["max_drawdown"] >= ROBUSTNESS_FILTERS["max_drawdown_floor"]).mean(),
            },
            {
                "statistic": "share_random_trials_passing_all_filters",
                "value": df["passes_robustness_filters"].mean(),
            },
        ]
    )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "robustness_summary.csv", index=False)

    return out


# ============================================================
# CHARTS
# ============================================================

def save_one_way_charts(df: pd.DataFrame) -> None:
    if df.empty:
        return

    metrics = ["sharpe", "cagr", "max_drawdown", "annualised_turnover"]

    for metric in metrics:
        if metric not in df.columns:
            continue

        plt.figure(figsize=(12, 7))

        for parameter, group in df.groupby("parameter"):
            group = group.sort_values("multiplier")
            plt.plot(
                group["multiplier"],
                group[metric],
                marker="o",
                label=parameter,
            )

        plt.axvline(1.0, linestyle="--", linewidth=1)
        plt.title(f"One-way sensitivity: {metric}")
        plt.xlabel("Parameter value / baseline value")
        plt.ylabel(metric)
        plt.legend(fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / f"01_one_way_{safe_filename(metric)}.png", dpi=160)
        plt.close()

    if "sharpe" in df.columns:
        tornado = (
            df.groupby("parameter")["sharpe"]
            .agg(["min", "max"])
            .reset_index()
        )

        tornado["range"] = tornado["max"] - tornado["min"]
        tornado = tornado.sort_values("range", ascending=True)

        plt.figure(figsize=(9, 6))
        plt.barh(tornado["parameter"], tornado["range"])
        plt.title("One-way Sharpe sensitivity range")
        plt.xlabel("Max Sharpe - Min Sharpe")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / "01_one_way_tornado_sharpe.png", dpi=160)
        plt.close()


def save_heatmap_charts(df: pd.DataFrame) -> None:
    if df.empty:
        return

    metrics = ["sharpe", "cagr", "max_drawdown", "annualised_turnover"]

    for heatmap_name, sub in df.groupby("heatmap_name"):
        x_param = sub["x_param"].iloc[0]
        y_param = sub["y_param"].iloc[0]

        for metric in metrics:
            if metric not in sub.columns:
                continue

            pivot = sub.pivot_table(
                index="y_value",
                columns="x_value",
                values=metric,
                aggfunc="mean",
            ).sort_index()

            if pivot.empty:
                continue

            plt.figure(figsize=(9, 7))
            image = plt.imshow(pivot.values, aspect="auto", origin="lower")
            plt.colorbar(image, label=metric)

            plt.xticks(
                range(len(pivot.columns)),
                [f"{x:.4g}" for x in pivot.columns],
                rotation=45,
                ha="right",
            )

            plt.yticks(
                range(len(pivot.index)),
                [f"{y:.4g}" for y in pivot.index],
            )

            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    value = pivot.iloc[i, j]

                    if pd.notna(value):
                        if metric in ["cagr", "max_drawdown", "annualised_turnover"]:
                            label = f"{value:.1%}"
                        else:
                            label = f"{value:.2f}"

                        plt.text(j, i, label, ha="center", va="center", fontsize=7)

            plt.title(f"{heatmap_name}: {metric}")
            plt.xlabel(x_param)
            plt.ylabel(y_param)
            plt.tight_layout()
            plt.savefig(
                CHARTS_DIR / f"02_heatmap_{safe_filename(heatmap_name)}_{safe_filename(metric)}.png",
                dpi=160,
            )
            plt.close()


def save_random_cloud_charts(
    random_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> None:
    if random_df.empty:
        return

    df = clean_metric_frame(random_df)

    baseline_row = baseline_df[
        baseline_df["baseline_type"] == "core_score_baseline"
    ]

    if baseline_row.empty:
        baseline_row = baseline_df.head(1)

    baseline = baseline_row.iloc[0]

    if {"cagr", "sharpe", "max_drawdown", "annualised_turnover"}.issubset(df.columns):
        size = df["annualised_turnover"].fillna(0.0)

        if size.max() > 0:
            size = 20 + 80 * (size / size.max())
        else:
            size = 30

        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            df["cagr"],
            df["sharpe"],
            c=df["max_drawdown"],
            s=size,
            alpha=0.65,
        )

        plt.colorbar(scatter, label="Max drawdown")

        plt.scatter(
            [baseline["cagr"]],
            [baseline["sharpe"]],
            marker="*",
            s=300,
            edgecolors="black",
            label="Core baseline",
        )

        plt.title("Random perturbation cloud")
        plt.xlabel("CAGR")
        plt.ylabel("Sharpe")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / "03_random_cloud_cagr_sharpe.png", dpi=180)
        plt.close()

    for metric in [
        "cagr",
        "sharpe",
        "max_drawdown",
        "annualised_volatility",
        "annualised_turnover",
        "average_exposure",
    ]:
        if metric not in df.columns or metric not in baseline.index:
            continue

        values = pd.to_numeric(df[metric], errors="coerce").dropna()

        if values.empty:
            continue

        baseline_value = safe_float(baseline[metric])

        plt.figure(figsize=(9, 5))
        plt.hist(values, bins=40, alpha=0.80)

        if np.isfinite(baseline_value):
            plt.axvline(
                baseline_value,
                linestyle="--",
                linewidth=2,
                label="Core baseline",
            )

        plt.title(f"Random perturbation distribution: {metric}")
        plt.xlabel(metric)
        plt.ylabel("Trial count")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / f"03_random_hist_{safe_filename(metric)}.png", dpi=160)
        plt.close()


def save_importance_charts(df: pd.DataFrame) -> None:
    if df.empty:
        print("  no parameter importance data to chart.", flush=True)
        return

    for metric in ["sharpe", "cagr", "max_drawdown", "annualised_turnover"]:
        sub = df[df["metric"] == metric].copy()

        if sub.empty:
            continue

        for col in [
            "ridge_standardised_beta",
            "spearman_rank_correlation",
            "pearson_correlation",
        ]:
            if col in sub.columns:
                sub[col] = pd.to_numeric(sub[col], errors="coerce")

        plot_col = "ridge_standardised_beta"

        if plot_col not in sub.columns or sub[plot_col].isna().all():
            plot_col = "spearman_rank_correlation"

        if plot_col not in sub.columns or sub[plot_col].isna().all():
            plot_col = "pearson_correlation"

        sub = sub.dropna(subset=[plot_col])

        if sub.empty:
            print(f"  no valid importance values for {metric}", flush=True)
            continue

        sub = sub.sort_values(plot_col, ascending=True)

        plt.figure(figsize=(10, 6))
        plt.barh(sub["parameter"], sub[plot_col])
        plt.axvline(0.0, linewidth=1)
        plt.title(f"Parameter importance: {metric}")
        plt.xlabel(plot_col)
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            CHARTS_DIR / f"04_parameter_importance_{safe_filename(metric)}.png",
            dpi=160,
        )
        plt.close()


def save_baseline_charts(df: pd.DataFrame) -> None:
    if df.empty:
        return

    for metric in ["cagr", "sharpe", "max_drawdown", "annualised_turnover"]:
        if metric not in df.columns:
            continue

        plot_df = df[["baseline_type", metric]].dropna()

        if plot_df.empty:
            continue

        plt.figure(figsize=(9, 5))
        plt.bar(plot_df["baseline_type"], plot_df[metric])
        plt.title(f"Baseline comparison: {metric}")
        plt.xlabel("Baseline")
        plt.ylabel(metric)
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / f"00_baseline_{safe_filename(metric)}.png", dpi=160)
        plt.close()


def save_all_charts(
    baseline_df: pd.DataFrame,
    one_way_df: pd.DataFrame,
    two_way_df: pd.DataFrame,
    random_df: pd.DataFrame,
    importance_df: pd.DataFrame,
) -> None:
    print("\nSaving charts.")

    save_baseline_charts(baseline_df)
    save_one_way_charts(one_way_df)
    save_heatmap_charts(two_way_df)
    save_random_cloud_charts(random_df, baseline_df)
    save_importance_charts(importance_df)


# ============================================================
# README REPORT
# ============================================================

def write_text_report(
    baseline_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
) -> None:
    lines = []

    lines.append("PARAMETER SENSITIVITY REPORT")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Interpretation:")
    lines.append("This is robustness evidence, not an optimisation pass.")
    lines.append("The correct result is not the highest random Sharpe.")
    lines.append("The correct result is a broad stable region around the frozen baseline.")
    lines.append("")

    if not baseline_df.empty:
        lines.append("BASELINES")
        lines.append("-" * 72)

        cols = [
            "baseline_type",
            "cagr",
            "annualised_volatility",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "annualised_turnover",
            "average_exposure",
            "average_cash",
            "final_equity",
        ]

        cols = [col for col in cols if col in baseline_df.columns]
        lines.append(baseline_df[cols].to_string(index=False))
        lines.append("")

    if not robustness_df.empty:
        lines.append("ROBUSTNESS SUMMARY")
        lines.append("-" * 72)
        lines.append(robustness_df.to_string(index=False))
        lines.append("")

    lines.append("Strong evidence:")
    lines.append("- Baseline is not an isolated top-1% spike.")
    lines.append("- Median random perturbation remains decent.")
    lines.append("- Heatmaps show broad stable regions, not cliffs.")
    lines.append("- Parameter importance is economically sensible.")
    lines.append("")
    lines.append("Weak evidence:")
    lines.append("- Baseline dominates almost all nearby trials.")
    lines.append("- Small parameter changes collapse Sharpe/CAGR.")
    lines.append("- One obscure parameter explains most performance.")
    lines.append("- Execution controls destroy returns.")

    path = OUTPUT_DIR / "README_parameter_sensitivity_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main() -> dict[str, pd.DataFrame]:
    prepare_dirs()

    print("\n" + "=" * 72)
    print("PARAMETER SENSITIVITY MAPS")
    print("=" * 72)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Charts directory: {CHARTS_DIR}")
    print(f"Random trials: {N_RANDOM_TRIALS:,}")
    print(f"Rebalance mode: {REBALANCE_MODE}")
    print(f"Total cost bps: {TOTAL_COST_BPS}")
    print("")

    scores, returns = load_inputs()

    baseline_df, production_raw_weights, benchmark_returns = run_baselines(
        scores=scores,
        returns=returns,
    )

    one_way_df = run_one_way_sensitivity(
        scores=scores,
        returns=returns,
        benchmark_returns=benchmark_returns,
    )

    two_way_df = run_two_way_heatmaps(
        scores=scores,
        returns=returns,
        benchmark_returns=benchmark_returns,
        production_raw_weights=production_raw_weights,
    )

    random_df = run_random_perturbation_cloud(
        scores=scores,
        returns=returns,
        benchmark_returns=benchmark_returns,
    )

    print(f"\nRandom perturbation rows loaded: {len(random_df):,}", flush=True)

    if len(random_df) < 100:
        print(
            "WARNING: random_df has fewer than 100 rows. "
            "Parameter importance will be weak and ridge beta may be unavailable.",
            flush=True,
        )

    importance_df = build_parameter_importance(random_df)

    robustness_df = build_robustness_summary(
        baseline_df=baseline_df,
        random_df=random_df,
    )

    save_all_charts(
        baseline_df=baseline_df,
        one_way_df=one_way_df,
        two_way_df=two_way_df,
        random_df=random_df,
        importance_df=importance_df,
    )

    write_text_report(
        baseline_df=baseline_df,
        robustness_df=robustness_df,
    )

    print("\n" + "=" * 72)
    print("PARAMETER SENSITIVITY COMPLETE")
    print("=" * 72)
    print(f"Saved CSVs to:   {OUTPUT_DIR}")
    print(f"Saved charts to: {CHARTS_DIR}")

    display_cols = [
        "baseline_type",
        "cagr",
        "annualised_volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "annualised_turnover",
        "average_exposure",
        "average_cash",
        "final_equity",
    ]

    display_cols = [col for col in display_cols if col in baseline_df.columns]

    print("\nBaseline metrics:")
    print(baseline_df[display_cols].to_string(index=False))

    print("\nRobustness summary:")
    print(robustness_df.to_string(index=False))

    return {
        "baseline": baseline_df,
        "one_way": one_way_df,
        "two_way": two_way_df,
        "random": random_df,
        "importance": importance_df,
        "robustness": robustness_df,
    }


if __name__ == "__main__":
    main()