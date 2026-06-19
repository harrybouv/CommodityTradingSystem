# commodity_strategy.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    PROCESSED_DATA_DIR,
    SCORE_WEIGHTS,
    UNIVERSE,
    MAX_ASSET_WEIGHT,
    MIN_SCORE_TO_HOLD,
    MAX_GROUP_WEIGHT,
    MAX_TOTAL_RISK_ASSET_EXPOSURE,
)

try:
    from config import WEIGHTS_PATH
except ImportError:
    WEIGHTS_PATH = PROCESSED_DATA_DIR / "target_weights.csv"


# ============================================================
# PATHS
# ============================================================

MOMENTUM_SCORES_PATH = PROCESSED_DATA_DIR / "momentum_scores.csv"
RELATIVE_STRENGTH_SCORES_PATH = PROCESSED_DATA_DIR / "relative_strength_scores.csv"
TREND_SCORES_PATH = PROCESSED_DATA_DIR / "trend_scores.csv"
VOLATILITY_SCORES_PATH = PROCESSED_DATA_DIR / "volatility_scores.csv"
RISK_SCORES_PATH = PROCESSED_DATA_DIR / "risk_scores.csv"
FINAL_SCORES_PATH = PROCESSED_DATA_DIR / "final_scores.csv"
TREND_PERSISTENCE_SCORES_PATH = PROCESSED_DATA_DIR / "trend_persistence_scores.csv"

# ============================================================
# SETTINGS
# ============================================================

MIN_VOL_FOR_SIZING = 0.05

# CONTROL SWITCHES
# First run with these both False.
# Then test USE_ASSET_ALLOCATION_OVERLAY=True.
# Only later test portfolio exposure overlay.
USE_ASSET_ALLOCATION_OVERLAY = False
USE_PORTFOLIO_EXPOSURE_OVERLAY = False

MIN_ASSET_ALLOCATION_MULTIPLIER = 0.70
MIN_PORTFOLIO_EXPOSURE_MULTIPLIER = 0.80


# ============================================================
# REQUIRED / OUTPUT COLUMNS
# ============================================================

TREND_PERSISTENCE_DIAGNOSTIC_COLUMNS = [
    "return_20d",
    "return_60d",
    "return_120d",
    "structural_uptrend_flag",
    "distance_from_60d_high",
    "distance_from_120d_high",
    "distance_from_252d_high",
    "new_60d_high",
    "new_120d_high",
    "breakout_score",
    "positive_day_share_60d",
    "positive_5d_share_60d",
    "trend_efficiency_60d",
    "trend_efficiency_120d",
    "trend_consistency_score",
    "pullback_from_20d_high",
    "pullback_from_60d_high",
    "mild_pullback_quality",
    "pullback_uptrend_score",
]

REQUIRED_SCORE_COLUMNS = [
    "date",
    "ticker",
    "adj_close",
    "momentum_score",
    "relative_strength_score",
    "trend_score",
    "trend_persistence_score",
    "volatility_score",
    "risk_score",
    "realised_vol_60d",
]

VOLATILITY_DIAGNOSTIC_COLUMNS = [
    "realised_vol_20d",
    "realised_vol_120d",
    "vol_ratio_20_60",
    "vol_ratio_60_120",
    "vol_acceleration_20_60",
    "vol_acceleration_60_120",
    "vol_20d_percentile_252d",
    "vol_60d_percentile_252d",
    "vol_stress_score",
    "vol_allocation_score",
]

RISK_DIAGNOSTIC_COLUMNS = [
    "drawdown_60d",
    "drawdown_120d",
    "downside_vol_20d",
    "downside_vol_60d",
    "downside_vol_120d",
    "downside_vol_ratio_20_60",
    "drawdown_persistence_60d",
    "deep_drawdown_persistence_60d",
    "tail_return_5pct_60d",
    "tail_return_5pct_120d",
    "downside_pressure_20d",
    "downside_pressure_60d",
    "risk_stress_score",
    "risk_allocation_score",
]

FINAL_SCORE_OUTPUT_COLUMNS = [
    "date",
    "ticker",
    "adj_close",
    "momentum_score",
    "relative_strength_score",
    "trend_score",
    "volatility_score",
    "risk_score",
    "final_score",
    "rank",
    "trend_persistence_score",
    "vol_stress_score",
    "risk_stress_score",
]

TARGET_WEIGHT_OUTPUT_COLUMNS = [
    "date",
    "ticker",
    "group",
    "adj_close",
    "trend_persistence_score",
    "momentum_score",
    "relative_strength_score",
    "trend_score",
    "volatility_score",
    "risk_score",
    "realised_vol_60d",

    "final_score",
    "rank",

    "signal_quality",
    "vol_stress_score",
    "risk_stress_score",
    "combined_stress_score",

    "vol_for_sizing",
    "vol_adjusted_signal",

    "risk_allocation_multiplier",
    "vol_allocation_multiplier",
    "asset_allocation_multiplier",
    "portfolio_exposure_multiplier",

    "base_raw_weight",
    "raw_weight",
    "target_weight",
    "cash_weight",
]


# ============================================================
# UTILITIES
# ============================================================

def _require_columns(
    df: pd.DataFrame,
    required_cols: list[str],
    name: str,
) -> None:
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out


def _safe_column(
    df: pd.DataFrame,
    col: str,
    default: float = 0.0,
) -> pd.Series:
    if col in df.columns:
        return df[col].replace([np.inf, -np.inf], np.nan).fillna(default)

    return pd.Series(default, index=df.index)


# ============================================================
# LOAD SCORE INPUTS
# ============================================================

def load_score_inputs(
    momentum_path: Path = MOMENTUM_SCORES_PATH,
    relative_strength_path: Path = RELATIVE_STRENGTH_SCORES_PATH,
    trend_path: Path = TREND_SCORES_PATH,
    trend_persistence_path: Path = TREND_PERSISTENCE_SCORES_PATH,
    volatility_path: Path = VOLATILITY_SCORES_PATH,
    risk_path: Path = RISK_SCORES_PATH,
) -> pd.DataFrame:

    momentum = _to_datetime(pd.read_csv(momentum_path))
    relative_strength = _to_datetime(pd.read_csv(relative_strength_path))
    trend = _to_datetime(pd.read_csv(trend_path))
    trend_persistence = _to_datetime(pd.read_csv(trend_persistence_path))
    volatility = _to_datetime(pd.read_csv(volatility_path))
    risk = _to_datetime(pd.read_csv(risk_path))

    _require_columns(
        momentum,
        ["date", "ticker", "adj_close", "momentum_score"],
        "momentum_scores.csv",
    )

    _require_columns(
        relative_strength,
        ["date", "ticker", "relative_strength_score"],
        "relative_strength_scores.csv",
    )

    _require_columns(
        trend,
        ["date", "ticker", "trend_score"],
        "trend_scores.csv",
    )

    _require_columns(
        trend_persistence,
        ["date", "ticker", "trend_persistence_score"],
        "trend_persistence_scores.csv",
    )

    _require_columns(
        volatility,
        ["date", "ticker", "volatility_score", "realised_vol_60d"],
        "volatility_scores.csv",
    )

    _require_columns(
        risk,
        ["date", "ticker", "risk_score"],
        "risk_scores.csv",
    )

    _require_columns(
        volatility,
        ["date", "ticker", "vol_stress_score", "vol_allocation_score"],
        "volatility_scores.csv allocation diagnostics",
    )

    _require_columns(
        risk,
        ["date", "ticker", "risk_stress_score", "risk_allocation_score"],
        "risk_scores.csv allocation diagnostics",
    )

    momentum = momentum[
        ["date", "ticker", "adj_close", "momentum_score"]
    ].copy()

    relative_strength = relative_strength[
        ["date", "ticker", "relative_strength_score"]
    ].copy()

    trend = trend[
        ["date", "ticker", "trend_score"]
    ].copy()

    trend_persistence_cols = [
        "date",
        "ticker",
        "trend_persistence_score",
    ] + [
        col for col in TREND_PERSISTENCE_DIAGNOSTIC_COLUMNS
        if col in trend_persistence.columns
    ]

    trend_persistence = trend_persistence[
        trend_persistence_cols
    ].copy()

    volatility_cols = [
        "date",
        "ticker",
        "volatility_score",
        "realised_vol_60d",
    ] + [
        col for col in VOLATILITY_DIAGNOSTIC_COLUMNS
        if col in volatility.columns and col not in ["realised_vol_60d"]
    ]

    risk_cols = [
        "date",
        "ticker",
        "risk_score",
    ] + [
        col for col in RISK_DIAGNOSTIC_COLUMNS
        if col in risk.columns
    ]

    volatility = volatility[volatility_cols].copy()
    risk = risk[risk_cols].copy()

    scores = momentum.merge(
        relative_strength,
        on=["date", "ticker"],
        how="inner",
    )

    scores = scores.merge(
        trend,
        on=["date", "ticker"],
        how="inner",
    )

    # Important:
    # Left merge so a zero-weight experimental signal cannot change
    # the trading universe or the date range.
    scores = scores.merge(
        trend_persistence,
        on=["date", "ticker"],
        how="left",
    )

    scores["trend_persistence_score"] = (
        scores["trend_persistence_score"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.50)
        .clip(0, 1)
    )

    for col in TREND_PERSISTENCE_DIAGNOSTIC_COLUMNS:
        if col in scores.columns:
            scores[col] = (
                scores[col]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

    scores = scores.merge(
        volatility,
        on=["date", "ticker"],
        how="inner",
    )

    scores = scores.merge(
        risk,
        on=["date", "ticker"],
        how="inner",
    )

    scores = scores.sort_values(["date", "ticker"]).reset_index(drop=True)

    _require_columns(scores, REQUIRED_SCORE_COLUMNS, "merged score inputs")

    if scores.empty:
        raise ValueError("Merged strategy score input is empty.")

    return scores


# ============================================================
# FINAL SCORE
# ============================================================

def build_weighted_final_score(
    scores: pd.DataFrame,
    score_weights: Optional[dict[str, float]] = None,
    normalise_weights: bool = False,
) -> pd.Series:

    data = scores.copy()

    if score_weights is None:
        score_weights = SCORE_WEIGHTS

    # Ignore missing columns only when the requested weight is exactly zero.
    active_weights = {
        col: weight
        for col, weight in score_weights.items()
        if abs(weight) > 1e-12
    }

    required = list(active_weights.keys())
    _require_columns(data, required, "score data")

    total_weight = sum(active_weights.values())

    if total_weight == 0:
        raise ValueError("Active score weights sum to zero.")

    final_score = pd.Series(0.0, index=data.index)

    for col, weight in active_weights.items():
        if normalise_weights:
            final_score += (weight / total_weight) * data[col]
        else:
            final_score += weight * data[col]

    return final_score.clip(0, 1)


def build_final_score_from_spec(
    scores: pd.DataFrame,
    spec: dict,
) -> pd.Series:

    data = scores.copy()
    custom = spec.get("custom")

    if custom == "signal_x_risk_modifier":
        signal_score = (
            0.60 * data["momentum_score"]
            + 0.40 * data["trend_score"]
        )

        risk_modifier = (
            0.50 * data["risk_score"]
            + 0.50 * data["volatility_score"]
        )

        return (signal_score * risk_modifier).clip(0, 1)

    if custom == "momentum_x_risk":
        return (
            data["momentum_score"]
            * data["risk_score"]
        ).clip(0, 1)

    if custom == "momentum_x_volatility":
        return (
            data["momentum_score"]
            * data["volatility_score"]
        ).clip(0, 1)

    if "weights" not in spec:
        raise ValueError(
            f"Ablation spec must contain either 'custom' or 'weights': {spec}"
        )

    return build_weighted_final_score(
        scores=data,
        score_weights=spec["weights"],
        normalise_weights=spec.get("normalise", True),
    )


def attach_final_score_and_rank(
    scores: pd.DataFrame,
    final_score: pd.Series,
) -> pd.DataFrame:

    out = scores.copy()
    out["final_score"] = final_score.clip(0, 1)

    out["rank"] = (
        out.groupby("date")["final_score"]
        .rank(ascending=False, method="first")
    )

    out = out.sort_values(["date", "rank"]).reset_index(drop=True)

    return out


def build_production_final_scores(scores: pd.DataFrame) -> pd.DataFrame:

    final_score = build_weighted_final_score(
        scores=scores,
        score_weights=SCORE_WEIGHTS,
        normalise_weights=False,
    )

    return attach_final_score_and_rank(scores, final_score)


# ============================================================
# GROUPS
# ============================================================

def get_ticker_group_map() -> dict[str, str]:
    return {
        ticker: meta.get("group", "unknown")
        for ticker, meta in UNIVERSE.items()
    }


def add_group_column(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()

    group_map = get_ticker_group_map()
    out["group"] = out["ticker"].map(group_map).fillna("unknown")

    return out


# ============================================================
# ALLOCATION OVERLAYS
# ============================================================

def add_signal_quality_and_stress(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()

    out["signal_quality"] = (
        0.35 * _safe_column(out, "momentum_score", 0.0)
        + 0.25 * _safe_column(out, "relative_strength_score", 0.0)
        + 0.20 * _safe_column(out, "trend_score", 0.0)
        + 0.20 * _safe_column(out, "final_score", 0.0)
    ).clip(0, 1)

    out["vol_stress_score"] = _safe_column(
        out,
        "vol_stress_score",
        0.0,
    ).clip(0, 1)

    out["risk_stress_score"] = _safe_column(
        out,
        "risk_stress_score",
        0.0,
    ).clip(0, 1)

    out["combined_stress_score"] = (
        0.55 * out["risk_stress_score"]
        + 0.45 * out["vol_stress_score"]
    ).clip(0, 1)

    return out


def add_asset_allocation_multipliers(
    data: pd.DataFrame,
    use_overlay: bool = USE_ASSET_ALLOCATION_OVERLAY,
    min_asset_multiplier: float = MIN_ASSET_ALLOCATION_MULTIPLIER,
) -> pd.DataFrame:

    out = add_signal_quality_and_stress(data)

    if not use_overlay:
        out["risk_allocation_multiplier"] = 1.0
        out["vol_allocation_multiplier"] = 1.0
        out["asset_allocation_multiplier"] = 1.0
        return out

    signal_quality = out["signal_quality"]
    weak_signal = 1.0 - signal_quality

    risk_stress = out["risk_stress_score"]
    vol_stress = out["vol_stress_score"]

    # Conditional logic:
    # high stress + weak signal = larger cut
    # high stress + strong signal = mild cut only
    risk_penalty = risk_stress * (
        0.30 * weak_signal
        + 0.08 * signal_quality
    )

    vol_penalty = vol_stress * (
        0.25 * weak_signal
        + 0.06 * signal_quality
    )

    out["risk_allocation_multiplier"] = (
        1.0 - risk_penalty
    ).clip(lower=0.75, upper=1.0)

    out["vol_allocation_multiplier"] = (
        1.0 - vol_penalty
    ).clip(lower=0.80, upper=1.0)

    out["asset_allocation_multiplier"] = (
        out["risk_allocation_multiplier"]
        * out["vol_allocation_multiplier"]
    ).clip(lower=min_asset_multiplier, upper=1.0)

    return out


def apply_portfolio_exposure_overlay(
    data: pd.DataFrame,
    use_overlay: bool = USE_PORTFOLIO_EXPOSURE_OVERLAY,
    min_portfolio_multiplier: float = MIN_PORTFOLIO_EXPOSURE_MULTIPLIER,
) -> pd.DataFrame:

    out = data.copy()

    if "combined_stress_score" not in out.columns:
        out = add_signal_quality_and_stress(out)

    out["pre_portfolio_overlay_weight"] = out["target_weight"]

    if not use_overlay:
        out["portfolio_exposure_multiplier"] = 1.0
        return out

    out["_weighted_stress"] = (
        out["target_weight"] * out["combined_stress_score"]
    )

    out["_weighted_quality"] = (
        out["target_weight"] * out["signal_quality"]
    )

    total_weight = (
        out.groupby("date")["target_weight"]
        .transform("sum")
    )

    stress_sum = (
        out.groupby("date")["_weighted_stress"]
        .transform("sum")
    )

    quality_sum = (
        out.groupby("date")["_weighted_quality"]
        .transform("sum")
    )

    portfolio_stress = np.where(
        total_weight > 0,
        stress_sum / total_weight,
        0.0,
    )

    portfolio_quality = np.where(
        total_weight > 0,
        quality_sum / total_weight,
        0.0,
    )

    portfolio_stress = pd.Series(
        portfolio_stress,
        index=out.index,
    ).clip(0, 1)

    portfolio_quality = pd.Series(
        portfolio_quality,
        index=out.index,
    ).clip(0, 1)

    portfolio_penalty = portfolio_stress * (
        0.25 * (1.0 - portfolio_quality)
        + 0.08 * portfolio_quality
    )

    out["portfolio_exposure_multiplier"] = (
        1.0 - portfolio_penalty
    ).clip(lower=min_portfolio_multiplier, upper=1.0)

    out["target_weight"] = (
        out["target_weight"] * out["portfolio_exposure_multiplier"]
    )

    out = out.drop(
        columns=["_weighted_stress", "_weighted_quality"],
        errors="ignore",
    )

    return out


# ============================================================
# WEIGHT CONSTRUCTION
# ============================================================

def calculate_raw_signals(
    data: pd.DataFrame,
    min_score_to_hold: float = MIN_SCORE_TO_HOLD,
    min_vol_for_sizing: float = MIN_VOL_FOR_SIZING,
    use_allocation_overlay: bool = USE_ASSET_ALLOCATION_OVERLAY,
) -> pd.DataFrame:

    out = data.copy()

    _require_columns(
        out,
        ["final_score", "realised_vol_60d"],
        "strategy input",
    )

    out = add_asset_allocation_multipliers(
        out,
        use_overlay=use_allocation_overlay,
    )

    out["vol_for_sizing"] = (
        out["realised_vol_60d"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(min_vol_for_sizing)
        .clip(lower=min_vol_for_sizing)
    )

    out["vol_adjusted_signal"] = (
        out["final_score"] / out["vol_for_sizing"]
    )

    out["base_raw_weight"] = np.where(
        out["final_score"] >= min_score_to_hold,
        out["vol_adjusted_signal"],
        0.0,
    )

    out["raw_weight"] = (
        out["base_raw_weight"] * out["asset_allocation_multiplier"]
    )

    out["raw_weight"] = out["raw_weight"].fillna(0.0).clip(lower=0.0)

    return out


def normalise_weights(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()

    out["signal_sum"] = (
        out.groupby("date")["raw_weight"]
        .transform("sum")
    )

    out["target_weight"] = np.where(
        out["signal_sum"] > 0,
        out["raw_weight"] / out["signal_sum"],
        0.0,
    )

    out["target_weight"] = out["target_weight"].fillna(0.0)

    return out


def apply_asset_caps(
    data: pd.DataFrame,
    max_asset_weight: float = MAX_ASSET_WEIGHT,
) -> pd.DataFrame:

    out = data.copy()

    out["target_weight"] = out["target_weight"].clip(
        lower=0.0,
        upper=max_asset_weight,
    )

    return out


def apply_group_caps(
    data: pd.DataFrame,
    max_group_weight: Optional[dict[str, float]] = None,
) -> pd.DataFrame:

    out = data.copy()

    if max_group_weight is None:
        max_group_weight = MAX_GROUP_WEIGHT

    if "group" not in out.columns:
        out = add_group_column(out)

    for _, date_idx in out.groupby("date").groups.items():
        date_data = out.loc[date_idx]

        for group, group_cap in max_group_weight.items():
            group_mask = date_data["group"] == group
            group_indices = date_data[group_mask].index

            if len(group_indices) == 0:
                continue

            group_weight = out.loc[group_indices, "target_weight"].sum()

            if group_weight > group_cap and group_weight > 0:
                scale = group_cap / group_weight
                out.loc[group_indices, "target_weight"] *= scale

    return out


def apply_total_exposure_cap(
    data: pd.DataFrame,
    max_total_exposure: float = MAX_TOTAL_RISK_ASSET_EXPOSURE,
) -> pd.DataFrame:

    out = data.copy()

    total_weight = (
        out.groupby("date")["target_weight"]
        .transform("sum")
    )

    scale = np.where(
        total_weight > max_total_exposure,
        max_total_exposure / total_weight,
        1.0,
    )

    out["target_weight"] = out["target_weight"] * scale

    return out


def calculate_cash_weight(data: pd.DataFrame) -> pd.DataFrame:

    out = data.copy()

    out["total_weight"] = (
        out.groupby("date")["target_weight"]
        .transform("sum")
    )

    out["cash_weight"] = 1.0 - out["total_weight"]
    out["cash_weight"] = out["cash_weight"].clip(lower=0.0, upper=1.0)

    return out


def build_target_weights(
    scored_data: pd.DataFrame,
    min_score_to_hold: float = MIN_SCORE_TO_HOLD,
    max_asset_weight: float = MAX_ASSET_WEIGHT,
    max_group_weight: Optional[dict[str, float]] = None,
    max_total_exposure: float = MAX_TOTAL_RISK_ASSET_EXPOSURE,
    min_vol_for_sizing: float = MIN_VOL_FOR_SIZING,
    use_asset_allocation_overlay: bool = USE_ASSET_ALLOCATION_OVERLAY,
    use_portfolio_exposure_overlay: bool = USE_PORTFOLIO_EXPOSURE_OVERLAY,
) -> pd.DataFrame:

    out = scored_data.copy()

    _require_columns(
        out,
        [
            "date",
            "ticker",
            "adj_close",
            "momentum_score",
            "relative_strength_score",
            "trend_score",
            "volatility_score",
            "risk_score",
            "realised_vol_60d",
            "final_score",
            "rank",
        ],
        "scored strategy data",
    )

    out = add_group_column(out)

    out = calculate_raw_signals(
        out,
        min_score_to_hold=min_score_to_hold,
        min_vol_for_sizing=min_vol_for_sizing,
        use_allocation_overlay=use_asset_allocation_overlay,
    )

    out = normalise_weights(out)

    out = apply_asset_caps(
        out,
        max_asset_weight=max_asset_weight,
    )

    out = apply_group_caps(
        out,
        max_group_weight=max_group_weight,
    )

    out = apply_total_exposure_cap(
        out,
        max_total_exposure=max_total_exposure,
    )

    out = apply_portfolio_exposure_overlay(
        out,
        use_overlay=use_portfolio_exposure_overlay,
    )

    out = calculate_cash_weight(out)

    out["target_weight"] = out["target_weight"].fillna(0.0)

    # Guarantee output columns exist.
    for col in TARGET_WEIGHT_OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    return (
        out[TARGET_WEIGHT_OUTPUT_COLUMNS]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


# ============================================================
# PUBLIC STRATEGY FUNCTIONS
# ============================================================

def build_production_strategy_weights(
    scores: Optional[pd.DataFrame] = None,
    save_final_scores: bool = False,
    save_target_weights: bool = False,
    final_scores_path: Path = FINAL_SCORES_PATH,
    target_weights_path: Path = WEIGHTS_PATH,
) -> pd.DataFrame:

    if scores is None:
        scores = load_score_inputs()

    scored = build_production_final_scores(scores)

    weights = build_target_weights(scored)

    if save_final_scores:
        final_scores_path.parent.mkdir(parents=True, exist_ok=True)

        for col in FINAL_SCORE_OUTPUT_COLUMNS:
            if col not in scored.columns:
                scored[col] = np.nan

        scored[FINAL_SCORE_OUTPUT_COLUMNS].to_csv(
            final_scores_path,
            index=False,
        )

        print(f"Saved final scores to: {final_scores_path}")

    if save_target_weights:
        target_weights_path.parent.mkdir(parents=True, exist_ok=True)
        weights.to_csv(target_weights_path, index=False)
        print(f"Saved target weights to: {target_weights_path}")

    return weights


def build_strategy_weights_from_spec(
    scores: pd.DataFrame,
    spec: dict,
    min_score_to_hold: float = MIN_SCORE_TO_HOLD,
    max_asset_weight: float = MAX_ASSET_WEIGHT,
    max_group_weight: Optional[dict[str, float]] = None,
    max_total_exposure: float = MAX_TOTAL_RISK_ASSET_EXPOSURE,
) -> pd.DataFrame:

    final_score = build_final_score_from_spec(scores, spec)

    scored = attach_final_score_and_rank(
        scores=scores,
        final_score=final_score,
    )

    return build_target_weights(
        scored_data=scored,
        min_score_to_hold=min_score_to_hold,
        max_asset_weight=max_asset_weight,
        max_group_weight=max_group_weight,
        max_total_exposure=max_total_exposure,
    )


def weights_long_to_matrix(weights: pd.DataFrame) -> pd.DataFrame:

    _require_columns(
        weights,
        ["date", "ticker", "target_weight"],
        "weights",
    )

    out = weights.copy()
    out["date"] = pd.to_datetime(out["date"])

    matrix = (
        out.pivot(
            index="date",
            columns="ticker",
            values="target_weight",
        )
        .sort_index()
        .fillna(0.0)
    )

    return matrix


def build_production_strategy_weight_matrix(
    scores: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:

    weights = build_production_strategy_weights(
        scores=scores,
        save_final_scores=False,
        save_target_weights=False,
    )

    return weights_long_to_matrix(weights)


# ============================================================
# DISPLAY / MAIN
# ============================================================

def print_latest_allocation(weights: pd.DataFrame) -> None:
    latest_date = weights["date"].max()
    latest = weights[weights["date"] == latest_date].copy()

    latest = latest.sort_values("target_weight", ascending=False)

    display_cols = [
        "ticker",
        "group",
        "final_score",
        "realised_vol_60d",
        "signal_quality",
        "combined_stress_score",
        "asset_allocation_multiplier",
        "portfolio_exposure_multiplier",
        "target_weight",
        "cash_weight",
    ]

    display_cols = [
        col for col in display_cols
        if col in latest.columns
    ]

    print(f"\nLatest target allocation: {latest_date.date()}")
    print(latest[display_cols].to_string(index=False))

    total_weight = latest["target_weight"].sum()
    cash_weight = 1.0 - total_weight

    print("\nPortfolio exposure:")
    print(f"Risk asset exposure: {total_weight:.2%}")
    print(f"Cash weight:          {cash_weight:.2%}")

    group_weights = (
        latest.groupby("group")["target_weight"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nGroup weights:")
    print(group_weights.to_string())

    print("\nOverlay switches:")
    print(f"Asset allocation overlay:     {USE_ASSET_ALLOCATION_OVERLAY}")
    print(f"Portfolio exposure overlay:   {USE_PORTFOLIO_EXPOSURE_OVERLAY}")


def main() -> pd.DataFrame:
    weights = build_production_strategy_weights(
        save_final_scores=True,
        save_target_weights=True,
    )

    print_latest_allocation(weights)

    return weights


if __name__ == "__main__":
    main()