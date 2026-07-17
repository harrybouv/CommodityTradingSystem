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
    USE_DYNAMIC_CASH_ALLOCATION,
    DYNAMIC_CASH_MAX_CUT,
    DYNAMIC_CASH_MIN_MULTIPLIER,
    DYNAMIC_CASH_STRESS_START,
    DYNAMIC_CASH_QUALITY_FLOOR,
    DYNAMIC_CASH_STRESS_WEIGHT,
    DYNAMIC_CASH_WEAK_QUALITY_WEIGHT,
    BEAR_SHORT_OVERLAY_ENABLED,
    BEAR_SHORT_SCORE_THRESHOLD,
    BEAR_SHORT_MAX_TOTAL_SHORT,
    BEAR_SHORT_MAX_SINGLE_SHORT,
    BEAR_SHORT_OVERRIDE_LONGS,
    BEAR_SHORT_SCALE_LONGS_TO_MAKE_ROOM,
    BEAR_SHORT_ALLOWED_REGIMES,
    BEAR_SHORT_REGIME_MIN_BEAR_VOTES,
    BEAR_SHORT_BASKET_MA_WINDOW,
    BEAR_SHORT_BASKET_RETURN_3M_THRESHOLD,
    BEAR_SHORT_BASKET_RETURN_6M_THRESHOLD,
    BEAR_SHORT_BASKET_RETURN_12M_THRESHOLD,
    BEAR_SHORT_BASKET_DRAWDOWN_THRESHOLD,
    BEAR_SHORT_BREADTH_ABOVE_MA_THRESHOLD,
    BEAR_SHORT_BREADTH_POS_3M_THRESHOLD,
    BEAR_SHORT_BREADTH_POS_6M_THRESHOLD,
)

from scoring.commodity_models.registry import build_commodity_model_scores
from scoring.commodity_models.Oil.USO_scoring import (
    USO_DIAGNOSTIC_COLUMNS,
)
from scoring.commodity_models.Gas.UNG_scoring import (
    UNG_DIAGNOSTIC_COLUMNS,
)
from scoring.commodity_models.Agriculture.DBA_scoring import (
    DBA_DIAGNOSTIC_COLUMNS,
)

try:
    from config import WEIGHTS_PATH
except ImportError:
    WEIGHTS_PATH = PROCESSED_DATA_DIR / "target_weights.csv"


MOMENTUM_SCORES_PATH = PROCESSED_DATA_DIR / "momentum_scores.csv"
RELATIVE_STRENGTH_SCORES_PATH = PROCESSED_DATA_DIR / "relative_strength_scores.csv"
TREND_SCORES_PATH = PROCESSED_DATA_DIR / "trend_scores.csv"
VOLATILITY_SCORES_PATH = PROCESSED_DATA_DIR / "volatility_scores.csv"
RISK_SCORES_PATH = PROCESSED_DATA_DIR / "risk_scores.csv"
FINAL_SCORES_PATH = PROCESSED_DATA_DIR / "final_scores.csv"
TREND_PERSISTENCE_SCORES_PATH = PROCESSED_DATA_DIR / "trend_persistence_scores.csv"
MACRO_SCORES_PATH = PROCESSED_DATA_DIR / "macro_scores.csv"

MIN_VOL_FOR_SIZING = 0.05

USE_ASSET_ALLOCATION_OVERLAY = False
USE_PORTFOLIO_EXPOSURE_OVERLAY = USE_DYNAMIC_CASH_ALLOCATION

MIN_ASSET_ALLOCATION_MULTIPLIER = 0.70
MIN_PORTFOLIO_EXPOSURE_MULTIPLIER = 0.80

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
    "macro_score",
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
GOLD_DIAGNOSTIC_COLUMNS = [
    "gld_base_score",
    "gold_overlay_score",
    "gold_final_score_pre_clip",
    "gold_overlay_enabled",
    "gold_feature_date",
    "gold_feature_age_days",
    "gold_core_data_quality_score",
    "gold_liquidity_squeeze_flag",
    "gold_real_yield_score",
    "gold_usd_score",
    "gold_stress_score",
    "gold_policy_rate_score",
    "gold_central_bank_score",
    "gold_positioning_score",
    "real_yield_10y",
    "real_yield_change_3m",
    "real_yield_z_3y",
    "usd_return_3m",
    "usd_z_3y",
    "vix_z_3y",
    "stlfsi_z_3y",
    "spy_drawdown_3m",
    "dgs2_change_3m",
    "managed_money_positioning_z_3y",
    "commodity_model",
    "commodity_model_score",
    "commodity_conviction_score",
    "commodity_data_quality_score",
    "commodity_model_version",
]

SILVER_DIAGNOSTIC_COLUMNS = [
    "slv_base_score",
    "silver_overlay_score",
    "silver_final_score_pre_clip",
    "silver_overlay_enabled",
    "silver_feature_date",
    "silver_feature_age_days",
    "silver_core_data_quality_score",

    "silver_gold_ratio_score",
    "silver_copper_ratio_score",
    "silver_gold_confirmation_score",
    "silver_usd_score",
    "silver_real_yield_score",
    "silver_macro_score",

    "gold_silver_ratio",
    "gold_silver_ratio_change_3m",
    "gold_silver_ratio_z_3y",
    "silver_vs_gold_return_1m",
    "silver_vs_gold_return_3m",
    "silver_vs_gold_return_6m",

    "silver_copper_ratio",
    "silver_copper_ratio_change_3m",
    "silver_copper_ratio_z_3y",
    "copper_momentum_score",
    "copper_trend_score",
    "silver_cheap_vs_copper_score",

    "gold_momentum_score",
    "gold_trend_score",

    "silver_trend_score",
    "silver_momentum_score",

    "usd_index",
    "usd_return_1m",
    "usd_return_3m",
    "usd_return_6m",
    "real_yield_10y",
    "real_yield_change_3m",
    "real_yield_z_3y",

    "commodity_model",
    "commodity_model_score",
    "commodity_conviction_score",
    "commodity_data_quality_score",
    "commodity_model_version",
]

COPPER_DIAGNOSTIC_COLUMNS = [
    "cper_base_score",
    "copper_overlay_score",
    "copper_final_score_pre_clip",
    "copper_overlay_enabled",
    "copper_feature_date",
    "copper_feature_age_days",
    "copper_core_data_quality_score",
    "copper_core_feature_count",

    "copper_china_electricity_score",
    "copper_china_cli_score",
    "copper_usd_score",
    "copper_broad_commodity_trend_score",
    "copper_oil_price_score",
    "copper_global_growth_score",

    "china_electricity_demand_twh",
    "china_electricity_yoy",
    "china_electricity_yoy_3m_avg",
    "china_electricity_yoy_change_3m",

    "china_cli",
    "china_cli_change_3m",
    "china_cli_z_3y",

    "usd_index",
    "usd_return_1m",
    "usd_return_3m",
    "usd_return_6m",
    "usd_z_3y",

    "dbc_return_1m",
    "dbc_return_3m",
    "dbc_return_6m",
    "dbc_trend_score",

    "uso_return_1m",
    "uso_return_3m",
    "uso_return_6m",
    "uso_trend_score",

    "spy_return_1m",
    "spy_return_3m",
    "spy_return_6m",
    "spy_trend_score",
    "vix_index",
    "vix_z_3y",

    "cper_return_1m",
    "cper_return_3m",
    "cper_return_6m",
    "cper_trend_score",
    "cper_momentum_score",

    "commodity_model",
    "commodity_model_score",
    "commodity_conviction_score",
    "commodity_data_quality_score",
    "commodity_model_version",
]

FINAL_SCORE_OUTPUT_COLUMNS = [
    "date",
    "ticker",
    "adj_close",
    "momentum_score",
    "relative_strength_score",
    "trend_score",
    "trend_persistence_score",
    "volatility_score",
    "risk_score",
    "macro_score",
    "final_score",
    "rank",
    "vol_stress_score",
    "risk_stress_score",
    "macro_group",
    "macro_regime",
    "usd_score",
    "rates_score",
    "inflation_score",
    "growth_score",
    "stress_score",
    "commodity_trend_score",
]

FINAL_SCORE_OUTPUT_COLUMNS += [
    col for col in GOLD_DIAGNOSTIC_COLUMNS
    if col not in FINAL_SCORE_OUTPUT_COLUMNS
]

FINAL_SCORE_OUTPUT_COLUMNS += [
    col for col in SILVER_DIAGNOSTIC_COLUMNS
    if col not in FINAL_SCORE_OUTPUT_COLUMNS
]

FINAL_SCORE_OUTPUT_COLUMNS += [
    col for col in COPPER_DIAGNOSTIC_COLUMNS
    if col not in FINAL_SCORE_OUTPUT_COLUMNS
]

FINAL_SCORE_OUTPUT_COLUMNS += [
    col for col in UNG_DIAGNOSTIC_COLUMNS
    if col not in FINAL_SCORE_OUTPUT_COLUMNS
]

FINAL_SCORE_OUTPUT_COLUMNS += [
    col for col in USO_DIAGNOSTIC_COLUMNS
    if col not in FINAL_SCORE_OUTPUT_COLUMNS
]

FINAL_SCORE_OUTPUT_COLUMNS += [
    col for col in DBA_DIAGNOSTIC_COLUMNS
    if col not in FINAL_SCORE_OUTPUT_COLUMNS
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
    "macro_score",
    "macro_group",
    "macro_regime",
    "usd_score",
    "rates_score",
    "inflation_score",
    "growth_score",
    "stress_score",
    "commodity_trend_score",
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

    "pre_short_target_weight",
    "commodity_regime",
    "bear_regime_flag",
    "bear_regime_votes",
    "bear_short_overlay_enabled",
    "bear_short_score_threshold",
    "bear_short_candidate",
    "bear_short_weight",
    "long_scale_after_short",
    "gross_exposure_after_short",
    "net_exposure_after_short",
    "short_exposure_after_short",

    "target_weight",
    "cash_weight",
]
TARGET_WEIGHT_OUTPUT_COLUMNS += [
    col for col in GOLD_DIAGNOSTIC_COLUMNS
    if col not in TARGET_WEIGHT_OUTPUT_COLUMNS
]

TARGET_WEIGHT_OUTPUT_COLUMNS += [
    col for col in SILVER_DIAGNOSTIC_COLUMNS
    if col not in TARGET_WEIGHT_OUTPUT_COLUMNS
]

TARGET_WEIGHT_OUTPUT_COLUMNS += [
    col for col in COPPER_DIAGNOSTIC_COLUMNS
    if col not in TARGET_WEIGHT_OUTPUT_COLUMNS
]

TARGET_WEIGHT_OUTPUT_COLUMNS += [
    col for col in USO_DIAGNOSTIC_COLUMNS
    if col not in TARGET_WEIGHT_OUTPUT_COLUMNS
]
TARGET_WEIGHT_OUTPUT_COLUMNS += [
    col for col in UNG_DIAGNOSTIC_COLUMNS
    if col not in TARGET_WEIGHT_OUTPUT_COLUMNS
]
TARGET_WEIGHT_OUTPUT_COLUMNS += [
    col for col in DBA_DIAGNOSTIC_COLUMNS
    if col not in TARGET_WEIGHT_OUTPUT_COLUMNS
]

MACRO_DIAGNOSTIC_COLUMNS = [
    "macro_group",
    "macro_regime",
    "usd_score",
    "rates_score",
    "inflation_score",
    "growth_score",
    "stress_score",
    "commodity_trend_score",
    "precious_metals_macro_score",
    "energy_macro_score",
    "industrial_metals_macro_score",
    "agriculture_macro_score",
    "broad_macro_score",
]


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


def _require_unique_date_ticker(df: pd.DataFrame, name: str) -> None:
    duplicate_count = df.duplicated(["date", "ticker"]).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"{name} has {duplicate_count} duplicate date/ticker rows."
        )


def load_score_inputs(
    momentum_path: Path = MOMENTUM_SCORES_PATH,
    relative_strength_path: Path = RELATIVE_STRENGTH_SCORES_PATH,
    trend_path: Path = TREND_SCORES_PATH,
    trend_persistence_path: Path = TREND_PERSISTENCE_SCORES_PATH,
    volatility_path: Path = VOLATILITY_SCORES_PATH,
    risk_path: Path = RISK_SCORES_PATH,
    macro_path: Path = MACRO_SCORES_PATH,
) -> pd.DataFrame:

    momentum = _to_datetime(pd.read_csv(momentum_path))
    relative_strength = _to_datetime(pd.read_csv(relative_strength_path))
    trend = _to_datetime(pd.read_csv(trend_path))
    trend_persistence = _to_datetime(pd.read_csv(trend_persistence_path))
    volatility = _to_datetime(pd.read_csv(volatility_path))
    risk = _to_datetime(pd.read_csv(risk_path))
    macro = _to_datetime(pd.read_csv(macro_path))

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

    _require_columns(
        macro,
        ["date", "ticker", "macro_score"],
        "macro_scores.csv",
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

    volatility = volatility[volatility_cols].copy()

    risk_cols = [
        "date",
        "ticker",
        "risk_score",
    ] + [
        col for col in RISK_DIAGNOSTIC_COLUMNS
        if col in risk.columns
    ]

    risk = risk[risk_cols].copy()

    macro_cols = [
        "date",
        "ticker",
        "macro_score",
    ] + [
        col for col in MACRO_DIAGNOSTIC_COLUMNS
        if col in macro.columns
    ]

    macro = macro[macro_cols].copy()

    macro_duplicate_count = macro.duplicated(["date", "ticker"]).sum()

    if macro_duplicate_count > 0:
        raise ValueError(
            f"macro_scores.csv has {macro_duplicate_count} duplicate date/ticker rows. "
            "This would structurally change the strategy merge."
        )

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

    scores_duplicate_count = scores.duplicated(["date", "ticker"]).sum()

    if scores_duplicate_count > 0:
        raise ValueError(
            f"merged base score table has {scores_duplicate_count} duplicate date/ticker rows "
            "before macro merge."
        )

    rows_before_macro = len(scores)

    scores = scores.merge(
        macro,
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )

    rows_after_macro = len(scores)

    if rows_after_macro != rows_before_macro:
        raise ValueError(
            f"Macro merge changed row count: {rows_before_macro} -> {rows_after_macro}"
        )

    scores["macro_score"] = (
        scores["macro_score"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.50)
        .clip(0, 1)
    )

    for col in MACRO_DIAGNOSTIC_COLUMNS:
        if col in scores.columns:
            if col in ["macro_group", "macro_regime"]:
                scores[col] = scores[col].fillna("unknown")
            else:
                scores[col] = (
                    scores[col]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.50)
                    .clip(0, 1)
                )

    scores = scores.sort_values(["date", "ticker"]).reset_index(drop=True)

    _require_columns(scores, REQUIRED_SCORE_COLUMNS, "merged score inputs")

    if scores.empty:
        raise ValueError("Merged strategy score input is empty.")

    return scores


def build_weighted_final_score(
    scores: pd.DataFrame,
    score_weights: Optional[dict[str, float]] = None,
    normalise_weights: bool = False,
) -> pd.Series:

    data = scores.copy()

    if score_weights is None:
        score_weights = SCORE_WEIGHTS

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

    commodity_scored = build_commodity_model_scores(scores)

    return attach_final_score_and_rank(
        scores=commodity_scored,
        final_score=commodity_scored["final_score"],
    )


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
    min_portfolio_multiplier: float = DYNAMIC_CASH_MIN_MULTIPLIER,
) -> pd.DataFrame:

    out = data.copy()

    if "combined_stress_score" not in out.columns:
        out = add_signal_quality_and_stress(out)

    out["pre_portfolio_overlay_weight"] = out["target_weight"]

    if not use_overlay:
        out["portfolio_exposure_multiplier"] = 1.0
        return out

    total_weight = (
        out.groupby("date")["target_weight"]
        .transform("sum")
    )

    out["_weighted_stress"] = (
        out["target_weight"] * out["combined_stress_score"]
    )

    out["_weighted_quality"] = (
        out["target_weight"] * out["signal_quality"]
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
        1.0,
    )

    portfolio_stress = pd.Series(
        portfolio_stress,
        index=out.index,
    ).clip(0.0, 1.0)

    portfolio_quality = pd.Series(
        portfolio_quality,
        index=out.index,
    ).clip(0.0, 1.0)

    stress_pressure = (
        (portfolio_stress - DYNAMIC_CASH_STRESS_START)
        / (1.0 - DYNAMIC_CASH_STRESS_START)
    ).clip(0.0, 1.0)

    weak_quality_pressure = (
        (DYNAMIC_CASH_QUALITY_FLOOR - portfolio_quality)
        / DYNAMIC_CASH_QUALITY_FLOOR
    ).clip(0.0, 1.0)

    cash_cut = DYNAMIC_CASH_MAX_CUT * (
        DYNAMIC_CASH_STRESS_WEIGHT * stress_pressure
        + DYNAMIC_CASH_WEAK_QUALITY_WEIGHT * weak_quality_pressure
    )

    portfolio_multiplier = (
        1.0 - cash_cut
    ).clip(lower=min_portfolio_multiplier, upper=1.0)

    out["portfolio_exposure_multiplier"] = portfolio_multiplier

    out["target_weight"] = (
        out["target_weight"] * out["portfolio_exposure_multiplier"]
    )

    out = out.drop(
        columns=["_weighted_stress", "_weighted_quality"],
        errors="ignore",
    )

    return out


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

def build_price_breadth_regime(
    data: pd.DataFrame,
    ma_window: int = BEAR_SHORT_BASKET_MA_WINDOW,
    min_bear_votes: int = BEAR_SHORT_REGIME_MIN_BEAR_VOTES,
) -> pd.DataFrame:

    _require_columns(
        data,
        ["date", "ticker", "adj_close"],
        "bear short regime input",
    )

    close = (
        data[["date", "ticker", "adj_close"]]
        .copy()
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .pivot(index="date", columns="ticker", values="adj_close")
        .sort_index()
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
    )

    returns = close.pct_change().replace([np.inf, -np.inf], np.nan)

    basket_return = returns.mean(axis=1).fillna(0.0)
    basket_index = (1.0 + basket_return).cumprod()

    basket_ma = basket_index.rolling(ma_window, min_periods=max(60, ma_window // 2)).mean()

    basket_return_3m = basket_index.pct_change(63)
    basket_return_6m = basket_index.pct_change(126)
    basket_return_12m = basket_index.pct_change(252)

    rolling_high_252 = basket_index.rolling(252, min_periods=126).max()
    basket_drawdown_252 = basket_index / rolling_high_252 - 1.0

    asset_ma = close.rolling(ma_window, min_periods=max(60, ma_window // 2)).mean()
    breadth_above_ma = (close > asset_ma).mean(axis=1)

    asset_return_3m = close.pct_change(63)
    asset_return_6m = close.pct_change(126)

    breadth_pos_3m = (asset_return_3m > 0.0).mean(axis=1)
    breadth_pos_6m = (asset_return_6m > 0.0).mean(axis=1)

    bear_votes = (
        (basket_index < basket_ma).astype(int)
        + (basket_return_3m < BEAR_SHORT_BASKET_RETURN_3M_THRESHOLD).astype(int)
        + (basket_return_6m < BEAR_SHORT_BASKET_RETURN_6M_THRESHOLD).astype(int)
        + (basket_return_12m < BEAR_SHORT_BASKET_RETURN_12M_THRESHOLD).astype(int)
        + (basket_drawdown_252 < BEAR_SHORT_BASKET_DRAWDOWN_THRESHOLD).astype(int)
        + (breadth_above_ma < BEAR_SHORT_BREADTH_ABOVE_MA_THRESHOLD).astype(int)
        + (breadth_pos_3m < BEAR_SHORT_BREADTH_POS_3M_THRESHOLD).astype(int)
        + (breadth_pos_6m < BEAR_SHORT_BREADTH_POS_6M_THRESHOLD).astype(int)
    )

    bull_votes = (
        (basket_index > basket_ma).astype(int)
        + (basket_return_3m > abs(BEAR_SHORT_BASKET_RETURN_3M_THRESHOLD)).astype(int)
        + (basket_return_6m > abs(BEAR_SHORT_BASKET_RETURN_6M_THRESHOLD)).astype(int)
        + (basket_return_12m > abs(BEAR_SHORT_BASKET_RETURN_12M_THRESHOLD)).astype(int)
        + (basket_drawdown_252 > -0.05).astype(int)
        + (breadth_above_ma > 0.60).astype(int)
        + (breadth_pos_3m > 0.60).astype(int)
        + (breadth_pos_6m > 0.60).astype(int)
    )

    regime = pd.Series("chop", index=close.index)
    regime.loc[bear_votes >= min_bear_votes] = "bear"
    regime.loc[(bear_votes < min_bear_votes) & (bull_votes >= 4)] = "bull"

    out = pd.DataFrame(
        {
            "date": close.index,
            "commodity_regime": regime.values,
            "bear_regime_flag": (regime.values == "bear"),
            "bear_regime_votes": bear_votes.values,
            "bull_regime_votes": bull_votes.values,
            "basket_return_3m": basket_return_3m.values,
            "basket_return_6m": basket_return_6m.values,
            "basket_return_12m": basket_return_12m.values,
            "basket_drawdown_252": basket_drawdown_252.values,
            "breadth_above_ma": breadth_above_ma.values,
            "breadth_pos_3m": breadth_pos_3m.values,
            "breadth_pos_6m": breadth_pos_6m.values,
        }
    )

    return out


def apply_bear_short_overlay(
    data: pd.DataFrame,
    use_overlay: bool = BEAR_SHORT_OVERLAY_ENABLED,
    score_threshold: float = BEAR_SHORT_SCORE_THRESHOLD,
    max_total_short: float = BEAR_SHORT_MAX_TOTAL_SHORT,
    max_single_short: float = BEAR_SHORT_MAX_SINGLE_SHORT,
    allowed_regimes: list[str] | tuple[str, ...] = BEAR_SHORT_ALLOWED_REGIMES,
    override_longs: bool = BEAR_SHORT_OVERRIDE_LONGS,
    scale_longs_to_make_room: bool = BEAR_SHORT_SCALE_LONGS_TO_MAKE_ROOM,
) -> pd.DataFrame:
    """
    Adds a capped bear-regime short overlay to existing long-only target weights.

    Important:
    - This does not create leverage. Gross exposure is capped at 100%.
    - Shorts only activate in allowed regimes, normally ["bear"].
    - If a ticker is shorted, its long weight is removed first.
    """

    out = data.copy()

    for col, default in {
        "pre_short_target_weight": np.nan,
        "commodity_regime": "unknown",
        "bear_regime_flag": False,
        "bear_regime_votes": 0,
        "bear_short_overlay_enabled": bool(use_overlay),
        "bear_short_score_threshold": float(score_threshold),
        "bear_short_candidate": False,
        "bear_short_weight": 0.0,
        "long_scale_after_short": 1.0,
        "gross_exposure_after_short": np.nan,
        "net_exposure_after_short": np.nan,
        "short_exposure_after_short": np.nan,
    }.items():
        out[col] = default

    out["pre_short_target_weight"] = out["target_weight"]

    regime = build_price_breadth_regime(out)
    out = out.merge(regime, on="date", how="left", suffixes=("", "_regime_calc"))

    # Keep canonical columns clean after merge.
    if "commodity_regime_regime_calc" in out.columns:
        out["commodity_regime"] = out["commodity_regime_regime_calc"].fillna("unknown")
    if "bear_regime_flag_regime_calc" in out.columns:
        out["bear_regime_flag"] = out["bear_regime_flag_regime_calc"].fillna(False).astype(bool)
    if "bear_regime_votes_regime_calc" in out.columns:
        out["bear_regime_votes"] = out["bear_regime_votes_regime_calc"].fillna(0).astype(int)

    out = out.drop(
        columns=[
            "commodity_regime_regime_calc",
            "bear_regime_flag_regime_calc",
            "bear_regime_votes_regime_calc",
        ],
        errors="ignore",
    )

    out["bear_short_overlay_enabled"] = bool(use_overlay)
    out["bear_short_score_threshold"] = float(score_threshold)

    if not use_overlay:
        return out

    allowed_regimes = set(allowed_regimes)

    out["bear_short_candidate"] = (
        out["commodity_regime"].isin(allowed_regimes)
        & (out["final_score"] < float(score_threshold))
    )

    for _, date_idx in out.groupby("date").groups.items():
        date_data = out.loc[date_idx]
        candidate_idx = date_data.index[date_data["bear_short_candidate"]]

        if len(candidate_idx) == 0:
            gross = float(out.loc[date_idx, "target_weight"].abs().sum())
            net = float(out.loc[date_idx, "target_weight"].sum())
            short = float(out.loc[date_idx, "target_weight"].clip(upper=0.0).abs().sum())

            out.loc[date_idx, "gross_exposure_after_short"] = gross
            out.loc[date_idx, "net_exposure_after_short"] = net
            out.loc[date_idx, "short_exposure_after_short"] = short
            continue

        if override_longs:
            out.loc[candidate_idx, "target_weight"] = 0.0

        n = len(candidate_idx)
        total_short = min(float(max_total_short), n * float(max_single_short))
        per_asset_short = min(float(max_single_short), total_short / n)

        out.loc[candidate_idx, "bear_short_weight"] = -per_asset_short

        long_weight_before_scale = out.loc[date_idx, "target_weight"].clip(lower=0.0)
        long_gross = float(long_weight_before_scale.sum())
        short_gross = float(out.loc[date_idx, "bear_short_weight"].abs().sum())

        if scale_longs_to_make_room:
            max_long_gross = max(0.0, 1.0 - short_gross)
            long_scale = min(1.0, max_long_gross / long_gross) if long_gross > 0 else 1.0
        else:
            long_scale = 1.0

        out.loc[date_idx, "target_weight"] = (
            out.loc[date_idx, "target_weight"].clip(lower=0.0) * long_scale
            + out.loc[date_idx, "bear_short_weight"]
        )

        gross = float(out.loc[date_idx, "target_weight"].abs().sum())

        if gross > 1.0:
            out.loc[date_idx, "target_weight"] = out.loc[date_idx, "target_weight"] / gross
            gross = 1.0

        net = float(out.loc[date_idx, "target_weight"].sum())
        short = float(out.loc[date_idx, "target_weight"].clip(upper=0.0).abs().sum())

        out.loc[date_idx, "long_scale_after_short"] = long_scale
        out.loc[date_idx, "gross_exposure_after_short"] = gross
        out.loc[date_idx, "net_exposure_after_short"] = net
        out.loc[date_idx, "short_exposure_after_short"] = short

    out["target_weight"] = (
        out["target_weight"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    return out

def calculate_cash_weight(data: pd.DataFrame) -> pd.DataFrame:

    out = data.copy()

    out["gross_exposure_after_short"] = (
        out.groupby("date")["target_weight"]
        .transform(lambda x: float(x.abs().sum()))
    )

    out["net_exposure_after_short"] = (
        out.groupby("date")["target_weight"]
        .transform("sum")
    )

    out["short_exposure_after_short"] = (
        out.groupby("date")["target_weight"]
        .transform(lambda x: float(x[x < 0].abs().sum()))
    )

    # Backward-compatible column name.
    out["total_weight"] = out["net_exposure_after_short"]

    out["cash_weight"] = 1.0 - out["gross_exposure_after_short"]
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

    out = apply_bear_short_overlay(out)
    out = calculate_cash_weight(out)
    out["target_weight"] = out["target_weight"].fillna(0.0)

    for col in TARGET_WEIGHT_OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    return (
        out[TARGET_WEIGHT_OUTPUT_COLUMNS]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


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

def build_production_strategy_score_history(
    scores: Optional[pd.DataFrame] = None,
    save: bool = False,
    output_path: Path = PROCESSED_DATA_DIR / "scores_history.csv",
) -> pd.DataFrame:
    """
    Historical score export for diagnostics.

    This does not change strategy behaviour. It rebuilds the exact production
    final scores and returns the long date/ticker score table used by V3
    diagnostics for feature IC, deciles, score buckets and decision audit.
    """

    if scores is None:
        scores = load_score_inputs()

    scored = build_production_final_scores(scores)

    for col in FINAL_SCORE_OUTPUT_COLUMNS:
        if col not in scored.columns:
            scored[col] = np.nan

    out = (
        scored[FINAL_SCORE_OUTPUT_COLUMNS]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )

    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
        print(f"Saved score history to: {output_path}")

    return out

def print_latest_allocation(weights: pd.DataFrame) -> None:
    latest_date = weights["date"].max()
    latest = weights[weights["date"] == latest_date].copy()

    latest = latest.sort_values("target_weight", ascending=False)

    display_cols = [
        "ticker",
        "group",
        "macro_group",
        "final_score",
        "macro_score",
        "macro_regime",
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