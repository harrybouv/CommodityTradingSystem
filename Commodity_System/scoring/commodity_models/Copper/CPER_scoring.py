# Commodity_System/scoring/commodity_models/CPER_scoring.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR
from scoring.commodity_models.base_scoring import (
    build_core_identity_commodity_score,
)


try:
    from config import (
        COPPER_OVERLAY_ENABLED,
        COPPER_OVERLAY_BLEND_WEIGHT,
        COPPER_OVERLAY_REQUIRE_FEATURES,
        COPPER_FEATURE_ASOF_TOLERANCE_DAYS,
        COPPER_USE_CHINA_ELECTRICITY,
        COPPER_USE_CHINA_CLI,
        COPPER_USE_USD,
        COPPER_USE_BROAD_COMMODITY_TREND,
        COPPER_USE_OIL_PRICE,
        COPPER_USE_GLOBAL_GROWTH,
        COPPER_COMPONENT_WEIGHTS,
    )
except ImportError:
    COPPER_OVERLAY_ENABLED = True
    COPPER_OVERLAY_BLEND_WEIGHT = 0.10
    COPPER_OVERLAY_REQUIRE_FEATURES = True
    COPPER_FEATURE_ASOF_TOLERANCE_DAYS = 35

    COPPER_USE_CHINA_ELECTRICITY = True
    COPPER_USE_CHINA_CLI = True
    COPPER_USE_USD = True
    COPPER_USE_BROAD_COMMODITY_TREND = True
    COPPER_USE_OIL_PRICE = True
    COPPER_USE_GLOBAL_GROWTH = True

    COPPER_COMPONENT_WEIGHTS = {
        "copper_china_electricity_score": 0.30,
        "copper_china_cli_score": 0.20,
        "copper_usd_score": 0.15,
        "copper_broad_commodity_trend_score": 0.15,
        "copper_oil_price_score": 0.10,
        "copper_global_growth_score": 0.10,
    }


# ============================================================
# CONSTANTS
# ============================================================

TICKER = "CPER"
MODEL_NAME = "copper"

COPPER_FEATURES_DAILY_PATH = (
    PROCESSED_DATA_DIR / "copper" / "copper_features_daily.csv"
)


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

    "commodity_model_version",
]


# ============================================================
# HELPERS
# ============================================================

def _clip01(s: pd.Series) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)


def _neutral_series(index: pd.Index, value: float = 0.50) -> pd.Series:
    return pd.Series(value, index=index, dtype="float64")


def _load_copper_features(path: Path = COPPER_FEATURES_DAILY_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Copper features not found: {path}. "
            "Run scoring/commodity_models/Copper/copper_features.py first."
        )

    features = pd.read_csv(path)

    if "date" not in features.columns:
        raise ValueError(f"Copper features file missing date column: {path}")

    features["date"] = pd.to_datetime(features["date"])
    features = (
        features
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    features = features.rename(columns={"date": "copper_feature_date"})

    return features


def _merge_copper_features_asof(
    cper: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    left = cper.copy()
    right = features.copy()

    left["date"] = pd.to_datetime(left["date"])
    right["copper_feature_date"] = pd.to_datetime(right["copper_feature_date"])

    left = left.sort_values("date").reset_index(drop=True)
    right = right.sort_values("copper_feature_date").reset_index(drop=True)

    out = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="copper_feature_date",
        direction="backward",
        tolerance=pd.Timedelta(days=COPPER_FEATURE_ASOF_TOLERANCE_DAYS),
    )

    out["copper_feature_age_days"] = (
        out["date"] - out["copper_feature_date"]
    ).dt.days

    return out


def _enabled_component_weights() -> dict[str, float]:
    toggles = {
        "copper_china_electricity_score": COPPER_USE_CHINA_ELECTRICITY,
        "copper_china_cli_score": COPPER_USE_CHINA_CLI,
        "copper_usd_score": COPPER_USE_USD,
        "copper_broad_commodity_trend_score": COPPER_USE_BROAD_COMMODITY_TREND,
        "copper_oil_price_score": COPPER_USE_OIL_PRICE,
        "copper_global_growth_score": COPPER_USE_GLOBAL_GROWTH,
    }

    weights = {
        col: float(COPPER_COMPONENT_WEIGHTS.get(col, 0.0))
        for col, enabled in toggles.items()
        if enabled and float(COPPER_COMPONENT_WEIGHTS.get(col, 0.0)) > 0.0
    }

    total = sum(weights.values())

    if total <= 0:
        return {}

    return {
        col: weight / total
        for col, weight in weights.items()
    }


def _build_overlay_score(data: pd.DataFrame) -> pd.Series:
    weights = _enabled_component_weights()

    if not weights:
        return _neutral_series(data.index)

    overlay = pd.Series(0.0, index=data.index, dtype="float64")

    for col, weight in weights.items():
        if col in data.columns:
            component = pd.to_numeric(data[col], errors="coerce")
        else:
            component = _neutral_series(data.index)

        component = _clip01(component.fillna(0.50))
        overlay += weight * component

    return _clip01(overlay.fillna(0.50))


def _attach_neutral_copper_diagnostics(
    out: pd.DataFrame,
    reason: str,
) -> pd.DataFrame:
    out = out.copy()

    out["cper_base_score"] = out["final_score"]
    out["copper_overlay_score"] = 0.50
    out["copper_final_score_pre_clip"] = out["final_score"]
    out["copper_overlay_enabled"] = False
    out["copper_feature_date"] = pd.NaT
    out["copper_feature_age_days"] = np.nan
    out["copper_core_data_quality_score"] = 0.0
    out["copper_core_feature_count"] = 0

    for col in [
        "copper_china_electricity_score",
        "copper_china_cli_score",
        "copper_usd_score",
        "copper_broad_commodity_trend_score",
        "copper_oil_price_score",
        "copper_global_growth_score",
    ]:
        out[col] = 0.50

    out["commodity_model"] = MODEL_NAME
    out["commodity_model_score"] = out["final_score"]
    out["commodity_conviction_score"] = out["copper_overlay_score"]
    out["commodity_data_quality_score"] = 0.0
    out["commodity_model_version"] = f"copper_overlay_neutral_{reason}"

    for col in COPPER_DIAGNOSTIC_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


# ============================================================
# MAIN SCORER
# ============================================================

def build_cper_scores(scores: pd.DataFrame) -> pd.DataFrame:
    base = build_core_identity_commodity_score(
        scores=scores,
        ticker=TICKER,
        model_name=MODEL_NAME,
    )

    if base.empty:
        return base

    base["cper_base_score"] = base["final_score"].copy()

    if not COPPER_OVERLAY_ENABLED or COPPER_OVERLAY_BLEND_WEIGHT <= 0:
        return _attach_neutral_copper_diagnostics(
            base,
            reason="disabled",
        )

    try:
        features = _load_copper_features()
    except FileNotFoundError:
        if COPPER_OVERLAY_REQUIRE_FEATURES:
            raise

        return _attach_neutral_copper_diagnostics(
            base,
            reason="missing_features",
        )

    out = _merge_copper_features_asof(
        cper=base,
        features=features,
    )

    missing_feature = out["copper_feature_date"].isna()

    if missing_feature.any() and COPPER_OVERLAY_REQUIRE_FEATURES:
        bad_dates = (
            out.loc[missing_feature, "date"]
            .head(5)
            .dt.date
            .tolist()
        )

        raise ValueError(
            "Copper overlay could not find as-of features for some CPER dates. "
            f"Examples: {bad_dates}"
        )

    overlay_score = _build_overlay_score(out)

    out["copper_overlay_score"] = overlay_score
    out["copper_overlay_enabled"] = True

    blend = float(COPPER_OVERLAY_BLEND_WEIGHT)

    if not 0.0 <= blend <= 1.0:
        raise ValueError(
            f"COPPER_OVERLAY_BLEND_WEIGHT must be in [0, 1], got {blend}"
        )

    out["copper_final_score_pre_clip"] = (
        (1.0 - blend) * out["cper_base_score"]
        + blend * out["copper_overlay_score"]
    )

    out["final_score"] = _clip01(out["copper_final_score_pre_clip"])

    out["commodity_model"] = MODEL_NAME
    out["commodity_model_score"] = out["final_score"]
    out["commodity_conviction_score"] = out["copper_overlay_score"]

    if "copper_core_data_quality_score" in out.columns:
        quality = pd.to_numeric(
            out["copper_core_data_quality_score"],
            errors="coerce",
        )
    else:
        quality = _neutral_series(out.index, 1.0)

    out["commodity_data_quality_score"] = (
        quality
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    out["commodity_model_version"] = (
        "copper_overlay_v1_china_cycle_usd_commodity_oil_growth"
    )

    for col in COPPER_DIAGNOSTIC_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    return out.sort_values(["date", "ticker"]).reset_index(drop=True)