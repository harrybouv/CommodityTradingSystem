from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# DIRECT-RUN PATH SETUP
# ============================================================

THIS_FILE = Path(__file__).resolve()
COMMODITY_ROOT = THIS_FILE.parents[3]

if str(COMMODITY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMODITY_ROOT))


from config import PROCESSED_DATA_DIR
from scoring.commodity_models.base_scoring import (
    build_core_identity_commodity_score,
)


try:
    from config import (
        UNG_OVERLAY_ENABLED,
        UNG_OVERLAY_BLEND_WEIGHT,
        UNG_OVERLAY_REQUIRE_FEATURES,
        UNG_FEATURE_ASOF_TOLERANCE_DAYS,

        UNG_USE_WEATHER_DEMAND,
        UNG_USE_STORAGE_TIGHTNESS,
        UNG_USE_STORAGE_MOMENTUM,
        UNG_USE_CURVE_ROLL,
        UNG_USE_SUPPLY_PRESSURE,
        UNG_USE_LNG_EXPORT_DEMAND,
        UNG_USE_OIL_RELATIVE_VALUE,
        UNG_USE_ENERGY_CONFIRMATION,

        UNG_COMPONENT_WEIGHTS,
    )
except ImportError:
    # Safe defaults. Production/testing values should live in config.py.
    UNG_OVERLAY_ENABLED = True
    UNG_OVERLAY_BLEND_WEIGHT = 0.10
    UNG_OVERLAY_REQUIRE_FEATURES = True
    UNG_FEATURE_ASOF_TOLERANCE_DAYS = 10

    UNG_USE_WEATHER_DEMAND = True
    UNG_USE_STORAGE_TIGHTNESS = True
    UNG_USE_STORAGE_MOMENTUM = True
    UNG_USE_CURVE_ROLL = True
    UNG_USE_SUPPLY_PRESSURE = True
    UNG_USE_LNG_EXPORT_DEMAND = False
    UNG_USE_OIL_RELATIVE_VALUE = True
    UNG_USE_ENERGY_CONFIRMATION = False

    UNG_COMPONENT_WEIGHTS = {
        "gas_weather_demand_score": 0.25,
        "gas_storage_tightness_score": 0.25,
        "gas_storage_momentum_score": 0.20,
        "gas_curve_roll_score": 0.15,
        "gas_supply_pressure_score": 0.10,
        "gas_lng_export_demand_score": 0.00,
        "gas_oil_relative_value_score": 0.05,
        "gas_energy_confirmation_score": 0.00,
    }


# ============================================================
# CONSTANTS
# ============================================================

TICKER = "UNG"
MODEL_NAME = "natural_gas"

GAS_FEATURES_DAILY_PATH = (
    PROCESSED_DATA_DIR / "gas" / "gas_features_daily.csv"
)


UNG_DIAGNOSTIC_COLUMNS = [
    "ung_base_score",
    "gas_overlay_score",
    "gas_final_score_pre_clip",
    "gas_overlay_enabled",
    "gas_feature_date",
    "gas_feature_age_days",
    "gas_core_data_quality_score",
    "gas_core_feature_count",

    # Core component scores
    "gas_weather_demand_score",
    "gas_storage_tightness_score",
    "gas_storage_momentum_score",
    "gas_curve_roll_score",
    "gas_supply_pressure_score",
    "gas_lng_export_demand_score",
    "gas_oil_relative_value_score",
    "gas_energy_confirmation_score",
    "gas_balance_score",
    "gas_expanded_balance_score",

    # Raw inputs
    "gas_storage_bcf",
    "gas_us_dry_production",
    "gas_lng_exports",
    "henry_hub_spot_price",
    "wti_spot_price",
    "hdd_utility_gas",
    "cdd_population",
    "unl_price",
    "dbc_price",

    # UNG / Henry Hub diagnostics
    "ung_return_1m",
    "ung_return_3m",
    "ung_return_6m",
    "ung_trend_score",
    "ung_momentum_score",
    "henry_hub_return_1m",
    "henry_hub_return_3m",
    "henry_hub_return_6m",
    "henry_hub_trend_score",
    "henry_hub_z_3y",

    # Weather
    "hdd_7d",
    "hdd_14d",
    "hdd_30d",
    "cdd_7d",
    "cdd_14d",
    "cdd_30d",
    "weather_demand_14d",
    "weather_demand_30d",
    "weather_demand_change_4w",
    "seasonal_weather_demand_14d",
    "seasonal_weather_demand_z_3y",
    "seasonal_weather_demand_change_4w",

    # Storage
    "gas_storage_seasonal_avg_5y",
    "gas_storage_vs_5y_seasonal_avg_bcf",
    "gas_storage_vs_5y_seasonal_avg_pct",
    "gas_storage_seasonal_z_5y",
    "gas_storage_seasonal_percentile_5y",
    "gas_storage_z_3y",
    "gas_storage_percentile_3y",
    "gas_storage_change_1w",
    "gas_storage_change_4w",
    "gas_storage_change_13w",
    "gas_storage_yoy_change",

    # Curve / roll
    "ung_unl_ratio",
    "ung_unl_ratio_return_1m",
    "ung_unl_ratio_return_3m",
    "ung_unl_ratio_return_6m",
    "ung_unl_ratio_z_3y",
    "ung_unl_ratio_trend_score",

    # Production / supply
    "gas_production_change_1m",
    "gas_production_change_3m",
    "gas_production_yoy_change",
    "gas_production_z_3y",

    # LNG
    "gas_lng_exports_change_1m",
    "gas_lng_exports_change_3m",
    "gas_lng_exports_yoy_change",
    "gas_lng_exports_z_3y",

    # Oil / energy confirmation
    "wti_per_mmbtu",
    "gas_oil_ratio",
    "gas_oil_ratio_z_3y",
    "gas_oil_ratio_return_3m",
    "uso_return_1m",
    "uso_return_3m",
    "uso_return_6m",
    "uso_trend_score",
    "dbc_return_1m",
    "dbc_return_3m",
    "dbc_return_6m",
    "dbc_trend_score",

    "commodity_model_version",
]


# ============================================================
# HELPERS
# ============================================================

def _clip01(s: pd.Series) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)


def _neutral_series(index: pd.Index, value: float = 0.50) -> pd.Series:
    return pd.Series(value, index=index, dtype="float64")


def _load_gas_features(path: Path = GAS_FEATURES_DAILY_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Gas features not found: {path}. "
            "Run scoring/commodity_models/NaturalGas/gas_features.py first."
        )

    features = pd.read_csv(path)

    if "date" not in features.columns:
        raise ValueError(f"Gas features file missing date column: {path}")

    features["date"] = pd.to_datetime(features["date"])
    features = (
        features
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    features = features.rename(columns={"date": "gas_feature_date"})

    return features


def _merge_gas_features_asof(
    ung: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    left = ung.copy()
    right = features.copy()

    left["date"] = pd.to_datetime(left["date"])
    right["gas_feature_date"] = pd.to_datetime(right["gas_feature_date"])

    left = left.sort_values("date").reset_index(drop=True)
    right = right.sort_values("gas_feature_date").reset_index(drop=True)

    out = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="gas_feature_date",
        direction="backward",
        tolerance=pd.Timedelta(days=UNG_FEATURE_ASOF_TOLERANCE_DAYS),
    )

    out["gas_feature_age_days"] = (
        out["date"] - out["gas_feature_date"]
    ).dt.days

    return out


def _enabled_component_weights() -> dict[str, float]:
    toggles = {
        "gas_weather_demand_score": UNG_USE_WEATHER_DEMAND,
        "gas_storage_tightness_score": UNG_USE_STORAGE_TIGHTNESS,
        "gas_storage_momentum_score": UNG_USE_STORAGE_MOMENTUM,
        "gas_curve_roll_score": UNG_USE_CURVE_ROLL,
        "gas_supply_pressure_score": UNG_USE_SUPPLY_PRESSURE,
        "gas_lng_export_demand_score": UNG_USE_LNG_EXPORT_DEMAND,
        "gas_oil_relative_value_score": UNG_USE_OIL_RELATIVE_VALUE,
        "gas_energy_confirmation_score": UNG_USE_ENERGY_CONFIRMATION,
    }

    weights = {
        col: float(UNG_COMPONENT_WEIGHTS.get(col, 0.0))
        for col, enabled in toggles.items()
        if enabled and float(UNG_COMPONENT_WEIGHTS.get(col, 0.0)) > 0.0
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


def _attach_neutral_gas_diagnostics(
    out: pd.DataFrame,
    reason: str,
) -> pd.DataFrame:
    out = out.copy()

    out["ung_base_score"] = out["final_score"]
    out["gas_overlay_score"] = 0.50
    out["gas_final_score_pre_clip"] = out["final_score"]
    out["gas_overlay_enabled"] = False
    out["gas_feature_date"] = pd.NaT
    out["gas_feature_age_days"] = np.nan
    out["gas_core_data_quality_score"] = 0.0
    out["gas_core_feature_count"] = 0

    for col in [
        "gas_weather_demand_score",
        "gas_storage_tightness_score",
        "gas_storage_momentum_score",
        "gas_curve_roll_score",
        "gas_supply_pressure_score",
        "gas_lng_export_demand_score",
        "gas_oil_relative_value_score",
        "gas_energy_confirmation_score",
        "gas_balance_score",
        "gas_expanded_balance_score",
    ]:
        out[col] = 0.50

    out["commodity_model"] = MODEL_NAME
    out["commodity_model_score"] = out["final_score"]
    out["commodity_conviction_score"] = out["gas_overlay_score"]
    out["commodity_data_quality_score"] = 0.0
    out["commodity_model_version"] = f"gas_overlay_neutral_{reason}"

    for col in UNG_DIAGNOSTIC_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


# ============================================================
# MAIN SCORER
# ============================================================

def build_ung_scores(scores: pd.DataFrame) -> pd.DataFrame:
    base = build_core_identity_commodity_score(
        scores=scores,
        ticker=TICKER,
        model_name=MODEL_NAME,
    )

    if base.empty:
        return base

    base["ung_base_score"] = base["final_score"].copy()

    if not UNG_OVERLAY_ENABLED or UNG_OVERLAY_BLEND_WEIGHT <= 0:
        return _attach_neutral_gas_diagnostics(
            base,
            reason="disabled",
        )

    try:
        features = _load_gas_features()
    except FileNotFoundError:
        if UNG_OVERLAY_REQUIRE_FEATURES:
            raise

        return _attach_neutral_gas_diagnostics(
            base,
            reason="missing_features",
        )

    out = _merge_gas_features_asof(
        ung=base,
        features=features,
    )

    missing_feature = out["gas_feature_date"].isna()

    if missing_feature.any() and UNG_OVERLAY_REQUIRE_FEATURES:
        bad_dates = (
            out.loc[missing_feature, "date"]
            .head(5)
            .dt.date
            .tolist()
        )

        raise ValueError(
            "Gas overlay could not find as-of features for some UNG dates. "
            f"Examples: {bad_dates}"
        )

    overlay_score = _build_overlay_score(out)

    out["gas_overlay_score"] = overlay_score
    out["gas_overlay_enabled"] = True

    blend = float(UNG_OVERLAY_BLEND_WEIGHT)

    if not 0.0 <= blend <= 1.0:
        raise ValueError(
            f"UNG_OVERLAY_BLEND_WEIGHT must be in [0, 1], got {blend}"
        )

    out["gas_final_score_pre_clip"] = (
        (1.0 - blend) * out["ung_base_score"]
        + blend * out["gas_overlay_score"]
    )

    out["final_score"] = _clip01(out["gas_final_score_pre_clip"])

    out["commodity_model"] = MODEL_NAME
    out["commodity_model_score"] = out["final_score"]
    out["commodity_conviction_score"] = out["gas_overlay_score"]

    if "gas_core_data_quality_score" in out.columns:
        quality = pd.to_numeric(
            out["gas_core_data_quality_score"],
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
        "gas_overlay_v1_weather_storage_curve_supply_oil"
    )

    for col in UNG_DIAGNOSTIC_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    return (
        out
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    print("UNG scoring module loaded successfully.")
    print(f"Gas features path: {GAS_FEATURES_DAILY_PATH}")
