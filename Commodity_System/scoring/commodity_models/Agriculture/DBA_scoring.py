# Commodity_System/scoring/commodity_models/DBA_scoring.py

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# DIRECT-RUN PATH SETUP
# ============================================================

THIS_FILE = Path(__file__).resolve()
COMMODITY_ROOT = THIS_FILE.parents[2]

if str(COMMODITY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMODITY_ROOT))


from config import PROCESSED_DATA_DIR
from scoring.commodity_models.base_scoring import (
    build_core_identity_commodity_score,
)


try:
    from config import (
        DBA_OVERLAY_ENABLED,
        DBA_OVERLAY_BLEND_WEIGHT,
        DBA_OVERLAY_REQUIRE_FEATURES,
        DBA_FEATURE_ASOF_TOLERANCE_DAYS,

        DBA_USE_USD,
        DBA_USE_RATES,
        DBA_USE_CROP_MOMENTUM,
        DBA_USE_CROP_RELATIVE_STRENGTH,
        DBA_USE_BROAD_COMMODITY_CONFIRMATION,
        DBA_USE_SEASONALITY,
        DBA_USE_EXPORT_DEMAND,

        DBA_COMPONENT_WEIGHTS,
    )
except ImportError:
    # Safe defaults. Production/testing values should live in config.py.
    DBA_OVERLAY_ENABLED = True
    DBA_OVERLAY_BLEND_WEIGHT = 0.10
    DBA_OVERLAY_REQUIRE_FEATURES = True
    DBA_FEATURE_ASOF_TOLERANCE_DAYS = 10

    DBA_USE_USD = True
    DBA_USE_RATES = True
    DBA_USE_CROP_MOMENTUM = True
    DBA_USE_CROP_RELATIVE_STRENGTH = True
    DBA_USE_BROAD_COMMODITY_CONFIRMATION = True
    DBA_USE_SEASONALITY = True

    # Keep exports off by default. ESR exports are useful, but noisy and
    # reset-prone, so they should be ablated before being trusted.
    DBA_USE_EXPORT_DEMAND = False

    DBA_COMPONENT_WEIGHTS = {
        "agri_usd_score": 0.20,
        "agri_rates_score": 0.07,
        "agri_crop_momentum_score": 0.25,
        "agri_crop_relative_strength_score": 0.25,
        "agri_broad_commodity_confirmation": 0.10,
        "agri_seasonality_score": 0.13,
        "agri_export_demand_score": 0.00,
    }


# ============================================================
# CONSTANTS
# ============================================================

TICKER = "DBA"
MODEL_NAME = "agriculture"

AGRI_FEATURES_DAILY_PATH = (
    PROCESSED_DATA_DIR / "agriculture" / "agriculture_features_daily.csv"
)


DBA_DIAGNOSTIC_COLUMNS = [
    "dba_base_score",
    "agri_overlay_score",
    "agri_final_score_pre_clip",
    "agri_overlay_enabled",
    "agri_feature_date",
    "agri_feature_age_days",
    "agri_core_data_quality_score",
    "agri_expanded_data_quality_score",
    "agri_core_feature_count",
    "agri_optional_feature_count",

    # Production component scores
    "agri_usd_score",
    "agri_rates_score",
    "agri_crop_momentum_score",
    "agri_crop_relative_strength_score",
    "agri_broad_commodity_confirmation",
    "agri_seasonality_score",
    "agri_export_demand_score",
    "agri_core_balance_score",
    "agri_expanded_balance_score",

    # DBA / broad commodity diagnostics
    "dba_price",
    "dba_return_1m",
    "dba_return_3m",
    "dba_return_6m",
    "dba_trend_score",
    "dba_momentum_score",
    "dbc_price",
    "dbc_return_1m",
    "dbc_return_3m",
    "dbc_return_6m",
    "dbc_trend_score",

    # USD / rates diagnostics
    "usd_index",
    "usd_return_1m",
    "usd_return_3m",
    "usd_return_6m",
    "usd_z_3y",
    "dgs10",
    "dgs2",
    "yield_curve_10y2y",
    "dgs10_change_1m",
    "dgs10_change_3m",
    "dgs2_change_1m",
    "dgs2_change_3m",
    "dgs10_z_3y",
    "dgs2_z_3y",
    "yield_curve_10y2y_change_3m",

    # Crop basket diagnostics
    "primary_crop_basket_index",
    "soft_crop_basket_index",
    "agri_crop_basket_index",
    "agri_crop_basket_return_1m",
    "agri_crop_basket_return_3m",
    "agri_crop_basket_return_6m",
    "agri_crop_basket_trend_score",

    # Individual crop diagnostics
    "corn_price",
    "corn_return_1m",
    "corn_return_3m",
    "corn_return_6m",
    "corn_trend_score",
    "corn_momentum_score",
    "wheat_price",
    "wheat_return_1m",
    "wheat_return_3m",
    "wheat_return_6m",
    "wheat_trend_score",
    "wheat_momentum_score",
    "soybeans_price",
    "soybeans_return_1m",
    "soybeans_return_3m",
    "soybeans_return_6m",
    "soybeans_trend_score",
    "soybeans_momentum_score",
    "sugar_price",
    "sugar_return_1m",
    "sugar_return_3m",
    "sugar_return_6m",
    "sugar_trend_score",
    "sugar_momentum_score",
    "coffee_price",
    "coffee_return_1m",
    "coffee_return_3m",
    "coffee_return_6m",
    "coffee_trend_score",
    "coffee_momentum_score",
    "cocoa_price",
    "cocoa_return_1m",
    "cocoa_return_3m",
    "cocoa_return_6m",
    "cocoa_trend_score",
    "cocoa_momentum_score",

    # Relative strength diagnostics
    "agri_crop_vs_dbc_ratio",
    "dba_vs_dbc_ratio",
    "agri_crop_vs_dba_ratio",
    "agri_crop_vs_dbc_return_3m",
    "dba_vs_dbc_return_3m",
    "agri_crop_vs_dba_return_3m",
    "agri_crop_vs_dbc_score",
    "dba_vs_dbc_score",
    "agri_crop_vs_dba_score",

    # Seasonality diagnostics
    "agri_seasonal_expected_return_1m",
    "agri_seasonal_expected_return_3m",

    # ESR export-demand diagnostics
    "corn_weekly_exports",
    "corn_current_my_net_sales",
    "corn_gross_new_sales",
    "corn_current_my_total_commitment",
    "corn_outstanding_sales",
    "corn_accumulated_exports",
    "corn_weekly_exports_4w_avg",
    "corn_net_sales_4w_avg",
    "corn_gross_sales_4w_avg",
    "corn_commitment_seasonal_percentile_5y",
    "corn_outstanding_sales_seasonal_percentile_5y",
    "corn_export_demand_score",

    "wheat_weekly_exports",
    "wheat_current_my_net_sales",
    "wheat_gross_new_sales",
    "wheat_current_my_total_commitment",
    "wheat_outstanding_sales",
    "wheat_accumulated_exports",
    "wheat_weekly_exports_4w_avg",
    "wheat_net_sales_4w_avg",
    "wheat_gross_sales_4w_avg",
    "wheat_commitment_seasonal_percentile_5y",
    "wheat_outstanding_sales_seasonal_percentile_5y",
    "wheat_export_demand_score",

    "soybeans_weekly_exports",
    "soybeans_current_my_net_sales",
    "soybeans_gross_new_sales",
    "soybeans_current_my_total_commitment",
    "soybeans_outstanding_sales",
    "soybeans_accumulated_exports",
    "soybeans_weekly_exports_4w_avg",
    "soybeans_net_sales_4w_avg",
    "soybeans_gross_sales_4w_avg",
    "soybeans_commitment_seasonal_percentile_5y",
    "soybeans_outstanding_sales_seasonal_percentile_5y",
    "soybeans_export_demand_score",

    "commodity_model_version",
]


# ============================================================
# HELPERS
# ============================================================

def _clip01(s: pd.Series) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)


def _neutral_series(index: pd.Index, value: float = 0.50) -> pd.Series:
    return pd.Series(value, index=index, dtype="float64")


def _load_agriculture_features(path: Path = AGRI_FEATURES_DAILY_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Agriculture features not found: {path}. "
            "Run scoring/commodity_models/Agriculture/agriculture_features.py first."
        )

    features = pd.read_csv(path)

    if "date" not in features.columns:
        raise ValueError(f"Agriculture features file missing date column: {path}")

    features["date"] = pd.to_datetime(features["date"])
    features = (
        features
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    features = features.rename(columns={"date": "agri_feature_date"})

    return features


def _merge_agriculture_features_asof(
    dba: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    left = dba.copy()
    right = features.copy()

    left["date"] = pd.to_datetime(left["date"])
    right["agri_feature_date"] = pd.to_datetime(right["agri_feature_date"])

    left = left.sort_values("date").reset_index(drop=True)
    right = right.sort_values("agri_feature_date").reset_index(drop=True)

    out = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="agri_feature_date",
        direction="backward",
        tolerance=pd.Timedelta(days=DBA_FEATURE_ASOF_TOLERANCE_DAYS),
    )

    out["agri_feature_age_days"] = (
        out["date"] - out["agri_feature_date"]
    ).dt.days

    return out


def _enabled_component_weights() -> dict[str, float]:
    toggles = {
        "agri_usd_score": DBA_USE_USD,
        "agri_rates_score": DBA_USE_RATES,
        "agri_crop_momentum_score": DBA_USE_CROP_MOMENTUM,
        "agri_crop_relative_strength_score": DBA_USE_CROP_RELATIVE_STRENGTH,
        "agri_broad_commodity_confirmation": DBA_USE_BROAD_COMMODITY_CONFIRMATION,
        "agri_seasonality_score": DBA_USE_SEASONALITY,
        "agri_export_demand_score": DBA_USE_EXPORT_DEMAND,
    }

    weights = {
        col: float(DBA_COMPONENT_WEIGHTS.get(col, 0.0))
        for col, enabled in toggles.items()
        if enabled and float(DBA_COMPONENT_WEIGHTS.get(col, 0.0)) > 0.0
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


def _attach_neutral_agriculture_diagnostics(
    out: pd.DataFrame,
    reason: str,
) -> pd.DataFrame:
    out = out.copy()

    out["dba_base_score"] = out["final_score"]
    out["agri_overlay_score"] = 0.50
    out["agri_final_score_pre_clip"] = out["final_score"]
    out["agri_overlay_enabled"] = False
    out["agri_feature_date"] = pd.NaT
    out["agri_feature_age_days"] = np.nan
    out["agri_core_data_quality_score"] = 0.0
    out["agri_expanded_data_quality_score"] = 0.0
    out["agri_core_feature_count"] = 0
    out["agri_optional_feature_count"] = 0

    for col in [
        "agri_usd_score",
        "agri_rates_score",
        "agri_crop_momentum_score",
        "agri_crop_relative_strength_score",
        "agri_broad_commodity_confirmation",
        "agri_seasonality_score",
        "agri_export_demand_score",
        "agri_core_balance_score",
        "agri_expanded_balance_score",
    ]:
        out[col] = 0.50

    out["commodity_model"] = MODEL_NAME
    out["commodity_model_score"] = out["final_score"]
    out["commodity_conviction_score"] = out["agri_overlay_score"]
    out["commodity_data_quality_score"] = 0.0
    out["commodity_model_version"] = f"agriculture_overlay_neutral_{reason}"

    for col in DBA_DIAGNOSTIC_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


# ============================================================
# MAIN SCORER
# ============================================================

def build_dba_scores(scores: pd.DataFrame) -> pd.DataFrame:
    base = build_core_identity_commodity_score(
        scores=scores,
        ticker=TICKER,
        model_name=MODEL_NAME,
    )

    if base.empty:
        return base

    base["dba_base_score"] = base["final_score"].copy()

    if not DBA_OVERLAY_ENABLED or DBA_OVERLAY_BLEND_WEIGHT <= 0:
        return _attach_neutral_agriculture_diagnostics(
            base,
            reason="disabled",
        )

    try:
        features = _load_agriculture_features()
    except FileNotFoundError:
        if DBA_OVERLAY_REQUIRE_FEATURES:
            raise

        return _attach_neutral_agriculture_diagnostics(
            base,
            reason="missing_features",
        )

    out = _merge_agriculture_features_asof(
        dba=base,
        features=features,
    )

    missing_feature = out["agri_feature_date"].isna()

    if missing_feature.any() and DBA_OVERLAY_REQUIRE_FEATURES:
        bad_dates = (
            out.loc[missing_feature, "date"]
            .head(5)
            .dt.date
            .tolist()
        )

        raise ValueError(
            "Agriculture overlay could not find as-of features for some DBA dates. "
            f"Examples: {bad_dates}"
        )

    overlay_score = _build_overlay_score(out)

    out["agri_overlay_score"] = overlay_score
    out["agri_overlay_enabled"] = True

    blend = float(DBA_OVERLAY_BLEND_WEIGHT)

    if not 0.0 <= blend <= 1.0:
        raise ValueError(
            f"DBA_OVERLAY_BLEND_WEIGHT must be in [0, 1], got {blend}"
        )

    out["agri_final_score_pre_clip"] = (
        (1.0 - blend) * out["dba_base_score"]
        + blend * out["agri_overlay_score"]
    )

    out["final_score"] = _clip01(out["agri_final_score_pre_clip"])

    out["commodity_model"] = MODEL_NAME
    out["commodity_model_score"] = out["final_score"]
    out["commodity_conviction_score"] = out["agri_overlay_score"]

    quality_col = (
        "agri_expanded_data_quality_score"
        if DBA_USE_EXPORT_DEMAND
        else "agri_core_data_quality_score"
    )

    if quality_col in out.columns:
        quality = pd.to_numeric(out[quality_col], errors="coerce")
    else:
        quality = _neutral_series(out.index, 1.0)

    out["commodity_data_quality_score"] = (
        quality
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    export_tag = "with_exports" if DBA_USE_EXPORT_DEMAND else "no_exports"
    out["commodity_model_version"] = (
        f"agriculture_overlay_v1_usd_rates_crop_momentum_rs_broad_seasonality_{export_tag}"
    )

    for col in DBA_DIAGNOSTIC_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    return out.sort_values(["date", "ticker"]).reset_index(drop=True)
