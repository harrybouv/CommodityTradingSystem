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
        USO_OVERLAY_ENABLED,
        USO_OVERLAY_BLEND_WEIGHT,
        USO_OVERLAY_REQUIRE_FEATURES,
        USO_FEATURE_ASOF_TOLERANCE_DAYS,
        USO_USE_INVENTORY_TIGHTNESS,
        USO_USE_CUSHING_TIGHTNESS,
        USO_USE_CURVE_ROLL,
        USO_USE_SUPPLY_REFINERY,
        USO_USE_GLOBAL_DEMAND,
        USO_USE_USD,
        USO_COMPONENT_WEIGHTS,
    )
except ImportError:
    USO_OVERLAY_ENABLED = True
    USO_OVERLAY_BLEND_WEIGHT = 0.10
    USO_OVERLAY_REQUIRE_FEATURES = True
    USO_FEATURE_ASOF_TOLERANCE_DAYS = 10

    USO_USE_INVENTORY_TIGHTNESS = True
    USO_USE_CUSHING_TIGHTNESS = True
    USO_USE_CURVE_ROLL = True
    USO_USE_SUPPLY_REFINERY = True
    USO_USE_GLOBAL_DEMAND = True
    USO_USE_USD = True

    USO_COMPONENT_WEIGHTS = {
        "oil_inventory_tightness_score": 0.30,
        "oil_cushing_tightness_score": 0.20,
        "oil_curve_roll_score": 0.25,
        "oil_supply_refinery_score": 0.15,
        "oil_global_demand_score": 0.07,
        "oil_usd_score": 0.03,
    }


# ============================================================
# CONSTANTS
# ============================================================

TICKER = "USO"
MODEL_NAME = "oil"

OIL_FEATURES_DAILY_PATH = (
    PROCESSED_DATA_DIR / "oil" / "oil_features_daily.csv"
)


USO_DIAGNOSTIC_COLUMNS = [
    "uso_base_score",
    "oil_overlay_score",
    "oil_final_score_pre_clip",
    "oil_overlay_enabled",
    "oil_feature_date",
    "oil_feature_age_days",
    "oil_core_data_quality_score",
    "oil_core_feature_count",

    "oil_inventory_tightness_score",
    "oil_cushing_tightness_score",
    "oil_curve_roll_score",
    "oil_supply_refinery_score",
    "oil_global_demand_score",
    "oil_usd_score",
    "oil_balance_score",

    "us_crude_stocks_ex_spr",
    "crude_stocks_change_1m",
    "crude_stocks_change_3m",
    "crude_stocks_z_3y",

    "cushing_crude_stocks",
    "cushing_stocks_change_1m",
    "cushing_stocks_change_3m",
    "cushing_stocks_z_3y",

    "wti_spot_price",
    "wti_return_1m",
    "wti_return_3m",
    "wti_return_6m",
    "wti_trend_score",

    "uso_return_1m",
    "uso_return_3m",
    "uso_return_6m",
    "uso_trend_score",
    "uso_momentum_score",

    "usl_price",
    "uso_usl_ratio",
    "uso_usl_ratio_return_1m",
    "uso_usl_ratio_return_3m",
    "uso_usl_ratio_return_6m",
    "uso_usl_ratio_z_3y",
    "uso_usl_ratio_trend_score",

    "bno_price",
    "bno_uso_ratio",
    "bno_uso_ratio_return_3m",

    "us_crude_production",
    "production_change_1m",
    "production_change_3m",
    "production_z_3y",
    "refinery_utilisation",
    "refinery_utilisation_change_1m",
    "refinery_utilisation_change_3m",
    "refinery_utilisation_z_3y",
    "oil_production_relief_score",
    "oil_refinery_usage_score",

    "dbc_return_1m",
    "dbc_return_3m",
    "dbc_return_6m",
    "dbc_trend_score",

    "spy_return_1m",
    "spy_return_3m",
    "spy_return_6m",
    "spy_trend_score",

    "cper_return_1m",
    "cper_return_3m",
    "cper_return_6m",
    "cper_trend_score",

    "indpro",
    "indpro_change_3m",
    "indpro_change_6m",
    "indpro_z_3y",

    "vix_index",
    "vix_z_3y",

    "usd_index",
    "usd_return_1m",
    "usd_return_3m",
    "usd_return_6m",
    "usd_z_3y",

    "commodity_model_version",
]


# ============================================================
# HELPERS
# ============================================================

def _clip01(s: pd.Series) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)


def _neutral_series(index: pd.Index, value: float = 0.50) -> pd.Series:
    return pd.Series(value, index=index, dtype="float64")


def _load_oil_features(path: Path = OIL_FEATURES_DAILY_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Oil features not found: {path}. "
            "Run scoring/commodity_models/Oil/oil_features.py first."
        )

    features = pd.read_csv(path)

    if "date" not in features.columns:
        raise ValueError(f"Oil features file missing date column: {path}")

    features["date"] = pd.to_datetime(features["date"])
    features = (
        features
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    features = features.rename(columns={"date": "oil_feature_date"})

    return features


def _merge_oil_features_asof(
    uso: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    left = uso.copy()
    right = features.copy()

    left["date"] = pd.to_datetime(left["date"])
    right["oil_feature_date"] = pd.to_datetime(right["oil_feature_date"])

    left = left.sort_values("date").reset_index(drop=True)
    right = right.sort_values("oil_feature_date").reset_index(drop=True)

    out = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="oil_feature_date",
        direction="backward",
        tolerance=pd.Timedelta(days=USO_FEATURE_ASOF_TOLERANCE_DAYS),
    )

    out["oil_feature_age_days"] = (
        out["date"] - out["oil_feature_date"]
    ).dt.days

    return out


def _enabled_component_weights() -> dict[str, float]:
    toggles = {
        "oil_inventory_tightness_score": USO_USE_INVENTORY_TIGHTNESS,
        "oil_cushing_tightness_score": USO_USE_CUSHING_TIGHTNESS,
        "oil_curve_roll_score": USO_USE_CURVE_ROLL,
        "oil_supply_refinery_score": USO_USE_SUPPLY_REFINERY,
        "oil_global_demand_score": USO_USE_GLOBAL_DEMAND,
        "oil_usd_score": USO_USE_USD,
    }

    weights = {
        col: float(USO_COMPONENT_WEIGHTS.get(col, 0.0))
        for col, enabled in toggles.items()
        if enabled and float(USO_COMPONENT_WEIGHTS.get(col, 0.0)) > 0.0
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


def _attach_neutral_oil_diagnostics(
    out: pd.DataFrame,
    reason: str,
) -> pd.DataFrame:
    out = out.copy()

    out["uso_base_score"] = out["final_score"]
    out["oil_overlay_score"] = 0.50
    out["oil_final_score_pre_clip"] = out["final_score"]
    out["oil_overlay_enabled"] = False
    out["oil_feature_date"] = pd.NaT
    out["oil_feature_age_days"] = np.nan
    out["oil_core_data_quality_score"] = 0.0
    out["oil_core_feature_count"] = 0

    for col in [
        "oil_inventory_tightness_score",
        "oil_cushing_tightness_score",
        "oil_curve_roll_score",
        "oil_supply_refinery_score",
        "oil_global_demand_score",
        "oil_usd_score",
        "oil_balance_score",
    ]:
        out[col] = 0.50

    out["commodity_model"] = MODEL_NAME
    out["commodity_model_score"] = out["final_score"]
    out["commodity_conviction_score"] = out["oil_overlay_score"]
    out["commodity_data_quality_score"] = 0.0
    out["commodity_model_version"] = f"oil_overlay_neutral_{reason}"

    for col in USO_DIAGNOSTIC_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


# ============================================================
# MAIN SCORER
# ============================================================

def build_uso_scores(scores: pd.DataFrame) -> pd.DataFrame:
    base = build_core_identity_commodity_score(
        scores=scores,
        ticker=TICKER,
        model_name=MODEL_NAME,
    )

    if base.empty:
        return base

    base["uso_base_score"] = base["final_score"].copy()

    if not USO_OVERLAY_ENABLED or USO_OVERLAY_BLEND_WEIGHT <= 0:
        return _attach_neutral_oil_diagnostics(
            base,
            reason="disabled",
        )

    try:
        features = _load_oil_features()
    except FileNotFoundError:
        if USO_OVERLAY_REQUIRE_FEATURES:
            raise

        return _attach_neutral_oil_diagnostics(
            base,
            reason="missing_features",
        )

    out = _merge_oil_features_asof(
        uso=base,
        features=features,
    )

    missing_feature = out["oil_feature_date"].isna()

    if missing_feature.any() and USO_OVERLAY_REQUIRE_FEATURES:
        bad_dates = (
            out.loc[missing_feature, "date"]
            .head(5)
            .dt.date
            .tolist()
        )

        raise ValueError(
            "Oil overlay could not find as-of features for some USO dates. "
            f"Examples: {bad_dates}"
        )

    overlay_score = _build_overlay_score(out)

    out["oil_overlay_score"] = overlay_score
    out["oil_overlay_enabled"] = True

    blend = float(USO_OVERLAY_BLEND_WEIGHT)

    if not 0.0 <= blend <= 1.0:
        raise ValueError(
            f"USO_OVERLAY_BLEND_WEIGHT must be in [0, 1], got {blend}"
        )

    out["oil_final_score_pre_clip"] = (
        (1.0 - blend) * out["uso_base_score"]
        + blend * out["oil_overlay_score"]
    )

    out["final_score"] = _clip01(out["oil_final_score_pre_clip"])

    out["commodity_model"] = MODEL_NAME
    out["commodity_model_score"] = out["final_score"]
    out["commodity_conviction_score"] = out["oil_overlay_score"]

    if "oil_core_data_quality_score" in out.columns:
        quality = pd.to_numeric(
            out["oil_core_data_quality_score"],
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
        "oil_overlay_v1_inventory_cushing_curve_supply_demand_usd"
    )

    for col in USO_DIAGNOSTIC_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    return (
        out
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    print("USO scoring module loaded successfully.")
    print(f"Oil features path: {OIL_FEATURES_DAILY_PATH}")