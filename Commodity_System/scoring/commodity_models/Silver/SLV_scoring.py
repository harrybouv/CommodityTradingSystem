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
from scoring.commodity_models.base_scoring import build_core_identity_commodity_score


try:
    from config import (
        SILVER_OVERLAY_ENABLED,
        SILVER_OVERLAY_BLEND_WEIGHT,
        SILVER_OVERLAY_REQUIRE_FEATURES,
        SILVER_FEATURE_ASOF_TOLERANCE_DAYS,
        SILVER_USE_GOLD_RATIO,
        SILVER_USE_COPPER_RATIO,
        SILVER_USE_GOLD_CONFIRMATION,
        SILVER_USE_USD,
        SILVER_USE_REAL_YIELD,
        SILVER_COMPONENT_WEIGHTS,
    )
except ImportError:
    SILVER_OVERLAY_ENABLED = False
    SILVER_OVERLAY_BLEND_WEIGHT = 0.15
    SILVER_OVERLAY_REQUIRE_FEATURES = True
    SILVER_FEATURE_ASOF_TOLERANCE_DAYS = 10

    SILVER_USE_GOLD_RATIO = True
    SILVER_USE_COPPER_RATIO = True
    SILVER_USE_GOLD_CONFIRMATION = True
    SILVER_USE_USD = True
    SILVER_USE_REAL_YIELD = True

    SILVER_COMPONENT_WEIGHTS = {
        "silver_gold_ratio_score": 0.30,
        "silver_copper_ratio_score": 0.25,
        "silver_gold_confirmation_score": 0.20,
        "silver_usd_score": 0.15,
        "silver_real_yield_score": 0.10,
    }


# ============================================================
# CONSTANTS
# ============================================================

TICKER = "SLV"
MODEL_NAME = "silver"

SILVER_FEATURES_DAILY_PATH = (
    PROCESSED_DATA_DIR / "silver" / "silver_features_daily.csv"
)


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

    "real_yield_10y",
    "real_yield_change_3m",
    "real_yield_z_3y",
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


def _load_silver_features(path: Path = SILVER_FEATURES_DAILY_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Silver features not found: {path}. "
            "Run scoring/commodity_models/Silver/silver_features.py first."
        )

    features = pd.read_csv(path)

    if "date" not in features.columns:
        raise ValueError(f"Silver features file missing date column: {path}")

    features["date"] = pd.to_datetime(features["date"])
    features = (
        features
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    features = features.rename(columns={"date": "silver_feature_date"})

    return features


def _merge_silver_features_asof(
    slv: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    left = slv.copy()
    right = features.copy()

    left["date"] = pd.to_datetime(left["date"])
    right["silver_feature_date"] = pd.to_datetime(right["silver_feature_date"])

    left = left.sort_values("date").reset_index(drop=True)
    right = right.sort_values("silver_feature_date").reset_index(drop=True)

    out = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="silver_feature_date",
        direction="backward",
        tolerance=pd.Timedelta(days=SILVER_FEATURE_ASOF_TOLERANCE_DAYS),
    )

    out["silver_feature_age_days"] = (
        out["date"] - out["silver_feature_date"]
    ).dt.days

    return out


def _enabled_component_weights() -> dict[str, float]:
    toggles = {
        "silver_gold_ratio_score": SILVER_USE_GOLD_RATIO,
        "silver_copper_ratio_score": SILVER_USE_COPPER_RATIO,
        "silver_gold_confirmation_score": SILVER_USE_GOLD_CONFIRMATION,
        "silver_usd_score": SILVER_USE_USD,
        "silver_real_yield_score": SILVER_USE_REAL_YIELD,
    }

    weights = {
        col: float(SILVER_COMPONENT_WEIGHTS.get(col, 0.0))
        for col, enabled in toggles.items()
        if enabled and float(SILVER_COMPONENT_WEIGHTS.get(col, 0.0)) > 0.0
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


def _attach_neutral_silver_diagnostics(
    out: pd.DataFrame,
    reason: str,
) -> pd.DataFrame:
    out = out.copy()

    out["slv_base_score"] = out["final_score"]
    out["silver_overlay_score"] = 0.50
    out["silver_final_score_pre_clip"] = out["final_score"]
    out["silver_overlay_enabled"] = False
    out["silver_feature_date"] = pd.NaT
    out["silver_feature_age_days"] = np.nan
    out["silver_core_data_quality_score"] = 0.0

    for col in [
        "silver_gold_ratio_score",
        "silver_copper_ratio_score",
        "silver_gold_confirmation_score",
        "silver_usd_score",
        "silver_real_yield_score",
        "silver_macro_score",
    ]:
        out[col] = 0.50

    out["commodity_model_version"] = f"silver_overlay_neutral_{reason}"

    return out


# ============================================================
# MAIN SCORER
# ============================================================

def build_slv_scores(scores: pd.DataFrame) -> pd.DataFrame:
    base = build_core_identity_commodity_score(
        scores=scores,
        ticker=TICKER,
        model_name=MODEL_NAME,
    )

    if base.empty:
        return base

    base["slv_base_score"] = base["final_score"].copy()

    if not SILVER_OVERLAY_ENABLED or SILVER_OVERLAY_BLEND_WEIGHT <= 0:
        return _attach_neutral_silver_diagnostics(
            base,
            reason="disabled",
        )

    try:
        features = _load_silver_features()
    except FileNotFoundError:
        if SILVER_OVERLAY_REQUIRE_FEATURES:
            raise

        return _attach_neutral_silver_diagnostics(
            base,
            reason="missing_features",
        )

    out = _merge_silver_features_asof(
        slv=base,
        features=features,
    )

    missing_feature = out["silver_feature_date"].isna()

    if missing_feature.any() and SILVER_OVERLAY_REQUIRE_FEATURES:
        bad_dates = (
            out.loc[missing_feature, "date"]
            .head(5)
            .dt.date
            .tolist()
        )

        raise ValueError(
            "Silver overlay could not find as-of features for some SLV dates. "
            f"Examples: {bad_dates}"
        )

    overlay_score = _build_overlay_score(out)

    out["silver_overlay_score"] = overlay_score
    out["silver_overlay_enabled"] = True

    blend = float(SILVER_OVERLAY_BLEND_WEIGHT)

    if not 0.0 <= blend <= 1.0:
        raise ValueError(
            f"SILVER_OVERLAY_BLEND_WEIGHT must be in [0, 1], got {blend}"
        )

    out["silver_final_score_pre_clip"] = (
        (1.0 - blend) * out["slv_base_score"]
        + blend * out["silver_overlay_score"]
    )

    out["final_score"] = _clip01(out["silver_final_score_pre_clip"])

    out["commodity_model"] = MODEL_NAME
    out["commodity_model_score"] = out["final_score"]
    out["commodity_conviction_score"] = out["silver_overlay_score"]

    if "silver_core_data_quality_score" in out.columns:
        quality = pd.to_numeric(
            out["silver_core_data_quality_score"],
            errors="coerce",
        )
    else:
        quality = _neutral_series(out.index, 1.0)

    out["commodity_data_quality_score"] = (
        quality
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    out["commodity_model_version"] = "silver_overlay_v1_ratio_copper_gold_macro"

    for col in SILVER_DIAGNOSTIC_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    return (
        out
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    print("SLV scoring module loaded successfully.")
    print(f"Silver features path: {SILVER_FEATURES_DAILY_PATH}")