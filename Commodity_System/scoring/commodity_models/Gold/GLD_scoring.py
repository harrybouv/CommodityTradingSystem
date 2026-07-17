from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR
from scoring.commodity_models.base_scoring import build_core_identity_commodity_score

try:
    from config import (
        GOLD_OVERLAY_ENABLED,
        GOLD_OVERLAY_BLEND_WEIGHT,
        GOLD_OVERLAY_REQUIRE_FEATURES,
        GOLD_FEATURE_ASOF_TOLERANCE_DAYS,
        GOLD_USE_REAL_YIELD,
        GOLD_USE_USD,
        GOLD_USE_STRESS,
        GOLD_USE_POLICY_RATE_REGIME,
        GOLD_USE_CENTRAL_BANK_DEMAND,
        GOLD_USE_POSITIONING_CROWDING,
        GOLD_COMPONENT_WEIGHTS,
    )
except ImportError:
    GOLD_OVERLAY_ENABLED = True
    GOLD_OVERLAY_BLEND_WEIGHT = 0.20
    GOLD_OVERLAY_REQUIRE_FEATURES = True
    GOLD_FEATURE_ASOF_TOLERANCE_DAYS = 10
    GOLD_USE_REAL_YIELD = True
    GOLD_USE_USD = True
    GOLD_USE_STRESS = True
    GOLD_USE_POLICY_RATE_REGIME = False
    GOLD_USE_CENTRAL_BANK_DEMAND = False
    GOLD_USE_POSITIONING_CROWDING = False
    GOLD_COMPONENT_WEIGHTS = {
        "gold_real_yield_score": 0.45,
        "gold_usd_score": 0.35,
        "gold_stress_score": 0.20,
        "gold_policy_rate_score": 0.00,
        "gold_central_bank_score": 0.00,
        "gold_positioning_score": 0.00,
    }


TICKER = "GLD"
MODEL_NAME = "gold"
GOLD_FEATURES_DAILY_PATH = PROCESSED_DATA_DIR / "gold" / "gold_features_daily.csv"


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
    "commodity_model_version",
]


def _clip01(s: pd.Series) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)


def _neutral_series(index: pd.Index, value: float = 0.50) -> pd.Series:
    return pd.Series(value, index=index, dtype="float64")


def _load_gold_features(path: Path = GOLD_FEATURES_DAILY_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Gold features not found: {path}. Run gold_features.py after gold_data.py."
        )

    features = pd.read_csv(path)
    if "date" not in features.columns:
        raise ValueError(f"Gold features file missing date column: {path}")

    features["date"] = pd.to_datetime(features["date"])
    features = features.sort_values("date").drop_duplicates("date", keep="last")

    rename_map = {
        "real_yield_score": "gold_real_yield_score",
        "usd_score": "gold_usd_score",
        "stress_score": "gold_stress_score",
        "policy_rate_score": "gold_policy_rate_score",
        "central_bank_score": "gold_central_bank_score",
        "positioning_score": "gold_positioning_score",
    }

    features = features.rename(
        columns={
            old: new
            for old, new in rename_map.items()
            if old in features.columns
        }
    )

    features = features.rename(columns={"date": "gold_feature_date"})
    return features


def _merge_gold_features_asof(gld: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    left = gld.copy()
    left["date"] = pd.to_datetime(left["date"])
    left = left.sort_values("date").reset_index(drop=True)

    right = features.copy().sort_values("gold_feature_date").reset_index(drop=True)

    out = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="gold_feature_date",
        direction="backward",
        tolerance=pd.Timedelta(days=GOLD_FEATURE_ASOF_TOLERANCE_DAYS),
    )

    out["gold_feature_age_days"] = (
        out["date"] - out["gold_feature_date"]
    ).dt.days

    return out


def _enabled_component_weights() -> dict[str, float]:
    toggles = {
        "gold_real_yield_score": GOLD_USE_REAL_YIELD,
        "gold_usd_score": GOLD_USE_USD,
        "gold_stress_score": GOLD_USE_STRESS,
        "gold_policy_rate_score": GOLD_USE_POLICY_RATE_REGIME,
        "gold_central_bank_score": GOLD_USE_CENTRAL_BANK_DEMAND,
        "gold_positioning_score": GOLD_USE_POSITIONING_CROWDING,
    }

    weights = {
        col: float(GOLD_COMPONENT_WEIGHTS.get(col, 0.0))
        for col, enabled in toggles.items()
        if enabled and float(GOLD_COMPONENT_WEIGHTS.get(col, 0.0)) > 0.0
    }

    total = sum(weights.values())
    if total <= 0:
        return {}

    return {col: weight / total for col, weight in weights.items()}


def _build_overlay_score(data: pd.DataFrame) -> pd.Series:
    weights = _enabled_component_weights()
    if not weights:
        return _neutral_series(data.index)

    overlay = pd.Series(0.0, index=data.index, dtype="float64")

    for col, weight in weights.items():
        component = data[col] if col in data.columns else _neutral_series(data.index)
        component = _clip01(pd.to_numeric(component, errors="coerce").fillna(0.50))
        overlay += weight * component

    return _clip01(overlay.fillna(0.50))


def _attach_neutral_gold_diagnostics(out: pd.DataFrame, reason: str) -> pd.DataFrame:
    out = out.copy()
    out["gld_base_score"] = out["final_score"]
    out["gold_overlay_score"] = 0.50
    out["gold_final_score_pre_clip"] = out["final_score"]
    out["gold_overlay_enabled"] = False
    out["gold_feature_date"] = pd.NaT
    out["gold_feature_age_days"] = np.nan
    out["gold_core_data_quality_score"] = 0.0
    out["gold_liquidity_squeeze_flag"] = 0

    for col in [
        "gold_real_yield_score",
        "gold_usd_score",
        "gold_stress_score",
        "gold_policy_rate_score",
        "gold_central_bank_score",
        "gold_positioning_score",
    ]:
        out[col] = 0.50

    out["commodity_model_version"] = f"gold_overlay_neutral_{reason}"
    return out


def build_gld_scores(scores: pd.DataFrame) -> pd.DataFrame:

    base = build_core_identity_commodity_score(
        scores=scores,
        ticker=TICKER,
        model_name=MODEL_NAME,
    )

    if base.empty:
        return base

    base["gld_base_score"] = base["final_score"].copy()

    if not GOLD_OVERLAY_ENABLED or GOLD_OVERLAY_BLEND_WEIGHT <= 0:
        return _attach_neutral_gold_diagnostics(base, reason="disabled")

    try:
        features = _load_gold_features()
    except FileNotFoundError:
        if GOLD_OVERLAY_REQUIRE_FEATURES:
            raise
        return _attach_neutral_gold_diagnostics(base, reason="missing_features")

    out = _merge_gold_features_asof(base, features)

    missing_feature = out["gold_feature_date"].isna()
    if missing_feature.any() and GOLD_OVERLAY_REQUIRE_FEATURES:
        bad_dates = out.loc[missing_feature, "date"].head(5).dt.date.tolist()
        raise ValueError(
            "Gold overlay could not find as-of features for some GLD dates. "
            f"Examples: {bad_dates}"
        )

    overlay_score = _build_overlay_score(out)
    out["gold_overlay_score"] = overlay_score
    out["gold_overlay_enabled"] = True

    blend = float(GOLD_OVERLAY_BLEND_WEIGHT)
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"GOLD_OVERLAY_BLEND_WEIGHT must be in [0, 1], got {blend}")

    out["gold_final_score_pre_clip"] = (
        (1.0 - blend) * out["gld_base_score"]
        + blend * out["gold_overlay_score"]
    )

    out["final_score"] = _clip01(out["gold_final_score_pre_clip"])

    out["commodity_model"] = MODEL_NAME
    out["commodity_model_score"] = out["final_score"]
    out["commodity_conviction_score"] = out["gold_overlay_score"]
    out["commodity_data_quality_score"] = out.get(
        "gold_core_data_quality_score",
        _neutral_series(out.index, 1.0),
    ).fillna(0.0).clip(0.0, 1.0)
    out["commodity_model_version"] = "gold_macro_overlay_v1_real_yield_usd_stress"

    for col in GOLD_DIAGNOSTIC_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    return out.sort_values(["date", "ticker"]).reset_index(drop=True)
