# Commodity_System/scoring/commodity_models/base_scoring.py

from __future__ import annotations

import numpy as np
import pandas as pd

from config import SCORE_WEIGHTS


def _require_columns(
    df: pd.DataFrame,
    required_cols: list[str],
    name: str,
) -> None:
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def build_core_identity_commodity_score(
    scores: pd.DataFrame,
    ticker: str,
    model_name: str,
    score_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Zero-behaviour-change commodity model.

    For now, every commodity-specific model reproduces the existing V2
    production score exactly.

    Later, this function can be replaced inside each ticker-specific model
    with commodity-specific logic while keeping the same output schema.
    """

    if score_weights is None:
        score_weights = SCORE_WEIGHTS

    out = scores[scores["ticker"] == ticker].copy()

    if out.empty:
        return out

    active_weights = {
        col: weight
        for col, weight in score_weights.items()
        if abs(weight) > 1e-12
    }

    _require_columns(
        out,
        ["date", "ticker"] + list(active_weights.keys()),
        f"{ticker} commodity model input",
    )

    final_score = pd.Series(0.0, index=out.index)

    for col, weight in active_weights.items():
        final_score += weight * (
            out[col]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    final_score = final_score.clip(0.0, 1.0)

    # Critical: this is the exact score currently used by the allocator.
    out["final_score"] = final_score

    # Diagnostics only. These are not used for allocation yet.
    out["commodity_model"] = model_name
    out["commodity_model_score"] = final_score
    out["commodity_conviction_score"] = 1.0
    out["commodity_risk_score"] = (
        out["risk_score"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.50)
        .clip(0.0, 1.0)
    )
    out["commodity_sizing_multiplier"] = 1.0
    out["commodity_data_quality_score"] = 1.0
    out["commodity_model_version"] = "core_v2_identity"

    return out.sort_values(["date", "ticker"]).reset_index(drop=True)