# Commodity_System/scoring/commodity_models/registry.py

from __future__ import annotations

import pandas as pd

from scoring.commodity_models.Gold.GLD_scoring import build_gld_scores
from scoring.commodity_models.Silver.SLV_scoring import build_slv_scores
from scoring.commodity_models.Oil.USO_scoring import build_uso_scores
from scoring.commodity_models.Gas.UNG_scoring import build_ung_scores
from scoring.commodity_models.Copper.CPER_scoring import build_cper_scores
from scoring.commodity_models.Agriculture.DBA_scoring import build_dba_scores
COMMODITY_SCORERS = {
    "GLD": build_gld_scores,
    "SLV": build_slv_scores,
    "USO": build_uso_scores,
    "UNG": build_ung_scores,
    "CPER": build_cper_scores,
    "DBA": build_dba_scores,
}


def build_commodity_model_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the production commodity-model score table.

    This is a refactor-only layer:
    - input rows must equal output rows
    - date/ticker structure must not change
    - final_score must match the old V2 weighted score exactly
    """

    required_cols = {"date", "ticker"}

    missing = required_cols.difference(scores.columns)

    if missing:
        raise ValueError(
            f"Commodity model input missing required columns: {sorted(missing)}"
        )

    input_keys = (
        scores[["date", "ticker"]]
        .copy()
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )

    frames = []

    available_tickers = set(scores["ticker"].unique())
    registered_tickers = set(COMMODITY_SCORERS.keys())

    missing_scorers = available_tickers.difference(registered_tickers)

    if missing_scorers:
        raise ValueError(
            "Missing commodity scorers for tickers: "
            f"{sorted(missing_scorers)}"
        )

    for ticker, scorer in COMMODITY_SCORERS.items():
        if ticker not in available_tickers:
            continue

        ticker_scores = scorer(scores)

        if ticker_scores.empty:
            continue

        frames.append(ticker_scores)

    if not frames:
        raise ValueError("No commodity model scores were produced.")

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])

    duplicate_count = out.duplicated(["date", "ticker"]).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Commodity model output has {duplicate_count} duplicate date/ticker rows."
        )

    output_keys = (
        out[["date", "ticker"]]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )

    if len(output_keys) != len(input_keys):
        raise ValueError(
            "Commodity model output row count changed: "
            f"{len(input_keys)} -> {len(output_keys)}"
        )

    if not output_keys.equals(input_keys):
        raise ValueError(
            "Commodity model output changed the date/ticker structure."
        )

    if "final_score" not in out.columns:
        raise ValueError("Commodity model output missing final_score.")

    out["final_score"] = (
        out["final_score"]
        .replace([float("inf"), float("-inf")], pd.NA)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    return (
        out
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )