# DELETE API KEY 0f9dc417-7950-289d-4bd3-a61ef1e52f50


from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


# ============================================================
# DIRECT-RUN PATH SETUP
# ============================================================

THIS_FILE = Path(__file__).resolve()
COMMODITY_ROOT = THIS_FILE.parents[3]

if str(COMMODITY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMODITY_ROOT))


from config import (
    END_DATE,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)

try:
    from config import (
        COPPER_CHINA_ELECTRICITY_RELEASE_LAG_DAYS,
        COPPER_CHINA_CLI_RELEASE_LAG_DAYS,
    )
except ImportError:
    COPPER_CHINA_ELECTRICITY_RELEASE_LAG_DAYS = 20
    COPPER_CHINA_CLI_RELEASE_LAG_DAYS = 15


# ============================================================
# PATHS
# ============================================================

COPPER_DATA_START_DATE = "2009-01-01"
COPPER_DATA_END_DATE = END_DATE

COPPER_RAW_DIR = RAW_DATA_DIR / "copper"
COPPER_PROCESSED_DIR = PROCESSED_DATA_DIR / "copper"

COPPER_FRED_RAW_PATH = COPPER_RAW_DIR / "copper_fred_raw.csv"
COPPER_EMBER_RAW_PATH = COPPER_RAW_DIR / "copper_ember_china_electricity_raw.csv"
COPPER_DATA_QUALITY_REPORT_PATH = COPPER_RAW_DIR / "copper_data_quality_report.csv"

COPPER_RAW_WIDE_PATH = COPPER_PROCESSED_DIR / "copper_raw_wide.csv"


# ============================================================
# SERIES DEFINITIONS
# ============================================================

COPPER_FRED_SERIES = {
    "CHNLOLITOAASTSAM": {
        "name": "OECD China CLI amplitude adjusted",
        "feature_group": "china_cycle",
        "frequency": "monthly",
        "description": "China/OECD industrial cycle proxy for copper demand.",
        "release_lag_days": COPPER_CHINA_CLI_RELEASE_LAG_DAYS,
    },
}

EMBER_API_BASE_URL = "https://api.ember-energy.org"
EMBER_API_KEY_ENV = "EMBER_API_KEY"
EMBER_API_KEY = os.getenv("EMBER_API_KEY")
EMBER_CHINA_ENTITY_CODE = "CHN"

EMBER_CHINA_ELECTRICITY_SERIES_ID = "EMBER_CHINA_ELECTRICITY_DEMAND_TWH"


# ============================================================
# GENERAL HELPERS
# ============================================================

def ensure_copper_data_dirs() -> None:
    COPPER_RAW_DIR.mkdir(parents=True, exist_ok=True)
    COPPER_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip()).strip("_").lower()
        for col in out.columns
    ]
    return out


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.replace(".", np.nan),
        errors="coerce",
    )


def _filter_period_date_range(
    df: pd.DataFrame,
    start_date: str | None = COPPER_DATA_START_DATE,
    end_date: str | None = COPPER_DATA_END_DATE,
) -> pd.DataFrame:
    out = df.copy()
    out["period_date"] = pd.to_datetime(out["period_date"])

    if start_date is not None:
        out = out[out["period_date"] >= pd.to_datetime(start_date)].copy()

    if end_date is not None:
        out = out[out["period_date"] <= pd.to_datetime(end_date)].copy()

    return out


def _month_available_date(
    period_date: pd.Series,
    lag_days: int,
) -> pd.Series:
    """
    Converts monthly economic period dates into conservative availability dates.

    This avoids lookahead bias. The model should not be able to use a January
    China electricity figure on 31 January if that value would only have been
    available weeks later.
    """
    dates = pd.to_datetime(period_date)

    return (
        dates
        + pd.offsets.MonthEnd(0)
        + pd.to_timedelta(lag_days, unit="D")
    )


def _extract_first_record_list(payload: Any) -> list[dict[str, Any]]:
    """
    Ember responses are JSON objects. This keeps the loader robust if the API
    response wrapper changes slightly.

    We only accept a list of dict records.
    """
    if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
        return payload

    if isinstance(payload, dict):
        for key in ["data", "records", "results", "result"]:
            value = payload.get(key)

            if isinstance(value, list) and all(isinstance(row, dict) for row in value):
                return value

            if isinstance(value, dict):
                try:
                    return _extract_first_record_list(value)
                except ValueError:
                    pass

        for value in payload.values():
            if isinstance(value, (dict, list)):
                try:
                    return _extract_first_record_list(value)
                except ValueError:
                    pass

    raise ValueError(
        "Could not find a list of data records in Ember API response. "
        "Print the raw JSON response and update _extract_first_record_list()."
    )


def _find_first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    columns = set(df.columns)

    for col in candidates:
        if col in columns:
            return col

    return None


# ============================================================
# FRED DATA
# ============================================================

def download_fred_series(
    series_id: str,
    start_date: str | None = COPPER_DATA_START_DATE,
    end_date: str | None = COPPER_DATA_END_DATE,
) -> pd.DataFrame:
    meta = COPPER_FRED_SERIES[series_id]
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    raw = pd.read_csv(url)

    if raw.empty or len(raw.columns) < 2:
        raise ValueError(f"FRED series {series_id} downloaded empty or malformed data.")

    date_col = raw.columns[0]
    value_col = raw.columns[1]

    out = raw[[date_col, value_col]].copy()
    out.columns = ["period_date", "value"]

    out["period_date"] = pd.to_datetime(out["period_date"])
    out["value"] = _safe_numeric(out["value"])

    out = _filter_period_date_range(
        out,
        start_date=start_date,
        end_date=end_date,
    )

    out = out.dropna(subset=["value"]).copy()

    out["available_date"] = _month_available_date(
        out["period_date"],
        lag_days=int(meta["release_lag_days"]),
    )

    out["series_id"] = series_id
    out["name"] = meta["name"]
    out["feature_group"] = meta["feature_group"]
    out["frequency"] = meta["frequency"]
    out["source"] = "FRED"
    out["description"] = meta["description"]

    return out[
        [
            "period_date",
            "available_date",
            "series_id",
            "name",
            "feature_group",
            "frequency",
            "source",
            "value",
            "description",
        ]
    ].sort_values("available_date").reset_index(drop=True)


def download_fred_data(
    series_ids: list[str] | None = None,
    start_date: str | None = COPPER_DATA_START_DATE,
    end_date: str | None = COPPER_DATA_END_DATE,
) -> pd.DataFrame:
    if series_ids is None:
        series_ids = list(COPPER_FRED_SERIES.keys())

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    print(f"Downloading FRED copper macro series: {', '.join(series_ids)}")

    for series_id in series_ids:
        try:
            frames.append(
                download_fred_series(
                    series_id=series_id,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        except Exception as e:
            failed.append(series_id)
            print(f"Failed to download FRED series {series_id}: {e}")

    if not frames:
        raise ValueError("No FRED copper macro data downloaded.")

    if failed:
        print(f"\nFailed FRED series: {failed}")

    return pd.concat(frames, ignore_index=True)


# ============================================================
# EMBER CHINA ELECTRICITY DEMAND
# ============================================================

def download_ember_china_electricity_demand(
    api_key: str | None = None,
    start_date: str | None = COPPER_DATA_START_DATE,
    end_date: str | None = COPPER_DATA_END_DATE,
) -> pd.DataFrame:
    """
    Downloads monthly China electricity demand from Ember.

    Requires an Ember API key in the EMBER_API_KEY environment variable.
    Do not hard-code the key in this file.
    """
    if api_key is None:
        api_key = EMBER_API_KEY

    if not api_key:
        raise RuntimeError(
            f"Missing Ember API key. Set environment variable {EMBER_API_KEY_ENV}. "
            "Example in PowerShell: $env:EMBER_API_KEY='your_key_here'"
        )

    endpoint = f"{EMBER_API_BASE_URL}/v1/electricity-demand/monthly"

    params: dict[str, str] = {
        "entity_code": EMBER_CHINA_ENTITY_CODE,
        "start_date": pd.to_datetime(start_date).strftime("%Y-%m"),
        "api_key": api_key,
    }

    if end_date is not None:
        params["end_date"] = pd.to_datetime(end_date).strftime("%Y-%m")

    response = requests.get(
        endpoint,
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Ember API request failed. "
            f"Status={response.status_code}. Body={response.text[:500]}"
        )

    records = _extract_first_record_list(response.json())

    raw = pd.DataFrame(records)
    raw = _normalise_columns(raw)

    if raw.empty:
        raise ValueError("Ember China electricity demand response contained no rows.")

    date_col = _find_first_existing_column(
        raw,
        ["date", "month", "period", "period_date"],
    )

    value_col = _find_first_existing_column(
        raw,
        [
            "value",
            "demand",
            "electricity_demand",
            "electricity_demand_twh",
            "demand_twh",
        ],
    )

    if date_col is None or value_col is None:
        raise ValueError(
            "Could not identify Ember date/value columns. "
            f"Available columns: {list(raw.columns)}"
        )

    out = raw[[date_col, value_col]].copy()
    out.columns = ["period_date", "value"]

    out["period_date"] = pd.to_datetime(out["period_date"])
    out["value"] = _safe_numeric(out["value"])

    out = _filter_period_date_range(
        out,
        start_date=start_date,
        end_date=end_date,
    )

    out = out.dropna(subset=["value"]).copy()

    out["available_date"] = _month_available_date(
        out["period_date"],
        lag_days=COPPER_CHINA_ELECTRICITY_RELEASE_LAG_DAYS,
    )

    out["series_id"] = EMBER_CHINA_ELECTRICITY_SERIES_ID
    out["name"] = "China electricity demand"
    out["feature_group"] = "china_electricity"
    out["frequency"] = "monthly"
    out["source"] = "Ember"
    out["description"] = (
        "Monthly China electricity demand in TWh from Ember. "
        "Used as an industrial demand proxy for copper."
    )

    return out[
        [
            "period_date",
            "available_date",
            "series_id",
            "name",
            "feature_group",
            "frequency",
            "source",
            "value",
            "description",
        ]
    ].sort_values("available_date").reset_index(drop=True)


# ============================================================
# CLEAN / SAVE
# ============================================================

def validate_long_data(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()

    required_cols = [
        "period_date",
        "available_date",
        "series_id",
        "source",
        "value",
    ]

    missing = [col for col in required_cols if col not in out.columns]

    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

    out["period_date"] = pd.to_datetime(out["period_date"])
    out["available_date"] = pd.to_datetime(out["available_date"])
    out["series_id"] = out["series_id"].astype(str).str.strip()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    out = out.dropna(subset=["period_date", "available_date", "series_id", "value"]).copy()

    duplicate_count = out.duplicated(["available_date", "series_id"]).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"{name} has {duplicate_count} duplicate available_date/series_id rows."
        )

    return out.sort_values(["series_id", "available_date"]).reset_index(drop=True)


def make_copper_raw_wide(long_data: pd.DataFrame) -> pd.DataFrame:
    data = validate_long_data(long_data, "combined copper long data")

    wide = (
        data
        .assign(date=lambda x: pd.to_datetime(x["available_date"]))
        .pivot(index="date", columns="series_id", values="value")
        .sort_index()
        .reset_index()
    )

    wide.columns.name = None

    return wide


def build_data_quality_report(long_data: pd.DataFrame) -> pd.DataFrame:
    data = validate_long_data(long_data, "combined copper long data")

    report = (
        data
        .groupby(["series_id", "source", "frequency", "feature_group"])
        .agg(
            first_period_date=("period_date", "min"),
            last_period_date=("period_date", "max"),
            first_available_date=("available_date", "min"),
            last_available_date=("available_date", "max"),
            rows=("value", "count"),
            missing_values=("value", lambda x: x.isna().sum()),
        )
        .reset_index()
    )

    report["passes_basic_history_filter"] = report["rows"] >= 36

    return report.sort_values("series_id").reset_index(drop=True)


def save_copper_data(
    fred: pd.DataFrame,
    ember: pd.DataFrame,
    wide: pd.DataFrame,
    quality_report: pd.DataFrame,
) -> None:
    ensure_copper_data_dirs()

    fred.to_csv(COPPER_FRED_RAW_PATH, index=False)
    ember.to_csv(COPPER_EMBER_RAW_PATH, index=False)
    wide.to_csv(COPPER_RAW_WIDE_PATH, index=False)
    quality_report.to_csv(COPPER_DATA_QUALITY_REPORT_PATH, index=False)

    print(f"\nSaved FRED copper data to: {COPPER_FRED_RAW_PATH}")
    print(f"Saved Ember China electricity data to: {COPPER_EMBER_RAW_PATH}")
    print(f"Saved copper raw wide data to: {COPPER_RAW_WIDE_PATH}")
    print(f"Saved copper data quality report to: {COPPER_DATA_QUALITY_REPORT_PATH}")


# ============================================================
# PIPELINE
# ============================================================

def run_copper_data_pipeline() -> pd.DataFrame:
    print("\nStarting copper data pipeline...")
    print(f"Project root: {COMMODITY_ROOT}")
    print(f"Raw dir:      {COPPER_RAW_DIR}")
    print(f"Processed:    {COPPER_PROCESSED_DIR}")

    fred = download_fred_data()
    ember = download_ember_china_electricity_demand()

    fred = validate_long_data(fred, "FRED copper data")
    ember = validate_long_data(ember, "Ember China electricity data")

    combined = pd.concat([fred, ember], ignore_index=True)

    wide = make_copper_raw_wide(combined)
    quality_report = build_data_quality_report(combined)

    save_copper_data(
        fred=fred,
        ember=ember,
        wide=wide,
        quality_report=quality_report,
    )

    print("\nCopper data pipeline complete.")
    print(f"Wide rows:   {len(wide):,}")
    print(f"Start date:  {wide['date'].min().date()}")
    print(f"End date:    {wide['date'].max().date()}")

    print("\nCopper data quality:")
    print(quality_report.to_string(index=False))

    return wide


if __name__ == "__main__":
    run_copper_data_pipeline()