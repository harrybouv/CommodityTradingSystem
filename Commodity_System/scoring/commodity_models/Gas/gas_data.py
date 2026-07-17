from __future__ import annotations

import os
import re
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf


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
    FORWARD_FILL_LIMIT,
)


# ============================================================
# PATHS
# ============================================================

# Pull earlier than the backtest start so rolling z-scores/regimes have
# warm-up history before the 2015 live/backtest period.
GAS_DATA_START_DATE = "2009-01-01"
GAS_DATA_END_DATE = END_DATE

GAS_RAW_DIR = RAW_DATA_DIR / "gas"
GAS_PROCESSED_DIR = PROCESSED_DATA_DIR / "gas"

GAS_EIA_RAW_PATH = GAS_RAW_DIR / "gas_eia_raw.csv"
GAS_FRED_RAW_PATH = GAS_RAW_DIR / "gas_fred_raw.csv"
GAS_YFINANCE_RAW_PATH = GAS_RAW_DIR / "gas_yfinance_raw.csv"
GAS_WEATHER_RAW_PATH = GAS_RAW_DIR / "gas_weather_raw.csv"
GAS_DATA_QUALITY_REPORT_PATH = GAS_RAW_DIR / "gas_data_quality_report.csv"

GAS_RAW_WIDE_PATH = GAS_PROCESSED_DIR / "gas_raw_wide.csv"


# ============================================================
# API SETTINGS
# ============================================================

EIA_API_KEY = os.getenv("EIA_API_KEY")

EIA_SERIESID_BASE_URL = "https://api.eia.gov/v2/seriesid"

NOAA_DEGREE_DAY_BASE_URL = (
    "https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data"
)


# ============================================================
# SERIES DEFINITIONS
# ============================================================

# EIA natural gas series.
#
# date in raw output = estimated public availability date.
# period_date = actual reported period date.
GAS_EIA_SERIES = {
    "NW2_EPG0_SWO_R48_BCF": {
        "api_series_id": "NG.NW2_EPG0_SWO_R48_BCF.W",
        "name": "Weekly Lower 48 working natural gas storage",
        "feature_group": "storage_tightness",
        "frequency": "weekly",
        "unit": "bcf",
        "release_lag_days": 6,
        "required": True,
        "description": (
            "Core storage/tightness input. Low storage versus seasonal normal "
            "is usually bullish natural gas."
        ),
    },
    "N9070US2": {
        "api_series_id": "NG.N9070US2.M",
        "name": "U.S. dry natural gas production",
        "feature_group": "production_supply",
        "frequency": "monthly",
        "unit": "million_cubic_feet",
        "release_lag_days": 45,
        "required": True,
        "description": (
            "Supply pressure input. Rising production can be bearish if demand "
            "and storage conditions do not absorb it."
        ),
    },
    "N9133US2": {
        "api_series_id": "NG.N9133US2.M",
        "name": "Liquefied U.S. natural gas exports",
        "feature_group": "lng_export_demand",
        "frequency": "monthly",
        "unit": "million_cubic_feet",
        "release_lag_days": 45,
        "required": False,
        "description": (
            "Optional LNG/export demand input. Higher exports can link Henry Hub "
            "more tightly to global gas demand."
        ),
    },
}


# FRED series for Henry Hub and oil relative value.
GAS_FRED_SERIES = {
    "DHHNGSP": {
        "name": "Henry Hub natural gas spot price",
        "feature_group": "henry_hub_price",
        "frequency": "daily",
        "unit": "dollars_per_mmbtu",
        "release_lag_days": 1,
        "required": True,
        "description": "Daily Henry Hub spot price. Used for gas price diagnostics and oil-relative-value features.",
    },
    "DCOILWTICO": {
        "name": "WTI crude oil spot price",
        "feature_group": "oil_relative_value",
        "frequency": "daily",
        "unit": "dollars_per_barrel",
        "release_lag_days": 1,
        "required": False,
        "description": "WTI spot price used for gas-versus-oil relative value.",
    },
}


# yfinance traded proxy series.
GAS_YFINANCE_UNIVERSE = {
    "UNL": {
        "name": "United States 12 Month Natural Gas Fund",
        "feature_group": "curve_roll",
        "description": "UNG vs UNL relative strength proxy for front-end natural gas curve / roll pressure.",
        "required": True,
    },
    "USO": {
        "name": "United States Oil Fund",
        "feature_group": "energy_confirmation",
        "description": "Optional broad energy/oil confirmation proxy.",
        "required": False,
    },
    "DBC": {
        "name": "Invesco DB Commodity Index Tracking Fund",
        "feature_group": "broad_commodity_confirmation",
        "description": "Optional broad commodity confirmation proxy.",
        "required": False,
    },
}


# NOAA CPC degree-day files.
#
# Heating:
#   UtilityGas.Heating.txt is ideal for gas heating demand because it is weighted
#   by utility gas usage rather than raw population.
#
# Cooling:
#   There is no UtilityGas.Cooling file in the CPC daily folders. Population.Cooling
#   is the clean automated CONUS proxy for cooling/power-burn demand.
GAS_NOAA_WEATHER_SERIES = {
    "NOAA_UTILITY_GAS_HDD": {
        "file_name": "UtilityGas.Heating.txt",
        "name": "Utility gas weighted heating degree days",
        "feature_group": "weather_demand",
        "frequency": "daily",
        "unit": "degree_days",
        "release_lag_days": 1,
        "required": True,
        "description": "Daily utility-gas-weighted heating degree days. Core winter gas demand input.",
    },
    "NOAA_POPULATION_CDD": {
        "file_name": "Population.Cooling.txt",
        "name": "Population weighted cooling degree days",
        "feature_group": "weather_demand",
        "frequency": "daily",
        "unit": "degree_days",
        "release_lag_days": 1,
        "required": True,
        "description": "Daily population-weighted cooling degree days. Core summer power-burn demand input.",
    },
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def ensure_gas_data_dirs() -> None:
    GAS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    GAS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip()).strip("_").lower()
        for col in out.columns
    ]
    return out


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).replace(".", np.nan),
        errors="coerce",
    )


def _filter_period_date_range(
    df: pd.DataFrame,
    start_date: str | None = GAS_DATA_START_DATE,
    end_date: str | None = GAS_DATA_END_DATE,
) -> pd.DataFrame:
    out = df.copy()
    out["period_date"] = pd.to_datetime(out["period_date"], errors="coerce")

    if start_date is not None:
        out = out[out["period_date"] >= pd.to_datetime(start_date)].copy()

    if end_date is not None:
        out = out[out["period_date"] <= pd.to_datetime(end_date)].copy()

    return out


def _filter_date_range(
    df: pd.DataFrame,
    start_date: str | None = GAS_DATA_START_DATE,
    end_date: str | None = GAS_DATA_END_DATE,
) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    if start_date is not None:
        out = out[out["date"] >= pd.to_datetime(start_date)].copy()

    if end_date is not None:
        out = out[out["date"] <= pd.to_datetime(end_date)].copy()

    return out


def _normalise_period_to_timestamp(value: Any) -> pd.Timestamp:
    """
    EIA/FRED periods can arrive as daily, weekly, or monthly strings.
    This keeps parsing consistent.
    """
    text = str(value).strip()

    # EIA monthly periods often arrive as YYYY-MM.
    if re.fullmatch(r"\d{4}-\d{2}", text):
        return pd.to_datetime(text + "-01", errors="coerce")

    return pd.to_datetime(text, errors="coerce")


# ============================================================
# EIA DATA
# ============================================================

def download_eia_series(
    series_id: str,
    start_date: str | None = GAS_DATA_START_DATE,
    end_date: str | None = GAS_DATA_END_DATE,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Download a single EIA natural gas series using the v2 /seriesid endpoint.
    """

    if api_key is None:
        api_key = EIA_API_KEY

    if not api_key:
        raise RuntimeError(
            f"Missing EIA API key. Set it once as an environment variable named {EIA_API_KEY_ENV}.\n"
            "PowerShell example:\n"
            "$env:EIA_API_KEY='your_key_here'"
        )

    meta = GAS_EIA_SERIES[series_id]
    api_series_id = meta["api_series_id"]

    response = requests.get(
        f"{EIA_SERIESID_BASE_URL}/{api_series_id}",
        params={"api_key": api_key},
        timeout=30,
    )

    response.raise_for_status()
    payload = response.json()

    data = payload.get("response", {}).get("data", [])

    if not data:
        raise ValueError(
            f"EIA API returned no data for {series_id} / {api_series_id}. "
            f"Payload keys: {list(payload.keys())}"
        )

    raw = pd.DataFrame(data)

    date_col = None
    for candidate in ["period", "date"]:
        if candidate in raw.columns:
            date_col = candidate
            break

    if date_col is None:
        raise ValueError(
            f"Could not find period/date column for EIA series {series_id}. "
            f"Available columns: {list(raw.columns)}"
        )

    value_col = None
    for candidate in ["value", "duoarea", "series"]:
        if candidate in raw.columns:
            converted = pd.to_numeric(raw[candidate], errors="coerce")
            if converted.notna().sum() > 10:
                value_col = candidate
                break

    if value_col is None:
        numeric_candidates = []
        for col in raw.columns:
            converted = pd.to_numeric(raw[col], errors="coerce")
            if converted.notna().sum() > 10:
                numeric_candidates.append(col)

        if not numeric_candidates:
            raise ValueError(
                f"Could not find numeric value column for EIA series {series_id}. "
                f"Available columns: {list(raw.columns)}"
            )

        value_col = numeric_candidates[0]

    out = raw[[date_col, value_col]].copy()
    out.columns = ["period_date", "value"]

    out["period_date"] = out["period_date"].apply(_normalise_period_to_timestamp)
    out["value"] = _safe_numeric(out["value"])

    out = out.dropna(subset=["period_date", "value"]).copy()

    out = _filter_period_date_range(
        out,
        start_date=start_date,
        end_date=end_date,
    )

    release_lag_days = int(meta.get("release_lag_days", 0))
    out["date"] = out["period_date"] + pd.Timedelta(days=release_lag_days)

    out["series_id"] = series_id
    out["name"] = meta["name"]
    out["feature_group"] = meta["feature_group"]
    out["frequency"] = meta["frequency"]
    out["unit"] = meta["unit"]
    out["source"] = "EIA"
    out["release_lag_days"] = release_lag_days
    out["required"] = bool(meta.get("required", False))
    out["description"] = meta["description"]

    return out[
        [
            "date",
            "period_date",
            "series_id",
            "name",
            "feature_group",
            "frequency",
            "unit",
            "source",
            "release_lag_days",
            "required",
            "value",
            "description",
        ]
    ].sort_values("date").reset_index(drop=True)


def download_eia_data(
    series_ids: list[str] | None = None,
    start_date: str | None = GAS_DATA_START_DATE,
    end_date: str | None = GAS_DATA_END_DATE,
) -> pd.DataFrame:
    if series_ids is None:
        series_ids = list(GAS_EIA_SERIES.keys())

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    print(f"Downloading EIA gas series: {', '.join(series_ids)}")

    for series_id in series_ids:
        meta = GAS_EIA_SERIES[series_id]
        required = bool(meta.get("required", False))

        try:
            frames.append(
                download_eia_series(
                    series_id=series_id,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        except Exception as e:
            failed.append(series_id)
            print(f"Failed to download/process EIA gas series {series_id}: {e}")

            if required:
                raise

    if not frames:
        raise ValueError("No EIA gas data downloaded.")

    if failed:
        print(f"\nFailed optional EIA gas series: {failed}")

    return pd.concat(frames, ignore_index=True)


def clean_eia_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce")
    df["series_id"] = df["series_id"].astype(str).str.upper().str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["date", "period_date", "series_id", "value"]).copy()

    duplicate_count = df.duplicated(["date", "series_id"]).sum()
    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate EIA gas date/series rows."
        )

    return df.sort_values(["series_id", "date"]).reset_index(drop=True)


def save_eia_data(df: pd.DataFrame, path: Path = GAS_EIA_RAW_PATH) -> None:
    ensure_gas_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved EIA gas data to: {path}")


# ============================================================
# FRED DATA
# ============================================================

def download_fred_series(
    series_id: str,
    start_date: str | None = GAS_DATA_START_DATE,
    end_date: str | None = GAS_DATA_END_DATE,
) -> pd.DataFrame:
    meta = GAS_FRED_SERIES[series_id]
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    raw = pd.read_csv(url)

    if raw.empty or len(raw.columns) < 2:
        raise ValueError(f"FRED series {series_id} downloaded empty or malformed data.")

    date_col = raw.columns[0]
    value_col = raw.columns[1]

    out = raw[[date_col, value_col]].copy()
    out.columns = ["period_date", "value"]

    out["period_date"] = pd.to_datetime(out["period_date"], errors="coerce")
    out["value"] = _safe_numeric(out["value"])

    out = out.dropna(subset=["period_date", "value"]).copy()

    out = _filter_period_date_range(
        out,
        start_date=start_date,
        end_date=end_date,
    )

    release_lag_days = int(meta.get("release_lag_days", 0))
    out["date"] = out["period_date"] + pd.Timedelta(days=release_lag_days)

    out["series_id"] = series_id
    out["name"] = meta["name"]
    out["feature_group"] = meta["feature_group"]
    out["frequency"] = meta["frequency"]
    out["unit"] = meta["unit"]
    out["source"] = "FRED"
    out["release_lag_days"] = release_lag_days
    out["required"] = bool(meta.get("required", False))
    out["description"] = meta["description"]

    return out[
        [
            "date",
            "period_date",
            "series_id",
            "name",
            "feature_group",
            "frequency",
            "unit",
            "source",
            "release_lag_days",
            "required",
            "value",
            "description",
        ]
    ].sort_values("date").reset_index(drop=True)


def download_fred_data(
    series_ids: list[str] | None = None,
    start_date: str | None = GAS_DATA_START_DATE,
    end_date: str | None = GAS_DATA_END_DATE,
) -> pd.DataFrame:
    if series_ids is None:
        series_ids = list(GAS_FRED_SERIES.keys())

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    print(f"Downloading FRED gas series: {', '.join(series_ids)}")

    for series_id in series_ids:
        meta = GAS_FRED_SERIES[series_id]
        required = bool(meta.get("required", False))

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
            print(f"Failed to download/process FRED gas series {series_id}: {e}")

            if required:
                raise

    if not frames:
        raise ValueError("No FRED gas data downloaded.")

    if failed:
        print(f"\nFailed optional FRED gas series: {failed}")

    return pd.concat(frames, ignore_index=True)


def clean_fred_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce")
    df["series_id"] = df["series_id"].astype(str).str.upper().str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["date", "period_date", "series_id", "value"]).copy()

    duplicate_count = df.duplicated(["date", "series_id"]).sum()
    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate FRED gas date/series rows."
        )

    return df.sort_values(["series_id", "date"]).reset_index(drop=True)


def save_fred_data(df: pd.DataFrame, path: Path = GAS_FRED_RAW_PATH) -> None:
    ensure_gas_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved FRED gas data to: {path}")


# ============================================================
# YFINANCE MARKET DATA
# ============================================================

def download_gas_yfinance_data(
    tickers: list[str] | None = None,
    start_date: str | None = GAS_DATA_START_DATE,
    end_date: str | None = GAS_DATA_END_DATE,
) -> pd.DataFrame:
    if tickers is None:
        tickers = list(GAS_YFINANCE_UNIVERSE.keys())

    tickers = [str(ticker).upper().strip() for ticker in tickers]

    print(f"Downloading yfinance gas overlay data for: {', '.join(tickers)}")

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=True,
        group_by="ticker",
        threads=True,
    )

    if raw.empty:
        raise ValueError("No yfinance gas overlay data downloaded.")

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for ticker in tickers:
        meta = GAS_YFINANCE_UNIVERSE.get(ticker, {})
        required = bool(meta.get("required", False))

        try:
            if isinstance(raw.columns, pd.MultiIndex):
                available_top_level = set(raw.columns.get_level_values(0))

                if ticker not in available_top_level:
                    raise ValueError(
                        f"{ticker} not found in downloaded MultiIndex columns."
                    )

                df = raw[ticker].copy()
            else:
                df = raw.copy()

            df = df.reset_index()
            df = _normalise_columns(df)

            if "adj_close" not in df.columns and "close" in df.columns:
                df["adj_close"] = df["close"]

            if "volume" not in df.columns:
                df["volume"] = 0.0

            if "close" not in df.columns:
                raise ValueError(f"{ticker} missing close column.")

            for col in ["open", "high", "low"]:
                if col not in df.columns:
                    df[col] = df["close"]

            required_cols = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
            ]

            missing = [col for col in required_cols if col not in df.columns]

            if missing:
                raise ValueError(f"{ticker} missing columns: {missing}")

            df = df[required_cols].copy()
            df["ticker"] = ticker
            df["name"] = meta.get("name", ticker)
            df["feature_group"] = meta.get("feature_group", "unknown")
            df["source"] = "yfinance"
            df["required"] = required
            df["description"] = meta.get("description", "")

            frames.append(df)

        except Exception as e:
            failed.append(ticker)
            print(f"Failed to process yfinance gas ticker {ticker}: {e}")

            if required:
                raise

    if not frames:
        raise ValueError("No valid yfinance gas overlay ticker data processed.")

    if failed:
        print(f"\nFailed optional yfinance gas tickers: {failed}")

    out = pd.concat(frames, ignore_index=True)

    return out[
        [
            "date",
            "ticker",
            "name",
            "feature_group",
            "source",
            "required",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "description",
        ]
    ].copy()


def clean_gas_yfinance_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    duplicate_count = df.duplicated(["date", "ticker"]).sum()
    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate yfinance gas date/ticker rows."
        )

    df = df[df["adj_close"] > 0].copy()
    df["volume"] = df["volume"].fillna(0.0)
    df = df[df["volume"] >= 0].copy()

    df[numeric_cols] = (
        df.groupby("ticker")[numeric_cols]
        .ffill(limit=FORWARD_FILL_LIMIT)
    )

    df = df.dropna(subset=["adj_close"]).copy()

    df["daily_return"] = (
        df.groupby("ticker")["adj_close"]
        .pct_change()
    )

    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def save_gas_yfinance_data(
    df: pd.DataFrame,
    path: Path = GAS_YFINANCE_RAW_PATH,
) -> None:
    ensure_gas_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved yfinance gas overlay data to: {path}")


def make_yfinance_adj_close_wide(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.pivot(index="date", columns="ticker", values="adj_close")
        .sort_index()
    )

    out.columns = [f"yf_{col}_adj_close" for col in out.columns]

    return out.reset_index()


# ============================================================
# NOAA DEGREE DAY WEATHER DATA
# ============================================================

def _parse_noaa_degree_day_text(
    text: str,
    series_id: str,
    meta: dict[str, Any],
) -> pd.DataFrame:
    """
    Parse NOAA CPC pipe-delimited daily degree-day files.

    The files are spreadsheet-friendly but not always parsed cleanly by pandas
    because the preamble and table header can appear on one line.

    We extract the table beginning at "Region|...", then use the CONUS row.
    """

    if "Region|" not in text:
        raise ValueError(
            f"Could not find Region table header in NOAA degree-day file for {series_id}."
        )

    table_text = text[text.find("Region|") :].strip()

    # Robust fallback if the server/browser gives one long line. Insert row
    # breaks before region rows such as "1|...", "2|...", and "CONUS|...".
    table_text = re.sub(
        r"\s+(CONUS|\d+)\|",
        r"\n\1|",
        table_text,
    )

    table_text = table_text.replace("\r\n", "\n").replace("\r", "\n")

    raw = pd.read_csv(StringIO(table_text), sep="|")

    if raw.empty:
        raise ValueError(f"Parsed NOAA table was empty for {series_id}.")

    first_col = raw.columns[0]
    raw = raw.rename(columns={first_col: "region"})
    raw["region"] = raw["region"].astype(str).str.strip().str.upper()

    conus = raw[raw["region"] == "CONUS"].copy()

    if conus.empty:
        raise ValueError(
            f"Could not find CONUS row in NOAA degree-day file for {series_id}. "
            f"Regions found: {raw['region'].head(20).tolist()}"
        )
    row = conus.iloc[0].drop(labels=["region"])

    out = pd.DataFrame(
        {
            "period_date": pd.to_datetime(
                row.index.astype(str),
                format="%Y%m%d",
                errors="coerce",
            ),
            "value": pd.to_numeric(row.values, errors="coerce"),
        }
    )

    out = out.dropna(subset=["period_date", "value"]).copy()

    release_lag_days = int(meta.get("release_lag_days", 1))
    out["date"] = out["period_date"] + pd.Timedelta(days=release_lag_days)

    out["series_id"] = series_id
    out["name"] = meta["name"]
    out["feature_group"] = meta["feature_group"]
    out["frequency"] = meta["frequency"]
    out["unit"] = meta["unit"]
    out["source"] = "NOAA_CPC"
    out["release_lag_days"] = release_lag_days
    out["required"] = bool(meta.get("required", False))
    out["description"] = meta["description"]

    return out[
        [
            "date",
            "period_date",
            "series_id",
            "name",
            "feature_group",
            "frequency",
            "unit",
            "source",
            "release_lag_days",
            "required",
            "value",
            "description",
        ]
    ].sort_values("date").reset_index(drop=True)


def download_noaa_degree_day_series_for_year(
    series_id: str,
    year: int,
) -> pd.DataFrame:
    meta = GAS_NOAA_WEATHER_SERIES[series_id]
    file_name = meta["file_name"]

    url = f"{NOAA_DEGREE_DAY_BASE_URL}/{year}/{file_name}"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    text = response.text

    if not text.strip():
        raise ValueError(f"NOAA degree-day file downloaded empty: {url}")

    out = _parse_noaa_degree_day_text(
        text=text,
        series_id=series_id,
        meta=meta,
    )

    return out


def download_noaa_weather_data(
    series_ids: list[str] | None = None,
    start_date: str | None = GAS_DATA_START_DATE,
    end_date: str | None = GAS_DATA_END_DATE,
) -> pd.DataFrame:
    if series_ids is None:
        series_ids = list(GAS_NOAA_WEATHER_SERIES.keys())

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date) if end_date is not None else pd.Timestamp.today()

    years = list(range(start_ts.year, end_ts.year + 1))

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    print(
        "Downloading NOAA CPC gas weather degree-day series: "
        f"{', '.join(series_ids)} for years {years[0]}-{years[-1]}"
    )

    for series_id in series_ids:
        meta = GAS_NOAA_WEATHER_SERIES[series_id]
        required = bool(meta.get("required", False))

        for year in years:
            try:
                yearly = download_noaa_degree_day_series_for_year(
                    series_id=series_id,
                    year=year,
                )

                yearly = _filter_period_date_range(
                    yearly,
                    start_date=start_date,
                    end_date=end_date,
                )

                if not yearly.empty:
                    frames.append(yearly)

            except Exception as e:
                label = f"{series_id}:{year}"
                failed.append(label)
                print(f"Failed NOAA weather download/process {label}: {e}")

                # If a required historical year fails, fail fast. For current
                # year, the folder can occasionally lag by a day, but normally exists.
                if required:
                    raise

    if not frames:
        raise ValueError("No NOAA gas weather data downloaded.")

    if failed:
        print(f"\nFailed optional NOAA weather downloads: {failed}")

    out = pd.concat(frames, ignore_index=True)

    duplicate_count = out.duplicated(["date", "series_id"]).sum()
    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate NOAA weather date/series rows."
        )

    return out.sort_values(["series_id", "date"]).reset_index(drop=True)


def clean_noaa_weather_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce")
    df["series_id"] = df["series_id"].astype(str).str.upper().str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["date", "period_date", "series_id", "value"]).copy()

    duplicate_count = df.duplicated(["date", "series_id"]).sum()
    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate NOAA weather date/series rows."
        )

    return df.sort_values(["series_id", "date"]).reset_index(drop=True)


def save_noaa_weather_data(
    df: pd.DataFrame,
    path: Path = GAS_WEATHER_RAW_PATH,
) -> None:
    ensure_gas_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved NOAA gas weather data to: {path}")


# ============================================================
# QUALITY REPORT
# ============================================================

def _quality_from_long_data(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = (
        df.groupby("series_id")
        .agg(
            source=("source", "first"),
            name=("name", "first"),
            feature_group=("feature_group", "first"),
            frequency=("frequency", "first"),
            unit=("unit", "first"),
            required=("required", "first"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            rows=("date", "count"),
            missing_value=("value", lambda x: x.isna().sum()),
            description=("description", "first"),
        )
        .reset_index()
    )

    out["data_layer"] = source_label
    out["passes_value_filter"] = out["missing_value"] == 0
    out["passes_history_filter"] = out["rows"] > 100
    out["keep"] = out["passes_value_filter"] & out["passes_history_filter"]

    def removal_reason(row: pd.Series) -> str:
        reasons: list[str] = []

        if not row["passes_value_filter"]:
            reasons.append("missing_values")

        if not row["passes_history_filter"]:
            reasons.append("insufficient_history")

        if not reasons:
            return "kept"

        return ";".join(reasons)

    out["removal_reason"] = out.apply(removal_reason, axis=1)

    return out


def _quality_from_yfinance_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = (
        df.groupby("ticker")
        .agg(
            source=("source", "first"),
            name=("name", "first"),
            feature_group=("feature_group", "first"),
            required=("required", "first"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            rows=("date", "count"),
            missing_adj_close=("adj_close", lambda x: x.isna().sum()),
            average_volume=("volume", "mean"),
            description=("description", "first"),
        )
        .reset_index()
        .rename(columns={"ticker": "series_id"})
    )

    out["frequency"] = "daily"
    out["unit"] = "adj_close"
    out["data_layer"] = "yfinance"
    out["passes_value_filter"] = out["missing_adj_close"] == 0
    out["passes_history_filter"] = out["rows"] > 100
    out["keep"] = out["passes_value_filter"] & out["passes_history_filter"]

    def removal_reason(row: pd.Series) -> str:
        reasons: list[str] = []

        if not row["passes_value_filter"]:
            reasons.append("missing_adj_close")

        if not row["passes_history_filter"]:
            reasons.append("insufficient_history")

        if not reasons:
            return "kept"

        return ";".join(reasons)

    out["removal_reason"] = out.apply(removal_reason, axis=1)

    return out


def build_gas_data_quality_report(
    eia: pd.DataFrame,
    fred: pd.DataFrame,
    weather: pd.DataFrame,
    yfinance_data: pd.DataFrame,
) -> pd.DataFrame:
    frames = [
        _quality_from_long_data(eia, "eia"),
        _quality_from_long_data(fred, "fred"),
        _quality_from_long_data(weather, "noaa_weather"),
        _quality_from_yfinance_data(yfinance_data),
    ]

    frames = [frame for frame in frames if not frame.empty]

    if not frames:
        raise ValueError("No gas data quality frames to combine.")

    out = pd.concat(frames, ignore_index=True)

    return out.sort_values(["data_layer", "series_id"]).reset_index(drop=True)


def save_gas_data_quality_report(
    df: pd.DataFrame,
    path: Path = GAS_DATA_QUALITY_REPORT_PATH,
) -> None:
    ensure_gas_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved gas data quality report to: {path}")


# ============================================================
# WIDE DATA
# ============================================================

def make_long_value_wide(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date"])

    out = (
        df.pivot(index="date", columns="series_id", values="value")
        .sort_index()
        .reset_index()
    )

    return out


def combine_wide_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    cleaned = []

    for frame in frames:
        if frame is None or frame.empty:
            continue

        out = frame.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).copy()
        cleaned.append(out)

    if not cleaned:
        raise ValueError("No wide gas data frames to combine.")

    combined = cleaned[0]

    for frame in cleaned[1:]:
        combined = combined.merge(
            frame,
            on="date",
            how="outer",
        )

    combined = combined.sort_values("date").reset_index(drop=True)

    duplicate_dates = combined["date"].duplicated().sum()
    if duplicate_dates > 0:
        raise ValueError(f"Found {duplicate_dates} duplicate dates in gas raw wide data.")

    return combined


def save_gas_raw_wide(
    df: pd.DataFrame,
    path: Path = GAS_RAW_WIDE_PATH,
) -> None:
    ensure_gas_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved processed gas raw wide data to: {path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_gas_data_pipeline() -> pd.DataFrame:
    ensure_gas_data_dirs()

    print("\n========== GAS DATA PIPELINE ==========")

    print("\n1. EIA natural gas data")
    eia_raw = download_eia_data()
    eia_clean = clean_eia_data(eia_raw)
    save_eia_data(eia_clean)

    print("\n2. FRED gas/oil price data")
    fred_raw = download_fred_data()
    fred_clean = clean_fred_data(fred_raw)
    save_fred_data(fred_clean)

    print("\n3. yfinance traded gas/energy proxy data")
    yf_raw = download_gas_yfinance_data()
    yf_clean = clean_gas_yfinance_data(yf_raw)
    save_gas_yfinance_data(yf_clean)

    print("\n4. NOAA CPC weather degree-day data")
    weather_raw = download_noaa_weather_data()
    weather_clean = clean_noaa_weather_data(weather_raw)
    save_noaa_weather_data(weather_clean)

    print("\n5. Data quality report")
    quality = build_gas_data_quality_report(
        eia=eia_clean,
        fred=fred_clean,
        weather=weather_clean,
        yfinance_data=yf_clean,
    )
    save_gas_data_quality_report(quality)

    print("\n6. Build processed raw-wide gas data")

    eia_wide = make_long_value_wide(eia_clean)
    fred_wide = make_long_value_wide(fred_clean)
    weather_wide = make_long_value_wide(weather_clean)
    yf_wide = make_yfinance_adj_close_wide(yf_clean)

    gas_wide = combine_wide_frames(
        [
            eia_wide,
            fred_wide,
            weather_wide,
            yf_wide,
        ]
    )

    save_gas_raw_wide(gas_wide)

    print("\nGas data ingestion complete.")
    print(f"Wide rows: {len(gas_wide):,}")
    print(f"Wide columns: {list(gas_wide.columns)}")
    print(f"Start date: {gas_wide['date'].min().date()}")
    print(f"End date: {gas_wide['date'].max().date()}")

    print("\nQuality report:")
    display_cols = [
        "data_layer",
        "series_id",
        "feature_group",
        "frequency",
        "required",
        "start_date",
        "end_date",
        "rows",
        "keep",
        "removal_reason",
    ]
    available_display_cols = [col for col in display_cols if col in quality.columns]
    print(quality[available_display_cols].to_string(index=False))

    return gas_wide


if __name__ == "__main__":
    run_gas_data_pipeline()