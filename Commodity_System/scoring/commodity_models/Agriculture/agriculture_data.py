from __future__ import annotations

import os
import re
import sys
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
    PRICE_DATA_PATH,
)

try:
    from config import (
        DBA_DOWNLOAD_ESR_EXPORTS,
        DBA_USE_CACHED_ESR_EXPORTS,
    )
except ImportError:
    DBA_DOWNLOAD_ESR_EXPORTS = False
    DBA_USE_CACHED_ESR_EXPORTS = True


# ============================================================
# PATHS
# ============================================================

AGRI_DATA_START_DATE = "2009-01-01"
AGRI_DATA_END_DATE = END_DATE

AGRI_RAW_DIR = RAW_DATA_DIR / "agriculture"
AGRI_PROCESSED_DIR = PROCESSED_DATA_DIR / "agriculture"

AGRI_EXISTING_PRICES_RAW_PATH = AGRI_RAW_DIR / "agriculture_existing_prices_raw.csv"
AGRI_FRED_RAW_PATH = AGRI_RAW_DIR / "agriculture_fred_raw.csv"
AGRI_YFINANCE_RAW_PATH = AGRI_RAW_DIR / "agriculture_yfinance_raw.csv"
AGRI_ESR_EXPORTS_RAW_PATH = AGRI_RAW_DIR / "agriculture_esr_exports_raw.csv"
AGRI_ESR_COMMODITIES_LOOKUP_PATH = AGRI_RAW_DIR / "agriculture_esr_commodities_lookup.csv"
AGRI_DATA_QUALITY_REPORT_PATH = AGRI_RAW_DIR / "agriculture_data_quality_report.csv"

AGRI_RAW_WIDE_PATH = AGRI_PROCESSED_DIR / "agriculture_raw_wide.csv"


# ============================================================
# FAS API SETTINGS
# ============================================================

# Hardcode locally if you want:
# FAS_API_KEY = "PASTE_YOUR_KEY_HERE"
#
# Safer version: set environment variable FAS_API_KEY.
# PowerShell:
# $env:FAS_API_KEY="your_key_here"
FAS_API_KEY = os.getenv("FAS_API_KEY")

FAS_API_BASE_URL = "https://api.fas.usda.gov/api"

ESR_COMMODITIES_URL = f"{FAS_API_BASE_URL}/esr/commodities"
ESR_EXPORTS_ALL_COUNTRIES_URL = (
    FAS_API_BASE_URL
    + "/esr/exports/commodityCode/{commodity_code}/allCountries/marketYear/{market_year}"
)


# ============================================================
# DATA DEFINITIONS
# ============================================================

AGRI_FRED_SERIES = {
    "DTWEXBGS": {
        "name": "Nominal Broad U.S. Dollar Index",
        "feature_group": "usd_strength",
        "frequency": "daily",
        "unit": "index",
        "release_lag_days": 1,
        "required": True,
        "description": "Broad USD strength proxy. Stronger USD is usually a headwind for globally priced agricultural commodities.",
    },
    "DGS10": {
        "name": "10-Year Treasury Constant Maturity Rate",
        "feature_group": "rates_pressure",
        "frequency": "daily",
        "unit": "percent",
        "release_lag_days": 1,
        "required": True,
        "description": "Long-rate proxy. Rising yields can tighten liquidity and pressure commodity risk assets indirectly.",
    },
    "DGS2": {
        "name": "2-Year Treasury Constant Maturity Rate",
        "feature_group": "rates_pressure",
        "frequency": "daily",
        "unit": "percent",
        "release_lag_days": 1,
        "required": True,
        "description": "Front-end rate proxy. Captures policy-rate pressure.",
    },
    "T10Y2Y": {
        "name": "10-Year Minus 2-Year Treasury Spread",
        "feature_group": "rates_pressure",
        "frequency": "daily",
        "unit": "percent",
        "release_lag_days": 1,
        "required": False,
        "description": "Yield-curve spread. Optional macro-growth/liquidity diagnostic.",
    },
}


AGRI_EXISTING_PRICE_TICKERS = {
    "DBA": {
        "name": "Invesco DB Agriculture Fund",
        "feature_group": "traded_proxy",
        "required": True,
        "description": "Main agriculture ETF/proxy already used by the core commodity system.",
    },
}


AGRI_YFINANCE_UNIVERSE = {
    "DBC": {
        "name": "Invesco DB Commodity Index Tracking Fund",
        "feature_group": "broad_commodity_confirmation",
        "required": False,
        "description": "Broad commodity trend confirmation proxy.",
    },
    "ZC=F": {
        "name": "Corn Futures",
        "feature_group": "crop_futures_proxy",
        "required": True,
        "description": "Corn futures proxy.",
    },
    "ZW=F": {
        "name": "Wheat Futures",
        "feature_group": "crop_futures_proxy",
        "required": True,
        "description": "Wheat futures proxy.",
    },
    "ZS=F": {
        "name": "Soybean Futures",
        "feature_group": "crop_futures_proxy",
        "required": True,
        "description": "Soybean futures proxy.",
    },
    "SB=F": {
        "name": "Sugar No. 11 Futures",
        "feature_group": "crop_futures_proxy",
        "required": False,
        "description": "Sugar futures proxy.",
    },
    "KC=F": {
        "name": "Coffee Futures",
        "feature_group": "crop_futures_proxy",
        "required": False,
        "description": "Coffee futures proxy.",
    },
    "CC=F": {
        "name": "Cocoa Futures",
        "feature_group": "crop_futures_proxy",
        "required": False,
        "description": "Cocoa futures proxy.",
    },
}


# ESR commodity names are found dynamically from /api/esr/commodities.
# This avoids hardcoding codes like 107, 401, etc.
#
# The pipeline will search these commodity names in the lookup table.
# If a commodity is not found, export data is skipped for that commodity.
AGRI_ESR_EXPORT_TARGETS = {
    "corn": {
        "name_patterns": ["corn"],
        "series_prefix": "ESR_CORN",
        "required": False,
    },
    "all_wheat": {
        "name_patterns": ["all wheat"],
        "series_prefix": "ESR_ALL_WHEAT",
        "required": False,
    },
    "soybeans": {
        "name_patterns": ["soybeans"],
        "series_prefix": "ESR_SOYBEANS",
        "required": False,
    },
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def ensure_agri_data_dirs() -> None:
    AGRI_RAW_DIR.mkdir(parents=True, exist_ok=True)
    AGRI_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip()).strip("_").lower()
        for col in out.columns
    ]
    return out


def _safe_series_name(value: str) -> str:
    return re.sub(
        r"[^0-9A-Za-z]+",
        "_",
        str(value).strip().upper(),
    ).strip("_")


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .replace(".", np.nan),
        errors="coerce",
    )


def _filter_date_range(
    df: pd.DataFrame,
    date_col: str = "date",
    start_date: str | None = AGRI_DATA_START_DATE,
    end_date: str | None = AGRI_DATA_END_DATE,
) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    if start_date is not None:
        out = out[out[date_col] >= pd.to_datetime(start_date)].copy()

    if end_date is not None:
        out = out[out[date_col] <= pd.to_datetime(end_date)].copy()

    return out


def _empty_long_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
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
    )


# ============================================================
# EXISTING SYSTEM PRICE DATA
# ============================================================

def load_existing_commodity_price_data(
    tickers: list[str] | None = None,
    path: Path = PRICE_DATA_PATH,
    start_date: str | None = AGRI_DATA_START_DATE,
    end_date: str | None = AGRI_DATA_END_DATE,
) -> pd.DataFrame:
    if tickers is None:
        tickers = list(AGRI_EXISTING_PRICE_TICKERS.keys())

    if not path.exists():
        raise FileNotFoundError(
            f"Commodity price data not found: {path}. Run data.py first."
        )

    prices = pd.read_csv(path)
    prices = _normalise_columns(prices)

    required_cols = ["date", "ticker", "adj_close"]
    missing = [col for col in required_cols if col not in prices.columns]

    if missing:
        raise ValueError(
            f"commodity_prices.csv missing required columns: {missing}. "
            f"Available columns: {list(prices.columns)}"
        )

    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")

    prices = prices[prices["ticker"].isin(tickers)].copy()
    prices = _filter_date_range(prices, "date", start_date, end_date)

    missing_tickers = [
        ticker for ticker in tickers
        if ticker not in set(prices["ticker"].dropna())
    ]

    if missing_tickers:
        raise ValueError(
            f"Existing commodity price data missing required agriculture tickers: {missing_tickers}"
        )

    if "close" not in prices.columns:
        prices["close"] = prices["adj_close"]

    for col in ["open", "high", "low"]:
        if col not in prices.columns:
            prices[col] = prices["close"]

    if "volume" not in prices.columns:
        prices["volume"] = 0.0

    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")

    rows = []

    for ticker, frame in prices.groupby("ticker"):
        meta = AGRI_EXISTING_PRICE_TICKERS.get(ticker, {})

        out = frame[
            [
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
            ]
        ].copy()

        out["name"] = meta.get("name", ticker)
        out["feature_group"] = meta.get("feature_group", "traded_proxy")
        out["source"] = "existing_commodity_prices"
        out["required"] = bool(meta.get("required", False))
        out["description"] = meta.get("description", "")

        rows.append(out)

    result = pd.concat(rows, ignore_index=True)

    return result[
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
    ].sort_values(["ticker", "date"]).reset_index(drop=True)


def save_existing_price_data(
    df: pd.DataFrame,
    path: Path = AGRI_EXISTING_PRICES_RAW_PATH,
) -> None:
    ensure_agri_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved existing agriculture price data to: {path}")


# ============================================================
# FRED DATA
# ============================================================

def download_fred_series(
    series_id: str,
    start_date: str | None = AGRI_DATA_START_DATE,
    end_date: str | None = AGRI_DATA_END_DATE,
) -> pd.DataFrame:
    meta = AGRI_FRED_SERIES[series_id]
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
    out = _filter_date_range(out, "period_date", start_date, end_date)

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
    start_date: str | None = AGRI_DATA_START_DATE,
    end_date: str | None = AGRI_DATA_END_DATE,
) -> pd.DataFrame:
    if series_ids is None:
        series_ids = list(AGRI_FRED_SERIES.keys())

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    print(f"Downloading FRED agriculture macro/rates series: {', '.join(series_ids)}")

    for series_id in series_ids:
        meta = AGRI_FRED_SERIES[series_id]
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
            print(f"Failed to download/process FRED agriculture series {series_id}: {e}")

            if required:
                raise

    if not frames:
        raise ValueError("No FRED agriculture data downloaded.")

    if failed:
        print(f"\nFailed optional FRED agriculture series: {failed}")

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
            f"Found {duplicate_count} duplicate FRED agriculture date/series rows."
        )

    return df.sort_values(["series_id", "date"]).reset_index(drop=True)


def save_fred_data(df: pd.DataFrame, path: Path = AGRI_FRED_RAW_PATH) -> None:
    ensure_agri_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved FRED agriculture data to: {path}")


# ============================================================
# YFINANCE MARKET DATA
# ============================================================

def download_agri_yfinance_data(
    tickers: list[str] | None = None,
    start_date: str | None = AGRI_DATA_START_DATE,
    end_date: str | None = AGRI_DATA_END_DATE,
) -> pd.DataFrame:
    if tickers is None:
        tickers = list(AGRI_YFINANCE_UNIVERSE.keys())

    tickers = [str(ticker).upper().strip() for ticker in tickers]

    print(f"Downloading yfinance agriculture overlay data for: {', '.join(tickers)}")

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
        raise ValueError("No yfinance agriculture overlay data downloaded.")

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for ticker in tickers:
        meta = AGRI_YFINANCE_UNIVERSE.get(ticker, {})
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

            if "date" not in df.columns and "datetime" in df.columns:
                df = df.rename(columns={"datetime": "date"})

            if "date" not in df.columns:
                raise ValueError(f"{ticker} missing date column after reset_index.")

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
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            for col in ["open", "high", "low", "close", "adj_close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["date", "adj_close"]).copy()
            df = _filter_date_range(df, "date", start_date, end_date)

            df["ticker"] = ticker
            df["name"] = meta.get("name", ticker)
            df["feature_group"] = meta.get("feature_group", "unknown")
            df["source"] = "yfinance"
            df["required"] = required
            df["description"] = meta.get("description", "")

            frames.append(df)

        except Exception as e:
            failed.append(ticker)
            print(f"Failed to process yfinance agriculture ticker {ticker}: {e}")

            if required:
                raise

    if not frames:
        raise ValueError("No valid yfinance agriculture ticker data processed.")

    if failed:
        print(f"\nFailed optional yfinance agriculture tickers: {failed}")

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
    ].sort_values(["ticker", "date"]).reset_index(drop=True)


def clean_agri_yfinance_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "ticker", "adj_close"]).copy()

    duplicate_count = df.duplicated(["date", "ticker"]).sum()
    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate yfinance agriculture date/ticker rows."
        )

    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def save_agri_yfinance_data(
    df: pd.DataFrame,
    path: Path = AGRI_YFINANCE_RAW_PATH,
) -> None:
    ensure_agri_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved yfinance agriculture data to: {path}")


# ============================================================
# ESR EXPORT SALES DATA
# ============================================================

def _has_fas_api_key() -> bool:
    return bool(FAS_API_KEY and FAS_API_KEY not in {"PASTE_YOUR_KEY_HERE", "REDACTED"})


def _fas_headers() -> dict[str, str]:
    return {
        "accept": "application/json",
        "X-Api-Key": FAS_API_KEY,
        "User-Agent": "CommodityTradingSystem/1.0",
    }


def _fas_get_json(url: str, timeout: int = 60) -> list[dict[str, Any]]:
    response = requests.get(
        url,
        headers=_fas_headers(),
        timeout=timeout,
    )

    if response.status_code >= 400:
        print("\nFAS API request failed.")
        print(f"URL: {response.url}")
        print(f"Status: {response.status_code}")
        print(f"Response preview: {response.text[:500]}")
        response.raise_for_status()

    payload = response.json()

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ["data", "results", "response"]:
            if key in payload and isinstance(payload[key], list):
                return payload[key]

        return [payload]

    raise ValueError(f"Unexpected FAS API payload type: {type(payload)}")


def download_esr_commodities_lookup() -> pd.DataFrame:
    if not _has_fas_api_key():
        print("No FAS API key found. Skipping ESR commodities lookup.")
        return pd.DataFrame()

    raw = _fas_get_json(ESR_COMMODITIES_URL)
    df = pd.DataFrame(raw)
    df = _normalise_columns(df)

    required_cols = ["commodity_code", "commodity_name"]

    # The API normally returns commodityCode / commodityName.
    if "commoditycode" in df.columns:
        df = df.rename(columns={"commoditycode": "commodity_code"})

    if "commodityname" in df.columns:
        df = df.rename(columns={"commodityname": "commodity_name"})

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"ESR commodities lookup missing columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df["commodity_code"] = pd.to_numeric(df["commodity_code"], errors="coerce")
    df["commodity_name"] = df["commodity_name"].astype(str)

    df = df.dropna(subset=["commodity_code", "commodity_name"]).copy()
    df["commodity_code"] = df["commodity_code"].astype(int)

    ensure_agri_data_dirs()
    df.to_csv(AGRI_ESR_COMMODITIES_LOOKUP_PATH, index=False)
    print(f"Saved ESR commodities lookup to: {AGRI_ESR_COMMODITIES_LOOKUP_PATH}")

    return df.sort_values("commodity_code").reset_index(drop=True)


def _find_esr_commodity_code(
    lookup: pd.DataFrame,
    name_patterns: list[str],
) -> int | None:
    if lookup.empty:
        return None

    names = lookup["commodity_name"].astype(str).str.lower()

    for pattern in name_patterns:
        pattern_l = pattern.lower()
        matched = lookup[names.str.contains(pattern_l, regex=False)].copy()

        if not matched.empty:
            # Prefer exact-ish shortest name.
            matched["name_len"] = matched["commodity_name"].astype(str).str.len()
            matched = matched.sort_values(["name_len", "commodity_code"])
            return int(matched.iloc[0]["commodity_code"])

    return None


def _extract_date_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "week_ending_date",
        "weekendingdate",
        "weekly_exports_date",
        "export_date",
        "date",
        "report_date",
        "reportdate",
        "release_date",
        "releasedate",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        col_l = col.lower()
        if "date" in col_l:
            return col

    return None


def _extract_numeric_export_columns(df: pd.DataFrame) -> list[str]:
    preferred_terms = [
        "weekly",
        "current",
        "exports",
        "outstanding",
        "sales",
        "commitment",
        "accumulated",
    ]

    numeric_cols = []

    for col in df.columns:
        col_l = col.lower()

        if any(term in col_l for term in preferred_terms):
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                numeric_cols.append(col)

    return numeric_cols


def download_esr_export_data_for_target(
    target_key: str,
    target_meta: dict[str, Any],
    lookup: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    commodity_code = _find_esr_commodity_code(
        lookup=lookup,
        name_patterns=target_meta["name_patterns"],
    )

    if commodity_code is None:
        print(f"Could not find ESR commodity code for {target_key}. Skipping.")
        return _empty_long_frame()

    frames: list[pd.DataFrame] = []
    failed: list[int] = []

    for market_year in range(int(start_year), int(end_year) + 1):
        url = ESR_EXPORTS_ALL_COUNTRIES_URL.format(
            commodity_code=int(commodity_code),
            market_year=int(market_year),
        )

        try:
            raw = _fas_get_json(url)
            if not raw:
                continue

            df = pd.DataFrame(raw)
            df = _normalise_columns(df)

            if df.empty:
                continue

            date_col = _extract_date_column(df)
            if date_col is None:
                print(
                    f"ESR {target_key} {market_year}: no date column found. "
                    f"Available columns: {list(df.columns)}"
                )
                continue

            df["date"] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=["date"]).copy()

            numeric_cols = _extract_numeric_export_columns(df)

            if not numeric_cols:
                print(
                    f"ESR {target_key} {market_year}: no useful numeric export columns found. "
                    f"Available columns: {list(df.columns)}"
                )
                continue

            rows = []

            for col in numeric_cols:
                values = pd.to_numeric(df[col], errors="coerce")

                if values.notna().sum() == 0:
                    continue

                series_id = f"{target_meta['series_prefix']}_{_safe_series_name(col)}"

                temp = pd.DataFrame(
                    {
                        "date": df["date"],
                        "period_date": df["date"],
                        "series_id": series_id,
                        "name": f"{target_key} {col}",
                        "feature_group": "export_demand",
                        "frequency": "weekly",
                        "unit": "unknown",
                        "source": "USDA_FAS_ESR_API",
                        "release_lag_days": 0,
                        "required": bool(target_meta.get("required", False)),
                        "value": values,
                        "description": (
                            "USDA FAS ESR weekly export-sales/export-demand data. "
                            "Used only as a secondary agriculture demand-confirmation feature."
                        ),
                    }
                )

                rows.append(temp)

            if rows:
                frames.append(pd.concat(rows, ignore_index=True))

        except Exception as e:
            failed.append(market_year)
            print(f"Failed ESR fetch/process for {target_key} {market_year}: {e}")

    if failed:
        print(f"ESR failures for {target_key}: {failed[:10]}")
        if len(failed) > 10:
            print(f"... plus {len(failed) - 10} more")

    if not frames:
        return _empty_long_frame()

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["date", "series_id", "value"]).copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"]).copy()

    # Multiple country rows may exist on the same date. Sum them.
    group_cols = [
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
        "description",
    ]

    out = (
        out.groupby(group_cols, dropna=False)["value"]
        .sum()
        .reset_index()
        .sort_values(["series_id", "date"])
        .reset_index(drop=True)
    )

    return out


def download_esr_export_data(
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    if not _has_fas_api_key():
        print("No FAS API key found. Skipping ESR export data.")
        return _empty_long_frame()

    if start_year is None:
        start_year = pd.to_datetime(AGRI_DATA_START_DATE).year

    if end_year is None:
        end_year = (
            pd.Timestamp.today().year + 1
            if AGRI_DATA_END_DATE is None
            else pd.to_datetime(AGRI_DATA_END_DATE).year + 1
        )

    print("\nDownloading USDA FAS ESR export-demand data")

    lookup = download_esr_commodities_lookup()

    if lookup.empty:
        print("Empty ESR commodities lookup. Skipping ESR export data.")
        return _empty_long_frame()

    frames = []

    for target_key, target_meta in AGRI_ESR_EXPORT_TARGETS.items():
        frame = download_esr_export_data_for_target(
            target_key=target_key,
            target_meta=target_meta,
            lookup=lookup,
            start_year=start_year,
            end_year=end_year,
        )

        if not frame.empty:
            frames.append(frame)

    if not frames:
        print("No ESR export data downloaded.")
        return _empty_long_frame()

    out = pd.concat(frames, ignore_index=True)
    out = _filter_date_range(out, "date", AGRI_DATA_START_DATE, AGRI_DATA_END_DATE)

    duplicate_count = out.duplicated(["date", "series_id"]).sum()
    if duplicate_count > 0:
        print(
            f"Warning: ESR export data had {duplicate_count} duplicate date/series rows after processing."
        )
        out = (
            out.groupby(
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
                    "description",
                ],
                dropna=False,
            )["value"]
            .sum()
            .reset_index()
        )

    return out.sort_values(["series_id", "date"]).reset_index(drop=True)


def save_esr_export_data(
    df: pd.DataFrame,
    path: Path = AGRI_ESR_EXPORTS_RAW_PATH,
) -> None:
    ensure_agri_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved ESR agriculture export data to: {path}")

def load_cached_esr_export_data(
    path: Path = AGRI_ESR_EXPORTS_RAW_PATH,
) -> pd.DataFrame:
    if not path.exists():
        print(f"No cached ESR export data found at: {path}")
        return _empty_long_frame()

    df = pd.read_csv(path)

    if df.empty:
        print(f"Cached ESR export data is empty: {path}")
        return _empty_long_frame()

    required_cols = [
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

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Cached ESR export data missing columns: {missing}. "
            f"Path: {path}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["date", "series_id", "value"]).copy()

    print(f"Loaded cached ESR export data from: {path}")
    print(f"Cached ESR rows: {len(df):,}")

    return df[required_cols].sort_values(["series_id", "date"]).reset_index(drop=True)

# ============================================================
# DATA QUALITY
# ============================================================

def _min_rows_for_frequency(frequency: str) -> int:
    frequency = str(frequency).lower()

    if frequency == "daily":
        return 252

    if frequency == "weekly":
        return 52

    if frequency == "monthly":
        return 24

    if frequency == "annual":
        return 5

    return 10


def _quality_from_long_data(
    df: pd.DataFrame,
    data_layer: str,
) -> pd.DataFrame:
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

    out["data_layer"] = data_layer
    out["min_required_rows"] = out["frequency"].apply(_min_rows_for_frequency)
    out["passes_value_filter"] = out["missing_value"] == 0
    out["passes_history_filter"] = out["rows"] >= out["min_required_rows"]
    out["keep"] = out["passes_value_filter"] & out["passes_history_filter"]

    def removal_reason(row: pd.Series) -> str:
        reasons = []

        if not row["passes_value_filter"]:
            reasons.append("missing_value")

        if not row["passes_history_filter"]:
            reasons.append("insufficient_history")

        if not reasons:
            return "kept"

        return ";".join(reasons)

    out["removal_reason"] = out.apply(removal_reason, axis=1)

    return out


def _quality_from_yfinance_data(
    df: pd.DataFrame,
    data_layer: str,
) -> pd.DataFrame:
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
    out["data_layer"] = data_layer
    out["passes_value_filter"] = out["missing_adj_close"] == 0
    out["passes_history_filter"] = out["rows"] >= 252
    out["keep"] = out["passes_value_filter"] & out["passes_history_filter"]

    def removal_reason(row: pd.Series) -> str:
        reasons = []

        if not row["passes_value_filter"]:
            reasons.append("missing_adj_close")

        if not row["passes_history_filter"]:
            reasons.append("insufficient_history")

        if not reasons:
            return "kept"

        return ";".join(reasons)

    out["removal_reason"] = out.apply(removal_reason, axis=1)

    return out


def build_agri_data_quality_report(
    existing_prices: pd.DataFrame,
    fred: pd.DataFrame,
    yfinance_data: pd.DataFrame,
    esr_exports: pd.DataFrame,
) -> pd.DataFrame:
    frames = [
        _quality_from_yfinance_data(existing_prices, "existing_commodity_prices"),
        _quality_from_long_data(fred, "fred"),
        _quality_from_yfinance_data(yfinance_data, "yfinance"),
        _quality_from_long_data(esr_exports, "fas_esr_exports"),
    ]

    frames = [frame for frame in frames if not frame.empty]

    if not frames:
        raise ValueError("No agriculture data quality frames to combine.")

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["data_layer", "series_id"]).reset_index(drop=True)


def save_agri_data_quality_report(
    df: pd.DataFrame,
    path: Path = AGRI_DATA_QUALITY_REPORT_PATH,
) -> None:
    ensure_agri_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved agriculture data quality report to: {path}")


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


def make_price_adj_close_wide(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date"])

    out = df.copy()
    out["ticker_clean"] = out["ticker"].apply(_safe_series_name)
    out["series_col"] = "yf_" + out["ticker_clean"] + "_adj_close"

    wide = (
        out.pivot(index="date", columns="series_col", values="adj_close")
        .sort_index()
        .reset_index()
    )

    return wide


def combine_wide_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    cleaned = []

    for frame in frames:
        if frame is None or frame.empty:
            continue

        out = frame.copy()

        if "date" not in out.columns:
            raise ValueError(f"Wide frame missing date column. Columns: {list(out.columns)}")

        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).copy()
        cleaned.append(out)

    if not cleaned:
        raise ValueError("No wide agriculture data frames to combine.")

    combined = cleaned[0]

    for frame in cleaned[1:]:
        overlapping_cols = [
            col for col in frame.columns
            if col != "date" and col in combined.columns
        ]

        if overlapping_cols:
            raise ValueError(
                f"Duplicate non-date columns while combining agriculture wide frames: {overlapping_cols}"
            )

        combined = combined.merge(frame, on="date", how="outer")

    combined = combined.sort_values("date").reset_index(drop=True)

    duplicate_dates = combined["date"].duplicated().sum()
    if duplicate_dates > 0:
        raise ValueError(
            f"Found {duplicate_dates} duplicate dates in agriculture raw wide data."
        )

    return combined


def save_agri_raw_wide(
    df: pd.DataFrame,
    path: Path = AGRI_RAW_WIDE_PATH,
) -> None:
    ensure_agri_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved processed agriculture raw wide data to: {path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_agriculture_data_pipeline() -> pd.DataFrame:
    ensure_agri_data_dirs()

    print("\n========== AGRICULTURE DATA PIPELINE ==========")

    print("\n1. Existing system DBA price data")
    existing_raw = load_existing_commodity_price_data()
    save_existing_price_data(existing_raw)

    print("\n2. FRED agriculture macro/rates data")
    fred_raw = download_fred_data()
    fred_clean = clean_fred_data(fred_raw)
    save_fred_data(fred_clean)

    print("\n3. yfinance agriculture futures/proxy data")
    yf_raw = download_agri_yfinance_data()
    yf_clean = clean_agri_yfinance_data(yf_raw)
    save_agri_yfinance_data(yf_clean)

    print("\n4. Optional USDA FAS ESR export-demand data")

    if DBA_DOWNLOAD_ESR_EXPORTS:
        print("Downloading fresh USDA FAS ESR export-demand data.")
        esr_raw = download_esr_export_data()
        save_esr_export_data(esr_raw)

    elif DBA_USE_CACHED_ESR_EXPORTS:
        print("Skipping fresh ESR download. Using cached ESR export data if available.")
        esr_raw = load_cached_esr_export_data()

    else:
        print("Skipping ESR export-demand data entirely.")
        esr_raw = _empty_long_frame()

    print("\n5. Data quality report")
    quality = build_agri_data_quality_report(
        existing_prices=existing_raw,
        fred=fred_clean,
        yfinance_data=yf_clean,
        esr_exports=esr_raw,
    )
    save_agri_data_quality_report(quality)

    print("\n6. Build processed raw-wide agriculture data")

    existing_wide = make_price_adj_close_wide(existing_raw)
    fred_wide = make_long_value_wide(fred_clean)
    yf_wide = make_price_adj_close_wide(yf_clean)
    esr_wide = make_long_value_wide(esr_raw)

    agri_wide = combine_wide_frames(
        [
            existing_wide,
            fred_wide,
            yf_wide,
            esr_wide,
        ]
    )

    save_agri_raw_wide(agri_wide)

    print("\nAgriculture data ingestion complete.")
    print(f"Wide rows: {len(agri_wide):,}")
    print(f"Wide columns: {list(agri_wide.columns)}")
    print(f"Start date: {agri_wide['date'].min().date()}")
    print(f"End date: {agri_wide['date'].max().date()}")

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

    return agri_wide


if __name__ == "__main__":
    run_agriculture_data_pipeline()