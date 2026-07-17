from __future__ import annotations

import re
import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd
import yfinance as yf
import requests



# ============================================================
# DIRECT-RUN PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import (
    END_DATE,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FORWARD_FILL_LIMIT,
)

EIA_API_KEY = os.getenv("EIA_API_KEY")

# ============================================================
# PATHS
# ============================================================

# Pull earlier than the backtest start so later rolling z-scores/regimes
# have warm-up history before the 2015 live/backtest period.
OIL_DATA_START_DATE = "2009-01-01"
OIL_DATA_END_DATE = END_DATE

OIL_RAW_DIR = RAW_DATA_DIR / "oil"
OIL_PROCESSED_DIR = PROCESSED_DATA_DIR / "oil"

OIL_EIA_RAW_PATH = OIL_RAW_DIR / "oil_eia_raw.csv"
OIL_FRED_RAW_PATH = OIL_RAW_DIR / "oil_fred_raw.csv"
OIL_YFINANCE_RAW_PATH = OIL_RAW_DIR / "oil_yfinance_raw.csv"
OIL_DATA_QUALITY_REPORT_PATH = OIL_RAW_DIR / "oil_data_quality_report.csv"
OIL_RAW_WIDE_PATH = OIL_PROCESSED_DIR / "oil_raw_wide.csv"


# ============================================================
# SERIES DEFINITIONS
# ============================================================

# EIA weekly petroleum data. These use the public EIA "Download Data (XLS)"
# endpoint rather than an API key.
#
# date in raw output = estimated public availability date, not period end date.
# period_date remains available as a separate column.
OIL_EIA_SERIES = {
    "WCESTUS1": {
        "api_series_id": "PET.WCESTUS1.W",
        "name": "Weekly U.S. commercial crude stocks excluding SPR",
        "feature_group": "inventory_tightness",
        "frequency": "weekly",
        "unit": "thousand_barrels",
        "release_lag_days": 5,
        "required": True,
        "description": "Core crude inventory tightness input. Low/falling stocks are bullish oil.",
    },
    "W_EPC0_SAX_YCUOK_MBBL": {
        "api_series_id": "PET.W_EPC0_SAX_YCUOK_MBBL.W",
        "name": "Weekly Cushing, OK crude stocks excluding SPR",
        "feature_group": "cushing_tightness",
        "frequency": "weekly",
        "unit": "thousand_barrels",
        "release_lag_days": 5,
        "required": True,
        "description": "WTI delivery-hub inventory tightness input. Low/falling Cushing stocks are bullish WTI/USO.",
    },
    "WCRFPUS2": {
        "api_series_id": "PET.WCRFPUS2.W",
        "name": "Weekly U.S. field production of crude oil",
        "feature_group": "production_supply",
        "frequency": "weekly",
        "unit": "thousand_barrels_per_day",
        "release_lag_days": 5,
        "required": True,
        "description": "Supply pressure input. Rising production can be bearish if demand/inventories do not absorb it.",
    },
    "WPULEUS3": {
        "api_series_id": "PET.WPULEUS3.W",
        "name": "Weekly U.S. refinery utilisation",
        "feature_group": "refinery_demand",
        "frequency": "weekly",
        "unit": "percent",
        "release_lag_days": 5,
        "required": True,
        "description": "Crude demand/refinery usage input. Strong utilisation supports crude demand.",
    },
}


# FRED series are used for market/economic data that FRED already serves cleanly.
OIL_FRED_SERIES = {
    "DCOILWTICO": {
        "name": "WTI crude oil spot price",
        "feature_group": "wti_price",
        "frequency": "daily",
        "release_lag_days": 1,
        "required": True,
        "description": "WTI spot price diagnostic and oil market level input.",
    },
    "INDPRO": {
        "name": "U.S. industrial production index",
        "feature_group": "demand_cycle",
        "frequency": "monthly",
        "release_lag_days": 15,
        "required": False,
        "description": "Slow demand-cycle proxy. Optional because global demand can also use existing macro proxies.",
    },
}


# yfinance is used for traded proxy series.
# USL is important because USO/USL relative strength can proxy front-end curve / roll conditions.
OIL_YFINANCE_UNIVERSE = {
    "USL": {
        "name": "United States 12 Month Oil Fund",
        "feature_group": "curve_roll",
        "description": "USO vs USL relative strength proxy for front-end futures curve / roll environment.",
        "required": True,
    },
    "BNO": {
        "name": "United States Brent Oil Fund",
        "feature_group": "brent_wti_spread_proxy",
        "description": "Optional Brent oil ETF proxy. Useful later for Brent/WTI relative pressure.",
        "required": False,
    },
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def ensure_oil_data_dirs() -> None:
    OIL_RAW_DIR.mkdir(parents=True, exist_ok=True)
    OIL_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


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
    start_date: str | None = OIL_DATA_START_DATE,
    end_date: str | None = OIL_DATA_END_DATE,
) -> pd.DataFrame:
    out = df.copy()
    out["period_date"] = pd.to_datetime(out["period_date"])

    if start_date is not None:
        out = out[out["period_date"] >= pd.to_datetime(start_date)].copy()

    if end_date is not None:
        out = out[out["period_date"] <= pd.to_datetime(end_date)].copy()

    return out


def _filter_date_range(
    df: pd.DataFrame,
    start_date: str | None = OIL_DATA_START_DATE,
    end_date: str | None = OIL_DATA_END_DATE,
) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    if start_date is not None:
        out = out[out["date"] >= pd.to_datetime(start_date)].copy()

    if end_date is not None:
        out = out[out["date"] <= pd.to_datetime(end_date)].copy()

    return out


# ============================================================
# EIA XLS DATA
# ============================================================

def _eia_hist_xls_url(series_id: str, frequency: str) -> str:
    suffix = {
        "weekly": "w",
        "monthly": "m",
        "annual": "a",
    }.get(frequency)

    if suffix is None:
        raise ValueError(f"Unsupported EIA frequency: {frequency}")

    return f"https://www.eia.gov/dnav/pet/hist_xls/{series_id}{suffix}.xls"

def download_eia_hist_xls_series(
    series_id: str,
    start_date: str | None = OIL_DATA_START_DATE,
    end_date: str | None = OIL_DATA_END_DATE,
) -> pd.DataFrame:
    """
    Download a single EIA petroleum series using the EIA API v2 series-id endpoint.

    This permanently avoids brittle parsing of EIA historical XLS files.
    """

    meta = OIL_EIA_SERIES[series_id]
    api_series_id = meta["api_series_id"]

    if not EIA_API_KEY or EIA_API_KEY == "PASTE_YOUR_EIA_API_KEY_HERE":
        raise ValueError(
            "Missing EIA API key. Set EIA_API_KEY near the top of oil_data.py."
        )

    url = f"https://api.eia.gov/v2/seriesid/{api_series_id}"

    response = requests.get(
        url,
        params={"api_key": EIA_API_KEY},
        timeout=30,
    )

    response.raise_for_status()
    payload = response.json()

    data = (
        payload
        .get("response", {})
        .get("data", [])
    )

    if not data:
        raise ValueError(
            f"EIA API returned no data for {series_id} / {api_series_id}. "
            f"Payload keys: {list(payload.keys())}"
        )

    raw = pd.DataFrame(data)

    date_col = None
    value_col = None

    for candidate in ["period", "date"]:
        if candidate in raw.columns:
            date_col = candidate
            break

    for candidate in ["value", "duoarea", "series"]:
        if candidate in raw.columns:
            candidate_values = pd.to_numeric(raw[candidate], errors="coerce")
            if candidate_values.notna().sum() > 50:
                value_col = candidate
                break

    if date_col is None:
        raise ValueError(
            f"Could not find date/period column for EIA series {series_id}. "
            f"Available columns: {list(raw.columns)}"
        )

    if value_col is None:
        numeric_candidates = []

        for col in raw.columns:
            converted = pd.to_numeric(raw[col], errors="coerce")
            if converted.notna().sum() > 50:
                numeric_candidates.append(col)

        if not numeric_candidates:
            raise ValueError(
                f"Could not find numeric value column for EIA series {series_id}. "
                f"Available columns: {list(raw.columns)}"
            )

        value_col = numeric_candidates[0]

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
    out["source"] = "EIA"
    out["release_lag_days"] = release_lag_days
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
            "value",
            "description",
        ]
    ].sort_values("date").reset_index(drop=True)


def download_eia_data(
    series_ids: list[str] | None = None,
    start_date: str | None = OIL_DATA_START_DATE,
    end_date: str | None = OIL_DATA_END_DATE,
) -> pd.DataFrame:
    if series_ids is None:
        series_ids = list(OIL_EIA_SERIES.keys())

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    print(f"Downloading EIA oil series: {', '.join(series_ids)}")

    for series_id in series_ids:
        meta = OIL_EIA_SERIES[series_id]
        required = bool(meta.get("required", False))

        try:
            frames.append(
                download_eia_hist_xls_series(
                    series_id=series_id,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        except Exception as e:
            failed.append(series_id)
            print(f"Failed to download/process EIA series {series_id}: {e}")

            if required:
                raise

    if not frames:
        raise ValueError("No EIA oil data downloaded.")

    if failed:
        print(f"\nFailed optional EIA series: {failed}")

    return pd.concat(frames, ignore_index=True)


def clean_eia_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    df["date"] = pd.to_datetime(df["date"])
    df["period_date"] = pd.to_datetime(df["period_date"])
    df["series_id"] = df["series_id"].astype(str).str.upper().str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    duplicate_count = df.duplicated(["date", "series_id"]).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate EIA date/series rows."
        )

    df = df.dropna(subset=["value"]).copy()

    return df.sort_values(["series_id", "date"]).reset_index(drop=True)


def save_eia_data(df: pd.DataFrame, path: Path = OIL_EIA_RAW_PATH) -> None:
    ensure_oil_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved EIA oil data to: {path}")


# ============================================================
# FRED DATA
# ============================================================

def download_fred_series(
    series_id: str,
    start_date: str | None = OIL_DATA_START_DATE,
    end_date: str | None = OIL_DATA_END_DATE,
) -> pd.DataFrame:
    meta = OIL_FRED_SERIES[series_id]
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    raw = pd.read_csv(url)

    if raw.empty or len(raw.columns) < 2:
        raise ValueError(
            f"FRED series {series_id} downloaded empty or malformed data."
        )

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
    out["unit"] = "index_or_price"
    out["source"] = "FRED"
    out["release_lag_days"] = release_lag_days
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
            "value",
            "description",
        ]
    ].sort_values("date").reset_index(drop=True)


def download_fred_data(
    series_ids: list[str] | None = None,
    start_date: str | None = OIL_DATA_START_DATE,
    end_date: str | None = OIL_DATA_END_DATE,
) -> pd.DataFrame:
    if series_ids is None:
        series_ids = list(OIL_FRED_SERIES.keys())

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    print(f"Downloading FRED oil macro/price series: {', '.join(series_ids)}")

    for series_id in series_ids:
        meta = OIL_FRED_SERIES[series_id]
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
            print(f"Failed to download FRED series {series_id}: {e}")

            if required:
                raise

    if not frames:
        raise ValueError("No FRED oil data downloaded.")

    if failed:
        print(f"\nFailed optional FRED series: {failed}")

    return pd.concat(frames, ignore_index=True)


def clean_fred_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    df["date"] = pd.to_datetime(df["date"])
    df["period_date"] = pd.to_datetime(df["period_date"])
    df["series_id"] = df["series_id"].astype(str).str.upper().str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    duplicate_count = df.duplicated(["date", "series_id"]).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate FRED date/series rows."
        )

    df = df.dropna(subset=["value"]).copy()

    return df.sort_values(["series_id", "date"]).reset_index(drop=True)


def save_fred_data(df: pd.DataFrame, path: Path = OIL_FRED_RAW_PATH) -> None:
    ensure_oil_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved FRED oil data to: {path}")


# ============================================================
# YFINANCE MARKET DATA
# ============================================================

def download_oil_yfinance_data(
    tickers: list[str] | None = None,
    start_date: str | None = OIL_DATA_START_DATE,
    end_date: str | None = OIL_DATA_END_DATE,
) -> pd.DataFrame:
    if tickers is None:
        tickers = list(OIL_YFINANCE_UNIVERSE.keys())

    tickers = [str(ticker).upper().strip() for ticker in tickers]

    print(f"Downloading yfinance oil overlay data for: {', '.join(tickers)}")

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
        raise ValueError("No yfinance oil overlay data downloaded.")

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for ticker in tickers:
        meta = OIL_YFINANCE_UNIVERSE.get(ticker, {})
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
            df["description"] = meta.get("description", "")

            frames.append(df)

        except Exception as e:
            failed.append(ticker)
            print(f"Failed to process yfinance ticker {ticker}: {e}")

            if required:
                raise

    if not frames:
        raise ValueError("No valid yfinance oil overlay ticker data processed.")

    if failed:
        print(f"\nFailed optional yfinance tickers: {failed}")

    out = pd.concat(frames, ignore_index=True)

    return out[
        [
            "date",
            "ticker",
            "name",
            "feature_group",
            "source",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "description",
        ]
    ].copy()


def clean_oil_yfinance_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    duplicate_count = df.duplicated(["date", "ticker"]).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate yfinance date/ticker rows."
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


def save_oil_yfinance_data(
    df: pd.DataFrame,
    path: Path = OIL_YFINANCE_RAW_PATH,
) -> None:
    ensure_oil_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved yfinance oil overlay data to: {path}")


# ============================================================
# WIDE TABLE BUILDING
# ============================================================

def make_series_wide(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.pivot(index="date", columns="series_id", values="value")
        .sort_index()
        .reset_index()
    )

    return out


def make_yfinance_adj_close_wide(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.pivot(index="date", columns="ticker", values="adj_close")
        .sort_index()
    )

    out.columns = [f"yf_{col}_adj_close" for col in out.columns]

    return out.reset_index()


def build_oil_raw_wide(
    eia: pd.DataFrame,
    fred: pd.DataFrame,
    yfinance_data: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if not eia.empty:
        frames.append(make_series_wide(eia))

    if not fred.empty:
        frames.append(make_series_wide(fred))

    if not yfinance_data.empty:
        frames.append(make_yfinance_adj_close_wide(yfinance_data))

    if not frames:
        raise ValueError("No oil data available for wide table.")

    wide = frames[0].copy()

    for frame in frames[1:]:
        wide = wide.merge(frame, on="date", how="outer")

    wide["date"] = pd.to_datetime(wide["date"])
    wide = (
        wide
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    return wide


def save_oil_raw_wide(
    df: pd.DataFrame,
    path: Path = OIL_RAW_WIDE_PATH,
) -> None:
    ensure_oil_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved oil raw wide data to: {path}")


# ============================================================
# QUALITY REPORT
# ============================================================

def build_series_quality_report(
    data: pd.DataFrame,
    source_name: str,
    min_rows: int = 200,
) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()

    report = (
        data.groupby("series_id")
        .agg(
            name=("name", "first"),
            source=("source", "first"),
            frequency=("frequency", "first"),
            feature_group=("feature_group", "first"),
            first_period_date=("period_date", "min"),
            last_period_date=("period_date", "max"),
            first_available_date=("date", "min"),
            last_available_date=("date", "max"),
            rows=("date", "count"),
            missing_values=("value", lambda x: x.isna().sum()),
            release_lag_days=("release_lag_days", "first"),
        )
        .reset_index()
    )

    report["source_type"] = source_name

    report["passes_basic_history_filter"] = (
        report["rows"] >= min_rows
    ) & (
        report["missing_values"] == 0
    )

    return report


def build_yfinance_quality_report(
    data: pd.DataFrame,
    min_rows: int = 200,
) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()

    report = (
        data.groupby("ticker")
        .agg(
            name=("name", "first"),
            source=("source", "first"),
            feature_group=("feature_group", "first"),
            first_available_date=("date", "min"),
            last_available_date=("date", "max"),
            rows=("date", "count"),
            missing_values=("adj_close", lambda x: x.isna().sum()),
            average_volume=("volume", "mean"),
        )
        .reset_index()
        .rename(columns={"ticker": "series_id"})
    )

    report["frequency"] = "daily"
    report["source_type"] = "yfinance"
    report["first_period_date"] = report["first_available_date"]
    report["last_period_date"] = report["last_available_date"]
    report["release_lag_days"] = 0

    report["passes_basic_history_filter"] = (
        report["rows"] >= min_rows
    ) & (
        report["missing_values"] == 0
    )

    return report[
        [
            "series_id",
            "name",
            "source",
            "frequency",
            "feature_group",
            "first_period_date",
            "last_period_date",
            "first_available_date",
            "last_available_date",
            "rows",
            "missing_values",
            "release_lag_days",
            "source_type",
            "average_volume",
            "passes_basic_history_filter",
        ]
    ]


def build_oil_data_quality_report(
    eia: pd.DataFrame,
    fred: pd.DataFrame,
    yfinance_data: pd.DataFrame,
) -> pd.DataFrame:
    reports = []

    eia_report = build_series_quality_report(
        eia,
        source_name="EIA",
        min_rows=200,
    )

    fred_report = build_series_quality_report(
        fred,
        source_name="FRED",
        min_rows=200,
    )

    yfinance_report = build_yfinance_quality_report(
        yfinance_data,
        min_rows=200,
    )

    for report in [eia_report, fred_report, yfinance_report]:
        if not report.empty:
            reports.append(report)

    if not reports:
        return pd.DataFrame()

    out = pd.concat(reports, ignore_index=True, sort=False)

    out = out.sort_values(
        ["source_type", "feature_group", "series_id"]
    ).reset_index(drop=True)

    return out


def save_oil_data_quality_report(
    report: pd.DataFrame,
    path: Path = OIL_DATA_QUALITY_REPORT_PATH,
) -> None:
    ensure_oil_data_dirs()
    report.to_csv(path, index=False)
    print(f"Saved oil data quality report to: {path}")


# ============================================================
# PIPELINE
# ============================================================

def run_oil_data_pipeline() -> pd.DataFrame:
    ensure_oil_data_dirs()

    print("\nStarting oil data pipeline...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw dir:      {OIL_RAW_DIR}")
    print(f"Processed:    {OIL_PROCESSED_DIR}")

    eia_raw = download_eia_data()
    eia = clean_eia_data(eia_raw)
    save_eia_data(eia)

    fred_raw = download_fred_data()
    fred = clean_fred_data(fred_raw)
    save_fred_data(fred)

    yfinance_raw = download_oil_yfinance_data()
    yfinance_data = clean_oil_yfinance_data(yfinance_raw)
    save_oil_yfinance_data(yfinance_data)

    wide = build_oil_raw_wide(
        eia=eia,
        fred=fred,
        yfinance_data=yfinance_data,
    )

    save_oil_raw_wide(wide)

    quality_report = build_oil_data_quality_report(
        eia=eia,
        fred=fred,
        yfinance_data=yfinance_data,
    )

    save_oil_data_quality_report(quality_report)

    print("\nOil data pipeline complete.")
    print(f"EIA rows:       {len(eia):,}")
    print(f"FRED rows:      {len(fred):,}")
    print(f"yfinance rows:  {len(yfinance_data):,}")
    print(f"Wide rows:      {len(wide):,}")
    print(f"Start date:     {wide['date'].min().date()}")
    print(f"End date:       {wide['date'].max().date()}")

    print("\nWide columns:")
    print(list(wide.columns))

    print("\nOil data quality:")
    if quality_report.empty:
        print("No quality report rows.")
    else:
        print(quality_report.to_string(index=False))

    return wide


def load_oil_raw_wide(path: Path = OIL_RAW_WIDE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Oil raw wide data not found: {path}. Run oil_data.py first."
        )

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    return df.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    run_oil_data_pipeline()