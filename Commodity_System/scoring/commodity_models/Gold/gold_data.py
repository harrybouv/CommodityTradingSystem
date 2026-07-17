from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    END_DATE,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FORWARD_FILL_LIMIT,
)
# Pull earlier than the backtest start so later rolling z-scores/regimes
# have proper warm-up history before the 2015 live/backtest period.
GOLD_DATA_START_DATE = "2009-01-01"
GOLD_DATA_END_DATE = END_DATE

GOLD_RAW_DIR = RAW_DATA_DIR / "gold"
GOLD_PROCESSED_DIR = PROCESSED_DATA_DIR / "gold"
GOLD_MANUAL_DIR = GOLD_RAW_DIR / "manual"

GOLD_FRED_RAW_PATH = GOLD_RAW_DIR / "gold_fred_raw.csv"
GOLD_YFINANCE_RAW_PATH = GOLD_RAW_DIR / "gold_yfinance_raw.csv"
GOLD_CENTRAL_BANK_STANDARDISED_PATH = GOLD_RAW_DIR / "gold_central_bank_demand_standardised.csv"
GOLD_COT_STANDARDISED_PATH = GOLD_RAW_DIR / "gold_cot_positioning_standardised.csv"
GOLD_DATA_QUALITY_REPORT_PATH = GOLD_RAW_DIR / "gold_data_quality_report.csv"
GOLD_RAW_WIDE_PATH = GOLD_PROCESSED_DIR / "gold_raw_wide.csv"

# Optional manual inputs. These will not exist at first. That is fine.
# The pipeline runs without them and prints a clear message.
GOLD_CENTRAL_BANK_MANUAL_PATH = GOLD_MANUAL_DIR / "gold_central_bank_demand.csv"
GOLD_COT_MANUAL_PATH = GOLD_MANUAL_DIR / "gold_cot_positioning.csv"


# ============================================================
# SERIES DEFINITIONS
# ============================================================

# FRED series are preferred for true macro variables. yfinance proxies are
# acceptable for market prices, but not for real yields, Fed funds, or credit spreads.
GOLD_FRED_SERIES = {
    "DFII10": {
        "name": "10Y TIPS real yield",
        "feature_group": "real_yield",
        "frequency": "daily",
        "description": "Primary gold opportunity-cost input: real yield level and changes.",
    },
    "DTWEXBGS": {
        "name": "Nominal broad US dollar index",
        "feature_group": "usd",
        "frequency": "daily",
        "description": "Primary USD strength input. Gold is usually pressured by USD strength.",
    },
    "VIXCLS": {
        "name": "CBOE VIX",
        "feature_group": "stress",
        "frequency": "daily",
        "description": "Equity volatility / market uncertainty input.",
    },
    "BAMLH0A0HYM2": {
        "name": "ICE BofA US High Yield OAS",
        "feature_group": "stress",
        "frequency": "daily",
        "description": "Credit stress input.",
    },
    "STLFSI4": {
        "name": "St Louis Fed Financial Stress Index",
        "feature_group": "stress",
        "frequency": "weekly",
        "description": "Composite financial stress index.",
    },
    "DGS2": {
        "name": "2Y Treasury yield",
        "feature_group": "policy_rates",
        "frequency": "daily",
        "description": "Market-implied Fed/policy regime proxy.",
    },
    "DGS10": {
        "name": "10Y Treasury yield",
        "feature_group": "policy_rates",
        "frequency": "daily",
        "description": "Nominal-rate and curve diagnostic input.",
    },
    "FEDFUNDS": {
        "name": "Effective federal funds rate",
        "feature_group": "policy_rates",
        "frequency": "monthly",
        "description": "Slow Fed tightening/easing regime input.",
    },
}

# yfinance is used only for market-traded price/index proxies needed by the
# gold overlay. SPY is required for equity drawdown stress. ^MOVE is optional:
# useful if Yahoo provides it, but the pipeline should not die if it disappears.
GOLD_YFINANCE_UNIVERSE = {
    "SPY": {
        "name": "S&P 500 ETF",
        "feature_group": "stress",
        "description": "Equity drawdown / risk appetite input.",
        "required": True,
    },
    "^MOVE": {
        "name": "ICE BofA MOVE Index",
        "feature_group": "stress",
        "description": "Bond-market volatility input. Optional because Yahoo availability can be unstable.",
        "required": False,
    },
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def ensure_gold_data_dirs() -> None:
    GOLD_RAW_DIR.mkdir(parents=True, exist_ok=True)
    GOLD_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    GOLD_MANUAL_DIR.mkdir(parents=True, exist_ok=True)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip()).strip("_").lower()
        for col in out.columns
    ]
    return out


def _filter_date_range(
    df: pd.DataFrame,
    start_date: str | None = GOLD_DATA_START_DATE,
    end_date: str | None = GOLD_DATA_END_DATE,
) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    if start_date is not None:
        out = out[out["date"] >= pd.to_datetime(start_date)].copy()

    if end_date is not None:
        out = out[out["date"] <= pd.to_datetime(end_date)].copy()

    return out


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace(".", np.nan), errors="coerce")


def _find_column(
    columns: list[str],
    required_terms: list[str],
    optional_terms: list[str] | None = None,
) -> str | None:
    optional_terms = optional_terms or []

    for col in columns:
        if all(term in col for term in required_terms):
            if not optional_terms or any(term in col for term in optional_terms):
                return col

    return None


# ============================================================
# FRED DATA
# ============================================================

def download_fred_series(
    series_id: str,
    start_date: str | None = GOLD_DATA_START_DATE,
    end_date: str | None = GOLD_DATA_END_DATE,
) -> pd.DataFrame:
    """
    Download a single FRED series using the public graph CSV endpoint.

    This deliberately avoids a FRED API key for now. If we later need vintages,
    real-time release handling, or bulk metadata, move this to the official API.
    """

    meta = GOLD_FRED_SERIES[series_id]
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    raw = pd.read_csv(url)

    if raw.empty or len(raw.columns) < 2:
        raise ValueError(f"FRED series {series_id} downloaded empty or malformed data.")

    date_col = raw.columns[0]
    value_col = raw.columns[1]

    out = raw[[date_col, value_col]].copy()
    out.columns = ["date", "value"]
    out["date"] = pd.to_datetime(out["date"])
    out["value"] = _safe_numeric(out["value"])

    out = _filter_date_range(out, start_date=start_date, end_date=end_date)

    out["series_id"] = series_id
    out["name"] = meta["name"]
    out["feature_group"] = meta["feature_group"]
    out["frequency"] = meta["frequency"]
    out["description"] = meta["description"]
    out["source"] = "FRED"

    return out[
        [
            "date",
            "series_id",
            "name",
            "feature_group",
            "frequency",
            "source",
            "value",
            "description",
        ]
    ].sort_values("date").reset_index(drop=True)


def download_fred_data(
    series_ids: list[str] | None = None,
    start_date: str | None = GOLD_DATA_START_DATE,
    end_date: str | None = GOLD_DATA_END_DATE,
) -> pd.DataFrame:
    if series_ids is None:
        series_ids = list(GOLD_FRED_SERIES.keys())

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    print(f"Downloading FRED gold macro series: {', '.join(series_ids)}")

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
        raise ValueError("No FRED gold macro data downloaded.")

    if failed:
        print(f"\nFailed FRED series: {failed}")

    return pd.concat(frames, ignore_index=True)


def clean_fred_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["series_id"] = df["series_id"].astype(str).str.upper().str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    duplicate_count = df.duplicated(["date", "series_id"]).sum()
    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count} duplicate FRED date/series rows.")

    # Keep true missing values out of the raw long table. Later feature-building
    # should use as-of logic consciously, not blind filling here.
    df = df.dropna(subset=["value"]).copy()

    return df.sort_values(["series_id", "date"]).reset_index(drop=True)


def save_fred_data(df: pd.DataFrame, path: Path = GOLD_FRED_RAW_PATH) -> None:
    ensure_gold_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved FRED gold macro data to: {path}")


def make_fred_wide(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.pivot(index="date", columns="series_id", values="value")
        .sort_index()
        .reset_index()
    )
    return out


# ============================================================
# YFINANCE MARKET DATA
# ============================================================

def download_gold_yfinance_data(
    tickers: list[str] | None = None,
    start_date: str | None = GOLD_DATA_START_DATE,
    end_date: str | None = GOLD_DATA_END_DATE,
) -> pd.DataFrame:
    if tickers is None:
        tickers = list(GOLD_YFINANCE_UNIVERSE.keys())

    tickers = [str(ticker).upper().strip() for ticker in tickers]

    print(f"Downloading yfinance gold overlay data for: {', '.join(tickers)}")

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
        raise ValueError("No yfinance gold overlay data downloaded.")

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for ticker in tickers:
        meta = GOLD_YFINANCE_UNIVERSE.get(ticker, {})
        required = bool(meta.get("required", False))

        try:
            if isinstance(raw.columns, pd.MultiIndex):
                available_top_level = set(raw.columns.get_level_values(0))

                if ticker not in available_top_level:
                    raise ValueError(f"{ticker} not found in downloaded MultiIndex columns.")

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
        raise ValueError("No valid yfinance gold overlay ticker data processed.")

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


def clean_gold_yfinance_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    duplicate_count = df.duplicated(["date", "ticker"]).sum()
    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count} duplicate yfinance date/ticker rows.")

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


def save_gold_yfinance_data(df: pd.DataFrame, path: Path = GOLD_YFINANCE_RAW_PATH) -> None:
    ensure_gold_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved yfinance gold overlay data to: {path}")


def make_yfinance_adj_close_wide(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.pivot(index="date", columns="ticker", values="adj_close")
        .sort_index()
    )

    out.columns = [f"yf_{col}_adj_close" for col in out.columns]

    return out.reset_index()


# ============================================================
# OPTIONAL MANUAL CENTRAL BANK DEMAND DATA
# ============================================================

def load_optional_central_bank_demand(
    path: Path = GOLD_CENTRAL_BANK_MANUAL_PATH,
) -> pd.DataFrame:
    """
    Optional manual WGC/IMF central bank demand data.

    Expected simple CSV format:
        date, net_purchases_tonnes

    Accepts some loose aliases, but do not rely on magic. Keep the manual CSV clean.
    """

    if not path.exists():
        print(
            "No manual central-bank demand file found. "
            f"Expected optional file at: {path}"
        )
        return pd.DataFrame(
            columns=[
                "date",
                "series_id",
                "name",
                "feature_group",
                "source",
                "value",
                "description",
            ]
        )

    raw = _normalise_columns(pd.read_csv(path))

    date_col = _find_column(list(raw.columns), ["date"]) or _find_column(list(raw.columns), ["month"])

    value_col = None
    for candidate in [
        "net_purchases_tonnes",
        "net_purchase_tonnes",
        "net_purchases",
        "net_purchases_sales",
        "net",
        "value",
        "tonnes",
    ]:
        if candidate in raw.columns:
            value_col = candidate
            break

    if date_col is None or value_col is None:
        raise ValueError(
            "Central-bank manual file must contain a date/month column and a net purchases tonnes column."
        )

    out = raw[[date_col, value_col]].copy()
    out.columns = ["date", "value"]
    out["date"] = pd.to_datetime(out["date"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"]).copy()

    out = _filter_date_range(out)

    out["series_id"] = "CENTRAL_BANK_NET_PURCHASES_TONNES"
    out["name"] = "Central bank net gold purchases"
    out["feature_group"] = "central_bank_demand"
    out["source"] = "manual_wgc_or_imf"
    out["description"] = "Manual WGC/IMF central-bank net purchases in tonnes. Lag before scoring."

    out = out[
        [
            "date",
            "series_id",
            "name",
            "feature_group",
            "source",
            "value",
            "description",
        ]
    ].sort_values("date").reset_index(drop=True)

    ensure_gold_data_dirs()
    out.to_csv(GOLD_CENTRAL_BANK_STANDARDISED_PATH, index=False)
    print(f"Saved standardised central-bank demand data to: {GOLD_CENTRAL_BANK_STANDARDISED_PATH}")

    return out


# ============================================================
# OPTIONAL MANUAL CFTC COT POSITIONING DATA
# ============================================================

def load_optional_gold_cot_positioning(
    path: Path = GOLD_COT_MANUAL_PATH,
) -> pd.DataFrame:
    """
    Optional manual CFTC disaggregated futures-and-options combined data.

    Preferred source: CFTC disaggregated futures-and-options combined historical file.
    Save the relevant CSV as:
        data/raw/gold/manual/gold_cot_positioning.csv

    This loader tries to standardise the usual CFTC columns:
        market_and_exchange_names
        report_date_as_yyyy_mm_dd
        open_interest_all
        m_money_positions_long_all
        m_money_positions_short_all
    """

    if not path.exists():
        print(
            "No manual CFTC gold positioning file found. "
            f"Expected optional file at: {path}"
        )
        return pd.DataFrame(
            columns=[
                "date",
                "series_id",
                "name",
                "feature_group",
                "source",
                "value",
                "description",
            ]
        )

    raw = _normalise_columns(pd.read_csv(path))
    cols = list(raw.columns)

    market_col = _find_column(cols, ["market", "exchange", "names"]) or _find_column(cols, ["market"])
    date_col = (
        _find_column(cols, ["report", "date", "yyyy"])
        or _find_column(cols, ["as", "of", "date"])
        or _find_column(cols, ["date"])
    )

    open_interest_col = _find_column(cols, ["open", "interest", "all"])
    mm_long_col = _find_column(cols, ["m_money", "positions", "long", "all"]) or _find_column(cols, ["managed", "money", "long"])
    mm_short_col = _find_column(cols, ["m_money", "positions", "short", "all"]) or _find_column(cols, ["managed", "money", "short"])

    required = {
        "market_col": market_col,
        "date_col": date_col,
        "open_interest_col": open_interest_col,
        "mm_long_col": mm_long_col,
        "mm_short_col": mm_short_col,
    }

    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise ValueError(
            "COT manual file is missing required CFTC-style columns: "
            f"{missing}. Available columns: {cols[:30]}..."
        )

    df = raw.copy()
    df = df[df[market_col].astype(str).str.contains("GOLD", case=False, na=False)].copy()

    if df.empty:
        raise ValueError("COT manual file loaded but no GOLD market rows were found.")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["market"] = df[market_col].astype(str)
    out["open_interest"] = pd.to_numeric(df[open_interest_col], errors="coerce")
    out["managed_money_long"] = pd.to_numeric(df[mm_long_col], errors="coerce")
    out["managed_money_short"] = pd.to_numeric(df[mm_short_col], errors="coerce")
    out["managed_money_net"] = out["managed_money_long"] - out["managed_money_short"]
    out["managed_money_net_pct_open_interest"] = np.where(
        out["open_interest"] > 0,
        out["managed_money_net"] / out["open_interest"],
        np.nan,
    )

    out = out.dropna(subset=["date", "managed_money_net_pct_open_interest"]).copy()
    out = _filter_date_range(out)
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    long = out.melt(
        id_vars=["date"],
        value_vars=[
            "open_interest",
            "managed_money_long",
            "managed_money_short",
            "managed_money_net",
            "managed_money_net_pct_open_interest",
        ],
        var_name="series_id",
        value_name="value",
    )

    long["name"] = long["series_id"]
    long["feature_group"] = "positioning_crowding"
    long["source"] = "manual_cftc_cot"
    long["description"] = "Manual CFTC gold COT positioning data. Lag before scoring."

    long = long[
        [
            "date",
            "series_id",
            "name",
            "feature_group",
            "source",
            "value",
            "description",
        ]
    ].sort_values(["series_id", "date"]).reset_index(drop=True)

    ensure_gold_data_dirs()
    long.to_csv(GOLD_COT_STANDARDISED_PATH, index=False)
    print(f"Saved standardised CFTC gold positioning data to: {GOLD_COT_STANDARDISED_PATH}")

    return long


# ============================================================
# QUALITY REPORT AND WIDE MERGE
# ============================================================

def build_gold_data_quality_report(
    fred: pd.DataFrame,
    yfinance_data: pd.DataFrame,
    central_bank: pd.DataFrame,
    cot: pd.DataFrame,
) -> pd.DataFrame:
    reports: list[pd.DataFrame] = []

    if not fred.empty:
        fred_report = (
            fred.groupby(["source", "feature_group", "series_id", "name"])
            .agg(
                start_date=("date", "min"),
                end_date=("date", "max"),
                rows=("date", "count"),
                missing_value=("value", lambda x: x.isna().sum()),
            )
            .reset_index()
        )
        reports.append(fred_report)

    if not yfinance_data.empty:
        yf_report = (
            yfinance_data.groupby(["source", "feature_group", "ticker", "name"])
            .agg(
                start_date=("date", "min"),
                end_date=("date", "max"),
                rows=("date", "count"),
                missing_value=("adj_close", lambda x: x.isna().sum()),
            )
            .reset_index()
            .rename(columns={"ticker": "series_id"})
        )
        reports.append(yf_report)

    for optional_name, optional_df in [
        ("central_bank_demand", central_bank),
        ("cot_positioning", cot),
    ]:
        if optional_df.empty:
            reports.append(
                pd.DataFrame(
                    {
                        "source": ["manual_missing"],
                        "feature_group": [optional_name],
                        "series_id": [optional_name],
                        "name": [optional_name],
                        "start_date": [pd.NaT],
                        "end_date": [pd.NaT],
                        "rows": [0],
                        "missing_value": [np.nan],
                    }
                )
            )
        else:
            opt_report = (
                optional_df.groupby(["source", "feature_group", "series_id", "name"])
                .agg(
                    start_date=("date", "min"),
                    end_date=("date", "max"),
                    rows=("date", "count"),
                    missing_value=("value", lambda x: x.isna().sum()),
                )
                .reset_index()
            )
            reports.append(opt_report)

    out = pd.concat(reports, ignore_index=True)
    out = out.sort_values(["feature_group", "source", "series_id"]).reset_index(drop=True)

    return out


def build_gold_raw_wide(
    fred: pd.DataFrame,
    yfinance_data: pd.DataFrame,
    central_bank: pd.DataFrame,
    cot: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if not fred.empty:
        frames.append(make_fred_wide(fred).set_index("date"))

    if not yfinance_data.empty:
        frames.append(make_yfinance_adj_close_wide(yfinance_data).set_index("date"))

    for optional_df in [central_bank, cot]:
        if not optional_df.empty:
            optional_wide = (
                optional_df.pivot(index="date", columns="series_id", values="value")
                .sort_index()
            )
            frames.append(optional_wide)

    if not frames:
        raise ValueError("No gold data available to build wide table.")

    wide = pd.concat(frames, axis=1).sort_index()

    # Do not forward-fill here. The feature layer should use explicit as-of and lag logic,
    # especially for weekly/monthly CFTC, Fed funds and central-bank data.
    wide = wide.reset_index()
    wide["date"] = pd.to_datetime(wide["date"])

    return wide


def save_gold_data_quality_report(df: pd.DataFrame, path: Path = GOLD_DATA_QUALITY_REPORT_PATH) -> None:
    ensure_gold_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved gold data quality report to: {path}")


def save_gold_raw_wide(df: pd.DataFrame, path: Path = GOLD_RAW_WIDE_PATH) -> None:
    ensure_gold_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved merged gold raw wide data to: {path}")


# ============================================================
# PIPELINE
# ============================================================

def run_gold_data_pipeline() -> pd.DataFrame:
    ensure_gold_data_dirs()

    fred_raw = download_fred_data()
    fred_clean = clean_fred_data(fred_raw)
    save_fred_data(fred_clean)

    yf_raw = download_gold_yfinance_data()
    yf_clean = clean_gold_yfinance_data(yf_raw)
    save_gold_yfinance_data(yf_clean)

    central_bank = load_optional_central_bank_demand()
    cot = load_optional_gold_cot_positioning()

    quality_report = build_gold_data_quality_report(
        fred=fred_clean,
        yfinance_data=yf_clean,
        central_bank=central_bank,
        cot=cot,
    )
    save_gold_data_quality_report(quality_report)

    wide = build_gold_raw_wide(
        fred=fred_clean,
        yfinance_data=yf_clean,
        central_bank=central_bank,
        cot=cot,
    )
    save_gold_raw_wide(wide)

    print("\nGold data ingestion complete.")
    print(f"FRED rows:       {len(fred_clean):,}")
    print(f"yfinance rows:   {len(yf_clean):,}")
    print(f"Central bank rows: {len(central_bank):,}")
    print(f"COT rows:          {len(cot):,}")
    print(f"Wide rows:       {len(wide):,}")
    print(f"Start date:      {wide['date'].min().date()}")
    print(f"End date:        {wide['date'].max().date()}")
    print("\nWide columns:")
    print(list(wide.columns))

    return wide


if __name__ == "__main__":
    run_gold_data_pipeline()
