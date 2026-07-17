# macro_data.py

from __future__ import annotations

import pandas as pd
import yfinance as yf

from config import (
    START_DATE,
    END_DATE,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FORWARD_FILL_LIMIT,
    MIN_PRICE_HISTORY_DAYS,
)

try:
    from config import (
        MACRO_UNIVERSE,
        MACRO_TICKERS,
        MACRO_PRICE_DATA_PATH,
        MACRO_DATA_QUALITY_REPORT_PATH,
        MIN_MACRO_PRICE_HISTORY_DAYS,
    )
except ImportError:
    # Fallbacks so this script can run before config.py is fully updated.
    MACRO_UNIVERSE = {
        "UUP": {
            "name": "US Dollar Bullish ETF",
            "macro_role": "usd",
            "description": "Tradable USD strength proxy",
        },
        "^TNX": {
            "name": "10-Year Treasury Yield",
            "macro_role": "rates",
            "description": "US 10-year yield proxy",
        },
        "TIP": {
            "name": "TIPS ETF",
            "macro_role": "inflation",
            "description": "Inflation-protected Treasuries proxy",
        },
        "IEF": {
            "name": "7-10 Year Treasury ETF",
            "macro_role": "nominal_rates",
            "description": "Nominal Treasury duration proxy used with TIP",
        },
        "SPY": {
            "name": "S&P 500 ETF",
            "macro_role": "growth_risk",
            "description": "Equity risk appetite proxy",
        },
        "^VIX": {
            "name": "VIX Index",
            "macro_role": "stress",
            "description": "Equity volatility / stress proxy",
        },
        "DBC": {
            "name": "Broad Commodity ETF",
            "macro_role": "broad_commodities",
            "description": "Broad commodity trend proxy",
        },
    }

    MACRO_TICKERS = list(MACRO_UNIVERSE.keys())
    MACRO_PRICE_DATA_PATH = RAW_DATA_DIR / "macro_prices.csv"
    MACRO_DATA_QUALITY_REPORT_PATH = RAW_DATA_DIR / "macro_data_quality_report.csv"
    MIN_MACRO_PRICE_HISTORY_DAYS = MIN_PRICE_HISTORY_DAYS


# ============================================================
# DIRECTORIES
# ============================================================

def ensure_macro_data_dirs() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DOWNLOAD
# ============================================================

def _normalise_ticker(ticker: str) -> str:
    return str(ticker).upper().strip()


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in out.columns
    ]

    return out


def download_macro_ohlcv_data(
    tickers: list[str] | None = None,
    start_date: str = START_DATE,
    end_date: str | None = END_DATE,
) -> pd.DataFrame:
    """
    Download raw OHLCV-style macro proxy data from yfinance.

    This mirrors data.py, but is intentionally separate because macro series
    include non-tradable index proxies like ^TNX and ^VIX, where volume is not
    a useful quality filter.
    """

    if tickers is None:
        tickers = MACRO_TICKERS

    tickers = [_normalise_ticker(ticker) for ticker in tickers]

    print(f"Downloading macro data for: {', '.join(tickers)}")

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
        raise ValueError("No macro data downloaded.")

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for ticker in tickers:
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

            # yfinance usually gives "adj_close" when auto_adjust=False.
            # Some macro/index series may not. For regime scoring, close is
            # acceptable as a fallback.
            if "adj_close" not in df.columns and "close" in df.columns:
                df["adj_close"] = df["close"]

            # Some index-like macro series may have missing/no volume.
            if "volume" not in df.columns:
                df["volume"] = 0.0

            # If OHLC columns are partially missing but close exists, use close
            # as a safe placeholder. The scoring layer mainly uses adj_close.
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

            meta = MACRO_UNIVERSE.get(ticker, {})
            df["name"] = meta.get("name", ticker)
            df["macro_role"] = meta.get("macro_role", "unknown")
            df["description"] = meta.get("description", "")

            frames.append(df)

        except Exception as e:
            failed.append(ticker)
            print(f"Failed to process {ticker}: {e}")

    if not frames:
        raise ValueError("No valid macro ticker data processed.")

    if failed:
        print(f"\nFailed macro tickers: {failed}")

    out = pd.concat(frames, ignore_index=True)

    out = out[
        [
            "date",
            "ticker",
            "name",
            "macro_role",
            "description",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ]
    ].copy()

    return out


# ============================================================
# QUALITY REPORT
# ============================================================

def build_macro_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build macro data quality report.

    Unlike commodity data, this does not require volume because ^TNX and ^VIX
    are index-style macro proxies.
    """

    report = (
        df.groupby("ticker")
        .agg(
            name=("name", "first"),
            macro_role=("macro_role", "first"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            rows=("date", "count"),
            average_volume=("volume", "mean"),
            missing_adj_close=("adj_close", lambda x: x.isna().sum()),
            missing_close=("close", lambda x: x.isna().sum()),
            missing_volume=("volume", lambda x: x.isna().sum()),
        )
        .reset_index()
    )

    report["history_days"] = report["rows"]

    report["passes_history_filter"] = (
        report["history_days"] >= MIN_MACRO_PRICE_HISTORY_DAYS
    )

    report["passes_price_filter"] = (
        report["missing_adj_close"] == 0
    )

    report["keep"] = (
        report["passes_history_filter"]
        & report["passes_price_filter"]
    )

    def removal_reason(row: pd.Series) -> str:
        reasons: list[str] = []

        if not row["passes_history_filter"]:
            reasons.append("insufficient_history")

        if not row["passes_price_filter"]:
            reasons.append("missing_adj_close")

        if not reasons:
            return "kept"

        return ";".join(reasons)

    report["removal_reason"] = report.apply(removal_reason, axis=1)

    return report.sort_values("ticker").reset_index(drop=True)


# ============================================================
# CLEANING
# ============================================================

def clean_macro_ohlcv_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["name"] = df["name"].astype(str)
    df["macro_role"] = df["macro_role"].astype(str)
    df["description"] = df["description"].astype(str)

    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    duplicate_count = df.duplicated(["date", "ticker"]).sum()
    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate macro date/ticker rows."
        )

    # Price must be positive. Volume is not filtered because macro indices
    # frequently have zero or missing volume.
    df = df[df["adj_close"] > 0].copy()

    df["volume"] = df["volume"].fillna(0.0)
    df = df[df["volume"] >= 0].copy()

    df[numeric_cols] = (
        df.groupby("ticker")[numeric_cols]
        .ffill(limit=FORWARD_FILL_LIMIT)
    )

    df = df.dropna(subset=["adj_close"]).copy()

    quality_report = build_macro_data_quality_report(df)

    ensure_macro_data_dirs()
    quality_report.to_csv(MACRO_DATA_QUALITY_REPORT_PATH, index=False)
    print(f"Saved macro data quality report to: {MACRO_DATA_QUALITY_REPORT_PATH}")

    kept_tickers = quality_report.loc[
        quality_report["keep"],
        "ticker",
    ].tolist()

    removed = quality_report.loc[
        ~quality_report["keep"],
        ["ticker", "macro_role", "removal_reason"],
    ]

    if not removed.empty:
        print("\nRemoved macro tickers:")
        print(removed.to_string(index=False))

    df = df[df["ticker"].isin(kept_tickers)].copy()

    if df.empty:
        raise ValueError("All macro tickers were removed by data quality filters.")

    df["daily_return"] = (
        df.groupby("ticker")["adj_close"]
        .pct_change()
    )

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    return df


# ============================================================
# SAVE / LOAD HELPERS
# ============================================================

def save_macro_price_data(
    df: pd.DataFrame,
    path=MACRO_PRICE_DATA_PATH,
) -> None:
    ensure_macro_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved macro price data to: {path}")


def load_macro_price_data(
    path=MACRO_PRICE_DATA_PATH,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def make_macro_price_matrix(
    df: pd.DataFrame,
    value_col: str = "adj_close",
) -> pd.DataFrame:
    matrix = (
        df.pivot(index="date", columns="ticker", values=value_col)
        .sort_index()
    )

    return matrix


def make_macro_return_matrix(df: pd.DataFrame) -> pd.DataFrame:
    prices = make_macro_price_matrix(df, value_col="adj_close")

    # Do not blindly fill missing macro returns. Missing values can reveal
    # availability issues and should be handled consciously in macro_features.py.
    returns = prices.pct_change(fill_method=None)

    return returns


# ============================================================
# PIPELINE
# ============================================================

def run_macro_data_pipeline() -> pd.DataFrame:
    ensure_macro_data_dirs()

    raw = download_macro_ohlcv_data()
    clean = clean_macro_ohlcv_data(raw)

    save_macro_price_data(clean)

    print("\nMacro data ingestion complete.")
    print(f"Rows: {len(clean):,}")
    print(f"Macro tickers kept: {sorted(clean['ticker'].unique())}")
    print(f"Start date: {clean['date'].min().date()}")
    print(f"End date: {clean['date'].max().date()}")

    requested = set(_normalise_ticker(ticker) for ticker in MACRO_TICKERS)
    kept = set(clean["ticker"].unique())
    missing_after_cleaning = sorted(requested - kept)

    if missing_after_cleaning:
        print(f"Requested but unavailable after cleaning: {missing_after_cleaning}")

    role_summary = (
        clean.groupby(["ticker", "macro_role"])
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            rows=("date", "count"),
        )
        .reset_index()
        .sort_values("ticker")
    )

    print("\nMacro series summary:")
    print(role_summary.to_string(index=False))

    return clean


if __name__ == "__main__":
    run_macro_data_pipeline()