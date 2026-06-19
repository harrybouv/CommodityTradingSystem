import pandas as pd
import yfinance as yf

from config import (
    TICKERS,
    START_DATE,
    END_DATE,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PRICE_DATA_PATH,
    FORWARD_FILL_LIMIT,
    MIN_AVERAGE_DAILY_VOLUME,
    MIN_PRICE_HISTORY_DAYS,
)


DATA_QUALITY_REPORT_PATH = RAW_DATA_DIR / "data_quality_report.csv"


def ensure_data_dirs() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_ohlcv_data(
    tickers: list[str] | None = None,
    start_date: str = START_DATE,
    end_date: str | None = END_DATE,
) -> pd.DataFrame:
    if tickers is None:
        tickers = TICKERS

    tickers = [ticker.upper().strip() for ticker in tickers]

    print(f"Downloading OHLCV data for: {', '.join(tickers)}")

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
        raise ValueError("No price data downloaded.")

    frames = []
    failed = []

    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                available_top_level = set(raw.columns.get_level_values(0))

                if ticker not in available_top_level:
                    raise ValueError(f"{ticker} not found in downloaded MultiIndex columns.")

                df = raw[ticker].copy()
            else:
                df = raw.copy()

            df = df.reset_index()

            df.columns = [
                str(col).strip().lower().replace(" ", "_")
                for col in df.columns
            ]

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

            frames.append(df)

        except Exception as e:
            failed.append(ticker)
            print(f"Failed to process {ticker}: {e}")

    if not frames:
        raise ValueError("No valid ticker data processed.")

    if failed:
        print(f"\nFailed tickers: {failed}")

    out = pd.concat(frames, ignore_index=True)

    out = out[
        ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    ].copy()

    return out


def build_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    report = (
        df.groupby("ticker")
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            rows=("date", "count"),
            average_volume=("volume", "mean"),
            missing_adj_close=("adj_close", lambda x: x.isna().sum()),
            missing_volume=("volume", lambda x: x.isna().sum()),
        )
        .reset_index()
    )

    report["history_days"] = report["rows"]

    report["passes_history_filter"] = (
        report["history_days"] >= MIN_PRICE_HISTORY_DAYS
    )

    report["passes_volume_filter"] = (
        report["average_volume"] >= MIN_AVERAGE_DAILY_VOLUME
    )

    report["keep"] = (
        report["passes_history_filter"]
        & report["passes_volume_filter"]
    )

    def removal_reason(row):
        reasons = []

        if not row["passes_history_filter"]:
            reasons.append("insufficient_history")

        if not row["passes_volume_filter"]:
            reasons.append("low_average_volume")

        if not reasons:
            return "kept"

        return ";".join(reasons)

    report["removal_reason"] = report.apply(removal_reason, axis=1)

    return report.sort_values("ticker").reset_index(drop=True)


def clean_ohlcv_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    duplicate_count = df.duplicated(["date", "ticker"]).sum()
    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count} duplicate date/ticker rows.")

    df = df[df["adj_close"] > 0].copy()
    df = df[df["volume"] >= 0].copy()

    df[numeric_cols] = (
        df.groupby("ticker")[numeric_cols]
        .ffill(limit=FORWARD_FILL_LIMIT)
    )

    df = df.dropna(subset=["adj_close", "volume"]).copy()

    quality_report = build_data_quality_report(df)

    ensure_data_dirs()
    quality_report.to_csv(DATA_QUALITY_REPORT_PATH, index=False)
    print(f"Saved data quality report to: {DATA_QUALITY_REPORT_PATH}")

    kept_tickers = quality_report.loc[
        quality_report["keep"],
        "ticker",
    ].tolist()

    removed = quality_report.loc[
        ~quality_report["keep"],
        ["ticker", "removal_reason"],
    ]

    if not removed.empty:
        print("\nRemoved tickers:")
        print(removed.to_string(index=False))

    df = df[df["ticker"].isin(kept_tickers)].copy()

    if df.empty:
        raise ValueError("All tickers were removed by data quality filters.")

    df["daily_return"] = (
        df.groupby("ticker")["adj_close"]
        .pct_change()
    )

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    return df


def save_price_data(df: pd.DataFrame, path=PRICE_DATA_PATH) -> None:
    ensure_data_dirs()
    df.to_csv(path, index=False)
    print(f"Saved price data to: {path}")


def load_price_data(path=PRICE_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def make_price_matrix(
    df: pd.DataFrame,
    value_col: str = "adj_close",
) -> pd.DataFrame:
    matrix = (
        df.pivot(index="date", columns="ticker", values=value_col)
        .sort_index()
    )

    return matrix


def make_return_matrix(df: pd.DataFrame) -> pd.DataFrame:
    prices = make_price_matrix(df, value_col="adj_close")

    # Important: do not blindly fill missing asset returns here.
    # Missing values contain useful information about asset availability.
    returns = prices.pct_change(fill_method=None)

    return returns


def run_data_pipeline() -> pd.DataFrame:
    ensure_data_dirs()

    raw = download_ohlcv_data()
    clean = clean_ohlcv_data(raw)

    save_price_data(clean)

    print("\nData ingestion complete.")
    print(f"Rows: {len(clean):,}")
    print(f"Tickers kept: {sorted(clean['ticker'].unique())}")
    print(f"Start date: {clean['date'].min().date()}")
    print(f"End date: {clean['date'].max().date()}")

    requested = set(TICKERS)
    kept = set(clean["ticker"].unique())
    missing_after_cleaning = sorted(requested - kept)

    if missing_after_cleaning:
        print(f"Requested but unavailable after cleaning: {missing_after_cleaning}")

    return clean


if __name__ == "__main__":
    run_data_pipeline()