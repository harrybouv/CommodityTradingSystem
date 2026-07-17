from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# DIRECT-RUN PATH SETUP
# ============================================================

THIS_FILE = Path(__file__).resolve()
COMMODITY_ROOT = THIS_FILE.parents[3]

if str(COMMODITY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMODITY_ROOT))


from config import (
    PROCESSED_DATA_DIR,
    PRICE_DATA_PATH,
)

try:
    from config import MACRO_PRICE_DATA_PATH
except ImportError:
    MACRO_PRICE_DATA_PATH = PROCESSED_DATA_DIR.parent / "raw" / "macro_prices.csv"


# ============================================================
# PATHS
# ============================================================

COPPER_PROCESSED_DIR = PROCESSED_DATA_DIR / "copper"

COPPER_RAW_WIDE_PATH = COPPER_PROCESSED_DIR / "copper_raw_wide.csv"
COPPER_FEATURES_DAILY_PATH = COPPER_PROCESSED_DIR / "copper_features_daily.csv"
COPPER_FEATURES_MONTHLY_PATH = COPPER_PROCESSED_DIR / "copper_features_monthly.csv"


# ============================================================
# SETTINGS
# ============================================================

TRADING_DAYS_PER_MONTH = 21
LOOKBACK_3M = 63
LOOKBACK_6M = 126
LOOKBACK_1Y = 252
LOOKBACK_3Y = 756

LOOKBACK_3M_MONTHLY = 3
LOOKBACK_1Y_MONTHLY = 12
LOOKBACK_3Y_MONTHLY = 36

FAST_MA = 50
SLOW_MA = 200

MONTHLY_ASOF_TOLERANCE_DAYS = 45
DAILY_ASOF_TOLERANCE_DAYS = 10

REQUIRED_COMMODITY_TICKERS = ["CPER", "USO"]
REQUIRED_MACRO_TICKERS = ["UUP", "DBC", "SPY", "^VIX"]

CHINA_CLI_COL = "CHNLOLITOAASTSAM"
CHINA_ELECTRICITY_COL = "EMBER_CHINA_ELECTRICITY_DEMAND_TWH"


# ============================================================
# HELPERS
# ============================================================

def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _clip01(s: pd.Series) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)


def _neutral_fill(s: pd.Series, neutral: float = 0.50) -> pd.Series:
    return _clip01(s.fillna(neutral))


def _return(s: pd.Series, periods: int) -> pd.Series:
    return _safe_numeric(s).pct_change(periods=periods, fill_method=None)


def _change(s: pd.Series, periods: int) -> pd.Series:
    return _safe_numeric(s).diff(periods)


def _rolling_z_score(
    s: pd.Series,
    window: int,
    min_periods: int,
) -> pd.Series:
    x = _safe_numeric(s)

    mean = x.rolling(
        window=window,
        min_periods=min_periods,
    ).mean()

    std = x.rolling(
        window=window,
        min_periods=min_periods,
    ).std(ddof=0)

    z = (x - mean) / std.replace(0.0, np.nan)

    return z.replace([np.inf, -np.inf], np.nan)


def _rolling_percentile(
    s: pd.Series,
    window: int,
    min_periods: int,
) -> pd.Series:
    x = _safe_numeric(s)

    def percentile_rank(values: np.ndarray) -> float:
        values = values[~np.isnan(values)]

        if len(values) < min_periods:
            return np.nan

        latest = values[-1]
        return float((values <= latest).mean())

    return x.rolling(
        window=window,
        min_periods=min_periods,
    ).apply(percentile_rank, raw=True)


def _score_high_is_good(
    s: pd.Series,
    window: int = LOOKBACK_3Y,
    min_periods: int = LOOKBACK_1Y,
) -> pd.Series:
    return _clip01(_rolling_percentile(s, window=window, min_periods=min_periods))


def _score_low_is_good(
    s: pd.Series,
    window: int = LOOKBACK_3Y,
    min_periods: int = LOOKBACK_1Y,
) -> pd.Series:
    return _clip01(1.0 - _rolling_percentile(s, window=window, min_periods=min_periods))


def _trend_score(price: pd.Series) -> pd.Series:
    price = _safe_numeric(price)

    fast_ma = price.rolling(
        window=FAST_MA,
        min_periods=FAST_MA // 2,
    ).mean()

    slow_ma = price.rolling(
        window=SLOW_MA,
        min_periods=SLOW_MA // 2,
    ).mean()

    above_slow = (price > slow_ma).astype(float)
    fast_above_slow = (fast_ma > slow_ma).astype(float)

    score = 0.60 * above_slow + 0.40 * fast_above_slow

    return _neutral_fill(score)


def _validate_not_constant(df: pd.DataFrame, columns: list[str]) -> None:
    print("\nFeature variation check:")

    for col in columns:
        if col not in df.columns:
            print(f"{col}: MISSING")
            continue

        non_na = df[col].dropna()
        unique_count = non_na.nunique()

        if unique_count <= 1:
            print(f"{col}: WARNING - constant / no variation")
        else:
            print(
                f"{col}: OK | min={non_na.min():.3f}, "
                f"mean={non_na.mean():.3f}, max={non_na.max():.3f}"
            )


# ============================================================
# LOADERS
# ============================================================

def load_commodity_price_matrix(path: Path = PRICE_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Commodity price data not found: {path}. Run data.py first."
        )

    prices = pd.read_csv(path)

    required_cols = ["date", "ticker", "adj_close"]
    missing = [col for col in required_cols if col not in prices.columns]

    if missing:
        raise ValueError(
            f"Commodity price data missing required columns: {missing}. "
            f"Available columns: {list(prices.columns)}"
        )

    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")

    matrix = (
        prices
        .pivot(index="date", columns="ticker", values="adj_close")
        .sort_index()
    )

    missing_tickers = [
        ticker for ticker in REQUIRED_COMMODITY_TICKERS
        if ticker not in matrix.columns
    ]

    if missing_tickers:
        raise ValueError(
            f"Copper feature pipeline requires {REQUIRED_COMMODITY_TICKERS}, "
            f"but missing: {missing_tickers}. "
            f"Available tickers: {list(matrix.columns)}"
        )

    return matrix[REQUIRED_COMMODITY_TICKERS].copy()


def load_macro_price_matrix(path: Path = MACRO_PRICE_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Macro price data not found: {path}. Run macro_data.py first."
        )

    prices = pd.read_csv(path)

    required_cols = ["date", "ticker", "adj_close"]
    missing = [col for col in required_cols if col not in prices.columns]

    if missing:
        raise ValueError(
            f"Macro price data missing required columns: {missing}. "
            f"Available columns: {list(prices.columns)}"
        )

    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")

    matrix = (
        prices
        .pivot(index="date", columns="ticker", values="adj_close")
        .sort_index()
    )

    missing_tickers = [
        ticker for ticker in REQUIRED_MACRO_TICKERS
        if ticker not in matrix.columns
    ]

    if missing_tickers:
        raise ValueError(
            f"Copper feature pipeline requires macro tickers {REQUIRED_MACRO_TICKERS}, "
            f"but missing: {missing_tickers}. "
            f"Available tickers: {list(matrix.columns)}"
        )

    return matrix[REQUIRED_MACRO_TICKERS].copy()


def load_copper_raw_wide(path: Path = COPPER_RAW_WIDE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Copper raw wide file not found: {path}. "
            "Run scoring/commodity_models/Copper/copper_data.py first."
        )

    raw = pd.read_csv(path)

    if "date" not in raw.columns:
        raise ValueError(f"Copper raw wide file missing date column: {path}")

    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    for col in raw.columns:
        if col != "date":
            raw[col] = _safe_numeric(raw[col])

    missing = [
        col for col in [CHINA_CLI_COL, CHINA_ELECTRICITY_COL]
        if col not in raw.columns
    ]

    if missing:
        raise ValueError(
            f"Copper raw wide missing required columns: {missing}. "
            f"Available columns: {list(raw.columns)}"
        )

    return raw


# ============================================================
# CHINA FEATURES
# ============================================================

def _build_monthly_china_electricity_features(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw[["date", CHINA_ELECTRICITY_COL]].dropna().copy()
    data = data.sort_values("date").reset_index(drop=True)

    if data.empty:
        raise ValueError("No China electricity demand data available.")

    demand = _safe_numeric(data[CHINA_ELECTRICITY_COL])

    out = pd.DataFrame({"date": data["date"]})
    out["china_electricity_demand_twh"] = demand.values
    out["china_electricity_yoy"] = demand.pct_change(12, fill_method=None).values
    out["china_electricity_yoy_3m_avg"] = (
        pd.Series(out["china_electricity_yoy"])
        .rolling(window=3, min_periods=2)
        .mean()
        .values
    )
    out["china_electricity_yoy_change_3m"] = (
        pd.Series(out["china_electricity_yoy_3m_avg"])
        .diff(3)
        .values
    )

    yoy_level_score = _score_high_is_good(
        pd.Series(out["china_electricity_yoy_3m_avg"]),
        window=LOOKBACK_3Y_MONTHLY,
        min_periods=LOOKBACK_1Y_MONTHLY,
    )

    yoy_acceleration_score = _score_high_is_good(
        pd.Series(out["china_electricity_yoy_change_3m"]),
        window=LOOKBACK_3Y_MONTHLY,
        min_periods=LOOKBACK_1Y_MONTHLY,
    )

    out["copper_china_electricity_score"] = _neutral_fill(
        0.70 * yoy_level_score
        + 0.30 * yoy_acceleration_score
    ).values

    return out


def _build_monthly_china_cli_features(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw[["date", CHINA_CLI_COL]].dropna().copy()
    data = data.sort_values("date").reset_index(drop=True)

    if data.empty:
        raise ValueError("No China/OECD CLI data available.")

    cli = _safe_numeric(data[CHINA_CLI_COL])

    out = pd.DataFrame({"date": data["date"]})
    out["china_cli"] = cli.values
    out["china_cli_change_3m"] = _change(cli, LOOKBACK_3M_MONTHLY).values
    out["china_cli_z_3y"] = _rolling_z_score(
        cli,
        window=LOOKBACK_3Y_MONTHLY,
        min_periods=LOOKBACK_1Y_MONTHLY,
    ).values

    cli_level_score = _score_high_is_good(
        cli,
        window=LOOKBACK_3Y_MONTHLY,
        min_periods=LOOKBACK_1Y_MONTHLY,
    )

    cli_change_score = _score_high_is_good(
        pd.Series(out["china_cli_change_3m"]),
        window=LOOKBACK_3Y_MONTHLY,
        min_periods=LOOKBACK_1Y_MONTHLY,
    )

    cli_z_score = _score_high_is_good(
        pd.Series(out["china_cli_z_3y"]),
        window=LOOKBACK_3Y_MONTHLY,
        min_periods=LOOKBACK_1Y_MONTHLY,
    )

    out["copper_china_cli_score"] = _neutral_fill(
        0.35 * cli_level_score
        + 0.50 * cli_change_score
        + 0.15 * cli_z_score
    ).values

    return out


def _merge_monthly_features_asof(
    daily_dates: pd.DataFrame,
    monthly_features: pd.DataFrame,
    tolerance_days: int = MONTHLY_ASOF_TOLERANCE_DAYS,
) -> pd.DataFrame:
    left = daily_dates.copy()
    right = monthly_features.copy()

    left["date"] = pd.to_datetime(left["date"])
    right["date"] = pd.to_datetime(right["date"])

    left = left.sort_values("date").reset_index(drop=True)
    right = right.sort_values("date").reset_index(drop=True)

    return pd.merge_asof(
        left,
        right,
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=tolerance_days),
    )


def build_china_features_daily(
    dates: pd.Series,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if raw is None:
        raw = load_copper_raw_wide()

    daily = pd.DataFrame({"date": pd.to_datetime(dates)})
    daily = daily.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    electricity = _build_monthly_china_electricity_features(raw)
    cli = _build_monthly_china_cli_features(raw)

    out = _merge_monthly_features_asof(daily, electricity)
    out = pd.merge_asof(
        out.sort_values("date").reset_index(drop=True),
        cli.sort_values("date").reset_index(drop=True),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=MONTHLY_ASOF_TOLERANCE_DAYS),
    )

    return out.sort_values("date").reset_index(drop=True)


# ============================================================
# MARKET FEATURES
# ============================================================

def build_market_features_daily(
    commodity_prices: pd.DataFrame | None = None,
    macro_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if commodity_prices is None:
        commodity_prices = load_commodity_price_matrix()

    if macro_prices is None:
        macro_prices = load_macro_price_matrix()

    commodity_prices = commodity_prices.copy().sort_index()
    macro_prices = macro_prices.copy().sort_index()

    common_dates = commodity_prices.index.intersection(macro_prices.index)

    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between commodity and macro price data.")

    commodity_prices = commodity_prices.loc[common_dates]
    macro_prices = macro_prices.loc[common_dates]

    cper = _safe_numeric(commodity_prices["CPER"])
    uso = _safe_numeric(commodity_prices["USO"])

    uup = _safe_numeric(macro_prices["UUP"])
    dbc = _safe_numeric(macro_prices["DBC"])
    spy = _safe_numeric(macro_prices["SPY"])
    vix = _safe_numeric(macro_prices["^VIX"])

    out = pd.DataFrame(index=common_dates)
    out.index.name = None
    out["date"] = common_dates

    # ------------------------------
    # CPER own diagnostics
    # ------------------------------

    out["cper_return_1m"] = _return(cper, TRADING_DAYS_PER_MONTH).values
    out["cper_return_3m"] = _return(cper, LOOKBACK_3M).values
    out["cper_return_6m"] = _return(cper, LOOKBACK_6M).values

    out["cper_trend_score"] = _trend_score(cper).values

    out["cper_momentum_score"] = _neutral_fill(
        0.50 * _score_high_is_good(_return(cper, LOOKBACK_3M))
        + 0.50 * _score_high_is_good(_return(cper, LOOKBACK_6M))
    ).values

    # ------------------------------
    # USD strength
    # Low / falling USD is good for copper.
    # ------------------------------

    out["usd_index"] = uup.values
    out["usd_return_1m"] = _return(uup, TRADING_DAYS_PER_MONTH).values
    out["usd_return_3m"] = _return(uup, LOOKBACK_3M).values
    out["usd_return_6m"] = _return(uup, LOOKBACK_6M).values
    out["usd_z_3y"] = _rolling_z_score(
        uup,
        window=LOOKBACK_3Y,
        min_periods=LOOKBACK_1Y,
    ).values

    usd_3m_score = _score_low_is_good(_return(uup, LOOKBACK_3M))
    usd_6m_score = _score_low_is_good(_return(uup, LOOKBACK_6M))
    usd_level_score = _score_low_is_good(uup)

    out["copper_usd_score"] = _neutral_fill(
        0.45 * usd_3m_score
        + 0.35 * usd_6m_score
        + 0.20 * usd_level_score
    ).values

    # ------------------------------
    # Broad commodity trend
    # DBC trend confirms commodity-wide regime.
    # ------------------------------

    out["dbc_return_1m"] = _return(dbc, TRADING_DAYS_PER_MONTH).values
    out["dbc_return_3m"] = _return(dbc, LOOKBACK_3M).values
    out["dbc_return_6m"] = _return(dbc, LOOKBACK_6M).values
    out["dbc_trend_score"] = _trend_score(dbc).values

    dbc_3m_score = _score_high_is_good(_return(dbc, LOOKBACK_3M))
    dbc_6m_score = _score_high_is_good(_return(dbc, LOOKBACK_6M))

    out["copper_broad_commodity_trend_score"] = _neutral_fill(
        0.50 * out["dbc_trend_score"]
        + 0.30 * dbc_3m_score
        + 0.20 * dbc_6m_score
    ).values

    # ------------------------------
    # Oil price / reflation confirmation
    # Oil strength is treated as cyclical/reflation confirmation, not a direct cause.
    # ------------------------------

    out["uso_return_1m"] = _return(uso, TRADING_DAYS_PER_MONTH).values
    out["uso_return_3m"] = _return(uso, LOOKBACK_3M).values
    out["uso_return_6m"] = _return(uso, LOOKBACK_6M).values
    out["uso_trend_score"] = _trend_score(uso).values

    uso_3m_score = _score_high_is_good(_return(uso, LOOKBACK_3M))
    uso_6m_score = _score_high_is_good(_return(uso, LOOKBACK_6M))

    out["copper_oil_price_score"] = _neutral_fill(
        0.50 * out["uso_trend_score"]
        + 0.35 * uso_3m_score
        + 0.15 * uso_6m_score
    ).values

    # ------------------------------
    # Global growth / risk appetite
    # SPY confirms growth/risk-on; low VIX confirms lower stress.
    # ------------------------------

    out["spy_return_1m"] = _return(spy, TRADING_DAYS_PER_MONTH).values
    out["spy_return_3m"] = _return(spy, LOOKBACK_3M).values
    out["spy_return_6m"] = _return(spy, LOOKBACK_6M).values
    out["spy_trend_score"] = _trend_score(spy).values

    out["vix_index"] = vix.values
    out["vix_z_3y"] = _rolling_z_score(
        vix,
        window=LOOKBACK_3Y,
        min_periods=LOOKBACK_1Y,
    ).values

    spy_3m_score = _score_high_is_good(_return(spy, LOOKBACK_3M))
    vix_low_score = _score_low_is_good(vix)

    out["copper_global_growth_score"] = _neutral_fill(
        0.45 * out["spy_trend_score"]
        + 0.35 * spy_3m_score
        + 0.20 * vix_low_score
    ).values

    return out.sort_values("date").reset_index(drop=True)


# ============================================================
# FINAL FEATURE BUILDING
# ============================================================

def build_copper_features_daily() -> pd.DataFrame:
    commodity_prices = load_commodity_price_matrix()
    macro_prices = load_macro_price_matrix()
    raw_copper = load_copper_raw_wide()

    market_features = build_market_features_daily(
        commodity_prices=commodity_prices,
        macro_prices=macro_prices,
    )

    china_features = build_china_features_daily(
        dates=market_features["date"],
        raw=raw_copper,
    )

    out = pd.merge(
        market_features,
        china_features,
        on="date",
        how="left",
    )

    core_cols = [
        "copper_china_electricity_score",
        "copper_china_cli_score",
        "copper_usd_score",
        "copper_broad_commodity_trend_score",
        "copper_oil_price_score",
        "copper_global_growth_score",
    ]

    raw_core_available = out[core_cols].notna().sum(axis=1)

    for col in core_cols:
        if col not in out.columns:
            out[col] = 0.50

        out[col] = _neutral_fill(pd.to_numeric(out[col], errors="coerce"))

    out["copper_core_feature_count"] = raw_core_available

    out["copper_core_data_quality_score"] = (
        out["copper_core_feature_count"] / len(core_cols)
    ).clip(0.0, 1.0)

    return out.sort_values("date").reset_index(drop=True)


def build_copper_features_monthly(features_daily: pd.DataFrame) -> pd.DataFrame:
    out = features_daily.copy()
    out["date"] = pd.to_datetime(out["date"])

    monthly = (
        out
        .set_index("date")
        .resample("ME")
        .last()
        .dropna(how="all")
        .reset_index()
    )

    return monthly


def save_copper_features(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
) -> None:
    COPPER_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    daily.to_csv(COPPER_FEATURES_DAILY_PATH, index=False)
    monthly.to_csv(COPPER_FEATURES_MONTHLY_PATH, index=False)

    print(f"\nSaved daily copper features to: {COPPER_FEATURES_DAILY_PATH}")
    print(f"Saved monthly copper features to: {COPPER_FEATURES_MONTHLY_PATH}")


# ============================================================
# PIPELINE
# ============================================================

def run_copper_feature_pipeline() -> pd.DataFrame:
    print("\nStarting copper feature pipeline.")
    print(f"Project root:       {COMMODITY_ROOT}")
    print(f"Commodity prices:   {PRICE_DATA_PATH}")
    print(f"Macro prices:       {MACRO_PRICE_DATA_PATH}")
    print(f"Copper raw wide:    {COPPER_RAW_WIDE_PATH}")

    daily = build_copper_features_daily()
    monthly = build_copper_features_monthly(daily)

    save_copper_features(daily, monthly)

    print("\nCopper feature pipeline complete.")
    print(f"Daily rows:   {len(daily):,}")
    print(f"Monthly rows: {len(monthly):,}")
    print(f"Start date:   {daily['date'].min().date()}")
    print(f"End date:     {daily['date'].max().date()}")

    score_cols = [
        "copper_china_electricity_score",
        "copper_china_cli_score",
        "copper_usd_score",
        "copper_broad_commodity_trend_score",
        "copper_oil_price_score",
        "copper_global_growth_score",
        "copper_core_data_quality_score",
    ]

    print("\nCore copper score columns:")
    print(daily[score_cols].describe().to_string())

    _validate_not_constant(daily, score_cols)

    return daily


if __name__ == "__main__":
    run_copper_feature_pipeline()