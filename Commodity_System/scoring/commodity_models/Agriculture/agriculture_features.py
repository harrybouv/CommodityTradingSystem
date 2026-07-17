from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)


# ============================================================
# DIRECT-RUN PATH SETUP
# ============================================================

THIS_FILE = Path(__file__).resolve()
COMMODITY_ROOT = THIS_FILE.parents[3]

if str(COMMODITY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMODITY_ROOT))


from config import PROCESSED_DATA_DIR


# ============================================================
# PATHS
# ============================================================

AGRI_PROCESSED_DIR = PROCESSED_DATA_DIR / "agriculture"

AGRI_RAW_WIDE_PATH = AGRI_PROCESSED_DIR / "agriculture_raw_wide.csv"
AGRI_FEATURES_DAILY_PATH = AGRI_PROCESSED_DIR / "agriculture_features_daily.csv"
AGRI_FEATURES_MONTHLY_PATH = AGRI_PROCESSED_DIR / "agriculture_features_monthly.csv"


# ============================================================
# SETTINGS
# ============================================================

TRADING_DAYS_PER_MONTH = 21
LOOKBACK_1M = 21
LOOKBACK_3M = 63
LOOKBACK_6M = 126
LOOKBACK_1Y = 252
LOOKBACK_3Y = 756

FAST_MA = 50
SLOW_MA = 200

DAILY_FFILL_LIMIT = 5
WEEKLY_FFILL_LIMIT = 10
MONTHLY_FFILL_LIMIT = 45

AGRI_ASOF_TOLERANCE_DAYS = 10

SEASONAL_LOOKBACK_YEARS = 8
SEASONAL_DAY_WINDOW = 14
SEASONAL_MIN_OBS = 10

EXPORT_SEASONAL_LOOKBACK_YEARS = 5
EXPORT_SEASONAL_DAY_WINDOW = 21
EXPORT_SEASONAL_MIN_OBS = 8


# ============================================================
# RAW-WIDE COLUMN NAMES
# ============================================================

DBA_COL = "yf_DBA_adj_close"
DBC_COL = "yf_DBC_adj_close"

CORN_COL = "yf_ZC_F_adj_close"
WHEAT_COL = "yf_ZW_F_adj_close"
SOYBEAN_COL = "yf_ZS_F_adj_close"
SUGAR_COL = "yf_SB_F_adj_close"
COFFEE_COL = "yf_KC_F_adj_close"
COCOA_COL = "yf_CC_F_adj_close"

USD_COL = "DTWEXBGS"
DGS10_COL = "DGS10"
DGS2_COL = "DGS2"
T10Y2Y_COL = "T10Y2Y"

PRIMARY_CROP_COLS = {
    "corn": CORN_COL,
    "wheat": WHEAT_COL,
    "soybeans": SOYBEAN_COL,
}

SOFT_CROP_COLS = {
    "sugar": SUGAR_COL,
    "coffee": COFFEE_COL,
    "cocoa": COCOA_COL,
}

EXPORT_SERIES = {
    "corn": {
        "weekly_exports": "ESR_CORN_WEEKLYEXPORTS",
        "net_sales": "ESR_CORN_CURRENTMYNETSALES",
        "gross_new_sales": "ESR_CORN_GROSSNEWSALES",
        "total_commitment": "ESR_CORN_CURRENTMYTOTALCOMMITMENT",
        "outstanding_sales": "ESR_CORN_OUTSTANDINGSALES",
        "accumulated_exports": "ESR_CORN_ACCUMULATEDEXPORTS",
    },
    "wheat": {
        "weekly_exports": "ESR_ALL_WHEAT_WEEKLYEXPORTS",
        "net_sales": "ESR_ALL_WHEAT_CURRENTMYNETSALES",
        "gross_new_sales": "ESR_ALL_WHEAT_GROSSNEWSALES",
        "total_commitment": "ESR_ALL_WHEAT_CURRENTMYTOTALCOMMITMENT",
        "outstanding_sales": "ESR_ALL_WHEAT_OUTSTANDINGSALES",
        "accumulated_exports": "ESR_ALL_WHEAT_ACCUMULATEDEXPORTS",
    },
    "soybeans": {
        "weekly_exports": "ESR_SOYBEANS_WEEKLYEXPORTS",
        "net_sales": "ESR_SOYBEANS_CURRENTMYNETSALES",
        "gross_new_sales": "ESR_SOYBEANS_GROSSNEWSALES",
        "total_commitment": "ESR_SOYBEANS_CURRENTMYTOTALCOMMITMENT",
        "outstanding_sales": "ESR_SOYBEANS_OUTSTANDINGSALES",
        "accumulated_exports": "ESR_SOYBEANS_ACCUMULATEDEXPORTS",
    },
}


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


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = _safe_numeric(numerator)
    den = _safe_numeric(denominator).replace(0.0, np.nan)
    return (num / den).replace([np.inf, -np.inf], np.nan)


def _rolling_z_score(
    s: pd.Series,
    window: int = LOOKBACK_3Y,
    min_periods: int = LOOKBACK_1Y,
) -> pd.Series:
    x = _safe_numeric(s)

    mean = x.rolling(window=window, min_periods=min_periods).mean()
    std = x.rolling(window=window, min_periods=min_periods).std(ddof=0)

    z = (x - mean) / std.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan)


def _rolling_percentile(
    s: pd.Series,
    window: int = LOOKBACK_3Y,
    min_periods: int = LOOKBACK_1Y,
) -> pd.Series:
    x = _safe_numeric(s)

    def percentile_rank(values: np.ndarray) -> float:
        values = values[~np.isnan(values)]

        if len(values) < min_periods:
            return np.nan

        latest = values[-1]
        return float((values <= latest).mean())

    return x.rolling(window=window, min_periods=min_periods).apply(
        percentile_rank,
        raw=True,
    )


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


def _ffill_existing(
    df: pd.DataFrame,
    columns: list[str],
    limit: int,
) -> pd.DataFrame:
    out = df.copy()
    existing = [col for col in columns if col in out.columns]

    if existing:
        out[existing] = out[existing].ffill(limit=limit)

    return out


def _series_or_nan(
    df: pd.DataFrame,
    column: str,
    index: pd.Index | None = None,
) -> pd.Series:
    if index is None:
        index = df.index

    if column not in df.columns:
        return pd.Series(np.nan, index=index, dtype="float64")

    s = _safe_numeric(df[column])

    if not s.index.equals(index):
        s = pd.Series(s.values, index=index)

    return s


def _weighted_mean_available(weighted_series: dict[str, tuple[pd.Series, float]]) -> pd.Series:
    if not weighted_series:
        return pd.Series(dtype="float64")

    first_series = next(iter(weighted_series.values()))[0]
    index = first_series.index

    numerator = pd.Series(0.0, index=index, dtype="float64")
    denominator = pd.Series(0.0, index=index, dtype="float64")

    for series, weight in weighted_series.values():
        s = _safe_numeric(series).reindex(index)
        valid = s.notna()
        numerator = numerator + s.fillna(0.0) * float(weight)
        denominator = denominator + valid.astype(float) * float(weight)

    return (numerator / denominator.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _rebased_price(price: pd.Series) -> pd.Series:
    s = _safe_numeric(price).copy()
    first_valid = s.dropna()

    if first_valid.empty:
        return pd.Series(np.nan, index=s.index, dtype="float64")

    base = float(first_valid.iloc[0])

    if base == 0.0 or np.isnan(base):
        return pd.Series(np.nan, index=s.index, dtype="float64")

    return (s / base * 100.0).replace([np.inf, -np.inf], np.nan)


def _rebased_basket(
    price_series: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
) -> pd.Series:
    if weights is None:
        weights = {name: 1.0 for name in price_series}

    rebased = {
        name: (_rebased_price(series), float(weights.get(name, 1.0)))
        for name, series in price_series.items()
    }

    return _weighted_mean_available(rebased)


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


def _seasonal_forward_return_expectation(
    dates: pd.Series,
    price: pd.Series,
    forward_days: int,
    lookback_years: int = SEASONAL_LOOKBACK_YEARS,
    day_window: int = SEASONAL_DAY_WINDOW,
    min_obs: int = SEASONAL_MIN_OBS,
) -> pd.Series:
    """
    Trailing no-lookahead seasonality signal.

    For each current date, this looks at previous years around the same
    day-of-year and asks: what was the realised forward return from those
    historical dates? Only historical observations whose forward return would
    already be known by the current row are used.
    """

    dt = pd.to_datetime(dates).reset_index(drop=True)
    values = _safe_numeric(price).reset_index(drop=True).to_numpy(dtype="float64")

    n = len(values)
    out = np.full(n, np.nan, dtype="float64")

    if n <= forward_days:
        return pd.Series(out, index=price.index, dtype="float64")

    forward_returns = np.full(n, np.nan, dtype="float64")
    valid = (
        np.isfinite(values[:-forward_days])
        & np.isfinite(values[forward_days:])
        & (values[:-forward_days] != 0.0)
    )
    forward_returns[:-forward_days][valid] = (
        values[forward_days:][valid] / values[:-forward_days][valid] - 1.0
    )

    ordinals = np.array([d.toordinal() for d in dt], dtype="int64")
    day_of_year = dt.dt.dayofyear.to_numpy(dtype="int64")

    for i in range(n):
        latest_idx = i - forward_days

        if latest_idx <= 0:
            continue

        start_date = dt.iloc[i] - pd.DateOffset(years=lookback_years)
        start_ordinal = start_date.toordinal()

        hist_slice = slice(0, latest_idx + 1)
        hist_returns = forward_returns[hist_slice]

        if np.isfinite(hist_returns).sum() < min_obs:
            continue

        doy_distance = np.minimum(
            np.abs(day_of_year[hist_slice] - day_of_year[i]),
            366 - np.abs(day_of_year[hist_slice] - day_of_year[i]),
        )

        mask = (
            (ordinals[hist_slice] >= start_ordinal)
            & (doy_distance <= day_window)
            & np.isfinite(hist_returns)
        )

        if mask.sum() < min_obs:
            continue

        out[i] = float(np.nanmean(hist_returns[mask]))

    return pd.Series(out, index=price.index, dtype="float64")


def _seasonal_level_percentile(
    dates: pd.Series,
    values: pd.Series,
    lookback_years: int = EXPORT_SEASONAL_LOOKBACK_YEARS,
    day_window: int = EXPORT_SEASONAL_DAY_WINDOW,
    min_obs: int = EXPORT_SEASONAL_MIN_OBS,
) -> pd.Series:
    """
    Percentile of the current level versus prior years around the same day.

    This is mainly for ESR total commitments / accumulated exports, because
    those series reset by market year and are not comparable as raw levels
    across the whole calendar.
    """

    dt = pd.to_datetime(dates).reset_index(drop=True)
    x = _safe_numeric(values).reset_index(drop=True).to_numpy(dtype="float64")

    n = len(x)
    out = np.full(n, np.nan, dtype="float64")

    ordinals = np.array([d.toordinal() for d in dt], dtype="int64")
    day_of_year = dt.dt.dayofyear.to_numpy(dtype="int64")

    for i in range(n):
        current_value = x[i]

        if not np.isfinite(current_value) or i == 0:
            continue

        start_date = dt.iloc[i] - pd.DateOffset(years=lookback_years)
        start_ordinal = start_date.toordinal()

        hist_slice = slice(0, i)
        hist_values = x[hist_slice]

        doy_distance = np.minimum(
            np.abs(day_of_year[hist_slice] - day_of_year[i]),
            366 - np.abs(day_of_year[hist_slice] - day_of_year[i]),
        )

        mask = (
            (ordinals[hist_slice] >= start_ordinal)
            & (doy_distance <= day_window)
            & np.isfinite(hist_values)
        )

        if mask.sum() < min_obs:
            continue

        seasonal_values = hist_values[mask]
        out[i] = float((seasonal_values <= current_value).mean())

    return pd.Series(out, index=values.index, dtype="float64")


# ============================================================
# LOAD / AS-OF PREPARATION
# ============================================================

def load_agriculture_raw_wide(path: Path = AGRI_RAW_WIDE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Agriculture raw wide file not found: {path}. "
            "Run scoring/commodity_models/Agriculture/agriculture_data.py first."
        )

    raw = pd.read_csv(path)

    if "date" not in raw.columns:
        raise ValueError(f"Agriculture raw wide file missing date column: {path}")

    raw["date"] = pd.to_datetime(raw["date"])
    raw = (
        raw
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    for col in raw.columns:
        if col != "date":
            raw[col] = _safe_numeric(raw[col])

    required_cols = [
        DBA_COL,
        DBC_COL,
        CORN_COL,
        WHEAT_COL,
        SOYBEAN_COL,
        USD_COL,
        DGS10_COL,
        DGS2_COL,
    ]

    missing = [col for col in required_cols if col not in raw.columns]

    if missing:
        raise ValueError(
            f"Agriculture raw wide missing required columns: {missing}. "
            f"Available columns: {list(raw.columns)}"
        )

    return raw


def prepare_agriculture_asof_inputs(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Creates an as-of table from mixed-frequency agriculture inputs.

    agriculture_raw_wide.csv already uses estimated availability dates from
    agriculture_data.py. The forward-fill limits below therefore preserve the
    release-lag discipline while making weekly / daily inputs usable on DBA
    trading dates.
    """

    out = raw.copy().sort_values("date").reset_index(drop=True)

    market_cols = [
        DBA_COL,
        DBC_COL,
        CORN_COL,
        WHEAT_COL,
        SOYBEAN_COL,
        SUGAR_COL,
        COFFEE_COL,
        COCOA_COL,
    ]

    fred_cols = [
        USD_COL,
        DGS10_COL,
        DGS2_COL,
        T10Y2Y_COL,
    ]

    esr_cols = [
        col
        for series in EXPORT_SERIES.values()
        for col in series.values()
    ]

    out = _ffill_existing(out, market_cols, DAILY_FFILL_LIMIT)
    out = _ffill_existing(out, fred_cols, DAILY_FFILL_LIMIT)
    out = _ffill_existing(out, esr_cols, WEEKLY_FFILL_LIMIT)

    return out


def get_agriculture_market_dates(raw: pd.DataFrame) -> pd.Series:
    market_dates = raw.loc[raw[DBA_COL].notna(), "date"].dropna().copy()

    if market_dates.empty:
        raise ValueError(
            f"No DBA market dates found in agriculture raw wide column {DBA_COL}."
        )

    return market_dates.sort_values().drop_duplicates().reset_index(drop=True)


def merge_agriculture_raw_to_market_dates(
    market_dates: pd.Series,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if raw is None:
        raw = load_agriculture_raw_wide()

    prepared = prepare_agriculture_asof_inputs(raw)

    left = pd.DataFrame({"date": pd.to_datetime(market_dates).sort_values().values})
    right = prepared.copy()
    right["date"] = pd.to_datetime(right["date"])

    left = left.sort_values("date").reset_index(drop=True)
    right = right.sort_values("date").reset_index(drop=True)

    out = pd.merge_asof(
        left,
        right,
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=AGRI_ASOF_TOLERANCE_DAYS),
    )

    return out.sort_values("date").reset_index(drop=True)


# ============================================================
# FEATURE BUILDING
# ============================================================

def _build_single_crop_momentum_features(
    out: pd.DataFrame,
    name: str,
    price: pd.Series,
) -> pd.Series:
    out[f"{name}_price"] = price.values
    out[f"{name}_return_1m"] = _return(price, LOOKBACK_1M).values
    out[f"{name}_return_3m"] = _return(price, LOOKBACK_3M).values
    out[f"{name}_return_6m"] = _return(price, LOOKBACK_6M).values
    out[f"{name}_trend_score"] = _trend_score(price).values

    score = _neutral_fill(
        0.50 * _score_high_is_good(_return(price, LOOKBACK_3M))
        + 0.30 * _score_high_is_good(_return(price, LOOKBACK_6M))
        + 0.20 * _trend_score(price)
    )

    out[f"{name}_momentum_score"] = score.values
    return score


def _build_single_export_score(
    out: pd.DataFrame,
    agri: pd.DataFrame,
    dates: pd.Series,
    crop_name: str,
    series_map: dict[str, str],
) -> pd.Series:
    index = agri.index

    weekly_exports = _series_or_nan(agri, series_map["weekly_exports"], index=index)
    net_sales = _series_or_nan(agri, series_map["net_sales"], index=index)
    gross_new_sales = _series_or_nan(agri, series_map["gross_new_sales"], index=index)
    total_commitment = _series_or_nan(agri, series_map["total_commitment"], index=index)
    outstanding_sales = _series_or_nan(agri, series_map["outstanding_sales"], index=index)
    accumulated_exports = _series_or_nan(agri, series_map["accumulated_exports"], index=index)

    out[f"{crop_name}_weekly_exports"] = weekly_exports.values
    out[f"{crop_name}_current_my_net_sales"] = net_sales.values
    out[f"{crop_name}_gross_new_sales"] = gross_new_sales.values
    out[f"{crop_name}_current_my_total_commitment"] = total_commitment.values
    out[f"{crop_name}_outstanding_sales"] = outstanding_sales.values
    out[f"{crop_name}_accumulated_exports"] = accumulated_exports.values

    weekly_exports_4w = weekly_exports.rolling(window=LOOKBACK_1M, min_periods=5).mean()
    net_sales_4w = net_sales.rolling(window=LOOKBACK_1M, min_periods=5).mean()
    gross_sales_4w = gross_new_sales.rolling(window=LOOKBACK_1M, min_periods=5).mean()

    out[f"{crop_name}_weekly_exports_4w_avg"] = weekly_exports_4w.values
    out[f"{crop_name}_net_sales_4w_avg"] = net_sales_4w.values
    out[f"{crop_name}_gross_sales_4w_avg"] = gross_sales_4w.values

    commitment_seasonal_pct = _seasonal_level_percentile(
        dates=dates,
        values=total_commitment.reset_index(drop=True),
    )
    outstanding_seasonal_pct = _seasonal_level_percentile(
        dates=dates,
        values=outstanding_sales.reset_index(drop=True),
    )

    commitment_seasonal_pct.index = index
    outstanding_seasonal_pct.index = index

    out[f"{crop_name}_commitment_seasonal_percentile_5y"] = commitment_seasonal_pct.values
    out[f"{crop_name}_outstanding_sales_seasonal_percentile_5y"] = outstanding_seasonal_pct.values

    weekly_score = _score_high_is_good(weekly_exports_4w)
    net_sales_score = _score_high_is_good(net_sales_4w)
    gross_sales_score = _score_high_is_good(gross_sales_4w)
    commitment_score = _clip01(commitment_seasonal_pct)
    outstanding_score = _clip01(outstanding_seasonal_pct)

    score = _neutral_fill(
        0.25 * weekly_score
        + 0.25 * net_sales_score
        + 0.15 * gross_sales_score
        + 0.25 * commitment_score
        + 0.10 * outstanding_score
    )

    out[f"{crop_name}_export_demand_score"] = score.values
    return score


def build_agriculture_features_daily(
    agri_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if agri_raw is None:
        agri_raw = load_agriculture_raw_wide()

    market_dates = get_agriculture_market_dates(agri_raw)

    agri = merge_agriculture_raw_to_market_dates(
        market_dates=market_dates,
        raw=agri_raw,
    )

    agri["date"] = pd.to_datetime(agri["date"])
    agri = agri.sort_values("date").drop_duplicates("date", keep="last")
    agri = agri.set_index("date")
    agri.index.name = None

    common_dates = agri.index

    dba = _safe_numeric(agri[DBA_COL])
    dbc = _safe_numeric(agri[DBC_COL])

    corn = _safe_numeric(agri[CORN_COL])
    wheat = _safe_numeric(agri[WHEAT_COL])
    soybeans = _safe_numeric(agri[SOYBEAN_COL])
    sugar = _series_or_nan(agri, SUGAR_COL, index=common_dates)
    coffee = _series_or_nan(agri, COFFEE_COL, index=common_dates)
    cocoa = _series_or_nan(agri, COCOA_COL, index=common_dates)

    usd = _safe_numeric(agri[USD_COL])
    dgs10 = _safe_numeric(agri[DGS10_COL])
    dgs2 = _safe_numeric(agri[DGS2_COL])
    t10y2y = _series_or_nan(agri, T10Y2Y_COL, index=common_dates)

    out = pd.DataFrame(index=common_dates)
    out.index.name = None
    out["date"] = common_dates

    # ------------------------------
    # Raw market / macro inputs
    # ------------------------------

    out["dba_price"] = dba.values
    out["dbc_price"] = dbc.values
    out["usd_index"] = usd.values
    out["dgs10"] = dgs10.values
    out["dgs2"] = dgs2.values
    out["yield_curve_10y2y"] = t10y2y.values

    # ------------------------------
    # DBA diagnostics
    # ------------------------------

    out["dba_return_1m"] = _return(dba, LOOKBACK_1M).values
    out["dba_return_3m"] = _return(dba, LOOKBACK_3M).values
    out["dba_return_6m"] = _return(dba, LOOKBACK_6M).values
    out["dba_trend_score"] = _trend_score(dba).values

    out["dba_momentum_score"] = _neutral_fill(
        0.50 * _score_high_is_good(_return(dba, LOOKBACK_3M))
        + 0.30 * _score_high_is_good(_return(dba, LOOKBACK_6M))
        + 0.20 * _trend_score(dba)
    ).values

    # ------------------------------
    # 1. USD score: weak/falling USD is bullish for agriculture.
    # ------------------------------

    out["usd_return_1m"] = _return(usd, LOOKBACK_1M).values
    out["usd_return_3m"] = _return(usd, LOOKBACK_3M).values
    out["usd_return_6m"] = _return(usd, LOOKBACK_6M).values
    out["usd_z_3y"] = _rolling_z_score(usd).values

    usd_3m_score = _score_low_is_good(_return(usd, LOOKBACK_3M))
    usd_6m_score = _score_low_is_good(_return(usd, LOOKBACK_6M))
    usd_level_score = _score_low_is_good(usd)

    out["agri_usd_score"] = _neutral_fill(
        0.45 * usd_3m_score
        + 0.35 * usd_6m_score
        + 0.20 * usd_level_score
    ).values

    # ------------------------------
    # 2. Rates score: lower/falling rates are treated as a modest tailwind.
    # This should remain lower-weight than crop momentum / relative strength.
    # ------------------------------

    out["dgs10_change_1m"] = _change(dgs10, LOOKBACK_1M).values
    out["dgs10_change_3m"] = _change(dgs10, LOOKBACK_3M).values
    out["dgs2_change_1m"] = _change(dgs2, LOOKBACK_1M).values
    out["dgs2_change_3m"] = _change(dgs2, LOOKBACK_3M).values
    out["dgs10_z_3y"] = _rolling_z_score(dgs10).values
    out["dgs2_z_3y"] = _rolling_z_score(dgs2).values
    out["yield_curve_10y2y_change_3m"] = _change(t10y2y, LOOKBACK_3M).values

    long_rate_level_score = _score_low_is_good(dgs10)
    long_rate_change_score = _score_low_is_good(_change(dgs10, LOOKBACK_3M))
    front_rate_change_score = _score_low_is_good(_change(dgs2, LOOKBACK_3M))
    curve_score = _score_high_is_good(t10y2y)

    out["agri_rates_score"] = _neutral_fill(
        0.35 * long_rate_level_score
        + 0.30 * long_rate_change_score
        + 0.20 * front_rate_change_score
        + 0.15 * curve_score
    ).values

    # ------------------------------
    # 3. Crop momentum score: core directional signal.
    # Uses primary grains/oilseeds heavily, with softs as secondary breadth.
    # ------------------------------

    crop_scores: dict[str, tuple[pd.Series, float]] = {}

    crop_scores["corn"] = (
        _build_single_crop_momentum_features(out, "corn", corn),
        1.0,
    )
    crop_scores["wheat"] = (
        _build_single_crop_momentum_features(out, "wheat", wheat),
        1.0,
    )
    crop_scores["soybeans"] = (
        _build_single_crop_momentum_features(out, "soybeans", soybeans),
        1.0,
    )

    crop_scores["sugar"] = (
        _build_single_crop_momentum_features(out, "sugar", sugar),
        0.35,
    )
    crop_scores["coffee"] = (
        _build_single_crop_momentum_features(out, "coffee", coffee),
        0.35,
    )
    crop_scores["cocoa"] = (
        _build_single_crop_momentum_features(out, "cocoa", cocoa),
        0.35,
    )

    out["agri_crop_momentum_score"] = _neutral_fill(
        _weighted_mean_available(crop_scores)
    ).values

    primary_crop_basket = _rebased_basket(
        {
            "corn": corn,
            "wheat": wheat,
            "soybeans": soybeans,
        }
    )

    soft_crop_basket = _rebased_basket(
        {
            "sugar": sugar,
            "coffee": coffee,
            "cocoa": cocoa,
        }
    )

    crop_basket = _weighted_mean_available(
        {
            "primary": (primary_crop_basket, 0.75),
            "softs": (soft_crop_basket, 0.25),
        }
    )

    out["primary_crop_basket_index"] = primary_crop_basket.values
    out["soft_crop_basket_index"] = soft_crop_basket.values
    out["agri_crop_basket_index"] = crop_basket.values
    out["agri_crop_basket_return_1m"] = _return(crop_basket, LOOKBACK_1M).values
    out["agri_crop_basket_return_3m"] = _return(crop_basket, LOOKBACK_3M).values
    out["agri_crop_basket_return_6m"] = _return(crop_basket, LOOKBACK_6M).values
    out["agri_crop_basket_trend_score"] = _trend_score(crop_basket).values

    # ------------------------------
    # 4. Crop relative-strength score: core cross-market confirmation.
    # Strong crop basket / DBA versus DBC means agriculture-specific strength,
    # not just a generic commodities beta move.
    # ------------------------------

    crop_vs_dbc_ratio = _safe_ratio(crop_basket, dbc)
    dba_vs_dbc_ratio = _safe_ratio(dba, dbc)
    crop_vs_dba_ratio = _safe_ratio(crop_basket, dba)

    out["agri_crop_vs_dbc_ratio"] = crop_vs_dbc_ratio.values
    out["dba_vs_dbc_ratio"] = dba_vs_dbc_ratio.values
    out["agri_crop_vs_dba_ratio"] = crop_vs_dba_ratio.values

    out["agri_crop_vs_dbc_return_3m"] = _return(crop_vs_dbc_ratio, LOOKBACK_3M).values
    out["dba_vs_dbc_return_3m"] = _return(dba_vs_dbc_ratio, LOOKBACK_3M).values
    out["agri_crop_vs_dba_return_3m"] = _return(crop_vs_dba_ratio, LOOKBACK_3M).values

    crop_vs_dbc_score = _neutral_fill(
        0.45 * _trend_score(crop_vs_dbc_ratio)
        + 0.35 * _score_high_is_good(_return(crop_vs_dbc_ratio, LOOKBACK_3M))
        + 0.20 * _score_high_is_good(_return(crop_vs_dbc_ratio, LOOKBACK_6M))
    )

    dba_vs_dbc_score = _neutral_fill(
        0.45 * _trend_score(dba_vs_dbc_ratio)
        + 0.35 * _score_high_is_good(_return(dba_vs_dbc_ratio, LOOKBACK_3M))
        + 0.20 * _score_high_is_good(_return(dba_vs_dbc_ratio, LOOKBACK_6M))
    )

    crop_vs_dba_score = _neutral_fill(
        0.45 * _trend_score(crop_vs_dba_ratio)
        + 0.35 * _score_high_is_good(_return(crop_vs_dba_ratio, LOOKBACK_3M))
        + 0.20 * _score_high_is_good(_return(crop_vs_dba_ratio, LOOKBACK_6M))
    )

    out["agri_crop_vs_dbc_score"] = crop_vs_dbc_score.values
    out["dba_vs_dbc_score"] = dba_vs_dbc_score.values
    out["agri_crop_vs_dba_score"] = crop_vs_dba_score.values

    out["agri_crop_relative_strength_score"] = _neutral_fill(
        0.55 * crop_vs_dbc_score
        + 0.30 * dba_vs_dbc_score
        + 0.15 * crop_vs_dba_score
    ).values

    # ------------------------------
    # 5. Broad commodity confirmation: supporting regime filter.
    # ------------------------------

    out["dbc_return_1m"] = _return(dbc, LOOKBACK_1M).values
    out["dbc_return_3m"] = _return(dbc, LOOKBACK_3M).values
    out["dbc_return_6m"] = _return(dbc, LOOKBACK_6M).values
    out["dbc_trend_score"] = _trend_score(dbc).values

    out["agri_broad_commodity_confirmation"] = _neutral_fill(
        0.50 * _trend_score(dbc)
        + 0.30 * _score_high_is_good(_return(dbc, LOOKBACK_3M))
        + 0.20 * _score_high_is_good(_return(dbc, LOOKBACK_6M))
    ).values

    # ------------------------------
    # 6. Seasonality score: supporting, no-lookahead historical seasonality.
    # ------------------------------

    seasonal_1m = _seasonal_forward_return_expectation(
        dates=pd.Series(common_dates),
        price=crop_basket.reset_index(drop=True),
        forward_days=LOOKBACK_1M,
    )
    seasonal_3m = _seasonal_forward_return_expectation(
        dates=pd.Series(common_dates),
        price=crop_basket.reset_index(drop=True),
        forward_days=LOOKBACK_3M,
    )

    seasonal_1m.index = common_dates
    seasonal_3m.index = common_dates

    out["agri_seasonal_expected_return_1m"] = seasonal_1m.values
    out["agri_seasonal_expected_return_3m"] = seasonal_3m.values

    out["agri_seasonality_score"] = _neutral_fill(
        0.60 * _score_high_is_good(seasonal_1m)
        + 0.40 * _score_high_is_good(seasonal_3m)
    ).values

    # ------------------------------
    # 7. ESR export-demand score: optional / low-weight / ablate carefully.
    # High or improving export sales/export shipments are supportive.
    # ------------------------------

    export_scores: dict[str, tuple[pd.Series, float]] = {}

    for crop_name, series_map in EXPORT_SERIES.items():
        export_scores[crop_name] = (
            _build_single_export_score(
                out=out,
                agri=agri,
                dates=pd.Series(common_dates),
                crop_name=crop_name,
                series_map=series_map,
            ),
            1.0,
        )

    out["agri_export_demand_score"] = _neutral_fill(
        _weighted_mean_available(export_scores)
    ).values

    # ------------------------------
    # Feature availability / diagnostic composite scores
    # ------------------------------

    core_score_cols = [
        "agri_usd_score",
        "agri_rates_score",
        "agri_crop_momentum_score",
        "agri_crop_relative_strength_score",
        "agri_broad_commodity_confirmation",
        "agri_seasonality_score",
    ]

    optional_score_cols = [
        "agri_export_demand_score",
    ]

    raw_core_available = out[core_score_cols].notna().sum(axis=1)
    raw_optional_available = out[optional_score_cols].notna().sum(axis=1)

    for col in core_score_cols + optional_score_cols:
        out[col] = _neutral_fill(pd.to_numeric(out[col], errors="coerce"))

    out["agri_core_feature_count"] = raw_core_available
    out["agri_optional_feature_count"] = raw_optional_available
    out["agri_core_data_quality_score"] = (
        out["agri_core_feature_count"] / len(core_score_cols)
    ).clip(0.0, 1.0)
    out["agri_expanded_data_quality_score"] = (
        (out["agri_core_feature_count"] + out["agri_optional_feature_count"])
        / (len(core_score_cols) + len(optional_score_cols))
    ).clip(0.0, 1.0)

    # Initial theory score only. Production blending should live in DBA_scoring.py.
    out["agri_core_balance_score"] = _neutral_fill(
        0.22 * out["agri_usd_score"]
        + 0.08 * out["agri_rates_score"]
        + 0.25 * out["agri_crop_momentum_score"]
        + 0.22 * out["agri_crop_relative_strength_score"]
        + 0.12 * out["agri_broad_commodity_confirmation"]
        + 0.11 * out["agri_seasonality_score"]
    )

    out["agri_expanded_balance_score"] = _neutral_fill(
        0.20 * out["agri_usd_score"]
        + 0.08 * out["agri_rates_score"]
        + 0.24 * out["agri_crop_momentum_score"]
        + 0.21 * out["agri_crop_relative_strength_score"]
        + 0.11 * out["agri_broad_commodity_confirmation"]
        + 0.10 * out["agri_seasonality_score"]
        + 0.06 * out["agri_export_demand_score"]
    )

    return out.sort_values("date").reset_index(drop=True)


# ============================================================
# MONTHLY / SAVE / PIPELINE
# ============================================================

def build_agriculture_features_monthly(features_daily: pd.DataFrame) -> pd.DataFrame:
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


def save_agriculture_features(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
) -> None:
    AGRI_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    daily.to_csv(AGRI_FEATURES_DAILY_PATH, index=False)
    monthly.to_csv(AGRI_FEATURES_MONTHLY_PATH, index=False)

    print(f"\nSaved daily agriculture features to: {AGRI_FEATURES_DAILY_PATH}")
    print(f"Saved monthly agriculture features to: {AGRI_FEATURES_MONTHLY_PATH}")


def run_agriculture_feature_pipeline() -> pd.DataFrame:
    print("\nStarting agriculture feature pipeline.")
    print(f"Project root:            {COMMODITY_ROOT}")
    print(f"Agriculture raw wide:    {AGRI_RAW_WIDE_PATH}")

    daily = build_agriculture_features_daily()
    monthly = build_agriculture_features_monthly(daily)

    save_agriculture_features(daily, monthly)

    print("\nAgriculture feature pipeline complete.")
    print(f"Daily rows:   {len(daily):,}")
    print(f"Monthly rows: {len(monthly):,}")
    print(f"Start date:   {daily['date'].min().date()}")
    print(f"End date:     {daily['date'].max().date()}")

    score_cols = [
        "agri_usd_score",
        "agri_rates_score",
        "agri_crop_momentum_score",
        "agri_crop_relative_strength_score",
        "agri_broad_commodity_confirmation",
        "agri_seasonality_score",
        "agri_export_demand_score",
        "agri_core_balance_score",
        "agri_expanded_balance_score",
        "agri_core_data_quality_score",
        "agri_expanded_data_quality_score",
    ]

    print("\nCore agriculture score columns:")
    print(daily[score_cols].describe().to_string())

    _validate_not_constant(daily, score_cols)

    return daily


if __name__ == "__main__":
    run_agriculture_feature_pipeline()