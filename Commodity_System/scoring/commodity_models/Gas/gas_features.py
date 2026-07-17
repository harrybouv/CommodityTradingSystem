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


# ============================================================
# PATHS
# ============================================================

GAS_PROCESSED_DIR = PROCESSED_DATA_DIR / "gas"

GAS_RAW_WIDE_PATH = GAS_PROCESSED_DIR / "gas_raw_wide.csv"
GAS_FEATURES_DAILY_PATH = GAS_PROCESSED_DIR / "gas_features_daily.csv"
GAS_FEATURES_MONTHLY_PATH = GAS_PROCESSED_DIR / "gas_features_monthly.csv"


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
WEATHER_FFILL_LIMIT = 3
WEEKLY_FFILL_LIMIT = 10
MONTHLY_FFILL_LIMIT = 45

GAS_ASOF_TOLERANCE_DAYS = 10

REQUIRED_COMMODITY_TICKERS = ["UNG", "USO"]

STORAGE_COL = "NW2_EPG0_SWO_R48_BCF"
PRODUCTION_COL = "N9070US2"
LNG_EXPORTS_COL = "N9133US2"

HENRY_HUB_COL = "DHHNGSP"
WTI_COL = "DCOILWTICO"

HDD_COL = "NOAA_UTILITY_GAS_HDD"
CDD_COL = "NOAA_POPULATION_CDD"

UNL_COL = "yf_UNL_adj_close"
YF_USO_COL = "yf_USO_adj_close"
DBC_COL = "yf_DBC_adj_close"


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
    window: int = LOOKBACK_3Y,
    min_periods: int = LOOKBACK_1Y,
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

    return x.rolling(
        window=window,
        min_periods=min_periods,
    ).apply(percentile_rank, raw=True)


def _score_high_is_good(
    s: pd.Series,
    window: int = LOOKBACK_3Y,
    min_periods: int = LOOKBACK_1Y,
) -> pd.Series:
    return _clip01(
        _rolling_percentile(
            s,
            window=window,
            min_periods=min_periods,
        )
    )


def _score_low_is_good(
    s: pd.Series,
    window: int = LOOKBACK_3Y,
    min_periods: int = LOOKBACK_1Y,
) -> pd.Series:
    return _clip01(
        1.0 - _rolling_percentile(
            s,
            window=window,
            min_periods=min_periods,
        )
    )


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

    existing = [
        col for col in columns
        if col in out.columns
    ]

    if existing:
        out[existing] = out[existing].ffill(limit=limit)

    return out


def _mean_available(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    existing = [col for col in cols if col in df.columns]

    if not existing:
        return pd.Series(np.nan, index=df.index)

    return df[existing].mean(axis=1, skipna=True)


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


def _seasonal_history_stats(
    dates: pd.Series,
    values: pd.Series,
    lookback_years: int = 5,
    day_window: int = 14,
    min_obs: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Trailing seasonal mean/std/percentile using prior observations only.

    This is used for gas storage because raw storage is deeply seasonal.
    The current value is compared to previous years' values around the same
    day-of-year, avoiding lookahead.
    """

    dates = pd.to_datetime(dates).reset_index(drop=True)
    values = _safe_numeric(values).reset_index(drop=True)

    means: list[float] = []
    stds: list[float] = []
    percentiles: list[float] = []

    for i, current_date in enumerate(dates):
        current_value = values.iloc[i]

        if pd.isna(current_value):
            means.append(np.nan)
            stds.append(np.nan)
            percentiles.append(np.nan)
            continue

        start_date = current_date - pd.DateOffset(years=lookback_years)

        mask = (
            (dates < current_date)
            & (dates >= start_date)
            & values.notna()
        )

        history_dates = dates[mask]
        history_values = values[mask]

        if history_values.empty:
            means.append(np.nan)
            stds.append(np.nan)
            percentiles.append(np.nan)
            continue

        current_doy = int(current_date.dayofyear)
        history_doy = history_dates.dt.dayofyear.astype(int)

        doy_distance = np.minimum(
            np.abs(history_doy - current_doy),
            366 - np.abs(history_doy - current_doy),
        )

        seasonal_values = history_values[doy_distance <= day_window]

        if len(seasonal_values) < min_obs:
            means.append(np.nan)
            stds.append(np.nan)
            percentiles.append(np.nan)
            continue

        seasonal_values = seasonal_values.astype(float)

        mean = float(seasonal_values.mean())
        std = float(seasonal_values.std(ddof=0))
        percentile = float((seasonal_values <= current_value).mean())

        means.append(mean)
        stds.append(std if std != 0.0 else np.nan)
        percentiles.append(percentile)

    return (
        pd.Series(means, index=values.index),
        pd.Series(stds, index=values.index),
        pd.Series(percentiles, index=values.index),
    )


# ============================================================
# LOADERS
# ============================================================

def load_gas_raw_wide(path: Path = GAS_RAW_WIDE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Gas raw wide file not found: {path}. "
            "Run scoring/commodity_models/NaturalGas/gas_data.py first."
        )

    raw = pd.read_csv(path)

    if "date" not in raw.columns:
        raise ValueError(f"Gas raw wide file missing date column: {path}")

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
        STORAGE_COL,
        PRODUCTION_COL,
        HENRY_HUB_COL,
        HDD_COL,
        CDD_COL,
        UNL_COL,
    ]

    missing = [
        col for col in required_cols
        if col not in raw.columns
    ]

    if missing:
        raise ValueError(
            f"Gas raw wide missing required columns: {missing}. "
            f"Available columns: {list(raw.columns)}"
        )

    return raw


def load_commodity_price_matrix(path: Path = PRICE_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Commodity price data not found: {path}. Run data.py first."
        )

    prices = pd.read_csv(path)

    required_cols = ["date", "ticker", "adj_close"]
    missing = [
        col for col in required_cols
        if col not in prices.columns
    ]

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
            f"Gas feature pipeline requires {REQUIRED_COMMODITY_TICKERS}, "
            f"but missing: {missing_tickers}. "
            f"Available tickers: {list(matrix.columns)}"
        )

    return matrix[REQUIRED_COMMODITY_TICKERS].copy()


# ============================================================
# AS-OF PREPARATION
# ============================================================

def prepare_gas_asof_inputs(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a daily/as-of table from mixed-frequency gas inputs.

    The date column in gas_raw_wide is already the estimated availability date,
    not the original economic period date. That preserves release-lag logic
    from gas_data.py.
    """

    out = raw.copy().sort_values("date").reset_index(drop=True)

    daily_cols = [
        HENRY_HUB_COL,
        WTI_COL,
        UNL_COL,
        YF_USO_COL,
        DBC_COL,
    ]

    weather_cols = [
        HDD_COL,
        CDD_COL,
    ]

    weekly_cols = [
        STORAGE_COL,
    ]

    monthly_cols = [
        PRODUCTION_COL,
        LNG_EXPORTS_COL,
    ]

    out = _ffill_existing(out, daily_cols, DAILY_FFILL_LIMIT)
    out = _ffill_existing(out, weather_cols, WEATHER_FFILL_LIMIT)
    out = _ffill_existing(out, weekly_cols, WEEKLY_FFILL_LIMIT)
    out = _ffill_existing(out, monthly_cols, MONTHLY_FFILL_LIMIT)

    return out


def merge_gas_raw_to_market_dates(
    market_dates: pd.Index,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if raw is None:
        raw = load_gas_raw_wide()

    prepared = prepare_gas_asof_inputs(raw)

    left = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series(market_dates)).sort_values().values
        }
    )

    right = prepared.copy()
    right["date"] = pd.to_datetime(right["date"])

    left = left.sort_values("date").reset_index(drop=True)
    right = right.sort_values("date").reset_index(drop=True)

    out = pd.merge_asof(
        left,
        right,
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=GAS_ASOF_TOLERANCE_DAYS),
    )

    return out.sort_values("date").reset_index(drop=True)


# ============================================================
# FEATURE BUILDING
# ============================================================

def build_gas_features_daily(
    commodity_prices: pd.DataFrame | None = None,
    gas_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if commodity_prices is None:
        commodity_prices = load_commodity_price_matrix()

    if gas_raw is None:
        gas_raw = load_gas_raw_wide()

    commodity_prices = commodity_prices.copy().sort_index()

    common_dates = commodity_prices.index

    if len(common_dates) == 0:
        raise ValueError("No commodity price dates available for gas features.")

    gas_asof = merge_gas_raw_to_market_dates(
        market_dates=common_dates,
        raw=gas_raw,
    )

    gas_asof["date"] = pd.to_datetime(gas_asof["date"])

    gas_asof = (
        gas_asof
        .sort_values("date")
        .set_index("date")
        .reindex(common_dates)
    )

    gas_asof.index.name = None

    ung = _safe_numeric(commodity_prices["UNG"])
    uso = _safe_numeric(commodity_prices["USO"])

    storage = _safe_numeric(gas_asof[STORAGE_COL])
    production = _safe_numeric(gas_asof[PRODUCTION_COL])
    henry_hub = _safe_numeric(gas_asof[HENRY_HUB_COL])
    hdd = _safe_numeric(gas_asof[HDD_COL])
    cdd = _safe_numeric(gas_asof[CDD_COL])
    unl = _safe_numeric(gas_asof[UNL_COL])

    if LNG_EXPORTS_COL in gas_asof.columns:
        lng_exports = _safe_numeric(gas_asof[LNG_EXPORTS_COL])
    else:
        lng_exports = pd.Series(np.nan, index=gas_asof.index)

    if WTI_COL in gas_asof.columns:
        wti = _safe_numeric(gas_asof[WTI_COL])
    else:
        wti = pd.Series(np.nan, index=gas_asof.index)

    if YF_USO_COL in gas_asof.columns:
        yf_uso = _safe_numeric(gas_asof[YF_USO_COL])
    else:
        yf_uso = uso.copy()

    if DBC_COL in gas_asof.columns:
        dbc = _safe_numeric(gas_asof[DBC_COL])
    else:
        dbc = pd.Series(np.nan, index=gas_asof.index)

    out = pd.DataFrame(index=common_dates)
    out.index.name = None
    out["date"] = common_dates

    # ------------------------------
    # Raw gas inputs
    # ------------------------------

    out["gas_storage_bcf"] = storage.values
    out["gas_us_dry_production"] = production.values
    out["gas_lng_exports"] = lng_exports.values
    out["henry_hub_spot_price"] = henry_hub.values
    out["wti_spot_price"] = wti.values
    out["hdd_utility_gas"] = hdd.values
    out["cdd_population"] = cdd.values
    out["unl_price"] = unl.values
    out["dbc_price"] = dbc.values

    # ------------------------------
    # UNG own diagnostics
    # ------------------------------

    out["ung_return_1m"] = _return(ung, LOOKBACK_1M).values
    out["ung_return_3m"] = _return(ung, LOOKBACK_3M).values
    out["ung_return_6m"] = _return(ung, LOOKBACK_6M).values
    out["ung_trend_score"] = _trend_score(ung).values

    out["ung_momentum_score"] = _neutral_fill(
        0.50 * _score_high_is_good(_return(ung, LOOKBACK_3M))
        + 0.50 * _score_high_is_good(_return(ung, LOOKBACK_6M))
    ).values

    # ------------------------------
    # Henry Hub diagnostics
    # ------------------------------

    out["henry_hub_return_1m"] = _return(henry_hub, LOOKBACK_1M).values
    out["henry_hub_return_3m"] = _return(henry_hub, LOOKBACK_3M).values
    out["henry_hub_return_6m"] = _return(henry_hub, LOOKBACK_6M).values
    out["henry_hub_trend_score"] = _trend_score(henry_hub).values
    out["henry_hub_z_3y"] = _rolling_z_score(henry_hub).values

    # ------------------------------
    # 1. Weather demand
    # Heating degree days matter most in winter.
    # Cooling degree days matter most in summer through power burn.
    # ------------------------------

    out["hdd_7d"] = hdd.rolling(window=7, min_periods=3).sum().values
    out["hdd_14d"] = hdd.rolling(window=14, min_periods=7).sum().values
    out["hdd_30d"] = hdd.rolling(window=30, min_periods=14).sum().values

    out["cdd_7d"] = cdd.rolling(window=7, min_periods=3).sum().values
    out["cdd_14d"] = cdd.rolling(window=14, min_periods=7).sum().values
    out["cdd_30d"] = cdd.rolling(window=30, min_periods=14).sum().values

    out["weather_demand_14d"] = out["hdd_14d"] + out["cdd_14d"]
    out["weather_demand_30d"] = out["hdd_30d"] + out["cdd_30d"]
    out["weather_demand_change_4w"] = _change(out["weather_demand_30d"], LOOKBACK_1M).values

    months = pd.Series(common_dates.month, index=common_dates)

    hdd_weight = pd.Series(0.25, index=common_dates)
    cdd_weight = pd.Series(0.25, index=common_dates)

    winter = months.isin([11, 12, 1, 2, 3])
    summer = months.isin([6, 7, 8, 9])
    shoulder = months.isin([4, 5, 10])

    hdd_weight.loc[winter] = 1.00
    cdd_weight.loc[winter] = 0.10

    hdd_weight.loc[summer] = 0.10
    cdd_weight.loc[summer] = 1.00

    hdd_weight.loc[shoulder] = 0.45
    cdd_weight.loc[shoulder] = 0.45

    seasonal_weather_demand = (
        hdd_weight.reset_index(drop=True) * pd.Series(out["hdd_14d"]).reset_index(drop=True)
        + cdd_weight.reset_index(drop=True) * pd.Series(out["cdd_14d"]).reset_index(drop=True)
    )
    seasonal_weather_demand.index = common_dates

    out["seasonal_weather_demand_14d"] = seasonal_weather_demand.values
    out["seasonal_weather_demand_z_3y"] = _rolling_z_score(seasonal_weather_demand).values
    out["seasonal_weather_demand_change_4w"] = _change(
        seasonal_weather_demand,
        LOOKBACK_1M,
    ).values

    weather_level_score = _score_high_is_good(seasonal_weather_demand)
    weather_change_score = _score_high_is_good(
        _change(seasonal_weather_demand, LOOKBACK_1M)
    )

    out["gas_weather_demand_score"] = _neutral_fill(
        0.70 * weather_level_score
        + 0.30 * weather_change_score
    ).values

    # ------------------------------
    # 2. Storage tightness
    # Storage is deeply seasonal, so compare it with prior years around the
    # same day-of-year.
    # ------------------------------

    seasonal_mean, seasonal_std, seasonal_pct = _seasonal_history_stats(
        dates=pd.Series(common_dates),
        values=storage.reset_index(drop=True),
        lookback_years=5,
        day_window=14,
        min_obs=20,
    )

    seasonal_mean.index = common_dates
    seasonal_std.index = common_dates
    seasonal_pct.index = common_dates

    out["gas_storage_seasonal_avg_5y"] = seasonal_mean.values
    out["gas_storage_vs_5y_seasonal_avg_bcf"] = (storage - seasonal_mean).values
    out["gas_storage_vs_5y_seasonal_avg_pct"] = (
        (storage - seasonal_mean) / seasonal_mean.replace(0.0, np.nan)
    ).values
    out["gas_storage_seasonal_z_5y"] = (
        (storage - seasonal_mean) / seasonal_std.replace(0.0, np.nan)
    ).values
    out["gas_storage_seasonal_percentile_5y"] = seasonal_pct.values

    out["gas_storage_z_3y"] = _rolling_z_score(storage).values
    out["gas_storage_percentile_3y"] = _rolling_percentile(storage).values

    seasonal_tightness_score = _clip01(1.0 - seasonal_pct)
    raw_level_tightness_score = _score_low_is_good(storage)

    out["gas_storage_tightness_score"] = _neutral_fill(
        0.75 * seasonal_tightness_score
        + 0.25 * raw_level_tightness_score
    ).values

    # ------------------------------
    # 3. Storage momentum / injection-withdrawal pressure
    # Falling storage or injections smaller than normal are bullish.
    # ------------------------------

    out["gas_storage_change_1w"] = _change(storage, 5).values
    out["gas_storage_change_4w"] = _change(storage, LOOKBACK_1M).values
    out["gas_storage_change_13w"] = _change(storage, LOOKBACK_3M).values
    out["gas_storage_yoy_change"] = _change(storage, LOOKBACK_1Y).values

    change_4w_score = _score_low_is_good(_change(storage, LOOKBACK_1M))
    change_13w_score = _score_low_is_good(_change(storage, LOOKBACK_3M))
    yoy_tightening_score = _score_low_is_good(_change(storage, LOOKBACK_1Y))

    out["gas_storage_momentum_score"] = _neutral_fill(
        0.45 * change_4w_score
        + 0.35 * change_13w_score
        + 0.20 * yoy_tightening_score
    ).values

    # ------------------------------
    # 4. Curve / roll proxy
    # UNG vs UNL relative strength proxies front-end curve/roll conditions.
    # Strong UNG relative to UNL = favourable for UNG.
    # ------------------------------

    ung_unl_ratio = ung.reset_index(drop=True) / unl.reset_index(drop=True)
    ung_unl_ratio.index = common_dates

    out["ung_unl_ratio"] = ung_unl_ratio.values
    out["ung_unl_ratio_return_1m"] = _return(ung_unl_ratio, LOOKBACK_1M).values
    out["ung_unl_ratio_return_3m"] = _return(ung_unl_ratio, LOOKBACK_3M).values
    out["ung_unl_ratio_return_6m"] = _return(ung_unl_ratio, LOOKBACK_6M).values
    out["ung_unl_ratio_z_3y"] = _rolling_z_score(ung_unl_ratio).values
    out["ung_unl_ratio_trend_score"] = _trend_score(ung_unl_ratio).values

    ratio_1m_score = _score_high_is_good(_return(ung_unl_ratio, LOOKBACK_1M))
    ratio_3m_score = _score_high_is_good(_return(ung_unl_ratio, LOOKBACK_3M))

    out["gas_curve_roll_score"] = _neutral_fill(
        0.45 * _trend_score(ung_unl_ratio)
        + 0.35 * ratio_3m_score
        + 0.20 * ratio_1m_score
    ).values

    # ------------------------------
    # 5. Production / supply pressure
    # Lower or falling production is supportive. Rising production is bearish.
    # ------------------------------

    out["gas_production_change_1m"] = _change(production, LOOKBACK_1M).values
    out["gas_production_change_3m"] = _change(production, LOOKBACK_3M).values
    out["gas_production_yoy_change"] = _change(production, LOOKBACK_1Y).values
    out["gas_production_z_3y"] = _rolling_z_score(production).values

    production_level_score = _score_low_is_good(production)
    production_3m_score = _score_low_is_good(_change(production, LOOKBACK_3M))
    production_yoy_score = _score_low_is_good(_change(production, LOOKBACK_1Y))

    out["gas_supply_pressure_score"] = _neutral_fill(
        0.45 * production_level_score
        + 0.35 * production_3m_score
        + 0.20 * production_yoy_score
    ).values

    # ------------------------------
    # 6. LNG export demand
    # Higher exports can tighten the domestic gas market.
    # Built as a diagnostic / optional later score.
    # ------------------------------

    out["gas_lng_exports_change_1m"] = _change(lng_exports, LOOKBACK_1M).values
    out["gas_lng_exports_change_3m"] = _change(lng_exports, LOOKBACK_3M).values
    out["gas_lng_exports_yoy_change"] = _change(lng_exports, LOOKBACK_1Y).values
    out["gas_lng_exports_z_3y"] = _rolling_z_score(lng_exports).values

    lng_level_score = _score_high_is_good(lng_exports)
    lng_3m_score = _score_high_is_good(_change(lng_exports, LOOKBACK_3M))
    lng_yoy_score = _score_high_is_good(_change(lng_exports, LOOKBACK_1Y))

    out["gas_lng_export_demand_score"] = _neutral_fill(
        0.45 * lng_level_score
        + 0.35 * lng_3m_score
        + 0.20 * lng_yoy_score
    ).values

    # ------------------------------
    # 7. Oil relative value / fuel-cost relationship
    # Gas cheap versus oil can be supportive, but this is secondary.
    # Convert WTI $/bbl into approximate $/MMBtu equivalent using 5.8 MMBtu/bbl.
    # ------------------------------

    wti_per_mmbtu = wti / 5.8
    gas_oil_ratio = henry_hub / wti_per_mmbtu.replace(0.0, np.nan)

    out["wti_per_mmbtu"] = wti_per_mmbtu.values
    out["gas_oil_ratio"] = gas_oil_ratio.values
    out["gas_oil_ratio_z_3y"] = _rolling_z_score(gas_oil_ratio).values
    out["gas_oil_ratio_return_3m"] = _return(gas_oil_ratio, LOOKBACK_3M).values

    gas_cheap_vs_oil_score = _score_low_is_good(gas_oil_ratio)
    gas_oil_ratio_falling_score = _score_low_is_good(_return(gas_oil_ratio, LOOKBACK_3M))

    out["gas_oil_relative_value_score"] = _neutral_fill(
        0.70 * gas_cheap_vs_oil_score
        + 0.30 * gas_oil_ratio_falling_score
    ).values

    # ------------------------------
    # 8. Broad energy / commodity confirmation
    # Optional diagnostic. Do not overweight in gas V1.
    # ------------------------------

    out["uso_return_1m"] = _return(uso, LOOKBACK_1M).values
    out["uso_return_3m"] = _return(uso, LOOKBACK_3M).values
    out["uso_return_6m"] = _return(uso, LOOKBACK_6M).values
    out["uso_trend_score"] = _trend_score(uso).values

    out["dbc_return_1m"] = _return(dbc, LOOKBACK_1M).values
    out["dbc_return_3m"] = _return(dbc, LOOKBACK_3M).values
    out["dbc_return_6m"] = _return(dbc, LOOKBACK_6M).values
    out["dbc_trend_score"] = _trend_score(dbc).values

    uso_confirmation_score = _neutral_fill(
        0.50 * _trend_score(uso)
        + 0.30 * _score_high_is_good(_return(uso, LOOKBACK_3M))
        + 0.20 * _score_high_is_good(_return(uso, LOOKBACK_6M))
    )

    dbc_confirmation_score = _neutral_fill(
        0.50 * _trend_score(dbc)
        + 0.30 * _score_high_is_good(_return(dbc, LOOKBACK_3M))
        + 0.20 * _score_high_is_good(_return(dbc, LOOKBACK_6M))
    )

    out["gas_energy_confirmation_score"] = _neutral_fill(
        0.60 * uso_confirmation_score
        + 0.40 * dbc_confirmation_score
    ).values

    # ------------------------------
    # Feature availability / core score
    # ------------------------------

    core_score_cols = [
        "gas_weather_demand_score",
        "gas_storage_tightness_score",
        "gas_storage_momentum_score",
        "gas_curve_roll_score",
        "gas_supply_pressure_score",
        "gas_oil_relative_value_score",
    ]

    optional_score_cols = [
        "gas_lng_export_demand_score",
        "gas_energy_confirmation_score",
    ]

    out["gas_core_feature_count"] = out[core_score_cols].notna().sum(axis=1)
    out["gas_core_data_quality_score"] = (
        out["gas_core_feature_count"] / len(core_score_cols)
    ).clip(0.0, 1.0)

    # Initial theory score. The scoring module can still use individual
    # components directly, but this is useful for diagnostics and quick tests.
    out["gas_balance_score"] = _neutral_fill(
        0.25 * out["gas_weather_demand_score"]
        + 0.25 * out["gas_storage_tightness_score"]
        + 0.20 * out["gas_storage_momentum_score"]
        + 0.15 * out["gas_curve_roll_score"]
        + 0.10 * out["gas_supply_pressure_score"]
        + 0.05 * out["gas_oil_relative_value_score"]
    )

    # Optional broader diagnostic score, not intended as production default.
    out["gas_expanded_balance_score"] = _neutral_fill(
        0.20 * out["gas_weather_demand_score"]
        + 0.22 * out["gas_storage_tightness_score"]
        + 0.18 * out["gas_storage_momentum_score"]
        + 0.14 * out["gas_curve_roll_score"]
        + 0.10 * out["gas_supply_pressure_score"]
        + 0.06 * out["gas_lng_export_demand_score"]
        + 0.05 * out["gas_oil_relative_value_score"]
        + 0.05 * out["gas_energy_confirmation_score"]
    )

    keep_cols = [
        "date",

        # Raw inputs
        "gas_storage_bcf",
        "gas_us_dry_production",
        "gas_lng_exports",
        "henry_hub_spot_price",
        "wti_spot_price",
        "hdd_utility_gas",
        "cdd_population",
        "unl_price",
        "dbc_price",

        # UNG / Henry Hub diagnostics
        "ung_return_1m",
        "ung_return_3m",
        "ung_return_6m",
        "ung_trend_score",
        "ung_momentum_score",
        "henry_hub_return_1m",
        "henry_hub_return_3m",
        "henry_hub_return_6m",
        "henry_hub_trend_score",
        "henry_hub_z_3y",

        # Weather
        "hdd_7d",
        "hdd_14d",
        "hdd_30d",
        "cdd_7d",
        "cdd_14d",
        "cdd_30d",
        "weather_demand_14d",
        "weather_demand_30d",
        "weather_demand_change_4w",
        "seasonal_weather_demand_14d",
        "seasonal_weather_demand_z_3y",
        "seasonal_weather_demand_change_4w",
        "gas_weather_demand_score",

        # Storage level
        "gas_storage_seasonal_avg_5y",
        "gas_storage_vs_5y_seasonal_avg_bcf",
        "gas_storage_vs_5y_seasonal_avg_pct",
        "gas_storage_seasonal_z_5y",
        "gas_storage_seasonal_percentile_5y",
        "gas_storage_z_3y",
        "gas_storage_percentile_3y",
        "gas_storage_tightness_score",

        # Storage change
        "gas_storage_change_1w",
        "gas_storage_change_4w",
        "gas_storage_change_13w",
        "gas_storage_yoy_change",
        "gas_storage_momentum_score",

        # Curve / roll
        "ung_unl_ratio",
        "ung_unl_ratio_return_1m",
        "ung_unl_ratio_return_3m",
        "ung_unl_ratio_return_6m",
        "ung_unl_ratio_z_3y",
        "ung_unl_ratio_trend_score",
        "gas_curve_roll_score",

        # Production / supply
        "gas_production_change_1m",
        "gas_production_change_3m",
        "gas_production_yoy_change",
        "gas_production_z_3y",
        "gas_supply_pressure_score",

        # LNG
        "gas_lng_exports_change_1m",
        "gas_lng_exports_change_3m",
        "gas_lng_exports_yoy_change",
        "gas_lng_exports_z_3y",
        "gas_lng_export_demand_score",

        # Oil / energy confirmation
        "wti_per_mmbtu",
        "gas_oil_ratio",
        "gas_oil_ratio_z_3y",
        "gas_oil_ratio_return_3m",
        "gas_oil_relative_value_score",
        "uso_return_1m",
        "uso_return_3m",
        "uso_return_6m",
        "uso_trend_score",
        "dbc_return_1m",
        "dbc_return_3m",
        "dbc_return_6m",
        "dbc_trend_score",
        "gas_energy_confirmation_score",

        # Aggregate diagnostics
        "gas_core_feature_count",
        "gas_core_data_quality_score",
        "gas_balance_score",
        "gas_expanded_balance_score",
    ]

    existing_keep_cols = [
        col for col in keep_cols
        if col in out.columns
    ]

    out = out[existing_keep_cols].copy()

    return out.sort_values("date").reset_index(drop=True)


# ============================================================
# SAVE / RUN
# ============================================================

def save_gas_features(
    daily: pd.DataFrame,
    daily_path: Path = GAS_FEATURES_DAILY_PATH,
    monthly_path: Path = GAS_FEATURES_MONTHLY_PATH,
) -> None:
    GAS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    daily.to_csv(daily_path, index=False)
    print(f"Saved daily gas features to: {daily_path}")

    monthly = (
        daily
        .set_index("date")
        .resample("ME")
        .last()
        .reset_index()
    )

    monthly.to_csv(monthly_path, index=False)
    print(f"Saved monthly gas features to: {monthly_path}")


def run_gas_feature_pipeline() -> pd.DataFrame:
    print("\n========== GAS FEATURE PIPELINE ==========")

    daily = build_gas_features_daily()
    save_gas_features(daily)

    score_cols = [
        "gas_weather_demand_score",
        "gas_storage_tightness_score",
        "gas_storage_momentum_score",
        "gas_curve_roll_score",
        "gas_supply_pressure_score",
        "gas_lng_export_demand_score",
        "gas_oil_relative_value_score",
        "gas_energy_confirmation_score",
        "gas_balance_score",
        "gas_expanded_balance_score",
        "gas_core_data_quality_score",
    ]

    _validate_not_constant(daily, score_cols)

    print("\nGas feature pipeline complete.")
    print(f"Daily rows: {len(daily):,}")
    print(f"Start date: {daily['date'].min().date()}")
    print(f"End date: {daily['date'].max().date()}")
    print(f"Columns: {list(daily.columns)}")

    return daily


if __name__ == "__main__":
    run_gas_feature_pipeline()
