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

OIL_PROCESSED_DIR = PROCESSED_DATA_DIR / "oil"

OIL_RAW_WIDE_PATH = OIL_PROCESSED_DIR / "oil_raw_wide.csv"
OIL_FEATURES_DAILY_PATH = OIL_PROCESSED_DIR / "oil_features_daily.csv"
OIL_FEATURES_MONTHLY_PATH = OIL_PROCESSED_DIR / "oil_features_monthly.csv"


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

OIL_ASOF_TOLERANCE_DAYS = 10

REQUIRED_COMMODITY_TICKERS = ["USO", "CPER"]
REQUIRED_MACRO_TICKERS = ["UUP", "DBC", "SPY", "^VIX"]

CRUDE_STOCKS_COL = "WCESTUS1"
CUSHING_STOCKS_COL = "W_EPC0_SAX_YCUOK_MBBL"
CRUDE_PRODUCTION_COL = "WCRFPUS2"
REFINERY_UTILISATION_COL = "WPULEUS3"
WTI_SPOT_COL = "DCOILWTICO"
INDPRO_COL = "INDPRO"
USL_COL = "yf_USL_adj_close"
BNO_COL = "yf_BNO_adj_close"


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

def load_oil_raw_wide(path: Path = OIL_RAW_WIDE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Oil raw wide file not found: {path}. "
            "Run scoring/commodity_models/Oil/oil_data.py first."
        )

    raw = pd.read_csv(path)

    if "date" not in raw.columns:
        raise ValueError(f"Oil raw wide file missing date column: {path}")

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
        CRUDE_STOCKS_COL,
        CUSHING_STOCKS_COL,
        CRUDE_PRODUCTION_COL,
        REFINERY_UTILISATION_COL,
        WTI_SPOT_COL,
        USL_COL,
    ]

    missing = [
        col for col in required_cols
        if col not in raw.columns
    ]

    if missing:
        raise ValueError(
            f"Oil raw wide missing required columns: {missing}. "
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
            f"Oil feature pipeline requires {REQUIRED_COMMODITY_TICKERS}, "
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
    missing = [
        col for col in required_cols
        if col not in prices.columns
    ]

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
            f"Oil feature pipeline requires macro tickers {REQUIRED_MACRO_TICKERS}, "
            f"but missing: {missing_tickers}. "
            f"Available tickers: {list(matrix.columns)}"
        )

    return matrix[REQUIRED_MACRO_TICKERS].copy()


# ============================================================
# AS-OF PREPARATION
# ============================================================

def prepare_oil_asof_inputs(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a daily/as-of table from mixed-frequency oil inputs.

    The date column in oil_raw_wide is already the estimated availability date,
    not the original economic period date. That preserves the release-lag logic
    from oil_data.py.
    """

    out = raw.copy().sort_values("date").reset_index(drop=True)

    daily_cols = [
        WTI_SPOT_COL,
        USL_COL,
        BNO_COL,
    ]

    weekly_cols = [
        CRUDE_STOCKS_COL,
        CUSHING_STOCKS_COL,
        CRUDE_PRODUCTION_COL,
        REFINERY_UTILISATION_COL,
    ]

    monthly_cols = [
        INDPRO_COL,
    ]

    out = _ffill_existing(out, daily_cols, DAILY_FFILL_LIMIT)
    out = _ffill_existing(out, weekly_cols, WEEKLY_FFILL_LIMIT)
    out = _ffill_existing(out, monthly_cols, MONTHLY_FFILL_LIMIT)

    return out


def merge_oil_raw_to_market_dates(
    market_dates: pd.Index,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if raw is None:
        raw = load_oil_raw_wide()

    prepared = prepare_oil_asof_inputs(raw)

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
        tolerance=pd.Timedelta(days=OIL_ASOF_TOLERANCE_DAYS),
    )

    return out.sort_values("date").reset_index(drop=True)


# ============================================================
# FEATURE BUILDING
# ============================================================

def build_oil_features_daily(
    commodity_prices: pd.DataFrame | None = None,
    macro_prices: pd.DataFrame | None = None,
    oil_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if commodity_prices is None:
        commodity_prices = load_commodity_price_matrix()

    if macro_prices is None:
        macro_prices = load_macro_price_matrix()

    if oil_raw is None:
        oil_raw = load_oil_raw_wide()

    commodity_prices = commodity_prices.copy().sort_index()
    macro_prices = macro_prices.copy().sort_index()

    common_dates = commodity_prices.index.intersection(macro_prices.index)

    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between commodity and macro price data.")

    commodity_prices = commodity_prices.loc[common_dates]
    macro_prices = macro_prices.loc[common_dates]

    oil_asof = merge_oil_raw_to_market_dates(
        market_dates=common_dates,
        raw=oil_raw,
    )

    oil_asof["date"] = pd.to_datetime(oil_asof["date"])

    oil_asof = (
        oil_asof
        .sort_values("date")
        .set_index("date")
        .reindex(common_dates)
    )

    oil_asof.index.name = None

    uso = _safe_numeric(commodity_prices["USO"])
    cper = _safe_numeric(commodity_prices["CPER"])

    uup = _safe_numeric(macro_prices["UUP"])
    dbc = _safe_numeric(macro_prices["DBC"])
    spy = _safe_numeric(macro_prices["SPY"])
    vix = _safe_numeric(macro_prices["^VIX"])

    out = pd.DataFrame(index=common_dates)
    out.index.name = None
    out["date"] = common_dates

    # ------------------------------
    # Raw oil/economic inputs
    # ------------------------------

    crude_stocks = _safe_numeric(oil_asof[CRUDE_STOCKS_COL])
    cushing_stocks = _safe_numeric(oil_asof[CUSHING_STOCKS_COL])
    production = _safe_numeric(oil_asof[CRUDE_PRODUCTION_COL])
    refinery_utilisation = _safe_numeric(oil_asof[REFINERY_UTILISATION_COL])
    wti_spot = _safe_numeric(oil_asof[WTI_SPOT_COL])
    usl = _safe_numeric(oil_asof[USL_COL])

    if BNO_COL in oil_asof.columns:
        bno = _safe_numeric(oil_asof[BNO_COL])
    else:
        bno = pd.Series(np.nan, index=oil_asof.index)

    if INDPRO_COL in oil_asof.columns:
        indpro = _safe_numeric(oil_asof[INDPRO_COL])
    else:
        indpro = pd.Series(np.nan, index=oil_asof.index)

    out["us_crude_stocks_ex_spr"] = crude_stocks.values
    out["cushing_crude_stocks"] = cushing_stocks.values
    out["us_crude_production"] = production.values
    out["refinery_utilisation"] = refinery_utilisation.values
    out["wti_spot_price"] = wti_spot.values
    out["usl_price"] = usl.values
    out["bno_price"] = bno.values
    out["indpro"] = indpro.values

    # ------------------------------
    # USO own diagnostics
    # ------------------------------

    out["uso_return_1m"] = _return(uso, LOOKBACK_1M).values
    out["uso_return_3m"] = _return(uso, LOOKBACK_3M).values
    out["uso_return_6m"] = _return(uso, LOOKBACK_6M).values
    out["uso_trend_score"] = _trend_score(uso).values

    out["uso_momentum_score"] = _neutral_fill(
        0.50 * _score_high_is_good(_return(uso, LOOKBACK_3M))
        + 0.50 * _score_high_is_good(_return(uso, LOOKBACK_6M))
    ).values

    # ------------------------------
    # WTI spot diagnostics
    # ------------------------------

    out["wti_return_1m"] = _return(wti_spot, LOOKBACK_1M).values
    out["wti_return_3m"] = _return(wti_spot, LOOKBACK_3M).values
    out["wti_return_6m"] = _return(wti_spot, LOOKBACK_6M).values
    out["wti_trend_score"] = _trend_score(wti_spot).values

    # ------------------------------
    # 1. Inventory tightness
    # Low/falling commercial crude inventories are bullish.
    # ------------------------------

    out["crude_stocks_change_1m"] = _change(crude_stocks, LOOKBACK_1M).values
    out["crude_stocks_change_3m"] = _change(crude_stocks, LOOKBACK_3M).values
    out["crude_stocks_z_3y"] = _rolling_z_score(crude_stocks).values

    crude_level_score = _score_low_is_good(crude_stocks)
    crude_change_1m_score = _score_low_is_good(_change(crude_stocks, LOOKBACK_1M))
    crude_change_3m_score = _score_low_is_good(_change(crude_stocks, LOOKBACK_3M))

    out["oil_inventory_tightness_score"] = _neutral_fill(
        0.50 * crude_level_score
        + 0.30 * crude_change_1m_score
        + 0.20 * crude_change_3m_score
    ).values

    # ------------------------------
    # 2. Cushing tightness
    # Cushing is especially relevant for WTI.
    # ------------------------------

    out["cushing_stocks_change_1m"] = _change(cushing_stocks, LOOKBACK_1M).values
    out["cushing_stocks_change_3m"] = _change(cushing_stocks, LOOKBACK_3M).values
    out["cushing_stocks_z_3y"] = _rolling_z_score(cushing_stocks).values

    cushing_level_score = _score_low_is_good(cushing_stocks)
    cushing_change_1m_score = _score_low_is_good(_change(cushing_stocks, LOOKBACK_1M))
    cushing_change_3m_score = _score_low_is_good(_change(cushing_stocks, LOOKBACK_3M))

    out["oil_cushing_tightness_score"] = _neutral_fill(
        0.60 * cushing_level_score
        + 0.25 * cushing_change_1m_score
        + 0.15 * cushing_change_3m_score
    ).values

    # ------------------------------
    # 3. Futures curve / roll proxy
    # USO vs USL relative strength proxies front-end curve/roll conditions.
    # Strong USO relative to USL = favourable for USO.
    # ------------------------------

    uso_usl_ratio = uso.reset_index(drop=True) / usl.reset_index(drop=True)
    uso_usl_ratio.index = common_dates

    out["uso_usl_ratio"] = uso_usl_ratio.values
    out["uso_usl_ratio_return_1m"] = _return(uso_usl_ratio, LOOKBACK_1M).values
    out["uso_usl_ratio_return_3m"] = _return(uso_usl_ratio, LOOKBACK_3M).values
    out["uso_usl_ratio_return_6m"] = _return(uso_usl_ratio, LOOKBACK_6M).values
    out["uso_usl_ratio_z_3y"] = _rolling_z_score(uso_usl_ratio).values
    out["uso_usl_ratio_trend_score"] = _trend_score(uso_usl_ratio).values

    ratio_1m_score = _score_high_is_good(_return(uso_usl_ratio, LOOKBACK_1M))
    ratio_3m_score = _score_high_is_good(_return(uso_usl_ratio, LOOKBACK_3M))

    out["oil_curve_roll_score"] = _neutral_fill(
        0.45 * _trend_score(uso_usl_ratio)
        + 0.35 * ratio_3m_score
        + 0.20 * ratio_1m_score
    ).values

    # Optional Brent/US oil relative diagnostics. Not used in V1 score.
    bno_uso_ratio = bno.reset_index(drop=True) / uso.reset_index(drop=True)
    bno_uso_ratio.index = common_dates

    out["bno_uso_ratio"] = bno_uso_ratio.values
    out["bno_uso_ratio_return_3m"] = _return(bno_uso_ratio, LOOKBACK_3M).values

    # ------------------------------
    # 4. Production / refinery usage
    # Falling production pressure + strong refinery utilisation = supportive balance.
    # ------------------------------

    out["production_change_1m"] = _change(production, LOOKBACK_1M).values
    out["production_change_3m"] = _change(production, LOOKBACK_3M).values
    out["production_z_3y"] = _rolling_z_score(production).values

    out["refinery_utilisation_change_1m"] = _change(refinery_utilisation, LOOKBACK_1M).values
    out["refinery_utilisation_change_3m"] = _change(refinery_utilisation, LOOKBACK_3M).values
    out["refinery_utilisation_z_3y"] = _rolling_z_score(refinery_utilisation).values

    production_relief_score = _neutral_fill(
        0.60 * _score_low_is_good(_change(production, LOOKBACK_3M))
        + 0.40 * _score_low_is_good(_change(production, LOOKBACK_1M))
    )

    refinery_usage_score = _neutral_fill(
        0.60 * _score_high_is_good(refinery_utilisation)
        + 0.40 * _score_high_is_good(_change(refinery_utilisation, LOOKBACK_1M))
    )

    out["oil_production_relief_score"] = production_relief_score.values
    out["oil_refinery_usage_score"] = refinery_usage_score.values

    out["oil_supply_refinery_score"] = _neutral_fill(
        0.45 * production_relief_score
        + 0.55 * refinery_usage_score
    ).values

    # ------------------------------
    # 5. Global performance / oil demand
    # DBC, SPY, CPER, INDPRO, low VIX.
    # Keep this modest because the main macro layer already captures broad regime.
    # ------------------------------

    out["dbc_return_1m"] = _return(dbc, LOOKBACK_1M).values
    out["dbc_return_3m"] = _return(dbc, LOOKBACK_3M).values
    out["dbc_return_6m"] = _return(dbc, LOOKBACK_6M).values
    out["dbc_trend_score"] = _trend_score(dbc).values

    dbc_score = _neutral_fill(
        0.50 * _trend_score(dbc)
        + 0.30 * _score_high_is_good(_return(dbc, LOOKBACK_3M))
        + 0.20 * _score_high_is_good(_return(dbc, LOOKBACK_6M))
    )

    out["spy_return_1m"] = _return(spy, LOOKBACK_1M).values
    out["spy_return_3m"] = _return(spy, LOOKBACK_3M).values
    out["spy_return_6m"] = _return(spy, LOOKBACK_6M).values
    out["spy_trend_score"] = _trend_score(spy).values

    spy_score = _neutral_fill(
        0.55 * _trend_score(spy)
        + 0.45 * _score_high_is_good(_return(spy, LOOKBACK_3M))
    )

    out["cper_return_1m"] = _return(cper, LOOKBACK_1M).values
    out["cper_return_3m"] = _return(cper, LOOKBACK_3M).values
    out["cper_return_6m"] = _return(cper, LOOKBACK_6M).values
    out["cper_trend_score"] = _trend_score(cper).values

    cper_score = _neutral_fill(
        0.55 * _trend_score(cper)
        + 0.45 * _score_high_is_good(_return(cper, LOOKBACK_3M))
    )

    out["indpro_change_3m"] = _change(indpro, LOOKBACK_3M).values
    out["indpro_change_6m"] = _change(indpro, LOOKBACK_6M).values
    out["indpro_z_3y"] = _rolling_z_score(indpro).values

    indpro_score = _neutral_fill(
        0.60 * _score_high_is_good(_change(indpro, LOOKBACK_3M))
        + 0.40 * _score_high_is_good(indpro)
    )

    out["vix_index"] = vix.values
    out["vix_z_3y"] = _rolling_z_score(vix).values
    vix_low_score = _score_low_is_good(vix)

    out["oil_global_demand_score"] = _neutral_fill(
        0.30 * dbc_score
        + 0.20 * spy_score
        + 0.20 * cper_score
        + 0.20 * indpro_score
        + 0.10 * vix_low_score
    ).values

    # ------------------------------
    # 6. USD strength
    # Weak/falling USD is supportive for dollar-priced commodities.
    # ------------------------------

    out["usd_index"] = uup.values
    out["usd_return_1m"] = _return(uup, LOOKBACK_1M).values
    out["usd_return_3m"] = _return(uup, LOOKBACK_3M).values
    out["usd_return_6m"] = _return(uup, LOOKBACK_6M).values
    out["usd_z_3y"] = _rolling_z_score(uup).values

    usd_1m_score = _score_low_is_good(_return(uup, LOOKBACK_1M))
    usd_3m_score = _score_low_is_good(_return(uup, LOOKBACK_3M))
    usd_6m_score = _score_low_is_good(_return(uup, LOOKBACK_6M))

    out["oil_usd_score"] = _neutral_fill(
        0.25 * usd_1m_score
        + 0.50 * usd_3m_score
        + 0.25 * usd_6m_score
    ).values

    # ------------------------------
    # Composite diagnostic only
    # This is not automatically used unless USO_scoring.py later chooses it.
    # ------------------------------

    out["oil_balance_score"] = _neutral_fill(
        0.30 * out["oil_inventory_tightness_score"]
        + 0.20 * out["oil_cushing_tightness_score"]
        + 0.25 * out["oil_curve_roll_score"]
        + 0.15 * out["oil_supply_refinery_score"]
        + 0.07 * out["oil_global_demand_score"]
        + 0.03 * out["oil_usd_score"]
    )

    # ------------------------------
    # Data quality diagnostics
    # ------------------------------

    availability = pd.DataFrame(index=out.index)

    availability["inventory_available"] = out["us_crude_stocks_ex_spr"].notna()
    availability["cushing_available"] = out["cushing_crude_stocks"].notna()
    availability["curve_available"] = out["usl_price"].notna() & uso.notna().values
    availability["supply_refinery_available"] = (
        out["us_crude_production"].notna()
        & out["refinery_utilisation"].notna()
    )
    availability["global_demand_available"] = (
        dbc.notna().values
        | spy.notna().values
        | cper.notna().values
        | out["indpro"].notna()
    )
    availability["usd_available"] = uup.notna().values

    out["oil_core_feature_count"] = availability.sum(axis=1)
    out["oil_core_data_quality_score"] = (
        out["oil_core_feature_count"] / 6.0
    ).clip(0.0, 1.0)

    core_cols = [
        "oil_inventory_tightness_score",
        "oil_cushing_tightness_score",
        "oil_curve_roll_score",
        "oil_supply_refinery_score",
        "oil_global_demand_score",
        "oil_usd_score",
        "oil_balance_score",
    ]

    for col in core_cols:
        out[col] = _neutral_fill(pd.to_numeric(out[col], errors="coerce"))

    return out.sort_values("date").reset_index(drop=True)


def build_oil_features_monthly(features_daily: pd.DataFrame) -> pd.DataFrame:
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


def save_oil_features(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
) -> None:
    OIL_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    daily.to_csv(OIL_FEATURES_DAILY_PATH, index=False)
    monthly.to_csv(OIL_FEATURES_MONTHLY_PATH, index=False)

    print(f"\nSaved daily oil features to: {OIL_FEATURES_DAILY_PATH}")
    print(f"Saved monthly oil features to: {OIL_FEATURES_MONTHLY_PATH}")


# ============================================================
# PIPELINE
# ============================================================

def run_oil_feature_pipeline() -> pd.DataFrame:
    print("\nStarting oil feature pipeline.")
    print(f"Project root:       {COMMODITY_ROOT}")
    print(f"Commodity prices:   {PRICE_DATA_PATH}")
    print(f"Macro prices:       {MACRO_PRICE_DATA_PATH}")
    print(f"Oil raw wide:       {OIL_RAW_WIDE_PATH}")

    daily = build_oil_features_daily()
    monthly = build_oil_features_monthly(daily)

    save_oil_features(daily, monthly)

    print("\nOil feature pipeline complete.")
    print(f"Daily rows:   {len(daily):,}")
    print(f"Monthly rows: {len(monthly):,}")
    print(f"Start date:   {daily['date'].min().date()}")
    print(f"End date:     {daily['date'].max().date()}")

    score_cols = [
        "oil_inventory_tightness_score",
        "oil_cushing_tightness_score",
        "oil_curve_roll_score",
        "oil_supply_refinery_score",
        "oil_global_demand_score",
        "oil_usd_score",
        "oil_balance_score",
        "oil_core_data_quality_score",
    ]

    print("\nCore oil score columns:")
    print(daily[score_cols].describe().to_string())

    _validate_not_constant(daily, score_cols)

    return daily


if __name__ == "__main__":
    run_oil_feature_pipeline()