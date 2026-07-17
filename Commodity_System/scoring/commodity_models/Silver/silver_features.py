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


from config import PROCESSED_DATA_DIR, PRICE_DATA_PATH


# ============================================================
# PATHS
# ============================================================

SILVER_PROCESSED_DIR = PROCESSED_DATA_DIR / "silver"

SILVER_FEATURES_DAILY_PATH = SILVER_PROCESSED_DIR / "silver_features_daily.csv"
SILVER_FEATURES_MONTHLY_PATH = SILVER_PROCESSED_DIR / "silver_features_monthly.csv"

GOLD_FEATURES_DAILY_PATH = PROCESSED_DATA_DIR / "gold" / "gold_features_daily.csv"


# ============================================================
# SETTINGS
# ============================================================

TRADING_DAYS_PER_MONTH = 21
LOOKBACK_3M = 63
LOOKBACK_6M = 126
LOOKBACK_1Y = 252
LOOKBACK_3Y = 756

FAST_MA = 50
SLOW_MA = 200

REQUIRED_TICKERS = ["GLD", "SLV", "CPER"]


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


def _score_high_is_good(s: pd.Series) -> pd.Series:
    return _clip01(_rolling_percentile(s))


def _score_low_is_good(s: pd.Series) -> pd.Series:
    return _clip01(1.0 - _rolling_percentile(s))


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

def load_price_matrix(path: Path = PRICE_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Price data not found: {path}. Run data.py first."
        )

    prices = pd.read_csv(path)

    required_cols = ["date", "ticker", "adj_close"]
    missing = [col for col in required_cols if col not in prices.columns]

    if missing:
        raise ValueError(
            f"Price data missing required columns: {missing}. "
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

    missing_tickers = [ticker for ticker in REQUIRED_TICKERS if ticker not in matrix.columns]

    if missing_tickers:
        raise ValueError(
            f"Silver feature pipeline requires {REQUIRED_TICKERS}, "
            f"but missing: {missing_tickers}. "
            f"Available tickers: {list(matrix.columns)}"
        )

    matrix = matrix[REQUIRED_TICKERS].copy()

    return matrix


def load_gold_macro_features(path: Path = GOLD_FEATURES_DAILY_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Gold features not found: {path}. "
            "Run scoring/commodity_models/Gold/gold_features.py first."
        )

    gold_features = pd.read_csv(path)

    if "date" not in gold_features.columns:
        raise ValueError(f"Gold features missing date column: {path}")

    gold_features["date"] = pd.to_datetime(gold_features["date"])
    gold_features = (
        gold_features
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    keep_cols = [
        "date",
        "real_yield_score",
        "usd_score",
        "real_yield_10y",
        "real_yield_change_3m",
        "real_yield_z_3y",
        "usd_index",
        "usd_return_1m",
        "usd_return_3m",
        "usd_return_6m",
        "usd_z_3y",
        "stress_score",
        "gold_liquidity_squeeze_flag",
    ]

    existing = [col for col in keep_cols if col in gold_features.columns]
    gold_features = gold_features[existing].copy()

    rename_map = {
        "real_yield_score": "silver_real_yield_score",
        "usd_score": "silver_usd_score",
        "stress_score": "silver_macro_stress_score",
    }

    gold_features = gold_features.rename(columns=rename_map)

    return gold_features


# ============================================================
# FEATURE BUILDING
# ============================================================

def build_silver_price_features(prices: pd.DataFrame | None = None) -> pd.DataFrame:
    if prices is None:
        prices = load_price_matrix()

    prices = prices.copy().sort_index()

    gld = _safe_numeric(prices["GLD"])
    slv = _safe_numeric(prices["SLV"])
    cper = _safe_numeric(prices["CPER"])

    out = pd.DataFrame({"date": prices.index})

    # ------------------------------
    # Basic return diagnostics
    # ------------------------------

    out["slv_return_1m"] = _return(slv, TRADING_DAYS_PER_MONTH).values
    out["slv_return_3m"] = _return(slv, LOOKBACK_3M).values
    out["slv_return_6m"] = _return(slv, LOOKBACK_6M).values

    out["gld_return_1m"] = _return(gld, TRADING_DAYS_PER_MONTH).values
    out["gld_return_3m"] = _return(gld, LOOKBACK_3M).values
    out["gld_return_6m"] = _return(gld, LOOKBACK_6M).values

    out["cper_return_1m"] = _return(cper, TRADING_DAYS_PER_MONTH).values
    out["cper_return_3m"] = _return(cper, LOOKBACK_3M).values
    out["cper_return_6m"] = _return(cper, LOOKBACK_6M).values

    # ------------------------------
    # Gold / silver ratio
    # ------------------------------

    gold_silver_ratio = gld / slv
    gold_silver_ratio_change_3m = _return(gold_silver_ratio, LOOKBACK_3M)

    silver_vs_gold_return_1m = _return(slv, TRADING_DAYS_PER_MONTH) - _return(gld, TRADING_DAYS_PER_MONTH)
    silver_vs_gold_return_3m = _return(slv, LOOKBACK_3M) - _return(gld, LOOKBACK_3M)
    silver_vs_gold_return_6m = _return(slv, LOOKBACK_6M) - _return(gld, LOOKBACK_6M)

    out["gold_silver_ratio"] = gold_silver_ratio.values
    out["gold_silver_ratio_change_3m"] = gold_silver_ratio_change_3m.values
    out["gold_silver_ratio_z_3y"] = _rolling_z_score(gold_silver_ratio).values

    out["silver_vs_gold_return_1m"] = silver_vs_gold_return_1m.values
    out["silver_vs_gold_return_3m"] = silver_vs_gold_return_3m.values
    out["silver_vs_gold_return_6m"] = silver_vs_gold_return_6m.values

    ratio_level_score = _score_high_is_good(gold_silver_ratio)
    ratio_falling_score = _score_low_is_good(gold_silver_ratio_change_3m)
    silver_catchup_score = _score_high_is_good(silver_vs_gold_return_3m)

    out["silver_gold_ratio_score"] = _neutral_fill(
        0.45 * ratio_level_score
        + 0.20 * ratio_falling_score
        + 0.35 * silver_catchup_score
    ).values

    # ------------------------------
    # Copper / industrial confirmation
    # ------------------------------

    silver_copper_ratio = slv / cper
    silver_copper_ratio_change_3m = _return(silver_copper_ratio, LOOKBACK_3M)

    out["silver_copper_ratio"] = silver_copper_ratio.values
    out["silver_copper_ratio_change_3m"] = silver_copper_ratio_change_3m.values
    out["silver_copper_ratio_z_3y"] = _rolling_z_score(silver_copper_ratio).values

    copper_momentum_score = _neutral_fill(
        0.60 * _score_high_is_good(_return(cper, LOOKBACK_3M))
        + 0.40 * _score_high_is_good(_return(cper, LOOKBACK_6M))
    )

    copper_trend_score = _trend_score(cper)
    silver_cheap_vs_copper_score = _score_low_is_good(silver_copper_ratio)

    out["copper_momentum_score"] = copper_momentum_score.values
    out["copper_trend_score"] = copper_trend_score.values
    out["silver_cheap_vs_copper_score"] = silver_cheap_vs_copper_score.values

    out["silver_copper_ratio_score"] = _neutral_fill(
        0.35 * silver_cheap_vs_copper_score
        + 0.40 * copper_momentum_score
        + 0.25 * copper_trend_score
    ).values

    # ------------------------------
    # Gold confirmation
    # ------------------------------

    gold_momentum_score = _neutral_fill(
        0.60 * _score_high_is_good(_return(gld, LOOKBACK_3M))
        + 0.40 * _score_high_is_good(_return(gld, LOOKBACK_6M))
    )

    gold_trend_score = _trend_score(gld)

    out["gold_momentum_score"] = gold_momentum_score.values
    out["gold_trend_score"] = gold_trend_score.values

    out["silver_gold_confirmation_score"] = _neutral_fill(
        0.60 * gold_momentum_score
        + 0.40 * gold_trend_score
    ).values

    # ------------------------------
    # Silver own diagnostics
    # ------------------------------

    out["silver_trend_score"] = _trend_score(slv).values

    out["silver_momentum_score"] = _neutral_fill(
        0.50 * _score_high_is_good(_return(slv, LOOKBACK_3M))
        + 0.50 * _score_high_is_good(_return(slv, LOOKBACK_6M))
    ).values

    return out.sort_values("date").reset_index(drop=True)


def merge_macro_features_asof(
    silver_features: pd.DataFrame,
    gold_macro_features: pd.DataFrame,
) -> pd.DataFrame:
    left = silver_features.copy()
    right = gold_macro_features.copy()

    left["date"] = pd.to_datetime(left["date"])
    right["date"] = pd.to_datetime(right["date"])

    left = left.sort_values("date").reset_index(drop=True)
    right = right.sort_values("date").reset_index(drop=True)

    out = pd.merge_asof(
        left,
        right,
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=10),
    )

    for col in ["silver_real_yield_score", "silver_usd_score"]:
        if col not in out.columns:
            out[col] = 0.50

        out[col] = _neutral_fill(pd.to_numeric(out[col], errors="coerce"))

    return out


def build_silver_features_daily() -> pd.DataFrame:
    prices = load_price_matrix()
    price_features = build_silver_price_features(prices)
    gold_macro = load_gold_macro_features()

    out = merge_macro_features_asof(
        silver_features=price_features,
        gold_macro_features=gold_macro,
    )

    out["silver_macro_score"] = _neutral_fill(
        0.55 * out["silver_usd_score"]
        + 0.45 * out["silver_real_yield_score"]
    )

    core_cols = [
        "silver_gold_ratio_score",
        "silver_copper_ratio_score",
        "silver_gold_confirmation_score",
        "silver_usd_score",
        "silver_real_yield_score",
        "silver_macro_score",
    ]

    for col in core_cols:
        if col not in out.columns:
            out[col] = 0.50

        out[col] = _neutral_fill(pd.to_numeric(out[col], errors="coerce"))

    out["silver_core_feature_count"] = out[core_cols].notna().sum(axis=1)

    out["silver_core_data_quality_score"] = (
        out["silver_core_feature_count"] / len(core_cols)
    ).clip(0.0, 1.0)

    return out.sort_values("date").reset_index(drop=True)


def build_silver_features_monthly(features_daily: pd.DataFrame) -> pd.DataFrame:
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


def save_silver_features(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
) -> None:
    SILVER_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    daily.to_csv(SILVER_FEATURES_DAILY_PATH, index=False)
    monthly.to_csv(SILVER_FEATURES_MONTHLY_PATH, index=False)

    print(f"\nSaved daily silver features to: {SILVER_FEATURES_DAILY_PATH}")
    print(f"Saved monthly silver features to: {SILVER_FEATURES_MONTHLY_PATH}")


# ============================================================
# PIPELINE
# ============================================================

def run_silver_feature_pipeline() -> pd.DataFrame:
    print("\nStarting silver feature pipeline...")
    print(f"Project root: {COMMODITY_ROOT}")
    print(f"Price data:   {PRICE_DATA_PATH}")
    print(f"Gold macro:   {GOLD_FEATURES_DAILY_PATH}")

    daily = build_silver_features_daily()
    monthly = build_silver_features_monthly(daily)

    save_silver_features(daily, monthly)

    print("\nSilver feature pipeline complete.")
    print(f"Daily rows:   {len(daily):,}")
    print(f"Monthly rows: {len(monthly):,}")
    print(f"Start date:   {daily['date'].min().date()}")
    print(f"End date:     {daily['date'].max().date()}")

    score_cols = [
        "silver_gold_ratio_score",
        "silver_copper_ratio_score",
        "silver_gold_confirmation_score",
        "silver_usd_score",
        "silver_real_yield_score",
        "silver_macro_score",
    ]

    print("\nCore silver score columns:")
    print(daily[score_cols].describe().to_string())

    _validate_not_constant(daily, score_cols)

    return daily


if __name__ == "__main__":
    run_silver_feature_pipeline()