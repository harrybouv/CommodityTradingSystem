from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR


# ============================================================
# PATHS
# ============================================================

GOLD_PROCESSED_DIR = PROCESSED_DATA_DIR / "gold"
GOLD_RAW_WIDE_PATH = GOLD_PROCESSED_DIR / "gold_raw_wide.csv"
GOLD_FEATURES_DAILY_PATH = GOLD_PROCESSED_DIR / "gold_features_daily.csv"
GOLD_FEATURES_MONTHLY_PATH = GOLD_PROCESSED_DIR / "gold_features_monthly.csv"


# ============================================================
# WINDOWS / SETTINGS
# ============================================================

TRADING_DAYS_PER_MONTH = 21
LOOKBACK_3M = 63
LOOKBACK_6M = 126
LOOKBACK_1Y = 252
LOOKBACK_3Y = 756

# Explicit as-of limits. These prevent silently dragging stale data forward forever.
DAILY_FFILL_LIMIT = 5
WEEKLY_FFILL_LIMIT = 10
MONTHLY_FFILL_LIMIT = 45
OPTIONAL_SLOW_FFILL_LIMIT = 95


# ============================================================
# HELPERS
# ============================================================

def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _clip01(s: pd.Series) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _rolling_z_score(
    s: pd.Series,
    window: int = LOOKBACK_3Y,
    min_periods: int = LOOKBACK_1Y,
) -> pd.Series:
    x = _safe_numeric(s)
    mean = x.rolling(window=window, min_periods=min_periods).mean()
    std = x.rolling(window=window, min_periods=min_periods).std(ddof=0)
    return ((x - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _rolling_percentile(
    s: pd.Series,
    window: int = LOOKBACK_3Y,
    min_periods: int = LOOKBACK_1Y,
) -> pd.Series:
    """
    Percentile rank of the latest value inside its trailing window.
    High value => close to 1. Low value => close to 0.
    """
    x = _safe_numeric(s)

    def pct_rank(a: np.ndarray) -> float:
        a = a[~np.isnan(a)]
        if len(a) < min_periods:
            return np.nan
        latest = a[-1]
        return float((a <= latest).mean())

    return x.rolling(window=window, min_periods=min_periods).apply(pct_rank, raw=True)


def _change(s: pd.Series, periods: int) -> pd.Series:
    return _safe_numeric(s).diff(periods)


def _return(s: pd.Series, periods: int) -> pd.Series:
    return _safe_numeric(s).pct_change(periods=periods, fill_method=None)


def _drawdown(s: pd.Series, window: int) -> pd.Series:
    price = _safe_numeric(s)
    high = price.rolling(window=window, min_periods=max(20, window // 3)).max()
    return (price / high - 1.0).replace([np.inf, -np.inf], np.nan)


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


def _mean_available(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    existing = [col for col in cols if col in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    return df[existing].mean(axis=1, skipna=True)


def _score_low_is_good(s: pd.Series) -> pd.Series:
    return _clip01(1.0 - _rolling_percentile(s))


def _score_high_is_good(s: pd.Series) -> pd.Series:
    return _clip01(_rolling_percentile(s))


def _neutral_fill(s: pd.Series, neutral: float = 0.50) -> pd.Series:
    return _clip01(s.fillna(neutral))


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def load_gold_raw_wide(path: Path = GOLD_RAW_WIDE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Gold raw wide file not found: {path}. Run gold_data.py first."
        )

    df = pd.read_csv(path)
    _require_columns(df, ["date"], "gold_raw_wide.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    for col in df.columns:
        if col != "date":
            df[col] = _safe_numeric(df[col])

    return df


def prepare_asof_inputs(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a daily/as-of table from mixed-frequency inputs.
    This is where missing data is handled deliberately rather than by blind global fill.
    """
    out = raw.copy().sort_values("date").reset_index(drop=True)

    daily_cols = [
        "DFII10",
        "DTWEXBGS",
        "VIXCLS",
        "BAMLH0A0HYM2",
        "DGS2",
        "DGS10",
        "yf_SPY_adj_close",
        "yf_^MOVE_adj_close",
    ]
    weekly_cols = ["STLFSI4"]
    monthly_cols = ["FEDFUNDS"]
    slow_optional_cols = [
        "CENTRAL_BANK_NET_PURCHASES_TONNES",
        "managed_money_net_pct_open_interest",
        "managed_money_net",
        "open_interest",
    ]

    out = _ffill_existing(out, daily_cols, DAILY_FFILL_LIMIT)
    out = _ffill_existing(out, weekly_cols, WEEKLY_FFILL_LIMIT)
    out = _ffill_existing(out, monthly_cols, MONTHLY_FFILL_LIMIT)
    out = _ffill_existing(out, slow_optional_cols, OPTIONAL_SLOW_FFILL_LIMIT)

    return out


def build_gold_features_daily(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    if raw is None:
        raw = load_gold_raw_wide()

    df = prepare_asof_inputs(raw)
    out = pd.DataFrame({"date": df["date"]})

    # ------------------------------
    # Real yields: core gold opportunity-cost variable
    # ------------------------------
    if "DFII10" in df.columns:
        out["real_yield_10y"] = df["DFII10"]
        out["real_yield_change_3m"] = _change(df["DFII10"], LOOKBACK_3M)
        out["real_yield_change_6m"] = _change(df["DFII10"], LOOKBACK_6M)
        out["real_yield_z_3y"] = _rolling_z_score(df["DFII10"])

        real_yield_level_score = _score_low_is_good(df["DFII10"])
        real_yield_3m_score = _score_low_is_good(out["real_yield_change_3m"])
        real_yield_6m_score = _score_low_is_good(out["real_yield_change_6m"])

        out["real_yield_score"] = _neutral_fill(
            0.40 * real_yield_level_score
            + 0.45 * real_yield_3m_score
            + 0.15 * real_yield_6m_score
        )
    else:
        out["real_yield_score"] = 0.50

    # ------------------------------
    # USD: gold usually dislikes strengthening dollar momentum
    # ------------------------------
    if "DTWEXBGS" in df.columns:
        out["usd_index"] = df["DTWEXBGS"]
        out["usd_return_1m"] = _return(df["DTWEXBGS"], TRADING_DAYS_PER_MONTH)
        out["usd_return_3m"] = _return(df["DTWEXBGS"], LOOKBACK_3M)
        out["usd_return_6m"] = _return(df["DTWEXBGS"], LOOKBACK_6M)
        out["usd_z_3y"] = _rolling_z_score(df["DTWEXBGS"])

        usd_1m_score = _score_low_is_good(out["usd_return_1m"])
        usd_3m_score = _score_low_is_good(out["usd_return_3m"])
        usd_6m_score = _score_low_is_good(out["usd_return_6m"])

        out["usd_score"] = _neutral_fill(
            0.20 * usd_1m_score
            + 0.55 * usd_3m_score
            + 0.25 * usd_6m_score
        )
    else:
        out["usd_score"] = 0.50

    # ------------------------------
    # Stress: safe-haven demand, gated against liquidity-squeeze regimes
    # ------------------------------
    stress_components: list[str] = []

    if "VIXCLS" in df.columns:
        out["vix"] = df["VIXCLS"]
        out["vix_z_3y"] = _rolling_z_score(df["VIXCLS"])
        out["vix_stress_score"] = _score_high_is_good(df["VIXCLS"])
        stress_components.append("vix_stress_score")

    if "STLFSI4" in df.columns:
        out["stlfsi"] = df["STLFSI4"]
        out["stlfsi_z_3y"] = _rolling_z_score(df["STLFSI4"])
        out["stlfsi_stress_score"] = _score_high_is_good(df["STLFSI4"])
        stress_components.append("stlfsi_stress_score")

    if "BAMLH0A0HYM2" in df.columns and df["BAMLH0A0HYM2"].notna().sum() >= LOOKBACK_1Y:
        out["hy_oas"] = df["BAMLH0A0HYM2"]
        out["hy_oas_z_3y"] = _rolling_z_score(df["BAMLH0A0HYM2"])
        out["hy_oas_stress_score"] = _score_high_is_good(df["BAMLH0A0HYM2"])
        stress_components.append("hy_oas_stress_score")

    if "yf_^MOVE_adj_close" in df.columns and df["yf_^MOVE_adj_close"].notna().sum() >= LOOKBACK_1Y:
        out["move"] = df["yf_^MOVE_adj_close"]
        out["move_z_3y"] = _rolling_z_score(df["yf_^MOVE_adj_close"])
        out["move_stress_score"] = _score_high_is_good(df["yf_^MOVE_adj_close"])
        stress_components.append("move_stress_score")

    if "yf_SPY_adj_close" in df.columns:
        out["spy_drawdown_3m"] = _drawdown(df["yf_SPY_adj_close"], LOOKBACK_3M)
        out["spy_drawdown_6m"] = _drawdown(df["yf_SPY_adj_close"], LOOKBACK_6M)
        out["spy_drawdown_stress_score"] = _score_high_is_good(-out["spy_drawdown_3m"])
        stress_components.append("spy_drawdown_stress_score")

    out["stress_raw_score"] = _neutral_fill(_mean_available(out, stress_components))

    # Gate: if real yields and USD are both rising hard, stress may be liquidity-squeeze toxic.
    real_yield_rising_pressure = _score_high_is_good(out.get("real_yield_change_3m", pd.Series(np.nan, index=out.index)))
    usd_rising_pressure = _score_high_is_good(out.get("usd_return_3m", pd.Series(np.nan, index=out.index)))

    out["gold_liquidity_squeeze_flag"] = (
        (real_yield_rising_pressure > 0.75)
        & (usd_rising_pressure > 0.75)
        & (out["stress_raw_score"] > 0.60)
    ).astype(int)

    out["stress_score"] = out["stress_raw_score"].where(
        out["gold_liquidity_squeeze_flag"] == 0,
        np.minimum(out["stress_raw_score"], 0.55),
    )
    out["stress_score"] = _neutral_fill(out["stress_score"])

    # ------------------------------
    # Policy/Fed regime: built now, toggle later
    # ------------------------------
    if "DGS2" in df.columns:
        out["dgs2"] = df["DGS2"]
        out["dgs2_change_3m"] = _change(df["DGS2"], LOOKBACK_3M)
        out["dgs2_change_6m"] = _change(df["DGS2"], LOOKBACK_6M)
        out["policy_rate_score"] = _neutral_fill(
            0.65 * _score_low_is_good(out["dgs2_change_3m"])
            + 0.35 * _score_low_is_good(out["dgs2_change_6m"])
        )
    else:
        out["policy_rate_score"] = 0.50

    if "DGS10" in df.columns and "DGS2" in df.columns:
        out["yield_curve_10y_2y"] = df["DGS10"] - df["DGS2"]

    if "FEDFUNDS" in df.columns:
        out["fedfunds"] = df["FEDFUNDS"]
        out["fedfunds_change_6m"] = _change(df["FEDFUNDS"], LOOKBACK_6M)

    # ------------------------------
    # Central-bank demand: built now, inactive until manual data is loaded/tested
    # ------------------------------
    if "CENTRAL_BANK_NET_PURCHASES_TONNES" in df.columns:
        out["central_bank_net_purchases_tonnes"] = df["CENTRAL_BANK_NET_PURCHASES_TONNES"]
        out["central_bank_purchases_12m"] = (
            df["CENTRAL_BANK_NET_PURCHASES_TONNES"]
            .rolling(window=252, min_periods=6)
            .sum()
        )
        out["central_bank_score"] = _neutral_fill(_score_high_is_good(out["central_bank_purchases_12m"]))
    else:
        out["central_bank_score"] = 0.50

    # ------------------------------
    # Positioning/crowding: built now, inactive until CFTC manual data is loaded/tested
    # ------------------------------
    if "managed_money_net_pct_open_interest" in df.columns:
        out["managed_money_net_pct_open_interest"] = df["managed_money_net_pct_open_interest"]
        out["managed_money_positioning_z_3y"] = _rolling_z_score(df["managed_money_net_pct_open_interest"])

        crowded_long = out["managed_money_positioning_z_3y"] > 1.5
        washed_out = out["managed_money_positioning_z_3y"] < -1.0

        out["positioning_score"] = 0.50
        out.loc[crowded_long, "positioning_score"] = 0.35
        out.loc[washed_out, "positioning_score"] = 0.65
        out["positioning_score"] = _neutral_fill(out["positioning_score"])
    else:
        out["positioning_score"] = 0.50

    # ------------------------------
    # Data quality diagnostics
    # ------------------------------
    core_cols = ["real_yield_score", "usd_score", "stress_score"]
    out["gold_core_feature_count"] = out[core_cols].notna().sum(axis=1)
    out["gold_core_data_quality_score"] = (out["gold_core_feature_count"] / len(core_cols)).clip(0.0, 1.0)

    # Keep warm-up rows for diagnostics, but scores are neutral until enough history exists.
    score_cols = [
        "real_yield_score",
        "usd_score",
        "stress_score",
        "policy_rate_score",
        "central_bank_score",
        "positioning_score",
    ]
    for col in score_cols:
        out[col] = _neutral_fill(out[col])

    return out.sort_values("date").reset_index(drop=True)


def build_gold_features_monthly(features_daily: pd.DataFrame) -> pd.DataFrame:
    out = features_daily.copy()
    out["date"] = pd.to_datetime(out["date"])
    monthly = (
        out.set_index("date")
        .resample("ME")
        .last()
        .dropna(how="all")
        .reset_index()
    )
    return monthly


def save_gold_features(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    daily_path: Path = GOLD_FEATURES_DAILY_PATH,
    monthly_path: Path = GOLD_FEATURES_MONTHLY_PATH,
) -> None:
    GOLD_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(daily_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    print(f"Saved daily gold features to: {daily_path}")
    print(f"Saved monthly gold features to: {monthly_path}")


def run_gold_feature_pipeline() -> pd.DataFrame:
    raw = load_gold_raw_wide()
    daily = build_gold_features_daily(raw)
    monthly = build_gold_features_monthly(daily)
    save_gold_features(daily, monthly)

    print("\nGold feature pipeline complete.")
    print(f"Daily rows:   {len(daily):,}")
    print(f"Monthly rows: {len(monthly):,}")
    print(f"Start date:   {daily['date'].min().date()}")
    print(f"End date:     {daily['date'].max().date()}")
    print("Core score columns:")
    print(daily[["real_yield_score", "usd_score", "stress_score"]].describe().to_string())

    return daily


if __name__ == "__main__":
    run_gold_feature_pipeline()
