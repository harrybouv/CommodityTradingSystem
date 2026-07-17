# macro_features.py

from __future__ import annotations

import numpy as np
import pandas as pd

from Commodity_System.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    PRICE_DATA_PATH,
    UNIVERSE,
)

try:
    from Commodity_System.config import MACRO_PRICE_DATA_PATH
except ImportError:
    MACRO_PRICE_DATA_PATH = RAW_DATA_DIR / "macro_prices.csv"


# ============================================================
# PATHS
# ============================================================

MACRO_SCORES_PATH = PROCESSED_DATA_DIR / "macro_scores.csv"
MACRO_COMPONENTS_PATH = PROCESSED_DATA_DIR / "macro_components.csv"


# ============================================================
# SETTINGS
# ============================================================

# Conservative default:
# score on date t uses macro data available up to t-1.
# If your backtester already shifts weights forward, this may be slightly conservative,
# but it avoids accidental same-close lookahead from macro series.
MACRO_SIGNAL_LAG_DAYS = 1

REQUIRED_MACRO_TICKERS = [
    "UUP",    # USD proxy
    "^TNX",   # 10Y yield proxy
    "TIP",    # inflation-linked bond proxy
    "IEF",    # nominal Treasury bond proxy
    "SPY",    # equity risk/growth proxy
    "^VIX",   # stress proxy
    "DBC",    # broad commodity trend proxy
]

NEUTRAL_SCORE = 0.50


# ============================================================
# UTILITIES
# ============================================================

def _ensure_output_dirs() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _clip01(x: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


def _to_score(
    value: pd.Series,
    scale: float,
    invert: bool = False,
) -> pd.Series:
    """
    Convert a signed signal into a 0-1 score.

    value = 0        -> 0.50
    value = +scale   -> 1.00
    value = -scale   -> 0.00

    If invert=True, positive values are treated as bad.
    """

    if scale <= 0:
        raise ValueError("scale must be positive.")

    x = value.copy()

    if invert:
        x = -x

    score = 0.50 + (x / (2.0 * scale))

    return score.replace([np.inf, -np.inf], np.nan).clip(0, 1)


def _weighted_average(
    components: list[tuple[pd.Series, float]],
    neutral: float = NEUTRAL_SCORE,
) -> pd.Series:
    """
    Weighted average that is robust to missing components.

    Missing component values are replaced with neutral rather than deleting dates.
    This matters because macro series can have slightly different trading calendars.
    """

    if not components:
        raise ValueError("No components supplied.")

    index = components[0][0].index
    total_weight = sum(weight for _, weight in components)

    if total_weight <= 0:
        raise ValueError("Component weights must sum to a positive value.")

    out = pd.Series(0.0, index=index)

    for series, weight in components:
        clean = (
            series
            .reindex(index)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(neutral)
            .clip(0, 1)
        )

        out += (weight / total_weight) * clean

    return out.clip(0, 1)


def _rolling_drawdown(series: pd.Series, window: int) -> pd.Series:
    rolling_high = series.rolling(window=window, min_periods=max(20, window // 4)).max()
    return (series / rolling_high) - 1.0


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = numerator / denominator.replace(0, np.nan)
    return ratio.replace([np.inf, -np.inf], np.nan)


def _normalise_ticker(ticker: str) -> str:
    return str(ticker).upper().strip()


def _require_columns(
    df: pd.DataFrame,
    required_cols: list[str],
    name: str,
) -> None:
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


# ============================================================
# LOAD DATA
# ============================================================

def load_macro_prices(path=MACRO_PRICE_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    _require_columns(
        df,
        ["date", "ticker", "adj_close"],
        "macro_prices.csv",
    )

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")

    df = (
        df.dropna(subset=["date", "ticker", "adj_close"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("macro_prices.csv is empty after cleaning.")

    return df


def load_commodity_dates_and_tickers(path=PRICE_DATA_PATH) -> tuple[pd.DatetimeIndex, list[str]]:
    prices = pd.read_csv(path)

    _require_columns(
        prices,
        ["date", "ticker"],
        "commodity price data",
    )

    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()

    dates = pd.DatetimeIndex(sorted(prices["date"].dropna().unique()))
    tickers = sorted(prices["ticker"].dropna().unique())

    if len(dates) == 0:
        raise ValueError("No commodity dates found.")

    if not tickers:
        raise ValueError("No commodity tickers found.")

    return dates, tickers


def make_macro_price_matrix(macro_prices: pd.DataFrame) -> pd.DataFrame:
    matrix = (
        macro_prices
        .pivot(index="date", columns="ticker", values="adj_close")
        .sort_index()
    )

    # Macro series can have small calendar mismatches.
    # Forward-fill lightly; do not fill huge gaps.
    matrix = matrix.ffill(limit=5)

    return matrix


def validate_macro_inputs(matrix: pd.DataFrame) -> None:
    available = set(matrix.columns)
    missing = [ticker for ticker in REQUIRED_MACRO_TICKERS if ticker not in available]

    if missing:
        raise ValueError(
            "Missing required macro tickers from macro_prices.csv: "
            f"{missing}\n"
            "Run macro_data.py first and inspect macro_data_quality_report.csv."
        )


# ============================================================
# COMPONENT SCORES
# ============================================================

def build_usd_score(prices: pd.DataFrame) -> pd.Series:
    """
    USD regime score.

    Theory:
    - Rising USD is usually hostile to commodities.
    - Falling USD is usually supportive.
    """

    uup = prices["UUP"]

    ret_20 = uup.pct_change(20, fill_method=None)
    ret_60 = uup.pct_change(60, fill_method=None)
    ret_120 = uup.pct_change(120, fill_method=None)

    ma_200 = uup.rolling(200, min_periods=100).mean()
    dist_200 = (uup / ma_200) - 1.0

    score = _weighted_average(
        [
            (_to_score(ret_20, scale=0.04, invert=True), 0.25),
            (_to_score(ret_60, scale=0.07, invert=True), 0.35),
            (_to_score(ret_120, scale=0.10, invert=True), 0.25),
            (_to_score(dist_200, scale=0.08, invert=True), 0.15),
        ]
    )

    return score.rename("usd_score")


def build_rates_score(prices: pd.DataFrame) -> pd.Series:
    """
    Rates regime score.

    Uses ^TNX as yfinance's 10Y yield proxy.

    Theory:
    - Rising yields pressure gold, liquidity and duration-sensitive assets.
    - Sharp yield spikes are especially hostile.
    - Falling/stable yields are supportive.
    """

    tnx = prices["^TNX"]

    change_20 = tnx.diff(20)
    change_60 = tnx.diff(60)
    change_120 = tnx.diff(120)

    ma_200 = tnx.rolling(200, min_periods=100).mean()
    dist_200 = (tnx / ma_200) - 1.0

    # ^TNX is quoted as yield * 10.
    # A 2-point move roughly corresponds to a 20bp move.
    score = _weighted_average(
        [
            (_to_score(change_20, scale=2.0, invert=True), 0.30),
            (_to_score(change_60, scale=4.0, invert=True), 0.35),
            (_to_score(change_120, scale=6.0, invert=True), 0.20),
            (_to_score(dist_200, scale=0.12, invert=True), 0.15),
        ]
    )

    return score.rename("rates_score")


def build_inflation_score(prices: pd.DataFrame) -> pd.Series:
    """
    Inflation / hard-asset pressure score.

    Uses TIP/IEF relative strength as a tradable market proxy.

    Theory:
    - TIP outperforming IEF suggests inflation-linked exposure is being rewarded.
    - This is not a perfect inflation measure, but it is liquid and timestamp-clean.
    """

    tip_ief = _safe_ratio(prices["TIP"], prices["IEF"])

    ret_20 = tip_ief.pct_change(20, fill_method=None)
    ret_60 = tip_ief.pct_change(60, fill_method=None)
    ret_120 = tip_ief.pct_change(120, fill_method=None)

    ma_200 = tip_ief.rolling(200, min_periods=100).mean()
    dist_200 = (tip_ief / ma_200) - 1.0

    score = _weighted_average(
        [
            (_to_score(ret_20, scale=0.025), 0.20),
            (_to_score(ret_60, scale=0.045), 0.35),
            (_to_score(ret_120, scale=0.065), 0.30),
            (_to_score(dist_200, scale=0.050), 0.15),
        ]
    )

    return score.rename("inflation_score")


def build_growth_score(prices: pd.DataFrame) -> pd.Series:
    """
    Growth / risk appetite score.

    Theory:
    - Healthy equity trend supports cyclical commodities like copper and energy.
    - Equity breakdowns are hostile to cyclical commodity risk.
    """

    spy = prices["SPY"]

    ret_20 = spy.pct_change(20, fill_method=None)
    ret_60 = spy.pct_change(60, fill_method=None)
    ret_120 = spy.pct_change(120, fill_method=None)

    ma_200 = spy.rolling(200, min_periods=100).mean()
    dist_200 = (spy / ma_200) - 1.0

    dd_120 = _rolling_drawdown(spy, 120)

    score = _weighted_average(
        [
            (_to_score(ret_20, scale=0.06), 0.15),
            (_to_score(ret_60, scale=0.12), 0.30),
            (_to_score(ret_120, scale=0.20), 0.25),
            (_to_score(dist_200, scale=0.10), 0.20),
            (_to_score(dd_120, scale=0.20), 0.10),
        ]
    )

    return score.rename("growth_score")


def build_stress_score(prices: pd.DataFrame) -> pd.Series:
    """
    Stress / liquidity score.

    Important:
    - This is scored as 1 = calm/supportive, 0 = severe stress.
    - It is not a gold safe-haven score.
    - V3 can later add more nuanced defensive/risk-off commodity-specific behaviour.
    """

    vix = prices["^VIX"]
    spy = prices["SPY"]
    uup = prices["UUP"]
    tnx = prices["^TNX"]

    vix_ma_60 = vix.rolling(60, min_periods=30).mean()
    vix_relative = (vix / vix_ma_60) - 1.0
    vix_change_20 = vix.pct_change(20, fill_method=None)

    spy_dd_60 = _rolling_drawdown(spy, 60)
    spy_dd_120 = _rolling_drawdown(spy, 120)

    usd_ret_20 = uup.pct_change(20, fill_method=None)
    rates_change_20 = tnx.diff(20)

    score = _weighted_average(
        [
            (_to_score(vix_relative, scale=0.80, invert=True), 0.25),
            (_to_score(vix_change_20, scale=0.60, invert=True), 0.20),
            (_to_score(spy_dd_60, scale=0.12), 0.20),
            (_to_score(spy_dd_120, scale=0.20), 0.15),
            (_to_score(usd_ret_20, scale=0.05, invert=True), 0.10),
            (_to_score(rates_change_20, scale=3.0, invert=True), 0.10),
        ]
    )

    return score.rename("stress_score")


def build_commodity_trend_score(prices: pd.DataFrame) -> pd.Series:
    """
    Broad commodity trend score.

    Theory:
    - If broad commodities are already trending positively, the environment is more
      supportive for commodity allocation.
    - This is a broad confirmation signal, not an individual commodity selector.
    """

    dbc = prices["DBC"]

    ret_20 = dbc.pct_change(20, fill_method=None)
    ret_60 = dbc.pct_change(60, fill_method=None)
    ret_120 = dbc.pct_change(120, fill_method=None)

    ma_200 = dbc.rolling(200, min_periods=100).mean()
    dist_200 = (dbc / ma_200) - 1.0

    score = _weighted_average(
        [
            (_to_score(ret_20, scale=0.08), 0.20),
            (_to_score(ret_60, scale=0.14), 0.35),
            (_to_score(ret_120, scale=0.22), 0.30),
            (_to_score(dist_200, scale=0.12), 0.15),
        ]
    )

    return score.rename("commodity_trend_score")


# ============================================================
# GROUP MACRO SCORES
# ============================================================

def build_macro_components(macro_prices: pd.DataFrame) -> pd.DataFrame:
    prices = make_macro_price_matrix(macro_prices)

    validate_macro_inputs(prices)

    components = pd.DataFrame(index=prices.index)

    components["usd_score"] = build_usd_score(prices)
    components["rates_score"] = build_rates_score(prices)
    components["inflation_score"] = build_inflation_score(prices)
    components["growth_score"] = build_growth_score(prices)
    components["stress_score"] = build_stress_score(prices)
    components["commodity_trend_score"] = build_commodity_trend_score(prices)

    # Apply conservative signal lag after components are built.
    if MACRO_SIGNAL_LAG_DAYS > 0:
        score_cols = [
            "usd_score",
            "rates_score",
            "inflation_score",
            "growth_score",
            "stress_score",
            "commodity_trend_score",
        ]

        components[score_cols] = components[score_cols].shift(MACRO_SIGNAL_LAG_DAYS)

    components = (
        components
        .replace([np.inf, -np.inf], np.nan)
        .fillna(NEUTRAL_SCORE)
        .clip(0, 1)
    )

    components = components.reset_index().rename(columns={"index": "date"})
    components["date"] = pd.to_datetime(components["date"])

    return components


def add_group_macro_scores(components: pd.DataFrame) -> pd.DataFrame:
    out = components.copy()

    out["precious_metals_macro_score"] = _weighted_average(
        [
            (out["usd_score"], 0.30),
            (out["rates_score"], 0.30),
            (out["inflation_score"], 0.20),
            (out["stress_score"], 0.10),
            (out["commodity_trend_score"], 0.10),
        ]
    )

    out["energy_macro_score"] = _weighted_average(
        [
            (out["inflation_score"], 0.30),
            (out["commodity_trend_score"], 0.25),
            (out["growth_score"], 0.20),
            (out["usd_score"], 0.15),
            (out["stress_score"], 0.10),
        ]
    )

    out["industrial_metals_macro_score"] = _weighted_average(
        [
            (out["growth_score"], 0.30),
            (out["usd_score"], 0.25),
            (out["commodity_trend_score"], 0.20),
            (out["inflation_score"], 0.15),
            (out["stress_score"], 0.10),
        ]
    )

    out["agriculture_macro_score"] = _weighted_average(
        [
            (out["inflation_score"], 0.35),
            (out["commodity_trend_score"], 0.30),
            (out["usd_score"], 0.25),
            (out["stress_score"], 0.10),
        ]
    )

    out["broad_macro_score"] = _weighted_average(
        [
            (out["usd_score"], 0.25),
            (out["rates_score"], 0.20),
            (out["inflation_score"], 0.25),
            (out["growth_score"], 0.10),
            (out["stress_score"], 0.10),
            (out["commodity_trend_score"], 0.10),
        ]
    )

    score_cols = [
        "precious_metals_macro_score",
        "energy_macro_score",
        "industrial_metals_macro_score",
        "agriculture_macro_score",
        "broad_macro_score",
    ]

    out[score_cols] = out[score_cols].clip(0, 1)

    return out


def classify_macro_regime(score: pd.Series) -> pd.Series:
    """
    Human-readable diagnostic label only.

    The strategy should use macro_score, not this string.
    """

    conditions = [
        score >= 0.70,
        score >= 0.58,
        score > 0.42,
        score > 0.30,
    ]

    choices = [
        "strong_supportive",
        "supportive",
        "neutral",
        "hostile",
    ]

    return pd.Series(
        np.select(conditions, choices, default="strong_hostile"),
        index=score.index,
    )


# ============================================================
# EXPAND TO COMMODITY TICKERS
# ============================================================

def get_ticker_group_map() -> dict[str, str]:
    return {
        _normalise_ticker(ticker): meta.get("group", "unknown")
        for ticker, meta in UNIVERSE.items()
    }


def map_group_to_macro_score(row: pd.Series) -> float:
    group = row["macro_group"]

    if group == "precious_metals":
        return row["precious_metals_macro_score"]

    if group == "energy":
        return row["energy_macro_score"]

    if group == "industrial_metals":
        return row["industrial_metals_macro_score"]

    if group == "agriculture":
        return row["agriculture_macro_score"]

    return row["broad_macro_score"]


def expand_macro_scores_to_commodity_universe(
    macro_components: pd.DataFrame,
    commodity_dates: pd.DatetimeIndex,
    commodity_tickers: list[str],
) -> pd.DataFrame:
    group_map = get_ticker_group_map()

    components = macro_components.copy()
    components["date"] = pd.to_datetime(components["date"])
    components = components.sort_values("date").set_index("date")

    # Align macro data to the exact commodity trading dates so later merges
    # do not accidentally delete valid commodity rows.
    components = (
        components
        .reindex(commodity_dates)
        .ffill(limit=5)
        .fillna(NEUTRAL_SCORE)
        .reset_index()
        .rename(columns={"index": "date"})
    )

    rows = []

    for ticker in commodity_tickers:
        clean_ticker = _normalise_ticker(ticker)

        temp = components.copy()
        temp["ticker"] = clean_ticker
        temp["macro_group"] = group_map.get(clean_ticker, "unknown")

        rows.append(temp)

    out = pd.concat(rows, ignore_index=True)

    out["macro_score"] = out.apply(map_group_to_macro_score, axis=1)
    out["macro_score"] = out["macro_score"].clip(0, 1)

    out["macro_regime"] = classify_macro_regime(out["macro_score"])

    output_cols = [
        "date",
        "ticker",
        "macro_group",
        "macro_score",
        "macro_regime",

        "usd_score",
        "rates_score",
        "inflation_score",
        "growth_score",
        "stress_score",
        "commodity_trend_score",

        "precious_metals_macro_score",
        "energy_macro_score",
        "industrial_metals_macro_score",
        "agriculture_macro_score",
        "broad_macro_score",
    ]

    for col in output_cols:
        if col not in out.columns:
            out[col] = np.nan

    return (
        out[output_cols]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


# ============================================================
# SAVE / PIPELINE
# ============================================================

def save_macro_outputs(
    macro_scores: pd.DataFrame,
    macro_components: pd.DataFrame,
) -> None:
    _ensure_output_dirs()

    macro_scores.to_csv(MACRO_SCORES_PATH, index=False)
    macro_components.to_csv(MACRO_COMPONENTS_PATH, index=False)

    print(f"Saved macro scores to: {MACRO_SCORES_PATH}")
    print(f"Saved macro components to: {MACRO_COMPONENTS_PATH}")


def print_macro_summary(macro_scores: pd.DataFrame) -> None:
    latest_date = macro_scores["date"].max()
    latest = macro_scores[macro_scores["date"] == latest_date].copy()

    print("\nLatest macro scores:")
    print(f"Date: {latest_date.date()}")

    display_cols = [
        "ticker",
        "macro_group",
        "macro_score",
        "macro_regime",
        "usd_score",
        "rates_score",
        "inflation_score",
        "growth_score",
        "stress_score",
        "commodity_trend_score",
    ]

    print(
        latest[display_cols]
        .sort_values("ticker")
        .to_string(index=False)
    )

    summary = (
        macro_scores
        .groupby("macro_group")["macro_score"]
        .agg(["mean", "min", "max"])
        .reset_index()
        .sort_values("macro_group")
    )

    print("\nMacro score summary by group:")
    print(summary.to_string(index=False))


def run_macro_feature_pipeline() -> pd.DataFrame:
    macro_prices = load_macro_prices()
    commodity_dates, commodity_tickers = load_commodity_dates_and_tickers()

    macro_components = build_macro_components(macro_prices)
    macro_components = add_group_macro_scores(macro_components)

    # Add diagnostic regimes to the date-level components too.
    for col in [
        "precious_metals_macro_score",
        "energy_macro_score",
        "industrial_metals_macro_score",
        "agriculture_macro_score",
        "broad_macro_score",
    ]:
        regime_col = col.replace("_score", "_regime")
        macro_components[regime_col] = classify_macro_regime(macro_components[col])

    macro_scores = expand_macro_scores_to_commodity_universe(
        macro_components=macro_components,
        commodity_dates=commodity_dates,
        commodity_tickers=commodity_tickers,
    )

    save_macro_outputs(
        macro_scores=macro_scores,
        macro_components=macro_components,
    )

    print("\nMacro feature pipeline complete.")
    print(f"Rows: {len(macro_scores):,}")
    print(f"Commodity tickers: {sorted(macro_scores['ticker'].unique())}")
    print(f"Start date: {macro_scores['date'].min().date()}")
    print(f"End date: {macro_scores['date'].max().date()}")

    print_macro_summary(macro_scores)

    return macro_scores


if __name__ == "__main__":
    run_macro_feature_pipeline()