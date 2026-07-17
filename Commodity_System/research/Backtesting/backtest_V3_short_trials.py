from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# ============================================================
# PATH SETUP
# ============================================================

THIS_FILE = Path(__file__).resolve()
BACKTESTING_DIR = THIS_FILE.parent
RESEARCH_DIR = BACKTESTING_DIR.parent
COMMODITY_ROOT = RESEARCH_DIR.parent
PROJECT_ROOT = COMMODITY_ROOT.parent

for path in [PROJECT_ROOT, COMMODITY_ROOT, RESEARCH_DIR, BACKTESTING_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# ============================================================
# IMPORT EXISTING REALISTIC ENGINE INFRASTRUCTURE
# ============================================================

try:
    import backtest_V2 as V2
except ImportError as exc:
    raise ImportError(
        "Could not import backtest_V2.py. Put this file in "
        "Commodity_System/research/Backtesting next to backtest_V2.py."
    ) from exc

try:
    from Commodity_System.commodity_strategy import build_production_strategy_score_history
except ImportError:
    try:
        from commodity_strategy import build_production_strategy_score_history
    except ImportError:
        build_production_strategy_score_history = None

try:
    from analytics import calculate_drawdown_series
except ImportError:
    calculate_drawdown_series = V2.calculate_drawdown_series


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = V2.RESULTS_DIR / "backtest_V3_short_trials"
CHARTS_DIR = OUTPUT_DIR / "charts"

# Conservative defaults. Override from config.py by defining the same names.
MAX_TOTAL_SHORT = float(V2.cfg("SHORT_TRIAL_MAX_TOTAL_SHORT", 0.25))
MAX_SINGLE_SHORT = float(V2.cfg("SHORT_TRIAL_MAX_SINGLE_SHORT", 0.08))
MAX_TARGET_GROSS_EXPOSURE = float(V2.cfg("SHORT_TRIAL_MAX_TARGET_GROSS_EXPOSURE", 1.00))
SHORT_BORROW_COST_ANNUAL = float(V2.cfg("SHORT_TRIAL_BORROW_COST_ANNUAL", 0.02))

REGIME_AVG_SCORE_BEAR = float(V2.cfg("SHORT_TRIAL_REGIME_AVG_SCORE_BEAR", 0.40))
REGIME_AVG_SCORE_BULL = float(V2.cfg("SHORT_TRIAL_REGIME_AVG_SCORE_BULL", 0.55))
REGIME_TREND_BEAR = float(V2.cfg("SHORT_TRIAL_REGIME_TREND_BEAR", 0.42))
REGIME_TREND_BULL = float(V2.cfg("SHORT_TRIAL_REGIME_TREND_BULL", 0.50))
REGIME_STRESS_BULL_MAX = float(V2.cfg("SHORT_TRIAL_REGIME_STRESS_BULL_MAX", 0.65))
REGIME_WEAK_SCORE_THRESHOLD = float(V2.cfg("SHORT_TRIAL_REGIME_WEAK_SCORE_THRESHOLD", 0.30))

MIN_SCORE_TO_HOLD = V2.cfg("MIN_SCORE_TO_HOLD", 0.50)
try:
    MIN_SCORE_TO_HOLD = float(MIN_SCORE_TO_HOLD)
except Exception:
    MIN_SCORE_TO_HOLD = 0.50

RISK_FREE_RATE_ANNUAL = 0.0

STRESS_PERIODS = {
    "commodity_bear_2013_2015": ("2013-01-01", "2015-12-31"),
    "oil_crash_2014_2015": ("2014-06-01", "2015-12-31"),
    "covid_crash_2020": ("2020-02-01", "2020-04-30"),
    "inflation_shock_2022": ("2022-01-01", "2022-12-31"),
}


@dataclass(frozen=True)
class ShortTrialConfig:
    name: str
    short_threshold: float
    allowed_regimes: tuple[str, ...]
    max_total_short: float = MAX_TOTAL_SHORT
    max_single_short: float = MAX_SINGLE_SHORT
    max_target_gross: float = MAX_TARGET_GROSS_EXPOSURE


SHORT_TRIALS = [
    ShortTrialConfig(
        name="short_020_all_regimes",
        short_threshold=0.20,
        allowed_regimes=("bull", "chop", "bear"),
    ),
    ShortTrialConfig(
        name="short_030_bear_only",
        short_threshold=0.30,
        allowed_regimes=("bear",),
    ),
    ShortTrialConfig(
        name="short_035_bear_only",
        short_threshold=0.35,
        allowed_regimes=("bear",),
    ),
    ShortTrialConfig(
        name="short_020_bear_chop_only",
        short_threshold=0.20,
        allowed_regimes=("bear", "chop"),
    ),
    # Slightly looser bear-only variant. The first run showed 0.20 was too harsh
    # at monthly rebalance dates. 0.40 is diagnostic only, not production approval.
    ShortTrialConfig(
        name="short_040_bear_only",
        short_threshold=0.40,
        allowed_regimes=("bear",),
    ),
]

FORCED_BEAR_TRIALS = [
    ShortTrialConfig(
        name="forced_2013_2015_short_020",
        short_threshold=0.20,
        allowed_regimes=("bear",),
    ),
    ShortTrialConfig(
        name="forced_2013_2015_short_030",
        short_threshold=0.30,
        allowed_regimes=("bear",),
    ),
    ShortTrialConfig(
        name="forced_2013_2015_short_035",
        short_threshold=0.35,
        allowed_regimes=("bear",),
    ),
    ShortTrialConfig(
        name="forced_2013_2015_short_040",
        short_threshold=0.40,
        allowed_regimes=("bear",),
    ),
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_to_csv(df: pd.DataFrame | pd.Series, path: Path, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df, pd.Series):
        df.to_frame().to_csv(path, index=index)
    else:
        df.to_csv(path, index=index)


def clean_numeric_series(s: pd.Series, default: float = 0.0) -> pd.Series:
    return (
        pd.to_numeric(s, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
        .astype(float)
    )


def signed_normalise_weights(
    weights: pd.Series,
    max_gross: float = MAX_TARGET_GROSS_EXPOSURE,
) -> pd.Series:
    """
    Normalise signed weights by gross exposure.

    This is the key difference versus V2.normalise_asset_weights(), which clips
    negative weights to zero because the production system is long-only.
    """
    out = weights.copy().astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    gross = float(out.abs().sum())
    max_gross = max(0.0, float(max_gross))

    if max_gross > 0 and gross > max_gross:
        out = out * (max_gross / gross)

    return out.fillna(0.0)


def clean_weights_for_output(weights: pd.DataFrame) -> pd.DataFrame:
    out = weights.copy()
    if "strategy" in out.columns:
        out = out.drop(columns=["strategy"])
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    return out.sort_index().fillna(0.0)


# ============================================================
# SCORE HISTORY / REGIME CLASSIFICATION
# ============================================================

def load_scores_history() -> pd.DataFrame:
    if build_production_strategy_score_history is None:
        raise ImportError(
            "Could not import build_production_strategy_score_history(). "
            "This short-trial script needs historical final_score data."
        )

    scores = build_production_strategy_score_history(save=False)

    if scores is None or scores.empty:
        raise ValueError("Score history is empty. Cannot build short candidates.")

    out = scores.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()

    if "final_score" not in out.columns:
        raise ValueError("Score history missing final_score column.")

    out["final_score"] = clean_numeric_series(out["final_score"])

    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_score_wide(scores_history: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    score_wide = (
        scores_history
        .pivot(index="date", columns="ticker", values="final_score")
        .sort_index()
        .reindex(columns=tickers)
    )

    return score_wide.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


def _mean_existing_columns(data: pd.DataFrame, columns: list[str], default: float = 0.50) -> pd.Series:
    existing = [col for col in columns if col in data.columns]

    if not existing:
        dates = pd.to_datetime(data["date"]).sort_values().unique()
        return pd.Series(default, index=pd.to_datetime(dates), dtype=float)

    tmp = data[["date"] + existing].copy()
    for col in existing:
        tmp[col] = clean_numeric_series(tmp[col], default=default)

    return tmp.groupby("date")[existing].mean().mean(axis=1).sort_index()


def build_regime_table(
    *,
    scores_history: pd.DataFrame,
    market_data: dict[str, pd.DataFrame],
    tickers: list[str],
) -> pd.DataFrame:
    """
    Price/breadth commodity-regime classifier for short-trial diagnostics.

    Do NOT base bear/bull primarily on average final_score. The score is clipped,
    blended, and compressed, so it is not a regime variable. This classifier uses
    the actual commodity basket trend, drawdown, momentum, and breadth.

    This is still research-only. It does not touch production weights or V4.
    """
    scores = scores_history.copy()
    scores["date"] = pd.to_datetime(scores["date"])
    scores["final_score"] = clean_numeric_series(scores["final_score"])

    returns = (
        market_data["returns"]
        .copy()
        .sort_index()
        .reindex(columns=tickers)
        .fillna(0.0)
    )

    close = (
        market_data.get("close", pd.DataFrame())
        .copy()
        .sort_index()
        .reindex(columns=tickers)
        .ffill()
    )

    if returns.empty:
        raise ValueError("Cannot build price/breadth regime table: market returns are empty.")

    out = pd.DataFrame(index=returns.index.copy())
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"

    basket_return = returns.mean(axis=1).fillna(0.0)
    basket_equity = (1.0 + basket_return).cumprod()

    out["basket_return"] = basket_return
    out["basket_equity"] = basket_equity
    out["basket_return_63d"] = basket_equity.pct_change(63)
    out["basket_return_126d"] = basket_equity.pct_change(126)
    out["basket_return_252d"] = basket_equity.pct_change(252)
    out["basket_realised_vol_63d"] = (
        basket_return.rolling(63, min_periods=30).std() * np.sqrt(V2.TRADING_DAYS_PER_YEAR)
    )

    rolling_high_252 = basket_equity.rolling(252, min_periods=60).max()
    out["basket_drawdown_252d"] = basket_equity / rolling_high_252 - 1.0

    basket_ma_200 = basket_equity.rolling(200, min_periods=100).mean()
    out["basket_above_200d"] = basket_equity > basket_ma_200

    if close.empty:
        out["breadth_above_200d"] = np.nan
        out["breadth_positive_63d"] = np.nan
        out["breadth_positive_126d"] = np.nan
    else:
        asset_ma_200 = close.rolling(200, min_periods=100).mean()
        asset_ret_63 = close.pct_change(63)
        asset_ret_126 = close.pct_change(126)

        out["breadth_above_200d"] = (close > asset_ma_200).mean(axis=1)
        out["breadth_positive_63d"] = (asset_ret_63 > 0).mean(axis=1)
        out["breadth_positive_126d"] = (asset_ret_126 > 0).mean(axis=1)

    grouped = scores.groupby("date")
    score_stats = pd.DataFrame(index=sorted(scores["date"].unique()))
    score_stats.index = pd.to_datetime(score_stats.index)
    score_stats.index.name = "date"
    score_stats["avg_final_score"] = grouped["final_score"].mean()
    score_stats["median_final_score"] = grouped["final_score"].median()
    score_stats["min_final_score"] = grouped["final_score"].min()
    score_stats["investable_score_share"] = grouped["final_score"].apply(
        lambda x: float((x >= MIN_SCORE_TO_HOLD).mean())
    )
    score_stats["weak_score_share_030"] = grouped["final_score"].apply(
        lambda x: float((x < 0.30).mean())
    )
    score_stats["weak_score_share_040"] = grouped["final_score"].apply(
        lambda x: float((x < 0.40).mean())
    )

    out = out.join(score_stats, how="left")
    score_cols = [
        "avg_final_score",
        "median_final_score",
        "min_final_score",
        "investable_score_share",
        "weak_score_share_030",
        "weak_score_share_040",
    ]
    out[score_cols] = out[score_cols].ffill()

    # Vote system. This prevents one noisy feature from dominating, but it also
    # avoids the previous failure mode where almost everything defaulted to chop.
    bear_votes = pd.DataFrame(index=out.index)
    bear_votes["basket_below_200d"] = ~out["basket_above_200d"].fillna(False)
    bear_votes["basket_return_126d_lt_minus_5"] = out["basket_return_126d"] < -0.05
    bear_votes["basket_return_252d_lt_minus_8"] = out["basket_return_252d"] < -0.08
    bear_votes["basket_drawdown_252d_lt_minus_10"] = out["basket_drawdown_252d"] < -0.10
    bear_votes["breadth_above_200d_lt_40"] = out["breadth_above_200d"] < 0.40
    bear_votes["breadth_positive_63d_lt_40"] = out["breadth_positive_63d"] < 0.40
    bear_votes["breadth_positive_126d_lt_40"] = out["breadth_positive_126d"] < 0.40

    bull_votes = pd.DataFrame(index=out.index)
    bull_votes["basket_above_200d"] = out["basket_above_200d"].fillna(False)
    bull_votes["basket_return_126d_gt_5"] = out["basket_return_126d"] > 0.05
    bull_votes["basket_return_252d_gt_8"] = out["basket_return_252d"] > 0.08
    bull_votes["basket_drawdown_252d_gt_minus_5"] = out["basket_drawdown_252d"] > -0.05
    bull_votes["breadth_above_200d_gt_60"] = out["breadth_above_200d"] > 0.60
    bull_votes["breadth_positive_63d_gt_60"] = out["breadth_positive_63d"] > 0.60
    bull_votes["breadth_positive_126d_gt_60"] = out["breadth_positive_126d"] > 0.60

    out["bear_vote_count"] = bear_votes.sum(axis=1).astype(int)
    out["bull_vote_count"] = bull_votes.sum(axis=1).astype(int)

    # Main labels. Bear takes precedence over bull if both are high, because the
    # purpose of this script is to prevent missing hostile commodity regimes.
    bear = out["bear_vote_count"] >= 3
    bull = (out["bull_vote_count"] >= 4) & ~bear

    out["regime"] = np.select([bear, bull], ["bear", "bull"], default="chop")

    # No median smoothing here. The previous smoothing/default-to-chop behaviour
    # hid the actual stress state. Monthly rebalancing already reduces noise.
    out["smoothed_regime"] = out["regime"]

    out["regime_model"] = "price_breadth_v2"

    return out.reset_index()


def force_bear_window(
    regime_table: pd.DataFrame,
    *,
    start: str = "2013-01-01",
    end: str = "2015-12-31",
    label: str = "forced_2013_2015_bear",
) -> pd.DataFrame:
    """
    Research-only counterfactual.

    This answers: if the system had recognised 2013-2015 as a commodity bear
    regime, would cash/short rules have helped? It is not production logic.
    """
    out = regime_table.copy()
    out["date"] = pd.to_datetime(out["date"])
    mask = (out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))
    out.loc[mask, "regime"] = "bear"
    out.loc[mask, "smoothed_regime"] = "bear"
    out["regime_model"] = label
    out["forced_bear_window"] = False
    out.loc[mask, "forced_bear_window"] = True
    return out


def build_regime_audit_tables(regime_tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    yearly_rows = []
    stress_rows = []

    for model_name, table in regime_tables.items():
        t = table.copy()
        t["date"] = pd.to_datetime(t["date"])
        t["year"] = t["date"].dt.year

        yearly = (
            t.groupby(["year", "smoothed_regime"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        for col in ["bull", "chop", "bear"]:
            if col not in yearly.columns:
                yearly[col] = 0
        yearly["regime_model"] = model_name
        yearly_rows.append(yearly[["regime_model", "year", "bull", "chop", "bear"]])

        for period_name, (start, end) in STRESS_PERIODS.items():
            section = t[(t["date"] >= pd.Timestamp(start)) & (t["date"] <= pd.Timestamp(end))]
            counts = section["smoothed_regime"].value_counts()
            stress_rows.append(
                {
                    "regime_model": model_name,
                    "period": period_name,
                    "start": start,
                    "end": end,
                    "days": int(len(section)),
                    "bull_days": int(counts.get("bull", 0)),
                    "chop_days": int(counts.get("chop", 0)),
                    "bear_days": int(counts.get("bear", 0)),
                    "bear_share": float(counts.get("bear", 0) / len(section)) if len(section) else np.nan,
                }
            )

    yearly_out = pd.concat(yearly_rows, ignore_index=True) if yearly_rows else pd.DataFrame()
    stress_out = pd.DataFrame(stress_rows)
    return yearly_out, stress_out


# ============================================================
# SHORT OVERLAY CONSTRUCTION
# ============================================================

def get_short_candidates(
    scores_today: pd.Series,
    regime_today: str,
    trial: ShortTrialConfig,
) -> pd.Index:
    if regime_today not in trial.allowed_regimes:
        return pd.Index([])

    scores_today = pd.to_numeric(scores_today, errors="coerce").dropna()
    candidates = scores_today[scores_today < trial.short_threshold].index

    return pd.Index(candidates)


def build_short_weights(
    scores_today: pd.Series,
    regime_today: str,
    trial: ShortTrialConfig,
) -> pd.Series:
    short_weights = pd.Series(0.0, index=scores_today.index, dtype=float)

    candidates = get_short_candidates(
        scores_today=scores_today,
        regime_today=regime_today,
        trial=trial,
    )

    if len(candidates) == 0:
        return short_weights

    n = len(candidates)
    total_short = min(float(trial.max_total_short), n * float(trial.max_single_short))
    per_asset_short = min(float(trial.max_single_short), total_short / n)

    short_weights.loc[candidates] = -per_asset_short

    return short_weights


def combine_long_short_weights(
    long_weights: pd.Series,
    short_weights: pd.Series,
    max_gross: float = MAX_TARGET_GROSS_EXPOSURE,
) -> pd.Series:
    """
    Conservative no-leverage combination.

    - Short candidates override long positions.
    - Existing long allocation is only scaled down if needed to make room.
    - Gross target exposure is kept <= max_gross.
    """
    tickers = long_weights.index.union(short_weights.index)
    longs = long_weights.reindex(tickers).fillna(0.0).clip(lower=0.0)
    shorts = short_weights.reindex(tickers).fillna(0.0).clip(upper=0.0)

    active_shorts = shorts[shorts < 0].index
    if len(active_shorts) > 0:
        longs.loc[active_shorts] = 0.0

    short_gross = float(shorts.abs().sum())
    long_gross = float(longs.sum())

    long_budget = max(0.0, float(max_gross) - short_gross)

    if long_gross > long_budget and long_gross > 0:
        longs = longs * (long_budget / long_gross)

    final = longs + shorts
    final = signed_normalise_weights(final, max_gross=max_gross)

    return final.reindex(long_weights.index).fillna(0.0)


def build_short_trial_weight_matrix(
    *,
    base_long_weights: pd.DataFrame,
    score_wide: pd.DataFrame,
    regime_table: pd.DataFrame,
    trial: ShortTrialConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = list(base_long_weights.columns)
    dates = pd.to_datetime(base_long_weights.index)

    scores_aligned = (
        score_wide
        .reindex(dates)
        .ffill()
        .reindex(columns=tickers)
        .fillna(0.0)
    )

    regimes = (
        regime_table
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
        .sort_index()["smoothed_regime"]
        .reindex(dates)
        .ffill()
        .fillna("chop")
    )

    weights_rows = []
    diagnostic_rows = []

    for date in dates:
        base_longs = base_long_weights.loc[date].reindex(tickers).fillna(0.0)
        scores_today = scores_aligned.loc[date].reindex(tickers).fillna(0.0)
        regime_today = str(regimes.loc[date])

        short_weights = build_short_weights(
            scores_today=scores_today,
            regime_today=regime_today,
            trial=trial,
        )

        final_weights = combine_long_short_weights(
            long_weights=base_longs,
            short_weights=short_weights,
            max_gross=trial.max_target_gross,
        )

        weights_rows.append(final_weights.rename(date))

        active_short = short_weights[short_weights < 0]
        active_long = final_weights[final_weights > 0]

        diagnostic_rows.append(
            {
                "date": date,
                "strategy": trial.name,
                "regime": regime_today,
                "short_threshold": trial.short_threshold,
                "allowed_regimes": ",".join(trial.allowed_regimes),
                "num_short_candidates": int(len(active_short)),
                "short_candidates": ",".join(active_short.index.tolist()),
                "long_gross": float(active_long.sum()),
                "short_gross": float(active_short.abs().sum()),
                "gross_exposure": float(final_weights.abs().sum()),
                "net_exposure": float(final_weights.sum()),
                "cash_weight": float(max(0.0, 1.0 - final_weights.abs().sum())),
                "avg_score": float(scores_today.mean()),
                "min_score": float(scores_today.min()),
                "max_score": float(scores_today.max()),
            }
        )

    weights = pd.DataFrame(weights_rows)
    weights.index.name = "date"
    weights = weights.sort_index().reindex(columns=tickers).fillna(0.0)

    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(["strategy", "date"])

    return weights, diagnostics


def build_bear_cash_brake_weights(
    *,
    base_long_weights: pd.DataFrame,
    regime_table: pd.DataFrame,
) -> pd.DataFrame:
    dates = pd.to_datetime(base_long_weights.index)

    regimes = (
        regime_table
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
        .sort_index()["smoothed_regime"]
        .reindex(dates)
        .ffill()
        .fillna("chop")
    )

    out = base_long_weights.copy().sort_index().fillna(0.0)
    bear_dates = regimes[regimes == "bear"].index
    out.loc[out.index.intersection(bear_dates), :] = 0.0

    return out


# ============================================================
# SIGNED-WEIGHT REALISTIC SIMULATOR
# ============================================================

def build_rebalance_signal_weights_signed(
    raw_weights: pd.DataFrame,
    market_dates: pd.Index,
    mode: str,
) -> pd.DataFrame:
    raw_weights = raw_weights.sort_index().fillna(0.0)

    daily_weights = (
        raw_weights
        .reindex(market_dates)
        .ffill()
        .fillna(0.0)
    )

    daily_weights = daily_weights.apply(signed_normalise_weights, axis=1)

    if mode == "daily":
        return daily_weights

    if mode == "weekly":
        periods = daily_weights.index.to_period("W-FRI")
    elif mode == "monthly":
        periods = daily_weights.index.to_period("M")
    else:
        raise ValueError("BACKTEST_REBALANCE_MODE must be 'daily', 'weekly', or 'monthly'.")

    signal_dates = (
        pd.Series(daily_weights.index, index=daily_weights.index)
        .groupby(periods)
        .last()
        .tolist()
    )

    return daily_weights.loc[signal_dates].copy()


def apply_volatility_targeting_signed(
    daily_weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    if not V2.VOL_TARGETING_ENABLED:
        return daily_weights.fillna(0.0)

    weights = daily_weights.sort_index().fillna(0.0)
    returns = returns.sort_index().fillna(0.0)

    common_dates = weights.index.intersection(returns.index)
    common_tickers = weights.columns.intersection(returns.columns)

    if len(common_dates) == 0 or len(common_tickers) == 0:
        return weights

    aligned_weights = weights.loc[common_dates, common_tickers]
    aligned_returns = returns.loc[common_dates, common_tickers]

    pre_target_returns = (aligned_weights.shift(1).fillna(0.0) * aligned_returns).sum(axis=1)

    realised_vol = (
        pre_target_returns
        .rolling(V2.VOL_TARGET_LOOKBACK_DAYS)
        .std()
        * np.sqrt(V2.TRADING_DAYS_PER_YEAR)
    )

    trigger_vol = V2.TARGET_PORTFOLIO_VOL * V2.VOL_TARGET_VOL_BUFFER
    raw_scale = V2.TARGET_PORTFOLIO_VOL / realised_vol

    scale = pd.Series(1.0, index=common_dates)
    scale = scale.where(realised_vol <= trigger_vol, raw_scale)
    scale = (
        scale
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=V2.VOL_TARGET_MIN_SCALE, upper=V2.VOL_TARGET_MAX_SCALE)
        .shift(1)
        .fillna(1.0)
    )

    out = weights.copy()
    out.loc[common_dates, common_tickers] = aligned_weights.multiply(scale, axis=0)
    out = out.apply(signed_normalise_weights, axis=1)

    return out.fillna(0.0)


def prepare_signal_weights_signed(
    raw_weights: pd.DataFrame,
    returns: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    market_dates = returns.index

    scheduled = build_rebalance_signal_weights_signed(
        raw_weights=raw_weights,
        market_dates=market_dates,
        mode=mode,
    )

    daily_scheduled = (
        scheduled
        .reindex(market_dates)
        .ffill()
        .fillna(0.0)
    )

    daily_scheduled = apply_volatility_targeting_signed(
        daily_weights=daily_scheduled,
        returns=returns,
    )

    signal_weights = daily_scheduled.loc[scheduled.index].copy()
    signal_weights = signal_weights.apply(signed_normalise_weights, axis=1)

    return signal_weights


def build_execution_plan_signed(
    signal_weights: pd.DataFrame,
    market_dates: pd.Index,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    delay = int(settings["execution_delay_days"])

    date_positions = pd.Series(range(len(market_dates)), index=market_dates)

    execution_rows = []
    execution_dates = []
    signal_dates = []

    for signal_date, row in signal_weights.iterrows():
        if signal_date not in date_positions.index:
            continue

        signal_pos = int(date_positions.loc[signal_date])
        execution_pos = signal_pos + delay

        if execution_pos >= len(market_dates):
            continue

        execution_date = market_dates[execution_pos]

        execution_rows.append(row)
        execution_dates.append(execution_date)
        signal_dates.append(signal_date)

    if not execution_rows:
        empty_plan = pd.DataFrame(columns=signal_weights.columns)
        empty_signal_dates = pd.Series(dtype="datetime64[ns]")
        return empty_plan, empty_signal_dates

    execution_plan = pd.DataFrame(
        execution_rows,
        index=pd.to_datetime(execution_dates),
        columns=signal_weights.columns,
    ).sort_index()

    signal_date_by_execution = pd.Series(
        pd.to_datetime(signal_dates),
        index=pd.to_datetime(execution_dates),
        name="signal_date",
    ).sort_index()

    if execution_plan.index.has_duplicates:
        execution_plan = execution_plan.groupby(level=0).last()
        signal_date_by_execution = signal_date_by_execution.groupby(level=0).last()

    execution_plan = execution_plan.apply(signed_normalise_weights, axis=1)

    return execution_plan, signal_date_by_execution


def simulate_strategy_signed_v3(
    name: str,
    raw_target_weights: pd.DataFrame,
    market_data: dict[str, pd.DataFrame],
    settings: dict[str, Any],
    initial_capital: float = V2.INITIAL_CAPITAL,
    short_borrow_cost_annual: float = SHORT_BORROW_COST_ANNUAL,
) -> dict[str, pd.DataFrame]:
    """
    V2-compatible simulator with signed weights.

    Reuses V2 settings, market data, transaction-cost assumptions, turnover
    controls, execution delay and liquidity caps. It does not use V2's long-only
    normalisation path because that clips shorts to zero.
    """
    returns = market_data["returns"].sort_index().fillna(0.0)
    adv = market_data["adv"].sort_index()

    tickers = list(returns.columns)
    market_dates = returns.index

    raw_target_weights = (
        raw_target_weights
        .sort_index()
        .reindex(columns=tickers)
        .fillna(0.0)
    )

    signal_weights = prepare_signal_weights_signed(
        raw_weights=raw_target_weights,
        returns=returns,
        mode=V2.BACKTEST_REBALANCE_MODE,
    )

    execution_plan, signal_date_by_execution = build_execution_plan_signed(
        signal_weights=signal_weights,
        market_dates=market_dates,
        settings=settings,
    )

    cost_table = V2.get_transaction_cost_assumptions(
        tickers=tickers,
        settings=settings,
    )

    cash_daily_return = (1.0 + V2.CASH_ANNUAL_YIELD) ** (1.0 / V2.TRADING_DAYS_PER_YEAR) - 1.0
    borrow_daily_rate = (1.0 + short_borrow_cost_annual) ** (1.0 / V2.TRADING_DAYS_PER_YEAR) - 1.0

    weights = pd.Series(0.0, index=tickers, dtype=float)
    equity = float(initial_capital)

    curve_rows = []
    trade_rows = []
    executed_weight_rows = []

    for date in market_dates:
        start_equity = equity
        asset_returns = returns.loc[date].reindex(tickers).fillna(0.0)

        starting_weights = weights.copy()
        starting_long_exposure = float(starting_weights.clip(lower=0.0).sum())
        starting_short_exposure = float(starting_weights.clip(upper=0.0).abs().sum())
        starting_gross_exposure = starting_long_exposure + starting_short_exposure
        starting_net_exposure = float(starting_weights.sum())
        starting_cash_weight = max(0.0, 1.0 - starting_gross_exposure)

        gross_return = float((starting_weights * asset_returns).sum())
        cash_return = float(starting_cash_weight * cash_daily_return)
        short_borrow_cost_drag = float(starting_short_exposure * borrow_daily_rate)
        pre_cost_return = gross_return + cash_return - short_borrow_cost_drag

        equity_before_costs = equity * (1.0 + pre_cost_return)

        if settings["use_portfolio_weight_drift"]:
            denominator = 1.0 + pre_cost_return
            if denominator > 0:
                drifted_weights = starting_weights * (1.0 + asset_returns) / denominator
                weights = drifted_weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        equity = equity_before_costs

        commission_cost = 0.0
        spread_cost = 0.0
        slippage_cost = 0.0
        legacy_flat_cost = 0.0
        total_trade_cost = 0.0

        raw_desired_turnover = 0.0
        turnover_after_no_trade_band = 0.0
        executed_turnover_before_liquidity = 0.0
        executed_turnover = 0.0
        no_trade_band_count = 0
        turnover_capped = False
        turnover_scale = 1.0
        partial_rebalance_fraction = 1.0
        execution_event = False
        signal_date = pd.NaT
        tracking_error_to_target = np.nan
        liquidity_capped_trade_count = 0

        if date in execution_plan.index:
            execution_event = True
            signal_date = signal_date_by_execution.loc[date]

            target_weights = execution_plan.loc[date].reindex(tickers).fillna(0.0)
            target_weights = signed_normalise_weights(target_weights)

            previous_weights_before_trade = weights.copy()
            desired_trades = target_weights - weights

            controlled_trades, turnover_diagnostics = V2.apply_turnover_controls(
                desired_trades=desired_trades,
                settings=settings,
            )

            raw_desired_turnover = turnover_diagnostics["raw_desired_turnover"]
            turnover_after_no_trade_band = turnover_diagnostics["turnover_after_no_trade_band"]
            executed_turnover_before_liquidity = turnover_diagnostics["executed_turnover_before_liquidity"]
            no_trade_band_count = turnover_diagnostics["no_trade_band_count"]
            turnover_capped = turnover_diagnostics["turnover_capped"]
            turnover_scale = turnover_diagnostics["turnover_scale"]
            partial_rebalance_fraction = turnover_diagnostics["partial_rebalance_fraction"]

            executed_trades, liquidity_diagnostics = V2.apply_liquidity_caps(
                trades=controlled_trades,
                date=date,
                equity=equity,
                initial_capital=initial_capital,
                adv=adv,
                settings=settings,
            )

            executed_turnover = float(executed_trades.abs().sum())

            for ticker in tickers:
                executed_trade_weight = float(executed_trades.get(ticker, 0.0))

                if abs(executed_trade_weight) < 1e-12:
                    continue

                cost_info = V2.calculate_trade_costs(
                    ticker=ticker,
                    executed_trade_weight=executed_trade_weight,
                    equity_before_costs=equity,
                    cost_table=cost_table,
                    settings=settings,
                )

                liq = liquidity_diagnostics.get(ticker, {})

                commission_cost += cost_info["commission_cost"]
                spread_cost += cost_info["spread_cost"]
                slippage_cost += cost_info["slippage_cost"]
                legacy_flat_cost += cost_info["legacy_flat_cost"]
                total_trade_cost += cost_info["total_trade_cost"]

                if bool(liq.get("liquidity_capped", False)):
                    liquidity_capped_trade_count += 1

                trade_rows.append(
                    {
                        "strategy": name,
                        "date": date,
                        "signal_date": signal_date,
                        "execution_date": date,
                        "execution_delay_days": settings["execution_delay_days"],
                        "cost_scenario": settings["cost_scenario"],
                        "ticker": ticker,
                        "previous_weight": float(previous_weights_before_trade.get(ticker, 0.0)),
                        "target_weight": float(target_weights.get(ticker, 0.0)),
                        "desired_trade_weight": float(desired_trades.get(ticker, 0.0)),
                        "controlled_trade_weight": float(controlled_trades.get(ticker, 0.0)),
                        "executed_trade_weight": executed_trade_weight,
                        "unfilled_trade_weight": float(liq.get("unfilled_trade_weight", 0.0)),
                        "trade_notional": cost_info["trade_notional"],
                        "liquidity_test_trade_notional": liq.get("liquidity_test_trade_notional", np.nan),
                        "adv_dollar": liq.get("adv_dollar", np.nan),
                        "max_trade_notional": liq.get("max_trade_notional", np.nan),
                        "participation_rate": liq.get("participation_rate", np.nan),
                        "fill_ratio": liq.get("fill_ratio", np.nan),
                        "liquidity_capped": bool(liq.get("liquidity_capped", False)),
                        "missing_adv": bool(liq.get("missing_adv", False)),
                        "commission_cost": cost_info["commission_cost"],
                        "spread_cost": cost_info["spread_cost"],
                        "slippage_cost": cost_info["slippage_cost"],
                        "legacy_flat_cost": cost_info["legacy_flat_cost"],
                        "total_trade_cost": cost_info["total_trade_cost"],
                    }
                )

            if total_trade_cost > 0:
                equity = max(0.0, equity - total_trade_cost)

            weights = weights + executed_trades
            weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            tracking_error_to_target = float((target_weights - weights).abs().sum())

        if start_equity > 0:
            net_return = (equity / start_equity) - 1.0
            commission_cost_drag = commission_cost / start_equity
            spread_cost_drag = spread_cost / start_equity
            slippage_cost_drag = slippage_cost / start_equity
            legacy_flat_cost_drag = legacy_flat_cost / start_equity
            total_transaction_cost_drag = total_trade_cost / start_equity
        else:
            net_return = 0.0
            commission_cost_drag = 0.0
            spread_cost_drag = 0.0
            slippage_cost_drag = 0.0
            legacy_flat_cost_drag = 0.0
            total_transaction_cost_drag = 0.0

        ending_long_exposure = float(weights.clip(lower=0.0).sum())
        ending_short_exposure = float(weights.clip(upper=0.0).abs().sum())
        ending_gross_exposure = ending_long_exposure + ending_short_exposure
        ending_net_exposure = float(weights.sum())
        ending_cash_weight = max(0.0, 1.0 - ending_gross_exposure)

        curve_rows.append(
            {
                "date": date,
                "strategy": name,
                "signal_date": signal_date,
                "execution_date": date if execution_event else pd.NaT,
                "execution_event": execution_event,
                "execution_delay_days": settings["execution_delay_days"],
                "cost_scenario": settings["cost_scenario"],
                "gross_return": gross_return,
                "cash_return": cash_return,
                "short_borrow_cost_drag": short_borrow_cost_drag,
                "pre_cost_return": pre_cost_return,
                "commission_cost_drag": commission_cost_drag,
                "spread_cost_drag": spread_cost_drag,
                "slippage_cost_drag": slippage_cost_drag,
                "legacy_flat_cost_drag": legacy_flat_cost_drag,
                "total_transaction_cost_drag": total_transaction_cost_drag,
                "net_return": net_return,
                "equity": equity,
                "raw_desired_turnover": raw_desired_turnover,
                "turnover_after_no_trade_band": turnover_after_no_trade_band,
                "executed_turnover_before_liquidity": executed_turnover_before_liquidity,
                "turnover": executed_turnover,
                "no_trade_band_count": no_trade_band_count,
                "turnover_capped": turnover_capped,
                "turnover_scale": turnover_scale,
                "partial_rebalance_fraction": partial_rebalance_fraction,
                "liquidity_capped_trade_count": liquidity_capped_trade_count,
                "tracking_error_to_target": tracking_error_to_target,
                "starting_long_exposure": starting_long_exposure,
                "starting_short_exposure": starting_short_exposure,
                "starting_gross_exposure": starting_gross_exposure,
                "starting_net_exposure": starting_net_exposure,
                "starting_cash_weight": starting_cash_weight,
                "long_exposure": ending_long_exposure,
                "short_exposure": ending_short_exposure,
                "gross_exposure": ending_gross_exposure,
                "net_exposure": ending_net_exposure,
                # Keep V2-compatible names. For signed strategies, exposure = gross exposure.
                "starting_exposure": starting_gross_exposure,
                "exposure": ending_gross_exposure,
                "cash_weight": ending_cash_weight,
            }
        )

        executed_weight_row = {"date": date, "strategy": name}
        executed_weight_row.update(weights.to_dict())
        executed_weight_rows.append(executed_weight_row)

    curve = pd.DataFrame(curve_rows)
    curve["date"] = pd.to_datetime(curve["date"])
    curve = curve.set_index("date").sort_index()
    curve["drawdown"] = calculate_drawdown_series(curve["net_return"])

    trade_log = pd.DataFrame(trade_rows)

    executed_weights = pd.DataFrame(executed_weight_rows)
    executed_weights["date"] = pd.to_datetime(executed_weights["date"])
    executed_weights = executed_weights.set_index("date").sort_index()

    return {
        "curve": curve,
        "trade_log": trade_log,
        "executed_weights": executed_weights,
        "signal_weights": signal_weights,
        "execution_plan": execution_plan,
    }


# ============================================================
# DIAGNOSTIC TABLES
# ============================================================

def build_engine_sync_check(
    v2_result: dict[str, pd.DataFrame],
    signed_result: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    v2_curve = v2_result["curve"].copy()
    signed_curve = signed_result["curve"].copy()

    common = v2_curve.index.intersection(signed_curve.index)
    if len(common) == 0:
        return pd.DataFrame()

    checks = []
    for col in ["net_return", "equity", "turnover", "total_transaction_cost_drag"]:
        if col not in v2_curve.columns or col not in signed_curve.columns:
            continue

        diff = (v2_curve.loc[common, col] - signed_curve.loc[common, col]).abs()
        checks.append(
            {
                "field": col,
                "max_abs_difference": float(diff.max()),
                "mean_abs_difference": float(diff.mean()),
            }
        )

    return pd.DataFrame(checks)


def build_yearly_returns(results: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []

    for name, result in results.items():
        returns = result["curve"]["net_return"].replace([np.inf, -np.inf], np.nan).dropna()
        if returns.empty:
            continue

        yearly = (1.0 + returns).resample("YE").prod() - 1.0

        for date, value in yearly.items():
            rows.append(
                {
                    "strategy": name,
                    "year": int(date.year),
                    "return": float(value),
                }
            )

    return pd.DataFrame(rows).sort_values(["strategy", "year"]).reset_index(drop=True)


def _period_metrics(returns: pd.Series, periods_per_year: int = V2.TRADING_DAYS_PER_YEAR) -> dict[str, float]:
    returns = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()

    if returns.empty:
        return {
            "days": 0,
            "total_return": np.nan,
            "annualised_return": np.nan,
            "annualised_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "hit_rate": np.nan,
        }

    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    ann_return = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if pd.notna(ann_vol) and ann_vol > 0 else np.nan

    return {
        "days": int(len(returns)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "annualised_return": float(ann_return),
        "annualised_volatility": float(ann_vol),
        "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((returns > 0).mean()),
    }


def build_stress_period_table(results: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []

    for strategy, result in results.items():
        curve = result["curve"].copy()
        curve.index = pd.to_datetime(curve.index)

        for period_name, (start, end) in STRESS_PERIODS.items():
            section = curve.loc[pd.Timestamp(start): pd.Timestamp(end)].copy()

            if section.empty:
                continue

            row = {
                "strategy": strategy,
                "period": period_name,
                "start": start,
                "end": end,
                **_period_metrics(section["net_return"]),
                "average_gross_exposure": float(section.get("gross_exposure", section["exposure"]).mean()),
                "average_net_exposure": float(section.get("net_exposure", section["exposure"]).mean()),
                "average_short_exposure": float(section.get("short_exposure", pd.Series(0.0, index=section.index)).mean()),
            }
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["period", "strategy"]).reset_index(drop=True)


def build_regime_performance_table(
    results: dict[str, dict[str, pd.DataFrame]],
    regime_table: pd.DataFrame,
) -> pd.DataFrame:
    regimes = (
        regime_table
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
        .sort_index()["smoothed_regime"]
    )

    rows = []

    for strategy, result in results.items():
        curve = result["curve"].copy()
        curve.index = pd.to_datetime(curve.index)

        joined = curve.join(regimes.rename("regime"), how="left")
        joined["regime"] = joined["regime"].ffill().fillna("chop")

        for regime, group in joined.groupby("regime"):
            metrics = _period_metrics(group["net_return"])
            rows.append(
                {
                    "strategy": strategy,
                    "regime": regime,
                    **metrics,
                    "average_gross_exposure": float(group.get("gross_exposure", group["exposure"]).mean()),
                    "average_net_exposure": float(group.get("net_exposure", group["exposure"]).mean()),
                    "average_short_exposure": float(group.get("short_exposure", pd.Series(0.0, index=group.index)).mean()),
                }
            )

    return pd.DataFrame(rows).sort_values(["strategy", "regime"]).reset_index(drop=True)


def build_contribution_tables(
    results: dict[str, dict[str, pd.DataFrame]],
    market_data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = market_data["returns"].sort_index().fillna(0.0)
    daily_rows = []
    summary_rows = []

    for strategy, result in results.items():
        weights = result["executed_weights"].copy()
        if "strategy" in weights.columns:
            weights = weights.drop(columns=["strategy"])

        tickers = [c for c in weights.columns if c in returns.columns]
        if not tickers:
            continue

        aligned_weights = weights[tickers].sort_index().reindex(returns.index).ffill().fillna(0.0)
        aligned_returns = returns[tickers].sort_index().reindex(aligned_weights.index).fillna(0.0)

        # Start-of-day weights drive that day's asset contribution.
        start_weights = aligned_weights.shift(1).fillna(0.0)
        long_weights = start_weights.clip(lower=0.0)
        short_weights = start_weights.clip(upper=0.0)

        long_contribution = (long_weights * aligned_returns).sum(axis=1)
        short_contribution = (short_weights * aligned_returns).sum(axis=1)
        asset_contribution = long_contribution + short_contribution

        daily = pd.DataFrame(
            {
                "date": aligned_weights.index,
                "strategy": strategy,
                "long_contribution": long_contribution.values,
                "short_contribution": short_contribution.values,
                "asset_contribution": asset_contribution.values,
                "gross_exposure_start": start_weights.abs().sum(axis=1).values,
                "net_exposure_start": start_weights.sum(axis=1).values,
                "short_exposure_start": short_weights.abs().sum(axis=1).values,
            }
        )
        daily_rows.append(daily)

        summary_rows.append(
            {
                "strategy": strategy,
                "sum_long_contribution": float(long_contribution.sum()),
                "sum_short_contribution": float(short_contribution.sum()),
                "sum_asset_contribution": float(asset_contribution.sum()),
                "average_gross_exposure_start": float(start_weights.abs().sum(axis=1).mean()),
                "average_net_exposure_start": float(start_weights.sum(axis=1).mean()),
                "average_short_exposure_start": float(short_weights.abs().sum(axis=1).mean()),
            }
        )

    daily_out = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    summary_out = pd.DataFrame(summary_rows)

    return daily_out, summary_out


# ============================================================
# CHARTS
# ============================================================

def setup_chart_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (12, 6),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_equity_curves(results: dict[str, dict[str, pd.DataFrame]]) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))

    preferred_order = [
        "baseline_v3_long_only",
        "bear_cash_brake",
        "forced_2013_2015_cash_brake",
        "short_020_all_regimes",
        "short_030_bear_only",
        "short_035_bear_only",
        "short_040_bear_only",
        "short_020_bear_chop_only",
        "forced_2013_2015_short_030",
        "forced_2013_2015_short_035",
        "forced_2013_2015_short_040",
        "equal_weight",
        "gold_only",
        "cash",
    ]

    for name in preferred_order:
        if name not in results:
            continue
        curve = results[name]["curve"]
        ax.plot(curve.index, curve["equity"], label=name)

    ax.set_title("V3 short-trial equity curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend(loc="best", fontsize=9)

    save_figure(fig, CHARTS_DIR / "equity_curves.png")


def plot_drawdowns(results: dict[str, dict[str, pd.DataFrame]]) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))

    for name in [
        "baseline_v3_long_only",
        "bear_cash_brake",
        "forced_2013_2015_cash_brake",
        "short_020_all_regimes",
        "short_030_bear_only",
        "short_035_bear_only",
        "short_040_bear_only",
        "short_020_bear_chop_only",
        "forced_2013_2015_short_030",
        "forced_2013_2015_short_035",
        "forced_2013_2015_short_040",
    ]:
        if name not in results:
            continue
        curve = results[name]["curve"]
        dd = curve["drawdown"] if "drawdown" in curve.columns else calculate_drawdown_series(curve["net_return"])
        ax.plot(dd.index, dd, label=name)

    ax.set_title("Drawdown comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="best", fontsize=9)

    save_figure(fig, CHARTS_DIR / "drawdowns.png")


def plot_metric_bars(performance_summary: pd.DataFrame) -> None:
    if performance_summary.empty or "strategy" not in performance_summary.columns:
        return

    strategies = [
        "baseline_v3_long_only",
        "bear_cash_brake",
        "forced_2013_2015_cash_brake",
        "short_020_all_regimes",
        "short_030_bear_only",
        "short_035_bear_only",
        "short_040_bear_only",
        "short_020_bear_chop_only",
        "forced_2013_2015_short_030",
        "forced_2013_2015_short_035",
        "forced_2013_2015_short_040",
    ]

    df = performance_summary[performance_summary["strategy"].isin(strategies)].copy()
    if df.empty:
        return

    for metric, title, filename, pct in [
        ("cagr", "CAGR by strategy", "cagr_comparison.png", True),
        ("sharpe", "Sharpe by strategy", "sharpe_comparison.png", False),
        ("max_drawdown", "Max drawdown by strategy", "max_drawdown_comparison.png", True),
    ]:
        if metric not in df.columns:
            continue

        plot_df = df[["strategy", metric]].dropna().sort_values(metric)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(plot_df["strategy"], plot_df[metric])
        ax.set_title(title)
        ax.set_xlabel(metric)
        if pct:
            ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        save_figure(fig, CHARTS_DIR / filename)


def plot_stress_period_returns(stress_table: pd.DataFrame) -> None:
    if stress_table.empty:
        return

    period = "commodity_bear_2013_2015"
    df = stress_table[stress_table["period"] == period].copy()
    if df.empty:
        return

    strategies = [
        "baseline_v3_long_only",
        "bear_cash_brake",
        "forced_2013_2015_cash_brake",
        "short_020_all_regimes",
        "short_030_bear_only",
        "short_035_bear_only",
        "short_040_bear_only",
        "short_020_bear_chop_only",
        "forced_2013_2015_short_030",
        "forced_2013_2015_short_035",
        "forced_2013_2015_short_040",
    ]
    df = df[df["strategy"].isin(strategies)].copy()
    df = df.sort_values("total_return")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(df["strategy"], df["total_return"])
    ax.set_title("2013-2015 commodity bear stress return")
    ax.set_xlabel("Total return")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))

    save_figure(fig, CHARTS_DIR / "stress_2013_2015_returns.png")


def plot_short_exposure(short_diagnostics: pd.DataFrame) -> None:
    if short_diagnostics.empty:
        return

    fig, ax = plt.subplots(figsize=(13, 7))

    for strategy, group in short_diagnostics.groupby("strategy"):
        group = group.sort_values("date")
        ax.plot(group["date"], group["short_gross"], label=strategy)

    ax.set_title("Target short exposure through time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Short gross exposure")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="best", fontsize=9)

    save_figure(fig, CHARTS_DIR / "target_short_exposure.png")


def build_charts(
    *,
    results: dict[str, dict[str, pd.DataFrame]],
    performance_summary: pd.DataFrame,
    stress_table: pd.DataFrame,
    short_diagnostics: pd.DataFrame,
) -> None:
    setup_chart_style()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_equity_curves(results)
    plot_drawdowns(results)
    plot_metric_bars(performance_summary)
    plot_stress_period_returns(stress_table)
    plot_short_exposure(short_diagnostics)


# ============================================================
# OUTPUT SAVING
# ============================================================

def save_all_outputs(
    *,
    results: dict[str, dict[str, pd.DataFrame]],
    performance_summary: pd.DataFrame,
    alpha_beta_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    regime_table: pd.DataFrame,
    short_diagnostics: pd.DataFrame,
    yearly_returns: pd.DataFrame,
    stress_periods: pd.DataFrame,
    regime_performance: pd.DataFrame,
    contribution_daily: pd.DataFrame,
    contribution_summary: pd.DataFrame,
    engine_sync_check: pd.DataFrame,
    trial_weights: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    all_curves = []
    all_trade_logs = []
    all_executed_weights = []

    for name, result in results.items():
        safe_name = name.replace("/", "_")

        curve_path = OUTPUT_DIR / f"{safe_name}_curve.csv"
        trade_log_path = OUTPUT_DIR / f"{safe_name}_trade_log.csv"
        executed_path = OUTPUT_DIR / f"{safe_name}_executed_weights.csv"
        signal_path = OUTPUT_DIR / f"{safe_name}_signal_weights.csv"
        execution_path = OUTPUT_DIR / f"{safe_name}_execution_plan.csv"

        result["curve"].reset_index().to_csv(curve_path, index=False)
        result["trade_log"].to_csv(trade_log_path, index=False)
        result["executed_weights"].reset_index().to_csv(executed_path, index=False)
        result["signal_weights"].reset_index().to_csv(signal_path, index=False)
        result["execution_plan"].reset_index().rename(columns={"index": "execution_date"}).to_csv(
            execution_path,
            index=False,
        )

        curve_out = result["curve"].reset_index()
        all_curves.append(curve_out)

        if not result["trade_log"].empty:
            all_trade_logs.append(result["trade_log"])

        executed_out = result["executed_weights"].reset_index()
        all_executed_weights.append(executed_out)

    for name, weights in trial_weights.items():
        path = OUTPUT_DIR / f"{name}_target_weights.csv"
        weights.reset_index().rename(columns={"index": "date"}).to_csv(path, index=False)

    paths["performance_summary"] = OUTPUT_DIR / "short_trial_performance_summary.csv"
    paths["alpha_beta_summary"] = OUTPUT_DIR / "short_trial_alpha_beta_summary.csv"
    paths["cost_summary"] = OUTPUT_DIR / "short_trial_cost_summary.csv"
    paths["regime_table"] = OUTPUT_DIR / "short_trial_regime_by_date.csv"
    paths["short_diagnostics"] = OUTPUT_DIR / "short_trial_target_diagnostics.csv"
    paths["yearly_returns"] = OUTPUT_DIR / "short_trial_yearly_returns.csv"
    paths["stress_periods"] = OUTPUT_DIR / "short_trial_stress_periods.csv"
    paths["regime_performance"] = OUTPUT_DIR / "short_trial_regime_performance.csv"
    paths["contribution_daily"] = OUTPUT_DIR / "short_trial_contribution_daily.csv"
    paths["contribution_summary"] = OUTPUT_DIR / "short_trial_contribution_summary.csv"
    paths["engine_sync_check"] = OUTPUT_DIR / "short_trial_engine_sync_check.csv"

    performance_summary.to_csv(paths["performance_summary"], index=False)
    alpha_beta_summary.to_csv(paths["alpha_beta_summary"], index=False)
    cost_summary.to_csv(paths["cost_summary"], index=False)
    regime_table.to_csv(paths["regime_table"], index=False)
    short_diagnostics.to_csv(paths["short_diagnostics"], index=False)
    yearly_returns.to_csv(paths["yearly_returns"], index=False)
    stress_periods.to_csv(paths["stress_periods"], index=False)
    regime_performance.to_csv(paths["regime_performance"], index=False)
    contribution_daily.to_csv(paths["contribution_daily"], index=False)
    contribution_summary.to_csv(paths["contribution_summary"], index=False)
    engine_sync_check.to_csv(paths["engine_sync_check"], index=False)

    if all_curves:
        paths["all_curves"] = OUTPUT_DIR / "short_trial_all_curves.csv"
        pd.concat(all_curves, ignore_index=True).to_csv(paths["all_curves"], index=False)

    if all_trade_logs:
        paths["all_trade_logs"] = OUTPUT_DIR / "short_trial_all_trade_logs.csv"
        pd.concat(all_trade_logs, ignore_index=True).to_csv(paths["all_trade_logs"], index=False)
    else:
        paths["all_trade_logs"] = OUTPUT_DIR / "short_trial_all_trade_logs.csv"
        pd.DataFrame().to_csv(paths["all_trade_logs"], index=False)

    if all_executed_weights:
        paths["all_executed_weights"] = OUTPUT_DIR / "short_trial_all_executed_weights.csv"
        pd.concat(all_executed_weights, ignore_index=True).to_csv(paths["all_executed_weights"], index=False)

    return paths


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n========== BACKTEST V3 SHORT TRIALS ==========")
    print("This is a research script. It does not modify production weights or V4 routing.")

    settings = V2.build_base_settings()

    print("\nMain execution settings inherited from V2/V3:")
    for key, value in settings.items():
        print(f"  {key}: {value}")

    print("\nShort-trial settings:")
    print(f"  max_total_short: {MAX_TOTAL_SHORT:.2%}")
    print(f"  max_single_short: {MAX_SINGLE_SHORT:.2%}")
    print(f"  max_target_gross_exposure: {MAX_TARGET_GROSS_EXPOSURE:.2%}")
    print(f"  short_borrow_cost_annual: {SHORT_BORROW_COST_ANNUAL:.2%}")

    market_data = V2.load_market_data(settings=settings)
    returns = market_data["returns"]

    base_model_weights = V2.load_target_weights()
    tickers = list(base_model_weights.columns)
    base_model_weights = base_model_weights.reindex(columns=tickers).fillna(0.0)

    scores_history = load_scores_history()
    score_wide = build_score_wide(scores_history=scores_history, tickers=tickers)

    regime_table = build_regime_table(
        scores_history=scores_history,
        market_data=market_data,
        tickers=tickers,
    )
    forced_regime_table = force_bear_window(regime_table)

    regime_yearly_audit, regime_stress_audit = build_regime_audit_tables(
        {
            "price_breadth_v2": regime_table,
            "forced_2013_2015_bear": forced_regime_table,
        }
    )

    print("\nNatural price/breadth regime counts:")
    print(regime_table["smoothed_regime"].value_counts().to_string())

    print("\nStress-period regime audit:")
    print(regime_stress_audit.to_string(index=False))

    bear_cash_brake_weights = build_bear_cash_brake_weights(
        base_long_weights=base_model_weights,
        regime_table=regime_table,
    )
    forced_bear_cash_brake_weights = build_bear_cash_brake_weights(
        base_long_weights=base_model_weights,
        regime_table=forced_regime_table,
    )

    trial_weights: dict[str, pd.DataFrame] = {
        "baseline_v3_long_only": base_model_weights,
        "bear_cash_brake": bear_cash_brake_weights,
        "forced_2013_2015_cash_brake": forced_bear_cash_brake_weights,
    }

    short_diagnostics_frames = []

    for trial in SHORT_TRIALS:
        weights, diagnostics = build_short_trial_weight_matrix(
            base_long_weights=base_model_weights,
            score_wide=score_wide,
            regime_table=regime_table,
            trial=trial,
        )
        trial_weights[trial.name] = weights
        short_diagnostics_frames.append(diagnostics)

    for trial in FORCED_BEAR_TRIALS:
        weights, diagnostics = build_short_trial_weight_matrix(
            base_long_weights=base_model_weights,
            score_wide=score_wide,
            regime_table=forced_regime_table,
            trial=trial,
        )
        trial_weights[trial.name] = weights
        short_diagnostics_frames.append(diagnostics)

    short_diagnostics = (
        pd.concat(short_diagnostics_frames, ignore_index=True)
        if short_diagnostics_frames
        else pd.DataFrame()
    )

    if not short_diagnostics.empty:
        activation = (
            short_diagnostics
            .groupby("strategy")
            .agg(
                rebalance_rows=("date", "count"),
                active_short_rows=("short_gross", lambda x: int((x > 0).sum())),
                max_short_gross=("short_gross", "max"),
                avg_short_gross=("short_gross", "mean"),
            )
            .reset_index()
        )
        print("\nShort activation audit:")
        print(activation.to_string(index=False))

    equal_weight = V2.make_equal_weight(index=base_model_weights.index, tickers=tickers)
    gold_only = V2.make_gold_only(index=base_model_weights.index, tickers=tickers)
    cash = V2.make_cash(index=base_model_weights.index, tickers=tickers)

    results: dict[str, dict[str, pd.DataFrame]] = {}

    # Exact long-only V2 controls.
    v2_control_strategies = {
        "baseline_v3_long_only": base_model_weights,
        "bear_cash_brake": bear_cash_brake_weights,
        "forced_2013_2015_cash_brake": forced_bear_cash_brake_weights,
        "equal_weight": equal_weight,
        "gold_only": gold_only,
        "cash": cash,
    }

    for name, weights in v2_control_strategies.items():
        print(f"\nRunning exact V2/V3 long-only engine: {name}")
        results[name] = V2.simulate_strategy_v2(
            name=name,
            raw_target_weights=weights,
            market_data=market_data,
            settings=settings,
            initial_capital=V2.INITIAL_CAPITAL,
        )

    # Signed strategies require the signed simulator, because V2 clips shorts.
    for trial in [*SHORT_TRIALS, *FORCED_BEAR_TRIALS]:
        print(f"\nRunning signed short-trial engine: {trial.name}")
        results[trial.name] = simulate_strategy_signed_v3(
            name=trial.name,
            raw_target_weights=trial_weights[trial.name],
            market_data=market_data,
            settings=settings,
            initial_capital=V2.INITIAL_CAPITAL,
            short_borrow_cost_annual=SHORT_BORROW_COST_ANNUAL,
        )

    # Engine sync check: signed simulator should match V2 for long-only baseline.
    print("\nRunning signed-engine baseline control for sync check.")
    signed_baseline_control = simulate_strategy_signed_v3(
        name="baseline_signed_engine_control",
        raw_target_weights=base_model_weights,
        market_data=market_data,
        settings=settings,
        initial_capital=V2.INITIAL_CAPITAL,
        short_borrow_cost_annual=0.0,
    )

    engine_sync_check = build_engine_sync_check(
        v2_result=results["baseline_v3_long_only"],
        signed_result=signed_baseline_control,
    )

    performance_summary = V2.build_performance_summary(
        results=results,
        benchmark_name="equal_weight",
    )

    alpha_beta_summary = V2.build_alpha_beta_summary(
        results=results,
        benchmark_names=["equal_weight", "gold_only", "cash", "baseline_v3_long_only"],
    )

    cost_summary = V2.build_cost_summary_table(
        results=results,
        settings=settings,
    )

    yearly_returns = build_yearly_returns(results)
    stress_periods = build_stress_period_table(results)
    regime_performance = build_regime_performance_table(results, regime_table)
    contribution_daily, contribution_summary = build_contribution_tables(results, market_data)

    paths = save_all_outputs(
        results=results,
        performance_summary=performance_summary,
        alpha_beta_summary=alpha_beta_summary,
        cost_summary=cost_summary,
        regime_table=regime_table,
        short_diagnostics=short_diagnostics,
        yearly_returns=yearly_returns,
        stress_periods=stress_periods,
        regime_performance=regime_performance,
        contribution_daily=contribution_daily,
        contribution_summary=contribution_summary,
        engine_sync_check=engine_sync_check,
        trial_weights=trial_weights,
    )

    forced_regime_path = OUTPUT_DIR / "short_trial_forced_2013_2015_regime_by_date.csv"
    regime_yearly_audit_path = OUTPUT_DIR / "short_trial_regime_yearly_audit.csv"
    regime_stress_audit_path = OUTPUT_DIR / "short_trial_regime_stress_audit.csv"
    forced_regime_table.to_csv(forced_regime_path, index=False)
    regime_yearly_audit.to_csv(regime_yearly_audit_path, index=False)
    regime_stress_audit.to_csv(regime_stress_audit_path, index=False)

    build_charts(
        results=results,
        performance_summary=performance_summary,
        stress_table=stress_periods,
        short_diagnostics=short_diagnostics,
    )

    cols = [
        "strategy",
        "benchmark",
        "final_equity",
        "cagr",
        "annualised_volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "average_daily_turnover",
        "annualised_turnover",
        "total_transaction_cost_drag",
        "average_exposure",
        "average_cash",
        "alpha_annualised",
        "beta",
        "information_ratio",
    ]
    cols = [col for col in cols if col in performance_summary.columns]

    print("\nShort-trial backtest complete.")
    print(f"Saved outputs to: {OUTPUT_DIR}")
    print(f"Saved charts to:  {CHARTS_DIR}")

    print("\nPerformance summary:")
    print(performance_summary[cols].to_string(index=False))

    if not engine_sync_check.empty:
        print("\nSigned-engine sync check versus V2 long-only baseline:")
        print(engine_sync_check.to_string(index=False))

        max_diff = engine_sync_check["max_abs_difference"].max()
        if pd.notna(max_diff) and max_diff < 1e-8:
            print("\nEngine sync check: PASS. Signed engine matches V2 baseline to numerical precision.")
        else:
            print(
                "\nEngine sync check: REVIEW. Differences are not automatically fatal, "
                "but inspect short_trial_engine_sync_check.csv before trusting short results."
            )

    print("\nKey files to inspect first:")
    for key in [
        "performance_summary",
        "stress_periods",
        "regime_performance",
        "contribution_summary",
        "short_diagnostics",
        "engine_sync_check",
    ]:
        print(f"  {key}: {paths[key]}")
    print(f"  forced_regime_table: {forced_regime_path}")
    print(f"  regime_yearly_audit: {regime_yearly_audit_path}")
    print(f"  regime_stress_audit: {regime_stress_audit_path}")

    print("\nKey charts:")
    for chart_name in [
        "equity_curves.png",
        "drawdowns.png",
        "stress_2013_2015_returns.png",
        "target_short_exposure.png",
    ]:
        print(f"  {CHARTS_DIR / chart_name}")


if __name__ == "__main__":
    main()
