from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

try:
    from config import MIN_SCORE_TO_HOLD
except Exception:
    MIN_SCORE_TO_HOLD = 0.65


CYCLICAL_BASKET = ["SLV", "USO", "CPER", "DBA"]
ALL_REGIMES = ["bull", "chop", "bear", "crisis"]
REGIME_RISK_RANK = {"bull": 0, "chop": 1, "bear": 2, "crisis": 3}


@dataclass(frozen=True)
class V4Policy:
    chop_max_exposure: float = 0.85
    chop_score_threshold_add: float = 0.00
    chop_max_single_position: float = 0.35
    bear_max_exposure: float = 0.30
    gold_max_drawdown_63d: float = -0.08


def _safe_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.sort_index().pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _drawdown(series: pd.Series, window: int) -> pd.Series:
    rolling_high = series.rolling(window, min_periods=max(5, window // 4)).max()
    return (series / rolling_high) - 1.0


def _rolling_percentile_last_value(series: pd.Series, window: int) -> pd.Series:
    def pct_rank(x: np.ndarray) -> float:
        s = pd.Series(x).dropna()
        if s.empty:
            return np.nan
        return float(s.rank(pct=True).iloc[-1])

    return series.rolling(window, min_periods=max(20, window // 4)).apply(pct_rank, raw=True)


def compute_regime_features(
    prices: pd.DataFrame,
    basket: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build commodity-regime features from an equal-weight cyclical commodity basket.

    Important: this function shifts all features by one day so that a regime label
    used on date T only uses information that was already available before T.
    """
    if basket is None:
        basket = CYCLICAL_BASKET

    prices = prices.copy().sort_index()
    prices.index = pd.to_datetime(prices.index)

    available = [ticker for ticker in basket if ticker in prices.columns]
    if len(available) < 2:
        raise ValueError(
            f"Need at least 2 regime-basket tickers. Found {available}."
        )

    basket_prices = prices[available].replace([np.inf, -np.inf], np.nan).ffill()
    basket_returns = _safe_returns(basket_prices).mean(axis=1)
    basket_index = (1.0 + basket_returns).cumprod()

    features = pd.DataFrame(index=prices.index)
    features["basket_ret_20"] = basket_index.pct_change(20)
    features["basket_ret_60"] = basket_index.pct_change(60)
    features["basket_ret_120"] = basket_index.pct_change(120)

    basket_ma_200 = basket_index.rolling(200, min_periods=100).mean()
    features["basket_above_ma_200"] = basket_index > basket_ma_200

    features["basket_dd_63"] = _drawdown(basket_index, 63)
    features["basket_dd_126"] = _drawdown(basket_index, 126)

    features["basket_vol_20"] = basket_returns.rolling(20, min_periods=10).std() * np.sqrt(252)
    features["basket_vol_pct_252"] = _rolling_percentile_last_value(features["basket_vol_20"], 252)

    ma_120 = basket_prices.rolling(120, min_periods=60).mean()
    above_120 = basket_prices > ma_120
    features["breadth_120"] = above_120.mean(axis=1)

    # No lookahead: today's policy uses yesterday's completed regime information.
    features = features.shift(1)

    features["basket_above_ma_200"] = features["basket_above_ma_200"].fillna(False).astype(bool)

    return features


def classify_raw_regime(features: pd.DataFrame) -> pd.Series:
    f = features.copy()

    raw = pd.Series("chop", index=f.index, dtype="object")

    crisis = (
        ((f["basket_ret_20"] <= -0.08) & (f["basket_vol_pct_252"] >= 0.80))
        | (f["basket_dd_63"] <= -0.15)
    )

    bear = (
        ~crisis
        & ((f["basket_ret_120"] <= -0.03) | (~f["basket_above_ma_200"]))
        & (f["breadth_120"] <= 0.40)
    )

    bull = (
        ~crisis
        & ~bear
        & (f["basket_ret_120"] >= 0.05)
        & (f["basket_above_ma_200"])
        & (f["breadth_120"] >= 0.60)
        & (f["basket_vol_pct_252"] <= 0.75)
    )

    raw.loc[bull.fillna(False)] = "bull"
    raw.loc[bear.fillna(False)] = "bear"
    raw.loc[crisis.fillna(False)] = "crisis"

    return raw


def smooth_regime(raw_regime: pd.Series) -> pd.Series:
    raw = raw_regime.astype(str).copy()
    out = pd.Series("chop", index=raw.index, dtype="object")

    is_crisis = raw.eq("crisis")
    is_bear_confirmed = raw.eq("bear").rolling(3, min_periods=3).sum().eq(3)
    is_bull_confirmed = raw.eq("bull").rolling(10, min_periods=10).sum().eq(10)

    out.loc[is_bull_confirmed.fillna(False)] = "bull"
    out.loc[is_bear_confirmed.fillna(False)] = "bear"
    out.loc[is_crisis.fillna(False)] = "crisis"

    return out


def compute_gold_defensive_gate(prices: pd.DataFrame) -> pd.Series:
    prices = prices.sort_index().copy()
    prices.index = pd.to_datetime(prices.index)

    if "GLD" not in prices.columns:
        return pd.Series(False, index=prices.index, name="gold_gate")

    gld = prices["GLD"].replace([np.inf, -np.inf], np.nan).ffill()
    gld_ma_200 = gld.rolling(200, min_periods=100).mean()
    gld_ret_60 = gld.pct_change(60)
    gld_dd_63 = _drawdown(gld, 63)

    gate = (gld > gld_ma_200) & (gld_ret_60 > 0) & (gld_dd_63 > -0.08)

    # No lookahead: today's gate uses yesterday's completed GLD information.
    return gate.shift(1).fillna(False).astype(bool).rename("gold_gate")


def build_score_matrix(scores: pd.DataFrame | None, market_dates: pd.Index, tickers: list[str]) -> pd.DataFrame:
    if scores is None or scores.empty:
        return pd.DataFrame(np.nan, index=market_dates, columns=tickers)

    if not {"date", "ticker", "final_score"}.issubset(scores.columns):
        return pd.DataFrame(np.nan, index=market_dates, columns=tickers)

    out = scores.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["final_score"] = pd.to_numeric(out["final_score"], errors="coerce")

    matrix = (
        out.pivot_table(index="date", columns="ticker", values="final_score", aggfunc="last")
        .sort_index()
        .reindex(market_dates)
        .ffill()
        .reindex(columns=tickers)
    )

    # Same no-lookahead rule as regime features.
    return matrix.shift(1)


def _cap_total_exposure(weights: pd.Series, max_exposure: float) -> pd.Series:
    out = weights.copy().astype(float).fillna(0.0).clip(lower=0.0)
    total = float(out.sum())

    if total > max_exposure and total > 0:
        out *= max_exposure / total

    return out


def apply_policy_row(
    v3_row: pd.Series,
    score_row: pd.Series,
    regime: str,
    gold_gate: bool,
    policy: V4Policy,
    variant: Literal["base", "no_chop", "no_gld_bear_exception", "crisis_only"] = "base",
) -> pd.Series:
    weights = v3_row.copy().astype(float).fillna(0.0).clip(lower=0.0)
    tickers = list(weights.index)

    if variant == "crisis_only":
        if regime == "crisis":
            return pd.Series(0.0, index=tickers)
        return _cap_total_exposure(weights, 1.0)

    if regime == "bull":
        return _cap_total_exposure(weights, 1.0)

    if regime == "chop":
        return _cap_total_exposure(weights, 1.0)

    if regime == "bear":
        out = pd.Series(0.0, index=tickers)

        if variant == "no_gld_bear_exception":
            return out

        if "GLD" in tickers and bool(gold_gate):
            out.loc["GLD"] = min(float(weights.get("GLD", 0.0)), policy.bear_max_exposure)

        return out

    if regime == "crisis":
        return pd.Series(0.0, index=tickers)

    # Defensive fallback.
    return _cap_total_exposure(weights, 1.0)


def build_v4_routed_weights(
    v3_weights: pd.DataFrame,
    prices: pd.DataFrame,
    scores: pd.DataFrame | None = None,
    variant: Literal["base", "no_chop", "no_gld_bear_exception", "crisis_only"] = "base",
    policy: V4Policy | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      routed_weights: daily commodity weights, no CASH column, suitable for V2 engine
      regime_diagnostics: regime labels + regime feature columns
      diagnostic_weights: routed weights plus CASH and gross exposure for audit
    """
    if policy is None:
        policy = V4Policy()

    prices = prices.copy().sort_index()
    prices.index = pd.to_datetime(prices.index)

    v3_weights = v3_weights.copy().sort_index()
    v3_weights.index = pd.to_datetime(v3_weights.index)

    tickers = [ticker for ticker in v3_weights.columns if ticker in prices.columns]
    if not tickers:
        raise ValueError("No overlap between V3 weight columns and price columns.")

    market_dates = prices.index

    daily_v3 = (
        v3_weights.reindex(columns=tickers)
        .reindex(market_dates)
        .ffill()
        .fillna(0.0)
        .clip(lower=0.0)
    )

    features = compute_regime_features(prices=prices)
    raw_regime = classify_raw_regime(features)
    smoothed_regime = smooth_regime(raw_regime)
    gold_gate = compute_gold_defensive_gate(prices=prices).reindex(market_dates).fillna(False)
    score_matrix = build_score_matrix(scores=scores, market_dates=market_dates, tickers=tickers)

    routed_rows = []

    for date in market_dates:
        routed = apply_policy_row(
            v3_row=daily_v3.loc[date],
            score_row=score_matrix.loc[date],
            regime=str(smoothed_regime.loc[date]),
            gold_gate=bool(gold_gate.loc[date]),
            policy=policy,
            variant=variant,
        )
        routed_rows.append(routed)

    routed_weights = pd.DataFrame(routed_rows, index=market_dates, columns=tickers).fillna(0.0)
    routed_weights = routed_weights.clip(lower=0.0)

    # Safety: V2 can handle sums below 1 as implicit cash, but never feed >100% exposure.
    exposure = routed_weights.sum(axis=1)
    scale = pd.Series(1.0, index=routed_weights.index)
    scale.loc[exposure > 1.0] = 1.0 / exposure.loc[exposure > 1.0]
    routed_weights = routed_weights.mul(scale, axis=0)

    regime_diagnostics = features.copy()
    regime_diagnostics.insert(0, "raw_regime", raw_regime)
    regime_diagnostics.insert(1, "smoothed_regime", smoothed_regime)
    regime_diagnostics["gold_gate"] = gold_gate
    regime_diagnostics["variant"] = variant

    diagnostic_weights = routed_weights.copy()
    diagnostic_weights["gross_commodity_exposure"] = routed_weights.sum(axis=1)
    diagnostic_weights["CASH"] = (1.0 - diagnostic_weights["gross_commodity_exposure"]).clip(lower=0.0, upper=1.0)
    diagnostic_weights.insert(0, "regime", smoothed_regime)
    diagnostic_weights.insert(1, "variant", variant)

    return routed_weights, regime_diagnostics, diagnostic_weights
