import numpy as np
import pandas as pd

from Commodity_System.config import (
    PRICE_DATA_PATH,
    PROCESSED_DATA_DIR,
)


# ============================================================
# HELPERS
# ============================================================

def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = numerator / denominator.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def rolling_path_efficiency(price: pd.Series, window: int) -> pd.Series:
    """
    Trend efficiency:
    How cleanly price travelled from point A to B.

    High positive value = smoother persistent uptrend.
    Low/zero value = choppy, flat, or negative movement.
    """
    net_change = price - price.shift(window)

    path_length = (
        price.diff()
        .abs()
        .rolling(window, min_periods=window)
        .sum()
    )

    efficiency = safe_divide(net_change, path_length)

    # We only want persistent upward movement.
    efficiency = efficiency.clip(lower=0.0, upper=1.0)

    return efficiency


# ============================================================
# LOAD DATA
# ============================================================

prices = pd.read_csv(PRICE_DATA_PATH)
prices["date"] = pd.to_datetime(prices["date"])
prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)


# ============================================================
# RETURNS
# ============================================================

prices["daily_return"] = (
    prices.groupby("ticker")["adj_close"]
    .pct_change()
)

prices["return_20d"] = (
    prices.groupby("ticker")["adj_close"]
    .pct_change(20)
)

prices["return_60d"] = (
    prices.groupby("ticker")["adj_close"]
    .pct_change(60)
)

prices["return_120d"] = (
    prices.groupby("ticker")["adj_close"]
    .pct_change(120)
)


# ============================================================
# MOVING AVERAGES / BASIC TREND STATE
# ============================================================

prices["ma_20d"] = (
    prices.groupby("ticker")["adj_close"]
    .transform(lambda x: x.rolling(20, min_periods=20).mean())
)

prices["ma_50d"] = (
    prices.groupby("ticker")["adj_close"]
    .transform(lambda x: x.rolling(50, min_periods=50).mean())
)

prices["ma_120d"] = (
    prices.groupby("ticker")["adj_close"]
    .transform(lambda x: x.rolling(120, min_periods=80).mean())
)

prices["ma_200d"] = (
    prices.groupby("ticker")["adj_close"]
    .transform(lambda x: x.rolling(200, min_periods=120).mean())
)

prices["above_50d"] = (
    prices["adj_close"] > prices["ma_50d"]
).astype(float)

prices["above_120d"] = (
    prices["adj_close"] > prices["ma_120d"]
).astype(float)

prices["ma_50_above_120"] = (
    prices["ma_50d"] > prices["ma_120d"]
).astype(float)

prices["ma_120_above_200"] = (
    prices["ma_120d"] > prices["ma_200d"]
).astype(float)

prices["structural_uptrend_flag"] = (
    (
        0.30 * prices["above_50d"]
        + 0.30 * prices["above_120d"]
        + 0.25 * prices["ma_50_above_120"]
        + 0.15 * prices["ma_120_above_200"]
    )
).clip(0, 1)


# ============================================================
# BREAKOUT / RANGE EXPANSION
# ============================================================

prices["rolling_high_20d"] = (
    prices.groupby("ticker")["adj_close"]
    .transform(lambda x: x.rolling(20, min_periods=20).max())
)

prices["rolling_high_60d"] = (
    prices.groupby("ticker")["adj_close"]
    .transform(lambda x: x.rolling(60, min_periods=60).max())
)

prices["rolling_high_120d"] = (
    prices.groupby("ticker")["adj_close"]
    .transform(lambda x: x.rolling(120, min_periods=80).max())
)

prices["rolling_high_252d"] = (
    prices.groupby("ticker")["adj_close"]
    .transform(lambda x: x.rolling(252, min_periods=126).max())
)

prices["distance_from_60d_high"] = (
    prices["adj_close"] / prices["rolling_high_60d"] - 1.0
)

prices["distance_from_120d_high"] = (
    prices["adj_close"] / prices["rolling_high_120d"] - 1.0
)

prices["distance_from_252d_high"] = (
    prices["adj_close"] / prices["rolling_high_252d"] - 1.0
)

prices["new_60d_high"] = (
    prices["adj_close"] >= prices["rolling_high_60d"]
).astype(float)

prices["new_120d_high"] = (
    prices["adj_close"] >= prices["rolling_high_120d"]
).astype(float)


# Closer to highs is better.
prices["distance_60d_high_score"] = (
    prices.groupby("date")["distance_from_60d_high"]
    .rank(pct=True)
)

prices["distance_120d_high_score"] = (
    prices.groupby("date")["distance_from_120d_high"]
    .rank(pct=True)
)

prices["distance_252d_high_score"] = (
    prices.groupby("date")["distance_from_252d_high"]
    .rank(pct=True)
)

prices["breakout_score"] = (
    0.25 * prices["distance_60d_high_score"]
    + 0.30 * prices["distance_120d_high_score"]
    + 0.20 * prices["distance_252d_high_score"]
    + 0.15 * prices["new_60d_high"]
    + 0.10 * prices["new_120d_high"]
)

prices["breakout_score"] = prices["breakout_score"].clip(0, 1)


# ============================================================
# TREND PERSISTENCE / CONSISTENCY
# ============================================================

prices["positive_day_share_60d"] = (
    prices.groupby("ticker")["daily_return"]
    .transform(lambda x: (x > 0).rolling(60, min_periods=40).mean())
)

# Uses weekly-ish 5-day returns without resampling.
prices["return_5d"] = (
    prices.groupby("ticker")["adj_close"]
    .pct_change(5)
)

prices["positive_5d_share_60d"] = (
    prices.groupby("ticker")["return_5d"]
    .transform(lambda x: (x > 0).rolling(60, min_periods=40).mean())
)

prices["trend_efficiency_60d"] = (
    prices.groupby("ticker")["adj_close"]
    .transform(lambda x: rolling_path_efficiency(x, 60))
)

prices["trend_efficiency_120d"] = (
    prices.groupby("ticker")["adj_close"]
    .transform(lambda x: rolling_path_efficiency(x, 120))
)


prices["positive_day_share_score"] = (
    prices.groupby("date")["positive_day_share_60d"]
    .rank(pct=True)
)

prices["positive_5d_share_score"] = (
    prices.groupby("date")["positive_5d_share_60d"]
    .rank(pct=True)
)

prices["trend_efficiency_60d_score"] = (
    prices.groupby("date")["trend_efficiency_60d"]
    .rank(pct=True)
)

prices["trend_efficiency_120d_score"] = (
    prices.groupby("date")["trend_efficiency_120d"]
    .rank(pct=True)
)

prices["trend_consistency_score"] = (
    0.25 * prices["positive_day_share_score"]
    + 0.25 * prices["positive_5d_share_score"]
    + 0.30 * prices["trend_efficiency_60d_score"]
    + 0.20 * prices["trend_efficiency_120d_score"]
)

prices["trend_consistency_score"] = prices["trend_consistency_score"].clip(0, 1)


# ============================================================
# PULLBACK-IN-UPTREND
# ============================================================

prices["pullback_from_20d_high"] = (
    prices["adj_close"] / prices["rolling_high_20d"] - 1.0
)

prices["pullback_from_60d_high"] = (
    prices["adj_close"] / prices["rolling_high_60d"] - 1.0
)

# We want mild pullbacks in strong trends, not crashes.
# Ideal: roughly 2% to 8% below recent high.
pullback_depth = -prices["pullback_from_20d_high"]

prices["mild_pullback_quality"] = np.where(
    pullback_depth < 0.02,
    0.20,
    np.where(
        pullback_depth <= 0.08,
        1.00,
        np.where(
            pullback_depth <= 0.14,
            0.50,
            0.00,
        ),
    ),
)

prices["recent_20d_weakness"] = (
    prices["return_20d"] < 0
).astype(float)

prices["medium_term_strength"] = (
    prices["return_60d"] > 0
).astype(float)

prices["pullback_uptrend_score"] = (
    prices["structural_uptrend_flag"]
    * prices["medium_term_strength"]
    * (
        0.65 * prices["mild_pullback_quality"]
        + 0.35 * prices["recent_20d_weakness"]
    )
)

prices["pullback_uptrend_score"] = prices["pullback_uptrend_score"].clip(0, 1)


# ============================================================
# FINAL TREND PERSISTENCE SCORE
# ============================================================

# This is the one score to plug into final_score.
#
# Interpretation:
# - breakout_score: is price pressing into / through its range?
# - trend_consistency_score: has strength been persistent rather than one-day noise?
# - pullback_uptrend_score: is there a mild pullback inside a valid uptrend?
#
# Pullback score has the smallest weight because monthly allocation should not
# become too tactical.

prices["trend_persistence_score"] = (
    0.45 * prices["trend_consistency_score"]
    + 0.40 * prices["breakout_score"]
    + 0.15 * prices["pullback_uptrend_score"]
)

prices["trend_persistence_score"] = (
    prices["trend_persistence_score"]
    .replace([np.inf, -np.inf], np.nan)
    .clip(0, 1)
)


# ============================================================
# OUTPUT
# ============================================================

output_columns = [
    "date",
    "ticker",
    "adj_close",

    "return_20d",
    "return_60d",
    "return_120d",

    "structural_uptrend_flag",

    "distance_from_60d_high",
    "distance_from_120d_high",
    "distance_from_252d_high",
    "new_60d_high",
    "new_120d_high",
    "breakout_score",

    "positive_day_share_60d",
    "positive_5d_share_60d",
    "trend_efficiency_60d",
    "trend_efficiency_120d",
    "trend_consistency_score",

    "pullback_from_20d_high",
    "pullback_from_60d_high",
    "mild_pullback_quality",
    "pullback_uptrend_score",

    "trend_persistence_score",
]

output = prices[output_columns].copy()

output = output.dropna(
    subset=[
        "breakout_score",
        "trend_consistency_score",
        "trend_persistence_score",
    ]
)

output_path = PROCESSED_DATA_DIR / "trend_persistence_scores.csv"
output.to_csv(output_path, index=False)

print(f"Saved trend persistence scores to {output_path}")
print(output.tail(12))