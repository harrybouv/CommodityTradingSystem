import numpy as np
import pandas as pd

from Commodity_System.config import (
    PRICE_DATA_PATH,
    PROCESSED_DATA_DIR,
    TRADING_DAYS_PER_YEAR,
)


# ============================================================
# HELPERS
# ============================================================

def rolling_percentile_of_last(window: pd.Series) -> float:
    """
    Percentile rank of the latest value inside its rolling window.
    Higher value = current value is high versus its own history.
    """
    window = pd.Series(window).dropna()

    if window.empty:
        return np.nan

    return window.rank(pct=True).iloc[-1]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = numerator / denominator.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


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


# ============================================================
# REALISED VOLATILITY
# ============================================================

prices["realised_vol_20d"] = (
    prices.groupby("ticker")["daily_return"]
    .transform(lambda x: x.rolling(20, min_periods=20).std() * np.sqrt(TRADING_DAYS_PER_YEAR))
)

prices["realised_vol_60d"] = (
    prices.groupby("ticker")["daily_return"]
    .transform(lambda x: x.rolling(60, min_periods=60).std() * np.sqrt(TRADING_DAYS_PER_YEAR))
)

prices["realised_vol_120d"] = (
    prices.groupby("ticker")["daily_return"]
    .transform(lambda x: x.rolling(120, min_periods=60).std() * np.sqrt(TRADING_DAYS_PER_YEAR))
)


# ============================================================
# ORIGINAL PRODUCTION VOLATILITY SCORE
# ============================================================

# Important:
# Keep this simple V1 score unchanged.
# The detailed volatility diagnostics are NOT used directly in final_score.

prices["vol_20d_rank"] = (
    prices.groupby("date")["realised_vol_20d"]
    .rank(pct=True, ascending=False)
)

prices["vol_60d_rank"] = (
    prices.groupby("date")["realised_vol_60d"]
    .rank(pct=True, ascending=False)
)

prices["volatility_score"] = (
    0.40 * prices["vol_20d_rank"]
    + 0.60 * prices["vol_60d_rank"]
)

prices["volatility_score"] = prices["volatility_score"].clip(0, 1)


# ============================================================
# ALLOCATION DIAGNOSTICS
# ============================================================

prices["vol_ratio_20_60"] = safe_divide(
    prices["realised_vol_20d"],
    prices["realised_vol_60d"],
)

prices["vol_ratio_60_120"] = safe_divide(
    prices["realised_vol_60d"],
    prices["realised_vol_120d"],
)

prices["vol_acceleration_20_60"] = (
    prices["realised_vol_20d"] - prices["realised_vol_60d"]
)

prices["vol_acceleration_60_120"] = (
    prices["realised_vol_60d"] - prices["realised_vol_120d"]
)


prices["vol_20d_percentile_252d"] = (
    prices.groupby("ticker")["realised_vol_20d"]
    .transform(
        lambda x: x.rolling(252, min_periods=126)
        .apply(rolling_percentile_of_last, raw=False)
    )
)

prices["vol_60d_percentile_252d"] = (
    prices.groupby("ticker")["realised_vol_60d"]
    .transform(
        lambda x: x.rolling(252, min_periods=126)
        .apply(rolling_percentile_of_last, raw=False)
    )
)


# ============================================================
# VOLATILITY STRESS SCORE
# ============================================================

# High = bad / unstable.
# This is NOT an alpha score. It is for allocation sizing only.

prices["short_vol_spike_stress"] = (
    (prices["vol_ratio_20_60"] - 1.0) / 0.75
).clip(lower=0.0, upper=1.0)

prices["medium_vol_spike_stress"] = (
    (prices["vol_ratio_60_120"] - 1.0) / 0.50
).clip(lower=0.0, upper=1.0)

prices["own_history_vol_stress"] = (
    0.35 * prices["vol_20d_percentile_252d"]
    + 0.65 * prices["vol_60d_percentile_252d"]
)

# High cross-sectional realised vol is also a mild stress marker.
prices["cross_sectional_vol_stress"] = (
    1.0 - prices["volatility_score"]
)

stress_components = [
    "short_vol_spike_stress",
    "medium_vol_spike_stress",
    "own_history_vol_stress",
    "cross_sectional_vol_stress",
]

for col in stress_components:
    prices[col] = prices[col].fillna(0.0).clip(0, 1)

prices["vol_stress_score"] = (
    0.35 * prices["short_vol_spike_stress"]
    + 0.20 * prices["medium_vol_spike_stress"]
    + 0.30 * prices["own_history_vol_stress"]
    + 0.15 * prices["cross_sectional_vol_stress"]
)

prices["vol_stress_score"] = prices["vol_stress_score"].fillna(0.0).clip(0, 1)

# High = good / safer from a volatility allocation perspective.
prices["vol_allocation_score"] = (
    1.0 - prices["vol_stress_score"]
).clip(0, 1)


# ============================================================
# OUTPUT
# ============================================================

output_columns = [
    "date",
    "ticker",
    "adj_close",
    "daily_return",

    "realised_vol_20d",
    "realised_vol_60d",
    "realised_vol_120d",

    "vol_20d_rank",
    "vol_60d_rank",
    "volatility_score",

    "vol_ratio_20_60",
    "vol_ratio_60_120",
    "vol_acceleration_20_60",
    "vol_acceleration_60_120",

    "vol_20d_percentile_252d",
    "vol_60d_percentile_252d",

    "short_vol_spike_stress",
    "medium_vol_spike_stress",
    "own_history_vol_stress",
    "cross_sectional_vol_stress",

    "vol_stress_score",
    "vol_allocation_score",
]

output = prices[output_columns].copy()

# Only drop rows required for the original production score.
# Do NOT drop rows just because long-window diagnostics are missing.
output = output.dropna(
    subset=[
        "daily_return",
        "realised_vol_20d",
        "realised_vol_60d",
        "volatility_score",
    ]
)

output_path = PROCESSED_DATA_DIR / "volatility_scores.csv"
output.to_csv(output_path, index=False)

print(f"Saved volatility scores to {output_path}")
print(output.tail(12))