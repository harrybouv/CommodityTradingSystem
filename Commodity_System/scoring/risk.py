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

prices["negative_return"] = prices["daily_return"].where(
    prices["daily_return"] < 0,
    0.0,
)


# ============================================================
# DRAWDOWN
# ============================================================

prices["rolling_high_60d"] = (
    prices.groupby("ticker")["adj_close"]
    .transform(lambda x: x.rolling(60, min_periods=60).max())
)

prices["rolling_high_120d"] = (
    prices.groupby("ticker")["adj_close"]
    .transform(lambda x: x.rolling(120, min_periods=60).max())
)

prices["drawdown_60d"] = (
    prices["adj_close"] / prices["rolling_high_60d"] - 1.0
)

prices["drawdown_120d"] = (
    prices["adj_close"] / prices["rolling_high_120d"] - 1.0
)


# ============================================================
# ORIGINAL PRODUCTION RISK SCORE
# ============================================================

# Important:
# Keep this simple V1 risk score unchanged.
# The detailed risk diagnostics are NOT used directly in final_score.

prices["downside_vol_60d"] = (
    prices.groupby("ticker")["negative_return"]
    .transform(lambda x: x.rolling(60, min_periods=60).std() * np.sqrt(TRADING_DAYS_PER_YEAR))
)

prices["drawdown_score"] = (
    prices.groupby("date")["drawdown_60d"]
    .rank(pct=True)
)

prices["downside_vol_score"] = (
    prices.groupby("date")["downside_vol_60d"]
    .rank(pct=True, ascending=False)
)

prices["risk_score"] = (
    0.60 * prices["drawdown_score"]
    + 0.40 * prices["downside_vol_score"]
)

prices["risk_score"] = prices["risk_score"].clip(0, 1)


# ============================================================
# ADDITIONAL RISK DIAGNOSTICS
# ============================================================

prices["downside_vol_20d"] = (
    prices.groupby("ticker")["negative_return"]
    .transform(lambda x: x.rolling(20, min_periods=20).std() * np.sqrt(TRADING_DAYS_PER_YEAR))
)

prices["downside_vol_120d"] = (
    prices.groupby("ticker")["negative_return"]
    .transform(lambda x: x.rolling(120, min_periods=60).std() * np.sqrt(TRADING_DAYS_PER_YEAR))
)

prices["downside_vol_ratio_20_60"] = safe_divide(
    prices["downside_vol_20d"],
    prices["downside_vol_60d"],
)


# Drawdown persistence.
prices["in_drawdown_2pct"] = (prices["drawdown_60d"] < -0.02).astype(float)
prices["in_drawdown_5pct"] = (prices["drawdown_60d"] < -0.05).astype(float)

prices["drawdown_persistence_60d"] = (
    prices.groupby("ticker")["in_drawdown_2pct"]
    .transform(lambda x: x.rolling(60, min_periods=30).mean())
)

prices["deep_drawdown_persistence_60d"] = (
    prices.groupby("ticker")["in_drawdown_5pct"]
    .transform(lambda x: x.rolling(60, min_periods=30).mean())
)


# Tail risk.
prices["tail_return_5pct_60d"] = (
    prices.groupby("ticker")["daily_return"]
    .transform(lambda x: x.rolling(60, min_periods=60).quantile(0.05))
)

prices["tail_return_5pct_120d"] = (
    prices.groupby("ticker")["daily_return"]
    .transform(lambda x: x.rolling(120, min_periods=60).quantile(0.05))
)


# Downside pressure.
prices["downside_pressure_20d"] = (
    prices.groupby("ticker")["negative_return"]
    .transform(lambda x: x.rolling(20, min_periods=20).sum())
)

prices["downside_pressure_60d"] = (
    prices.groupby("ticker")["negative_return"]
    .transform(lambda x: x.rolling(60, min_periods=60).sum())
)


# ============================================================
# RISK STRESS SCORE
# ============================================================

# High = bad / unstable.
# This is NOT an alpha score. It is for allocation sizing only.

prices["drawdown_120d_score"] = (
    prices.groupby("date")["drawdown_120d"]
    .rank(pct=True)
)

prices["drawdown_stress_score"] = (
    1.0
    - (
        0.70 * prices["drawdown_score"]
        + 0.30 * prices["drawdown_120d_score"]
    )
)

prices["downside_vol_20d_score"] = (
    prices.groupby("date")["downside_vol_20d"]
    .rank(pct=True, ascending=False)
)

prices["downside_vol_120d_score"] = (
    prices.groupby("date")["downside_vol_120d"]
    .rank(pct=True, ascending=False)
)

prices["downside_vol_combined_score"] = (
    0.20 * prices["downside_vol_20d_score"]
    + 0.60 * prices["downside_vol_score"]
    + 0.20 * prices["downside_vol_120d_score"]
)

prices["downside_vol_stress_score"] = (
    1.0 - prices["downside_vol_combined_score"]
)


# Higher persistence = worse.
prices["persistence_stress_score"] = (
    0.65
    * prices.groupby("date")["drawdown_persistence_60d"]
    .rank(pct=True)
    + 0.35
    * prices.groupby("date")["deep_drawdown_persistence_60d"]
    .rank(pct=True)
)


# More negative tail return = worse.
prices["tail_risk_60d_stress"] = (
    prices.groupby("date")["tail_return_5pct_60d"]
    .rank(pct=True, ascending=False)
)

prices["tail_risk_120d_stress"] = (
    prices.groupby("date")["tail_return_5pct_120d"]
    .rank(pct=True, ascending=False)
)

prices["tail_risk_stress_score"] = (
    0.70 * prices["tail_risk_60d_stress"]
    + 0.30 * prices["tail_risk_120d_stress"]
)


# More negative recent downside pressure = worse.
prices["downside_pressure_20d_stress"] = (
    prices.groupby("date")["downside_pressure_20d"]
    .rank(pct=True, ascending=False)
)

prices["downside_pressure_60d_stress"] = (
    prices.groupby("date")["downside_pressure_60d"]
    .rank(pct=True, ascending=False)
)

prices["downside_pressure_stress_score"] = (
    0.65 * prices["downside_pressure_20d_stress"]
    + 0.35 * prices["downside_pressure_60d_stress"]
)


stress_components = [
    "drawdown_stress_score",
    "downside_vol_stress_score",
    "persistence_stress_score",
    "tail_risk_stress_score",
    "downside_pressure_stress_score",
]

for col in stress_components:
    prices[col] = prices[col].fillna(0.0).clip(0, 1)

prices["risk_stress_score"] = (
    0.35 * prices["drawdown_stress_score"]
    + 0.25 * prices["downside_vol_stress_score"]
    + 0.20 * prices["persistence_stress_score"]
    + 0.15 * prices["tail_risk_stress_score"]
    + 0.05 * prices["downside_pressure_stress_score"]
)

prices["risk_stress_score"] = prices["risk_stress_score"].fillna(0.0).clip(0, 1)

# High = good / safer from a risk allocation perspective.
prices["risk_allocation_score"] = (
    1.0 - prices["risk_stress_score"]
).clip(0, 1)


# ============================================================
# OUTPUT
# ============================================================

output_columns = [
    "date",
    "ticker",
    "adj_close",

    "daily_return",
    "negative_return",

    "drawdown_60d",
    "drawdown_120d",
    "downside_vol_20d",
    "downside_vol_60d",
    "downside_vol_120d",
    "downside_vol_ratio_20_60",

    "drawdown_persistence_60d",
    "deep_drawdown_persistence_60d",

    "tail_return_5pct_60d",
    "tail_return_5pct_120d",

    "downside_pressure_20d",
    "downside_pressure_60d",

    "drawdown_score",
    "downside_vol_score",
    "risk_score",

    "drawdown_stress_score",
    "downside_vol_stress_score",
    "persistence_stress_score",
    "tail_risk_stress_score",
    "downside_pressure_stress_score",

    "risk_stress_score",
    "risk_allocation_score",
]

output = prices[output_columns].copy()

# Only drop rows required for the original production score.
# Do NOT drop rows just because long-window diagnostics are missing.
output = output.dropna(
    subset=[
        "drawdown_60d",
        "downside_vol_60d",
        "drawdown_score",
        "downside_vol_score",
        "risk_score",
    ]
)

output_path = PROCESSED_DATA_DIR / "risk_scores.csv"
output.to_csv(output_path, index=False)

print(f"Saved risk scores to {output_path}")
print(output.tail(12))