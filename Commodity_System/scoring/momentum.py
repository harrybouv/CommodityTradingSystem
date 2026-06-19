import pandas as pd
from Commodity_System.config import PRICE_DATA_PATH, PROCESSED_DATA_DIR


prices = pd.read_csv(PRICE_DATA_PATH)
prices["date"] = pd.to_datetime(prices["date"])
prices = prices.sort_values(["ticker", "date"])

for window in [20, 60, 120]:
    prices[f"momentum_{window}d"] = (
        prices.groupby("ticker")["adj_close"]
        .pct_change(window)
    )

prices["momentum_20d_rank"] = prices.groupby("date")["momentum_20d"].rank(pct=True)
prices["momentum_60d_rank"] = prices.groupby("date")["momentum_60d"].rank(pct=True)
prices["momentum_120d_rank"] = prices.groupby("date")["momentum_120d"].rank(pct=True)

prices["relative_momentum_score"] = (
    0.25 * prices["momentum_20d_rank"]
    + 0.35 * prices["momentum_60d_rank"]
    + 0.40 * prices["momentum_120d_rank"]
)

prices["absolute_momentum_score"] = (
    (prices["momentum_20d"] > 0).astype(int)
    + (prices["momentum_60d"] > 0).astype(int)
    + (prices["momentum_120d"] > 0).astype(int)
) / 3

prices["base_momentum_score"] = (
    0.70 * prices["relative_momentum_score"]
    + 0.30 * prices["absolute_momentum_score"]
)

prices["momentum_acceleration_20_60"] = (
    prices["momentum_20d"] - prices["momentum_60d"]
)

prices["momentum_acceleration_60_120"] = (
    prices["momentum_60d"] - prices["momentum_120d"]
)

prices["momentum_acceleration_20_60_rank"] = (
    prices.groupby("date")["momentum_acceleration_20_60"]
    .rank(pct=True)
)

prices["momentum_acceleration_60_120_rank"] = (
    prices.groupby("date")["momentum_acceleration_60_120"]
    .rank(pct=True)
)

prices["momentum_acceleration_score"] = (
    0.60 * prices["momentum_acceleration_20_60_rank"]
    + 0.40 * prices["momentum_acceleration_60_120_rank"]
)

prices["momentum_score"] = (
    0.75 * prices["base_momentum_score"]
    + 0.25 * prices["momentum_acceleration_score"]
)

prices["momentum_score"] = prices["momentum_score"].clip(0, 1)


output = prices[
    [
        "date",
        "ticker",
        "adj_close",
        "momentum_20d",
        "momentum_60d",
        "momentum_120d",
        "relative_momentum_score",
        "absolute_momentum_score",
        "momentum_score",
    ]
].dropna()


output_path = PROCESSED_DATA_DIR / "momentum_scores.csv"
output.to_csv(output_path, index=False)


print(f"Saved momentum scores to {output_path}")
print(output.tail(12))