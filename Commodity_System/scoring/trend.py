import pandas as pd
from Commodity_System.config import PRICE_DATA_PATH, PROCESSED_DATA_DIR


prices = pd.read_csv(PRICE_DATA_PATH)
prices["date"] = pd.to_datetime(prices["date"])
prices = prices.sort_values(["ticker", "date"])


prices["ma_20d"] = (
    prices.groupby("ticker")["adj_close"]
    .transform(lambda x: x.rolling(20).mean())
)

prices["ma_50d"] = (
    prices.groupby("ticker")["adj_close"]
    .transform(lambda x: x.rolling(50).mean())
)

prices["ma_200d"] = (
    prices.groupby("ticker")["adj_close"]
    .transform(lambda x: x.rolling(200).mean())
)

prices["price_above_50d"] = (
    prices["adj_close"] > prices["ma_50d"]
).astype(int)

prices["price_above_200d"] = (
    prices["adj_close"] > prices["ma_200d"]
).astype(int)

prices["ma_20_above_50"] = (
    prices["ma_20d"] > prices["ma_50d"]
).astype(int)

prices["ma_50_above_200"] = (
    prices["ma_50d"] > prices["ma_200d"]
).astype(int)

prices["distance_from_50d"] = (
    prices["adj_close"] / prices["ma_50d"] - 1
)

prices["distance_from_200d"] = (
    prices["adj_close"] / prices["ma_200d"] - 1
)

prices["ma_50d_slope_20d"] = (
    prices.groupby("ticker")["ma_50d"]
    .pct_change(20)
)

prices["ma_200d_slope_20d"] = (
    prices.groupby("ticker")["ma_200d"]
    .pct_change(20)
)

prices["distance_rank"] = (
    prices.groupby("date")["distance_from_200d"]
    .rank(pct=True)
)

prices["ma_50d_slope_rank"] = (
    prices.groupby("date")["ma_50d_slope_20d"]
    .rank(pct=True)
)

prices["ma_200d_slope_rank"] = (
    prices.groupby("date")["ma_200d_slope_20d"]
    .rank(pct=True)
)

prices["trend_structure_score"] = (
    0.25 * prices["price_above_50d"]
    + 0.30 * prices["price_above_200d"]
    + 0.20 * prices["ma_20_above_50"]
    + 0.25 * prices["ma_50_above_200"]
)

prices["trend_slope_score"] = (
    0.65 * prices["ma_50d_slope_rank"]
    + 0.35 * prices["ma_200d_slope_rank"]
)

prices["trend_score"] = (
    0.50 * prices["trend_structure_score"]
    + 0.30 * prices["trend_slope_score"]
    + 0.20 * prices["distance_rank"]
)

prices["trend_score"] = prices["trend_score"].clip(0, 1)


output = prices[
    [
        "date",
        "ticker",
        "adj_close",
        "ma_50d",
        "ma_200d",
        "price_above_200d",
        "ma_50_above_200",
        "distance_from_200d",
        "trend_score",
    ]
].dropna()


output_path = PROCESSED_DATA_DIR / "trend_scores.csv"
output.to_csv(output_path, index=False)

print(f"Saved trend scores to {output_path}")
print(output.tail(12))