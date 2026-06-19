import pandas as pd
from Commodity_System.config import PRICE_DATA_PATH, PROCESSED_DATA_DIR


prices = pd.read_csv(PRICE_DATA_PATH)
prices["date"] = pd.to_datetime(prices["date"])
prices = prices.sort_values(["ticker", "date"])

for window in [20, 60, 120]:
    prices[f"return_{window}d"] = (
        prices.groupby("ticker")["adj_close"]
        .pct_change(window)
    )

    prices[f"universe_return_{window}d"] = (
        prices.groupby("date")[f"return_{window}d"]
        .transform("mean")
    )

    prices[f"relative_strength_{window}d"] = (
        prices[f"return_{window}d"]
        - prices[f"universe_return_{window}d"]
    )

    prices[f"relative_strength_{window}d_rank"] = (
        prices.groupby("date")[f"relative_strength_{window}d"]
        .rank(pct=True)
    )

prices["relative_strength_core_score"] = (
    0.25 * prices["relative_strength_20d_rank"]
    + 0.35 * prices["relative_strength_60d_rank"]
    + 0.40 * prices["relative_strength_120d_rank"]
)

prices["relative_strength_consistency_score"] = (
    (prices["relative_strength_20d"] > 0).astype(int)
    + (prices["relative_strength_60d"] > 0).astype(int)
    + (prices["relative_strength_120d"] > 0).astype(int)
) / 3

prices["relative_strength_acceleration_20_60"] = (
    prices["relative_strength_20d"]
    - prices["relative_strength_60d"]
)

prices["relative_strength_acceleration_60_120"] = (
    prices["relative_strength_60d"]
    - prices["relative_strength_120d"]
)

prices["relative_strength_acceleration_20_60_rank"] = (
    prices.groupby("date")["relative_strength_acceleration_20_60"]
    .rank(pct=True)
)

prices["relative_strength_acceleration_60_120_rank"] = (
    prices.groupby("date")["relative_strength_acceleration_60_120"]
    .rank(pct=True)
)

prices["relative_strength_acceleration_score"] = (
    0.60 * prices["relative_strength_acceleration_20_60_rank"]
    + 0.40 * prices["relative_strength_acceleration_60_120_rank"]
)

prices["relative_strength_score"] = (
    0.70 * prices["relative_strength_core_score"]
    + 0.20 * prices["relative_strength_consistency_score"]
    + 0.10 * prices["relative_strength_acceleration_score"]
)

prices["relative_strength_score"] = prices["relative_strength_score"].clip(0, 1)


output = prices[
    [
        "date",
        "ticker",
        "adj_close",
        "return_20d",
        "return_60d",
        "return_120d",
        "universe_return_20d",
        "universe_return_60d",
        "universe_return_120d",
        "relative_strength_20d",
        "relative_strength_60d",
        "relative_strength_120d",
        "relative_strength_core_score",
        "relative_strength_consistency_score",
        "relative_strength_acceleration_score",
        "relative_strength_score",
    ]
].dropna()


output_path = PROCESSED_DATA_DIR / "relative_strength_scores.csv"
output.to_csv(output_path, index=False)


print(f"Saved relative strength scores to {output_path}")
print(output.tail(12))