from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

SCORES_PATH = ROOT / "data" / "processed" / "final_scores.csv"
ASSET_RETURNS_PATH = ROOT / "results" / "backtest_V3" / "asset_returns_V3.csv"

OUT_DIR = ROOT / "results" / "regime_score_bucket_test"
CHART_DIR = OUT_DIR / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
TICKERS = ["GLD", "SLV", "USO", "UNG", "CPER", "DBA"]
CYCLICAL_TICKERS = ["SLV", "USO", "UNG", "CPER", "DBA"]

BULL_THRESHOLD = 0.02
BEAR_THRESHOLD = -0.02


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def load_scores() -> pd.DataFrame:
    scores = read_required(SCORES_PATH)
    scores["date"] = pd.to_datetime(scores["date"])
    scores["ticker"] = scores["ticker"].astype(str).str.upper()

    if "final_score" not in scores.columns:
        raise ValueError("final_scores.csv must contain final_score.")

    return scores[["date", "ticker", "final_score"]].dropna()


def load_asset_returns() -> pd.DataFrame:
    returns = read_required(ASSET_RETURNS_PATH)
    returns["date"] = pd.to_datetime(returns["date"])
    returns = returns.set_index("date").sort_index()

    cols = [c for c in TICKERS if c in returns.columns]
    returns = returns[cols].apply(pd.to_numeric, errors="coerce")

    return returns.replace([np.inf, -np.inf], np.nan)


def build_realised_monthly_regime(asset_returns: pd.DataFrame) -> pd.DataFrame:
    available = [t for t in CYCLICAL_TICKERS if t in asset_returns.columns]
    if not available:
        raise ValueError("No cyclical tickers available.")

    cyc_daily = asset_returns[available].mean(axis=1, skipna=True)
    monthly_cyc_return = (1 + cyc_daily).resample("ME").prod() - 1

    monthly_regime = pd.Series(index=monthly_cyc_return.index, dtype="object")
    monthly_regime[monthly_cyc_return > BULL_THRESHOLD] = "Bull"
    monthly_regime[monthly_cyc_return < BEAR_THRESHOLD] = "Bear"
    monthly_regime[
        (monthly_cyc_return >= BEAR_THRESHOLD)
        & (monthly_cyc_return <= BULL_THRESHOLD)
    ] = "Chop"

    regime_by_month = monthly_regime.copy()
    regime_by_month.index = regime_by_month.index.to_period("M")

    regimes = pd.DataFrame(index=asset_returns.index)
    regimes["month"] = regimes.index.to_period("M")
    regimes["regime"] = regimes["month"].map(regime_by_month)
    regimes = regimes.drop(columns=["month"])

    return regimes.dropna()


def build_forward_return_panel(asset_returns: pd.DataFrame) -> pd.DataFrame:
    fwd = asset_returns.shift(-1)
    out = fwd.stack().reset_index()
    out.columns = ["date", "ticker", "next_return"]
    out["ticker"] = out["ticker"].astype(str).str.upper()
    return out.dropna()


def run_bucket_test(scores: pd.DataFrame, asset_returns: pd.DataFrame, regimes: pd.DataFrame):
    fwd = build_forward_return_panel(asset_returns)

    merged = scores.merge(fwd, on=["date", "ticker"], how="inner")
    merged = merged.merge(
        regimes.reset_index().rename(columns={"index": "date"}),
        on="date",
        how="inner",
    )

    merged = merged.dropna(subset=["final_score", "next_return", "regime"])

    # Global buckets so bucket 1/10 mean the same thing in every regime.
    merged["score_bucket"] = pd.qcut(
        merged["final_score"],
        q=10,
        labels=list(range(1, 11)),
        duplicates="drop",
    ).astype(int)

    bucket_direction = (
        merged.groupby("score_bucket")
        .agg(
            avg_score=("final_score", "mean"),
            min_score=("final_score", "min"),
            max_score=("final_score", "max"),
            avg_next_return=("next_return", "mean"),
            median_next_return=("next_return", "median"),
            count=("next_return", "count"),
            hit_rate=("next_return", lambda x: (x > 0).mean()),
        )
        .reset_index()
    )

    regime_bucket = (
        merged.groupby(["regime", "score_bucket"])
        .agg(
            avg_score=("final_score", "mean"),
            avg_next_return=("next_return", "mean"),
            annualised_avg_next_return=("next_return", lambda x: x.mean() * TRADING_DAYS),
            median_next_return=("next_return", "median"),
            count=("next_return", "count"),
            hit_rate=("next_return", lambda x: (x > 0).mean()),
        )
        .reset_index()
    )

    ic_rows = []
    for regime, g in merged.groupby("regime"):
        ic_rows.append({
            "regime": regime,
            "observations": len(g),
            "pearson_ic": g["final_score"].corr(g["next_return"]),
            "rank_ic": g["final_score"].rank().corr(g["next_return"].rank()),
            "avg_next_return": g["next_return"].mean(),
        })

    ic_summary = pd.DataFrame(ic_rows)

    return merged, bucket_direction, regime_bucket, ic_summary


def plot_regime_bucket_returns(regime_bucket: pd.DataFrame) -> Path:
    pivot = (
        regime_bucket
        .pivot(index="score_bucket", columns="regime", values="avg_next_return")
        .reindex(index=list(range(1, 11)))
        .reindex(columns=["Bull", "Chop", "Bear"])
    )

    ax = pivot.plot(kind="bar", figsize=(12, 6), width=0.8)
    ax.set_title("Average Next-Day Return by Final Score Bucket and Regime")
    ax.set_xlabel("Final score bucket: 1 = lowest, 10 = highest")
    ax.set_ylabel("Average next-day return")
    ax.axhline(0, color="black", linewidth=1)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.2%}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Regime")

    path = CHART_DIR / "regime_score_bucket_returns.png"
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()
    return path


def plot_rank_ic(ic_summary: pd.DataFrame) -> Path:
    order = ["Bull", "Chop", "Bear"]
    df = ic_summary.set_index("regime").reindex(order).dropna(how="all")

    ax = df["rank_ic"].plot(kind="bar", figsize=(9, 5))
    ax.set_title("Final Score Rank IC by Regime")
    ax.set_xlabel("Regime")
    ax.set_ylabel("Rank IC")
    ax.axhline(0, color="black", linewidth=1)
    ax.grid(axis="y", alpha=0.3)

    path = CHART_DIR / "regime_rank_ic.png"
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()
    return path


def main():
    scores = load_scores()
    asset_returns = load_asset_returns()
    regimes = build_realised_monthly_regime(asset_returns)

    merged, bucket_direction, regime_bucket, ic_summary = run_bucket_test(
        scores=scores,
        asset_returns=asset_returns,
        regimes=regimes,
    )

    bucket_direction.to_csv(OUT_DIR / "bucket_direction_check.csv", index=False)
    regime_bucket.to_csv(OUT_DIR / "regime_score_bucket_returns.csv", index=False)
    ic_summary.to_csv(OUT_DIR / "regime_score_ic_summary.csv", index=False)
    merged.to_csv(OUT_DIR / "score_forward_return_panel.csv", index=False)

    chart1 = plot_regime_bucket_returns(regime_bucket)
    chart2 = plot_rank_ic(ic_summary)

    print("\nBucket direction check:")
    print(bucket_direction.to_string(index=False))

    print("\nRegime IC summary:")
    print(ic_summary.to_string(index=False))

    print("\nSaved:")
    print(OUT_DIR / "bucket_direction_check.csv")
    print(OUT_DIR / "regime_score_bucket_returns.csv")
    print(OUT_DIR / "regime_score_ic_summary.csv")
    print(chart1)
    print(chart2)


if __name__ == "__main__":
    main()