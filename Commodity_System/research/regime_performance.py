from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

CURVES_PATH = ROOT / "results" / "backtest_V3" / "all_curves_V3.csv"
PRICES_PATH = ROOT / "data" / "raw" / "commodity_prices.csv"
WEIGHTS_PATH = ROOT / "results" / "backtest_V3" / "weights_history_V3.csv"

OUT_DIR = ROOT / "results" / "cyclical_regime_performance"
CHART_DIR = OUT_DIR / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252

CYCLICAL_TICKERS = ["SLV", "USO", "UNG", "CPER", "DBA"]  # exclude GLD


def clean_returns(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def cagr(r):
    r = clean_returns(r)
    if r.empty:
        return np.nan
    total = (1 + r).prod() - 1
    years = len(r) / TRADING_DAYS
    return (1 + total) ** (1 / years) - 1 if years > 0 else np.nan


def sharpe(r):
    r = clean_returns(r)
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return r.mean() / r.std() * np.sqrt(TRADING_DAYS)


def max_drawdown(r):
    r = clean_returns(r)
    if r.empty:
        return np.nan
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1
    return dd.min()

def load_cyclical_regime():
    prices = pd.read_csv(PRICES_PATH)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].astype(str).str.upper()

    missing = sorted(set(CYCLICAL_TICKERS) - set(prices["ticker"].unique()))
    if missing:
        raise ValueError(f"Missing cyclical tickers from price data: {missing}")

    wide = (
        prices[prices["ticker"].isin(CYCLICAL_TICKERS)]
        .pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
        .sort_index()
    )

    # Equal-weight ex-gold cyclical commodity basket.
    basket_returns = wide.pct_change().mean(axis=1, skipna=True)
    basket_price = (1 + basket_returns.fillna(0)).cumprod()

    df = pd.DataFrame({"cyclical_basket": basket_price})

    # 12-month cyclical commodity return.
    df["cyclical_ret_252"] = df["cyclical_basket"].pct_change(252)

    # Lag by one day to avoid same-day lookahead.
    ret_252_lag = df["cyclical_ret_252"].shift(1)

    df["regime"] = np.select(
        [
            ret_252_lag > 0.05,
            ret_252_lag < -0.05,
        ],
        [
            "Bull",
            "Bear",
        ],
        default="Chop",
    )

    return df[[
        "regime",
        "cyclical_basket",
        "cyclical_ret_252",
    ]]

def load_curves():
    curves = pd.read_csv(CURVES_PATH)
    curves["date"] = pd.to_datetime(curves["date"])

    if "strategy" not in curves.columns:
        raise ValueError("all_curves_V3.csv must contain 'strategy' column.")
    if "net_return" not in curves.columns:
        raise ValueError("all_curves_V3.csv must contain 'net_return' column.")

    return curves


def load_avg_gld_by_regime(regimes):
    if not WEIGHTS_PATH.exists():
        return pd.DataFrame(columns=["regime", "avg_gld_weight"])

    weights = pd.read_csv(WEIGHTS_PATH)
    weights["date"] = pd.to_datetime(weights["date"])

    # Handle either wide or long format.
    if "GLD" in weights.columns:
        gld = weights[["date", "GLD"]].rename(columns={"GLD": "gld_weight"})
    elif {"ticker", "weight"}.issubset(weights.columns):
        gld = weights[weights["ticker"].astype(str).str.upper() == "GLD"][["date", "weight"]]
        gld = gld.rename(columns={"weight": "gld_weight"})
    elif {"ticker", "executed_weight"}.issubset(weights.columns):
        gld = weights[weights["ticker"].astype(str).str.upper() == "GLD"][["date", "executed_weight"]]
        gld = gld.rename(columns={"executed_weight": "gld_weight"})
    else:
        return pd.DataFrame(columns=["regime", "avg_gld_weight"])

    joined = gld.merge(regimes.reset_index()[["date", "regime"]], on="date", how="inner")

    return (
        joined.groupby("regime")["gld_weight"]
        .mean()
        .reset_index()
        .rename(columns={"gld_weight": "avg_gld_weight"})
    )


def build_summary(curves, regimes):
    df = curves.merge(
        regimes.reset_index()[["date", "regime"]],
        on="date",
        how="inner",
    )

    rows = []

    for (strategy, regime), group in df.groupby(["strategy", "regime"]):
        r = group["net_return"]
        rows.append({
            "strategy": strategy,
            "regime": regime,
            "days": len(group),
            "cagr": cagr(r),
            "sharpe": sharpe(r),
            "max_drawdown": max_drawdown(r),
            "total_return": (1 + clean_returns(r)).prod() - 1,
            "volatility": clean_returns(r).std() * np.sqrt(TRADING_DAYS),
        })

    summary = pd.DataFrame(rows)

    gld_by_regime = load_avg_gld_by_regime(regimes)
    if not gld_by_regime.empty:
        summary = summary.merge(gld_by_regime, on="regime", how="left")

    order = {"Bull": 0, "Chop": 1, "Bear": 2}
    summary["regime_order"] = summary["regime"].map(order)
    summary = summary.sort_values(["regime_order", "strategy"]).drop(columns=["regime_order"])

    return summary


def plot_cagr(summary):
    keep = ["model", "equal_weight", "gold_only", "cash"]
    plot_df = summary[summary["strategy"].isin(keep)].copy()

    regime_order = ["Bull", "Chop", "Bear"]
    strategy_order = [s for s in keep if s in plot_df["strategy"].unique()]

    pivot = (
        plot_df.pivot(index="regime", columns="strategy", values="cagr")
        .reindex(regime_order)
        .reindex(columns=strategy_order)
    )

    ax = pivot.plot(kind="bar", figsize=(12, 6), width=0.78)

    ax.set_title("Performance by Ex-Gold Cyclical Commodity Momentum Regime")
    ax.set_ylabel("Conditional CAGR")
    ax.set_xlabel("Regime")
    ax.axhline(0, color="black", linewidth=1)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Strategy")

    plt.tight_layout()
    out_path = CHART_DIR / "cyclical_regime_cagr_by_strategy.png"
    plt.savefig(out_path, dpi=170)
    plt.close()

    return out_path


def main():
    regimes = load_cyclical_regime()
    curves = load_curves()

    summary = build_summary(curves, regimes)

    summary_path = OUT_DIR / "cyclical_regime_performance_summary.csv"
    summary.to_csv(summary_path, index=False)

    chart_path = plot_cagr(summary)

    print("\nCyclical commodity regime performance:")
    print(summary.to_string(index=False))

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved chart:   {chart_path}")


if __name__ == "__main__":
    main()