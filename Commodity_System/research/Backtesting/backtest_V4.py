from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

THIS_FILE = Path(__file__).resolve()
BACKTESTING_DIR = THIS_FILE.parent
RESEARCH_DIR = BACKTESTING_DIR.parent
COMMODITY_ROOT = RESEARCH_DIR.parent
PROJECT_ROOT = COMMODITY_ROOT.parent

for path in [PROJECT_ROOT, COMMODITY_ROOT, RESEARCH_DIR, BACKTESTING_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ============================================================
# IMPORT EXISTING V2 ENGINE + NEW ROUTER
# ============================================================

try:
    import backtest_V2 as V2
except ImportError as exc:
    raise ImportError(
        "Could not import backtest_V2.py. Put backtest_V4.py in "
        "Commodity_System/research/Backtesting next to backtest_V2.py."
    ) from exc

try:
    from regime_router import build_v4_routed_weights
except ImportError as exc:
    raise ImportError(
        "Could not import regime_router.py. Put regime_router.py in "
        "Commodity_System/research/Backtesting next to backtest_V4.py."
    ) from exc

try:
    from Commodity_System.commodity_strategy import build_production_strategy_score_history
except ImportError:
    try:
        from commodity_strategy import build_production_strategy_score_history
    except ImportError:
        build_production_strategy_score_history = None


# ============================================================
# V4 SETTINGS
# ============================================================

OUTPUT_DIR = V2.RESULTS_DIR / "backtest_V4"

V4_VARIANTS = {
    "v4_router": "base",
    "v4_no_chop_reduction": "no_chop",
    "v4_no_gld_bear_exception": "no_gld_bear_exception",
    "v4_crisis_only": "crisis_only",
}


# ============================================================
# HELPERS
# ============================================================

def safe_to_csv(df: pd.DataFrame | pd.Series, path: Path, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df, pd.Series):
        df.to_frame().to_csv(path, index=index)
    else:
        df.to_csv(path, index=index)


def load_scores_history() -> pd.DataFrame:
    if build_production_strategy_score_history is None:
        print("Warning: score-history function unavailable. Chop threshold filter will be weaker.")
        return pd.DataFrame(columns=["date", "ticker", "final_score"])

    try:
        scores = build_production_strategy_score_history(save=False)
        print("Loaded production final-score history for V4 routing.")
        return scores
    except Exception as exc:
        print(f"Warning: failed to build score history: {exc}")
        print("V4 will still run, but chop threshold filter will be weaker.")
        return pd.DataFrame(columns=["date", "ticker", "final_score"])


def simulate_with_rebalance_mode(
    *,
    name: str,
    weights: pd.DataFrame,
    market_data: dict[str, pd.DataFrame],
    settings: dict[str, Any],
    mode: str,
) -> dict[str, pd.DataFrame]:
    """
    V2 stores rebalance mode as a module-level variable, not inside settings.
    This wrapper lets V3 baselines stay monthly while V4 can test daily defensive routing.
    """
    old_mode = V2.BACKTEST_REBALANCE_MODE
    V2.BACKTEST_REBALANCE_MODE = mode

    try:
        return V2.simulate_strategy_v2(
            name=name,
            raw_target_weights=weights,
            market_data=market_data,
            settings=settings,
            initial_capital=V2.INITIAL_CAPITAL,
        )
    finally:
        V2.BACKTEST_REBALANCE_MODE = old_mode


def build_regime_performance_table(
    results: dict[str, dict[str, pd.DataFrame]],
    regime_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    if regime_diagnostics.empty or "smoothed_regime" not in regime_diagnostics.columns:
        return pd.DataFrame()

    regimes = regime_diagnostics["smoothed_regime"].copy()
    regimes.index = pd.to_datetime(regimes.index)

    rows = []

    for strategy, result in results.items():
        curve = result["curve"].copy()
        curve.index = pd.to_datetime(curve.index)

        joined = curve[["net_return", "exposure"]].join(regimes.rename("regime"), how="left")
        joined = joined.dropna(subset=["regime"])

        for regime, group in joined.groupby("regime"):
            returns = group["net_return"].replace([np.inf, -np.inf], np.nan).dropna()

            if returns.empty:
                continue

            equity = (1.0 + returns).cumprod()
            dd = equity / equity.cummax() - 1.0
            ann_return = returns.mean() * V2.TRADING_DAYS_PER_YEAR
            ann_vol = returns.std() * np.sqrt(V2.TRADING_DAYS_PER_YEAR)
            sharpe = ann_return / ann_vol if ann_vol and ann_vol > 0 else np.nan

            rows.append(
                {
                    "strategy": strategy,
                    "regime": regime,
                    "days": len(returns),
                    "total_return": float(equity.iloc[-1] - 1.0),
                    "annualised_return": float(ann_return),
                    "annualised_volatility": float(ann_vol),
                    "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
                    "max_drawdown": float(dd.min()),
                    "hit_rate": float((returns > 0).mean()),
                    "average_exposure": float(group["exposure"].mean()),
                }
            )

    return pd.DataFrame(rows).sort_values(["strategy", "regime"]).reset_index(drop=True)


def save_v4_outputs(
    *,
    results: dict[str, dict[str, pd.DataFrame]],
    routed_weights_by_name: dict[str, pd.DataFrame],
    diagnostic_weights_by_name: dict[str, pd.DataFrame],
    regime_diagnostics_by_name: dict[str, pd.DataFrame],
    performance_summary: pd.DataFrame,
    alpha_beta_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    regime_performance: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_curves = []
    all_trade_logs = []
    all_executed_weights = []

    for name, result in results.items():
        curve = result["curve"].reset_index()
        all_curves.append(curve)

        trade_log = result["trade_log"]
        if trade_log is not None and not trade_log.empty:
            all_trade_logs.append(trade_log)

        executed = result["executed_weights"].copy().reset_index()
        all_executed_weights.append(executed)

        safe_name = name.replace("/", "_")

        result["curve"].reset_index().to_csv(OUTPUT_DIR / f"{safe_name}_curve.csv", index=False)
        result["trade_log"].to_csv(OUTPUT_DIR / f"{safe_name}_trade_log.csv", index=False)
        result["executed_weights"].reset_index().to_csv(OUTPUT_DIR / f"{safe_name}_executed_weights.csv", index=False)
        result["signal_weights"].reset_index().to_csv(OUTPUT_DIR / f"{safe_name}_signal_weights.csv", index=False)
        result["execution_plan"].reset_index().rename(columns={"index": "execution_date"}).to_csv(
            OUTPUT_DIR / f"{safe_name}_execution_plan.csv",
            index=False,
        )

    if all_curves:
        pd.concat(all_curves, ignore_index=True).to_csv(OUTPUT_DIR / "all_curves_V4.csv", index=False)

    if all_trade_logs:
        pd.concat(all_trade_logs, ignore_index=True).to_csv(OUTPUT_DIR / "all_trade_logs_V4.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / "all_trade_logs_V4.csv", index=False)

    if all_executed_weights:
        pd.concat(all_executed_weights, ignore_index=True).to_csv(OUTPUT_DIR / "all_executed_weights_V4.csv", index=False)

    for name, weights in routed_weights_by_name.items():
        weights.reset_index().rename(columns={"index": "date"}).to_csv(
            OUTPUT_DIR / f"{name}_routed_weights.csv",
            index=False,
        )

    for name, weights in diagnostic_weights_by_name.items():
        weights.reset_index().rename(columns={"index": "date"}).to_csv(
            OUTPUT_DIR / f"{name}_diagnostic_weights.csv",
            index=False,
        )

    for name, diagnostics in regime_diagnostics_by_name.items():
        diagnostics.reset_index().rename(columns={"index": "date"}).to_csv(
            OUTPUT_DIR / f"{name}_regime_diagnostics.csv",
            index=False,
        )

    performance_summary.to_csv(OUTPUT_DIR / "performance_summary_V4.csv", index=False)
    alpha_beta_summary.to_csv(OUTPUT_DIR / "alpha_beta_summary_V4.csv", index=False)
    cost_summary.to_csv(OUTPUT_DIR / "cost_summary_V4.csv", index=False)
    regime_performance.to_csv(OUTPUT_DIR / "regime_performance_V4.csv", index=False)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n========== BACKTEST V4: FROZEN V3 + COMMODITY REGIME ROUTER ==========")
    print("V4 does not change the V3 scorer. It routes V3 weights by commodity regime.")

    settings = V2.build_base_settings()

    print("\nMain execution settings:")
    for key, value in settings.items():
        print(f"  {key}: {value}")

    market_data = V2.load_market_data(settings=settings)
    close = market_data["close"]

    v3_weights = V2.load_target_weights()
    tickers = [ticker for ticker in v3_weights.columns if ticker in market_data["returns"].columns]
    v3_weights = v3_weights.reindex(columns=tickers).fillna(0.0)

    scores_history = load_scores_history()

    equal_weight = V2.make_equal_weight(index=v3_weights.index, tickers=tickers)
    gold_only = V2.make_gold_only(index=v3_weights.index, tickers=tickers)
    cash = V2.make_cash(index=v3_weights.index, tickers=tickers)

    routed_weights_by_name: dict[str, pd.DataFrame] = {}
    diagnostic_weights_by_name: dict[str, pd.DataFrame] = {}
    regime_diagnostics_by_name: dict[str, pd.DataFrame] = {}

    for strategy_name, variant in V4_VARIANTS.items():
        routed, regime_diag, diagnostic_weights = build_v4_routed_weights(
            v3_weights=v3_weights,
            prices=close,
            scores=scores_history,
            variant=variant,
        )

        routed_weights_by_name[strategy_name] = routed
        regime_diagnostics_by_name[strategy_name] = regime_diag
        diagnostic_weights_by_name[strategy_name] = diagnostic_weights

    strategies = {
        # Core controls
        "frozen_v3": {"weights": v3_weights, "mode": "monthly"},
        "equal_weight": {"weights": equal_weight, "mode": "monthly"},
        "gold_only": {"weights": gold_only, "mode": "monthly"},
        "cash": {"weights": cash, "mode": "monthly"},

        # Clean V4 tests: all monthly-only for fair comparison
        "v4_router_monthly": {
            "weights": routed_weights_by_name["v4_router"],
            "mode": "monthly",
        },
        "v4_no_chop_reduction_monthly": {
            "weights": routed_weights_by_name["v4_no_chop_reduction"],
            "mode": "monthly",
        },
        "v4_no_gld_bear_exception_monthly": {
            "weights": routed_weights_by_name["v4_no_gld_bear_exception"],
            "mode": "monthly",
        },
        "v4_crisis_only_monthly": {
            "weights": routed_weights_by_name["v4_crisis_only"],
            "mode": "monthly",
        },

        # Keep this only as a warning/experimental line.
        # This is NOT a valid final implementation because it causes daily rebalancing.
        "v4_router_daily_experimental": {
            "weights": routed_weights_by_name["v4_router"],
            "mode": "daily",
        },
    }

    results: dict[str, dict[str, pd.DataFrame]] = {}

    for name, spec in strategies.items():
        print(f"\nRunning strategy: {name} | rebalance mode: {spec['mode']}")

        results[name] = simulate_with_rebalance_mode(
            name=name,
            weights=spec["weights"],
            market_data=market_data,
            settings=settings,
            mode=spec["mode"],
        )

    performance_summary = V2.build_performance_summary(
        results=results,
        benchmark_name="equal_weight",
    )

    alpha_beta_summary = V2.build_alpha_beta_summary(
        results=results,
        benchmark_names=["equal_weight", "gold_only", "cash", "frozen_v3"],
    )

    cost_summary = V2.build_cost_summary_table(
        results=results,
        settings=settings,
    )

    regime_performance = build_regime_performance_table(
        results=results,
        regime_diagnostics=regime_diagnostics_by_name["v4_router"],
    )

    save_v4_outputs(
        results=results,
        routed_weights_by_name=routed_weights_by_name,
        diagnostic_weights_by_name=diagnostic_weights_by_name,
        regime_diagnostics_by_name=regime_diagnostics_by_name,
        performance_summary=performance_summary,
        alpha_beta_summary=alpha_beta_summary,
        cost_summary=cost_summary,
        regime_performance=regime_performance,
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
        "average_exposure",
        "average_cash",
        "average_daily_turnover",
        "annualised_turnover",
        "total_transaction_cost_drag",
        "alpha_annualised",
        "beta",
        "information_ratio",
    ]
    cols = [col for col in cols if col in performance_summary.columns]

    print("\nBacktest V4 complete.")
    print(f"Saved V4 outputs to: {OUTPUT_DIR}")

    print("\nPerformance summary:")
    print(performance_summary[cols].to_string(index=False))

    if not regime_performance.empty:
        regime_cols = [
            "strategy",
            "regime",
            "days",
            "total_return",
            "annualised_return",
            "annualised_volatility",
            "sharpe",
            "max_drawdown",
            "average_exposure",
        ]
        print("\nRegime performance summary:")
        print(regime_performance[regime_cols].to_string(index=False))

    print("\nKey files to inspect first:")
    print(OUTPUT_DIR / "performance_summary_V4.csv")
    print(OUTPUT_DIR / "regime_performance_V4.csv")
    print(OUTPUT_DIR / "v4_router_regime_diagnostics.csv")
    print(OUTPUT_DIR / "v4_router_diagnostic_weights.csv")
    print(OUTPUT_DIR / "v4_router_curve.csv")


if __name__ == "__main__":
    main()
