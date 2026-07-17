from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

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
# IMPORT V2 ENGINE
# ============================================================
# Important:
# V3 deliberately reuses V2's execution engine. Do not rewrite the
# backtest unless there is a specific bug. The purpose of V3 is:
#   1. same strategy
#   2. same realistic execution logic
#   3. better saved data for diagnostics/reporting

try:
    import backtest_V2 as V2
except ImportError as exc:
    raise ImportError(
        "Could not import backtest_V2.py. Make sure backtest_V3.py is in "
        "Commodity_System/research/Backtesting next to backtest_V2.py."
    ) from exc


# ============================================================
# OPTIONAL DIAGNOSTICS IMPORTS
# ============================================================

try:
    from diagnostics import DiagnosticsConfig, generate_full_diagnostics_report
except ImportError:
    DiagnosticsConfig = None
    generate_full_diagnostics_report = None


# ============================================================
# V3 SETTINGS
# ============================================================

OUTPUT_DIR = V2.RESULTS_DIR / "backtest_V3"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"

RUN_DIAGNOSTICS = bool(V2.cfg("BACKTEST_V3_RUN_DIAGNOSTICS", True))
RUN_SCENARIO_TESTS = bool(V2.cfg("BACKTEST_V3_RUN_SCENARIO_TESTS", V2.cfg("BACKTEST_V2_RUN_SCENARIO_TESTS", True)))

# Keep this False unless you have implemented diagnostics_report.py.
RUN_HTML_REPORT = bool(V2.cfg("BACKTEST_V3_RUN_HTML_REPORT", False))

RISK_FREE_RATE_ANNUAL = 0.0

RUN_RISK_METRICS = bool(V2.cfg("BACKTEST_V3_RUN_RISK_METRICS", False))


# ============================================================
# SMALL HELPERS
# ============================================================

def safe_to_csv(df: pd.DataFrame | pd.Series, path: Path, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(df, pd.Series):
        df.to_frame().to_csv(path, index=index)
    else:
        df.to_csv(path, index=index)


def clean_weights_for_diagnostics(weights: pd.DataFrame) -> pd.DataFrame:
    """
    V2 executed_weights includes a 'strategy' column.
    Diagnostics wants wide date-indexed ticker weights.
    """
    out = weights.copy()

    if "strategy" in out.columns:
        out = out.drop(columns=["strategy"])

    out.index = pd.to_datetime(out.index)
    out.index.name = "date"

    return out.sort_index().fillna(0.0)


def build_benchmark_returns(results: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Build benchmark return matrix for diagnostics.py.

    Columns:
      equal_weight
      gold_only
      cash
    """
    benchmark_cols = {}

    for name in ["equal_weight", "gold_only", "cash"]:
        if name in results:
            curve = results[name]["curve"]
            benchmark_cols[name] = curve["net_return"]

    if not benchmark_cols:
        return pd.DataFrame()

    out = pd.DataFrame(benchmark_cols)
    out.index.name = "date"
    return out.sort_index()


def build_asset_returns(market_data: dict[str, pd.DataFrame], tickers: list[str]) -> pd.DataFrame:
    returns = market_data["returns"].copy()
    returns = returns.reindex(columns=tickers).fillna(0.0)
    returns.index = pd.to_datetime(returns.index)
    returns.index.name = "date"
    return returns.sort_index()


def build_price_data_for_audit(market_data: dict[str, pd.DataFrame], tickers: list[str]) -> pd.DataFrame:
    close = market_data.get("close", pd.DataFrame()).copy()

    if close.empty:
        return pd.DataFrame()

    close = close.reindex(columns=tickers)
    close.index = pd.to_datetime(close.index)
    close.index.name = "date"

    return close.sort_index()


# ============================================================
# SCORE HISTORY CAPTURE
# ============================================================

def try_import_function(module_name: str, function_name: str):
    try:
        module = __import__(module_name, fromlist=[function_name])
        return getattr(module, function_name, None)
    except Exception:
        return None


def normalise_score_history(scores: pd.DataFrame | None) -> pd.DataFrame:
    """
    Expected ideal format:
      date | ticker | final_score | momentum_score | ... component scores

    This function is intentionally forgiving because your scoring pipeline has
    evolved. If no proper score history exists yet, V3 still runs; diagnostics
    will simply skip feature attribution.
    """
    if scores is None or scores.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    out = scores.copy()

    # If date is index, expose it.
    if "date" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={out.index.name or "index": "date"})
        elif "Date" in out.columns:
            out = out.rename(columns={"Date": "date"})

    # Standardise ticker column.
    if "ticker" not in out.columns:
        for candidate in ["symbol", "asset", "Ticker"]:
            if candidate in out.columns:
                out = out.rename(columns={candidate: "ticker"})
                break

    # Wide score matrix fallback:
    # date | GLD | SLV | ...
    # becomes date | ticker | final_score
    if "ticker" not in out.columns:
        possible_date_col = "date" if "date" in out.columns else None

        if possible_date_col is not None:
            value_cols = [c for c in out.columns if c != possible_date_col]

            if value_cols:
                out = out.melt(
                    id_vars=[possible_date_col],
                    value_vars=value_cols,
                    var_name="ticker",
                    value_name="final_score",
                )

    if "date" not in out.columns or "ticker" not in out.columns:
        return pd.DataFrame(columns=["date", "ticker"])

    out["date"] = pd.to_datetime(out["date"])
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()

    for col in out.columns:
        if col not in ["date", "ticker"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)

    return out


def try_build_scores_history() -> pd.DataFrame:
    """
    Attempts to collect historical score data without touching production logic.

    This is deliberately optional. V2 did not need score history; diagnostics do.
    If your strategy module later exposes a clean function like
    build_production_strategy_score_history(), this will pick it up.
    """

    # 1. Preferred: strategy/scoring functions if they exist.
    candidate_functions = [
        ("Commodity_System.commodity_strategy", "build_production_strategy_score_history"),
        ("Commodity_System.commodity_strategy", "build_strategy_score_history"),
        ("Commodity_System.commodity_strategy", "build_score_history"),
        ("commodity_strategy", "build_production_strategy_score_history"),
        ("commodity_strategy", "build_strategy_score_history"),
        ("commodity_strategy", "build_score_history"),
        ("Commodity_System.scoring.score", "build_score_history"),
        ("scoring.score", "build_score_history"),
    ]

    for module_name, function_name in candidate_functions:
        fn = try_import_function(module_name, function_name)

        if fn is None:
            continue

        try:
            scores = fn()
            normalised = normalise_score_history(scores)

            if not normalised.empty:
                print(f"Loaded score history from {module_name}.{function_name}()")
                return normalised

        except Exception as exc:
            print(f"Warning: failed score history function {module_name}.{function_name}(): {exc}")

    # 2. Fallback: look for already-saved score history CSVs.
    candidate_paths = [
        V2.RESULTS_DIR / "scores_history.csv",
        V2.RESULTS_DIR / "score_history.csv",
        V2.RESULTS_DIR / "final_scores_history.csv",
        V2.RESULTS_DIR / "backtest" / "scores_history.csv",
        V2.RESULTS_DIR / "backtest_V2" / "scores_history_V2.csv",
        V2.RESULTS_DIR / "backtest_V2" / "scores_history.csv",
        V2.COMMODITY_ROOT / "results" / "scores_history.csv",
        V2.COMMODITY_ROOT / "results" / "final_scores_history.csv",
    ]

    for path in candidate_paths:
        try:
            if path.exists():
                scores = pd.read_csv(path)
                normalised = normalise_score_history(scores)

                if not normalised.empty:
                    print(f"Loaded score history from CSV: {path}")
                    return normalised

        except Exception as exc:
            print(f"Warning: failed reading score history CSV {path}: {exc}")

    print(
        "Warning: no score history found. "
        "V3 will still run, but feature attribution charts will be skipped until "
        "the scoring pipeline exposes historical component scores."
    )

    return pd.DataFrame(columns=["date", "ticker"])


# ============================================================
# V3 OUTPUT SAVING
# ============================================================

def save_v3_outputs(
    *,
    results: dict[str, dict[str, pd.DataFrame]],
    market_data: dict[str, pd.DataFrame],
    model_weights: pd.DataFrame,
    scores_history: pd.DataFrame,
    performance_summary: pd.DataFrame,
    alpha_beta_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    scenario_summary: Optional[pd.DataFrame],
) -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    all_curves = []
    all_trade_logs = []
    all_executed_weights = []

    for name, result in results.items():
        curve = result["curve"].reset_index()
        all_curves.append(curve)

        trade_log = result["trade_log"]
        if trade_log is not None and not trade_log.empty:
            all_trade_logs.append(trade_log)

        executed_weights = clean_weights_for_diagnostics(result["executed_weights"])
        executed_weights_out = executed_weights.reset_index()
        executed_weights_out.insert(1, "strategy", name)
        all_executed_weights.append(executed_weights_out)

        if name == "model":
            model_curve_path = OUTPUT_DIR / "model_curve_V3.csv"
            trade_log_path = OUTPUT_DIR / "model_trade_log_V3.csv"
            weights_path = OUTPUT_DIR / "weights_history_V3.csv"
            executed_model_weights_path = OUTPUT_DIR / "executed_model_weights_V3.csv"
            signal_weights_path = OUTPUT_DIR / "target_signal_weights_V3.csv"
            execution_plan_path = OUTPUT_DIR / "execution_plan_V3.csv"

            result["curve"].reset_index().to_csv(model_curve_path, index=False)
            result["trade_log"].to_csv(trade_log_path, index=False)

            # Canonical diagnostics name.
            executed_weights.reset_index().to_csv(weights_path, index=False)

            # Familiar V2-style name too.
            executed_weights.reset_index().to_csv(executed_model_weights_path, index=False)

            result["signal_weights"].reset_index().to_csv(signal_weights_path, index=False)

            result["execution_plan"].reset_index().rename(
                columns={"index": "execution_date"}
            ).to_csv(execution_plan_path, index=False)

            paths["model_curve"] = model_curve_path
            paths["model_trade_log"] = trade_log_path
            paths["weights_history"] = weights_path
            paths["executed_model_weights"] = executed_model_weights_path
            paths["target_signal_weights"] = signal_weights_path
            paths["execution_plan"] = execution_plan_path

    if all_curves:
        all_curves_path = OUTPUT_DIR / "all_curves_V3.csv"
        pd.concat(all_curves, ignore_index=True).to_csv(all_curves_path, index=False)
        paths["all_curves"] = all_curves_path

    all_trade_logs_path = OUTPUT_DIR / "all_trade_logs_V3.csv"
    if all_trade_logs:
        pd.concat(all_trade_logs, ignore_index=True).to_csv(all_trade_logs_path, index=False)
    else:
        pd.DataFrame().to_csv(all_trade_logs_path, index=False)
    paths["all_trade_logs"] = all_trade_logs_path

    if all_executed_weights:
        all_weights_path = OUTPUT_DIR / "all_executed_weights_V3.csv"
        pd.concat(all_executed_weights, ignore_index=True).to_csv(all_weights_path, index=False)
        paths["all_executed_weights"] = all_weights_path

    asset_returns = build_asset_returns(
        market_data=market_data,
        tickers=list(model_weights.columns),
    )

    asset_returns_path = OUTPUT_DIR / "asset_returns_V3.csv"
    asset_returns.reset_index().to_csv(asset_returns_path, index=False)
    paths["asset_returns"] = asset_returns_path

    price_data = build_price_data_for_audit(
        market_data=market_data,
        tickers=list(model_weights.columns),
    )

    if not price_data.empty:
        price_data_path = OUTPUT_DIR / "price_data_V3.csv"
        price_data.reset_index().to_csv(price_data_path, index=False)
        paths["price_data"] = price_data_path

    benchmark_returns = build_benchmark_returns(results)

    benchmark_returns_path = OUTPUT_DIR / "benchmark_returns_V3.csv"
    benchmark_returns.reset_index().to_csv(benchmark_returns_path, index=False)
    paths["benchmark_returns"] = benchmark_returns_path

    scores_path = OUTPUT_DIR / "scores_history_V3.csv"
    scores_history.to_csv(scores_path, index=False)
    paths["scores_history"] = scores_path

    performance_path = OUTPUT_DIR / "performance_summary_V3.csv"
    alpha_beta_path = OUTPUT_DIR / "alpha_beta_summary_V3.csv"
    cost_path = OUTPUT_DIR / "cost_summary_V3.csv"

    performance_summary.to_csv(performance_path, index=False)
    alpha_beta_summary.to_csv(alpha_beta_path, index=False)
    cost_summary.to_csv(cost_path, index=False)

    paths["performance_summary"] = performance_path
    paths["alpha_beta_summary"] = alpha_beta_path
    paths["cost_summary"] = cost_path

    if scenario_summary is not None:
        scenario_path = OUTPUT_DIR / "scenario_summary_V3.csv"
        scenario_summary.to_csv(scenario_path, index=False)
        paths["scenario_summary"] = scenario_path

    return paths


# ============================================================
# OPTIONAL V2 COMPARISON
# ============================================================

def compare_against_existing_v2(performance_summary_v3: pd.DataFrame) -> pd.DataFrame:
    """
    Optional sanity check.

    If the old V2 output exists, this compares model headline metrics.
    V3 should be extremely close to V2 because it uses the same engine.
    """
    v2_path = V2.RESULTS_DIR / "backtest_V2" / "performance_summary_V2.csv"

    if not v2_path.exists():
        return pd.DataFrame()

    try:
        v2 = pd.read_csv(v2_path)
    except Exception:
        return pd.DataFrame()

    v3 = performance_summary_v3.copy()

    if "strategy" not in v2.columns or "strategy" not in v3.columns:
        return pd.DataFrame()

    v2_model = v2[v2["strategy"] == "model"].copy()
    v3_model = v3[v3["strategy"] == "model"].copy()

    if v2_model.empty or v3_model.empty:
        return pd.DataFrame()

    metrics = [
        "final_equity",
        "cagr",
        "annualised_volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "average_exposure",
        "average_cash",
    ]

    rows = []

    for metric in metrics:
        if metric not in v2_model.columns or metric not in v3_model.columns:
            continue

        old = pd.to_numeric(v2_model.iloc[0][metric], errors="coerce")
        new = pd.to_numeric(v3_model.iloc[0][metric], errors="coerce")

        rows.append(
            {
                "metric": metric,
                "v2": old,
                "v3": new,
                "difference": new - old,
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty:
        out.to_csv(OUTPUT_DIR / "v2_v3_model_comparison.csv", index=False)

    return out


# ============================================================
# DIAGNOSTICS CALL
# ============================================================

def run_diagnostics_if_available(
    *,
    results: dict[str, dict[str, pd.DataFrame]],
    market_data: dict[str, pd.DataFrame],
    model_weights: pd.DataFrame,
    scores_history: pd.DataFrame,
) -> Optional[dict[str, Any]]:
    if not RUN_DIAGNOSTICS:
        print("\nDiagnostics disabled by BACKTEST_V3_RUN_DIAGNOSTICS = False.")
        return None

    if generate_full_diagnostics_report is None or DiagnosticsConfig is None:
        print(
            "\nDiagnostics skipped: could not import diagnostics.py. "
            "Make sure Commodity_System/research/diagnostics.py exists."
        )
        return None

    print("\nGenerating V3 diagnostics package.")

    model_result = results["model"]

    model_curve = model_result["curve"].reset_index()
    weights_history = clean_weights_for_diagnostics(model_result["executed_weights"]).reset_index()

    asset_returns = build_asset_returns(
        market_data=market_data,
        tickers=list(model_weights.columns),
    ).reset_index()

    benchmark_returns = build_benchmark_returns(results).reset_index()

    trade_log = model_result["trade_log"].copy()

    min_score_to_hold = V2.cfg("MIN_SCORE_TO_HOLD", None)

    try:
        if min_score_to_hold is not None:
            min_score_to_hold = float(min_score_to_hold)
    except Exception:
        min_score_to_hold = None

    diagnostics_config = DiagnosticsConfig(
        initial_capital=V2.INITIAL_CAPITAL,
        periods_per_year=V2.TRADING_DAYS_PER_YEAR,
        rolling_window=252,
        short_rolling_window=126,
        risk_free_rate=RISK_FREE_RATE_ANNUAL,
        min_score_to_hold=min_score_to_hold,
    )

    manifest = generate_full_diagnostics_report(
        model_curve=model_curve,
        weights_history=weights_history,
        scores_history=scores_history,
        asset_returns=asset_returns,
        benchmark_returns=benchmark_returns,
        trade_log=trade_log,
        output_dir=DIAGNOSTICS_DIR,
        config=diagnostics_config,
    )

    print(f"Diagnostics saved to: {DIAGNOSTICS_DIR}")

    return manifest


# ============================================================
# OPTIONAL HTML REPORT CALL
# ============================================================

def run_html_report_if_available() -> None:
    """
    This is intentionally optional because diagnostics_report.py may not exist yet.

    Once diagnostics_report.py is written, we can standardise its function name.
    For now this tries a few sensible names and fails softly.
    """
    if not RUN_HTML_REPORT:
        return

    try:
        import diagnostics_report
    except ImportError:
        print("\nHTML report skipped: diagnostics_report.py not available yet.")
        return

    candidate_functions = [
        "build_diagnostics_report",
        "generate_diagnostics_report",
        "generate_html_report",
        "main_from_folder",
    ]

    for name in candidate_functions:
        fn = getattr(diagnostics_report, name, None)

        if fn is None:
            continue

        try:
            print(f"\nGenerating HTML diagnostics report using diagnostics_report.{name}().")
            fn(DIAGNOSTICS_DIR)
            return
        except TypeError:
            try:
                fn(input_dir=DIAGNOSTICS_DIR, output_dir=DIAGNOSTICS_DIR)
                return
            except Exception as exc:
                print(f"HTML report function {name} failed: {exc}")
        except Exception as exc:
            print(f"HTML report function {name} failed: {exc}")

    print("\nHTML report skipped: no compatible function found in diagnostics_report.py.")

def run_risk_metrics_if_available() -> None:
    if not RUN_RISK_METRICS:
        return

    try:
        from risk_metrics import generate_risk_package
    except ImportError:
        print("\nRisk metrics skipped: risk_metrics.py not available.")
        return

    try:
        print("\nGenerating risk metrics package from V3 outputs.")
        generate_risk_package(
            v3_output_dir=OUTPUT_DIR,
            output_dir=V2.RESULTS_DIR / "risk",
        )
    except Exception as exc:
        print(f"\nRisk metrics skipped due to error: {exc}")

# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n========== BACKTEST V3: V2 ENGINE + DIAGNOSTIC DATA CAPTURE ==========")
    print("V3 uses the same execution logic as V2. If strategy/settings are unchanged,")
    print("model performance should match V2 apart from output naming.")

    settings = V2.build_base_settings()

    print("\nMain settings:")
    for key, value in settings.items():
        print(f"  {key}: {value}")

    # ------------------------------------------------------------
    # Load same market data and same production weights as V2.
    # ------------------------------------------------------------
    market_data = V2.load_market_data(settings=settings)
    returns = market_data["returns"]

    model_weights = V2.load_target_weights()
    tickers = list(model_weights.columns)
    model_weights = model_weights.reindex(columns=tickers).fillna(0.0)

    # ------------------------------------------------------------
    # Same benchmarks as V2.
    # ------------------------------------------------------------
    equal_weight = V2.make_equal_weight(
        index=model_weights.index,
        tickers=tickers,
    )

    gold_only = V2.make_gold_only(
        index=model_weights.index,
        tickers=tickers,
    )

    cash = V2.make_cash(
        index=model_weights.index,
        tickers=tickers,
    )

    strategies = {
        "model": model_weights,
        "equal_weight": equal_weight,
        "gold_only": gold_only,
        "cash": cash,
    }

    # ------------------------------------------------------------
    # Run same V2 engine.
    # ------------------------------------------------------------
    results: dict[str, dict[str, pd.DataFrame]] = {}

    for name, weights in strategies.items():
        print(f"\nRunning V3 strategy using V2 engine: {name}")

        result = V2.simulate_strategy_v2(
            name=name,
            raw_target_weights=weights,
            market_data=market_data,
            settings=settings,
            initial_capital=V2.INITIAL_CAPITAL,
        )

        results[name] = result

    # ------------------------------------------------------------
    # Same V2 summary logic.
    # ------------------------------------------------------------
    performance_summary = V2.build_performance_summary(
        results=results,
        benchmark_name="equal_weight",
    )

    alpha_beta_summary = V2.build_alpha_beta_summary(
        results=results,
        benchmark_names=["equal_weight", "gold_only"],
    )

    cost_summary = V2.build_cost_summary_table(
        results=results,
        settings=settings,
    )

    # ------------------------------------------------------------
    # Same scenario logic as V2, controlled by V3 flag.
    # ------------------------------------------------------------
    scenario_summary = None

    if RUN_SCENARIO_TESTS:
        print("\nRunning V3 model scenario tests using V2 scenario engine.")
        scenario_summary = V2.run_model_scenarios(
            model_weights=model_weights,
            market_data=market_data,
        )

    # ------------------------------------------------------------
    # Score history is optional for now.
    # ------------------------------------------------------------
    scores_history = try_build_scores_history()

    # ------------------------------------------------------------
    # Save V3 canonical outputs.
    # ------------------------------------------------------------
    output_paths = save_v3_outputs(
        results=results,
        market_data=market_data,
        model_weights=model_weights,
        scores_history=scores_history,
        performance_summary=performance_summary,
        alpha_beta_summary=alpha_beta_summary,
        cost_summary=cost_summary,
        scenario_summary=scenario_summary,
    )

    # ------------------------------------------------------------
    # Optional sanity check against existing V2 output.
    # ------------------------------------------------------------
    v2_comparison = compare_against_existing_v2(performance_summary)

    # ------------------------------------------------------------
    # Generate diagnostics package.
    # ------------------------------------------------------------
    run_diagnostics_if_available(
        results=results,
        market_data=market_data,
        model_weights=model_weights,
        scores_history=scores_history,
    )

    run_html_report_if_available()
    run_risk_metrics_if_available()

    # ------------------------------------------------------------
    # Console output.
    # ------------------------------------------------------------
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
        "average_daily_turnover",
        "annualised_turnover",
        "total_transaction_cost_drag",
        "annualised_transaction_cost_drag",
        "average_exposure",
        "average_cash",
        "alpha_annualised",
        "beta",
        "information_ratio",
    ]

    cols = [col for col in cols if col in performance_summary.columns]

    print("\nBacktest V3 complete.")
    print(f"Saved V3 outputs to: {OUTPUT_DIR}")

    print("\nKey output files:")
    for name, path in output_paths.items():
        print(f"  {name}: {path}")

    print("\nPerformance summary:")
    print(performance_summary[cols].to_string(index=False))

    print("\nCost summary:")
    print(cost_summary.to_string(index=False))

    if not v2_comparison.empty:
        print("\nV2 vs V3 model comparison:")
        print(v2_comparison.to_string(index=False))

        max_abs_diff = v2_comparison["difference"].abs().max()

        if pd.notna(max_abs_diff) and max_abs_diff < 1e-10:
            print("\nV2/V3 check: PASS. Model results match to numerical precision.")
        else:
            print(
                "\nV2/V3 check: inspect differences. "
                "If settings/strategy changed since the old V2 run, differences may be expected."
            )

    if scenario_summary is not None and not scenario_summary.empty:
        scenario_cols = [
            "scenario",
            "cost_scenario",
            "execution_delay_days",
            "final_equity",
            "cagr",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "total_cost_drag",
            "annualised_turnover",
            "liquidity_capped_trades",
            "average_fill_ratio",
            "average_tracking_error_to_target",
        ]

        scenario_cols = [col for col in scenario_cols if col in scenario_summary.columns]

        print("\nScenario summary:")
        print(scenario_summary[scenario_cols].to_string(index=False))


if __name__ == "__main__":
    main()