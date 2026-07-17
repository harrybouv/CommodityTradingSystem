from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter


# ============================================================
# DIAGNOSTICS CONFIG
# ============================================================

@dataclass
class DiagnosticsConfig:
    """
    Config for diagnostics only.

    This file must not change the strategy. It audits the strategy outputs.
    backtest_V3.py should pass model_curve, weights_history, scores_history,
    asset_returns, benchmark_returns and trade_log into generate_full_diagnostics_report().
    """

    initial_capital: float = 10_000.0
    periods_per_year: int = 12
    rolling_window: int = 12
    short_rolling_window: int = 6
    risk_free_rate: float = 0.0

    # Weight/contribution behaviour
    use_shifted_weights_for_attribution: bool = True
    held_weight_epsilon: float = 1e-6
    cash_column_names: tuple[str, ...] = ("CASH", "cash", "Cash", "cash_weight")

    # Optional score config
    min_score_to_hold: Optional[float] = None
    feature_weights: dict[str, float] = field(default_factory=dict)
    final_score_candidates: tuple[str, ...] = ("final_score", "score", "total_score", "combined_score")

    # Chart appearance
    chart_dpi: int = 160
    chart_facecolor: str = "#0f1117"
    axes_facecolor: str = "#151923"
    text_color: str = "#e8e8e8"
    grid_color: str = "#2c3340"
    accent: str = "#6ee7b7"
    accent_2: str = "#60a5fa"
    accent_3: str = "#f59e0b"
    negative: str = "#fb7185"
    neutral: str = "#a3a3a3"
    cmap: str = "viridis"

    # Red flag thresholds
    top_asset_contribution_warning: float = 0.50
    top_two_asset_contribution_warning: float = 0.75
    strategy_gld_corr_warning: float = 0.75
    avg_held_corr_warning: float = 0.65
    feature_corr_warning: float = 0.85
    annual_turnover_warning: float = 3.0
    cost_drag_pct_gross_warning: float = 0.20
    weak_ic_warning: float = 0.02
    min_effective_positions_warning: float = 2.5


# ============================================================
# GENERAL UTILITIES
# ============================================================

def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _ensure_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    base = _as_path(output_dir)
    dirs = {
        "base": base,
        "data": base / "data",
        "tables": base / "tables",
        "charts": base / "charts",
        "summary": base / "summary",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _find_col(df: pd.DataFrame, candidates: list[str] | tuple[str, ...]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if str(cand).lower() in lower_map:
            return lower_map[str(cand).lower()]
    return None


def _coerce_datetime_index(
    df: pd.DataFrame,
    date_candidates: tuple[str, ...] = ("date", "Date", "datetime", "timestamp"),
) -> pd.DataFrame:
    out = df.copy()
    date_col = _find_col(out, date_candidates)

    if date_col is not None:
        out[date_col] = pd.to_datetime(out[date_col])
        out = out.set_index(date_col)
    elif not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index)
        except Exception as exc:
            raise ValueError("Could not infer a datetime index or date column.") from exc

    out = out.sort_index()
    out.index.name = "date"
    return out


def _safe_to_csv(df: pd.DataFrame | pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df, pd.Series):
        df.to_frame().to_csv(path)
    else:
        df.to_csv(path)


def _save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _clean_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _clean_numeric_series(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna()

def _safe_corr(a: pd.Series, b: pd.Series, method: str = "pearson") -> float:
    """
    Correlation helper that avoids NumPy warnings from constant/all-NaN columns.
    """
    df = pd.concat(
        [
            pd.to_numeric(a, errors="coerce"),
            pd.to_numeric(b, errors="coerce"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) < 3:
        return np.nan

    if df.iloc[:, 0].nunique() <= 1 or df.iloc[:, 1].nunique() <= 1:
        return np.nan

    return float(df.iloc[:, 0].corr(df.iloc[:, 1], method=method))

def _annualise_return(periodic_returns: pd.Series, periods_per_year: int) -> float:
    r = _clean_numeric_series(periodic_returns)
    if r.empty:
        return np.nan

    total = float((1.0 + r).prod())
    if total <= 0:
        return np.nan

    years = len(r) / periods_per_year
    if years <= 0:
        return np.nan

    return total ** (1.0 / years) - 1.0


def _max_drawdown_from_returns(periodic_returns: pd.Series) -> float:
    r = _clean_numeric_series(periodic_returns)
    if r.empty:
        return np.nan

    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def _format_pct_axis(ax) -> None:
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

def _format_pct_x_axis(ax) -> None:
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))

def _format_money_axis(ax) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))


def _setup_plot_style(cfg: DiagnosticsConfig) -> None:
    plt.rcParams.update({
        "figure.facecolor": cfg.chart_facecolor,
        "axes.facecolor": cfg.axes_facecolor,
        "axes.edgecolor": cfg.grid_color,
        "axes.labelcolor": cfg.text_color,
        "xtick.color": cfg.text_color,
        "ytick.color": cfg.text_color,
        "text.color": cfg.text_color,
        "axes.titleweight": "bold",
        "grid.color": cfg.grid_color,
        "grid.alpha": 0.55,
        "legend.facecolor": cfg.axes_facecolor,
        "legend.edgecolor": cfg.grid_color,
        "legend.labelcolor": cfg.text_color,
        "font.size": 10,
    })


def _save_fig(fig: plt.Figure, path: Path, cfg: DiagnosticsConfig) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=cfg.chart_dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return str(path)


def _plot_no_data(path: Path, title: str, message: str, cfg: DiagnosticsConfig) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(title)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color=cfg.text_color,
    )
    ax.set_axis_off()
    return _save_fig(fig, path, cfg)


# ============================================================
# DATA NORMALISATION
# ============================================================

def normalise_returns(data: pd.DataFrame | pd.Series | None, name: str = "return") -> pd.DataFrame:
    """
    Accepts either:
      - wide return dataframe indexed by date
      - long dataframe with date/ticker/return columns
      - series indexed by date

    Returns wide dataframe indexed by date.
    """
    if data is None:
        return pd.DataFrame()

    if isinstance(data, pd.Series):
        out = data.to_frame(name)
        out = _coerce_datetime_index(out)
        return _clean_numeric_df(out)

    df = data.copy()
    if df.empty:
        return pd.DataFrame()

    columns_lower = {str(c).lower(): c for c in df.columns}
    has_ticker = (
        "ticker" in columns_lower
        or "symbol" in columns_lower
        or "asset" in columns_lower
    )
    ret_col = _find_col(
        df,
        ("return", "returns", "asset_return", "period_return", "monthly_return", "ret"),
    )

    if has_ticker and ret_col is not None:
        ticker_col = _find_col(df, ("ticker", "symbol", "asset"))
        date_col = _find_col(df, ("date", "Date", "datetime", "timestamp"))

        if date_col is None:
            raise ValueError("Long returns data needs a date column.")

        df[date_col] = pd.to_datetime(df[date_col])
        wide = df.pivot_table(
            index=date_col,
            columns=ticker_col,
            values=ret_col,
            aggfunc="last",
        )
        wide.index.name = "date"
        return _clean_numeric_df(wide.sort_index())

    out = _coerce_datetime_index(df)
    return _clean_numeric_df(out)


def returns_from_prices(price_data: pd.DataFrame, price_col: Optional[str] = None) -> pd.DataFrame:
    """
    Accepts either:
      - wide price dataframe indexed by date, columns=tickers
      - long dataframe with date/ticker/close columns

    Returns wide return dataframe indexed by date.
    """
    if price_data is None or price_data.empty:
        return pd.DataFrame()

    df = price_data.copy()
    ticker_col = _find_col(df, ("ticker", "symbol", "asset"))
    date_col = _find_col(df, ("date", "Date", "datetime", "timestamp"))
    price_col = price_col or _find_col(
        df,
        ("close", "Close", "adj_close", "Adj Close", "price", "last"),
    )

    if ticker_col and date_col and price_col:
        df[date_col] = pd.to_datetime(df[date_col])
        prices = df.pivot_table(
            index=date_col,
            columns=ticker_col,
            values=price_col,
            aggfunc="last",
        )
        prices = _clean_numeric_df(prices.sort_index())
        return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")

    prices = _coerce_datetime_index(df)
    prices = _clean_numeric_df(prices)
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")


def normalise_weights(weights_history: pd.DataFrame | None, cfg: DiagnosticsConfig) -> pd.DataFrame:
    """
    Accepts either:
      - wide weights indexed by date, columns=tickers
      - long dataframe with date/ticker/weight or target_weight columns

    Returns wide dataframe indexed by date.
    """
    if weights_history is None or weights_history.empty:
        return pd.DataFrame()

    df = weights_history.copy()
    ticker_col = _find_col(df, ("ticker", "symbol", "asset"))
    weight_col = _find_col(
        df,
        ("weight", "target_weight", "executed_weight", "actual_weight", "new_weight", "allocation"),
    )
    date_col = _find_col(df, ("date", "Date", "datetime", "timestamp"))

    if ticker_col and weight_col and date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        wide = df.pivot_table(
            index=date_col,
            columns=ticker_col,
            values=weight_col,
            aggfunc="last",
        )
        wide.index.name = "date"
        return _clean_numeric_df(wide.sort_index()).fillna(0.0)

    out = _coerce_datetime_index(df)
    return _clean_numeric_df(out).fillna(0.0)


def normalise_scores(scores_history: pd.DataFrame | None) -> pd.DataFrame:
    """
    Returns long scores dataframe with date/ticker columns where possible.

    Expected columns:
      date, ticker, final_score, *_score components
    """
    if scores_history is None or scores_history.empty:
        return pd.DataFrame()

    df = scores_history.copy()
    date_col = _find_col(df, ("date", "Date", "datetime", "timestamp"))
    ticker_col = _find_col(df, ("ticker", "symbol", "asset"))

    if date_col is None or ticker_col is None:
        return _coerce_datetime_index(df)

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date", ticker_col: "ticker"})
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    for col in df.columns:
        if col not in ("date", "ticker"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.replace([np.inf, -np.inf], np.nan)


def extract_strategy_returns(model_curve: pd.DataFrame | pd.Series | None) -> pd.Series:
    """
    Extracts strategy returns from model_curve.

    Preferred columns:
      net_return, strategy_return, model_return, return

    Falls back to equity pct_change if equity-like columns exist.
    """
    if model_curve is None:
        return pd.Series(dtype=float, name="strategy")

    if isinstance(model_curve, pd.Series):
        s = model_curve.copy()
        s.index = pd.to_datetime(s.index)
        s.name = "strategy"
        return _clean_numeric_series(s)

    df = _coerce_datetime_index(model_curve.copy())
    ret_col = _find_col(
        df,
        ("net_return", "strategy_return", "model_return", "return", "returns", "monthly_return"),
    )

    if ret_col is not None:
        s = pd.to_numeric(df[ret_col], errors="coerce")
        s.name = "strategy"
        return _clean_numeric_series(s)

    eq_col = _find_col(
        df,
        ("net_equity", "equity", "model_equity", "portfolio_value", "value", "final_equity"),
    )

    if eq_col is not None:
        s = pd.to_numeric(df[eq_col], errors="coerce").pct_change()
        s.name = "strategy"
        return _clean_numeric_series(s)

    return pd.Series(dtype=float, name="strategy")


def extract_benchmark_returns(
    model_curve: pd.DataFrame | None = None,
    benchmark_returns: pd.DataFrame | pd.Series | None = None,
) -> pd.DataFrame:
    """Combines explicit benchmark_returns with benchmark return columns found in model_curve."""
    parts = []

    if benchmark_returns is not None:
        b = normalise_returns(benchmark_returns)
        if not b.empty:
            parts.append(b)

    if model_curve is not None and not isinstance(model_curve, pd.Series):
        df = _coerce_datetime_index(model_curve.copy())
        candidate_cols = []

        for c in df.columns:
            lc = str(c).lower()
            if lc.endswith("_return") and lc not in {
                "net_return",
                "gross_return",
                "strategy_return",
                "model_return",
            }:
                candidate_cols.append(c)

        if candidate_cols:
            found = _clean_numeric_df(df[candidate_cols])
            rename = {
                c: str(c).replace("_return", "").replace("_", " ").title()
                for c in candidate_cols
            }
            found = found.rename(columns=rename)
            parts.append(found)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, axis=1).sort_index()
    out = out.loc[:, ~out.columns.duplicated()]
    return _clean_numeric_df(out)


def get_feature_columns(scores: pd.DataFrame, cfg: DiagnosticsConfig) -> list[str]:
    """
    Main-report feature list.

    Deliberately restricted to production-level model features.
    Do NOT include every overlay/component diagnostic column here.
    Overlay diagnostics should be separate later.
    """
    if scores is None or scores.empty:
        return []

    preferred = [
        "momentum_score",
        "relative_strength_score",
        "trend_score",
        "trend_persistence_score",
        "volatility_score",
        "risk_score",
        "macro_score",
        "commodity_model_score",
    ]

    features = []

    for col in preferred:
        if col not in scores.columns:
            continue

        s = pd.to_numeric(scores[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

        # Need enough data and actual variation.
        if s.notna().sum() < 50:
            continue

        if s.dropna().nunique() <= 1:
            continue

        features.append(col)

    return features


def get_final_score_column(scores: pd.DataFrame, cfg: DiagnosticsConfig) -> Optional[str]:
    if scores is None or scores.empty:
        return None
    return _find_col(scores, cfg.final_score_candidates)


# ============================================================
# PERFORMANCE DIAGNOSTICS
# ============================================================

def calculate_performance_metrics(returns: pd.Series, cfg: DiagnosticsConfig) -> pd.Series:
    r = _clean_numeric_series(returns)
    if r.empty:
        return pd.Series(dtype=float)

    ann_return = _annualise_return(r, cfg.periods_per_year)
    ann_vol = float(r.std(ddof=1) * math.sqrt(cfg.periods_per_year)) if len(r) > 1 else np.nan
    excess_ann_return = ann_return - cfg.risk_free_rate if pd.notna(ann_return) else np.nan
    sharpe = excess_ann_return / ann_vol if ann_vol and ann_vol > 0 else np.nan

    downside = r[r < 0]
    downside_vol = (
        float(downside.std(ddof=1) * math.sqrt(cfg.periods_per_year))
        if len(downside) > 1
        else np.nan
    )
    sortino = excess_ann_return / downside_vol if downside_vol and downside_vol > 0 else np.nan

    max_dd = _max_drawdown_from_returns(r)
    calmar = ann_return / abs(max_dd) if pd.notna(max_dd) and max_dd < 0 else np.nan

    return pd.Series({
        "periods": len(r),
        "start": r.index.min(),
        "end": r.index.max(),
        "total_return": float((1.0 + r).prod() - 1.0),
        "cagr": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": float((r > 0).mean()),
        "avg_period_return": float(r.mean()),
        "best_period": float(r.max()),
        "worst_period": float(r.min()),
    })


def calculate_benchmark_comparison(
    strategy_returns: pd.Series,
    benchmark_returns: pd.DataFrame,
    cfg: DiagnosticsConfig,
) -> pd.DataFrame:
    rows = []
    all_returns = pd.concat(
        [strategy_returns.rename("Strategy"), benchmark_returns],
        axis=1,
    ).dropna(how="all")

    for col in all_returns.columns:
        metrics = calculate_performance_metrics(all_returns[col], cfg)
        metrics.name = col
        rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)

    if "Strategy" in table.index:
        strategy = table.loc["Strategy"]
        for metric in ["cagr", "sharpe", "sortino", "max_drawdown", "calmar", "ann_vol"]:
            table[f"excess_{metric}_vs_strategy"] = table[metric] - strategy.get(metric, np.nan)

    return table


def calculate_rolling_metrics(strategy_returns: pd.Series, cfg: DiagnosticsConfig) -> pd.DataFrame:
    r = _clean_numeric_series(strategy_returns)
    if r.empty:
        return pd.DataFrame()

    w = cfg.rolling_window
    sw = cfg.short_rolling_window

    out = pd.DataFrame(index=r.index)
    out[f"rolling_return_{sw}p"] = (1.0 + r).rolling(sw).apply(np.prod, raw=True) - 1.0
    out[f"rolling_return_{w}p"] = (1.0 + r).rolling(w).apply(np.prod, raw=True) - 1.0
    out["rolling_ann_vol"] = r.rolling(w).std() * math.sqrt(cfg.periods_per_year)
    out["rolling_ann_return"] = (
        (1.0 + r).rolling(w).apply(np.prod, raw=True) ** (cfg.periods_per_year / w)
        - 1.0
    )
    out["rolling_sharpe"] = (
        (out["rolling_ann_return"] - cfg.risk_free_rate)
        / out["rolling_ann_vol"]
    )

    def rolling_mdd(x: np.ndarray) -> float:
        eq = np.cumprod(1.0 + x)
        dd = eq / np.maximum.accumulate(eq) - 1.0
        return float(np.min(dd))

    out["rolling_max_drawdown"] = r.rolling(w).apply(rolling_mdd, raw=True)
    return out.replace([np.inf, -np.inf], np.nan)


def calculate_monthly_return_table(strategy_returns: pd.Series) -> pd.DataFrame:
    r = _clean_numeric_series(strategy_returns)
    if r.empty:
        return pd.DataFrame()

    df = r.to_frame("return")
    df["year"] = df.index.year
    df["month"] = df.index.strftime("%b")

    month_order = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    table = df.pivot_table(index="year", columns="month", values="return", aggfunc="sum")
    table = table.reindex(columns=month_order)
    table["Year"] = r.groupby(r.index.year).apply(lambda x: (1.0 + x).prod() - 1.0)
    return table


def calculate_drawdown_periods(strategy_returns: pd.Series, top_n: int = 10) -> pd.DataFrame:
    r = _clean_numeric_series(strategy_returns)
    if r.empty:
        return pd.DataFrame()

    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    in_dd = dd < 0

    periods = []
    start = None
    trough_date = None
    trough_dd = 0.0

    for date, is_dd in in_dd.items():
        if is_dd and start is None:
            start = date
            trough_date = date
            trough_dd = float(dd.loc[date])
        elif is_dd and start is not None:
            if dd.loc[date] < trough_dd:
                trough_dd = float(dd.loc[date])
                trough_date = date
        elif not is_dd and start is not None:
            periods.append({
                "start": start,
                "trough": trough_date,
                "recovery": date,
                "max_drawdown": trough_dd,
                "duration_periods": len(dd.loc[start:date]),
            })
            start = None
            trough_date = None
            trough_dd = 0.0

    if start is not None:
        periods.append({
            "start": start,
            "trough": trough_date,
            "recovery": pd.NaT,
            "max_drawdown": trough_dd,
            "duration_periods": len(dd.loc[start:]),
        })

    out = pd.DataFrame(periods)
    if out.empty:
        return out

    return out.sort_values("max_drawdown").head(top_n).reset_index(drop=True)


# ============================================================
# ATTRIBUTION DIAGNOSTICS
# ============================================================

def align_returns_and_weights(
    asset_returns: pd.DataFrame,
    weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if asset_returns.empty or weights.empty:
        return pd.DataFrame(), pd.DataFrame()

    common_cols = [c for c in asset_returns.columns if c in weights.columns]
    if not common_cols:
        return pd.DataFrame(), pd.DataFrame()

    idx = asset_returns.index.union(weights.index).sort_values()
    r = asset_returns.reindex(idx)[common_cols]
    w = weights.reindex(idx)[common_cols].ffill().fillna(0.0)

    common_idx = r.dropna(how="all").index.intersection(w.index)
    return r.loc[common_idx], w.loc[common_idx]


def calculate_ticker_attribution(
    strategy_returns: pd.Series,
    asset_returns: pd.DataFrame,
    weights: pd.DataFrame,
    cfg: DiagnosticsConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    r, w = align_returns_and_weights(asset_returns, weights)
    if r.empty or w.empty:
        return pd.DataFrame(), pd.DataFrame()

    w_for_returns = w.shift(1).fillna(0.0) if cfg.use_shifted_weights_for_attribution else w
    contribution = (w_for_returns * r).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    held = w_for_returns > cfg.held_weight_epsilon

    total_strategy_return = float(strategy_returns.reindex(contribution.index).fillna(0.0).sum())

    rows = []
    for ticker in contribution.columns:
        c = contribution[ticker]
        asset_r = r[ticker]
        h = held[ticker]

        selected_returns = asset_r[h].dropna()
        rejected_returns = asset_r[~h].dropna()
        total_c = float(c.sum())

        rows.append({
            "ticker": ticker,
            "average_weight": float(w[ticker].mean()),
            "max_weight": float(w[ticker].max()),
            "months_held": int(h.sum()),
            "pct_months_held": float(h.mean()) if len(h) else np.nan,
            "total_contribution": total_c,
            "contribution_share_of_strategy_sum": (
                total_c / total_strategy_return
                if abs(total_strategy_return) > 1e-12
                else np.nan
            ),
            "avg_monthly_contribution": float(c.mean()),
            "best_month_contribution": float(c.max()),
            "worst_month_contribution": float(c.min()),
            "hit_rate_when_held": (
                float((selected_returns > 0).mean())
                if len(selected_returns)
                else np.nan
            ),
            "avg_return_when_held": (
                float(selected_returns.mean())
                if len(selected_returns)
                else np.nan
            ),
            "avg_return_when_not_held": (
                float(rejected_returns.mean())
                if len(rejected_returns)
                else np.nan
            ),
            "selected_minus_rejected_return": (
                float(selected_returns.mean() - rejected_returns.mean())
                if len(selected_returns) and len(rejected_returns)
                else np.nan
            ),
            "vol_contribution_proxy": float(c.std(ddof=1)) if len(c) > 1 else np.nan,
        })

    table = pd.DataFrame(rows).sort_values(
        "total_contribution",
        ascending=False,
    ).reset_index(drop=True)

    contribution.index.name = "date"
    return table, contribution


def calculate_return_concentration(ticker_attribution: pd.DataFrame) -> pd.Series:
    if ticker_attribution.empty or "total_contribution" not in ticker_attribution.columns:
        return pd.Series(dtype=float)

    contrib = ticker_attribution.set_index("ticker")["total_contribution"].dropna()
    positive_total = contrib[contrib > 0].sum()
    abs_total = contrib.abs().sum()

    if abs_total == 0:
        return pd.Series(dtype=float)

    sorted_abs = contrib.abs().sort_values(ascending=False)
    weights = sorted_abs / abs_total

    return pd.Series({
        "top_asset_abs_contribution_share": float(weights.iloc[0]) if len(weights) else np.nan,
        "top_two_asset_abs_contribution_share": float(weights.iloc[:2].sum()) if len(weights) else np.nan,
        "return_concentration_hhi": float((weights ** 2).sum()),
        "positive_contribution_total": float(positive_total),
        "absolute_contribution_total": float(abs_total),
    })


def calculate_exposure_diagnostics(
    weights: pd.DataFrame,
    cfg: DiagnosticsConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    if weights.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    cash_cols = [c for c in weights.columns if c in cfg.cash_column_names]
    asset_cols = [c for c in weights.columns if c not in cash_cols]

    asset_w = weights[asset_cols].clip(lower=0.0)
    exposure = asset_w.sum(axis=1)
    cash = weights[cash_cols].sum(axis=1) if cash_cols else 1.0 - exposure

    norm_w = asset_w.div(asset_w.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    effective_n = 1.0 / (norm_w.pow(2).sum(axis=1).replace(0.0, np.nan))

    history = pd.DataFrame({
        "gross_exposure": exposure,
        "cash_weight": cash,
        "number_positions": (asset_w > cfg.held_weight_epsilon).sum(axis=1),
        "effective_number_positions": effective_n,
        "max_single_weight": asset_w.max(axis=1),
    })

    summary = pd.Series({
        "average_exposure": float(exposure.mean()),
        "average_cash": float(cash.mean()),
        "max_cash": float(cash.max()),
        "min_cash": float(cash.min()),
        "months_fully_invested": int((cash <= 0.01).sum()),
        "months_cash_heavy_25pct_plus": int((cash >= 0.25).sum()),
        "average_number_positions": float(history["number_positions"].mean()),
        "average_effective_number_positions": float(effective_n.mean()),
        "minimum_effective_number_positions": float(effective_n.min()),
    })

    return history, summary


# ============================================================
# FEATURE / SCORE DIAGNOSTICS
# ============================================================

def build_forward_return_panel(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Builds date/ticker/forward_return panel.

    Uses next-period returns, so scores at date t are tested against
    asset returns at t+1. This avoids same-period lookahead.
    """
    if asset_returns is None or asset_returns.empty:
        return pd.DataFrame(columns=["date", "ticker", "forward_return"])

    fwd = asset_returns.copy()
    fwd.index = pd.to_datetime(fwd.index)
    fwd = fwd.sort_index()

    # Next-period return for each asset.
    fwd = fwd.shift(-1)

    # Pandas newer versions do not allow stack(dropna=False).
    # We drop NaNs after stacking anyway, so plain stack() is correct.
    out = fwd.stack().rename("forward_return").reset_index()
    out.columns = ["date", "ticker", "forward_return"]

    out["date"] = pd.to_datetime(out["date"])
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["forward_return"] = pd.to_numeric(out["forward_return"], errors="coerce")

    return (
        out.dropna(subset=["forward_return"])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def calculate_feature_attribution(
    scores: pd.DataFrame,
    asset_returns: pd.DataFrame,
    weights: pd.DataFrame,
    cfg: DiagnosticsConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = normalise_scores(scores)

    if scores.empty or "ticker" not in scores.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    features = get_feature_columns(scores, cfg)
    if not features:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    fwd = build_forward_return_panel(asset_returns)
    merged = scores.merge(fwd, on=["date", "ticker"], how="left")

    if weights is not None and not weights.empty:
        weights_long = weights.stack().rename("weight").reset_index()
        weights_long.columns = ["date", "ticker", "weight"]
        merged = merged.merge(weights_long, on=["date", "ticker"], how="left")
    else:
        merged["weight"] = np.nan

    rows = []
    ic_rows = []

    for feature in features:
        x = pd.to_numeric(merged[feature], errors="coerce")
        y = pd.to_numeric(merged["forward_return"], errors="coerce")

        valid = merged.loc[
            x.notna() & y.notna(),
            ["date", "ticker", feature, "forward_return", "weight"],
        ].copy()

        if valid.empty:
            continue

        held = valid["weight"].fillna(0.0) > cfg.held_weight_epsilon

        weighted_contribution = (
            valid[feature] * cfg.feature_weights.get(feature, 1.0)
            if cfg.feature_weights
            else valid[feature]
        )

        per_date_ic = []
        per_date_rank_ic = []

        for date, g in valid.groupby("date"):
            if len(g) < 3:
                continue

            ic = _safe_corr(g[feature], g["forward_return"])
            rank_ic = _safe_corr(
                g[feature].rank(),
                g["forward_return"].rank(),
            )

            if pd.notna(ic):
                per_date_ic.append(ic)
                ic_rows.append({
                    "date": date,
                    "feature": feature,
                    "ic": ic,
                    "rank_ic": rank_ic,
                })

            if pd.notna(rank_ic):
                per_date_rank_ic.append(rank_ic)

        decile_spread = np.nan
        top_decile_ret = np.nan
        bottom_decile_ret = np.nan

        try:
            valid["decile"] = pd.qcut(
                valid[feature],
                q=10,
                labels=False,
                duplicates="drop",
            ) + 1

            decile_means = valid.groupby("decile")["forward_return"].mean()

            if len(decile_means) >= 2:
                bottom_decile_ret = float(decile_means.iloc[0])
                top_decile_ret = float(decile_means.iloc[-1])
                decile_spread = top_decile_ret - bottom_decile_ret

        except Exception:
            pass

        rows.append({
            "feature": feature,
            "average_score": float(valid[feature].mean()),
            "average_score_when_held": (
                float(valid.loc[held, feature].mean())
                if held.any()
                else np.nan
            ),
            "average_score_when_not_held": (
                float(valid.loc[~held, feature].mean())
                if (~held).any()
                else np.nan
            ),
            "average_weighted_contribution_to_score": float(weighted_contribution.mean()),
                "full_sample_corr_with_forward_return": _safe_corr(
                valid[feature],
                valid["forward_return"],
            ),
            "average_cross_sectional_ic": (
                float(np.nanmean(per_date_ic))
                if per_date_ic
                else np.nan
            ),
            "average_cross_sectional_rank_ic": (
                float(np.nanmean(per_date_rank_ic))
                if per_date_rank_ic
                else np.nan
            ),
            "top_decile_forward_return": top_decile_ret,
            "bottom_decile_forward_return": bottom_decile_ret,
            "decile_spread": decile_spread,
            "observations": int(len(valid)),
        })

    feature_table = pd.DataFrame(rows)

    if not feature_table.empty:
        feature_table = feature_table.sort_values(
            "average_cross_sectional_rank_ic",
            ascending=False,
        )

    ic_table = pd.DataFrame(ic_rows)

    decile_rows = []
    for feature in features:
        valid = merged[[feature, "forward_return"]].dropna().copy()

        if len(valid) < 20:
            continue

        try:
            valid["decile"] = pd.qcut(
                valid[feature],
                q=10,
                labels=False,
                duplicates="drop",
            ) + 1
        except Exception:
            continue

        dec = valid.groupby("decile")["forward_return"].agg(
            ["mean", "median", "count"],
        ).reset_index()

        dec["feature"] = feature
        decile_rows.append(dec)

    decile_table = (
        pd.concat(decile_rows, ignore_index=True)
        if decile_rows
        else pd.DataFrame()
    )

    return feature_table.reset_index(drop=True), ic_table, decile_table


def calculate_score_threshold_diagnostics(
    scores: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cfg: DiagnosticsConfig,
) -> pd.DataFrame:
    scores = normalise_scores(scores)
    final_col = get_final_score_column(scores, cfg)

    if scores.empty or final_col is None or "ticker" not in scores.columns:
        return pd.DataFrame()

    fwd = build_forward_return_panel(asset_returns)

    merged = scores[["date", "ticker", final_col]].merge(
        fwd,
        on=["date", "ticker"],
        how="left",
    ).dropna()

    if merged.empty:
        return pd.DataFrame()

    rows = []

    try:
        merged["score_bucket"] = pd.qcut(
            merged[final_col],
            q=10,
            labels=False,
            duplicates="drop",
        ) + 1

        bucketed = merged.groupby("score_bucket")["forward_return"].agg(
            ["mean", "median", "count"],
        )

        bucketed["hit_rate"] = merged.groupby("score_bucket")["forward_return"].apply(
            lambda x: (x > 0).mean()
        )

        bucketed = bucketed.reset_index().rename(columns={
            "mean": "avg_forward_return",
            "median": "median_forward_return",
        })

        rows.append(bucketed)

    except Exception:
        pass

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    if cfg.min_score_to_hold is not None:
        above = merged[merged[final_col] >= cfg.min_score_to_hold]
        below = merged[merged[final_col] < cfg.min_score_to_hold]

        summary = pd.DataFrame([
            {
                "score_bucket": "above_threshold",
                "avg_forward_return": above["forward_return"].mean(),
                "median_forward_return": above["forward_return"].median(),
                "count": len(above),
                "hit_rate": (above["forward_return"] > 0).mean() if len(above) else np.nan,
            },
            {
                "score_bucket": "below_threshold",
                "avg_forward_return": below["forward_return"].mean(),
                "median_forward_return": below["forward_return"].median(),
                "count": len(below),
                "hit_rate": (below["forward_return"] > 0).mean() if len(below) else np.nan,
            },
        ])

        out = pd.concat([out, summary], ignore_index=True)

    return out


# ============================================================
# REGIME DIAGNOSTICS
# ============================================================

def _rolling_average_pairwise_corr(returns: pd.DataFrame, window: int) -> pd.Series:
    vals = []
    dates = []

    for i in range(len(returns)):
        if i + 1 < window:
            vals.append(np.nan)
            dates.append(returns.index[i])
            continue

        chunk = returns.iloc[i + 1 - window:i + 1]
        corr = chunk.corr()

        if corr.shape[0] < 2:
            vals.append(np.nan)
        else:
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
            vals.append(float(upper.mean()) if len(upper) else np.nan)

        dates.append(returns.index[i])

    return pd.Series(vals, index=dates, name="rolling_avg_pairwise_corr")


def classify_price_regimes(
    asset_returns: pd.DataFrame,
    weights: pd.DataFrame,
    cfg: DiagnosticsConfig,
) -> pd.DataFrame:
    if asset_returns.empty:
        return pd.DataFrame()

    r = asset_returns.copy().sort_index()
    r = r.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Keep rows where at least one asset return exists.
    r = r.dropna(how="all")

    if r.empty:
        return pd.DataFrame()

    ew = r.mean(axis=1, skipna=True)

    broad_mom = (1.0 + ew.fillna(0.0)).rolling(126, min_periods=60).apply(np.prod, raw=True) - 1.0
    broad_vol = ew.rolling(252, min_periods=60).std() * math.sqrt(cfg.periods_per_year)
    dispersion = r.std(axis=1, skipna=True)
    avg_corr = _rolling_average_pairwise_corr(r.fillna(0.0), min(252, max(20, cfg.rolling_window)))

    regimes = pd.DataFrame(index=r.index)

    # Broad direction regimes
    regimes["broad_commodity_uptrend"] = broad_mom > 0
    regimes["broad_commodity_downtrend"] = broad_mom <= 0

    # Volatility regimes
    vol_median = broad_vol.expanding(min_periods=60).median()
    regimes["high_volatility"] = broad_vol > vol_median
    regimes["low_volatility"] = broad_vol <= vol_median

    # Dispersion regimes
    disp_median = dispersion.expanding(min_periods=60).median()
    regimes["high_dispersion"] = dispersion > disp_median
    regimes["low_dispersion"] = dispersion <= disp_median

    # Correlation regimes
    corr_median = avg_corr.expanding(min_periods=60).median()
    regimes["high_correlation"] = avg_corr > corr_median
    regimes["low_correlation"] = avg_corr <= corr_median

    # Asset leadership regimes.
    # Important: only call idxmax on rows where at least one non-NaN momentum exists.
    six_month_asset_mom = (
        (1.0 + r.fillna(0.0))
        .rolling(126, min_periods=60)
        .apply(np.prod, raw=True)
        - 1.0
    )

    valid_leader_rows = six_month_asset_mom.notna().any(axis=1)
    leader = pd.Series(index=six_month_asset_mom.index, dtype="object")

    if valid_leader_rows.any():
        leader.loc[valid_leader_rows] = six_month_asset_mom.loc[valid_leader_rows].idxmax(axis=1)

    regimes["gold_led"] = leader.eq("GLD")
    regimes["silver_led"] = leader.eq("SLV")
    regimes["energy_led"] = leader.isin(["USO", "UNG"])
    regimes["copper_led"] = leader.eq("CPER")
    regimes["agriculture_led"] = leader.eq("DBA")

    # Portfolio state regimes
    if weights is not None and not weights.empty:
        w = weights.copy()
        w.index = pd.to_datetime(w.index)
        w = w.sort_index().reindex(r.index).ffill().fillna(0.0)

        cash_cols = [c for c in w.columns if c in cfg.cash_column_names]
        asset_cols = [c for c in w.columns if c not in cash_cols]

        exposure = w[asset_cols].clip(lower=0.0).sum(axis=1)

        if cash_cols:
            cash = w[cash_cols].sum(axis=1)
        else:
            cash = 1.0 - exposure

        regimes["cash_heavy"] = cash >= 0.25
        regimes["fully_invested"] = cash <= 0.05

    # Replace early warm-up NaNs with False. This is correct:
    # unknown regime should not be treated as active.
    regimes = regimes.fillna(False).astype(bool)

    return regimes

def calculate_regime_summary(
    strategy_returns: pd.Series,
    regimes: pd.DataFrame,
    weights: pd.DataFrame,
    contribution: pd.DataFrame,
    cfg: DiagnosticsConfig,
) -> pd.DataFrame:
    if regimes.empty:
        return pd.DataFrame()

    r = _clean_numeric_series(strategy_returns).reindex(regimes.index)
    rows = []

    for regime in regimes.columns:
        mask = regimes[regime].astype(bool)
        reg_r = r[mask].dropna()

        if reg_r.empty:
            continue

        row = calculate_performance_metrics(reg_r, cfg).to_dict()
        row["regime"] = regime
        row["periods"] = int(mask.sum())
        row["avg_period_return"] = float(reg_r.mean())
        row["total_return_sum"] = float(reg_r.sum())

        if weights is not None and not weights.empty:
            w = weights.reindex(regimes.index).ffill().fillna(0.0)
            cash_cols = [c for c in w.columns if c in cfg.cash_column_names]
            exposure = w[[c for c in w.columns if c not in cash_cols]].clip(lower=0.0).sum(axis=1)
            cash = w[cash_cols].sum(axis=1) if cash_cols else 1.0 - exposure

            row["average_exposure"] = float(exposure[mask].mean())
            row["average_cash"] = float(cash[mask].mean())

        if contribution is not None and not contribution.empty:
            c = contribution.reindex(regimes.index).fillna(0.0)
            reg_contrib = c[mask].sum().sort_values(ascending=False)

            row["best_ticker"] = reg_contrib.index[0] if len(reg_contrib) else None
            row["best_ticker_contribution"] = (
                float(reg_contrib.iloc[0])
                if len(reg_contrib)
                else np.nan
            )
            row["worst_ticker"] = reg_contrib.index[-1] if len(reg_contrib) else None
            row["worst_ticker_contribution"] = (
                float(reg_contrib.iloc[-1])
                if len(reg_contrib)
                else np.nan
            )

        rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    return out.sort_values("avg_period_return", ascending=False).reset_index(drop=True)


# ============================================================
# CORRELATION DIAGNOSTICS
# ============================================================

def calculate_correlation_summary(
    strategy_returns: pd.Series,
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    weights: pd.DataFrame,
    cfg: DiagnosticsConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    r_strategy = _clean_numeric_series(strategy_returns)

    if not asset_returns.empty:
        asset_clean = asset_returns.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

        good_asset_cols = []
        for col in asset_clean.columns:
            s = asset_clean[col].dropna()
            if len(s) >= 50 and s.nunique() > 1:
                good_asset_cols.append(col)

        asset_corr = asset_clean[good_asset_cols].corr() if len(good_asset_cols) >= 2 else pd.DataFrame()

        for ticker in asset_returns.columns:
            aligned = pd.concat([r_strategy, asset_returns[ticker]], axis=1).dropna()

            if len(aligned) > 2:
                corr = _safe_corr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                factor_var = aligned.iloc[:, 1].var()
                beta = (
                    aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / factor_var
                    if pd.notna(factor_var) and factor_var > 0
                    else np.nan
            )
            else:
                corr = np.nan
                beta = np.nan

            rows.append({
                "factor": ticker,
                "type": "asset",
                "strategy_corr": corr,
                "strategy_beta": beta,
            })

    else:
        asset_corr = pd.DataFrame()

    if not benchmark_returns.empty:
        for bench in benchmark_returns.columns:
            aligned = pd.concat([r_strategy, benchmark_returns[bench]], axis=1).dropna()

            if len(aligned) > 2:
                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                beta = (
                    aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var()
                    if aligned.iloc[:, 1].var() > 0
                    else np.nan
                )
            else:
                corr = np.nan
                beta = np.nan

            rows.append({
                "factor": bench,
                "type": "benchmark",
                "strategy_corr": corr,
                "strategy_beta": beta,
            })

    corr_summary = pd.DataFrame(rows)

    held_corr = pd.DataFrame()

    if not asset_returns.empty and not weights.empty:
        r, w = align_returns_and_weights(asset_returns, weights)

        vals = []
        for date in r.index:
            lookback = r.loc[:date].tail(cfg.rolling_window)

            held_assets = [
                c for c in w.columns
                if c in lookback.columns and w.loc[date, c] > cfg.held_weight_epsilon
            ]

            if len(held_assets) < 2 or len(lookback) < 3:
                vals.append({
                    "date": date,
                    "avg_held_pairwise_corr": np.nan,
                    "held_assets": len(held_assets),
                })
                continue

            corr = lookback[held_assets].corr()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()

            vals.append({
                "date": date,
                "avg_held_pairwise_corr": float(upper.mean()) if len(upper) else np.nan,
                "held_assets": len(held_assets),
            })

        held_corr = pd.DataFrame(vals).set_index("date")

    return corr_summary, asset_corr, held_corr


def calculate_feature_correlation_matrix(
    scores: pd.DataFrame,
    cfg: DiagnosticsConfig,
) -> pd.DataFrame:
    scores = normalise_scores(scores)
    features = get_feature_columns(scores, cfg)

    if not features:
        return pd.DataFrame()

    data = scores[features].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)

    # Drop useless columns before correlation.
    good_cols = []
    for col in data.columns:
        s = data[col].dropna()
        if len(s) >= 50 and s.nunique() > 1:
            good_cols.append(col)

    if len(good_cols) < 2:
        return pd.DataFrame()

    return data[good_cols].corr()


# ============================================================
# TURNOVER / COST DIAGNOSTICS
# ============================================================

def calculate_turnover_from_weights(weights: pd.DataFrame) -> pd.Series:
    if weights.empty:
        return pd.Series(dtype=float, name="turnover")

    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    turnover.name = "turnover"
    return turnover


def calculate_turnover_cost_summary(
    weights: pd.DataFrame,
    trade_log: pd.DataFrame | None,
    model_curve: pd.DataFrame | None,
    strategy_returns: pd.Series,
    cfg: DiagnosticsConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    turnover = calculate_turnover_from_weights(weights)

    history = pd.DataFrame(index=turnover.index)
    history["turnover"] = turnover

    # ------------------------------------------------------------
    # Cost extraction from trade log
    # ------------------------------------------------------------
    if trade_log is not None and not trade_log.empty:
        tl = trade_log.copy()
        date_col = _find_col(tl, ("date", "Date", "datetime", "timestamp"))

        if date_col is not None:
            tl[date_col] = pd.to_datetime(tl[date_col])

            preferred_cost_cols = [
                "total_trade_cost",
                "total_cost",
                "transaction_cost",
                "trading_cost",
            ]

            cost_cols = [c for c in preferred_cost_cols if c in tl.columns]

            if cost_cols:
                numeric_costs = tl[cost_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

                if "total_trade_cost" in numeric_costs.columns:
                    cost_by_date = numeric_costs["total_trade_cost"].groupby(tl[date_col]).sum()
                elif "total_cost" in numeric_costs.columns:
                    cost_by_date = numeric_costs["total_cost"].groupby(tl[date_col]).sum()
                elif "transaction_cost" in numeric_costs.columns:
                    cost_by_date = numeric_costs["transaction_cost"].groupby(tl[date_col]).sum()
                else:
                    cost_by_date = numeric_costs.groupby(tl[date_col]).sum().sum(axis=1)

                history["total_cost"] = cost_by_date.reindex(history.index).fillna(0.0)

    # ------------------------------------------------------------
    # Fallback from model curve cost drag
    # ------------------------------------------------------------
    if "total_cost" not in history.columns and model_curve is not None and not isinstance(model_curve, pd.Series):
        mc = _coerce_datetime_index(model_curve.copy())

        drag_col = _find_col(
            mc,
            (
                "total_transaction_cost_drag",
                "transaction_cost",
                "cost_drag",
                "trading_cost_drag",
            ),
        )

        if drag_col is not None:
            drag = pd.to_numeric(mc[drag_col], errors="coerce").fillna(0.0)
            equity_col = _find_col(mc, ("equity", "net_equity", "model_equity", "portfolio_value"))

            if equity_col is not None:
                equity = pd.to_numeric(mc[equity_col], errors="coerce").ffill()
                history["total_cost"] = (drag * equity).reindex(history.index).fillna(0.0)
            else:
                history["total_cost"] = drag.reindex(history.index).fillna(0.0)

    if "total_cost" not in history.columns:
        history["total_cost"] = 0.0

    history["cumulative_cost"] = history["total_cost"].cumsum()

    total_cost = float(history["total_cost"].sum())
    avg_monthly_turnover = float(history["turnover"].mean()) if len(history) else np.nan
    annualised_turnover = avg_monthly_turnover * cfg.periods_per_year if pd.notna(avg_monthly_turnover) else np.nan

    # Clean cost ratios. These are interpretable.
    total_cost_pct_initial_capital = (
        total_cost / cfg.initial_capital
        if cfg.initial_capital and cfg.initial_capital > 0
        else np.nan
    )

    final_equity = np.nan
    gross_net_gap = np.nan

    if model_curve is not None and not isinstance(model_curve, pd.Series) and not model_curve.empty:
        mc = _coerce_datetime_index(model_curve.copy())

        equity_col = _find_col(mc, ("equity", "net_equity", "model_equity", "portfolio_value"))
        if equity_col is not None:
            final_equity = pd.to_numeric(mc[equity_col], errors="coerce").dropna().iloc[-1]

        gross_col = _find_col(mc, ("gross_equity", "equity_gross", "raw_equity", "pre_cost_equity"))
        if gross_col is not None and equity_col is not None:
            gross_final = pd.to_numeric(mc[gross_col], errors="coerce").dropna().iloc[-1]
            net_final = pd.to_numeric(mc[equity_col], errors="coerce").dropna().iloc[-1]
            if gross_final > 0:
                gross_net_gap = (gross_final - net_final) / gross_final

    total_cost_pct_final_equity = (
        total_cost / final_equity
        if pd.notna(final_equity) and final_equity > 0
        else np.nan
    )

    summary = pd.Series({
        "average_period_turnover": avg_monthly_turnover,
        "annualised_turnover": annualised_turnover,
        "max_period_turnover": float(history["turnover"].max()) if len(history) else np.nan,
        "total_cost": total_cost,
        "average_period_cost": float(history["total_cost"].mean()) if len(history) else np.nan,
        "total_cost_pct_initial_capital": total_cost_pct_initial_capital,
        "total_cost_pct_final_equity": total_cost_pct_final_equity,
        "gross_net_equity_gap": gross_net_gap,
    })

    return history, summary

# ============================================================
# DECISION DIAGNOSTICS
# ============================================================

def build_enriched_decision_log(
    scores: pd.DataFrame,
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cfg: DiagnosticsConfig,
) -> pd.DataFrame:
    scores = normalise_scores(scores)

    if scores.empty or weights.empty or "ticker" not in scores.columns:
        return pd.DataFrame()

    features = get_feature_columns(scores, cfg)
    final_col = get_final_score_column(scores, cfg)

    weights_long = weights.stack().rename("new_weight").reset_index()
    weights_long.columns = ["date", "ticker", "new_weight"]
    weights_long = weights_long.sort_values(["ticker", "date"])
    weights_long["old_weight"] = weights_long.groupby("ticker")["new_weight"].shift(1).fillna(0.0)
    weights_long["trade_weight"] = weights_long["new_weight"] - weights_long["old_weight"]

    fwd = build_forward_return_panel(asset_returns)

    out = scores.merge(weights_long, on=["date", "ticker"], how="left")
    out = out.merge(fwd, on=["date", "ticker"], how="left")

    out["new_weight"] = out["new_weight"].fillna(0.0)
    out["old_weight"] = out["old_weight"].fillna(0.0)
    out["trade_weight"] = out["trade_weight"].fillna(0.0)
    out["next_period_contribution"] = out["new_weight"] * out["forward_return"]

    if features:
        feature_values = out[features].copy()
        out["main_positive_feature"] = feature_values.idxmax(axis=1)
        out["main_negative_feature"] = feature_values.idxmin(axis=1)
    else:
        out["main_positive_feature"] = None
        out["main_negative_feature"] = None

    trade_eps = max(cfg.held_weight_epsilon, 1e-4)

    conditions = [
        (out["old_weight"] <= trade_eps) & (out["new_weight"] > trade_eps),
        (out["old_weight"] > trade_eps) & (out["new_weight"] <= trade_eps),
        (out["old_weight"] > trade_eps) & (out["new_weight"] > out["old_weight"] + trade_eps),
        (out["old_weight"] > trade_eps) & (out["new_weight"] < out["old_weight"] - trade_eps),
    ]

    choices = ["entry", "exit", "increase", "decrease"]
    out["decision_type"] = np.select(conditions, choices, default="hold/no_trade")

    if final_col is not None:
        out = out.sort_values(["date", final_col], ascending=[True, False])

    return out


def identify_best_worst_decisions(
    decision_log: pd.DataFrame,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if decision_log.empty or "next_period_contribution" not in decision_log.columns:
        return pd.DataFrame(), pd.DataFrame()

    candidates = decision_log[
        decision_log["decision_type"].isin(["entry", "resize", "hold/no_trade"])
    ].dropna(subset=["next_period_contribution"])

    best = candidates.sort_values("next_period_contribution", ascending=False).head(top_n)
    worst = candidates.sort_values("next_period_contribution", ascending=True).head(top_n)

    return best, worst


# ============================================================
# RED FLAGS
# ============================================================

def generate_red_flags(
    ticker_attribution: pd.DataFrame,
    concentration: pd.Series,
    corr_summary: pd.DataFrame,
    held_corr: pd.DataFrame,
    feature_table: pd.DataFrame,
    feature_corr: pd.DataFrame,
    exposure_summary: pd.Series,
    cost_summary: pd.Series,
    cfg: DiagnosticsConfig,
) -> pd.DataFrame:
    flags = []

    def add_flag(
        category: str,
        severity: str,
        message: str,
        value: Any = None,
        threshold: Any = None,
    ):
        flags.append({
            "category": category,
            "severity": severity,
            "message": message,
            "value": value,
            "threshold": threshold,
        })

    if not concentration.empty:
        top = concentration.get("top_asset_abs_contribution_share", np.nan)
        top2 = concentration.get("top_two_asset_abs_contribution_share", np.nan)

        if pd.notna(top) and top > cfg.top_asset_contribution_warning:
            add_flag(
                "Concentration",
                "High",
                "Top asset contributes too much of absolute return contribution.",
                top,
                cfg.top_asset_contribution_warning,
            )

        if pd.notna(top2) and top2 > cfg.top_two_asset_contribution_warning:
            add_flag(
                "Concentration",
                "High",
                "Top two assets dominate absolute return contribution.",
                top2,
                cfg.top_two_asset_contribution_warning,
            )

    if not corr_summary.empty:
        gld_rows = corr_summary[corr_summary["factor"].astype(str).str.upper().eq("GLD")]

        if not gld_rows.empty:
            gld_corr = float(gld_rows.iloc[0]["strategy_corr"])

            if pd.notna(gld_corr) and abs(gld_corr) > cfg.strategy_gld_corr_warning:
                add_flag(
                    "Correlation",
                    "Medium",
                    "Strategy is highly correlated with GLD.",
                    gld_corr,
                    cfg.strategy_gld_corr_warning,
                )

    if not held_corr.empty and "avg_held_pairwise_corr" in held_corr.columns:
        avg_held = held_corr["avg_held_pairwise_corr"].mean()

        if pd.notna(avg_held) and avg_held > cfg.avg_held_corr_warning:
            add_flag(
                "Correlation",
                "Medium",
                "Held assets are highly correlated on average.",
                avg_held,
                cfg.avg_held_corr_warning,
            )

    if not feature_table.empty and "average_cross_sectional_rank_ic" in feature_table.columns:
        weak_features = feature_table[
            feature_table["average_cross_sectional_rank_ic"].abs() < cfg.weak_ic_warning
        ]

        if len(weak_features) > 0:
            add_flag(
                "Features",
                "Medium",
                f"{len(weak_features)} feature(s) have weak average rank IC.",
                len(weak_features),
                cfg.weak_ic_warning,
            )

    if not feature_corr.empty:
        upper = feature_corr.abs().where(
            np.triu(np.ones(feature_corr.shape), k=1).astype(bool)
        ).stack()

        high = upper[upper > cfg.feature_corr_warning]

        if len(high) > 0:
            add_flag(
                "Features",
                "Medium",
                "Some score components are highly correlated; signal set may be redundant.",
                float(high.max()),
                cfg.feature_corr_warning,
            )

    if not exposure_summary.empty:
        min_eff = exposure_summary.get("minimum_effective_number_positions", np.nan)
        avg_eff = exposure_summary.get("average_effective_number_positions", np.nan)

        if pd.notna(avg_eff) and avg_eff < cfg.min_effective_positions_warning:
            add_flag(
                "Diversification",
                "Medium",
                "Average effective number of positions is low.",
                avg_eff,
                cfg.min_effective_positions_warning,
            )

        if pd.notna(min_eff) and min_eff < 1.5:
            add_flag(
                "Diversification",
                "Low",
                "At times the portfolio is close to a single-asset bet.",
                min_eff,
                1.5,
            )

    if not cost_summary.empty:
        turnover = cost_summary.get("annualised_turnover", np.nan)
        cost_pct_final = cost_summary.get("total_cost_pct_final_equity", np.nan)
        gross_net_gap = cost_summary.get("gross_net_equity_gap", np.nan)

        if pd.notna(turnover) and turnover > cfg.annual_turnover_warning:
            add_flag(
                "Implementation",
                "Medium",
                "Annualised turnover is high for a monthly system.",
                turnover,
                cfg.annual_turnover_warning,
            )

        if pd.notna(cost_pct_final) and cost_pct_final > 0.05:
            add_flag(
                "Implementation",
                "Medium",
                "Cumulative trading costs are more than 5% of final equity.",
                cost_pct_final,
                0.05,
            )

        if pd.notna(gross_net_gap) and gross_net_gap > cfg.cost_drag_pct_gross_warning:
            add_flag(
                "Implementation",
                "High",
                "Gross-to-net equity gap is large.",
                gross_net_gap,
                cfg.cost_drag_pct_gross_warning,
            )

    if not flags:
        add_flag(
            "Overall",
            "Info",
            "No major automatic diagnostic red flags triggered. Still inspect charts manually.",
            None,
            None,
        )

    return pd.DataFrame(flags)


# ============================================================
# CHART FUNCTIONS
# ============================================================

def plot_equity_vs_benchmarks(
    strategy_returns: pd.Series,
    benchmark_returns: pd.DataFrame,
    output_path: Path,
    cfg: DiagnosticsConfig,
) -> str:
    r = pd.concat(
        [strategy_returns.rename("Strategy"), benchmark_returns],
        axis=1,
    ).dropna(how="all")

    if r.empty:
        return _plot_no_data(
            output_path,
            "Equity vs benchmarks",
            "No strategy/benchmark returns available.",
            cfg,
        )

    equity = cfg.initial_capital * (1.0 + r.fillna(0.0)).cumprod()

    fig, ax = plt.subplots(figsize=(12, 6))

    for col in equity.columns:
        lw = 2.6 if col == "Strategy" else 1.6
        alpha = 1.0 if col == "Strategy" else 0.75
        ax.plot(equity.index, equity[col], label=col, linewidth=lw, alpha=alpha)

    ax.set_title("Equity curve vs benchmarks")
    ax.set_ylabel("Portfolio value")
    _format_money_axis(ax)
    ax.grid(True)
    ax.legend(loc="best")

    return _save_fig(fig, output_path, cfg)


def plot_drawdown_vs_benchmarks(
    strategy_returns: pd.Series,
    benchmark_returns: pd.DataFrame,
    output_path: Path,
    cfg: DiagnosticsConfig,
) -> str:
    r = pd.concat(
        [strategy_returns.rename("Strategy"), benchmark_returns],
        axis=1,
    ).dropna(how="all")

    if r.empty:
        return _plot_no_data(
            output_path,
            "Drawdown vs benchmarks",
            "No strategy/benchmark returns available.",
            cfg,
        )

    equity = (1.0 + r.fillna(0.0)).cumprod()
    dd = equity / equity.cummax() - 1.0

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for col in dd.columns:
        lw = 2.4 if col == "Strategy" else 1.4
        ax.plot(dd.index, dd[col], label=col, linewidth=lw, alpha=0.9)

    ax.set_title("Drawdown vs benchmarks")
    ax.set_ylabel("Drawdown")
    _format_pct_axis(ax)
    ax.grid(True)
    ax.legend(loc="lower left")

    return _save_fig(fig, output_path, cfg)


def plot_rolling_metrics(
    rolling_metrics: pd.DataFrame,
    output_path: Path,
    cfg: DiagnosticsConfig,
) -> str:
    if rolling_metrics.empty:
        return _plot_no_data(
            output_path,
            "Rolling metrics",
            "No rolling metrics available.",
            cfg,
        )

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for col in [
        "rolling_sharpe",
        "rolling_ann_return",
        "rolling_ann_vol",
        "rolling_max_drawdown",
    ]:
        if col in rolling_metrics.columns:
            ax.plot(
                rolling_metrics.index,
                rolling_metrics[col],
                label=col.replace("_", " ").title(),
                linewidth=1.8,
            )

    ax.axhline(0, color=cfg.grid_color, linewidth=1)
    ax.set_title("Rolling performance diagnostics")
    ax.grid(True)
    ax.legend(loc="best")

    return _save_fig(fig, output_path, cfg)


def plot_monthly_return_heatmap(
    monthly_table: pd.DataFrame,
    output_path: Path,
    cfg: DiagnosticsConfig,
) -> str:
    if monthly_table.empty:
        return _plot_no_data(
            output_path,
            "Monthly return heatmap",
            "No monthly return table available.",
            cfg,
        )

    month_cols = [c for c in monthly_table.columns if c != "Year"]
    data = monthly_table[month_cols].astype(float)

    abs_max = max(abs(np.nanmin(data.values)), abs(np.nanmax(data.values)))
    if not np.isfinite(abs_max) or abs_max == 0:
        abs_max = 0.01

    fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * len(data))))
    im = ax.imshow(
        data.values,
        aspect="auto",
        cmap="RdYlGn",
        vmin=-abs_max,
        vmax=abs_max,
    )

    ax.set_title("Monthly return heatmap")
    ax.set_xticks(range(len(month_cols)))
    ax.set_xticklabels(month_cols)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index.astype(str))

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iat[i, j]
            if pd.notna(val):
                ax.text(
                    j,
                    i,
                    f"{val:.1%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                )

    return _save_fig(fig, output_path, cfg)


def plot_allocation_heatmap(
    weights: pd.DataFrame,
    output_path: Path,
    cfg: DiagnosticsConfig,
) -> str:
    if weights.empty:
        return _plot_no_data(
            output_path,
            "Allocation heatmap",
            "No weights available.",
            cfg,
        )

    cash_cols = [c for c in weights.columns if c in cfg.cash_column_names]
    cols = [c for c in weights.columns if c not in cash_cols] + cash_cols

    data = weights[cols].T

    fig, ax = plt.subplots(figsize=(13, max(4, 0.6 * len(cols))))
    im = ax.imshow(
        data.values,
        aspect="auto",
        cmap=cfg.cmap,
        vmin=0,
        vmax=max(0.01, np.nanmax(data.values)),
    )

    ax.set_title("Portfolio allocation heatmap")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)

    n_dates = len(data.columns)
    step = max(1, n_dates // 8)
    tick_positions = list(range(0, n_dates, step))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [data.columns[i].strftime("%Y-%m") for i in tick_positions],
        rotation=45,
        ha="right",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    return _save_fig(fig, output_path, cfg)


def plot_ticker_contribution(
    ticker_attribution: pd.DataFrame,
    output_path: Path,
    cfg: DiagnosticsConfig,
) -> str:
    if ticker_attribution.empty or "total_contribution" not in ticker_attribution.columns:
        return _plot_no_data(
            output_path,
            "Ticker contribution",
            "No ticker attribution available.",
            cfg,
        )

    data = ticker_attribution.copy()
    data["total_contribution"] = pd.to_numeric(data["total_contribution"], errors="coerce")
    data = data.dropna(subset=["total_contribution"])
    data = data.sort_values("total_contribution", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [cfg.negative if x < 0 else cfg.accent for x in data["total_contribution"]]

    ax.barh(data["ticker"], data["total_contribution"], color=colors)
    ax.axvline(0, color=cfg.grid_color, linewidth=1)
    ax.set_title("Total return contribution by ticker")
    ax.set_xlabel("Contribution to strategy return")
    ax.set_ylabel("Ticker")
    _format_pct_x_axis(ax)
    ax.grid(True, axis="x")

    return _save_fig(fig, output_path, cfg)


def plot_average_weights(
    ticker_attribution: pd.DataFrame,
    output_path: Path,
    cfg: DiagnosticsConfig,
) -> str:
    if ticker_attribution.empty or "average_weight" not in ticker_attribution.columns:
        return _plot_no_data(
            output_path,
            "Average weights",
            "No average weight data available.",
            cfg,
        )

    data = ticker_attribution.copy()
    data["average_weight"] = pd.to_numeric(data["average_weight"], errors="coerce")
    data = data.dropna(subset=["average_weight"])
    data = data.sort_values("average_weight", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(data["ticker"], data["average_weight"], color=cfg.accent_2)
    ax.set_title("Average allocation by ticker")
    ax.set_xlabel("Average portfolio weight")
    ax.set_ylabel("Ticker")
    _format_pct_x_axis(ax)
    ax.grid(True, axis="x")

    return _save_fig(fig, output_path, cfg)


def plot_selected_vs_rejected(
    ticker_attribution: pd.DataFrame,
    output_path: Path,
    cfg: DiagnosticsConfig,
) -> str:
    needed = {"ticker", "avg_return_when_held", "avg_return_when_not_held"}

    if ticker_attribution.empty or not needed.issubset(ticker_attribution.columns):
        return _plot_no_data(
            output_path,
            "Selected vs rejected returns",
            "No selected/rejected return data available.",
            cfg,
        )

    data = ticker_attribution.set_index("ticker")[
        ["avg_return_when_held", "avg_return_when_not_held"]
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(data.index))
    width = 0.36

    ax.bar(
        x - width / 2,
        data["avg_return_when_held"],
        width,
        label="When held",
        color=cfg.accent,
    )
    ax.bar(
        x + width / 2,
        data["avg_return_when_not_held"],
        width,
        label="When not held",
        color=cfg.accent_3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(data.index)
    ax.set_title("Average forward return when selected vs rejected")
    ax.set_ylabel("Average period return")
    _format_pct_axis(ax)
    ax.grid(True, axis="y")
    ax.legend()

    return _save_fig(fig, output_path, cfg)


def plot_feature_ic(
    feature_table: pd.DataFrame,
    output_path: Path,
    cfg: DiagnosticsConfig,
) -> str:
    col = "average_cross_sectional_rank_ic"

    if feature_table.empty or col not in feature_table.columns:
        return _plot_no_data(
            output_path,
            "Feature rank IC",
            "No feature IC data available.",
            cfg,
        )

    data = feature_table.sort_values(col, ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [cfg.negative if x < 0 else cfg.accent for x in data[col]]

    ax.barh(data["feature"], data[col], color=colors)
    ax.axvline(0, color=cfg.grid_color, linewidth=1)
    ax.set_title("Average cross-sectional rank IC by feature")
    ax.set_xlabel("Rank IC vs next-period return")
    ax.grid(True, axis="x")

    return _save_fig(fig, output_path, cfg)


def plot_feature_deciles(
    decile_table: pd.DataFrame,
    output_path: Path,
    cfg: DiagnosticsConfig,
) -> str:
    if decile_table.empty or not {"feature", "decile", "mean"}.issubset(decile_table.columns):
        return _plot_no_data(
            output_path,
            "Feature decile returns",
            "No feature decile data available.",
            cfg,
        )

    data = decile_table.copy()
    data["decile"] = pd.to_numeric(data["decile"], errors="coerce")
    data["mean"] = pd.to_numeric(data["mean"], errors="coerce")
    data = data.dropna(subset=["feature", "decile", "mean"])

    if data.empty:
        return _plot_no_data(
            output_path,
            "Feature decile returns",
            "No valid feature decile data available.",
            cfg,
        )

    # Restrict to the main production features only.
    allowed = [
        "momentum_score",
        "relative_strength_score",
        "trend_score",
        "trend_persistence_score",
        "volatility_score",
        "risk_score",
        "macro_score",
        "commodity_model_score",
    ]

    data = data[data["feature"].isin(allowed)]

    if data.empty:
        return _plot_no_data(
            output_path,
            "Feature decile returns",
            "No core production feature decile data available.",
            cfg,
        )

    fig, ax = plt.subplots(figsize=(11, 6))

    for feature, g in data.groupby("feature"):
        g = g.sort_values("decile")
        ax.plot(
            g["decile"],
            g["mean"],
            marker="o",
            linewidth=2,
            label=feature.replace("_score", "").replace("_", " ").title(),
        )

    ax.axhline(0, color=cfg.grid_color, linewidth=1)
    ax.set_title("Forward return by core feature decile")
    ax.set_xlabel("Feature decile, low to high")
    ax.set_ylabel("Average next-period return")
    _format_pct_axis(ax)
    ax.grid(True)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=True,
    )

    return _save_fig(fig, output_path, cfg)


def plot_score_bucket_returns(
    score_table: pd.DataFrame,
    output_path: Path,
    cfg: DiagnosticsConfig,
) -> str:
    if score_table.empty or "avg_forward_return" not in score_table.columns:
        return _plot_no_data(
            output_path,
            "Score bucket forward returns",
            "No score bucket data available.",
            cfg,
        )

    numeric = score_table[
        pd.to_numeric(score_table["score_bucket"], errors="coerce").notna()
    ].copy()

    if numeric.empty:
        return _plot_no_data(
            output_path,
            "Score bucket forward returns",
            "No numeric score buckets available.",
            cfg,
        )

    numeric["score_bucket"] = numeric["score_bucket"].astype(int)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [cfg.negative if x < 0 else cfg.accent for x in numeric["avg_forward_return"]]

    ax.bar(numeric["score_bucket"].astype(str), numeric["avg_forward_return"], color=colors)
    ax.axhline(0, color=cfg.grid_color, linewidth=1)
    ax.set_title("Average next-period return by final score bucket")
    ax.set_xlabel("Final score bucket, low to high")
    ax.set_ylabel("Average next-period return")
    _format_pct_axis(ax)
    ax.grid(True, axis="y")

    return _save_fig(fig, output_path, cfg)


def plot_regime_summary(
    regime_summary: pd.DataFrame,
    output_path: Path,
    cfg: DiagnosticsConfig,
) -> str:
    if regime_summary.empty or "avg_period_return" not in regime_summary.columns:
        return _plot_no_data(
            output_path,
            "Regime contribution",
            "No regime summary available.",
            cfg,
        )

    data = regime_summary.copy()
    data["avg_period_return"] = pd.to_numeric(data["avg_period_return"], errors="coerce")
    data = data.dropna(subset=["avg_period_return"])
    data = data.sort_values("avg_period_return", ascending=True)

    fig, ax = plt.subplots(figsize=(11, max(5.5, 0.35 * len(data))))
    colors = [cfg.negative if x < 0 else cfg.accent for x in data["avg_period_return"]]

    ax.barh(data["regime"], data["avg_period_return"], color=colors)
    ax.axvline(0, color=cfg.grid_color, linewidth=1)
    ax.set_title("Average strategy return by price-derived regime")
    ax.set_xlabel("Average period return")
    ax.set_ylabel("Regime")
    _format_pct_x_axis(ax)
    ax.grid(True, axis="x")

    return _save_fig(fig, output_path, cfg)

def plot_correlation_heatmap(
    corr: pd.DataFrame,
    output_path: Path,
    cfg: DiagnosticsConfig,
    title: str,
) -> str:
    if corr is None or corr.empty:
        return _plot_no_data(
            output_path,
            title,
            "No correlation data available.",
            cfg,
        )

    data = corr.copy()
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna(axis=0, how="all").dropna(axis=1, how="all")

    common = [c for c in data.index if c in data.columns]
    if common:
        data = data.loc[common, common]

    if data.empty:
        return _plot_no_data(
            output_path,
            title,
            "No valid numeric correlation data available.",
            cfg,
        )

    n = len(data.columns)

    # Prevent unreadable monster heatmaps.
    if n > 14:
        avg_abs = (
            data.abs()
            .where(~np.eye(n, dtype=bool))
            .mean(axis=1)
            .sort_values(ascending=False)
        )
        keep = avg_abs.head(14).index.tolist()
        data = data.loc[keep, keep]
        n = len(data.columns)
        title = title + " — top 14 shown"

    fig_size = max(7, min(13, 0.75 * n + 3))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(data.values, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_title(title)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)

    fig.colorbar(im, ax=ax)

    if n <= 10:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data.iat[i, j]
                if pd.notna(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                    )

    return _save_fig(fig, output_path, cfg)



def plot_held_correlation(
    held_corr: pd.DataFrame,
    output_path: Path,
    cfg: DiagnosticsConfig,
) -> str:
    if held_corr.empty or "avg_held_pairwise_corr" not in held_corr.columns:
        return _plot_no_data(
            output_path,
            "Rolling held-asset correlation",
            "No held correlation data available.",
            cfg,
        )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        held_corr.index,
        held_corr["avg_held_pairwise_corr"],
        color=cfg.accent,
        linewidth=2,
    )

    ax.axhline(0, color=cfg.grid_color, linewidth=1)
    ax.set_title("Rolling average pairwise correlation of held assets")
    ax.set_ylabel("Correlation")
    ax.grid(True)

    return _save_fig(fig, output_path, cfg)


def plot_exposure_cash(
    exposure_history: pd.DataFrame,
    output_path: Path,
    cfg: DiagnosticsConfig,
) -> str:
    if exposure_history.empty:
        return _plot_no_data(
            output_path,
            "Exposure and cash",
            "No exposure data available.",
            cfg,
        )

    fig, ax = plt.subplots(figsize=(12, 5.5))

    if "gross_exposure" in exposure_history.columns:
        ax.plot(
            exposure_history.index,
            exposure_history["gross_exposure"],
            label="Gross exposure",
            color=cfg.accent,
            linewidth=2,
        )

    if "cash_weight" in exposure_history.columns:
        ax.plot(
            exposure_history.index,
            exposure_history["cash_weight"],
            label="Cash",
            color=cfg.accent_3,
            linewidth=2,
        )

    if "effective_number_positions" in exposure_history.columns:
        ax2 = ax.twinx()
        ax2.plot(
            exposure_history.index,
            exposure_history["effective_number_positions"],
            label="Effective positions",
            color=cfg.accent_2,
            linewidth=1.5,
            alpha=0.8,
        )
        ax2.set_ylabel("Effective positions", color=cfg.text_color)
        ax2.tick_params(colors=cfg.text_color)

    ax.set_title("Exposure, cash and effective diversification")
    ax.set_ylabel("Weight")
    _format_pct_axis(ax)
    ax.grid(True)
    ax.legend(loc="upper left")

    return _save_fig(fig, output_path, cfg)


def plot_turnover_and_costs(
    turnover_history: pd.DataFrame,
    output_path: Path,
    cfg: DiagnosticsConfig,
) -> str:
    if turnover_history.empty:
        return _plot_no_data(
            output_path,
            "Turnover and costs",
            "No turnover/cost data available.",
            cfg,
        )

    fig, ax = plt.subplots(figsize=(12, 5.5))

    if "turnover" in turnover_history.columns:
        ax.bar(
            turnover_history.index,
            turnover_history["turnover"],
            width=20,
            label="Turnover",
            color=cfg.accent_2,
            alpha=0.75,
        )
        ax.set_ylabel("Turnover")
        _format_pct_axis(ax)

    if "cumulative_cost" in turnover_history.columns:
        ax2 = ax.twinx()
        ax2.plot(
            turnover_history.index,
            turnover_history["cumulative_cost"],
            label="Cumulative cost",
            color=cfg.negative,
            linewidth=2,
        )
        ax2.set_ylabel("Cumulative cost/drag", color=cfg.text_color)
        ax2.tick_params(colors=cfg.text_color)

    ax.set_title("Turnover and cumulative cost drag")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left")

    return _save_fig(fig, output_path, cfg)


def plot_gross_vs_net_equity(
    model_curve: pd.DataFrame | None,
    output_path: Path,
    cfg: DiagnosticsConfig,
) -> str:
    if model_curve is None or isinstance(model_curve, pd.Series) or model_curve.empty:
        return _plot_no_data(
            output_path,
            "Gross vs net equity",
            "No model curve available.",
            cfg,
        )

    mc = _coerce_datetime_index(model_curve.copy())

    gross_col = _find_col(mc, ("gross_equity", "equity_gross", "raw_equity", "pre_cost_equity"))
    net_col = _find_col(mc, ("net_equity", "equity", "model_equity", "portfolio_value"))

    # Reconstruct gross equity if V3 only has pre-cost return.
    if gross_col is None:
        pre_cost_col = _find_col(mc, ("pre_cost_return", "gross_return"))
        if pre_cost_col is not None:
            pre_cost_returns = pd.to_numeric(mc[pre_cost_col], errors="coerce").fillna(0.0)
            mc["gross_equity_reconstructed"] = cfg.initial_capital * (1.0 + pre_cost_returns).cumprod()
            gross_col = "gross_equity_reconstructed"

    if net_col is None:
        return _plot_no_data(
            output_path,
            "Gross vs net equity",
            "Need net_equity/equity column.",
            cfg,
        )

    if gross_col is None:
        return _plot_no_data(
            output_path,
            "Gross vs net equity",
            "Need gross_equity or pre_cost_return/gross_return to reconstruct gross equity.",
            cfg,
        )

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(mc.index, mc[gross_col], label="Gross / pre-cost", color=cfg.accent_2, linewidth=2)
    ax.plot(mc.index, mc[net_col], label="Net / after costs", color=cfg.accent, linewidth=2.4)

    ax.set_title("Gross vs net equity after costs")
    ax.set_ylabel("Portfolio value")
    _format_money_axis(ax)
    ax.grid(True)
    ax.legend()

    return _save_fig(fig, output_path, cfg)


def plot_decision_outcomes(
    decision_log: pd.DataFrame,
    output_path: Path,
    cfg: DiagnosticsConfig,
) -> str:
    if (
        decision_log.empty
        or "decision_type" not in decision_log.columns
        or "next_period_contribution" not in decision_log.columns
    ):
        return _plot_no_data(
            output_path,
            "Decision outcomes",
            "No decision outcome data available.",
            cfg,
        )

    actual = decision_log[
        decision_log["decision_type"].isin(["entry", "exit", "increase", "decrease"])
    ].copy()

    actual["next_period_contribution"] = pd.to_numeric(
        actual["next_period_contribution"],
        errors="coerce",
    )

    actual = actual.dropna(subset=["next_period_contribution"])

    if actual.empty:
        return _plot_no_data(
            output_path,
            "Decision outcomes",
            "No actual entry/exit/increase/decrease decisions available.",
            cfg,
        )

    data = (
        actual.groupby("decision_type")["next_period_contribution"]
        .agg(["mean", "count"])
        .sort_values("mean")
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [cfg.negative if x < 0 else cfg.accent for x in data["mean"]]

    ax.barh(data.index, data["mean"], color=colors)
    ax.axvline(0, color=cfg.grid_color, linewidth=1)
    ax.set_title("Average next-period contribution by actual decision type")
    ax.set_xlabel("Average next-period contribution")
    ax.set_ylabel("Decision type")
    _format_pct_x_axis(ax)
    ax.grid(True, axis="x")

    for i, (decision_type, row) in enumerate(data.iterrows()):
        ax.text(
            row["mean"],
            i,
            f" n={int(row['count'])}",
            va="center",
            ha="left" if row["mean"] >= 0 else "right",
            fontsize=8,
            color=cfg.text_color,
        )

    return _save_fig(fig, output_path, cfg)


def plot_parameter_sensitivity_heatmaps(
    sensitivity_results: pd.DataFrame | None,
    output_dir: Path,
    cfg: DiagnosticsConfig,
) -> list[str]:
    """
    Optional.

    Expects a dataframe containing at least two parameter columns and metric columns.

    Common metric columns:
      sharpe, cagr, max_drawdown, calmar, turnover, final_equity

    The runner can create this later; diagnostics.py is ready for it now.
    """
    if sensitivity_results is None or sensitivity_results.empty:
        return []

    df = sensitivity_results.copy()

    metric_candidates = ["sharpe", "cagr", "max_drawdown", "calmar", "turnover", "final_equity"]
    metrics = [m for m in metric_candidates if m in df.columns]

    param_cols = [
        c for c in df.columns
        if c not in metrics and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(param_cols) < 2 or not metrics:
        return []

    x_param, y_param = param_cols[0], param_cols[1]
    paths = []

    for metric in metrics:
        pivot = df.pivot_table(
            index=y_param,
            columns=x_param,
            values=metric,
            aggfunc="mean",
        ).sort_index(ascending=True)

        fig, ax = plt.subplots(figsize=(9, 6.5))
        im = ax.imshow(pivot.values, aspect="auto", cmap=cfg.cmap)

        ax.set_title(f"Parameter sensitivity: {metric}")
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.3g}" for x in pivot.columns], rotation=45, ha="right")

        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{y:.3g}" for y in pivot.index])

        fig.colorbar(im, ax=ax)

        path = output_dir / f"parameter_sensitivity_{metric}.png"
        paths.append(_save_fig(fig, path, cfg))

    return paths


# ============================================================
# MASTER ORCHESTRATION
# ============================================================

def generate_full_diagnostics_report(
    *,
    model_curve: pd.DataFrame | pd.Series,
    weights_history: pd.DataFrame,
    scores_history: pd.DataFrame | None = None,
    asset_returns: pd.DataFrame | None = None,
    benchmark_returns: pd.DataFrame | pd.Series | None = None,
    trade_log: pd.DataFrame | None = None,
    price_data: pd.DataFrame | None = None,
    sensitivity_results: pd.DataFrame | None = None,
    output_dir: str | Path,
    config: DiagnosticsConfig | None = None,
) -> dict[str, Any]:
    """
    Main function called by backtest_V3.py.

    It saves:
      - normalised input data to /data
      - summary diagnostic tables to /tables
      - charts to /charts
      - manifest + red flags to /summary

    It intentionally does not build the final HTML. diagnostics_report.py will consume these files.
    """
    cfg = config or DiagnosticsConfig()
    _setup_plot_style(cfg)
    dirs = _ensure_output_dirs(output_dir)

    # ---------- Normalise inputs ----------
    strategy_returns = extract_strategy_returns(model_curve)
    weights = normalise_weights(weights_history, cfg)
    scores = normalise_scores(scores_history) if scores_history is not None else pd.DataFrame()

    if asset_returns is not None:
        asset_r = normalise_returns(asset_returns)
    elif price_data is not None:
        asset_r = returns_from_prices(price_data)
    else:
        asset_r = pd.DataFrame()

    benches = extract_benchmark_returns(
        model_curve=model_curve if isinstance(model_curve, pd.DataFrame) else None,
        benchmark_returns=benchmark_returns,
    )

    if not strategy_returns.empty:
        strategy_returns = strategy_returns.sort_index()

    if not asset_r.empty:
        asset_r = asset_r.sort_index()

    if not benches.empty:
        benches = benches.sort_index()

    # Save normalised audit data.
    _safe_to_csv(strategy_returns.rename("strategy_return"), dirs["data"] / "strategy_returns.csv")

    if not weights.empty:
        _safe_to_csv(weights, dirs["data"] / "weights_history.csv")

    if not scores.empty:
        _safe_to_csv(scores, dirs["data"] / "scores_history.csv")

    if not asset_r.empty:
        _safe_to_csv(asset_r, dirs["data"] / "asset_returns.csv")

    if not benches.empty:
        _safe_to_csv(benches, dirs["data"] / "benchmark_returns.csv")

    tables: dict[str, str] = {}
    charts: dict[str, str] = {}

    # ---------- Performance ----------
    headline = calculate_performance_metrics(strategy_returns, cfg).to_frame("Strategy").T
    benchmark_comparison = calculate_benchmark_comparison(strategy_returns, benches, cfg)
    rolling = calculate_rolling_metrics(strategy_returns, cfg)
    monthly_table = calculate_monthly_return_table(strategy_returns)
    drawdown_periods = calculate_drawdown_periods(strategy_returns)

    table_map = {
        "headline_summary": headline,
        "benchmark_comparison": benchmark_comparison,
        "rolling_metrics": rolling,
        "monthly_returns": monthly_table,
        "worst_drawdown_periods": drawdown_periods,
    }

    # ---------- Attribution ----------
    ticker_attr, contribution = calculate_ticker_attribution(
        strategy_returns,
        asset_r,
        weights,
        cfg,
    )
    concentration = calculate_return_concentration(ticker_attr)
    exposure_history, exposure_summary = calculate_exposure_diagnostics(weights, cfg)

    table_map.update({
        "ticker_attribution": ticker_attr,
        "ticker_contribution_history": contribution,
        "return_concentration": concentration.to_frame("value"),
        "exposure_history": exposure_history,
        "exposure_summary": exposure_summary.to_frame("value"),
    })

    # ---------- Features / scores ----------
    feature_table, feature_ic_history, feature_deciles = calculate_feature_attribution(
        scores,
        asset_r,
        weights,
        cfg,
    )
    score_threshold = calculate_score_threshold_diagnostics(scores, asset_r, cfg)
    feature_corr = calculate_feature_correlation_matrix(scores, cfg)

    table_map.update({
        "feature_attribution": feature_table,
        "feature_ic_history": feature_ic_history,
        "feature_deciles": feature_deciles,
        "score_threshold_diagnostics": score_threshold,
        "feature_correlation_matrix": feature_corr,
    })

    # ---------- Regimes ----------
    regimes = classify_price_regimes(asset_r, weights, cfg)
    regime_summary = calculate_regime_summary(
        strategy_returns,
        regimes,
        weights,
        contribution,
        cfg,
    )

    table_map.update({
        "price_regimes": regimes,
        "regime_summary": regime_summary,
    })

    # ---------- Correlations ----------
    corr_summary, asset_corr, held_corr = calculate_correlation_summary(
        strategy_returns,
        asset_r,
        benches,
        weights,
        cfg,
    )

    table_map.update({
        "correlation_summary": corr_summary,
        "asset_correlation_matrix": asset_corr,
        "held_asset_correlation_history": held_corr,
    })

    # ---------- Costs / turnover ----------
    turnover_history, cost_summary = calculate_turnover_cost_summary(
        weights,
        trade_log,
        model_curve if isinstance(model_curve, pd.DataFrame) else None,
        strategy_returns,
        cfg,
    )

    table_map.update({
        "turnover_cost_history": turnover_history,
        "turnover_cost_summary": cost_summary.to_frame("value"),
    })

    # ---------- Decisions ----------
    decision_log = build_enriched_decision_log(scores, weights, asset_r, cfg)
    best_decisions, worst_decisions = identify_best_worst_decisions(decision_log)

    table_map.update({
        "decision_log_enriched": decision_log,
        "best_decisions": best_decisions,
        "worst_decisions": worst_decisions,
    })

    # ---------- Optional parameter sensitivity ----------
    if sensitivity_results is not None and not sensitivity_results.empty:
        table_map["parameter_sensitivity_summary"] = sensitivity_results

    # Save tables.
    for name, table in table_map.items():
        if table is None:
            continue

        path = dirs["tables"] / f"{name}.csv"

        try:
            _safe_to_csv(table, path)
            tables[name] = str(path)
        except Exception as exc:
            warnings.warn(f"Could not save table {name}: {exc}")

    # ---------- Charts ----------
    charts["equity_vs_benchmarks"] = plot_equity_vs_benchmarks(
        strategy_returns,
        benches,
        dirs["charts"] / "01_equity_vs_benchmarks.png",
        cfg,
    )

    charts["drawdown_vs_benchmarks"] = plot_drawdown_vs_benchmarks(
        strategy_returns,
        benches,
        dirs["charts"] / "02_drawdown_vs_benchmarks.png",
        cfg,
    )

    charts["rolling_metrics"] = plot_rolling_metrics(
        rolling,
        dirs["charts"] / "03_rolling_metrics.png",
        cfg,
    )

    charts["monthly_return_heatmap"] = plot_monthly_return_heatmap(
        monthly_table,
        dirs["charts"] / "04_monthly_return_heatmap.png",
        cfg,
    )

    charts["allocation_heatmap"] = plot_allocation_heatmap(
        weights,
        dirs["charts"] / "05_allocation_heatmap.png",
        cfg,
    )

    charts["ticker_contribution"] = plot_ticker_contribution(
        ticker_attr,
        dirs["charts"] / "06_ticker_contribution.png",
        cfg,
    )

    charts["average_weights"] = plot_average_weights(
        ticker_attr,
        dirs["charts"] / "07_average_weights.png",
        cfg,
    )

    charts["selected_vs_rejected"] = plot_selected_vs_rejected(
        ticker_attr,
        dirs["charts"] / "08_selected_vs_rejected_returns.png",
        cfg,
    )

    charts["feature_ic"] = plot_feature_ic(
        feature_table,
        dirs["charts"] / "09_feature_ic.png",
        cfg,
    )

    charts["feature_deciles"] = plot_feature_deciles(
        feature_deciles,
        dirs["charts"] / "10_feature_decile_returns.png",
        cfg,
    )

    charts["score_bucket_returns"] = plot_score_bucket_returns(
        score_threshold,
        dirs["charts"] / "11_score_bucket_returns.png",
        cfg,
    )

    charts["regime_summary"] = plot_regime_summary(
        regime_summary,
        dirs["charts"] / "12_regime_summary.png",
        cfg,
    )

    charts["asset_correlation_heatmap"] = plot_correlation_heatmap(
        asset_corr,
        dirs["charts"] / "13_asset_correlation_heatmap.png",
        cfg,
        "Asset return correlation matrix",
    )

    charts["feature_correlation_heatmap"] = plot_correlation_heatmap(
        feature_corr,
        dirs["charts"] / "14_feature_correlation_heatmap.png",
        cfg,
        "Feature correlation matrix",
    )

    charts["held_correlation"] = plot_held_correlation(
        held_corr,
        dirs["charts"] / "15_held_asset_correlation.png",
        cfg,
    )

    charts["exposure_cash"] = plot_exposure_cash(
        exposure_history,
        dirs["charts"] / "16_exposure_cash.png",
        cfg,
    )

    charts["turnover_costs"] = plot_turnover_and_costs(
        turnover_history,
        dirs["charts"] / "17_turnover_costs.png",
        cfg,
    )

    charts["gross_vs_net"] = plot_gross_vs_net_equity(
        model_curve if isinstance(model_curve, pd.DataFrame) else None,
        dirs["charts"] / "18_gross_vs_net_equity.png",
        cfg,
    )

    charts["decision_outcomes"] = plot_decision_outcomes(
        decision_log,
        dirs["charts"] / "19_decision_outcomes.png",
        cfg,
    )

    sensitivity_chart_paths = plot_parameter_sensitivity_heatmaps(
        sensitivity_results,
        dirs["charts"],
        cfg,
    )

    for i, p in enumerate(sensitivity_chart_paths, start=1):
        charts[f"parameter_sensitivity_{i}"] = p

    # ---------- Red flags ----------
    red_flags = generate_red_flags(
        ticker_attribution=ticker_attr,
        concentration=concentration,
        corr_summary=corr_summary,
        held_corr=held_corr,
        feature_table=feature_table,
        feature_corr=feature_corr,
        exposure_summary=exposure_summary,
        cost_summary=cost_summary,
        cfg=cfg,
    )

    red_flag_path = dirs["summary"] / "red_flags.csv"
    _safe_to_csv(red_flags, red_flag_path)
    tables["red_flags"] = str(red_flag_path)

    # ---------- Human-readable summary ----------
    summary_lines = []

    if not headline.empty:
        h = headline.iloc[0]

        summary_lines.append("V3 DIAGNOSTICS SUMMARY")
        summary_lines.append("======================")
        summary_lines.append(f"CAGR: {h.get('cagr', np.nan):.2%}")
        summary_lines.append(f"Sharpe: {h.get('sharpe', np.nan):.2f}")
        summary_lines.append(f"Sortino: {h.get('sortino', np.nan):.2f}")
        summary_lines.append(f"Max drawdown: {h.get('max_drawdown', np.nan):.2%}")
        summary_lines.append(f"Win rate: {h.get('win_rate', np.nan):.2%}")

    if not concentration.empty:
        summary_lines.append("")
        summary_lines.append("CONCENTRATION")
        summary_lines.append(
            f"Top asset abs contribution share: "
            f"{concentration.get('top_asset_abs_contribution_share', np.nan):.2%}"
        )
        summary_lines.append(
            f"Top 2 asset abs contribution share: "
            f"{concentration.get('top_two_asset_abs_contribution_share', np.nan):.2%}"
        )

    if not exposure_summary.empty:
        summary_lines.append("")
        summary_lines.append("EXPOSURE")
        summary_lines.append(
            f"Average exposure: {exposure_summary.get('average_exposure', np.nan):.2%}"
        )
        summary_lines.append(
            f"Average cash: {exposure_summary.get('average_cash', np.nan):.2%}"
        )
        summary_lines.append(
            f"Average effective positions: "
            f"{exposure_summary.get('average_effective_number_positions', np.nan):.2f}"
        )

    if not red_flags.empty:
        summary_lines.append("")
        summary_lines.append("RED FLAGS")

        for _, row in red_flags.iterrows():
            summary_lines.append(f"[{row['severity']}] {row['category']}: {row['message']}")

    summary_path = dirs["summary"] / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    manifest = {
        "base_dir": str(dirs["base"]),
        "data_dir": str(dirs["data"]),
        "tables_dir": str(dirs["tables"]),
        "charts_dir": str(dirs["charts"]),
        "summary_dir": str(dirs["summary"]),
        "tables": tables,
        "charts": charts,
        "summary_text": str(summary_path),
        "red_flags": str(red_flag_path),
    }

    manifest_path = dirs["summary"] / "diagnostics_manifest.json"
    _save_json(manifest, manifest_path)
    manifest["manifest"] = str(manifest_path)

    return manifest


# ============================================================
# CONVENIENCE: LOAD FROM V3 OUTPUT FOLDER
# ============================================================

def generate_from_v3_folder(
    v3_output_dir: str | Path,
    diagnostics_output_dir: str | Path | None = None,
    config: DiagnosticsConfig | None = None,
) -> dict[str, Any]:
    """
    Convenience function once backtest_V3.py saves canonical CSVs.

    Expected files if available:
      model_curve_V3.csv
      weights_history_V3.csv
      scores_history_V3.csv
      asset_returns_V3.csv
      benchmark_returns_V3.csv
      trade_log_V3.csv
      parameter_sensitivity_summary.csv
    """
    base = _as_path(v3_output_dir)
    out = _as_path(diagnostics_output_dir) if diagnostics_output_dir else base / "diagnostics"

    def read_optional(name: str) -> pd.DataFrame | None:
        path = base / name
        if not path.exists():
            return None
        return pd.read_csv(path)

    model_curve = read_optional("model_curve_V3.csv")
    weights = read_optional("weights_history_V3.csv")

    if model_curve is None or weights is None:
        raise FileNotFoundError(
            "Need at least model_curve_V3.csv and weights_history_V3.csv in the V3 output folder."
        )

    return generate_full_diagnostics_report(
        model_curve=model_curve,
        weights_history=weights,
        scores_history=read_optional("scores_history_V3.csv"),
        asset_returns=read_optional("asset_returns_V3.csv"),
        benchmark_returns=read_optional("benchmark_returns_V3.csv"),
        trade_log=read_optional("trade_log_V3.csv"),
        sensitivity_results=read_optional("parameter_sensitivity_summary.csv"),
        output_dir=out,
        config=config,
    )