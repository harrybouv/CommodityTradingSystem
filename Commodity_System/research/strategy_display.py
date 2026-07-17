from __future__ import annotations

import html
import json
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter


# ============================================================
# PATH SETUP
# ============================================================

THIS_FILE = Path(__file__).resolve()
RESEARCH_DIR = THIS_FILE.parent
COMMODITY_ROOT = RESEARCH_DIR.parent
PROJECT_ROOT = COMMODITY_ROOT.parent

for path in [PROJECT_ROOT, COMMODITY_ROOT, RESEARCH_DIR]:
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


# ============================================================
# CONFIG
# ============================================================

try:
    from Commodity_System import config as CFG
except Exception:
    try:
        import config as CFG
    except Exception:
        CFG = None


def cfg(name: str, default: Any) -> Any:
    if CFG is None:
        return default
    return getattr(CFG, name, default)


RESULTS_DIR = Path(cfg("RESULTS_DIR", COMMODITY_ROOT / "results"))
INITIAL_CAPITAL = float(cfg("INITIAL_CAPITAL", 10_000.0))
TRADING_DAYS_PER_YEAR = int(cfg("TRADING_DAYS_PER_YEAR", 252))

BACKTEST_V3_DIR = RESULTS_DIR / "backtest_V3"
DIAGNOSTICS_DIR = BACKTEST_V3_DIR / "diagnostics"
DIAG_TABLES_DIR = DIAGNOSTICS_DIR / "tables"
DIAG_CHARTS_DIR = DIAGNOSTICS_DIR / "charts"
STRESS_DIR = RESULTS_DIR / "stress_tests"
WALK_FORWARD_DIR = RESULTS_DIR / "walk_forward"
RISK_DIR = RESULTS_DIR / "risk"

OUTPUT_DIR = RESULTS_DIR / "final_strategy_report"

# Theme can be set in config.py:
#   STRATEGY_REPORT_THEME = "light"
# or
#   STRATEGY_REPORT_THEME = "dark"
DEFAULT_REPORT_THEME = str(
    cfg("STRATEGY_REPORT_THEME", cfg("CHART_THEME", "light"))
).strip().lower()

REPORT_THEME = "light"
CHARTS_DIR = OUTPUT_DIR / "charts_light"
REPORT_PATH = OUTPUT_DIR / "strategy_report_light.html"

# ============================================================
# VISUAL SYSTEM — PROFESSIONAL LIGHT RESEARCH PACK
# ============================================================
# ============================================================
# VISUAL SYSTEM — REPORT THEMES
# ============================================================

THEMES: dict[str, dict[str, Any]] = {
    "light": {
        # Clean Google Doc / research-report mode
        "paper": "#ffffff",
        "panel": "#ffffff",
        "panel_alt": "#f9fafb",
        "ink": "#111827",
        "muted": "#4b5563",
        "faint": "#9ca3af",
        "grid": "#e5e7eb",
        "rule": "#d1d5db",
        "rule_soft": "#e5e7eb",

        "navy": "#1f4e79",
        "blue": "#2f80ed",
        "teal": "#0f766e",
        "green": "#15803d",
        "amber": "#d97706",
        "red": "#b91c1c",
        "purple": "#6d28d9",
        "brown": "#92400e",
        "gold": "#b45309",

        "bar_edge": "#ffffff",
        "alloc_cmap": "YlGnBu",
        "corr_cmap": "RdBu_r",

        # HTML report styling
        "html_body_bg": "#ffffff",
        "html_hero_bg": "linear-gradient(135deg, #ffffff, #f8fafc)",
        "html_hero_overlay": "linear-gradient(90deg, rgba(31,78,121,0.035), transparent 45%, rgba(180,83,9,0.04))",
        "html_pill_bg": "rgba(255,255,255,.78)",
        "html_notice_bg": "#fff7ed",
        "html_row_even": "rgba(249,250,251,.70)",
        "html_shadow": "0 12px 30px rgba(17,24,39,.06)",
        "html_hero_shadow": "0 18px 50px rgba(17,24,39,.08)",

        "palette": [
            "#1f4e79",
            "#0f766e",
            "#b45309",
            "#6d28d9",
            "#b91c1c",
            "#2f80ed",
            "#92400e",
            "#15803d",
        ],
    },

    "dark": {
        # Existing dashboard / screen-review mode
        "paper": "#0f1117",
        "panel": "#151923",
        "panel_alt": "#111827",
        "ink": "#e8e8e8",
        "muted": "#a3a3a3",
        "faint": "#6b7280",
        "grid": "#2c3340",
        "rule": "#374151",
        "rule_soft": "#273244",

        "navy": "#60a5fa",
        "blue": "#3b82f6",
        "teal": "#2dd4bf",
        "green": "#6ee7b7",
        "amber": "#f59e0b",
        "red": "#fb7185",
        "purple": "#c084fc",
        "brown": "#d6a76c",
        "gold": "#fbbf24",

        "bar_edge": "#2c3340",
        "alloc_cmap": "viridis",
        "corr_cmap": "RdBu_r",

        # HTML report styling
        "html_body_bg": "radial-gradient(circle at 12% 0%, rgba(96,165,250,0.12), transparent 28%), radial-gradient(circle at 88% 8%, rgba(245,158,11,0.10), transparent 30%), #0f1117",
        "html_hero_bg": "linear-gradient(135deg, #151923, #111827)",
        "html_hero_overlay": "linear-gradient(90deg, rgba(96,165,250,0.08), transparent 40%, rgba(245,158,11,0.07))",
        "html_pill_bg": "rgba(15,17,23,.62)",
        "html_notice_bg": "#1f2937",
        "html_row_even": "rgba(31,41,55,.42)",
        "html_shadow": "0 18px 45px rgba(0,0,0,.26)",
        "html_hero_shadow": "0 22px 70px rgba(0,0,0,.34)",

        "palette": [
            "#60a5fa",
            "#6ee7b7",
            "#f59e0b",
            "#c084fc",
            "#fb7185",
            "#3b82f6",
            "#d6a76c",
            "#2dd4bf",
        ],
    },
}

COLORS: dict[str, Any] = {}
PALETTE: list[str] = []


def select_report_theme(theme: str | None = None) -> None:
    """
    Selects the chart/report theme and updates output paths.

    Light mode is for Google Docs / written research.
    Dark mode is for dashboard screenshots / internal visual review.
    """
    global REPORT_THEME, COLORS, PALETTE, CHARTS_DIR, REPORT_PATH

    selected = str(theme or DEFAULT_REPORT_THEME or "light").strip().lower()

    if selected not in THEMES:
        raise ValueError(
            f"Unknown STRATEGY_REPORT_THEME={selected!r}. "
            "Use 'light' or 'dark'."
        )

    REPORT_THEME = selected
    COLORS = THEMES[selected]
    PALETTE = list(COLORS["palette"])

    CHARTS_DIR = OUTPUT_DIR / f"charts_{selected}"
    REPORT_PATH = OUTPUT_DIR / f"strategy_report_{selected}.html"


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["paper"],
            "axes.facecolor": COLORS["panel"],
            "savefig.facecolor": COLORS["paper"],
            "axes.edgecolor": COLORS["rule"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.facecolor": COLORS["panel"],
            "legend.edgecolor": COLORS["rule"],
            "legend.labelcolor": COLORS["ink"],
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.65 if REPORT_THEME == "light" else 0.55,
        }
    )


def heatmap_text_color(value: float, threshold: float = 0.55) -> str:
    """
    Keeps heatmap labels readable on both pale and dark cells.
    """
    try:
        v = abs(float(value))
    except Exception:
        return COLORS["ink"]

    return "#ffffff" if v >= threshold else "#111827"


select_report_theme(DEFAULT_REPORT_THEME)


# ============================================================
# GENERAL HELPERS
# ============================================================


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def esc(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return "N/A"
    return html.escape(str(x))


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def fmt_num(x: Any, digits: int = 2) -> str:
    v = safe_float(x)
    if v is None:
        return "N/A"
    return f"{v:,.{digits}f}"


def fmt_pct(x: Any, digits: int = 2) -> str:
    v = safe_float(x)
    if v is None:
        return "N/A"
    return f"{v:.{digits}%}"


def fmt_money(x: Any, digits: int = 0) -> str:
    v = safe_float(x)
    if v is None:
        return "N/A"
    return f"${v:,.{digits}f}"


def fmt_date(x: Any) -> str:
    if x is None:
        return "N/A"
    try:
        return pd.to_datetime(x).strftime("%Y-%m-%d")
    except Exception:
        return esc(x)


def find_col(df: pd.DataFrame, candidates: tuple[str, ...] | list[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        key = str(c).lower()
        if key in lower:
            return lower[key]
    return None


def read_csv_safe(path: Path | str | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "name"})
        return df
    except Exception as exc:
        print(f"Warning: failed to read {p}: {exc}")
        return pd.DataFrame()


def load_diag_table(name: str) -> pd.DataFrame:
    return read_csv_safe(DIAG_TABLES_DIR / f"{name}.csv")


def coerce_date_index(df: pd.DataFrame, date_candidates: tuple[str, ...] = ("date", "Date", "datetime", "timestamp", "index")) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    date_col = find_col(out, date_candidates)
    if date_col is not None:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col]).set_index(date_col).sort_index()
    else:
        try:
            out.index = pd.to_datetime(out.index)
            out = out.sort_index()
        except Exception:
            return pd.DataFrame()
    out.index.name = "date"
    return out


def first_row(df: pd.DataFrame, strategy: str | None = None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=object)
    d = df.copy()
    if strategy and "strategy" in d.columns:
        m = d[d["strategy"].astype(str).str.lower() == strategy.lower()]
        if not m.empty:
            return m.iloc[0]
    return d.iloc[0]


def series_from_name_value(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=object)
    d = df.copy()
    if "metric" in d.columns and "value" in d.columns:
        return pd.Series(d["value"].values, index=d["metric"].astype(str))
    if "name" in d.columns and "value" in d.columns:
        return pd.Series(d["value"].values, index=d["name"].astype(str))
    if d.shape[0] == 1:
        return pd.Series(d.iloc[0].to_dict())
    if d.shape[1] >= 2:
        return pd.Series(d.iloc[:, 1].values, index=d.iloc[:, 0].astype(str))
    return pd.Series(dtype=object)


def numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def extract_returns(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    d = coerce_date_index(df)
    if d.empty:
        return pd.Series(dtype=float)
    ret_col = find_col(d, ("net_return", "strategy_return", "model_return", "return", "returns", "period_return"))
    if ret_col is not None:
        out = numeric_series(d[ret_col])
        out.name = "return"
        return out
    eq_col = find_col(d, ("equity", "net_equity", "model_equity", "portfolio_value", "value"))
    if eq_col is not None:
        out = pd.to_numeric(d[eq_col], errors="coerce").pct_change()
        out = numeric_series(out)
        out.name = "return"
        return out
    return pd.Series(dtype=float)


def drawdown_from_returns(returns: pd.Series) -> pd.Series:
    r = numeric_series(returns)
    if r.empty:
        return pd.Series(dtype=float)
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    dd.name = "drawdown"
    return dd


def save_fig(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=185, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


def no_data_chart(filename: str, title: str, message: str = "Required data was not available.") -> str:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_title(title)
    ax.text(0.5, 0.52, message, ha="center", va="center", transform=ax.transAxes, fontsize=11, color=COLORS["muted"])
    ax.set_axis_off()
    return save_fig(fig, CHARTS_DIR / filename)


def rel_path(path: str | Path | None) -> Optional[str]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return p.resolve().relative_to(OUTPUT_DIR.resolve()).as_posix()
    except Exception:
        return p.resolve().as_uri()


def format_cell(value: Any, col: str = "") -> str:
    col_l = str(col).lower()
    if isinstance(value, str):
        if any(k in col_l for k in ["date", "start", "end"]):
            return fmt_date(value)
        return esc(value)
    v = safe_float(value)
    if v is None:
        if any(k in col_l for k in ["date", "start", "end"]):
            return fmt_date(value)
        return "N/A"
    if any(k in col_l for k in ["date", "start", "end"]):
        return fmt_date(value)
    if any(k in col_l for k in ["sharpe", "sortino", "calmar", "beta", "ratio", "ic", "corr", "r_squared", "positions"]):
        return f"{v:,.2f}"
    if any(k in col_l for k in ["equity", "capital", "dollar", "cost", "value", "notional"]):
        return f"${v:,.2f}"
    if any(k in col_l for k in ["return", "cagr", "vol", "drawdown", "weight", "cash", "exposure", "turnover", "drag", "contribution", "alpha", "var", "cvar", "loss", "rate", "threshold", "impact", "spread"]):
        if "pct" in col_l and abs(v) > 2.0:
            return f"{v:,.2f}%"
        return f"{v:.2%}"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:,.3f}"


# ============================================================
# CHARTS
# ============================================================


def chart_equity_curve(all_curves: pd.DataFrame, model_curve: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(12.5, 6.2))

    plotted = False
    d = all_curves.copy()
    if not d.empty:
        date_col = find_col(d, ("date", "Date", "datetime", "timestamp", "index"))
        strat_col = find_col(d, ("strategy", "name"))
        eq_col = find_col(d, ("equity", "net_equity", "model_equity", "portfolio_value", "value"))
        if date_col and strat_col and eq_col:
            d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
            d = d.dropna(subset=[date_col])
            order = ["model", "equal_weight", "gold_only", "cash"]
            names = list(d[strat_col].dropna().astype(str).unique())
            names = [n for n in order if n in names] + [n for n in names if n not in order]
            for i, name in enumerate(names):
                g = d[d[strat_col].astype(str) == name].sort_values(date_col)
                ax.plot(g[date_col], pd.to_numeric(g[eq_col], errors="coerce"), label=name.replace("_", " ").title(), lw=2.25 if name == "model" else 1.65, color=PALETTE[i % len(PALETTE)], alpha=0.95)
                plotted = True

    if not plotted and not model_curve.empty:
        d = coerce_date_index(model_curve)
        eq_col = find_col(d, ("equity", "net_equity", "model_equity", "portfolio_value", "value"))
        if eq_col:
            ax.plot(d.index, pd.to_numeric(d[eq_col], errors="coerce"), label="Model", color=COLORS["navy"], lw=2.4)
            plotted = True

    if not plotted:
        plt.close(fig)
        return no_data_chart("equity_curve.png", "Equity curve")

    ax.set_title("Equity Curve — Strategy vs Benchmarks")
    ax.set_ylabel("Portfolio value")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", ncols=2)
    return save_fig(fig, CHARTS_DIR / "equity_curve.png")


def chart_drawdowns(all_curves: pd.DataFrame, model_curve: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    plotted = False

    d = all_curves.copy()
    if not d.empty:
        date_col = find_col(d, ("date", "Date", "datetime", "timestamp", "index"))
        strat_col = find_col(d, ("strategy", "name"))
        ret_col = find_col(d, ("net_return", "strategy_return", "model_return", "return", "returns"))
        if date_col and strat_col and ret_col:
            d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
            for i, (name, g) in enumerate(d.dropna(subset=[date_col]).groupby(strat_col)):
                s = pd.Series(pd.to_numeric(g[ret_col], errors="coerce").values, index=g[date_col]).dropna().sort_index()
                dd = drawdown_from_returns(s)
                ax.plot(dd.index, dd.values, label=str(name).replace("_", " ").title(), lw=2.15 if str(name) == "model" else 1.45, color=PALETTE[i % len(PALETTE)])
                plotted = True

    if not plotted:
        r = extract_returns(model_curve)
        if not r.empty:
            dd = drawdown_from_returns(r)
            ax.plot(dd.index, dd.values, label="Model", color=COLORS["red"], lw=2.1)
            plotted = True

    if not plotted:
        plt.close(fig)
        return no_data_chart("drawdowns.png", "Drawdown")

    ax.set_title("Drawdown Profile")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis="y")
    ax.legend(loc="lower left", ncols=2)
    return save_fig(fig, CHARTS_DIR / "drawdowns.png")


def chart_rolling_metric(model_curve: pd.DataFrame, metric: str, filename: str, title: str) -> str:
    r = extract_returns(model_curve)
    if r.empty:
        return no_data_chart(filename, title)

    window = 252
    fig, ax = plt.subplots(figsize=(12.5, 4.8))

    if metric == "sharpe":
        roll_ret = r.rolling(window).mean() * TRADING_DAYS_PER_YEAR
        roll_vol = r.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        y = roll_ret / roll_vol.replace(0, np.nan)
        ax.axhline(0, color=COLORS["rule"], lw=1.0)
        ax.set_ylabel("Sharpe")
    elif metric == "return":
        y = r.rolling(window).mean() * TRADING_DAYS_PER_YEAR
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.axhline(0, color=COLORS["rule"], lw=1.0)
        ax.set_ylabel("Annualised return")
    elif metric == "volatility":
        y = r.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylabel("Annualised volatility")
    elif metric == "drawdown":
        y = drawdown_from_returns(r)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylabel("Drawdown")
    else:
        y = r
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylabel("Return")

    color = COLORS["navy"] if metric in ["sharpe", "return"] else COLORS["red"] if metric == "drawdown" else COLORS["teal"]
    ax.plot(y.index, y.values, color=color, lw=2.0)
    ax.set_title(title)
    ax.grid(True, axis="y")
    return save_fig(fig, CHARTS_DIR / filename)


def chart_allocation_heatmap(weights: pd.DataFrame) -> str:
    d = coerce_date_index(weights)
    if d.empty:
        return no_data_chart("allocation_heatmap.png", "Allocation heatmap")
    numeric_cols = [c for c in d.columns if c.lower() not in ["strategy"]]
    d = d[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if d.empty:
        return no_data_chart("allocation_heatmap.png", "Allocation heatmap")
    if len(d) > 420:
        d = d.resample("ME").last().dropna(how="all")
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    im = ax.imshow(
        d.T.values,
        aspect="auto",
        interpolation="nearest",
        cmap=COLORS["alloc_cmap"],
        vmin=0,
        vmax=max(
            0.35,
            float(np.nanmax(d.values)) if np.isfinite(d.values).any() else 1.0,
        ),
    )
    ax.set_title("Allocation Heatmap")
    ax.set_yticks(range(len(d.columns)))
    ax.set_yticklabels(d.columns)
    xticks = np.linspace(0, max(len(d.index) - 1, 0), min(8, len(d.index))).astype(int)
    ax.set_xticks(xticks)
    ax.set_xticklabels([d.index[i].strftime("%Y-%m") for i in xticks], rotation=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    return save_fig(fig, CHARTS_DIR / "allocation_heatmap.png")


def chart_exposure_cash(weights: pd.DataFrame) -> str:
    d = coerce_date_index(weights)
    if d.empty:
        return no_data_chart("exposure_cash.png", "Exposure and cash")
    cols = [c for c in d.columns if str(c).lower() not in ["strategy", "cash", "cash_weight"]]
    asset_w = d[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    exposure = asset_w.sum(axis=1).clip(0, 1)
    cash = (1.0 - exposure).clip(0, 1)
    fig, ax = plt.subplots(figsize=(12.5, 4.9))
    ax.fill_between(exposure.index, 0, exposure.values, color=COLORS["navy"], alpha=0.25, label="Gross exposure")
    ax.plot(exposure.index, exposure.values, color=COLORS["navy"], lw=2.0)
    ax.plot(cash.index, cash.values, color=COLORS["gold"], lw=2.0, label="Cash")
    ax.set_title("Portfolio Exposure and Cash")
    ax.set_ylabel("Weight")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis="y")
    ax.legend(loc="upper left")
    return save_fig(fig, CHARTS_DIR / "exposure_cash.png")


def chart_average_weights(weights: pd.DataFrame, ticker_attr: pd.DataFrame) -> str:
    if ticker_attr is not None and not ticker_attr.empty and {"ticker", "average_weight"}.issubset(ticker_attr.columns):
        s = ticker_attr.set_index("ticker")["average_weight"].sort_values(ascending=True)
    else:
        d = coerce_date_index(weights)
        if d.empty:
            return no_data_chart("average_weights.png", "Average allocation")
        cols = [c for c in d.columns if str(c).lower() not in ["strategy"]]
        s = d[cols].apply(pd.to_numeric, errors="coerce").mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.barh(s.index.astype(str), s.values, color=COLORS["teal"], alpha=0.9)
    ax.set_title("Average Allocation by Asset")
    ax.set_xlabel("Average portfolio weight")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis="x")
    return save_fig(fig, CHARTS_DIR / "average_weights.png")


def chart_contribution(ticker_attr: pd.DataFrame) -> str:
    if ticker_attr is None or ticker_attr.empty or "ticker" not in ticker_attr.columns or "total_contribution" not in ticker_attr.columns:
        return no_data_chart("ticker_contribution.png", "Return contribution")
    d = ticker_attr.copy()
    d["total_contribution"] = pd.to_numeric(d["total_contribution"], errors="coerce")
    d = d.dropna(subset=["total_contribution"]).sort_values("total_contribution")
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    colors = [COLORS["red"] if x < 0 else COLORS["green"] for x in d["total_contribution"]]
    ax.barh(d["ticker"].astype(str), d["total_contribution"], color=colors, alpha=0.9)
    ax.set_title("Cumulative Return Contribution by Asset")
    ax.set_xlabel("Return contribution")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.axvline(0, color=COLORS["rule"], lw=1)
    ax.grid(True, axis="x")
    return save_fig(fig, CHARTS_DIR / "ticker_contribution.png")


def chart_selected_vs_rejected(ticker_attr: pd.DataFrame) -> str:
    needed = {"ticker", "avg_return_when_held", "avg_return_when_not_held"}
    if ticker_attr is None or ticker_attr.empty or not needed.issubset(set(ticker_attr.columns)):
        return no_data_chart("selected_vs_rejected.png", "Selected vs rejected returns")
    d = ticker_attr.copy().sort_values("ticker")
    x = np.arange(len(d))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    held = pd.to_numeric(d["avg_return_when_held"], errors="coerce")
    rejected = pd.to_numeric(d["avg_return_when_not_held"], errors="coerce")
    ax.bar(x - width/2, held, width, label="Selected", color=COLORS["green"], alpha=0.9)
    ax.bar(x + width/2, rejected, width, label="Rejected", color=COLORS["brown"], alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(d["ticker"].astype(str))
    ax.set_title("Average Forward Return When Selected vs Rejected")
    ax.set_ylabel("Average forward return")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.axhline(0, color=COLORS["rule"], lw=1)
    ax.grid(True, axis="y")
    ax.legend(loc="upper left")
    return save_fig(fig, CHARTS_DIR / "selected_vs_rejected.png")


def chart_feature_ic(feature_attr: pd.DataFrame) -> str:
    if feature_attr is None or feature_attr.empty or "feature" not in feature_attr.columns:
        return no_data_chart("feature_ic.png", "Feature rank IC")
    col = find_col(feature_attr, ("average_cross_sectional_rank_ic", "average_cross_sectional_ic", "full_sample_corr_with_forward_return"))
    if col is None:
        return no_data_chart("feature_ic.png", "Feature rank IC")
    d = feature_attr.copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=[col]).sort_values(col)
    if d.empty:
        return no_data_chart("feature_ic.png", "Feature rank IC")
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    colors = [COLORS["red"] if x < 0 else COLORS["navy"] for x in d[col]]
    ax.barh(d["feature"].astype(str), d[col], color=colors, alpha=0.9)
    ax.axvline(0, color=COLORS["rule"], lw=1)
    ax.set_title("Feature Information Coefficient")
    ax.set_xlabel(col.replace("_", " ").title())
    ax.grid(True, axis="x")
    return save_fig(fig, CHARTS_DIR / "feature_ic.png")


def chart_asset_correlation(asset_returns: pd.DataFrame) -> str:
    d = coerce_date_index(asset_returns)

    if d.empty:
        return no_data_chart("asset_correlation.png", "Asset return correlation")

    d = d.apply(pd.to_numeric, errors="coerce").dropna(how="all")

    if d.shape[1] < 2:
        return no_data_chart("asset_correlation.png", "Asset return correlation")

    corr = d.corr()

    fig, ax = plt.subplots(figsize=(8.5, 7.2))

    im = ax.imshow(
        corr.values,
        cmap=COLORS["corr_cmap"],
        vmin=-1,
        vmax=1,
    )

    ax.set_title("Asset Return Correlation Matrix")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            val = corr.iloc[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=heatmap_text_color(val),
                fontsize=9,
                fontweight="700" if abs(val) >= 0.55 else "500",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_edgecolor(COLORS["rule"])
    cbar.ax.tick_params(colors=COLORS["ink"])

    return save_fig(fig, CHARTS_DIR / "asset_correlation.png")


def chart_walk_forward_equity(wf_equity: pd.DataFrame, wf_returns: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    plotted = False
    d = wf_equity.copy()
    if not d.empty:
        date_col = find_col(d, ("date", "Date", "datetime", "timestamp"))
        strat_col = find_col(d, ("strategy",))
        eq_col = find_col(d, ("equity", "strategy_equity", "model_equity", "portfolio_value", "value"))
        if date_col and eq_col:
            d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
            if strat_col:
                for i, (name, g) in enumerate(d.dropna(subset=[date_col]).groupby(strat_col)):
                    ax.plot(g[date_col], pd.to_numeric(g[eq_col], errors="coerce"), label=str(name).replace("_", " ").title(), lw=2, color=PALETTE[i % len(PALETTE)])
                    plotted = True
            else:
                ax.plot(d[date_col], pd.to_numeric(d[eq_col], errors="coerce"), label="Walk-forward", lw=2, color=COLORS["navy"])
                plotted = True

    if not plotted and not wf_returns.empty:
        r = extract_returns(wf_returns)
        if not r.empty:
            eq = INITIAL_CAPITAL * (1 + r).cumprod()
            ax.plot(eq.index, eq.values, label="Walk-forward", lw=2, color=COLORS["navy"])
            plotted = True

    if not plotted:
        plt.close(fig)
        return no_data_chart("walk_forward_equity.png", "Walk-forward equity")
    ax.set_title("Walk-Forward Equity Curve")
    ax.set_ylabel("Portfolio value")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, axis="y")
    ax.legend(loc="upper left")
    return save_fig(fig, CHARTS_DIR / "walk_forward_equity.png")


def chart_walk_forward_periods(wf_periods: pd.DataFrame) -> str:
    if wf_periods is None or wf_periods.empty:
        return no_data_chart("walk_forward_periods.png", "Walk-forward periods")
    d = wf_periods.copy()
    label_col = find_col(d, ("test_start", "start_date", "period", "window"))
    if label_col is None:
        d["period"] = np.arange(len(d)) + 1
        label_col = "period"
    metrics = [c for c in ["cagr", "sharpe", "max_drawdown"] if c in d.columns]
    if not metrics:
        return no_data_chart("walk_forward_periods.png", "Walk-forward periods")
    labels = [fmt_date(x) if "date" in str(label_col).lower() or "start" in str(label_col).lower() else str(x) for x in d[label_col]]
    x = np.arange(len(d))
    fig, ax1 = plt.subplots(figsize=(12.5, 5.7))
    if "cagr" in metrics:
        ax1.bar(x, pd.to_numeric(d["cagr"], errors="coerce"), color=COLORS["navy"], alpha=0.78, label="CAGR")
    if "max_drawdown" in metrics:
        ax1.plot(x, pd.to_numeric(d["max_drawdown"], errors="coerce"), color=COLORS["red"], marker="o", lw=2.0, label="Max drawdown")
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_ylabel("Return / drawdown")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax2 = ax1.twinx()
    if "sharpe" in metrics:
        ax2.plot(x, pd.to_numeric(d["sharpe"], errors="coerce"), color=COLORS["gold"], marker="s", lw=2.0, label="Sharpe")
        ax2.set_ylabel("Sharpe")
    ax1.set_title("Walk-Forward Test Window Summary")
    ax1.grid(True, axis="y")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    return save_fig(fig, CHARTS_DIR / "walk_forward_periods.png")


def chart_stress_costs(cost_df: pd.DataFrame) -> str:
    if cost_df is None or cost_df.empty or "total_cost_bps" not in cost_df.columns:
        return no_data_chart("stress_costs.png", "Cost sensitivity")
    d = cost_df.copy().sort_values("total_cost_bps")
    fig, ax1 = plt.subplots(figsize=(11.5, 5.4))
    x = pd.to_numeric(d["total_cost_bps"], errors="coerce")
    if "cagr" in d.columns:
        ax1.plot(x, pd.to_numeric(d["cagr"], errors="coerce"), color=COLORS["navy"], marker="o", lw=2.2, label="CAGR")
    if "max_drawdown" in d.columns:
        ax1.plot(x, pd.to_numeric(d["max_drawdown"], errors="coerce"), color=COLORS["red"], marker="o", lw=2.0, label="Max drawdown")
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_xlabel("Total cost assumption (bps)")
    ax1.set_ylabel("Return / drawdown")
    ax2 = ax1.twinx()
    if "sharpe" in d.columns:
        ax2.plot(x, pd.to_numeric(d["sharpe"], errors="coerce"), color=COLORS["gold"], marker="s", lw=2.0, label="Sharpe")
        ax2.set_ylabel("Sharpe")
    ax1.set_title("Stress Test — Cost Sensitivity")
    ax1.grid(True, axis="y")
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper right")
    return save_fig(fig, CHARTS_DIR / "stress_costs.png")


def chart_stress_delay(delay_df: pd.DataFrame) -> str:
    if delay_df is None or delay_df.empty or "execution_delay_days" not in delay_df.columns:
        return no_data_chart("stress_delay.png", "Execution delay sensitivity")
    d = delay_df.copy().sort_values("execution_delay_days")
    fig, ax1 = plt.subplots(figsize=(11.5, 5.4))
    x = pd.to_numeric(d["execution_delay_days"], errors="coerce")
    if "cagr" in d.columns:
        ax1.plot(x, pd.to_numeric(d["cagr"], errors="coerce"), color=COLORS["navy"], marker="o", lw=2.2, label="CAGR")
    if "max_drawdown" in d.columns:
        ax1.plot(x, pd.to_numeric(d["max_drawdown"], errors="coerce"), color=COLORS["red"], marker="o", lw=2.0, label="Max drawdown")
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_xlabel("Execution delay (trading days)")
    ax1.set_ylabel("Return / drawdown")
    ax2 = ax1.twinx()
    if "sharpe" in d.columns:
        ax2.plot(x, pd.to_numeric(d["sharpe"], errors="coerce"), color=COLORS["gold"], marker="s", lw=2.0, label="Sharpe")
        ax2.set_ylabel("Sharpe")
    ax1.set_title("Stress Test — Execution Delay")
    ax1.grid(True, axis="y")
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper right")
    return save_fig(fig, CHARTS_DIR / "stress_delay.png")


def chart_stress_shocks(shocks_df: pd.DataFrame) -> str:
    if shocks_df is None or shocks_df.empty or "scenario" not in shocks_df.columns:
        return no_data_chart("stress_shocks.png", "Asset shock scenarios")
    impact_col = find_col(shocks_df, ("portfolio_return", "portfolio_return_pct", "equity_impact"))
    if impact_col is None:
        return no_data_chart("stress_shocks.png", "Asset shock scenarios")
    d = shocks_df.copy()
    d[impact_col] = pd.to_numeric(d[impact_col], errors="coerce")
    if impact_col.endswith("pct"):
        d[impact_col] = d[impact_col] / 100.0
    d = d.dropna(subset=[impact_col]).sort_values(impact_col)
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    colors = [COLORS["red"] if x < 0 else COLORS["green"] for x in d[impact_col]]
    ax.barh(d["scenario"].astype(str).str.replace("_", " "), d[impact_col], color=colors, alpha=0.88)
    ax.set_title("Stress Test — Latest Portfolio Shock Impact")
    ax.set_xlabel("Estimated portfolio return impact")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.axvline(0, color=COLORS["rule"], lw=1)
    ax.grid(True, axis="x")
    return save_fig(fig, CHARTS_DIR / "stress_shocks.png")


def chart_var_distribution(var_summary: pd.DataFrame, risk_breaches: pd.DataFrame, model_curve: pd.DataFrame) -> str:
    r = extract_returns(model_curve)
    if r.empty and risk_breaches is not None and not risk_breaches.empty:
        d = coerce_date_index(risk_breaches)
        col = find_col(d, ("strategy_return", "return", "net_return"))
        if col:
            r = numeric_series(d[col])
    if r.empty:
        return no_data_chart("var_distribution.png", "Return distribution and VaR")
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    ax.hist(
        r.values,
        bins=80,
        color=COLORS["blue"],
        edgecolor=COLORS["bar_edge"],
        alpha=0.82,
    )
    if var_summary is not None and not var_summary.empty:
        d = var_summary[var_summary.get("horizon_days", pd.Series(dtype=object)).astype(str).isin(["1", "1.0"])] if "horizon_days" in var_summary.columns else var_summary.copy()
        for _, row in d.iterrows():
            conf = safe_float(row.get("confidence_level"))
            var = safe_float(row.get("var"))
            if conf is not None and var is not None:
                color = COLORS["amber"] if conf < 0.99 else COLORS["red"]
                ax.axvline(-var, color=color, lw=2.0, ls="--", label=f"{int(conf*100)}% VaR")
    ax.axvline(0, color=COLORS["rule"], lw=1)
    ax.set_title("Daily Return Distribution and VaR Thresholds")
    ax.set_xlabel("Daily return")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis="y")
    ax.legend(loc="upper left")
    return save_fig(fig, CHARTS_DIR / "var_distribution.png")


def chart_var_rolling(rolling_var: pd.DataFrame) -> str:
    d = coerce_date_index(rolling_var)
    if d.empty:
        return no_data_chart("rolling_var_cvar.png", "Rolling VaR and CVaR")
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    cols = [c for c in d.columns if "rolling_var" in c.lower() or "rolling_cvar" in c.lower()]
    if not cols:
        return no_data_chart("rolling_var_cvar.png", "Rolling VaR and CVaR")
    for i, col in enumerate(cols):
        line_style = "--" if "cvar" in col.lower() else "-"
        ax.plot(d.index, pd.to_numeric(d[col], errors="coerce"), label=col.replace("_", " ").title(), color=PALETTE[i % len(PALETTE)], lw=1.8, ls=line_style)
    ax.set_title("Rolling Historical VaR / CVaR")
    ax.set_ylabel("Loss threshold")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", ncols=2)
    return save_fig(fig, CHARTS_DIR / "rolling_var_cvar.png")


def chart_worst_losses(worst_losses: pd.DataFrame) -> str:
    if worst_losses is None or worst_losses.empty:
        return no_data_chart("worst_losses.png", "Worst realised losses")
    d = worst_losses.copy()
    loss_col = find_col(d, ("worst_return", "worst_loss"))
    label_col = find_col(d, ("window", "horizon_days"))
    if loss_col is None or label_col is None:
        return no_data_chart("worst_losses.png", "Worst realised losses")
    vals = pd.to_numeric(d[loss_col], errors="coerce")
    if "loss" in loss_col.lower() and vals.mean(skipna=True) > 0:
        vals = -vals
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.bar(d[label_col].astype(str), vals, color=COLORS["red"], alpha=0.88)
    ax.set_title("Worst Realised Loss by Holding Window")
    ax.set_ylabel("Worst return")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.axhline(0, color=COLORS["rule"], lw=1)
    ax.grid(True, axis="y")
    return save_fig(fig, CHARTS_DIR / "worst_losses.png")


# ============================================================
# HTML COMPONENTS
# ============================================================


def status_class(status: str) -> str:
    s = status.lower()
    if s in {"pass", "strong", "ok"}:
        return "pass"
    if s in {"watch"}:
        return "watch"
    if s in {"fail", "weak", "review"}:
        return "fail"
    if s in {"incomplete"}:
        return "neutral"
    return "neutral"


def badge(status: str) -> str:
    cls = status_class(status)
    return f'<span class="badge {cls}">{esc(status.upper())}</span>'


def metric_card(title: str, value: str, label: str = "", status: str | None = None) -> str:
    status_html = badge(status) if status else ""
    return f"""
    <div class="metric-card">
      <div class="metric-top"><span>{esc(title)}</span>{status_html}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-label">{esc(label)}</div>
    </div>
    """


def chart_card(path: str | Path | None, title: str, caption: str = "") -> str:
    src = rel_path(path)
    if not src:
        return f"""
        <div class="chart-card missing">
          <div class="chart-title">{esc(title)}</div>
          <div class="missing-text">Chart unavailable</div>
        </div>
        """
    caption_html = f'<div class="chart-caption">{esc(caption)}</div>' if caption else ""
    return f"""
    <div class="chart-card">
      <div class="chart-title">{esc(title)}</div>
      <img src="{esc(src)}" alt="{esc(title)}" />
      {caption_html}
    </div>
    """


def table_html(df: pd.DataFrame, title: str = "", columns: list[str] | None = None, max_rows: int = 12) -> str:
    if df is None or df.empty:
        return f"""
        <div class="table-card">
          <div class="table-title">{esc(title)}</div>
          <div class="missing-text">No table data available.</div>
        </div>
        """
    d = df.copy()
    if columns:
        keep = [c for c in columns if c in d.columns]
        if keep:
            d = d[keep]
    if max_rows is not None and len(d) > max_rows:
        d = d.head(max_rows)
    header = "".join(f"<th>{esc(c)}</th>" for c in d.columns)
    rows = []
    for _, row in d.iterrows():
        cells = []
        for col in d.columns:
            cls = ""
            col_l = str(col).lower()
            v = safe_float(row[col])
            if v is not None and any(k in col_l for k in ["return", "cagr", "drawdown", "contribution", "alpha", "loss", "impact"]):
                cls = "pos" if v > 0 else "neg" if v < 0 else ""
                if "drawdown" in col_l or "loss" in col_l:
                    cls = "neg" if v < 0 or v > 0 else ""
            cells.append(f'<td class="{cls}">{format_cell(row[col], col)}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
    <div class="table-card">
      <div class="table-title">{esc(title)}</div>
      <div class="table-scroll">
        <table>
          <thead><tr>{header}</tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </div>
    """


def section(section_id: str, title: str, subtitle: str, body: str) -> str:
    return f"""
    <section class="section" id="{esc(section_id)}">
      <div class="section-header">
        <div>
          <div class="section-kicker">{esc(section_id)}</div>
          <h2>{esc(title)}</h2>
          <p>{esc(subtitle)}</p>
        </div>
      </div>
      {body}
    </section>
    """


# ============================================================
# STATUS LOGIC
# ============================================================


def status_performance(row: pd.Series) -> str:
    sharpe = safe_float(row.get("sharpe"))
    cagr = safe_float(row.get("cagr"))
    dd = safe_float(row.get("max_drawdown"))
    if sharpe is None or cagr is None or dd is None:
        return "Incomplete"
    if sharpe >= 1.25 and cagr > 0.08 and dd > -0.18:
        return "Pass"
    if sharpe >= 0.75 and cagr > 0 and dd > -0.30:
        return "Watch"
    return "Review"


def status_walk_forward(row: pd.Series) -> str:
    if row.empty:
        return "Incomplete"
    sharpe = safe_float(row.get("sharpe"))
    cagr = safe_float(row.get("cagr"))
    dd = safe_float(row.get("max_drawdown"))
    if sharpe is None or cagr is None:
        return "Incomplete"
    if sharpe >= 1.0 and cagr > 0.06 and (dd is None or dd > -0.22):
        return "Pass"
    if sharpe >= 0.5 and cagr > 0:
        return "Watch"
    return "Review"


def status_var(breach_summary: pd.DataFrame) -> str:
    if breach_summary is None or breach_summary.empty:
        return "Incomplete"
    if "calibration_status" in breach_summary.columns:
        statuses = set(breach_summary["calibration_status"].astype(str).str.lower())
        if "fail" in statuses:
            return "Review"
        if "watch" in statuses:
            return "Watch"
        return "Pass"
    actual_col = find_col(breach_summary, ("actual_breach_rate",))
    expected_col = find_col(breach_summary, ("expected_breach_rate",))
    if actual_col and expected_col:
        actual = pd.to_numeric(breach_summary[actual_col], errors="coerce")
        expected = pd.to_numeric(breach_summary[expected_col], errors="coerce")
        ratio = (actual / expected.replace(0, np.nan)).max()
        if pd.notna(ratio) and ratio <= 1.5:
            return "Pass"
        if pd.notna(ratio) and ratio <= 2.5:
            return "Watch"
        return "Review"
    return "Incomplete"


def status_cost(cost_df: pd.DataFrame) -> str:
    row = first_row(cost_df, "model")
    turnover = safe_float(row.get("annualised_turnover"))
    drag = safe_float(row.get("annualised_transaction_cost_drag"))
    if turnover is None and drag is None:
        return "Incomplete"
    if (turnover is None or turnover < 3.0) and (drag is None or abs(drag) < 0.02):
        return "Pass"
    if (turnover is None or turnover < 5.0) and (drag is None or abs(drag) < 0.05):
        return "Watch"
    return "Review"


def status_stress(stress_cost: pd.DataFrame, stress_delay: pd.DataFrame) -> str:
    if (stress_cost is None or stress_cost.empty) and (stress_delay is None or stress_delay.empty):
        return "Incomplete"
    for df in [stress_cost, stress_delay]:
        if df is None or df.empty:
            continue
        if "sharpe" in df.columns:
            s = pd.to_numeric(df["sharpe"], errors="coerce").min()
            if pd.notna(s) and s < 0.5:
                return "Review"
    return "Pass"


# ============================================================
# REPORT GENERATION
# ============================================================


def load_all_data() -> dict[str, Any]:
    return {
        "model_curve": read_csv_safe(BACKTEST_V3_DIR / "model_curve_V3.csv"),
        "all_curves": read_csv_safe(BACKTEST_V3_DIR / "all_curves_V3.csv"),
        "weights": read_csv_safe(BACKTEST_V3_DIR / "weights_history_V3.csv"),
        "asset_returns": read_csv_safe(BACKTEST_V3_DIR / "asset_returns_V3.csv"),
        "benchmarks": read_csv_safe(BACKTEST_V3_DIR / "benchmark_returns_V3.csv"),
        "performance": read_csv_safe(BACKTEST_V3_DIR / "performance_summary_V3.csv"),
        "costs": read_csv_safe(BACKTEST_V3_DIR / "cost_summary_V3.csv"),
        "scenario_summary": read_csv_safe(BACKTEST_V3_DIR / "scenario_summary_V3.csv"),
        "ticker_attr": load_diag_table("ticker_attribution"),
        "feature_attr": load_diag_table("feature_attribution"),
        "feature_deciles": load_diag_table("feature_deciles"),
        "score_threshold": load_diag_table("score_threshold_diagnostics"),
        "regime_summary": load_diag_table("regime_summary"),
        "correlation_summary": load_diag_table("correlation_summary"),
        "return_concentration": load_diag_table("return_concentration"),
        "exposure_summary": load_diag_table("exposure_summary"),
        "turnover_cost_summary": load_diag_table("turnover_cost_summary"),
        "red_flags": load_diag_table("red_flags"),
        "stress_baseline": read_csv_safe(STRESS_DIR / "stress_baseline_summary.csv"),
        "stress_historical": read_csv_safe(STRESS_DIR / "stress_historical_windows.csv"),
        "stress_cost": read_csv_safe(STRESS_DIR / "stress_cost_sensitivity.csv"),
        "stress_delay": read_csv_safe(STRESS_DIR / "stress_execution_delay.csv"),
        "stress_worst": read_csv_safe(STRESS_DIR / "stress_worst_rolling_windows.csv"),
        "stress_shocks": read_csv_safe(STRESS_DIR / "stress_asset_shocks.csv"),
        "stress_contribution": read_csv_safe(STRESS_DIR / "stress_asset_contribution.csv"),
        "wf_summary": read_csv_safe(WALK_FORWARD_DIR / "walk_forward_summary.csv"),
        "wf_periods": read_csv_safe(WALK_FORWARD_DIR / "walk_forward_periods.csv"),
        "wf_returns": read_csv_safe(WALK_FORWARD_DIR / "walk_forward_returns.csv"),
        "wf_equity": read_csv_safe(WALK_FORWARD_DIR / "walk_forward_equity.csv"),
        "wf_regime": read_csv_safe(WALK_FORWARD_DIR / "walk_forward_regime_summary.csv"),
        "var_summary": read_csv_safe(RISK_DIR / "var_summary.csv"),
        "var_breach_summary": read_csv_safe(RISK_DIR / "var_breach_summary.csv"),
        "rolling_var": read_csv_safe(RISK_DIR / "rolling_var.csv"),
        "var_breaches": read_csv_safe(RISK_DIR / "var_breaches.csv"),
        "worst_losses": read_csv_safe(RISK_DIR / "worst_losses.csv"),
        "tail_events": read_csv_safe(RISK_DIR / "tail_events.csv"),
    }


def generate_all_charts(data: dict[str, Any]) -> dict[str, str]:
    charts = {}
    charts["equity"] = chart_equity_curve(data["all_curves"], data["model_curve"])
    charts["drawdown"] = chart_drawdowns(data["all_curves"], data["model_curve"])
    charts["rolling_sharpe"] = chart_rolling_metric(data["model_curve"], "sharpe", "rolling_sharpe.png", "Rolling 252-Day Sharpe")
    charts["rolling_return"] = chart_rolling_metric(data["model_curve"], "return", "rolling_return.png", "Rolling 252-Day Annualised Return")
    charts["rolling_vol"] = chart_rolling_metric(data["model_curve"], "volatility", "rolling_volatility.png", "Rolling 252-Day Annualised Volatility")
    charts["rolling_drawdown"] = chart_rolling_metric(data["model_curve"], "drawdown", "rolling_drawdown.png", "Strategy Drawdown")
    charts["allocation_heatmap"] = chart_allocation_heatmap(data["weights"])
    charts["exposure_cash"] = chart_exposure_cash(data["weights"])
    charts["average_weights"] = chart_average_weights(data["weights"], data["ticker_attr"])
    charts["contribution"] = chart_contribution(data["ticker_attr"])
    charts["selected_vs_rejected"] = chart_selected_vs_rejected(data["ticker_attr"])
    charts["feature_ic"] = chart_feature_ic(data["feature_attr"])
    charts["asset_corr"] = chart_asset_correlation(data["asset_returns"])
    charts["wf_equity"] = chart_walk_forward_equity(data["wf_equity"], data["wf_returns"])
    charts["wf_periods"] = chart_walk_forward_periods(data["wf_periods"])
    charts["stress_costs"] = chart_stress_costs(data["stress_cost"])
    charts["stress_delay"] = chart_stress_delay(data["stress_delay"])
    charts["stress_shocks"] = chart_stress_shocks(data["stress_shocks"])
    charts["var_distribution"] = chart_var_distribution(data["var_summary"], data["var_breaches"], data["model_curve"])
    charts["rolling_var"] = chart_var_rolling(data["rolling_var"])
    charts["worst_losses"] = chart_worst_losses(data["worst_losses"])
    return charts


def build_missing_outputs(data: dict[str, Any]) -> str:
    critical = {
        "Backtest V3 model curve": data["model_curve"],
        "Backtest V3 performance summary": data["performance"],
        "Weights history": data["weights"],
        "Diagnostics ticker attribution": data["ticker_attr"],
        "Stress cost sensitivity": data["stress_cost"],
        "Walk-forward summary": data["wf_summary"],
        "Risk / VaR summary": data["var_summary"],
    }
    missing = [name for name, df in critical.items() if df is None or df.empty]
    if not missing:
        return ""
    items = "".join(f"<li>{esc(m)}</li>" for m in missing)
    return f"""
    <div class="notice warn">
      <strong>Missing output groups</strong>
      <ul>{items}</ul>
    </div>
    """


def build_report_html(data: dict[str, Any], charts: dict[str, str]) -> str:
    perf_row = first_row(data["performance"], "model")
    wf_row = first_row(data["wf_summary"])
    cost_row = first_row(data["costs"], "model")
    exposure = series_from_name_value(data["exposure_summary"])
    concentration = series_from_name_value(data["return_concentration"])

    perf_status = status_performance(perf_row)
    wf_status = status_walk_forward(wf_row)
    stress_status = status_stress(data["stress_cost"], data["stress_delay"])
    var_status = status_var(data["var_breach_summary"])
    cost_status = status_cost(data["costs"])

    model_start = "N/A"
    model_end = "N/A"
    mc = coerce_date_index(data["model_curve"])
    if not mc.empty:
        model_start = mc.index.min().strftime("%Y-%m-%d")
        model_end = mc.index.max().strftime("%Y-%m-%d")

    exec_cards = "".join([
        metric_card("CAGR", fmt_pct(perf_row.get("cagr")), "Full-sample backtest", perf_status),
        metric_card("Sharpe", fmt_num(perf_row.get("sharpe")), "Annualised risk-adjusted return", perf_status),
        metric_card("Sortino", fmt_num(perf_row.get("sortino")), "Downside-risk adjusted", perf_status),
        metric_card("Max drawdown", fmt_pct(perf_row.get("max_drawdown")), "Largest peak-to-trough loss", perf_status),
        metric_card("Walk-forward Sharpe", fmt_num(wf_row.get("sharpe")), "Stitched out-of-sample validation", wf_status),
        metric_card("Walk-forward CAGR", fmt_pct(wf_row.get("cagr")), "Stitched out-of-sample validation", wf_status),
        metric_card("Average exposure", fmt_pct(perf_row.get("average_exposure", exposure.get("average_exposure", np.nan))), "Capital deployed", None),
        metric_card("Annual turnover", fmt_pct(cost_row.get("annualised_turnover")), "Execution intensity", cost_status),
    ])

    validation_rows = pd.DataFrame([
        {"dimension": "Backtest performance", "status": perf_status, "primary_metric": "Sharpe", "value": perf_row.get("sharpe")},
        {"dimension": "Walk-forward validation", "status": wf_status, "primary_metric": "Sharpe", "value": wf_row.get("sharpe")},
        {"dimension": "Stress robustness", "status": stress_status, "primary_metric": "Cost/delay sensitivity", "value": "see stress section"},
        {"dimension": "VaR calibration", "status": var_status, "primary_metric": "Breach rate", "value": "see risk section"},
        {"dimension": "Implementation cost", "status": cost_status, "primary_metric": "Annual turnover", "value": cost_row.get("annualised_turnover")},
    ])

    overview = section(
        "01",
        "Executive Summary",
        "Headline performance, validation status and core implementation metrics.",
        f"""
        <div class="metric-grid">{exec_cards}</div>
        <div class="two-col">
          {table_html(validation_rows, "Validation Status", max_rows=10)}
          {table_html(data["performance"], "Backtest Summary", columns=["strategy", "final_equity", "cagr", "annualised_volatility", "sharpe", "sortino", "calmar", "max_drawdown", "average_exposure", "average_cash"], max_rows=6)}
        </div>
        """,
    )

    performance = section(
        "02",
        "Performance and Benchmarking",
        "Backtest equity, drawdown and rolling risk/return behaviour.",
        f"""
        <div class="chart-grid single">{chart_card(charts['equity'], 'Equity Curve')}</div>
        <div class="chart-grid single">{chart_card(charts['drawdown'], 'Drawdown')}</div>
        <div class="chart-grid two">
          {chart_card(charts['rolling_sharpe'], 'Rolling Sharpe')}
          {chart_card(charts['rolling_return'], 'Rolling Return')}
          {chart_card(charts['rolling_vol'], 'Rolling Volatility')}
          {chart_card(charts['rolling_drawdown'], 'Rolling Drawdown')}
        </div>
        """,
    )

    walk_forward = section(
        "03",
        "Walk-Forward Validation",
        "Out-of-sample stitched results and test-window stability.",
        f"""
        <div class="chart-grid single">{chart_card(charts['wf_equity'], 'Walk-Forward Equity')}</div>
        <div class="chart-grid single">{chart_card(charts['wf_periods'], 'Walk-Forward Test Windows')}</div>
        <div class="two-col">
          {table_html(data['wf_summary'], 'Walk-Forward Summary', columns=['strategy', 'final_equity', 'cagr', 'annualised_volatility', 'sharpe', 'sortino', 'max_drawdown', 'alpha_annualised', 'beta', 'average_exposure'], max_rows=5)}
          {table_html(data['wf_periods'], 'Walk-Forward Periods', columns=['strategy', 'train_start', 'train_end', 'test_start', 'test_end', 'cagr', 'sharpe', 'sortino', 'max_drawdown', 'final_equity'], max_rows=10)}
        </div>
        """,
    )

    stress = section(
        "04",
        "Stress and Execution Robustness",
        "Cost, delay, historical-window and shock sensitivity.",
        f"""
        <div class="chart-grid two">
          {chart_card(charts['stress_costs'], 'Cost Sensitivity')}
          {chart_card(charts['stress_delay'], 'Execution Delay')}
          {chart_card(charts['stress_shocks'], 'Asset Shock Scenarios')}
          {chart_card(charts['worst_losses'], 'Worst Loss Windows')}
        </div>
        <div class="two-col">
          {table_html(data['stress_cost'], 'Cost Sensitivity', columns=['total_cost_bps', 'final_equity', 'cagr', 'sharpe', 'sortino', 'max_drawdown', 'annualised_turnover'], max_rows=8)}
          {table_html(data['stress_delay'], 'Execution Delay', columns=['execution_delay_days', 'final_equity', 'cagr', 'sharpe', 'sortino', 'max_drawdown'], max_rows=8)}
        </div>
        {table_html(data['stress_historical'], 'Historical Regime Windows', columns=['scenario', 'strategy', 'scenario_start', 'scenario_end', 'final_equity', 'cagr', 'sharpe', 'sortino', 'max_drawdown'], max_rows=18)}
        """,
    )

    risk = section(
        "05",
        "Tail Risk and VaR",
        "Historical VaR, CVaR, breach calibration and realised loss windows.",
        f"""
        <div class="chart-grid two">
          {chart_card(charts['var_distribution'], 'Return Distribution and VaR')}
          {chart_card(charts['rolling_var'], 'Rolling VaR / CVaR')}
        </div>
        <div class="two-col">
          {table_html(data['var_summary'], 'VaR / CVaR Summary', columns=['confidence_label', 'horizon_days', 'observations', 'var', 'cvar', 'return_threshold', 'worst_return', 'var_dollars_initial_capital', 'cvar_dollars_initial_capital'], max_rows=12)}
          {table_html(data['var_breach_summary'], 'VaR Breach Calibration', columns=['confidence_label', 'rolling_window_days', 'valid_observations', 'breach_count', 'expected_breach_rate', 'actual_breach_rate', 'calibration_status'], max_rows=8)}
        </div>
        <div class="two-col">
          {table_html(data['worst_losses'], 'Worst Rolling Losses', max_rows=10)}
          {table_html(data['tail_events'], 'Largest One-Day Tail Events', columns=['date', 'strategy_return', 'loss', 'equity', 'dollar_loss_on_equity'], max_rows=10)}
        </div>
        """,
    )

    allocation = section(
        "06",
        "Allocation, Contribution and Diversification",
        "Realised exposure, asset weights, return contribution and asset correlation.",
        f"""
        <div class="chart-grid single">{chart_card(charts['allocation_heatmap'], 'Allocation Heatmap')}</div>
        <div class="chart-grid two">
          {chart_card(charts['exposure_cash'], 'Exposure and Cash')}
          {chart_card(charts['average_weights'], 'Average Weights')}
          {chart_card(charts['contribution'], 'Return Contribution')}
          {chart_card(charts['asset_corr'], 'Asset Correlation')}
        </div>
        <div class="two-col">
          {table_html(data['ticker_attr'], 'Ticker Attribution', columns=['ticker', 'average_weight', 'max_weight', 'pct_months_held', 'total_contribution', 'contribution_share_of_strategy_sum', 'hit_rate_when_held', 'avg_return_when_held', 'avg_return_when_not_held', 'selected_minus_rejected_return'], max_rows=12)}
          {table_html(pd.DataFrame([concentration.to_dict()]) if not concentration.empty else pd.DataFrame(), 'Contribution Concentration', max_rows=5)}
        </div>
        """,
    )

    signals = section(
        "07",
        "Signal Diagnostics",
        "Selection behaviour, feature IC and score-quality diagnostics.",
        f"""
        <div class="chart-grid two">
          {chart_card(charts['selected_vs_rejected'], 'Selected vs Rejected')}
          {chart_card(charts['feature_ic'], 'Feature Information Coefficient')}
        </div>
        <div class="two-col">
          {table_html(data['feature_attr'], 'Feature Attribution', columns=['feature', 'average_cross_sectional_rank_ic', 'average_cross_sectional_ic', 'full_sample_corr_with_forward_return', 'top_decile_forward_return', 'bottom_decile_forward_return', 'decile_spread', 'observations'], max_rows=12)}
          {table_html(data['score_threshold'], 'Score Threshold Diagnostics', max_rows=12)}
        </div>
        """,
    )

    implementation = section(
        "08",
        "Costs, Turnover and Implementation",
        "Execution assumptions, trading intensity, scenario results and flagged diagnostics.",
        f"""
        <div class="two-col">
          {table_html(data['costs'], 'Cost Summary', columns=['strategy', 'total_transaction_cost', 'total_transaction_cost_drag', 'annualised_transaction_cost_drag', 'average_daily_turnover', 'annualised_turnover'], max_rows=8)}
          {table_html(data['scenario_summary'], 'V3 Scenario Summary', columns=['scenario', 'cost_scenario', 'execution_delay_days', 'final_equity', 'cagr', 'sharpe', 'sortino', 'max_drawdown', 'total_cost_drag', 'annualised_turnover'], max_rows=12)}
        </div>
        {table_html(data['red_flags'], 'Diagnostics Flags', max_rows=20)}
        """,
    )

    missing = build_missing_outputs(data)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Commodity Strategy Research Pack</title>
  <style>
    :root {{
      --paper: {COLORS["paper"]};
      --panel: {COLORS["panel"]};
      --panel-alt: {COLORS["panel_alt"]};
      --ink: {COLORS["ink"]};
      --muted: {COLORS["muted"]};
      --faint: {COLORS["faint"]};
      --rule: {COLORS["rule"]};
      --rule-soft: {COLORS["rule_soft"]};
      --navy: {COLORS["navy"]};
      --blue: {COLORS["blue"]};
      --green: {COLORS["green"]};
      --amber: {COLORS["amber"]};
      --red: {COLORS["red"]};
      --gold: {COLORS["gold"]};
    }}

    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: {COLORS["html_body_bg"]};
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}

    .page {{ max-width: 1480px; margin: 0 auto; padding: 34px 34px 70px; }}

    .hero {{
      border: 1px solid var(--rule);
      background: {COLORS["html_hero_bg"]};
      border-radius: 28px;
      padding: 34px 38px;
      box-shadow: {COLORS["html_hero_shadow"]};
      position: relative;
      overflow: hidden;
    }}
    .hero:before {{
      content: "";
      position: absolute;
      inset: 0;
      background: {COLORS["html_hero_overlay"]};
      pointer-events: none;
    }}
    .hero-content {{ position: relative; z-index: 1; }}
    .eyebrow {{ letter-spacing: .14em; text-transform: uppercase; color: var(--blue); font-size: 12px; font-weight: 800; }}
    h1 {{ margin: 8px 0 8px; font-size: clamp(34px, 5vw, 64px); line-height: .98; letter-spacing: -0.045em; }}
    .subtitle {{ max-width: 980px; color: var(--muted); font-size: 17px; margin: 0 0 24px; }}
    .meta-row {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .pill {{
      border: 1px solid var(--rule);
      background: {COLORS["html_pill_bg"]};
      color: var(--ink);
      padding: 8px 11px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
    }}

    .notice {{
      margin-top: 18px;
      border-radius: 18px;
      border: 1px solid var(--rule);
      background: {COLORS["html_notice_bg"]};
      padding: 16px 18px;
      color: var(--ink);
    }}
    .notice ul {{ margin: 8px 0 0; }}

    .section {{ margin-top: 32px; }}
    .section-header {{ margin-bottom: 14px; display: flex; align-items: end; justify-content: space-between; gap: 20px; }}
    .section-kicker {{ color: var(--gold); font-weight: 900; font-size: 12px; letter-spacing: .16em; text-transform: uppercase; }}
    h2 {{ font-size: 27px; letter-spacing: -.025em; margin: 3px 0 2px; }}
    .section-header p {{ margin: 0; color: var(--muted); }}

    .metric-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .metric-card {{
      background: var(--panel);
      border: 1px solid var(--rule);
      border-radius: 20px;
      padding: 18px 18px 16px;
      box-shadow: {COLORS["html_shadow"]};
        }}
    .metric-top {{ display: flex; justify-content: space-between; gap: 10px; align-items: start; color: var(--muted); font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: .06em; }}
    .metric-value {{ margin-top: 16px; font-size: 31px; font-weight: 900; letter-spacing: -.04em; color: var(--ink); }}
    .metric-label {{ margin-top: 4px; color: var(--muted); font-size: 13px; }}

    .badge {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 4px 8px; font-size: 10px; font-weight: 900; letter-spacing: .06em; border: 1px solid; white-space: nowrap; }}
    .badge.pass {{ color: var(--green); border-color: rgba(47,111,78,.35); background: rgba(47,111,78,.08); }}
    .badge.watch {{ color: var(--amber); border-color: rgba(183,121,31,.38); background: rgba(183,121,31,.10); }}
    .badge.fail, .badge.review {{ color: var(--red); border-color: rgba(155,34,38,.35); background: rgba(155,34,38,.08); }}
    .badge.neutral, .badge.incomplete {{ color: var(--muted); border-color: var(--rule); background: rgba(255,255,255,.45); }}

    .chart-grid {{ display: grid; gap: 16px; margin-bottom: 16px; }}
    .chart-grid.single {{ grid-template-columns: 1fr; }}
    .chart-grid.two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .two-col {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-bottom: 16px; }}

    .chart-card, .table-card {{
      background: var(--panel);
      border: 1px solid var(--rule);
      border-radius: 22px;
      padding: 16px;
      box-shadow: {COLORS["html_shadow"]};
          }}
    .chart-title, .table-title {{ font-weight: 900; letter-spacing: -.015em; margin: 1px 0 12px; color: var(--navy); }}
    .chart-card img {{ width: 100%; display: block; border-radius: 14px; border: 1px solid var(--rule-soft); background: var(--paper); }}
    .chart-caption {{ color: var(--muted); margin-top: 10px; font-size: 13px; }}
    .missing-text {{ color: var(--muted); padding: 18px; border: 1px dashed var(--rule); border-radius: 14px; background: var(--panel-alt); }}

    .table-scroll {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{
      text-align: left;
      padding: 10px 9px;
      color: var(--blue);
      border-bottom: 1px solid var(--rule);
      background: var(--panel-alt);
      white-space: nowrap;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .05em;
    }}
    td {{ padding: 9px; border-bottom: 1px solid var(--rule-soft); white-space: nowrap; }}
    tr:nth-child(even) td {{ background: {COLORS["html_row_even"]}; }}
    td.pos {{ color: var(--green); font-weight: 750; }}
    td.neg {{ color: var(--red); font-weight: 750; }}

    footer {{ margin-top: 34px; color: var(--muted); font-size: 12px; text-align: center; }}

    @media (max-width: 1100px) {{
      .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .chart-grid.two, .two-col {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 720px) {{
      .page {{ padding: 18px; }}
      .metric-grid {{ grid-template-columns: 1fr; }}
      .hero {{ padding: 26px 22px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <div class="hero-content">
        <div class="eyebrow">Commodity Allocation and Risk Engine</div>
        <h1>Strategy Research Pack</h1>
        <p class="subtitle">Backtest V3, walk-forward validation, stress testing, tail-risk diagnostics, allocation analysis and signal attribution in one consolidated report.</p>
        <div class="meta-row">
          <span class="pill">Generated {esc(datetime.now().strftime('%Y-%m-%d %H:%M'))}</span>
          <span class="pill">Backtest: {esc(model_start)} → {esc(model_end)}</span>
          <span class="pill">Initial capital: {fmt_money(INITIAL_CAPITAL)}</span>
          <span class="pill">Trading days/year: {TRADING_DAYS_PER_YEAR}</span>
        </div>
      </div>
    </header>

    {missing}
    {overview}
    {performance}
    {walk_forward}
    {stress}
    {risk}
    {allocation}
    {signals}
    {implementation}

    <footer>Generated from files in {esc(str(RESULTS_DIR))}. This report is a research and validation artifact; it does not alter strategy logic.</footer>
  </main>
</body>
</html>"""
    return html_doc


def generate_report(open_browser: bool = True, theme: str | None = None) -> Path:
    select_report_theme(theme or DEFAULT_REPORT_THEME)
    ensure_dirs()
    setup_plot_style()

    data = load_all_data()
    charts = generate_all_charts(data)
    report = build_report_html(data, charts)

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("\n========== FINAL STRATEGY REPORT ==========")
    print(f"Theme: {REPORT_THEME}")
    print(f"Saved report: {REPORT_PATH}")
    print(f"Saved charts: {CHARTS_DIR}")

    if open_browser:
        try:
            webbrowser.open(REPORT_PATH.resolve().as_uri())
        except Exception as exc:
            print(f"Could not open browser automatically: {exc}")

    return REPORT_PATH

def main() -> None:
    open_browser = True
    theme: str | None = None

    args = [a.strip().lower() for a in sys.argv[1:]]

    if any(a in {"--no-open", "no-open", "false"} for a in args):
        open_browser = False

    for i, arg in enumerate(args):
        if arg in {"light", "dark"}:
            theme = arg
        elif arg.startswith("--theme="):
            theme = arg.split("=", 1)[1].strip()
        elif arg == "--theme" and i + 1 < len(args):
            theme = args[i + 1].strip()

    generate_report(open_browser=open_browser, theme=theme)


if __name__ == "__main__":
    main()
