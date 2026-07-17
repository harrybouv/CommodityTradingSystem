from __future__ import annotations

import sys
import json
import html
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

THIS_FILE = Path(__file__).resolve()
RESEARCH_DIR = THIS_FILE.parent
COMMODITY_ROOT = RESEARCH_DIR.parent
PROJECT_ROOT = COMMODITY_ROOT.parent

for path in [PROJECT_ROOT, COMMODITY_ROOT, RESEARCH_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_INPUT_DIR = COMMODITY_ROOT / "results" / "backtest_V3" / "diagnostics"
DEFAULT_OUTPUT_FILE = DEFAULT_INPUT_DIR / "report.html"


# ============================================================
# FORMAT HELPERS
# ============================================================

PCT_COL_HINTS = [
    "return", "cagr", "vol", "drawdown", "weight", "cash", "exposure",
    "turnover", "cost_drag", "contribution", "alpha", "beta",
    "hit_rate", "win_rate", "corr", "ic", "sharpe", "sortino", "calmar",
]

MONEY_COL_HINTS = [
    "equity", "capital", "value", "notional", "cost", "dollar",
]


def esc(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return "N/A"
    return html.escape(str(x))


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def fmt_num(x: Any, digits: int = 2) -> str:
    val = safe_float(x)
    if val is None:
        return "N/A"
    return f"{val:,.{digits}f}"


def fmt_pct(x: Any, digits: int = 2) -> str:
    val = safe_float(x)
    if val is None:
        return "N/A"
    return f"{val:.{digits}%}"


def fmt_money(x: Any, digits: int = 2) -> str:
    val = safe_float(x)
    if val is None:
        return "N/A"
    return f"${val:,.{digits}f}"


def fmt_value(value: Any, col: str = "") -> str:
    col_l = str(col).lower()

    if isinstance(value, str):
        return esc(value)

    val = safe_float(value)

    if val is None:
        return "N/A"

    if "date" in col_l or "start" in col_l or "end" in col_l:
        return esc(value)

    # These are ratios/returns usually stored as decimals.
    if any(h in col_l for h in PCT_COL_HINTS):
        # Do not format Sharpe/Sortino/Calmar/Beta/IC as percentages.
        if any(h in col_l for h in ["sharpe", "sortino", "calmar", "beta", "ic", "corr", "r_squared"]):
            return f"{val:,.2f}"
        return f"{val:.2%}"

    if any(h in col_l for h in MONEY_COL_HINTS):
        return f"${val:,.2f}"

    if abs(val) >= 1000:
        return f"{val:,.0f}"

    return f"{val:,.3f}"


def pct_class(x: Any) -> str:
    val = safe_float(x)
    if val is None:
        return "muted"
    if val > 0:
        return "pos"
    if val < 0:
        return "neg"
    return "neutral"


def rel_path(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        try:
            return path.resolve().relative_to(base.parent.resolve()).as_posix()
        except Exception:
            return path.as_posix()


# ============================================================
# FILE LOADING
# ============================================================

def load_manifest(input_dir: Path) -> dict[str, Any]:
    manifest_path = input_dir / "summary" / "diagnostics_manifest.json"

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback if manifest does not exist.
    tables_dir = input_dir / "tables"
    charts_dir = input_dir / "charts"
    summary_dir = input_dir / "summary"

    tables = {
        p.stem: str(p)
        for p in tables_dir.glob("*.csv")
    } if tables_dir.exists() else {}

    charts = {
        p.stem: str(p)
        for p in charts_dir.glob("*.png")
    } if charts_dir.exists() else {}

    return {
        "base_dir": str(input_dir),
        "tables_dir": str(tables_dir),
        "charts_dir": str(charts_dir),
        "summary_dir": str(summary_dir),
        "tables": tables,
        "charts": charts,
        "summary_text": str(summary_dir / "summary.txt"),
        "red_flags": str(summary_dir / "red_flags.csv"),
    }


def read_csv_safe(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()

    p = Path(path)

    if not p.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(p)

        # Common pattern from pandas saving an index.
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "name"})

        return df
    except Exception:
        return pd.DataFrame()


def get_table(manifest: dict[str, Any], name: str) -> pd.DataFrame:
    tables = manifest.get("tables", {})
    path = tables.get(name)
    return read_csv_safe(path)


def get_chart_path(manifest: dict[str, Any], key: str) -> Optional[Path]:
    charts = manifest.get("charts", {})
    path = charts.get(key)

    if path is None:
        return None

    p = Path(path)

    if p.exists():
        return p

    return None


# ============================================================
# HTML COMPONENTS
# ============================================================

def card(title: str, value: str, subtitle: str = "", extra_class: str = "") -> str:
    return f"""
    <div class="metric-card {extra_class}">
      <div class="metric-label">{esc(title)}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-subtitle">{esc(subtitle)}</div>
    </div>
    """


def section(title: str, body: str, subtitle: str = "") -> str:
    sub = f'<div class="section-subtitle">{esc(subtitle)}</div>' if subtitle else ""

    return f"""
    <section class="panel">
      <div class="panel-header">
        <span class="toggle">▣</span>
        <h2>{esc(title)}</h2>
      </div>
      {sub}
      <div class="panel-body">
        {body}
      </div>
    </section>
    """


def note_box(title: str, text: str, kind: str = "note") -> str:
    return f"""
    <div class="note-box {kind}">
      <div class="note-title">{esc(title)}</div>
      <div class="note-text">{esc(text)}</div>
    </div>
    """


def image_block(
    manifest: dict[str, Any],
    key: str,
    title: str,
    caption: str,
    report_dir: Path,
) -> str:
    p = get_chart_path(manifest, key)

    if p is None:
        return f"""
        <div class="chart-card missing">
          <div class="chart-title">{esc(title)}</div>
          <div class="missing-text">Chart not available: {esc(key)}</div>
        </div>
        """

    src = rel_path(p, report_dir)

    return f"""
    <div class="chart-card">
      <div class="chart-title">{esc(title)}</div>
      <img src="{esc(src)}" alt="{esc(title)}">
      <div class="chart-caption">{esc(caption)}</div>
    </div>
    """


def df_to_html_table(
    df: pd.DataFrame,
    max_rows: int = 12,
    columns: Optional[list[str]] = None,
    title: str = "",
    table_class: str = "",
) -> str:
    if df is None or df.empty:
        return f"""
        <div class="table-wrap empty">
          <div class="table-title">{esc(title)}</div>
          <div class="empty-msg">No table data available.</div>
        </div>
        """

    show = df.copy()

    if columns:
        existing = [c for c in columns if c in show.columns]
        if existing:
            show = show[existing]

    if max_rows is not None and len(show) > max_rows:
        show = show.head(max_rows)

    header = "".join(f"<th>{esc(c)}</th>" for c in show.columns)

    rows = []

    for _, row in show.iterrows():
        cells = []

        for col in show.columns:
            val = row[col]
            rendered = fmt_value(val, col)

            cls = ""
            col_l = str(col).lower()

            if any(x in col_l for x in ["return", "contribution", "drawdown", "p/l", "cost", "alpha"]):
                cls = pct_class(val)

            cells.append(f'<td class="{cls}">{rendered}</td>')

        rows.append("<tr>" + "".join(cells) + "</tr>")

    title_html = f'<div class="table-title">{esc(title)}</div>' if title else ""

    return f"""
    <div class="table-wrap {table_class}">
      {title_html}
      <table>
        <thead><tr>{header}</tr></thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    """


# ============================================================
# INTERPRETATION HELPERS
# ============================================================

def extract_headline_metrics(headline: pd.DataFrame) -> dict[str, Any]:
    if headline.empty:
        return {}

    row = headline.iloc[0].to_dict()

    return {
        "cagr": row.get("cagr"),
        "sharpe": row.get("sharpe"),
        "sortino": row.get("sortino"),
        "max_drawdown": row.get("max_drawdown"),
        "calmar": row.get("calmar"),
        "win_rate": row.get("win_rate"),
        "total_return": row.get("total_return"),
        "ann_vol": row.get("ann_vol") or row.get("annualised_volatility"),
    }


def build_executive_summary(
    headline: pd.DataFrame,
    exposure_summary: pd.DataFrame,
    concentration: pd.DataFrame,
    cost_summary: pd.DataFrame,
) -> str:
    m = extract_headline_metrics(headline)

    avg_exposure = None
    avg_cash = None
    eff_pos = None

    if not exposure_summary.empty:
        if {"name", "value"}.issubset(exposure_summary.columns):
            exp_map = dict(zip(exposure_summary["name"], exposure_summary["value"]))
            avg_exposure = exp_map.get("average_exposure")
            avg_cash = exp_map.get("average_cash")
            eff_pos = exp_map.get("average_effective_number_positions")

    top_share = None
    top_two_share = None

    if not concentration.empty and {"name", "value"}.issubset(concentration.columns):
        con_map = dict(zip(concentration["name"], concentration["value"]))
        top_share = con_map.get("top_asset_abs_contribution_share")
        top_two_share = con_map.get("top_two_asset_abs_contribution_share")

    annual_turnover = None
    total_cost = None

    if not cost_summary.empty and {"name", "value"}.issubset(cost_summary.columns):
        cost_map = dict(zip(cost_summary["name"], cost_summary["value"]))
        annual_turnover = cost_map.get("annualised_turnover")
        total_cost = cost_map.get("total_cost")

    return f"""
    <div class="metric-grid">
      {card("CAGR", fmt_pct(m.get("cagr")), "annualised strategy return")}
      {card("Sharpe", fmt_num(m.get("sharpe")), "risk-adjusted return")}
      {card("Max drawdown", fmt_pct(m.get("max_drawdown")), "worst peak-to-trough loss", "danger")}
      {card("Sortino", fmt_num(m.get("sortino")), "downside-risk adjusted")}
      {card("Average exposure", fmt_pct(avg_exposure), "how invested the model usually is")}
      {card("Average cash", fmt_pct(avg_cash), "defensive allocation")}
      {card("Effective positions", fmt_num(eff_pos), "diversification quality")}
      {card("Top asset share", fmt_pct(top_share), "absolute contribution concentration")}
      {card("Top 2 asset share", fmt_pct(top_two_share), "concentration check")}
      {card("Annual turnover", fmt_pct(annual_turnover), "implementation churn")}
     {card("Total cost", fmt_money(total_cost), "estimated cumulative cost drag")}
     {card("Win rate", fmt_pct(m.get("win_rate")), "positive periods")}
    </div>
    """


def build_red_flags_section(red_flags: pd.DataFrame) -> str:
    if red_flags.empty:
        return note_box(
            "No red flags file found",
            "Diagnostics did not produce a red flag table. This is not proof of robustness; inspect charts manually.",
            "warn",
        )

    rows = []

    for _, r in red_flags.iterrows():
        severity = str(r.get("severity", "Info"))
        category = str(r.get("category", "General"))
        message = str(r.get("message", ""))

        sev_class = severity.lower()

        rows.append(f"""
        <div class="flag {sev_class}">
          <div class="flag-top">
            <span class="flag-severity">{esc(severity)}</span>
            <span class="flag-category">{esc(category)}</span>
          </div>
          <div class="flag-message">{esc(message)}</div>
        </div>
        """)

    return f'<div class="flag-grid">{"".join(rows)}</div>'


def build_ticker_interpretation(ticker_attr: pd.DataFrame) -> str:
    if ticker_attr.empty:
        return note_box(
            "Ticker attribution unavailable",
            "No ticker contribution table was found. This usually means asset returns or weights were not saved correctly.",
            "warn",
        )

    top = ticker_attr.sort_values("total_contribution", ascending=False).head(1)

    if top.empty:
        return ""

    row = top.iloc[0]
    ticker = row.get("ticker", "N/A")
    contribution = row.get("total_contribution")
    avg_weight = row.get("average_weight")
    held = row.get("pct_months_held")
    selected_gap = row.get("selected_minus_rejected_return")

    text = (
        f"Top contributor is {ticker}. It contributed {fmt_pct(contribution)} "
        f"with an average weight of {fmt_pct(avg_weight)} and was held {fmt_pct(held)} of periods. "
        f"The selected-vs-rejected return gap is {fmt_pct(selected_gap)}. "
        "If this gap is weak or negative, the allocation logic is not clearly selecting better assets."
    )

    return note_box("Ticker attribution read", text, "note")


def build_feature_interpretation(feature_attr: pd.DataFrame) -> str:
    if feature_attr.empty:
        return note_box(
            "Feature attribution unavailable",
            "No feature attribution table was produced. This likely means historical component scores are not being exported yet. This is the biggest remaining diagnostic gap.",
            "warn",
        )

    sort_col = "average_cross_sectional_rank_ic"

    if sort_col not in feature_attr.columns:
        return note_box(
            "Feature attribution partial",
            "Feature table exists, but rank IC was not found. Feature predictiveness is therefore not fully tested.",
            "warn",
        )

    top = feature_attr.sort_values(sort_col, ascending=False).head(1).iloc[0]
    bottom = feature_attr.sort_values(sort_col, ascending=True).head(1).iloc[0]

    text = (
        f"Best feature by average rank IC is {top.get('feature')} at {fmt_num(top.get(sort_col))}. "
        f"Weakest feature is {bottom.get('feature')} at {fmt_num(bottom.get(sort_col))}. "
        "Rank IC should not be treated as gospel, but if most features are near zero, the model may be more theoretical than predictive."
    )

    return note_box("Feature quality read", text, "note")


def build_regime_interpretation(regime_summary: pd.DataFrame) -> str:
    if regime_summary.empty:
        return note_box(
            "Regime diagnostics unavailable",
            "No regime table was produced. Price-derived regime classification may have failed due to missing asset returns.",
            "warn",
        )

    if "avg_period_return" not in regime_summary.columns:
        return ""

    best = regime_summary.sort_values("avg_period_return", ascending=False).head(1).iloc[0]
    worst = regime_summary.sort_values("avg_period_return", ascending=True).head(1).iloc[0]

    text = (
        f"Best regime by average period return is {best.get('regime')} at {fmt_pct(best.get('avg_period_return'))}. "
        f"Worst regime is {worst.get('regime')} at {fmt_pct(worst.get('avg_period_return'))}. "
        "This is where you test whether returns come from a broad process or from one favourable market environment."
    )

    return note_box("Regime read", text, "note")


# ============================================================
# REPORT BUILDERS
# ============================================================

def build_header() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""
    <header class="hero">
      <div>
        <h1>COMMODITY ENGINE</h1>
        <div class="subtitle">V3 DIAGNOSTICS / BACKTEST AUDIT REPORT</div>
      </div>

      <div class="status-box">
        <div class="status-row"><span>MODE</span><b>LOCAL RESEARCH</b></div>
        <div class="status-row"><span>REPORT</span><b>DIAGNOSTICS</b></div>
        <div class="status-row"><span>ENGINE</span><b>BACKTEST V3</b></div>
        <div class="status-row"><span>BUILT</span><b>{esc(now)}</b></div>
      </div>
    </header>
    """


def build_performance_section(
    manifest: dict[str, Any],
    report_dir: Path,
    headline: pd.DataFrame,
    benchmark: pd.DataFrame,
    monthly: pd.DataFrame,
    drawdowns: pd.DataFrame,
) -> str:
    body = ""

    body += image_block(
        manifest,
        "equity_vs_benchmarks",
        "Equity curve vs benchmarks",
        "This is the first credibility test: the model must beat simple alternatives on risk-adjusted terms, not just look busy.",
        report_dir,
    )

    body += image_block(
        manifest,
        "drawdown_vs_benchmarks",
        "Drawdown vs benchmarks",
        "The key question is not just return. It is whether the system reduces bad periods versus simple commodity exposure.",
        report_dir,
    )

    body += image_block(
        manifest,
        "rolling_metrics",
        "Rolling metrics",
        "Checks whether the system was persistently useful or only carried by one favourable window.",
        report_dir,
    )

    body += df_to_html_table(
        benchmark,
        max_rows=8,
        title="Benchmark comparison",
    )

    body += df_to_html_table(
        drawdowns,
        max_rows=8,
        title="Worst drawdown periods",
    )

    body += image_block(
        manifest,
        "monthly_return_heatmap",
        "Monthly return heatmap",
        "Useful for spotting clustered pain, one-off lucky periods, and whether returns are spread across time.",
        report_dir,
    )

    return section(
        "Performance & Benchmarks",
        body,
        "Headline returns, drawdowns, rolling quality, and comparison against simple baselines.",
    )


def build_allocation_section(
    manifest: dict[str, Any],
    report_dir: Path,
    exposure_summary: pd.DataFrame,
) -> str:
    body = ""

    body += image_block(
        manifest,
        "allocation_heatmap",
        "Allocation heatmap",
        "Shows whether the model is genuinely rotating or simply sitting in one asset/cash most of the time.",
        report_dir,
    )

    body += image_block(
        manifest,
        "exposure_cash",
        "Exposure, cash and effective diversification",
        "Cash is part of the strategy. This checks whether the model is risk-managing or just underinvested.",
        report_dir,
    )

    body += df_to_html_table(
        exposure_summary,
        max_rows=20,
        title="Exposure summary",
    )

    return section(
        "Allocation, Cash & Diversification",
        body,
        "Human-readable view of how much risk the system is actually taking.",
    )


def build_ticker_section(
    manifest: dict[str, Any],
    report_dir: Path,
    ticker_attr: pd.DataFrame,
    concentration: pd.DataFrame,
) -> str:
    body = ""

    body += build_ticker_interpretation(ticker_attr)

    body += image_block(
        manifest,
        "ticker_contribution",
        "Ticker contribution",
        "If one asset dominates this chart, the system may be narrower than the headline suggests.",
        report_dir,
    )

    body += image_block(
        manifest,
        "average_weights",
        "Average weights",
        "Compares contribution against capital actually allocated. High contribution with low weight is strong; high contribution with high weight may just be exposure.",
        report_dir,
    )

    body += image_block(
        manifest,
        "selected_vs_rejected",
        "Selected vs rejected returns",
        "This is a direct sanity check: assets selected by the model should generally do better than when they are rejected.",
        report_dir,
    )

    body += df_to_html_table(
        ticker_attr,
        max_rows=12,
        columns=[
            "ticker",
            "average_weight",
            "max_weight",
            "pct_months_held",
            "total_contribution",
            "contribution_share_of_strategy_sum",
            "hit_rate_when_held",
            "avg_return_when_held",
            "avg_return_when_not_held",
            "selected_minus_rejected_return",
        ],
        title="Ticker attribution summary",
    )

    body += df_to_html_table(
        concentration,
        max_rows=12,
        title="Return concentration",
    )

    return section(
        "Ticker Attribution",
        body,
        "Identifies which commodities actually drove the backtest.",
    )


def build_feature_section(
    manifest: dict[str, Any],
    report_dir: Path,
    feature_attr: pd.DataFrame,
    score_threshold: pd.DataFrame,
) -> str:
    body = ""

    body += build_feature_interpretation(feature_attr)

    body += image_block(
        manifest,
        "feature_ic",
        "Feature rank IC",
        "Tests whether individual score components have useful relationship with next-period returns.",
        report_dir,
    )

    body += image_block(
        manifest,
        "feature_deciles",
        "Feature decile forward returns",
        "A good feature should show some monotonic improvement from low deciles to high deciles. Noisy charts here are a warning.",
        report_dir,
    )

    body += image_block(
        manifest,
        "score_bucket_returns",
        "Final score bucket returns",
        "Checks whether the final score threshold actually separates stronger assets from weaker assets.",
        report_dir,
    )

    body += df_to_html_table(
        feature_attr,
        max_rows=14,
        columns=[
            "feature",
            "average_score",
            "average_score_when_held",
            "average_score_when_not_held",
            "full_sample_corr_with_forward_return",
            "average_cross_sectional_ic",
            "average_cross_sectional_rank_ic",
            "top_decile_forward_return",
            "bottom_decile_forward_return",
            "decile_spread",
            "observations",
        ],
        title="Feature attribution summary",
    )

    body += df_to_html_table(
        score_threshold,
        max_rows=14,
        title="Score threshold diagnostics",
    )

    return section(
        "Feature & Score Diagnostics",
        body,
        "This is the evidence layer for whether your scoring system is predictive or merely plausible.",
    )


def build_regime_section(
    manifest: dict[str, Any],
    report_dir: Path,
    regime_summary: pd.DataFrame,
) -> str:
    body = ""

    body += build_regime_interpretation(regime_summary)

    body += image_block(
        manifest,
        "regime_summary",
        "Regime contribution",
        "Shows which market states pay the strategy and which punish it.",
        report_dir,
    )

    body += df_to_html_table(
        regime_summary,
        max_rows=18,
        columns=[
            "regime",
            "periods",
            "avg_period_return",
            "cagr",
            "sharpe",
            "sortino",
            "max_drawdown",
            "win_rate",
            "average_exposure",
            "average_cash",
            "best_ticker",
            "worst_ticker",
        ],
        title="Regime summary",
    )

    return section(
        "Regime Diagnostics",
        body,
        "Price-derived regimes. This does not require external macro data and is enough to expose major regime dependence.",
    )


def build_correlation_section(
    manifest: dict[str, Any],
    report_dir: Path,
    corr_summary: pd.DataFrame,
) -> str:
    body = ""

    body += image_block(
        manifest,
        "asset_correlation_heatmap",
        "Asset correlation matrix",
        "Checks whether apparent diversification is real or just different names for the same macro exposure.",
        report_dir,
    )

    body += image_block(
        manifest,
        "feature_correlation_heatmap",
        "Feature correlation matrix",
        "Highly correlated features are not independent signals. They may still help, but they should not be oversold.",
        report_dir,
    )

    body += image_block(
        manifest,
        "held_correlation",
        "Held-asset rolling correlation",
        "Tracks whether the actual portfolio becomes more concentrated during stress.",
        report_dir,
    )

    body += df_to_html_table(
        corr_summary,
        max_rows=20,
        title="Strategy correlation / beta summary",
    )

    return section(
        "Correlation & Hidden Concentration",
        body,
        "The section that stops this becoming a disguised gold/inflation bet without you noticing.",
    )


def build_cost_section(
    manifest: dict[str, Any],
    report_dir: Path,
    cost_summary: pd.DataFrame,
) -> str:
    body = ""

    body += image_block(
        manifest,
        "turnover_costs",
        "Turnover and cost drag",
        "Measures whether the system is implementable or quietly eating returns through churn.",
        report_dir,
    )

    body += image_block(
        manifest,
        "gross_vs_net",
        "Gross vs net equity",
        "If this gap becomes large, the strategy is too fragile to execution assumptions.",
        report_dir,
    )

    body += df_to_html_table(
        cost_summary,
        max_rows=20,
        title="Turnover and cost summary",
    )

    return section(
        "Turnover, Costs & Realism",
        body,
        "Bridge between spreadsheet backtest and something that can be paper-traded or implemented.",
    )


def build_decision_section(
    manifest: dict[str, Any],
    report_dir: Path,
    best_decisions: pd.DataFrame,
    worst_decisions: pd.DataFrame,
) -> str:
    body = ""

    body += image_block(
        manifest,
        "decision_outcomes",
        "Decision outcomes",
        "Checks whether entries, exits and resizes are adding value after the fact.",
        report_dir,
    )

    body += df_to_html_table(
        best_decisions,
        max_rows=10,
        columns=[
            "date",
            "ticker",
            "decision_type",
            "new_weight",
            "final_score",
            "main_positive_feature",
            "main_negative_feature",
            "forward_return",
            "next_period_contribution",
        ],
        title="Best decisions",
    )

    body += df_to_html_table(
        worst_decisions,
        max_rows=10,
        columns=[
            "date",
            "ticker",
            "decision_type",
            "new_weight",
            "final_score",
            "main_positive_feature",
            "main_negative_feature",
            "forward_return",
            "next_period_contribution",
        ],
        title="Worst decisions",
    )

    return section(
        "Decision Audit",
        body,
        "Best and worst decisions. This is useful for memos and for finding systematic model failure modes.",
    )


def build_appendix_section(manifest: dict[str, Any], report_dir: Path) -> str:
    tables = manifest.get("tables", {})
    charts = manifest.get("charts", {})

    table_links = []

    for name, path in sorted(tables.items()):
        p = Path(path)
        if p.exists():
            table_links.append(
                f'<a class="file-link" href="{esc(rel_path(p, report_dir))}">{esc(name)}.csv</a>'
            )

    chart_links = []

    for name, path in sorted(charts.items()):
        p = Path(path)
        if p.exists():
            chart_links.append(
                f'<a class="file-link" href="{esc(rel_path(p, report_dir))}">{esc(name)}.png</a>'
            )

    body = f"""
    <div class="appendix-grid">
      <div>
        <h3>Tables</h3>
        <div class="file-list">{''.join(table_links) if table_links else '<span class="muted">No tables found.</span>'}</div>
      </div>
      <div>
        <h3>Charts</h3>
        <div class="file-list">{''.join(chart_links) if chart_links else '<span class="muted">No charts found.</span>'}</div>
      </div>
    </div>
    """

    return section(
        "Appendix / Raw Files",
        body,
        "Raw diagnostic outputs. Use these for auditing, not for day-to-day interpretation.",
    )


# ============================================================
# CSS
# ============================================================

def build_css() -> str:
    return """
    <style>
      :root {
        --bg: #050505;
        --panel: #111111;
        --panel-2: #151515;
        --border: #3d3d3d;
        --border-soft: #242424;
        --text: #f2f2f2;
        --muted: #9a9a9a;
        --dim: #686868;
        --green: #b8d7b1;
        --blue: #89aabd;
        --yellow: #d9c68b;
        --red: #e69a9a;
        --orange: #d6a05f;
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        background:
          linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px),
          var(--bg);
        background-size: 100% 4px, 4px 100%;
        color: var(--text);
        font-family: Consolas, "Courier New", ui-monospace, monospace;
        font-size: 14px;
        line-height: 1.45;
      }

      .page {
        max-width: 1880px;
        margin: 28px auto;
        padding: 0 16px 32px;
      }

      .shell {
        border: 1px solid var(--border);
        background: rgba(10, 10, 10, 0.96);
        box-shadow: 0 0 0 1px #111, 0 0 32px rgba(255,255,255,0.04);
      }

      .hero {
        display: flex;
        justify-content: space-between;
        gap: 32px;
        align-items: flex-end;
        padding: 32px;
        min-height: 265px;
        border-bottom: 1px solid var(--border);
      }

      h1 {
        font-family: Georgia, "Times New Roman", serif;
        font-size: clamp(38px, 4vw, 72px);
        line-height: 0.95;
        margin: 0;
        letter-spacing: 0.045em;
        text-shadow: 3px 3px 0 #555, 6px 6px 0 #1f1f1f;
      }

      .subtitle {
        margin-top: 22px;
        color: var(--muted);
        letter-spacing: 0.16em;
        font-size: 15px;
      }

      .status-box {
        min-width: 420px;
        border: 1px solid var(--border);
        background: var(--panel);
        padding: 16px;
      }

      .status-row {
        display: flex;
        justify-content: space-between;
        gap: 24px;
        border-bottom: 1px dotted var(--border-soft);
        padding: 6px 0;
      }

      .status-row:last-child {
        border-bottom: 0;
      }

      .status-row span {
        color: var(--muted);
      }

      .status-row b {
        color: var(--text);
        font-weight: 700;
      }

      .panel {
        margin: 18px;
        border: 1px solid var(--border);
        background: var(--panel);
      }

      .panel-header {
        display: flex;
        align-items: center;
        gap: 10px;
        border-bottom: 1px solid var(--border);
        padding: 10px 14px;
        background: var(--panel-2);
      }

      .panel-header h2 {
        margin: 0;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 16px;
        letter-spacing: 0.055em;
        text-transform: uppercase;
      }

      .toggle {
        color: var(--muted);
      }

      .section-subtitle {
        color: var(--muted);
        padding: 12px 16px 0;
        font-size: 13px;
      }

      .panel-body {
        padding: 16px;
      }

      .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(180px, 1fr));
        gap: 10px;
      }

      .metric-card {
        border: 1px solid var(--border);
        background: #090909;
        padding: 14px;
        min-height: 95px;
      }

      .metric-label {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 13px;
      }

      .metric-value {
        margin-top: 12px;
        font-size: 26px;
        font-weight: 800;
        color: var(--text);
      }

      .metric-card.danger .metric-value {
        color: var(--red);
      }

      .metric-subtitle {
        margin-top: 8px;
        color: var(--dim);
        font-size: 12px;
      }

      .chart-card {
        border: 1px solid var(--border);
        background: #080808;
        padding: 14px;
        margin-bottom: 16px;
      }

      .chart-title {
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 800;
        margin-bottom: 10px;
      }

      .chart-card img {
        width: 100%;
        display: block;
        border: 1px solid var(--border-soft);
        background: #050505;
      }

      .chart-caption {
        color: var(--muted);
        margin-top: 10px;
        font-size: 13px;
      }

      .missing-text,
      .empty-msg {
        color: var(--muted);
        padding: 24px;
        border: 1px dashed var(--border);
      }

      .note-box {
        border: 1px solid var(--border);
        padding: 12px 14px;
        margin-bottom: 16px;
        background: #090909;
      }

      .note-box.note {
        border-left: 4px solid var(--blue);
      }

      .note-box.warn {
        border-left: 4px solid var(--yellow);
      }

      .note-box.danger {
        border-left: 4px solid var(--red);
      }

      .note-title {
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 6px;
      }

      .note-text {
        color: var(--muted);
      }

      .table-wrap {
        border: 1px solid var(--border);
        background: #080808;
        margin: 16px 0;
        overflow-x: auto;
      }

      .table-title {
        padding: 10px 12px;
        border-bottom: 1px solid var(--border);
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }

      table {
        width: 100%;
        border-collapse: collapse;
        min-width: 760px;
      }

      th {
        text-align: left;
        padding: 10px;
        border-bottom: 1px solid var(--border);
        border-right: 1px solid var(--border-soft);
        background: var(--panel-2);
        color: var(--text);
        white-space: nowrap;
      }

      td {
        padding: 9px 10px;
        border-bottom: 1px solid var(--border-soft);
        border-right: 1px solid var(--border-soft);
        color: var(--text);
        white-space: nowrap;
      }

      tr:hover td {
        background: #131313;
      }

      .pos {
        color: var(--green);
      }

      .neg {
        color: var(--red);
      }

      .neutral {
        color: var(--text);
      }

      .muted {
        color: var(--muted);
      }

      .flag-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(260px, 1fr));
        gap: 10px;
      }

      .flag {
        border: 1px solid var(--border);
        padding: 12px;
        background: #090909;
      }

      .flag.high {
        border-left: 4px solid var(--red);
      }

      .flag.medium {
        border-left: 4px solid var(--yellow);
      }

      .flag.low {
        border-left: 4px solid var(--blue);
      }

      .flag.info {
        border-left: 4px solid var(--green);
      }

      .flag-top {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 8px;
      }

      .flag-severity {
        font-weight: 800;
        color: var(--text);
      }

      .flag-category {
        color: var(--muted);
      }

      .flag-message {
        color: var(--muted);
      }

      .appendix-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 18px;
      }

      .appendix-grid h3 {
        margin-top: 0;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }

      .file-list {
        display: grid;
        gap: 7px;
      }

      .file-link {
        color: var(--blue);
        text-decoration: none;
        border-bottom: 1px dotted var(--blue);
        width: fit-content;
      }

      .file-link:hover {
        color: var(--green);
        border-bottom-color: var(--green);
      }

      .footer {
        color: var(--dim);
        padding: 24px 32px 32px;
        border-top: 1px solid var(--border);
      }

      @media (max-width: 1100px) {
        .hero {
          flex-direction: column;
          align-items: stretch;
        }

        .status-box {
          min-width: 0;
        }

        .metric-grid {
          grid-template-columns: repeat(2, minmax(180px, 1fr));
        }

        .flag-grid,
        .appendix-grid {
          grid-template-columns: 1fr;
        }
      }

      @media (max-width: 640px) {
        .metric-grid {
          grid-template-columns: 1fr;
        }

        .panel {
          margin: 10px;
        }

        .hero {
          padding: 20px;
        }
      }
    </style>
    """


# ============================================================
# MAIN REPORT FUNCTION
# ============================================================

def build_diagnostics_report(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path | None = None,
) -> Path:
    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "report.html"

    manifest = load_manifest(input_dir)

    # Core tables
    headline = get_table(manifest, "headline_summary")
    benchmark = get_table(manifest, "benchmark_comparison")
    monthly = get_table(manifest, "monthly_returns")
    drawdowns = get_table(manifest, "worst_drawdown_periods")

    ticker_attr = get_table(manifest, "ticker_attribution")
    concentration = get_table(manifest, "return_concentration")
    exposure_summary = get_table(manifest, "exposure_summary")

    feature_attr = get_table(manifest, "feature_attribution")
    score_threshold = get_table(manifest, "score_threshold_diagnostics")

    regime_summary = get_table(manifest, "regime_summary")
    corr_summary = get_table(manifest, "correlation_summary")
    cost_summary = get_table(manifest, "turnover_cost_summary")

    best_decisions = get_table(manifest, "best_decisions")
    worst_decisions = get_table(manifest, "worst_decisions")
    red_flags = get_table(manifest, "red_flags")

    body = ""

    body += build_header()

    body += section(
        "Executive Summary",
        build_executive_summary(
            headline=headline,
            exposure_summary=exposure_summary,
            concentration=concentration,
            cost_summary=cost_summary,
        ),
        "Fast read. If these cards look weak, do not hide behind detailed charts.",
    )

    body += section(
        "Red Flags",
        build_red_flags_section(red_flags),
        "Automatic warnings. These are not final verdicts, but they tell you where to inspect brutally.",
    )

    body += build_performance_section(
        manifest=manifest,
        report_dir=output_dir,
        headline=headline,
        benchmark=benchmark,
        monthly=monthly,
        drawdowns=drawdowns,
    )

    body += build_allocation_section(
        manifest=manifest,
        report_dir=output_dir,
        exposure_summary=exposure_summary,
    )

    body += build_ticker_section(
        manifest=manifest,
        report_dir=output_dir,
        ticker_attr=ticker_attr,
        concentration=concentration,
    )

    body += build_feature_section(
        manifest=manifest,
        report_dir=output_dir,
        feature_attr=feature_attr,
        score_threshold=score_threshold,
    )

    body += build_regime_section(
        manifest=manifest,
        report_dir=output_dir,
        regime_summary=regime_summary,
    )

    body += build_correlation_section(
        manifest=manifest,
        report_dir=output_dir,
        corr_summary=corr_summary,
    )

    body += build_cost_section(
        manifest=manifest,
        report_dir=output_dir,
        cost_summary=cost_summary,
    )

    body += build_decision_section(
        manifest=manifest,
        report_dir=output_dir,
        best_decisions=best_decisions,
        worst_decisions=worst_decisions,
    )

    body += build_appendix_section(
        manifest=manifest,
        report_dir=output_dir,
    )

    body += """
    <div class="footer">
      Diagnostic report generated from V3 outputs. This report audits the strategy; it does not alter model logic.
    </div>
    """

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Commodity Engine — V3 Diagnostics Report</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {build_css()}
</head>
<body>
  <main class="page">
    <div class="shell">
      {body}
    </div>
  </main>
</body>
</html>
"""

    report_path.write_text(html_doc, encoding="utf-8")

    print(f"\nDiagnostics HTML report built:")
    print(f"  {report_path}")

    return report_path


# Alias names so backtest_V3.py can call whichever exists.
generate_diagnostics_report = build_diagnostics_report
generate_html_report = build_diagnostics_report


def main_from_folder(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path | None = None,
) -> Path:
    return build_diagnostics_report(input_dir=input_dir, output_dir=output_dir)


def main() -> None:
    if len(sys.argv) > 1:
        input_dir = Path(sys.argv[1])
    else:
        input_dir = DEFAULT_INPUT_DIR

    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = None

    build_diagnostics_report(input_dir=input_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()