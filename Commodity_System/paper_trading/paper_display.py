# Commodity_System/paper_trading/paper_display.py

from __future__ import annotations

import html as html_lib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

THIS_FILE = Path(__file__).resolve()
PAPER_TRADING_DIR = THIS_FILE.parent
COMMODITY_ROOT = THIS_FILE.parents[1]

for path in [COMMODITY_ROOT, PAPER_TRADING_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# ============================================================
# IMPORTS
# ============================================================

import paper_config as pc

try:
    from config import (
        UNIVERSE,
        MIN_SCORE_TO_HOLD,
        MAX_GROUP_WEIGHT,
    )
except ImportError:
    UNIVERSE = {}
    MIN_SCORE_TO_HOLD = 0.65
    MAX_GROUP_WEIGHT = {}


# ============================================================
# PATHS
# ============================================================

PAPER_STATE_DIR = getattr(pc, "PAPER_STATE_DIR", PAPER_TRADING_DIR / "state")
PAPER_REPORTS_DIR = getattr(pc, "PAPER_REPORTS_DIR", PAPER_TRADING_DIR / "reports")
PAPER_SNAPSHOTS_DIR = getattr(pc, "PAPER_SNAPSHOTS_DIR", PAPER_TRADING_DIR / "snapshots")

PAPER_DECISION_SNAPSHOT_DIR = getattr(
    pc,
    "PAPER_DECISION_SNAPSHOT_DIR",
    getattr(pc, "PAPER_DECISION_SNAPSHOTS_DIR", PAPER_SNAPSHOTS_DIR / "decisions"),
)
PAPER_FINAL_SCORE_SNAPSHOT_DIR = getattr(
    pc,
    "PAPER_FINAL_SCORE_SNAPSHOT_DIR",
    getattr(pc, "PAPER_FINAL_SCORE_SNAPSHOTS_DIR", PAPER_SNAPSHOTS_DIR / "final_scores"),
)

FINAL_SCORES_PATH = Path(
    getattr(
        pc,
        "FINAL_SCORES_PATH",
        getattr(
            pc,
            "FINAL_SCORES_SOURCE_PATH",
            COMMODITY_ROOT / "data" / "processed" / "final_scores.csv",
        ),
    )
)

PAPER_POSITIONS_PATH = getattr(
    pc,
    "PAPER_POSITIONS_PATH",
    PAPER_STATE_DIR / "paper_positions.csv",
)

PAPER_TRADES_PATH = getattr(
    pc,
    "PAPER_TRADES_PATH",
    PAPER_STATE_DIR / "paper_trades.csv",
)

PAPER_EQUITY_CURVE_PATH = getattr(
    pc,
    "PAPER_EQUITY_CURVE_PATH",
    PAPER_STATE_DIR / "paper_equity_curve.csv",
)

PAPER_STATE_PATH = getattr(
    pc,
    "PAPER_STATE_PATH",
    PAPER_STATE_DIR / "paper_state.csv",
)

LATEST_ORDERS_PATH = getattr(
    pc,
    "LATEST_ORDERS_PATH",
    PAPER_REPORTS_DIR / "latest_orders.csv",
)

LATEST_PORTFOLIO_PATH = getattr(
    pc,
    "LATEST_PORTFOLIO_PATH",
    PAPER_REPORTS_DIR / "latest_portfolio.csv",
)

LATEST_SUMMARY_PATH = getattr(
    pc,
    "LATEST_SUMMARY_PATH",
    PAPER_REPORTS_DIR / "latest_summary.txt",
)

DASHBOARD_HTML_PATH = PAPER_REPORTS_DIR / "paper_dashboard.html"
DASHBOARD_TEXT_PATH = PAPER_REPORTS_DIR / "paper_dashboard_summary.txt"


# ============================================================
# SETTINGS
# ============================================================

PAPER_TICKERS = getattr(pc, "PAPER_TICKERS", list(UNIVERSE.keys()))
DATE_FORMAT = getattr(pc, "DATE_FORMAT", "%Y-%m-%d")

REPORT_TITLE = "COMMODITY ENGINE"
REPORT_SUBTITLE = "LOCAL PAPER ALLOCATION MONITOR"

ASSET_COLOURS = {
    "GLD": "#b8b8b8",
    "SLV": "#8e8e8e",
    "USO": "#8a6b58",
    "UNG": "#7a6a97",
    "CPER": "#7d9aad",
    "DBA": "#a48a5f",
    "CASH": "#2f2f2f",
}

POSITIVE_COLOUR = "#9fbe9b"
NEGATIVE_COLOUR = "#c27d7d"
WARNING_COLOUR = "#c9b27a"
INFO_COLOUR = "#9caab8"

CORE_SCORE_COLUMNS = [
    "momentum_score",
    "relative_strength_score",
    "trend_score",
    "trend_persistence_score",
    "volatility_score",
    "risk_score",
    "macro_score",
]

CORE_SCORE_LABELS = {
    "momentum_score": "Momentum",
    "relative_strength_score": "Rel strength",
    "trend_score": "Trend",
    "trend_persistence_score": "Persistence",
    "volatility_score": "Volatility",
    "risk_score": "Risk",
    "macro_score": "Macro",
}

ASSET_SPECIFIC_SCORE_MAP = {
    "GLD": {
        "base_score": "gld_base_score",
        "overlay_score": "gold_overlay_score",
        "pre_clip_score": "gold_final_score_pre_clip",
        "asset_quality_score": "gold_core_data_quality_score",
    },
    "SLV": {
        "base_score": "slv_base_score",
        "overlay_score": "silver_overlay_score",
        "pre_clip_score": "silver_final_score_pre_clip",
        "asset_quality_score": "silver_core_data_quality_score",
    },
    "CPER": {
        "base_score": "cper_base_score",
        "overlay_score": "copper_overlay_score",
        "pre_clip_score": "copper_final_score_pre_clip",
        "asset_quality_score": "copper_core_data_quality_score",
    },
    "USO": {
        "base_score": "uso_base_score",
        "overlay_score": "oil_overlay_score",
        "pre_clip_score": "oil_final_score_pre_clip",
        "asset_quality_score": "oil_core_data_quality_score",
    },
    "DBA": {
        "base_score": "dba_base_score",
        "overlay_score": "agri_overlay_score",
        "pre_clip_score": "agri_final_score_pre_clip",
        "asset_quality_score": "agri_core_data_quality_score",
    },
    # UNG currently has no full gas-specific overlay table in the uploaded final_scores file.
    # If you later add gas_overlay_score / ung_base_score, add them here.
    "UNG": {},
}

# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_output_directories() -> None:
    PAPER_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def clean_ticker(value: Any) -> str:
    return str(value).upper().strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if np.isnan(out) or np.isinf(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def escape(value: Any) -> str:
    return html_lib.escape(str(value))


def money(value: Any) -> str:
    x = safe_float(value, 0.0)
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


def money_signed(value: Any) -> str:
    x = safe_float(value, 0.0)
    sign = "+" if x > 0 else "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


def pct(value: Any) -> str:
    x = safe_float(value, 0.0)
    return f"{x:.2%}"


def pct_signed(value: Any) -> str:
    x = safe_float(value, 0.0)
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.2%}"


def num(value: Any, dp: int = 3) -> str:
    x = safe_float(value, 0.0)
    return f"{x:.{dp}f}"


def date_text(value: Any) -> str:
    if value is None or value == "":
        return "N/A"

    parsed = pd.to_datetime(value, errors="coerce")

    if pd.isna(parsed):
        return str(value)

    return pd.Timestamp(parsed).strftime(DATE_FORMAT)


def file_age_text(path: Path) -> str:
    if not path.exists():
        return "missing"

    modified = pd.Timestamp.fromtimestamp(path.stat().st_mtime)
    return modified.strftime("%Y-%m-%d %H:%M")


def ticker_name(ticker: str) -> str:
    meta = UNIVERSE.get(ticker, {})
    return meta.get("name", ticker)


def ticker_group(ticker: str) -> str:
    meta = UNIVERSE.get(ticker, {})
    return meta.get("group", "unknown")


def group_cap(group: str) -> float | None:
    if isinstance(MAX_GROUP_WEIGHT, dict):
        value = MAX_GROUP_WEIGHT.get(group)
        if value is not None:
            return safe_float(value, np.nan)

    return None


def tone_for_value(value: Any) -> str:
    x = safe_float(value, 0.0)
    if x > 1e-8:
        return "positive"
    if x < -1e-8:
        return "negative"
    return "neutral"


# ============================================================
# LOADERS
# ============================================================

def load_state() -> dict[str, str]:
    state_raw = read_csv_if_exists(PAPER_STATE_PATH)

    if state_raw.empty or not {"key", "value"}.issubset(state_raw.columns):
        return {}

    return {
        str(row["key"]): str(row["value"])
        for _, row in state_raw.iterrows()
    }


def load_positions() -> pd.DataFrame:
    positions = read_csv_if_exists(PAPER_POSITIONS_PATH)

    if positions.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "shares",
                "avg_cost",
                "cost_basis",
                "price_date",
                "price",
                "market_value",
                "current_weight",
                "unrealised_pnl",
                "unrealised_return",
            ]
        )

    positions = positions.copy()

    if "ticker" in positions.columns:
        positions["ticker"] = positions["ticker"].map(clean_ticker)

    for col in [
        "shares",
        "avg_cost",
        "cost_basis",
        "price",
        "market_value",
        "abs_market_value",
        "current_weight",
        "abs_current_weight",
        "unrealised_pnl",
        "unrealised_return",
    ]:
        if col not in positions.columns:
            positions[col] = 0.0

        positions[col] = pd.to_numeric(
            positions[col],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "cost_basis" not in positions.columns or positions["cost_basis"].abs().sum() <= 0:
        positions["cost_basis"] = positions["shares"] * positions["avg_cost"]

    if "unrealised_return" not in positions.columns:
        positions["unrealised_return"] = np.where(
            positions["cost_basis"].abs() > 1e-12,
            positions["unrealised_pnl"] / positions["cost_basis"].abs(),
            0.0,
        )

    if "price_date" not in positions.columns:
        positions["price_date"] = ""

    if "position_type" not in positions.columns:
        positions["position_type"] = np.where(
            positions["shares"] > 1e-10,
            "LONG",
            np.where(positions["shares"] < -1e-10, "SHORT", "FLAT"),
        )

    if "abs_market_value" not in positions.columns:
        positions["abs_market_value"] = positions["market_value"].abs()

    if "abs_current_weight" not in positions.columns:
        positions["abs_current_weight"] = positions["current_weight"].abs()

    positions["name"] = positions["ticker"].map(ticker_name)
    positions["group"] = positions["ticker"].map(ticker_group)

    return positions.sort_values("ticker").reset_index(drop=True)


def load_trades() -> pd.DataFrame:
    trades = read_csv_if_exists(PAPER_TRADES_PATH)

    if trades.empty:
        return pd.DataFrame()

    trades = trades.copy()

    if "ticker" in trades.columns:
        trades["ticker"] = trades["ticker"].map(clean_ticker)

    date_col = "accounting_date" if "accounting_date" in trades.columns else "run_date"

    if date_col in trades.columns:
        trades["_sort_date"] = pd.to_datetime(trades[date_col], errors="coerce")
        trades = trades.sort_values("_sort_date").drop(columns=["_sort_date"])

    return trades.reset_index(drop=True)


def load_equity_curve() -> pd.DataFrame:
    equity = read_csv_if_exists(PAPER_EQUITY_CURVE_PATH)

    if equity.empty:
        return pd.DataFrame()

    equity = equity.copy()

    if "date" in equity.columns:
        equity["date"] = pd.to_datetime(equity["date"], errors="coerce")
        equity = equity.dropna(subset=["date"]).sort_values("date")

    for col in [
        "equity",
        "cash",
        "invested_value",
        "total_exposure",
        "net_exposure",
        "gross_exposure",
        "long_exposure",
        "short_exposure",
        "cash_weight",
        "cash_buffer_weight",
        "daily_pnl",
        "cumulative_pnl",
        "since_rebalance_pnl",
        "rebalance_anchor_equity",
        "unrealised_pnl",
        "cost_basis",
        "interest_earned",
        "days_interest",
        "trade_count",
    ]:
        if col in equity.columns:
            equity[col] = pd.to_numeric(
                equity[col],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan)

    return equity.reset_index(drop=True)


def load_latest_orders() -> pd.DataFrame:
    orders = read_csv_if_exists(LATEST_ORDERS_PATH)

    if orders.empty:
        return pd.DataFrame()

    if "ticker" in orders.columns:
        orders["ticker"] = orders["ticker"].map(clean_ticker)

    return orders


def find_latest_decision_file() -> Path | None:
    if not PAPER_DECISION_SNAPSHOT_DIR.exists():
        return None

    files = sorted(PAPER_DECISION_SNAPSHOT_DIR.glob("decision_*.csv"))

    if not files:
        return None

    return max(files, key=lambda p: p.stat().st_mtime)


def normalise_score_columns(decision: pd.DataFrame) -> pd.DataFrame:
    out = decision.copy()

    # Clean up possible old merge suffixes.
    for base in ["final_score", "rank"]:
        if base not in out.columns:
            candidates = [
                c for c in out.columns
                if c.startswith(base + "_")
            ]
            if candidates:
                out[base] = out[candidates[0]]

    return out


def load_latest_decision() -> tuple[pd.DataFrame, Path | None]:
    path = find_latest_decision_file()

    if path is None:
        return pd.DataFrame(), None

    decision = read_csv_if_exists(path)

    if decision.empty:
        return pd.DataFrame(), path

    decision = normalise_score_columns(decision)

    if "ticker" in decision.columns:
        decision["ticker"] = decision["ticker"].map(clean_ticker)

    if "current_weight" not in decision.columns:
        if "current_weight_before" in decision.columns:
            decision["current_weight"] = decision["current_weight_before"]
        else:
            decision["current_weight"] = 0.0

    if "target_weight" not in decision.columns:
        decision["target_weight"] = 0.0

    if "weight_gap" not in decision.columns:
        decision["weight_gap"] = (
            pd.to_numeric(decision["target_weight"], errors="coerce").fillna(0.0)
            - pd.to_numeric(decision["current_weight"], errors="coerce").fillna(0.0)
        )

    for col in [
        "target_weight",
        "current_weight",
        "weight_gap",
        "abs_target_weight",
        "abs_current_weight",
        "final_score",
        "rank",
        "raw_notional",
        "requested_shares",
    ]:
        if col in decision.columns:
            decision[col] = pd.to_numeric(
                decision[col],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan)

    if "status" not in decision.columns:
        decision["status"] = np.where(
            decision["target_weight"].fillna(0.0) > 0.001,
            "HELD",
            np.where(
                decision.get("final_score", pd.Series(np.nan, index=decision.index)).fillna(-1.0)
                >= MIN_SCORE_TO_HOLD - 0.05,
                "WATCH",
                "NO HOLD",
            ),
        )

    decision["name"] = decision["ticker"].map(ticker_name)
    decision["group"] = decision["ticker"].map(ticker_group)

    sort_cols = ["target_weight"]
    ascending = [False]

    if "rank" in decision.columns:
        sort_cols.append("rank")
        ascending.append(True)

    if "final_score" in decision.columns:
        sort_cols.append("final_score")
        ascending.append(False)

    decision = decision.sort_values(sort_cols, ascending=ascending)

    return decision.reset_index(drop=True), path

def find_latest_final_scores_file() -> Path | None:
    """
    Find the most recent final_scores file.

    Priority is based on the latest date inside the CSV, not just modified time.
    This avoids accidentally using an old signal file that was touched recently.
    """
    candidates: list[Path] = []

    if FINAL_SCORES_PATH.exists():
        candidates.append(FINAL_SCORES_PATH)

    if PAPER_FINAL_SCORE_SNAPSHOT_DIR.exists():
        candidates.extend(PAPER_FINAL_SCORE_SNAPSHOT_DIR.glob("final_scores*.csv"))

    unique_candidates: list[Path] = []
    seen: set[str] = set()

    for path in candidates:
        if not path.exists():
            continue

        key = str(path.resolve())

        if key not in seen:
            unique_candidates.append(path)
            seen.add(key)

    scored_candidates: list[dict[str, Any]] = []

    for path in unique_candidates:
        raw = read_csv_if_exists(path)

        if raw.empty or "date" not in raw.columns or "ticker" not in raw.columns:
            continue

        dates = pd.to_datetime(raw["date"], errors="coerce").dropna()

        if dates.empty:
            continue

        scored_candidates.append(
            {
                "path": path,
                "latest_date": pd.Timestamp(dates.max()).normalize(),
                "modified": path.stat().st_mtime,
            }
        )

    if not scored_candidates:
        return None

    latest = max(
        scored_candidates,
        key=lambda item: (item["latest_date"], item["modified"]),
    )

    return latest["path"]


def load_latest_final_scores_for_display() -> tuple[pd.DataFrame, Path | None]:
    """
    Load the newest available final_scores file and keep only the latest signal date.

    This is deliberately separate from the decision snapshot because the decision
    file may not contain all the score-contributor columns.
    """
    path = find_latest_final_scores_file()

    if path is None:
        return pd.DataFrame(), None

    scores = read_csv_if_exists(path)

    if scores.empty:
        return pd.DataFrame(), path

    if "ticker" not in scores.columns:
        return pd.DataFrame(), path

    scores = scores.copy()
    scores["ticker"] = scores["ticker"].map(clean_ticker)

    if "date" in scores.columns:
        scores["date"] = pd.to_datetime(scores["date"], errors="coerce")
        scores = scores.dropna(subset=["date"]).copy()

        if not scores.empty:
            latest_date = pd.Timestamp(scores["date"].max()).normalize()
            scores = scores[scores["date"] == latest_date].copy()

    scores = scores[scores["ticker"].isin(PAPER_TICKERS)].copy()

    if scores.empty:
        return pd.DataFrame(), path

    numeric_cols = [
        col for col in scores.columns
        if (
            col in ["rank", "final_score"]
            or col.endswith("_score")
            or col.endswith("_weight")
            or col.endswith("_return")
            or col.endswith("_z_3y")
            or col.endswith("_change_3m")
            or col.endswith("_age_days")
        )
    ]

    for col in numeric_cols:
        scores[col] = pd.to_numeric(
            scores[col],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

    if "name" not in scores.columns:
        scores["name"] = scores["ticker"].map(ticker_name)

    if "group" not in scores.columns:
        scores["group"] = scores["ticker"].map(ticker_group)

    sort_cols = []
    ascending = []

    if "rank" in scores.columns:
        sort_cols.append("rank")
        ascending.append(True)

    if "final_score" in scores.columns:
        sort_cols.append("final_score")
        ascending.append(False)

    if sort_cols:
        scores = scores.sort_values(sort_cols, ascending=ascending)
    else:
        scores = scores.sort_values("ticker")

    return scores.reset_index(drop=True), path


def score_value(value: Any) -> str:
    if pd.isna(value):
        return "<span class='dim'>N/A</span>"

    return f"{safe_float(value):.3f}"


def score_delta(value: Any) -> str:
    if pd.isna(value):
        return "<span class='dim'>N/A</span>"

    x = safe_float(value, 0.0)
    tone = tone_for_value(x)

    return f"<span class='tone-{tone}'>{x:+.3f}</span>"


def score_driver_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return "<span class='dim'>N/A</span>"

    return f"<span class='score-driver'>{escape(value)}</span>"


def score_driver_summary(row: pd.Series) -> str:
    contributors: list[tuple[str, float]] = []

    for col in CORE_SCORE_COLUMNS:
        if col not in row.index:
            continue

        value = safe_float(row.get(col), np.nan)

        if pd.notna(value):
            contributors.append((CORE_SCORE_LABELS.get(col, col), value))

    if not contributors:
        return "No score contributors available."

    strongest = sorted(contributors, key=lambda item: item[1], reverse=True)[:2]
    weakest = sorted(contributors, key=lambda item: item[1])[:2]

    strong_text = ", ".join(
        f"{label} {value:.3f}"
        for label, value in strongest
    )

    weak_text = ", ".join(
        f"{label} {value:.3f}"
        for label, value in weakest
    )

    parts = [f"Strongest: {strong_text}.", f"Weakest: {weak_text}."]

    base_score = safe_float(row.get("base_score"), np.nan)
    overlay_score = safe_float(row.get("overlay_score"), np.nan)
    final_score = safe_float(row.get("final_score"), np.nan)

    if pd.notna(base_score) and pd.notna(overlay_score):
        parts.append(f"Asset model: base {base_score:.3f}, overlay {overlay_score:.3f}.")

    if pd.notna(final_score):
        if final_score >= MIN_SCORE_TO_HOLD:
            parts.append("Clears hold threshold.")
        else:
            parts.append("Below hold threshold.")

    return " ".join(parts)


def build_score_attribution(
    final_scores: pd.DataFrame,
    decision: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the table explaining why each asset received its final score.

    final_scores is the source of truth for score components.
    decision is used only to add target/current weights and order status.
    """
    if final_scores.empty:
        return pd.DataFrame()

    out = final_scores.copy()

    decision_cols = [
        "ticker",
        "target_weight",
        "current_weight",
        "weight_gap",
        "side",
        "status",
    ]

    if not decision.empty and "ticker" in decision.columns:
        merge_cols = [col for col in decision_cols if col in decision.columns]

        for col in merge_cols:
            if col != "ticker" and col in out.columns:
                out = out.drop(columns=[col])

        out = out.merge(
            decision[merge_cols],
            on="ticker",
            how="left",
            validate="one_to_one",
        )

    for col in ["target_weight", "current_weight", "weight_gap"]:
        if col not in out.columns:
            out[col] = np.nan

        out[col] = pd.to_numeric(
            out[col],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

    if "status" not in out.columns:
        out["status"] = np.where(
            out.get("target_weight", pd.Series(np.nan, index=out.index)).fillna(0.0) > 0.001,
            "HELD",
            np.where(
                out.get("final_score", pd.Series(np.nan, index=out.index)).fillna(-1.0)
                >= MIN_SCORE_TO_HOLD - 0.05,
                "WATCH",
                "NO HOLD",
            ),
        )

    for new_col in [
        "base_score",
        "overlay_score",
        "pre_clip_score",
        "asset_quality_score",
    ]:
        out[new_col] = np.nan

    for idx, row in out.iterrows():
        ticker = clean_ticker(row.get("ticker", ""))
        mapping = ASSET_SPECIFIC_SCORE_MAP.get(ticker, {})

        for new_col, source_col in mapping.items():
            if source_col in out.columns:
                out.at[idx, new_col] = safe_float(row.get(source_col), np.nan)

    if "final_score" in out.columns:
        out["score_gap_to_hold"] = out["final_score"] - MIN_SCORE_TO_HOLD
    else:
        out["score_gap_to_hold"] = np.nan

    out["final_minus_base"] = np.where(
        out["base_score"].notna() & out.get("final_score", pd.Series(np.nan, index=out.index)).notna(),
        out["final_score"] - out["base_score"],
        np.nan,
    )

    out["driver_summary"] = out.apply(score_driver_summary, axis=1)

    sort_cols = []
    ascending = []

    if "rank" in out.columns:
        sort_cols.append("rank")
        ascending.append(True)

    if "final_score" in out.columns:
        sort_cols.append("final_score")
        ascending.append(False)

    if sort_cols:
        out = out.sort_values(sort_cols, ascending=ascending)
    else:
        out = out.sort_values("ticker")

    return out.reset_index(drop=True)

# ============================================================
# CONTEXT BUILDERS
# ============================================================

def latest_equity_stats(
    state: dict[str, str],
    positions: pd.DataFrame,
    equity_curve: pd.DataFrame,
) -> dict[str, float | str]:
    initial_capital = safe_float(
        state.get("initial_capital"),
        safe_float(getattr(pc, "PAPER_INITIAL_CAPITAL", 10000), 10000),
    )

    if not equity_curve.empty:
        last = equity_curve.iloc[-1]

        equity = safe_float(last.get("equity"), 0.0)
        cash = safe_float(last.get("cash"), 0.0)
        invested = safe_float(last.get("invested_value"), 0.0)
        exposure = safe_float(last.get("total_exposure", last.get("gross_exposure")), 0.0)
        net_exposure = safe_float(last.get("net_exposure"), exposure)
        gross_exposure = safe_float(last.get("gross_exposure"), exposure)
        long_exposure = safe_float(last.get("long_exposure"), max(net_exposure, 0.0))
        short_exposure = safe_float(last.get("short_exposure"), 0.0)
        cash_weight = safe_float(last.get("cash_weight"), 0.0)
        cash_buffer_weight = safe_float(last.get("cash_buffer_weight"), max(0.0, 1.0 - gross_exposure))
        action = str(last.get("action", "N/A"))
        signal_date = date_text(last.get("signal_date", state.get("last_signal_date", "")))
        last_date = date_text(last.get("date", state.get("last_run_date", "")))
        latest_price_date = date_text(last.get("latest_price_date", state.get("last_price_date", "")))
        daily_pnl = safe_float(last.get("daily_pnl"), 0.0)
        since_rebalance_pnl = safe_float(last.get("since_rebalance_pnl"), 0.0)
        interest_earned = safe_float(last.get("interest_earned"), 0.0)
        days_interest = safe_float(last.get("days_interest"), 0.0)
        trade_count = safe_float(last.get("trade_count"), 0.0)
        unrealised_pnl = safe_float(last.get("unrealised_pnl"), positions.get("unrealised_pnl", pd.Series(dtype=float)).sum())
    else:
        cash = safe_float(state.get("cash"), 0.0)
        invested = float(positions.get("market_value", pd.Series(dtype=float)).sum())
        gross_invested = float(positions.get("market_value", pd.Series(dtype=float)).abs().sum())
        long_value = float(positions.get("market_value", pd.Series(dtype=float)).clip(lower=0.0).sum())
        short_value = float(positions.get("market_value", pd.Series(dtype=float)).clip(upper=0.0).abs().sum())
        equity = cash + invested
        net_exposure = invested / equity if equity > 0 else 0.0
        gross_exposure = gross_invested / equity if equity > 0 else 0.0
        long_exposure = long_value / equity if equity > 0 else 0.0
        short_exposure = short_value / equity if equity > 0 else 0.0
        exposure = gross_exposure
        cash_weight = cash / equity if equity > 0 else 0.0
        cash_buffer_weight = max(0.0, 1.0 - gross_exposure)
        action = "N/A"
        signal_date = date_text(state.get("last_signal_date", ""))
        last_date = date_text(state.get("last_run_date", ""))
        latest_price_date = date_text(state.get("last_price_date", ""))
        daily_pnl = 0.0
        since_rebalance_pnl = 0.0
        interest_earned = 0.0
        days_interest = 0.0
        trade_count = 0.0
        unrealised_pnl = float(positions.get("unrealised_pnl", pd.Series(dtype=float)).sum())

    cumulative_pnl = equity - initial_capital
    cumulative_return = cumulative_pnl / initial_capital if initial_capital > 0 else 0.0
    daily_return = daily_pnl / (equity - daily_pnl) if (equity - daily_pnl) > 0 else 0.0
    since_rebalance_return = since_rebalance_pnl / (equity - since_rebalance_pnl) if (equity - since_rebalance_pnl) > 0 else 0.0

    return {
        "equity": equity,
        "cash": cash,
        "invested": invested,
        "exposure": exposure,
        "net_exposure": net_exposure,
        "gross_exposure": gross_exposure,
        "long_exposure": long_exposure,
        "short_exposure": short_exposure,
        "cash_weight": cash_weight,
        "cash_buffer_weight": cash_buffer_weight,
        "initial_capital": initial_capital,
        "cumulative_pnl": cumulative_pnl,
        "cumulative_return": cumulative_return,
        "daily_pnl": daily_pnl,
        "daily_return": daily_return,
        "since_rebalance_pnl": since_rebalance_pnl,
        "since_rebalance_return": since_rebalance_return,
        "unrealised_pnl": unrealised_pnl,
        "interest_earned": interest_earned,
        "days_interest": days_interest,
        "trade_count": trade_count,
        "action": action,
        "signal_date": signal_date,
        "last_date": last_date,
        "latest_price_date": latest_price_date,
    }


def build_position_performance(positions: pd.DataFrame, decision: pd.DataFrame) -> pd.DataFrame:
    active = positions[positions["current_weight"].abs() > 0.0005].copy()

    if active.empty:
        return active

    active["entry_value"] = active["cost_basis"].abs()
    active["current_value"] = active["market_value"].abs()
    active["abs_current_weight"] = active["current_weight"].abs()
    active["pnl"] = active["unrealised_pnl"]
    active["pnl_pct"] = active["unrealised_return"]

    if not decision.empty and "target_weight" in decision.columns:
        active = active.merge(
            decision[["ticker", "target_weight"]],
            on="ticker",
            how="left",
        )
    else:
        active["target_weight"] = np.nan

    active["weight_drift"] = active["current_weight"] - active["target_weight"].fillna(0.0)

    return active.sort_values("abs_current_weight", ascending=False).reset_index(drop=True)


def build_status_flags(
    decision: pd.DataFrame,
    positions: pd.DataFrame,
    stats: dict[str, float | str],
) -> list[tuple[str, str]]:
    flags: list[tuple[str, str]] = []

    exposure = safe_float(stats.get("gross_exposure", stats.get("exposure")), 0.0)
    cash_weight = safe_float(stats.get("cash_weight"), 0.0)

    held_positions = positions[positions["current_weight"].abs() > 0.001].copy()

    if held_positions.empty:
        flags.append(("WARNING", "No active commodity positions. Portfolio is effectively cash."))

    if len(held_positions) == 1:
        ticker = held_positions.iloc[0]["ticker"]
        weight = held_positions.iloc[0]["current_weight"]
        pnl = safe_float(held_positions.iloc[0].get("unrealised_pnl"), 0.0)
        ret = safe_float(held_positions.iloc[0].get("unrealised_return"), 0.0)
        flags.append(("INFO", f"Single active holding: {ticker} at {weight:.2%}. Position P/L {money_signed(pnl)} ({pct_signed(ret)})."))

    if cash_weight >= 0.50:
        flags.append(("INFO", f"High cash allocation: {cash_weight:.2%}."))

    if exposure > 0.95:
        flags.append(("RISK", "Portfolio is nearly fully invested."))

    if not decision.empty and "final_score" in decision.columns:
        near = decision[
            (decision["target_weight"].fillna(0.0) <= 0.001)
            & (decision["final_score"].fillna(0.0) >= MIN_SCORE_TO_HOLD - 0.05)
            & (decision["final_score"].fillna(0.0) < MIN_SCORE_TO_HOLD)
        ].copy()

        if not near.empty:
            tickers = ", ".join(
                f"{row['ticker']} {safe_float(row['final_score']):.3f}"
                for _, row in near.iterrows()
            )
            flags.append(("WATCH", f"Near-threshold assets below hold line ({MIN_SCORE_TO_HOLD:.2f}): {tickers}."))

    if not decision.empty:
        capped_rows = []

        for _, row in decision.iterrows():
            ticker = row.get("ticker", "")
            group = row.get("group", "unknown")
            target_weight = safe_float(row.get("target_weight"), 0.0)
            cap = group_cap(group)

            if cap is not None and not np.isnan(cap):
                if target_weight > 0 and abs(target_weight - cap) < 0.0025:
                    capped_rows.append(f"{ticker} at {group} cap {cap:.0%}")

        if capped_rows:
            flags.append(("CAP", "; ".join(capped_rows)))

    if not flags:
        flags.append(("OK", "No structural warnings detected."))

    return flags


def build_live_analytics_status(equity_curve: pd.DataFrame) -> dict[str, Any]:
    observations = len(equity_curve)

    if observations < 2 or "equity" not in equity_curve.columns:
        return {
            "observations": observations,
            "status": "insufficient",
            "return": np.nan,
            "max_drawdown": np.nan,
            "best_check": np.nan,
            "worst_check": np.nan,
        }

    eq = equity_curve.copy().dropna(subset=["equity"])
    eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
    eq = eq.dropna(subset=["equity"])

    if len(eq) < 2:
        return {
            "observations": len(eq),
            "status": "insufficient",
            "return": np.nan,
            "max_drawdown": np.nan,
            "best_check": np.nan,
            "worst_check": np.nan,
        }

    returns = eq["equity"].pct_change().dropna()
    drawdown = eq["equity"] / eq["equity"].cummax() - 1.0

    return {
        "observations": len(eq),
        "status": "early" if len(eq) < 20 else "usable",
        "return": eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1.0,
        "max_drawdown": drawdown.min(),
        "best_check": returns.max() if not returns.empty else np.nan,
        "worst_check": returns.min() if not returns.empty else np.nan,
    }


def build_memo_lines(
    decision: pd.DataFrame,
    positions: pd.DataFrame,
    stats: dict[str, float | str],
    flags: list[tuple[str, str]],
) -> list[str]:
    lines: list[str] = []

    held = positions[positions["current_weight"].abs() > 0.001].copy()
    held["abs_current_weight"] = held["current_weight"].abs()
    held = held.sort_values("abs_current_weight", ascending=False)

    lines.append("WEEKLY PAPER MEMO")
    lines.append("")
    lines.append("Current allocation:")

    if held.empty:
        lines.append("- No active commodity positions.")
    else:
        for _, row in held.iterrows():
            pos_type = row.get("position_type", "LONG" if safe_float(row.get("current_weight")) > 0 else "SHORT")
            lines.append(
                f"- {row['ticker']} / {row['name']}: "
                f"{pos_type} {safe_float(row['current_weight']):+.2%} current weight, "
                f"P/L {money_signed(row.get('unrealised_pnl', 0.0))} "
                f"({pct_signed(row.get('unrealised_return', 0.0))})."
            )

    lines.append(f"- Cash: {safe_float(stats.get('cash_weight')):.2%}.")
    lines.append("")

    lines.append("Checkup performance:")
    lines.append(f"- Latest check P/L: {money_signed(stats.get('daily_pnl'))} ({pct_signed(stats.get('daily_return'))}).")
    lines.append(f"- Since current rebalance: {money_signed(stats.get('since_rebalance_pnl'))} ({pct_signed(stats.get('since_rebalance_return'))}).")
    lines.append(f"- Since paper start: {money_signed(stats.get('cumulative_pnl'))} ({pct_signed(stats.get('cumulative_return'))}).")
    lines.append("")

    lines.append("System interpretation:")

    if not decision.empty and "final_score" in decision.columns:
        active = decision[decision["target_weight"].fillna(0.0).abs() > 0.001].copy()
        inactive = decision[decision["target_weight"].fillna(0.0).abs() <= 0.001].copy()

        if active.empty:
            lines.append("- No asset currently clears allocation after score/cap/risk rules.")
        else:
            active_names = ", ".join(
                f"{row['ticker']} {safe_float(row.get('target_weight')):+.1%} ({safe_float(row.get('final_score')):.3f})"
                if pd.notna(row.get("final_score"))
                else f"{row['ticker']} {safe_float(row.get('target_weight')):+.1%}"
                for _, row in active.iterrows()
            )
            lines.append(f"- Active signed allocation is concentrated in: {active_names}.")

        below = inactive[
            inactive["final_score"].fillna(-1.0) < MIN_SCORE_TO_HOLD
        ].copy()

        if not below.empty:
            top_below = below.sort_values("final_score", ascending=False).head(3)
            names = ", ".join(
                f"{row['ticker']} ({safe_float(row['final_score']):.3f})"
                for _, row in top_below.iterrows()
                if pd.notna(row.get("final_score"))
            )
            if names:
                lines.append(f"- Closest non-held assets below {MIN_SCORE_TO_HOLD:.2f} threshold: {names}.")

    else:
        lines.append("- Decision snapshot unavailable or missing score columns.")

    lines.append("")
    lines.append("Risks / watch items:")
    for label, message in flags:
        lines.append(f"- [{label}] {message}")

    lines.append("")
    lines.append("Operating rule:")
    lines.append("- Display is read-only. Do not alter strategy parameters from this report.")

    return lines


# ============================================================
# HTML COMPONENTS
# ============================================================

def metric_box(label: str, value: str, sub: str = "", tone: str = "") -> str:
    tone_class = f" tone-{tone}" if tone else ""

    return f"""
    <div class="metric{tone_class}">
        <div class="metric-label">{escape(label)}</div>
        <div class="metric-value">{escape(value)}</div>
        <div class="metric-sub">{escape(sub)}</div>
    </div>
    """


def status_badge(label: str, value: str, tone: str = "neutral") -> str:
    return f"""
    <div class="status-row">
        <span class="status-label">{escape(label)}</span>
        <span class="status-value tone-{escape(tone)}">{escape(value)}</span>
    </div>
    """


def table_html(
    df: pd.DataFrame,
    columns: list[str],
    headers: dict[str, str] | None = None,
    formatters: dict[str, Any] | None = None,
    max_rows: int | None = None,
    empty_text: str = "No data available.",
) -> str:
    if headers is None:
        headers = {}

    if formatters is None:
        formatters = {}

    if df.empty:
        return f'<div class="empty">{escape(empty_text)}</div>'

    visible = [col for col in columns if col in df.columns]

    if not visible:
        return f'<div class="empty">{escape(empty_text)}</div>'

    data = df[visible].copy()

    if max_rows is not None:
        data = data.head(max_rows)

    head = "".join(f"<th>{escape(headers.get(col, col))}</th>" for col in visible)
    rows = []

    for _, row in data.iterrows():
        cells = []

        for col in visible:
            value = row[col]
            formatter = formatters.get(col)

            if formatter is not None:
                rendered = formatter(value)
            elif isinstance(value, float):
                rendered = num(value, 4)
            else:
                rendered = escape(value)

            cells.append(f"<td>{rendered}</td>")

        rows.append("<tr>" + "".join(cells) + "</tr>")

    return f"""
    <table>
        <thead><tr>{head}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    """


def allocation_bar(positions: pd.DataFrame, cash_weight: float) -> str:
    segments = []

    active = positions[positions["current_weight"].abs() > 0.0005].copy()
    active["abs_current_weight"] = active["current_weight"].abs()
    active = active.sort_values("abs_current_weight", ascending=False)

    for _, row in active.iterrows():
        ticker = row["ticker"]
        weight = safe_float(row["current_weight"])
        width = max(weight * 100, 0.25)
        colour = ASSET_COLOURS.get(ticker, "#777777")

        segments.append(
            f"""
            <div class="allocation-segment"
                 style="width:{width:.4f}%; background:{colour};">
                 <span>{escape(ticker)} {weight:.1%}</span>
            </div>
            """
        )

    if cash_weight > 0.0005:
        width = max(cash_weight * 100, 0.25)
        segments.append(
            f"""
            <div class="allocation-segment cash"
                 style="width:{width:.4f}%; background:{ASSET_COLOURS['CASH']};">
                 <span>CASH {cash_weight:.1%}</span>
            </div>
            """
        )

    if not segments:
        segments.append(
            """
            <div class="allocation-segment cash" style="width:100%; background:#2f2f2f;">
                <span>NO EXPOSURE</span>
            </div>
            """
        )

    return f"<div class='allocation-bar'>{''.join(segments)}</div>"


def score_bar(value: Any) -> str:
    if pd.isna(value):
        return "<span class='dim'>N/A</span>"

    score = safe_float(value, 0.0)
    score = max(min(score, 1.0), 0.0)
    width = score * 100

    if score >= MIN_SCORE_TO_HOLD:
        colour = POSITIVE_COLOUR
    elif score >= MIN_SCORE_TO_HOLD - 0.05:
        colour = WARNING_COLOUR
    else:
        colour = "#7f7f7f"

    return f"""
    <div class="score-cell">
        <span>{score:.3f}</span>
        <div class="score-track">
            <div class="score-fill" style="width:{width:.2f}%; background:{colour};"></div>
        </div>
    </div>
    """


def pnl_span(value: Any, money_mode: bool = True) -> str:
    tone = tone_for_value(value)
    text = money_signed(value) if money_mode else pct_signed(value)
    return f"<span class='tone-{tone}'>{escape(text)}</span>"


def make_equity_svg(equity_curve: pd.DataFrame) -> str:
    if equity_curve.empty or "equity" not in equity_curve.columns:
        return '<div class="empty chart-empty">No equity curve yet.</div>'

    data = equity_curve.copy()
    data = data.dropna(subset=["equity"])

    if data.empty:
        return '<div class="empty chart-empty">No equity curve yet.</div>'

    if "date" not in data.columns:
        data["date"] = range(len(data))

    width = 980
    height = 260
    pad_left = 58
    pad_right = 22
    pad_top = 20
    pad_bottom = 36

    values = data["equity"].astype(float).tolist()

    if len(values) == 1:
        value = values[0]
        return f"""
        <div class="single-point-chart">
            <div class="single-point-label">EQUITY CURVE INITIALISED</div>
            <div class="single-point-value">{money(value)}</div>
            <div class="single-point-sub">More points will appear as paper tracking continues.</div>
        </div>
        """

    y_min = min(values)
    y_max = max(values)

    if y_min == y_max:
        y_min *= 0.99
        y_max *= 1.01

    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad

    inner_w = width - pad_left - pad_right
    inner_h = height - pad_top - pad_bottom

    points = []

    for i, value in enumerate(values):
        x = pad_left + (i / (len(values) - 1)) * inner_w
        y = pad_top + (1.0 - ((value - y_min) / (y_max - y_min))) * inner_h
        points.append(f"{x:.2f},{y:.2f}")

    first_date = date_text(data["date"].iloc[0])
    last_date = date_text(data["date"].iloc[-1])

    latest_value = values[-1]
    first_value = values[0]
    change = latest_value - first_value
    tone = "positive" if change >= 0 else "negative"

    line_colour = POSITIVE_COLOUR if change >= 0 else NEGATIVE_COLOUR

    return f"""
    <svg class="equity-svg" viewBox="0 0 {width} {height}" role="img">
        <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" class="axis" />
        <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" class="axis" />

        <line x1="{pad_left}" y1="{pad_top}" x2="{width - pad_right}" y2="{pad_top}" class="gridline" />
        <line x1="{pad_left}" y1="{pad_top + inner_h / 2}" x2="{width - pad_right}" y2="{pad_top + inner_h / 2}" class="gridline" />
        <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" class="gridline" />

        <polyline points="{' '.join(points)}" class="equity-line" style="stroke:{line_colour};" fill="none" />

        <text x="{pad_left}" y="{height - 10}" class="axis-label">{escape(first_date)}</text>
        <text x="{width - pad_right}" y="{height - 10}" text-anchor="end" class="axis-label">{escape(last_date)}</text>

        <text x="8" y="{pad_top + 5}" class="axis-label">{money(y_max)}</text>
        <text x="8" y="{height - pad_bottom}" class="axis-label">{money(y_min)}</text>

        <circle cx="{points[-1].split(',')[0]}" cy="{points[-1].split(',')[1]}" r="3.5" class="latest-dot" style="fill:{line_colour};" />
        <text x="{width - pad_right}" y="{pad_top + 14}" text-anchor="end" class="chart-stat {tone}">
            Latest {money(latest_value)} / Δ {money_signed(change)}
        </text>
    </svg>
    """


def position_performance_bars(position_perf: pd.DataFrame) -> str:
    if position_perf.empty:
        return '<div class="empty">No active position performance yet.</div>'

    rows = []

    for _, row in position_perf.iterrows():
        ticker = row["ticker"]
        entry_value = safe_float(row.get("entry_value"), 0.0)
        current_value = safe_float(row.get("current_value"), 0.0)
        pnl = safe_float(row.get("pnl"), 0.0)
        pnl_pct = safe_float(row.get("pnl_pct"), 0.0)

        baseline = max(entry_value, current_value, 1.0)
        entry_width = max((entry_value / baseline) * 100, 1.0)
        current_width = max((current_value / baseline) * 100, 1.0)
        tone = tone_for_value(pnl)
        colour = POSITIVE_COLOUR if pnl >= 0 else NEGATIVE_COLOUR

        rows.append(
            f"""
            <div class="position-bar-row">
                <div class="position-bar-head">
                    <span class="position-ticker">{escape(ticker)}</span>
                    <span class="position-pnl tone-{tone}">{money_signed(pnl)} / {pct_signed(pnl_pct)}</span>
                </div>
                <div class="value-line">
                    <span class="value-label">ENTRY</span>
                    <div class="value-track"><div class="value-fill entry" style="width:{entry_width:.2f}%"></div></div>
                    <span class="value-number">{money(entry_value)}</span>
                </div>
                <div class="value-line">
                    <span class="value-label">NOW</span>
                    <div class="value-track"><div class="value-fill current" style="width:{current_width:.2f}%; background:{colour};"></div></div>
                    <span class="value-number">{money(current_value)}</span>
                </div>
            </div>
            """
        )

    return "\n".join(rows)


def make_group_exposure_bars(decision: pd.DataFrame, positions: pd.DataFrame) -> str:
    groups = sorted({
        *[ticker_group(ticker) for ticker in PAPER_TICKERS],
        *positions.get("group", pd.Series(dtype=str)).dropna().astype(str).tolist(),
    })

    rows = []

    for group in groups:
        if not group or group == "unknown":
            continue

        current = 0.0
        target = 0.0

        if not positions.empty and "group" in positions.columns:
            current = safe_float(positions.loc[positions["group"] == group, "current_weight"].sum())

        if not decision.empty and "group" in decision.columns:
            target = safe_float(decision.loc[decision["group"] == group, "target_weight"].sum())

        cap = group_cap(group)
        cap_value = cap if cap is not None and not np.isnan(cap) else 1.0
        cap_value = max(cap_value, 0.0001)
        current_width = min(current / cap_value * 100, 100)
        target_width = min(target / cap_value * 100, 100)

        rows.append(
            f"""
            <div class="group-row">
                <div class="group-label">{escape(group)}</div>
                <div class="group-bars">
                    <div class="group-track">
                        <div class="group-fill current" style="width:{current_width:.2f}%"></div>
                        <div class="group-fill target" style="width:{target_width:.2f}%"></div>
                    </div>
                </div>
                <div class="group-values">
                    {pct(current)} / cap {pct(cap_value)}
                </div>
            </div>
            """
        )

    return "\n".join(rows) if rows else '<div class="empty">No group exposure data available.</div>'


def build_flags_html(flags: list[tuple[str, str]]) -> str:
    cards = []

    for label, message in flags:
        tone = label.lower()
        cards.append(
            f"""
            <div class="flag flag-{escape(tone)}">
                <span class="flag-label">{escape(label)}</span>
                <span class="flag-text">{escape(message)}</span>
            </div>
            """
        )

    return "\n".join(cards)


def build_memo_html(lines: list[str]) -> str:
    escaped = "\n".join(escape(line) for line in lines)
    return f"<pre class='memo'>{escaped}</pre>"


def analytics_status_html(live: dict[str, Any]) -> str:
    observations = int(live.get("observations", 0))
    status = str(live.get("status", "insufficient"))

    if status == "insufficient":
        status_text = "INSUFFICIENT LIVE SAMPLE"
        detail = "Operational tracking only. Sharpe/maxDD are not meaningful yet."
        tone = "warning"
    elif status == "early":
        status_text = "EARLY LIVE SAMPLE"
        detail = "Useful for behaviour checks, not statistical proof."
        tone = "info"
    else:
        status_text = "USABLE LIVE SAMPLE"
        detail = "Enough observations to start reading basic live diagnostics."
        tone = "positive"

    return f"""
    <div class="analytics-box">
        <div class="analytics-head tone-{tone}">{escape(status_text)}</div>
        <div class="analytics-detail">{escape(detail)}</div>
        <div class="analytics-grid">
            <div><span>OBS</span><b>{observations}</b></div>
            <div><span>LIVE RETURN</span><b>{pct_signed(live.get("return")) if pd.notna(live.get("return")) else "N/A"}</b></div>
            <div><span>MAX DD</span><b>{pct_signed(live.get("max_drawdown")) if pd.notna(live.get("max_drawdown")) else "N/A"}</b></div>
            <div><span>BEST CHECK</span><b>{pct_signed(live.get("best_check")) if pd.notna(live.get("best_check")) else "N/A"}</b></div>
            <div><span>WORST CHECK</span><b>{pct_signed(live.get("worst_check")) if pd.notna(live.get("worst_check")) else "N/A"}</b></div>
        </div>
    </div>
    """


# ============================================================
# CSS / HTML
# ============================================================

def build_css() -> str:
    return """
    :root {
        --bg: #050505;
        --panel: #0d0d0d;
        --panel-2: #111111;
        --border: #8a8a8a;
        --border-dim: #3c3c3c;
        --text: #d8d8d8;
        --text-soft: #9c9c9c;
        --text-dim: #6f6f6f;
        --white: #f2f2f2;
        --positive: #9fbe9b;
        --warning: #c9b27a;
        --negative: #c27d7d;
        --info: #9caab8;
        --accent-blue: #7d9aad;
    }

    * {
        box-sizing: border-box;
    }

    body {
        margin: 0;
        background:
            repeating-linear-gradient(
                to bottom,
                rgba(255, 255, 255, 0.025) 0px,
                rgba(255, 255, 255, 0.025) 1px,
                transparent 1px,
                transparent 4px
            ),
            var(--bg);
        color: var(--text);
        font-family: "IBM Plex Mono", "JetBrains Mono", "Consolas", "Courier New", monospace;
        font-size: 13px;
        line-height: 1.35;
    }

    .page {
        max-width: 1500px;
        margin: 0 auto;
        padding: 22px;
    }

    .terminal-frame {
        border: 1px solid var(--border);
        background: rgba(10, 10, 10, 0.96);
        box-shadow: inset 0 0 0 1px #1e1e1e, 0 0 0 1px #000;
    }

    .header {
        padding: 24px 26px 18px 26px;
        border-bottom: 1px solid var(--border);
        display: grid;
        grid-template-columns: 1fr 440px;
        gap: 18px;
        align-items: end;
    }

    .title {
        margin: 0;
        font-family: "Courier New", monospace;
        font-size: 48px;
        letter-spacing: 0.045em;
        line-height: 0.95;
        color: var(--white);
        font-weight: 900;
        text-shadow: 2px 0 #707070, 4px 0 #333333, 6px 0 #151515;
    }

    .subtitle {
        margin-top: 8px;
        color: var(--text-soft);
        font-size: 14px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .system-meta {
        border: 1px solid var(--border-dim);
        padding: 10px 12px;
        background: var(--panel-2);
    }

    .status-row {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        padding: 3px 0;
        border-bottom: 1px dotted #2c2c2c;
    }

    .status-row:last-child {
        border-bottom: 0;
    }

    .status-label {
        color: var(--text-soft);
    }

    .status-value {
        color: var(--white);
        text-align: right;
    }

    .section {
        border: 1px solid var(--border-dim);
        background: var(--panel);
        margin: 14px;
    }

    .section-title {
        padding: 7px 10px;
        border-bottom: 1px solid var(--border-dim);
        background: #151515;
        color: var(--white);
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-family: "Courier New", monospace;
    }

    .section-title::before {
        content: "▣ ";
        color: var(--text-soft);
    }

    .section-body {
        padding: 12px;
        overflow-x: auto;
    }

    .grid {
        display: grid;
        grid-template-columns: 1.08fr 1fr;
        gap: 0;
    }

    .metrics {
        display: grid;
        grid-template-columns: repeat(4, minmax(130px, 1fr));
        gap: 8px;
    }

    .metrics.checkup {
        grid-template-columns: repeat(5, minmax(120px, 1fr));
    }

    .metric {
        border: 1px solid var(--border-dim);
        background: #080808;
        padding: 10px;
        min-height: 76px;
    }

    .metric-label {
        color: var(--text-soft);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .metric-value {
        margin-top: 8px;
        font-size: 22px;
        color: var(--white);
        font-weight: 800;
    }

    .metric-sub {
        margin-top: 4px;
        color: var(--text-dim);
        font-size: 11px;
    }

    .tone-positive { color: var(--positive) !important; }
    .tone-warning { color: var(--warning) !important; }
    .tone-negative { color: var(--negative) !important; }
    .tone-info { color: var(--info) !important; }
    .tone-neutral { color: var(--white) !important; }
    .dim { color: var(--text-dim); }

    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
    }

    th {
        color: var(--white);
        background: #171717;
        border: 1px solid var(--border-dim);
        padding: 6px 7px;
        text-align: left;
        font-weight: 800;
        letter-spacing: 0.04em;
        white-space: nowrap;
    }

    td {
        border: 1px solid #282828;
        padding: 6px 7px;
        color: var(--text);
        white-space: nowrap;
    }

    tr:nth-child(even) td { background: #0a0a0a; }
    tr:hover td { background: #161616; }

    .allocation-bar {
        width: 100%;
        height: 42px;
        display: flex;
        border: 1px solid var(--border);
        background: #050505;
        overflow: hidden;
    }

    .allocation-segment {
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #f0f0f0;
        border-right: 1px solid #0b0b0b;
        min-width: 1px;
        overflow: hidden;
        text-shadow: 1px 1px #000;
        font-size: 12px;
        font-weight: 800;
    }

    .allocation-segment span {
        padding: 0 6px;
        white-space: nowrap;
    }

    .score-cell {
        min-width: 118px;
        display: grid;
        grid-template-columns: 44px 1fr;
        align-items: center;
        gap: 7px;
    }

    .score-track {
        height: 8px;
        background: #222;
        border: 1px solid #3a3a3a;
        overflow: hidden;
    }

    .score-fill {
        height: 100%;
    }
     .score-driver {
        display: block;
        min-width: 360px;
        max-width: 620px;
        white-space: normal;
        line-height: 1.35;
        color: var(--text-soft);
    }

    .score-source-note {
        margin-bottom: 10px;
        color: var(--text-dim);
        font-size: 11px;
        letter-spacing: 0.04em;
    }
    .equity-svg {
        width: 100%;
        height: auto;
        border: 1px solid var(--border-dim);
        background: #070707;
    }

    .axis {
        stroke: #777;
        stroke-width: 1;
    }

    .gridline {
        stroke: #252525;
        stroke-width: 1;
    }

    .equity-line {
        stroke-width: 2;
    }

    .latest-dot {
        fill: #f0f0f0;
    }

    .axis-label {
        fill: #8c8c8c;
        font-size: 11px;
        font-family: monospace;
    }

    .chart-stat {
        fill: #cfcfcf;
        font-size: 12px;
        font-family: monospace;
        font-weight: 800;
    }

    .single-point-chart {
        min-height: 190px;
        border: 1px solid var(--border-dim);
        background: #070707;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        letter-spacing: 0.05em;
    }

    .single-point-label {
        color: var(--text-soft);
        font-size: 12px;
    }

    .single-point-value {
        margin-top: 8px;
        color: var(--white);
        font-size: 30px;
        font-weight: 900;
    }

    .single-point-sub {
        margin-top: 8px;
        color: var(--text-dim);
        font-size: 11px;
    }

    .position-bar-row {
        border: 1px solid var(--border-dim);
        background: #080808;
        padding: 10px;
        margin-bottom: 8px;
    }

    .position-bar-head {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        font-weight: 800;
    }

    .position-ticker {
        color: var(--white);
    }

    .value-line {
        display: grid;
        grid-template-columns: 58px 1fr 105px;
        gap: 8px;
        align-items: center;
        margin-top: 5px;
    }

    .value-label {
        color: var(--text-soft);
        font-size: 11px;
    }

    .value-track {
        height: 14px;
        background: #1a1a1a;
        border: 1px solid #333;
        overflow: hidden;
    }

    .value-fill {
        height: 100%;
    }

    .value-fill.entry {
        background: #6b6b6b;
    }

    .value-number {
        text-align: right;
        color: var(--text);
        font-size: 11px;
    }

    .group-row {
        display: grid;
        grid-template-columns: 150px 1fr 130px;
        gap: 8px;
        align-items: center;
        padding: 7px 0;
        border-bottom: 1px solid #222;
    }

    .group-row:last-child {
        border-bottom: 0;
    }

    .group-label {
        color: var(--text);
    }

    .group-track {
        height: 14px;
        background: #181818;
        border: 1px solid #333;
        position: relative;
    }

    .group-fill.current {
        height: 100%;
        background: var(--accent-blue);
        opacity: 0.85;
    }

    .group-fill.target {
        height: 100%;
        background: var(--warning);
        opacity: 0.45;
        position: absolute;
        top: 0;
        left: 0;
    }

    .group-values {
        color: var(--text-soft);
        text-align: right;
        font-size: 11px;
    }

    .flags {
        display: grid;
        gap: 7px;
    }

    .flag {
        border: 1px solid var(--border-dim);
        padding: 8px 9px;
        background: #080808;
        display: grid;
        grid-template-columns: 82px 1fr;
        gap: 10px;
    }

    .flag-label {
        font-weight: 900;
        color: var(--white);
    }

    .flag-text {
        color: var(--text);
    }

    .flag-warning .flag-label,
    .flag-watch .flag-label,
    .flag-cap .flag-label {
        color: var(--warning);
    }

    .flag-risk .flag-label {
        color: var(--negative);
    }

    .flag-ok .flag-label {
        color: var(--positive);
    }

    .flag-info .flag-label {
        color: var(--info);
    }

    .analytics-box {
        border: 1px solid var(--border-dim);
        background: #080808;
        padding: 11px;
    }

    .analytics-head {
        font-weight: 900;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }

    .analytics-detail {
        color: var(--text-soft);
        margin-bottom: 10px;
    }

    .analytics-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 8px;
    }

    .analytics-grid div {
        border: 1px solid #303030;
        padding: 8px;
        background: #0b0b0b;
    }

    .analytics-grid span {
        display: block;
        color: var(--text-soft);
        font-size: 10px;
        margin-bottom: 4px;
    }

    .analytics-grid b {
        color: var(--white);
        font-size: 14px;
    }

    .memo {
        margin: 0;
        white-space: pre-wrap;
        color: var(--text);
        font-family: "IBM Plex Mono", "JetBrains Mono", "Consolas", "Courier New", monospace;
        font-size: 12px;
        line-height: 1.45;
    }

    .empty {
        color: var(--text-dim);
        border: 1px dashed #333;
        padding: 14px;
        background: #080808;
    }

    .footer {
        color: var(--text-dim);
        padding: 14px;
        border-top: 1px solid var(--border);
        font-size: 11px;
        display: flex;
        justify-content: space-between;
        gap: 18px;
    }

    @media (max-width: 1100px) {
        .header {
            grid-template-columns: 1fr;
        }

        .grid {
            grid-template-columns: 1fr;
        }

        .metrics,
        .metrics.checkup {
            grid-template-columns: repeat(2, minmax(130px, 1fr));
        }

        .analytics-grid {
            grid-template-columns: repeat(2, 1fr);
        }

        .title {
            font-size: 34px;
        }

        table {
            font-size: 11px;
        }
    }
    """


def build_html(
    *,
    state: dict[str, str],
    positions: pd.DataFrame,
    trades: pd.DataFrame,
    latest_orders: pd.DataFrame,
    equity_curve: pd.DataFrame,
    decision: pd.DataFrame,
    decision_path: Path | None,
    final_scores: pd.DataFrame,
    final_scores_path: Path | None,
    stats: dict[str, float | str],
    position_perf: pd.DataFrame,
    flags: list[tuple[str, str]],
    memo_lines: list[str],
    live_analytics: dict[str, Any],
) -> str:

    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    last_rebalance = state.get("last_rebalance_period", "N/A")
    last_signal = stats.get("signal_date", "N/A")
    last_action = stats.get("action", "N/A")
    latest_price_date = stats.get("latest_price_date", "N/A")
    last_date = stats.get("last_date", "N/A")

    pnl = safe_float(stats.get("cumulative_pnl"), 0.0)
    pnl_tone = tone_for_value(pnl)

    active_positions = positions[positions["current_weight"].abs() > 0.0005].copy()
    active_positions = active_positions.sort_values("current_weight", ascending=False)

    score_df = decision.copy()

    score_attribution_df = build_score_attribution(
        final_scores=final_scores,
        decision=decision,
    )

    if not score_df.empty:
        if "status" not in score_df.columns:
            score_df["status"] = np.where(
                score_df["target_weight"].fillna(0.0) > 0.001,
                "HELD",
                np.where(
                    score_df.get("final_score", pd.Series(-1.0, index=score_df.index)).fillna(-1.0)
                    >= MIN_SCORE_TO_HOLD - 0.05,
                    "WATCH",
                    "NO HOLD",
                ),
            )

    trades_display = trades.tail(12).copy()

    if not trades_display.empty:
        if "accounting_date" not in trades_display.columns and "run_date" in trades_display.columns:
            trades_display["accounting_date"] = trades_display["run_date"]

        if "filled_notional" not in trades_display.columns and "fill_notional" in trades_display.columns:
            trades_display["filled_notional"] = trades_display["fill_notional"]

    decision_file_text = str(decision_path.name) if decision_path is not None else "none"
    final_scores_file_text = (
        f"{final_scores_path.name} / modified {file_age_text(final_scores_path)}"
        if final_scores_path is not None
        else "none"
    )

    html = f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>{escape(REPORT_TITLE)} — Paper Monitor</title>
        <style>{build_css()}</style>
    </head>
    <body>
        <div class="page">
            <div class="terminal-frame">

                <header class="header">
                    <div>
                        <h1 class="title">{escape(REPORT_TITLE)}</h1>
                        <div class="subtitle">{escape(REPORT_SUBTITLE)}</div>
                    </div>

                    <div class="system-meta">
                        {status_badge("MODE", "LOCAL PAPER", "positive")}
                        {status_badge("ACCOUNTING DATE", str(last_date), "neutral")}
                        {status_badge("LATEST SIGNAL", str(last_signal), "neutral")}
                        {status_badge("LATEST PRICE", str(latest_price_date), "neutral")}
                        {status_badge("LAST REBALANCE", str(last_rebalance), "neutral")}
                        {status_badge("LAST ACTION", str(last_action), "info")}
                        {status_badge("REPORT BUILT", now, "neutral")}
                    </div>
                </header>

                <section class="section">
                    <div class="section-title">Account Summary</div>
                    <div class="section-body">
                        <div class="metrics">
                            {metric_box("Equity", money(stats.get("equity")), "paper account value")}
                            {metric_box("Cash", money(stats.get("cash")), pct(stats.get("cash_weight")) + " actual cash/equity")}
                            {metric_box("Gross", pct(stats.get("gross_exposure")), "net " + pct_signed(stats.get("net_exposure")))}
                            {metric_box("Short", pct(stats.get("short_exposure")), "long " + pct(stats.get("long_exposure")))}
                            {metric_box("Total P/L", money_signed(stats.get("cumulative_pnl")), pct_signed(stats.get("cumulative_return")) + " since start", pnl_tone)}
                        </div>
                    </div>
                </section>

                <section class="section">
                    <div class="section-title">Paper Checkup</div>
                    <div class="section-body">
                        <div class="metrics checkup">
                            {metric_box("Check P/L", money_signed(stats.get("daily_pnl")), pct_signed(stats.get("daily_return")) + " since last check", tone_for_value(stats.get("daily_pnl")))}
                            {metric_box("Rebalance P/L", money_signed(stats.get("since_rebalance_pnl")), pct_signed(stats.get("since_rebalance_return")) + " current allocation", tone_for_value(stats.get("since_rebalance_pnl")))}
                            {metric_box("Unrealised P/L", money_signed(stats.get("unrealised_pnl")), "open positions only", tone_for_value(stats.get("unrealised_pnl")))}
                            {metric_box("Cash Interest", money_signed(stats.get("interest_earned")), str(int(safe_float(stats.get("days_interest")))) + " day accrual", tone_for_value(stats.get("interest_earned")))}
                            {metric_box("Trades", str(int(safe_float(stats.get("trade_count")))), "this check run")}
                        </div>
                    </div>
                </section>

                <section class="section">
                    <div class="section-title">Current Allocation</div>
                    <div class="section-body">
                        {allocation_bar(positions, safe_float(stats.get("cash_weight")))}
                    </div>
                </section>

                <div class="grid">
                    <section class="section">
                        <div class="section-title">Position Value Change</div>
                        <div class="section-body">
                            {position_performance_bars(position_perf)}
                        </div>
                    </section>

                    <section class="section">
                        <div class="section-title">Live Analytics Status</div>
                        <div class="section-body">
                            {analytics_status_html(live_analytics)}
                        </div>
                    </section>
                </div>

                <section class="section">
                    <div class="section-title">Position Performance</div>
                    <div class="section-body">
                        {table_html(
                            position_perf,
                            [
                                "ticker",
                                "name",
                                "shares",
                                "avg_cost",
                                "price",
                                "entry_value",
                                "current_value",
                                "pnl",
                                "pnl_pct",
                                "target_weight",
                                "current_weight",
                                "weight_drift",
                                "price_date",
                            ],
                            headers={
                                "ticker": "TICKER",
                                "name": "ASSET",
                                "shares": "SHARES",
                                "avg_cost": "AVG COST",
                                "price": "PRICE",
                                "entry_value": "ENTRY VALUE",
                                "current_value": "CURRENT VALUE",
                                "pnl": "UNREAL P/L",
                                "pnl_pct": "UNREAL %",
                                "target_weight": "TARGET",
                                "current_weight": "CURRENT",
                                "weight_drift": "DRIFT",
                                "price_date": "PRICE DATE",
                            },
                            formatters={
                                "shares": lambda x: f"{safe_float(x):,.6f}",
                                "avg_cost": money,
                                "price": money,
                                "entry_value": money,
                                "current_value": money,
                                "pnl": lambda x: pnl_span(x, True),
                                "pnl_pct": lambda x: pnl_span(x, False),
                                "target_weight": pct,
                                "current_weight": pct,
                                "weight_drift": lambda x: pnl_span(x, False),
                                "price_date": date_text,
                            },
                            empty_text="No active open positions.",
                        )}
                    </div>
                </section>

                <section class="section">
                    <div class="section-title">Equity Curve</div>
                    <div class="section-body">
                        {make_equity_svg(equity_curve)}
                    </div>
                </section>

                <section class="section">
                    <div class="section-title">Signal Board</div>
                    <div class="section-body">
                        {table_html(
                            score_df,
                            [
                                "rank",
                                "ticker",
                                "name",
                                "group",
                                "final_score",
                                "target_weight",
                                "current_weight",
                                "weight_gap",
                                "side",
                                "raw_notional",
                                "status",
                            ],
                            headers={
                                "rank": "RANK",
                                "ticker": "TICKER",
                                "name": "ASSET",
                                "group": "GROUP",
                                "final_score": "SCORE",
                                "target_weight": "TARGET",
                                "current_weight": "CURRENT",
                                "weight_gap": "GAP",
                                "side": "ORDER",
                                "raw_notional": "NOTIONAL",
                                "status": "STATUS",
                            },
                            formatters={
                                "rank": lambda x: "" if pd.isna(x) else str(int(float(x))),
                                "final_score": score_bar,
                                "target_weight": pct,
                                "current_weight": pct,
                                "weight_gap": lambda x: pnl_span(x, False),
                                "raw_notional": money,
                            },
                            empty_text="No decision snapshot available yet.",
                        )}
                    </div>
                                </section>

                <section class="section">
                    <div class="section-title">Score Attribution</div>
                    <div class="section-body">
                        <div class="score-source-note">
                            SCORE SOURCE: {escape(final_scores_file_text)}
                        </div>

                        {table_html(
                            score_attribution_df,
                            [
                                "rank",
                                "ticker",
                                "name",
                                "group",
                                "final_score",
                                "score_gap_to_hold",
                                "target_weight",
                                "base_score",
                                "overlay_score",
                                "final_minus_base",
                                "commodity_model_score",
                                "commodity_conviction_score",
                                "commodity_data_quality_score",
                                "asset_quality_score",
                                "momentum_score",
                                "relative_strength_score",
                                "trend_score",
                                "trend_persistence_score",
                                "volatility_score",
                                "risk_score",
                                "macro_score",
                                "driver_summary",
                            ],
                            headers={
                                "rank": "RANK",
                                "ticker": "TICKER",
                                "name": "ASSET",
                                "group": "GROUP",
                                "final_score": "FINAL",
                                "score_gap_to_hold": "VS HOLD",
                                "target_weight": "TARGET",
                                "base_score": "BASE",
                                "overlay_score": "OVERLAY",
                                "final_minus_base": "FINAL - BASE",
                                "commodity_model_score": "MODEL",
                                "commodity_conviction_score": "CONVICTION",
                                "commodity_data_quality_score": "DATA Q",
                                "asset_quality_score": "ASSET Q",
                                "momentum_score": "MOM",
                                "relative_strength_score": "REL STR",
                                "trend_score": "TREND",
                                "trend_persistence_score": "PERSIST",
                                "volatility_score": "VOL",
                                "risk_score": "RISK",
                                "macro_score": "MACRO",
                                "driver_summary": "INTERPRETATION",
                            },
                            formatters={
                                "rank": lambda x: "" if pd.isna(x) else str(int(float(x))),
                                "final_score": score_bar,
                                "score_gap_to_hold": score_delta,
                                "target_weight": pct,
                                "base_score": score_value,
                                "overlay_score": score_value,
                                "final_minus_base": score_delta,
                                "commodity_model_score": score_value,
                                "commodity_conviction_score": score_value,
                                "commodity_data_quality_score": score_value,
                                "asset_quality_score": score_value,
                                "momentum_score": score_value,
                                "relative_strength_score": score_value,
                                "trend_score": score_value,
                                "trend_persistence_score": score_value,
                                "volatility_score": score_value,
                                "risk_score": score_value,
                                "macro_score": score_value,
                                "driver_summary": score_driver_text,
                            },
                            empty_text="No final_scores attribution file available yet.",
                        )}
                    </div>
                </section>

                <div class="grid">
                    <section class="section">
                        <div class="section-title">Group Exposure</div>
                        <div class="section-body">
                            {make_group_exposure_bars(decision, positions)}
                        </div>
                    </section>

                    <section class="section">
                        <div class="section-title">System Flags</div>
                        <div class="section-body">
                            <div class="flags">
                                {build_flags_html(flags)}
                            </div>
                        </div>
                    </section>
                </div>

                <section class="section">
                    <div class="section-title">Recent Paper Trades</div>
                    <div class="section-body">
                        {table_html(
                            trades_display,
                            [
                                "accounting_date",
                                "signal_date",
                                "ticker",
                                "side",
                                "filled_shares",
                                "fill_price",
                                "filled_notional",
                                "commission",
                                "slippage_cost",
                                "realised_pnl",
                                "reason",
                            ],
                            headers={
                                "accounting_date": "DATE",
                                "signal_date": "SIGNAL",
                                "ticker": "TICKER",
                                "side": "SIDE",
                                "filled_shares": "SHARES",
                                "fill_price": "PRICE",
                                "filled_notional": "NOTIONAL",
                                "commission": "COMM",
                                "slippage_cost": "SLIP",
                                "realised_pnl": "REAL P/L",
                                "reason": "REASON",
                            },
                            formatters={
                                "accounting_date": date_text,
                                "signal_date": date_text,
                                "filled_shares": lambda x: f"{safe_float(x):,.6f}",
                                "fill_price": money,
                                "filled_notional": money,
                                "commission": money,
                                "slippage_cost": money,
                                "realised_pnl": lambda x: pnl_span(x, True),
                            },
                            empty_text="No paper trades logged yet.",
                        )}
                    </div>
                </section>

                <section class="section">
                    <div class="section-title">Weekly Memo Block</div>
                    <div class="section-body">
                        {build_memo_html(memo_lines)}
                    </div>
                </section>

                <footer class="footer">
                    <div>
                        STATE: {escape(PAPER_STATE_PATH.name)} /
                        POSITIONS: {escape(PAPER_POSITIONS_PATH.name)} /
                        TRADES: {escape(PAPER_TRADES_PATH.name)}
                    </div>
                    <div>
                        DECISION SNAPSHOT: {escape(decision_file_text)} /
                        SCORES: {escape(final_scores_file_text)} /
                        MODIFIED: {escape(file_age_text(PAPER_EQUITY_CURVE_PATH))}
                    </div>
                </footer>

            </div>
        </div>
    </body>
    </html>
    """

    return html


# ============================================================
# TEXT SUMMARY
# ============================================================

def write_text_summary(
    stats: dict[str, float | str],
    positions: pd.DataFrame,
    flags: list[tuple[str, str]],
    memo_lines: list[str],
) -> None:
    lines: list[str] = []

    lines.append("COMMODITY ENGINE — PAPER DISPLAY SUMMARY")
    lines.append("=" * 58)
    lines.append(f"Accounting date: {stats.get('last_date', 'N/A')}")
    lines.append(f"Latest signal:   {stats.get('signal_date', 'N/A')}")
    lines.append(f"Latest price:    {stats.get('latest_price_date', 'N/A')}")
    lines.append(f"Last action:     {stats.get('action', 'N/A')}")
    lines.append(f"Equity:          {money(stats.get('equity'))}")
    lines.append(f"Cash:            {money(stats.get('cash'))} ({pct(stats.get('cash_weight'))})")
    lines.append(f"Net invested:    {money(stats.get('invested'))} ({pct_signed(stats.get('net_exposure'))})")
    lines.append(f"Gross exposure:  {pct(stats.get('gross_exposure'))} | Long {pct(stats.get('long_exposure'))} | Short {pct(stats.get('short_exposure'))}")
    lines.append(f"Check P/L:       {money_signed(stats.get('daily_pnl'))} ({pct_signed(stats.get('daily_return'))})")
    lines.append(f"Rebalance P/L:   {money_signed(stats.get('since_rebalance_pnl'))} ({pct_signed(stats.get('since_rebalance_return'))})")
    lines.append(f"Total P/L:       {money_signed(stats.get('cumulative_pnl'))} ({pct_signed(stats.get('cumulative_return'))})")
    lines.append("")

    lines.append("POSITIONS")
    lines.append("-" * 58)

    active = positions[positions["current_weight"].abs() > 0.0005].copy()
    active["abs_current_weight"] = active["current_weight"].abs()
    active = active.sort_values("abs_current_weight", ascending=False)

    if active.empty:
        lines.append("No active positions.")
    else:
        for _, row in active.iterrows():
            lines.append(
                f"{row['ticker']:>4} "
                f"{str(row.get('position_type', '')):>5} "
                f"{safe_float(row['current_weight']):>+7.2%} "
                f"{money(row['market_value']):>14} "
                f"P/L {money_signed(row.get('unrealised_pnl', 0.0)):>12} "
                f"({pct_signed(row.get('unrealised_return', 0.0))})"
            )

    lines.append("")
    lines.append("FLAGS")
    lines.append("-" * 58)

    for label, message in flags:
        lines.append(f"[{label}] {message}")

    lines.append("")
    lines.extend(memo_lines)

    DASHBOARD_TEXT_PATH.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_output_directories()

    state = load_state()
    positions = load_positions()
    trades = load_trades()
    latest_orders = load_latest_orders()
    equity_curve = load_equity_curve()
    decision, decision_path = load_latest_decision()
    final_scores, final_scores_path = load_latest_final_scores_for_display()
    stats = latest_equity_stats(
        state=state,
        positions=positions,
        equity_curve=equity_curve,
    )

    position_perf = build_position_performance(
        positions=positions,
        decision=decision,
    )

    flags = build_status_flags(
        decision=decision,
        positions=positions,
        stats=stats,
    )

    live_analytics = build_live_analytics_status(equity_curve)

    memo_lines = build_memo_lines(
        decision=decision,
        positions=positions,
        stats=stats,
        flags=flags,
    )

    html = build_html(
        state=state,
        positions=positions,
        trades=trades,
        latest_orders=latest_orders,
        equity_curve=equity_curve,
        decision=decision,
        decision_path=decision_path,
        stats=stats,
        position_perf=position_perf,
        flags=flags,
        memo_lines=memo_lines,
        live_analytics=live_analytics,
        final_scores=final_scores,
        final_scores_path=final_scores_path,
    )

    DASHBOARD_HTML_PATH.write_text(html, encoding="utf-8")

    write_text_summary(
        stats=stats,
        positions=positions,
        flags=flags,
        memo_lines=memo_lines,
    )

    print("\n" + "=" * 80)
    print("PAPER DISPLAY COMPLETE")
    print("=" * 80)
    print(f"HTML dashboard: {DASHBOARD_HTML_PATH}")
    print(f"Text summary:   {DASHBOARD_TEXT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
