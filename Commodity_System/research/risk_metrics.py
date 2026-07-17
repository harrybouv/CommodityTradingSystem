from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# ============================================================
# PATH SETUP
# ============================================================

THIS_FILE = Path(__file__).resolve()
RESEARCH_DIR = THIS_FILE.parent
COMMODITY_ROOT = RESEARCH_DIR.parent
PROJECT_ROOT = COMMODITY_ROOT.parent

for path in [PROJECT_ROOT, COMMODITY_ROOT, RESEARCH_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# ============================================================
# CONFIG IMPORT
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


# ============================================================
# DEFAULT PATHS / SETTINGS
# ============================================================

RESULTS_DIR = Path(cfg("RESULTS_DIR", COMMODITY_ROOT / "results"))
INITIAL_CAPITAL = float(cfg("INITIAL_CAPITAL", 10_000.0))
TRADING_DAYS_PER_YEAR = int(cfg("TRADING_DAYS_PER_YEAR", 252))

DEFAULT_V3_OUTPUT_DIR = RESULTS_DIR / "backtest_V3"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "risk"

MODEL_CURVE_FILENAME = "model_curve_V3.csv"

CONFIDENCE_LEVELS = [0.95, 0.99]
SUMMARY_HORIZONS = [1, 5, 20]
ROLLING_WINDOW_DAYS = 252
ROLLING_MIN_OBS = 126

WORST_LOSS_HORIZONS = {
    "1_day": 1,
    "5_day": 5,
    "10_day": 10,
    "20_day": 20,
    "63_day": 63,
    "126_day": 126,
    "252_day": 252,
}


# ============================================================
# STYLE
# ============================================================

COLORS = {
    "bg": "#0f1117",
    "axes": "#151923",
    "grid": "#2c3340",
    "text": "#e8e8e8",
    "muted": "#a3a3a3",
    "green": "#6ee7b7",
    "blue": "#60a5fa",
    "amber": "#f59e0b",
    "red": "#fb7185",
    "purple": "#c084fc",
}


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["axes"],
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "text.color": COLORS["text"],
            "axes.titleweight": "bold",
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.55,
            "legend.facecolor": COLORS["axes"],
            "legend.edgecolor": COLORS["grid"],
            "legend.labelcolor": COLORS["text"],
            "font.size": 10,
        }
    )


def save_fig(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=170,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return str(path)


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "base": output_dir,
        "charts": output_dir / "charts",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"Run backtest_V3.py first so {MODEL_CURVE_FILENAME} exists."
        )

    return pd.read_csv(path)


def find_col(df: pd.DataFrame, candidates: list[str] | tuple[str, ...]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}

    for cand in candidates:
        if cand in df.columns:
            return cand

        cand_l = str(cand).lower()
        if cand_l in lower_map:
            return lower_map[cand_l]

    return None


def coerce_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    date_col = find_col(
        out,
        (
            "date",
            "Date",
            "datetime",
            "timestamp",
            "index",
            "Unnamed: 0",
        ),
    )

    if date_col is not None:
        out[date_col] = pd.to_datetime(out[date_col])
        out = out.set_index(date_col)
    else:
        try:
            out.index = pd.to_datetime(out.index)
        except Exception as exc:
            raise ValueError("Could not infer a date column/index from model curve.") from exc

    out = out.sort_index()
    out.index.name = "date"

    return out


def clean_series(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna()
    return out


def extract_strategy_returns(model_curve: pd.DataFrame) -> pd.Series:
    df = coerce_datetime_index(model_curve)

    ret_col = find_col(
        df,
        (
            "net_return",
            "strategy_return",
            "model_return",
            "return",
            "returns",
            "daily_return",
            "period_return",
        ),
    )

    if ret_col is not None:
        r = clean_series(df[ret_col])
        r.name = "strategy_return"
        return r

    equity_col = find_col(
        df,
        (
            "equity",
            "net_equity",
            "model_equity",
            "portfolio_value",
            "value",
        ),
    )

    if equity_col is not None:
        r = pd.to_numeric(df[equity_col], errors="coerce").pct_change()
        r = clean_series(r)
        r.name = "strategy_return"
        return r

    raise ValueError(
        "Could not find strategy returns in model_curve_V3.csv. "
        "Expected one of: net_return, strategy_return, model_return, return, returns, "
        "or an equity column."
    )


def extract_equity(model_curve: pd.DataFrame, returns: pd.Series) -> pd.Series:
    df = coerce_datetime_index(model_curve)

    equity_col = find_col(
        df,
        (
            "equity",
            "net_equity",
            "model_equity",
            "portfolio_value",
            "value",
        ),
    )

    if equity_col is not None:
        eq = clean_series(df[equity_col])
        eq.name = "equity"
        return eq.reindex(returns.index).ffill()

    eq = INITIAL_CAPITAL * (1.0 + returns).cumprod()
    eq.name = "equity"
    return eq


def compound_horizon_returns(returns: pd.Series, horizon_days: int) -> pd.Series:
    """
    Overlapping realised horizon returns.

    Example:
      horizon_days=5 means each point is the compounded 5-trading-day return
      ending on that date.
    """
    r = clean_series(returns)

    if horizon_days <= 1:
        out = r.copy()
        out.name = "horizon_return"
        return out

    out = (
        (1.0 + r)
        .rolling(window=horizon_days, min_periods=horizon_days)
        .apply(np.prod, raw=True)
        - 1.0
    )

    out = out.dropna()
    out.name = "horizon_return"

    return out


def historical_var_cvar(
    returns: pd.Series,
    confidence: float,
) -> tuple[float, float]:
    """
    Returns positive loss numbers.

    If 95% VaR = 0.012, that means a 1.2% loss threshold.
    """
    r = clean_series(returns)

    if r.empty:
        return np.nan, np.nan

    losses = -r
    var = float(losses.quantile(confidence))

    tail = losses[losses >= var]
    cvar = float(tail.mean()) if len(tail) > 0 else np.nan

    return var, cvar


def rolling_var_cvar(
    returns: pd.Series,
    confidence: float,
    window: int = ROLLING_WINDOW_DAYS,
    min_obs: int = ROLLING_MIN_OBS,
) -> pd.DataFrame:
    """
    Rolling historical VaR/CVaR.

    Important:
    The estimate is shifted by one day so today's VaR is based only on
    information available before today's realised return.
    """
    r = clean_series(returns)
    losses = -r

    rolling_var = losses.rolling(
        window=window,
        min_periods=min_obs,
    ).quantile(confidence)

    def cvar_func(x: pd.Series) -> float:
        s = pd.Series(x).dropna()

        if s.empty:
            return np.nan

        var = s.quantile(confidence)
        tail = s[s >= var]

        if tail.empty:
            return np.nan

        return float(tail.mean())

    rolling_cvar = losses.rolling(
        window=window,
        min_periods=min_obs,
    ).apply(cvar_func, raw=False)

    out = pd.DataFrame(
        {
            f"rolling_var_{int(confidence * 100)}": rolling_var.shift(1),
            f"rolling_cvar_{int(confidence * 100)}": rolling_cvar.shift(1),
        }
    )

    return out


def drawdown_from_returns(returns: pd.Series) -> pd.Series:
    r = clean_series(returns)
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    dd.name = "drawdown"
    return dd


def classify_breach_rate(
    actual_rate: float,
    expected_rate: float,
) -> str:
    if pd.isna(actual_rate) or pd.isna(expected_rate):
        return "insufficient_data"

    if actual_rate <= expected_rate * 0.50:
        return "conservative"
    if actual_rate <= expected_rate * 1.50:
        return "ok"
    if actual_rate <= expected_rate * 2.50:
        return "watch"
    return "fail"


# ============================================================
# TABLE BUILDERS
# ============================================================

def build_var_summary(
    returns: pd.Series,
    equity: pd.Series,
) -> pd.DataFrame:
    rows = []

    final_equity = float(equity.dropna().iloc[-1]) if not equity.dropna().empty else np.nan

    for horizon in SUMMARY_HORIZONS:
        horizon_returns = compound_horizon_returns(returns, horizon)

        for confidence in CONFIDENCE_LEVELS:
            var, cvar = historical_var_cvar(
                returns=horizon_returns,
                confidence=confidence,
            )

            rows.append(
                {
                    "confidence_level": confidence,
                    "confidence_label": f"{int(confidence * 100)}%",
                    "horizon_days": horizon,
                    "observations": int(horizon_returns.dropna().shape[0]),
                    "var": var,
                    "cvar": cvar,
                    "return_threshold": -var if pd.notna(var) else np.nan,
                    "worst_return": float(horizon_returns.min()) if not horizon_returns.empty else np.nan,
                    "best_return": float(horizon_returns.max()) if not horizon_returns.empty else np.nan,
                    "mean_return": float(horizon_returns.mean()) if not horizon_returns.empty else np.nan,
                    "volatility": float(horizon_returns.std(ddof=1)) if len(horizon_returns) > 1 else np.nan,
                    "var_dollars_initial_capital": var * INITIAL_CAPITAL if pd.notna(var) else np.nan,
                    "cvar_dollars_initial_capital": cvar * INITIAL_CAPITAL if pd.notna(cvar) else np.nan,
                    "var_dollars_final_equity": var * final_equity if pd.notna(var) and pd.notna(final_equity) else np.nan,
                    "cvar_dollars_final_equity": cvar * final_equity if pd.notna(cvar) and pd.notna(final_equity) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def build_rolling_var_tables(
    returns: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r = clean_series(returns)
    losses = -r

    rolling_parts = [
        pd.DataFrame(
            {
                "strategy_return": r,
                "loss": losses,
            }
        )
    ]

    for confidence in CONFIDENCE_LEVELS:
        rolling_parts.append(
            rolling_var_cvar(
                returns=r,
                confidence=confidence,
                window=ROLLING_WINDOW_DAYS,
                min_obs=ROLLING_MIN_OBS,
            )
        )

    rolling = pd.concat(rolling_parts, axis=1).sort_index()

    breach_cols = {
        "strategy_return": rolling["strategy_return"],
        "loss": rolling["loss"],
    }

    breach_summary_rows = []

    for confidence in CONFIDENCE_LEVELS:
        label = int(confidence * 100)
        var_col = f"rolling_var_{label}"
        breach_col = f"breach_{label}"

        valid = rolling[var_col].notna()
        breach = (rolling["loss"] > rolling[var_col]) & valid

        breach_cols[var_col] = rolling[var_col]
        breach_cols[breach_col] = breach.astype(int)

        valid_observations = int(valid.sum())
        breach_count = int(breach.sum())
        actual_rate = breach_count / valid_observations if valid_observations > 0 else np.nan
        expected_rate = 1.0 - confidence

        breach_summary_rows.append(
            {
                "confidence_level": confidence,
                "confidence_label": f"{label}%",
                "rolling_window_days": ROLLING_WINDOW_DAYS,
                "min_observations": ROLLING_MIN_OBS,
                "valid_observations": valid_observations,
                "breach_count": breach_count,
                "expected_breach_rate": expected_rate,
                "actual_breach_rate": actual_rate,
                "calibration_status": classify_breach_rate(
                    actual_rate=actual_rate,
                    expected_rate=expected_rate,
                ),
            }
        )

    breaches = pd.DataFrame(breach_cols).sort_index()
    breach_summary = pd.DataFrame(breach_summary_rows)

    return rolling, breaches, breach_summary


def build_worst_losses(
    returns: pd.Series,
) -> pd.DataFrame:
    r = clean_series(returns)
    rows = []

    for label, horizon in WORST_LOSS_HORIZONS.items():
        horizon_returns = compound_horizon_returns(r, horizon)

        if horizon_returns.empty:
            rows.append(
                {
                    "window": label,
                    "horizon_days": horizon,
                    "end_date": pd.NaT,
                    "start_date": pd.NaT,
                    "worst_return": np.nan,
                    "worst_loss": np.nan,
                }
            )
            continue

        end_date = horizon_returns.idxmin()
        end_pos = r.index.get_loc(end_date)

        if isinstance(end_pos, slice):
            end_pos = end_pos.stop - 1

        start_pos = max(0, int(end_pos) - horizon + 1)
        start_date = r.index[start_pos]

        worst_return = float(horizon_returns.loc[end_date])

        rows.append(
            {
                "window": label,
                "horizon_days": horizon,
                "start_date": start_date,
                "end_date": end_date,
                "worst_return": worst_return,
                "worst_loss": -worst_return,
                "loss_dollars_initial_capital": -worst_return * INITIAL_CAPITAL,
            }
        )

    return pd.DataFrame(rows)


def build_tail_events(
    returns: pd.Series,
    equity: pd.Series,
    n: int = 25,
) -> pd.DataFrame:
    r = clean_series(returns)

    df = pd.DataFrame(
        {
            "strategy_return": r,
            "loss": -r,
        }
    )

    df["equity"] = equity.reindex(df.index).ffill()
    df["dollar_loss_on_equity"] = df["loss"] * df["equity"].shift(1)

    df = df.sort_values("loss", ascending=False).head(n)
    df = df.reset_index().rename(columns={"index": "date"})

    return df


def build_drawdown_tail_summary(
    returns: pd.Series,
) -> pd.DataFrame:
    dd = drawdown_from_returns(returns)

    if dd.empty:
        return pd.DataFrame()

    rows = [
        {
            "metric": "max_drawdown",
            "value": float(dd.min()),
        },
        {
            "metric": "average_drawdown",
            "value": float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0,
        },
        {
            "metric": "drawdown_5th_percentile",
            "value": float(dd.quantile(0.05)),
        },
        {
            "metric": "drawdown_1st_percentile",
            "value": float(dd.quantile(0.01)),
        },
        {
            "metric": "pct_days_in_drawdown",
            "value": float((dd < 0).mean()),
        },
    ]

    return pd.DataFrame(rows)


# ============================================================
# CHARTS
# ============================================================

def plot_return_distribution(
    returns: pd.Series,
    var_summary: pd.DataFrame,
    charts_dir: Path,
) -> str:
    r = clean_series(returns)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(
        r,
        bins=80,
        color=COLORS["blue"],
        alpha=0.72,
        edgecolor=COLORS["grid"],
    )

    one_day = var_summary[var_summary["horizon_days"] == 1].copy()

    for _, row in one_day.iterrows():
        confidence = int(row["confidence_level"] * 100)
        var = row["var"]

        if pd.notna(var):
            color = COLORS["amber"] if confidence == 95 else COLORS["red"]
            ax.axvline(
                -var,
                color=color,
                linewidth=2.2,
                linestyle="--",
                label=f"{confidence}% VaR threshold",
            )

    ax.axvline(
        0,
        color=COLORS["muted"],
        linewidth=1.1,
        alpha=0.75,
    )

    ax.set_title("Daily strategy return distribution with VaR thresholds")
    ax.set_xlabel("Daily return")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis="y")
    ax.legend(loc="upper left")

    return save_fig(fig, charts_dir / "return_distribution_var.png")


def plot_rolling_var_cvar(
    rolling: pd.DataFrame,
    charts_dir: Path,
) -> str:
    fig, ax = plt.subplots(figsize=(14, 6.5))

    columns = rolling.columns

    if "rolling_var_95" in columns:
        ax.plot(
            rolling.index,
            rolling["rolling_var_95"],
            color=COLORS["amber"],
            linewidth=1.8,
            label="Rolling 95% VaR",
        )

    if "rolling_cvar_95" in columns:
        ax.plot(
            rolling.index,
            rolling["rolling_cvar_95"],
            color=COLORS["red"],
            linewidth=1.8,
            label="Rolling 95% CVaR",
        )

    if "rolling_var_99" in columns:
        ax.plot(
            rolling.index,
            rolling["rolling_var_99"],
            color=COLORS["blue"],
            linewidth=1.5,
            alpha=0.85,
            label="Rolling 99% VaR",
        )

    if "rolling_cvar_99" in columns:
        ax.plot(
            rolling.index,
            rolling["rolling_cvar_99"],
            color=COLORS["purple"],
            linewidth=1.5,
            alpha=0.85,
            label="Rolling 99% CVaR",
        )

    ax.set_title("Rolling historical VaR and CVaR")
    ax.set_xlabel("Date")
    ax.set_ylabel("Estimated loss threshold")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True)
    ax.legend(loc="upper left", ncols=2)

    return save_fig(fig, charts_dir / "rolling_var_cvar.png")


def plot_var_breaches(
    breaches: pd.DataFrame,
    charts_dir: Path,
) -> str:
    fig, ax = plt.subplots(figsize=(14, 6.5))

    ax.plot(
        breaches.index,
        breaches["strategy_return"],
        color=COLORS["muted"],
        linewidth=0.75,
        alpha=0.55,
        label="Daily strategy return",
    )

    if "rolling_var_95" in breaches.columns:
        ax.plot(
            breaches.index,
            -breaches["rolling_var_95"],
            color=COLORS["amber"],
            linewidth=1.5,
            alpha=0.95,
            label="Negative 95% VaR threshold",
        )

    if "rolling_var_99" in breaches.columns:
        ax.plot(
            breaches.index,
            -breaches["rolling_var_99"],
            color=COLORS["red"],
            linewidth=1.4,
            alpha=0.95,
            label="Negative 99% VaR threshold",
        )

    if "breach_95" in breaches.columns:
        b95 = breaches[breaches["breach_95"] == 1]

        ax.scatter(
            b95.index,
            b95["strategy_return"],
            color=COLORS["amber"],
            s=28,
            alpha=0.90,
            label="95% VaR breach",
            zorder=5,
        )

    if "breach_99" in breaches.columns:
        b99 = breaches[breaches["breach_99"] == 1]

        ax.scatter(
            b99.index,
            b99["strategy_return"],
            color=COLORS["red"],
            s=42,
            alpha=0.95,
            label="99% VaR breach",
            zorder=6,
        )

    ax.axhline(
        0,
        color=COLORS["grid"],
        linewidth=1.0,
    )

    ax.set_title("VaR breach map: realised returns vs rolling VaR thresholds")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily return")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True)
    ax.legend(loc="lower left", ncols=2)

    return save_fig(fig, charts_dir / "var_breaches.png")


def plot_worst_rolling_losses(
    worst_losses: pd.DataFrame,
    charts_dir: Path,
) -> str:
    df = worst_losses.copy()

    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title("Worst rolling losses")
        ax.text(
            0.5,
            0.5,
            "No worst-loss data available.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return save_fig(fig, charts_dir / "worst_rolling_losses.png")

    df = df.sort_values("horizon_days")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        df["window"],
        df["worst_return"],
        color=COLORS["red"],
        alpha=0.88,
    )

    ax.axhline(
        0,
        color=COLORS["grid"],
        linewidth=1.0,
    )

    ax.set_title("Worst realised rolling loss by horizon")
    ax.set_xlabel("Window")
    ax.set_ylabel("Worst compounded return")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis="y")

    for i, row in df.iterrows():
        val = row["worst_return"]

        if pd.notna(val):
            ax.text(
                x=list(df.index).index(i),
                y=val,
                s=f"{val:.1%}",
                ha="center",
                va="top",
                fontsize=10,
                color=COLORS["text"],
            )

    return save_fig(fig, charts_dir / "worst_rolling_losses.png")


# ============================================================
# SAVE HELPERS
# ============================================================

def save_csv(df: pd.DataFrame | pd.Series, path: Path, index: bool = True) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(df, pd.Series):
        df.to_frame().to_csv(path, index=index)
    else:
        df.to_csv(path, index=index)

    return str(path)


def save_json(obj: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

    return str(path)


# ============================================================
# MASTER FUNCTION
# ============================================================

def generate_risk_package(
    v3_output_dir: str | Path = DEFAULT_V3_OUTPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """
    Build VaR/CVaR/tail-risk diagnostics from the existing V3 model curve.

    This does not rerun the strategy.
    It audits realised model returns saved by backtest_V3.py.
    """
    setup_plot_style()

    v3_output_dir = Path(v3_output_dir)
    output_dir = Path(output_dir)

    dirs = ensure_output_dirs(output_dir)

    model_curve_path = v3_output_dir / MODEL_CURVE_FILENAME
    model_curve = read_csv_required(model_curve_path)

    returns = extract_strategy_returns(model_curve)
    equity = extract_equity(model_curve, returns)

    if returns.empty:
        raise ValueError("No valid strategy returns found. Cannot compute VaR package.")

    # ---------- Tables ----------
    var_summary = build_var_summary(
        returns=returns,
        equity=equity,
    )

    rolling_var, var_breaches, var_breach_summary = build_rolling_var_tables(
        returns=returns,
    )

    worst_losses = build_worst_losses(
        returns=returns,
    )

    tail_events = build_tail_events(
        returns=returns,
        equity=equity,
        n=25,
    )

    drawdown_tail_summary = build_drawdown_tail_summary(
        returns=returns,
    )

    # ---------- Save tables ----------
    table_paths = {
        "var_summary": save_csv(
            var_summary,
            output_dir / "var_summary.csv",
            index=False,
        ),
        "rolling_var": save_csv(
            rolling_var.reset_index(),
            output_dir / "rolling_var.csv",
            index=False,
        ),
        "var_breaches": save_csv(
            var_breaches.reset_index(),
            output_dir / "var_breaches.csv",
            index=False,
        ),
        "var_breach_summary": save_csv(
            var_breach_summary,
            output_dir / "var_breach_summary.csv",
            index=False,
        ),
        "worst_losses": save_csv(
            worst_losses,
            output_dir / "worst_losses.csv",
            index=False,
        ),
        "tail_events": save_csv(
            tail_events,
            output_dir / "tail_events.csv",
            index=False,
        ),
        "drawdown_tail_summary": save_csv(
            drawdown_tail_summary,
            output_dir / "drawdown_tail_summary.csv",
            index=False,
        ),
    }

    # ---------- Charts ----------
    chart_paths = {
        "return_distribution_var": plot_return_distribution(
            returns=returns,
            var_summary=var_summary,
            charts_dir=dirs["charts"],
        ),
        "rolling_var_cvar": plot_rolling_var_cvar(
            rolling=rolling_var,
            charts_dir=dirs["charts"],
        ),
        "var_breaches": plot_var_breaches(
            breaches=var_breaches,
            charts_dir=dirs["charts"],
        ),
        "worst_rolling_losses": plot_worst_rolling_losses(
            worst_losses=worst_losses,
            charts_dir=dirs["charts"],
        ),
    }

    # ---------- Manifest ----------
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_model_curve": str(model_curve_path),
        "output_dir": str(output_dir),
        "initial_capital": INITIAL_CAPITAL,
        "trading_days_per_year": TRADING_DAYS_PER_YEAR,
        "rolling_window_days": ROLLING_WINDOW_DAYS,
        "rolling_min_observations": ROLLING_MIN_OBS,
        "confidence_levels": CONFIDENCE_LEVELS,
        "summary_horizons": SUMMARY_HORIZONS,
        "tables": table_paths,
        "charts": chart_paths,
    }

    manifest_path = output_dir / "risk_manifest.json"
    save_json(manifest, manifest_path)
    manifest["manifest"] = str(manifest_path)

    # ---------- Console summary ----------
    print("\n========== RISK METRICS PACKAGE ==========")
    print(f"Source: {model_curve_path}")
    print(f"Output: {output_dir}")

    print("\nVaR / CVaR summary:")
    display_cols = [
        "confidence_label",
        "horizon_days",
        "observations",
        "var",
        "cvar",
        "return_threshold",
        "worst_return",
        "var_dollars_initial_capital",
        "cvar_dollars_initial_capital",
    ]
    display_cols = [c for c in display_cols if c in var_summary.columns]
    print(var_summary[display_cols].to_string(index=False))

    print("\nVaR breach calibration:")
    print(var_breach_summary.to_string(index=False))

    print("\nWorst realised rolling losses:")
    print(worst_losses.to_string(index=False))

    print("\nSaved files:")
    for name, path in table_paths.items():
        print(f"  table/{name}: {path}")
    for name, path in chart_paths.items():
        print(f"  chart/{name}: {path}")
    print(f"  manifest: {manifest_path}")

    return manifest


# ============================================================
# CLI
# ============================================================

def main() -> None:
    """
    Usage:
      python Commodity_System/research/risk_metrics.py

    Optional:
      python Commodity_System/research/risk_metrics.py <v3_output_dir> <output_dir>
    """
    if len(sys.argv) >= 2:
        v3_output_dir = Path(sys.argv[1])
    else:
        v3_output_dir = DEFAULT_V3_OUTPUT_DIR

    if len(sys.argv) >= 3:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = DEFAULT_OUTPUT_DIR

    generate_risk_package(
        v3_output_dir=v3_output_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()