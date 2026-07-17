from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# BASIC OVERLAY CORRELATION CHARTS
# ============================================================
# Purpose:
#   Simple visual sanity checks:
#   - Do overlay indicators move with price?
#   - Do they relate to next 1m / 3m / 6m returns?
#   - Are components basically duplicate signals?
#
# This is not proof of alpha. It is a quick "does this look sane?"
# inspection script.


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "results" / "basic_charts" / "overlay_correlations"

TICKERS = ["GLD", "SLV", "CPER", "USO", "UNG", "DBA"]

SEARCH_DIRS = [
    BASE_DIR,
    BASE_DIR / "data" / "raw",
    BASE_DIR / "data" / "processed",
    BASE_DIR / "results",
    BASE_DIR / "results" / "backtest",
]

CORR_METHOD = "spearman"   # "spearman" is usually better for noisy indicators than "pearson".
MONTHLY_FREQ = "ME"        # falls back to "M" if your pandas version is older.
MIN_OBS = 24               # require at least this many monthly observations for a useful chart.


# Keep these deliberately limited. The old script became noisy because it plotted too much.
OVERLAY_SPECS = {
    "GLD": {
        "name": "gold",
        "feature_patterns": ["gold_features_daily.csv", "gold_features_daily*.csv"],
        "feature_cols": [
            "real_yield_10y",
            "real_yield_change_3m",
            "real_yield_z_3y",
            "real_yield_score",
            "usd_index",
            "usd_return_3m",
            "usd_z_3y",
            "usd_score",
            "vix_z_3y",
            "stlfsi_z_3y",
            "spy_drawdown_3m",
            "stress_score",
        ],
        "final_score_cols": [
            "gold_overlay_score",
            "gold_real_yield_score",
            "gold_usd_score",
            "gold_stress_score",
            "gld_base_score",
            "final_score",
        ],
    },
    "SLV": {
        "name": "silver",
        "feature_patterns": ["silver_features_daily.csv", "silver_features_daily*.csv"],
        "feature_cols": [
            "gold_silver_ratio",
            "gold_silver_ratio_change_3m",
            "gold_silver_ratio_z_3y",
            "silver_vs_gold_return_3m",
            "silver_gold_ratio_score",
            "silver_copper_ratio",
            "silver_copper_ratio_change_3m",
            "silver_copper_ratio_z_3y",
            "silver_copper_ratio_score",
            "silver_gold_confirmation_score",
            "silver_real_yield_score",
            "silver_macro_score",
        ],
        "final_score_cols": [
            "silver_overlay_score",
            "silver_gold_ratio_score",
            "silver_copper_ratio_score",
            "silver_gold_confirmation_score",
            "silver_real_yield_score",
            "slv_base_score",
            "final_score",
        ],
    },
    "CPER": {
        "name": "copper",
        "feature_patterns": ["copper_features_daily.csv", "copper_features_daily*.csv"],
        "feature_cols": [
            "china_cli",
            "china_cli_change_3m",
            "china_cli_z_3y",
            "copper_china_cli_score",
            "china_electricity_yoy",
            "china_electricity_yoy_3m_avg",
            "china_electricity_yoy_change_3m",
            "copper_china_electricity_score",
            "copper_usd_score",
            "copper_broad_commodity_trend_score",
            "copper_oil_price_score",
            "copper_global_growth_score",
        ],
        "final_score_cols": [
            "copper_overlay_score",
            "copper_china_cli_score",
            "copper_china_electricity_score",
            "copper_usd_score",
            "cper_base_score",
            "final_score",
        ],
    },
    "USO": {
        "name": "oil",
        "feature_patterns": ["oil_features_daily.csv", "oil_features_daily*.csv"],
        "feature_cols": [
            "wti_spot_price",
            "wti_return_3m",
            "oil_inventory_tightness_score",
            "oil_cushing_tightness_score",
            "uso_usl_ratio",
            "uso_usl_ratio_return_3m",
            "uso_usl_ratio_z_3y",
            "oil_curve_roll_score",
            "refinery_utilisation",
            "refinery_utilisation_change_3m",
            "oil_supply_refinery_score",
            "oil_global_demand_score",
            "oil_usd_score",
        ],
        "final_score_cols": [
            "oil_overlay_score",
            "oil_curve_roll_score",
            "oil_supply_refinery_score",
            "oil_usd_score",
            "oil_inventory_tightness_score",
            "oil_cushing_tightness_score",
            "uso_base_score",
            "final_score",
        ],
    },
    "UNG": {
        "name": "gas",
        "feature_patterns": ["gas_features_daily.csv", "gas_features_daily*.csv"],
        "feature_cols": [
            "henry_hub_spot_price",
            "henry_hub_return_3m",
            "gas_weather_demand_score",
            "gas_storage_tightness_score",
            "gas_storage_momentum_score",
            "ung_unl_ratio",
            "ung_unl_ratio_return_3m",
            "ung_unl_ratio_z_3y",
            "gas_curve_roll_score",
            "gas_supply_pressure_score",
            "gas_lng_export_demand_score",
            "gas_oil_relative_value_score",
            "gas_energy_confirmation_score",
            "gas_balance_score",
        ],
        "final_score_cols": [
            # UNG overlay may be disabled in production, so these may not exist.
            "gas_overlay_score",
            "gas_storage_tightness_score",
            "gas_storage_momentum_score",
            "gas_curve_roll_score",
            "final_score",
        ],
    },
    "DBA": {
        "name": "agriculture",
        "feature_patterns": ["agriculture_features_daily.csv", "agriculture_features_daily*.csv"],
        "feature_cols": [
            "dba_price",
            "dba_return_3m",
            "agri_crop_basket_index",
            "agri_crop_basket_return_3m",
            "agri_crop_basket_trend_score",
            "agri_crop_momentum_score",
            "agri_crop_relative_strength_score",
            "agri_crop_vs_dbc_score",
            "dba_vs_dbc_score",
            "agri_crop_vs_dba_score",
            "agri_seasonality_score",
            "agri_export_demand_score",
        ],
        "final_score_cols": [
            "agri_overlay_score",
            "agri_crop_momentum_score",
            "agri_crop_relative_strength_score",
            "agri_seasonality_score",
            "agri_export_demand_score",
            "dba_base_score",
            "final_score",
        ],
    },
}


TARGET_COLS = [
    "price_level",
    "return_1m",
    "return_3m",
    "next_return_1m",
    "next_return_3m",
    "next_return_6m",
]


def find_file(patterns: list[str]) -> Path | None:
    matches: list[Path] = []

    for folder in SEARCH_DIRS:
        if not folder.exists():
            continue

        for pattern in patterns:
            matches.extend(sorted(folder.glob(pattern)))

    if not matches:
        for pattern in patterns:
            matches.extend(
                sorted(
                    p for p in BASE_DIR.rglob(pattern)
                    if ".venv" not in p.parts and "__pycache__" not in p.parts
                )
            )

    if not matches:
        return None

    exact = [p for p in matches if "(" not in p.name and ")" not in p.name]
    return exact[0] if exact else matches[0]


def load_csv(label: str, patterns: list[str]) -> pd.DataFrame | None:
    path = find_file(patterns)

    if path is None:
        print(f"[skip] {label}: file not found")
        return None

    df = pd.read_csv(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

    print(f"[load] {label}: {path} | rows={len(df):,}, cols={len(df.columns):,}")
    return df


def to_numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    existing = [c for c in cols if c in df.columns]
    out = pd.DataFrame(index=df.index)

    for col in existing:
        out[col] = pd.to_numeric(df[col], errors="coerce")

    return out.replace([np.inf, -np.inf], np.nan)


def monthly_last(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("monthly_last expects a DatetimeIndex")

    try:
        return df.resample(MONTHLY_FREQ).last()
    except ValueError:
        return df.resample("M").last()


def clean_for_corr(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are all missing or constant.
    good_cols = []
    for col in out.columns:
        s = pd.to_numeric(out[col], errors="coerce").dropna()
        if len(s) >= MIN_OBS and s.nunique() > 1:
            good_cols.append(col)

    return out[good_cols]


def load_price_matrix() -> pd.DataFrame | None:
    prices = load_csv("commodity prices", ["commodity_prices.csv", "commodity_prices*.csv"])

    if prices is None:
        return None

    required = {"date", "ticker", "adj_close"}
    if not required.issubset(prices.columns):
        print(f"[skip] commodity prices missing: {required - set(prices.columns)}")
        return None

    matrix = (
        prices[prices["ticker"].isin(TICKERS)]
        .pivot(index="date", columns="ticker", values="adj_close")
        .sort_index()
    )

    matrix = matrix[[t for t in TICKERS if t in matrix.columns]]
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    return matrix


def load_final_scores() -> pd.DataFrame | None:
    scores = load_csv("final scores", ["final_scores.csv", "final_scores*.csv"])

    if scores is None:
        # target_weights contains almost all of the same diagnostic columns, so use it as a fallback.
        scores = load_csv("target weights fallback", ["target_weights.csv", "target_weights*.csv"])

    if scores is None:
        return None

    if "ticker" not in scores.columns or "date" not in scores.columns:
        print("[skip] final scores / target weights missing date or ticker")
        return None

    return scores


def save_heatmap(
    corr: pd.DataFrame,
    title: str,
    filename: str,
    annotate: bool = True,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    corr = corr.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if corr.empty:
        print(f"[skip] {filename}: empty correlation table")
        return

    rows, cols = corr.shape

    width = min(max(7.0, 0.85 * cols + 3.0), 13.0)
    height = min(max(4.0, 0.38 * rows + 2.0), 12.0)

    fig, ax = plt.subplots(figsize=(width, height))

    im = ax.imshow(corr.values, aspect="auto", vmin=vmin, vmax=vmax, cmap="RdBu_r")

    ax.set_title(title)
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)

    if annotate and rows <= 24 and cols <= 12:
        for i in range(rows):
            for j in range(cols):
                val = corr.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    path = OUT_DIR / filename
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[save] {path}")


def save_bar(series: pd.Series, title: str, filename: str) -> None:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        print(f"[skip] {filename}: empty")
        return

    series = series.reindex(series.abs().sort_values(ascending=True).index)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(series) + 1.5)))
    ax.barh(series.index.astype(str), series.values)
    ax.axvline(0.0, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(f"{CORR_METHOD.title()} correlation")
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[save] {path}")


def save_etf_return_corr(price_matrix: pd.DataFrame | None) -> None:
    if price_matrix is None or price_matrix.empty:
        return

    monthly_prices = monthly_last(price_matrix)
    returns = monthly_prices.pct_change(fill_method=None)

    corr = returns.corr(method="pearson")
    corr.to_csv(OUT_DIR / "00_etf_monthly_return_corr.csv")

    save_heatmap(
        corr,
        "ETF monthly return correlation",
        "00_etf_monthly_return_corr.png",
        annotate=True,
    )

    level_corr = monthly_prices.corr(method="pearson")
    level_corr.to_csv(OUT_DIR / "01_etf_price_level_corr_spurious_check.csv")

    save_heatmap(
        level_corr,
        "ETF price-level correlation - useful only as a rough spurious-beta check",
        "01_etf_price_level_corr_spurious_check.png",
        annotate=True,
    )


def build_price_targets(ticker: str, price_matrix: pd.DataFrame) -> pd.DataFrame:
    px = monthly_last(price_matrix[[ticker]].dropna()).rename(columns={ticker: "price_level"})

    out = px.copy()
    out["return_1m"] = out["price_level"].pct_change(1)
    out["return_3m"] = out["price_level"].pct_change(3)

    out["next_return_1m"] = out["price_level"].shift(-1) / out["price_level"] - 1.0
    out["next_return_3m"] = out["price_level"].shift(-3) / out["price_level"] - 1.0
    out["next_return_6m"] = out["price_level"].shift(-6) / out["price_level"] - 1.0

    return out


def build_overlay_frame(
    ticker: str,
    spec: dict,
    price_matrix: pd.DataFrame,
    final_scores: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    pieces = []

    feature_df = load_csv(f"{ticker} {spec['name']} features", spec["feature_patterns"])

    if feature_df is not None and "date" in feature_df.columns:
        feature_df = feature_df.set_index("date").sort_index()
        feature_part = to_numeric_frame(feature_df, spec["feature_cols"])
        feature_part = monthly_last(feature_part)
        pieces.append(feature_part)

    if final_scores is not None:
        one = final_scores[final_scores["ticker"] == ticker].copy()

        if not one.empty:
            one = one.set_index("date").sort_index()
            score_part = to_numeric_frame(one, spec["final_score_cols"])
            score_part = monthly_last(score_part)
            pieces.append(score_part)

    if not pieces:
        return pd.DataFrame(), [], []

    features = pd.concat(pieces, axis=1)

    # Remove duplicate columns while preserving order.
    features = features.loc[:, ~features.columns.duplicated()]
    features = clean_for_corr(features)

    if ticker not in price_matrix.columns:
        print(f"[skip] {ticker}: no price data")
        return pd.DataFrame(), [], []

    targets = build_price_targets(ticker, price_matrix)

    data = features.join(targets, how="inner")
    data = clean_for_corr(data)

    feature_cols = [c for c in features.columns if c in data.columns]
    target_cols = [c for c in TARGET_COLS if c in data.columns]

    return data, feature_cols, target_cols


def save_overlay_corrs(
    ticker: str,
    spec: dict,
    price_matrix: pd.DataFrame,
    final_scores: pd.DataFrame | None,
) -> None:
    data, feature_cols, target_cols = build_overlay_frame(
        ticker=ticker,
        spec=spec,
        price_matrix=price_matrix,
        final_scores=final_scores,
    )

    if data.empty or len(feature_cols) == 0 or len(target_cols) == 0:
        print(f"[skip] {ticker}: no usable overlay/price correlation data")
        return

    if len(data.dropna(how="all")) < MIN_OBS:
        print(f"[skip] {ticker}: less than {MIN_OBS} usable monthly observations")
        return

    corr = data[feature_cols + target_cols].corr(method=CORR_METHOD)

    target_corr = corr.loc[feature_cols, target_cols]
    target_corr.to_csv(OUT_DIR / f"{ticker}_overlay_vs_price_target_corr.csv")

    save_heatmap(
        target_corr,
        f"{ticker}: overlay indicators vs price/forward returns ({CORR_METHOD})",
        f"{ticker}_overlay_vs_price_target_corr.png",
        annotate=True,
    )

    # A tighter matrix that excludes raw price level. This is usually the serious one.
    serious_targets = [c for c in ["return_1m", "return_3m", "next_return_1m", "next_return_3m", "next_return_6m"] if c in target_corr.columns]
    if serious_targets:
        save_heatmap(
            target_corr[serious_targets],
            f"{ticker}: overlay indicators vs return targets ({CORR_METHOD})",
            f"{ticker}_overlay_vs_return_targets_only.png",
            annotate=True,
        )

    # Full matrix, but only if not huge. Huge full matrices are exactly what made the old version useless.
    full_cols = feature_cols + target_cols
    if len(full_cols) <= 22:
        full_corr = corr.loc[full_cols, full_cols]
        full_corr.to_csv(OUT_DIR / f"{ticker}_overlay_full_corr.csv")

        save_heatmap(
            full_corr,
            f"{ticker}: full overlay correlation matrix ({CORR_METHOD})",
            f"{ticker}_overlay_full_corr.png",
            annotate=False,
        )

    # Rank the features by relationship to future returns.
    for target in ["next_return_1m", "next_return_3m", "next_return_6m"]:
        if target not in target_corr.columns:
            continue

        ranked = target_corr[target].dropna()
        ranked = ranked.reindex(ranked.abs().sort_values(ascending=False).index).head(12)

        ranked.to_csv(OUT_DIR / f"{ticker}_top_corr_{target}.csv")

        save_bar(
            ranked,
            f"{ticker}: strongest overlay correlations with {target}",
            f"{ticker}_top_corr_{target}.png",
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("OVERLAY CORRELATION MATRICES")
    print("=" * 80)
    print(f"Base dir:   {BASE_DIR}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Method:     {CORR_METHOD}")
    print(f"Min obs:    {MIN_OBS} monthly observations")

    price_matrix = load_price_matrix()
    if price_matrix is None or price_matrix.empty:
        raise ValueError("No usable commodity price matrix found.")

    final_scores = load_final_scores()

    save_etf_return_corr(price_matrix)

    for ticker, spec in OVERLAY_SPECS.items():
        print("\n" + "-" * 80)
        print(f"{ticker}: {spec['name']}")
        print("-" * 80)
        save_overlay_corrs(
            ticker=ticker,
            spec=spec,
            price_matrix=price_matrix,
            final_scores=final_scores,
        )

    print("\n" + "=" * 80)
    print(f"DONE. Charts saved to: {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
