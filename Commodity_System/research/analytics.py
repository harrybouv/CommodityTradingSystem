# research/analytics.py

import numpy as np
import pandas as pd


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_returns(returns: pd.Series) -> pd.Series:
    returns = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    return returns.astype(float)


def build_equity_curve(
    returns: pd.Series,
    initial_capital: float = 10_000,
) -> pd.Series:
    returns = clean_returns(returns)
    return initial_capital * (1 + returns).cumprod()


def calculate_total_return(returns: pd.Series) -> float:
    returns = clean_returns(returns)
    if returns.empty:
        return np.nan
    return (1 + returns).prod() - 1


def calculate_cagr(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    returns = clean_returns(returns)
    if returns.empty:
        return np.nan

    total_return = calculate_total_return(returns)
    years = len(returns) / periods_per_year

    if years <= 0:
        return np.nan

    return (1 + total_return) ** (1 / years) - 1


def calculate_annualised_return(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    returns = clean_returns(returns)
    if returns.empty:
        return np.nan
    return returns.mean() * periods_per_year


def calculate_annualised_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    returns = clean_returns(returns)
    if len(returns) < 2:
        return np.nan
    return returns.std() * np.sqrt(periods_per_year)


# ============================================================
# RISK-ADJUSTED RETURNS
# ============================================================

def annual_to_periodic_rate(
    annual_rate: float,
    periods_per_year: int = 252,
) -> float:
    return (1 + annual_rate) ** (1 / periods_per_year) - 1


def calculate_sharpe(
    returns: pd.Series,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    returns = clean_returns(returns)
    if len(returns) < 2:
        return np.nan

    rf_periodic = annual_to_periodic_rate(risk_free_rate_annual, periods_per_year)
    excess = returns - rf_periodic

    if excess.std() == 0:
        return np.nan

    return excess.mean() / excess.std() * np.sqrt(periods_per_year)


def calculate_sortino(
    returns: pd.Series,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    returns = clean_returns(returns)
    if len(returns) < 2:
        return np.nan

    rf_periodic = annual_to_periodic_rate(risk_free_rate_annual, periods_per_year)
    excess = returns - rf_periodic

    downside = excess[excess < 0]

    if len(downside) < 2 or downside.std() == 0:
        return np.nan

    return excess.mean() / downside.std() * np.sqrt(periods_per_year)


def calculate_calmar(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    cagr = calculate_cagr(returns, periods_per_year)
    max_dd = calculate_max_drawdown(returns)

    if pd.isna(cagr) or pd.isna(max_dd) or max_dd == 0:
        return np.nan

    return cagr / abs(max_dd)


def calculate_information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    df = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if df.empty:
        return np.nan

    active = df.iloc[:, 0] - df.iloc[:, 1]
    tracking_error = active.std() * np.sqrt(periods_per_year)

    if tracking_error == 0:
        return np.nan

    active_return = active.mean() * periods_per_year
    return active_return / tracking_error


# ============================================================
# DRAWDOWN
# ============================================================

def calculate_drawdown_series(returns: pd.Series) -> pd.Series:
    returns = clean_returns(returns)
    equity = (1 + returns).cumprod()
    running_high = equity.cummax()
    return equity / running_high - 1


def calculate_max_drawdown(returns: pd.Series) -> float:
    dd = calculate_drawdown_series(returns)
    if dd.empty:
        return np.nan
    return dd.min()


def calculate_max_drawdown_date(returns: pd.Series):
    dd = calculate_drawdown_series(returns)
    if dd.empty:
        return pd.NaT
    return dd.idxmin()


def calculate_drawdown_duration(returns: pd.Series) -> int:
    dd = calculate_drawdown_series(returns)

    if dd.empty:
        return 0

    underwater = dd < 0
    max_duration = 0
    current_duration = 0

    for value in underwater:
        if value:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return max_duration


# ============================================================
# WIN / LOSS STATS
# ============================================================

def calculate_hit_rate(returns: pd.Series) -> float:
    returns = clean_returns(returns)
    if returns.empty:
        return np.nan
    return (returns > 0).mean()


def calculate_average_win(returns: pd.Series) -> float:
    returns = clean_returns(returns)
    winners = returns[returns > 0]
    if winners.empty:
        return np.nan
    return winners.mean()


def calculate_average_loss(returns: pd.Series) -> float:
    returns = clean_returns(returns)
    losers = returns[returns < 0]
    if losers.empty:
        return np.nan
    return losers.mean()


def calculate_win_loss_ratio(returns: pd.Series) -> float:
    avg_win = calculate_average_win(returns)
    avg_loss = calculate_average_loss(returns)

    if pd.isna(avg_win) or pd.isna(avg_loss) or avg_loss == 0:
        return np.nan

    return avg_win / abs(avg_loss)


def calculate_profit_factor(returns: pd.Series) -> float:
    returns = clean_returns(returns)

    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())

    if gross_loss == 0:
        return np.nan

    return gross_profit / gross_loss


# ============================================================
# DISTRIBUTION / TAIL RISK
# ============================================================

def calculate_skew(returns: pd.Series) -> float:
    returns = clean_returns(returns)
    return returns.skew()


def calculate_kurtosis(returns: pd.Series) -> float:
    returns = clean_returns(returns)
    return returns.kurtosis()


def calculate_var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    returns = clean_returns(returns)
    if returns.empty:
        return np.nan
    return returns.quantile(1 - confidence)


def calculate_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    returns = clean_returns(returns)
    if returns.empty:
        return np.nan

    var = calculate_var(returns, confidence)
    tail = returns[returns <= var]

    if tail.empty:
        return np.nan

    return tail.mean()


# ============================================================
# TURNOVER / COSTS / EXPOSURE
# ============================================================

def calculate_turnover(weights: pd.DataFrame) -> pd.Series:
    weights = weights.fillna(0)
    turnover = weights.diff().abs().sum(axis=1)

    if not turnover.empty:
        turnover.iloc[0] = weights.iloc[0].abs().sum()

    return turnover


def calculate_cost_drag(
    turnover: pd.Series,
    total_cost_bps: float,
) -> pd.Series:
    return turnover * (total_cost_bps / 10_000)


def calculate_exposure(weights: pd.DataFrame) -> pd.Series:
    return weights.fillna(0).sum(axis=1)


def calculate_exposure_stats(exposure: pd.Series) -> dict:
    exposure = pd.Series(exposure).replace([np.inf, -np.inf], np.nan).dropna()

    if exposure.empty:
        return {
            "average_exposure": np.nan,
            "max_exposure": np.nan,
            "min_exposure": np.nan,
            "average_cash": np.nan,
        }

    return {
        "average_exposure": exposure.mean(),
        "max_exposure": exposure.max(),
        "min_exposure": exposure.min(),
        "average_cash": 1 - exposure.mean(),
    }


# ============================================================
# ALPHA / BETA / REGRESSION
# ============================================================

def calculate_alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    df = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    df.columns = ["strategy", "benchmark"]

    if len(df) < 3:
        return {
            "alpha_daily": np.nan,
            "alpha_annualised": np.nan,
            "beta": np.nan,
            "r_squared": np.nan,
            "correlation": np.nan,
            "tracking_error": np.nan,
            "information_ratio": np.nan,
            "alpha_t_stat": np.nan,
            "beta_t_stat": np.nan,
        }

    rf_periodic = annual_to_periodic_rate(risk_free_rate_annual, periods_per_year)

    y = df["strategy"] - rf_periodic
    x = df["benchmark"] - rf_periodic

    if x.var() == 0:
        return {
            "alpha_daily": np.nan,
            "alpha_annualised": np.nan,
            "beta": np.nan,
            "r_squared": np.nan,
            "correlation": np.nan,
            "tracking_error": np.nan,
            "information_ratio": np.nan,
            "alpha_t_stat": np.nan,
            "beta_t_stat": np.nan,
        }

    beta = y.cov(x) / x.var()
    alpha_daily = y.mean() - beta * x.mean()
    alpha_annualised = (1 + alpha_daily) ** periods_per_year - 1

    y_hat = alpha_daily + beta * x
    residual = y - y_hat

    sse = (residual ** 2).sum()
    sst = ((y - y.mean()) ** 2).sum()

    r_squared = 1 - sse / sst if sst != 0 else np.nan
    correlation = y.corr(x)

    tracking_error = residual.std() * np.sqrt(periods_per_year)
    information_ratio = (
        alpha_annualised / tracking_error
        if tracking_error != 0
        else np.nan
    )

    n = len(df)
    x_centered_sum_sq = ((x - x.mean()) ** 2).sum()

    if n > 2 and x_centered_sum_sq != 0:
        residual_variance = sse / (n - 2)
        se_beta = np.sqrt(residual_variance / x_centered_sum_sq)
        se_alpha = np.sqrt(
            residual_variance
            * (
                1 / n
                + (x.mean() ** 2) / x_centered_sum_sq
            )
        )

        beta_t_stat = beta / se_beta if se_beta != 0 else np.nan
        alpha_t_stat = alpha_daily / se_alpha if se_alpha != 0 else np.nan
    else:
        alpha_t_stat = np.nan
        beta_t_stat = np.nan

    return {
        "alpha_daily": alpha_daily,
        "alpha_annualised": alpha_annualised,
        "beta": beta,
        "r_squared": r_squared,
        "correlation": correlation,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "alpha_t_stat": alpha_t_stat,
        "beta_t_stat": beta_t_stat,
    }


# ============================================================
# ROLLING METRICS
# ============================================================

def calculate_rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    returns = clean_returns(returns)
    rf_periodic = annual_to_periodic_rate(risk_free_rate_annual, periods_per_year)
    excess = returns - rf_periodic

    return (
        excess.rolling(window).mean()
        / excess.rolling(window).std()
        * np.sqrt(periods_per_year)
    )


def calculate_rolling_volatility(
    returns: pd.Series,
    window: int = 252,
    periods_per_year: int = 252,
) -> pd.Series:
    returns = clean_returns(returns)
    return returns.rolling(window).std() * np.sqrt(periods_per_year)


def calculate_rolling_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 252,
) -> pd.Series:
    df = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    df.columns = ["strategy", "benchmark"]

    rolling_cov = df["strategy"].rolling(window).cov(df["benchmark"])
    rolling_var = df["benchmark"].rolling(window).var()

    return rolling_cov / rolling_var


# ============================================================
# MONTHLY / YEARLY RETURNS
# ============================================================

def calculate_monthly_returns(returns: pd.Series) -> pd.DataFrame:
    returns = clean_returns(returns)

    monthly = (1 + returns).resample("ME").prod() - 1

    table = monthly.to_frame("monthly_return")
    table["year"] = table.index.year
    table["month"] = table.index.month

    return table


def calculate_monthly_returns_pivot(returns: pd.Series) -> pd.DataFrame:
    monthly = calculate_monthly_returns(returns)

    pivot = monthly.pivot(
        index="year",
        columns="month",
        values="monthly_return",
    )

    pivot = pivot.sort_index()
    return pivot


def calculate_annual_returns(returns: pd.Series) -> pd.Series:
    returns = clean_returns(returns)
    return (1 + returns).resample("YE").prod() - 1


# ============================================================
# CONTRIBUTION
# ============================================================

def calculate_asset_contribution(
    used_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
) -> pd.DataFrame:
    used_weights = used_weights.reindex(asset_returns.index).fillna(0)
    asset_returns = asset_returns.reindex(used_weights.index).fillna(0)

    common_cols = used_weights.columns.intersection(asset_returns.columns)
    used_weights = used_weights[common_cols]
    asset_returns = asset_returns[common_cols]

    daily_contribution = used_weights * asset_returns
    total_contribution = daily_contribution.sum().sort_values(ascending=False)

    out = total_contribution.rename("total_return_contribution").reset_index()
    out = out.rename(columns={"index": "ticker"})

    total_abs = out["total_return_contribution"].abs().sum()
    if total_abs != 0:
        out["contribution_share_abs"] = (
            out["total_return_contribution"].abs() / total_abs
        )
    else:
        out["contribution_share_abs"] = np.nan

    return out


# ============================================================
# FULL SUMMARY
# ============================================================

def calculate_full_summary(
    returns: pd.Series,
    equity: pd.Series | None = None,
    turnover: pd.Series | None = None,
    transaction_cost: pd.Series | None = None,
    exposure: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
    strategy_name: str = "strategy",
    benchmark_name: str | None = None,
    initial_capital: float = 10_000,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    returns = clean_returns(returns)

    if equity is None:
        equity = build_equity_curve(returns, initial_capital=initial_capital)

    drawdown = calculate_drawdown_series(returns)

    summary = {
        "strategy": strategy_name,
        "benchmark": benchmark_name,
        "start_date": returns.index.min().date() if len(returns) else None,
        "end_date": returns.index.max().date() if len(returns) else None,
        "observations": len(returns),

        "initial_capital": initial_capital,
        "final_equity": equity.iloc[-1] if len(equity) else np.nan,

        "total_return": calculate_total_return(returns),
        "cagr": calculate_cagr(returns, periods_per_year),
        "annualised_return": calculate_annualised_return(returns, periods_per_year),
        "annualised_volatility": calculate_annualised_volatility(returns, periods_per_year),

        "sharpe": calculate_sharpe(
            returns,
            risk_free_rate_annual=risk_free_rate_annual,
            periods_per_year=periods_per_year,
        ),
        "sortino": calculate_sortino(
            returns,
            risk_free_rate_annual=risk_free_rate_annual,
            periods_per_year=periods_per_year,
        ),
        "calmar": calculate_calmar(returns, periods_per_year),

        "max_drawdown": calculate_max_drawdown(returns),
        "max_drawdown_date": calculate_max_drawdown_date(returns),
        "max_drawdown_duration_days": calculate_drawdown_duration(returns),

        "hit_rate": calculate_hit_rate(returns),
        "average_win": calculate_average_win(returns),
        "average_loss": calculate_average_loss(returns),
        "win_loss_ratio": calculate_win_loss_ratio(returns),
        "profit_factor": calculate_profit_factor(returns),

        "best_day": returns.max() if len(returns) else np.nan,
        "worst_day": returns.min() if len(returns) else np.nan,

        "skew": calculate_skew(returns),
        "kurtosis": calculate_kurtosis(returns),
        "var_95": calculate_var(returns, confidence=0.95),
        "cvar_95": calculate_cvar(returns, confidence=0.95),
        "var_99": calculate_var(returns, confidence=0.99),
        "cvar_99": calculate_cvar(returns, confidence=0.99),
    }

    if turnover is not None:
        turnover = pd.Series(turnover).replace([np.inf, -np.inf], np.nan).dropna()
        summary.update(
            {
                "average_daily_turnover": turnover.mean(),
                "median_daily_turnover": turnover.median(),
                "max_daily_turnover": turnover.max(),
                "annualised_turnover": turnover.mean() * periods_per_year,
            }
        )

    if transaction_cost is not None:
        transaction_cost = pd.Series(transaction_cost).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        summary.update(
            {
                "total_transaction_cost_drag": transaction_cost.sum(),
                "average_daily_transaction_cost": transaction_cost.mean(),
                "annualised_transaction_cost_drag": (
                    transaction_cost.mean() * periods_per_year
                ),
            }
        )

    if exposure is not None:
        summary.update(calculate_exposure_stats(exposure))

    if benchmark_returns is not None:
        regression = calculate_alpha_beta(
            strategy_returns=returns,
            benchmark_returns=benchmark_returns,
            risk_free_rate_annual=risk_free_rate_annual,
            periods_per_year=periods_per_year,
        )

        summary.update(regression)

        summary["information_ratio_active"] = calculate_information_ratio(
            returns,
            benchmark_returns,
            periods_per_year=periods_per_year,
        )

    return summary