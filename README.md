# Systematic Commodity Allocation Engine

A research and paper-trading system that converts release-aware market and fundamental data into comparable commodity scores, constrained portfolio allocations and monitored trading decisions.

The project was built to test whether a systematic, risk-controlled allocation process can outperform simple commodity benchmarks without relying on a single asset, parameter set or backtest period.

<img width="1672" height="941" alt="pipeline" src="https://github.com/user-attachments/assets/bb5294ef-d6c1-48ec-ab86-066afcefa27f" />

## Methodology

The model ranks liquid commodity ETFs using a shared base score:

- Momentum — 23%
- Relative strength — 16%
- Trend — 3%
- Trend persistence — 10%
- Volatility — 19%
- Risk — 29%

These signals are combined with asset-specific overlays for gold, silver, copper, oil, natural gas and agriculture. Inputs include prices, real yields, the US dollar, Chinese industrial activity, EIA balances, weather and agricultural data.

All non-price data are aligned to the date on which they were available using as-of joins and explicit release lags. Portfolio construction applies score thresholds, position and group caps, cash allocation, turnover controls, execution delay and liquidity limits.

## Backtest and validation

The strategy was evaluated using:

- Realistic commissions, bid–ask spreads, slippage and delayed execution
- Equal-weight, inverse-volatility, trend-following and gold-only benchmarks
- Walk-forward testing across multiple market periods
- Backward out-of-sample holdouts, including the 2013–2015 commodity bear market
- Feature and overlay ablation tests
- 1,000 random parameter perturbations
- Cost, liquidity and regime stress tests
- Selected-versus-rejected forward-return analysis
- VaR, CVaR, drawdown, correlation and turnover diagnostics

<img width="1741" height="944" alt="equitycurve" src="https://github.com/user-attachments/assets/e1cf519e-e849-4b68-91a8-31fe72b41445" />

## Out-of-sample failure and rectification

A backward holdout was deliberately extended into the 2013–2015 commodity bear market. The original long-only model failed this regime, producing approximately **-3.9% CAGR**, **-0.78 Sharpe** and a **-21.0% maximum drawdown** over the hostile window. The failure showed that low scores were effective for avoiding exposure, but the strategy could not profit from persistent broad commodity declines.

A defensive regime router was tested first. It reduced drawdown by holding more cash, but was rejected because the improvement came with a material loss of return. A narrower **bear-only short overlay** was then added without changing the underlying scoring model.

Across the full 2013–2026 test, the final model improved the long-only baseline from **12.1% to 13.0% CAGR**, increased cash-adjusted Sharpe from **0.87 to 0.93** and reduced maximum drawdown from **-21.0% to -15.5%**. Results over 2015–2026 were broadly unchanged, indicating that the overlay primarily addressed the missing bear-regime behaviour rather than inflating the original sample.

## Results

The table below uses the realistic full-period 2013–2026 backtest, including estimated costs and the hostile backward holdout.

| Strategy | CAGR | Adjusted Sharpe | Max drawdown | Average cash |
|---|---:|---:|---:|---:|
| Frozen model | 13.0% | 0.93 | -15.5% | 50.3% |
| Long-only model | 12.1% | 0.87 | -21.0% | 53.1% |
| Gold only | 9.3% | 0.40 | -26.2% | 7.1% |
| Trend following | 5.4% | 0.16 | -41.2% | 17.2% |
| Equal weight | 2.3% | -0.02 | -55.8% | 7.6% |

*Adjusted Sharpe is calculated using cash-adjusted excess returns.*

<img width="2296" height="980" alt="drawdowns" src="https://github.com/user-attachments/assets/985dc4b5-3cd2-49d2-a6f4-eadaf7b829c0" />

## Robustness

Walk-forward results test performance across changing market environments rather than one full-sample fit. Random perturbations test whether performance survives nearby parameter choices, while ablations isolate the contribution of individual signals and overlays.

<img width="2296" height="1037" alt="walk_forward_periods" src="https://github.com/user-attachments/assets/e869c171-b532-4a8d-a954-a0db303ec787" />

<img width="1800" height="1260" alt="perturbations" src="https://github.com/user-attachments/assets/62d0f53e-04ff-4f0e-903d-44d6639dc24a" />

The selected-versus-rejected test provides a direct check of whether the ranking process separates stronger subsequent returns from weaker opportunities.

<img width="2048" height="950" alt="selectedrejected" src="https://github.com/user-attachments/assets/cbebce8a-cd5c-4c04-9c99-b4da0e4a3e5f" />

Forward returns are measured over the next rebalance period (1 month). Selected assets subsequently outperformed rejected assets across the universe, providing a direct test of efficacy.

<img width="2048" height="968" alt="var" src="https://github.com/user-attachments/assets/d466136a-a442-476c-b213-95ad6547ae4b" />

The distribution of net daily returns remains concentrated around zero, with historical 95% and 99% VaR thresholds used to quantify left-tail risk. Note fat tails can be observed.

<img width="1485" height="1313" alt="heatmap" src="https://github.com/user-attachments/assets/6531740d-c5c1-4c05-9464-bdc3680c6a4a" />

Daily asset returns were generally weakly correlated, supporting cross-commodity diversification, the principal exception was 0.78 between gold and silver - a very expected result.

<img width="1923" height="943" alt="returns" src="https://github.com/user-attachments/assets/57ffc3a2-8764-4515-aaf2-47058660ac60" />

All six assets contributed positively to cumulative returns. Gold was the largest contributor, but performance was not solely dependent on precious metals.

## Full research log

The complete development history—including rejected experiments, ablation,
walk-forward validation, parameter sensitivity, execution realism, backward
holdout failure and bear-regime rectification—is available in the
[Commodity Research Log](Commodity%20Research%20Log.pdf).

## Research findings

- Risk controls and selective exposure were more important than continuous full investment.
- Results were not explained solely by gold exposure or a generic trend strategy.
- Backward out-of-sample testing exposed a genuine bear-regime failure that was not visible in the stronger 2015+ sample.
- A broad defensive router was rejected; targeted bear-only shorting produced a better return-risk trade-off.
- Release-aware alignment, transaction costs and liquidity constraints materially reduced headline backtest performance but produced a more defensible result.
- Robustness tests support the broader model region rather than one precisely tuned configuration.

## Paper trading

The frozen configuration, including the conditional bear-only short overlay, runs through a scheduled paper-trading process with monthly rebalancing. The system records scores, target weights, cash, trades, execution diagnostics and portfolio state separately from the research environment.

No live-capital performance is claimed. Backtested results are not evidence of future returns.

## Limitations

- Commodity ETFs introduce roll yield, tracking error and structural differences from futures exposure.
- Fundamental datasets have different histories, revisions and release schedules.
- Some asset-specific overlays have limited independent sample sizes.
- Walk-forward and perturbation tests reduce, but do not eliminate, overfitting risk.
- Paper trading cannot fully reproduce live liquidity, market impact or operational failure.

## Status

The research model is frozen and in paper trading. Current work focuses on monitoring live decision behaviour, maintaining the data pipeline and comparing paper results with backtest expectations.

---

*Research project only. Nothing in this repository constitutes investment advice.*
