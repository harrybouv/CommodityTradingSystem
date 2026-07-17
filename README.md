# Commodity Allocation and Risk Engine

A systematic multi-commodity research and paper-trading system built in Python. The project combines asset-level fundamental and macroeconomic signals with trend, momentum, relative strength, volatility and portfolio-risk controls across gold, silver, copper, crude oil, natural gas and agricultural commodities.

## What the system does

- Builds automated market, macroeconomic and commodity-specific data pipelines
- Engineers asset-level features and converts them into comparable model scores
- Constructs a diversified portfolio with volatility, exposure and risk controls
- Models transaction costs, execution delays, turnover and liquidity constraints
- Runs walk-forward validation, feature ablation and parameter-sensitivity analysis
- Evaluates historical regimes, asset shocks and benchmark decomposition
- Supports regime-aware allocation, short-overlay research and paper trading
- Exports portfolio state, allocations and diagnostics for monitoring

## Research workflow

The strategy was developed across multiple backtest generations rather than as a single fitted model. Validation includes walk-forward testing, feature ablation, parameter perturbation, execution-cost sensitivity, delayed-execution tests, historical-window stress tests and asset-contribution analysis.

Historical outputs are research results rather than live trading performance. They depend on the selected period, available data, execution assumptions and model configuration.

## Repository structure

```text
Commodity_System/
├── scoring/                  # Asset models, features and score aggregation
├── research/
│   ├── Backtesting/          # Backtests, stress tests and walk-forward analysis
│   └── Parameter Testing/    # Sensitivity and overlay research
├── paper_trading/            # Paper-trading runner and reporting tools
├── commodity_strategy.py     # Portfolio strategy logic
├── macro_data.py             # Macroeconomic data pipeline
├── config.py                 # Central configuration and paths
└── backtest_runner.py        # Main backtest entry point
```

Generated datasets, backtest outputs, logs and mutable paper-trading state are excluded from version control.

## Installation

```bash
git clone https://github.com/harrybouv/CommodityTradingSystem.git
cd CommodityTradingSystem
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## API configuration

Create a `.env` file in the repository root:

```dotenv
EIA_API_KEY=your_eia_key
EMBER_API_KEY=your_ember_key
FAS_API_KEY=your_fas_key
```

API credentials are loaded locally and must never be committed.

## Disclaimer

This is an independent research and educational project. It is not investment advice, and historical or simulated performance does not guarantee future results.
