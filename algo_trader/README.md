# Algo Trader (Current Version)

This is the production-ready automated trading system, superseding the legacy `trading-bot`. It features a modular architecture, sophisticated risk management, and an advanced vectorized backtesting engine.

## Architecture

The system is organized into specialized modules for better maintainability and testing:

```text
algo_trader/
├── backtest/        # Vectorized simulation engine and metrics
├── config/          # Centralized settings and ticker universe
├── data/            # Data fetching (historical/stream) and indicators
├── execution/       # Order submission and reconciliation
├── monitoring/      # Logging, Discord alerts, and live dashboard
├── risk/            # Position sizing, PDT tracker, and circuit breakers
└── strategy/        # Signal generation, ranking, and regime detection
```

## Core Functionality

### 1. Strategy Logic
The bot operates primarily on **15-minute bars** and evaluates multiple technical indicators:
- **EMA Crossover**: 9 EMA (Fast) vs 21 EMA (Slow).
- **RSI Momentum**: Cross-over triggers (55 long) or bounces from oversold (35).
- **Volume Surge**: Requires volume >= 1.5x the 20-period SMA.
- **VWAP Filter**: Only takes long trades when price is above VWAP.
- **Trend Filter**: Price must be above the 50 EMA (Bias).
- **Regime Detection**: SPY daily trend classification (BULLISH, CAUTIOUS, BEARISH, CRISIS) to filter trades and adjust position sizes.

### 2. Risk Management
- **Risk-Based Sizing**: Calculates shares based on a 1% risk-per-trade model (Entry - Stop Loss).
- **Multi-Factor Scaling**: Position sizes are automatically reduced during:
  - **Bearish Regimes** (SPY trend).
  - **High Equity Drawdown** (Trailing peak equity check).
  - **Bearish QQQ Trend** (Directional risk reduction).
- **PDT Tracker**: Automatically enforces the 3-day-trade-per-5-day limit for accounts under $25,000.
- **Circuit Breaker**: Monitors daily, weekly, and monthly P&L to halt trading if loss limits are hit.
- **Portfolio Heat**: Manages total account exposure and sector-specific concentration.

### 3. Backtesting
The **Vectorized Backtesting Engine** (`backtest/engine.py`) allows for rapid simulation of years of data:
- **Realistic Costs**: Accounts for slippage, spread, and regulatory fees.
- **Overnight Support**: Options to hold positions overnight (`--swing`) or close by EOD.
- **Metrics**: Provides Sharpe Ratio, Sortino Ratio, Max Drawdown, and monthly P&L summaries.

## Usage

### Running the Live Bot
```bash
python main.py --paper  # Run in paper trading mode
python main.py --live   # Run in live mode (requires confirmation)
```

### Running a Backtest
```bash
python backtest_runner.py --start 2024-01-01 --end 2024-12-31 --tickers AAPL NVDA
```
*Add `--swing` to simulate swing trading (holding overnight).*

## Live Strategy vs. Backtesting
The backtesting engine is designed to mirror the live strategy logic. Note that:
- **Confirmations**: The live strategy requires 4 out of 6 technical confirmations (including EMA spread), while the backtester defaults to 3 out of 5 for broader signal capture.
- **Fills**: Backtests fill at the **open** of the next bar to prevent look-ahead bias, identical to how the live bot reacts to bar closures.

---
**Warning**: Trading stocks involves significant risk of loss. Always test thoroughly in paper mode before deploying real capital.
