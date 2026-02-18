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
├── strategy/        # Signal generation, ranking, and regime detection
```

## Core Functionality

### 1. Strategy Logic
The bot operates primarily on **15-minute bars** and evaluates multiple technical indicators:
- **EMA Crossover**: 9 EMA (Fast) vs 21 EMA (Slow).
- **RSI Momentum**: Cross-over triggers (55 long) or bounces from oversold (35).
- **Volume Surge**: Requires volume >= 1.2x the 20-bar average.
- **VWAP Filter**: Only takes long trades when price is above VWAP.
- **Trend Filter**: Price must be above the 50 EMA (Bias).
- **Regime Detection**: SPY daily trend classification (BULLISH, CAUTIOUS, BEARISH, CRISIS) to filter trades and adjust position sizes.

### 2. Risk Management
- **Risk-Based Sizing**: Calculates shares based on a dynamic risk model (6.7% risk per trade).
- **Multi-Factor Scaling**: Position sizes are automatically reduced during:
  - **Bearish Regimes** (SPY trend).
  - **High Equity Drawdown** (Trailing peak equity check).
  - **Volatility Spikes**: SPY ATR > 1.5x average (Volatility Gating).
- **PDT Tracker**: Automatically enforces the 3-day-trade-per-5-day limit for accounts under $25,000.
- **Circuit Breaker**: Monitors daily, weekly, and monthly P&L to halt trading if loss limits are hit.
- **Portfolio Heat**: Manages total account exposure (max 90%) and sector-concentration.

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
python backtest_runner.py --start 2026-01-15 --end 2026-02-15 --tickers AAPL MSFT NVDA
```

## The Trading Decision Flow (How the Bot Decides)

The bot follows a 5-step funnel to move from 70+ symbols down to specific risk-managed trades:

### Phase 1: Market Regime Identification
*   **File**: [`strategy/regime.py`](file:///E:/trading/algo_trader/strategy/regime.py)
*   The system analyzes SPY and VIX to determine the market environment:
    *   **BULLISH**: Aggressive risk (6.7%), loose confirmations.
    *   **CAUTIOUS**: Moderate risk (5.2%), Trend + Mean Reversion logic.
    *   **BEARISH**: Defensive risk (2.5%), Mean Reversion focus.
    *   **CRISIS**: Trading halted (VIX > 40).

### Phase 2: Rolling Leadership Ranking (Elite Selection)
*   **File**: [`strategy/signals.py`](file:///E:/trading/algo_trader/strategy/signals.py) (Logic) & [`main.py`](file:///E:/trading/algo_trader/main.py) (Calculation)
*   **Hindsight-Free Dynamic Tiering** (Phase 108):
    *   **Elite Selection**: Only the **Top 20** ranked stocks (by 125-day ROC) are allowed to trigger signals.
    *   **Tiered Targets**: 
        *   **Rank 1-12 (Elite Leaders)**: 25.0x ATR Target.
        *   **Rank 13-20 (Runners)**: 15.0x ATR Target.

### Phase 3: Technical Signal Confirmation
*   **File**: [`strategy/signals.py`](file:///E:/trading/algo_trader/strategy/signals.py)
*   **Trend Following** (BULLISH/CAUTIOUS):
    *   **Trend**: Price > Bias EMA (50) and Fast EMA > Slow EMA.
    *   **Momentum**: RSI crossing 55 or bouncing from 35.
    *   **Trend Strength**: ADX >= 30.0 and rising.
    *   **Volume**: Volume >= 1.2x average.
*   **Mean Reversion** (CAUTIOUS/BEARISH):
    *   **Trigger**: RSI(2) < 10 AND Price < Lower Bollinger Band.

### Phase 4: Position Sizing & Volatility Gating
*   **File**: [`risk/position_sizer.py`](file:///E:/trading/algo_trader/risk/position_sizer.py)
*   **Dynamic Sizing**: Uses a risk-based model (e.g., 6.7% account risk per trade).
*   **Volatility Gating** (Phase 112): Position size is cut by **50%** if SPY's current ATR is > 1.5x its average (Volatility spike).
*   **Safety Caps**: Max 60% position pct, 90% total portfolio heat, and sector limits.

### Phase 5: Trade management & Advanced Exits
*   **File**: [`main.py`](file:///E:/trading/algo_trader/main.py)
*   **EOD Profit Locking**: Profitable positions are closed starting at 3:30 PM ET to lock in gains.
*   **Dead Money Stop**: Exit if held > 48 hours with < 1.0x ATR profit.
*   **Break-Even Guardrail**: Move SL to BE + offset after significant ATR expansion.
*   **Partial Exit**: Sell 10-20% at 3.5x ATR to "fireproof" the trade.

---
**Warning**: Trading stocks involves significant risk of loss. Always test thoroughly in paper mode before deploying real capital.
