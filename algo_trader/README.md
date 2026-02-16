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

## The Trading Decision Flow (How the Bot Decides)

The bot follows a 5-step funnel to move from 70+ symbols down to specific risk-managed trades:

### Phase 1: Market Regime Identification
*   **File**: [`strategy/regime.py`](file:///Users/keith/trading/algo_trader/strategy/regime.py)
*   The system analyzes SPY and VIX to determine the market environment:
    *   **BULLISH**: Aggressive risk (6.7%), loose confirmations.
    *   **CAUTIOUS**: Moderate risk (5.2%), tighter confirmations.
    *   **BEARISH**: Defensive risk (2.5%), Trend + Mean Reversion logic.
    *   **CRISIS**: Trading halted (VIX > 40).

### Phase 2: Rolling Leadership Ranking
*   **File**: [`strategy/ranker.py`](file:///Users/keith/trading/algo_trader/strategy/ranker.py) & `_update_leaderboard` in `engine.py`
*   Instead of guessing which stocks will win, the bot uses **Hindsight-Free Dynamic Tiering**:
    *   It calculates a **120-Day ROC** (Relative Strength) for all 70 symbols.
    *   Only the **Top 25%** are granted "Leader" status.
    *   Leaders get the **25x ATR "Dagger Runner"** targets. Laggards get 10x ATR targets.

### Phase 3: Technical Signal Confirmation
*   **File**: [`strategy/signals.py`](file:///Users/keith/trading/algo_trader/strategy/signals.py)
*   Each 15-min bar must pass a "strike system" (3-5 required confirmations):
    *   **Trend Confirmation**: Price > Bias EMA (50) and Fast EMA > Slow EMA.
    *   **Strength Confirmation**: ADX > 30.0 (High Conviction Gate).
    *   **Momentum Confirmation**: RSI crossing 55 or bouncing from 35.
    *   **Liquidity Filter**: Volume > 1.2x average.

### Phase 4: Position Sizing & Safety Caps
*   **File**: [`risk/position_sizer.py`](file:///Users/keith/trading/algo_trader/risk/position_sizer.py)
*   The bot calculates exact shares based on the **6.7% Bull Risk** model:
    *   **ATR-Based Stops**: If volatility increases, position size automatically decreases.
    *   **Portfolio Heat**: Total account-wide risk is capped at 90% of equity.
    *   **Sector Guard**: No more than 2 open positions in the same sector.

### Phase 5: Trade Management & Exit
*   **File**: [`backtest/engine.py`](file:///Users/keith/trading/algo_trader/backtest/engine.py)
*   Once a trade is live, it is managed dynamically:
    *   **Partial Exit**: 10% of shares are sold at **3.5x ATR** profit to "fireproof" the trade.
    *   **Trailing Stop**: Activates at 2.0x ATR to lock in gains on runners.
    *   **Time Stop**: Exit if the trade remains stagnant for >48 hours.

---
**Warning**: Trading stocks involves significant risk of loss. Always test thoroughly in paper mode before deploying real capital.

