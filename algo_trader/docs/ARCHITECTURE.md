# AlgoTrader Architecture

## System Overview
AlgoTrader is a real-time, event-driven algorithmic trading system built for the Alpaca platform. It utilizes a multi-layered resolution strategy (1-Hour trend filtering with 15-Minute mean reversion) and robust risk management gating.

## Core Components

### 1. Data Layer
- **DataStream**: WebSocket-based client for real-time Bar and Quote data. Implements auto-reconnect with exponential backoff.
- **HistoricalDataClient**: REST-based client for historical data retrieval, indicator warm-up, and caching.
- **DataValidator**: Real-time safety filter that rejects stale, malformed (NaN), or outlier market data.

### 2. Strategy Layer
- **Signals**: Multi-timeframe signal generation. Checks RSI, ADX, EMAs, and Volume profiles.
- **Regime**: SPY-based market filter. Categorizes market into BULLISH, CAUTIOUS, or VOLATILE regimes to adjust position sizing.
- **Ranker**: Prioritizes signals based on Strength, PDT constraints, and portfolio heat.

### 3. Execution Layer
- **OrderManager**: Handles order submission (Bracket Orders), tracking, and lifecycle. Implements Trailing Stop logic and EOD liquidation.
- **Reconciler**: Periodically synchronizes local state with Alpaca's reality. Fixes state drift after disconnects.

### 4. Risk Layer
- **CircuitBreaker**: Hard stops for Daily, Weekly, and Monthly P&L drawdown.
- **PortfolioHeatManager**: Enforces sector concentration limits and total dollar-at-risk caps.
- **PreTradeValidator**: The "7-Point" final gate that validates every order against all risk rules at the moment of execution.

## Data Flow
1. **Bars** arrive via `DataStream`.
2. `DataValidator` checks bar for sanity.
3. `Signals` updates indicators and checks for entries.
4. `Ranker` selects the best trades within current heat/PDT limits.
5. `calculate_position_size` applies Regime and Drawdown multipliers.
6. `PreTradeValidator` performs the final 7-point check.
7. `OrderManager` submits to Alpaca.

## Error Handling
- **API Resilience**: All REST calls are wrapped in `@with_retry` decorators.
- **Connectivity**: WebSocket auto-reconnects and triggers `Reconciler` upon re-establishment.
- **Safety**: Circuit Breakers are checked every heartbeat and pre-trade.
