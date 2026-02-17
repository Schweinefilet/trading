# Paper Trading Validation Plan (8-Week)

## Objectives
Before transitioning to live capital, the bot must undergo a rigorous 8-week paper trading period to verify that the **Production Hardening** (Phase 113) and **Risk Mitigation** (Phase 112) features perform correctly under live market dynamics.

## Phase 1: Stability & Execution (Weeks 1-2)
- **Goal**: Zero system crashes and correct order execution.
- **Success Criteria**:
    1.  Bot runs continuously from 9:00 AM to 4:10 PM without unhandled exceptions.
    2.  `Reconciler` successfully handles at least 2 simulated disconnects.
    3.  `DataValidator` successfully flags and logs at least 5 instances of stale bars or outliers (if they occur).
    4.  All bracket orders (TP/SL) are correctly placed and tracked.

## Phase 2: Risk Gating & PDT (Weeks 3-4)
- **Goal**: Verify that all safety gates effectively prevent over-trading.
- **Success Criteria**:
    1.  `PreTradeValidator` blocks at least 1 trade due to Spread > 0.5% or Heat Limit.
    2.  PDT Tracker correctly locks the bot when day trades hit 0.
    3.  Regime switching (BULLISH -> CAUTIOUS) correctly reduces position sizes by 50%.
    4.  Circuit Breaker halts the bot if a simulated Daily P&L drop of > 3% occurs.

## Phase 3: Performance & Drift (Weeks 5-8)
- **Goal**: Align Paper results with Backtest expectations and monitor slippage.
- **Success Criteria**:
    1.  Paper P&L is within +/- 15% of the backtest-predicted P&L for the same period.
    2.  Average slippage (Entry Price vs. Bid/Ask at time of signal) is < 0.1%.
    3.  Sharp Ratio remains > 1.5 during the paper period.
    4.  Dashboard telemetry shows 100% data stream uptime.

## Final Review
At the end of Week 8, a **Go/No-Go** meeting will be held. Life deployment is ONLY allowed if all Phase 1-3 criteria are met with **"GREEN"** status.

> [!IMPORTANT]
> Any code change during the 8-week period resets the stability timer (Phase 1) to 1 week.
