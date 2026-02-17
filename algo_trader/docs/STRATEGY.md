# AlgoTrader Strategy Document

## Overview
The strategy is a **Trend-Following Mean Reversion** hybrid. It seeks to capture explosive moves in high-liquid tech stocks while using a regime filter to avoid choppy markets.

## Timeframes
- **Trend (Anchor)**: 1-Hour Chart (EMA 20/50, ADX).
- **Execution**: 15-Minute Chart (RSI, VWAP, ATR).

## Signal Logic

### Entry (Long)
1. **Regime**: Must be `BULLISH` or `CAUTIOUS`.
2. **Trend**: 1H EMA 20 > 1H EMA 50 AND ADX(14) > 20.
3. **Execution**: 15m RSI(14) crosses below 30 (oversold) then turns up, OR 15m Price > VWAP with Volume > 1.5x SMA.

### Exit
- **Take Profit**: 15% (Initial) or dynamic based on 3x ATR.
- **Stop Loss**: 5% (Initial) or dynamic based on 2x ATR.
- **Trailing Stop**: Activated when profit exceeds 2.5x ATR.

## Risk Management

### Position Sizing
- **Dynamic Risk**: Base risk 6.7% of equity, reduced to 4.0% or 2.5% as equity passes $100k/$250k thresholds.
- **Regime Multiplier**: 
    - `BULLISH`: 1.0x size.
    - `CAUTIOUS`: 0.5x size.
    - `VOLATILE`: 0.0x size (No new trades).
- **Drawdown Gating**: If current drawdown > 5%, reduce all new position sizes by 50%.

### Portfolio Limits
- **Max Positions**: 10
- **Sector Cap**: Max 3 positions in the same Sector.
- **Heat Cap**: Total Portfolio Heat (dollar risk) < 15% of equity.

## Universe
The bot trades only the most liquid tickers defined in `config/tickers.py`, primarily focused on high-momentum technology and growth sectors.
