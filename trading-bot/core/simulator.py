import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config.settings import settings
from config.symbols import symbol_manager
from data.market_data import MarketDataClient
from data.indicators import calculate_all_indicators
from core.strategy import SwingPullbackStrategy, Signal, SignalType
from utils.logger import logger

class PositionTracker:
    """Track all open positions and verify capital constraints for Swing Trading."""
    
    def __init__(self, starting_capital: float):
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.positions = {}  # symbol -> position info
        self.equity_curve = []
        
    def can_afford(self, shares: int, price: float) -> bool:
        return (shares * price) <= self.cash
    
    def open_position(self, symbol: str, shares: int, price: float, timestamp: datetime, stop_loss: float, take_profit: float, reason: str):
        cost = shares * price
        if cost > self.cash:
            return False
            
        self.cash -= cost
        self.positions[symbol] = {
            'qty': shares,
            'entry_price': price,
            'entry_time': timestamp,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'highest_close': price,
            'hold_days': 0,
            'reason': reason,
            'cost': cost
        }
        return True
        
    def close_position(self, symbol: str, exit_price: float, timestamp: datetime, reason: str):
        if symbol not in self.positions:
            return None
            
        pos = self.positions[symbol]
        proceeds = pos['qty'] * exit_price
        pnl = proceeds - pos['cost']
        pnl_pct = (pnl / pos['cost']) * 100 if pos['cost'] > 0 else 0
        
        self.cash += proceeds
        
        trade_record = {
            'symbol': symbol,
            'qty': pos['qty'],
            'entry_time': pos['entry_time'],
            'exit_time': timestamp,
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'hold_days': pos['hold_days']
        }
        
        del self.positions[symbol]
        return trade_record

    def get_equity(self, current_prices: Dict[str, float]) -> float:
        pos_value = 0
        for sym, pos in self.positions.items():
            price = current_prices.get(sym, pos['entry_price'])
            pos_value += pos['qty'] * price
        return self.cash + pos_value

def simulate_swing_period(
    start_date: datetime,
    end_date: datetime,
    symbols: List[str],
    capital: float = 10000.0,
    verbose: bool = True
):
    """
    Simulate swing trading over a period using daily bars.
    """
    data_client = MarketDataClient()
    strategy = SwingPullbackStrategy()
    tracker = PositionTracker(capital)
    all_trades = []
    queued_entries = [] # List of (symbol, signal) to enter at next day's open
    
    # --- Handle Timezones ---
    import pytz
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=pytz.UTC)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=pytz.UTC)

    # 1. Fetch Daily Data
    # Buffer for indicators (Need > 200 days for 200-EMA)
    fetch_start = start_date - timedelta(days=350)
    fetch_symbols = list(set(symbols + ["SPY"]))
    
    if verbose:
        logger.info(f"Fetching daily bars for {len(fetch_symbols)} symbols...")
        
    try:
        raw_data = data_client.get_bars(
            symbols=fetch_symbols,
            start=fetch_start,
            end=end_date,
            timeframe=TimeFrame.Day
        )
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

    # Pre-calculate indicators for all symbols
    processed_data = {}
    for sym, df in raw_data.items():
        if not df.empty:
            processed_data[sym] = calculate_all_indicators(df)

    if "SPY" not in processed_data:
        logger.error("SPY data missing - required for market filter")
        return None

    spy_df = processed_data["SPY"]
    trading_days = spy_df[spy_df.index >= start_date].index.tolist()

    if verbose:
        logger.info(f"Starting simulation over {len(trading_days)} trading days...")

    for i, today in enumerate(trading_days):
        # Current market prices for equity tracking
        current_prices = {s: processed_data[s].loc[today, 'close'] for s in processed_data if today in processed_data[s].index}
        
        # --- A. EXECUTE QUEUED ENTRIES (Next Day's Open) ---
        for sym, signal in queued_entries[:]:
            if i < len(trading_days) and sym in processed_data and today in processed_data[sym].index:
                entry_price = processed_data[sym].loc[today, 'open']
                
                # Rule: Max 1 new entry per day
                if len([t for t in all_trades if t['entry_time'] == today]) >= settings.SWING_MAX_ENTRIES_PER_DAY:
                    continue
                
                # Rule: Max 3 positions concurrent
                if len(tracker.positions) >= settings.SWING_MAX_POSITIONS:
                    continue

                # Position Sizing: Risk 2% of equity
                current_equity = tracker.get_equity(current_prices)
                risk_amt = current_equity * settings.SWING_RISK_PER_TRADE
                stop_dist = abs(entry_price - signal.stop_loss)
                
                if stop_dist > 0:
                    shares = int(risk_amt / stop_dist)
                    # Cap at 25% of equity
                    max_shares = int(current_equity * settings.SWING_MAX_POSITION_PCT / entry_price)
                    shares = min(shares, max_shares)
                    
                    if shares > 0 and tracker.can_afford(shares, entry_price):
                        tracker.open_position(sym, shares, entry_price, today, signal.stop_loss, signal.take_profit, signal.reason)
                        if verbose:
                            logger.info(f"[ENTRY] {today.date()} | {sym} @ ${entry_price:.2f} | Stop: ${signal.stop_loss:.2f} | Target: ${signal.take_profit:.2f}")
            
            queued_entries.remove((sym, signal))

        # --- B. CHECK STOPS/TARGETS ON OPEN POSITIONS ---
        active_syms = list(tracker.positions.keys())
        for sym in active_syms:
            pos = tracker.positions[sym]
            df = processed_data[sym]
            if today not in df.index: continue
            
            day_data = df.loc[today]
            pos['hold_days'] += 1
            
            exit_reason = None
            exit_price = None
            
            # 1. Stop Loss Check (Gap-down aware)
            if day_data['open'] <= pos['stop_loss']:
                exit_reason = "Stop Loss (Gap)"
                exit_price = day_data['open']
            elif day_data['low'] <= pos['stop_loss']:
                exit_reason = "Stop Loss"
                exit_price = pos['stop_loss']
                
            # 2. Take Profit Check (Gap-up aware)
            elif day_data['open'] >= pos['take_profit']:
                exit_reason = "Take Profit (Gap)"
                exit_price = day_data['open']
            elif day_data['high'] >= pos['take_profit']:
                exit_reason = "Take Profit"
                exit_price = pos['take_profit']
                
            # 3. Trailing Stop Check
            # Update highest close for trailing
            if day_data['close'] > pos['highest_close']:
                pos['highest_close'] = day_data['close']
            
            atr = day_data['atr']
            profit_dist = (pos['highest_close'] - pos['entry_price'])
            if profit_dist > (settings.SWING_TRAILING_ACTIVATION * atr):
                trail_stop = pos['highest_close'] - (settings.SWING_TRAILING_ATR_MULT * atr)
                if day_data['low'] <= trail_stop:
                    exit_reason = "Trailing Stop"
                    exit_price = max(trail_stop, day_data['open'])
            
            # 4. Trend Break (Close below 50-EMA)
            ema_trend = day_data[f"ema_{settings.SWING_EMA_TREND}"]
            if day_data['close'] < ema_trend:
                exit_reason = "Trend Break"
                exit_price = day_data['close']
                
            # 5. Time Stop (7 days, flat)
            if pos['hold_days'] >= settings.SWING_TIME_STOP_DAYS:
                pnl_pct = (day_data['close'] - pos['entry_price']) / pos['entry_price']
                if abs(pnl_pct) < settings.SWING_TIME_STOP_MIN_PNL:
                    exit_reason = "Time Stop"
                    exit_price = day_data['close']

            if exit_reason:
                trade = tracker.close_position(sym, exit_price, today, exit_reason)
                all_trades.append(trade)
                if verbose:
                    logger.info(f"[EXIT] {today.date()} | {sym} @ ${exit_price:.2f} | Reason: {exit_reason} | P&L: ${trade['pnl']:.2f} ({trade['pnl_pct']:.1f}%)")

        # --- C. GENERATE NEW SIGNALS (At Today's Close) ---
        # No more new entries if portfolio drawdown > 6%
        current_equity = tracker.get_equity(current_prices)
        if (current_equity / capital) < (1 - settings.SWING_MAX_PORTFOLIO_DD):
            if i % 10 == 0: logger.warning(f"Portfolio Drawdown Exceeded ({settings.SWING_MAX_PORTFOLIO_DD:.1%}). Suspending entries.")
            continue

        spy_price = spy_df.loc[today, 'close']
        spy_ema = spy_df.loc[today, f"ema_{settings.SWING_EMA_TREND}"]
        market_context = {"spy_price": spy_price, "spy_ema": spy_ema}
        
        for sym in symbols:
            if sym in tracker.positions or sym == "SPY": continue
            if sym not in processed_data or today not in processed_data[sym].index: continue
            
            df = processed_data[sym]
            # Strategy needs enough data for indicators
            ts_idx = df.index.get_loc(today)
            if ts_idx < 60: continue
            
            window = df.iloc[max(0, ts_idx-100):ts_idx+1]
            signal = strategy.generate_signals(sym, window, market_context=market_context)
            
            if signal.is_entry:
                queued_entries.append((sym, signal))

        tracker.equity_curve.append((today, current_equity))

    return {
        "starting_capital": capital,
        "ending_capital": tracker.cash + sum(p['qty'] * current_prices.get(s, p['entry_price']) for s, p in tracker.positions.items()),
        "trades": all_trades,
        "equity_curve": tracker.equity_curve
    }
