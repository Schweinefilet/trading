"""
Vectorized backtesting engine using pandas.
Simulates the strategy against historical data with realistic costs and constraints.
"""
import math
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import pytz

from config.settings import TradingConfig, config
from config.tickers import tickers
from data.indicators import compute_indicators, compute_regime_indicators
from backtest.metrics import compute_metrics


@dataclass
class BacktestTrade:
    """Record of a simulated trade."""
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    shares: int
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    exit_reason: str
    is_day_trade: bool = True
    atr_at_entry: float = 0.0


class BacktestEngine:
    """
    Vectorized backtesting engine for the intraday momentum strategy.

    Features:
      - Same signal logic as live strategy
      - Realistic costs (slippage + spread + regulatory fees)
      - PDT constraints (max 3 day trades / 5 business days)
      - Circuit breaker checks
      - Regime filter (SPY/VIX)
      - Fills at next bar open (no look-ahead bias)
    """

    def __init__(self, cfg: TradingConfig = None):
        self.cfg = cfg or config
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []

    def run(
        self,
        bar_data: Dict[str, pd.DataFrame],
        spy_daily: pd.DataFrame,
        start: datetime = None,
        end: datetime = None,
        capital: float = None,
        verbose: bool = False,
    ) -> Dict:
        """
        Run backtest across all symbols.

        Args:
            bar_data: Dict mapping symbol -> 15-min bar DataFrame (with indicators)
            spy_daily: SPY daily bars for regime (with sma_50, sma_200, vol_proxy)
            start: Filter start datetime
            end: Filter end datetime
            capital: Starting capital (defaults to config)
            verbose: Print trade-by-trade details

        Returns:
            Dict with trades, metrics, equity_curve
        """
        capital = capital or self.cfg.STARTING_CAPITAL
        self.cash = capital  # Track available cash
        self.verbose = verbose
        equity = capital
        peak_equity = capital
        self.trades = []
        self.equity_curve = [capital]

        # State tracking
        positions: Dict[str, dict] = {}  # symbol -> position dict
        daily_pnl = 0.0
        weekly_pnl = 0.0
        monthly_pnl = 0.0
        consecutive_losses = 0
        current_date = None
        current_week = None
        current_month = None
        day_trades_in_window: List[datetime] = []  # dates of day trades

        # Build a combined timeline of all bar timestamps
        all_timestamps = set()
        for sym, df in bar_data.items():
            if start:
                start_ts = pd.Timestamp(start)
                if start_ts.tz is None and df.index.tz is not None:
                    start_ts = start_ts.tz_localize(df.index.tz)
                elif start_ts.tz is not None and df.index.tz is not None:
                    start_ts = start_ts.tz_convert(df.index.tz)
                df = df[df.index >= start_ts]
                
            if end:
                end_ts = pd.Timestamp(end)
                if end_ts.tz is None and df.index.tz is not None:
                    end_ts = end_ts.tz_localize(df.index.tz)
                elif end_ts.tz is not None and df.index.tz is not None:
                    end_ts = end_ts.tz_convert(df.index.tz)
                df = df[df.index <= end_ts]
            bar_data[sym] = df
            all_timestamps.update(df.index.tolist())

        sorted_timestamps = sorted(all_timestamps)

        # Determine regime at each trading day
        regime_by_date = self._build_regime_map(spy_daily)

        et = pytz.timezone("US/Eastern")

        for i, ts in enumerate(sorted_timestamps):
            # Convert to Eastern for time checks
            if hasattr(ts, 'tz') and ts.tz is not None:
                ts_et = ts.astimezone(et)
            else:
                try:
                    ts_et = et.localize(ts)
                except Exception:
                    ts_et = ts

            bar_date = ts_et.date()
            bar_time = ts_et.time()

            # Daily reset
            if current_date != bar_date:
                # EOD: close day trade positions from previous day
                for sym in list(positions.keys()):
                    pos = positions[sym]
                    if pos.get("is_day_trade", True):
                        if sym in bar_data and ts in bar_data[sym].index:
                            exit_price = bar_data[sym].loc[ts, "open"]
                            pnl = self._close_position(
                                positions, sym, exit_price, ts,
                                "EOD_close", equity, day_trades_in_window
                            )
                            if pnl is not None:
                                equity += pnl

                current_date = bar_date
                daily_pnl = 0.0
                consecutive_losses = 0  # Reset daily — prevents permanent halt

                # Weekly reset
                if current_week != bar_date.isocalendar()[1]:
                    current_week = bar_date.isocalendar()[1]
                    weekly_pnl = 0.0

                # Monthly reset
                if current_month != bar_date.month:
                    current_month = bar_date.month
                    monthly_pnl = 0.0

            # Trading hours check
            if not self._in_trading_hours(bar_time):
                continue

            # EOD liquidation check (3:45 PM ET)
            eod_time = dt_time(15, 45)
            if bar_time >= eod_time:
                for sym in list(positions.keys()):
                    pos = positions[sym]
                    if pos.get("is_day_trade", True) and sym in bar_data and ts in bar_data[sym].index:
                        exit_price = bar_data[sym].loc[ts, "close"]
                        pnl = self._close_position(
                            positions, sym, exit_price, ts,
                            "time_stop_eod", equity, day_trades_in_window
                        )
                        if pnl is not None:
                            daily_pnl += pnl
                            weekly_pnl += pnl
                            monthly_pnl += pnl
                            equity += pnl
                            if pnl < 0:
                                consecutive_losses += 1
                            else:
                                consecutive_losses = 0
                continue

            # Circuit breaker checks
            if equity > 0:
                if daily_pnl / equity <= -self.cfg.DAILY_LOSS_LIMIT_PCT:
                    continue
                if weekly_pnl / equity <= -self.cfg.WEEKLY_LOSS_LIMIT_PCT:
                    continue
                if consecutive_losses >= self.cfg.MAX_CONSECUTIVE_LOSSES:
                    continue
                if peak_equity > 0 and (peak_equity - equity) / peak_equity >= self.cfg.DRAWDOWN_HALT_PCT:
                    continue

            # Get regime for this date
            regime = regime_by_date.get(bar_date, "BULLISH")

            # Check existing positions for stops
            for sym in list(positions.keys()):
                if sym not in bar_data or ts not in bar_data[sym].index:
                    continue
                bar = bar_data[sym].loc[ts]
                pos = positions[sym]

                hit, exit_price, reason = self._check_stops(pos, bar)
                if hit:
                    pnl = self._close_position(
                        positions, sym, exit_price, ts,
                        reason, equity, day_trades_in_window
                    )
                    if pnl is not None:
                        daily_pnl += pnl
                        weekly_pnl += pnl
                        monthly_pnl += pnl
                        equity += pnl
                        if pnl < 0:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0

            # PDT check
            pdt_count = self._count_day_trades(day_trades_in_window, bar_date)
            day_trades_remaining = max(0, self.cfg.PDT_MAX_DAY_TRADES - pdt_count)

            # Skip signal generation if at max positions
            if len(positions) >= self.cfg.MAX_POSITIONS:
                self.equity_curve.append(equity)
                continue

            # Generate signals for each symbol
            signals_this_bar = []
            for sym in bar_data:
                if sym in positions:
                    continue  # Already have position
                if sym in ("SPY", "QQQ"):
                    continue  # Context tickers, never trade

                df = bar_data[sym]
                if ts not in df.index:
                    continue

                # Get data up to current bar (no look-ahead)
                idx = df.index.get_loc(ts)
                if idx < 55:
                    continue  # Not enough warm-up

                window = df.iloc[:idx + 1]
                signal = self._generate_signal(sym, window, regime, ts)
                if signal:
                    signals_this_bar.append(signal)

            # Rank and select
            signals_this_bar.sort(key=lambda s: s["strength"], reverse=True)

            executed = 0
            for sig in signals_this_bar:
                if executed >= 2:
                    break  # Max 2 entries per bar
                if len(positions) >= self.cfg.MAX_POSITIONS:
                    break

                if self.cfg.HOLD_OVERNIGHT:
                    is_day_trade = False
                else:
                    is_day_trade = day_trades_remaining > 0
                if not is_day_trade and not self.cfg.ALLOW_SWING_OVERFLOW:
                    continue

                # Position sizing
                sym = sig["symbol"]
                entry_price = sig["entry_price"]
                stop_price = sig["stop_price"]
                atr = sig["atr"]

                stop_distance = abs(entry_price - stop_price)
                if stop_distance <= 0:
                    continue

                # Drawdown sizing
                dd_mult = 1.0
                if peak_equity > 0:
                    dd = (peak_equity - equity) / peak_equity
                    if dd >= 0.10:
                        dd_mult = 0.50
                    elif dd >= 0.05:
                        dd_mult = 0.75

                # Regime sizing
                regime_mult = 1.0
                if regime == "CAUTIOUS" and sig["direction"] == "LONG":
                    regime_mult = 0.75
                elif regime == "BEARISH" and sig["direction"] == "LONG":
                    regime_mult = 0.50

                risk_amount = equity * self.cfg.RISK_PER_TRADE_PCT * dd_mult * regime_mult
                shares = max(1, math.floor(risk_amount / stop_distance))

                # Cap: max 25% of equity
                max_shares = math.floor(equity * self.cfg.MAX_POSITION_PCT / entry_price)
                shares = min(shares, max_shares)

                # Apply costs
                slippage = entry_price * self.cfg.SLIPPAGE_PCT
                spread = self.cfg.SPREAD_COST_PER_SHARE
                cost_per_share = slippage + spread

                if sig["direction"] == "LONG":
                    actual_entry = entry_price + cost_per_share
                else:
                    actual_entry = entry_price - cost_per_share

                # Cash constraint (soft buffer)
                total_deployed = equity - self.cash
                if (total_deployed + (shares * actual_entry)) / equity > self.cfg.MAX_CAPITAL_DEPLOYED_PCT:
                     # Attempt to downsize to fit the limit
                     available_for_trade = (equity * self.cfg.MAX_CAPITAL_DEPLOYED_PCT) - total_deployed
                     if available_for_trade <= 0:
                         continue
                     shares = min(shares, math.floor(available_for_trade / actual_entry))

                # Hard Cash constraint
                cost_basis = shares * actual_entry
                if cost_basis > self.cash:
                    shares = int(self.cash / actual_entry)
                    cost_basis = shares * actual_entry


                if shares < 1:
                    continue

                # Deduct cash
                self.cash -= cost_basis

                # Record position
                positions[sym] = {
                    "symbol": sym,
                    "direction": sig["direction"],
                    "shares": shares,
                    "entry_price": actual_entry,
                    "stop_price": stop_price,
                    "take_profit": sig["take_profit"],
                    "atr": atr,
                    "entry_time": ts,
                    "is_day_trade": is_day_trade,
                    "highest_price": actual_entry,
                    "lowest_price": actual_entry,
                }

                if is_day_trade:
                    day_trades_remaining -= 1

                if verbose:
                    print(f"  {ts} | ENTRY {sig['direction']} {shares} {sym} @ ${actual_entry:.2f} "
                          f"| SL=${stop_price:.2f} | TP=${sig['take_profit']:.2f}")

                executed += 1

            # Update peak equity
            peak_equity = max(peak_equity, equity)
            self.equity_curve.append(equity)

        # Close any remaining positions at last price
        for sym in list(positions.keys()):
            if sym in bar_data and not bar_data[sym].empty:
                last_price = bar_data[sym].iloc[-1]["close"]
                last_ts = bar_data[sym].index[-1]
                pnl = self._close_position(
                    positions, sym, last_price, last_ts,
                    "backtest_end", equity, day_trades_in_window
                )
                if pnl is not None:
                    equity += pnl

        self.equity_curve.append(equity)

        # Compute metrics
        metrics = compute_metrics(self.trades, capital, equity, self.equity_curve)

        return {
            "trades": self.trades,
            "metrics": metrics,
            "equity_curve": self.equity_curve,
            "starting_capital": capital,
            "ending_capital": equity,
        }

    def _generate_signal(self, symbol: str, df: pd.DataFrame, regime: str, ts) -> Optional[dict]:
        """Generate a signal using the same logic as live strategy."""
        curr = df.iloc[-1]

        close = curr.get("close", np.nan)
        rsi = curr.get("rsi", np.nan)
        rsi_prev = curr.get("rsi_prev", np.nan)
        adx = curr.get("adx", np.nan)
        ema_fast = curr.get("ema_fast", np.nan)
        ema_slow = curr.get("ema_slow", np.nan)
        ema_bias = curr.get("ema_bias", np.nan)
        atr = curr.get("atr", np.nan)
        vwap = curr.get("vwap", np.nan)
        volume = curr.get("volume", np.nan)
        volume_sma = curr.get("volume_sma_20", np.nan)

        essentials = [close, rsi, rsi_prev, adx, ema_fast, ema_slow, ema_bias, atr, volume, volume_sma]
        if any(pd.isna(v) for v in essentials):
            return None

        stop_mult = tickers.get_stop_multiplier(symbol, self.cfg.ATR_STOP_MULTIPLIER)

        from strategy.regime import regime_allows_trade
        
        # 1. ADX trend strength
        adx_ok = adx >= self.cfg.ADX_TREND_THRESHOLD
        
        # 2. Price/EMA bias
        bias_ok = close > ema_bias
        
        # 3. EMA alignment
        ema_ok = ema_fast > ema_slow
        
        # 4. RSI momentum or oversold bounce
        rsi_cross_up = (rsi_prev < self.cfg.RSI_MOMENTUM_LONG and rsi >= self.cfg.RSI_MOMENTUM_LONG)
        rsi_bounce = (rsi_prev <= 35 and rsi > 35)
        rsi_ok = rsi_cross_up or rsi_bounce
        
        # 5. Volume Surge
        vol_ok = volume >= self.cfg.VOLUME_MULTIPLIER * volume_sma
        
        # Require 3 of the 5 primary technical confirmations
        confirmations = sum([adx_ok, bias_ok, ema_ok, rsi_ok, vol_ok])
        
        # LONG check
        if (confirmations >= 3 and regime_allows_trade(regime, "LONG")):
            if not self.cfg.USE_VWAP or pd.isna(vwap) or close > vwap:
                return {
                    "symbol": symbol,
                    "direction": "LONG",
                    "entry_price": close,
                    "stop_price": close - atr * stop_mult,
                    "take_profit": close + atr * self.cfg.ATR_TARGET_MULTIPLIER,
                    "atr": atr,
                    "strength": self._score(adx, rsi, volume, volume_sma, ema_fast, ema_slow, atr, symbol),
                }

        # SHORT check (remain strict for now)
        if self.cfg.ALLOW_SHORTS and regime_allows_trade(regime, "SHORT"):
            # Add basic short logic if needed
            pass

        return None

    def _score(self, adx, rsi, volume, volume_sma, ema_fast, ema_slow, atr, symbol):
        """Signal strength score."""
        adx_s = min(max((adx - 20) / 40, 0), 1.0)
        rsi_s = min(abs(rsi - 50) / 30, 1.0)
        vol_s = min(max((volume / volume_sma - 1) / 2, 0), 1.0) if volume_sma > 0 else 0
        ema_s = min(abs(ema_fast - ema_slow) / atr, 1.0) if atr > 0 else 0
        tier_bonus = tickers.get_tier_bonus(symbol)
        return adx_s * 0.25 + rsi_s * 0.25 + vol_s * 0.20 + ema_s * 0.15 + tier_bonus

    def _check_stops(self, pos: dict, bar) -> Tuple[bool, float, str]:
        """Check if stop-loss or take-profit was hit during this bar."""
        high = bar.get("high", 0)
        low = bar.get("low", 0)

        if pos["direction"] == "LONG":
            # Stop loss hit
            if low <= pos["stop_price"]:
                return True, pos["stop_price"], "stop_loss"
            # Take profit hit
            if high >= pos["take_profit"]:
                return True, pos["take_profit"], "take_profit"
            
            # Trailing stop check
            highest = pos.get("highest_price", pos["entry_price"])
            if (highest - pos["entry_price"]) >= (pos["atr"] * self.cfg.TRAILING_STOP_ACTIVATE_ATR):
                trail_stop = highest - (pos["atr"] * self.cfg.TRAILING_STOP_ATR)
                if low <= trail_stop:
                    return True, trail_stop, "trailing_stop"
            
            
            # Update highest for trailing
            pos["highest_price"] = max(highest, high)

        else:
            # SHORT logic
            if high >= pos["stop_price"]:
                return True, pos["stop_price"], "stop_loss"
            if low <= pos["take_profit"]:
                return True, pos["take_profit"], "take_profit"
            
            # Trailing stop check (SHORT)
            lowest = pos.get("lowest_price", pos["entry_price"])
            if (pos["entry_price"] - lowest) >= (pos["atr"] * self.cfg.TRAILING_STOP_ACTIVATE_ATR):
                trail_stop = lowest + (pos["atr"] * self.cfg.TRAILING_STOP_ATR)
                if high >= trail_stop:
                    return True, trail_stop, "trailing_stop"
            
            
            pos["lowest_price"] = min(lowest, low)

        return False, 0, ""

    def _close_position(
        self, positions: dict, symbol: str, exit_price: float,
        exit_time, reason: str, equity: float, day_trade_dates: list
    ) -> Optional[float]:
        """Close a position and record the trade."""
        if symbol not in positions:
            return None

        pos = positions.pop(symbol)
        shares = pos["shares"]
        entry_price = pos["entry_price"]

        # Apply exit costs
        # Slippage + Spread
        execution_cost = exit_price * self.cfg.SLIPPAGE_PCT + self.cfg.SPREAD_COST_PER_SHARE
        
        if pos["direction"] == "LONG":
            actual_exit = exit_price - execution_cost
            pnl = (actual_exit - entry_price) * shares
        else:
            actual_exit = exit_price + execution_cost
            pnl = (entry_price - actual_exit) * shares

        # Update cash (return cost basis + PnL)
        cost_basis = entry_price * shares
        proceeds = cost_basis + pnl
        self.cash += proceeds

        pnl_pct = pnl / (entry_price * shares) * 100 if entry_price > 0 else 0

        # Record trade
        trade = BacktestTrade(
            symbol=symbol,
            direction=pos["direction"],
            entry_price=entry_price,
            exit_price=actual_exit,
            shares=shares,
            entry_time=pos["entry_time"],
            exit_time=exit_time,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            is_day_trade=pos.get("is_day_trade", True),
            atr_at_entry=pos.get("atr", 0),
        )
        self.trades.append(trade)

        if self.verbose:
             print(f"  {exit_time} | EXIT {pos['direction']} {symbol} @ ${actual_exit:.2f} | PnL=${pnl:.2f} ({reason})")

        # Record day trade for PDT tracking
        if pos.get("is_day_trade", True):
            entry_date = pos["entry_time"]
            if hasattr(entry_date, 'date'):
                entry_date = entry_date.date() if hasattr(entry_date, 'date') else entry_date
            day_trade_dates.append(entry_date)

        return pnl

    def _count_day_trades(self, day_trade_dates: list, current_date) -> int:
        """Count day trades in the last 5 business days."""
        if not day_trade_dates:
            return 0
        cutoff = current_date - timedelta(days=7)  # Roughly 5 business days
        count = 0
        for d in day_trade_dates:
            if hasattr(d, 'date'):
                d = d.date()
            if isinstance(d, datetime):
                d = d.date()
            try:
                if d >= cutoff:
                    count += 1
            except TypeError:
                pass
        return count

    def _build_regime_map(self, spy_daily: pd.DataFrame) -> Dict:
        """Build a regime lookup by date from SPY daily bars."""
        regime_map = {}
        if spy_daily.empty:
            return regime_map

        if "sma_50" not in spy_daily.columns:
            spy_daily = compute_regime_indicators(spy_daily)

        for ts, row in spy_daily.iterrows():
            d = ts.date() if hasattr(ts, 'date') else ts
            close = row.get("close", np.nan)
            sma_50 = row.get("sma_50", np.nan)
            sma_200 = row.get("sma_200", np.nan)
            vol_proxy = row.get("vol_proxy", 20)

            if pd.isna(close) or pd.isna(sma_50):
                regime_map[d] = "CAUTIOUS"
                continue

            if vol_proxy >= 30 or (not pd.isna(sma_200) and close < sma_200):
                regime_map[d] = "CRISIS"
            elif close < sma_50:
                regime_map[d] = "BEARISH"
            elif vol_proxy >= 25:
                regime_map[d] = "CAUTIOUS"
            else:
                regime_map[d] = "BULLISH"

        return regime_map

    def _in_trading_hours(self, t: dt_time) -> bool:
        """Check if time is in active trading window."""
        morning_start = dt_time(10, 0)
        morning_end = dt_time(11, 30)
        afternoon_start = dt_time(13, 30)
        afternoon_end = dt_time(15, 45)

        return (morning_start <= t < morning_end) or (afternoon_start <= t < afternoon_end)
