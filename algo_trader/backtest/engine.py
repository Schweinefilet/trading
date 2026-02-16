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
        self.leaderboard_cache: Dict = {} # date -> {symbol: rank_percentile}


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
        self.spy_daily = spy_daily
        self.bar_data = bar_data # Store for internal method access (Phase 103)
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

        # Build QQQ trend map for risk-sizing filter (Phase 6A)
        qqq_trend_bullish = {}
        if "QQQ" in bar_data and not bar_data["QQQ"].empty:
            qqq_df = bar_data["QQQ"]
            qqq_ema_fast = qqq_df["close"].ewm(span=self.cfg.QQQ_EMA_FAST, adjust=False).mean()
            qqq_ema_slow = qqq_df["close"].ewm(span=self.cfg.QQQ_EMA_SLOW, adjust=False).mean()
            for ts_q, row_q in qqq_df.iterrows():
                idx_q = qqq_df.index.get_loc(ts_q)
                if idx_q >= self.cfg.QQQ_EMA_SLOW:
                    qqq_trend_bullish[ts_q] = qqq_ema_fast.iloc[idx_q] > qqq_ema_slow.iloc[idx_q]

        et = pytz.timezone("US/Eastern")

        for i, ts in enumerate(sorted_timestamps):
            # Phase 103: Dynamic Leaderboard Cache Reset (daily)
            # (Handled inside _generate_signal now, but initialized here for safety)
            ts_date = ts.date() if hasattr(ts, 'date') else None

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

                # Partial exit check (Phase 7)
                partial_pnl = self._check_partial_exit(pos, bar, ts, equity, day_trades_in_window)
                if partial_pnl is not None:
                    daily_pnl += partial_pnl
                    weekly_pnl += partial_pnl
                    monthly_pnl += partial_pnl
                    equity += partial_pnl
                    if partial_pnl > 0:
                        consecutive_losses = 0
                    # If position was fully closed during partial (shouldn't happen), clean up
                    if sym not in positions:
                        continue

                hit, exit_price, reason = self._check_stops(pos, bar, current_ts=ts)
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

            # Entry time sub-filter (Phase 7) — only generate signals in tighter windows
            # Check time window & day filter
            if not self._in_entry_window(ts_et):
                # Temporary: disabling Tuesday skip to test if it's the bottleneck
                # if ts_et.weekday() in self.cfg.ENTRY_SKIPPED_DAYS:
                #     continue 
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

                # Correlation Guard (Phase 3)
                sector = tickers.get_sector(sig["symbol"])
                sector_count = sum(1 for s, p in positions.items() if tickers.get_sector(s) == sector)
                if sector_count >= self.cfg.MAX_SECTOR_POSITIONS:
                    continue

                # Position sizing
                sym = sig["symbol"]
                entry_price = sig["entry_price"]
                stop_price = sig["stop_price"]
                atr = sig["atr"]

                if self.cfg.USE_RISK_BASED_SIZING:
                    # Risk-based sizing (1% risk per trade)
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

                    # QQQ directional risk sizing
                    qqq_mult = 1.0
                    if ts in qqq_trend_bullish and not qqq_trend_bullish[ts]:
                        qqq_mult = self.cfg.QQQ_RISK_REDUCTION_BEARISH

                    # Regime-based risk scaling (Phase 100)
                    risk_pct = self.cfg.RISK_PER_TRADE_PCT
                    if regime == "BULLISH":
                        risk_pct = self.cfg.RISK_PER_TRADE_BULLISH
                    elif regime == "CAUTIOUS":
                        risk_pct = self.cfg.RISK_PER_TRADE_CAUTIOUS
                    elif regime == "BEARISH":
                        risk_pct = self.cfg.RISK_PER_TRADE_BEARISH

                    # Complex risk multipliers (Phase 6 toggle)
                    if getattr(self.cfg, "USE_COMPLEX_RISK_MULTIPLIERS", True):
                        risk_amount = equity * risk_pct * dd_mult * regime_mult * qqq_mult
                    else:
                        risk_amount = equity * risk_pct

                    shares = max(1, math.floor(risk_amount / stop_distance))
                else:
                    # Fixed-dollar sizing
                    shares = max(1, math.floor(self.cfg.FIXED_POSITION_DOLLAR / entry_price))

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
                    "strategy": sig.get("strategy", "trend_following"), # Track strategy for special exit logic
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
        
        # RSI acceleration (Phase 7)
        rsi_roc = curr.get("rsi_roc", 0)
        if pd.isna(rsi_roc):
            rsi_roc = 0
        
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
        
        # 6. RSI Acceleration (Phase 3 - Moved from Score to Confirmation)
        rsi_accel_ok = rsi_roc > 0
        
        # 7. ATR Expansion (Phase 3 - Volatility confirmation)
        atr_prev = curr.get("atr_prev", np.nan)
        atr_expanding = (not pd.isna(atr_prev) and atr > atr_prev) or pd.isna(atr_prev)
        
        # Adaptive confirmation threshold (Phase 3: BULLISH 4, CAUTIOUS 5, BEARISH 6)
        if regime == "BULLISH":
            min_confirms = self.cfg.CONFIRMATIONS_BULLISH
        elif regime == "CAUTIOUS":
            min_confirms = self.cfg.CONFIRMATIONS_CAUTIOUS
        else:
            min_confirms = self.cfg.CONFIRMATIONS_BEARISH
        
        # Phase 7: Revert to Core 5 Confirmations (RSI Accel/ATR Exp moved to Score)
        confirmations = sum([adx_ok, bias_ok, ema_ok, rsi_ok, vol_ok])
        
        # Relative Strength (Stock ROC vs SPY ROC)
        spy_roc = 0
        stock_roc = curr.get("roc_20", 0)
        
        # Look up SPY ROC for the current date
        spy_ts = pd.Timestamp(ts.date())
        if self.spy_daily is not None and not self.spy_daily.empty and spy_ts in self.spy_daily.index:
            spy_roc = self.spy_daily.loc[spy_ts].get("roc_20", 0)
            
        rel_strength = stock_roc - spy_roc
        
        # LONG check
        if (confirmations >= min_confirms and regime_allows_trade(regime, "LONG")):
            if not self.cfg.USE_VWAP or pd.isna(vwap) or close > vwap:
                # ADX Rising check (Phase 100)
                adx_prev = curr.get("adx_prev", np.nan)
                adx_rising = (not pd.isna(adx_prev) and adx > adx_prev) or pd.isna(adx_prev)
                
                if adx_rising:
                    # Dynamic Take-Profit Ranking (Phase 103)
                    target_mult = 10.0 # Default
                    
                    # Get current date's rankings
                    ts_date = ts.date()
                    if ts_date not in self.leaderboard_cache:
                        self._update_leaderboard(ts_date, self.bar_data)
                    
                    rank = self.leaderboard_cache[ts_date].get(symbol, 0.5) # Default to 50th percentile
                    
                    if rank <= 0.25: # Top 25% (Leaders)
                        target_mult = 25.0
                    elif rank <= 0.50: # Top 25-50%
                        target_mult = 15.0
                    else: # Bottom 50%
                        target_mult = 10.0

                    return {
                        "symbol": symbol,
                        "direction": "LONG",
                        "entry_price": close,
                        "stop_price": close - atr * stop_mult,
                        "take_profit": close + atr * target_mult,
                        "atr": atr,
                        "strength": self._score(adx, rsi, volume, volume_sma, ema_fast, ema_slow, atr, symbol, rsi_roc, rel_strength, atr_expanding, ts),
                        "strategy": "trend_following",
                        "regime_at_entry": regime # Track for partial exit scaling
                    }

        # 2. Mean Reversion Logic (Phase 2)
        if regime in ["CAUTIOUS", "BEARISH"]:
            rsi2 = curr.get("rsi_2", np.nan)
            bb_lower = curr.get("bb_lower", np.nan)
            if not pd.isna(rsi2) and not pd.isna(bb_lower):
                # Trigger: RSI(2) < 10 (tighter) AND Price < Lower BB
                if rsi2 < 10 and close < bb_lower:
                    return {
                        "symbol": symbol,
                        "direction": "LONG",
                        "entry_price": close,
                        "stop_price": close - atr * 1.0,
                        "take_profit": close + (atr * 4.0), # Phase 4: Higher target for MR
                        "atr": atr,
                        "strength": 0.85,
                        "strategy": "mean_reversion"
                    }
        
        return None

    def _update_leaderboard(self, date, bar_data_in: Dict[str, pd.DataFrame]):
        """Calculate Relative Strength rankings for all symbols on a specific date."""
        scores = {}
        for sym, df in bar_data_in.items():
            if sym in ["SPY", "QQQ"]: continue
            
            # Find the most recent bar on or before this date
            day_data = df[df.index.date <= date]
            if not day_data.empty:
                val = day_data.iloc[-1].get("roc_125", np.nan)
                if not pd.isna(val):
                    scores[sym] = val
        
        if not scores:
            self.leaderboard_cache[date] = {}
            return
            
        # Rank symbols (higher ROC = better rank 1)
        sorted_syms = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        num_syms = len(sorted_syms)
        
        self.leaderboard_cache[date] = {
            sym: (i + 1) / num_syms for i, sym in enumerate(sorted_syms)
        }


    def _score(self, adx, rsi, volume, volume_sma, ema_fast, ema_slow, atr, symbol, rsi_roc=0, rel_strength=0.0, atr_expanding=False, ts=None):
        """Signal strength score with Phase 103 Dynamic RS bonuses."""
        adx_s = min(max((adx - 20) / 40, 0), 1.0)
        rsi_s = min(abs(rsi - 50) / 30, 1.0)
        vol_s = min(max((volume / volume_sma - 1) / 2, 0), 1.0) if volume_sma > 0 else 0
        ema_s = min(abs(ema_fast - ema_slow) / atr, 1.0) if atr > 0 else 0
        
        # Phase 103: Dynamic RS Bonus (Hindsight-Free)
        rs_score_bonus = 0.0
        if ts is not None:
            ts_date = ts.date()
            if ts_date in self.leaderboard_cache:
                rank = self.leaderboard_cache[ts_date].get(symbol, 0.5)
                if rank <= 0.25: # Dynamic Tier 1
                    rs_score_bonus = 0.15
                elif rank <= 0.50: # Dynamic Tier 2
                    rs_score_bonus = 0.10
        
        # RSI acceleration bonus
        rsi_accel_bonus = 0.05 if (rsi_roc and rsi_roc > 0) else 0
        # ATR expansion bonus (Phase 7)
        atr_bonus = 0.05 if atr_expanding else 0
        # Relative Strength (Intraday) bonus
        rs_intraday_bonus = 0.05 if rel_strength > 0 else 0
        
        return adx_s * 0.25 + rsi_s * 0.25 + vol_s * 0.20 + ema_s * 0.10 + rs_score_bonus + rsi_accel_bonus + rs_intraday_bonus + atr_bonus


    def _check_stops(self, pos: dict, bar, current_ts=None) -> Tuple[bool, float, str]:
        """Check if stop-loss or take-profit was hit during this bar."""
        high = bar.get("high", 0)
        low = bar.get("low", 0)

        if pos["direction"] == "LONG":
            # Stop loss hit
            if low <= pos["stop_price"]:
                reason = "break_even" if pos.get("break_even_set") else "stop_loss"
                return True, pos["stop_price"], reason
            # Take profit hit
            if high >= pos["take_profit"]:
                return True, pos["take_profit"], "take_profit"
            
            # Trailing stop check
            highest = pos.get("highest_price", pos["entry_price"])
            if (highest - pos["entry_price"]) >= (pos["atr"] * self.cfg.TRAILING_STOP_ACTIVATE_ATR):
                trail_stop = highest - (pos["atr"] * self.cfg.TRAILING_STOP_ATR)
                if low <= trail_stop:
                    return True, trail_stop, "trailing_stop"
            
            # Break-even guardrail (Phase 6A)
            # After hold time threshold, if price reached activation level, move stop to break-even + offset
            if current_ts is not None and not pos.get("break_even_set", False):
                hold_delta = current_ts - pos["entry_time"]
                hold_mins = hold_delta.total_seconds() / 60
                profit_atr = (highest - pos["entry_price"]) / pos["atr"] if pos["atr"] > 0 else 0
                if hold_mins >= self.cfg.BREAK_EVEN_MIN_HOLD_MINS and profit_atr >= self.cfg.BREAK_EVEN_ACTIVATE_ATR:
                    be_stop = pos["entry_price"] + (pos["atr"] * self.cfg.BREAK_EVEN_OFFSET_ATR)
                    if be_stop > pos["stop_price"]:
                        pos["stop_price"] = be_stop
                        pos["break_even_set"] = True
                        # Re-check if current bar triggers the new BE stop
                        if low <= be_stop:
                            return True, be_stop, "break_even"

            # Stale Trade Temporal Exit (Phase 3)
            if current_ts is not None:
                hold_hours = (current_ts - pos["entry_time"]).total_seconds() / 3600
                max_hold = 48 if pos.get("strategy") == "mean_reversion" else self.cfg.MAX_HOLD_HOURS
                if hold_hours >= max_hold:
                    return True, bar.get("close", pos["entry_price"]), "time_stop_stale"
            
            # Update highest for trailing
            pos["highest_price"] = max(highest, high)

            # --- Phase 2: Mean Reversion Special Exits ---
            if pos.get("strategy") == "mean_reversion":
                rsi2 = bar.get("rsi_2", np.nan)
                sma5 = bar.get("sma_5", np.nan)
                close = bar.get("close", 0)
                
                # Exit on RSI(2) overbought or price touching SMA(5)
                if (not pd.isna(rsi2) and rsi2 > 70) or (not pd.isna(sma5) and close > sma5):
                    return True, close, "mean_reversion_exit"

        else:
            # SHORT logic
            if high >= pos["stop_price"]:
                reason = "break_even" if pos.get("break_even_set") else "stop_loss"
                return True, pos["stop_price"], reason
            if low <= pos["take_profit"]:
                return True, pos["take_profit"], "take_profit"
            
            # Trailing stop check (SHORT)
            lowest = pos.get("lowest_price", pos["entry_price"])
            if (pos["entry_price"] - lowest) >= (pos["atr"] * self.cfg.TRAILING_STOP_ACTIVATE_ATR):
                trail_stop = lowest + (pos["atr"] * self.cfg.TRAILING_STOP_ATR)
                if high >= trail_stop:
                    return True, trail_stop, "trailing_stop"
            
            # Break-even guardrail (SHORT, Phase 6A)
            if current_ts is not None and not pos.get("break_even_set", False):
                hold_delta = current_ts - pos["entry_time"]
                hold_mins = hold_delta.total_seconds() / 60
                profit_atr = (pos["entry_price"] - lowest) / pos["atr"] if pos["atr"] > 0 else 0
                if hold_mins >= self.cfg.BREAK_EVEN_MIN_HOLD_MINS and profit_atr >= self.cfg.BREAK_EVEN_ACTIVATE_ATR:
                    be_stop = pos["entry_price"] - (pos["atr"] * self.cfg.BREAK_EVEN_OFFSET_ATR)
                    if be_stop < pos["stop_price"]:
                        pos["stop_price"] = be_stop
                        pos["break_even_set"] = True
                        if high >= be_stop:
                            return True, be_stop, "break_even"
            
            pos["lowest_price"] = min(lowest, low)

        return False, 0, ""

    def _close_position(
        self, positions: dict, symbol: str, exit_price: float,
        exit_time, reason: str, equity: float, day_trade_dates: list
    ) -> Optional[float]:
        """Close a position and record the trade (includes accumulated partial P&L)."""
        if symbol not in positions:
            return None

        pos = positions.pop(symbol)
        shares = pos["shares"]  # Remaining shares after any partial exit
        entry_price = pos["entry_price"]

        # Apply exit costs
        # Slippage + Spread
        execution_cost_per_share = exit_price * self.cfg.SLIPPAGE_PCT + self.cfg.SPREAD_COST_PER_SHARE
        
        if pos["direction"] == "LONG":
            actual_exit = exit_price - execution_cost_per_share
            gross_pnl = (actual_exit - entry_price) * shares
        else:
            actual_exit = exit_price + execution_cost_per_share
            gross_pnl = (entry_price - actual_exit) * shares

        # Regulatory Fees (Sell-side only)
        exit_proceeds = shares * actual_exit
        sec_fee = math.ceil(exit_proceeds * self.cfg.SEC_FEE_RATE * 100) / 100
        finra_taf = min(math.ceil(shares * self.cfg.FINRA_TAF_RATE * 100) / 100, self.cfg.FINRA_TAF_CAP)
        total_fees = sec_fee + finra_taf
        
        remaining_pnl = gross_pnl - total_fees
        
        # Add accumulated partial P&L (Phase 7)
        partial_pnl = pos.get("partial_pnl_realized", 0)
        partial_shares_exited = pos.get("partial_shares_exited", 0)
        total_pnl = remaining_pnl + partial_pnl
        
        # Total shares = remaining + any partial exits already done
        total_shares = shares + partial_shares_exited

        # Update cash (return cost basis + PnL for remaining shares)
        cost_basis = entry_price * shares
        proceeds = cost_basis + remaining_pnl
        self.cash += proceeds

        pnl_pct = total_pnl / (entry_price * total_shares) * 100 if entry_price > 0 else 0
        
        # Determine exit reason suffix if partial was taken
        final_reason = f"{reason}+partial" if partial_pnl != 0 else reason

        # Record trade with combined P&L
        trade = BacktestTrade(
            symbol=symbol,
            direction=pos["direction"],
            entry_price=entry_price,
            exit_price=actual_exit,
            shares=total_shares,
            entry_time=pos["entry_time"],
            exit_time=exit_time,
            pnl=total_pnl,
            pnl_pct=pnl_pct,
            exit_reason=final_reason,
            is_day_trade=pos.get("is_day_trade", True),
            atr_at_entry=pos.get("atr", 0),
        )
        self.trades.append(trade)

        if self.verbose:
             print(f"  {exit_time} | EXIT {pos['direction']} {shares} {symbol} @ ${actual_exit:.2f} | "
                   f"Remaining=${remaining_pnl:.2f} | Partial=${partial_pnl:.2f} | Total PnL=${total_pnl:.2f} ({final_reason})")

        # Record day trade for PDT tracking
        if pos.get("is_day_trade", True):
            entry_date = pos["entry_time"]
            if hasattr(entry_date, 'date'):
                entry_date = entry_date.date() if hasattr(entry_date, 'date') else entry_date
            day_trade_dates.append(entry_date)

        return total_pnl

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
        """Check if time is in active trading window (for stop checks etc.)."""
        morning_start = dt_time(10, 0)
        morning_end = dt_time(11, 30)
        afternoon_start = dt_time(13, 30)
        afternoon_end = dt_time(15, 45)

        return (morning_start <= t < morning_end) or (afternoon_start <= t < afternoon_end)

    def _in_entry_window(self, dt: datetime) -> bool:
        """
        Check if datetime is within allowable trading window and day.
        Excludes open/close noise and specific skipped days (e.g. Tuesdays).
        """
        # 1. Day Filter (Phase 2)
        if dt.weekday() in self.cfg.ENTRY_SKIPPED_DAYS:
            return False

        t = dt.time()
        bo = self.cfg.ENTRY_BLACKOUT_OPEN_MINS
        bc = self.cfg.ENTRY_BLACKOUT_CLOSE_MINS
        
        # Morning window: 10:00+bo to 11:30-bc
        m_start = dt_time(10, bo)
        m_end_h, m_end_m = divmod(11 * 60 + 30 - bc, 60)
        m_end = dt_time(m_start.hour if m_end_h > 12 else m_end_h, m_end_m)
        
        # Afternoon window: 13:30+bo to 15:45-bc
        a_start_total = 13 * 60 + 30 + bo
        a_start = dt_time(*divmod(a_start_total, 60))
        a_end_total = 15 * 60 + 45 - bc
        a_end = dt_time(*divmod(a_end_total, 60))
        
        return (m_start <= t < m_end) or (a_start <= t < a_end)

    def _check_partial_exit(self, pos: dict, bar, current_ts, equity: float, day_trade_dates: list):
        """Check if partial exit should trigger at configured ATR profit (Phase 7).
        
        Instead of recording a separate trade, accumulates P&L within the position.
        The combined P&L is included when the remaining position finally closes.
        Returns realized P&L of the partial close, or None if no partial taken.
        """
        if not self.cfg.USE_PARTIAL_EXITS:
            return None
        if pos.get("partial_taken", False):
            return None
        
        high = bar.get("high", 0)
        low = bar.get("low", 0)
        atr = pos.get("atr", 0)
        if atr <= 0:
            return None
        
        if pos["direction"] == "LONG":
            profit_atr = (high - pos["entry_price"]) / atr
        else:
            profit_atr = (pos["entry_price"] - low) / atr
        
        if profit_atr < self.cfg.PARTIAL_EXIT_ATR_TRIGGER:
            return None
        
        # Trigger partial exit
        # Phase 100: Regime-based partial exit size
        regime = pos.get("regime_at_entry", "CAUTIOUS")
        part_pct = self.cfg.PARTIAL_EXIT_PCT_BULLISH if regime == "BULLISH" else self.cfg.PARTIAL_EXIT_PCT_DEFAULT
        
        total_shares = pos["shares"]
        partial_shares = math.floor(total_shares * part_pct)
        remaining_shares = total_shares - partial_shares
        
        if remaining_shares < 1:
            return None  # Don't partial if it would close the whole position
        
        # Partial exit price = entry + trigger ATR
        if pos["direction"] == "LONG":
            partial_exit_price = pos["entry_price"] + atr * self.cfg.PARTIAL_EXIT_ATR_TRIGGER
        else:
            partial_exit_price = pos["entry_price"] - atr * self.cfg.PARTIAL_EXIT_ATR_TRIGGER
        
        # Apply exit costs on partial
        execution_cost_per_share = partial_exit_price * self.cfg.SLIPPAGE_PCT + self.cfg.SPREAD_COST_PER_SHARE
        if pos["direction"] == "LONG":
            actual_partial_exit = partial_exit_price - execution_cost_per_share
            gross_pnl = (actual_partial_exit - pos["entry_price"]) * partial_shares
        else:
            actual_partial_exit = partial_exit_price + execution_cost_per_share
            gross_pnl = (pos["entry_price"] - actual_partial_exit) * partial_shares
        
        # Regulatory fees on partial
        exit_proceeds = partial_shares * actual_partial_exit
        sec_fee = math.ceil(exit_proceeds * self.cfg.SEC_FEE_RATE * 100) / 100
        finra_taf = min(math.ceil(partial_shares * self.cfg.FINRA_TAF_RATE * 100) / 100, self.cfg.FINRA_TAF_CAP)
        total_fees = sec_fee + finra_taf
        pnl = gross_pnl - total_fees
        
        # Accumulate partial P&L in position (don't create a separate trade)
        pos["partial_pnl_realized"] = pos.get("partial_pnl_realized", 0) + pnl
        pos["partial_shares_exited"] = pos.get("partial_shares_exited", 0) + partial_shares
        # Update shares and proceeds
        pos["shares"] = remaining_shares
        # Mark as partial taken
        pos["partial_exit_taken"] = True
        
        # Phase 4: Restore dynamic TP from config for the remainder
        target_mult = self.cfg.ATR_TARGET_MULTIPLIER
        pos["take_profit"] = pos["entry_price"] + atr * target_mult if pos["direction"] == "LONG" \
            else pos["entry_price"] - atr * target_mult
        
        # Move stop to break-even + lock profit offset after partial
        be_stop = pos["entry_price"] + (atr * self.cfg.BREAK_EVEN_OFFSET_ATR) if pos["direction"] == "LONG" \
            else pos["entry_price"] - (atr * self.cfg.BREAK_EVEN_OFFSET_ATR)
        if pos["direction"] == "LONG" and be_stop > pos["stop_price"]:
            pos["stop_price"] = be_stop
            pos["break_even_set"] = True
        elif pos["direction"] == "SHORT" and be_stop < pos["stop_price"]:
            pos["stop_price"] = be_stop
            pos["break_even_set"] = True
        
        # Return cash from partial close
        partial_cost_basis = pos["entry_price"] * partial_shares
        self.cash += partial_cost_basis + pnl
        
        if self.verbose:
            print(f"  {current_ts} | PARTIAL EXIT {pos['direction']} {partial_shares}/{total_shares} {pos['symbol']} "
                  f"@ ${actual_partial_exit:.2f} | PnL=${pnl:.2f} | Remaining={remaining_shares}")
        
        return pnl

