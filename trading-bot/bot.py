#!/usr/bin/env python3
"""
24/5 Automated Day Trading Bot
Main orchestrator with lifecycle management and scheduling

This bot runs as a long-lived process managing:
- Pre-market screening (8:00-9:25 AM ET)  
- Active trading (9:45 AM - 3:55 PM ET)
- End-of-day liquidation (3:55 PM ET)
- Daily summaries and reporting
"""

import asyncio
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import threading
import time

from alpaca.data.timeframe import TimeFrame

from config.settings import settings
from config.symbols import symbol_manager
from data.market_data import market_data
from data.screener import screener
from data.indicators import (
    calculate_all_indicators,
    get_trailing_stop_price,
)
from core.risk_manager import risk_manager
from core.order_manager import order_manager
from core.portfolio import portfolio
from core.strategy import strategy, Signal, SignalType
from utils.logger import logger, trade_logger
from utils.notifier import notifier
from utils.helpers import (
    now_et, today_et,
    is_market_open, is_premarket, is_trading_day,
    get_market_open_time, get_market_close_time,
    get_next_trading_day,
    is_opening_range, is_eod_liquidation_period,
    minutes_until_market_open, minutes_until_market_close,
)


class TradingBot:
    """
    Main trading bot orchestrator.
    
    Lifecycle:
    - 4:00 AM ET: Wake up, system health check
    - 8:00 AM ET: Pre-market screener runs
    - 9:30 AM ET: Market open, observe opening range
    - 9:45 AM ET: Active trading begins
    - 3:55 PM ET: Close all positions (EOD liquidation)
    - 4:00 PM ET: Market close, generate daily summary
    - 4:10 PM ET: Sleep until next trading day
    """
    
    def __init__(self):
        """Initialize the trading bot."""
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._watchlist: List[str] = []
        self._bars_cache: Dict[str, any] = {}  # symbol -> latest bars DataFrame
        self._last_signals: Dict[str, Signal] = {}
        
        # Setup signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_event.set()
    
    # ========================================================================
    # Startup and Shutdown
    # ========================================================================
    
    async def start(self) -> None:
        """Start the trading bot."""
        logger.info("=" * 60)
        logger.info("Starting 24/5 Trading Bot")
        logger.info(f"Mode: {'PAPER' if settings.is_paper else 'LIVE'}")
        logger.info(f"Account Type: {settings.ACCOUNT_TYPE}")
        logger.info("=" * 60)
        
        try:
            # Initialize components
            await self._initialize()
            
            # Send startup notification
            await notifier.send_system_event(
                "Bot Started",
                f"Trading mode: {'Paper' if settings.is_paper else 'LIVE'}\n"
                f"Account type: {settings.ACCOUNT_TYPE}"
            )
            
            self._running = True
            
            # Main loop
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            await notifier.send_risk_event("Bot Error", str(e), "critical")
            
        finally:
            await self._shutdown()
    
    async def _initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing components...")
        
        # Connect to Alpaca
        if not order_manager.connect():
            raise RuntimeError("Failed to connect to Alpaca API")
        
        # Get account info
        account = order_manager.get_account()
        if not account:
            raise RuntimeError("Failed to get account info")
        
        equity = account["equity"]
        logger.info(f"Account equity: ${equity:,.2f}")
        
        # Initialize risk manager with current equity
        risk_manager.update_equity(equity)
        risk_manager.set_day_start_equity(equity)
        
        # Sync portfolio
        portfolio.sync_with_broker(equity, account["cash"])
        
        # Check for existing positions
        positions = order_manager.get_positions()
        if positions:
            logger.info(f"Found {len(positions)} existing positions:")
            for symbol, pos in positions.items():
                logger.info(f"  {symbol}: {pos['qty']} shares @ ${pos['entry_price']:.2f}")
        
        # Cancel any stale orders
        open_orders = order_manager.get_open_orders()
        if open_orders:
            logger.info(f"Cancelling {len(open_orders)} stale orders...")
            order_manager.cancel_all_orders()
        
        # Load static watchlist
        self._watchlist = list(symbol_manager.STATIC_WATCHLIST)
        logger.info(f"Loaded {len(self._watchlist)} symbols in static watchlist")
        
        # Start order sync loop
        order_manager.start_sync_loop()
        
        logger.info("Initialization complete")
    
    async def _shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down...")
        
        self._running = False
        
        try:
            # Stop streaming
            await market_data.stop_streaming()
            
            # Stop order sync
            order_manager.stop_sync_loop()
            
            # Optionally close positions (configurable)
            # For safety, we just cancel orders but leave positions
            order_manager.cancel_all_orders()
            
            # Close notifier
            await notifier.close()
            
            # Send shutdown notification
            await notifier.send_system_event("Bot Stopped", "Graceful shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("Shutdown complete")
    
    # ========================================================================
    # Main Loop
    # ========================================================================
    
    async def _main_loop(self) -> None:
        """Main trading loop."""
        while self._running and not self._shutdown_event.is_set():
            try:
                now = now_et()
                
                # Check what phase we're in
                if not is_trading_day():
                    # Not a trading day, sleep until next one
                    next_day = get_next_trading_day()
                    logger.info(f"Not a trading day. Next trading day: {next_day}")
                    await self._sleep_until_next_day()
                    continue
                
                # Pre-market phase (8:00 - 9:30 AM)
                if is_premarket():
                    await self._premarket_phase()
                
                # Opening range phase (9:30 - 9:45 AM)
                elif is_market_open() and is_opening_range(settings.OPENING_RANGE_MINUTES):
                    await self._opening_range_phase()
                
                # Active trading phase (9:45 AM - 3:55 PM)
                elif is_market_open() and not is_eod_liquidation_period(settings.EOD_LIQUIDATION_MINUTES):
                    await self._active_trading_phase()
                
                # EOD liquidation (3:55 - 4:00 PM)
                elif is_market_open() and is_eod_liquidation_period(settings.EOD_LIQUIDATION_MINUTES):
                    await self._eod_liquidation_phase()
                
                # After hours
                else:
                    await self._after_hours_phase()
                
                # Check for shutdown
                if self._shutdown_event.is_set():
                    break
                
                # Small delay to prevent tight loop
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(10)  # Wait before retrying
    
    # ========================================================================
    # Trading Phases
    # ========================================================================
    
    async def _premarket_phase(self) -> None:
        """Pre-market phase: run screener and prepare watchlist."""
        now = now_et()
        
        # Only run screener between 8:00 and 9:25 AM
        if now.hour == 8 and now.minute == 0:
            logger.info("Pre-market phase started")
            
            if settings.SCREENER_ENABLED:
                logger.info("Running pre-market screener...")
                try:
                    results = await screener.scan()
                    
                    # Update watchlist with screener results
                    screener_symbols = [r["symbol"] for r in results]
                    symbol_manager.set_dynamic_watchlist(screener_symbols)
                    self._watchlist = symbol_manager.watchlist
                    
                    logger.info(f"Watchlist updated: {len(self._watchlist)} symbols")
                    
                    # Log top movers
                    for r in results[:5]:
                        logger.info(
                            f"  {r['symbol']}: gap={r['gap_pct']:+.1f}%, "
                            f"rvol={r['relative_volume']:.1f}x"
                        )
                        
                except Exception as e:
                    logger.error(f"Screener error: {e}")
        
        # Wait until market opens
        mins = minutes_until_market_open()
        if mins and mins > 5:
            # Sleep in chunks to remain responsive
            await asyncio.sleep(min(60, mins * 60 / 2))
        else:
            await asyncio.sleep(10)
    
    async def _opening_range_phase(self) -> None:
        """Opening range phase: observe market, don't trade."""
        logger.info("Opening range phase - observing market")
        
        # Subscribe to data for watchlist
        if self._watchlist:
            await market_data.subscribe(self._watchlist, bars=True, quotes=False)
        
        # Just observe, don't trade
        remaining = settings.OPENING_RANGE_MINUTES * 60
        while remaining > 0 and is_opening_range(settings.OPENING_RANGE_MINUTES):
            await asyncio.sleep(10)
            remaining -= 10
            
            if self._shutdown_event.is_set():
                return
        
        logger.info("Opening range complete, starting active trading")
    
    async def _active_trading_phase(self) -> None:
        """Active trading phase: execute strategy."""
        logger.info(f"Active trading - processing {len(self._watchlist)} symbols...")
        
        # Refresh account state
        account = order_manager.get_account()
        if account:
            risk_manager.update_equity(account["equity"])
        
        signals_generated = 0
        errors = 0
        
        # Process each symbol in watchlist
        for symbol in self._watchlist:
            if self._shutdown_event.is_set():
                return
            
            if risk_manager.is_trading_halted:
                logger.warning(f"Trading halted: {risk_manager._state.halt_reason}")
                await asyncio.sleep(60)
                return
            
            try:
                signal = await self._process_symbol(symbol)
                if signal and signal.is_entry:
                    signals_generated += 1
                    logger.info(f"📈 {symbol}: {signal.signal_type.value} signal (confidence: {signal.confidence:.0%})")
            except Exception as e:
                errors += 1
                logger.error(f"Error processing {symbol}: {e}")
        
        logger.info(f"Cycle complete: {signals_generated} signals, {errors} errors")
        
        # Update existing positions (trailing stops, etc.)
        await self._update_positions()
        
        # Small delay between cycles
        await asyncio.sleep(5)
    
    async def _process_symbol(self, symbol: str) -> Optional[Signal]:
        """Process a single symbol for trading signals."""
        # Get recent bars
        try:
            bars_dict = market_data.get_bars(
                [symbol],
                timeframe=TimeFrame.Minute,
                limit=100,
            )
            
            if symbol not in bars_dict or bars_dict[symbol].empty:
                logger.debug(f"{symbol}: No bar data available")
                return None
            
            bars = bars_dict[symbol]
            self._bars_cache[symbol] = bars
            
        except Exception as e:
            logger.warning(f"Failed to get bars for {symbol}: {e}")
            return None
        
        # Get current position
        current_position = portfolio.get_position(symbol)
        position_dict = None
        if current_position:
            position_dict = {
                "side": current_position.side,
                "entry_price": current_position.entry_price,
                "stop_loss": current_position.stop_loss,
                "take_profit": current_position.take_profit,
                "highest_price": current_position.highest_price,
                "lowest_price": current_position.lowest_price,
                "trailing_stop": current_position.trailing_stop,
            }
        
        # Generate signal
        signal = strategy.generate_signals(symbol, bars, position_dict)
        self._last_signals[symbol] = signal
        
        # Log the signal result
        if signal.is_entry:
            logger.info(f"🔔 {symbol}: {signal.signal_type.value} @ ${signal.price:.2f} - {signal.reason}")
        
        # Act on signal
        if signal.is_entry:
            await self._handle_entry_signal(symbol, signal)
        elif signal.is_exit:
            await self._handle_exit_signal(symbol, signal)
        
        return signal
    
    async def _handle_entry_signal(self, symbol: str, signal: Signal) -> None:
        """Handle an entry signal."""
        # Skip if we already have a position
        if portfolio.has_position(symbol):
            return
        
        # Get quote for smart pricing
        quotes = market_data.get_latest_quotes([symbol])
        if symbol not in quotes:
            logger.warning(f"No quote available for {symbol}")
            return
        
        quote = quotes[symbol]
        bid = quote["bid_price"]
        ask = quote["ask_price"]
        
        if bid <= 0 or ask <= 0:
            return
        
        # Calculate position size
        shares = risk_manager.calculate_position_size(
            entry_price=signal.price,
            stop_loss=signal.stop_loss,
        )
        
        if shares <= 0:
            logger.debug(f"Position size too small for {symbol}")
            return
        
        # Check if we can open the position
        account = order_manager.get_account()
        buying_power = account["buying_power"] if account else 0
        
        can_trade, reason = risk_manager.can_open_position(
            symbol=symbol,
            shares=shares,
            price=signal.price,
            buying_power=buying_power,
        )
        
        if not can_trade:
            logger.info(f"Cannot open {symbol}: {reason}")
            return
        
        # Place order
        side = "BUY" if signal.signal_type == SignalType.BUY else "SELL"
        
        order_id = order_manager.place_smart_entry(
            symbol=symbol,
            side=side,
            qty=shares,
            bid_price=bid,
            ask_price=ask,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            use_bracket=True,
        )
        
        if order_id:
            trade_logger.log_signal(
                symbol=symbol,
                signal_type=side,
                confidence=signal.confidence,
                indicators=signal.indicators or {},
            )
            
            # Register with risk manager
            risk_manager.register_position(
                symbol=symbol,
                shares=shares,
                entry_price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                side="long" if side == "BUY" else "short",
            )
            
            # Open in portfolio
            portfolio.open_position(
                symbol=symbol,
                side="long" if side == "BUY" else "short",
                qty=shares,
                entry_price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )
            
            # Send notification
            await notifier.send_trade_entry(
                symbol=symbol,
                side=side,
                qty=shares,
                price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy=strategy.name,
            )
    
    async def _handle_exit_signal(self, symbol: str, signal: Signal) -> None:
        """Handle an exit signal."""
        if not portfolio.has_position(symbol):
            return
        
        # Close position via broker
        if order_manager.close_position(symbol):
            pos = portfolio.get_position(symbol)
            
            # Record trade in portfolio
            trade = portfolio.close_position(
                symbol=symbol,
                exit_price=signal.price,
                exit_reason=signal.reason,
            )
            
            if trade:
                # Update risk manager
                risk_manager.close_position(
                    symbol=symbol,
                    exit_price=signal.price,
                    is_day_trade=True,  # All our trades are day trades
                )
                
                # Send notification
                await notifier.send_trade_exit(
                    symbol=symbol,
                    side=trade.side,
                    qty=trade.qty,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    pnl=trade.pnl,
                    r_multiple=trade.r_multiple,
                    exit_reason=trade.exit_reason,
                )
    
    async def _update_positions(self) -> None:
        """Update existing positions (trailing stops, price updates)."""
        positions = portfolio.get_all_positions()
        
        for symbol, pos in positions.items():
            try:
                # Get current price
                current_price = market_data.get_current_price(symbol)
                if current_price:
                    portfolio.update_position_price(symbol, current_price)
                    risk_manager.update_position_price(symbol, current_price)
                    
                    # Check trailing stop
                    if symbol in self._bars_cache:
                        bars = self._bars_cache[symbol]
                        if "atr" in bars.columns:
                            atr = bars["atr"].iloc[-1]
                            trailing = get_trailing_stop_price(
                                current_price=current_price,
                                highest_price=pos.highest_price,
                                atr=atr,
                                entry_price=pos.entry_price,
                                side=pos.side.upper(),
                            )
                            if trailing:
                                pos.trailing_stop = trailing
                                
            except Exception as e:
                logger.debug(f"Error updating {symbol}: {e}")
    
    async def _eod_liquidation_phase(self) -> None:
        """End of day: close all positions."""
        logger.info("EOD liquidation phase - closing all positions")
        
        positions = portfolio.get_all_positions()
        
        for symbol in list(positions.keys()):
            try:
                current_price = market_data.get_current_price(symbol) or positions[symbol].current_price
                
                if order_manager.close_position(symbol):
                    trade = portfolio.close_position(
                        symbol=symbol,
                        exit_price=current_price,
                        exit_reason="EOD liquidation",
                    )
                    
                    if trade:
                        risk_manager.close_position(symbol, current_price, is_day_trade=True)
                        
                        await notifier.send_trade_exit(
                            symbol=symbol,
                            side=trade.side,
                            qty=trade.qty,
                            entry_price=trade.entry_price,
                            exit_price=trade.exit_price,
                            pnl=trade.pnl,
                            r_multiple=trade.r_multiple,
                            exit_reason="EOD liquidation",
                        )
                        
            except Exception as e:
                logger.error(f"Error closing {symbol}: {e}")
        
        # Cancel any remaining orders
        order_manager.cancel_all_orders()
        
        logger.info("EOD liquidation complete")
        
        # Wait for market close
        await asyncio.sleep(60)
    
    async def _after_hours_phase(self) -> None:
        """After hours: generate summary and sleep."""
        logger.info("After hours phase")
        
        # Generate daily summary
        await self._send_daily_summary()
        
        # Reset daily state for tomorrow
        risk_manager.reset_daily_state()
        portfolio.reset_daily_stats()
        symbol_manager.clear_dynamic()
        self._bars_cache.clear()
        self._last_signals.clear()
        
        # Sleep until next trading day
        await self._sleep_until_next_day()
    
    async def _send_daily_summary(self) -> None:
        """Send daily trading summary via Discord."""
        stats = portfolio.get_daily_stats()
        account = order_manager.get_account()
        
        await notifier.send_daily_summary(
            total_trades=stats["total_trades"],
            winning_trades=stats["winning_trades"],
            total_pnl=stats["total_pnl"],
            best_trade=stats["best_trade"],
            worst_trade=stats["worst_trade"],
            portfolio_value=account["equity"] if account else 0,
        )
        
        logger.info(
            f"Daily Summary: {stats['total_trades']} trades, "
            f"Win rate: {stats['win_rate']:.1f}%, "
            f"P&L: ${stats['total_pnl']:+.2f}"
        )
    
    async def _sleep_until_next_day(self) -> None:
        """Sleep until the next trading day."""
        next_day = get_next_trading_day()
        wake_time = datetime.combine(next_day, datetime.min.time().replace(hour=4))
        
        now = now_et()
        sleep_seconds = (wake_time - now.replace(tzinfo=None)).total_seconds()
        
        if sleep_seconds > 0:
            logger.info(f"Sleeping until {wake_time.strftime('%Y-%m-%d %H:%M')} ET")
            
            # Sleep in chunks to remain responsive to shutdown
            while sleep_seconds > 0 and not self._shutdown_event.is_set():
                chunk = min(300, sleep_seconds)  # 5 minute chunks
                await asyncio.sleep(chunk)
                sleep_seconds -= chunk


def main():
    """Main entry point."""
    bot = TradingBot()
    
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
