"""
Main entry point and orchestration for the live trading bot.
Handles startup, reconciliation, the main event loop, and shutdown.

Usage:
    python main.py [--paper] [--live]
"""
import sys
import time
import signal
import argparse
from datetime import datetime, date, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import config
from config.tickers import tickers
from data.historical import HistoricalDataClient, parse_timeframe
from data.indicators import compute_indicators, compute_regime_indicators
from data.stream import DataStream
from data.validation import DataValidator
from strategy.signals import generate_signals
from strategy.regime import get_regime, get_position_size_multiplier, regime_allows_trade
from strategy.ranker import SignalRanker
from risk.position_sizer import calculate_position_size, calculate_risk_dollars
from risk.circuit_breaker import CircuitBreaker
from risk.pdt_tracker import PDTTracker
from risk.validation import PreTradeValidator
from risk.portfolio_heat import PortfolioHeatManager, Position
from execution.order_manager import OrderManager
from execution.reconciliation import Reconciler
from monitoring.logger import (
    logger, log_startup, log_signal, log_trade_entry, log_pdt_status,
    log_circuit_breaker, log_heartbeat, log_daily_summary
)
from monitoring.alerts import alerts
from monitoring.dashboard import print_dashboard


class TradingBot:
    """
    Main trading bot orchestrator.
    Manages the full lifecycle: startup → reconcile → run → shutdown.
    """

    def __init__(self, paper: bool = True):
        self.paper = paper
        config.PAPER_TRADING = paper

        # Core components
        self.data_stream = DataStream()
        self.order_manager = OrderManager()
        self.reconciler = Reconciler()
        self.circuit_breaker = CircuitBreaker()
        self.pdt_tracker = PDTTracker()
        self.heat_manager = PortfolioHeatManager()
        self.ranker = SignalRanker()
        self.hist_client = HistoricalDataClient()
        self.validator = DataValidator()
        self.pre_trade_validator = PreTradeValidator()

        # State
        self.equity = config.STARTING_CAPITAL
        self.running = False
        self.regime_state = "BULLISH"
        self.spy_bars = None
        self.bar_data = {}  # Warm-up data per symbol
        self.quotes = {}    # Latest bid/ask per symbol
        self.last_heartbeat = time.monotonic()
        self.daily_pnl = 0.0
        self.daily_trades = 0

    def start(self):
        """Full startup sequence."""
        logger.info("=" * 60)
        logger.info(f"STARTING {'PAPER' if self.paper else 'LIVE'} TRADING BOT")
        logger.info("=" * 60)

        # Initialize Alpaca clients
        self._init_clients()

        # Reconcile state
        self._reconcile()

        # Warm up indicators
        self._warm_up()

        # Compute initial regime
        self._update_regime()

        # Start data stream
        self.data_stream.on_bar(self._on_bar)
        self.data_stream.on_quote(self._on_quote)
        self.data_stream.on_connect(self._reconcile)
        self.data_stream.start()

        # Reset daily circuit breaker
        self.circuit_breaker.reset_daily()
        self.circuit_breaker.reset_weekly()
        self.circuit_breaker.reset_monthly()
        self.circuit_breaker.update_peak_equity(self.equity)

        # Log startup
        pdt_status = self.pdt_tracker.get_status(self.equity)
        log_startup(
            {"equity": self.equity},
            len(self.heat_manager.positions),
            self.regime_state,
        )
        log_pdt_status(pdt_status["day_trades_used"], pdt_status["day_trades_remaining"], pdt_status["pdt_limit"])

        self.running = True
        logger.info("Startup complete. Entering main loop.")

    def _init_clients(self):
        """Initialize Alpaca API clients."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient

            self.trading_client = TradingClient(
                api_key=config.ALPACA_API_KEY,
                secret_key=config.ALPACA_SECRET_KEY,
                paper=self.paper,
            )

            self.order_manager.set_client(self.trading_client)
            self.reconciler.set_client(self.trading_client)

            # Fetch account equity
            account = trading_client.get_account()
            self.equity = float(account.equity)
            logger.info(f"Account initialized. Equity: ${self.equity:,.2f}")

        except Exception as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
            logger.info("Running in offline/simulation mode.")
            self.equity = config.STARTING_CAPITAL

    def _update_account_equity(self):
        """Fetch latest equity from Alpaca."""
        if hasattr(self, 'trading_client'):
            try:
                account = self.trading_client.get_account()
                self.equity = float(account.equity)
                # Note: Equity is used globally for position sizing
            except Exception as e:
                logger.error(f"Failed to update equity: {e}")

    def _reconcile(self):
        """Reconcile local state with server."""
        logger.info("Reconciling state with Alpaca server...")

        recon = self.reconciler.reconcile(
            local_positions={sym: {} for sym in self.heat_manager.positions},
            local_orders={},
        )

        # Update equity
        if recon["account"]:
            self.equity = recon["account"].get("equity", self.equity)

        # Update OrderManager (which also updates positions locally)
        self.order_manager.sync_state(recon["positions"], recon["orders"])

        # Update HeatManager
        self.heat_manager.clear()
        for sym, pos_data in recon["positions"].items():
            # Create a Position object for heat tracking
            # We don't have stop_price from Alpaca's simple position fetch, 
            # we'll have to either find it from orders or use a default.
            # For now, we'll use a 0.0 stop_price which means maximum risk (conservative for heat)
            pos = Position(
                symbol=sym,
                shares=pos_data["qty"],
                entry_price=pos_data["entry_price"],
                stop_price=0.0, # Unknown
                side=pos_data["side"].upper(),
            )
            self.heat_manager.add_position(pos)

        logger.info(f"Reconciliation complete. Positions: {len(self.heat_manager.positions)}")

        # Rebuild PDT counter from fills
        fills = self.reconciler.get_recent_fills(days=7)
        if fills:
            self.pdt_tracker.rebuild_from_fills(fills)

        for w in recon.get("warnings", []):
            logger.warning(f"Reconciliation: {w}")

    def _warm_up(self):
        """Warm up indicator buffers for all symbols."""
        logger.info("Warming up indicators...")
        tf = parse_timeframe(config.PRIMARY_TIMEFRAME)

        all_symbols = tickers.TRADE_TICKERS + tickers.CONTEXT_TICKERS
        self.bar_data = self.hist_client.warm_indicators(all_symbols, tf, config.LOOKBACK_BARS)

        for sym in self.bar_data:
            self.bar_data[sym] = compute_indicators(self.bar_data[sym])

        logger.info(f"Warmed {len(self.bar_data)} symbols with {config.LOOKBACK_BARS} bars each.")

        # SPY daily for regime
        end = datetime.now()
        start = end - timedelta(days=300)
        spy_tf = parse_timeframe("1Day")
        self.spy_bars = self.hist_client.fetch_bars("SPY", spy_tf, start, end)
        if not self.spy_bars.empty:
            self.spy_bars = compute_regime_indicators(self.spy_bars)

    def _update_regime(self):
        """Update market regime state."""
        if self.spy_bars is not None and not self.spy_bars.empty:
            regime = get_regime(self.spy_bars)
            self.regime_state = regime.value
            logger.info(f"Regime: {self.regime_state}")

    def _on_bar(self, symbol: str, bar_dict: dict):
        """Handle incoming bar from stream."""
        # Phase 113: Validate incoming bar
        is_valid, reason = self.validator.validate_bar(symbol, bar_dict)
        if not is_valid:
            logger.warning(f"Skipping invalid bar for {symbol}: {reason}")
            return

        try:
            # Update bar data
            import pandas as pd
            new_row = pd.DataFrame([bar_dict])
            if "timestamp" in new_row.columns:
                new_row = new_row.set_index("timestamp")

            if symbol in self.bar_data:
                self.bar_data[symbol] = pd.concat([self.bar_data[symbol], new_row])
                if len(self.bar_data[symbol]) > config.LOOKBACK_BARS:
                    self.bar_data[symbol] = self.bar_data[symbol].iloc[-config.LOOKBACK_BARS:]
                # Recompute indicators
                self.bar_data[symbol] = compute_indicators(self.bar_data[symbol])
            else:
                self.bar_data[symbol] = new_row

        except Exception as e:
            logger.error(f"Error processing bar for {symbol}: {e}")

    def _on_quote(self, symbol: str, quote_dict: dict):
        """Handle incoming quote from stream."""
        self.quotes[symbol] = quote_dict

    def run_loop(self):
        """Main trading loop. Runs until stopped."""
        while self.running:
            try:
                # Wait for new bar data
                got_bar = self.data_stream.wait_for_bar(timeout=config.HEARTBEAT_INTERVAL_SEC)

                if not got_bar:
                    # Heartbeat
                    self._heartbeat()
                    continue

                # Process signals
                self._process_signals()

                # Check advanced exits (EOD Profit Lock, Dead Money)
                self._check_advanced_exits()

                # Heartbeat check
                now = time.monotonic()
                if now - self.last_heartbeat >= config.HEARTBEAT_INTERVAL_SEC:
                    self._heartbeat()

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received.")
                self.stop()
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                alerts.error(str(e), "main_loop")
                time.sleep(5)

    def _process_signals(self):
        """Generate, rank, and execute signals."""
        # Circuit breaker check
        can_trade, reason = self.circuit_breaker.can_trade(self.equity)
        if not can_trade:
            log_circuit_breaker(reason, {"daily_pnl": self.daily_pnl}, "HALT")
            return

        # Compute leaderboard/rankings for Elite Selection
        leaderboard = self._get_leaderboard()

        # PDT check
        pdt_status = self.pdt_tracker.get_status(self.equity)
        day_trades_remaining = pdt_status["day_trades_remaining"]

        # Generate signals for all symbols
        signals = []
        for sym in tickers.TRADE_TICKERS:
            if sym not in self.bar_data:
                continue

            df = self.bar_data[sym]
            
            # Pass ranking to signal generator
            rank_data = leaderboard.get(sym, {"rank": 999, "percentile": 1.0})
            
            signal = generate_signals(
                sym, df, self.regime_state, 
                rank_int=rank_data["rank"], 
                rank_percentile=rank_data["percentile"]
            )
            
            if signal:
                log_signal(
                    sym, signal.direction.value, signal.signal_strength,
                    {"rsi": df.iloc[-1].get("rsi", 0), "adx": df.iloc[-1].get("adx", 0),
                     "volume_ratio": df.iloc[-1].get("volume", 0) / max(df.iloc[-1].get("volume_sma_20", 1), 1)}
                )
                signals.append(signal)

        if not signals:
            return

        # Rank and select
        heat_summary = self.heat_manager.get_summary(self.equity)
        day_trades, swing_trades = self.ranker.rank_and_select(
            signals=signals,
            open_positions={sym: {} for sym in self.heat_manager.positions},
            sector_positions=heat_summary.get("sectors", {}),
            semi_super_count=heat_summary.get("semi_super_count", 0),
            portfolio_heat=heat_summary.get("heat_dollars", 0),
            equity=self.equity,
            day_trades_remaining=day_trades_remaining,
        )

        # Execute selected signals
        for sig in day_trades + swing_trades:
            self._execute_signal(sig)

    def _execute_signal(self, sig):
        """Execute a single trade signal."""
        # Position sizing
        regime_mult = get_position_size_multiplier(
            __import__('strategy.regime', fromlist=['RegimeState']).RegimeState(self.regime_state),
            sig.direction.value
        )
        dd_mult = self.circuit_breaker.get_size_multiplier(self.equity)

        heat_summary = self.heat_manager.get_summary(self.equity)
        
        # SPY ATR for Volatility Gating
        spy_atr = 0.0
        spy_atr_sma = 0.0
        if "SPY" in self.bar_data:
            spy_df = self.bar_data["SPY"]
            if not spy_df.empty:
                spy_atr = spy_df.iloc[-1].get("atr", 0)
                spy_atr_sma = spy_df.iloc[-15:-1]["atr"].mean() if len(spy_df) >= 15 else spy_atr

        shares = calculate_position_size(
            equity=self.equity,
            entry_price=sig.entry_price,
            stop_price=sig.stop_loss,
            atr=sig.atr,
            regime_multiplier=regime_mult * dd_mult,
            peak_equity=self.circuit_breaker.state.peak_equity,
            current_deployed=heat_summary.get("deployed_dollars", 0),
            current_portfolio_heat=heat_summary.get("heat_dollars", 0),
            spy_atr=spy_atr,
            spy_atr_sma=spy_atr_sma,
        )

        if shares < 1:
            return

        # Check heat limits
        risk_dollars = calculate_risk_dollars(shares, sig.entry_price, sig.stop_loss)
        can_add, reason = self.heat_manager.can_add_position(sig.symbol, risk_dollars, self.equity)
        if not can_add:
            logger.info(f"Skipping {sig.symbol}: {reason}")
            return

        # Phase 113: Final 7-Point Validation Gate
        pdt_status = self.pdt_tracker.get_status(self.equity)
        can_trade_cb, _ = self.circuit_breaker.can_trade(self.equity)
        
        # Determine last bar time for freshness check
        last_bar_time = datetime.now()
        if sig.symbol in self.bar_data:
            last_bar_time = self.bar_data[sig.symbol].index[-1]

        is_valid, reason = self.pre_trade_validator.validate(
            symbol=sig.symbol,
            side="buy" if sig.direction.value == "LONG" else "sell",
            qty=shares,
            price=sig.entry_price,
            stop_loss=sig.stop_loss,
            equity=self.equity,
            pdt_status=pdt_status,
            heat_summary=heat_summary,
            circuit_breaker_ok=can_trade_cb,
            last_bar_time=last_bar_time,
            quote=self.quotes.get(sig.symbol),
        )

        if not is_valid:
            logger.warning(f"Trade REJECTED by 7-Point Validation for {sig.symbol}: {reason}")
            alerts.error(f"Trade Rejected: {sig.symbol} - {reason}", "risk")
            return

        # Submit bracket order
        side = "buy" if sig.direction.value == "LONG" else "sell"
        order_id = self.order_manager.submit_bracket_order(
            symbol=sig.symbol,
            side=side,
            qty=shares,
            limit_price=sig.entry_price,
            stop_loss_price=sig.stop_loss,
            take_profit_price=sig.take_profit,
            is_day_trade=not sig.is_swing,
        )

        if order_id:
            # Add to heat manager
            pos = Position(
                symbol=sig.symbol,
                shares=shares,
                entry_price=sig.entry_price,
                stop_price=sig.stop_loss,
                side=sig.direction.value,
            )
            self.heat_manager.add_position(pos)

            log_trade_entry(
                sig.symbol, sig.direction.value, shares, sig.entry_price,
                sig.stop_loss, sig.take_profit, shares * sig.entry_price, sig.reason,
            )
            alerts.trade_executed(sig.symbol, side, shares, sig.entry_price)
            self.daily_trades += 1

    def _heartbeat(self):
        """Periodic heartbeat: log status and check health."""
        self.last_heartbeat = time.monotonic()
        connected = self.data_stream.is_connected

        # Refresh equity from Alpaca periodically
        self._update_account_equity()

        log_heartbeat(len(self.heat_manager.positions), self.daily_pnl, connected)

        # Print dashboard
        pdt_status = self.pdt_tracker.get_status(self.equity)
        heat_summary = self.heat_manager.get_summary(self.equity)

        positions_dict = {}
        for sym, pos in self.heat_manager.positions.items():
            positions_dict[sym] = {
                "side": pos.side,
                "qty": pos.shares,
                "entry_price": pos.entry_price,
                "current_price": pos.entry_price,  # Would need market data for real value
                "unrealized_pl": 0,
            }

        print_dashboard(
            equity=self.equity,
            daily_pnl=self.daily_pnl,
            positions=positions_dict,
            heat_pct=heat_summary.get("heat_pct", 0),
            pdt_status=pdt_status,
            regime=self.regime_state,
            connected=connected,
        )

        # telemetry and state saving
        telemetry = {
            "equity": self.equity,
            "daily_pnl": self.daily_pnl,
            "positions_count": len(self.heat_manager.positions),
            "heat_pct": heat_summary.get("heat_pct", 0),
            "regime": self.regime_state,
            "connected": connected,
            "trades_today": self.daily_trades,
            "day_trades_used": pdt_status.get("day_trades_used", 0)
        }
        alerts.performance_telemetry(telemetry)

        # Save current state for dashboard
        try:
            import json
            state_file = config.BASE_DIR / "logs" / "state.json"
            state = {
                "last_update": datetime.now().isoformat(),
                "metrics": telemetry,
                "positions": positions_dict,
                "pdt": pdt_status,
                "regime": self.regime_state
            }
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

        # Check stream health
        if connected and self.data_stream.seconds_since_last_data() > 120:
            logger.warning("No data received in 2 minutes. Stream may need reconnect.")

    def _get_leaderboard(self) -> dict:
        """Compute RS ranking (roc_125) for all tickers."""
        scores = {}
        for sym, df in self.bar_data.items():
            if sym in ["SPY", "QQQ"]:
                continue
            if not df.empty:
                val = df.iloc[-1].get("roc_125", np.nan)
                if not np.isnan(val):
                    scores[sym] = val
        
        if not scores:
            return {}
            
        # Rank: higher roc = better (rank 1)
        sorted_syms = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        num = len(sorted_syms)
        return {
            sym: {"rank": i + 1, "percentile": (i + 1) / num}
            for i, sym in enumerate(sorted_syms)
        }

    def _check_advanced_exits(self):
        """Implement EOD Profit Lock and Dead Money time-stops."""
        import pytz
        et = pytz.timezone("US/Eastern")
        now_et = datetime.now(et)
        
        for sym, pos in list(self.heat_manager.positions.items()):
            # 1. 3:30 PM Profit Locking
            if (now_et.hour == 15 and now_et.minute >= 30) or (now_et.hour >= 16):
                # Check current price vs entry
                curr_price = self.quotes.get(sym, {}).get("bid") if pos.side == "LONG" else self.quotes.get(sym, {}).get("ask")
                if curr_price:
                    if (pos.side == "LONG" and curr_price > pos.entry_price) or \
                       (pos.side == "SHORT" and curr_price < pos.entry_price):
                        logger.info(f"EOD Profit Lock triggered for {sym} at {curr_price}")
                        self._liquidate_position(sym, "eod_profit_lock")
                        continue

            # 2. Dead Money (48h + < 1.0x ATR profit)
            hold_hours = (datetime.now() - pos.entry_time).total_seconds() / 3600
            if hold_hours >= config.MAX_HOLD_HOURS:
                curr_price = self.quotes.get(sym, {}).get("bid") if pos.side == "LONG" else self.quotes.get(sym, {}).get("ask")
                if curr_price and pos.atr > 0:
                    profit_atr = abs(curr_price - pos.entry_price) / pos.atr
                    if profit_atr < 1.0:
                        logger.info(f"Dead Money exit triggered for {sym} (Hold: {hold_hours:.1f}h, Profit: {profit_atr:.2f} ATR)")
                        self._liquidate_position(sym, "time_stop_dead_money")

    def _liquidate_position(self, symbol: str, reason: str):
        """Close a position immediately."""
        pos = self.heat_manager.positions.get(symbol)
        if not pos:
            return
            
        side = "sell" if pos.side == "LONG" else "buy"
        order_id = self.order_manager.submit_market_close(symbol, pos.shares, side)
        if order_id:
            logger.info(f"Liquidated {symbol}: {reason}")
            # OrderManager.handle_fill will eventually remove it from heat_manager if fully synced
            # But for safety, we should really track it better. 
            # In main.py, heat_manager is updated via sync_state/reconcile.

    def stop(self):
        """Graceful shutdown."""
        logger.info("Shutting down trading bot...")
        self.running = False

        # Close day trade positions
        self.order_manager.close_day_trade_positions()

        # Stop stream
        self.data_stream.stop()

        # Final summary
        log_daily_summary(
            self.daily_pnl, self.daily_trades,
            len([t for t in self.heat_manager.positions.values()]),
            0,  # wins/losses tracked elsewhere
            self.heat_manager.calculate_heat() / max(self.equity, 1) * 100,
            self.equity,
        )

        alerts.daily_summary(self.daily_pnl, self.daily_trades, self.equity)
        logger.info("Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="Algo Trader Bot")
    parser.add_argument("--paper", action="store_true", default=True, help="Paper trading mode (default)")
    parser.add_argument("--live", action="store_true", help="LIVE trading mode (WARNING: uses real money)")
    args = parser.parse_args()

    paper = not args.live

    if not paper:
        confirm = input("\n⚠️  LIVE TRADING MODE — This uses REAL MONEY. Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            print("Aborted.")
            return

    bot = TradingBot(paper=paper)

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        bot.start()
        bot.run_loop()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        alerts.error(str(e), "fatal")
        bot.stop()
        raise


if __name__ == "__main__":
    main()
