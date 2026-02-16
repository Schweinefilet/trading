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
from strategy.signals import generate_signals
from strategy.regime import get_regime, get_position_size_multiplier, regime_allows_trade
from strategy.ranker import SignalRanker
from risk.position_sizer import calculate_position_size, calculate_risk_dollars
from risk.circuit_breaker import CircuitBreaker
from risk.pdt_tracker import PDTTracker
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

        # State
        self.equity = config.STARTING_CAPITAL
        self.running = False
        self.regime_state = "BULLISH"
        self.spy_bars = None
        self.bar_data = {}  # Warm-up data per symbol
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

            trading_client = TradingClient(
                api_key=config.ALPACA_API_KEY,
                secret_key=config.ALPACA_SECRET_KEY,
                paper=self.paper,
            )

            self.order_manager.set_client(trading_client)
            self.reconciler.set_client(trading_client)

            # Fetch account equity
            account = trading_client.get_account()
            self.equity = float(account.equity)
            logger.info(f"Account initialized. Equity: ${self.equity:,.2f}")

        except Exception as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
            logger.info("Running in offline/simulation mode.")
            self.equity = config.STARTING_CAPITAL

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

        # Update positions
        self.heat_manager.clear()
        for sym, pos_data in recon["positions"].items():
            pos = Position(
                symbol=sym,
                shares=pos_data.get("qty", 0),
                entry_price=pos_data.get("entry_price", 0),
                stop_price=0,  # Will be updated from order data
                side=pos_data.get("side", "long"),
            )
            self.heat_manager.add_position(pos)

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

        # PDT check
        pdt_status = self.pdt_tracker.get_status(self.equity)
        day_trades_remaining = pdt_status["day_trades_remaining"]

        # Generate signals for all symbols
        signals = []
        for sym in tickers.TRADE_TICKERS:
            if sym not in self.bar_data:
                continue

            df = self.bar_data[sym]
            signal = generate_signals(sym, df, self.regime_state)
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
        shares = calculate_position_size(
            equity=self.equity,
            entry_price=sig.entry_price,
            stop_price=sig.stop_loss,
            atr=sig.atr,
            regime_multiplier=regime_mult * dd_mult,
            peak_equity=self.circuit_breaker.state.peak_equity,
            current_deployed=heat_summary.get("deployed_dollars", 0),
            current_portfolio_heat=heat_summary.get("heat_dollars", 0),
        )

        if shares < 1:
            return

        # Check heat limits
        risk_dollars = calculate_risk_dollars(shares, sig.entry_price, sig.stop_loss)
        can_add, reason = self.heat_manager.can_add_position(sig.symbol, risk_dollars, self.equity)
        if not can_add:
            logger.info(f"Skipping {sig.symbol}: {reason}")
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

        # Check stream health
        if connected and self.data_stream.seconds_since_last_data() > 120:
            logger.warning("No data received in 2 minutes. Stream may need reconnect.")

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
