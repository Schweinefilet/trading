#!/usr/bin/env python3
"""
Backtesting Engine for Trading Bot
Runs strategy logic against historical data

Usage:
    python backtest.py --symbol AAPL --start 2024-01-01 --end 2024-12-31 --timeframe 5Min
"""

import argparse
import csv
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config.settings import settings
from data.market_data import MarketDataClient, get_timeframe
from data.indicators import calculate_all_indicators
from core.strategy import MultiIndicatorMomentumStrategy, Signal, SignalType
from utils.helpers import format_pnl, format_percent


@dataclass
class BacktestTrade:
    """Record of a simulated trade."""
    symbol: str
    side: str
    qty: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    r_multiple: float
    exit_reason: str


@dataclass
class BacktestResult:
    """Backtesting results summary."""
    symbol: str
    start_date: str
    end_date: str
    timeframe: str
    
    # Performance
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    avg_r_multiple: float
    
    # Trade details
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float  # minutes
    
    trades: List[BacktestTrade] = field(default_factory=list)


class Backtester:
    """
    Backtesting engine for evaluating trading strategies.
    
    Features:
    - Historical data fetching from Alpaca
    - Same strategy logic as live trading
    - Configurable slippage simulation
    - Comprehensive metrics calculation
    """
    
    def __init__(
        self,
        strategy: MultiIndicatorMomentumStrategy = None,
        initial_capital: float = 10000.0,
        slippage_pct: float = 0.0005,  # 0.05% default
        commission_per_share: float = 0.0,  # Alpaca is commission-free
    ):
        """
        Initialize backtester.
        
        Args:
            strategy: Strategy instance to test
            initial_capital: Starting capital
            slippage_pct: Slippage as percentage of price
            commission_per_share: Commission per share traded
        """
        self.strategy = strategy or MultiIndicatorMomentumStrategy()
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.commission_per_share = commission_per_share
        
        self._data_client = MarketDataClient()
    
    def run(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.Minute,
    ) -> BacktestResult:
        """
        Run backtest for a symbol over a date range.
        
        Args:
            symbol: Ticker symbol
            start: Start date
            end: End date
            timeframe: Bar timeframe
            
        Returns:
            BacktestResult with all metrics
        """
        print(f"\nBacktesting {symbol} from {start.date()} to {end.date()}")
        print(f"Timeframe: {timeframe}")
        print(f"Initial capital: ${self.initial_capital:,.2f}")
        print("-" * 50)
        
        # Fetch historical data
        print("Fetching historical data...")
        bars = self._fetch_data(symbol, start, end, timeframe)
        
        if bars.empty:
            raise ValueError(f"No data available for {symbol}")
        
        print(f"Loaded {len(bars)} bars")
        
        # Calculate indicators
        print("Calculating indicators...")
        df = calculate_all_indicators(bars)
        
        # Run simulation
        print("Running simulation...")
        trades, equity_curve = self._simulate(symbol, df)
        
        # Calculate metrics
        print("Calculating metrics...")
        result = self._calculate_metrics(
            symbol=symbol,
            start=start,
            end=end,
            timeframe=str(timeframe),
            trades=trades,
            equity_curve=equity_curve,
        )
        
        return result
    
    def _fetch_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame,
    ) -> pd.DataFrame:
        """Fetch historical bar data."""
        # Fetch in chunks if date range is large
        all_bars = []
        current_start = start
        chunk_days = 30
        
        while current_start < end:
            chunk_end = min(current_start + timedelta(days=chunk_days), end)
            
            try:
                bars = self._data_client.get_bars_df(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=current_start,
                    end=chunk_end,
                    limit=10000,
                )
                
                if not bars.empty:
                    all_bars.append(bars)
                    
            except Exception as e:
                print(f"  Warning: Failed to fetch {current_start.date()} - {chunk_end.date()}: {e}")
            
            current_start = chunk_end
        
        if not all_bars:
            return pd.DataFrame()
        
        return pd.concat(all_bars).sort_index()
    
    def _simulate(
        self,
        symbol: str,
        df: pd.DataFrame,
    ) -> Tuple[List[BacktestTrade], List[float]]:
        """
        Simulate trading on historical data.
        
        Returns:
            Tuple of (trades, equity_curve)
        """
        trades = []
        equity_curve = [self.initial_capital]
        
        capital = self.initial_capital
        position: Optional[Dict] = None
        
        # Need enough bars for indicators
        lookback = max(settings.EMA_SLOW, settings.ATR_PERIOD, 26) + 5
        
        for i in range(lookback, len(df)):
            # Get data window
            window = df.iloc[i - lookback:i + 1].copy()
            current_bar = df.iloc[i]
            current_price = current_bar["close"]
            current_time = current_bar.name if isinstance(current_bar.name, datetime) else df.index[i]
            
            # Generate signal
            position_dict = None
            if position:
                position_dict = {
                    "side": position["side"],
                    "entry_price": position["entry_price"],
                    "stop_loss": position["stop_loss"],
                    "take_profit": position["take_profit"],
                    "highest_price": position["highest_price"],
                    "lowest_price": position["lowest_price"],
                }
            
            signal = self.strategy.generate_signals(symbol, window, position_dict)
            
            # Handle exits
            if position and signal.is_exit:
                trade = self._close_position(
                    position, current_price, current_time, signal.reason
                )
                trades.append(trade)
                capital += trade.pnl
                position = None
            
            # Check stop/target hits for open position
            elif position:
                exit_price, exit_reason = self._check_stops(position, current_bar)
                if exit_price:
                    trade = self._close_position(
                        position, exit_price, current_time, exit_reason
                    )
                    trades.append(trade)
                    capital += trade.pnl
                    position = None
                else:
                    # Update high/low tracking
                    if position["side"] == "long":
                        position["highest_price"] = max(position["highest_price"], current_bar["high"])
                    else:
                        position["lowest_price"] = min(position["lowest_price"], current_bar["low"])
            
            # Handle entries
            if not position and signal.is_entry and signal.stop_loss:
                position = self._open_position(
                    symbol=symbol,
                    signal=signal,
                    capital=capital,
                    current_time=current_time,
                )
            
            # Track equity
            if position:
                unrealized = self._calculate_unrealized(position, current_price)
                equity_curve.append(capital + unrealized)
            else:
                equity_curve.append(capital)
        
        # Close any remaining position at end
        if position and len(df) > 0:
            final_price = df["close"].iloc[-1]
            final_time = df.index[-1]
            trade = self._close_position(position, final_price, final_time, "End of backtest")
            trades.append(trade)
            capital += trade.pnl
            equity_curve.append(capital)
        
        return trades, equity_curve
    
    def _open_position(
        self,
        symbol: str,
        signal: Signal,
        capital: float,
        current_time: datetime,
    ) -> Dict:
        """Open a simulated position."""
        # Apply slippage
        if signal.signal_type == SignalType.BUY:
            entry_price = signal.price * (1 + self.slippage_pct)
        else:
            entry_price = signal.price * (1 - self.slippage_pct)
        
        # Calculate position size (1% risk per trade)
        risk_per_share = abs(entry_price - signal.stop_loss)
        risk_amount = capital * settings.PER_TRADE_RISK_PCT
        qty = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        
        # Cap at max position size
        max_qty = int(capital * settings.MAX_POSITION_PCT / entry_price)
        qty = min(qty, max_qty)
        
        if qty <= 0:
            return None
        
        return {
            "symbol": symbol,
            "side": "long" if signal.signal_type == SignalType.BUY else "short",
            "qty": qty,
            "entry_price": entry_price,
            "entry_time": current_time,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "highest_price": entry_price,
            "lowest_price": entry_price,
        }
    
    def _close_position(
        self,
        position: Dict,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
    ) -> BacktestTrade:
        """Close a simulated position."""
        # Apply slippage
        if position["side"] == "long":
            actual_exit = exit_price * (1 - self.slippage_pct)
            pnl = (actual_exit - position["entry_price"]) * position["qty"]
        else:
            actual_exit = exit_price * (1 + self.slippage_pct)
            pnl = (position["entry_price"] - actual_exit) * position["qty"]
        
        # Apply commission
        pnl -= self.commission_per_share * position["qty"] * 2  # Entry + exit
        
        # Calculate metrics
        pnl_pct = (pnl / (position["entry_price"] * position["qty"])) * 100
        
        risk = abs(position["entry_price"] - position["stop_loss"])
        reward = actual_exit - position["entry_price"] if position["side"] == "long" else position["entry_price"] - actual_exit
        r_multiple = reward / risk if risk > 0 else 0
        
        return BacktestTrade(
            symbol=position["symbol"],
            side=position["side"],
            qty=position["qty"],
            entry_price=position["entry_price"],
            exit_price=actual_exit,
            entry_time=position["entry_time"],
            exit_time=exit_time,
            pnl=pnl,
            pnl_pct=pnl_pct,
            r_multiple=r_multiple,
            exit_reason=exit_reason,
        )
    
    def _check_stops(
        self,
        position: Dict,
        bar: pd.Series,
    ) -> Tuple[Optional[float], str]:
        """Check if stop-loss or take-profit was hit."""
        if position["side"] == "long":
            if bar["low"] <= position["stop_loss"]:
                return position["stop_loss"], "Stop-loss"
            if bar["high"] >= position["take_profit"]:
                return position["take_profit"], "Take-profit"
        else:
            if bar["high"] >= position["stop_loss"]:
                return position["stop_loss"], "Stop-loss"
            if bar["low"] <= position["take_profit"]:
                return position["take_profit"], "Take-profit"
        
        return None, ""
    
    def _calculate_unrealized(self, position: Dict, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if position["side"] == "long":
            return (current_price - position["entry_price"]) * position["qty"]
        else:
            return (position["entry_price"] - current_price) * position["qty"]
    
    def _calculate_metrics(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
        trades: List[BacktestTrade],
        equity_curve: List[float],
    ) -> BacktestResult:
        """Calculate comprehensive backtest metrics."""
        final_capital = equity_curve[-1] if equity_curve else self.initial_capital
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Trade statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # P&L statistics
        if trades:
            wins = [t.pnl for t in trades if t.pnl > 0]
            losses = [t.pnl for t in trades if t.pnl < 0]
            
            avg_trade_pnl = sum(t.pnl for t in trades) / len(trades)
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            largest_win = max(wins) if wins else 0
            largest_loss = min(losses) if losses else 0
            avg_r = sum(t.r_multiple for t in trades) / len(trades)
            
            # Profit factor
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
            
            # Average trade duration
            durations = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in trades]
            avg_duration = sum(durations) / len(durations) if durations else 0
        else:
            avg_trade_pnl = avg_win = avg_loss = largest_win = largest_loss = 0
            avg_r = profit_factor = avg_duration = 0
        
        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0
        max_dd_pct = 0
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = (dd / peak) * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
        
        # Sharpe ratio (simplified, assuming daily returns)
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        return BacktestResult(
            symbol=symbol,
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            timeframe=timeframe,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_r_multiple=avg_r,
            avg_trade_pnl=avg_trade_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration=avg_duration,
            trades=trades,
        )
    
    def print_results(self, result: BacktestResult) -> None:
        """Print backtest results to console."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        
        print(f"\nSymbol: {result.symbol}")
        print(f"Period: {result.start_date} to {result.end_date}")
        print(f"Timeframe: {result.timeframe}")
        
        print("\n--- Performance ---")
        print(f"Initial Capital:  ${result.initial_capital:,.2f}")
        print(f"Final Capital:    ${result.final_capital:,.2f}")
        print(f"Total Return:     {format_pnl(result.total_return)} ({format_percent(result.total_return_pct)})")
        
        print("\n--- Trade Statistics ---")
        print(f"Total Trades:     {result.total_trades}")
        print(f"Winning Trades:   {result.winning_trades}")
        print(f"Losing Trades:    {result.losing_trades}")
        print(f"Win Rate:         {result.win_rate:.1f}%")
        
        print("\n--- Risk Metrics ---")
        print(f"Max Drawdown:     {format_pnl(-result.max_drawdown)} ({format_percent(-result.max_drawdown_pct)})")
        print(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
        print(f"Profit Factor:    {result.profit_factor:.2f}")
        print(f"Avg R-Multiple:   {result.avg_r_multiple:+.2f}")
        
        print("\n--- Trade Details ---")
        print(f"Avg Trade P&L:    {format_pnl(result.avg_trade_pnl)}")
        print(f"Avg Win:          {format_pnl(result.avg_win)}")
        print(f"Avg Loss:         {format_pnl(result.avg_loss)}")
        print(f"Largest Win:      {format_pnl(result.largest_win)}")
        print(f"Largest Loss:     {format_pnl(result.largest_loss)}")
        print(f"Avg Duration:     {result.avg_trade_duration:.1f} minutes")
        
        print("\n" + "=" * 60)
    
    def save_results(self, result: BacktestResult, output_dir: Path = None) -> str:
        """
        Save backtest results to CSV.
        
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = settings.BACKTEST_DIR
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = output_dir / f"backtest_{result.symbol}_{timestamp}.csv"
        
        with open(summary_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Symbol", result.symbol])
            writer.writerow(["Start Date", result.start_date])
            writer.writerow(["End Date", result.end_date])
            writer.writerow(["Timeframe", result.timeframe])
            writer.writerow(["Initial Capital", f"{result.initial_capital:.2f}"])
            writer.writerow(["Final Capital", f"{result.final_capital:.2f}"])
            writer.writerow(["Total Return", f"{result.total_return:.2f}"])
            writer.writerow(["Total Return %", f"{result.total_return_pct:.2f}"])
            writer.writerow(["Total Trades", result.total_trades])
            writer.writerow(["Win Rate %", f"{result.win_rate:.2f}"])
            writer.writerow(["Max Drawdown", f"{result.max_drawdown:.2f}"])
            writer.writerow(["Max Drawdown %", f"{result.max_drawdown_pct:.2f}"])
            writer.writerow(["Sharpe Ratio", f"{result.sharpe_ratio:.2f}"])
            writer.writerow(["Profit Factor", f"{result.profit_factor:.2f}"])
            writer.writerow(["Avg R-Multiple", f"{result.avg_r_multiple:.2f}"])
        
        # Save trades
        if result.trades:
            trades_file = output_dir / f"backtest_{result.symbol}_{timestamp}_trades.csv"
            
            with open(trades_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "symbol", "side", "qty", "entry_price", "exit_price",
                    "entry_time", "exit_time", "pnl", "pnl_pct", "r_multiple", "exit_reason"
                ])
                
                for t in result.trades:
                    writer.writerow([
                        t.symbol, t.side, t.qty, f"{t.entry_price:.2f}", f"{t.exit_price:.2f}",
                        t.entry_time, t.exit_time, f"{t.pnl:.2f}", f"{t.pnl_pct:.2f}",
                        f"{t.r_multiple:.2f}", t.exit_reason
                    ])
        
        print(f"\nResults saved to: {summary_file}")
        return str(summary_file)


def parse_timeframe(tf_str: str) -> TimeFrame:
    """Parse timeframe string like '5Min', '1Hour', '1Day'."""
    import re
    
    match = re.match(r"(\d+)(\w+)", tf_str)
    if not match:
        raise ValueError(f"Invalid timeframe format: {tf_str}")
    
    value = int(match.group(1))
    unit = match.group(2).lower()
    
    return get_timeframe(value, unit)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Backtest trading strategy")
    parser.add_argument("--symbol", required=True, help="Stock symbol to backtest")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", default="5Min", help="Bar timeframe (e.g., 1Min, 5Min, 1Hour)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage percentage")
    
    args = parser.parse_args()
    
    # Parse dates
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    timeframe = parse_timeframe(args.timeframe)
    
    # Run backtest
    backtester = Backtester(
        initial_capital=args.capital,
        slippage_pct=args.slippage,
    )
    
    result = backtester.run(
        symbol=args.symbol.upper(),
        start=start,
        end=end,
        timeframe=timeframe,
    )
    
    # Print and save results
    backtester.print_results(result)
    backtester.save_results(result)


if __name__ == "__main__":
    main()
