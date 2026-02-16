"""
Backtest runner entry point.
Fetches historical data, computes indicators, and runs the backtest engine.

Usage:
    python backtest_runner.py [--walk-forward] [--verbose]
"""
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import config
from config.tickers import tickers
from data.historical import HistoricalDataClient, parse_timeframe
from data.indicators import compute_indicators, compute_regime_indicators
from backtest.engine import BacktestEngine
from backtest.metrics import print_metrics
from backtest.walk_forward import walk_forward_analysis, print_walk_forward_report


def main():
    parser = argparse.ArgumentParser(description="Algo Trader Backtester")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward analysis")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print trade-by-trade details")
    parser.add_argument("--start", default=config.BACKTEST_START, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=config.BACKTEST_END, help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=config.STARTING_CAPITAL, help="Starting capital")
    parser.add_argument("--tickers", nargs="+", default=None, help="Override tickers (space-separated)")
    parser.add_argument("--swing", action="store_true", help="Enable Swing Trading (bold overnight)")
    parser.add_argument("--no-regime", action="store_true", help="Disable Regime Filter (trade anytime)")
    args = parser.parse_args()

    if args.swing:
        config.HOLD_OVERNIGHT = True
    if args.no_regime:
        config.USE_REGIME_FILTER = False

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    symbols = args.tickers or tickers.all_symbols

    print(f"\n{'=' * 60}")
    print(f"  ALGO TRADER BACKTEST")
    print(f"  Period: {args.start} to {args.end}")
    print(f"  Capital: ${args.capital:,.2f}")
    print(f"  Symbols: {len(symbols)} tickers")
    print(f"  Mode: {'Walk-Forward' if args.walk_forward else 'Standard'}")
    print(f"{'=' * 60}")
    
    # Print all parameters (User requirement)
    print("\n  CONFIGURATION SETTINGS:")
    # Filter out methods, constants, and built-ins to show only parameters
    for key, value in config.__dict__.items():
        if not key.startswith("_") and not callable(value):
            print(f"    {key:<30}: {value}")
    print(f"{'=' * 60}\n")

    # Fetch historical data
    print("Step 1: Fetching historical data...")
    client = HistoricalDataClient()
    tf = parse_timeframe(config.PRIMARY_TIMEFRAME)

    bar_data = client.fetch_bars_multi(symbols, tf, start, end)
    print(f"  Loaded {len(bar_data)} symbols with data.\n")

    if not bar_data:
        print("  ERROR: No data fetched. Check API keys and date range.")
        sys.exit(1)

    # Compute indicators
    print("Step 2: Computing indicators...")
    for sym in bar_data:
        bar_data[sym] = compute_indicators(bar_data[sym])
        if args.verbose:
            print(f"  {sym}: {len(bar_data[sym])} bars, indicators computed")
    print()

    # Fetch SPY daily for regime detection
    print("Step 3: Fetching SPY daily data for regime filter...")
    spy_start = start - __import__('datetime').timedelta(days=300)  # Need 200-day SMA warm-up
    spy_tf = parse_timeframe("1Day")
    spy_daily = client.fetch_bars("SPY", spy_tf, spy_start, end)
    if not spy_daily.empty:
        spy_daily = compute_regime_indicators(spy_daily)
        print(f"  SPY: {len(spy_daily)} daily bars with regime indicators.\n")
    else:
        print("  WARNING: No SPY data. Regime filter will default to BULLISH.\n")
        spy_daily = __import__('pandas').DataFrame()

    if args.walk_forward:
        # Walk-forward analysis
        print("Step 4: Running Walk-Forward Analysis...")
        results = walk_forward_analysis(
            bar_data, spy_daily,
            capital=args.capital,
            verbose=args.verbose,
        )
        print_walk_forward_report(results)

    else:
        # Standard backtest
        print("Step 4: Running backtest...")
        engine = BacktestEngine()
        results = engine.run(
            bar_data, spy_daily,
            start=start, end=end,
            capital=args.capital,
            verbose=args.verbose,
        )

        print_metrics(results["metrics"])

        # Print ALL trades
        trades = results["trades"]
        if trades and args.verbose:
            print(f"\n  All Trades ({len(trades)}):")
            print(f"  {'Date':<19s} {'Symbol':<6s} {'Dir':<6s} {'Entry$':>8s} {'Exit$':>8s} {'Shares':>6s} {'Amt$':>9s} {'PnL':>10s} {'Reason':<15s}")
            print(f"  {'-' * 19} {'-' * 6} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 6} {'-' * 9} {'-' * 10} {'-' * 15}")
            for t in trades:
                sign = "+" if t.pnl >= 0 else ""
                entry_amt = t.entry_price * t.shares
                print(f"  {t.entry_time.strftime('%Y-%m-%d %H:%M'):<19s} {t.symbol:<6s} {t.direction:<6s} ${t.entry_price:>7.2f} ${t.exit_price:>7.2f} "
                      f"{t.shares:>6d} ${entry_amt:>8.2f} {sign}${t.pnl:>9.2f} {t.exit_reason:<15s}")

    print("\nBacktest complete.")


if __name__ == "__main__":
    main()
