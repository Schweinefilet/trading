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
from backtest.monthly_report import print_monthly_tsv


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
    parser.add_argument("--monthly-deposit", type=float, default=0.0, help="Simulate monthly account funding")
    args = parser.parse_args()

    if args.swing:
        config.HOLD_OVERNIGHT = True
    if args.no_regime:
        config.USE_REGIME_FILTER = False
    
    # Sync config with CLI overrides so the printout below is accurate
    config.BACKTEST_START = args.start
    config.BACKTEST_END = args.end
    config.STARTING_CAPITAL = args.capital

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
    # Sensitive keys to redact from logs
    _sensitive_keys = {
        "ALPACA_API_KEY", "ALPACA_SECRET_KEY",
        "DISCORD_WEBHOOK_URL", "EMAIL_PASS",
    }
    # Filter out methods, constants, and built-ins to show only parameters
    for key, value in config.__dict__.items():
        if not key.startswith("_") and not callable(value):
            if key in _sensitive_keys:
                print(f"    {key:<30}: ****")
            else:
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

    # Fetch SPY daily for regime detection + benchmark data
    print("Step 3: Fetching benchmark daily data (SPY, DIA, QQQ)...")
    spy_start = start - __import__('datetime').timedelta(days=300)  # Need 200-day SMA warm-up
    spy_tf = parse_timeframe("1Day")
    
    # Fetch all benchmarks in one batch
    benchmark_daily = client.fetch_bars_multi(["SPY", "DIA", "QQQ"], spy_tf, spy_start, end)
    spy_daily = benchmark_daily.get("SPY", __import__('pandas').DataFrame())
    
    if not spy_daily.empty:
        spy_daily = compute_regime_indicators(spy_daily)
        print(f"  SPY: {len(spy_daily)} daily bars with regime indicators.")
    else:
        print("  WARNING: No SPY data. Regime filter will default to BULLISH.")
        spy_daily = __import__('pandas').DataFrame()
    
    dia_daily = benchmark_daily.get("DIA", __import__('pandas').DataFrame())
    qqq_daily = benchmark_daily.get("QQQ", __import__('pandas').DataFrame())
    print(f"  DIA: {len(dia_daily)} bars, QQQ: {len(qqq_daily)} bars\n")

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
            monthly_deposit=args.monthly_deposit,
            verbose=args.verbose,
        )

        print_metrics(results["metrics"])

        # Monthly TSV report with benchmarks
        print("\n" + "=" * 60)
        print("  MONTHLY PERFORMANCE (TSV - copy into Google Sheets)")
        print("=" * 60 + "\n")
        
        # Build benchmark bar_data dict for monthly_report
        bench_data = {}
        if not dia_daily.empty:
            bench_data["DIA"] = dia_daily
        if not qqq_daily.empty:
            bench_data["QQQ"] = qqq_daily
        
        print_monthly_tsv(
            trades=results["trades"],
            starting_capital=args.capital,
            spy_daily=spy_daily,
            bar_data=bench_data,
        )

    print("\nBacktest complete.")


if __name__ == "__main__":
    main()
