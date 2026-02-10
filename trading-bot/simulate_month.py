#!/usr/bin/env python3
"""
Multi-Day Continuous Simulation
Runs the strategy across multiple trading days with rolling capital.

Usage:
    python simulate_month.py                                        # Past month
    python simulate_month.py --start 2026-01-06 --end 2026-02-06   # Custom range
    python simulate_month.py --days 5                               # Last N trading days
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from config.settings import settings
from config.symbols import symbol_manager
from simulate_day import simulate_day


def get_trading_days(start: datetime, end: datetime) -> List[datetime]:
    """Generate list of weekdays (trading days) between start and end."""
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Mon-Fri
            days.append(current)
        current += timedelta(days=1)
    return days


def simulate_period(
    start_date: datetime,
    end_date: datetime,
    symbols: List[str],
    capital: float = 10000.0,
    verbose: bool = False,
    quiet: bool = False,
    data_cache: Dict[str, pd.DataFrame] = None,
):
    """
    Run simulation across multiple trading days with rolling capital.
    Each day's ending capital becomes the next day's starting capital.
    """
    trading_days = get_trading_days(start_date, end_date)
    
    if not trading_days:
        if not quiet:
            print("  ❌ No trading days in the specified range.")
        return
    
    if not quiet:
        print(f"\n{'='*80}")
        print(f"  MULTI-DAY SIMULATION")
        print(f"  Period: {start_date.strftime('%b %d, %Y')} → {end_date.strftime('%b %d, %Y')}")
        print(f"  Trading Days: {len(trading_days)}  |  Starting Capital: ${capital:,.2f}")
        print(f"  Symbols: {len(symbols)}  |  Strategy: SwingStrategy_v2")
        print(f"{'='*80}\n")
    
    # Rolling state
    current_capital = capital
    all_trades = []
    daily_results = []
    total_max_dd = 0
    
    for i, day in enumerate(trading_days):
        day_label = day.strftime("%a %b %d")
        if not quiet:
            print(f"  [{i+1}/{len(trading_days)}] {day_label} | Capital: ${current_capital:,.2f} ... ", end="", flush=True)
        
        try:
            result = simulate_day(
                date=day,
                symbols=symbols,
                capital=current_capital,
                verbose=verbose,
                data_cache=data_cache,
            )
            
            if result is None:
                if not quiet:
                    print("⚠ No data / market closed")
                daily_results.append({
                    "date": day,
                    "trades": 0,
                    "win_rate": 0,
                    "pnl": 0,
                    "pnl_pct": 0,
                    "capital": current_capital,
                })
                continue
            
            # Roll capital forward
            current_capital = result["ending_capital"]
            all_trades.extend(result["trades"])
            
            if result["max_drawdown"] > total_max_dd:
                total_max_dd = result["max_drawdown"]
            
            # Emoji for P&L
            if result["pnl"] > 0:
                emoji = "✅"
            elif result["pnl"] < 0:
                emoji = "❌"
            else:
                emoji = "➖"
            
            if not quiet:
                print(f"{emoji} {result['total_trades']} trades | WR: {result['win_rate']:.0f}% | P&L: ${result['pnl']:+,.2f} ({result['pnl_pct']:+.2f}%) | Capital: ${current_capital:,.2f}")
            
            daily_results.append({
                "date": day,
                "trades": result["total_trades"],
                "win_rate": result["win_rate"],
                "pnl": result["pnl"],
                "pnl_pct": result["pnl_pct"],
                "capital": current_capital,
            })
            
        except Exception as e:
            if not quiet:
                print(f"⚠ Error: {e}")
            daily_results.append({
                "date": day,
                "trades": 0,
                "win_rate": 0,
                "pnl": 0,
                "pnl_pct": 0,
                "capital": current_capital,
            })
    
    # --- Summary ---
    total_pnl = current_capital - capital
    total_pnl_pct = (total_pnl / capital) * 100
    total_trades = len(all_trades)
    winning_trades = [t for t in all_trades if t["pnl"] > 0]
    losing_trades = [t for t in all_trades if t["pnl"] <= 0]
    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
    
    trading_days_with_trades = [d for d in daily_results if d["trades"] > 0]
    profitable_days = [d for d in daily_results if d["pnl"] > 0]
    losing_days = [d for d in daily_results if d["pnl"] < 0]
    
    avg_win = sum(t["pnl"] for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t["pnl"] for t in losing_trades) / len(losing_trades) if losing_trades else 0
    
    best_day = max(daily_results, key=lambda d: d["pnl"]) if daily_results else None
    worst_day = min(daily_results, key=lambda d: d["pnl"]) if daily_results else None
    
    if not quiet:
        print(f"\n{'='*80}")
        print(f"  PERIOD SUMMARY")
        print(f"{'='*80}")
        print(f"  Starting Capital:   ${capital:,.2f}")
        print(f"  Ending Capital:     ${current_capital:,.2f}")
        print(f"  Total P&L:          ${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)")
        print(f"  Max Drawdown:       ${total_max_dd:,.2f}")
        print(f"{'─'*80}")
        print(f"  Total Trades:       {total_trades}")
        print(f"  Win Rate:           {win_rate:.1f}%  ({len(winning_trades)}W / {len(losing_trades)}L)")
        print(f"  Avg Winner:         ${avg_win:+,.2f}")
        print(f"  Avg Loser:          ${avg_loss:+,.2f}")
        print(f"{'─'*80}")
        print(f"  Trading Days:       {len(trading_days_with_trades)}")
        print(f"  Profitable Days:    {len(profitable_days)}")
        print(f"  Losing Days:        {len(losing_days)}")
        if best_day:
            print(f"  Best Day:           {best_day['date'].strftime('%b %d')} (${best_day['pnl']:+,.2f})")
        if worst_day:
            print(f"  Worst Day:          {worst_day['date'].strftime('%b %d')} (${worst_day['pnl']:+,.2f})")
        print(f"{'='*80}\n")
    
    return {
        "starting_capital": capital,
        "ending_capital": current_capital,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "max_drawdown": total_max_dd,
        "daily_results": daily_results,
        "all_trades": all_trades,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-day trading simulation")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--days", type=int, help="Last N trading days", default=None)
    parser.add_argument("--capital", type=float, help="Starting capital", default=settings.INITIAL_CAPITAL)
    parser.add_argument("--symbols", type=str, help="Comma-sep symbols", default=None)
    parser.add_argument("--verbose", action="store_true", help="Show individual trade logs")
    
    args = parser.parse_args()
    
    # Determine date range
    if args.days:
        end_date = datetime.now() - timedelta(days=1)
        while end_date.weekday() >= 5:
            end_date -= timedelta(days=1)
        # Go back N trading days
        start_date = end_date
        days_counted = 0
        while days_counted < args.days:
            start_date -= timedelta(days=1)
            if start_date.weekday() < 5:
                days_counted += 1
    elif args.start and args.end:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    elif args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.now() - timedelta(days=1)
        while end_date.weekday() >= 5:
            end_date -= timedelta(days=1)
    else:
        # Default: past month
        end_date = datetime.now() - timedelta(days=1)
        while end_date.weekday() >= 5:
            end_date -= timedelta(days=1)
        start_date = end_date - timedelta(days=30)
    
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = list(symbol_manager.STATIC_WATCHLIST)
    
    simulate_period(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        capital=args.capital,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
