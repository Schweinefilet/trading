"""
Monthly performance report with benchmark comparisons.
Outputs TSV tables for direct copy-paste into Google Sheets.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import OrderedDict

import pandas as pd
import numpy as np


def compute_monthly_pnl(trades: list, starting_capital: float) -> Dict[str, dict]:
    """
    Compute monthly P&L and cumulative returns from trades.
    
    Returns:
        OrderedDict keyed by 'YYYY-MM' with:
            pnl: dollar P&L for that month
            pct: percentage return for that month (relative to starting equity of that month)
            cumulative_pct: cumulative return from start
    """
    if not trades:
        return {}

    # Group trades by exit month
    monthly_pnl = {}
    for t in trades:
        try:
            key = t.exit_time.strftime("%Y-%m")
        except Exception:
            continue
        monthly_pnl[key] = monthly_pnl.get(key, 0) + t.pnl

    if not monthly_pnl:
        return {}

    # Sort chronologically
    sorted_months = sorted(monthly_pnl.keys())
    
    result = OrderedDict()
    running_equity = starting_capital
    cumulative_return = 0.0
    
    for month in sorted_months:
        pnl = monthly_pnl[month]
        pct = (pnl / running_equity * 100) if running_equity > 0 else 0
        running_equity += pnl
        cumulative_return = ((running_equity - starting_capital) / starting_capital) * 100
        
        result[month] = {
            "pnl": pnl,
            "pct": pct,
            "cumulative_pct": cumulative_return,
        }
    
    return result


def compute_benchmark_monthly_returns(
    spy_daily: pd.DataFrame,
    start_date: str,
    end_date: str,
    bar_data: Dict[str, pd.DataFrame] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute monthly returns for benchmarks: S&P 500 (SPY), Dow (DIA), Nasdaq (QQQ).
    
    Returns:
        Dict keyed by benchmark name, each containing:
            Dict keyed by 'YYYY-MM' -> monthly return %
    """
    benchmarks = {}
    
    # SPY = S&P 500 proxy
    if spy_daily is not None and not spy_daily.empty:
        benchmarks["spy"] = _monthly_returns_from_df(spy_daily)
    
    # DIA = Dow proxy, QQQ = Nasdaq proxy
    if bar_data:
        for sym, label in [("DIA", "dia"), ("QQQ", "qqq")]:
            if sym in bar_data and not bar_data[sym].empty:
                df = bar_data[sym]
                # Resample to daily if intraday
                if hasattr(df.index, 'hour'):
                    daily = df.resample('D').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna(subset=['close'])
                else:
                    daily = df
                benchmarks[label] = _monthly_returns_from_df(daily)
    
    return benchmarks


def _monthly_returns_from_df(df: pd.DataFrame) -> Dict[str, float]:
    """Compute monthly returns from a daily price DataFrame."""
    if df.empty or "close" not in df.columns:
        return {}
    
    # Get last close of each month
    monthly_close = df["close"].resample("ME").last().dropna()
    
    if len(monthly_close) < 2:
        return {}
    
    returns = {}
    for i in range(1, len(monthly_close)):
        prev = monthly_close.iloc[i - 1]
        curr = monthly_close.iloc[i]
        if prev > 0:
            pct = ((curr - prev) / prev) * 100
            key = monthly_close.index[i].strftime("%Y-%m")
            returns[key] = pct
    
    return returns


def compute_benchmark_cumulative(monthly_returns: Dict[str, float]) -> Dict[str, float]:
    """Compute cumulative returns from monthly returns."""
    cumulative = {}
    running = 1.0
    for month in sorted(monthly_returns.keys()):
        running *= (1 + monthly_returns[month] / 100)
        cumulative[month] = (running - 1) * 100
    return cumulative


def print_monthly_tsv(
    trades: list,
    starting_capital: float,
    spy_daily: pd.DataFrame = None,
    bar_data: Dict[str, pd.DataFrame] = None,
):
    """
    Print two TSV tables:
    TABLE 1 - Monthly P&L with benchmarks
    TABLE 2 - Cumulative Returns with benchmarks
    """
    # Compute bot monthly data
    bot_monthly = compute_monthly_pnl(trades, starting_capital)
    if not bot_monthly:
        print("No monthly data available.")
        return
    
    # Compute benchmark data
    benchmarks = compute_benchmark_monthly_returns(spy_daily, None, None, bar_data)
    
    spy_monthly = benchmarks.get("spy", {})
    dia_monthly = benchmarks.get("dia", {})
    qqq_monthly = benchmarks.get("qqq", {})
    
    # Get all months from bot data
    all_months = sorted(bot_monthly.keys())
    
    # Filter benchmark months to only include months in the bot's data
    # so cumulative returns align with the backtest period
    bot_months_set = set(all_months)
    
    spy_filtered = {m: v for m, v in spy_monthly.items() if m in bot_months_set}
    dia_filtered = {m: v for m, v in dia_monthly.items() if m in bot_months_set}
    qqq_filtered = {m: v for m, v in qqq_monthly.items() if m in bot_months_set}
    
    # Compute cumulative for benchmarks (only over the backtest window)
    spy_cum = compute_benchmark_cumulative(spy_filtered)
    dia_cum = compute_benchmark_cumulative(dia_filtered)
    qqq_cum = compute_benchmark_cumulative(qqq_filtered)
    
    # === TABLE 1: Monthly P&L ===
    print("Month\tBOT P&L\tBOT %\tS&P 500 %\tDow %\tNasdaq %")
    for month in all_months:
        data = bot_monthly[month]
        pnl = data["pnl"]
        pct = data["pct"]
        
        # Format P&L
        if pnl >= 0:
            pnl_str = f"${pnl:,.2f}"
        else:
            pnl_str = f"-${abs(pnl):,.2f}"
        
        # Format bot %
        pct_str = f"{pct:.2f}%"
        
        # Benchmark %
        spy_str = f"{spy_monthly[month]:.2f}%" if month in spy_monthly else ""
        dia_str = f"{dia_monthly[month]:.2f}%" if month in dia_monthly else ""
        qqq_str = f"{qqq_monthly[month]:.2f}%" if month in qqq_monthly else ""
        
        print(f"{month}\t{pnl_str}\t{pct_str}\t{spy_str}\t{dia_str}\t{qqq_str}")
    
    print()  # Blank line between tables
    
    # === TABLE 2: Cumulative Returns ===
    print("Month\tYour Cumulative %\tS&P 500 Cumulative %\tDow Cumulative %\tNasdaq Cumulative %")
    for month in all_months:
        data = bot_monthly[month]
        bot_cum_str = f"{data['cumulative_pct']:.2f}%"
        spy_cum_str = f"{spy_cum[month]:.2f}%" if month in spy_cum else ""
        dia_cum_str = f"{dia_cum[month]:.2f}%" if month in dia_cum else ""
        qqq_cum_str = f"{qqq_cum[month]:.2f}%" if month in qqq_cum else ""
        
        print(f"{month}\t{bot_cum_str}\t{spy_cum_str}\t{dia_cum_str}\t{qqq_cum_str}")
