#!/usr/bin/env python3
"""
Strategy Optimizer - Grid Search Parameter Tuning
Backtests multiple parameter combinations over historical data to find optimal settings.
Supports multiprocessing for faster execution.

Usage:
    python optimize.py --tier 1           # Run Tier 1 (High Impact) optimization
    python optimize.py --months 1         # Optimize over past 1 month
    python optimize.py --apply            # Auto-apply best params to settings
"""

import argparse
import sys
import os
import itertools
import multiprocessing
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

sys.path.insert(0, os.path.dirname(__file__))

# We need to load dotenv in global scope for workers, but settings object
# must be updated per-worker.
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from config.settings import settings
from config.symbols import symbol_manager
from simulate_month import simulate_period, get_trading_days
from data.market_data import MarketDataClient
from alpaca.data.timeframe import TimeFrame

# Global cache for workers
_shared_data = None

def init_pool(data):
    """Initialize worker pool with shared data cache."""
    global _shared_data
    _shared_data = data


# --- Parameter Grids ---

TIER_1_GRID = {
    "MIN_SIGNALS_REQUIRED": [2],
    "MIN_CONFIDENCE_SCORE": [1.5, 2.0],
    "ATR_SL_MULTIPLIER": [1.5, 2.0, 2.5],
    "ATR_TP_MULTIPLIER": [3.0, 4.0, 5.0],    # Better Risk/Reward (2:1 or higher)
    "TRAILING_STOP_ACTIVATION": [0.015, 0.02],
    "TRAILING_STOP_PCT": [0.0075, 0.01],
    "MIN_HOLD_MINUTES": [15, 30],
}

TIER_2_GRID = {
    "RSI_PERIOD": [10, 14],
    "RSI_OVERSOLD": [25, 30, 35],
    "MIN_VOLUME_RATIO": [1.2, 1.5, 2.0],
    "TAKE_PROFIT_PCT": [0.03, 0.05],
    "STOP_LOSS_PCT": [0.015, 0.025],
    "PARTIAL_EXIT_PCT": [0.0, 0.5],
}

TIER_3_GRID = {
    "SPY_TREND_FILTER": [True, False],
    "VIX_MAX": [25.0, 30.0, 100.0],
    "PER_TRADE_RISK_PCT": [0.01, 0.015, 0.02],  # "RISK_PER_TRADE"
    "MAX_TRADES_PER_DAY": [5, 10, 20],         # "MAX_OPEN_POSITIONS" proxy/related
}


def apply_params(params: Dict[str, Any]):
    """Apply params to settings object."""
    for key, value in params.items():
        if hasattr(settings, key):
            setattr(settings, key, value)


def calculate_score(result: Dict) -> float:
    """Score a parameter combination using a Profit-Factor weighted approach."""
    if result is None or result["total_trades"] < 5:
        return -999.0
    
    pnl_pct = result["total_pnl_pct"]
    win_rate = result["win_rate"] / 100
    trades = result["total_trades"]
    
    # Calculate simple Profit Factor proxy
    # (Avg Win * Win Rate) / (Avg Loss * (1-Win Rate))
    # We'll use a simplified version: P&L per trade weight
    avg_pnl = result.get("avg_trade", 0)
    
    # Penalize large drawdowns heavily
    dd_penalty = 0
    if result["max_drawdown"] > 0 and result["starting_capital"] > 0:
        dd_pct = (result["max_drawdown"] / result["starting_capital"]) * 100
        dd_penalty = dd_pct * 1.5  # Increased penalty
    
    # Reward Expectancy: Average P&L per trade is key for robustness
    expectancy_bonus = avg_pnl * 0.5
    
    # Penalize low trade count (statistically insignificant)
    trade_weight = min(trades / 50, 1.0)
    
    score = (pnl_pct * trade_weight) + (win_rate * 10) + expectancy_bonus - dd_penalty
    return round(score, 2)


def run_simulation_wrapper(args):
    """
    Worker function for multiprocessing.
    args: (params, start_date, end_date, symbols, capital)
    """
    params, start_date, end_date, symbols, capital = args
    
    # Apply settings in this process
    apply_params(params)
    
    try:
        # Pass global shared data to the simulation
        result = simulate_period(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            capital=capital,
            verbose=False,
            quiet=True,
            data_cache=_shared_data
        )
        return params, result
    except Exception as e:
        return params, None


def optimize(
    tier: int = 1,
    months: float = 3,
    capital: float = 10000.0,
    symbols: List[str] = None,
    processes: int = 4,
):
    """Run grid search optimization with multiprocessing."""
    
    # Select grid
    if tier == 1:
        grid = TIER_1_GRID
        print(f"  [TIER 1] OPTIMIZING CORE PARAMETERS")
    elif tier == 2:
        grid = TIER_2_GRID
        print(f"  [TIER 2] OPTIMIZING REFINEMENT PARAMETERS")
    elif tier == 3:
        grid = TIER_3_GRID
        print(f"  [TIER 3] OPTIMIZING RISK PARAMETERS")
    else:
        print("  [ERROR] Invalid tier. Select 1, 2, or 3.")
        return None

    # Calculate date range
    end_date = datetime.now() - timedelta(days=1)
    while end_date.weekday() >= 5:
        end_date -= timedelta(days=1)
    start_date = end_date - timedelta(days=int(months * 30))
    
    if symbols is None:
        symbols = list(symbol_manager.STATIC_WATCHLIST)
    
    # Generate combinations
    keys = list(grid.keys())
    # Deduplicate and sort values to ensure unique combinations
    values = [sorted(list(set(v))) for v in grid.values()]
    combos = list(itertools.product(*values))
    
    print(f"  {'-'*60}")
    print(f"  Period: {start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}")
    print(f"  Combinations: {len(combos)}")
    print(f"  Processes:    {processes}")
    print(f"  {'-'*60}\n")
    
    # Prepare arguments for workers
    worker_args = []
    for combo in combos:
        params = dict(zip(keys, combo))
        worker_args.append((params, start_date, end_date, symbols, capital))
    
    results = []
    start_time = time.time()
    
    # --- Pre-fetch Market Data ---
    print(f"  [DATA] Pre-fetching market data for {len(symbols)} symbols...")
    data_client = MarketDataClient()
    
    # We need data from start_date - 10 days to end_date + 1 day for indicators Warmup
    fetch_start = start_date - timedelta(days=10)
    fetch_end = end_date + timedelta(days=1)
    
    # Include SPY and VIXY for context
    fetch_symbols = list(set(symbols + ["SPY", "VIXY"]))
    
    try:
        all_data = data_client.get_bars(
            symbols=fetch_symbols,
            start=fetch_start,
            end=fetch_end,
            timeframe=TimeFrame.Minute
        )
        print(f"  [OK] Data loaded: {sum(len(df) for df in all_data.values()):,} total bars")
    except Exception as e:
        print(f"  [ERROR] Failed to pre-fetch data: {e}")
        return None

    # Run multiprocessing pool
    with multiprocessing.Pool(processes=processes, initializer=init_pool, initargs=(all_data,)) as pool:
        # returns list of (params, result) tuples
        print(f"  [RUN] Starting {len(combos)} simulations...")
        
        # Use imap_unordered to track progress
        completed = 0
        if tqdm:
            iterator = tqdm(pool.imap_unordered(run_simulation_wrapper, worker_args), total=len(combos), unit="sim", desc="  Running")
        else:
            iterator = pool.imap_unordered(run_simulation_wrapper, worker_args)
            
        completed = 0
        for params_out, result_out in iterator:
            completed += 1
            if not tqdm:
                if completed % 10 == 0 or completed == len(combos):
                    pct = (completed / len(combos)) * 100
                    elapsed = time.time() - start_time
                    print(f"  [{completed}/{len(combos)}] {pct:.1f}% complete ({elapsed:.1f}s)")
            
            if result_out:
                score = calculate_score(result_out)
                results.append({
                    "params": params_out,
                    "result": result_out,
                    "score": score,
                })

    total_time = time.time() - start_time
    print(f"\n  [OK] Optimization complete in {total_time:.1f}s\n")
    
    if not results:
        print("  [ERROR] No valid results.")
        return None
    
    # Sort and Display
    results.sort(key=lambda x: x["score"], reverse=True)
    
    print(f"{'='*80}")
    print(f"  TOP 5 CONFIGURATIONS (TIER {tier})")
    print(f"{'='*80}")
    print(f"  {'Rank':<5} {'Score':>6} {'P&L %':>8} {'WR %':>6} {'Trades':>7} {'MaxDD':>9}")
    print(f"  {'-'*55}")
    
    for rank, r in enumerate(results[:5], 1):
        res = r["result"]
        print(f"  #{rank:<4} {r['score']:>6.2f} {res['total_pnl_pct']:>8.2f}% {res['win_rate']:>6.0f}% {res['total_trades']:>7} ${res['max_drawdown']:>8.0f}")
        # Print diff from default/current settings??
        # Just print params
        params_str = " | ".join(f"{k}={v}" for k, v in r["params"].items())
        print(f"   |-- {params_str}")
        print()
        
    return results


def apply_best_to_env(params: Dict[str, Any]):
    """Update .env file with the best parameters."""
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    
    # Read current lines
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Map keys to update
    params_to_write = params.copy()
    
    new_lines = []
    seen_keys = set()
    
    for line in lines:
        if "=" in line:
            key = line.split("=")[0].strip()
            if key in params_to_write:
                new_lines.append(f"{key}={params_to_write[key]}\n")
                seen_keys.add(key)
                continue
        new_lines.append(line)
    
    # Append missing
    for k, v in params_to_write.items():
        if k not in seen_keys:
            new_lines.append(f"{k}={v}\n")
            
    with open(env_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"  [SAVE] Saved best parameters to .env")


def main():
    # Fix for Windows multiprocessing
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="Strategy parameter optimizer")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=1, help="Optimization tier (1=Core, 2=Refine, 3=Risk)")
    parser.add_argument("--months", type=float, help="Months of history", default=1)
    parser.add_argument("--capital", type=float, default=settings.INITIAL_CAPITAL)
    parser.add_argument("--processes", type=int, default=os.cpu_count() or 4, help="Number of parallel processes")
    parser.add_argument("--apply", action="store_true", help="Auto-apply best params to .env")
    
    args = parser.parse_args()
    
    # Warn if Tier 1 has too many combos
    if args.tier == 1:
        # check combo count
        combo_count = 1
        for v in TIER_1_GRID.values():
            combo_count *= len(v)
        if combo_count > 500:
            print(f"  ⚠ WARNING: Tier 1 has {combo_count} combinations.")
            print(f"  This may take a long time. Consider reducing grid or months.")
            # We proceeded anyway, but maybe user wants to know.

    results = optimize(
        tier=args.tier,
        months=args.months,
        capital=args.capital,
        processes=args.processes,
    )
    
    if results and args.apply:
        best_params = results[0]["params"]
        apply_best_to_env(best_params)


if __name__ == "__main__":
    main()
