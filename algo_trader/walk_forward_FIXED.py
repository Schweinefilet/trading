"""
FIXED Walk-Forward Analysis with ACTUAL Parameter Optimization

This replaces your broken walk_forward.py that just returns defaults.
"""
import itertools
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd

from config.settings import config
from backtest.engine import BacktestEngine


def optimize_parameters(
    bar_data: Dict[str, pd.DataFrame],
    spy_daily: pd.DataFrame,
    start: datetime,
    end: datetime,
    capital: float,
    verbose: bool = False
) -> Tuple[Dict, float]:
    """
    ACTUALLY optimize parameters via grid search.
    
    Returns:
        (best_params, best_sharpe)
    """
    # Define parameter grid - KEEP IT SMALL to avoid overfitting
    param_grid = {
        'RSI_PERIOD': [10, 14, 20],
        'RSI_MOMENTUM_LONG': [45, 50, 55, 60],  # Lower thresholds
        'ADX_TREND_THRESHOLD': [15, 20, 25],
        'ATR_STOP_MULTIPLIER': [2.0, 2.5, 3.0, 3.5],  # Wider stops
        'VOLUME_MULTIPLIER': [0.8, 1.0, 1.5],
    }
    
    best_sharpe = -999
    best_params = None
    test_count = 0
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    total_combos = 1
    for v in values:
        total_combos *= len(v)
    
    print(f"    Testing {total_combos} parameter combinations...")
    
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        
        # Apply params to config
        original_vals = {}
        for key, val in params.items():
            original_vals[key] = getattr(config, key)
            setattr(config, key, val)
        
        try:
            # Run backtest on in-sample period
            engine = BacktestEngine()
            results = engine.run(
                bar_data, spy_daily,
                start=start, end=end,
                capital=capital,
                verbose=False
            )
            
            sharpe = results['metrics'].get('sharpe_ratio', -999)
            trades = results['metrics'].get('total_trades', 0)
            
            # Require minimum trades to avoid lucky flukes
            if trades < 10:
                sharpe = -999
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params.copy()
                
        except Exception as e:
            if verbose:
                print(f"    Error with params {params}: {e}")
        finally:
            # Restore original config
            for key, val in original_vals.items():
                setattr(config, key, val)
        
        test_count += 1
        if test_count % 50 == 0:
            print(f"    Progress: {test_count}/{total_combos} ({test_count/total_combos*100:.1f}%)")
    
    if best_params is None:
        # Fallback to defaults if all failed
        best_params = {key: getattr(config, key) for key in param_grid.keys()}
        best_sharpe = 0.0
    
    return best_params, best_sharpe


def walk_forward_analysis(
    bar_data: Dict[str, pd.DataFrame],
    spy_daily: pd.DataFrame,
    capital: float = 10000,
    verbose: bool = False,
    in_sample_months: int = 3,
    out_sample_months: int = 1,
) -> List[Dict]:
    """
    Proper walk-forward with rolling windows.
    
    Each window:
    1. Optimize on in-sample period
    2. Test on out-of-sample period
    3. Record WFE (Walk-Forward Efficiency)
    """
    results = []
    
    # Find earliest and latest dates across all symbols
    all_dates = []
    for df in bar_data.values():
        if not df.empty:
            all_dates.extend(df.index.tolist())
    
    if not all_dates:
        print("ERROR: No data available")
        return results
    
    start_date = min(all_dates)
    end_date = max(all_dates)
    
    current = start_date
    window_num = 1
    
    while current < end_date:
        # Define IS (in-sample) period
        is_start = current
        is_end = current + timedelta(days=in_sample_months * 30)
        
        # Define OOS (out-of-sample) period
        oos_start = is_end
        oos_end = is_end + timedelta(days=out_sample_months * 30)
        
        if oos_end > end_date:
            break
        
        print(f"\n  [WalkForward] Window {window_num}: IS={is_start.date()} to {is_end.date()} | OOS={oos_start.date()} to {oos_end.date()}")
        
        # Step 1: Optimize on in-sample
        best_params, is_sharpe = optimize_parameters(
            bar_data, spy_daily,
            is_start, is_end,
            capital, verbose
        )
        
        # Step 2: Apply best params and test on out-of-sample
        original_vals = {}
        for key, val in best_params.items():
            original_vals[key] = getattr(config, key)
            setattr(config, key, val)
        
        try:
            engine = BacktestEngine()
            oos_results = engine.run(
                bar_data, spy_daily,
                start=oos_start, end=oos_end,
                capital=capital,
                verbose=False
            )
            
            oos_sharpe = oos_results['metrics'].get('sharpe_ratio', 0)
            oos_return = oos_results['metrics'].get('total_return_pct', 0)
            oos_trades = oos_results['metrics'].get('total_trades', 0)
            
            # Calculate Walk-Forward Efficiency
            wfe = oos_sharpe / is_sharpe if is_sharpe > 0 else 0
            
            flag = "PASS" if wfe > 0.5 and oos_trades >= 5 else "FAIL"
            
            print(f"    IS Sharpe={is_sharpe:.2f} | OOS Sharpe={oos_sharpe:.2f} | OOS Return={oos_return:.2f}% | WFE={wfe:.2f} | Trades={oos_trades} | {flag}")
            
            results.append({
                'window': window_num,
                'is_start': is_start,
                'is_end': is_end,
                'oos_start': oos_start,
                'oos_end': oos_end,
                'best_params': best_params,
                'is_sharpe': is_sharpe,
                'oos_sharpe': oos_sharpe,
                'oos_return_pct': oos_return,
                'oos_trades': oos_trades,
                'wfe': wfe,
                'flag': flag,
            })
            
        finally:
            # Restore config
            for key, val in original_vals.items():
                setattr(config, key, val)
        
        # Move to next window
        current = oos_start
        window_num += 1
    
    return results


def print_walk_forward_report(results: List[Dict]):
    """Print summary of walk-forward analysis."""
    print("\n" + "="*70)
    print("  WALK-FORWARD ANALYSIS REPORT")
    print("="*70)
    
    if not results:
        print("  No results to display.")
        return
    
    print(f"\n   Win | IS Sharpe | OOS Sharpe | OOS Ret% |   WFE | Trades | Flag")
    print(f"  {'-'*4} | {'-'*9} | {'-'*10} | {'-'*8} | {'-'*5} | {'-'*6} | {'-'*4}")
    
    for r in results:
        print(f"  {r['window']:4d} | {r['is_sharpe']:9.2f} | {r['oos_sharpe']:10.2f} | {r['oos_return_pct']:7.2f}% | {r['wfe']:5.2f} | {r['oos_trades']:6d} | {r['flag']}")
    
    # Calculate averages
    avg_wfe = sum(r['wfe'] for r in results) / len(results)
    avg_oos_sharpe = sum(r['oos_sharpe'] for r in results) / len(results)
    fail_count = sum(1 for r in results if r['flag'] == 'FAIL')
    
    print(f"\n  {'-'*70}")
    print(f"  Avg OOS Sharpe:     {avg_oos_sharpe:.2f}")
    print(f"  Avg WFE:            {avg_wfe:.2f}")
    print(f"  Flagged Windows:    {fail_count} / {len(results)}")
    
    if avg_wfe < 0.5:
        print(f"  Strategy Robust:    NO (WFE < 0.5 indicates overfitting)")
    elif avg_oos_sharpe < 0.5:
        print(f"  Strategy Robust:    NO (Avg OOS Sharpe < 0.5)")
    else:
        print(f"  Strategy Robust:    YES")
    
    print("="*70)
