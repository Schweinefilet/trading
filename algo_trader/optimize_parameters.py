"""
AUTOMATED PARAMETER OPTIMIZATION SCRIPT
=======================================
Multi-process grid search with progress tracking and centralized results.

Usage:
    python optimize_parameters.py --cores 11 --output optimization_results.csv

Features:
    - Tests thousands of parameter combinations
    - Parallel execution across CPU cores
    - Real-time progress bar
    - Exports all results to CSV for analysis
    - Flags configurations meeting objectives
"""

import itertools
import multiprocessing as mp
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from tqdm import tqdm

# Import your backtesting components
from config.settings import config
from config.tickers import tickers
from data.fetcher import fetch_bars_bulk
from data.indicators import compute_indicators, compute_regime_indicators
from backtest.engine import BacktestEngine


@dataclass
class OptimizationConfig:
    """Single parameter configuration to test."""
    # Entry Signal Parameters
    adx_threshold: float = 30.0
    confirmations_bullish: int = 3
    confirmations_cautious: int = 3
    confirmations_bearish: int = 3
    volume_multiplier: float = 1.2
    rsi_momentum_long: float = 50.0
    elite_selection_top: int = 20
    
    # Exit Parameters
    atr_stop_multiplier: float = 3.0
    partial_exit_atr_trigger: float = 3.5
    partial_exit_pct_bullish: float = 0.10
    partial_exit_pct_default: float = 0.20
    break_even_activate_atr: float = 1.2
    max_hold_hours: int = 48
    target_mult_elite: float = 25.0
    target_mult_runners: float = 15.0
    
    # Risk Management
    risk_per_trade_pct: float = 0.067
    max_position_pct: float = 0.60
    volatility_gate_mult: float = 1.5
    
    # Indicator Settings
    rsi_period: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    atr_period: int = 14
    
    # Regime & Time Filters
    crisis_vix_threshold: float = 30.0
    cautious_vix_threshold: float = 25.0
    entry_blackout_open_mins: int = 5
    entry_blackout_close_mins: int = 15
    skip_tuesday: bool = True
    skip_monday: bool = False
    skip_friday: bool = False
    
    # Configuration ID
    config_id: str = ""


@dataclass
class OptimizationResult:
    """Results from a single backtest run."""
    config_id: str
    
    # Primary Objectives
    win_rate: float
    total_return_pct: float
    
    # Constraints
    total_trades: int
    avg_win: float
    sharpe_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    
    # Diagnostic Metrics
    avg_loss: float
    largest_win: float
    largest_loss: float
    annualized_return: float
    sortino_ratio: float
    expectancy: float
    
    # Success Flags
    meets_objectives: bool
    meets_constraints: bool
    
    # All parameters (for export)
    parameters: Dict[str, Any] = None


def generate_parameter_grid() -> List[OptimizationConfig]:
    """
    Generate comprehensive parameter grid for testing.
    Returns list of OptimizationConfig objects.
    """
    
    # Define parameter ranges with fine granularity
    param_grid = {
        # HIGH PRIORITY: Entry Signal Tightening
        'adx_threshold': [25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0],
        'confirmations': [
            (3, 3, 3),  # Baseline
            (3, 4, 4),  # Stricter in uncertain regimes
            (4, 4, 4),  # Strict everywhere
            (3, 4, 5),  # Very defensive
            (4, 4, 5),  # Maximum selectivity
            (4, 5, 5),  # Ultra-selective
        ],
        'volume_multiplier': [1.0, 1.2, 1.3, 1.5, 1.7, 2.0, 2.3, 2.5],
        'rsi_momentum_long': [48.0, 50.0, 52.0, 55.0, 58.0, 60.0],
        'elite_selection_top': [8, 10, 12, 15, 18, 20, 25],
        
        # HIGH PRIORITY: Exit Optimization
        'atr_stop_multiplier': [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        'partial_exit_atr_trigger': [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
        'partial_exit_pct': [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        'break_even_activate_atr': [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
        'max_hold_hours': [24, 36, 48, 60, 72, 96],
        'target_multipliers': [
            (20.0, 12.0),  # Conservative
            (25.0, 15.0),  # Baseline
            (30.0, 18.0),  # Aggressive
            (20.0, 15.0),  # Balanced
            (30.0, 20.0),  # Very aggressive
        ],
        
        # MEDIUM PRIORITY: Risk Management
        'risk_per_trade_pct': [0.04, 0.05, 0.06, 0.067, 0.075, 0.08, 0.09],
        'max_position_pct': [0.40, 0.50, 0.60, 0.70, 0.80],
        'volatility_gate_mult': [1.2, 1.3, 1.5, 1.8, 2.0, 2.5],
        
        # LOW PRIORITY: Indicator Fine-tuning
        'rsi_period': [10, 12, 14, 16, 18],
        'ema_pairs': [(8, 21), (9, 21), (10, 21), (9, 26), (12, 26)],
        'atr_period': [10, 14, 20],
        
        # MEDIUM PRIORITY: Regime Filters
        'crisis_vix_threshold': [25.0, 30.0, 35.0, 40.0],
        'cautious_vix_threshold': [20.0, 25.0, 28.0, 30.0],
        
        # Time Filters
        'entry_blackout_open_mins': [5, 10, 15, 20],
        'entry_blackout_close_mins': [5, 10, 15, 20, 30],
        'day_filters': [
            (True, False, False),   # Skip Tuesday only (baseline)
            (False, False, False),  # No skip
            (True, True, False),    # Skip Mon+Tue
            (False, False, True),   # Skip Friday only
            (True, False, True),    # Skip Tue+Fri
        ],
    }
    
    # Generate all combinations (with intelligent pruning to keep it manageable)
    # Strategy: Test in phases
    
    configs = []
    config_counter = 0
    
    # Phase 1: Test entry filters exhaustively (most impact on win rate)
    print("Generating Phase 1: Entry Filter Combinations...")
    for adx in param_grid['adx_threshold']:
        for conf in param_grid['confirmations']:
            for vol_mult in param_grid['volume_multiplier']:
                for rsi_mom in param_grid['rsi_momentum_long']:
                    for elite_top in param_grid['elite_selection_top']:
                        config_obj = OptimizationConfig(
                            adx_threshold=adx,
                            confirmations_bullish=conf[0],
                            confirmations_cautious=conf[1],
                            confirmations_bearish=conf[2],
                            volume_multiplier=vol_mult,
                            rsi_momentum_long=rsi_mom,
                            elite_selection_top=elite_top,
                            config_id=f"P1_{config_counter:06d}"
                        )
                        configs.append(config_obj)
                        config_counter += 1
    
    # Phase 2: Test exit optimization with baseline entry filters
    print("Generating Phase 2: Exit Optimization Combinations...")
    # baseline_entry = OptimizationConfig()  # Default values (not used, just conceptually)
    for stop_mult in param_grid['atr_stop_multiplier']:
        for partial_trigger in param_grid['partial_exit_atr_trigger']:
            for partial_pct in param_grid['partial_exit_pct']:
                for be_activate in param_grid['break_even_activate_atr']:
                    for max_hold in param_grid['max_hold_hours']:
                        for targets in param_grid['target_multipliers']:
                            config_obj = OptimizationConfig(
                                atr_stop_multiplier=stop_mult,
                                partial_exit_atr_trigger=partial_trigger,
                                partial_exit_pct_bullish=partial_pct,
                                partial_exit_pct_default=partial_pct,
                                break_even_activate_atr=be_activate,
                                max_hold_hours=max_hold,
                                target_mult_elite=targets[0],
                                target_mult_runners=targets[1],
                                config_id=f"P2_{config_counter:06d}"
                            )
                            configs.append(config_obj)
                            config_counter += 1
    
    # Phase 3: Risk & Regime combinations with baseline entry/exit
    print("Generating Phase 3: Risk & Regime Combinations...")
    for risk_pct in param_grid['risk_per_trade_pct']:
        for max_pos in param_grid['max_position_pct']:
            for vol_gate in param_grid['volatility_gate_mult']:
                for crisis_vix in param_grid['crisis_vix_threshold']:
                    for cautious_vix in param_grid['cautious_vix_threshold']:
                        config_obj = OptimizationConfig(
                            risk_per_trade_pct=risk_pct,
                            max_position_pct=max_pos,
                            volatility_gate_mult=vol_gate,
                            crisis_vix_threshold=crisis_vix,
                            cautious_vix_threshold=cautious_vix,
                            config_id=f"P3_{config_counter:06d}"
                        )
                        configs.append(config_obj)
                        config_counter += 1
    
    # Phase 4: Indicator & Time Filter sweep
    print("Generating Phase 4: Indicator & Time Filter Combinations...")
    for rsi_p in param_grid['rsi_period']:
        for ema_pair in param_grid['ema_pairs']:
            for atr_p in param_grid['atr_period']:
                for bo_open in param_grid['entry_blackout_open_mins']:
                    for bo_close in param_grid['entry_blackout_close_mins']:
                        for day_filt in param_grid['day_filters']:
                            config_obj = OptimizationConfig(
                                rsi_period=rsi_p,
                                ema_fast=ema_pair[0],
                                ema_slow=ema_pair[1],
                                atr_period=atr_p,
                                entry_blackout_open_mins=bo_open,
                                entry_blackout_close_mins=bo_close,
                                skip_tuesday=day_filt[0],
                                skip_monday=day_filt[1],
                                skip_friday=day_filt[2],
                                config_id=f"P4_{config_counter:06d}"
                            )
                            configs.append(config_obj)
                            config_counter += 1
    
    print(f"\nTotal configurations generated: {len(configs):,}")
    return configs


def apply_config_to_settings(opt_config: OptimizationConfig):
    """Apply optimization config to global TradingConfig instance."""
    
    # Entry signals
    config.ADX_TREND_THRESHOLD = opt_config.adx_threshold
    config.CONFIRMATIONS_BULLISH = opt_config.confirmations_bullish
    config.CONFIRMATIONS_CAUTIOUS = opt_config.confirmations_cautious
    config.CONFIRMATIONS_BEARISH = opt_config.confirmations_bearish
    config.VOLUME_MULTIPLIER = opt_config.volume_multiplier
    config.RSI_MOMENTUM_LONG = opt_config.rsi_momentum_long
    
    # New Parameters: Elite Selection & Tiered Targets
    config.ELITE_SELECTION_TOP = opt_config.elite_selection_top
    config.TARGET_MULT_ELITE = opt_config.target_mult_elite
    config.TARGET_MULT_RUNNERS = opt_config.target_mult_runners
    
    # Exits
    config.ATR_STOP_MULTIPLIER = opt_config.atr_stop_multiplier
    config.PARTIAL_EXIT_ATR_TRIGGER = opt_config.partial_exit_atr_trigger
    config.PARTIAL_EXIT_PCT_BULLISH = opt_config.partial_exit_pct_bullish
    config.PARTIAL_EXIT_PCT_DEFAULT = opt_config.partial_exit_pct_default
    config.BREAK_EVEN_ACTIVATE_ATR = opt_config.break_even_activate_atr
    config.MAX_HOLD_HOURS = opt_config.max_hold_hours
    
    # Risk
    config.RISK_PER_TRADE_PCT = opt_config.risk_per_trade_pct
    config.MAX_POSITION_PCT = opt_config.max_position_pct
    config.VOLATILITY_GATE_ATR_MULT = opt_config.volatility_gate_mult
    
    # Indicators
    config.RSI_PERIOD = opt_config.rsi_period
    config.EMA_FAST = opt_config.ema_fast
    config.EMA_SLOW = opt_config.ema_slow
    config.ATR_PERIOD = opt_config.atr_period
    
    # Time filters
    config.ENTRY_BLACKOUT_OPEN_MINS = opt_config.entry_blackout_open_mins
    config.ENTRY_BLACKOUT_CLOSE_MINS = opt_config.entry_blackout_close_mins
    
    # Day filters
    skip_days = []
    if opt_config.skip_monday:
        skip_days.append(0)
    if opt_config.skip_tuesday:
        skip_days.append(1)
    if opt_config.skip_friday:
        skip_days.append(4)
    config.ENTRY_SKIPPED_DAYS = skip_days


# Global variables for worker processes
_worker_bar_data = None
_worker_spy_daily = None

def init_worker(bar_data, spy_daily):
    """Initialize worker process with shared data."""
    global _worker_bar_data, _worker_spy_daily
    _worker_bar_data = bar_data
    _worker_spy_daily = spy_daily

def run_single_backtest(opt_config: OptimizationConfig) -> OptimizationResult:
    """
    Run a single backtest with given configuration.
    Uses globally initialized data to avoid pickling overhead on Windows.
    """
    # Access data from global scope (initialized once per worker)
    global _worker_bar_data, _worker_spy_daily
    
    # Fail gracefully if data missing (shouldn't happen if initialized correctly)
    if _worker_bar_data is None:
        return OptimizationResult(
            config_id=opt_config.config_id,
            win_rate=0, total_return_pct=0, total_trades=0, avg_win=0, sharpe_ratio=0,
            max_drawdown_pct=0, profit_factor=0, avg_loss=0, largest_win=0, largest_loss=0,
            annualized_return=0, sortino_ratio=0, expectancy=0,
            meets_objectives=False, meets_constraints=False,
            parameters={'error': 'Worker data not initialized'}
        )

    try:
        # Apply configuration
        apply_config_to_settings(opt_config)
        
        # Run backtest
        engine = BacktestEngine(config)
        results = engine.run(
            bar_data=_worker_bar_data,
            spy_daily=_worker_spy_daily,
            start=datetime(2016, 6, 6),
            end=datetime(2026, 2, 16),
            capital=25_000,
            verbose=False
        )
        
        metrics = results['metrics']
        
        # Extract key metrics
        win_rate = metrics.get('win_rate', 0)
        total_return_pct = metrics.get('total_return_pct', 0)
        total_trades = metrics.get('total_trades', 0)
        avg_win = metrics.get('avg_win', 0)
        avg_loss = metrics.get('avg_loss', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown_pct', 0)
        profit_factor = metrics.get('profit_factor', 0)
        largest_win = metrics.get('largest_win', 0)
        largest_loss = metrics.get('largest_loss', 0)
        annual_return = metrics.get('annualized_return', 0)
        sortino = metrics.get('sortino_ratio', 0)
        expectancy = metrics.get('expectancy', 0)
        
        # Check objectives
        meets_objectives = (win_rate >= 70.0 and total_return_pct >= 2147.71)
        
        # Check constraints
        meets_constraints = (
            total_trades >= 1200 and
            avg_win >= 1400 and
            sharpe >= 1.3 and
            max_dd <= 20.0 and
            profit_factor >= 1.5
        )
        
        result = OptimizationResult(
            config_id=opt_config.config_id,
            win_rate=win_rate,
            total_return_pct=total_return_pct,
            total_trades=total_trades,
            avg_win=avg_win,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            profit_factor=profit_factor,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            annualized_return=annual_return,
            sortino_ratio=sortino,
            expectancy=expectancy,
            meets_objectives=meets_objectives,
            meets_constraints=meets_constraints,
            parameters=asdict(opt_config)
        )
        
        return result
        
    except Exception as e:
        # Return error result
        return OptimizationResult(
            config_id=opt_config.config_id,
            win_rate=0,
            total_return_pct=0,
            total_trades=0,
            avg_win=0,
            sharpe_ratio=0,
            max_drawdown_pct=100.0,
            profit_factor=0,
            avg_loss=0,
            largest_win=0,
            largest_loss=0,
            annualized_return=0,
            sortino_ratio=0,
            expectancy=0,
            meets_objectives=False,
            meets_constraints=False,
            parameters={'error': str(e)}
        )


def load_backtest_data() -> Tuple[Dict, pd.DataFrame]:
    """Load and prepare all data needed for backtesting."""
    print("\n" + "="*60)
    print("LOADING BACKTEST DATA")
    print("="*60)
    
    # Use tickers.TRADE_TICKERS for the robust list (or context tickers if needed)
    # Check if config has overridden list (from --test flag)
    if hasattr(config, 'TRADE_TICKERS') and config.TRADE_TICKERS:
        symbols = config.TRADE_TICKERS
    else:
        symbols = tickers.TRADE_TICKERS
        
    print(f"Loading data for {len(symbols)} symbols...")
    
    # Fetch bars
    bar_data = fetch_bars_bulk(
        symbols=symbols,
        timeframe="1Hour",
        start="2016-06-01",
        end="2026-02-17"
    )
    
    print(f"Loaded {len(bar_data)} symbols with data")
    
    # Compute indicators for each symbol
    print("Computing indicators...")
    for sym in tqdm(bar_data.keys(), desc="Indicators"):
        bar_data[sym] = compute_indicators(bar_data[sym])
    
    # Load SPY daily for regime
    spy_daily_dict = fetch_bars_bulk(
        symbols=["SPY"],
        timeframe="1Day",
        start="2016-05-01",
        end="2026-02-17"
    )
    spy_daily = spy_daily_dict.get("SPY", pd.DataFrame())
    
    if not spy_daily.empty:
        spy_daily = compute_regime_indicators(spy_daily)
    
    print(f"✓ Data loaded: {len(bar_data)} symbols, SPY daily regime ready\n")
    
    return bar_data, spy_daily


def main():
    """Main optimization execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated Parameter Optimization')
    parser.add_argument('--cores', type=int, default=11, help='Number of CPU cores to use')
    parser.add_argument('--output', type=str, default='optimization_results.csv', 
                        help='Output CSV filename')
    parser.add_argument('--limit', type=int, default=None, 
                        help='Limit number of configurations (for testing)')
    parser.add_argument('--test', action='store_true', 
                        help='Run in fast test mode (few tickers, few configs)')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SWING TRADING BOT - PARAMETER OPTIMIZATION")
    print("="*60)
    print(f"CPU Cores: {args.cores}")
    print(f"Output File: {args.output}")
    print("="*60 + "\n")
    
    # Generate parameter grid
    configs = generate_parameter_grid()
    
    if args.limit:
        configs = configs[:args.limit]
        print(f"TESTING MODE: Limited to {args.limit} configurations\n")
    
    if args.test:
        print("TEST MODE ENABLED: Limiting to top 5 tickers and 5 DIVERSE configs")
        config.TRADE_TICKERS = tickers.TRADE_TICKERS[:5]
        # Pick 5 configs from different parts of the list to show variety
        step = len(configs) // 5
        configs = [configs[i] for i in range(0, len(configs), step)][:5]

    # Load data once (shared across all processes)
    bar_data, spy_daily = load_backtest_data()
    
    # Prepare arguments for multiprocessing
    # NOTE: On Windows, we refrain from passing the huge bar_data dict in the args list.
    # Instead, we use the init_worker approach to share it once per process.
    print(f"\nPreparing {len(configs):,} backtest configurations...\n")
    
    # Run parallel backtests
    print("="*60)
    print("RUNNING OPTIMIZATION")
    print("="*60)
    
    results = []
    
    # If using multiprocessing, ensure freeze_support for Windows
    mp.freeze_support()
    
    with mp.Pool(processes=args.cores, initializer=init_worker, initargs=(bar_data, spy_daily)) as pool:
        # Use imap_unordered for better performance with progress bar
        for result in tqdm(
            pool.imap_unordered(run_single_backtest, configs),
            total=len(configs),
            desc="Backtesting",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        ):
            results.append(result)
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60 + "\n")
    
    # Convert results to DataFrame
    results_data = []
    for r in results:
        row = {
            'config_id': r.config_id,
            'win_rate': r.win_rate,
            'total_return_pct': r.total_return_pct,
            'total_trades': r.total_trades,
            'avg_win': r.avg_win,
            'avg_loss': r.avg_loss,
            'sharpe_ratio': r.sharpe_ratio,
            'sortino_ratio': r.sortino_ratio,
            'max_drawdown_pct': r.max_drawdown_pct,
            'profit_factor': r.profit_factor,
            'largest_win': r.largest_win,
            'largest_loss': r.largest_loss,
            'annualized_return': r.annualized_return,
            'expectancy': r.expectancy,
            'meets_objectives': r.meets_objectives,
            'meets_constraints': r.meets_constraints,
        }
        
        # Add all parameters
        if r.parameters:
            row.update(r.parameters)
        
        results_data.append(row)
    
    df = pd.DataFrame(results_data)
    
    # Save to CSV
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    print(f"✓ Results saved to: {output_path}\n")
    
    # Print summary statistics
    print("="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total Configurations Tested: {len(results):,}")
    print(f"Configurations Meeting Objectives: {df['meets_objectives'].sum()}")
    print(f"Configurations Meeting Constraints: {df['meets_constraints'].sum()}")
    print(f"Perfect Configs (Both): {(df['meets_objectives'] & df['meets_constraints']).sum()}")
    print()
    
    # Best by win rate
    if not df.empty:
        best_wr = df.nlargest(5, 'win_rate')[['config_id', 'win_rate', 'total_return_pct', 'total_trades']]
        print("\nTop 5 by Win Rate:")
        print(best_wr.to_string(index=False))
        
        # Best by total return
        best_ret = df.nlargest(5, 'total_return_pct')[['config_id', 'win_rate', 'total_return_pct', 'total_trades']]
        print("\nTop 5 by Total Return:")
        print(best_ret.to_string(index=False))
        
        # Perfect configs
        perfect = df[df['meets_objectives'] & df['meets_constraints']]
        if len(perfect) > 0:
            print("\n" + "="*60)
            print("SUCCESS! FOUND PERFECT CONFIGURATIONS:")
            print("="*60)
            print(perfect[['config_id', 'win_rate', 'total_return_pct', 'total_trades', 'sharpe_ratio']].to_string(index=False))
        else:
            print("\nNo configurations met both objectives and constraints.")
            print("Review the CSV for near-misses and trade-offs.\n")
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
