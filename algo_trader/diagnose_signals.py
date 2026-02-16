"""
Diagnostic script to debug why walk-forward generates 0 trades.
This will help identify if the issue is:
1. Signal generation
2. Regime filtering
3. Parameter optimization
4. Data quality
"""
import pandas as pd
from datetime import datetime, timedelta

# Test 1: Check if signals are being generated at all
def test_signal_generation():
    """Test if signal generation works with default parameters."""
    print("\n" + "-"*60)
    print("TEST 1: Signal Generation")
    print("-"*60)
    
    # Mock some bar data
    dates = pd.date_range(start='2025-01-01', periods=100, freq='15min')
    mock_data = pd.DataFrame({
        'open': 100 + pd.Series(range(100)).apply(lambda x: x * 0.1),
        'high': 101 + pd.Series(range(100)).apply(lambda x: x * 0.1),
        'low': 99 + pd.Series(range(100)).apply(lambda x: x * 0.1),
        'close': 100 + pd.Series(range(100)).apply(lambda x: x * 0.1),
        'volume': 1000000,
    }, index=dates)
    
    # Add basic indicators
    mock_data['rsi'] = 50  # Neutral RSI
    mock_data['adx'] = 25  # Trending
    mock_data['atr'] = 2.0
    mock_data['volume_sma_20'] = 1000000
    
    print(f"PASS Created mock data: {len(mock_data)} bars")
    print(f"  RSI range: {mock_data['rsi'].min():.1f} - {mock_data['rsi'].max():.1f}")
    print(f"  ADX: {mock_data['adx'].iloc[-1]:.1f}")
    
    # Try to generate a signal
    try:
        from strategy.signals import generate_signals
        
        # Test with BULLISH regime
        signal = generate_signals('TEST', mock_data, 'BULLISH')
        if signal:
            print(f"PASS BULLISH regime generated signal: {signal.direction.value}")
        else:
            print("FAIL BULLISH regime: NO SIGNAL generated")
            print("  -> Check RSI/ADX thresholds in strategy/signals.py")
        
        # Test with NEUTRAL regime
        signal = generate_signals('TEST', mock_data, 'NEUTRAL')
        if signal:
            print(f"PASS NEUTRAL regime generated signal: {signal.direction.value}")
        else:
            print("FAIL NEUTRAL regime: NO SIGNAL generated")
            
    except Exception as e:
        print(f"FAIL ERROR in signal generation: {e}")
        import traceback
        traceback.print_exc()


# Test 2: Check regime filter behavior
def test_regime_filter():
    """Check what regime SPY was in during 2025."""
    print("\n" + "-"*60)
    print("TEST 2: Regime Filter Analysis")
    print("-"*60)
    
    try:
        from data.historical import HistoricalDataClient, parse_timeframe
        from data.indicators import compute_regime_indicators
        from strategy.regime import get_regime
        
        client = HistoricalDataClient()
        tf = parse_timeframe("1Day")
        start = datetime(2024, 6, 1)
        end = datetime(2026, 1, 1)
        
        spy_data = client.fetch_bars("SPY", tf, start, end)
        if spy_data.empty:
            print("FAIL No SPY data fetched")
            return
            
        spy_data = compute_regime_indicators(spy_data)
        
        # Check regime for each month in 2025
        regime_counts = {}
        for month in range(1, 13):
            month_data = spy_data[
                (spy_data.index.year == 2025) & 
                (spy_data.index.month == month)
            ]
            if not month_data.empty:
                regime = get_regime(month_data)
                regime_counts[f"2025-{month:02d}"] = regime.value
        
        print("Regime by month in 2025:")
        for month, regime in regime_counts.items():
            print(f"  {month}: {regime}")
        
        # Count regime distribution
        from collections import Counter
        regime_dist = Counter(regime_counts.values())
        print(f"\nRegime distribution:")
        for regime, count in regime_dist.items():
            print(f"  {regime}: {count} months ({count/12*100:.1f}%)")
        
        if regime_dist.get('BEARISH', 0) + regime_dist.get('NEUTRAL', 0) > 6:
            print("\nWARNING  WARNING: Market was NOT BULLISH most of 2025")
            print("   If USE_REGIME_FILTER=True and strategy only trades LONG in BULLISH,")
            print("   this explains why walk-forward generated 0 trades!")
            
    except Exception as e:
        print(f"FAIL ERROR in regime analysis: {e}")
        import traceback
        traceback.print_exc()


# Test 3: Check if parameter optimization is working
def test_parameter_optimization():
    """Test if walk-forward actually searches parameter space."""
    print("\n" + "-"*60)
    print("TEST 3: Parameter Optimization")
    print("-"*60)
    
    # Check if walk_forward.py has actual optimization logic
    try:
        with open('backtest/walk_forward.py', 'r') as f:
            code = f.read()
        
        # Look for optimization keywords
        has_grid_search = 'itertools' in code or 'product' in code
        has_optimization = 'optimize' in code.lower()
        has_parameter_sweep = any(x in code for x in ['for param', 'param_grid', 'parameter_combinations'])
        
        print(f"Grid search imports: {'PASS' if has_grid_search else 'FAIL'}")
        print(f"Optimization logic: {'PASS' if has_optimization else 'FAIL'}")
        print(f"Parameter sweeping: {'PASS' if has_parameter_sweep else 'FAIL'}")
        
        if not (has_grid_search or has_parameter_sweep):
            print("\nWARNING  WARNING: walk_forward.py likely NOT optimizing parameters!")
            print("   It's probably just returning config defaults.")
            print("   This explains why all windows show identical 'Best Params'.")
            
    except FileNotFoundError:
        print("FAIL backtest/walk_forward.py not found")
    except Exception as e:
        print(f"FAIL ERROR reading walk_forward.py: {e}")


# Test 4: Check capital constraints
def test_capital_enforcement():
    """Check if backtest engine properly enforces capital limits."""
    print("\n" + "-"*60)
    print("TEST 4: Capital Constraint Enforcement")
    print("-"*60)
    
    # Analyze the backtest log for overlapping positions
    trades_on_jan17 = [
        ('15:00', 'VRT', 2434.28),
        ('19:00', 'MA', 2105.01),
        ('19:15', 'TXN', 2495.17),
        ('20:15', 'MU', 2420.92),
        ('20:15', 'LRCX', 2420.41),
    ]
    
    print("Trades on 2025-01-17:")
    running_total = 0
    for time, sym, amt in trades_on_jan17:
        running_total += amt
        status = "WARNING EXCEEDS $10k" if running_total > 10000 else "PASS"
        print(f"  {time} {sym:6s} ${amt:8.2f}  Running: ${running_total:9.2f} {status}")
    
    if running_total > 10000:
        print(f"\nWARNING  WARNING: Backtest allowed ${running_total:,.2f} in positions!")
        print("   This exceeds $10k capital. Engine not enforcing limits properly.")
        print("   Check backtest/engine.py position tracking logic.")


# Test 5: Check if data has required indicators
def test_indicator_computation():
    """Verify indicators are computed correctly."""
    print("\n" + "-"*60)
    print("TEST 5: Indicator Computation")
    print("-"*60)
    
    try:
        from data.historical import HistoricalDataClient, parse_timeframe
        from data.indicators import compute_indicators
        
        client = HistoricalDataClient()
        tf = parse_timeframe("15Min")
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 31)
        
        data = client.fetch_bars("NVDA", tf, start, end)
        if data.empty:
            print("FAIL No data fetched for NVDA")
            return
        
        data = compute_indicators(data)
        
        required = ['rsi', 'adx', 'atr', 'ema_9', 'ema_21', 'volume_sma_20']
        print("Required indicators:")
        for ind in required:
            if ind in data.columns:
                non_null = data[ind].notna().sum()
                pct = non_null / len(data) * 100
                print(f"  {ind:20s}: PASS ({non_null}/{len(data)} bars, {pct:.1f}%)")
            else:
                print(f"  {ind:20s}: FAIL MISSING")
        
        # Check for NaN issues
        last_row = data.iloc[-1]
        nan_indicators = [col for col in required if col in data.columns and pd.isna(last_row[col])]
        if nan_indicators:
            print(f"\nWARNING  WARNING: Last bar has NaN in: {nan_indicators}")
            print("   Signal generation will fail if indicators are NaN!")
            
    except Exception as e:
        print(f"✗ ERROR in indicator test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "-"*70)
    print("  ALGO TRADER DIAGNOSTIC SUITE")
    print("  Identifying why walk-forward generates 0 trades")
    print("-"*70)
    
    test_signal_generation()
    test_regime_filter()
    test_parameter_optimization()
    test_capital_enforcement()
    test_indicator_computation()
    
    print("\n" + "-"*70)
    print("  DIAGNOSIS COMPLETE")
    print("-"*70)
    print("\nNext steps:")
    print("1. Fix any failures identified above")
    print("2. If regime filter is too restrictive, set USE_REGIME_FILTER=False")
    print("3. Implement actual parameter optimization in walk_forward.py")
    print("4. Lower RSI/ADX thresholds if no signals are being generated")
    print("5. Fix capital enforcement in backtest engine")
