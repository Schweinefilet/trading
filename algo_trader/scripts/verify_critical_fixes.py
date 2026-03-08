
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock

sys.path.insert(0, ".")

from config.settings import config
from config.tickers import tickers
from backtest.engine import BacktestEngine

class TestCriticalFixes(unittest.TestCase):
    
    def test_tier_default(self):
        print("\nTesting Tier Default...")
        # "UNKNOWN_SYMBOL" should be Tier 3 (value 3), providing 0.0 bonus (only Tier 1/2 get bonus)
        # Wait, get_tier_bonus returns 0.0 for Tier 3.
        # But previously it returned Tier 4 (also 0.0).
        # The user wanted to avoid "penalizing".
        # Let's check get_tier value directly.
        tier = tickers.get_tier("UNKNOWN_SYMBOL")
        print(f"Tier for UNKNOWN_SYMBOL: {tier}")
        self.assertEqual(tier, 3)

    def test_regime_fallback(self):
        print("\nTesting Regime Fallback...")
        engine = BacktestEngine()
        
        # Create empty SPY data with some dates but missing columns/values
        dates = pd.date_range("2025-01-01", periods=3)
        spy_daily = pd.DataFrame(index=dates, columns=["close", "sma_50", "sma_200", "vol_proxy"])
        # No values (NaN), so it should fallback
        spy_daily["close"] = np.nan
        spy_daily["sma_50"] = np.nan
        
        regime_map = engine._build_regime_map(spy_daily)
        print(f"Regime Map: {regime_map}")
        
        # Should be BEARISH
        for d, r in regime_map.items():
            self.assertEqual(r, "BEARISH")

    def test_gap_stop_loss(self):
        print("\nTesting Gap Stop Loss...")
        engine = BacktestEngine()
        
        # Position: LONG @ 100, Stop @ 95
        pos = {
            "symbol": "TEST",
            "direction": "LONG",
            "entry_price": 100.0,
            "stop_price": 95.0,
            "take_profit": 110.0,
            "atr": 1.0
        }
        
        # Scenario 1: Open 94 (Gap Down below 95) -> Should exit at 94
        bar_gap = {"open": 94.0, "high": 96.0, "low": 93.0, "close": 95.0}
        hit, price, reason = engine._check_stops(pos, bar_gap)
        print(f"Gap Down: Hit={hit}, Price={price}, Reason={reason}")
        self.assertTrue(hit)
        self.assertEqual(price, 94.0)

        # Scenario 2: Open 95.5 (No Gap below stop), Low 94 -> Should exit at 95
        bar_intraday = {"open": 95.5, "high": 96.0, "low": 94.0, "close": 95.0}
        hit, price, reason = engine._check_stops(pos, bar_intraday)
        print(f"Intraday: Hit={hit}, Price={price}, Reason={reason}")
        self.assertTrue(hit)
        self.assertEqual(price, 95.0)

    def test_short_gap_stop_loss(self):
        print("\nTesting SHORT Gap Stop Loss...")
        engine = BacktestEngine()
        
        # Position: SHORT @ 100, Stop @ 105
        pos = {
            "symbol": "TEST",
            "direction": "SHORT",
            "entry_price": 100.0,
            "stop_price": 105.0,
            "take_profit": 90.0,
            "atr": 1.0
        }
        
        # Scenario 1: Open 106 (Gap Up above 105) -> Should exit at 106
        bar_gap = {"open": 106.0, "high": 107.0, "low": 104.0, "close": 105.0}
        hit, price, reason = engine._check_stops(pos, bar_gap)
        print(f"Gap Up: Hit={hit}, Price={price}, Reason={reason}")
        self.assertTrue(hit)
        self.assertEqual(price, 106.0)

        # Scenario 2: Open 104 (No Gap), High 106 -> Should exit at 105
        bar_intraday = {"open": 104.0, "high": 106.0, "low": 103.0, "close": 105.0}
        hit, price, reason = engine._check_stops(pos, bar_intraday)
        print(f"Intraday: Hit={hit}, Price={price}, Reason={reason}")
        self.assertTrue(hit)
        self.assertEqual(price, 105.0)

    def test_atr_zero(self):
        print("\nTesting Zero ATR Handling...")
        from risk.position_sizer import calculate_position_size
        
        # Should return 0 shares if ATR is 0 (or stop distance is 0)
        shares = calculate_position_size(100000, 100, 100, 0.0)
        print(f"Shares with ATR=0 (Stop Dist=0): {shares}")
        self.assertEqual(shares, 0)
        
        # Explicit zero check in calculate_position_size uses stop_distance
        # If stop_distance > 0 but ATR=0? ATR is used only for trailing stop / partial exit logic inside engine?
        # calculate_position_size takes atr arg but doesn't seem to use it for sizing if stop_price is provided?
        # Let's check. It uses stop_distance.
        # But if stop distance is 0, returns 0.
        
        # Test Negative Equity
        shares_neg = calculate_position_size(-5000, 100, 95, 1.0)
        print(f"Shares with Negative Equity: {shares_neg}")
        self.assertEqual(shares_neg, 0)

if __name__ == '__main__':
    unittest.main()
