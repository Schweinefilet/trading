
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, ".")

from config.settings import config
from risk.position_sizer import calculate_position_size
from strategy.ranker import SignalRanker
from execution.order_manager import OrderManager, PositionState

class TestConfigWiring(unittest.TestCase):
    
    def test_config_values(self):
        print("\nTesting Config Values...")
        self.assertEqual(config.CONFIRMATIONS_BULLISH, 3)
        self.assertEqual(config.CONFIRMATIONS_CAUTIOUS, 4)
        self.assertEqual(config.CONFIRMATIONS_BEARISH, 5)
        self.assertEqual(config.ENTRY_BLACKOUT_OPEN_MINS, 15)
        self.assertEqual(config.PARTIAL_EXIT_ATR_TRIGGER, 2.0)
        self.assertEqual(config.PARTIAL_EXIT_PCT, 0.20)
        print("Config values verified.")

    def test_position_sizing_risk(self):
        print("\nTesting Position Sizing Risk...")
        print(f"DEBUG: config.RISK_PER_TRADE_PCT = {config.RISK_PER_TRADE_PCT}")
        equity = 100000
        entry = 100
        stop = 80 # $20 risk
        
        # Test Default (6.7%)
        shares = calculate_position_size(equity, 100, 80, 5.0)
        print(f"Default Risk Shares: {shares} (Expected ~335)")
        # 100k * 0.067 = 6700 risk. 6700 / 20 = 335.
        
        self.assertEqual(shares, 335)
        
        # Test Bearish Risk (2.5%) explicitly passed
        # Risk 2.5% = 2,500. Stop $20. Shares = 125.
        shares_bearish = calculate_position_size(equity, 100, 80, 5.0, risk_pct=0.025)
        print(f"Bearish Risk Shares: {shares_bearish}")
        self.assertEqual(shares_bearish, 125)
        
    def test_ranker_sector_limits(self):
        print("\nTesting Ranker Sector Limits...")
        ranker = SignalRanker()
        
        # Mock signals
        class MockSignal:
            def __init__(self, sym, strength):
                self.symbol = sym
                self.signal_strength = strength
                
        signals = [MockSignal("AAPL", 0.9), MockSignal("MSFT", 0.8), MockSignal("GOOG", 0.7)]
        
        # config.MAX_SECTOR_POSITIONS is 2
        # Say we have 2 tech positions already
        sector_positions = {"Technology": 2}
        open_positions = {}
        
        # All signals should be filtered out
        with patch('config.tickers.tickers.get_sector', return_value="Technology"):
            day, swing = ranker.rank_and_select(
                signals, open_positions, sector_positions, 0, 0, 100000, 3
            )
            print(f"Signals selected with max sector constraints: {len(day) + len(swing)}")
            self.assertEqual(len(day) + len(swing), 0)
            
        # With 1 position, should allow 1 more (max 2 entries per cycle, checks ranker logic)
        sector_positions = {"Technology": 1}
        with patch('config.tickers.tickers.get_sector', return_value="Technology"):
             day, swing = ranker.rank_and_select(
                signals, open_positions, sector_positions, 0, 0, 100000, 3
            )
             print(f"Signals selected with 1 sector position: {len(day) + len(swing)}")
             self.assertTrue(len(day) + len(swing) > 0)

    def test_order_manager_time_stop(self):
        print("\nTesting Order Manager Time Stop...")
        om = OrderManager()
        om.submit_market_close = MagicMock()
        
        # Create a stale position
        stale_time = datetime.now() - timedelta(hours=config.MAX_HOLD_HOURS + 1)
        om._positions["STALE"] = PositionState(
            symbol="STALE", side="long", qty=100, entry_price=100,
            entry_time=stale_time, stop_price=90, take_profit_price=110, atr=1
        )
        
        closed = om.check_time_stops()
        print(f"Closed symbols: {closed}")
        self.assertIn("STALE", closed)
        om.submit_market_close.assert_called()
        
    def test_order_manager_partial_exit(self):
        print("\nTesting Order Manager Partial Exit...")
        om = OrderManager()
        om.submit_market_close = MagicMock()
        
        entry_price = 100
        atr = 2.0
        # Target is 2.0 ATR = +4.0 price -> 104.0
        
        om._positions["WINNER"] = PositionState(
            symbol="WINNER", side="long", qty=100, entry_price=entry_price,
            entry_time=datetime.now(), stop_price=90, take_profit_price=110, atr=atr
        )
        
        # Current price 105 (2.5 ATR profit)
        current_prices = {"WINNER": 105.0}
        
        partials = om.check_partial_exits(current_prices)
        print(f"Partial exit symbols: {partials}")
        self.assertIn("WINNER", partials)
        self.assertTrue(om._positions["WINNER"].partial_exit_taken)
        # Should close 20% of 100 = 20
        om.submit_market_close.assert_called_with("WINNER", 20, "sell")

if __name__ == '__main__':
    unittest.main()
