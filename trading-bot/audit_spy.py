from data.market_data import MarketDataClient
from data.indicators import calculate_all_indicators
from datetime import datetime
import pandas as pd
from alpaca.data.timeframe import TimeFrame

def audit_spy_nvda():
    client = MarketDataClient()
    start = datetime(2023, 8, 1) # Start earlier for EMA warm up
    end = datetime(2024, 12, 31)
    raw = client.get_bars(["NVDA", "SPY"], timeframe=TimeFrame.Day, start=start, end=end)
    
    if "NVDA" in raw and "SPY" in raw:
        nvda = calculate_all_indicators(raw["NVDA"])
        spy = calculate_all_indicators(raw["SPY"])
        
        # Merge on index
        df = pd.merge(nvda[['close', 'rsi', 'ema_50']], spy[['close', 'ema_50']], left_index=True, right_index=True, suffixes=('_nvda', '_spy'))
        df['is_nvda_uptrend'] = df['close_nvda'] > df['ema_50_nvda']
        df['is_spy_bullish'] = df['close_spy'] > df['ema_50_spy']
        df['is_nvda_rsi_low'] = df['rsi'] < 40
        
        pullbacks = df[(df['is_nvda_rsi_low']) & (df.index >= '2024-01-01')]
        print(f"NVDA Pullbacks in 2024 (RSI < 40):")
        print(pullbacks[['close_nvda', 'rsi', 'is_nvda_uptrend', 'is_spy_bullish']])
        
        potential_entries = pullbacks[(pullbacks['is_nvda_uptrend']) & (pullbacks['is_spy_bullish'])]
        print(f"\nPotential Entries (Uptrend + SPY Bullish): {len(potential_entries)}")
    else:
        print("Missing data")

if __name__ == "__main__":
    audit_spy_nvda()
