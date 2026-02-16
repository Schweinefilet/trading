from data.market_data import MarketDataClient
from data.indicators import calculate_all_indicators
from datetime import datetime
import pandas as pd
from alpaca.data.timeframe import TimeFrame

def audit_ema_200():
    client = MarketDataClient()
    start = datetime(2023, 1, 1) # Plenty of history for 200 EMA
    end = datetime(2024, 12, 31)
    raw = client.get_bars(["NVDA", "SPY"], timeframe=TimeFrame.Day, start=start, end=end)
    
    if "NVDA" in raw and "SPY" in raw:
        nvda = calculate_all_indicators(raw["NVDA"])
        spy = calculate_all_indicators(raw["SPY"])
        
        # Merge on index
        df = pd.merge(nvda[['close', 'rsi', 'ema_200']], spy[['close', 'ema_200']], left_index=True, right_index=True, suffixes=('_nvda', '_spy'))
        df['is_nvda_uptrend_200'] = df['close_nvda'] > df['ema_200_nvda']
        df['is_spy_bullish_200'] = df['close_spy'] > df['ema_200_spy']
        df['is_nvda_rsi_low'] = df['rsi'] < 40
        
        pullbacks = df[(df['is_nvda_rsi_low']) & (df.index >= '2024-01-01')]
        print(f"NVDA Pullbacks in 2024 (RSI < 40):")
        
        potential_entries = pullbacks[(pullbacks['is_nvda_uptrend_200']) & (pullbacks['is_spy_bullish_200'])]
        print(f"\nPotential Entries with 200-EMA Filter: {len(potential_entries)}")
        if len(potential_entries) > 0:
            print(potential_entries[['close_nvda', 'rsi', 'is_nvda_uptrend_200', 'is_spy_bullish_200']].head(10))
    else:
        print("Missing data")

if __name__ == "__main__":
    audit_ema_200()
