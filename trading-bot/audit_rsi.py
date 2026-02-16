from data.market_data import MarketDataClient
from data.indicators import calculate_all_indicators
from datetime import datetime
import pandas as pd
from alpaca.data.timeframe import TimeFrame

def audit_rsi():
    client = MarketDataClient()
    start = datetime(2024, 1, 1)
    end = datetime(2024, 12, 31)
    raw = client.get_bars(["NVDA"], timeframe=TimeFrame.Day, start=start, end=end)
    if "NVDA" in raw:
        df = calculate_all_indicators(raw["NVDA"])
        print(f"NVDA 2024 RSI Summary:")
        print(df["rsi"].describe())
        print(f"Lowest RSI: {df['rsi'].min()}")
        print(f"Days with RSI < 40: {len(df[df['rsi'] < 40])}")
        print(f"Days with RSI < 45: {len(df[df['rsi'] < 45])}")
    else:
        print("No data")

if __name__ == "__main__":
    audit_rsi()
