from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from pathlib import Path

# Load env
env_path = Path("e:/trading/.env")
load_dotenv(env_path)

client = StockHistoricalDataClient(
    api_key=os.getenv("ALPACA_API_KEY"),
    secret_key=os.getenv("ALPACA_SECRET_KEY")
)

start = datetime.now() - timedelta(days=5)
end = datetime.now()

print(f"Fetching 5-minute bars for NVDA from {start} to {end}...")
try:
    request = StockBarsRequest(
        symbol_or_symbols=["NVDA"],
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start,
        end=end
    )
    response = client.get_stock_bars(request)
    print(f"Response type: {type(response)}")
    print(f"Token related attributes: {[a for a in dir(response) if 'token' in a.lower()]}")
    if hasattr(response, 'data'):
        print(f"Success! Received {len(response.data.get('NVDA', []))} bars.")
except Exception as e:
    print(f"Error: {e}")
