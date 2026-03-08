"""
Data fetcher wrapper.
Provides simplified interface for bulk data loading, wrapping data.historical.
"""
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

from data.historical import HistoricalDataClient, parse_timeframe
from config.tickers import tickers


def fetch_bars_bulk(
    symbols: List[str] = None,
    timeframe: str = "1Hour",
    start: str = None,
    end: str = None,
    use_cache: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical bars for multiple symbols.
    
    Args:
        symbols: List of symbols (defaults to all active trade tickers)
        timeframe: "15Min", "1Hour", "1Day", etc.
        start: Start date string "YYYY-MM-DD"
        end: End date string "YYYY-MM-DD" (defaults to now)
        use_cache: Whether to use local parquet cache
    
    Returns:
        Dict mapping symbol -> DataFrame
    """
    if symbols is None:
        # Use all tradeable tickers + context
        symbols = tickers.all_symbols
        
    # Convert dates
    start_dt = datetime.strptime(start, "%Y-%m-%d") if start else datetime(2023, 1, 1)
    end_dt = datetime.strptime(end, "%Y-%m-%d") if end else datetime.now()
    
    # Parse timeframe
    tf = parse_timeframe(timeframe)
    
    # Initialize client
    client = HistoricalDataClient()
    
    print(f"Fetching {timeframe} data for {len(symbols)} symbols from {start_dt.date()} to {end_dt.date()}...")
    
    # Fetch
    data = client.fetch_bars_multi(
        symbols=symbols,
        timeframe=tf,
        start=start_dt,
        end=end_dt,
        use_cache=use_cache
    )
    
    return data
