import pandas as pd
import openpyxl
from openai import OpenAI
import os
import sys
from pathlib import Path

# Add tracker dir to path to import config
sys.path.append(str(Path(__file__).parent / "webull_tracker"))
import config

def get_current_positions(excel_path):
    """
    Reads the Trade Log and calculates current open positions.
    """
    if not Path(excel_path).exists():
        return "No trading log found."
    
    try:
        df = pd.read_excel(excel_path, sheet_name="Trade Log")
        if df.empty:
            return "No trades recorded."
        
        # Simple position tracking logic
        positions = {}
        for _, row in df.iterrows():
            symbol = row['Symbol']
            side = str(row['Side']).lower()
            qty = float(row['Qty'])
            
            if symbol not in positions:
                positions[symbol] = 0.0
            
            if 'buy' in side:
                positions[symbol] += qty
            elif 'sell' in side:
                positions[symbol] -= qty
        
        # Filter out closed positions
        open_positions = {k: round(v, 4) for k, v in positions.items() if abs(v) > 0.0001}
        return open_positions
    except Exception as e:
        return f"Error reading log: {e}"

def get_recent_performance(excel_path):
    """
    Reads the Daily Performance sheet to get a summary.
    """
    try:
        df = pd.read_excel(excel_path, sheet_name="Daily Performance")
        if df.empty:
            return "No performance data."
        
        # Last 5 days of performance
        recent = df.tail(5).to_string(index=False)
        return recent
    except Exception as e:
        return f"Error reading performance: {e}"

import yfinance as yf
from datetime import datetime

def get_market_data(symbols):
    """
    Fetches current price, day change %, and recent news for a list of symbols.
    """
    if not symbols:
        return {}
    
    print(f"[AI Trading Advisor] Fetching data & news for: {', '.join(symbols)}...")
    data = {}
    try:
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            price = round(info.last_price, 2)
            prev_close = ticker.info.get('previousClose', price)
            change = round(((price - prev_close) / prev_close) * 100, 2)
            
            # Fetch recent news (titles only to save tokens)
            news = []
            try:
                raw_news = ticker.news[:3] # Last 3 news items
                news = [item.get('title') for item in raw_news if item.get('title')]
            except:
                pass
                
            data[symbol] = {
                "price": price, 
                "change_pct": change,
                "recent_news": news
            }
    except Exception as e:
        print(f"Error fetching market data: {e}")
    return data

def get_trading_advice():
    # Check if API key is still a placeholder or empty
    placeholders = ["YOUR_API_KEY_HERE", "PASTE_YOUR_OPENAI_API_KEY_HERE"]
    if not config.OPENAI_API_KEY or any(p in config.OPENAI_API_KEY for p in placeholders):
        print("\n[!] OpenAI API Key missing or invalid.")
        print(f"Please update the OPENAI_API_KEY in: {Path(config.__file__).absolute()}")
        return

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    excel_path = config.MASTER_EXCEL_PATH
    positions = get_current_positions(excel_path)
    performance = get_recent_performance(excel_path)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Market Watchlist for new opportunities (Non-Crypto, Shariah-Compliant focus)
    WATCHLIST = ["AAPL", "MSFT", "NVDA", "CRM", "GOOGL", "AMD", "META", "NFLX", "ADBE", "INTC"]
    
    # Fetch real-time market data and news for BOTH positions and watchlist
    research_symbols = []
    if isinstance(positions, dict):
        research_symbols.extend(list(positions.keys()))
    
    # Add watchlist to research (avoid duplicates)
    for s in WATCHLIST:
        if s not in research_symbols:
            research_symbols.append(s)
            
    market_data = get_market_data(research_symbols)
    
    prompt = f"""
You are an expert algorithmic trading advisor specializing in Shariah-compliant and non-crypto equities. 
Current Date and Time: {current_time}

Based on the trade history, current positions, market watchlist, real-time data, and recent news, provide a comprehensive trading strategy for today.

# CRITICAL CONSTRAINTS:
1. **NON-CRYPTO ONLY**: Do NOT recommend any cryptocurrencies, crypto-linked stocks (like MSTR, COIN), or crypto ETFs.
2. **SHARIAH COMPLIANCE**: Only recommend stocks that pass Shariah screening:
   - **Business Screening**: No companies involved in prohibited activities (e.g., interest-based finance/banking, conventional insurance, gambling, alcohol, pork, non-halal food, adult entertainment, tobacco, weapons).
   - **Financial Screening**: Total debt/Market Cap < 33%, Interest-bearing cash/Market Cap < 33%, and Interest-bearing debt/Market Cap < 33%.
   - **Note**: If a stock in the watchlist or current positions does not meet these criteria, recommend CLOSING it (if held) or IGNORING it (if in watchlist).

# Current Open Positions (Actual Account):
{positions}

# Market Data & News (Portfolio + Watchlist):
{market_data}

# Recent Account Performance:
{performance}

# Required Tasks:
1. **Manage Existing Positions**: For every current position, first verify Shariah compliance and non-crypto status. Then determine if we should HOLD, CLOSE, or adjust SL/TP.
2. **Scout for New Opportunities (Day Trading)**: Recommend at least **1 NEW day trade** (1-2 day outlook) that is strictly Shariah-compliant and non-crypto.
3. **Scout for New Opportunities (Long Term)**: Recommend at least **1 NEW long-term trade** (weeks/months outlook) that is strictly Shariah-compliant and non-crypto.

# Suggestion Format:
For EVERY suggested trade (Current + New), you MUST include:
1. **Ticker & Action** (e.g., AAPL: BUY)
2. **Trade Type** (CURRENT / DAY TRADE / LONG TERM)
3. **Compliance Status** (Confirm if it is Shariah-compliant and why)
4. **Levels**: Recommended Quantity (assume $1k risk), Take Profit, and Stop Loss.
5. **Rationale**: Detailed explanation including news or trend context, and justification for the trade type (day vs long term).
"""

    print("\n[AI Trading Advisor] Consulting ChatGPT...")
    try:
        response = client.chat.completions.create(
            model=config.ADVISOR_MODEL,
            messages=[
                {"role": "system", "content": "You are a top-tier financial analyst and Shariah compliance expert. Provide precise, professional advice in a human-readable format."},
                {"role": "user", "content": prompt}
            ]
        )
        advice = response.choices[0].message.content
        print("\n" + "="*50)
        print(f"TRADING STRATEGY (ADVISOR) - {current_time}")
        print("="*50)
        print(advice)
        print("="*50)
        
    except Exception as e:
        print(f"Error consulting AI: {e}")

if __name__ == "__main__":
    get_trading_advice()
