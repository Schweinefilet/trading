from openai import OpenAI
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import yfinance as yf

# Add tracker dir to path to import config
sys.path.append(str(Path(__file__).parent / "webull_tracker"))
import config

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

def get_alpaca_context():
    """
    Fetches current account balance and positions from Alpaca.
    """
    if not config.ALPACA_API_KEY or config.ALPACA_API_KEY == "YOUR_ALPACA_KEY_ID":
        return None, "Alpaca API Key missing."

    try:
        client = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=True)
        account = client.get_account()
        positions = client.get_all_positions()
        
        pos_summary = {p.symbol: round(float(p.qty), 4) for p in positions}
        context = {
            "equity": account.equity,
            "buying_power": account.buying_power,
            "positions": pos_summary
        }
        return context, None
    except Exception as e:
        return None, f"Error fetching Alpaca context: {e}"

def get_market_data(symbols):
    """
    Fetches current price and news for a list of symbols using batch download.
    """
    if not symbols:
        return {}
    
    print(f"[Paper Trader] Batch fetching market data for {len(symbols)} symbols...")
    data = {}
    try:
        # Batch download prices (1 day period to get latest)
        # Using threads=True for parallel download
        tickers = yf.download(symbols, period="1d", group_by='ticker', threads=True, progress=False)
        
        for symbol in symbols:
            try:
                # Handle varying dataframe structures depending on yfinance version/result count
                if len(symbols) == 1:
                    # If single symbol, tickers is just the DF
                    df = tickers
                    recent_price = df['Close'].iloc[-1]
                else:
                    # If multiple, it's MultiIndex, access by symbol
                    # Check if symbol exists in columns
                    if symbol not in tickers.columns.levels[0]:
                        print(f"No data for {symbol}")
                        continue
                    df = tickers[symbol]
                    recent_price = df['Close'].iloc[-1]
                
                # News still needs individual fetching unfortunately, or we skip/minimize it for speed
                # For optimization, we'll fetch news only for top positions or skip to save time 
                # OR we can keep it as is if news is critical. 
                # To be super fast, let's skip detailed news fetching for the broad watchlist 
                # and only do it if absolutely needed or accept the trade-off. 
                # For now, let's keep it simple and just get price + metadata.
                
                # Note: 'fast_info' is faster but doesn't work well in batch context as easily without iterating Ticker objects.
                # If we really want news, we must iterate. 
                # Compromise: Batch fetch PRICES (slowest part usually), iterate for news if needed.
                
                # Actually, iterating Ticker objects for news is what makes it slow. 
                # Let's drop news fetching for the optimization to prioritize speed, 
                # as the AI mainly acts on price/trends in this simplified version.
                
                data[symbol] = {
                    "price": round(float(recent_price), 2),
                    "recent_news": [] # Skipped for performance
                }
            except Exception as e:
                # print(f"Error processing {symbol}: {e}")
                pass
                
    except Exception as e:
        print(f"Error fetching market data: {e}")
    return data

def execute_alpaca_trades(trades):
    """
    Executes trades on Alpaca paper trading account.
    """
    print("\n[Alpaca] Executing trades...")
    try:
        client = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=True)
        current_positions = {p.symbol: p for p in client.get_all_positions()}
        
        for trade in trades:
            symbol = trade.get('ticker')
            action = trade.get('action', '').upper()
            qty_val = trade.get('quantity')
            
            if not symbol or not action or qty_val is None:
                continue
                
            try:
                qty = float(qty_val)
                if qty <= 0: continue
            except:
                continue
            
            if action == 'HOLD':
                continue
            
            side = None
            if action == 'BUY':
                side = OrderSide.BUY
            elif action == 'SELL' or action == 'CLOSE':
                side = OrderSide.SELL if symbol in current_positions else None
                if action == 'CLOSE' and side is None:
                    print(f"[Alpaca] Skipping CLOSE for {symbol} (no position).")
                    continue
            
            if side:
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.GTC
                )
                order = client.submit_order(order_data=order_data)
                print(f"[Alpaca] Order submitted: {side} {qty} {symbol} (ID: {order.id})")
                print(f"[Alpaca] Order submitted: {side} {qty} {symbol} (ID: {order.id})")
    except Exception as e:
        error_msg = str(e)
        if "insufficient buying power" in error_msg.lower():
             print(f"[Alpaca] Warning: Insufficient buying power to execute remaining trades.")
        else:
            print(f"[Alpaca] Error: {e}")

def run_paper_trader():
    print("--- Alpaca Automated Paper Trader ---")
    
    # Check OpenAI Key
    if not config.OPENAI_API_KEY:
        print("[!] OpenAI API Key missing from config.")
        return

    if not config.OPENAI_API_KEY or "sk-proj-..." in config.OPENAI_API_KEY:
        print("[!] OpenAI API Key missing.")
        return

    # Get Alpaca Context
    context, error = get_alpaca_context()
    if error:
        print(f"[!] {error}")
        return

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # EXCLUDED / BOYCOTT LIST (BDS/Ethical)
    EXCLUDED_TICKERS = ["GOOG", "GOOGL", "MSFT", "CSCO", "CVX", "CAT", "BKNG", "KO"]

    WATCHLIST = [
        "NVDA", "AAPL", "AVGO", "TSLA", "LLY", "XOM", "JNJ", "AMD", "ABBV", "ORCL",
        "HD", "PG", "MU", "CRM", "MRK", "ABT", "TMO", "LRCX", "ISRG", "LIN",
        "PEP", "UBER", "ADBE", "TXN", "ASML", "QCOM", "AMAT", "COST", "MCD", "ACN",
        "UNH", "VZ", "AZN", "COP", "HON", "DE", "NOW", "ADP", "SBUX", "BSX",
        "TJX", "NKE", "MDT", "PANW", "KLAC", "MDLZ", "CL", "ADI", "ITW", "SHW"
    ]
    
    # Symbols to research
    research_symbols = list(context['positions'].keys())
    for s in WATCHLIST:
        if s not in research_symbols: research_symbols.append(s)
    # research_symbols = research_symbols[:15] # Removed limit to scan full watchlist

    market_data = get_market_data(research_symbols)
    
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    prompt = f"""
You are an automated Shariah-compliant trading bot.
Current Date: {current_time}
Account Summary: {context}
Market Data: {market_data}

# CONSTRAINTS:
1. NON-CRYPTO ONLY.
2. SHARIAH COMPLIANCE:
   - Business: No banking, alcohol, gambling, pork, etc.
   - Financial: Debt < 33%, Interest-bearing cash < 33%.
   - Close non-compliant positions immediately.
3. EXCLUDED / BOYCOTT LIST:
   - The following tickers are STRICTLY FORBIDDEN due to ethical/BDS constraints: {EXCLUDED_TICKERS}
   - If any of these symbols appear in "Account Summary" positions, you MUST generate a CLOSE order for them IMMEDIATELY, regardless of profit/loss.
   - DO NOT BUY these under any circumstances.

# REQUIRED TASKS:
1. **Aggressive Capital Allocation**: Distribute and spend up to **100% of the available buying power** (~$10,000) across your recommendations.
2. Manage positions (HOLD, CLOSE, SL/TP).
3. Recommend 2-4 new Shariah-compliant trades (Day or Long Term) to ensure full capital utilization.

# RESPONSE FORMAT (JSON ONLY):
Respond ONLY with a JSON object:
{{
  "strategy": "...",
  "trades": [
    {{
      "ticker": "...",
      "action": "BUY/SELL/HOLD/CLOSE",
      "trade_type": "DAY TRADE/LONG TERM/CURRENT",
      "compliance_status": "...",
      "quantity": 10,
      "tp_price": 0.0,
      "sl_price": 0.0,
      "rationale": "..."
    }}
  ]
}}
"""

    print("[AI] Consulting Strategy Engine...")
    try:
        response = client.chat.completions.create(
            model=config.ADVISOR_MODEL,
            messages=[
                {"role": "system", "content": "You are a Shariah-compliant automated trader. Respond in raw JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        
        print(f"\nStrategy: {data.get('strategy')}")
        for t in data.get('trades', []):
            print(f"[{t['trade_type']}] {t['ticker']}: {t['action']} (Qty: {t['quantity']}) - {t['rationale']}")
        
        execute_alpaca_trades(data.get('trades', []))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_paper_trader()
