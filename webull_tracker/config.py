from pathlib import Path

# User-configurable paths
DOWNLOADS_FOLDER = Path(r"E:\Downloads E")
MASTER_EXCEL_PATH = Path(r"E:\trading\webull_tracker\webull_trading_log.xlsx")

# File detection patterns
WEBULL_PATTERNS = [
    r"TradeHistory.*\.csv$",
    r"OrderHistory.*\.csv$",
    r"trade_history.*\.csv$",
    r"order_history.*\.csv$",
    r"webull.*\.csv$",
    r"Webull_Orders.*\.csv$"
]

# Currency Settings
USD_VND_RATE = 25400 # Default assumption, user can change

import os
from dotenv import load_dotenv

# Load environment variables from parent directory (e:\trading\.env)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

# AI Advisor Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADVISOR_MODEL = "gpt-4o"

# Alpaca API Settings (Paper Trading)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"
