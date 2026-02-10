# 24/5 Automated Day Trading Bot

A fully functional, production-ready automated day trading bot using the Alpaca Markets API. Designed to run 24/5 covering all US market hours, pre-market, and after-hours.

## Features

- **Multi-Indicator Momentum Strategy**: Combines EMA crossover, RSI, VWAP, MACD, and ATR
- **Comprehensive Risk Management**: Position sizing, PDT compliance, daily loss limits, circuit breaker
- **Pre-Market Screener**: Scans for high-momentum stocks before market open
- **Real-Time WebSocket Streaming**: Live market data via Alpaca Data API v2
- **Discord Notifications**: Trade alerts, risk events, and daily summaries
- **Backtesting Engine**: Validate strategies against historical data
- **Production-Ready**: Graceful shutdown, logging, error handling

## Architecture

```
trading-bot/
├── config/
│   ├── settings.py      # Environment variables and configuration
│   └── symbols.py       # Watchlist management
├── core/
│   ├── strategy.py      # Trading strategy (modular, swappable)
│   ├── risk_manager.py  # Position sizing, PDT, circuit breaker
│   ├── order_manager.py # Alpaca order operations
│   └── portfolio.py     # P&L tracking and trade journal
├── data/
│   ├── market_data.py   # Real-time and historical data
│   ├── indicators.py    # Technical indicators (EMA, RSI, VWAP, MACD, ATR)
│   └── screener.py      # Pre-market stock screening
├── utils/
│   ├── logger.py        # Structured JSON logging
│   ├── notifier.py      # Discord webhook notifications
│   └── helpers.py       # Time utilities, retries, formatting
├── bot.py               # Main orchestrator
├── backtest.py          # Backtesting engine
└── requirements.txt
```

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Alpaca Markets account (paper or live)
- Discord webhook URL (optional, for notifications)

### 2. Installation

```bash
cd trading-bot

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Copy `.env.example` to the parent directory's `.env` file (or create `trading-bot/.env`):

```bash
copy .env.example ..\.env
```

Edit `.env` with your credentials:

```env
# Required
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Trading Mode
TRADING_MODE=paper  # Start with paper trading!

# Optional: Discord notifications
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

### 4. Run the Bot

```bash
python bot.py
```

The bot will:
1. Connect to Alpaca and verify credentials
2. Wait for the next trading window
3. Run pre-market screener at 8:00 AM ET
4. Begin trading at 9:45 AM ET (after opening range)
5. Close all positions at 3:55 PM ET
6. Send daily summary and sleep until next trading day

### 5. Backtesting

Test the strategy against historical data:

```bash
python backtest.py --symbol AAPL --start 2024-01-01 --end 2024-06-30 --timeframe 5Min
```

Options:
- `--symbol`: Stock ticker (required)
- `--start`: Start date YYYY-MM-DD (required)
- `--end`: End date YYYY-MM-DD (required)
- `--timeframe`: Bar size - 1Min, 5Min, 1Hour, 1Day (default: 5Min)
- `--capital`: Starting capital (default: 10000)
- `--slippage`: Slippage percentage (default: 0.05%)

## Trading Strategy

### Entry Conditions (Long)
1. 9 EMA crosses above 21 EMA
2. Price is above VWAP
3. RSI is between 30-65
4. MACD histogram is positive and increasing
5. Minimum average volume threshold met (500K default)

### Exit Conditions
- **Stop-loss**: 1.5x ATR below entry
- **Take-profit**: 2.5x ATR above entry (1.67 R:R ratio)
- **Trailing stop**: Activates at 1.5x ATR profit, trails at 1x ATR
- **Time-based**: Close all positions at 3:55 PM ET
- **Signal reversal**: EMA bearish crossover

## Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_POSITION_PCT` | 5% | Maximum single position size |
| `MAX_EXPOSURE_PCT` | 30% | Maximum total portfolio exposure |
| `MAX_DAILY_LOSS_PCT` | 2% | Daily loss limit (halts trading) |
| `PER_TRADE_RISK_PCT` | 1% | Risk per trade for position sizing |
| `MAX_CONCURRENT_POSITIONS` | 6 | Maximum open positions |

### PDT Compliance
- Tracks day trades in rolling 5-business-day window
- Enforces 3 day-trade limit if equity < $25,000
- Set `ACCOUNT_TYPE=cash` to bypass PDT (settlement wait applies)

### Circuit Breaker
- Pauses trading for 30 minutes after 3 consecutive losses
- Automatically resets after timeout

## Switching to Live Trading

> ⚠️ **WARNING**: Live trading involves real money. Test thoroughly with paper trading first!

1. Update `.env`:
```env
TRADING_MODE=live
ALPACA_BASE_URL=https://api.alpaca.markets
```

2. Use live API credentials (different from paper)

3. Start with reduced position sizes:
```env
MAX_POSITION_PCT=0.02
MAX_EXPOSURE_PCT=0.10
```

## Logs and Trade Journal

- **Logs**: `logs/trading_bot.log` (JSON format, daily rotation)
- **Trade Journal**: `trades/trade_log.csv`
- **Backtest Results**: `backtest_results/`

## Docker Deployment

```bash
docker build -t trading-bot .
docker run -d --env-file .env --name trading-bot trading-bot
```

## Customizing the Strategy

The strategy is modular. To create a custom strategy:

1. Create a new class inheriting from `Strategy` in `core/strategy.py`
2. Implement `generate_signals()` method
3. Import and use in `bot.py`

```python
from core.strategy import Strategy, Signal, SignalType

class MyCustomStrategy(Strategy):
    @property
    def name(self) -> str:
        return "MyStrategy"
    
    def generate_signals(self, symbol, bars, position=None) -> Signal:
        # Your logic here
        return Signal(...)
```

## Troubleshooting

**"Failed to connect to Alpaca API"**
- Verify API keys in `.env`
- Check if using correct base URL for paper/live

**"PDT limit reached"**
- Account equity below $25,000 with 3+ day trades
- Wait for rolling window to clear or switch to cash account

**No trades executing**
- Check if market is open
- Verify watchlist symbols are valid
- Review logs for signal generation

## License

MIT License - Use at your own risk. The authors are not responsible for any financial losses.
