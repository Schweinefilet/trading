"""
Structured logging using loguru.
Rotated files, 30-day retention, standardized event logging.
"""
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger as _loguru_logger

from config.settings import config


# Remove default handler
_loguru_logger.remove()

# Console handler (colorized)
_loguru_logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
    level=config.LOG_LEVEL,
    colorize=True,
)

# File handler (rotated)
log_dir = config.BASE_DIR / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

_loguru_logger.add(
    str(log_dir / "trading_{time:YYYY-MM-DD}.log"),
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {module}:{function}:{line} | {message}",
    level="DEBUG",
    rotation="50 MB",
    retention="30 days",
    compression="zip",
)

# Export as 'logger'
logger = _loguru_logger


def log_startup(account_info: dict, positions: int, regime: str):
    """Log startup summary."""
    logger.info("=" * 60)
    logger.info("ALGO TRADER — STARTUP")
    logger.info(f"  Equity: ${account_info.get('equity', 0):,.2f}")
    logger.info(f"  Positions: {positions}")
    logger.info(f"  Regime: {regime}")
    logger.info(f"  Mode: {'PAPER' if config.PAPER_TRADING else 'LIVE'}")
    logger.info("=" * 60)


def log_signal(symbol: str, direction: str, strength: float, indicators: dict):
    """Log a generated signal."""
    logger.info(
        f"SIGNAL | {symbol} | {direction} | strength={strength:.3f} | "
        f"RSI={indicators.get('rsi', 0):.1f} | ADX={indicators.get('adx', 0):.1f} | "
        f"Vol={indicators.get('volume_ratio', 0):.1f}x"
    )


def log_trade_entry(symbol: str, direction: str, shares: int, price: float,
                    stop: float, target: float, value: float, reason: str):
    """Log a trade entry."""
    logger.info(
        f"TRADE_ENTRY | {symbol} | {direction} | {shares} shares @ ${price:.2f} | "
        f"SL=${stop:.2f} | TP=${target:.2f} | Value=${value:,.2f} | {reason}"
    )


def log_trade_exit(symbol: str, direction: str, shares: int, price: float,
                   pnl: float, pnl_pct: float, hold_time: str, reason: str):
    """Log a trade exit."""
    logger.info(
        f"TRADE_EXIT | {symbol} | {direction} | {shares} shares @ ${price:.2f} | "
        f"PnL=${pnl:+,.2f} ({pnl_pct:+.2f}%) | Hold={hold_time} | {reason}"
    )


def log_pdt_status(used: int, remaining: int, total: int):
    """Log PDT status."""
    logger.info(f"PDT_STATUS | Used={used}/{total} | Remaining={remaining}")


def log_circuit_breaker(trigger: str, values: dict, action: str):
    """Log circuit breaker event."""
    logger.critical(
        f"CIRCUIT_BREAKER | {trigger} | Daily=${values.get('daily_pnl', 0):+,.2f} | "
        f"Weekly=${values.get('weekly_pnl', 0):+,.2f} | Action={action}"
    )


def log_daily_summary(total_pnl: float, trades: int, wins: int, losses: int,
                      heat_pct: float, equity: float):
    """Log daily summary."""
    logger.info("=" * 60)
    logger.info(f"DAILY_SUMMARY | PnL=${total_pnl:+,.2f} | Trades={trades} | "
                f"W={wins} L={losses} | Heat={heat_pct:.1f}% | Equity=${equity:,.2f}")
    logger.info("=" * 60)


def log_heartbeat(positions: int, daily_pnl: float, connected: bool):
    """Log heartbeat (every 60 seconds)."""
    logger.debug(
        f"HEARTBEAT | Alive | Positions={positions} | "
        f"DailyPnL=${daily_pnl:+,.2f} | Stream={'OK' if connected else 'DISCONNECTED'}"
    )
