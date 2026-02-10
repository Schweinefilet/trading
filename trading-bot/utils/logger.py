# Structured Logging Module
# JSON-formatted logging with daily rotation

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    """Custom formatter that outputs JSON-structured log entries."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, "symbol"):
            log_entry["symbol"] = record.symbol
        if hasattr(record, "order_id"):
            log_entry["order_id"] = record.order_id
        if hasattr(record, "trade_data"):
            log_entry["trade_data"] = record.trade_data
        if hasattr(record, "signal"):
            log_entry["signal"] = record.signal
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors for console output."""
        color = self.COLORS.get(record.levelname, self.RESET)
        
        # Format timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Build message
        msg = f"{color}[{timestamp}] [{record.levelname:8}]{self.RESET} "
        
        # Add symbol if present
        if hasattr(record, "symbol"):
            msg += f"[{record.symbol}] "
        
        msg += record.getMessage()
        
        # Add exception if present
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"
        
        return msg


def setup_logger(
    name: str = "trading_bot",
    log_dir: Optional[Path] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files (default: ./logs)
        console_level: Logging level for console output
        file_level: Logging level for file output
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # Set up log directory
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    
    # File handler with JSON formatting and daily rotation
    log_file = log_dir / "trading_bot.log"
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=30,  # Keep 30 days of logs
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(JsonFormatter())
    file_handler.suffix = "%Y-%m-%d"
    logger.addHandler(file_handler)
    
    return logger


class TradeLogger:
    """Specialized logger for trade-specific logging."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize trade logger."""
        self.logger = logger or setup_logger()
    
    def log_signal(
        self, 
        symbol: str, 
        signal_type: str, 
        confidence: float,
        indicators: Dict[str, Any]
    ) -> None:
        """Log a trading signal."""
        self.logger.info(
            f"Signal: {signal_type} (confidence: {confidence:.2f})",
            extra={
                "symbol": symbol,
                "signal": {
                    "type": signal_type,
                    "confidence": confidence,
                    "indicators": indicators
                }
            }
        )
    
    def log_order(
        self,
        symbol: str,
        order_id: str,
        side: str,
        qty: float,
        order_type: str,
        price: Optional[float] = None,
        status: str = "submitted"
    ) -> None:
        """Log an order event."""
        msg = f"Order {status}: {side} {qty} @ {price if price else 'MKT'}"
        self.logger.info(
            msg,
            extra={
                "symbol": symbol,
                "order_id": order_id,
                "trade_data": {
                    "side": side,
                    "qty": qty,
                    "order_type": order_type,
                    "price": price,
                    "status": status
                }
            }
        )
    
    def log_fill(
        self,
        symbol: str,
        order_id: str,
        side: str,
        qty: float,
        fill_price: float,
        pnl: Optional[float] = None
    ) -> None:
        """Log an order fill."""
        msg = f"Fill: {side} {qty} @ ${fill_price:.2f}"
        if pnl is not None:
            msg += f" | P&L: ${pnl:+.2f}"
        
        self.logger.info(
            msg,
            extra={
                "symbol": symbol,
                "order_id": order_id,
                "trade_data": {
                    "side": side,
                    "qty": qty,
                    "fill_price": fill_price,
                    "pnl": pnl
                }
            }
        )
    
    def log_risk_event(self, event_type: str, details: str) -> None:
        """Log a risk management event."""
        self.logger.warning(f"Risk Event [{event_type}]: {details}")
    
    def log_error(self, message: str, exc_info: bool = False) -> None:
        """Log an error."""
        self.logger.error(message, exc_info=exc_info)


# Default logger instance
logger = setup_logger()
trade_logger = TradeLogger(logger)
