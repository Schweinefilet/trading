# Utility Helper Functions
# Time checks, retries, formatting, and common operations

import functools
import time
import asyncio
from datetime import datetime, date, timedelta
from typing import Any, Callable, List, Optional, TypeVar, Union
import pytz
import exchange_calendars as xcals


# Eastern timezone for US markets
ET = pytz.timezone("America/New_York")
UTC = pytz.UTC

# NYSE calendar for market hours
_nyse_calendar = None


def get_nyse_calendar():
    """Get NYSE calendar singleton."""
    global _nyse_calendar
    if _nyse_calendar is None:
        _nyse_calendar = xcals.get_calendar("XNYS")
    return _nyse_calendar


# ============================================================================
# Time and Market Hours Functions
# ============================================================================

def now_et() -> datetime:
    """Get current time in Eastern timezone."""
    return datetime.now(ET)


def now_utc() -> datetime:
    """Get current time in UTC."""
    return datetime.now(UTC)


def to_et(dt: datetime) -> datetime:
    """Convert a datetime to Eastern timezone."""
    if dt.tzinfo is None:
        dt = ET.localize(dt)
    return dt.astimezone(ET)


def today_et() -> date:
    """Get today's date in Eastern timezone."""
    return now_et().date()


def is_market_open() -> bool:
    """Check if the market is currently open."""
    calendar = get_nyse_calendar()
    now = now_et()
    
    # Check if today is a trading day
    if not calendar.is_session(now.date()):
        return False
    
    # Get market open/close times for today
    # session_open_close returns (open_time, close_time) tuple
    open_time, close_time = calendar.session_open_close(now.date())
    market_open = to_et(open_time.to_pydatetime())
    market_close = to_et(close_time.to_pydatetime())
    
    return market_open <= now <= market_close


def is_premarket() -> bool:
    """Check if we're in pre-market hours (4:00 AM - 9:30 AM ET)."""
    now = now_et()
    premarket_open = now.replace(hour=4, minute=0, second=0, microsecond=0)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    
    return premarket_open <= now < market_open


def is_afterhours() -> bool:
    """Check if we're in after-hours (4:00 PM - 8:00 PM ET)."""
    now = now_et()
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    afterhours_close = now.replace(hour=20, minute=0, second=0, microsecond=0)
    
    return market_close <= now <= afterhours_close


def is_trading_day(check_date: Optional[date] = None) -> bool:
    """Check if a given date is a trading day."""
    if check_date is None:
        check_date = today_et()
    calendar = get_nyse_calendar()
    return calendar.is_session(check_date)


def get_next_trading_day(from_date: Optional[date] = None) -> date:
    """Get the next trading day after a given date."""
    if from_date is None:
        from_date = today_et()
    calendar = get_nyse_calendar()
    
    # Start from next day
    check_date = from_date + timedelta(days=1)
    
    # Find next valid session
    while not calendar.is_session(check_date):
        check_date += timedelta(days=1)
    
    return check_date


def get_market_open_time(for_date: Optional[date] = None) -> Optional[datetime]:
    """Get market open time for a specific date."""
    if for_date is None:
        for_date = today_et()
    
    calendar = get_nyse_calendar()
    if not calendar.is_session(for_date):
        return None
    
    open_time, close_time = calendar.session_open_close(for_date)
    return to_et(open_time.to_pydatetime())


def get_market_close_time(for_date: Optional[date] = None) -> Optional[datetime]:
    """Get market close time for a specific date."""
    if for_date is None:
        for_date = today_et()
    
    calendar = get_nyse_calendar()
    if not calendar.is_session(for_date):
        return None
    
    open_time, close_time = calendar.session_open_close(for_date)
    return to_et(close_time.to_pydatetime())


def minutes_until_market_open() -> Optional[int]:
    """Get minutes until market opens. Returns None if market is open or not a trading day."""
    now = now_et()
    market_open = get_market_open_time(now.date())
    
    if market_open is None:
        return None
    
    if now >= market_open:
        return None  # Already open or past
    
    delta = market_open - now
    return int(delta.total_seconds() / 60)


def minutes_until_market_close() -> Optional[int]:
    """Get minutes until market closes. Returns None if market is closed."""
    now = now_et()
    market_close = get_market_close_time(now.date())
    
    if market_close is None:
        return None
    
    if now >= market_close:
        return None  # Already closed
    
    delta = market_close - now
    return int(delta.total_seconds() / 60)


def is_opening_range(minutes: int = 15) -> bool:
    """Check if we're in the opening range period."""
    now = now_et()
    market_open = get_market_open_time(now.date())
    
    if market_open is None:
        return False
    
    opening_range_end = market_open + timedelta(minutes=minutes)
    return market_open <= now < opening_range_end


def is_eod_liquidation_period(minutes_before_close: int = 5) -> bool:
    """Check if we're in the end-of-day liquidation period."""
    remaining = minutes_until_market_close()
    if remaining is None:
        return False
    return remaining <= minutes_before_close


# ============================================================================
# Retry and Error Handling
# ============================================================================

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential: Whether to use exponential backoff
        exceptions: Tuple of exception types to catch
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    # Calculate delay
                    if exponential:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                    else:
                        delay = base_delay
                    
                    print(f"Retry {attempt + 1}/{max_retries} for {func.__name__} after {delay:.1f}s: {e}")
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


def async_retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True,
    exceptions: tuple = (Exception,),
):
    """Async version of retry_with_backoff decorator."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    if exponential:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                    else:
                        delay = base_delay
                    
                    print(f"Retry {attempt + 1}/{max_retries} for {func.__name__} after {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


# ============================================================================
# Formatting Functions
# ============================================================================

def format_price(price: float, decimals: int = 2) -> str:
    """Format price with dollar sign."""
    return f"${price:,.{decimals}f}"


def format_pnl(pnl: float, decimals: int = 2) -> str:
    """Format P&L with sign and dollar sign."""
    return f"${pnl:+,.{decimals}f}"


def format_percent(value: float, decimals: int = 2) -> str:
    """Format percentage with sign."""
    return f"{value:+.{decimals}f}%"


def format_quantity(qty: float) -> str:
    """Format quantity, showing decimals only if fractional."""
    if qty == int(qty):
        return str(int(qty))
    return f"{qty:.4f}"


def format_large_number(num: float) -> str:
    """Format large numbers with K, M, B suffixes."""
    if abs(num) >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif abs(num) >= 1_000:
        return f"{num / 1_000:.2f}K"
    return f"{num:.2f}"


# ============================================================================
# Trading Calculations
# ============================================================================

def calculate_r_multiple(
    entry_price: float,
    exit_price: float,
    stop_loss: float,
    side: str = "BUY"
) -> float:
    """
    Calculate R-multiple (risk-adjusted return).
    
    Args:
        entry_price: Entry price
        exit_price: Exit price
        stop_loss: Stop loss price
        side: 'BUY' or 'SELL'
        
    Returns:
        R-multiple value
    """
    if side == "BUY":
        risk = entry_price - stop_loss
        reward = exit_price - entry_price
    else:  # SELL/SHORT
        risk = stop_loss - entry_price
        reward = entry_price - exit_price
    
    if risk == 0:
        return 0.0
    
    return reward / risk


def calculate_position_value(qty: float, price: float) -> float:
    """Calculate total position value."""
    return qty * price


def calculate_pnl(
    qty: float,
    entry_price: float,
    exit_price: float,
    side: str = "BUY"
) -> float:
    """
    Calculate profit/loss for a trade.
    
    Args:
        qty: Position quantity
        entry_price: Entry price
        exit_price: Exit price
        side: 'BUY' or 'SELL'
        
    Returns:
        P&L in dollars
    """
    if side == "BUY":
        return qty * (exit_price - entry_price)
    else:  # SELL/SHORT
        return qty * (entry_price - exit_price)


def calculate_pnl_percent(
    entry_price: float,
    exit_price: float,
    side: str = "BUY"
) -> float:
    """
    Calculate P&L percentage.
    
    Args:
        entry_price: Entry price
        exit_price: Exit price
        side: 'BUY' or 'SELL'
        
    Returns:
        P&L as percentage
    """
    if side == "BUY":
        return ((exit_price - entry_price) / entry_price) * 100
    else:
        return ((entry_price - exit_price) / entry_price) * 100
