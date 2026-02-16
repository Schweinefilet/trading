"""
Performance analytics for backtesting results.
"""
import math
from typing import Dict, List, Optional

import numpy as np


def compute_metrics(trades: list, starting_capital: float, ending_capital: float,
                    equity_curve: list) -> Dict:
    """
    Compute comprehensive backtest metrics.

    Args:
        trades: List of BacktestTrade objects
        starting_capital: Initial capital
        ending_capital: Final capital
        equity_curve: List of equity values over time

    Returns:
        Dict of all performance metrics
    """
    total_return = ending_capital - starting_capital
    total_return_pct = (total_return / starting_capital) * 100 if starting_capital > 0 else 0

    if not trades:
        return {
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "total_trades": 0,
            "win_rate": 0,
            "sharpe_ratio": 0,
            "sortino_ratio": 0,
            "max_drawdown_pct": 0,
            "profit_factor": 0,
            "expectancy": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "avg_holding_period_hours": 0,
            "day_trades": 0,
            "swing_trades": 0,
        }

    pnls = [t.pnl for t in trades]
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]

    win_rate = len(winners) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t.pnl for t in winners]) if winners else 0
    avg_loss = np.mean([t.pnl for t in losers]) if losers else 0
    largest_win = max(pnls) if pnls else 0
    largest_loss = min(pnls) if pnls else 0

    gross_profit = sum(t.pnl for t in winners)
    gross_loss = abs(sum(t.pnl for t in losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Expectancy per trade
    expectancy = np.mean(pnls) if pnls else 0

    # Holding period
    hold_hours = []
    for t in trades:
        try:
            delta = t.exit_time - t.entry_time
            hold_hours.append(delta.total_seconds() / 3600)
        except Exception:
            pass
    avg_holding = np.mean(hold_hours) if hold_hours else 0

    # Day/swing trade counts
    day_trades = sum(1 for t in trades if t.is_day_trade)
    swing_trades = len(trades) - day_trades

    # Drawdown
    max_dd_pct = _max_drawdown(equity_curve)

    # Sharpe ratio (annualized)
    sharpe = _sharpe_ratio(pnls, starting_capital)

    # Sortino ratio
    sortino = _sortino_ratio(pnls, starting_capital)

    # Annualized return
    if trades:
        try:
            first_ts = trades[0].entry_time
            last_ts = trades[-1].exit_time
            days = (last_ts - first_ts).days
            years = max(days / 365.25, 0.01)
            annualized = ((ending_capital / starting_capital) ** (1 / years) - 1) * 100
        except Exception:
            annualized = 0
    else:
        annualized = 0

    # Monthly returns
    monthly = _monthly_returns(trades)

    return {
        "total_return": total_return,
        "total_return_pct": total_return_pct,
        "annualized_return_pct": annualized,
        "total_trades": len(trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_dd_pct,
        "avg_holding_period_hours": avg_holding,
        "day_trades": day_trades,
        "swing_trades": swing_trades,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "monthly_returns": monthly,
        "worst_single_trade": largest_loss,
    }


def _max_drawdown(equity_curve: list) -> float:
    """Calculate maximum drawdown as a percentage."""
    if not equity_curve:
        return 0
    peak = equity_curve[0]
    max_dd = 0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _sharpe_ratio(pnls: list, capital: float, risk_free_annual: float = 0.045) -> float:
    """Annualized Sharpe ratio."""
    if not pnls or capital <= 0:
        return 0
    returns = [p / capital for p in pnls]
    if len(returns) < 2:
        return 0

    # Assume ~250 trading days, ~26 bars per day for 15-min
    trades_per_year = len(returns) * (252 / max(len(returns), 1))
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    if std_ret == 0:
        return 0

    # Annualize
    annualized_return = mean_ret * trades_per_year
    annualized_std = std_ret * math.sqrt(trades_per_year)

    return (annualized_return - risk_free_annual) / annualized_std


def _sortino_ratio(pnls: list, capital: float, risk_free_annual: float = 0.045) -> float:
    """Annualized Sortino ratio (uses downside deviation only)."""
    if not pnls or capital <= 0:
        return 0
    returns = [p / capital for p in pnls]
    if len(returns) < 2:
        return 0

    trades_per_year = len(returns) * (252 / max(len(returns), 1))
    mean_ret = np.mean(returns)
    downside = [r for r in returns if r < 0]

    if not downside:
        return float('inf')

    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0

    annualized_return = mean_ret * trades_per_year
    annualized_downside = downside_std * math.sqrt(trades_per_year)

    return (annualized_return - risk_free_annual) / annualized_downside


def _monthly_returns(trades: list) -> Dict[str, float]:
    """Compute monthly P&L from trades."""
    monthly = {}
    for t in trades:
        try:
            key = t.exit_time.strftime("%Y-%m")
        except Exception:
            continue
        monthly[key] = monthly.get(key, 0) + t.pnl
    return monthly


def print_metrics(metrics: dict, title: str = "BACKTEST RESULTS"):
    """Print formatted metrics table."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  Total Return:         ${metrics['total_return']:+,.2f} ({metrics['total_return_pct']:+.2f}%)")
    print(f"  Annualized Return:    {metrics.get('annualized_return_pct', 0):+.2f}%")
    print(f"  Sharpe Ratio:         {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio:        {metrics['sortino_ratio']:.2f}")
    print(f"  Max Drawdown:         {metrics['max_drawdown_pct']:.2f}%")
    print(f"  {'-' * 56}")
    print(f"  Total Trades:         {metrics['total_trades']}")
    print(f"  Win Rate:             {metrics['win_rate']:.1f}%")
    print(f"  Avg Win:              ${metrics['avg_win']:+,.2f}")
    print(f"  Avg Loss:             ${metrics['avg_loss']:+,.2f}")
    print(f"  Largest Win:          ${metrics['largest_win']:+,.2f}")
    print(f"  Largest Loss:         ${metrics['largest_loss']:+,.2f}")
    print(f"  Profit Factor:        {metrics['profit_factor']:.2f}")
    print(f"  Expectancy:           ${metrics['expectancy']:+,.2f}")
    print(f"  {'-' * 56}")
    print(f"  Day Trades:           {metrics.get('day_trades', 0)}")
    print(f"  Swing Trades:         {metrics.get('swing_trades', 0)}")
    print(f"  Avg Holding:          {metrics.get('avg_holding_period_hours', 0):.1f} hours")
    print(f"{'=' * 60}")

    # Monthly breakdown
    monthly = metrics.get("monthly_returns", {})
    if monthly:
        print(f"\n  Monthly P&L:")
        for month in sorted(monthly.keys()):
            pnl = monthly[month]
            bar = "#" * int(min(abs(pnl) / 50, 20))
            sign = "+" if pnl >= 0 else ""
            color_bar = f"{'(+)' if pnl >= 0 else '(-)'} {bar}"
            print(f"    {month}: {sign}${pnl:,.2f}  {color_bar}")
    print()
