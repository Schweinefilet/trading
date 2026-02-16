"""
Console status dashboard.
Prints a formatted table of current positions, P&L, heat, PDT status, and regime.
"""
from datetime import datetime
from typing import Dict, List, Optional


def print_dashboard(
    equity: float,
    daily_pnl: float,
    positions: Dict[str, dict],
    heat_pct: float,
    pdt_status: dict,
    regime: str,
    connected: bool,
):
    """
    Print a formatted console status dashboard.

    Args:
        equity: Current account equity
        daily_pnl: Today's P&L
        positions: Dict of open positions {symbol: {side, qty, entry_price, unrealized_pl, current_price}}
        heat_pct: Portfolio heat as percentage
        pdt_status: PDT tracker status dict
        regime: Current regime state name
        connected: WebSocket connection status
    """
    now = datetime.now().strftime("%H:%M:%S")
    pnl_sign = "+" if daily_pnl >= 0 else ""

    print()
    print("┌──────────────────────────────────────────────────────────────────┐")
    print(f"│  ALGO TRADER DASHBOARD                             {now}  │")
    print("├──────────────────────────────────────────────────────────────────┤")
    print(f"│  Equity: ${equity:>10,.2f}  │  Daily PnL: {pnl_sign}${daily_pnl:>9,.2f}          │")
    print(f"│  Regime: {regime:<12s}  │  Heat: {heat_pct:>5.1f}%                      │")
    print(f"│  PDT: {pdt_status.get('day_trades_used', 0)}/{pdt_status.get('pdt_limit', 3)} used      │  Stream: {'🟢 OK' if connected else '🔴 DOWN'}                     │")
    print("├──────────────────────────────────────────────────────────────────┤")

    if positions:
        print("│  SYM     SIDE   QTY    ENTRY       NOW      UNRL PnL           │")
        print("│  ─────   ────   ───   ──────   ──────────   ─────────           │")
        for sym, pos in positions.items():
            side = pos.get("side", "?")[:4].upper()
            qty = pos.get("qty", 0)
            entry = pos.get("entry_price", 0)
            current = pos.get("current_price", entry)
            upl = pos.get("unrealized_pl", 0)
            upl_sign = "+" if upl >= 0 else ""
            print(f"│  {sym:<6s}  {side:<5s}  {qty:>3d}   ${entry:>7.2f}   ${current:>9.2f}   {upl_sign}${upl:>8.2f}   │")
    else:
        print("│  No open positions                                              │")

    print("└──────────────────────────────────────────────────────────────────┘")
    print()
