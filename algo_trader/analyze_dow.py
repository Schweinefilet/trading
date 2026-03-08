import re
from datetime import datetime
import pandas as pd
from collections import defaultdict

# Path to the log file
log_file = "outputphase2_verbose.txt"

# Regex to capture Exit lines with PnL
# Example line:
# 2026-02-12 16:15:00+00:00 | EXIT LONG 6 VRT @ $242.97 | Remaining=$5.02 | Partial=$43.57 | Total PnL=$48.60 (break_even+partial)
# We need the timestamp and the Total PnL.
pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}) .*\| EXIT .* Total PnL=\$([-+]?[\d,]+\.\d{2})")

stats = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0, "losses": 0})

try:
    with open(log_file, "r") as f:
        for line in f:
            if "Total PnL=$" in line and "EXIT" in line:
                try:
                    # Parse Date
                    # 2026-02-12 16:15:00+00:00 ...
                    date_part = line.split()[0] # 2026-02-12
                    
                    # Parse PnL
                    # ... | Total PnL=$-3.36 (stop_loss)
                    # Split by "Total PnL=$"
                    pnl_part = line.split("Total PnL=$")[1]
                    # Get the number part before the next space or paren
                    # "-3.36 (stop_loss)" -> "-3.36"
                    pnl_str = pnl_part.split()[0].replace(",", "")
                    if "(" in pnl_str: # Clean trailing paren if tight
                        pnl_str = pnl_str.split("(")[0]
                    
                    pnl = float(pnl_str)
                    
                    dt = datetime.strptime(date_part, "%Y-%m-%d")
                    weekday = dt.weekday() # 0-6
                    day_name = dt.strftime("%A")
                    
                    stats[day_name]["pnl"] += pnl
                    stats[day_name]["trades"] += 1
                    if pnl > 0:
                        stats[day_name]["wins"] += 1
                    elif pnl < 0:
                        stats[day_name]["losses"] += 1
                except Exception as e:
                    print(f"DEBUG: Parse error on line: {line.strip()} | Error: {e}")

    # Print results
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    print(f"{'Day':<10} | {'Trades':<8} | {'PnL':<15} | {'Win Rate':<8} | {'Avg PnL':<10}")
    print("-" * 65)
    
    total_pnl = 0
    total_trades = 0
    
    for day in days:
        s = stats[day]
        trades = s["trades"]
        pnl = s["pnl"]
        wins = s["wins"]
        win_rate = (wins / trades * 100) if trades > 0 else 0
        avg_pnl = (pnl / trades) if trades > 0 else 0
        
        total_pnl += pnl
        total_trades += trades
        
        print(f"{day:<10} | {trades:<8} | ${pnl:<14,.2f} | {win_rate:6.1f}% | ${avg_pnl:<9.2f}")
        
    print("-" * 65)
    print(f"TOTAL      | {total_trades:<8} | ${total_pnl:<14,.2f}")

except FileNotFoundError:
    print(f"Error: {log_file} not found.")
