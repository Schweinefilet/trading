import re
from collections import defaultdict

log_file = "output_audit_1h_phase112_stabilization.txt"
trades = []

# Regex to match exit lines
# 2025-01-10 16:00    AAPL   LONG   $ 234.89 $ 236.22    864 $202944.96 +$  1144.78 mean_reversion_exit
pattern = re.compile(r'(\d{4}-\d{2}-\d{2}).*Total PnL=\$([-]?\d+\.\d+)\s+\((.*)\)')

with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            date, pnl, reason = match.groups()
            pnl = float(pnl)
            
            # Categorize strategy
            strat = "mean_reversion" if "mean_reversion" in reason else "trend_following"
            trades.append({"date": date, "pnl": pnl, "strat": strat, "reason": reason})

# 1. Strategy Breakdown
stats = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0})
for t in trades:
    s = t["strat"]
    stats[s]["count"] += 1
    stats[s]["pnl"] += t["pnl"]
    if t["pnl"] > 0:
        stats[s]["wins"] += 1

print("--- Strategy Breakdown ---")
for s, data in stats.items():
    wr = (data["wins"] / data["count"] * 100) if data["count"] > 0 else 0
    print(f"{s}: Count={data['count']}, PnL=${data['pnl']:.2f}, WinRate={wr:.1f}%")

# 2. 2025 Breakdown
print("\n--- 2025 Diagnostics ---")
trades_2025 = [t for t in trades if t["date"].startswith("2025")]
pnl_2025 = sum(t["pnl"] for t in trades_2025)
print(f"Total Trades 2025: {len(trades_2025)}")
print(f"Total PnL 2025: ${pnl_2025:.2f}")

# 3. Max Consecutive Losses
max_consecutive = 0
current_consecutive = 0
for t in trades:
    if t["pnl"] < 0:
        current_consecutive += 1
        max_consecutive = max(max_consecutive, current_consecutive)
    else:
        current_consecutive = 0

print(f"\nMax Consecutive Losses: {max_consecutive}")
