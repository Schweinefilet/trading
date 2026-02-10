---
description: Run the trading bot day simulation
---

To simulate a trading day using historical data:

1. Open a terminal in the `trading-bot` directory:
   ```powershell
   cd e:\trading\trading-bot
   ```

2. Run the simulation script. 

   **Default (Last trading day, $10,000 capital):**
   ```powershell
   // turbo
   python simulate_day.py
   ```

   **Specific options:**
   ```powershell
   # Specify date and capital
   // turbo
   python simulate_day.py --date 2026-02-06 --capital 10000

   # Quiet mode (summary only)
   // turbo
   python simulate_day.py --quiet
   ```

   **Arguments:**
   - `--date YYYY-MM-DD`: Date to simulate (defaults to last weekday)
   - `--capital FLOAT`: Starting capital (defaults to 10000)
   - `--symbols SYM1,SYM2`: Specific symbols to simulate
   - `--quiet`: Suppress trade-by-trade logging
