# AlgoTrader Operational Runbook

## Daily Operations

### 1. Pre-Market Check (9:00 AM EST)
- Ensure `.env` contains valid Alpaca credentials.
- Verify internet connectivity.
- Start the bot: `python main.py --paper` (recommended first).

### 2. During Trading Hours (9:30 AM - 4:00 PM EST)
- **Monitoring**: Keep the Streamlit dashboard open: `streamlit run monitoring/dashboard_st.py`.
- **Alerts**: Monitor Discord/Console for `CRITICAL` Circuit Breaker alerts.
- **Heartbeat**: Check `logs/state.json` or console to ensure the "Last Update" time is current.

### 3. Post-Market Cleanup (4:05 PM EST)
- The bot automatically closes Day Trade positions at 3:55 PM.
- Verify `Daily Summary` in alerts/logs.
- Review `logs/trading_{date}.log` for any `WARNING` level retries or data gaps.

## Incident Response

### Bot Crashes
1. **Restart**: Simply run `python main.py`.
2. **Reconciliation**: The bot will automatically call Alpaca on startup to sync positions. No manual intervention is required for state recovery.

### WebSocket Disconnects
- The bot will attempt to reconnect infinitely with exponential backoff.
- If data is missing for > 2 minutes, a `WARNING` is logged. If persistence fails, kill and restart.

### Panic Stop
If you need to close everything IMMEDIATELY:
1. Run `python scripts/panic_close.py` (if available) OR
2. Log into Alpaca Dashboard and click **"Close All Positions"**.
3. Stop the bot with `Ctrl+C`.

### Wide Spreads / Data Issues
- The `PreTradeValidator` will automatically reject trades if Bid/Ask spread > 0.5% or data is > 5 minutes stale.

## Maintenance
- **Logs**: Periodically clear `logs/` directory or zip old `.log` and `.parquet` files.
- **Updates**: Run `pip install -r requirements.txt` after pulling new code.
