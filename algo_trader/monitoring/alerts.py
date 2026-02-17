"""
Alert dispatch: console + file log + optional Discord webhook.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import config


class AlertManager:
    """Dispatch alerts for critical trading events."""

    def __init__(self):
        self._alert_log = config.BASE_DIR / "logs" / "alerts.log"
        self._alert_log.parent.mkdir(parents=True, exist_ok=True)

    def send(self, level: str, title: str, message: str, data: Optional[dict] = None):
        """
        Send an alert via all configured channels.
        """
        timestamp = datetime.now().isoformat()

        # Console
        prefix = "🔴" if level == "CRITICAL" else "🟡" if level == "WARNING" else "🟢"
        print(f"\n{prefix} ALERT [{level}] {title}: {message}\n")

        # File log
        self._log_to_file(timestamp, level, title, message, data)

        # Discord webhook
        if config.DISCORD_WEBHOOK_URL:
            self._send_discord(level, title, message, data)
            
        # Email (Production only / when enabled)
        if config.EMAIL_ENABLE:
            self._send_email(level, title, message, data)

    def _log_to_file(self, timestamp: str, level: str, title: str, message: str, data: Optional[dict]):
        """Append alert to file log."""
        try:
            entry = {
                "timestamp": timestamp,
                "level": level,
                "title": title,
                "message": message,
                "data": data,
            }
            with open(self._alert_log, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"  [Alerts] File log error: {e}")

    def _send_discord(self, level: str, title: str, message: str, data: Optional[dict]):
        """Send alert via Discord webhook."""
        try:
            import aiohttp
            import asyncio

            color = 0xFF0000 if level == "CRITICAL" else 0xFFFF00 if level == "WARNING" else 0x00FF00

            embed = {
                "title": f"[{level}] {title}",
                "description": message,
                "color": color,
                "timestamp": datetime.now().isoformat(),
                "footer": {"text": "AlgoTrader v1.0"}
            }

            if data:
                fields = []
                for k, v in data.items():
                    fields.append({"name": str(k), "value": str(v), "inline": True})
                embed["fields"] = fields[:25]

            payload = {"embeds": [embed]}

            async def _post():
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(config.DISCORD_WEBHOOK_URL, json=payload) as resp:
                            if resp.status not in (200, 204):
                                print(f"  [Alerts] Discord webhook returned {resp.status}")
                except Exception as e:
                    print(f"  [Alerts] Discord POST error: {e}")

            # Safe async execution
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(_post())
                else:
                    loop.run_until_complete(_post())
            except Exception:
                asyncio.run(_post())

        except Exception as e:
            print(f"  [Alerts] Discord error: {e}")

    def _send_email(self, level: str, title: str, message: str, data: Optional[dict]):
        """Send alert via Email (SMTP)."""
        if not all([config.EMAIL_SMTP_HOST, config.EMAIL_USER, config.EMAIL_PASS, config.EMAIL_TO]):
            return

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.utils import formatdate

            body = f"Alert Level: {level}\nTitle: {title}\nMessage: {message}\n"
            if data:
                body += "\nData:\n" + json.dumps(data, indent=2)

            msg = MIMEText(body)
            msg["Subject"] = f"AlgoTrader ALERT: {title}"
            msg["From"] = config.EMAIL_USER
            msg["To"] = config.EMAIL_TO
            msg["Date"] = formatdate(localtime=True)

            with smtplib.SMTP(config.EMAIL_SMTP_HOST, config.EMAIL_SMTP_PORT) as server:
                server.starttls()
                server.login(config.EMAIL_USER, config.EMAIL_PASS)
                server.send_message(msg)

        except Exception as e:
            print(f"  [Alerts] Email error: {e}")

    def performance_telemetry(self, metrics: dict):
        """Log performance metrics for dashboard consumption."""
        try:
            telemetry_file = config.BASE_DIR / "logs" / "telemetry.log"
            metrics["timestamp"] = datetime.now().isoformat()
            with open(telemetry_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception as e:
            print(f"  [Alerts] Telemetry log error: {e}")

    # Convenience methods
    def trade_executed(self, symbol: str, side: str, qty: int, price: float, pnl: float = None):
        if config.ALERT_ON_TRADE:
            msg = f"{side.upper()} {qty} {symbol} @ ${price:.2f}"
            if pnl is not None:
                msg += f" | PnL=${pnl:+,.2f}"
            self.send("INFO", "Trade Executed", msg, {"symbol": symbol, "side": side, "qty": qty, "price": price})

    def circuit_breaker_triggered(self, reason: str, daily_pnl: float):
        if config.ALERT_ON_CIRCUIT_BREAKER:
            self.send("CRITICAL", "Circuit Breaker", reason, {"daily_pnl": f"${daily_pnl:+,.2f}"})

    def error(self, error_msg: str, context: str = ""):
        if config.ALERT_ON_ERROR:
            self.send("WARNING", "Error", error_msg, {"context": context})

    def daily_summary(self, pnl: float, trades: int, equity: float):
        self.send("INFO", "Daily Summary",
                  f"PnL=${pnl:+,.2f} | Trades={trades} | Equity=${equity:,.2f}",
                  {"pnl": f"${pnl:+,.2f}", "trades": trades, "equity": f"${equity:,.2f}"})


# Singleton
alerts = AlertManager()
