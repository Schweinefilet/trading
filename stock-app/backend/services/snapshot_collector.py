"""
Collects minute-by-minute portfolio snapshots.
This service fetches current 1-minute candle prices and calculates portfolio value.
"""
import threading
import time
from datetime import datetime, timedelta
import pytz
from .data_fetcher import DataFetcher
from models import PortfolioPosition, PortfolioSnapshot, db

class SnapshotCollector:
    def __init__(self, app=None):
        self.app = app
        self.fetcher = DataFetcher()
        self.running = False
        self.thread = None
    
    def start(self, app=None):
        """Start the background snapshot collection thread."""
        if self.running:
            return
        if app:
            self.app = app
        self.running = True
        self.thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.thread.start()
        print("[SnapshotCollector] Started")
    
    def stop(self):
        """Stop the background thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("[SnapshotCollector] Stopped")
    
    def _collect_loop(self):
        """Main collection loop - runs every minute during market hours."""
        while self.running:
            try:
                now = datetime.now(pytz.timezone('US/Eastern'))
                
                # Only collect during market hours: 9:30 AM - 4:00 PM ET, Mon-Fri
                if self._is_market_hours(now):
                    self._collect_snapshot()
                
                # Wait until the next minute boundary
                time.sleep(self._seconds_until_next_minute())
            except Exception as e:
                print(f"[SnapshotCollector] Error: {e}")
                time.sleep(5)
    
    def _is_market_hours(self, dt):
        """Check if given datetime is during US market hours (9:30 AM - 4:00 PM ET)."""
        # Monday=0, Sunday=6
        if dt.weekday() >= 5:  # Saturday or Sunday
            return False
        
        et_time = dt.time()
        market_open = datetime.strptime('09:30', '%H:%M').time()
        market_close = datetime.strptime('16:00', '%H:%M').time()
        
        return market_open <= et_time < market_close
    
    def _seconds_until_next_minute(self):
        """Calculate seconds to wait until next minute boundary."""
        now = datetime.now()
        next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        seconds = (next_minute - now).total_seconds()
        return max(1, seconds)
    
    def _collect_snapshot(self):
        """Fetch current portfolio value and store as snapshot."""
        if not self.app:
            return
        
        try:
            with self.app.app_context():
                # Get all positions
                positions = PortfolioPosition.query.all()
                if not positions:
                    return
                
                # Calculate portfolio value
                total_value = 0
                for pos in positions:
                    try:
                        # Get the latest 1-minute candle
                        hist = self.fetcher.get_history(pos.ticker, period='1d', interval='1m')
                        if hist and len(hist) > 0:
                            # Get the last (most recent) candle
                            latest = hist[-1]
                            current_price = latest.get('Close', latest.get('close', 0))
                            position_value = current_price * pos.shares
                            total_value += position_value
                    except Exception as e:
                        print(f"[SnapshotCollector] Error fetching {pos.ticker}: {e}")
                        continue
                
                # Create timestamp in 'YYYY-MM-DD HH:MM' format
                now = datetime.now()
                timestamp = now.strftime('%Y-%m-%d %H:%M')
                
                # Check if snapshot already exists for this minute
                existing = PortfolioSnapshot.query.filter_by(timestamp=timestamp).first()
                if existing:
                    # Update existing snapshot
                    existing.total_value = total_value
                    existing.fetched_at = datetime.utcnow()
                else:
                    # Create new snapshot
                    snapshot = PortfolioSnapshot(timestamp=timestamp, total_value=total_value)
                    db.session.add(snapshot)
                
                db.session.commit()
                print(f"[SnapshotCollector] Saved snapshot at {timestamp}: ${total_value:.2f}")
        
        except Exception as e:
            print(f"[SnapshotCollector] Error saving snapshot: {e}")
            try:
                db.session.rollback()
            except:
                pass


# Global instance
_collector = None

def get_snapshot_collector():
    global _collector
    if _collector is None:
        _collector = SnapshotCollector()
    return _collector
