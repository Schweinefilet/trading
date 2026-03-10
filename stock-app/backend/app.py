import os
import atexit
import logging
from flask import Flask
from flask_cors import CORS
from models import db
from routes.market import market_bp
from routes.portfolio import portfolio_bp
from routes.analytics import analytics_bp
from routes.brokerage_sync import brokerage_bp

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# Database Configuration
# sqlite file lives at backend/data/stock_cache.db
basedir = os.path.abspath(os.path.dirname(__file__))
db_dir = os.path.join(basedir, 'data')
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

db_path = os.path.join(db_dir, 'stock_cache.db')

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# Register Blueprints
app.register_blueprint(market_bp, url_prefix='/api')
app.register_blueprint(portfolio_bp, url_prefix='/api')
app.register_blueprint(analytics_bp, url_prefix='/api')
app.register_blueprint(brokerage_bp, url_prefix='/api/brokerage')

@app.route('/api/health')
def health_check():
    return {'status': 'healthy'}

with app.app_context():
    db.create_all()
    
    # Handle schema migration: update old PortfolioSnapshot table if needed
    from sqlalchemy import inspect, text
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()
    
    if 'portfolio_snapshots' in tables:
        columns = [col['name'] for col in inspector.get_columns('portfolio_snapshots')]
        # If table has old 'date' column, migrate to new 'timestamp' column
        if 'date' in columns and 'timestamp' not in columns:
            print("[Schema Migration] Migrating portfolio_snapshots table...")
            try:
                # SQLite doesn't support ALTER TABLE to change columns easily
                # So we'll create a new table and copy data
                db.session.execute(text("""
                    CREATE TABLE portfolio_snapshots_new (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT NOT NULL UNIQUE,
                        total_value FLOAT NOT NULL,
                        fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(timestamp)
                    )
                """))
                # Copy date data to timestamp (date becomes timestamp with 00:00)
                db.session.execute(text("""
                    INSERT INTO portfolio_snapshots_new (id, timestamp, total_value, fetched_at)
                    SELECT id, date || ' 00:00', total_value, fetched_at
                    FROM portfolio_snapshots
                """))
                # Drop old table and rename
                db.session.execute(text("DROP TABLE portfolio_snapshots"))
                db.session.execute(text("ALTER TABLE portfolio_snapshots_new RENAME TO portfolio_snapshots"))
                db.session.commit()
                print("[Schema Migration] Successfully migrated portfolio_snapshots table")
            except Exception as e:
                print(f"[Schema Migration] Error during migration: {e}")
                db.session.rollback()
                # If migration fails, just drop and recreate
                try:
                    db.session.execute(text("DROP TABLE IF EXISTS portfolio_snapshots"))
                    db.session.execute(text("DROP TABLE IF EXISTS portfolio_snapshots_new"))
                    db.session.commit()
                    db.create_all()
                    print("[Schema Migration] Recreated portfolio_snapshots table")
                except Exception as e2:
                    print(f"[Schema Migration] Error recreating table: {e2}")
                    db.session.rollback()

# ---------------------------------------------------------------------------
# Portfolio Snapshot Collector (minute-by-minute intraday snapshots)
# ---------------------------------------------------------------------------

from services.snapshot_collector import get_snapshot_collector

# Only start collector if not in testing mode and if we have positions
try:
    snapshot_collector = get_snapshot_collector()
    # Delay start slightly to ensure database is ready
    def start_collector_delayed():
        try:
            import time
            time.sleep(2)
            snapshot_collector.start(app)
        except Exception as e:
            logging.error(f"[SnapshotCollector] Failed to start: {e}")
    
    import threading
    collector_thread = threading.Thread(target=start_collector_delayed, daemon=True)
    collector_thread.start()
    atexit.register(lambda: snapshot_collector.stop())
    logging.info("[SnapshotCollector] Portfolio snapshot collector will start shortly")
except Exception as e:
    logging.error(f"[SnapshotCollector] Initialization error: {e}")

# ---------------------------------------------------------------------------
# Background sync scheduler (APScheduler)
# ---------------------------------------------------------------------------

def scheduled_sync():
    """Runs every 15 minutes to keep brokerage data fresh."""
    with app.app_context():
        try:
            from models import SnapTradeUser
            from services.snaptrade_service import sync_all_accounts, decrypt_secret

            user = SnapTradeUser.query.first()
            if not user:
                logging.info("[SnapTrade Scheduler] No SnapTrade user registered — skipping sync")
                return

            user_secret = decrypt_secret(user.user_secret)
            result = sync_all_accounts(user.snaptrade_user_id, user_secret, db.session)
            logging.info(
                "[SnapTrade Scheduler] Sync complete — accounts: %d, positions: %d, errors: %s",
                result.get("accounts_synced", 0),
                result.get("positions_synced", 0),
                result.get("errors", []),
            )
        except Exception as e:
            logging.error("[SnapTrade Scheduler] Unexpected error: %s", e)


try:
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler()
    scheduler.add_job(scheduled_sync, 'interval', minutes=15)
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown(wait=False))
    logging.info("[SnapTrade Scheduler] Started — syncing every 15 minutes")
except Exception as _sched_err:
    logging.warning("[SnapTrade Scheduler] Could not start scheduler: %s", _sched_err)


if __name__ == '__main__':
    # Default Flask port is 5000, but macOS uses it for AirPlay. Using 5001.
    app.run(host='0.0.0.0', port=5001, debug=True)
