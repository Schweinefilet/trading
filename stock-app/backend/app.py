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
