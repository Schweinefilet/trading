from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class StockCache(db.Model):
    __tablename__ = 'stock_cache'
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), index=True)
    data_type = db.Column(db.String(50)) # quote, history, fundamentals, financials
    key_params = db.Column(db.String(255)) # e.g. "period=1y&interval=1d"
    data = db.Column(db.JSON)
    fetched_at = db.Column(db.DateTime, default=datetime.utcnow)

class PortfolioPosition(db.Model):
    __tablename__ = 'portfolio_positions'
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), unique=True, nullable=False)
    shares = db.Column(db.Float, nullable=False)
    avg_cost = db.Column(db.Float, nullable=False)
    date_added = db.Column(db.DateTime, default=datetime.utcnow)

# --- SnapTrade Integration Models ---

class SnapTradeUser(db.Model):
    __tablename__ = 'snaptrade_users'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String, unique=True, nullable=False)          # app-side identifier
    snaptrade_user_id = db.Column(db.String, unique=True, nullable=False) # SnapTrade's userId
    user_secret = db.Column(db.String, nullable=False)                    # encrypted userSecret
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class BrokerageAccount(db.Model):
    __tablename__ = 'brokerage_accounts'
    id = db.Column(db.Integer, primary_key=True)
    snaptrade_user_id = db.Column(db.String, db.ForeignKey('snaptrade_users.snaptrade_user_id'))
    account_id = db.Column(db.String, unique=True, nullable=False)  # SnapTrade's accountId
    brokerage_name = db.Column(db.String, nullable=False)            # e.g. "Webull"
    account_name = db.Column(db.String)                              # e.g. "Individual"
    account_number = db.Column(db.String)                            # masked, e.g. "****1234"
    account_type = db.Column(db.String)                              # e.g. "MARGIN", "CASH"
    is_active = db.Column(db.Boolean, default=True)
    last_synced = db.Column(db.DateTime)
    last_sync_error = db.Column(db.String)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SyncedPosition(db.Model):
    __tablename__ = 'synced_positions'
    id = db.Column(db.Integer, primary_key=True)
    account_id = db.Column(db.String, db.ForeignKey('brokerage_accounts.account_id'))
    ticker = db.Column(db.String, nullable=False)
    symbol_description = db.Column(db.String)
    units = db.Column(db.Float)                   # shares held
    average_purchase_price = db.Column(db.Float)  # cost basis per share
    current_price = db.Column(db.Float)
    current_market_value = db.Column(db.Float)
    open_pnl = db.Column(db.Float)                # unrealized P&L
    open_pnl_percent = db.Column(db.Float)
    currency = db.Column(db.String, default='USD')
    asset_type = db.Column(db.String)             # EQUITY, ETF, etc.
    last_synced = db.Column(db.DateTime, default=datetime.utcnow)

class SyncedAccountBalance(db.Model):
    __tablename__ = 'synced_account_balances'
    id = db.Column(db.Integer, primary_key=True)
    account_id = db.Column(db.String, db.ForeignKey('brokerage_accounts.account_id'))
    currency = db.Column(db.String, default='USD')
    cash = db.Column(db.Float)
    market_value = db.Column(db.Float)      # total value of all positions
    total_value = db.Column(db.Float)       # cash + market_value
    buying_power = db.Column(db.Float)
    maintenance_excess = db.Column(db.Float)
    fetched_at = db.Column(db.DateTime, default=datetime.utcnow)
