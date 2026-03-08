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
