from flask import Blueprint, jsonify
from services.portfolio_analytics import PortfolioAnalytics
from models import PortfolioPosition

analytics_bp = Blueprint('analytics', __name__)

@analytics_bp.route('/portfolio/analytics')
def get_portfolio_analytics():
    positions = PortfolioPosition.query.all()
    pos_dicts = [{'ticker': p.ticker, 'shares': p.shares, 'avg_cost': p.avg_cost} for p in positions]
    
    pa = PortfolioAnalytics(pos_dicts)
    metrics = pa.get_all_metrics()
    return jsonify(metrics)

@analytics_bp.route('/portfolio/correlation')
def get_portfolio_correlation():
    positions = PortfolioPosition.query.all()
    pos_dicts = [{'ticker': p.ticker, 'shares': p.shares, 'avg_cost': p.avg_cost} for p in positions]
    
    pa = PortfolioAnalytics(pos_dicts)
    corr = pa.get_correlation_matrix()
    return jsonify(corr)
