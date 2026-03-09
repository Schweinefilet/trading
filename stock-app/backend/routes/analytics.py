from flask import Blueprint, jsonify, request
from services.portfolio_analytics import PortfolioAnalytics
from services.cache import CacheService
from models import PortfolioPosition

analytics_bp = Blueprint('analytics', __name__)

@analytics_bp.route('/portfolio/analytics')
def get_portfolio_analytics():
    # Accept timeframe parameter (default to 1y)
    timeframe = request.args.get('timeframe', '1y')
    
    # Validate timeframe
    valid_timeframes = ['3mo', '6mo', '1y', '2y']
    if timeframe not in valid_timeframes:
        timeframe = '1y'
    
    cache_ticker = '__portfolio__'
    cache_key = f'v1_{timeframe}'
    cached = CacheService.get(cache_ticker, 'portfolio_analytics', cache_key)
    if cached:
        return jsonify(cached)

    positions = PortfolioPosition.query.all()
    pos_dicts = [{'ticker': p.ticker, 'shares': p.shares, 'avg_cost': p.avg_cost} for p in positions]
    
    pa = PortfolioAnalytics(pos_dicts)
    metrics = pa.get_all_metrics(timeframe=timeframe)
    CacheService.set(cache_ticker, 'portfolio_analytics', metrics, cache_key)
    return jsonify(metrics)

@analytics_bp.route('/portfolio/correlation')
def get_portfolio_correlation():
    positions = PortfolioPosition.query.all()
    pos_dicts = [{'ticker': p.ticker, 'shares': p.shares, 'avg_cost': p.avg_cost} for p in positions]
    
    pa = PortfolioAnalytics(pos_dicts)
    corr = pa.get_correlation_matrix()
    return jsonify(corr)
