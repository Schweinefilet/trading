from flask import Blueprint, jsonify, request
from services.data_fetcher import DataFetcher
from services.technical import TechnicalService

market_bp = Blueprint('market', __name__)
fetcher = DataFetcher()

@market_bp.route('/stock/<ticker>/quote')
def get_quote(ticker):
    data = fetcher.get_quote(ticker)
    if not data:
        return jsonify({'error': 'Stock not found'}), 404
    return jsonify(data)

@market_bp.route('/stock/<ticker>/history')
def get_history(ticker):
    period = request.args.get('period', '1y')
    interval = request.args.get('interval', '1d')
    
    # Pad period to ensure we have enough historical data for 200-day SMAs
    # length estimates: 1mo~21, 3mo~63, 6mo~126, 1y~252, 2y~504, 5y~1260
    pad_mapping = {
        '1mo': ('1y', 22),
        '3mo': ('1y', 65),
        '6mo': ('2y', 130),
        '1y': ('2y', 253),
        '2y': ('5y', 505),
        '5y': ('10y', 1265)
    }
    
    fetch_period = period
    target_len = None
    if interval == '1d' and period in pad_mapping:
        fetch_period, target_len = pad_mapping[period]
        
    data = fetcher.get_history(ticker, fetch_period, interval)
    if not data:
        return jsonify({'error': 'History not found'}), 404
        
    # Compute technical indicators on the padded dataset
    # By default, use our expanded service with a standard set
    enriched_data = TechnicalService.compute_indicators_from_list(data)
    
    # Trim down to the originally requested period length
    if target_len and len(enriched_data) > target_len:
        enriched_data = enriched_data[-target_len:]
        
    return jsonify(enriched_data)

@market_bp.route('/stock/<ticker>/indicators')
def get_stock_indicators(ticker):
    period = request.args.get('period', '1y')
    interval = request.args.get('interval', '1d')
    indicators_str = request.args.get('indicators', '')
    
    indicators = [i.strip() for i in indicators_str.split(',') if i.strip()]
    
    # Standard padding logic
    pad_mapping = {
        '1mo': ('1y', 22),
        '3mo': ('1y', 65),
        '6mo': ('2y', 130),
        '1y': ('2y', 253),
        '2y': ('5y', 505),
        '5y': ('10y', 1265)
    }
    
    fetch_period = period
    target_len = None
    if interval == '1d' and period in pad_mapping:
        fetch_period, target_len = pad_mapping[period]
        
    data = fetcher.get_history(ticker, fetch_period, interval)
    if not data:
        return jsonify({'error': 'History not found'}), 404
        
    # Compute specific technical indicators
    enriched_data = TechnicalService.compute_indicators_from_list(data, indicators)
    
    # Trim down
    if target_len and len(enriched_data) > target_len:
        enriched_data = enriched_data[-target_len:]
        
    return jsonify(enriched_data)

@market_bp.route('/stock/<ticker>/fundamentals')
def get_fundamentals(ticker):
    data = fetcher.get_fundamentals(ticker)
    if not data:
        return jsonify({'error': 'Fundamentals not found'}), 404
    return jsonify(data)

@market_bp.route('/stock/<ticker>/financials')
def get_financials(ticker):
    data = fetcher.get_financials(ticker)
    if not data:
        return jsonify({'error': 'Financials not found'}), 404
    return jsonify(data)

@market_bp.route('/stock/<ticker>/analyst')
def get_analyst(ticker):
    refresh = str(request.args.get('refresh', '')).lower() in {'1', 'true', 'yes'}
    data = fetcher.get_analyst(ticker, force_refresh=refresh)
    if not data:
        return jsonify({'error': 'Analyst data not found'}), 404
    return jsonify(data)
