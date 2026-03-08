from flask import Blueprint, jsonify, request
from models import PortfolioPosition, db

portfolio_bp = Blueprint('portfolio', __name__)

@portfolio_bp.route('/portfolio')
def get_portfolio():
    positions = PortfolioPosition.query.all()
    res = []
    for p in positions:
        res.append({
            'ticker': p.ticker,
            'shares': p.shares,
            'avg_cost': p.avg_cost,
            'date_added': p.date_added.strftime('%Y-%m-%d')
        })
    return jsonify(res)

@portfolio_bp.route('/portfolio/add', methods=['POST'])
def add_position():
    data = request.json
    ticker = data.get('ticker')
    shares = data.get('shares')
    avg_cost = data.get('avg_cost')
    
    if not ticker or shares is None or avg_cost is None:
        return jsonify({'error': 'Missing data'}), 400
        
    pos = PortfolioPosition.query.filter_by(ticker=ticker.upper()).first()
    if pos:
        # Update existing (weighted average cost)
        current_shares = pos.shares
        new_shares = float(shares)
        total_shares = current_shares + new_shares
        
        if total_shares > 0:
            pos.avg_cost = ((pos.avg_cost * current_shares) + (float(avg_cost) * new_shares)) / total_shares
        
        pos.shares = total_shares
    else:
        # Create new
        pos = PortfolioPosition(
            ticker=ticker.upper(),
            shares=float(shares),
            avg_cost=float(avg_cost)
        )
        db.session.add(pos)
        
    db.session.commit()
    return jsonify({'success': True})

@portfolio_bp.route('/portfolio/<ticker>', methods=['DELETE', 'PUT'])
def manage_position(ticker):
    pos = PortfolioPosition.query.filter_by(ticker=ticker.upper()).first()
    if not pos:
        return jsonify({'error': 'Position not found'}), 404
        
    if request.method == 'DELETE':
        db.session.delete(pos)
        db.session.commit()
        return jsonify({'success': True})
        
    if request.method == 'PUT':
        data = request.json
        pos.shares = float(data.get('shares', pos.shares))
        pos.avg_cost = float(data.get('avg_cost', pos.avg_cost))
        db.session.commit()
        return jsonify({'success': True})
