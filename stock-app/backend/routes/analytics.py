from flask import Blueprint, jsonify, request
from datetime import datetime
from services.portfolio_analytics import PortfolioAnalytics
from services.cache import CacheService
from models import PortfolioPosition, SyncedPosition, BrokerageAccount, SyncedAccountBalance, PortfolioSnapshot, db

analytics_bp = Blueprint('analytics', __name__)

# Maps UI label → (yfinance period, interval)
_TIMEFRAME_MAP = {
    '1d':  ('1d',   '1m'),
    '1w':  ('5d',   '15m'),
    '1m':  ('1mo',  '1d'),
    '3mo': ('3mo',  '1d'),
    '6mo': ('6mo',  '1d'),
    '1y':  ('1y',   '1d'),
    '2y':  ('2y',   '1d'),
}


def _build_unified_positions():
    """
    Merge manual PortfolioPositions with synced brokerage SyncedPositions.
    Positions from the same ticker are combined: shares are summed, avg_cost
    is recalculated as a weighted average.
    Returns a list of {ticker, shares, avg_cost} dicts.
    """
    merged = {}  # ticker -> {shares, cost_basis_total}

    # Manual positions
    for p in PortfolioPosition.query.all():
        t = p.ticker.upper()
        merged[t] = {
            'shares': p.shares,
            'cost_basis_total': p.shares * p.avg_cost,
        }

    # Synced brokerage positions — aggregate across all active accounts
    active_ids = {
        a.account_id
        for a in BrokerageAccount.query.filter_by(is_active=True).all()
    }
    for sp in SyncedPosition.query.filter(SyncedPosition.account_id.in_(active_ids)).all():
        t = sp.ticker.upper()
        units = sp.units or 0.0
        cost = (sp.average_purchase_price or 0.0) * units
        if t in merged:
            merged[t]['shares'] += units
            merged[t]['cost_basis_total'] += cost
        else:
            merged[t] = {'shares': units, 'cost_basis_total': cost}

    result = []
    for ticker, v in merged.items():
        if v['shares'] <= 0:
            continue
        result.append({
            'ticker': ticker,
            'shares': v['shares'],
            'avg_cost': v['cost_basis_total'] / v['shares'],
        })
    return result


def _get_total_cash():
    """Sum latest cash balance across all active brokerage accounts."""
    active_ids = {
        a.account_id
        for a in BrokerageAccount.query.filter_by(is_active=True).all()
    }
    total = 0.0
    for account_id in active_ids:
        bal = (
            SyncedAccountBalance.query
            .filter_by(account_id=account_id)
            .order_by(SyncedAccountBalance.fetched_at.desc())
            .first()
        )
        if bal and bal.cash:
            total += bal.cash
    return total


def _get_start_date():
    """Earliest date any position was added (manual or via brokerage connection)."""
    candidates = []
    for p in PortfolioPosition.query.all():
        if p.date_added:
            candidates.append(p.date_added)
    for a in BrokerageAccount.query.filter_by(is_active=True).all():
        if a.created_at:
            candidates.append(a.created_at)
    return min(candidates) if candidates else None


def _save_snapshot(total_value: float):
    """Upsert today's portfolio value snapshot for historical chart persistence."""
    try:
        today = datetime.utcnow().strftime('%Y-%m-%d')
        snap = PortfolioSnapshot.query.filter_by(date=today).first()
        if snap:
            snap.total_value = total_value
            snap.fetched_at = datetime.utcnow()
        else:
            snap = PortfolioSnapshot(date=today, total_value=total_value)
            db.session.add(snap)
        db.session.commit()
    except Exception:
        db.session.rollback()


def _load_snapshots() -> dict:
    """Return all stored snapshots as {date_str: total_value}."""
    return {s.date: s.total_value for s in PortfolioSnapshot.query.all()}


@analytics_bp.route('/portfolio/analytics')
def get_portfolio_analytics():
    label = request.args.get('timeframe', '1y')
    if label not in _TIMEFRAME_MAP:
        label = '1y'
    period, interval = _TIMEFRAME_MAP[label]

    cache_ticker = '__portfolio__'
    cache_key = f'v1_{label}'
    force = request.args.get('force', '0') == '1'
    if force:
        CacheService.delete(cache_ticker, 'portfolio_analytics', cache_key)
    else:
        cached = CacheService.get(cache_ticker, 'portfolio_analytics', cache_key)
        if cached:
            return jsonify(cached)

    pos_dicts = _build_unified_positions()
    total_cash = _get_total_cash()
    start_date = _get_start_date()
    snapshots = _load_snapshots()

    pa = PortfolioAnalytics(pos_dicts)
    metrics = pa.get_all_metrics(
        timeframe=period,
        interval=interval,
        total_cash=total_cash,
        start_date=start_date,
        snapshots=snapshots,
    )

    # Persist today's total value for future historical accuracy.
    _save_snapshot(metrics['summary']['total_value'])

    CacheService.set(cache_ticker, 'portfolio_analytics', metrics, cache_key)
    return jsonify(metrics)


@analytics_bp.route('/portfolio/correlation')
def get_portfolio_correlation():
    pos_dicts = _build_unified_positions()
    pa = PortfolioAnalytics(pos_dicts)
    corr = pa.get_correlation_matrix()
    return jsonify(corr)
