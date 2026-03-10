from flask import Blueprint, jsonify, request
from datetime import datetime
from services.portfolio_analytics import PortfolioAnalytics
from services.cache import CacheService
from models import (
    PortfolioPosition,
    SyncedPosition,
    BrokerageAccount,
    SyncedAccountBalance,
    PortfolioSnapshot,
    PortfolioBackfillPoint,
    PortfolioBackfillMeta,
    db,
)
from services.snaptrade_service import decrypt_secret, get_account_activities, normalize_activity_record
from models import SnapTradeUser

analytics_bp = Blueprint('analytics', __name__)

# Maps UI label → (yfinance period, interval)
_TIMEFRAME_MAP = {
    '1d':  ('1d',   '1m'),
    '1w':  ('7d',   '1d'),
    '1m':  ('1mo',  '1d'),
    '3mo': ('3mo',  '1d'),
    '6mo': ('6mo',  '1d'),
    '1y':  ('1y',   '1d'),
    '2y':  ('2y',   '1d'),
    'max': ('max',  '1d'),
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
    """Upsert current minute's portfolio value snapshot for intraday chart persistence."""
    try:
        now = datetime.utcnow()
        timestamp = now.strftime('%Y-%m-%d %H:%M')
        snap = PortfolioSnapshot.query.filter_by(timestamp=timestamp).first()
        if snap:
            snap.total_value = total_value
            snap.fetched_at = now
        else:
            snap = PortfolioSnapshot(timestamp=timestamp, total_value=total_value)
            db.session.add(snap)
        db.session.commit()
    except Exception:
        db.session.rollback()


def _load_snapshots() -> dict:
    """Return all stored snapshots as {timestamp_str: total_value}."""
    try:
        return {s.timestamp: s.total_value for s in PortfolioSnapshot.query.all()}
    except Exception as e:
        print(f"[Analytics] Error loading snapshots: {e}")
        return {}


def _load_trade_activities() -> list:
    """Load normalized BUY/SELL trade activities across all active SnapTrade accounts."""
    user = SnapTradeUser.query.first()
    if not user:
        return []

    active_accounts = BrokerageAccount.query.filter_by(
        snaptrade_user_id=user.snaptrade_user_id,
        is_active=True,
    ).all()
    if not active_accounts:
        return []

    try:
        user_secret = decrypt_secret(user.user_secret)
    except Exception:
        return []

    activities = []
    for account in active_accounts:
        try:
            raw = get_account_activities(
                snaptrade_user_id=user.snaptrade_user_id,
                user_secret=user_secret,
                account_id=account.account_id,
                start_date=None,
                end_date=None,
                activity_types='BUY,SELL',
            )
        except Exception:
            continue

        for activity in raw:
            marker = normalize_activity_record(activity, account_id=account.account_id)
            if marker:
                activities.append(marker)

    activities.sort(key=lambda a: a.get('occurred_at') or '')
    return activities


def _load_backfill_series() -> list:
    """Load persisted all-time reconstructed daily series."""
    rows = PortfolioBackfillPoint.query.order_by(PortfolioBackfillPoint.date.asc()).all()
    return [{'date': r.date, 'value': float(r.total_value)} for r in rows]


def _save_backfill_series(series: list, signature: str):
    """Persist reconstructed all-time daily series (replace existing)."""
    try:
        PortfolioBackfillPoint.query.delete()
        for point in series:
            date_str = str(point.get('date', ''))[:10]
            value = float(point.get('value', 0.0))
            if not date_str:
                continue
            db.session.add(PortfolioBackfillPoint(date=date_str, total_value=value))

        meta = PortfolioBackfillMeta.query.first()
        if meta is None:
            meta = PortfolioBackfillMeta(signature=signature)
            db.session.add(meta)
        else:
            meta.signature = signature
            meta.generated_at = datetime.utcnow()
        db.session.commit()
    except Exception:
        db.session.rollback()


@analytics_bp.route('/portfolio/analytics')
def get_portfolio_analytics():
    label = request.args.get('timeframe', '1y')
    if label not in _TIMEFRAME_MAP:
        label = '1y'
    period, interval = _TIMEFRAME_MAP[label]

    cache_ticker = '__portfolio__'
    cache_key = f'v1_{label}'
    force = request.args.get('force', '0') == '1'
    rebuild_backfill = request.args.get('rebuild_backfill', '0') == '1'
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

    # Daily+ views can use persisted reconstructed backfill (compute once, reuse).
    needs_daily_backfill = interval == '1d' and label != '1d'
    backfill_series = _load_backfill_series() if needs_daily_backfill else []
    trade_activities = []

    if needs_daily_backfill and (not backfill_series or rebuild_backfill):
        trade_activities = _load_trade_activities()
        current_total = (pa.get_basic_metrics().get('total_value', 0.0) or 0.0) + total_cash
        rebuilt = pa.build_trade_backfill_series(
            trade_activities=trade_activities,
            current_total_value=current_total,
            total_cash=total_cash,
        )
        if rebuilt:
            signature = f"trades:{len(trade_activities)}|asof:{datetime.utcnow().strftime('%Y-%m-%d')}"
            _save_backfill_series(rebuilt, signature=signature)
            backfill_series = rebuilt

    print(
        f"[Analytics] Positions: {len(pos_dicts)}, Cash: {total_cash}, "
        f"Snapshots: {len(snapshots)}, BackfillPoints: {len(backfill_series)}, TradesLoaded: {len(trade_activities)}"
    )

    metrics = pa.get_all_metrics(
        timeframe=period,
        interval=interval,
        total_cash=total_cash,
        start_date=start_date,
        snapshots=snapshots,
        trade_activities=trade_activities,
        precomputed_trade_backfill=backfill_series,
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
