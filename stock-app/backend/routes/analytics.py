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
    db,
)
from services import snaptrade_service
from services.snaptrade_service import decrypt_secret, get_account_activities, normalize_activity_record
from models import SnapTradeUser

analytics_bp = Blueprint('analytics', __name__)

# Known anchor: exact portfolio value at end of 2026-03-11
_ANCHOR_DATE  = '2026-03-11'
_ANCHOR_VALUE = 11_563.18

# Absolute floor: portfolio has never dropped below this value in the window.
# Used to clamp inferred history to prevent physically-impossible values.
_VALUE_FLOOR  = 10_500.0

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


def _load_all_activities() -> list:
    """
    Load all relevant activity types across all active SnapTrade accounts:
    BUY, SELL, DEPOSIT, WITHDRAWAL, DIVIDEND.
    Returns a list of normalized dicts with 'activity_type' included.
    """
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
                activity_types='BUY,SELL,DEPOSIT,WITHDRAWAL,DIVIDEND',
            )
        except Exception:
            continue

        for activity in raw:
            marker = normalize_activity_record(activity, account_id=account.account_id)
            if marker:
                activities.append(marker)

    activities.sort(key=lambda a: a.get('occurred_at') or '')
    return activities


def _load_daily_snapshot_eod_series() -> list:
    """Aggregate minute snapshots to daily EOD using the last snapshot per day."""
    rows = PortfolioSnapshot.query.order_by(PortfolioSnapshot.timestamp.asc()).all()
    by_day = {}
    for row in rows:
        ts = str(row.timestamp or '')
        if len(ts) < 10:
            continue
        day = ts[:10]
        prev = by_day.get(day)
        if prev is None or ts > prev['timestamp']:
            by_day[day] = {'timestamp': ts, 'value': float(row.total_value)}
    return [{'date': d, 'value': v['value']} for d, v in sorted(by_day.items())]


def _load_daily_snaptrade_balance_series() -> list:
    """
    Build daily total portfolio values from synced SnapTrade account balances.
    For each account/day, keep latest fetched snapshot, then sum across accounts.
    """
    active_ids = {
        a.account_id
        for a in BrokerageAccount.query.filter_by(is_active=True).all()
    }
    rows = (
        SyncedAccountBalance.query
        .filter(SyncedAccountBalance.account_id.in_(active_ids))
        .order_by(SyncedAccountBalance.fetched_at.asc())
        .all()
    )

    # account_id -> day -> latest total_value for that day
    per_account = {}
    for row in rows:
        if not row.fetched_at:
            continue
        day = row.fetched_at.strftime('%Y-%m-%d')
        account_days = per_account.setdefault(row.account_id, {})
        prev = account_days.get(day)
        if prev is None or row.fetched_at > prev['fetched_at']:
            total_value = row.total_value
            if total_value is None:
                cash = float(row.cash or 0.0)
                mv = float(row.market_value or 0.0)
                total_value = cash + mv
            account_days[day] = {
                'fetched_at': row.fetched_at,
                'total_value': float(total_value),
            }

    # day -> aggregated total across active accounts
    day_totals = {}
    for _, day_map in per_account.items():
        for day, payload in day_map.items():
            day_totals[day] = day_totals.get(day, 0.0) + float(payload['total_value'])

    return [{'date': day, 'value': float(value)} for day, value in sorted(day_totals.items())]


def _log_snaptrade_diagnostics(activities: list):
    """
    Diagnostic logging for activities coverage and SDK capability.
    Logs DEPOSIT event presence/range and available balance/history methods.
    """
    try:
        client = snaptrade_service._get_client()
        methods = [
            name for name in dir(client.account_information)
            if ('balance' in name.lower() or 'history' in name.lower())
        ]
        print(f"[Analytics][Diag] SnapTrade account_information balance/history methods: {methods}")
    except Exception as e:
        print(f"[Analytics][Diag] Could not inspect SnapTrade SDK methods: {e}")

    type_counts = {'BUY': 0, 'SELL': 0, 'DEPOSIT': 0, 'WITHDRAWAL': 0, 'DIVIDEND': 0}
    deposit_dates = []
    for item in activities:
        activity_type = str(item.get('activity_type') or item.get('type') or '').upper()
        if activity_type in type_counts:
            type_counts[activity_type] += 1
        if activity_type == 'DEPOSIT':
            dt = str(item.get('occurred_at') or item.get('trade_date') or '')
            if dt:
                deposit_dates.append(dt)

    deposit_dates.sort()
    dep_start = deposit_dates[0] if deposit_dates else None
    dep_end = deposit_dates[-1] if deposit_dates else None
    print(
        "[Analytics][Diag] Activity counts "
        f"BUY={type_counts['BUY']} SELL={type_counts['SELL']} "
        f"DEPOSIT={type_counts['DEPOSIT']} WITHDRAWAL={type_counts['WITHDRAWAL']} DIVIDEND={type_counts['DIVIDEND']} "
        f"DEPOSIT_RANGE=({dep_start}, {dep_end})"
    )

    # Explicitly query DEPOSIT/WITHDRAWAL only and log returned date coverage per account.
    try:
        user = SnapTradeUser.query.first()
        if user:
            user_secret = decrypt_secret(user.user_secret)
            active_accounts = BrokerageAccount.query.filter_by(
                snaptrade_user_id=user.snaptrade_user_id,
                is_active=True,
            ).all()
            for account in active_accounts:
                raw = get_account_activities(
                    snaptrade_user_id=user.snaptrade_user_id,
                    user_secret=user_secret,
                    account_id=account.account_id,
                    start_date=None,
                    end_date=None,
                    activity_types='DEPOSIT,WITHDRAWAL',
                )
                normalized = [
                    normalize_activity_record(item, account_id=account.account_id)
                    for item in raw
                ]
                normalized = [n for n in normalized if n]
                dates = sorted([
                    str(n.get('occurred_at') or n.get('trade_date') or '')
                    for n in normalized
                    if (n.get('activity_type') or '').upper() in {'DEPOSIT', 'WITHDRAWAL'}
                ])
                print(
                    f"[Analytics][Diag] Account {account.account_id} DEPOSIT/WITHDRAWAL raw={len(raw)} normalized={len(normalized)} "
                    f"range=({dates[0] if dates else None}, {dates[-1] if dates else None})"
                )
    except Exception as e:
        print(f"[Analytics][Diag] Failed DEPOSIT/WITHDRAWAL diagnostics: {e}")


@analytics_bp.route('/portfolio/analytics')
def get_portfolio_analytics():
    label = request.args.get('timeframe', '1y')
    if label not in _TIMEFRAME_MAP:
        label = '1y'
    period, interval = _TIMEFRAME_MAP[label]

    anchor_start_date = request.args.get('anchor_start_date', _ANCHOR_DATE)
    anchor_start_value = request.args.get('anchor_start_value', str(_ANCHOR_VALUE))
    parsed_anchor_date = None
    parsed_anchor_value = None
    if anchor_start_date and anchor_start_value is not None:
        try:
            parsed_anchor_date = datetime.strptime(anchor_start_date, '%Y-%m-%d').date()
            parsed_anchor_value = float(anchor_start_value)
        except (ValueError, TypeError):
            parsed_anchor_date = None
            parsed_anchor_value = None
    value_floor = float(request.args.get('value_floor', str(_VALUE_FLOOR)))

    cache_ticker = '__portfolio__'
    cache_key = f'v3_{label}'
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

    needs_daily_history = interval == '1d' and label != '1d'
    all_activities = []
    cash_history_approximate = False
    has_data_gaps = False
    daily_balance_history = []
    daily_snapshot_history = []

    if needs_daily_history:
        all_activities = _load_all_activities()
        _log_snaptrade_diagnostics(all_activities)

        has_deposit_events = any(
            str(a.get('activity_type') or a.get('type') or '').upper() == 'DEPOSIT'
            for a in all_activities
        )
        cash_history_approximate = not has_deposit_events

        daily_balance_history = _load_daily_snaptrade_balance_series()
        daily_snapshot_history = _load_daily_snapshot_eod_series()

    print(
        f"[Analytics] Positions: {len(pos_dicts)}, Cash: {total_cash}, "
        f"Snapshots: {len(snapshots)}, DailyBalancePoints: {len(daily_balance_history)}, "
        f"DailySnapshotPoints: {len(daily_snapshot_history)}, ActivitiesLoaded: {len(all_activities)}, "
        f"CashApprox: {cash_history_approximate}"
    )

    metrics = pa.get_all_metrics(
        timeframe=period,
        interval=interval,
        total_cash=total_cash,
        start_date=start_date,
        snapshots=snapshots,
        daily_balance_history=daily_balance_history,
        daily_snapshot_history=daily_snapshot_history,
        activity_events=all_activities,
        anchor_start_date=parsed_anchor_date,
        anchor_start_value=parsed_anchor_value,
        value_floor=value_floor,
    )

    metrics['cash_history_approximate'] = bool(cash_history_approximate)
    metrics['has_data_gaps'] = bool(metrics.get('has_data_gaps', False))

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
