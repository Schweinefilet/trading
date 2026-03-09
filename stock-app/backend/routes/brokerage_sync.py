"""
Brokerage sync routes — all endpoints under /api/brokerage.
Handles SnapTrade user registration, connection portal, sync, and account management.
"""

import logging
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request
from models import db, SnapTradeUser, BrokerageAccount, SyncedPosition, SyncedAccountBalance
from services.cache import CacheService

logger = logging.getLogger(__name__)

brokerage_bp = Blueprint("brokerage", __name__)


def _bust_analytics_cache():
    for tf in ['1d', '1w', '1m', '3mo', '6mo', '1y', '2y']:
        CacheService.delete('__portfolio__', 'portfolio_analytics', f'v1_{tf}')


# ---------------------------------------------------------------------------
# Guard helper
# ---------------------------------------------------------------------------

def _get_snaptrade_user():
    """
    Return (snaptrade_user, user_secret_decrypted) or raise a 404 response tuple.
    Returns None if not found so callers can return the error early.
    """
    from services.snaptrade_service import decrypt_secret
    user = SnapTradeUser.query.first()
    if not user:
        return None, None
    return user, decrypt_secret(user.user_secret)


def _not_registered():
    return jsonify({"error": "not_registered"}), 404


def _fmt_dt(dt):
    """Format a datetime as ISO 8601 UTC string or None."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# POST /register
# ---------------------------------------------------------------------------

@brokerage_bp.route("/register", methods=["POST"])
def register():
    """
    Check if a SnapTrade user already exists.
    If not, register one and persist to DB.
    Returns {already_registered, snaptrade_user_id}.
    """
    from services.snaptrade_service import register_snaptrade_user, encrypt_secret

    existing = SnapTradeUser.query.first()
    if existing:
        return jsonify({
            "already_registered": True,
            "snaptrade_user_id": existing.snaptrade_user_id,
        })

    # Use a stable local user identifier
    local_user_id = "local_user"
    try:
        result = register_snaptrade_user(local_user_id)
    except Exception as e:
        logger.error("SnapTrade registration failed: %s", e)
        _code = _snaptrade_http_code(e)

        # SnapTrade error 1010 / HTTP 400: user already exists on their side.
        # This happens when our DB was wiped but SnapTrade still has the user,
        # or when React StrictMode double-fires the effect and two register
        # requests race. Re-check the DB — the first request may have just
        # committed the row.
        if _code in (400, 1010) or "already exists" in str(e).lower():
            re_check = SnapTradeUser.query.first()
            if re_check:
                return jsonify({
                    "already_registered": True,
                    "snaptrade_user_id": re_check.snaptrade_user_id,
                })
            # User exists in SnapTrade but we lost the secret — auto-recover
            # by deleting the stale SnapTrade user and re-registering.
            from services.snaptrade_service import delete_snaptrade_user
            try:
                delete_snaptrade_user(local_user_id)
                logger.info("Deleted stale SnapTrade user '%s', re-registering...", local_user_id)
            except Exception as del_err:
                logger.warning("Could not delete stale SnapTrade user: %s — attempting re-register anyway", del_err)
            try:
                result = register_snaptrade_user(local_user_id)
            except Exception as rereg_err:
                logger.error("Re-registration failed: %s", rereg_err)
                return jsonify({"error": str(rereg_err)}), 500
            new_user = SnapTradeUser(
                user_id=local_user_id,
                snaptrade_user_id=result["snaptrade_user_id"],
                user_secret=encrypt_secret(result["user_secret"]),
            )
            db.session.add(new_user)
            db.session.commit()
            return jsonify({
                "already_registered": False,
                "snaptrade_user_id": result["snaptrade_user_id"],
            })

        if _code == 401:
            return jsonify({"error": "snaptrade_auth_expired"}), 401
        if _code == 429:
            return jsonify({"error": "rate_limited", "retry_after": 60}), 429
        return jsonify({"error": str(e)}), 500

    new_user = SnapTradeUser(
        user_id=local_user_id,
        snaptrade_user_id=result["snaptrade_user_id"],
        user_secret=encrypt_secret(result["user_secret"]),
    )
    db.session.add(new_user)
    db.session.commit()

    return jsonify({
        "already_registered": False,
        "snaptrade_user_id": result["snaptrade_user_id"],
    })


# ---------------------------------------------------------------------------
# GET /connect-url
# ---------------------------------------------------------------------------

@brokerage_bp.route("/connect-url", methods=["GET"])
def connect_url():
    """
    Generate a SnapTrade connection portal URL.
    Optional ?broker=WEBULL query param to open directly to a broker.
    Returns {url}.
    """
    from services.snaptrade_service import get_connection_portal_url

    user, user_secret = _get_snaptrade_user()
    if user is None:
        return _not_registered()

    broker_slug = request.args.get("broker")  # e.g. "WEBULL"
    try:
        url = get_connection_portal_url(user.snaptrade_user_id, user_secret, broker_slug)
    except Exception as e:
        logger.error("Failed to get connection portal URL: %s", e)
        _code = _snaptrade_http_code(e)
        if _code == 401:
            return jsonify({"error": "snaptrade_auth_expired"}), 401
        if _code == 429:
            return jsonify({"error": "rate_limited", "retry_after": 60}), 429
        return jsonify({"error": str(e)}), 500

    return jsonify({"url": url})


# ---------------------------------------------------------------------------
# POST /sync
# ---------------------------------------------------------------------------

@brokerage_bp.route("/sync", methods=["POST"])
def sync_all():
    """
    Trigger a full sync across all connected brokerage accounts.
    Returns {accounts_synced, positions_synced, errors, synced_at}.
    """
    from services.snaptrade_service import sync_all_accounts

    user, user_secret = _get_snaptrade_user()
    if user is None:
        return _not_registered()

    try:
        result = sync_all_accounts(user.snaptrade_user_id, user_secret, db.session)
    except Exception as e:
        logger.error("Full sync failed: %s", e)
        _code = _snaptrade_http_code(e)
        if _code == 401:
            return jsonify({"error": "snaptrade_auth_expired"}), 401
        if _code == 429:
            return jsonify({"error": "rate_limited", "retry_after": 60}), 429
        return jsonify({"error": str(e)}), 500

    result["synced_at"] = _fmt_dt(datetime.utcnow())
    _bust_analytics_cache()
    return jsonify(result)


# ---------------------------------------------------------------------------
# POST /sync/<account_id>
# ---------------------------------------------------------------------------

@brokerage_bp.route("/sync/<account_id>", methods=["POST"])
def sync_account(account_id):
    """Sync a single specific account."""
    from services.snaptrade_service import sync_single_account

    user, user_secret = _get_snaptrade_user()
    if user is None:
        return _not_registered()

    try:
        result = sync_single_account(user.snaptrade_user_id, user_secret, account_id, db.session)
    except Exception as e:
        logger.error("Single account sync failed for %s: %s", account_id, e)
        _code = _snaptrade_http_code(e)
        if _code == 401:
            return jsonify({"error": "snaptrade_auth_expired"}), 401
        if _code == 429:
            return jsonify({"error": "rate_limited", "retry_after": 60}), 429
        return jsonify({"error": str(e)}), 500

    result["synced_at"] = _fmt_dt(datetime.utcnow())
    _bust_analytics_cache()
    return jsonify(result)


# ---------------------------------------------------------------------------
# GET /accounts
# ---------------------------------------------------------------------------

@brokerage_bp.route("/accounts", methods=["GET"])
def get_accounts():
    """
    Return all active brokerage accounts with last_synced timestamp.
    """
    user, _ = _get_snaptrade_user()
    if user is None:
        return _not_registered()

    accounts = BrokerageAccount.query.filter_by(
        snaptrade_user_id=user.snaptrade_user_id,
        is_active=True,
    ).all()

    return jsonify([
        {
            "account_id":     a.account_id,
            "brokerage_name": a.brokerage_name,
            "account_name":   a.account_name,
            "account_number": a.account_number,
            "account_type":   a.account_type,
            "last_synced":    _fmt_dt(a.last_synced),
            "last_sync_error": a.last_sync_error,
        }
        for a in accounts
    ])


# ---------------------------------------------------------------------------
# GET /positions
# ---------------------------------------------------------------------------

@brokerage_bp.route("/positions", methods=["GET"])
def get_positions():
    """
    Return synced positions.
    ?account_id=<id>  — filter to one account
    Omit account_id   — return all positions with brokerage metadata joined.
    """
    user, _ = _get_snaptrade_user()
    if user is None:
        return _not_registered()

    account_id = request.args.get("account_id")

    if account_id:
        positions = SyncedPosition.query.filter_by(account_id=account_id).all()
        # Get account metadata
        acct = BrokerageAccount.query.filter_by(account_id=account_id).first()
        brokerage_name = acct.brokerage_name if acct else ""
        account_name   = acct.account_name   if acct else ""
        return jsonify([_fmt_position(p, brokerage_name, account_name) for p in positions])

    # All accounts
    active_accounts = BrokerageAccount.query.filter_by(
        snaptrade_user_id=user.snaptrade_user_id,
        is_active=True,
    ).all()
    acct_map = {a.account_id: a for a in active_accounts}

    all_positions = (
        SyncedPosition.query
        .filter(SyncedPosition.account_id.in_(acct_map.keys()))
        .all()
    )

    return jsonify([
        _fmt_position(
            p,
            acct_map[p.account_id].brokerage_name if p.account_id in acct_map else "",
            acct_map[p.account_id].account_name   if p.account_id in acct_map else "",
        )
        for p in all_positions
    ])


def _fmt_position(p, brokerage_name, account_name):
    return {
        "id":                     p.id,
        "account_id":             p.account_id,
        "brokerage_name":         brokerage_name,
        "account_name":           account_name,
        "ticker":                 p.ticker,
        "symbol_description":     p.symbol_description,
        "units":                  p.units,
        "average_purchase_price": p.average_purchase_price,
        "current_price":          p.current_price,
        "current_market_value":   p.current_market_value,
        "open_pnl":               p.open_pnl,
        "open_pnl_percent":       p.open_pnl_percent,
        "currency":               p.currency,
        "asset_type":             p.asset_type,
        "last_synced":            _fmt_dt(p.last_synced),
    }


# ---------------------------------------------------------------------------
# GET /balances
# ---------------------------------------------------------------------------

@brokerage_bp.route("/balances", methods=["GET"])
def get_balances():
    """
    Return latest balance snapshot(s).
    ?account_id=<id>  — single account
    Omit              — all accounts as an array.
    """
    user, _ = _get_snaptrade_user()
    if user is None:
        return _not_registered()

    account_id = request.args.get("account_id")

    if account_id:
        bal = (
            SyncedAccountBalance.query
            .filter_by(account_id=account_id)
            .order_by(SyncedAccountBalance.fetched_at.desc())
            .first()
        )
        if not bal:
            return jsonify(None)
        return jsonify(_fmt_balance(bal))

    # All active accounts — return latest balance per account
    active_accounts = BrokerageAccount.query.filter_by(
        snaptrade_user_id=user.snaptrade_user_id,
        is_active=True,
    ).all()

    result = []
    for acct in active_accounts:
        bal = (
            SyncedAccountBalance.query
            .filter_by(account_id=acct.account_id)
            .order_by(SyncedAccountBalance.fetched_at.desc())
            .first()
        )
        if bal:
            d = _fmt_balance(bal)
            d["brokerage_name"] = acct.brokerage_name
            d["account_name"]   = acct.account_name
            result.append(d)

    return jsonify(result)


def _fmt_balance(b):
    return {
        "account_id":        b.account_id,
        "currency":          b.currency,
        "cash":              b.cash,
        "market_value":      b.market_value,
        "total_value":       b.total_value,
        "buying_power":      b.buying_power,
        "maintenance_excess":b.maintenance_excess,
        "fetched_at":        _fmt_dt(b.fetched_at),
    }


# ---------------------------------------------------------------------------
# GET /summary
# ---------------------------------------------------------------------------

@brokerage_bp.route("/summary", methods=["GET"])
def get_summary():
    """
    Aggregate portfolio summary across ALL connected accounts.
    """
    user, _ = _get_snaptrade_user()
    if user is None:
        return _not_registered()

    active_accounts = BrokerageAccount.query.filter_by(
        snaptrade_user_id=user.snaptrade_user_id,
        is_active=True,
    ).all()

    if not active_accounts:
        return jsonify({
            "total_market_value": 0.0,
            "total_cash": 0.0,
            "total_portfolio_value": 0.0,
            "total_unrealized_pnl": 0.0,
            "total_unrealized_pnl_pct": 0.0,
            "by_brokerage": {},
            "last_synced": None,
            "account_count": 0,
        })

    total_market_value = 0.0
    total_cash = 0.0
    total_unrealized_pnl = 0.0
    total_cost_basis = 0.0
    last_synced = None
    by_brokerage = {}

    for acct in active_accounts:
        # Positions for this account
        positions = SyncedPosition.query.filter_by(account_id=acct.account_id).all()
        acct_mv  = sum((p.current_market_value or 0.0) for p in positions)
        acct_pnl = sum((p.open_pnl or 0.0) for p in positions)
        acct_cost = sum(
            ((p.units or 0.0) * (p.average_purchase_price or 0.0))
            for p in positions
        )

        # Latest balance for cash
        bal = (
            SyncedAccountBalance.query
            .filter_by(account_id=acct.account_id)
            .order_by(SyncedAccountBalance.fetched_at.desc())
            .first()
        )
        acct_cash = bal.cash or 0.0 if bal else 0.0

        total_market_value += acct_mv
        total_cash += acct_cash
        total_unrealized_pnl += acct_pnl
        total_cost_basis += acct_cost

        if acct.last_synced:
            if last_synced is None or acct.last_synced > last_synced:
                last_synced = acct.last_synced

        name = acct.brokerage_name
        if name not in by_brokerage:
            by_brokerage[name] = {"value": 0.0, "pnl": 0.0, "cash": 0.0}
        by_brokerage[name]["value"] += acct_mv + acct_cash
        by_brokerage[name]["pnl"]   += acct_pnl
        by_brokerage[name]["cash"]  += acct_cash

    total_portfolio_value = total_market_value + total_cash
    # Weighted P&L %: open_pnl / cost_basis
    total_unrealized_pnl_pct = (
        (total_unrealized_pnl / total_cost_basis * 100)
        if total_cost_basis > 0 else 0.0
    )

    return jsonify({
        "total_market_value":       round(total_market_value, 2),
        "total_cash":               round(total_cash, 2),
        "total_portfolio_value":    round(total_portfolio_value, 2),
        "total_unrealized_pnl":     round(total_unrealized_pnl, 2),
        "total_unrealized_pnl_pct": round(total_unrealized_pnl_pct, 4),
        "by_brokerage":             {
            k: {kk: round(vv, 2) for kk, vv in v.items()}
            for k, v in by_brokerage.items()
        },
        "last_synced":              _fmt_dt(last_synced),
        "account_count":            len(active_accounts),
    })


# ---------------------------------------------------------------------------
# DELETE /accounts/<account_id>
# ---------------------------------------------------------------------------

@brokerage_bp.route("/accounts/<account_id>", methods=["DELETE"])
def delete_account(account_id):
    """
    Disconnect a brokerage account.
    Calls SnapTrade to remove the connection, then soft-deletes the DB row
    and purges associated positions and balances.
    """
    from services.snaptrade_service import disconnect_account as st_disconnect

    user, user_secret = _get_snaptrade_user()
    if user is None:
        return _not_registered()

    acct = BrokerageAccount.query.filter_by(account_id=account_id).first()
    if not acct:
        return jsonify({"error": "account_not_found"}), 404

    # Try SnapTrade disconnect (best-effort — don't abort if it fails)
    try:
        st_disconnect(user.snaptrade_user_id, user_secret, account_id)
    except Exception as e:
        logger.warning("SnapTrade disconnect call failed for %s: %s — proceeding with local cleanup", account_id, e)

    # Delete associated positions and balances
    SyncedPosition.query.filter_by(account_id=account_id).delete()
    SyncedAccountBalance.query.filter_by(account_id=account_id).delete()

    # Soft-delete the account row
    acct.is_active = False
    db.session.commit()

    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Helper: extract HTTP status code from SnapTrade exceptions
# ---------------------------------------------------------------------------

def _snaptrade_http_code(exc) -> int:
    """Try to extract an HTTP status code from a SnapTrade exception."""
    # SnapTrade SDK exceptions carry the HTTP status in the exception message
    # in the form "(400)\nReason: Bad Request\n..."
    import re
    msg = str(exc)
    m = re.match(r'^\((\d+)\)', msg)
    if m:
        return int(m.group(1))
    for attr in ("status_code", "status", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    msg_lower = msg.lower()
    if "429" in msg or "rate limit" in msg_lower:
        return 429
    if "401" in msg or "unauthorized" in msg_lower or "unauthenticated" in msg_lower:
        return 401
    if "400" in msg or "already exists" in msg_lower:
        return 400
    return 500
