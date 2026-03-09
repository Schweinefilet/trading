"""
SnapTrade service layer — the sole file that communicates with the SnapTrade SDK.
All other backend files import from here. Never import the SDK directly elsewhere.

Tested against snaptrade-python-sdk v11.
"""

import os
import base64
import hashlib
import logging
from datetime import datetime

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported brokers registry
# ---------------------------------------------------------------------------
SUPPORTED_BROKERS = {
    "webull":    {"slug": "WEBULL",    "display_name": "Webull",    "color": "#04A6E0"},
    "robinhood": {"slug": "ROBINHOOD", "display_name": "Robinhood", "color": "#00C805"},
}

# ---------------------------------------------------------------------------
# Encryption helpers (user_secret stored encrypted at rest)
# ---------------------------------------------------------------------------

def _get_fernet() -> Fernet:
    """Build a Fernet key derived from SNAPTRADE_CONSUMER_KEY."""
    key_material = os.getenv("SNAPTRADE_CONSUMER_KEY", "default_fallback_key_change_me")
    key_bytes = hashlib.sha256(key_material.encode()).digest()
    fernet_key = base64.urlsafe_b64encode(key_bytes)
    return Fernet(fernet_key)


def encrypt_secret(secret: str) -> str:
    return _get_fernet().encrypt(secret.encode()).decode()


def decrypt_secret(encrypted: str) -> str:
    return _get_fernet().decrypt(encrypted.encode()).decode()


# ---------------------------------------------------------------------------
# SnapTrade client initialization
# ---------------------------------------------------------------------------

def _get_client():
    """Return an initialised SnapTrade client."""
    from snaptrade_client import SnapTrade
    return SnapTrade(
        consumer_key=os.getenv("SNAPTRADE_CONSUMER_KEY"),
        client_id=os.getenv("SNAPTRADE_CLIENT_ID"),
    )


# ---------------------------------------------------------------------------
# Response body normalisation helpers
# ---------------------------------------------------------------------------

def _body(response):
    """Extract .body from a response, returning it directly if already a plain object."""
    return response.body if hasattr(response, "body") else response


def _attr_or_key(obj, *keys, default=None):
    """Try attribute then dict-key access across multiple candidate key names."""
    for key in keys:
        try:
            if isinstance(obj, dict):
                v = obj.get(key)
            else:
                v = getattr(obj, key, None)
            if v is not None:
                return v
        except Exception:
            pass
    return default


def _to_float(v) -> "float | None":
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# User registration
# ---------------------------------------------------------------------------

def register_snaptrade_user(user_id: str) -> dict:
    """
    Register a new user with SnapTrade (v11 SDK).
    Returns {snaptrade_user_id, user_secret}.
    """
    client = _get_client()
    response = client.authentication.register_snap_trade_user(user_id=user_id)
    body = _body(response)
    return {
        "snaptrade_user_id": _attr_or_key(body, "userId", "user_id") or "",
        "user_secret":       _attr_or_key(body, "userSecret", "user_secret") or "",
    }


# ---------------------------------------------------------------------------
# Connection portal URL
# ---------------------------------------------------------------------------

def get_connection_portal_url(snaptrade_user_id: str, user_secret: str, broker_slug: str = None) -> str:
    """
    Generate a SnapTrade hosted connection portal URL (v11 SDK).
    If broker_slug is provided (e.g. "WEBULL"), opens directly to that broker.
    """
    client = _get_client()
    kwargs = dict(
        user_id=snaptrade_user_id,
        user_secret=user_secret,
    )
    if broker_slug:
        kwargs["broker"] = broker_slug

    response = client.authentication.login_snap_trade_user(**kwargs)
    body = _body(response)
    return (
        _attr_or_key(body, "redirectURI", "redirect_uri", "loginLink", "connectionPortalUrl")
        or ""
    )


# ---------------------------------------------------------------------------
# Account fetching
# ---------------------------------------------------------------------------

def get_user_accounts(snaptrade_user_id: str, user_secret: str) -> list:
    """Return all connected brokerage accounts for a user."""
    client = _get_client()
    response = client.account_information.list_user_accounts(
        user_id=snaptrade_user_id,
        user_secret=user_secret,
    )
    body = _body(response)
    return body if isinstance(body, list) else []


# ---------------------------------------------------------------------------
# Position fetching
# ---------------------------------------------------------------------------

def get_account_positions(snaptrade_user_id: str, user_secret: str, account_id: str) -> list:
    """Return all positions for a specific account."""
    client = _get_client()
    response = client.account_information.get_user_account_positions(
        user_id=snaptrade_user_id,
        user_secret=user_secret,
        account_id=account_id,
    )
    body = _body(response)
    return body if isinstance(body, list) else []


# ---------------------------------------------------------------------------
# Balance fetching
# ---------------------------------------------------------------------------

def get_account_balances(snaptrade_user_id: str, user_secret: str, account_id: str):
    """Return balance data for an account (list or dict depending on API)."""
    client = _get_client()
    response = client.account_information.get_user_account_balance(
        user_id=snaptrade_user_id,
        user_secret=user_secret,
        account_id=account_id,
    )
    return _body(response)


# ---------------------------------------------------------------------------
# Account disconnection
# ---------------------------------------------------------------------------

def disconnect_account(snaptrade_user_id: str, user_secret: str, account_id: str) -> bool:
    """
    Remove a connected brokerage account.
    Finds the brokerage authorization for the account and deletes it.
    Returns True on success.
    """
    client = _get_client()

    # List authorizations to find the one matching this account
    auths_response = client.connections.list_brokerage_authorizations(
        user_id=snaptrade_user_id,
        user_secret=user_secret,
    )
    auths = _body(auths_response)
    if not isinstance(auths, list):
        auths = [auths] if auths else []

    auth_id = None
    for auth in auths:
        # Each auth object may reference the account via .account or nested
        auth_accounts = _attr_or_key(auth, "accounts", "brokerage_accounts") or []
        if not isinstance(auth_accounts, list):
            auth_accounts = [auth_accounts]
        for acct in auth_accounts:
            if _attr_or_key(acct, "id") == account_id:
                auth_id = _attr_or_key(auth, "id")
                break
        # Also try direct account reference
        if not auth_id:
            direct_acct_id = _attr_or_key(auth, "account", default={})
            if _attr_or_key(direct_acct_id, "id") == account_id:
                auth_id = _attr_or_key(auth, "id")
        if auth_id:
            break

    if not auth_id:
        # Fallback: try to use account_id directly as authorization_id
        logger.warning(
            "Could not find authorization for account %s — attempting direct removal", account_id
        )
        auth_id = account_id

    client.connections.remove_brokerage_authorization(
        authorization_id=auth_id,
        user_id=snaptrade_user_id,
        user_secret=user_secret,
    )
    return True


# ---------------------------------------------------------------------------
# Account/Position/Balance parsing helpers
# ---------------------------------------------------------------------------

def _parse_account(acct) -> dict:
    """Normalise a raw SnapTrade account object into a flat dict."""
    institution_name = (
        _attr_or_key(acct, "institution_name")
        or _attr_or_key(_attr_or_key(acct, "brokerage"), "name")
        or _attr_or_key(_attr_or_key(acct, "institution"), "name")
        or "Unknown"
    )
    return {
        "account_id":     _attr_or_key(acct, "id", default=""),
        "brokerage_name": str(institution_name),
        "account_name":   _attr_or_key(acct, "name", default=""),
        "account_number": _attr_or_key(acct, "number", "account_number", default=""),
        "account_type":   _attr_or_key(acct, "type", "account_type", default=""),
    }


def _parse_position(pos) -> dict:
    """Normalise a raw SnapTrade position object into a flat dict."""
    symbol_obj = _attr_or_key(pos, "symbol")
    inner = _attr_or_key(symbol_obj, "symbol") if symbol_obj else None
    ticker = (
        _attr_or_key(inner, "ticker", "symbol", "raw_symbol")
        or _attr_or_key(symbol_obj, "ticker", "symbol")
        or _attr_or_key(pos, "ticker", "symbol")
        or "UNKNOWN"
    )
    description = (
        _attr_or_key(inner, "description")
        or _attr_or_key(symbol_obj, "description")
        or ""
    )
    asset_type_obj = _attr_or_key(inner, "type") if inner else None
    asset_type = (
        _attr_or_key(asset_type_obj, "code", "description")
        or _attr_or_key(pos, "asset_type", "security_type")
        or "EQUITY"
    )
    currency_obj = _attr_or_key(inner, "currency") if inner else None
    currency = (
        (_attr_or_key(currency_obj, "code") if not isinstance(currency_obj, str) else currency_obj)
        or _attr_or_key(pos, "currency")
        or "USD"
    )
    return {
        "ticker":                 str(ticker).upper(),
        "symbol_description":     str(description),
        "units":                  _to_float(_attr_or_key(pos, "units", "quantity")),
        "average_purchase_price": _to_float(_attr_or_key(pos, "average_purchase_price", "cost_basis")),
        "current_price":          _to_float(_attr_or_key(pos, "price", "current_price")),
        "current_market_value":   _to_float(_attr_or_key(pos, "current_market_value", "market_value")),
        "open_pnl":               _to_float(_attr_or_key(pos, "open_pnl", "unrealized_pnl")),
        "open_pnl_percent":       _to_float(_attr_or_key(pos, "open_pnl_percent", "unrealized_pnl_percent")),
        "currency":               str(currency),
        "asset_type":             str(asset_type),
    }


def _parse_balance(bal) -> dict:
    """Normalise a raw balance object or list into a flat dict."""
    if isinstance(bal, list):
        merged = {}
        for item in bal:
            for k in ("cash", "buying_power", "maintenance_excess"):
                v = _to_float(_attr_or_key(item, k))
                if v is not None:
                    merged[k] = merged.get(k, 0.0) + v
        first = bal[0] if bal else {}
        c_obj = _attr_or_key(first, "currency")
        merged["currency"] = (
            (_attr_or_key(c_obj, "code") if not isinstance(c_obj, str) else c_obj)
            or "USD"
        )
        merged["market_value"] = _to_float(_attr_or_key(first, "market_value")) or 0.0
        merged["total_value"] = _to_float(_attr_or_key(first, "total_value")) or (
            merged.get("cash", 0.0) + merged.get("market_value", 0.0)
        )
        return merged

    c_obj = _attr_or_key(bal, "currency")
    currency = (
        (_attr_or_key(c_obj, "code") if c_obj and not isinstance(c_obj, str) else c_obj)
        or "USD"
    )
    cash = _to_float(_attr_or_key(bal, "cash")) or 0.0
    mv   = _to_float(_attr_or_key(bal, "market_value")) or 0.0
    tv   = _to_float(_attr_or_key(bal, "total_value")) or (cash + mv)
    return {
        "currency":           str(currency),
        "cash":               cash,
        "market_value":       mv,
        "total_value":        tv,
        "buying_power":       _to_float(_attr_or_key(bal, "buying_power")),
        "maintenance_excess": _to_float(_attr_or_key(bal, "maintenance_excess")),
    }


# ---------------------------------------------------------------------------
# Full sync
# ---------------------------------------------------------------------------

def sync_all_accounts(snaptrade_user_id: str, user_secret: str, db_session) -> dict:
    """
    Fetch all accounts, then for each account fetch positions and balances.
    Upserts everything into SQLite.
    Returns {accounts_synced, positions_synced, errors}.
    """
    from models import BrokerageAccount, SyncedPosition, SyncedAccountBalance

    errors = []
    accounts_synced = 0
    positions_synced = 0

    try:
        raw_accounts = get_user_accounts(snaptrade_user_id, user_secret)
    except Exception as e:
        logger.error("Failed to fetch accounts: %s", e)
        return {"accounts_synced": 0, "positions_synced": 0, "errors": [str(e)]}

    for raw_acct in raw_accounts:
        parsed = _parse_account(raw_acct)
        acct_id = parsed["account_id"]
        if not acct_id:
            continue

        try:
            # ---- Upsert account row ----
            acct_row = db_session.query(BrokerageAccount).filter_by(account_id=acct_id).first()
            if acct_row is None:
                acct_row = BrokerageAccount(
                    account_id=acct_id,
                    snaptrade_user_id=snaptrade_user_id,
                )
                db_session.add(acct_row)
            acct_row.brokerage_name  = parsed["brokerage_name"]
            acct_row.account_name    = parsed["account_name"]
            acct_row.account_number  = parsed["account_number"]
            acct_row.account_type    = parsed["account_type"]
            acct_row.is_active       = True
            acct_row.last_sync_error = None

            now = datetime.utcnow()

            # ---- Positions ----
            raw_positions = get_account_positions(snaptrade_user_id, user_secret, acct_id)
            db_session.query(SyncedPosition).filter_by(account_id=acct_id).delete()
            for raw_pos in raw_positions:
                p = _parse_position(raw_pos)
                db_session.add(SyncedPosition(
                    account_id=acct_id,
                    ticker=p["ticker"],
                    symbol_description=p["symbol_description"],
                    units=p["units"],
                    average_purchase_price=p["average_purchase_price"],
                    current_price=p["current_price"],
                    current_market_value=p["current_market_value"],
                    open_pnl=p["open_pnl"],
                    open_pnl_percent=p["open_pnl_percent"],
                    currency=p["currency"],
                    asset_type=p["asset_type"],
                    last_synced=now,
                ))
                positions_synced += 1

            # ---- Balance ----
            raw_bal = get_account_balances(snaptrade_user_id, user_secret, acct_id)
            bal = _parse_balance(raw_bal)
            db_session.add(SyncedAccountBalance(
                account_id=acct_id,
                currency=bal.get("currency", "USD"),
                cash=bal.get("cash"),
                market_value=bal.get("market_value"),
                total_value=bal.get("total_value"),
                buying_power=bal.get("buying_power"),
                maintenance_excess=bal.get("maintenance_excess"),
                fetched_at=now,
            ))

            acct_row.last_synced = now
            db_session.commit()
            accounts_synced += 1

        except Exception as e:
            db_session.rollback()
            logger.error("Error syncing account %s: %s", acct_id, e)
            try:
                err_row = db_session.query(BrokerageAccount).filter_by(account_id=acct_id).first()
                if err_row:
                    err_row.last_sync_error = str(e)
                    db_session.commit()
            except Exception:
                db_session.rollback()
            errors.append(f"Account {acct_id}: {e}")

    return {
        "accounts_synced":  accounts_synced,
        "positions_synced": positions_synced,
        "errors":           errors,
    }


def sync_single_account(snaptrade_user_id: str, user_secret: str, account_id: str, db_session) -> dict:
    """Sync positions and balance for one specific account."""
    from models import BrokerageAccount, SyncedPosition, SyncedAccountBalance

    acct_row = db_session.query(BrokerageAccount).filter_by(account_id=account_id).first()
    if not acct_row:
        return {
            "accounts_synced": 0,
            "positions_synced": 0,
            "errors": [f"Account {account_id} not found in DB"],
        }

    positions_synced = 0
    errors = []
    try:
        now = datetime.utcnow()

        raw_positions = get_account_positions(snaptrade_user_id, user_secret, account_id)
        db_session.query(SyncedPosition).filter_by(account_id=account_id).delete()
        for raw_pos in raw_positions:
            p = _parse_position(raw_pos)
            db_session.add(SyncedPosition(
                account_id=account_id,
                ticker=p["ticker"],
                symbol_description=p["symbol_description"],
                units=p["units"],
                average_purchase_price=p["average_purchase_price"],
                current_price=p["current_price"],
                current_market_value=p["current_market_value"],
                open_pnl=p["open_pnl"],
                open_pnl_percent=p["open_pnl_percent"],
                currency=p["currency"],
                asset_type=p["asset_type"],
                last_synced=now,
            ))
            positions_synced += 1

        raw_bal = get_account_balances(snaptrade_user_id, user_secret, account_id)
        bal = _parse_balance(raw_bal)
        db_session.add(SyncedAccountBalance(
            account_id=account_id,
            currency=bal.get("currency", "USD"),
            cash=bal.get("cash"),
            market_value=bal.get("market_value"),
            total_value=bal.get("total_value"),
            buying_power=bal.get("buying_power"),
            maintenance_excess=bal.get("maintenance_excess"),
            fetched_at=now,
        ))

        acct_row.last_synced = now
        acct_row.last_sync_error = None
        db_session.commit()

    except Exception as e:
        db_session.rollback()
        logger.error("Error syncing account %s: %s", account_id, e)
        try:
            acct_row.last_sync_error = str(e)
            db_session.commit()
        except Exception:
            db_session.rollback()
        errors.append(str(e))

    return {
        "accounts_synced":  0 if errors else 1,
        "positions_synced": positions_synced,
        "errors":           errors,
    }
