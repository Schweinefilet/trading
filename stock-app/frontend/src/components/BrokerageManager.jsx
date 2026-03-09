/**
 * BrokerageManager — account connection, sync, and management UI.
 * Rendered as a section on the Portfolio page.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { Plus, RefreshCw, X, Wifi, WifiOff, AlertTriangle, Check } from 'lucide-react';

const BROKER_META = {
    Webull:    { color: '#04A6E0', bg: 'rgba(4,166,224,0.15)',  initials: 'WB' },
    Robinhood: { color: '#00C805', bg: 'rgba(0,200,5,0.15)',   initials: 'RH' },
};

const DEFAULT_META = { color: '#9ca3af', bg: 'rgba(156,163,175,0.15)', initials: '??' };

function getBrokerMeta(name = '') {
    return BROKER_META[name] || DEFAULT_META;
}

// ---------------------------------------------------------------------------
// Toast
// ---------------------------------------------------------------------------
function Toast({ message, type, onClose }) {
    useEffect(() => {
        const t = setTimeout(onClose, 4000);
        return () => clearTimeout(t);
    }, [onClose]);

    const colors = {
        success: { bg: 'rgba(0,200,5,0.15)',   border: 'rgba(0,200,5,0.4)',   text: '#4ade80' },
        error:   { bg: 'rgba(239,68,68,0.15)',  border: 'rgba(239,68,68,0.4)', text: '#f87171' },
        warning: { bg: 'rgba(251,191,36,0.15)', border: 'rgba(251,191,36,0.4)',text: '#fbbf24' },
        info:    { bg: 'rgba(59,130,246,0.15)', border: 'rgba(59,130,246,0.4)',text: '#60a5fa' },
    };
    const c = colors[type] || colors.info;

    return (
        <div
            className="fixed bottom-6 right-6 z-50 flex items-center gap-3 px-4 py-3 rounded-xl shadow-xl"
            style={{
                background: c.bg,
                border: `1px solid ${c.border}`,
                backdropFilter: 'blur(12px)',
                WebkitBackdropFilter: 'blur(12px)',
                maxWidth: '360px',
            }}
        >
            <span className="text-sm font-medium flex-1" style={{ color: c.text }}>{message}</span>
            <button onClick={onClose} style={{ color: c.text, opacity: 0.7 }}>
                <X className="h-4 w-4" />
            </button>
        </div>
    );
}

// ---------------------------------------------------------------------------
// Spinner
// ---------------------------------------------------------------------------
function Spinner({ size = 14 }) {
    return (
        <div
            className="animate-spin rounded-full flex-shrink-0"
            style={{
                width: size, height: size,
                border: '2px solid rgba(255,255,255,0.2)',
                borderTopColor: 'var(--accent)',
            }}
        />
    );
}

// ---------------------------------------------------------------------------
// Relative time helper
// ---------------------------------------------------------------------------
function relativeTime(isoString) {
    if (!isoString) return 'never';
    const diff = (Date.now() - new Date(isoString)) / 1000;
    if (diff < 60) return 'just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
}

// ---------------------------------------------------------------------------
// AddAccountModal
// ---------------------------------------------------------------------------
function AddAccountModal({ onClose, onGetUrl, onDoneLink }) {
    const [loading, setLoading] = useState(null); // broker slug being loaded

    const handleBroker = async (slug) => {
        setLoading(slug);
        try {
            const url = await onGetUrl(slug);
            if (url) window.open(url, '_blank', 'noopener,noreferrer');
        } catch {
            // silently fall through — parent handles errors
        } finally {
            setLoading(null);
        }
    };

    const handleAllBrokers = async () => {
        setLoading('ALL');
        try {
            const url = await onGetUrl(null);
            if (url) window.open(url, '_blank', 'noopener,noreferrer');
        } finally {
            setLoading(null);
        }
    };

    return createPortal(
        <div
            className="fixed inset-0 flex items-center justify-center z-50 p-4"
            style={{ background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)' }}
            onClick={onClose}
        >
            <div
                className="glass w-full max-w-sm p-6 space-y-5"
                style={{ borderRadius: '20px' }}
                onClick={e => e.stopPropagation()}
            >
                <div className="flex justify-between items-center">
                    <h3 className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>Connect a Broker</h3>
                    <button onClick={onClose} style={{ color: 'var(--text-tertiary)' }}>
                        <X className="h-5 w-5" />
                    </button>
                </div>

                <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                    You'll be redirected to SnapTrade's secure portal to link your account.
                    Your credentials are never shared with this app.
                </p>

                <div className="space-y-3">
                    {/* Webull */}
                    <button
                        onClick={() => handleBroker('WEBULL')}
                        disabled={loading !== null}
                        className="w-full flex items-center gap-3 px-4 py-3 rounded-xl font-bold transition-opacity disabled:opacity-60"
                        style={{ background: 'rgba(4,166,224,0.12)', border: '1px solid rgba(4,166,224,0.3)', color: '#04A6E0' }}
                    >
                        {loading === 'WEBULL' ? <Spinner /> : (
                            <span className="w-6 h-6 rounded flex items-center justify-center text-[10px] font-black" style={{ background: '#04A6E0', color: '#000' }}>WB</span>
                        )}
                        Connect Webull
                    </button>

                    {/* Robinhood */}
                    <button
                        onClick={() => handleBroker('ROBINHOOD')}
                        disabled={loading !== null}
                        className="w-full flex items-center gap-3 px-4 py-3 rounded-xl font-bold transition-opacity disabled:opacity-60"
                        style={{ background: 'rgba(0,200,5,0.12)', border: '1px solid rgba(0,200,5,0.3)', color: '#00C805' }}
                    >
                        {loading === 'ROBINHOOD' ? <Spinner /> : (
                            <span className="w-6 h-6 rounded flex items-center justify-center text-[10px] font-black" style={{ background: '#00C805', color: '#000' }}>RH</span>
                        )}
                        Connect Robinhood
                    </button>

                    {/* All brokers */}
                    <button
                        onClick={handleAllBrokers}
                        disabled={loading !== null}
                        className="w-full text-sm text-center py-2 transition-opacity disabled:opacity-60"
                        style={{ color: 'var(--text-secondary)' }}
                    >
                        {loading === 'ALL' ? (
                            <span className="flex items-center justify-center gap-2"><Spinner size={12} /> Opening broker list...</span>
                        ) : (
                            'Browse all supported brokers →'
                        )}
                    </button>
                </div>

                {/* Done linking prompt */}
                <div
                    className="flex items-center justify-between rounded-xl px-4 py-3"
                    style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)' }}
                >
                    <span className="text-xs" style={{ color: 'var(--text-tertiary)' }}>Done linking in the other tab?</span>
                    <button
                        onClick={() => { onDoneLink(); onClose(); }}
                        className="text-xs font-bold flex items-center gap-1"
                        style={{ color: 'var(--accent)' }}
                    >
                        <Check className="h-3.5 w-3.5" /> Refresh
                    </button>
                </div>
            </div>
        </div>,
        document.body
    );
}

// ---------------------------------------------------------------------------
// DisconnectModal
// ---------------------------------------------------------------------------
function DisconnectModal({ account, onConfirm, onClose }) {
    const [loading, setLoading] = useState(false);

    const handleConfirm = async () => {
        setLoading(true);
        await onConfirm(account.account_id);
        setLoading(false);
    };

    return createPortal(
        <div
            className="fixed inset-0 flex items-center justify-center z-50 p-4"
            style={{ background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)' }}
            onClick={onClose}
        >
            <div
                className="glass w-full max-w-sm p-6 space-y-5"
                style={{ borderRadius: '20px' }}
                onClick={e => e.stopPropagation()}
            >
                <div className="flex items-center gap-3">
                    <AlertTriangle className="h-5 w-5 flex-shrink-0" style={{ color: 'var(--negative)' }} />
                    <h3 className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>
                        Disconnect {account.brokerage_name}?
                    </h3>
                </div>
                <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                    This will remove all synced positions from{' '}
                    <strong>{account.account_name || account.account_number || account.brokerage_name}</strong>.
                    Your brokerage account itself is not affected.
                </p>
                <div className="flex gap-3">
                    <button
                        onClick={onClose}
                        className="flex-1 py-2.5 rounded-xl font-medium"
                        style={{ background: 'rgba(255,255,255,0.08)', color: 'var(--text-secondary)', border: '1px solid rgba(255,255,255,0.12)' }}
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleConfirm}
                        disabled={loading}
                        className="flex-1 py-2.5 rounded-xl font-bold flex items-center justify-center gap-2 disabled:opacity-60"
                        style={{ background: 'rgba(239,68,68,0.2)', color: '#f87171', border: '1px solid rgba(239,68,68,0.3)' }}
                    >
                        {loading ? <Spinner size={14} /> : null}
                        {loading ? 'Disconnecting...' : 'Disconnect'}
                    </button>
                </div>
            </div>
        </div>,
        document.body
    );
}

// ---------------------------------------------------------------------------
// AccountCard
// ---------------------------------------------------------------------------
function AccountCard({ account, positionCount, totalValue, isSyncing, onSync, onDisconnect }) {
    const meta = getBrokerMeta(account.brokerage_name);

    return (
        <div
            className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 py-4 px-4"
            style={{ borderBottom: '1px solid rgba(255,255,255,0.07)' }}
        >
            {/* Left: broker identity */}
            <div className="flex items-center gap-3 min-w-0">
                <div
                    className="w-10 h-10 rounded-xl flex items-center justify-center text-xs font-black flex-shrink-0"
                    style={{ background: meta.bg, color: meta.color }}
                >
                    {meta.initials}
                </div>
                <div className="min-w-0">
                    <div className="font-bold text-sm" style={{ color: 'var(--text-primary)' }}>
                        {account.brokerage_name}
                        {account.account_name && (
                            <span className="ml-1 font-normal text-xs" style={{ color: 'var(--text-tertiary)' }}>
                                — {account.account_name}
                            </span>
                        )}
                        {account.account_number && (
                            <span className="ml-1 text-xs" style={{ color: 'var(--text-tertiary)' }}>
                                {account.account_number}
                            </span>
                        )}
                    </div>
                    <div className="text-xs mt-0.5" style={{ color: 'var(--text-tertiary)' }}>
                        {totalValue != null ? `$${totalValue.toLocaleString(undefined, { maximumFractionDigits: 2 })} · ` : ''}
                        {positionCount} position{positionCount !== 1 ? 's' : ''}
                        {' · '}Last synced {relativeTime(account.last_synced)}
                    </div>
                    {account.last_sync_error && (
                        <div
                            className="mt-1 text-xs font-medium flex items-center gap-1"
                            style={{ color: '#fbbf24' }}
                        >
                            <AlertTriangle className="h-3 w-3" />
                            Reconnect required
                        </div>
                    )}
                </div>
            </div>

            {/* Right: actions */}
            <div className="flex items-center gap-2 flex-shrink-0">
                <button
                    onClick={() => onSync(account.account_id)}
                    disabled={isSyncing}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-opacity disabled:opacity-50"
                    style={{
                        background: 'rgba(255,255,255,0.07)',
                        color: 'var(--text-secondary)',
                        border: '1px solid rgba(255,255,255,0.10)',
                    }}
                >
                    {isSyncing ? <Spinner size={12} /> : <RefreshCw className="h-3.5 w-3.5" />}
                    Sync
                </button>
                <button
                    onClick={() => onDisconnect(account)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-opacity"
                    style={{
                        background: 'rgba(239,68,68,0.08)',
                        color: '#f87171',
                        border: '1px solid rgba(239,68,68,0.2)',
                    }}
                >
                    <WifiOff className="h-3.5 w-3.5" />
                    Disconnect
                </button>
            </div>
        </div>
    );
}

// ---------------------------------------------------------------------------
// BrokerageManager
// ---------------------------------------------------------------------------
export default function BrokerageManager({
    accounts,
    positions,
    balances,
    summary,
    isSyncing,
    syncingAccountId,
    syncAll,
    syncAccount,
    disconnect,
    fetchAccounts,
    getConnectUrl,
}) {
    const [showAddModal, setShowAddModal] = useState(false);
    const [disconnectTarget, setDisconnectTarget] = useState(null);
    const [toast, setToast] = useState(null);

    const showToast = useCallback((message, type = 'info') => {
        setToast({ message, type });
    }, []);

    // Position count per account
    const posByAccount = positions.reduce((acc, p) => {
        acc[p.account_id] = (acc[p.account_id] || 0) + 1;
        return acc;
    }, {});

    // Total value per account (cash + market_value; cash-only accounts have market_value=0)
    const valByAccount = balances.reduce((acc, b) => {
        acc[b.account_id] = b.total_value ?? b.market_value ?? b.cash;
        return acc;
    }, {});

    const handleSyncAll = async () => {
        const result = await syncAll();
        if (result.success) {
            const d = result.data;
            showToast(
                `Synced successfully — ${d.positions_synced} positions updated across ${d.accounts_synced} accounts`,
                'success',
            );
        } else {
            if (result.error === 'snaptrade_auth_expired') {
                showToast('Authentication expired — reconnect your accounts', 'warning');
            } else {
                showToast('Sync failed — check your connection', 'error');
            }
        }
    };

    const handleSyncAccount = async (accountId) => {
        const result = await syncAccount(accountId);
        if (result.success) {
            showToast(`Synced successfully — ${result.data.positions_synced} positions updated`, 'success');
        } else {
            if (result.error === 'snaptrade_auth_expired') {
                showToast('Authentication expired — reconnect this account', 'warning');
            } else if (result.error === 'rate_limited') {
                showToast(`Too many requests — auto-retry in ${result.data?.retry_after || 60}s`, 'warning');
            } else {
                showToast('Sync failed — check your connection', 'error');
            }
        }
    };

    const handleDisconnect = async (accountId) => {
        const result = await disconnect(accountId);
        setDisconnectTarget(null);
        if (result.success) {
            showToast('Account disconnected', 'info');
        } else {
            showToast('Failed to disconnect account', 'error');
        }
    };

    const handleDoneLink = useCallback(async () => {
        const result = await syncAll();
        if (result.success) {
            const d = result.data;
            showToast(
                d.accounts_synced > 0
                    ? `Connected! Found ${d.accounts_synced} account(s) — ${d.positions_synced} positions synced`
                    : 'Synced — no new accounts found yet. SnapTrade may need a moment.',
                d.accounts_synced > 0 ? 'success' : 'warning',
            );
        } else {
            showToast('Sync failed — try "Sync All" manually', 'error');
        }
    }, [syncAll, showToast]);

    const handleGetUrl = async (brokerSlug) => {
        try {
            return await getConnectUrl(brokerSlug);
        } catch (err) {
            const status = err.response?.status;
            if (status === 401) showToast('Authentication expired — please re-register', 'warning');
            else if (status === 429) showToast('Rate limited — try again in a moment', 'warning');
            else showToast('Failed to get connection URL', 'error');
            return null;
        }
    };

    const isEmpty = accounts.length === 0;

    return (
        <div className="glass overflow-hidden" style={{ padding: 0 }}>
            {/* Header */}
            <div
                className="px-5 py-4 flex items-center justify-between"
                style={{ borderBottom: '1px solid rgba(255,255,255,0.07)' }}
            >
                <div>
                    <h3 className="text-sm font-bold uppercase tracking-widest" style={{ color: 'var(--text-tertiary)' }}>
                        Connected Accounts
                    </h3>
                    {!isEmpty && (
                        <p className="text-xs mt-0.5" style={{ color: 'var(--text-tertiary)' }}>
                            {accounts.length} broker{accounts.length !== 1 ? 's' : ''} · auto-sync every 15 min
                        </p>
                    )}
                </div>
                <button
                    onClick={() => setShowAddModal(true)}
                    className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-bold transition-opacity"
                    style={{ background: 'var(--accent)', color: '#000' }}
                >
                    <Plus className="h-3.5 w-3.5" />
                    Add Account
                </button>
            </div>

            {/* Account cards */}
            {isEmpty ? (
                <div className="flex flex-col items-center justify-center py-12 px-6 text-center">
                    <Wifi className="h-10 w-10 mb-4" style={{ color: 'rgba(255,255,255,0.12)' }} />
                    <p className="text-sm font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>No brokerages connected</p>
                    <p className="text-xs mb-4" style={{ color: 'var(--text-tertiary)' }}>
                        Link Webull, Robinhood, or any SnapTrade-supported broker to sync your holdings automatically.
                    </p>
                    <button
                        onClick={() => setShowAddModal(true)}
                        className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-bold"
                        style={{ background: 'var(--accent)', color: '#000' }}
                    >
                        <Plus className="h-4 w-4" />
                        Connect a Broker
                    </button>
                </div>
            ) : (
                <>
                    {accounts.map(acct => (
                        <AccountCard
                            key={acct.account_id}
                            account={acct}
                            positionCount={posByAccount[acct.account_id] || 0}
                            totalValue={valByAccount[acct.account_id]}
                            isSyncing={syncingAccountId === acct.account_id}
                            onSync={handleSyncAccount}
                            onDisconnect={setDisconnectTarget}
                        />
                    ))}

                    {/* Portfolio summary footer */}
                    {summary && (
                        <div
                            className="px-5 py-4 space-y-3"
                            style={{ borderTop: '1px solid rgba(255,255,255,0.07)', background: 'rgba(255,255,255,0.02)' }}
                        >
                            <div className="flex items-start justify-between gap-2">
                                <div>
                                    <p className="text-xs uppercase font-bold tracking-widest mb-1" style={{ color: 'var(--text-tertiary)' }}>
                                        Portfolio Total Across All Accounts
                                    </p>
                                    <p className="text-xl font-black" style={{ color: 'var(--text-primary)' }}>
                                        ${summary.total_portfolio_value?.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                                    </p>
                                    <p
                                        className="text-sm font-bold mt-0.5"
                                        style={{ color: (summary.total_unrealized_pnl || 0) >= 0 ? 'var(--positive)' : 'var(--negative)' }}
                                    >
                                        {(summary.total_unrealized_pnl || 0) >= 0 ? '+' : ''}
                                        ${Math.abs(summary.total_unrealized_pnl || 0).toFixed(2)} unrealized
                                        {' '}
                                        ({(summary.total_unrealized_pnl || 0) >= 0 ? '+' : ''}
                                        {(summary.total_unrealized_pnl_pct || 0).toFixed(2)}%)
                                    </p>
                                </div>
                                <div className="flex flex-col items-end gap-2">
                                    <button
                                        onClick={handleSyncAll}
                                        disabled={isSyncing}
                                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold disabled:opacity-50"
                                        style={{
                                            background: 'var(--accent)',
                                            color: '#000',
                                        }}
                                    >
                                        {isSyncing ? <Spinner size={12} /> : <RefreshCw className="h-3.5 w-3.5" />}
                                        Sync All
                                    </button>
                                    <span className="text-[10px]" style={{ color: 'var(--text-tertiary)' }}>
                                        auto every 15m
                                    </span>
                                </div>
                            </div>

                            {/* Per-brokerage breakdown */}
                            {summary.by_brokerage && Object.keys(summary.by_brokerage).length > 0 && (
                                <div className="flex flex-wrap gap-3">
                                    {Object.entries(summary.by_brokerage).map(([name, data]) => {
                                        const meta = getBrokerMeta(name);
                                        return (
                                            <div
                                                key={name}
                                                className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs"
                                                style={{ background: meta.bg, border: `1px solid ${meta.color}30` }}
                                            >
                                                <span className="font-black" style={{ color: meta.color }}>{name}</span>
                                                <span style={{ color: 'var(--text-secondary)' }}>
                                                    ${data.value?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                                </span>
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </div>
                    )}
                </>
            )}

            {/* Modals */}
            {showAddModal && (
                <AddAccountModal
                    onClose={() => setShowAddModal(false)}
                    onGetUrl={handleGetUrl}
                    onDoneLink={handleDoneLink}
                />
            )}

            {disconnectTarget && (
                <DisconnectModal
                    account={disconnectTarget}
                    onConfirm={handleDisconnect}
                    onClose={() => setDisconnectTarget(null)}
                />
            )}

            {/* Toast */}
            {toast && (
                <Toast
                    message={toast.message}
                    type={toast.type}
                    onClose={() => setToast(null)}
                />
            )}
        </div>
    );
}
