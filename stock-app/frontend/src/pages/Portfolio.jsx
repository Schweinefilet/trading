import React, { useState, useEffect, useCallback, useMemo } from 'react';
import axios from 'axios';
import {
    PieChart, Pie, Cell, ResponsiveContainer,
    Tooltip as RechartsTooltip, LineChart, Line,
    XAxis, YAxis, CartesianGrid, Area, AreaChart
} from 'recharts';
import { TrendingUp, TrendingDown, Target, Shield, Trash2, Pencil, Check, X, Plus, Zap, ChevronDown, ChevronUp, RefreshCw, Wifi } from 'lucide-react';
import { Link } from 'react-router-dom';
import ParticlesBg from '../components/ParticlesBg';
import BrokerageManager from '../components/BrokerageManager';
import StockIndicatorsModal from '../components/StockIndicatorsModal';
import { useBrokerageSync } from '../hooks/useBrokerageSync';

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#6366f1', '#14b8a6', '#f97316'];

const MetricCard = ({ icon: Icon, label, value, color }) => (
    <div className="glass p-5 flex flex-col items-center text-center">
        <div className={`w-11 h-11 rounded-full flex items-center justify-center mb-3 ${color}`}>
            <Icon className="h-5 w-5" />
        </div>
        <p className="text-2xl font-black tabular-nums" style={{ color: 'var(--text-primary)' }}>{value}</p>
        <p className="text-[10px] uppercase font-bold tracking-widest mt-1" style={{ color: 'var(--text-tertiary)' }}>{label}</p>
    </div>
);

const EditableRow = ({ holding, onDelete, onUpdate, onRowClick, brokerTags = [] }) => {
    const [editing, setEditing] = useState(false);
    const [shares, setShares] = useState(holding.shares);
    const [cost, setCost] = useState(holding.avg_cost);
    const [hovered, setHovered] = useState(false);

    const handleSave = async () => {
        await onUpdate(holding.ticker, parseFloat(shares), parseFloat(cost));
        setEditing(false);
    };

    const isPositive = holding.gain_loss >= 0;

    return (
        <tr
            className="transition-colors cursor-pointer"
            style={{
                borderBottom: '1px solid rgba(255,255,255,0.07)',
                background: hovered ? 'rgba(255,255,255,0.02)' : 'transparent',
            }}
            onMouseEnter={() => setHovered(true)}
            onMouseLeave={() => setHovered(false)}
            onClick={(e) => {
                // Don't trigger row click if clicking on action buttons or input fields
                if (e.target.closest('button') || e.target.closest('input') || e.target.closest('a')) {
                    return;
                }
                if (onRowClick) {
                    onRowClick(holding.ticker);
                }
            }}
        >
            <td className="py-4 px-3">
                <Link
                    to={`/stock/${holding.ticker}`}
                    className="font-bold transition-colors hover:opacity-80"
                    style={{ color: 'var(--accent)' }}
                >
                    {holding.ticker}
                </Link>
                <div className="mt-0.5">
                    <p className="text-[11px] truncate" style={{ color: 'var(--text-tertiary)', maxWidth: '220px' }}>
                        {holding.long_name || holding.ticker}
                    </p>
                    {holding.sector && (
                        <span
                            className="inline-block mt-1 text-[10px] px-1.5 py-0.5 rounded"
                            style={{ background: 'rgba(255,255,255,0.08)', color: 'var(--text-secondary)' }}
                        >
                            {holding.sector}
                        </span>
                    )}
                    {brokerTags.map(broker => (
                        <span
                            key={broker}
                            className="inline-block mt-1 ml-1 text-[10px] px-1.5 py-0.5 rounded font-semibold"
                            style={{ background: 'rgba(59,130,246,0.18)', color: '#60a5fa', border: '1px solid rgba(59,130,246,0.3)' }}
                        >
                            {broker}
                        </span>
                    ))}
                </div>
            </td>
            <td className="py-4 px-3 text-right">
                {editing ? (
                    <input
                        type="number"
                        value={shares}
                        onChange={e => setShares(e.target.value)}
                        className="w-24 rounded-lg px-2 py-1 text-white text-right text-sm outline-none"
                        style={{ background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.14)' }}
                    />
                ) : (
                    <span style={{ color: 'var(--text-secondary)' }}>{holding.shares}</span>
                )}
            </td>
            <td className="py-4 px-3 text-right">
                {editing ? (
                    <input
                        type="number"
                        value={cost}
                        onChange={e => setCost(e.target.value)}
                        className="w-24 rounded-lg px-2 py-1 text-white text-right text-sm outline-none"
                        style={{ background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.14)' }}
                    />
                ) : (
                    <span style={{ color: 'var(--text-secondary)' }}>${holding.avg_cost.toFixed(2)}</span>
                )}
            </td>
            <td className="py-4 px-3 text-right font-medium" style={{ color: 'var(--text-primary)' }}>
                ${holding.current_price.toFixed(2)}
            </td>
            <td className="py-4 px-3 text-right font-bold" style={{ color: 'var(--text-primary)' }}>
                ${holding.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
            </td>
            <td className="py-4 px-3 text-right font-bold" style={{ color: isPositive ? 'var(--positive)' : 'var(--negative)' }}>
                <div className="flex flex-col items-end">
                    <span>{isPositive ? '+' : ''}{holding.gain_loss.toFixed(2)}</span>
                    <span className="text-xs opacity-75">{isPositive ? '+' : ''}{holding.gain_loss_pct.toFixed(2)}%</span>
                </div>
            </td>
            <td className="py-4 px-3 text-right">
                <div className="flex flex-col items-end">
                    <span className="text-sm" style={{ color: 'var(--text-tertiary)' }}>{(holding.weight * 100).toFixed(1)}%</span>
                    <div className="w-16 h-1 rounded mt-1" style={{ background: 'rgba(255,255,255,0.10)' }}>
                        <div className="h-1 rounded" style={{ width: `${holding.weight * 100}%`, background: 'var(--accent)' }} />
                    </div>
                </div>
            </td>
            <td className="py-4 px-3 text-right">
                <div className="flex items-center justify-end space-x-1">
                    {editing ? (
                        <>
                            <button onClick={handleSave} className="p-1 transition-colors" style={{ color: 'var(--positive)' }}>
                                <Check className="h-4 w-4" />
                            </button>
                            <button onClick={() => setEditing(false)} className="p-1 transition-colors" style={{ color: 'var(--text-tertiary)' }}>
                                <X className="h-4 w-4" />
                            </button>
                        </>
                    ) : (
                        <>
                            <button
                                onClick={() => setEditing(true)}
                                className="p-1 transition-colors"
                                style={{ color: 'var(--text-tertiary)' }}
                                onMouseEnter={e => e.currentTarget.style.color = 'var(--text-primary)'}
                                onMouseLeave={e => e.currentTarget.style.color = 'var(--text-tertiary)'}
                            >
                                <Pencil className="h-4 w-4" />
                            </button>
                            <button
                                onClick={() => onDelete(holding.ticker)}
                                className="p-1 transition-colors"
                                style={{ color: 'var(--text-tertiary)' }}
                                onMouseEnter={e => e.currentTarget.style.color = 'var(--negative)'}
                                onMouseLeave={e => e.currentTarget.style.color = 'var(--text-tertiary)'}
                            >
                                <Trash2 className="h-4 w-4" />
                            </button>
                        </>
                    )}
                </div>
            </td>
        </tr>
    );
};

const AddPositionModal = ({ onClose, onAdded }) => {
    const [ticker, setTicker] = useState('');
    const [shares, setShares] = useState('');
    const [cost, setCost] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleAdd = async () => {
        if (!ticker || !shares || !cost) { setError('All fields are required.'); return; }
        setLoading(true);
        setError('');
        try {
            await axios.post('/api/portfolio/add', {
                ticker: ticker.toUpperCase(),
                shares: parseFloat(shares),
                avg_cost: parseFloat(cost)
            });
            onAdded();
            onClose();
        } catch (e) {
            setError('Failed to add position. Check the ticker and try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div
            className="fixed inset-0 flex items-center justify-center z-50 p-4"
            style={{ background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)' }}
        >
            <div className="glass w-full max-w-sm p-6" style={{ borderRadius: '20px' }}>
                <div className="flex justify-between items-center mb-6">
                    <h3 className="text-xl font-bold" style={{ color: 'var(--text-primary)' }}>Add Position</h3>
                    <button
                        onClick={onClose}
                        className="transition-colors"
                        style={{ color: 'var(--text-tertiary)' }}
                        onMouseEnter={e => e.currentTarget.style.color = 'var(--text-primary)'}
                        onMouseLeave={e => e.currentTarget.style.color = 'var(--text-tertiary)'}
                    >
                        <X className="h-5 w-5" />
                    </button>
                </div>
                <div className="space-y-4">
                    <div>
                        <label className="block text-xs mb-1 uppercase font-bold" style={{ color: 'var(--text-secondary)' }}>
                            Ticker Symbol
                        </label>
                        <input
                            type="text"
                            placeholder="e.g. AAPL"
                            className="glass-input uppercase"
                            value={ticker}
                            onChange={e => setTicker(e.target.value.toUpperCase())}
                        />
                    </div>
                    <div>
                        <label className="block text-xs mb-1 uppercase font-bold" style={{ color: 'var(--text-secondary)' }}>
                            Number of Shares
                        </label>
                        <input
                            type="number"
                            placeholder="e.g. 10"
                            className="glass-input"
                            value={shares}
                            onChange={e => setShares(e.target.value)}
                        />
                    </div>
                    <div>
                        <label className="block text-xs mb-1 uppercase font-bold" style={{ color: 'var(--text-secondary)' }}>
                            Average Cost Basis (per share)
                        </label>
                        <input
                            type="number"
                            placeholder="e.g. 175.00"
                            className="glass-input"
                            value={cost}
                            onChange={e => setCost(e.target.value)}
                        />
                    </div>
                    {error && <p className="text-sm" style={{ color: 'var(--negative)' }}>{error}</p>}
                    <div className="flex space-x-3 pt-2">
                        <button
                            onClick={onClose}
                            className="flex-1 py-2.5 rounded-xl font-medium transition-colors"
                            style={{
                                background: 'rgba(255,255,255,0.08)',
                                color: 'var(--text-secondary)',
                                border: '1px solid rgba(255,255,255,0.12)',
                            }}
                        >
                            Cancel
                        </button>
                        <button
                            disabled={loading}
                            onClick={handleAdd}
                            className="flex-1 py-2.5 font-bold transition-colors disabled:opacity-50"
                            style={{ background: 'var(--accent)', color: '#000', borderRadius: 'var(--radius-btn)' }}
                        >
                            {loading ? 'Adding...' : 'Add Position'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

const CustomPieTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
        const { ticker, label, value, weight } = payload[0].payload;
        return (
            <div className="glass p-3" style={{ borderRadius: '12px' }}>
                <p className="font-bold" style={{ color: 'var(--text-primary)' }}>{label || ticker}</p>
                <p style={{ color: 'var(--text-secondary)' }}>${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}</p>
                <p style={{ color: 'var(--text-secondary)' }}>{(weight * 100).toFixed(1)}%</p>
            </div>
        );
    }
    return null;
};

const scoreToColor = (score) => {
    if (!Number.isFinite(score)) return 'var(--text-primary)';
    const s = Math.max(0, Math.min(1, score));
    const hue = Math.round(s * 120); // 0=red, 120=green
    return `hsl(${hue}, 82%, 58%)`;
};

const RiskLineItem = ({ code, label, value, definition, score }) => (
    <div className="py-3" style={{ borderBottom: '1px solid rgba(255,255,255,0.07)' }}>
        <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-[10px] uppercase font-black tracking-widest" style={{ color: 'var(--text-tertiary)' }}>
                        {code}
                    </span>
                    <span className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
                        {label}
                    </span>
                    <span
                        className="relative group text-[10px] uppercase tracking-wider cursor-help"
                        style={{ color: 'var(--text-tertiary)' }}
                        aria-label={`${label}. ${definition}`}
                    >
                        Info
                        <span
                            className="pointer-events-none absolute left-0 top-full mt-1 w-96 rounded-lg px-2 py-1.5 text-[11px] normal-case opacity-0 group-hover:opacity-100 transition-opacity z-20"
                            style={{
                                background: 'rgba(0,0,0,0.92)',
                                color: 'var(--text-secondary)',
                                border: '1px solid rgba(255,255,255,0.15)',
                                whiteSpace: 'pre-line',
                            }}
                        >
                            {definition}
                        </span>
                    </span>
                </div>
            </div>
            <div className="text-lg font-black tabular-nums text-right" style={{ color: scoreToColor(score) }}>
                {value}
            </div>
        </div>
    </div>
);

const asNumber = (value, decimals = 2) => {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(decimals) : 'No Data';
};

const asPercent = (value, decimals = 2) => {
    const n = Number(value);
    return Number.isFinite(n) ? `${(n * 100).toFixed(decimals)}%` : 'No Data';
};

const asCurrency = (value) => {
    const n = Number(value);
    return Number.isFinite(n)
        ? `$${n.toLocaleString(undefined, { maximumFractionDigits: 2 })}`
        : 'No Data';
};

const asCapture = (up, down) => {
    const u = Number(up);
    const d = Number(down);
    if (!Number.isFinite(u) || !Number.isFinite(d)) return 'No Data';
    return `Up ${u.toFixed(2)} Down ${d.toFixed(2)}`;
};

const asRawPercentNumber = (value, decimals = 2) => {
    const n = toFinite(value);
    return n === null ? 'No Data' : n.toFixed(decimals);
};

const toFinite = (value) => {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
};

const sharpeBucketLabel = (value) => {
    const n = toFinite(value);
    if (n === null) return 'No Data';
    if (n < 1) return 'Sharpe <1';
    if (n < 2) return 'Sharpe 1-2';
    if (n < 3) return 'Sharpe 2-3';
    return 'Sharpe >3';
};

const withRiskFallback = (risk = {}, holdings = [], correlation = {}, summary = {}) => {
    const merged = { ...risk };

    if (!Number.isFinite(Number(merged.win_rate)) && holdings.length) {
        const winners = holdings.filter(h => Number(h.gain_loss) > 0).length;
        merged.win_rate = winners / holdings.length;
    }

    if (!Number.isFinite(Number(merged.effective_n)) && holdings.length > 0) {
        const weights = holdings.map(h => Number(h.weight) || 0);
        const tickers = holdings.map(h => h.ticker);
        const hasCorr = tickers.every(t => correlation?.[t]);
        if (hasCorr) {
            let denom = 0;
            for (let i = 0; i < tickers.length; i += 1) {
                for (let j = 0; j < tickers.length; j += 1) {
                    const c = Number(correlation?.[tickers[i]]?.[tickers[j]]);
                    if (Number.isFinite(c)) denom += weights[i] * weights[j] * c;
                }
            }
            if (denom > 0) merged.effective_n = 1 / denom;
        }
    }

    if (!Number.isFinite(Number(merged.calmar_ratio))) {
        const annual = Number(summary.total_gain_loss_pct) / 100;
        const mdd = Number(merged.max_drawdown);
        if (Number.isFinite(annual) && Number.isFinite(mdd) && Math.abs(mdd) > 1e-12) {
            merged.calmar_ratio = annual / Math.abs(mdd);
        }
    }

    return merged;
};

const riskScore = (code, risk = {}, summary = {}) => {
    const toScoreHigher = (v, bad, good) => {
        if (!Number.isFinite(v)) return null;
        if (good === bad) return null;
        return (v - bad) / (good - bad);
    };
    const toScoreLower = (v, good, bad) => {
        if (!Number.isFinite(v)) return null;
        if (good === bad) return null;
        return (bad - v) / (bad - good);
    };
    const clamp01 = (v) => Math.max(0, Math.min(1, v));

    if (code === 'BETA') {
        const b = toFinite(risk.beta);
        if (b === null) return null;
        if (b <= 0.3 || b >= 1.5) return 0;
        if (b < 1) return clamp01((b - 0.3) / (1 - 0.3));
        return clamp01((1.5 - b) / (1.5 - 1));
    }
    if (code === 'SHARPE') return clamp01(toScoreHigher(toFinite(risk.sharpe_ratio), 0.0, 3.0));
    if (code === 'VOL') return clamp01(toScoreLower(Math.abs((toFinite(risk.volatility) ?? NaN) * 100), 14, 25));
    if (code === 'MDD') return clamp01(toScoreLower(Math.abs((toFinite(risk.max_drawdown) ?? NaN) * 100), 10, 30));
    if (code === 'SORTINO') return clamp01(toScoreHigher(toFinite(risk.sortino_ratio), 0.0, 2.0));
    if (code === 'CALMAR') return clamp01(toScoreHigher(toFinite(risk.calmar_ratio), 0.25, 3.0));
    if (code === 'VAR95') {
        const nav = toFinite(summary.total_value);
        const v = toFinite(risk.var_95_dollar);
        if (nav === null || v === null || nav <= 0) return null;
        return clamp01(toScoreLower((Math.abs(v) / nav) * 100, 1.5, 3.0));
    }
    if (code === 'CVAR') {
        const v = toFinite(risk.var_95_dollar);
        const c = toFinite(risk.cvar_95_dollar);
        if (v === null || c === null || Math.abs(v) < 1e-12) return null;
        const ratio = Math.abs(c) / Math.abs(v);
        return clamp01(toScoreLower(ratio, 1.35, 2.0));
    }
    if (code === 'CDD') return clamp01(toScoreLower(Math.abs((toFinite(risk.current_drawdown) ?? NaN) * 100), 5, 25));
    if (code === 'SKEW') return clamp01(toScoreHigher(toFinite(risk.skewness), -0.5, 0.8));
    if (code === 'EN') return clamp01(toScoreHigher(toFinite(risk.effective_n), 2, 8));
    if (code === 'ALPHA') return clamp01(toScoreHigher(toFinite(risk.jensen_alpha), -0.02, 0.06));
    if (code === 'CAPTURE') {
        const up = toFinite(risk.up_capture);
        const down = toFinite(risk.down_capture);
        if (up === null || down === null) return null;
        return clamp01(toScoreHigher(up - down, 0.0, 0.35));
    }
    if (code === 'WIN') return clamp01(toScoreHigher(toFinite(risk.win_rate), 0.4, 0.7));
    if (code === 'ULCER') return clamp01(toScoreLower((toFinite(risk.ulcer_index) ?? NaN) * 100, 5, 15));
    return null;
};

const RISK_EXPLANATIONS = {
    BETA: `Good: 0.8-1.2 for growth, 0.4-0.8 for conservative. Bad: >1.5 excessive market risk, <0.3 too defensive to generate meaningful returns.`,
    SHARPE: `Good: >1.0 solid, >2.0 excellent, >3.0 elite. Bad: <0.5 not being compensated for the risk taken, <0 losing money on a risk-adjusted basis.`,
    VOL: `Good: 10-18% for a diversified portfolio. Bad: >25% approaches single-stock risk territory, <8% likely too conservative to outperform.`,
    MDD: `Good: <10% conservative, <20% acceptable for aggressive growth. Bad: >30% most investors capitulate before recovery, >50% catastrophic.`,
    SORTINO: `Good: >1.0 solid, >2.0 excellent. Bad: <0.5 downside swings are consuming returns, <0 losing money net of downside risk.`,
    CALMAR: `Good: >1.0 solid, >3.0 excellent. Bad: <0.5 max pain not justified by returns, <0.25 poor risk-adjusted performance.`,
    VAR95: `Good: <1.5% of total portfolio value on a bad day. Bad: >3% of NAV means a single bad day causes structural damage to the portfolio.`,
    CVAR: `Good: 1.2-1.5x your VaR figure. Bad: >2x VaR signals extreme fat tails and blow-up risk on worst days.`,
    CDD: `Good: <5% from peak in normal conditions. Bad: >15% and still declining suggests something structural is broken, >25% is a crisis.`,
    SKEW: `Good: >0.3 positive skew, winners meaningfully outrun losers. Bad: <-0.5 left-tail blow-up risk, your worst days significantly exceed your best days.`,
    EN: `Good: >5 meaningful diversification, >8 well-diversified. Bad: <3 concentrated 2-3 stock bet regardless of ticker count, <2 essentially a single-position portfolio.`,
    ALPHA: `Good: >2% annualized consistently suggests genuine edge. Bad: <0% SPY would have outperformed you on a risk-adjusted basis, <-2% significant value destruction.`,
    CAPTURE: `Good: Up capture exceeds Down capture by 15+ points (e.g., 1.20 up / 0.85 down). Bad: Down capture >= Up capture means no asymmetry, you are a leveraged index fund with fees.`,
    WIN: `Good: >50% with average winner near equal to average loser, or 40-50% if average winner is 2x+ average loser. Bad: <40% with winners less than 1.5x losers is negative expectancy by math.`,
    ULCER: `Good: <5% drawdowns are shallow or short-lived. Bad: >10% spending significant time deep underwater, >15% chronic drawdown pattern.`,
};

const BROKER_COLORS = {
    Webull:    '#04A6E0',
    Robinhood: '#00C805',
};

const DEFAULT_BROKER_COLOR = '#9ca3af';

function getBrokerColor(name) {
    return BROKER_COLORS[name] || DEFAULT_BROKER_COLOR;
}

function relativeTime(isoString) {
    if (!isoString) return 'never';
    const diff = (Date.now() - new Date(isoString)) / 1000;
    if (diff < 60)    return 'just now';
    if (diff < 3600)  return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
}

const SyncedHoldingsSection = ({ accounts, positions, accountFilter, onFilterChange, onRowClick }) => {
    const hasAccounts = accounts.length > 0;
    const hasPositions = positions.length > 0;

    const filteredPositions = useMemo(() => {
        if (!accountFilter || accountFilter === 'all') return positions;
        return positions.filter(p => p.account_id === accountFilter);
    }, [positions, accountFilter]);

    // Most recent last_synced across filtered positions
    const lastSynced = useMemo(() => {
        const dates = filteredPositions.map(p => p.last_synced).filter(Boolean);
        if (!dates.length) return null;
        return dates.reduce((a, b) => (a > b ? a : b));
    }, [filteredPositions]);

    return (
        <div className="glass overflow-hidden" style={{ padding: 0 }}>
            <div
                className="px-5 py-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3"
                style={{ borderBottom: '1px solid rgba(255,255,255,0.07)' }}
            >
                <div>
                    <h3 className="text-sm font-bold uppercase tracking-widest" style={{ color: 'var(--text-tertiary)' }}>
                        Synced Holdings
                    </h3>
                    {lastSynced && (
                        <p className="text-xs mt-0.5" style={{ color: 'var(--text-tertiary)' }}>
                            Last updated {relativeTime(lastSynced)}
                        </p>
                    )}
                </div>
                {hasAccounts && (
                    <select
                        value={accountFilter}
                        onChange={e => onFilterChange(e.target.value)}
                        className="text-xs font-bold rounded-lg px-3 py-1.5 outline-none"
                        style={{
                            background: 'rgba(255,255,255,0.08)',
                            color: 'var(--text-secondary)',
                            border: '1px solid rgba(255,255,255,0.14)',
                        }}
                    >
                        <option value="all">All accounts</option>
                        {accounts.map(a => (
                            <option key={a.account_id} value={a.account_id}>
                                {a.brokerage_name}{a.account_name ? ` — ${a.account_name}` : ''}{a.account_number ? ` ${a.account_number}` : ''}
                            </option>
                        ))}
                    </select>
                )}
            </div>

            {!hasAccounts ? (
                <div className="flex flex-col items-center justify-center py-10 px-6 text-center">
                    <Wifi className="h-8 w-8 mb-3" style={{ color: 'rgba(255,255,255,0.12)' }} />
                    <p className="text-sm" style={{ color: 'var(--text-tertiary)' }}>
                        Not connected — link a brokerage above to see synced positions here.
                    </p>
                </div>
            ) : !hasPositions ? (
                <div className="flex items-center justify-center py-10">
                    <p className="text-sm" style={{ color: 'var(--text-tertiary)' }}>
                        No positions synced yet — click Sync All to fetch your holdings.
                    </p>
                </div>
            ) : filteredPositions.length === 0 ? (
                <div className="flex items-center justify-center py-10">
                    <p className="text-sm" style={{ color: 'var(--text-tertiary)' }}>No positions for selected account.</p>
                </div>
            ) : (
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr
                                className="text-[10px] uppercase font-bold tracking-widest"
                                style={{ color: 'var(--text-tertiary)', borderBottom: '1px solid rgba(255,255,255,0.07)' }}
                            >
                                <th className="py-3 px-3 text-left">Ticker</th>
                                <th className="py-3 px-3 text-left">Brokerage</th>
                                <th className="py-3 px-3 text-left">Account</th>
                                <th className="py-3 px-3 text-right">Shares</th>
                                <th className="py-3 px-3 text-right">Avg Cost</th>
                                <th className="py-3 px-3 text-right">Current</th>
                                <th className="py-3 px-3 text-right">Market Value</th>
                                <th className="py-3 px-3 text-right">Unrealized P&L</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredPositions.map(p => {
                                const isPos = (p.open_pnl || 0) >= 0;
                                const color = getBrokerColor(p.brokerage_name);
                                return (
                                    <tr
                                        key={p.id}
                                        className="transition-colors cursor-pointer"
                                        style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}
                                        onClick={(e) => {
                                            // Don't trigger row click if clicking on link
                                            if (e.target.closest('a')) {
                                                return;
                                            }
                                            if (onRowClick) {
                                                onRowClick(p.ticker);
                                            }
                                        }}
                                    >
                                        <td className="py-3 px-3">
                                            <Link
                                                to={`/stock/${p.ticker}`}
                                                className="font-bold transition-colors hover:opacity-80"
                                                style={{ color: 'var(--accent)' }}
                                            >
                                                {p.ticker}
                                            </Link>
                                            {p.symbol_description && (
                                                <p className="text-[10px] mt-0.5 truncate max-w-[160px]" style={{ color: 'var(--text-tertiary)' }}>
                                                    {p.symbol_description}
                                                </p>
                                            )}
                                        </td>
                                        <td className="py-3 px-3">
                                            <span
                                                className="inline-block px-2 py-0.5 rounded text-[10px] font-black uppercase tracking-wider"
                                                style={{ background: `${color}20`, color: color, border: `1px solid ${color}40` }}
                                            >
                                                {p.brokerage_name}
                                            </span>
                                        </td>
                                        <td className="py-3 px-3 text-xs" style={{ color: 'var(--text-tertiary)' }}>
                                            {p.account_name || '—'}
                                        </td>
                                        <td className="py-3 px-3 text-right" style={{ color: 'var(--text-secondary)' }}>
                                            {p.units != null ? p.units.toLocaleString(undefined, { maximumFractionDigits: 6 }) : '—'}
                                        </td>
                                        <td className="py-3 px-3 text-right" style={{ color: 'var(--text-secondary)' }}>
                                            {p.average_purchase_price != null ? `$${p.average_purchase_price.toFixed(2)}` : '—'}
                                        </td>
                                        <td className="py-3 px-3 text-right font-medium" style={{ color: 'var(--text-primary)' }}>
                                            {p.current_price != null ? `$${p.current_price.toFixed(2)}` : '—'}
                                        </td>
                                        <td className="py-3 px-3 text-right font-bold" style={{ color: 'var(--text-primary)' }}>
                                            {p.current_market_value != null
                                                ? `$${p.current_market_value.toLocaleString(undefined, { maximumFractionDigits: 2 })}`
                                                : '—'}
                                        </td>
                                        <td className="py-3 px-3 text-right">
                                            <div className="flex flex-col items-end">
                                                <span className="font-bold" style={{ color: isPos ? 'var(--positive)' : 'var(--negative)' }}>
                                                    {isPos ? '+' : ''}{p.open_pnl != null ? `$${p.open_pnl.toFixed(2)}` : '—'}
                                                </span>
                                                {p.open_pnl_percent != null && (
                                                    <span className="text-xs opacity-75" style={{ color: isPos ? 'var(--positive)' : 'var(--negative)' }}>
                                                        {isPos ? '+' : ''}{(p.open_pnl_percent * 100).toFixed(2)}%
                                                    </span>
                                                )}
                                            </div>
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};

const Portfolio = () => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [showAddModal, setShowAddModal] = useState(false);
    const [allocationMode, setAllocationMode] = useState('ticker');
    const [showAllRiskMetrics, setShowAllRiskMetrics] = useState(false);
    const [selectedTicker, setSelectedTicker] = useState(null);
    const [valueHistoryTimeframe, setValueHistoryTimeframe] = useState('1y');

    // Brokerage sync state (shared hook)
    const brokerage = useBrokerageSync({ lazy: true });

    useEffect(() => {
        const timer = setTimeout(() => {
            brokerage.start();
        }, 700);
        return () => clearTimeout(timer);
    }, [brokerage.start]);

    const fetchPortfolio = useCallback(async (timeframe = '1y') => {
        setLoading(true);
        try {
            const res = await axios.get('/api/portfolio/analytics', {
                params: { timeframe }
            });
            setData(res.data);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    }, []);

    // Silent background refresh — no loading spinner, used for polling and forced updates
    const refreshPortfolio = useCallback(async (timeframe = '1y', force = false) => {
        try {
            const params = { timeframe };
            if (force) params.force = '1';
            const res = await axios.get('/api/portfolio/analytics', { params });
            setData(res.data);
        } catch (e) {
            console.error(e);
        }
    }, []);

    useEffect(() => { fetchPortfolio(valueHistoryTimeframe); }, [fetchPortfolio, valueHistoryTimeframe]);

    // Poll every 60 seconds when tab is visible
    useEffect(() => {
        const interval = setInterval(() => {
            if (document.visibilityState === 'visible') {
                refreshPortfolio(valueHistoryTimeframe);
            }
        }, 60_000);
        return () => clearInterval(interval);
    }, [refreshPortfolio, valueHistoryTimeframe]);

    const handleDelete = async (ticker) => {
        if (!window.confirm(`Remove ${ticker} from portfolio?`)) return;
        try { await axios.delete(`/api/portfolio/${ticker}`); fetchPortfolio(valueHistoryTimeframe); } catch (e) { console.error(e); }
    };

    const handleUpdate = async (ticker, shares, avg_cost) => {
        try { await axios.put(`/api/portfolio/${ticker}`, { shares, avg_cost }); fetchPortfolio(valueHistoryTimeframe); } catch (e) { console.error(e); }
    };

    const { summary, holdings, value_history, risk, correlation } = data || { summary: {}, holdings: [], value_history: [], risk: {}, correlation: {} };
    const safeSummary = (summary && typeof summary === 'object') ? summary : {};
    const safeRisk = (risk && typeof risk === 'object') ? risk : {};
    const safeHoldings = Array.isArray(holdings) ? holdings : [];
    const safeValueHistory = Array.isArray(value_history) ? value_history : [];

    // Build a map of ticker -> [broker names] from synced positions
    const brokerByTicker = useMemo(() => {
        const map = {};
        for (const pos of (brokerage.positions || [])) {
            const t = pos.ticker?.toUpperCase();
            if (!t) continue;
            if (!map[t]) map[t] = [];
            const name = pos.brokerage_name;
            if (name && !map[t].includes(name)) map[t].push(name);
        }
        return map;
    }, [brokerage.positions]);
    const safeCorrelation = (correlation && typeof correlation === 'object') ? correlation : {};

    const isEmpty = safeHoldings.length === 0;
    const displayRisk = withRiskFallback(safeRisk, safeHoldings, safeCorrelation, safeSummary);

    const allocationData = useMemo(() => {
        const items = safeHoldings;
        const totalValue = items.reduce((sum, h) => sum + (Number(h.value) || 0), 0);

        if (allocationMode === 'ticker') {
            return items.map((h) => ({
                label: h.ticker,
                value: Number(h.value) || 0,
                weight: totalValue > 0 ? (Number(h.value) || 0) / totalValue : 0,
            }));
        }

        const grouped = {};
        items.forEach((h) => {
            const key = allocationMode === 'sector'
                ? (h.sector || 'Unknown')
                : sharpeBucketLabel(h.holding_sharpe);
            grouped[key] = (grouped[key] || 0) + (Number(h.value) || 0);
        });

        return Object.entries(grouped)
            .map(([label, value]) => ({
                label,
                value,
                weight: totalValue > 0 ? value / totalValue : 0,
            }))
            .sort((a, b) => b.value - a.value);
    }, [safeHoldings, allocationMode]);

    const valueHistoryChartData = useMemo(() => {
        return safeValueHistory
            .map((p) => {
                const value = Number(p?.value);
                const date = p?.date;
                if (!Number.isFinite(value) || !date) return null;
                const shortDate = valueHistoryTimeframe === '1d'
                    ? String(date).slice(11, 16)           // "HH:MM"
                    : valueHistoryTimeframe === '1w'
                        ? String(date).slice(5, 16)        // "MM-DD HH:MM"
                        : String(date).slice(5, 10);       // "MM-DD"
                return { date, shortDate, value };
            })
            .filter(Boolean);
    }, [safeValueHistory, valueHistoryTimeframe]);

    const riskItems = [
        {
            code: 'BETA',
            label: 'Portfolio Beta',
            value: asNumber(displayRisk.beta),
            definition: RISK_EXPLANATIONS.BETA,
        },
        {
            code: 'SHARPE',
            label: 'Sharpe Ratio',
            value: asNumber(displayRisk.sharpe_ratio),
            definition: RISK_EXPLANATIONS.SHARPE,
        },
        {
            code: 'VOL',
            label: 'Annualized Volatility',
            value: asPercent(displayRisk.volatility, 1),
            definition: RISK_EXPLANATIONS.VOL,
        },
        {
            code: 'MDD',
            label: 'Max Drawdown',
            value: asPercent(displayRisk.max_drawdown, 1),
            definition: RISK_EXPLANATIONS.MDD,
        },
        {
            code: 'SORTINO',
            label: 'Sortino Ratio',
            value: asNumber(displayRisk.sortino_ratio),
            definition: RISK_EXPLANATIONS.SORTINO,
        },
        {
            code: 'CALMAR',
            label: 'Calmar Ratio',
            value: asNumber(displayRisk.calmar_ratio),
            definition: RISK_EXPLANATIONS.CALMAR,
        },
        {
            code: 'VAR95',
            label: 'Value at Risk 95',
            value: asCurrency(displayRisk.var_95_dollar),
            definition: RISK_EXPLANATIONS.VAR95,
        },
        {
            code: 'CVAR',
            label: 'CVaR Expected Shortfall',
            value: asCurrency(displayRisk.cvar_95_dollar),
            definition: RISK_EXPLANATIONS.CVAR,
        },
        {
            code: 'CDD',
            label: 'Current Drawdown',
            value: asPercent(displayRisk.current_drawdown, 1),
            definition: RISK_EXPLANATIONS.CDD,
        },
        {
            code: 'SKEW',
            label: 'Skewness',
            value: asNumber(displayRisk.skewness),
            definition: RISK_EXPLANATIONS.SKEW,
        },
        {
            code: 'EN',
            label: 'Effective N',
            value: asNumber(displayRisk.effective_n),
            definition: RISK_EXPLANATIONS.EN,
        },
        {
            code: 'ALPHA',
            label: 'Jensen Alpha',
            value: asPercent(displayRisk.jensen_alpha, 2),
            definition: RISK_EXPLANATIONS.ALPHA,
        },
        {
            code: 'CAPTURE',
            label: 'Up Down Capture vs SPY',
            value: asCapture(displayRisk.up_capture, displayRisk.down_capture),
            definition: RISK_EXPLANATIONS.CAPTURE,
        },
        {
            code: 'WIN',
            label: 'Win Rate',
            value: asPercent(displayRisk.win_rate, 1),
            definition: RISK_EXPLANATIONS.WIN,
        },
        {
            code: 'ULCER',
            label: 'Ulcer Index',
            value: asPercent(displayRisk.ulcer_index, 2),
            definition: RISK_EXPLANATIONS.ULCER,
        },
    ];

    const visibleRiskItems = showAllRiskMetrics ? riskItems : riskItems.slice(0, 8);

    if (loading) return (
        <div className="flex items-center justify-center h-[60vh]">
            <div
                className="w-10 h-10 rounded-full animate-spin"
                style={{ border: '4px solid var(--accent)', borderTopColor: 'transparent' }}
            />
        </div>
    );

    return (
        <>
        <ParticlesBg canvasId="particles-portfolio" />
        <div className="relative space-y-8" style={{ zIndex: 1 }}>
            {/* Header */}
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div>
                    <h1 className="text-3xl font-black tracking-tight" style={{ color: 'var(--text-primary)' }}>My Portfolio</h1>
                    <p className="mt-1 text-sm" style={{ color: 'var(--text-secondary)' }}>Real-time performance & risk analytics</p>
                </div>
                <button
                    onClick={() => setShowAddModal(true)}
                    className="flex items-center space-x-2 font-bold px-5 py-2.5 transition-colors"
                    style={{ background: 'var(--accent)', color: '#000', borderRadius: 'var(--radius-btn)' }}
                >
                    <Plus className="h-5 w-5" />
                    <span>Add Position</span>
                </button>
            </div>

            {/* Summary Bar */}
            {!isEmpty && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {/* Portfolio total — analytics already unifies manual + synced + cash */}
                    <div className="col-span-2 glass p-5 relative overflow-hidden">
                        <p className="text-xs uppercase font-bold tracking-widest" style={{ color: 'var(--text-tertiary)' }}>Total Portfolio</p>
                        <p className="text-3xl font-black mt-1" style={{ color: 'var(--text-primary)' }}>
                            ${toFinite(safeSummary.total_value)?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}
                        </p>
                        <div
                            className="flex items-center mt-2 text-sm font-bold"
                            style={{ color: (toFinite(safeSummary.total_gain_loss) ?? 0) >= 0 ? 'var(--positive)' : 'var(--negative)' }}
                        >
                            {(toFinite(safeSummary.total_gain_loss) ?? 0) >= 0
                                ? <TrendingUp className="h-4 w-4 mr-1" />
                                : <TrendingDown className="h-4 w-4 mr-1" />
                            }
                            {(toFinite(safeSummary.total_gain_loss) ?? 0) >= 0 ? '+' : ''}${Math.abs(toFinite(safeSummary.total_gain_loss) ?? 0).toFixed(2)} ({asRawPercentNumber(safeSummary.total_gain_loss_pct, 2)}%) Total Return
                        </div>

                        {/* Breakdown — informational only, already included in total above */}
                        {(() => {
                            const totalCash = brokerage.summary?.total_cash || 0;
                            const totalV = toFinite(safeSummary.total_value) || 0;
                            // stocks_value from analytics (live prices); fall back to total - cash
                            const stocksV = toFinite(safeSummary.stocks_value) ?? Math.max(0, totalV - totalCash);
                            if (totalCash <= 0 && stocksV <= 0) return null;
                            return (
                                <div className="mt-3 pt-3 space-y-1" style={{ borderTop: '1px solid rgba(255,255,255,0.07)' }}>
                                    <div className="flex items-center justify-between text-xs">
                                        <span style={{ color: 'var(--text-tertiary)' }}>Positions value</span>
                                        <span className="font-bold" style={{ color: 'var(--text-secondary)' }}>
                                            ${stocksV.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                                        </span>
                                    </div>
                                    {totalCash > 0 && (
                                        <div className="flex items-center justify-between text-xs">
                                            <span style={{ color: 'var(--text-tertiary)' }}>Cash (dry powder)</span>
                                            <span className="font-bold" style={{ color: 'var(--text-secondary)' }}>
                                                ${totalCash.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                                            </span>
                                        </div>
                                    )}
                                </div>
                            );
                        })()}
                    </div>
                    <MetricCard icon={Target} label="Beta" value={asNumber(displayRisk.beta)} color="bg-blue-500/10 text-blue-400" />
                    <MetricCard icon={Shield} label="Sharpe Ratio" value={asNumber(displayRisk.sharpe_ratio)} color="bg-emerald-500/10 text-emerald-400" />
                </div>
            )}

            {!isEmpty && valueHistoryChartData.length > 0 && (
                <div className="glass p-5">
                    <div className="flex items-center justify-between mb-4 gap-4 flex-wrap">
                        <h3 className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--text-tertiary)' }}>
                            Portfolio Value Over Time
                        </h3>
                        <div className="flex items-center gap-2">
                            {[
                                { key: '1d', label: '1D' },
                                { key: '1w', label: '1W' },
                                { key: '1m', label: '1M' },
                                { key: '3mo', label: '3M' },
                                { key: '6mo', label: '6M' },
                                { key: '1y', label: '1Y' },
                                { key: '2y', label: '2Y' },
                            ].map(({ key, label }) => (
                                <button
                                    key={key}
                                    onClick={() => setValueHistoryTimeframe(key)}
                                    className="text-xs font-bold px-3 py-1.5 rounded-lg transition-all"
                                    style={{
                                        background: valueHistoryTimeframe === key ? 'var(--accent)' : 'rgba(255,255,255,0.08)',
                                        color: valueHistoryTimeframe === key ? '#000' : 'var(--text-secondary)',
                                        border: valueHistoryTimeframe === key ? 'none' : '1px solid rgba(255,255,255,0.12)'
                                    }}
                                >
                                    {label}
                                </button>
                            ))}
                        </div>
                    </div>
                    <div className="h-64 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={valueHistoryChartData}>
                                <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
                                <XAxis dataKey="shortDate" tick={{ fill: '#8b95a5', fontSize: 11 }} axisLine={false} tickLine={false} />
                                <YAxis
                                    tick={{ fill: '#8b95a5', fontSize: 11 }}
                                    axisLine={false}
                                    tickLine={false}
                                    width={84}
                                    domain={['auto', 'auto']}
                                    tickFormatter={(v) => `$${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                                />
                                <RechartsTooltip
                                    formatter={(v) => [`$${Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 })}`, 'Portfolio Value']}
                                    labelFormatter={(l, payload) => payload?.[0]?.payload?.date || l}
                                    contentStyle={{
                                        background: 'rgba(0,0,0,0.9)',
                                        border: '1px solid rgba(255,255,255,0.15)',
                                        borderRadius: '10px',
                                        color: '#fff',
                                    }}
                                />
                                <Line type="monotone" dataKey="value" stroke="var(--accent)" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}

            {isEmpty ? (
                /* Empty State */
                <div className="space-y-8">
                    <div
                        className="flex flex-col items-center justify-center py-20"
                        style={{
                            borderRadius: '20px',
                            border: '2px dashed rgba(255,255,255,0.12)',
                            background: 'rgba(255,255,255,0.03)',
                        }}
                    >
                        <Zap className="h-16 w-16 mb-6" style={{ color: 'rgba(255,255,255,0.15)' }} />
                        <h2 className="text-xl font-bold mb-2" style={{ color: 'var(--text-secondary)' }}>Portfolio is empty</h2>
                        <p className="mb-8 text-sm text-center max-w-sm" style={{ color: 'var(--text-tertiary)' }}>
                            Connect a brokerage account or add a position manually to track your investments.
                        </p>
                        <button
                            onClick={() => setShowAddModal(true)}
                            className="flex items-center space-x-2 font-bold px-6 py-3 transition-colors"
                            style={{ background: 'var(--accent)', color: '#000', borderRadius: 'var(--radius-btn)' }}
                        >
                            <Plus className="h-5 w-5" />
                            <span>Add Your First Position</span>
                        </button>
                    </div>
                    <BrokerageManager
                        accounts={brokerage.accounts}
                        positions={brokerage.positions}
                        balances={brokerage.balances}
                        summary={brokerage.summary}
                        isSyncing={brokerage.isSyncing}
                        syncingAccountId={brokerage.syncingAccountId}
                        syncAll={brokerage.syncAll}
                        syncAccount={brokerage.syncAccount}
                        disconnect={brokerage.disconnect}
                        fetchAccounts={brokerage.fetchAccounts}
                        getConnectUrl={brokerage.getConnectUrl}
                    />
                </div>
            ) : (
                <>
                    {/* Holdings Table */}
                    <div className="glass overflow-hidden" style={{ padding: 0 }}>
                        <div
                            className="px-5 py-4 flex items-center justify-between"
                            style={{ borderBottom: '1px solid rgba(255,255,255,0.07)' }}
                        >
                            <h3 className="text-sm font-bold uppercase tracking-widest" style={{ color: 'var(--text-tertiary)' }}>Holdings</h3>
                            <span className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
                                {safeHoldings.length} position{safeHoldings.length !== 1 ? 's' : ''}
                            </span>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr
                                        className="text-[10px] uppercase font-bold tracking-widest"
                                        style={{ color: 'var(--text-tertiary)', borderBottom: '1px solid rgba(255,255,255,0.07)' }}
                                    >
                                        <th className="py-3 px-3 text-left">Ticker</th>
                                        <th className="py-3 px-3 text-right">Shares</th>
                                        <th className="py-3 px-3 text-right">Avg Cost</th>
                                        <th className="py-3 px-3 text-right">Current</th>
                                        <th className="py-3 px-3 text-right">Value</th>
                                        <th className="py-3 px-3 text-right">Gain/Loss</th>
                                        <th className="py-3 px-3 text-right">Weight</th>
                                        <th className="py-3 px-3"></th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {safeHoldings.map(h => (
                                        <EditableRow
                                            key={h.ticker}
                                            holding={h}
                                            onDelete={handleDelete}
                                            onUpdate={handleUpdate}
                                            onRowClick={setSelectedTicker}
                                            brokerTags={brokerByTicker[h.ticker] || []}
                                        />
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* Allocation + Risk */}
                    <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
                        <div className="glass p-5 lg:col-span-2">
                            <div className="flex items-center justify-between mb-4 gap-3">
                                <h3 className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--text-tertiary)' }}>
                                    Portfolio Allocation
                                </h3>
                                <select
                                    value={allocationMode}
                                    onChange={(e) => setAllocationMode(e.target.value)}
                                    className="text-xs font-bold uppercase tracking-wide rounded-md px-2 py-1 outline-none"
                                    style={{
                                        background: 'rgba(255,255,255,0.08)',
                                        color: 'var(--text-secondary)',
                                        border: '1px solid rgba(255,255,255,0.14)'
                                    }}
                                >
                                    <option value="ticker">By Ticker</option>
                                    <option value="sector">By Sector</option>
                                    <option value="sharpe">By Sharpe Bucket</option>
                                </select>
                            </div>
                            <div className="h-56 w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <PieChart>
                                        <Pie
                                            data={allocationData}
                                            cx="50%" cy="50%"
                                            innerRadius={55} outerRadius={85}
                                            paddingAngle={3}
                                            dataKey="value"
                                            nameKey="label"
                                        >
                                            {allocationData.map((_, i) => (
                                                <Cell key={i} fill={COLORS[i % COLORS.length]} stroke="none" />
                                            ))}
                                        </Pie>
                                        <RechartsTooltip content={<CustomPieTooltip />} />
                                    </PieChart>
                                </ResponsiveContainer>
                            </div>
                            <div className="space-y-2 mt-2">
                                {allocationData.map((h, i) => (
                                    <div key={h.label} className="flex items-center justify-between text-xs">
                                        <div className="flex items-center space-x-2">
                                            <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                                            <span className="font-bold" style={{ color: 'var(--text-secondary)' }}>{h.label}</span>
                                        </div>
                                        <span style={{ color: 'var(--text-tertiary)' }}>{(h.weight * 100).toFixed(1)}%</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="glass p-5 lg:col-span-3">
                            <h3 className="text-xs font-bold uppercase tracking-widest mb-5" style={{ color: 'var(--text-tertiary)' }}>
                                Risk Profile
                            </h3>
                            <div>
                                {visibleRiskItems.map((item) => (
                                    <RiskLineItem
                                        key={item.code}
                                        code={item.code}
                                        label={item.label}
                                        value={item.value}
                                        definition={item.definition}
                                        score={riskScore(item.code, displayRisk, safeSummary)}
                                    />
                                ))}
                            </div>
                            {riskItems.length > 8 && (
                                <button
                                    onClick={() => setShowAllRiskMetrics(prev => !prev)}
                                    className="mt-3 flex items-center gap-2 text-xs font-bold uppercase tracking-wider hover:opacity-80 transition-opacity"
                                    style={{ color: 'var(--text-secondary)' }}
                                >
                                    {showAllRiskMetrics ? (
                                        <ChevronUp className="w-4 h-4" />
                                    ) : (
                                        <ChevronDown className="w-4 h-4" />
                                    )}
                                    {showAllRiskMetrics ? 'Show less' : 'Show more'}
                                </button>
                            )}
                        </div>
                    </div>

                    {/* Connected Accounts */}
                    <BrokerageManager
                        accounts={brokerage.accounts}
                        positions={brokerage.positions}
                        balances={brokerage.balances}
                        summary={brokerage.summary}
                        isSyncing={brokerage.isSyncing}
                        syncingAccountId={brokerage.syncingAccountId}
                        syncAll={brokerage.syncAll}
                        syncAccount={brokerage.syncAccount}
                        disconnect={brokerage.disconnect}
                        fetchAccounts={brokerage.fetchAccounts}
                        getConnectUrl={brokerage.getConnectUrl}
                    />

                </>
            )}

            {showAddModal && (
                <AddPositionModal onClose={() => setShowAddModal(false)} onAdded={fetchPortfolio} />
            )}

            {selectedTicker && (
                <StockIndicatorsModal
                    ticker={selectedTicker}
                    onClose={() => {
                        setSelectedTicker(null);
                        refreshPortfolio(valueHistoryTimeframe, true);
                    }}
                />
            )}
        </div>
        </>
    );
};

export default Portfolio;
