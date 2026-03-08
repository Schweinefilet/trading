import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
    PieChart, Pie, Cell, ResponsiveContainer,
    Tooltip as RechartsTooltip, LineChart, Line,
    XAxis, YAxis, CartesianGrid, Area, AreaChart
} from 'recharts';
import { TrendingUp, TrendingDown, Target, Shield, Activity, AlertTriangle, Trash2, Pencil, Check, X, Plus, Zap } from 'lucide-react';
import { Link } from 'react-router-dom';
import ParticlesBg from '../components/ParticlesBg';

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

const EditableRow = ({ holding, onDelete, onUpdate }) => {
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
            className="transition-colors"
            style={{
                borderBottom: '1px solid rgba(255,255,255,0.07)',
                background: hovered ? 'rgba(255,255,255,0.05)' : 'transparent',
            }}
            onMouseEnter={() => setHovered(true)}
            onMouseLeave={() => setHovered(false)}
        >
            <td className="py-4 px-3">
                <Link
                    to={`/stock/${holding.ticker}`}
                    className="font-bold transition-colors hover:opacity-80"
                    style={{ color: 'var(--accent)' }}
                >
                    {holding.ticker}
                </Link>
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
        const { ticker, value, weight } = payload[0].payload;
        return (
            <div className="glass p-3" style={{ borderRadius: '12px' }}>
                <p className="font-bold" style={{ color: 'var(--text-primary)' }}>{ticker}</p>
                <p style={{ color: 'var(--text-secondary)' }}>${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}</p>
                <p style={{ color: 'var(--text-secondary)' }}>{(weight * 100).toFixed(1)}%</p>
            </div>
        );
    }
    return null;
};

const Portfolio = () => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [showAddModal, setShowAddModal] = useState(false);

    const fetchPortfolio = useCallback(async () => {
        setLoading(true);
        try {
            const res = await axios.get('/api/portfolio/analytics');
            setData(res.data);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { fetchPortfolio(); }, [fetchPortfolio]);

    const handleDelete = async (ticker) => {
        if (!window.confirm(`Remove ${ticker} from portfolio?`)) return;
        try { await axios.delete(`/api/portfolio/${ticker}`); fetchPortfolio(); } catch (e) { console.error(e); }
    };

    const handleUpdate = async (ticker, shares, avg_cost) => {
        try { await axios.put(`/api/portfolio/${ticker}`, { shares, avg_cost }); fetchPortfolio(); } catch (e) { console.error(e); }
    };

    if (loading) return (
        <div className="flex items-center justify-center h-[60vh]">
            <div
                className="w-10 h-10 rounded-full animate-spin"
                style={{ border: '4px solid var(--accent)', borderTopColor: 'transparent' }}
            />
        </div>
    );

    const { summary, holdings, risk, correlation } = data || { summary: {}, holdings: [], risk: {}, correlation: {} };
    const isEmpty = !holdings || holdings.length === 0;

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
                    <div className="col-span-2 glass p-5 relative overflow-hidden">
                        <p className="text-xs uppercase font-bold tracking-widest" style={{ color: 'var(--text-tertiary)' }}>Total Market Value</p>
                        <p className="text-3xl font-black mt-1" style={{ color: 'var(--text-primary)' }}>
                            ${summary.total_value?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}
                        </p>
                        <div
                            className="flex items-center mt-2 text-sm font-bold"
                            style={{ color: summary.total_gain_loss >= 0 ? 'var(--positive)' : 'var(--negative)' }}
                        >
                            {summary.total_gain_loss >= 0
                                ? <TrendingUp className="h-4 w-4 mr-1" />
                                : <TrendingDown className="h-4 w-4 mr-1" />
                            }
                            {summary.total_gain_loss >= 0 ? '+' : ''}${Math.abs(summary.total_gain_loss).toFixed(2)} ({summary.total_gain_loss_pct?.toFixed(2)}%) Total Return
                        </div>
                        <TrendingUp
                            className="absolute -right-6 -bottom-6 h-32 w-32 select-none pointer-events-none"
                            style={{ color: 'rgba(10,132,255,0.08)' }}
                        />
                    </div>
                    <MetricCard icon={Target} label="Beta" value={risk.beta?.toFixed(2) || '—'} color="bg-blue-500/10 text-blue-400" />
                    <MetricCard icon={Shield} label="Sharpe Ratio" value={risk.sharpe_ratio?.toFixed(2) || '—'} color="bg-emerald-500/10 text-emerald-400" />
                </div>
            )}

            {isEmpty ? (
                /* Empty State */
                <div
                    className="flex flex-col items-center justify-center py-32"
                    style={{
                        borderRadius: '20px',
                        border: '2px dashed rgba(255,255,255,0.12)',
                        background: 'rgba(255,255,255,0.03)',
                    }}
                >
                    <Zap className="h-16 w-16 mb-6" style={{ color: 'rgba(255,255,255,0.15)' }} />
                    <h2 className="text-xl font-bold mb-2" style={{ color: 'var(--text-secondary)' }}>Portfolio is empty</h2>
                    <p className="mb-8 text-sm text-center max-w-sm" style={{ color: 'var(--text-tertiary)' }}>
                        Start by adding your first stock position to track your investments and see risk analytics.
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
            ) : (
                <>
                    {/* Allocation + Risk */}
                    <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
                        <div className="glass p-5 lg:col-span-2">
                            <h3 className="text-xs font-bold uppercase tracking-widest mb-4" style={{ color: 'var(--text-tertiary)' }}>
                                Portfolio Allocation
                            </h3>
                            <div className="h-56 w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <PieChart>
                                        <Pie
                                            data={holdings}
                                            cx="50%" cy="50%"
                                            innerRadius={55} outerRadius={85}
                                            paddingAngle={3}
                                            dataKey="value"
                                            nameKey="ticker"
                                        >
                                            {holdings.map((_, i) => (
                                                <Cell key={i} fill={COLORS[i % COLORS.length]} stroke="none" />
                                            ))}
                                        </Pie>
                                        <RechartsTooltip content={<CustomPieTooltip />} />
                                    </PieChart>
                                </ResponsiveContainer>
                            </div>
                            <div className="space-y-2 mt-2">
                                {holdings.map((h, i) => (
                                    <div key={h.ticker} className="flex items-center justify-between text-xs">
                                        <div className="flex items-center space-x-2">
                                            <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                                            <span className="font-bold" style={{ color: 'var(--text-secondary)' }}>{h.ticker}</span>
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
                            <div className="grid grid-cols-2 gap-4">
                                <MetricCard
                                    icon={Target} label="Portfolio Beta"
                                    value={risk.beta?.toFixed(2) || '—'}
                                    color="bg-blue-500/10 text-blue-400"
                                />
                                <MetricCard
                                    icon={Shield} label="Sharpe Ratio"
                                    value={risk.sharpe_ratio?.toFixed(2) || '—'}
                                    color="bg-emerald-500/10 text-emerald-400"
                                />
                                <MetricCard
                                    icon={Activity} label="Ann. Volatility"
                                    value={risk.volatility ? `${(risk.volatility * 100).toFixed(1)}%` : '—'}
                                    color="bg-amber-500/10 text-amber-400"
                                />
                                <MetricCard
                                    icon={AlertTriangle} label="Max Drawdown"
                                    value={risk.max_drawdown ? `${(risk.max_drawdown * 100).toFixed(1)}%` : '—'}
                                    color="bg-rose-500/10 text-rose-400"
                                />
                            </div>
                        </div>
                    </div>

                    {/* Holdings Table */}
                    <div className="glass overflow-hidden" style={{ padding: 0 }}>
                        <div
                            className="px-5 py-4 flex items-center justify-between"
                            style={{ borderBottom: '1px solid rgba(255,255,255,0.07)' }}
                        >
                            <h3 className="text-sm font-bold uppercase tracking-widest" style={{ color: 'var(--text-tertiary)' }}>Holdings</h3>
                            <span className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
                                {holdings.length} position{holdings.length !== 1 ? 's' : ''}
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
                                    {holdings.map(h => (
                                        <EditableRow
                                            key={h.ticker}
                                            holding={h}
                                            onDelete={handleDelete}
                                            onUpdate={handleUpdate}
                                        />
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* Correlation Heatmap */}
                    {holdings.length >= 2 && correlation && Object.keys(correlation).length > 0 && (
                        <div className="glass p-5">
                            <div className="flex items-center justify-between mb-5">
                                <h3 className="text-sm font-bold uppercase tracking-widest" style={{ color: 'var(--text-tertiary)' }}>
                                    Correlation Matrix
                                </h3>
                                <span className="text-xs" style={{ color: 'var(--text-tertiary)' }}>1-Year Daily Returns</span>
                            </div>
                            <div className="overflow-x-auto">
                                <table className="text-xs mx-auto">
                                    <thead>
                                        <tr>
                                            <th className="p-2 w-16"></th>
                                            {holdings.map(h => (
                                                <th key={h.ticker} className="p-2 font-black uppercase w-16 text-center" style={{ color: 'var(--text-secondary)' }}>
                                                    {h.ticker}
                                                </th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {holdings.map(h1 => (
                                            <tr key={h1.ticker}>
                                                <td className="p-2 font-black uppercase text-right pr-3" style={{ color: 'var(--text-secondary)' }}>
                                                    {h1.ticker}
                                                </td>
                                                {holdings.map(h2 => {
                                                    const val = correlation?.[h1.ticker]?.[h2.ticker] ?? 0;
                                                    const isIdentity = h1.ticker === h2.ticker;
                                                    let bg = 'rgba(255,255,255,0.06)';
                                                    let textColor = 'var(--text-tertiary)';
                                                    if (isIdentity) {
                                                        bg = 'rgba(255,255,255,0.10)'; textColor = 'var(--text-secondary)';
                                                    } else if (val > 0.7) {
                                                        bg = 'rgba(48,209,88,0.28)'; textColor = '#a8f5be';
                                                    } else if (val > 0.3) {
                                                        bg = 'rgba(48,209,88,0.14)'; textColor = 'var(--positive)';
                                                    } else if (val < -0.3) {
                                                        bg = 'rgba(255,69,58,0.24)'; textColor = '#ffa5a0';
                                                    } else if (val < -0.1) {
                                                        bg = 'rgba(255,69,58,0.11)'; textColor = 'var(--negative)';
                                                    }
                                                    return (
                                                        <td key={h2.ticker} className="p-1">
                                                            <div
                                                                className="w-14 h-10 flex items-center justify-center rounded-md"
                                                                style={{ background: bg }}
                                                            >
                                                                <span className="font-bold" style={{ color: textColor }}>{val.toFixed(2)}</span>
                                                            </div>
                                                        </td>
                                                    );
                                                })}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                            <p className="text-xs mt-4 text-center" style={{ color: 'var(--text-tertiary)' }}>
                                🟢 High positive correlation · ⬜ Low correlation · 🔴 Negative correlation
                            </p>
                        </div>
                    )}
                </>
            )}

            {showAddModal && (
                <AddPositionModal onClose={() => setShowAddModal(false)} onAdded={fetchPortfolio} />
            )}
        </div>
        </>
    );
};

export default Portfolio;
