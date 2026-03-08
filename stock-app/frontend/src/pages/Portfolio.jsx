import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
    PieChart, Pie, Cell, ResponsiveContainer,
    Tooltip as RechartsTooltip, LineChart, Line,
    XAxis, YAxis, CartesianGrid, Area, AreaChart
} from 'recharts';
import { TrendingUp, TrendingDown, Target, Shield, Activity, AlertTriangle, Trash2, Pencil, Check, X, Plus, Zap } from 'lucide-react';
import { Link } from 'react-router-dom';

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#6366f1', '#14b8a6', '#f97316'];

const MetricCard = ({ icon: Icon, label, value, color }) => (
    <div className="bg-slate-800/60 rounded-xl p-5 flex flex-col items-center text-center group hover:bg-slate-800 transition-colors border border-slate-700/50">
        <div className={`w-11 h-11 rounded-full flex items-center justify-center mb-3 ${color}`}>
            <Icon className="h-5 w-5" />
        </div>
        <p className="text-2xl font-black text-white tabular-nums">{value}</p>
        <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mt-1">{label}</p>
    </div>
);

const EditableRow = ({ holding, onDelete, onUpdate }) => {
    const [editing, setEditing] = useState(false);
    const [shares, setShares] = useState(holding.shares);
    const [cost, setCost] = useState(holding.avg_cost);

    const handleSave = async () => {
        await onUpdate(holding.ticker, parseFloat(shares), parseFloat(cost));
        setEditing(false);
    };

    const isPositive = holding.gain_loss >= 0;

    return (
        <tr className="hover:bg-slate-800/50 transition-colors border-b border-slate-800/80">
            <td className="py-4 px-3">
                <Link to={`/stock/${holding.ticker}`} className="font-bold text-blue-400 hover:text-blue-300 transition-colors">
                    {holding.ticker}
                </Link>
            </td>
            <td className="py-4 px-3 text-right">
                {editing ? (
                    <input
                        type="number"
                        value={shares}
                        onChange={e => setShares(e.target.value)}
                        className="w-24 bg-slate-700 border border-slate-600 rounded px-2 py-1 text-white text-right text-sm"
                    />
                ) : (
                    <span className="text-slate-300">{holding.shares}</span>
                )}
            </td>
            <td className="py-4 px-3 text-right">
                {editing ? (
                    <input
                        type="number"
                        value={cost}
                        onChange={e => setCost(e.target.value)}
                        className="w-24 bg-slate-700 border border-slate-600 rounded px-2 py-1 text-white text-right text-sm"
                    />
                ) : (
                    <span className="text-slate-300">${holding.avg_cost.toFixed(2)}</span>
                )}
            </td>
            <td className="py-4 px-3 text-right font-medium text-white">${holding.current_price.toFixed(2)}</td>
            <td className="py-4 px-3 text-right font-bold text-white">
                ${holding.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
            </td>
            <td className={`py-4 px-3 text-right font-bold ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
                <div className="flex flex-col items-end">
                    <span>{isPositive ? '+' : ''}{holding.gain_loss.toFixed(2)}</span>
                    <span className="text-xs opacity-75">{isPositive ? '+' : ''}{holding.gain_loss_pct.toFixed(2)}%</span>
                </div>
            </td>
            <td className="py-4 px-3 text-right">
                <div className="flex flex-col items-end">
                    <span className="text-slate-400 text-sm">{(holding.weight * 100).toFixed(1)}%</span>
                    <div className="w-16 h-1 bg-slate-700 rounded mt-1">
                        <div className="h-1 bg-blue-500 rounded" style={{ width: `${holding.weight * 100}%` }} />
                    </div>
                </div>
            </td>
            <td className="py-4 px-3 text-right">
                <div className="flex items-center justify-end space-x-1">
                    {editing ? (
                        <>
                            <button onClick={handleSave} className="p-1 text-emerald-400 hover:text-emerald-300 transition-colors"><Check className="h-4 w-4" /></button>
                            <button onClick={() => setEditing(false)} className="p-1 text-slate-500 hover:text-slate-300 transition-colors"><X className="h-4 w-4" /></button>
                        </>
                    ) : (
                        <>
                            <button onClick={() => setEditing(true)} className="p-1 text-slate-600 hover:text-slate-300 transition-colors"><Pencil className="h-4 w-4" /></button>
                            <button onClick={() => onDelete(holding.ticker)} className="p-1 text-slate-600 hover:text-rose-500 transition-colors"><Trash2 className="h-4 w-4" /></button>
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
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <div className="bg-slate-800 border border-slate-700 rounded-2xl p-6 w-full max-w-sm shadow-2xl">
                <div className="flex justify-between items-center mb-6">
                    <h3 className="text-xl font-bold">Add Position</h3>
                    <button onClick={onClose} className="text-slate-500 hover:text-slate-200"><X className="h-5 w-5" /></button>
                </div>
                <div className="space-y-4">
                    <div>
                        <label className="block text-xs text-slate-400 mb-1 uppercase font-bold">Ticker Symbol</label>
                        <input
                            type="text"
                            placeholder="e.g. AAPL"
                            className="w-full bg-slate-700 border border-slate-600 rounded-lg p-2.5 text-white uppercase focus:outline-none focus:ring-2 focus:ring-blue-500"
                            value={ticker}
                            onChange={e => setTicker(e.target.value.toUpperCase())}
                        />
                    </div>
                    <div>
                        <label className="block text-xs text-slate-400 mb-1 uppercase font-bold">Number of Shares</label>
                        <input
                            type="number"
                            placeholder="e.g. 10"
                            className="w-full bg-slate-700 border border-slate-600 rounded-lg p-2.5 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                            value={shares}
                            onChange={e => setShares(e.target.value)}
                        />
                    </div>
                    <div>
                        <label className="block text-xs text-slate-400 mb-1 uppercase font-bold">Average Cost Basis (per share)</label>
                        <input
                            type="number"
                            placeholder="e.g. 175.00"
                            className="w-full bg-slate-700 border border-slate-600 rounded-lg p-2.5 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                            value={cost}
                            onChange={e => setCost(e.target.value)}
                        />
                    </div>
                    {error && <p className="text-rose-400 text-sm">{error}</p>}
                    <div className="flex space-x-3 pt-2">
                        <button onClick={onClose} className="flex-1 py-2.5 rounded-lg bg-slate-700 hover:bg-slate-600 text-white transition-colors">Cancel</button>
                        <button
                            disabled={loading}
                            onClick={handleAdd}
                            className="flex-1 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-bold transition-colors disabled:opacity-50"
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
            <div className="bg-slate-900 border border-slate-700 p-3 rounded-lg text-sm">
                <p className="font-bold text-white">{ticker}</p>
                <p className="text-slate-400">${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}</p>
                <p className="text-slate-400">{(weight * 100).toFixed(1)}%</p>
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
            <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
        </div>
    );

    const { summary, holdings, risk, correlation } = data || { summary: {}, holdings: [], risk: {}, correlation: {} };
    const isEmpty = !holdings || holdings.length === 0;

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div>
                    <h1 className="text-3xl font-black text-white tracking-tight">My Portfolio</h1>
                    <p className="text-slate-400 mt-1 text-sm">Real-time performance & risk analytics</p>
                </div>
                <button
                    onClick={() => setShowAddModal(true)}
                    className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-500 text-white font-bold px-4 py-2.5 rounded-xl transition-colors shadow-lg shadow-blue-600/20"
                >
                    <Plus className="h-5 w-5" />
                    <span>Add Position</span>
                </button>
            </div>

            {/* Summary Bar */}
            {!isEmpty && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="col-span-2 bg-gradient-to-br from-blue-600/20 to-slate-800 border border-blue-500/20 rounded-2xl p-5 relative overflow-hidden">
                        <p className="text-xs text-slate-500 uppercase font-bold tracking-widest">Total Market Value</p>
                        <p className="text-3xl font-black text-white mt-1">
                            ${summary.total_value?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}
                        </p>
                        <div className={`flex items-center mt-2 text-sm font-bold ${summary.total_gain_loss >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {summary.total_gain_loss >= 0 ? <TrendingUp className="h-4 w-4 mr-1" /> : <TrendingDown className="h-4 w-4 mr-1" />}
                            {summary.total_gain_loss >= 0 ? '+' : ''}${Math.abs(summary.total_gain_loss).toFixed(2)} ({summary.total_gain_loss_pct?.toFixed(2)}%) Total Return
                        </div>
                        <TrendingUp className="absolute -right-6 -bottom-6 text-blue-500/10 h-32 w-32 select-none pointer-events-none" />
                    </div>
                    <MetricCard icon={Target} label="Beta" value={risk.beta?.toFixed(2) || '—'} color="bg-blue-500/10 text-blue-400" />
                    <MetricCard icon={Shield} label="Sharpe Ratio" value={risk.sharpe_ratio?.toFixed(2) || '—'} color="bg-emerald-500/10 text-emerald-400" />
                </div>
            )}

            {isEmpty ? (
                /* Empty State */
                <div className="flex flex-col items-center justify-center py-32 border-2 border-dashed border-slate-700 rounded-2xl">
                    <Zap className="h-16 w-16 text-slate-700 mb-6" />
                    <h2 className="text-xl font-bold text-slate-400 mb-2">Portfolio is empty</h2>
                    <p className="text-slate-600 mb-8 text-sm text-center max-w-sm">
                        Start by adding your first stock position to track your investments and see risk analytics.
                    </p>
                    <button
                        onClick={() => setShowAddModal(true)}
                        className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-500 text-white font-bold px-6 py-3 rounded-xl transition-colors"
                    >
                        <Plus className="h-5 w-5" />
                        <span>Add Your First Position</span>
                    </button>
                </div>
            ) : (
                <>
                    {/* Allocation + Risk */}
                    <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
                        <div className="bg-slate-800 border border-slate-700/50 rounded-2xl p-5 lg:col-span-2">
                            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Portfolio Allocation</h3>
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
                                            <span className="font-bold text-slate-200">{h.ticker}</span>
                                        </div>
                                        <span className="text-slate-500">{(h.weight * 100).toFixed(1)}%</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="bg-slate-800 border border-slate-700/50 rounded-2xl p-5 lg:col-span-3">
                            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-5">Risk Profile</h3>
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
                    <div className="bg-slate-800 border border-slate-700/50 rounded-2xl overflow-hidden">
                        <div className="px-5 py-4 border-b border-slate-700/50 flex items-center justify-between">
                            <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest">Holdings</h3>
                            <span className="text-xs text-slate-500">{holdings.length} position{holdings.length !== 1 ? 's' : ''}</span>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="text-[10px] uppercase text-slate-500 border-b border-slate-700/50">
                                        <th className="py-3 px-3 text-left font-bold tracking-widest">Ticker</th>
                                        <th className="py-3 px-3 text-right font-bold tracking-widest">Shares</th>
                                        <th className="py-3 px-3 text-right font-bold tracking-widest">Avg Cost</th>
                                        <th className="py-3 px-3 text-right font-bold tracking-widest">Current</th>
                                        <th className="py-3 px-3 text-right font-bold tracking-widest">Value</th>
                                        <th className="py-3 px-3 text-right font-bold tracking-widest">Gain/Loss</th>
                                        <th className="py-3 px-3 text-right font-bold tracking-widest">Weight</th>
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
                        <div className="bg-slate-800 border border-slate-700/50 rounded-2xl p-5">
                            <div className="flex items-center justify-between mb-5">
                                <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest">Correlation Matrix</h3>
                                <span className="text-xs text-slate-500">1-Year Daily Returns</span>
                            </div>
                            <div className="overflow-x-auto">
                                <table className="text-xs mx-auto">
                                    <thead>
                                        <tr>
                                            <th className="p-2 w-16"></th>
                                            {holdings.map(h => (
                                                <th key={h.ticker} className="p-2 text-slate-400 font-black uppercase w-16 text-center">{h.ticker}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {holdings.map(h1 => (
                                            <tr key={h1.ticker}>
                                                <td className="p-2 text-slate-400 font-black uppercase text-right pr-3">{h1.ticker}</td>
                                                {holdings.map(h2 => {
                                                    const val = correlation?.[h1.ticker]?.[h2.ticker] ?? 0;
                                                    const isIdentity = h1.ticker === h2.ticker;
                                                    let bg = 'bg-slate-700';
                                                    let textColor = 'text-slate-400';
                                                    if (isIdentity) {
                                                        bg = 'bg-slate-600'; textColor = 'text-slate-300';
                                                    } else if (val > 0.7) {
                                                        bg = 'bg-emerald-500/40'; textColor = 'text-emerald-200';
                                                    } else if (val > 0.3) {
                                                        bg = 'bg-emerald-500/20'; textColor = 'text-emerald-300';
                                                    } else if (val < -0.3) {
                                                        bg = 'bg-rose-500/30'; textColor = 'text-rose-300';
                                                    } else if (val < -0.1) {
                                                        bg = 'bg-rose-500/15'; textColor = 'text-rose-400';
                                                    }
                                                    return (
                                                        <td key={h2.ticker} className="p-1">
                                                            <div className={`w-14 h-10 flex items-center justify-center rounded-md ${bg}`}>
                                                                <span className={`font-bold ${textColor}`}>{val.toFixed(2)}</span>
                                                            </div>
                                                        </td>
                                                    );
                                                })}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                            <p className="text-xs text-slate-600 mt-4 text-center">
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
    );
};

export default Portfolio;
