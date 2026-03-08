import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { X } from 'lucide-react';

// Format large numbers: e.g. 2940000000000 -> "$2.94T"
function fmtNum(v) {
    if (v == null || v === '' || isNaN(v)) return '—';
    const n = Number(v);
    if (Math.abs(n) >= 1e12) return `$${(n / 1e12).toFixed(2)}T`;
    if (Math.abs(n) >= 1e9)  return `$${(n / 1e9).toFixed(1)}B`;
    if (Math.abs(n) >= 1e6)  return `$${(n / 1e6).toFixed(1)}M`;
    if (Math.abs(n) >= 1e3)  return `$${(n / 1e3).toFixed(1)}K`;
    return `$${n.toFixed(2)}`;
}

// Format percentages
function fmtPct(v) {
    if (v == null || v === '' || isNaN(v)) return '—';
    return `${(Number(v) * 100).toFixed(2)}%`;
}

// Format plain numbers (like ratios)
function fmtRatio(v, decimals = 2) {
    if (v == null || v === '' || isNaN(v)) return '—';
    return Number(v).toFixed(decimals);
}

function MetricCard({ label, value }) {
    return (
        <div className="bg-[#1e222d] rounded-lg p-3">
            <div className="text-slate-500 text-xs mb-1">{label}</div>
            <div className="text-white font-bold text-sm">{value}</div>
        </div>
    );
}

function RatingsBar({ label, count, total, color }) {
    const pct = total > 0 ? (count / total) * 100 : 0;
    return (
        <div className="flex items-center gap-2 py-1">
            <div className="text-xs text-slate-400 w-24 flex-shrink-0">{label}</div>
            <div className="flex-grow bg-[#2a2e39] rounded-full h-3 overflow-hidden">
                <div
                    className="h-3 rounded-full transition-all"
                    style={{ width: `${pct}%`, backgroundColor: color }}
                />
            </div>
            <div className="text-xs text-slate-300 w-6 text-right flex-shrink-0">{count}</div>
        </div>
    );
}

function gradeColor(grade) {
    if (!grade) return 'text-slate-400';
    const g = grade.toLowerCase();
    if (g.includes('strong buy') || g.includes('strongbuy')) return 'text-emerald-400';
    if (g.includes('buy') || g.includes('outperform') || g.includes('overweight') || g.includes('positive')) return 'text-green-400';
    if (g.includes('strong sell') || g.includes('strongsell') || g.includes('underperform') || g.includes('underweight') || g.includes('negative')) return 'text-red-400';
    if (g.includes('sell')) return 'text-red-400';
    return 'text-slate-400';
}

const StatsPanel = ({ ticker, onClose }) => {
    const [tab, setTab] = useState('overview');
    const [fundamentals, setFundamentals] = useState(null);
    const [analyst, setAnalyst] = useState(null);
    const [loadingFund, setLoadingFund] = useState(true);
    const [loadingAnalyst, setLoadingAnalyst] = useState(true);

    useEffect(() => {
        setLoadingFund(true);
        setLoadingAnalyst(true);
        setFundamentals(null);
        setAnalyst(null);

        axios.get(`http://localhost:5001/api/stock/${ticker}/fundamentals`)
            .then(r => setFundamentals(r.data))
            .catch(() => setFundamentals(null))
            .finally(() => setLoadingFund(false));

        axios.get(`http://localhost:5001/api/stock/${ticker}/analyst`)
            .then(r => setAnalyst(r.data))
            .catch(() => setAnalyst(null))
            .finally(() => setLoadingAnalyst(false));
    }, [ticker]);

    const f = fundamentals || {};

    // 52-week range visual
    const week52Low = f.fifty_two_week_low;
    const week52High = f.fifty_two_week_high;
    // Use market_cap as a proxy; we need current price — grab from quote or use fundamentals info
    // We'll compute a rough current price from context; fundamentals may not have it directly
    // Use a neutral fallback — the bar just shows low/high range with no current marker if unavailable

    const renderOverview = () => {
        if (loadingFund) {
            return (
                <div className="flex items-center justify-center h-40">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500" />
                </div>
            );
        }
        if (!fundamentals) {
            return <div className="text-slate-400 text-sm p-4">Fundamentals data unavailable.</div>;
        }

        return (
            <div className="p-4 space-y-4">
                {/* Company header */}
                <div className="bg-[#1e222d] rounded-lg p-4">
                    <div className="text-white font-bold text-base">{f.longName || ticker}</div>
                    {(f.sector || f.industry) && (
                        <div className="flex flex-wrap gap-2 mt-2">
                            {f.sector && (
                                <span className="px-2 py-0.5 bg-[#2a2e39] rounded text-xs text-slate-300">{f.sector}</span>
                            )}
                            {f.industry && (
                                <span className="px-2 py-0.5 bg-[#2a2e39] rounded text-xs text-slate-400">{f.industry}</span>
                            )}
                        </div>
                    )}
                </div>

                {/* Valuation */}
                <div>
                    <div className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-2 px-1">Valuation</div>
                    <div className="grid grid-cols-2 gap-2">
                        <MetricCard label="P/E Ratio" value={fmtRatio(f.pe_ratio)} />
                        <MetricCard label="Forward P/E" value={fmtRatio(f.forward_pe)} />
                        <MetricCard label="P/B Ratio" value={fmtRatio(f.pb_ratio)} />
                        <MetricCard label="P/S Ratio" value={fmtRatio(f.ps_ratio)} />
                        <MetricCard label="EV/EBITDA" value={fmtRatio(f.ev_ebitda)} />
                        <MetricCard label="PEG Ratio" value={fmtRatio(f.peg_ratio)} />
                    </div>
                </div>

                {/* Per Share */}
                <div>
                    <div className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-2 px-1">Per Share</div>
                    <div className="grid grid-cols-2 gap-2">
                        <MetricCard label="EPS (TTM)" value={fmtRatio(f.eps_ttm)} />
                        <MetricCard label="EPS (Forward)" value={fmtRatio(f.eps_forward)} />
                    </div>
                </div>

                {/* Financials */}
                <div>
                    <div className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-2 px-1">Financials</div>
                    <div className="grid grid-cols-2 gap-2">
                        <MetricCard label="Revenue (TTM)" value={fmtNum(f.revenue_ttm)} />
                        <MetricCard label="Market Cap" value={fmtNum(f.market_cap)} />
                        <MetricCard label="Net Income" value={fmtNum(f.net_income_ttm)} />
                        <MetricCard label="Free Cash Flow" value={fmtNum(f.free_cash_flow)} />
                        <MetricCard label="Profit Margin" value={fmtPct(f.profit_margin)} />
                        <MetricCard label="ROE" value={fmtPct(f.roe)} />
                        <MetricCard label="ROA" value={fmtPct(f.roa)} />
                        <MetricCard label="Debt / Equity" value={fmtRatio(f.debt_to_equity)} />
                    </div>
                </div>

                {/* Risk / Other */}
                <div>
                    <div className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-2 px-1">Risk & Other</div>
                    <div className="grid grid-cols-2 gap-2">
                        <MetricCard label="Beta" value={fmtRatio(f.beta)} />
                        <MetricCard
                            label="Dividend Yield"
                            value={f.dividend_yield != null ? `${Number(f.dividend_yield).toFixed(2)}%` : '—'}
                        />
                    </div>

                    {/* 52-Week Range visual */}
                    {week52Low != null && week52High != null && (
                        <div className="bg-[#1e222d] rounded-lg p-3 mt-2">
                            <div className="text-slate-500 text-xs mb-2">52-Week Range</div>
                            <div className="flex items-center justify-between text-xs text-slate-400 mb-1.5">
                                <span>${Number(week52Low).toFixed(2)}</span>
                                <span>${Number(week52High).toFixed(2)}</span>
                            </div>
                            <div className="relative h-2 bg-[#2a2e39] rounded-full">
                                <div className="absolute inset-0 rounded-full bg-gradient-to-r from-red-500 via-yellow-400 to-green-500 opacity-60" />
                            </div>
                        </div>
                    )}
                </div>
            </div>
        );
    };

    const renderAnalyst = () => {
        if (loadingAnalyst) {
            return (
                <div className="flex items-center justify-center h-40">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500" />
                </div>
            );
        }
        if (!analyst) {
            return <div className="text-slate-400 text-sm p-4">Analyst data unavailable.</div>;
        }

        const ratings = analyst.ratings || [];
        const currentRatings = ratings[0] || {};
        const totalRatings = (currentRatings.strongBuy || 0) + (currentRatings.buy || 0) +
            (currentRatings.hold || 0) + (currentRatings.sell || 0) + (currentRatings.strongSell || 0);

        const pt = analyst.price_targets || {};
        const upgrades = analyst.upgrades_downgrades || [];

        const ptLow = pt.low != null ? Number(pt.low) : null;
        const ptMean = pt.mean != null ? Number(pt.mean) : null;
        const ptHigh = pt.high != null ? Number(pt.high) : null;
        const ptCurrent = pt.current != null ? Number(pt.current) : null;

        // Compute % upside from current to mean
        let upsidePct = null;
        if (ptMean != null && ptCurrent != null && ptCurrent !== 0) {
            upsidePct = ((ptMean - ptCurrent) / ptCurrent) * 100;
        }

        // Compute positions for the range bar
        let currentPct = null;
        let meanPct = null;
        if (ptLow != null && ptHigh != null && ptHigh > ptLow) {
            if (ptCurrent != null) {
                currentPct = Math.min(100, Math.max(0, ((ptCurrent - ptLow) / (ptHigh - ptLow)) * 100));
            }
            if (ptMean != null) {
                meanPct = Math.min(100, Math.max(0, ((ptMean - ptLow) / (ptHigh - ptLow)) * 100));
            }
        }

        return (
            <div className="p-4 space-y-4">
                {/* Ratings Distribution */}
                {totalRatings > 0 && (
                    <div>
                        <div className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-2 px-1">
                            Ratings Distribution
                            {currentRatings.period && (
                                <span className="ml-2 normal-case text-slate-600">({currentRatings.period})</span>
                            )}
                        </div>
                        <div className="bg-[#1e222d] rounded-lg p-3 space-y-1">
                            <RatingsBar label="Strong Buy" count={currentRatings.strongBuy || 0} total={totalRatings} color="#26a69a" />
                            <RatingsBar label="Buy" count={currentRatings.buy || 0} total={totalRatings} color="#4caf50" />
                            <RatingsBar label="Hold" count={currentRatings.hold || 0} total={totalRatings} color="#ff9800" />
                            <RatingsBar label="Sell" count={currentRatings.sell || 0} total={totalRatings} color="#ef5350" />
                            <RatingsBar label="Strong Sell" count={currentRatings.strongSell || 0} total={totalRatings} color="#b71c1c" />
                        </div>
                    </div>
                )}

                {/* Price Targets */}
                {(ptLow != null || ptMean != null || ptHigh != null) && (
                    <div>
                        <div className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-2 px-1">
                            Price Targets
                            {pt.numberOfAnalysts != null && (
                                <span className="ml-2 normal-case text-slate-600">{pt.numberOfAnalysts} analysts</span>
                            )}
                        </div>
                        <div className="bg-[#1e222d] rounded-lg p-3">
                            <div className="grid grid-cols-3 gap-2 mb-4">
                                {ptLow != null && (
                                    <div className="bg-[#2a0a0a] border border-red-900 rounded p-2 text-center">
                                        <div className="text-xs text-red-400 mb-1">Low</div>
                                        <div className="text-white font-bold text-sm">${ptLow.toFixed(2)}</div>
                                    </div>
                                )}
                                {ptMean != null && (
                                    <div className="bg-[#0d1a2a] border border-blue-800 rounded p-2 text-center">
                                        <div className="text-xs text-blue-400 mb-1">Mean</div>
                                        <div className="text-white font-bold text-sm">${ptMean.toFixed(2)}</div>
                                        {upsidePct != null && (
                                            <div className={`text-[10px] mt-0.5 ${upsidePct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                {upsidePct >= 0 ? '+' : ''}{upsidePct.toFixed(1)}%
                                            </div>
                                        )}
                                    </div>
                                )}
                                {ptHigh != null && (
                                    <div className="bg-[#0a2a0d] border border-green-900 rounded p-2 text-center">
                                        <div className="text-xs text-green-400 mb-1">High</div>
                                        <div className="text-white font-bold text-sm">${ptHigh.toFixed(2)}</div>
                                    </div>
                                )}
                            </div>

                            {/* Visual range bar */}
                            {ptLow != null && ptHigh != null && (
                                <div>
                                    <div className="relative h-2 rounded-full overflow-visible" style={{ background: 'linear-gradient(to right, #ef5350, #ff9800, #26a69a)' }}>
                                        {currentPct != null && (
                                            <div
                                                className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full border-2 border-[#131722] shadow"
                                                style={{ left: `${currentPct}%`, transform: 'translate(-50%, -50%)' }}
                                                title={`Current: $${ptCurrent?.toFixed(2)}`}
                                            />
                                        )}
                                        {meanPct != null && (
                                            <div
                                                className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-blue-400 rounded-full border-2 border-[#131722] shadow"
                                                style={{ left: `${meanPct}%`, transform: 'translate(-50%, -50%)' }}
                                                title={`Mean target: $${ptMean?.toFixed(2)}`}
                                            />
                                        )}
                                    </div>
                                    <div className="flex justify-between text-[10px] text-slate-500 mt-2">
                                        <span>${ptLow.toFixed(2)}</span>
                                        <span className="text-slate-600 text-[9px]">
                                            <span className="inline-flex items-center gap-1">
                                                <span className="w-2 h-2 rounded-full bg-white inline-block" /> Current
                                                <span className="w-2 h-2 rounded-full bg-blue-400 inline-block ml-1" /> Mean
                                            </span>
                                        </span>
                                        <span>${ptHigh.toFixed(2)}</span>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* Recent Analyst Actions */}
                {upgrades.length > 0 && (
                    <div>
                        <div className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-2 px-1">Recent Analyst Actions</div>
                        <div className="bg-[#1e222d] rounded-lg overflow-hidden">
                            <table className="w-full text-xs">
                                <thead>
                                    <tr className="border-b border-[#2a2e39]">
                                        <th className="text-left px-3 py-2 text-slate-500 font-medium">Date</th>
                                        <th className="text-left px-3 py-2 text-slate-500 font-medium">Firm</th>
                                        <th className="text-left px-3 py-2 text-slate-500 font-medium">Grade</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {upgrades.slice(0, 8).map((row, i) => (
                                        <tr key={i} className="border-b border-[#2a2e39] last:border-0 hover:bg-[#2a2e39] transition-colors">
                                            <td className="px-3 py-2 text-slate-400">
                                                {row.GradeDate ? String(row.GradeDate).slice(0, 10) : '—'}
                                            </td>
                                            <td className="px-3 py-2 text-slate-300 max-w-[100px] truncate">{row.Firm || '—'}</td>
                                            <td className="px-3 py-2">
                                                {row.FromGrade && (
                                                    <span className={`${gradeColor(row.FromGrade)} opacity-60`}>{row.FromGrade}</span>
                                                )}
                                                {row.FromGrade && row.ToGrade && (
                                                    <span className="text-slate-600 mx-1">→</span>
                                                )}
                                                {row.ToGrade && (
                                                    <span className={gradeColor(row.ToGrade)}>{row.ToGrade}</span>
                                                )}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}

                {!totalRatings && !ptLow && !ptMean && !ptHigh && upgrades.length === 0 && (
                    <div className="text-slate-400 text-sm p-4">No analyst data available for this ticker.</div>
                )}
            </div>
        );
    };

    return (
        <div className="h-full bg-[#131722] border-l border-[#2a2e39] flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-[#2a2e39] flex-shrink-0">
                <div className="flex items-center gap-1">
                    <button
                        onClick={() => setTab('overview')}
                        className={`px-3 py-1 rounded text-xs font-semibold transition-colors ${tab === 'overview' ? 'bg-[#2962ff] text-white' : 'text-slate-400 hover:text-white hover:bg-[#2a2e39]'}`}
                    >
                        Overview
                    </button>
                    <button
                        onClick={() => setTab('analyst')}
                        className={`px-3 py-1 rounded text-xs font-semibold transition-colors ${tab === 'analyst' ? 'bg-[#2962ff] text-white' : 'text-slate-400 hover:text-white hover:bg-[#2a2e39]'}`}
                    >
                        Analyst
                    </button>
                </div>
                <button
                    onClick={onClose}
                    className="p-1 hover:bg-[#2a2e39] rounded text-slate-400 transition-colors"
                >
                    <X className="w-4 h-4" />
                </button>
            </div>

            {/* Scrollable content */}
            <div className="flex-grow overflow-y-auto">
                {tab === 'overview' ? renderOverview() : renderAnalyst()}
            </div>
        </div>
    );
};

export default StatsPanel;
