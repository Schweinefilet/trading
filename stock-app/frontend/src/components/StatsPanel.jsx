import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { X } from 'lucide-react';

// ── Formatters ────────────────────────────────────────────────────────────────
function fmtNum(v) {
    if (v == null || v === '' || isNaN(v)) return '—';
    const n = Number(v);
    if (Math.abs(n) >= 1e12) return `$${(n / 1e12).toFixed(2)}T`;
    if (Math.abs(n) >= 1e9)  return `$${(n / 1e9).toFixed(1)}B`;
    if (Math.abs(n) >= 1e6)  return `$${(n / 1e6).toFixed(1)}M`;
    if (Math.abs(n) >= 1e3)  return `$${(n / 1e3).toFixed(1)}K`;
    return `$${n.toFixed(2)}`;
}
function fmtPct(v) {
    if (v == null || v === '' || isNaN(v)) return '—';
    return `${(Number(v) * 100).toFixed(2)}%`;
}
function fmtRatio(v, decimals = 2) {
    if (v == null || v === '' || isNaN(v)) return '—';
    return Number(v).toFixed(decimals);
}
function upsideStr(target, current) {
    if (target == null || current == null || current === 0) return null;
    const pct = ((target - current) / current) * 100;
    return { pct, str: `${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%`, positive: pct >= 0 };
}

// ── MetricCard ────────────────────────────────────────────────────────────────
function MetricCard({ label, value }) {
    return (
        <div className="bg-[#1e222d] rounded-lg p-3">
            <div className="text-slate-500 text-xs mb-1">{label}</div>
            <div className="text-white font-bold text-sm">{value}</div>
        </div>
    );
}

// ── RatingsBar ────────────────────────────────────────────────────────────────
function RatingsBar({ label, count, total, color }) {
    const pct = total > 0 ? (count / total) * 100 : 0;
    return (
        <div className="flex items-center gap-2 py-1">
            <div className="text-xs text-slate-400 w-24 flex-shrink-0">{label}</div>
            <div className="flex-grow bg-[#2a2e39] rounded-full h-2.5 overflow-hidden">
                <div className="h-2.5 rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: color }} />
            </div>
            <div className="text-xs text-slate-300 w-6 text-right flex-shrink-0">{count}</div>
        </div>
    );
}

// ── Grade color helper ────────────────────────────────────────────────────────
function gradeColor(grade) {
    if (!grade) return 'text-slate-400';
    const g = grade.toLowerCase();
    if (g.includes('strong buy') || g.includes('strongbuy')) return 'text-emerald-400';
    if (g.includes('buy') || g.includes('outperform') || g.includes('overweight') || g.includes('positive')) return 'text-green-400';
    if (g.includes('strong sell') || g.includes('strongsell') || g.includes('underperform') || g.includes('underweight') || g.includes('negative') || g.includes('sell')) return 'text-rose-400';
    return 'text-slate-400';
}

function actionBadge(action) {
    if (!action) return null;
    const a = action.toLowerCase();
    if (a === 'up' || a === 'upgrade') return 'bg-emerald-500/15 text-emerald-400';
    if (a === 'down' || a === 'downgrade') return 'bg-rose-500/15 text-rose-400';
    return 'bg-slate-700 text-slate-400';
}

// ── KDE Price Target Distribution Chart ──────────────────────────────────────
function PriceTargetKDE({ low, mean, high, current, ticker }) {
    const W = 288, H = 82;
    const PAD = { l: 2, r: 2, t: 12, b: 20 };
    const iW = W - PAD.l - PAD.r;
    const iH = H - PAD.t - PAD.b;

    // Gaussian σ from range – wider range = wider bell
    const sigma = Math.max((high - low) / 3.5, 1);

    // X axis extends 8% beyond low/high to show tails
    const margin = (high - low) * 0.08;
    const xMin = Math.min(low - margin, current != null ? current - margin : low - margin);
    const xMax = Math.max(high + margin, current != null ? current + margin : high + margin);
    const xSpan = xMax - xMin;

    // Gaussian PDF (unnormalized, peaks at 1)
    const gaussian = (x) => Math.exp(-0.5 * ((x - mean) / sigma) ** 2);

    // Generate 120 sample points
    const N = 120;
    const pts = Array.from({ length: N + 1 }, (_, i) => ({
        x: xMin + (i / N) * xSpan,
        y: gaussian(xMin + (i / N) * xSpan),
    }));

    const toX = (x) => PAD.l + ((x - xMin) / xSpan) * iW;
    const toY = (y) => PAD.t + iH - y * iH; // y is 0..1, peaks at 1

    // SVG path strings
    const linePts = pts.map(p => `${toX(p.x).toFixed(1)},${toY(p.y).toFixed(1)}`).join(' L ');
    const linePath = `M ${linePts}`;
    const fillPath = `${linePath} L ${toX(pts[pts.length - 1].x).toFixed(1)},${toY(0).toFixed(1)} L ${toX(pts[0].x).toFixed(1)},${toY(0).toFixed(1)} Z`;

    const gradId = `kde_${ticker}_${String(low).replace('.', '')}_${String(high).replace('.', '')}`;
    const baseY = toY(0);
    const curX  = current != null ? toX(current) : null;
    const meanX = toX(mean);
    const lowX  = toX(low);
    const highX = toX(high);

    // Tick label helper — avoid overlapping
    const labels = [
        current != null && { x: curX, val: current, color: 'rgba(255,255,255,0.75)', anchor: 'middle' },
        { x: lowX,  val: low,  color: '#ef5350', anchor: 'middle' },
        { x: meanX, val: mean, color: '#4da3ff', anchor: 'middle' },
        { x: highX, val: high, color: '#26a69a', anchor: 'middle' },
    ].filter(Boolean).sort((a, b) => a.x - b.x);

    // Deduplicate labels that are too close (< 30px apart)
    const deduped = labels.reduce((acc, lbl) => {
        if (acc.length === 0 || lbl.x - acc[acc.length - 1].x > 28) acc.push(lbl);
        return acc;
    }, []);

    return (
        <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', overflow: 'visible' }}>
            <defs>
                <linearGradient id={gradId} x1="0" x2="1" y1="0" y2="0">
                    <stop offset="0%"   stopColor="#ef5350" stopOpacity="0.18" />
                    <stop offset="35%"  stopColor="#ff9800" stopOpacity="0.28" />
                    <stop offset="65%"  stopColor="#26a69a" stopOpacity="0.28" />
                    <stop offset="100%" stopColor="#26a69a" stopOpacity="0.12" />
                </linearGradient>
                <linearGradient id={`${gradId}_stroke`} x1="0" x2="1" y1="0" y2="0">
                    <stop offset="0%"   stopColor="#ef5350" />
                    <stop offset="40%"  stopColor="#ff9800" />
                    <stop offset="100%" stopColor="#26a69a" />
                </linearGradient>
            </defs>

            {/* Y-axis label */}
            <text x={PAD.l} y={PAD.t - 2} fill="#475569" fontSize="7" textAnchor="start">Frequency</text>

            {/* Fill area */}
            <path d={fillPath} fill={`url(#${gradId})`} />

            {/* Curve line */}
            <path d={linePath} fill="none" stroke={`url(#${gradId}_stroke)`} strokeWidth="1.5" strokeLinejoin="round" />

            {/* Baseline */}
            <line x1={PAD.l} y1={baseY} x2={W - PAD.r} y2={baseY} stroke="#2a2e39" strokeWidth="1" />

            {/* Low dashed marker */}
            <line x1={lowX} y1={PAD.t} x2={lowX} y2={baseY} stroke="#ef5350" strokeWidth="1" strokeDasharray="2,2" opacity="0.7" />

            {/* High dashed marker */}
            <line x1={highX} y1={PAD.t} x2={highX} y2={baseY} stroke="#26a69a" strokeWidth="1" strokeDasharray="2,2" opacity="0.7" />

            {/* Mean dashed marker */}
            <line x1={meanX} y1={PAD.t} x2={meanX} y2={baseY} stroke="#4da3ff" strokeWidth="1.5" strokeDasharray="3,2" opacity="0.9" />

            {/* Current price solid marker */}
            {curX != null && (
                <line x1={curX} y1={PAD.t} x2={curX} y2={baseY} stroke="white" strokeWidth="1.5" strokeDasharray="3,2" opacity="0.75" />
            )}

            {/* X-axis tick labels */}
            {deduped.map((lbl, i) => (
                <text key={i} x={lbl.x} y={baseY + 11} textAnchor={lbl.anchor} fill={lbl.color} fontSize="7.5" fontWeight="500">
                    ${lbl.val.toFixed(0)}
                </text>
            ))}
        </svg>
    );
}

// ── StatsPanel ────────────────────────────────────────────────────────────────
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
    const week52Low  = f.fifty_two_week_low;
    const week52High = f.fifty_two_week_high;

    // ── Overview ──────────────────────────────────────────────────────────────
    const renderOverview = () => {
        if (loadingFund) return (
            <div className="flex items-center justify-center h-40">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500" />
            </div>
        );
        if (!fundamentals) return (
            <div className="text-slate-400 text-sm p-4">Fundamentals data unavailable.</div>
        );

        return (
            <div className="p-4 space-y-4">
                {/* Company header */}
                <div className="bg-[#1e222d] rounded-lg p-4">
                    <div className="text-white font-bold text-base">{f.longName || ticker}</div>
                    {(f.sector || f.industry) && (
                        <div className="flex flex-wrap gap-2 mt-2">
                            {f.sector   && <span className="px-2 py-0.5 bg-[#2a2e39] rounded text-xs text-slate-300">{f.sector}</span>}
                            {f.industry && <span className="px-2 py-0.5 bg-[#2a2e39] rounded text-xs text-slate-400">{f.industry}</span>}
                        </div>
                    )}
                </div>

                {/* Valuation */}
                <section>
                    <SectionHeader>Valuation</SectionHeader>
                    <div className="grid grid-cols-2 gap-2">
                        <MetricCard label="P/E Ratio"    value={fmtRatio(f.pe_ratio)} />
                        <MetricCard label="Forward P/E"  value={fmtRatio(f.forward_pe)} />
                        <MetricCard label="P/B Ratio"    value={fmtRatio(f.pb_ratio)} />
                        <MetricCard label="P/S Ratio"    value={fmtRatio(f.ps_ratio)} />
                        <MetricCard label="EV/EBITDA"    value={fmtRatio(f.ev_ebitda)} />
                        <MetricCard label="PEG Ratio"    value={fmtRatio(f.peg_ratio)} />
                    </div>
                </section>

                {/* Per Share */}
                <section>
                    <SectionHeader>Per Share</SectionHeader>
                    <div className="grid grid-cols-2 gap-2">
                        <MetricCard label="EPS (TTM)"     value={fmtRatio(f.eps_ttm)} />
                        <MetricCard label="EPS (Forward)" value={fmtRatio(f.eps_forward)} />
                    </div>
                </section>

                {/* Financials */}
                <section>
                    <SectionHeader>Financials</SectionHeader>
                    <div className="grid grid-cols-2 gap-2">
                        <MetricCard label="Revenue (TTM)"   value={fmtNum(f.revenue_ttm)} />
                        <MetricCard label="Market Cap"      value={fmtNum(f.market_cap)} />
                        <MetricCard label="Net Income"      value={fmtNum(f.net_income_ttm)} />
                        <MetricCard label="Free Cash Flow"  value={fmtNum(f.free_cash_flow)} />
                        <MetricCard label="Profit Margin"   value={fmtPct(f.profit_margin)} />
                        <MetricCard label="ROE"             value={fmtPct(f.roe)} />
                        <MetricCard label="ROA"             value={fmtPct(f.roa)} />
                        <MetricCard label="Debt / Equity"   value={fmtRatio(f.debt_to_equity)} />
                    </div>
                </section>

                {/* Risk / Other */}
                <section>
                    <SectionHeader>Risk & Other</SectionHeader>
                    <div className="grid grid-cols-2 gap-2">
                        <MetricCard label="Beta"           value={fmtRatio(f.beta)} />
                        <MetricCard label="Dividend Yield" value={f.dividend_yield != null ? `${Number(f.dividend_yield).toFixed(2)}%` : '—'} />
                        <MetricCard label="Current Ratio"  value={fmtRatio(f.current_ratio)} />
                        <MetricCard label="Quick Ratio"    value={fmtRatio(f.quick_ratio)} />
                    </div>

                    {/* 52-Week Range */}
                    {week52Low != null && week52High != null && (
                        <div className="bg-[#1e222d] rounded-lg p-3 mt-2">
                            <div className="text-slate-500 text-xs mb-2">52-Week Range</div>
                            <div className="flex items-center justify-between text-xs text-slate-400 mb-1.5">
                                <span>${Number(week52Low).toFixed(2)}</span>
                                <span>${Number(week52High).toFixed(2)}</span>
                            </div>
                            <div className="relative h-2 bg-[#2a2e39] rounded-full">
                                <div className="absolute inset-0 rounded-full opacity-60" style={{ background: 'linear-gradient(to right, #ef5350, #ff9800, #26a69a)' }} />
                            </div>
                        </div>
                    )}
                </section>
            </div>
        );
    };

    // ── Analyst ───────────────────────────────────────────────────────────────
    const renderAnalyst = () => {
        if (loadingAnalyst) return (
            <div className="flex items-center justify-center h-40">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500" />
            </div>
        );
        if (!analyst) return (
            <div className="text-slate-400 text-sm p-4">Analyst data unavailable.</div>
        );

        const ratings = analyst.ratings || [];
        const cur = ratings[0] || {};
        const total = (cur.strongBuy || 0) + (cur.buy || 0) + (cur.hold || 0) + (cur.sell || 0) + (cur.strongSell || 0);

        const pt       = analyst.price_targets || {};
        const upgrades = analyst.upgrades_downgrades || [];

        const ptLow     = pt.low     != null ? Number(pt.low)     : null;
        const ptMean    = pt.mean    != null ? Number(pt.mean)    : null;
        const ptHigh    = pt.high    != null ? Number(pt.high)    : null;
        const ptCurrent = pt.current != null ? Number(pt.current) : null;

        const lowUpside  = upsideStr(ptLow, ptCurrent);
        const meanUpside = upsideStr(ptMean, ptCurrent);
        const highUpside = upsideStr(ptHigh, ptCurrent);

        const noData = !total && ptLow == null && ptMean == null && ptHigh == null && upgrades.length === 0;
        if (noData) return (
            <div className="text-slate-400 text-sm p-4">No analyst data available for this ticker.</div>
        );

        return (
            <div className="p-4 space-y-5">

                {/* ── Ratings Distribution ── */}
                {total > 0 && (
                    <section>
                        <SectionHeader suffix={cur.period ? `(${cur.period})` : null}>
                            Ratings Distribution
                        </SectionHeader>
                        <div className="bg-[#1e222d] rounded-lg p-3 space-y-0.5">
                            <RatingsBar label="Strong Buy"  count={cur.strongBuy  || 0} total={total} color="#26a69a" />
                            <RatingsBar label="Buy"         count={cur.buy        || 0} total={total} color="#4caf50" />
                            <RatingsBar label="Hold"        count={cur.hold       || 0} total={total} color="#ff9800" />
                            <RatingsBar label="Sell"        count={cur.sell       || 0} total={total} color="#ef5350" />
                            <RatingsBar label="Strong Sell" count={cur.strongSell || 0} total={total} color="#b71c1c" />
                        </div>
                    </section>
                )}

                {/* ── Price Targets ── */}
                {(ptLow != null || ptMean != null || ptHigh != null) && (
                    <section>
                        <SectionHeader suffix={pt.numberOfAnalysts != null ? `${pt.numberOfAnalysts} analysts` : null}>
                            Price Targets
                        </SectionHeader>

                        <div className="bg-[#1e222d] rounded-xl p-3 space-y-3">
                            {/* Three cards */}
                            <div className="grid grid-cols-3 gap-2">
                                {/* Low */}
                                <div className="bg-[#1a0a0a] border border-red-900/60 rounded-lg p-2.5 text-center">
                                    <div className="text-[11px] text-red-400 font-semibold mb-1">Low</div>
                                    <div className="text-white font-bold text-sm">{ptLow != null ? `$${ptLow.toFixed(2)}` : '—'}</div>
                                    {lowUpside && (
                                        <div className={`text-[10px] mt-0.5 font-medium ${lowUpside.positive ? 'text-green-400' : 'text-red-400'}`}>
                                            {lowUpside.str}
                                        </div>
                                    )}
                                </div>

                                {/* Mean */}
                                <div className="bg-[#0a1628] border border-blue-700/60 rounded-lg p-2.5 text-center">
                                    <div className="text-[11px] text-blue-400 font-semibold mb-1">Mean</div>
                                    <div className="text-white font-bold text-sm">{ptMean != null ? `$${ptMean.toFixed(2)}` : '—'}</div>
                                    {meanUpside && (
                                        <div className={`text-[10px] mt-0.5 font-medium ${meanUpside.positive ? 'text-green-400' : 'text-red-400'}`}>
                                            {meanUpside.str}
                                        </div>
                                    )}
                                </div>

                                {/* High */}
                                <div className="bg-[#0a1a0a] border border-green-900/60 rounded-lg p-2.5 text-center">
                                    <div className="text-[11px] text-green-400 font-semibold mb-1">High</div>
                                    <div className="text-white font-bold text-sm">{ptHigh != null ? `$${ptHigh.toFixed(2)}` : '—'}</div>
                                    {highUpside && (
                                        <div className={`text-[10px] mt-0.5 font-medium ${highUpside.positive ? 'text-green-400' : 'text-red-400'}`}>
                                            {highUpside.str}
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* KDE distribution chart */}
                            {ptLow != null && ptMean != null && ptHigh != null && ptHigh > ptLow && (
                                <div>
                                    <div className="text-slate-600 text-[10px] mb-1 flex items-center gap-3">
                                        <span className="flex items-center gap-1">
                                            <span className="inline-block w-3 border-t border-dashed border-white/50" />
                                            Current
                                        </span>
                                        <span className="flex items-center gap-1">
                                            <span className="inline-block w-3 border-t border-dashed border-blue-400/80" />
                                            Mean target
                                        </span>
                                    </div>
                                    <div className="bg-[#131722] rounded-lg p-2 pt-1">
                                        <PriceTargetKDE
                                            low={ptLow}
                                            mean={ptMean}
                                            high={ptHigh}
                                            current={ptCurrent}
                                            ticker={ticker}
                                        />
                                    </div>
                                </div>
                            )}
                        </div>
                    </section>
                )}

                {/* ── Recent Analyst Actions ── */}
                {upgrades.length > 0 && (
                    <section>
                        <SectionHeader suffix={`${upgrades.length} actions`}>Recent Analyst Actions</SectionHeader>
                        <div className="bg-[#1e222d] rounded-lg overflow-hidden">
                            <table className="w-full text-xs">
                                <thead>
                                    <tr className="border-b border-[#2a2e39]">
                                        <th className="text-left px-3 py-2 text-slate-500 font-medium">Date</th>
                                        <th className="text-left px-3 py-2 text-slate-500 font-medium">Firm</th>
                                        <th className="text-left px-3 py-2 text-slate-500 font-medium">Action</th>
                                        <th className="text-left px-3 py-2 text-slate-500 font-medium">Grade</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {upgrades.slice(0, 50).map((row, i) => (
                                        <tr key={i} className="border-b border-[#2a2e39]/60 last:border-0 hover:bg-[#2a2e39] transition-colors">
                                            <td className="px-3 py-1.5 text-slate-400 whitespace-nowrap">
                                                {row.GradeDate ? String(row.GradeDate).slice(0, 10) : '—'}
                                            </td>
                                            <td className="px-3 py-1.5 text-slate-300 max-w-[90px] truncate">{row.Firm || '—'}</td>
                                            <td className="px-3 py-1.5">
                                                {row.Action && (
                                                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold capitalize ${actionBadge(row.Action)}`}>
                                                        {row.Action}
                                                    </span>
                                                )}
                                            </td>
                                            <td className="px-3 py-1.5">
                                                <div className="flex items-center gap-1 flex-wrap">
                                                    {row.FromGrade && (
                                                        <span className={`${gradeColor(row.FromGrade)} opacity-50 truncate max-w-[60px]`}>{row.FromGrade}</span>
                                                    )}
                                                    {row.FromGrade && row.ToGrade && (
                                                        <span className="text-slate-600">→</span>
                                                    )}
                                                    {row.ToGrade && (
                                                        <span className={`${gradeColor(row.ToGrade)} truncate max-w-[60px]`}>{row.ToGrade}</span>
                                                    )}
                                                </div>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </section>
                )}
            </div>
        );
    };

    return (
        <div className="h-full bg-[#131722] border-l border-[#2a2e39] flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-[#2a2e39] flex-shrink-0">
                <div className="flex items-center gap-1">
                    {[
                        { id: 'overview', label: 'Overview' },
                        { id: 'analyst',  label: 'Analyst' },
                    ].map(t => (
                        <button
                            key={t.id}
                            onClick={() => setTab(t.id)}
                            className={`px-3 py-1 rounded text-xs font-semibold transition-colors ${tab === t.id ? 'bg-[#2962ff] text-white' : 'text-slate-400 hover:text-white hover:bg-[#2a2e39]'}`}
                        >
                            {t.label}
                        </button>
                    ))}
                </div>
                <button onClick={onClose} className="p-1 hover:bg-[#2a2e39] rounded text-slate-400 transition-colors">
                    <X className="w-4 h-4" />
                </button>
            </div>

            {/* Content */}
            <div className="flex-grow overflow-y-auto">
                {tab === 'overview' ? renderOverview() : renderAnalyst()}
            </div>
        </div>
    );
};

// ── Section header helper ─────────────────────────────────────────────────────
function SectionHeader({ children, suffix }) {
    return (
        <div className="flex items-center justify-between mb-2 px-1">
            <div className="text-slate-500 text-xs font-semibold uppercase tracking-wider">{children}</div>
            {suffix && <div className="text-slate-600 text-[10px] normal-case">{suffix}</div>}
        </div>
    );
}

export default StatsPanel;
