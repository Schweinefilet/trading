import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import {
    Search, TrendingUp, BarChart2, Activity, Clock, Layers,
    Settings, Camera, Maximize2, Star, Eye, Edit, Trash2, X, Plus,
    ChevronDown, BarChart, LineChart, AreaChart, Box, History, AlertTriangle
} from 'lucide-react';
import StockChart from '../components/StockChart';
import IndicatorModal from '../components/IndicatorModal';
import StatsPanel from '../components/StatsPanel';
import { computeHeikinAshi } from '../utils/heikinAshi';

const TIMEFRAMES = [
    { label: '1m', value: '1m' },
    { label: '5m', value: '5m' },
    { label: '15m', value: '15m' },
    { label: '30m', value: '30m' },
    { label: '1H', value: '1h' },
    { label: '1D', value: '1d' },
    { label: '1W', value: '1wk' },
    { label: '1M', value: '1mo' },
];

const PERIODS = [
    { label: '1M', value: '1mo' },
    { label: '3M', value: '3mo' },
    { label: '6M', value: '6mo' },
    { label: '1Y', value: '1y' },
    { label: '2Y', value: '2y' },
    { label: '5Y', value: '5y' },
];

// Max periods yfinance supports per interval
const INTERVAL_MAX_PERIOD = {
    '1m':  '5d',
    '5m':  '1mo',
    '15m': '1mo',
    '30m': '1mo',
    '1h':  '3mo',
};

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }
    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }
    componentDidCatch(error, errorInfo) {
        console.error("[ErrorBoundary] Caught error:", error, errorInfo);
    }
    render() {
        if (this.state.hasError) {
            return (
                <div className="flex flex-col items-center justify-center h-full bg-[#131722] text-red-400 p-8 text-center gap-4">
                    <AlertTriangle className="w-12 h-12" />
                    <h2 className="text-xl font-bold">Chart Component Crashed</h2>
                    <p className="text-sm text-slate-400 max-w-md">{this.state.error?.message || "An unexpected error occurred while rendering the chart."}</p>
                    <button
                        onClick={() => window.location.reload()}
                        className="px-4 py-2 bg-[#2962ff] text-white rounded hover:bg-[#1e4bd8] transition-colors"
                    >
                        Reload Page
                    </button>
                </div>
            );
        }
        return this.props.children;
    }
}

const StockDetail = () => {
    const { ticker } = useParams();
    const navigate = useNavigate();
    const [searchInput, setSearchInput] = useState('');
    const [data, setData] = useState([]);
    const [indicatorData, setIndicatorData] = useState([]);
    const [quote, setQuote] = useState(null);
    const [period, setPeriod] = useState('5y');
    const [interval, setInterval] = useState('1d');

    const handleIntervalChange = (newInterval) => {
        setInterval(newInterval);
        if (INTERVAL_MAX_PERIOD[newInterval]) {
            // Intraday: clamp to its max period
            setPeriod(INTERVAL_MAX_PERIOD[newInterval]);
        } else {
            // Switching back to daily/weekly/monthly: restore default
            setPeriod('5y');
        }
    };
    const [chartType, setChartType] = useState('candlestick');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [showIndicatorModal, setShowIndicatorModal] = useState(false);
    const [showChartTypeDropdown, setShowChartTypeDropdown] = useState(false);
    const chartTypeDropdownRef = useRef(null);
    const [activeIndicators, setActiveIndicators] = useState(['rsi', 'macd', 'sma_20', 'sma_50']);
    const [hoveredBar, setHoveredBar] = useState(null);
    const [showStatsPanel, setShowStatsPanel] = useState(false);

    // Reset hovered bar on ticker change
    useEffect(() => { setHoveredBar(null); }, [ticker]);

    // Close chart-type dropdown on outside click
    useEffect(() => {
        if (!showChartTypeDropdown) return;
        const handler = (e) => {
            if (chartTypeDropdownRef.current && !chartTypeDropdownRef.current.contains(e.target)) {
                setShowChartTypeDropdown(false);
            }
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, [showChartTypeDropdown]);

    // Remove an oscillator pane by its pane id
    const handleRemovePane = useCallback((paneId) => {
        setActiveIndicators(prev => prev.filter(id => id !== paneId && id.split('_')[0] !== paneId));
    }, []);

    // Remove a main-pane overlay (SMA, EMA, VWAP, bbands…) by its indicator id
    const handleRemoveOverlay = useCallback((indicatorId) => {
        setActiveIndicators(prev => prev.filter(id => id !== indicatorId));
    }, []);

    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            try {
                const indicatorsParam = activeIndicators.join(',');
                const [historyRes, indicatorRes, quoteRes] = await Promise.all([
                    axios.get(`http://localhost:5001/api/stock/${ticker}/history?period=${period}&interval=${interval}`),
                    axios.get(`http://localhost:5001/api/stock/${ticker}/indicators?period=${period}&interval=${interval}&indicators=${indicatorsParam}`),
                    axios.get(`http://localhost:5001/api/stock/${ticker}/quote`)
                ]);

                const hData = Array.isArray(historyRes.data) ? historyRes.data : [];
                const iData = Array.isArray(indicatorRes.data) ? indicatorRes.data : [];

                console.log(`[StockDetail] Fetched ${hData.length} records for ${ticker}`);

                setData(hData);
                setIndicatorData(iData);
                setQuote(quoteRes.data);
                setError(null);
            } catch (err) {
                console.error("Fetch error:", err);
                setError("Market data unavailable for this ticker or interval.");
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [ticker, period, interval, activeIndicators]);

    return (
        <div className="flex flex-col h-screen bg-[#131722] text-[#d1d4dc] overflow-hidden">
            <header className="h-12 border-b border-[#2a2e39] flex items-center px-4 gap-4 flex-shrink-0">
                {/* Logo — clickable home navigation */}
                <button
                    onClick={() => navigate('/')}
                    className="flex items-center gap-2 pr-4 border-r border-[#2a2e39] hover:opacity-80 transition-opacity"
                >
                    <TrendingUp className="text-blue-500 w-5 h-5" />
                    <span className="font-bold text-white tracking-widest text-sm">TV_CLONE</span>
                </button>

                <form onSubmit={(e) => { e.preventDefault(); const t = searchInput.trim().toUpperCase(); if (t) { navigate(`/stock/${t}`); setSearchInput(''); } }} className="relative min-w-[160px] flex-grow max-w-xs">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                    <input
                        type="text"
                        placeholder={ticker}
                        className="w-full bg-[#1e222d] border border-transparent focus:border-[#2962ff] rounded pl-9 pr-3 py-1 text-sm outline-none uppercase"
                        style={{ color: '#d1d4dc' }}
                        value={searchInput}
                        onChange={(e) => setSearchInput(e.target.value)}
                    />
                </form>

                <div className="flex bg-[#1e222d] p-0.5 rounded border border-[#2a2e39] ml-2">
                    {TIMEFRAMES.map(tf => (
                        <button
                            key={tf.label}
                            onClick={() => handleIntervalChange(tf.value)}
                            className={`px-3 py-1 text-[11px] font-bold rounded transition-colors ${interval === tf.value ? 'bg-[#2962ff] text-white' : 'hover:bg-[#2a2e39] text-[#d1d4dc]'}`}
                        >
                            {tf.label}
                        </button>
                    ))}
                </div>

                {!INTERVAL_MAX_PERIOD[interval] && (
                    <div className="flex bg-[#1e222d] p-0.5 rounded border border-[#2a2e39]">
                        {PERIODS.map(p => (
                            <button
                                key={p.label}
                                onClick={() => setPeriod(p.value)}
                                className={`px-2.5 py-1 text-[11px] font-bold rounded transition-colors ${period === p.value ? 'bg-[#2962ff] text-white' : 'hover:bg-[#2a2e39] text-[#d1d4dc]'}`}
                            >
                                {p.label}
                            </button>
                        ))}
                    </div>
                )}

                <div className="relative ml-2" ref={chartTypeDropdownRef}>
                    <button
                        onClick={() => setShowChartTypeDropdown(!showChartTypeDropdown)}
                        className="flex items-center gap-1 px-3 py-1.5 bg-[#1e222d] hover:bg-[#2a2e39] border border-[#2a2e39] rounded text-[11px] font-bold transition-colors"
                    >
                        {chartType === 'candlestick' && <BarChart2 className="w-4 h-4 text-blue-500" />}
                        {chartType === 'heikin_ashi' && <Activity className="w-4 h-4 text-blue-500" />}
                        {chartType === 'line' && <LineChart className="w-4 h-4 text-blue-500" />}
                        {chartType === 'area' && <AreaChart className="w-4 h-4 text-blue-500" />}
                        {chartType === 'bar' && <BarChart className="w-4 h-4 text-blue-500" />}
                        <ChevronDown className="w-3 h-3 text-slate-500" />
                    </button>

                    {showChartTypeDropdown && (
                        <div className="absolute top-full left-0 mt-1 w-40 bg-[#1e222d] border border-[#2a2e39] rounded shadow-2xl z-50 py-1">
                            {[
                                { id: 'candlestick', label: 'Candles', icon: BarChart2 },
                                { id: 'heikin_ashi', label: 'Heikin Ashi', icon: Activity },
                                { id: 'bar', label: 'Bars', icon: BarChart },
                                { id: 'line', label: 'Line', icon: LineChart },
                                { id: 'area', label: 'Area', icon: AreaChart },
                            ].map(type => (
                                <button
                                    key={type.id}
                                    onClick={() => { setChartType(type.id); setShowChartTypeDropdown(false); }}
                                    className="w-full flex items-center gap-3 px-3 py-2 text-xs hover:bg-[#2a2e39] text-[#d1d4dc] transition-colors"
                                >
                                    <type.icon className={`w-4 h-4 ${chartType === type.id ? 'text-blue-500' : 'text-slate-400'}`} />
                                    {type.label}
                                </button>
                            ))}
                        </div>
                    )}
                </div>

                <button
                    onClick={() => setShowIndicatorModal(true)}
                    className="ml-2 px-3 py-1.5 bg-[#1e222d] hover:bg-[#2a2e39] border border-[#2a2e39] rounded text-[11px] font-bold text-[#d1d4dc] transition-colors"
                >
                    Indicators
                </button>

                {/* Stats toggle button */}
                <button
                    onClick={() => setShowStatsPanel(prev => !prev)}
                    className={`ml-2 flex items-center gap-1.5 px-3 py-1.5 border rounded text-[11px] font-bold transition-colors ${
                        showStatsPanel
                            ? 'bg-[#2962ff] border-[#2962ff] text-white'
                            : 'bg-[#1e222d] hover:bg-[#2a2e39] border-[#2a2e39] text-[#d1d4dc]'
                    }`}
                >
                    <BarChart2 className="w-3.5 h-3.5" />
                    Stats
                </button>

                <div className="flex items-center gap-2 ml-auto">
                    <button className="p-2 hover:bg-[#2a2e39] rounded text-slate-400"><Maximize2 className="w-5 h-5" /></button>
                    <button className="p-2 hover:bg-[#2a2e39] rounded text-slate-400"><Settings className="w-5 h-5" /></button>
                    <button className="p-2 hover:bg-[#2a2e39] rounded text-slate-400"><Camera className="w-5 h-5" /></button>
                </div>
            </header>

            <div className="flex flex-grow overflow-hidden">
                <aside className="w-12 border-r border-[#2a2e39] flex flex-col items-center py-4 gap-4 bg-[#131722] flex-shrink-0">
                    <button className="p-2 hover:bg-[#2a2e39] rounded text-blue-500"><Maximize2 className="w-5 h-5" /></button>
                    <div className="w-6 h-px bg-[#2a2e39]" />
                    <button className="p-2 hover:bg-[#2a2e39] rounded text-slate-400"><Edit className="w-5 h-5" /></button>
                    <button className="p-2 hover:bg-[#2a2e39] rounded text-slate-400"><Plus className="w-5 h-5" /></button>
                </aside>

                <main className="flex-grow relative bg-[#131722] overflow-hidden" style={{ minHeight: '400px' }}>
                    {loading ? (
                        <div className="absolute inset-0 flex items-center justify-center bg-[#131722]/50 z-50">
                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                        </div>
                    ) : error ? (
                        <div className="absolute inset-0 flex items-center justify-center flex-col gap-4 text-center px-4">
                            <History className="w-12 h-12 text-slate-600" />
                            <p className="text-slate-400 max-w-xs">{error}</p>
                        </div>
                    ) : (
                        <ErrorBoundary>
                            <StockChart
                                data={chartType === 'heikin_ashi' ? computeHeikinAshi(data) : data}
                                indicatorData={indicatorData}
                                activeIndicators={activeIndicators}
                                chartType={chartType === 'heikin_ashi' ? 'candlestick' : chartType}
                                onCrosshairMove={setHoveredBar}
                                onRemovePane={handleRemovePane}
                                onRemoveOverlay={handleRemoveOverlay}
                            />
                        </ErrorBoundary>
                    )}
                </main>

                {showStatsPanel && (
                    <div className="w-[360px] flex-shrink-0 border-l border-[#2a2e39] overflow-y-auto">
                        <StatsPanel ticker={ticker} onClose={() => setShowStatsPanel(false)} />
                    </div>
                )}

                {showIndicatorModal && (
                    <IndicatorModal
                        onClose={() => setShowIndicatorModal(false)}
                        onSelect={(result) => {
                            const toAdd = (Array.isArray(result) ? result : [result]).filter(id => !activeIndicators.includes(id));
                            if (toAdd.length > 0) setActiveIndicators(prev => [...prev, ...toAdd]);
                            setShowIndicatorModal(false);
                        }}
                    />
                )}
            </div>

            <footer className="h-8 border-t border-[#2a2e39] bg-[#131722] flex items-center px-4 gap-6 text-[10px] font-medium text-slate-400 flex-shrink-0">
                <div className="flex gap-4">
                    {(() => {
                        const bar = hoveredBar || data[data.length - 1];
                        const o = (hoveredBar?.open ?? bar?.Open)?.toFixed(2) || '0.00';
                        const h = (hoveredBar?.high ?? bar?.High)?.toFixed(2) || '0.00';
                        const l = (hoveredBar?.low  ?? bar?.Low)?.toFixed(2)  || '0.00';
                        const c = (hoveredBar?.close ?? bar?.Close)?.toFixed(2) || '0.00';
                        return (<>
                            <span>O <span className="text-white">{o}</span></span>
                            <span>H <span className="text-white">{h}</span></span>
                            <span>L <span className="text-white">{l}</span></span>
                            <span>C <span className="text-white">{c}</span></span>
                        </>);
                    })()}
                </div>
            </footer>
        </div>
    );
};

export default StockDetail;
