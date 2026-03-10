import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import html2canvas from 'html2canvas';
import {
    Search, TrendingUp, BarChart2, Activity, Clock, Layers,
    Settings, Camera, Maximize2, Star, Eye, Trash2, X,
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
                <div className="flex flex-col items-center justify-center h-full bg-black text-red-400 p-8 text-center gap-4">
                    <AlertTriangle className="w-12 h-12" />
                    <h2 className="text-xl font-bold">Chart Component Crashed</h2>
                    <p className="text-sm text-slate-400 max-w-md">{this.state.error?.message || "An unexpected error occurred while rendering the chart."}</p>
                    <button
                        onClick={() => window.location.reload()}
                        className="px-4 py-2 bg-white text-black rounded hover:bg-white/90 transition-colors"
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
    const [period, setPeriod] = useState('1mo');
    const [interval, setInterval] = useState('5m');
    const [position, setPosition] = useState(null);
    const [tradeMarkers, setTradeMarkers] = useState([]);

    const handleIntervalChange = (newInterval) => {
        setInterval(newInterval);
        if (INTERVAL_MAX_PERIOD[newInterval]) {
            // Intraday: clamp to its max period
            setPeriod(INTERVAL_MAX_PERIOD[newInterval]);
        } else {
            // Switching back to daily/weekly/monthly: restore default
            setPeriod('1y');
        }
    };
    const [chartType, setChartType] = useState('hollow_candle');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [showIndicatorModal, setShowIndicatorModal] = useState(false);
    const [showChartTypeDropdown, setShowChartTypeDropdown] = useState(false);
    const chartTypeDropdownRef = useRef(null);
    const [activeIndicators, setActiveIndicators] = useState(['rsi', 'sma_20', 'sma_50']);
    const [hoveredBar, setHoveredBar] = useState(null);
    const [showStatsPanel, setShowStatsPanel] = useState(false);
    const [showSettingsDropdown, setShowSettingsDropdown] = useState(false);
    const [showFooterBar, setShowFooterBar] = useState(true);
    const [captureBusy, setCaptureBusy] = useState(false);
    const settingsDropdownRef = useRef(null);
    const captureFrameRef = useRef(null);
    const requestSeqRef = useRef(0);

    const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

    const getWithRetry = useCallback(async (url, config = {}, maxAttempts = 3) => {
        let attempt = 0;
        while (attempt < maxAttempts) {
            try {
                return await axios.get(url, config);
            } catch (err) {
                if (err?.code === 'ERR_CANCELED') throw err;
                const status = err?.response?.status;
                const retryAfter = Number(err?.response?.headers?.['retry-after']);
                const isRateLimited = status === 429;
                attempt += 1;

                if (!isRateLimited || attempt >= maxAttempts) {
                    throw err;
                }

                const backoffMs = Number.isFinite(retryAfter) && retryAfter > 0
                    ? retryAfter * 1000
                    : 700 * (2 ** (attempt - 1));
                await sleep(backoffMs);
            }
        }
    }, []);

    // Reset hovered bar on ticker change
    useEffect(() => { setHoveredBar(null); }, [ticker]);

    // Fetch portfolio position for this ticker and calculate P&L from current price
    useEffect(() => {
        const fetchPosition = async () => {
            try {
                const portfolioRes = await axios.get('http://localhost:5001/api/portfolio');
                const positions = Array.isArray(portfolioRes.data) ? portfolioRes.data : [];
                const pos = positions.find(p => p.ticker === ticker.toUpperCase());
                
                if (pos && data.length > 0) {
                    // Get current price from latest data point
                    const currentPrice = data[data.length - 1]?.close || data[data.length - 1]?.Close || 0;
                    const pnl = (currentPrice - pos.avg_cost) * pos.shares;
                    setPosition({
                        shares: pos.shares,
                        entryPrice: pos.avg_cost,
                        currentPrice: currentPrice,
                        pnl: pnl
                    });
                } else {
                    setPosition(null);
                }
            } catch (err) {
                console.error('Failed to fetch position:', err);
                setPosition(null);
            }
        };
        
        if (ticker && data.length > 0) {
            fetchPosition();
        }
    }, [ticker, data]);

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

    // Close settings dropdown on outside click
    useEffect(() => {
        if (!showSettingsDropdown) return;
        const handler = (e) => {
            if (settingsDropdownRef.current && !settingsDropdownRef.current.contains(e.target)) {
                setShowSettingsDropdown(false);
            }
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, [showSettingsDropdown]);

    // Remove an oscillator pane by its pane id
    const handleRemovePane = useCallback((paneId) => {
        setActiveIndicators(prev => prev.filter(id => id !== paneId && id.split('_')[0] !== paneId));
    }, []);

    // Remove a main-pane overlay (SMA, EMA, VWAP, bbands…) by its indicator id
    const handleRemoveOverlay = useCallback((indicatorId) => {
        setActiveIndicators(prev => prev.filter(id => id !== indicatorId));
    }, []);

    const handleCaptureChart = useCallback(async () => {
        if (!captureFrameRef.current || captureBusy) return;
        setShowSettingsDropdown(false);
        setCaptureBusy(true);
        try {
            const canvas = await html2canvas(captureFrameRef.current, {
                backgroundColor: '#000000',
                scale: Math.min(window.devicePixelRatio || 1, 2),
                useCORS: true,
                logging: false,
            });

            const stamp = new Date().toISOString().replace(/[:.]/g, '-');
            const link = document.createElement('a');
            link.href = canvas.toDataURL('image/png');
            link.download = `${ticker}_chart_${stamp}.png`;
            link.click();
        } catch (err) {
            console.error('Failed to capture chart image:', err);
        } finally {
            setCaptureBusy(false);
        }
    }, [ticker, captureBusy]);

    useEffect(() => {
        const requestId = requestSeqRef.current + 1;
        requestSeqRef.current = requestId;
        const controller = new AbortController();

        const fetchData = async () => {
            setLoading(true);
            try {
                const indicatorsParam = activeIndicators.join(',');
                const [historyRes, indicatorRes, quoteRes, markersRes] = await Promise.all([
                    getWithRetry(`http://localhost:5001/api/stock/${ticker}/history?period=${period}&interval=${interval}`, { signal: controller.signal }),
                    getWithRetry(`http://localhost:5001/api/stock/${ticker}/indicators?period=${period}&interval=${interval}&indicators=${indicatorsParam}`, { signal: controller.signal }),
                    getWithRetry(`http://localhost:5001/api/stock/${ticker}/quote`, { signal: controller.signal }),
                    getWithRetry(`http://localhost:5001/api/stock/${ticker}/trade-markers?period=${period}&interval=${interval}`, { signal: controller.signal })
                        .catch(() => ({ data: [] })),
                ]);

                if (requestSeqRef.current !== requestId) {
                    return;
                }

                const hData = Array.isArray(historyRes.data) ? historyRes.data : [];
                const iData = Array.isArray(indicatorRes.data) ? indicatorRes.data : [];

                console.log(`[StockDetail] Fetched ${hData.length} records for ${ticker}`);

                setData(hData);
                setIndicatorData(iData);
                setQuote(quoteRes.data);
                setTradeMarkers(Array.isArray(markersRes.data) ? markersRes.data : []);
                setError(null);
            } catch (err) {
                if (err?.code === 'ERR_CANCELED') {
                    return;
                }
                console.error("Fetch error:", err);
                setTradeMarkers([]);
                const status = err?.response?.status;
                if (status === 429) {
                    setError("Rate limited while loading market data. Retrying shortly usually resolves it.");
                } else {
                    setError("Market data unavailable for this ticker or interval.");
                }
            } finally {
                if (requestSeqRef.current === requestId) {
                    setLoading(false);
                }
            }
        };
        fetchData();

        return () => {
            controller.abort();
        };
    }, [ticker, period, interval, activeIndicators, getWithRetry]);

    return (
        <div className="flex flex-col h-screen bg-black text-[#d1d4dc] overflow-hidden">
            <header className="h-12 border-b border-white/10 flex items-center px-4 gap-4 flex-shrink-0">
                {/* Logo — clickable home navigation */}
                <button
                    onClick={() => navigate('/')}
                    className="flex items-center gap-2 pr-4 border-r border-white/10 hover:opacity-80 transition-opacity"
                >
                    <TrendingUp className="text-white w-5 h-5" />
                    <span className="font-bold text-white tracking-widest text-sm">Tradr</span>
                </button>

                <form onSubmit={(e) => { e.preventDefault(); const t = searchInput.trim().toUpperCase(); if (t) { navigate(`/stock/${t}`); setSearchInput(''); } }} className="relative min-w-[160px] flex-grow max-w-xs">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                    <input
                        type="text"
                        placeholder={ticker}
                        className="w-full border border-white/10 focus:border-white/40 rounded pl-9 pr-3 py-1 text-sm outline-none uppercase transition-colors"
                        style={{ color: '#d1d4dc', backgroundColor: 'rgba(255,255,255,0.07)' }}
                        value={searchInput}
                        onChange={(e) => setSearchInput(e.target.value)}
                    />
                </form>

                <div className="hidden lg:flex flex-col min-w-0 max-w-sm">
                    <span className="text-xs truncate" style={{ color: 'var(--text-secondary)' }}>
                        {quote?.longName || ticker}
                    </span>
                    {quote?.sector && (
                        <span className="text-[10px] truncate" style={{ color: 'var(--text-tertiary)' }}>
                            {quote.sector}
                        </span>
                    )}
                </div>

                {position && (
                    <div className="flex items-center gap-1 px-3 py-1.5 bg-white/7 border border-white/10 rounded text-xs font-semibold text-white ml-2">
                        <span>{position.shares}@{position.entryPrice}</span>
                        <span className={position.pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                            ({position.pnl >= 0 ? '+' : ''}{position.pnl.toFixed(2)})
                        </span>
                    </div>
                )}

                <div className="flex bg-white/7 p-0.5 rounded border border-white/10 ml-2">
                    {TIMEFRAMES.map(tf => (
                        <button
                            key={tf.label}
                            onClick={() => handleIntervalChange(tf.value)}
                            className={`px-3 py-1 text-[11px] font-bold rounded transition-colors ${interval === tf.value ? 'bg-white text-black' : 'hover:bg-white/10 text-[#d1d4dc]'}`}
                        >
                            {tf.label}
                        </button>
                    ))}
                </div>

                {!INTERVAL_MAX_PERIOD[interval] && (
                    <div className="flex bg-white/7 p-0.5 rounded border border-white/10">
                        {PERIODS.map(p => (
                            <button
                                key={p.label}
                                onClick={() => setPeriod(p.value)}
                                className={`px-2.5 py-1 text-[11px] font-bold rounded transition-colors ${period === p.value ? 'bg-white text-black' : 'hover:bg-white/10 text-[#d1d4dc]'}`}
                            >
                                {p.label}
                            </button>
                        ))}
                    </div>
                )}

                <div className="relative ml-2" ref={chartTypeDropdownRef}>
                    <button
                        onClick={() => setShowChartTypeDropdown(!showChartTypeDropdown)}
                        className="flex items-center gap-1 px-3 py-1.5 bg-white/7 hover:bg-white/10 border border-white/10 rounded text-[11px] font-bold transition-colors"
                    >
                        {chartType === 'candlestick'   && <BarChart2  className="w-4 h-4 text-white" />}
                        {chartType === 'hollow_candle' && <Box       className="w-4 h-4 text-white" />}
                        {chartType === 'heikin_ashi'   && <Activity  className="w-4 h-4 text-white" />}
                        {chartType === 'line'          && <LineChart  className="w-4 h-4 text-white" />}
                        {chartType === 'area'          && <AreaChart  className="w-4 h-4 text-white" />}
                        {chartType === 'bar'           && <BarChart   className="w-4 h-4 text-white" />}
                        <ChevronDown className="w-3 h-3 text-slate-500" />
                    </button>

                    {showChartTypeDropdown && (
                        <div className="absolute top-full left-0 mt-1 w-40 bg-white/7 border border-white/10 rounded shadow-2xl z-50 py-1">
                            {[
                                { id: 'hollow_candle', label: 'Hollow Candles', icon: Box },
                                { id: 'candlestick',   label: 'Candles',        icon: BarChart2 },
                                { id: 'heikin_ashi',   label: 'Heikin Ashi',    icon: Activity },
                                { id: 'bar',           label: 'Bars',           icon: BarChart },
                                { id: 'line',          label: 'Line',           icon: LineChart },
                                { id: 'area',          label: 'Area',           icon: AreaChart },
                            ].map(type => (
                                <button
                                    key={type.id}
                                    onClick={() => { setChartType(type.id); setShowChartTypeDropdown(false); }}
                                    className="w-full flex items-center gap-3 px-3 py-2 text-xs hover:bg-white/10 text-[#d1d4dc] transition-colors"
                                >
                                    <type.icon className={`w-4 h-4 ${chartType === type.id ? 'text-white' : 'text-slate-400'}`} />
                                    {type.label}
                                </button>
                            ))}
                        </div>
                    )}
                </div>

                <button
                    onClick={() => setShowIndicatorModal(true)}
                    className="ml-2 px-3 py-1.5 bg-white/7 hover:bg-white/10 border border-white/10 rounded text-[11px] font-bold text-[#d1d4dc] transition-colors"
                >
                    Indicators
                </button>

                {/* Stats toggle button */}
                <button
                    onClick={() => setShowStatsPanel(prev => !prev)}
                    className={`ml-2 flex items-center gap-1.5 px-3 py-1.5 border rounded text-[11px] font-bold transition-colors ${
                        showStatsPanel
                            ? 'bg-white border-white text-black'
                            : 'bg-white/7 hover:bg-white/10 border-white/10 text-[#d1d4dc]'
                    }`}
                >
                    <BarChart2 className="w-3.5 h-3.5" />
                    Stats
                </button>

                <div className="flex items-center gap-2 ml-auto relative" ref={settingsDropdownRef}>
                    <button
                        onClick={() => setShowSettingsDropdown(prev => !prev)}
                        className={`p-2 rounded transition-colors ${showSettingsDropdown ? 'bg-white/15 text-white' : 'hover:bg-white/10 text-slate-400'}`}
                        title="Chart settings"
                    >
                        <Settings className="w-5 h-5" />
                    </button>
                    {showSettingsDropdown && (
                        <div className="absolute top-12 right-0 w-56 bg-black/95 border border-white/15 rounded-lg shadow-2xl z-50 p-2">
                            <p className="text-[10px] uppercase font-bold tracking-widest px-2 py-1 text-slate-400">Chart Settings</p>
                            <button
                                onClick={() => setShowFooterBar(prev => !prev)}
                                className="w-full text-left text-xs px-2 py-2 rounded hover:bg-white/10 text-slate-200"
                            >
                                {showFooterBar ? 'Hide' : 'Show'} OHLC footer
                            </button>
                            <button
                                onClick={() => setShowStatsPanel(prev => !prev)}
                                className="w-full text-left text-xs px-2 py-2 rounded hover:bg-white/10 text-slate-200"
                            >
                                {showStatsPanel ? 'Hide' : 'Show'} stats panel
                            </button>
                            <p className="text-[10px] px-2 py-1 text-slate-500">More options coming soon.</p>
                        </div>
                    )}
                    <button
                        onClick={handleCaptureChart}
                        disabled={captureBusy}
                        className={`p-2 rounded transition-colors ${captureBusy ? 'text-slate-600 cursor-not-allowed' : 'hover:bg-white/10 text-slate-400'}`}
                        title="Download chart snapshot"
                    >
                        <Camera className="w-5 h-5" />
                    </button>
                </div>
            </header>

            <div className="flex flex-grow overflow-hidden">
                <main ref={captureFrameRef} className="flex-grow relative bg-black overflow-hidden" style={{ minHeight: '400px' }}>
                    {loading ? (
                        <div className="absolute inset-0 flex items-center justify-center bg-black/60 z-50">
                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
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
                                position={position}
                                tradeMarkers={tradeMarkers}
                            />
                        </ErrorBoundary>
                    )}
                </main>

                {showStatsPanel && (
                    <div className="w-[460px] flex-shrink-0 border-l border-white/10 overflow-y-auto">
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

            {showFooterBar && (
            <footer className="h-8 border-t border-white/10 bg-black flex items-center px-4 gap-6 text-[10px] font-medium text-slate-400 flex-shrink-0">
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
            )}
        </div>
    );
};

export default StockDetail;
