import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import {
    Search, TrendingUp, BarChart2, Activity, Clock, Layers,
    Settings, Camera, Maximize2, Star, Eye, Edit, Trash2, X, Plus,
    ChevronDown, BarChart, LineChart, AreaChart, Box, History, AlertTriangle
} from 'lucide-react';
import StockChart from '../components/StockChart';
import IndicatorModal from '../components/IndicatorModal';
import { computeHeikinAshi } from '../utils/heikinAshi';

const TIMEFRAMES = [
    { label: '1m', value: '1m' },
    { label: '5m', value: '5m' },
    { label: '15m', value: '15m' },
    { label: '30m', value: '30m' },
    { label: '1H', value: '1h' },
    { label: '4H', value: '4h' },
    { label: '1D', value: '1d' },
    { label: '1W', value: '1wk' },
    { label: '1M', value: '1mo' },
];

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
    const [data, setData] = useState([]);
    const [indicatorData, setIndicatorData] = useState([]);
    const [quote, setQuote] = useState(null);
    const [period, setPeriod] = useState('1y');
    const [interval, setInterval] = useState('1d');
    const [chartType, setChartType] = useState('candlestick');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [showIndicatorModal, setShowIndicatorModal] = useState(false);
    const [showChartTypeDropdown, setShowChartTypeDropdown] = useState(false);
    const [activeIndicators, setActiveIndicators] = useState(['rsi', 'macd', 'sma_20', 'sma_50']);

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
                <div className="flex items-center gap-2 pr-4 border-r border-[#2a2e39]">
                    <TrendingUp className="text-blue-500 w-5 h-5" />
                    <span className="font-bold text-white tracking-widest text-sm">TV_CLONE</span>
                </div>

                <div className="relative flex-grow max-w-xs">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                    <input
                        type="text"
                        placeholder="Search symbol..."
                        className="w-full bg-[#1e222d] border border-transparent focus:border-[#2962ff] rounded px-9 py-1 text-sm outline-none"
                        value={ticker}
                        readOnly
                    />
                </div>

                <div className="flex bg-[#1e222d] p-0.5 rounded border border-[#2a2e39] ml-2">
                    {TIMEFRAMES.map(tf => (
                        <button
                            key={tf.label}
                            onClick={() => setInterval(tf.value)}
                            className={`px-3 py-1 text-[11px] font-bold rounded transition-colors ${interval === tf.value ? 'bg-[#2962ff] text-white' : 'hover:bg-[#2a2e39] text-[#d1d4dc]'}`}
                        >
                            {tf.label}
                        </button>
                    ))}
                </div>

                <div className="relative ml-2">
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
                            />
                        </ErrorBoundary>
                    )}
                </main>

                {showIndicatorModal && (
                    <IndicatorModal
                        onClose={() => setShowIndicatorModal(false)}
                        onSelect={(indicator) => {
                            if (!activeIndicators.includes(indicator)) {
                                setActiveIndicators([...activeIndicators, indicator]);
                            }
                            setShowIndicatorModal(false);
                        }}
                    />
                )}
            </div>

            <footer className="h-8 border-t border-[#2a2e39] bg-[#131722] flex items-center px-4 gap-6 text-[10px] font-medium text-slate-400 flex-shrink-0">
                <div className="flex gap-4">
                    <span>O <span className="text-white">{data[data.length - 1]?.Open?.toFixed(2) || '0.00'}</span></span>
                    <span>H <span className="text-white">{data[data.length - 1]?.High?.toFixed(2) || '0.00'}</span></span>
                    <span>L <span className="text-white">{data[data.length - 1]?.Low?.toFixed(2) || '0.00'}</span></span>
                    <span>C <span className="text-white">{data[data.length - 1]?.Close?.toFixed(2) || '0.00'}</span></span>
                </div>
            </footer>
        </div>
    );
};

export default StockDetail;
