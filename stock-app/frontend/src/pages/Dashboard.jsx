import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import axios from 'axios';
import { TrendingUp, TrendingDown, Clock, Plus, X, Check, ChevronDown, List } from 'lucide-react';
import { Link } from 'react-router-dom';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip as RechartsTooltip } from 'recharts';
import ParticlesBg from '../components/ParticlesBg';

// ── Market strip tickers ──────────────────────────────────────────────────────
const MARKET_TICKERS = [
    { ticker: '^GSPC',    label: 'S&P 500' },
    { ticker: '^IXIC',    label: 'NASDAQ' },
    { ticker: '^DJI',     label: 'DOW' },
    { ticker: 'ES=F',     label: 'S&P Fut.' },
    { ticker: 'NQ=F',     label: 'NQ Fut.' },
    { ticker: 'YM=F',     label: 'Dow Fut.' },
    { ticker: 'DX=F',     label: 'DXY' },
    { ticker: 'GC=F',     label: 'Gold' },
    { ticker: 'SI=F',     label: 'Silver' },
    { ticker: 'CL=F',     label: 'Oil WTI' },
];

// ── Default watchlist contents ────────────────────────────────────────────────
const DEFAULT_MAIN     = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN'];
const DEFAULT_EXTENDED = [
    'A','AAON','ABT','ADBE','ADMA','ADSK','ADUS','AEO','AIT','ALC','ALG','ALGM','ALGN','ALLE',
    'AMAT','AMBA','AMD','AME','AMRZ','ANF','AOS','APLS','APPF','ARCB','ARLP','AS','ASAN','ASML',
    'ASO','ATGE','ATR','AVDL','AVGO','AWI','AXSM','AYI','AZN','AZO','BBY','BCC','BCPC','BFAM',
    'BIRK','BKR','BLLN','BLSH','BMI','BOBS','BOOT','BRC','BROS','BRZE','BSX','BSY','BURL','CAI',
    'CALX','CARG','CDNS','CELH','CGNX','CHD','CHE','CHRW','CHT','CL','CLX','CNR','COCO','COLM',
    'COMP','COR','CPRI','CROX','CRS','CRUL','CSI','CTAS','CTVA','CVX','DASH','DCI','DD','DECK',
    'DGII','DHR','DHT','DKS','DLO','DOCU','DORN','DOX','DSGX','DT','DV','DXCM','DY','ECG','ECL',
    'EFX','EL','ELF','EME','ENSG','EOG','EPAC','EPAM','ETN','EW','EXPD','EXTR','EYE','FAST','FDP',
    'FELE','FERG','FFIV','FICO','FIX','FIZZ','FLOC','FN','FND','FOLD','FORM','FPS','FRPT','FSLR',
    'FSS','FTNT','FTV','GEV','GGG','GKOS','GMED','GNRC','GNTX','GRDN','GTR','GWW','HBNB','HD',
    'HOLX','HSY','HUBB','HUBG','HUBS','HWKN','IAS','IBP','ICLR','IDXX','IESC','IEX','IMO','INSP',
    'INTA','INVX','IOSP','IOT','IPAR','IR','ITT','ITW','JBHT','JCI','JNJ','JSG','KADA','KAI',
    'KEX','KLAC','KMB','KN','KOF','LAUR','LB','LECO','LFST','LFUS','LIN','LLY','LNTH','LOGI',
    'LOPE','LPX','LRCX','LSTR','LULU','MA','MANH','MASI','MATX','MBLY','MCHP','MCK','MCO','MEDP',
    'MGRC','MGY','MICC','MKC','MLM','MNST','MOD','MPWR','MRVL','MSA','MSCI','MSI','MSM','MTD',
    'MTRN','MU','MYRG','NDSN','NICE','NKE','NNNN','NOVT','NTRA','NVDA','NVO','NVR','NVS','NVT',
    'NXPI','NXT','ODFL','OLLI','OMAB','OMCL','ON','ONON','ORLY','PACS','PAY','PCOR','PEGA','PEN',
    'PG','PH','PLUS','PNR','PODD','POOL','POWI','PRVA','PSTG','PTC','PTRN','PWR','QCOM','RBA',
    'RBC','RDY','RELX','RELY','RMD','ROG','ROK','ROL','ROST','RPM','RS','RVLV','RXO','SAIA',
    'SAIL','SEDG','SHW','SIMO','SITE','SKY','SLAB','SMTC','SN','SNDK','SNDR','SNPS','SNY','SONO',
    'SPGI','SPSC','SPXC','SSD','STE','STM','STNG','STRA','STRL','STVN','STX','SUNC','SWKS','SYK',
    'SYM','TCGL','TEAM','TECH','TEL','TENB','TER','TGLS','TGTX','THO','TILE','TJX','TPL','TREX',
    'TRMB','TRNO','TS','TSCO','TSLA','TSM','TT','TTC','TTD','TTEK','TXN','TYL','UFPI','UI','UII',
    'ULS','ULTA','UNF','VCEL','VLTO','VMC','VMI','VRSN','VRT','VSEC','WAT','WCN','WDFC','WFG',
    'WGS','WHD','WINA','WMS','WOR','WRBY','WS','WSM','WSO','WST','WTS','XOM','XPRO','XYL','YETI',
    'ZBRA','ZETA','ZTS','ZWS',
];
const DEFAULT_INDEXES  = ['^GSPC', '^IXIC', '^DJI', '^RUT', '^VIX', 'GC=F', 'SI=F', 'CL=F', 'DX=F', '^TNX'];

// ── Watchlist tab config ──────────────────────────────────────────────────────
const WATCHLIST_TABS = [
    { id: 'portfolio', label: 'Portfolio', storageKey: null, defaultList: [] },
    { id: 'main',     label: 'Watchlist',  storageKey: 'tradr_watchlist_main',     defaultList: DEFAULT_MAIN },
    { id: 'extended', label: 'Extended',   storageKey: 'tradr_watchlist_extended',  defaultList: DEFAULT_EXTENDED },
    { id: 'indexes',  label: 'Indexes',    storageKey: 'tradr_watchlist_indexes',   defaultList: DEFAULT_INDEXES },
];

// ── Sort options ──────────────────────────────────────────────────────────────
const SORT_OPTIONS = [
    { value: 'pct_desc',    label: '% Change ↑' },
    { value: 'pct_asc',     label: '% Change ↓' },
    { value: 'price_desc',  label: 'Price ↑' },
    { value: 'price_asc',   label: 'Price ↓' },
    { value: 'change_desc', label: 'Change $ ↑' },
    { value: 'change_asc',  label: 'Change $ ↓' },
    { value: 'sector_asc',  label: 'Sector A→Z' },
    { value: 'sector_desc', label: 'Sector Z→A' },
    { value: 'rsi_low',     label: 'RSI Oversold' },
    { value: 'rsi_high',    label: 'RSI Overbought' },
    { value: 'macd_bull',   label: 'MACD Bullish X' },
    { value: 'macd_bear',   label: 'MACD Bearish X' },
    { value: 'default',     label: 'Default Order' },
];

// ── Helpers ───────────────────────────────────────────────────────────────────
const getRSI = (indData) => {
    if (!Array.isArray(indData) || indData.length === 0) return null;
    const last = indData[indData.length - 1];
    if (!last) return null;
    const col = Object.keys(last).find(k => k.toUpperCase().startsWith('RSI'));
    return col ? last[col] : null;
};

// 1 = bullish cross, -1 = bearish cross, 0 = none
const getMACDCross = (indData) => {
    if (!Array.isArray(indData) || indData.length < 2) return 0;
    const prev = indData[indData.length - 2];
    const curr = indData[indData.length - 1];
    if (!prev || !curr) return 0;
    const col = Object.keys(curr).find(k => /^MACDh_/i.test(k));
    if (!col) return 0;
    const p = prev[col];
    const c = curr[col];
    if (p != null && c != null) {
        if (p <= 0 && c > 0) return 1;
        if (p >= 0 && c < 0) return -1;
    }
    return 0;
};

// ── Market Indices Chart ──────────────────────────────────────────────────────
const INDICES = [
    { ticker: '^GSPC', label: 'S&P 500' },
    { ticker: '^IXIC', label: 'NASDAQ' },
    { ticker: '^DJI', label: 'DOW' },
];

const TIMEFRAMES = [
    { value: '1d', label: '1D', period: '1d', interval: '5m' },
    { value: '1w', label: '1W', period: '5d', interval: '1h' },
    { value: '1m', label: '1M', period: '1mo', interval: '1d' },
    { value: '3m', label: '3M', period: '3mo', interval: '1d' },
    { value: '6m', label: '6M', period: '6mo', interval: '1d' },
    { value: '1y', label: '1Y', period: '1y', interval: '1d' },
];

const MarketIndicesChart = () => {
    const [selectedIndex, setSelectedIndex] = useState('^GSPC'); // S&P 500
    const [selectedTimeframe, setSelectedTimeframe] = useState('6m'); // 6 months
    const [chartData, setChartData] = useState([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            try {
                const tf = TIMEFRAMES.find(t => t.value === selectedTimeframe);
                const response = await axios.get(`/api/stock/${selectedIndex}/history`, {
                    params: { period: tf.period, interval: tf.interval }
                });
                
                const formatted = response.data.map(item => ({
                    date: item.Date,
                    shortDate: String(item.Date).slice(5, 10),
                    close: item.Close,
                })).filter(d => d.close != null);
                
                setChartData(formatted);
            } catch (error) {
                console.error('Error fetching index data:', error);
                setChartData([]);
            } finally {
                setLoading(false);
            }
        };
        
        fetchData();
    }, [selectedIndex, selectedTimeframe]);

    const currentIndex = INDICES.find(idx => idx.ticker === selectedIndex);

    return (
        <div className="glass overflow-hidden relative" style={{ padding: '24px' }}>
            <div className="relative z-10">
                <h2 className="font-medium flex items-center mb-4" style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                    <TrendingUp className="h-4 w-4 mr-2" style={{ color: 'var(--accent)' }} />
                    Market Indices
                </h2>

                {/* Index Selector */}
                <div className="flex items-center gap-2 mb-4">
                    {INDICES.map(idx => (
                        <button
                            key={idx.ticker}
                            onClick={() => setSelectedIndex(idx.ticker)}
                            className="text-sm font-bold px-3 py-1.5 rounded-lg transition-all"
                            style={{
                                background: selectedIndex === idx.ticker ? 'var(--accent)' : 'rgba(255,255,255,0.08)',
                                color: selectedIndex === idx.ticker ? '#000' : 'var(--text-secondary)',
                                border: selectedIndex === idx.ticker ? 'none' : '1px solid rgba(255,255,255,0.12)'
                            }}
                        >
                            {idx.label}
                        </button>
                    ))}
                </div>

                {/* Timeframe Selector */}
                <div className="flex items-center gap-1.5 mb-4">
                    {TIMEFRAMES.map(tf => (
                        <button
                            key={tf.value}
                            onClick={() => setSelectedTimeframe(tf.value)}
                            className="text-xs font-bold px-2.5 py-1 rounded-lg transition-all"
                            style={{
                                background: selectedTimeframe === tf.value ? 'var(--accent)' : 'rgba(255,255,255,0.08)',
                                color: selectedTimeframe === tf.value ? '#000' : 'var(--text-secondary)',
                                border: selectedTimeframe === tf.value ? 'none' : '1px solid rgba(255,255,255,0.12)'
                            }}
                        >
                            {tf.label}
                        </button>
                    ))}
                </div>

                {/* Chart */}
                <div className="h-64">
                    {loading ? (
                        <div className="flex items-center justify-center h-full">
                            <span className="text-sm animate-pulse" style={{ color: 'var(--text-tertiary)' }}>Loading...</span>
                        </div>
                    ) : chartData.length > 0 ? (
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={chartData}>
                                <XAxis 
                                    dataKey="shortDate" 
                                    tick={{ fill: '#8b95a5', fontSize: 11 }} 
                                    axisLine={false} 
                                    tickLine={false} 
                                />
                                <YAxis
                                    tick={{ fill: '#8b95a5', fontSize: 11 }}
                                    axisLine={false}
                                    tickLine={false}
                                    width={80}
                                    domain={['auto', 'auto']}
                                    tickFormatter={(v) => `${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                                />
                                <RechartsTooltip
                                    formatter={(v) => [`${Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 })}`, currentIndex?.label]}
                                    labelFormatter={(l, payload) => payload?.[0]?.payload?.date || l}
                                    contentStyle={{
                                        background: 'rgba(0,0,0,0.9)',
                                        border: '1px solid rgba(255,255,255,0.15)',
                                        borderRadius: '10px',
                                        color: '#fff',
                                    }}
                                />
                                <Line 
                                    type="monotone" 
                                    dataKey="close" 
                                    stroke="var(--accent)" 
                                    strokeWidth={2.5} 
                                    dot={false} 
                                    isAnimationActive={false} 
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    ) : (
                        <div className="flex items-center justify-center h-full">
                            <span className="text-sm" style={{ color: 'var(--text-tertiary)' }}>No data available</span>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

// ── Market Strip ──────────────────────────────────────────────────────────────
const SCROLL_SPEED = 0.45; // px per animation frame

// Pure display cell — receives pre-fetched quote as prop
const MarketCell = ({ label, quote }) => {
    const isPos = (quote?.change ?? 0) >= 0;
    return (
        <div className="flex items-center gap-3 px-4 py-2 border-r border-white/10 flex-shrink-0">
            <span className="text-[11px] font-bold uppercase tracking-wider" style={{ color: 'var(--text-tertiary)' }}>{label}</span>
            {quote ? (
                <>
                    <span className="text-sm font-bold tabular-nums" style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
                        {quote.current_price?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </span>
                    <span className="text-[11px] font-bold tabular-nums" style={{ color: isPos ? 'var(--positive)' : 'var(--negative)' }}>
                        {isPos ? '+' : ''}{quote.percent_change?.toFixed(2)}%
                    </span>
                </>
            ) : (
                <span className="text-xs animate-pulse" style={{ color: 'var(--text-tertiary)' }}>···</span>
            )}
        </div>
    );
};

const MarketStrip = () => {
    const [quotes, setQuotes] = useState({});
    const trackRef = useRef(null);
    const containerRef = useRef(null);
    const offsetRef = useRef(0);
    const isDragging = useRef(false);
    const dragStartX = useRef(0);
    const dragStartOffset = useRef(0);
    const rafRef = useRef(null);

    // Fetch quotes once — only 10 calls instead of doubling with duplicate components
    useEffect(() => {
        MARKET_TICKERS.forEach(({ ticker }) => {
            axios.get(`/api/stock/${encodeURIComponent(ticker)}/quote`)
                .then(r => setQuotes(prev => ({ ...prev, [ticker]: r.data })))
                .catch(() => {});
        });
    }, []);

    // Wrap offset so it stays in [-singleWidth, 0)
    const normalize = useCallback((val) => {
        const track = trackRef.current;
        if (!track) return val;
        const w = track.scrollWidth / 2;
        if (w <= 0) return val;
        val = val % w;
        if (val > 0) val -= w;
        return val;
    }, []);

    // Animation loop — runs continuously, pauses while user is dragging
    useEffect(() => {
        const tick = () => {
            if (!isDragging.current && trackRef.current) {
                offsetRef.current = normalize(offsetRef.current - SCROLL_SPEED);
                trackRef.current.style.transform = `translateX(${offsetRef.current}px)`;
            }
            rafRef.current = requestAnimationFrame(tick);
        };
        rafRef.current = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(rafRef.current);
    }, [normalize]);

    const handlePointerDown = useCallback((e) => {
        isDragging.current = true;
        dragStartX.current = e.clientX;
        dragStartOffset.current = offsetRef.current;
        e.currentTarget.style.cursor = 'grabbing';
        e.currentTarget.setPointerCapture(e.pointerId);
    }, []);

    const handlePointerMove = useCallback((e) => {
        if (!isDragging.current) return;
        const delta = e.clientX - dragStartX.current;
        const newOffset = normalize(dragStartOffset.current + delta);
        offsetRef.current = newOffset;
        if (trackRef.current) trackRef.current.style.transform = `translateX(${newOffset}px)`;
    }, [normalize]);

    const handlePointerUp = useCallback((e) => {
        isDragging.current = false;
        if (e.currentTarget) e.currentTarget.style.cursor = 'grab';
    }, []);

    return (
        <div
            ref={containerRef}
            className="glass overflow-hidden select-none"
            style={{ padding: '4px 0', cursor: 'grab' }}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerUp}
        >
            {/* Two copies side-by-side for seamless infinite loop */}
            <div ref={trackRef} className="flex" style={{ willChange: 'transform' }}>
                {[0, 1].flatMap(copy =>
                    MARKET_TICKERS.map(m => (
                        <MarketCell key={`${m.ticker}-${copy}`} label={m.label} quote={quotes[m.ticker]} />
                    ))
                )}
            </div>
        </div>
    );
};

// ── Watchlist Card ────────────────────────────────────────────────────────────
const WatchlistCard = ({ ticker, onRemove, quote, indData, chartData, showChart = false, allowRemove = true, loading = false }) => {

    if (loading) return <div className="glass animate-pulse h-28" />;
    if (!quote) {
        return (
            <div className="glass h-28 flex items-center justify-center" style={{ borderRadius: '16px' }}>
                <span className="text-xs" style={{ color: 'var(--text-tertiary)' }}>No data yet</span>
            </div>
        );
    }

    const isPositive = (quote.change ?? 0) >= 0;
    const rsi = getRSI(indData);
    const macdCross = getMACDCross(indData);
    const sector = quote.sector || null;

    return (
        <div className="glass flex flex-col justify-between group relative" style={{ padding: '18px', borderRadius: '16px' }}>
            {allowRemove && (
                <button
                    onClick={(e) => { e.preventDefault(); onRemove(ticker); }}
                    className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-all z-10"
                    style={{ color: 'var(--text-tertiary)' }}
                    title="Remove"
                >
                    <X className="h-3.5 w-3.5" />
                </button>
            )}
            <Link to={`/stock/${ticker}`} className="flex flex-col justify-between h-full">
                <div className="flex justify-between items-start">
                    <div className="min-w-0 flex-1 pr-2">
                        <h3 className="font-bold text-lg uppercase" style={{ color: '#fff', fontWeight: 700, fontSize: '0.95rem', letterSpacing: '-0.01em' }}>{ticker}</h3>
                        <p className="text-xs truncate" style={{ color: 'var(--text-tertiary)' }}>{quote.longName}</p>
                        {sector && (
                            <span className="inline-block mt-1 text-[10px] font-medium px-1.5 py-0.5 rounded truncate max-w-full" style={{ background: 'rgba(255,255,255,0.08)', color: 'var(--text-secondary)' }}>
                                {sector}
                            </span>
                        )}
                    </div>
                    <div
                        className="px-2 py-1 rounded text-xs font-bold flex-shrink-0"
                        style={isPositive
                            ? { background: 'rgba(48,209,88,0.18)', color: 'var(--positive)', border: '1px solid rgba(48,209,88,0.3)', borderRadius: '8px', padding: '2px 8px', fontSize: '0.75rem', fontWeight: 600 }
                            : { background: 'rgba(255,69,58,0.18)', color: 'var(--negative)', border: '1px solid rgba(255,69,58,0.3)', borderRadius: '8px', padding: '2px 8px', fontSize: '0.75rem', fontWeight: 600 }
                        }
                    >
                        {isPositive ? '+' : ''}{quote.percent_change?.toFixed(2)}%
                    </div>
                </div>
                <div className="mt-3 flex justify-between items-end">
                    <span className="text-2xl font-bold" style={{ color: '#fff', fontWeight: 700, fontSize: '1.35rem' }}>${quote.current_price?.toFixed(2)}</span>
                    <div className="flex items-center" style={{ color: isPositive ? 'var(--positive)' : 'var(--negative)' }}>
                        {isPositive ? <TrendingUp className="h-4 w-4 mr-1" /> : <TrendingDown className="h-4 w-4 mr-1" />}
                        <span className="text-sm font-medium">{quote.change?.toFixed(2)}</span>
                    </div>
                </div>
                {/* RSI + MACD badges */}
                <div className="mt-2 flex gap-1.5 flex-wrap">
                    {rsi != null && (
                        <span
                            className="text-[10px] font-bold px-1.5 py-0.5 rounded tabular-nums"
                            style={rsi < 30
                                ? { background: 'rgba(48,209,88,0.15)', color: 'var(--positive)', borderRadius: '6px', padding: '2px 6px', fontSize: '0.7rem' }
                                : rsi > 70
                                    ? { background: 'rgba(255,69,58,0.15)', color: 'var(--negative)', borderRadius: '6px', padding: '2px 6px', fontSize: '0.7rem' }
                                    : { background: 'rgba(255,255,255,0.08)', color: 'var(--text-secondary)', borderRadius: '6px', padding: '2px 6px', fontSize: '0.7rem' }
                            }
                        >
                            RSI {rsi.toFixed(1)}
                        </span>
                    )}
                    {macdCross === 1 && (
                        <span className="text-[10px] font-bold px-1.5 py-0.5 rounded" style={{ background: 'rgba(48,209,88,0.15)', color: 'var(--positive)', borderRadius: '6px', padding: '2px 6px', fontSize: '0.7rem' }}>MACD ↑</span>
                    )}
                    {macdCross === -1 && (
                        <span className="text-[10px] font-bold px-1.5 py-0.5 rounded" style={{ background: 'rgba(255,69,58,0.15)', color: 'var(--negative)', borderRadius: '6px', padding: '2px 6px', fontSize: '0.7rem' }}>MACD ↓</span>
                    )}
                </div>
                {showChart && Array.isArray(chartData) && chartData.length > 1 && (
                    <div className="mt-2 h-10">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={chartData}>
                                <Line
                                    type="monotone"
                                    dataKey="close"
                                    stroke={isPositive ? 'var(--positive)' : 'var(--negative)'}
                                    strokeWidth={1.6}
                                    dot={false}
                                    isAnimationActive={false}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                )}
            </Link>
        </div>
    );
};

// ── Sort Dropdown ─────────────────────────────────────────────────────────────
const SortDropdown = ({ value, onChange }) => {
    const [open, setOpen] = useState(false);
    const selected = SORT_OPTIONS.find(o => o.value === value);
    return (
        <div className="relative">
            <button
                onClick={() => setOpen(o => !o)}
                className="flex items-center gap-1.5 text-sm transition-colors"
                style={{ color: 'var(--text-secondary)' }}
            >
                Sort: <span className="font-medium" style={{ color: 'var(--text-primary)' }}>{selected?.label}</span>
                <ChevronDown className={`h-3.5 w-3.5 transition-transform ${open ? 'rotate-180' : ''}`} />
            </button>
            {open && (
                <div className="glass absolute right-0 top-full mt-1 w-44 z-20 py-1 overflow-hidden">
                    {SORT_OPTIONS.map(opt => (
                        <button
                            key={opt.value}
                            onClick={() => { onChange(opt.value); setOpen(false); }}
                            className="w-full text-left px-4 py-2 text-sm transition-colors"
                            style={opt.value === value
                                ? { color: '#fff', background: 'rgba(255,255,255,0.12)' }
                                : { color: 'var(--text-secondary)' }
                            }
                        >
                            {opt.label}
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
};

// ── Watchlist Customize Modal ─────────────────────────────────────────────────
const WatchlistModal = ({ title, watchlist, onClose, onSave }) => {
    const [items, setItems] = useState([...watchlist]);
    const [input, setInput] = useState('');
    const [error, setError] = useState('');

    const add = () => {
        const t = input.trim().toUpperCase();
        if (!t) return;
        if (items.includes(t)) { setError(`${t} is already in the list`); return; }
        setItems(prev => [...prev, t]);
        setInput('');
        setError('');
    };
    const remove = (t) => setItems(prev => prev.filter(i => i !== t));

    return (
        <div className="fixed inset-0 flex items-center justify-center z-50 p-4" style={{ background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(8px)' }}>
            <div className="glass p-6 w-full max-w-sm" style={{ padding: '24px' }}>
                <div className="flex justify-between items-center mb-5">
                    <h3 className="text-lg font-bold">Customize — {title}</h3>
                    <button onClick={onClose} style={{ color: 'var(--text-tertiary)' }}><X className="h-5 w-5" /></button>
                </div>
                <div className="flex gap-2 mb-4">
                    <input
                        type="text"
                        placeholder="Add ticker (e.g. AAPL)"
                        className="glass-input uppercase text-sm flex-1"
                        style={{ padding: '8px 12px', color: 'white' }}
                        value={input}
                        onChange={e => { setInput(e.target.value.toUpperCase()); setError(''); }}
                        onKeyDown={e => e.key === 'Enter' && add()}
                    />
                    <button onClick={add} className="px-3 py-2 rounded-xl text-black transition-colors" style={{ background: 'var(--accent)', borderRadius: '12px' }}>
                        <Plus className="h-4 w-4" />
                    </button>
                </div>
                {error && <p className="text-sm mb-3" style={{ color: 'var(--negative)' }}>{error}</p>}
                <div className="space-y-1 max-h-64 overflow-y-auto mb-5">
                    {items.map(t => (
                        <div key={t} className="flex items-center justify-between px-3 py-2 rounded-lg" style={{ background: 'rgba(255,255,255,0.07)' }}>
                            <span className="font-bold text-sm text-white">{t}</span>
                            <button onClick={() => remove(t)} style={{ color: 'var(--text-tertiary)' }} className="hover:text-rose-400 transition-colors">
                                <X className="h-4 w-4" />
                            </button>
                        </div>
                    ))}
                    {items.length === 0 && (
                        <p className="text-sm text-center py-4" style={{ color: 'var(--text-tertiary)' }}>No tickers yet</p>
                    )}
                </div>
                <div className="flex gap-3">
                    <button onClick={onClose} className="flex-1 py-2 rounded-xl text-white text-sm transition-colors" style={{ background: 'rgba(255,255,255,0.08)', borderRadius: 'var(--radius-btn)' }}>
                        Cancel
                    </button>
                    <button
                        onClick={() => { onSave(items); onClose(); }}
                        className="flex-1 py-2 text-black text-sm font-bold transition-colors flex items-center justify-center gap-2"
                        style={{ background: 'var(--accent)', borderRadius: 'var(--radius-btn)' }}
                    >
                        <Check className="h-4 w-4" /> Save
                    </button>
                </div>
            </div>
        </div>
    );
};

// ── Dashboard ─────────────────────────────────────────────────────────────────
const Dashboard = () => {
    const [portfolio, setPortfolio] = useState(null);
    const [portfolioTimeframe, setPortfolioTimeframe] = useState('1w');

    // Load all three watchlists from localStorage
    const [watchlists, setWatchlists] = useState(() => {
        const result = {};
        for (const tab of WATCHLIST_TABS) {
            if (!tab.storageKey) {
                result[tab.id] = tab.defaultList;
                continue;
            }
            try {
                const saved = localStorage.getItem(tab.storageKey);
                result[tab.id] = saved ? JSON.parse(saved) : tab.defaultList;
            } catch {
                result[tab.id] = tab.defaultList;
            }
        }
        return result;
    });

    const [activeTab, setActiveTab] = useState('portfolio');
    const [showCustomize, setShowCustomize] = useState(false);
    const [sortBy, setSortBy] = useState('pct_desc');
    const [quotesMap, setQuotesMap] = useState({});
    const [indicatorsMap, setIndicatorsMap] = useState({});
    const [miniChartsMap, setMiniChartsMap] = useState({});
    const [loadingMap, setLoadingMap] = useState({});
    const pendingRef = useRef(new Set());
    const fetchedVersionRef = useRef({});  // ticker -> pollTick at last fetch
    const [pollTick, setPollTick] = useState(0);

    const portfolioTabTickers = useMemo(() => {
        const holdings = Array.isArray(portfolio?.holdings) ? portfolio.holdings : [];
        const seen = new Set();
        return holdings
            .map(h => String(h?.ticker || '').toUpperCase())
            .filter(t => t && !seen.has(t) && seen.add(t));
    }, [portfolio]);

    const watchlist = activeTab === 'portfolio' ? portfolioTabTickers : (watchlists[activeTab] ?? []);
    const activeTabConfig = WATCHLIST_TABS.find(t => t.id === activeTab);

    useEffect(() => {
        axios.get('/api/portfolio/analytics', {
            params: { timeframe: portfolioTimeframe }
        })
            .then(r => setPortfolio(r.data))
            .catch(() => {});
    }, [portfolioTimeframe]);

    // Poll portfolio + quotes every 60 seconds
    useEffect(() => {
        const id = setInterval(() => {
            if (document.visibilityState !== 'visible') return;
            axios.get('/api/portfolio/analytics', { params: { timeframe: portfolioTimeframe } })
                .then(r => setPortfolio(r.data))
                .catch(() => {});
            setPollTick(t => t + 1);
        }, 60_000);
        return () => clearInterval(id);
    }, [portfolioTimeframe]);

    useEffect(() => {
        const activeTickers = watchlist;
        if (activeTickers.length === 0) return;

        const originalIndex = new Map(activeTickers.map((t, i) => [t, i]));
        const ordered = [...activeTickers].sort((a, b) => {
            if (activeTab !== 'extended') {
                return (originalIndex.get(a) ?? 0) - (originalIndex.get(b) ?? 0);
            }

            const capA = Number(quotesMap[a]?.market_cap);
            const capB = Number(quotesMap[b]?.market_cap);
            const hasA = Number.isFinite(capA);
            const hasB = Number.isFinite(capB);

            if (hasA && hasB && capA !== capB) return capB - capA;
            if (hasA && !hasB) return -1;
            if (!hasA && hasB) return 1;
            return (originalIndex.get(a) ?? 0) - (originalIndex.get(b) ?? 0);
        });

        const toFetch = ordered.filter((ticker) => {
            if (pendingRef.current.has(ticker)) return false;
            const fetchedAt = fetchedVersionRef.current[ticker] ?? -1;
            // Re-fetch on poll tick change OR if never fetched OR if indicators missing
            return fetchedAt < pollTick || !indicatorsMap[ticker];
        });

        if (toFetch.length === 0) return;

        const CONCURRENCY = 4;
        let cursor = 0;

        const worker = async () => {
            while (cursor < toFetch.length) {
                const index = cursor;
                cursor += 1;
                const ticker = toFetch[index];
                pendingRef.current.add(ticker);
                setLoadingMap(prev => ({ ...prev, [ticker]: true }));

                const enc = encodeURIComponent(ticker);
                try {
                    const [quoteRes, indRes] = await Promise.all([
                        axios.get(`/api/stock/${enc}/quote`),
                        axios.get(`/api/stock/${enc}/indicators?period=3mo&interval=1d&indicators=rsi,macd`),
                    ]);
                    fetchedVersionRef.current[ticker] = pollTick;
                    setQuotesMap(prev => ({ ...prev, [ticker]: quoteRes.data }));
                    setIndicatorsMap(prev => ({ ...prev, [ticker]: Array.isArray(indRes.data) ? indRes.data : [] }));
                } catch {
                    // Leave card in 'no data yet' state on transient failures.
                } finally {
                    pendingRef.current.delete(ticker);
                    setLoadingMap(prev => ({ ...prev, [ticker]: false }));
                }
            }
        };

        const workers = Array.from({ length: Math.min(CONCURRENCY, toFetch.length) }, () => worker());
        Promise.all(workers).catch(() => {});
    }, [watchlist, activeTab, quotesMap, indicatorsMap, pollTick]);

    const saveWatchlist = useCallback((newList) => {
        if (activeTab === 'portfolio') return;
        const tab = WATCHLIST_TABS.find(t => t.id === activeTab);
        if (!tab?.storageKey) return;
        setWatchlists(prev => ({ ...prev, [activeTab]: newList }));
        localStorage.setItem(tab.storageKey, JSON.stringify(newList));
    }, [activeTab]);

    const removeFromWatchlist = useCallback((ticker) => {
        saveWatchlist(watchlist.filter(t => t !== ticker));
    }, [watchlist, saveWatchlist]);

    const sortedWatchlist = useMemo(() => {
        if (sortBy === 'default') return watchlist;
        return [...watchlist].sort((a, b) => {
            const qa = quotesMap[a];
            const qb = quotesMap[b];
            const ia = indicatorsMap[a];
            const ib = indicatorsMap[b];
            switch (sortBy) {
                case 'pct_desc':    return (qb?.percent_change ?? -Infinity) - (qa?.percent_change ?? -Infinity);
                case 'pct_asc':     return (qa?.percent_change ?? Infinity)  - (qb?.percent_change ?? Infinity);
                case 'price_desc':  return (qb?.current_price ?? -Infinity)  - (qa?.current_price ?? -Infinity);
                case 'price_asc':   return (qa?.current_price ?? Infinity)   - (qb?.current_price ?? Infinity);
                case 'change_desc': return (qb?.change ?? -Infinity)         - (qa?.change ?? -Infinity);
                case 'change_asc':  return (qa?.change ?? Infinity)          - (qb?.change ?? Infinity);
                case 'rsi_low': {
                    const ra = getRSI(ia) ?? Infinity;
                    const rb = getRSI(ib) ?? Infinity;
                    return ra - rb;
                }
                case 'rsi_high': {
                    const ra = getRSI(ia) ?? -Infinity;
                    const rb = getRSI(ib) ?? -Infinity;
                    return rb - ra;
                }
                case 'sector_asc': {
                    const sa = (quotesMap[a]?.sector || 'zzz').toLowerCase();
                    const sb = (quotesMap[b]?.sector || 'zzz').toLowerCase();
                    return sa < sb ? -1 : sa > sb ? 1 : 0;
                }
                case 'sector_desc': {
                    const sa = (quotesMap[a]?.sector || '').toLowerCase();
                    const sb = (quotesMap[b]?.sector || '').toLowerCase();
                    return sa > sb ? -1 : sa < sb ? 1 : 0;
                }
                case 'macd_bull': return getMACDCross(ib) - getMACDCross(ia);
                case 'macd_bear': return getMACDCross(ia) - getMACDCross(ib);
                default: return 0;
            }
        });
    }, [watchlist, sortBy, quotesMap, indicatorsMap]);

    const top12ForMiniCharts = useMemo(() => {
        return [...watchlist]
            .sort((a, b) => (quotesMap[b]?.percent_change ?? -Infinity) - (quotesMap[a]?.percent_change ?? -Infinity))
            .slice(0, 12);
    }, [watchlist, quotesMap]);

    useEffect(() => {
        const toFetch = top12ForMiniCharts.filter((ticker) => !miniChartsMap[ticker]);
        if (toFetch.length === 0) return;

        const CONCURRENCY = 4;
        let cursor = 0;

        const worker = async () => {
            while (cursor < toFetch.length) {
                const i = cursor;
                cursor += 1;
                const ticker = toFetch[i];
                try {
                    const res = await axios.get(`/api/stock/${encodeURIComponent(ticker)}/history`, {
                        params: { period: '1d', interval: '5m' }
                    });
                    const series = Array.isArray(res.data)
                        ? res.data
                            .map((d) => ({ close: Number(d?.close) }))
                            .filter((d) => Number.isFinite(d.close))
                        : [];
                    setMiniChartsMap((prev) => ({ ...prev, [ticker]: series }));
                } catch {
                    setMiniChartsMap((prev) => ({ ...prev, [ticker]: [] }));
                }
            }
        };

        const workers = Array.from({ length: Math.min(CONCURRENCY, toFetch.length) }, () => worker());
        Promise.all(workers).catch(() => {});
    }, [top12ForMiniCharts, miniChartsMap]);

    const portfolioValueHistory = useMemo(() => {
        const series = Array.isArray(portfolio?.value_history) ? portfolio.value_history : [];
        return series
            .map((p) => {
                const value = Number(p?.value);
                const date = p?.date;
                if (!Number.isFinite(value) || !date) return null;
                const shortDate = portfolioTimeframe === '1d'
                    ? String(date).slice(11, 16)
                    : portfolioTimeframe === '1w'
                        ? String(date).slice(5, 16)
                        : portfolioTimeframe === 'max'
                            ? String(date).slice(2, 10)
                        : String(date).slice(5, 10);
                return { date, shortDate, value };
            })
            .filter(Boolean);
    }, [portfolio, portfolioTimeframe]);

    return (
        <>
        <ParticlesBg canvasId="particles-dashboard" />
        <div className="relative space-y-8" style={{ zIndex: 1 }}>
            {/* Market Strip */}
            <MarketStrip />

            {/* Portfolio + Market Indices */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Portfolio Overview card */}
                <div className="glass overflow-hidden relative" style={{ padding: '24px' }}>
                    <div className="relative z-10">
                        <h2 className="font-medium flex items-center" style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                            <Clock className="h-4 w-4 mr-2" style={{ color: 'var(--accent)' }} />
                            Portfolio Overview
                        </h2>
                        <div className="mt-4 flex flex-col sm:flex-row sm:items-end sm:space-x-8">
                            <div>
                                <p className="text-3xl sm:text-4xl font-black text-white">
                                    ${portfolio?.summary.total_value?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}
                                </p>
                                <p className="text-sm mt-1 uppercase tracking-wider" style={{ color: 'var(--text-tertiary)' }}>Total Value</p>
                            </div>
                            <div className="mt-4 sm:mt-0">
                                <p className="text-xl font-bold" style={{ color: portfolio?.summary.total_gain_loss >= 0 ? 'var(--positive)' : 'var(--negative)' }}>
                                    {portfolio?.summary.total_gain_loss >= 0 ? '+' : ''}
                                    ${portfolio?.summary.total_gain_loss?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}
                                </p>
                                <p className="text-sm mt-1 uppercase tracking-wider" style={{ color: 'var(--text-tertiary)' }}>Total G/L</p>
                            </div>
                            <div className="mt-4 sm:mt-0">
                                <p className="text-xl font-bold" style={{ color: portfolio?.summary.total_gain_loss_pct >= 0 ? 'var(--positive)' : 'var(--negative)' }}>
                                    {portfolio?.summary.total_gain_loss_pct >= 0 ? '+' : ''}
                                    {portfolio?.summary.total_gain_loss_pct?.toFixed(2) || '0.00'}%
                                </p>
                                <p className="text-sm mt-1 uppercase tracking-wider" style={{ color: 'var(--text-tertiary)' }}>Percentage</p>
                            </div>
                        </div>

                        {portfolioValueHistory.length > 1 && (
                            <>
                                <div className="flex items-center justify-between mt-5 mb-3" style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '12px' }}>
                                    <span className="text-xs font-bold uppercase tracking-wider" style={{ color: 'var(--text-tertiary)' }}>Value History</span>
                                    <div className="flex items-center gap-1.5">
                                        {[
                                            { key: '1d', label: '1D' },
                                            { key: '1w', label: '1W' },
                                            { key: '1m', label: '1M' },
                                            { key: '3mo', label: '3M' },
                                            { key: '6mo', label: '6M' },
                                            { key: '1y', label: '1Y' },
                                            { key: '2y', label: '2Y' },
                                            { key: 'max', label: 'ALL' },
                                        ].map(({ key, label }) => (
                                            <button
                                                key={key}
                                                onClick={() => setPortfolioTimeframe(key)}
                                                className="text-xs font-bold px-2.5 py-1 rounded-lg transition-all"
                                                style={{
                                                    background: portfolioTimeframe === key ? 'var(--accent)' : 'rgba(255,255,255,0.08)',
                                                    color: portfolioTimeframe === key ? '#000' : 'var(--text-secondary)',
                                                    border: portfolioTimeframe === key ? 'none' : '1px solid rgba(255,255,255,0.12)'
                                                }}
                                            >
                                                {label}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                                <div className="h-44">
                                    <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={portfolioValueHistory}>
                                        <XAxis dataKey="shortDate" tick={{ fill: '#8b95a5', fontSize: 11 }} axisLine={false} tickLine={false} />
                                        <YAxis
                                            tick={{ fill: '#8b95a5', fontSize: 11 }}
                                            axisLine={false}
                                            tickLine={false}
                                            width={80}
                                            domain={['auto', 'auto']}
                                            tickFormatter={(v) => `$${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                                        />
                                        <RechartsTooltip
                                            formatter={(v) => [`$${Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 })}`, 'Value']}
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
                            </>
                        )}
                    </div>
                </div>

                {/* Market Indices Chart */}
                <MarketIndicesChart />
            </div>

            {/* Watchlist section */}
            <section>
                {/* Section header */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                    {/* Tab pills */}
                    <div className="glass-pill">
                        {WATCHLIST_TABS.map(tab => (
                            <button
                                key={tab.id}
                                onClick={() => { setActiveTab(tab.id); setSortBy('pct_desc'); }}
                                className="flex items-center gap-1.5 rounded-[10px] text-sm transition-all"
                                style={activeTab === tab.id
                                    ? { background: 'rgba(255,255,255,0.15)', borderRadius: '10px', color: '#fff', padding: '6px 14px', fontWeight: 600 }
                                    : { color: 'rgba(255,255,255,0.55)', background: 'transparent', padding: '6px 14px' }
                                }
                            >
                                <List className="h-3.5 w-3.5" />
                                {tab.label}
                                <span
                                    className="text-[10px] font-bold px-1.5 py-0.5 rounded-full"
                                    style={activeTab === tab.id
                                        ? { background: 'rgba(255,255,255,0.20)', color: '#fff' }
                                        : { background: 'rgba(255,255,255,0.08)', color: 'var(--text-tertiary)' }
                                    }
                                >
                                    {tab.id === 'portfolio' ? portfolioTabTickers.length : (watchlists[tab.id] ?? []).length}
                                </span>
                            </button>
                        ))}
                    </div>

                    {/* Right controls */}
                    <div className="flex items-center gap-4">
                        <SortDropdown value={sortBy} onChange={setSortBy} />
                        {activeTab !== 'portfolio' && (
                            <button
                                onClick={() => setShowCustomize(true)}
                                className="text-sm font-medium hover:underline transition-colors"
                                style={{ color: 'var(--text-secondary)' }}
                            >
                                Customize
                            </button>
                        )}
                    </div>
                </div>

                {/* Cards grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                    {sortedWatchlist.map(ticker => (
                        <WatchlistCard
                            key={`${activeTab}-${ticker}`}
                            ticker={ticker}
                            onRemove={removeFromWatchlist}
                            quote={quotesMap[ticker]}
                            indData={indicatorsMap[ticker]}
                            chartData={miniChartsMap[ticker]}
                            showChart={top12ForMiniCharts.includes(ticker)}
                            allowRemove={activeTab !== 'portfolio'}
                            loading={!!loadingMap[ticker]}
                        />
                    ))}
                    {watchlist.length === 0 && (
                        <button
                            onClick={() => setShowCustomize(true)}
                            className="col-span-full flex items-center justify-center gap-2 py-12 rounded-xl transition-colors"
                            style={{ border: '1.5px dashed rgba(255,255,255,0.20)', color: 'var(--text-tertiary)' }}
                            disabled={activeTab === 'portfolio'}
                        >
                            <Plus className="h-5 w-5" /> {activeTab === 'portfolio' ? 'No portfolio holdings yet' : `Add stocks to ${activeTabConfig?.label}`}
                        </button>
                    )}
                </div>
            </section>

            {showCustomize && activeTab !== 'portfolio' && (
                <WatchlistModal
                    title={activeTabConfig?.label ?? 'Watchlist'}
                    watchlist={watchlist}
                    onClose={() => setShowCustomize(false)}
                    onSave={saveWatchlist}
                />
            )}
        </div>
        </>
    );
};

export default Dashboard;
