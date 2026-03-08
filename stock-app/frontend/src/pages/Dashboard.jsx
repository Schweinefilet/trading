import React, { useState, useEffect, useCallback, useMemo } from 'react';
import axios from 'axios';
import { TrendingUp, TrendingDown, Clock, Plus, X, Check, ChevronDown, List } from 'lucide-react';
import { Link } from 'react-router-dom';

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

// ── Market Strip ──────────────────────────────────────────────────────────────
const MarketItem = ({ ticker, label }) => {
    const [quote, setQuote] = useState(null);
    useEffect(() => {
        axios.get(`/api/stock/${encodeURIComponent(ticker)}/quote`)
            .then(r => setQuote(r.data))
            .catch(() => {});
    }, [ticker]);
    const isPos = (quote?.change ?? 0) >= 0;
    return (
        <div className="flex items-center gap-3 px-4 py-2 border-r border-slate-700/50 flex-shrink-0">
            <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">{label}</span>
            {quote ? (
                <>
                    <span className="text-sm font-bold text-white tabular-nums">
                        {quote.current_price?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </span>
                    <span className={`text-[11px] font-bold tabular-nums ${isPos ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {isPos ? '+' : ''}{quote.percent_change?.toFixed(2)}%
                    </span>
                </>
            ) : (
                <span className="text-xs text-slate-600 animate-pulse">···</span>
            )}
        </div>
    );
};

const MarketStrip = () => (
    <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl overflow-x-auto flex scrollbar-none">
        {MARKET_TICKERS.map(m => (
            <MarketItem key={m.ticker} ticker={m.ticker} label={m.label} />
        ))}
    </div>
);

// ── Watchlist Card ────────────────────────────────────────────────────────────
const WatchlistCard = ({ ticker, onRemove, quote: quoteProp, indData }) => {
    const [quote, setQuote] = useState(quoteProp || null);
    const [loading, setLoading] = useState(!quoteProp);

    useEffect(() => {
        if (quoteProp) { setQuote(quoteProp); setLoading(false); return; }
        setLoading(true);
        axios.get(`/api/stock/${encodeURIComponent(ticker)}/quote`)
            .then(r => setQuote(r.data))
            .catch(() => {})
            .finally(() => setLoading(false));
    }, [ticker, quoteProp]);

    if (loading) return <div className="card animate-pulse h-28" />;
    if (!quote) return null;

    const isPositive = (quote.change ?? 0) >= 0;
    const rsi = getRSI(indData);
    const macdCross = getMACDCross(indData);
    const sector = quote.sector || null;

    return (
        <div className="card hover:border-slate-500 transition-colors flex flex-col justify-between group relative">
            <button
                onClick={(e) => { e.preventDefault(); onRemove(ticker); }}
                className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 text-slate-600 hover:text-rose-400 transition-all z-10"
                title="Remove"
            >
                <X className="h-3.5 w-3.5" />
            </button>
            <Link to={`/stock/${ticker}`} className="flex flex-col justify-between h-full">
                <div className="flex justify-between items-start">
                    <div className="min-w-0 flex-1 pr-2">
                        <h3 className="font-bold text-lg text-white group-hover:text-blue-400 transition-colors uppercase">{ticker}</h3>
                        <p className="text-xs text-slate-500 truncate">{quote.longName}</p>
                        {sector && (
                            <span className="inline-block mt-1 text-[10px] font-medium px-1.5 py-0.5 rounded bg-slate-700/80 text-slate-400 truncate max-w-full">
                                {sector}
                            </span>
                        )}
                    </div>
                    <div className={`px-2 py-1 rounded text-xs font-bold flex-shrink-0 ${isPositive ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                        {isPositive ? '+' : ''}{quote.percent_change?.toFixed(2)}%
                    </div>
                </div>
                <div className="mt-3 flex justify-between items-end">
                    <span className="text-2xl font-bold">${quote.current_price?.toFixed(2)}</span>
                    <div className={`flex items-center ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {isPositive ? <TrendingUp className="h-4 w-4 mr-1" /> : <TrendingDown className="h-4 w-4 mr-1" />}
                        <span className="text-sm font-medium">{quote.change?.toFixed(2)}</span>
                    </div>
                </div>
                {/* RSI + MACD badges */}
                <div className="mt-2 flex gap-1.5 flex-wrap">
                    {rsi != null && (
                        <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded tabular-nums ${
                            rsi < 30 ? 'bg-emerald-500/20 text-emerald-400' :
                            rsi > 70 ? 'bg-rose-500/20 text-rose-400' :
                            'bg-slate-700 text-slate-400'
                        }`}>
                            RSI {rsi.toFixed(1)}
                        </span>
                    )}
                    {macdCross === 1 && (
                        <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400">MACD ↑</span>
                    )}
                    {macdCross === -1 && (
                        <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-400">MACD ↓</span>
                    )}
                </div>
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
                className="flex items-center gap-1.5 text-sm text-slate-400 hover:text-slate-200 transition-colors"
            >
                Sort: <span className="text-slate-200 font-medium">{selected?.label}</span>
                <ChevronDown className={`h-3.5 w-3.5 transition-transform ${open ? 'rotate-180' : ''}`} />
            </button>
            {open && (
                <div className="absolute right-0 top-full mt-1 w-44 bg-slate-800 border border-slate-700 rounded-xl shadow-2xl z-20 py-1 overflow-hidden">
                    {SORT_OPTIONS.map(opt => (
                        <button
                            key={opt.value}
                            onClick={() => { onChange(opt.value); setOpen(false); }}
                            className={`w-full text-left px-4 py-2 text-sm transition-colors ${
                                opt.value === value ? 'text-blue-400 bg-blue-500/10' : 'text-slate-300 hover:bg-slate-700'
                            }`}
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
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <div className="bg-slate-800 border border-slate-700 rounded-2xl p-6 w-full max-w-sm shadow-2xl">
                <div className="flex justify-between items-center mb-5">
                    <h3 className="text-lg font-bold">Customize — {title}</h3>
                    <button onClick={onClose} className="text-slate-500 hover:text-slate-200"><X className="h-5 w-5" /></button>
                </div>
                <div className="flex gap-2 mb-4">
                    <input
                        type="text"
                        placeholder="Add ticker (e.g. AAPL)"
                        className="flex-1 bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 uppercase text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                        style={{ color: 'white' }}
                        value={input}
                        onChange={e => { setInput(e.target.value.toUpperCase()); setError(''); }}
                        onKeyDown={e => e.key === 'Enter' && add()}
                    />
                    <button onClick={add} className="px-3 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-white transition-colors">
                        <Plus className="h-4 w-4" />
                    </button>
                </div>
                {error && <p className="text-rose-400 text-xs mb-3">{error}</p>}
                <div className="space-y-1 max-h-64 overflow-y-auto mb-5">
                    {items.map(t => (
                        <div key={t} className="flex items-center justify-between px-3 py-2 bg-slate-700/50 rounded-lg">
                            <span className="font-bold text-sm text-white">{t}</span>
                            <button onClick={() => remove(t)} className="text-slate-500 hover:text-rose-400 transition-colors">
                                <X className="h-4 w-4" />
                            </button>
                        </div>
                    ))}
                    {items.length === 0 && (
                        <p className="text-slate-500 text-sm text-center py-4">No tickers yet</p>
                    )}
                </div>
                <div className="flex gap-3">
                    <button onClick={onClose} className="flex-1 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-white text-sm transition-colors">
                        Cancel
                    </button>
                    <button
                        onClick={() => { onSave(items); onClose(); }}
                        className="flex-1 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-sm font-bold transition-colors flex items-center justify-center gap-2"
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

    // Load all three watchlists from localStorage
    const [watchlists, setWatchlists] = useState(() => {
        const result = {};
        for (const tab of WATCHLIST_TABS) {
            try {
                const saved = localStorage.getItem(tab.storageKey);
                result[tab.id] = saved ? JSON.parse(saved) : tab.defaultList;
            } catch {
                result[tab.id] = tab.defaultList;
            }
        }
        return result;
    });

    const [activeTab, setActiveTab] = useState('main');
    const [showCustomize, setShowCustomize] = useState(false);
    const [sortBy, setSortBy] = useState('pct_desc');
    const [quotesMap, setQuotesMap] = useState({});
    const [indicatorsMap, setIndicatorsMap] = useState({});

    const watchlist = watchlists[activeTab] ?? [];
    const activeTabConfig = WATCHLIST_TABS.find(t => t.id === activeTab);

    useEffect(() => {
        axios.get('/api/portfolio/analytics')
            .then(r => setPortfolio(r.data))
            .catch(() => {});
    }, []);

    // Fetch quotes + indicators for all tickers across all watchlists (deduplicated)
    const allTickers = useMemo(() => {
        const set = new Set();
        for (const tab of WATCHLIST_TABS) {
            for (const t of (watchlists[tab.id] ?? [])) set.add(t);
        }
        return [...set];
    }, [watchlists]);

    useEffect(() => {
        if (allTickers.length === 0) return;
        allTickers.forEach(ticker => {
            // Skip if already fetched
            if (quotesMap[ticker]) return;
            const enc = encodeURIComponent(ticker);
            axios.get(`/api/stock/${enc}/quote`)
                .then(r => setQuotesMap(prev => ({ ...prev, [ticker]: r.data })))
                .catch(() => {});
            axios.get(`/api/stock/${enc}/indicators?period=3mo&interval=1d&indicators=rsi,macd`)
                .then(r => setIndicatorsMap(prev => ({ ...prev, [ticker]: Array.isArray(r.data) ? r.data : [] })))
                .catch(() => {});
        });
    }, [allTickers]); // eslint-disable-line react-hooks/exhaustive-deps

    const saveWatchlist = useCallback((newList) => {
        const tab = WATCHLIST_TABS.find(t => t.id === activeTab);
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

    return (
        <div className="space-y-8">
            {/* Market Strip */}
            <MarketStrip />

            {/* Portfolio + CTA */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="card md:col-span-2 bg-gradient-to-br from-blue-600/20 to-slate-800 border-blue-500/30 overflow-hidden relative">
                    <div className="relative z-10">
                        <h2 className="text-slate-400 font-medium flex items-center">
                            <Clock className="h-4 w-4 mr-2" />
                            Portfolio Overview
                        </h2>
                        <div className="mt-4 flex flex-col sm:flex-row sm:items-end sm:space-x-8">
                            <div>
                                <p className="text-3xl sm:text-4xl font-black text-white">
                                    ${portfolio?.summary.total_value?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}
                                </p>
                                <p className="text-sm text-slate-400 mt-1 uppercase tracking-wider">Total Value</p>
                            </div>
                            <div className="mt-4 sm:mt-0">
                                <p className={`text-xl font-bold ${portfolio?.summary.total_gain_loss >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                    {portfolio?.summary.total_gain_loss >= 0 ? '+' : ''}
                                    ${portfolio?.summary.total_gain_loss?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}
                                </p>
                                <p className="text-sm text-slate-400 mt-1 uppercase tracking-wider">Total G/L</p>
                            </div>
                            <div className="mt-4 sm:mt-0">
                                <p className={`text-xl font-bold ${portfolio?.summary.total_gain_loss_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                    {portfolio?.summary.total_gain_loss_pct >= 0 ? '+' : ''}
                                    {portfolio?.summary.total_gain_loss_pct?.toFixed(2) || '0.00'}%
                                </p>
                                <p className="text-sm text-slate-400 mt-1 uppercase tracking-wider">Percentage</p>
                            </div>
                        </div>
                    </div>
                    <div className="absolute top-0 right-0 p-8 text-blue-500/10 -rotate-12 select-none pointer-events-none">
                        <TrendingUp size={160} />
                    </div>
                </div>

                <div className="card flex flex-col justify-center items-center text-center p-8 border-dashed border-2 border-slate-700">
                    <p className="text-slate-400 mb-4">Start managing and tracking your investments today.</p>
                    <Link to="/portfolio" className="btn-primary w-full">View My Portfolio</Link>
                </div>
            </div>

            {/* Watchlist section */}
            <section>
                {/* Section header */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                    {/* Tab pills */}
                    <div className="flex items-center gap-1 bg-slate-800/70 border border-slate-700/60 rounded-xl p-1">
                        {WATCHLIST_TABS.map(tab => (
                            <button
                                key={tab.id}
                                onClick={() => { setActiveTab(tab.id); setSortBy('default'); }}
                                className={`flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-sm font-semibold transition-all ${
                                    activeTab === tab.id
                                        ? 'bg-blue-600 text-white shadow'
                                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'
                                }`}
                            >
                                <List className="h-3.5 w-3.5" />
                                {tab.label}
                                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full ${
                                    activeTab === tab.id ? 'bg-white/20 text-white' : 'bg-slate-700 text-slate-400'
                                }`}>
                                    {(watchlists[tab.id] ?? []).length}
                                </span>
                            </button>
                        ))}
                    </div>

                    {/* Right controls */}
                    <div className="flex items-center gap-4">
                        <SortDropdown value={sortBy} onChange={setSortBy} />
                        <button
                            onClick={() => setShowCustomize(true)}
                            className="text-sm text-blue-400 font-medium hover:underline"
                        >
                            Customize
                        </button>
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
                        />
                    ))}
                    {watchlist.length === 0 && (
                        <button
                            onClick={() => setShowCustomize(true)}
                            className="col-span-full flex items-center justify-center gap-2 py-12 border-2 border-dashed border-slate-700 rounded-xl text-slate-500 hover:text-slate-300 hover:border-slate-500 transition-colors"
                        >
                            <Plus className="h-5 w-5" /> Add stocks to {activeTabConfig?.label}
                        </button>
                    )}
                </div>
            </section>

            {showCustomize && (
                <WatchlistModal
                    title={activeTabConfig?.label ?? 'Watchlist'}
                    watchlist={watchlist}
                    onClose={() => setShowCustomize(false)}
                    onSave={saveWatchlist}
                />
            )}
        </div>
    );
};

export default Dashboard;
