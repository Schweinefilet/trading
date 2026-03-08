import React, { useState } from 'react';
import { Search, X, Star, ChevronRight } from 'lucide-react';

const INDICATOR_CATEGORIES = {
    'Trend': [
        { id: 'sma', name: 'SMA (Simple Moving Average)', desc: 'Average price over a period' },
        { id: 'ema', name: 'EMA (Exponential Moving Average)', desc: 'Weighted average giving more weight to recent prices' },
        { id: 'bbands', name: 'Bollinger Bands', desc: 'Volatility bands above and below a moving average' },
        { id: 'ichimoku', name: 'Ichimoku Cloud', desc: 'Comprehensive trend and momentum indicator' },
        { id: 'psar', name: 'Parabolic SAR', desc: 'Stop and reverse trend follower' },
        { id: 'supertrend', name: 'SuperTrend', desc: 'Trend following indicator based on ATR' },
    ],
    'Momentum': [
        { id: 'rsi', name: 'RSI (Relative Strength Index)', desc: 'Measures the speed and change of price movements' },
        { id: 'macd', name: 'MACD', desc: 'Trend-following momentum indicator' },
        { id: 'stoch', name: 'Stochastic Oscillator', desc: 'Compares closing price to price range' },
        { id: 'cci', name: 'CCI (Commodity Channel Index)', desc: 'Identifies cyclical trends' },
    ],
    'Volatility': [
        { id: 'atr', name: 'ATR (Average True Range)', desc: 'Measures market volatility' },
        { id: 'stdev', name: 'Standard Deviation', desc: 'Statistical measure of market volatility' },
    ],
    'Volume': [
        { id: 'obv', name: 'OBV (On Balance Volume)', desc: 'Uses volume flow to predict price changes' },
        { id: 'vwap', name: 'VWAP', desc: 'Volume Weighted Average Price' },
        { id: 'cmf', name: 'Chaikin Money Flow', desc: 'Measures Accumulation-Distribution over a period' },
    ]
};

const IndicatorModal = ({ onClose, onSelect }) => {
    const [search, setSearch] = useState('');
    const [activeCategory, setActiveCategory] = useState('Trend');

    const filteredIndicators = Object.entries(INDICATOR_CATEGORIES).reduce((acc, [cat, items]) => {
        const filtered = items.filter(i =>
            i.name.toLowerCase().includes(search.toLowerCase()) ||
            cat.toLowerCase().includes(search.toLowerCase())
        );
        if (filtered.length > 0) acc[cat] = filtered;
        return acc;
    }, {});

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
            <div className="bg-[#1e222d] w-full max-w-3xl h-[600px] rounded-lg border border-[#2a2e39] flex flex-col shadow-2xl overflow-hidden">
                {/* Header */}
                <div className="p-4 border-b border-[#2a2e39] flex items-center justify-between">
                    <h2 className="text-lg font-bold text-white tracking-tight">Indicators & Strategies</h2>
                    <button onClick={onClose} className="p-1 hover:bg-[#2a2e39] rounded text-slate-400 transition-colors">
                        <X className="w-6 h-6" />
                    </button>
                </div>

                {/* Search */}
                <div className="p-4 bg-[#131722] border-b border-[#2a2e39]">
                    <div className="relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                        <input
                            type="text"
                            placeholder="Search..."
                            autoFocus
                            className="w-full bg-[#1e222d] border border-transparent focus:border-[#2962ff] rounded-md px-10 py-2.5 text-sm outline-none transition-all"
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                        />
                    </div>
                </div>

                {/* Body */}
                <div className="flex flex-grow overflow-hidden">
                    {/* Sidebar */}
                    <aside className="w-48 border-r border-[#2a2e39] bg-[#131722] py-2 flex-shrink-0">
                        {Object.keys(INDICATOR_CATEGORIES).map(cat => (
                            <button
                                key={cat}
                                onClick={() => setActiveCategory(cat)}
                                className={`w-full text-left px-6 py-2.5 text-xs font-semibold transition-colors ${activeCategory === cat && !search ? 'bg-[#2a2e39] text-[#2962ff]' : 'text-slate-400 hover:text-slate-200'}`}
                            >
                                {cat}
                            </button>
                        ))}
                    </aside>

                    {/* List */}
                    <main className="flex-grow overflow-y-auto p-2 bg-[#1e222d]">
                        {Object.entries(search ? filteredIndicators : { [activeCategory]: INDICATOR_CATEGORIES[activeCategory] }).map(([cat, items]) => (
                            <div key={cat} className="mb-4">
                                {search && <div className="px-4 py-2 text-[10px] font-bold text-slate-500 uppercase tracking-widest">{cat}</div>}
                                {items.map(indicator => (
                                    <button
                                        key={indicator.id}
                                        onClick={() => onSelect(indicator.id)}
                                        className="w-full text-left px-4 py-3 rounded hover:bg-[#2a2e39] group flex items-start gap-4 transition-colors"
                                    >
                                        <div className="flex-grow">
                                            <div className="text-sm font-medium text-[#d1d4dc] group-hover:text-white">{indicator.name}</div>
                                            <div className="text-[11px] text-slate-500 group-hover:text-slate-400 mt-0.5">{indicator.desc}</div>
                                        </div>
                                        <Star className="w-4 h-4 text-slate-600 hover:text-yellow-500 mt-1 opacity-0 group-hover:opacity-100 transition-opacity" />
                                    </button>
                                ))}
                            </div>
                        ))}
                    </main>
                </div>
            </div>
        </div>
    );
};

export default IndicatorModal;
