import React, { useState } from 'react';
import { Search, X, Star, ChevronLeft, Plus } from 'lucide-react';

const INDICATOR_CATEGORIES = {
    'Trend': [
        { id: 'sma', name: 'SMA (Simple Moving Average)', desc: 'Average price over a period', hasPeriod: true },
        { id: 'ema', name: 'EMA (Exponential Moving Average)', desc: 'Weighted average giving more weight to recent prices', hasPeriod: true },
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

const PRESET_PERIODS = [9, 20, 50, 100, 200];

const PeriodPicker = ({ indicatorBase, onBack, onSelect }) => {
    const [selectedPeriods, setSelectedPeriods] = useState([20, 50]);
    const [customInput, setCustomInput] = useState('');

    const togglePeriod = (p) => {
        setSelectedPeriods(prev =>
            prev.includes(p) ? prev.filter(x => x !== p) : [...prev, p]
        );
    };

    const addCustom = () => {
        const val = parseInt(customInput, 10);
        if (!isNaN(val) && val > 0 && !selectedPeriods.includes(val)) {
            setSelectedPeriods(prev => [...prev, val].sort((a, b) => a - b));
        }
        setCustomInput('');
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') addCustom();
    };

    const handleAddSelected = () => {
        if (selectedPeriods.length === 0) return;
        const results = selectedPeriods.map(p => `${indicatorBase}_${p}`);
        onSelect(results);
    };

    const label = indicatorBase === 'sma' ? 'SMA' : 'EMA';

    return (
        <div className="flex flex-col h-full">
            {/* Sub-header */}
            <div className="p-4 border-b border-[#2a2e39] flex items-center gap-3">
                <button
                    onClick={onBack}
                    className="p-1 hover:bg-[#2a2e39] rounded text-slate-400 transition-colors"
                >
                    <ChevronLeft className="w-5 h-5" />
                </button>
                <div>
                    <div className="text-sm font-bold text-white">{label} Period Picker</div>
                    <div className="text-[11px] text-slate-500">Select one or more periods</div>
                </div>
            </div>

            <div className="flex-grow overflow-y-auto p-4 space-y-4">
                {/* Preset buttons */}
                <div>
                    <div className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-2">Preset Periods</div>
                    <div className="flex flex-wrap gap-2">
                        {PRESET_PERIODS.map(p => (
                            <button
                                key={p}
                                onClick={() => togglePeriod(p)}
                                className={`px-4 py-2 rounded font-bold text-sm transition-colors ${
                                    selectedPeriods.includes(p)
                                        ? 'bg-[#2962ff] text-white'
                                        : 'bg-[#2a2e39] text-slate-300 hover:bg-[#363b4a]'
                                }`}
                            >
                                {p}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Custom period input */}
                <div>
                    <div className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-2">Custom Period</div>
                    <div className="flex gap-2">
                        <input
                            type="number"
                            min="1"
                            placeholder="e.g. 15"
                            value={customInput}
                            onChange={(e) => setCustomInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            className="flex-grow bg-[#2a2e39] border border-transparent focus:border-[#2962ff] rounded px-3 py-2 text-sm outline-none text-white"
                        />
                        <button
                            onClick={addCustom}
                            className="px-3 py-2 bg-[#2a2e39] hover:bg-[#363b4a] rounded text-slate-300 transition-colors"
                        >
                            <Plus className="w-4 h-4" />
                        </button>
                    </div>
                </div>

                {/* Selected periods preview */}
                {selectedPeriods.length > 0 && (
                    <div>
                        <div className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-2">Selected</div>
                        <div className="flex flex-wrap gap-1.5">
                            {selectedPeriods.map(p => (
                                <div
                                    key={p}
                                    className="flex items-center gap-1 px-2.5 py-1 bg-[#1a2a4a] border border-blue-800 rounded text-xs text-blue-300"
                                >
                                    <span>{label} {p}</span>
                                    <button
                                        onClick={() => togglePeriod(p)}
                                        className="text-blue-500 hover:text-white ml-0.5"
                                    >
                                        <X className="w-3 h-3" />
                                    </button>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>

            {/* Add Selected button */}
            <div className="p-4 border-t border-[#2a2e39] flex-shrink-0">
                <button
                    onClick={handleAddSelected}
                    disabled={selectedPeriods.length === 0}
                    className="w-full py-2.5 bg-[#2962ff] hover:bg-[#1e4bd8] disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold text-sm rounded transition-colors"
                >
                    Add Selected ({selectedPeriods.length})
                </button>
            </div>
        </div>
    );
};

const IndicatorModal = ({ onClose, onSelect }) => {
    const [search, setSearch] = useState('');
    const [activeCategory, setActiveCategory] = useState('Trend');
    const [periodPicker, setPeriodPicker] = useState(null); // 'sma' | 'ema' | null

    const handleIndicatorClick = (indicator) => {
        if (indicator.hasPeriod) {
            setPeriodPicker(indicator.id);
        } else {
            onSelect(indicator.id);
        }
    };

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
                <div className="p-4 border-b border-[#2a2e39] flex items-center justify-between flex-shrink-0">
                    <h2 className="text-lg font-bold text-white tracking-tight">Indicators & Strategies</h2>
                    <button onClick={onClose} className="p-1 hover:bg-[#2a2e39] rounded text-slate-400 transition-colors">
                        <X className="w-6 h-6" />
                    </button>
                </div>

                {periodPicker ? (
                    <PeriodPicker
                        indicatorBase={periodPicker}
                        onBack={() => setPeriodPicker(null)}
                        onSelect={(results) => {
                            onSelect(results);
                        }}
                    />
                ) : (
                    <>
                        {/* Search */}
                        <div className="p-4 bg-[#131722] border-b border-[#2a2e39] flex-shrink-0">
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
                                                onClick={() => handleIndicatorClick(indicator)}
                                                className="w-full text-left px-4 py-3 rounded hover:bg-[#2a2e39] group flex items-start gap-4 transition-colors"
                                            >
                                                <div className="flex-grow">
                                                    <div className="text-sm font-medium text-[#d1d4dc] group-hover:text-white">{indicator.name}</div>
                                                    <div className="text-[11px] text-slate-500 group-hover:text-slate-400 mt-0.5">{indicator.desc}</div>
                                                </div>
                                                {indicator.hasPeriod ? (
                                                    <span className="text-[10px] text-slate-500 group-hover:text-slate-300 mt-1 border border-[#2a2e39] rounded px-1.5 py-0.5 flex-shrink-0">periods</span>
                                                ) : (
                                                    <Star className="w-4 h-4 text-slate-600 hover:text-yellow-500 mt-1 opacity-0 group-hover:opacity-100 transition-opacity" />
                                                )}
                                            </button>
                                        ))}
                                    </div>
                                ))}
                            </main>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default IndicatorModal;
