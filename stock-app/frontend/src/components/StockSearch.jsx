import React, { useState, useEffect, useRef } from 'react';
import { Search, X } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

// Common popular symbols for quick lookup
const POPULAR = [
    { ticker: 'AAPL', name: 'Apple Inc.' },
    { ticker: 'MSFT', name: 'Microsoft Corp.' },
    { ticker: 'GOOGL', name: 'Alphabet Inc.' },
    { ticker: 'AMZN', name: 'Amazon.com Inc.' },
    { ticker: 'TSLA', name: 'Tesla Inc.' },
    { ticker: 'NVDA', name: 'NVIDIA Corp.' },
    { ticker: 'META', name: 'Meta Platforms Inc.' },
    { ticker: 'BRK-B', name: 'Berkshire Hathaway' },
    { ticker: 'JPM', name: 'JPMorgan Chase' },
    { ticker: 'V', name: 'Visa Inc.' },
    { ticker: 'UNH', name: 'UnitedHealth Group' },
    { ticker: 'XOM', name: 'Exxon Mobil Corp.' },
    { ticker: 'LLY', name: 'Eli Lilly & Co.' },
    { ticker: 'JNJ', name: 'Johnson & Johnson' },
    { ticker: 'SPY', name: 'S&P 500 ETF Trust' },
    { ticker: 'QQQ', name: 'Nasdaq 100 ETF' },
    { ticker: 'AMD', name: 'Advanced Micro Devices' },
    { ticker: 'COINBASE', name: 'Coinbase Global' },
    { ticker: 'COIN', name: 'Coinbase Global Inc.' },
    { ticker: 'PLTR', name: 'Palantir Technologies' },
];

const StockSearch = () => {
    const [query, setQuery] = useState('');
    const [suggestions, setSuggestions] = useState([]);
    const [showDropdown, setShowDropdown] = useState(false);
    const [activeIndex, setActiveIndex] = useState(-1);
    const inputRef = useRef(null);
    const dropdownRef = useRef(null);
    const navigate = useNavigate();

    useEffect(() => {
        if (!query.trim()) {
            setSuggestions([]);
            setShowDropdown(false);
            return;
        }

        const q = query.toUpperCase();
        const filtered = POPULAR.filter(
            s => s.ticker.startsWith(q) || s.name.toUpperCase().includes(q)
        ).slice(0, 6);
        setSuggestions(filtered);
        setShowDropdown(true);
        setActiveIndex(-1);
    }, [query]);

    // Close dropdown on outside click
    useEffect(() => {
        const handler = (e) => {
            if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
                setShowDropdown(false);
            }
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, []);

    const navigateTo = (ticker) => {
        navigate(`/stock/${ticker.toUpperCase()}`);
        setQuery('');
        setShowDropdown(false);
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (activeIndex >= 0 && suggestions[activeIndex]) {
            navigateTo(suggestions[activeIndex].ticker);
        } else if (query.trim()) {
            navigateTo(query.trim());
        }
    };

    const handleKeyDown = (e) => {
        if (!showDropdown || suggestions.length === 0) return;
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            setActiveIndex(i => Math.min(i + 1, suggestions.length - 1));
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            setActiveIndex(i => Math.max(i - 1, -1));
        } else if (e.key === 'Escape') {
            setShowDropdown(false);
        }
    };

    return (
        <div className="relative w-full max-w-md" ref={dropdownRef}>
            <form onSubmit={handleSubmit}>
                <div className="relative">
                    <input
                        ref={inputRef}
                        type="text"
                        placeholder="Search ticker or company..."
                        className="w-full bg-zinc-700/85 border border-zinc-600 rounded-full py-2 pl-10 pr-8 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:border-transparent text-white placeholder-zinc-400 text-sm transition-all"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={handleKeyDown}
                        onFocus={() => query && setShowDropdown(true)}
                        autoComplete="off"
                    />
                    <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-400 pointer-events-none" />
                    {query && (
                        <button
                            type="button"
                            onClick={() => { setQuery(''); setSuggestions([]); setShowDropdown(false); }}
                            className="absolute right-3 top-2.5 text-slate-500 hover:text-slate-300"
                        >
                            <X className="h-4 w-4" />
                        </button>
                    )}
                </div>
            </form>

            {showDropdown && suggestions.length > 0 && (
                <div className="absolute z-50 top-full mt-2 w-full bg-slate-800 border border-slate-700 rounded-xl shadow-2xl overflow-hidden">
                    {suggestions.map((s, i) => (
                        <button
                            key={s.ticker}
                            onMouseDown={() => navigateTo(s.ticker)}
                            className={`w-full flex items-center justify-between px-4 py-2.5 text-left hover:bg-slate-700 transition-colors ${i === activeIndex ? 'bg-slate-700' : ''
                                }`}
                        >
                            <div>
                                <span className="font-bold text-white text-sm">{s.ticker}</span>
                                <span className="text-slate-400 text-xs ml-2">{s.name}</span>
                            </div>
                            <span className="text-[10px] font-bold text-slate-500 uppercase border border-slate-600 px-1.5 py-0.5 rounded">
                                STOCK
                            </span>
                        </button>
                    ))}
                    {!suggestions.some(s => s.ticker === query.toUpperCase()) && (
                        <button
                            onMouseDown={() => navigateTo(query)}
                            className="w-full flex items-center space-x-2 px-4 py-2.5 text-left hover:bg-blue-600/20 border-t border-slate-700 transition-colors"
                        >
                            <Search className="h-4 w-4 text-blue-400" />
                            <span className="text-blue-400 text-sm">Search "<strong>{query.toUpperCase()}</strong>"</span>
                        </button>
                    )}
                </div>
            )}
        </div>
    );
};

export default StockSearch;
