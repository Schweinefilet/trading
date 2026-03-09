import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { X, TrendingUp, TrendingDown, BarChart2, ExternalLink, Loader } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const StockIndicatorsModal = ({ ticker, onClose }) => {
    const navigate = useNavigate();
    const [loading, setLoading] = useState(true);
    const [quote, setQuote] = useState(null);
    const [indicators, setIndicators] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!ticker) return;
        
        const fetchData = async () => {
            setLoading(true);
            setError(null);
            
            try {
                // Fetch quote and indicators in parallel
                const [quoteRes, indicatorsRes] = await Promise.all([
                    axios.get(`/api/stock/${ticker}/quote`),
                    axios.get(`/api/stock/${ticker}/indicators`, {
                        params: {
                            period: '1y',
                            interval: '1d',
                            indicators: 'rsi_14,macd,sma_20,sma_50,sma_200,ema_12,ema_26,bbands'
                        }
                    })
                ]);
                
                setQuote(quoteRes.data);
                
                // Get the most recent indicators
                const latestIndicators = indicatorsRes.data && indicatorsRes.data.length > 0 
                    ? indicatorsRes.data[indicatorsRes.data.length - 1] 
                    : null;
                    
                setIndicators(latestIndicators);
            } catch (err) {
                console.error('Error fetching stock data:', err);
                setError('Failed to load stock data. Please try again.');
            } finally {
                setLoading(false);
            }
        };
        
        fetchData();
    }, [ticker]);

    const handleViewFullChart = () => {
        navigate(`/stock/${ticker}`);
        onClose();
    };

    const isPositive = quote && quote.change >= 0;

    const formatValue = (value, decimals = 2) => {
        if (value === null || value === undefined || !Number.isFinite(value)) {
            return 'N/A';
        }
        return typeof value === 'number' ? value.toFixed(decimals) : value;
    };

    // Get the actual price - backend returns current_price
    const displayPrice = quote?.current_price;
    const displayChange = quote?.change;
    const displayChangePercent = quote?.percent_change;

    return (
        <div
            className="fixed inset-0 flex items-center justify-center z-50 p-4"
            style={{ background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)' }}
            onClick={onClose}
        >
            <div
                className="glass w-full max-w-2xl max-h-[90vh] overflow-y-auto"
                style={{ borderRadius: '20px' }}
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div className="sticky top-0 glass p-5 flex justify-between items-start gap-4" style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', zIndex: 10 }}>
                    <div className="flex-1">
                        <h2 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>
                            {ticker}
                        </h2>
                        {quote && (
                            <>
                                <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>
                                    {quote.longName || quote.shortName || ticker}
                                </p>
                                <div className="flex items-baseline gap-3 mt-2">
                                    <span className="text-3xl font-black" style={{ color: 'var(--text-primary)' }}>
                                        ${formatValue(displayPrice, 2)}
                                    </span>
                                    <span
                                        className="text-lg font-bold flex items-center gap-1"
                                        style={{ color: isPositive ? 'var(--positive)' : 'var(--negative)' }}
                                    >
                                        {isPositive ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                                        {isPositive ? '+' : ''}{formatValue(displayChange, 2)} ({isPositive ? '+' : ''}{formatValue(displayChangePercent, 2)}%)
                                    </span>
                                </div>
                            </>
                        )}
                    </div>
                    <button
                        onClick={onClose}
                        className="transition-colors p-2"
                        style={{ color: 'var(--text-tertiary)' }}
                        onMouseEnter={e => e.currentTarget.style.color = 'var(--text-primary)'}
                        onMouseLeave={e => e.currentTarget.style.color = 'var(--text-tertiary)'}
                    >
                        <X className="h-6 w-6" />
                    </button>
                </div>

                {/* Content */}
                <div className="p-5">
                    {loading ? (
                        <div className="flex flex-col items-center justify-center py-16">
                            <Loader className="h-12 w-12 animate-spin mb-4" style={{ color: 'var(--accent)' }} />
                            <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>Loading indicators...</p>
                        </div>
                    ) : error ? (
                        <div className="flex flex-col items-center justify-center py-16">
                            <p className="text-sm" style={{ color: 'var(--negative)' }}>{error}</p>
                        </div>
                    ) : (
                        <>
                            {/* Technical Indicators */}
                            <div className="mb-6">
                                <h3 className="text-xs uppercase font-bold tracking-widest mb-4" style={{ color: 'var(--text-tertiary)' }}>
                                    Technical Indicators
                                </h3>
                                
                                {indicators ? (
                                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                        {/* RSI */}
                                        {indicators.rsi_14 !== null && indicators.rsi_14 !== undefined && (
                                            <div className="glass p-4">
                                                <div className="flex justify-between items-start mb-2">
                                                    <div>
                                                        <p className="text-xs uppercase font-bold" style={{ color: 'var(--text-tertiary)' }}>RSI (14)</p>
                                                        <p className="text-sm text-gray-400 mt-0.5">Relative Strength Index</p>
                                                    </div>
                                                    <span className="text-2xl font-black" style={{ 
                                                        color: indicators.rsi_14 > 70 ? 'var(--negative)' : indicators.rsi_14 < 30 ? 'var(--positive)' : 'var(--text-primary)' 
                                                    }}>
                                                        {formatValue(indicators.rsi_14, 1)}
                                                    </span>
                                                </div>
                                                <div className="w-full h-2 rounded mt-2" style={{ background: 'rgba(255,255,255,0.1)' }}>
                                                    <div 
                                                        className="h-2 rounded transition-all" 
                                                        style={{ 
                                                            width: `${Math.min(100, Math.max(0, indicators.rsi_14))}%`,
                                                            background: indicators.rsi_14 > 70 ? 'var(--negative)' : indicators.rsi_14 < 30 ? 'var(--positive)' : 'var(--accent)'
                                                        }} 
                                                    />
                                                </div>
                                                <p className="text-xs mt-2" style={{ color: 'var(--text-tertiary)' }}>
                                                    {indicators.rsi_14 > 70 ? 'Overbought' : indicators.rsi_14 < 30 ? 'Oversold' : 'Neutral'}
                                                </p>
                                            </div>
                                        )}

                                        {/* MACD */}
                                        {(indicators.macd !== null && indicators.macd !== undefined) && (
                                            <div className="glass p-4">
                                                <div className="flex justify-between items-start mb-2">
                                                    <div>
                                                        <p className="text-xs uppercase font-bold" style={{ color: 'var(--text-tertiary)' }}>MACD</p>
                                                        <p className="text-sm text-gray-400 mt-0.5">Moving Average Convergence</p>
                                                    </div>
                                                    <span className="text-2xl font-black" style={{ 
                                                        color: indicators.macd > 0 ? 'var(--positive)' : 'var(--negative)' 
                                                    }}>
                                                        {formatValue(indicators.macd, 2)}
                                                    </span>
                                                </div>
                                                {indicators.macd_signal !== null && indicators.macd_signal !== undefined && (
                                                    <div className="flex justify-between text-xs mt-2" style={{ color: 'var(--text-secondary)' }}>
                                                        <span>Signal: {formatValue(indicators.macd_signal, 2)}</span>
                                                        {indicators.macd_hist !== null && indicators.macd_hist !== undefined && (
                                                            <span>Hist: {formatValue(indicators.macd_hist, 2)}</span>
                                                        )}
                                                    </div>
                                                )}
                                            </div>
                                        )}

                                        {/* SMA 20 */}
                                        {indicators.sma_20 !== null && indicators.sma_20 !== undefined && (
                                            <div className="glass p-4">
                                                <div className="flex justify-between items-start">
                                                    <div>
                                                        <p className="text-xs uppercase font-bold" style={{ color: 'var(--text-tertiary)' }}>SMA 20</p>
                                                        <p className="text-sm text-gray-400 mt-0.5">20-Day Simple Moving Avg</p>
                                                    </div>
                                                    <span className="text-2xl font-black" style={{ color: 'var(--text-primary)' }}>
                                                        ${formatValue(indicators.sma_20, 2)}
                                                    </span>
                                                </div>
                                                {quote && displayPrice && (
                                                    <p className="text-xs mt-2" style={{ color: 'var(--text-tertiary)' }}>
                                                        Price is {displayPrice > indicators.sma_20 ? 'above' : 'below'} SMA
                                                    </p>
                                                )}
                                            </div>
                                        )}

                                        {/* SMA 50 */}
                                        {indicators.sma_50 !== null && indicators.sma_50 !== undefined && (
                                            <div className="glass p-4">
                                                <div className="flex justify-between items-start">
                                                    <div>
                                                        <p className="text-xs uppercase font-bold" style={{ color: 'var(--text-tertiary)' }}>SMA 50</p>
                                                        <p className="text-sm text-gray-400 mt-0.5">50-Day Simple Moving Avg</p>
                                                    </div>
                                                    <span className="text-2xl font-black" style={{ color: 'var(--text-primary)' }}>
                                                        ${formatValue(indicators.sma_50, 2)}
                                                    </span>
                                                </div>
                                                {quote && displayPrice && (
                                                    <p className="text-xs mt-2" style={{ color: 'var(--text-tertiary)' }}>
                                                        Price is {displayPrice > indicators.sma_50 ? 'above' : 'below'} SMA
                                                    </p>
                                                )}
                                            </div>
                                        )}

                                        {/* SMA 200 */}
                                        {indicators.sma_200 !== null && indicators.sma_200 !== undefined && (
                                            <div className="glass p-4">
                                                <div className="flex justify-between items-start">
                                                    <div>
                                                        <p className="text-xs uppercase font-bold" style={{ color: 'var(--text-tertiary)' }}>SMA 200</p>
                                                        <p className="text-sm text-gray-400 mt-0.5">200-Day Simple Moving Avg</p>
                                                    </div>
                                                    <span className="text-2xl font-black" style={{ color: 'var(--text-primary)' }}>
                                                        ${formatValue(indicators.sma_200, 2)}
                                                    </span>
                                                </div>
                                                {quote && displayPrice && (
                                                    <p className="text-xs mt-2" style={{ color: 'var(--text-tertiary)' }}>
                                                        Price is {displayPrice > indicators.sma_200 ? 'above' : 'below'} SMA
                                                    </p>
                                                )}
                                            </div>
                                        )}

                                        {/* Bollinger Bands */}
                                        {indicators.bb_upper !== null && indicators.bb_upper !== undefined && (
                                            <div className="glass p-4">
                                                <div className="flex justify-between items-start mb-2">
                                                    <div>
                                                        <p className="text-xs uppercase font-bold" style={{ color: 'var(--text-tertiary)' }}>Bollinger Bands</p>
                                                        <p className="text-sm text-gray-400 mt-0.5">Volatility indicator</p>
                                                    </div>
                                                </div>
                                                <div className="flex justify-between text-xs" style={{ color: 'var(--text-secondary)' }}>
                                                    <div>
                                                        <p className="text-gray-400">Upper</p>
                                                        <p className="font-bold">${formatValue(indicators.bb_upper, 2)}</p>
                                                    </div>
                                                    {indicators.bb_middle !== null && indicators.bb_middle !== undefined && (
                                                        <div>
                                                            <p className="text-gray-400">Middle</p>
                                                            <p className="font-bold">${formatValue(indicators.bb_middle, 2)}</p>
                                                        </div>
                                                    )}
                                                    {indicators.bb_lower !== null && indicators.bb_lower !== undefined && (
                                                        <div>
                                                            <p className="text-gray-400">Lower</p>
                                                            <p className="font-bold">${formatValue(indicators.bb_lower, 2)}</p>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                ) : (
                                    <p className="text-sm text-center py-8" style={{ color: 'var(--text-tertiary)' }}>
                                        No indicator data available
                                    </p>
                                )}
                            </div>

                            {/* Quote Details */}
                            {quote && (
                                <div className="mb-6">
                                    <h3 className="text-xs uppercase font-bold tracking-widest mb-4" style={{ color: 'var(--text-tertiary)' }}>
                                        Quote Details
                                    </h3>
                                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                                        <div>
                                            <p className="text-xs" style={{ color: 'var(--text-tertiary)' }}>Open</p>
                                            <p className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
                                                ${formatValue(quote.open, 2)}
                                            </p>
                                        </div>
                                        <div>
                                            <p className="text-xs" style={{ color: 'var(--text-tertiary)' }}>High</p>
                                            <p className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
                                                ${formatValue(quote.high, 2)}
                                            </p>
                                        </div>
                                        <div>
                                            <p className="text-xs" style={{ color: 'var(--text-tertiary)' }}>Low</p>
                                            <p className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
                                                ${formatValue(quote.low, 2)}
                                            </p>
                                        </div>
                                        <div>
                                            <p className="text-xs" style={{ color: 'var(--text-tertiary)' }}>Volume</p>
                                            <p className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
                                                {quote.volume ? (quote.volume / 1000000).toFixed(2) + 'M' : 'N/A'}
                                            </p>
                                        </div>
                                        {quote.market_cap && (
                                            <div>
                                                <p className="text-xs" style={{ color: 'var(--text-tertiary)' }}>Market Cap</p>
                                                <p className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
                                                    ${(quote.market_cap / 1000000000).toFixed(2)}B
                                                </p>
                                            </div>
                                        )}
                                        {quote.pe_ratio && (
                                            <div>
                                                <p className="text-xs" style={{ color: 'var(--text-tertiary)' }}>P/E Ratio</p>
                                                <p className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
                                                    {formatValue(quote.pe_ratio, 2)}
                                                </p>
                                            </div>
                                        )}
                                        {quote.dividend_yield && (
                                            <div>
                                                <p className="text-xs" style={{ color: 'var(--text-tertiary)' }}>Div Yield</p>
                                                <p className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
                                                    {formatValue(quote.dividend_yield * 100, 2)}%
                                                </p>
                                            </div>
                                        )}
                                        {quote.fifty_two_week_high && (
                                            <div>
                                                <p className="text-xs" style={{ color: 'var(--text-tertiary)' }}>52W High</p>
                                                <p className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
                                                    ${formatValue(quote.fifty_two_week_high, 2)}
                                                </p>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* View Full Chart Button */}
                            <button
                                onClick={handleViewFullChart}
                                className="w-full py-3 font-bold flex items-center justify-center gap-2 transition-all"
                                style={{
                                    background: 'var(--accent)',
                                    color: '#000',
                                    borderRadius: 'var(--radius-btn)'
                                }}
                                onMouseEnter={e => e.currentTarget.style.opacity = '0.9'}
                                onMouseLeave={e => e.currentTarget.style.opacity = '1'}
                            >
                                <BarChart2 className="h-5 w-5" />
                                View Full Chart & Analysis
                                <ExternalLink className="h-4 w-4" />
                            </button>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
};

export default StockIndicatorsModal;
