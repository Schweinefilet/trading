import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { TrendingUp, TrendingDown, Clock } from 'lucide-react';
import { Link } from 'react-router-dom';

const WatchlistCard = ({ ticker }) => {
    const [quote, setQuote] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchQuote = async () => {
            try {
                const res = await axios.get(`/api/stock/${ticker}/quote`);
                setQuote(res.data);
            } catch (e) {
                console.error(e);
            } finally {
                setLoading(false);
            }
        };
        fetchQuote();
    }, [ticker]);

    if (loading) return <div className="card animate-pulse h-28"></div>;
    if (!quote) return null;

    const isPositive = quote.change >= 0;

    return (
        <Link to={`/stock/${ticker}`} className="card hover:border-slate-500 transition-colors flex flex-col justify-between group">
            <div className="flex justify-between items-start">
                <div>
                    <h3 className="font-bold text-lg text-white group-hover:text-blue-400 transition-colors uppercase">{ticker}</h3>
                    <p className="text-xs text-slate-500 truncate max-w-[120px]">{quote.longName}</p>
                </div>
                <div className={`px-2 py-1 rounded text-xs font-bold ${isPositive ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                    {isPositive ? '+' : ''}{quote.percent_change?.toFixed(2)}%
                </div>
            </div>
            <div className="mt-4 flex justify-between items-end">
                <span className="text-2xl font-bold">${quote.current_price?.toFixed(2)}</span>
                <div className={`flex items-center ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {isPositive ? <TrendingUp className="h-4 w-4 mr-1" /> : <TrendingDown className="h-4 w-4 mr-1" />}
                    <span className="text-sm font-medium">{quote.change?.toFixed(2)}</span>
                </div>
            </div>
        </Link>
    );
};

const Dashboard = () => {
    const [portfolio, setPortfolio] = useState(null);
    const [watchlist, setWatchlist] = useState(['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN']);

    useEffect(() => {
        const fetchPortfolio = async () => {
            try {
                const res = await axios.get('/api/portfolio/analytics');
                setPortfolio(res.data);
            } catch (e) {
                console.error(e);
            }
        };
        fetchPortfolio();
    }, []);

    return (
        <div className="space-y-8">
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
                    <p className="text-slate-400 mb-4">Start managing your and tracking your investments today.</p>
                    <Link to="/portfolio" className="btn-primary w-full">View My Portfolio</Link>
                </div>
            </div>

            <section>
                <div className="flex justify-between items-end mb-4">
                    <h2 className="text-xl font-bold">Watchlist</h2>
                    <button className="text-sm text-blue-400 font-medium hover:underline">Customize</button>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                    {watchlist.map(ticker => (
                        <WatchlistCard key={ticker} ticker={ticker} />
                    ))}
                </div>
            </section>
        </div>
    );
};

export default Dashboard;
