import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, Wallet, TrendingUp } from 'lucide-react';
import StockSearch from './StockSearch';

const Layout = ({ children }) => {
    const location = useLocation();

    const navItems = [
        { path: '/', label: 'Dashboard', icon: LayoutDashboard },
        { path: '/portfolio', label: 'Portfolio', icon: Wallet },
    ];

    return (
        <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col">
            <header className="bg-slate-800 border-b border-slate-700 h-16 flex items-center px-4 md:px-8 justify-between sticky top-0 z-40">
                <Link to="/" className="flex items-center space-x-2 text-blue-500 font-bold text-xl">
                    <TrendingUp className="h-8 w-8" />
                    <span className="hidden sm:inline">Tradr</span>
                </Link>

                <div className="flex-1 max-w-lg mx-4">
                    <StockSearch />
                </div>

                <nav className="flex space-x-1 sm:space-x-4">
                    {navItems.map((item) => {
                        const Icon = item.icon;
                        const isActive = location.pathname === item.path;
                        return (
                            <Link
                                key={item.path}
                                to={item.path}
                                className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors ${isActive
                                        ? 'bg-slate-700 text-blue-400 font-medium'
                                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'
                                    }`}
                            >
                                <Icon className="h-5 w-5" />
                                <span className="hidden md:inline">{item.label}</span>
                            </Link>
                        );
                    })}
                </nav>
            </header>

            <main className="flex-1 p-4 md:p-8 max-w-7xl mx-auto w-full">
                {children}
            </main>

            <footer className="p-8 border-t border-slate-800 text-slate-500 text-center text-sm">
                Tradr Analytics &copy; {new Date().getFullYear()} &bull; Built with yfinance & pandas-ta
            </footer>
        </div>
    );
};

export default Layout;
