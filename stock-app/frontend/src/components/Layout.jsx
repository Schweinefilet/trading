import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, Wallet, TrendingUp } from 'lucide-react';
import StockSearch from './StockSearch';

const Layout = ({ children }) => {
    const location = useLocation();

    const navItems = [
        { path: '/',          label: 'Dashboard', icon: LayoutDashboard },
        { path: '/portfolio', label: 'Portfolio',  icon: Wallet },
    ];

    return (
        <div className="min-h-screen text-white flex flex-col">
            <header
                className="glass h-16 flex items-center px-4 md:px-8 justify-between"
                style={{
                    borderRadius: 0,
                    borderLeft: 'none',
                    borderRight: 'none',
                    borderTop: 'none',
                    position: 'sticky',
                    top: 0,
                    zIndex: 100,
                }}
            >
                <Link to="/" className="flex items-center space-x-2 font-bold text-xl text-white">
                    <TrendingUp className="h-8 w-8" style={{ color: 'var(--accent)' }} />
                    <span className="hidden sm:inline" style={{ color: 'var(--text-primary)' }}>Tradr</span>
                </Link>

                <div className="flex-1 max-w-lg mx-4">
                    <StockSearch />
                </div>

                <nav className="flex space-x-1 sm:space-x-2">
                    {navItems.map((item) => {
                        const Icon = item.icon;
                        const isActive = location.pathname === item.path;
                        return (
                            <Link
                                key={item.path}
                                to={item.path}
                                className="flex items-center space-x-2 px-3 py-2 rounded-xl transition-all"
                                style={isActive
                                    ? { background: 'rgba(255,255,255,0.15)', color: '#fff', fontWeight: 600 }
                                    : { color: 'rgba(255,255,255,0.60)' }
                                }
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

            <footer className="p-8 border-t border-white/10 text-center text-sm" style={{ color: 'var(--text-tertiary)' }}>
                Tradr Analytics &copy; {new Date().getFullYear()} &bull; Built with yfinance &amp; pandas-ta
            </footer>
        </div>
    );
};

export default Layout;
