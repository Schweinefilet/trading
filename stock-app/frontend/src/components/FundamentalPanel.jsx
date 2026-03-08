import React from 'react';

const FundamentalPanel = ({ data }) => {
    if (!data) return null;

    const formatValue = (val, type) => {
        if (val === null || val === undefined) return 'N/A';
        if (type === 'currency') {
            return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', notation: 'compact' }).format(val);
        }
        if (type === 'percent') {
            return (val * 100).toFixed(2) + '%';
        }
        return val.toLocaleString();
    };

    const metrics = [
        { label: 'Market Cap', value: formatValue(data.market_cap, 'currency'), category: 'valuation' },
        { label: 'P/E Ratio', value: data.pe_ratio?.toFixed(2), category: 'valuation' },
        { label: 'Forward P/E', value: data.forward_pe?.toFixed(2), category: 'valuation' },
        { label: 'P/S Ratio', value: data.ps_ratio?.toFixed(2), category: 'valuation' },
        { label: 'P/B Ratio', value: data.pb_ratio?.toFixed(2), category: 'valuation' },
        { label: 'EV/EBITDA', value: data.ev_ebitda?.toFixed(2), category: 'valuation' },

        { label: 'Revenue (TTM)', value: formatValue(data.revenue_ttm, 'currency'), category: 'health' },
        { label: 'Net Income (TTM)', value: formatValue(data.net_income_ttm, 'currency'), category: 'health' },
        { label: 'Profit Margin', value: formatValue(data.profit_margin, 'percent'), category: 'health' },
        { label: 'ROE', value: formatValue(data.roe, 'percent'), category: 'health' },
        { label: 'Debt/Equity', value: data.debt_to_equity?.toFixed(2), category: 'health' },
        { label: 'Current Ratio', value: data.current_ratio?.toFixed(2), category: 'health' },
    ];

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <div className="card">
                <h3 className="text-lg font-semibold mb-4 text-slate-300 border-b border-slate-700 pb-2">Valuation Metrics</h3>
                <div className="grid grid-cols-2 gap-4">
                    {metrics.filter(m => m.category === 'valuation').map(m => (
                        <div key={m.label}>
                            <p className="text-xs text-slate-500 uppercase">{m.label}</p>
                            <p className="text-lg font-medium text-slate-100">{m.value}</p>
                        </div>
                    ))}
                </div>
            </div>
            <div className="card">
                <h3 className="text-lg font-semibold mb-4 text-slate-300 border-b border-slate-700 pb-2">Financial Health</h3>
                <div className="grid grid-cols-2 gap-4">
                    {metrics.filter(m => m.category === 'health').map(m => (
                        <div key={m.label}>
                            <p className="text-xs text-slate-500 uppercase">{m.label}</p>
                            <p className="text-lg font-medium text-slate-100">{m.value}</p>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default FundamentalPanel;
