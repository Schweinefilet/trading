import React, { useState } from 'react';
import axios from 'axios';

const PortfolioManager = ({ ticker, onAdded }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [shares, setShares] = useState(0);
    const [cost, setCost] = useState(0);
    const [loading, setLoading] = useState(false);

    const handleAdd = async () => {
        setLoading(true);
        try {
            await axios.post('/api/portfolio/add', {
                ticker,
                shares: parseFloat(shares),
                avg_cost: parseFloat(cost)
            });
            setIsOpen(false);
            if (onAdded) onAdded();
        } catch (error) {
            console.error("Error adding to portfolio:", error);
            alert("Failed to add position");
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <button
                onClick={() => setIsOpen(true)}
                className="btn-primary"
            >
                Add to Portfolio
            </button>

            {isOpen && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
                    <div className="card w-full max-w-sm">
                        <h3 className="text-xl font-bold mb-4">Add {ticker} to Portfolio</h3>
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm text-slate-400 mb-1">Shares</label>
                                <input
                                    type="number"
                                    className="w-full bg-slate-700 border border-slate-600 rounded p-2 text-white"
                                    value={shares}
                                    onChange={(e) => setShares(e.target.value)}
                                />
                            </div>
                            <div>
                                <label className="block text-sm text-slate-400 mb-1">Avg Cost Basis</label>
                                <input
                                    type="number"
                                    className="w-full bg-slate-700 border border-slate-600 rounded p-2 text-white"
                                    value={cost}
                                    onChange={(e) => setCost(e.target.value)}
                                />
                            </div>
                            <div className="flex space-x-3 pt-4">
                                <button
                                    disabled={loading}
                                    onClick={() => setIsOpen(false)}
                                    className="flex-1 py-2 rounded bg-slate-700 hover:bg-slate-600 text-white transition-colors"
                                >
                                    Cancel
                                </button>
                                <button
                                    disabled={loading}
                                    onClick={handleAdd}
                                    className="flex-1 py-2 rounded bg-blue-600 hover:bg-blue-500 text-white transition-colors"
                                >
                                    {loading ? 'Adding...' : 'Add Position'}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
};

export default PortfolioManager;
