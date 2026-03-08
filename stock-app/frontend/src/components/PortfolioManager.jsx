import React, { useState } from 'react';
import axios from 'axios';
import { X } from 'lucide-react';

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
                className="font-medium px-4 py-2 transition-colors"
                style={{ background: 'var(--accent)', color: '#000', borderRadius: 'var(--radius-btn)' }}
            >
                Add to Portfolio
            </button>

            {isOpen && (
                <div
                    className="fixed inset-0 flex items-center justify-center z-50 p-4"
                    style={{ background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)' }}
                >
                    <div className="glass w-full max-w-sm p-6" style={{ borderRadius: '20px' }}>
                        <div className="flex items-center justify-between mb-5">
                            <h3 className="text-xl font-bold" style={{ color: 'var(--text-primary)' }}>
                                Add {ticker} to Portfolio
                            </h3>
                            <button
                                onClick={() => setIsOpen(false)}
                                className="transition-colors"
                                style={{ color: 'var(--text-tertiary)' }}
                                onMouseEnter={e => e.currentTarget.style.color = 'var(--text-primary)'}
                                onMouseLeave={e => e.currentTarget.style.color = 'var(--text-tertiary)'}
                            >
                                <X className="h-5 w-5" />
                            </button>
                        </div>
                        <div className="space-y-4">
                            <div>
                                <label className="block text-xs mb-1 uppercase font-bold" style={{ color: 'var(--text-secondary)' }}>
                                    Shares
                                </label>
                                <input
                                    type="number"
                                    className="glass-input"
                                    value={shares}
                                    onChange={(e) => setShares(e.target.value)}
                                />
                            </div>
                            <div>
                                <label className="block text-xs mb-1 uppercase font-bold" style={{ color: 'var(--text-secondary)' }}>
                                    Avg Cost Basis
                                </label>
                                <input
                                    type="number"
                                    className="glass-input"
                                    value={cost}
                                    onChange={(e) => setCost(e.target.value)}
                                />
                            </div>
                            <div className="flex space-x-3 pt-2">
                                <button
                                    disabled={loading}
                                    onClick={() => setIsOpen(false)}
                                    className="flex-1 py-2.5 rounded-xl font-medium transition-colors disabled:opacity-50"
                                    style={{
                                        background: 'rgba(255,69,58,0.12)',
                                        color: 'var(--negative)',
                                        border: '1px solid rgba(255,69,58,0.25)',
                                    }}
                                >
                                    Cancel
                                </button>
                                <button
                                    disabled={loading}
                                    onClick={handleAdd}
                                    className="flex-1 py-2.5 font-bold transition-colors disabled:opacity-50"
                                    style={{ background: 'var(--accent)', color: '#000', borderRadius: 'var(--radius-btn)' }}
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
