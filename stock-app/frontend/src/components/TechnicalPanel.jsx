import React from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    BarChart,
    Bar
} from 'recharts';

const IndicatorChart = ({ data, dataKey, color, label, height = 150 }) => (
    <div className="mt-4">
        <p className="text-sm font-medium text-slate-400 mb-2">{label}</p>
        <div style={{ height }} className="w-full">
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                    <XAxis dataKey="Date" hide />
                    <YAxis stroke="#94a3b8" tick={{ fontSize: 10 }} orientation="right" domain={[0, 100]} />
                    <Tooltip
                        contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155' }}
                        itemStyle={{ color }}
                    />
                    <Line type="monotone" dataKey={dataKey} stroke={color} dot={false} strokeWidth={1.5} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    </div>
);

const TechnicalPanel = ({ data }) => {
    if (!data || data.length === 0) return null;

    return (
        <div className="card mt-6">
            <h3 className="text-lg font-semibold mb-2 text-slate-300 border-b border-slate-700 pb-2">Technical Indicators</h3>
            <div className="space-y-6">
                <IndicatorChart
                    data={data}
                    dataKey="RSI_14"
                    color="#a78bfa"
                    label="RSI (14)"
                />
                <div className="mt-4">
                    <p className="text-sm font-medium text-slate-400 mb-2">MACD (12, 26, 9)</p>
                    <div className="h-[150px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={data}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                                <XAxis dataKey="Date" hide />
                                <YAxis stroke="#94a3b8" tick={{ fontSize: 10 }} orientation="right" domain={['auto', 'auto']} />
                                <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155' }} />
                                <Line type="monotone" dataKey="MACD_12_26_9" stroke="#3b82f6" dot={false} />
                                <Line type="monotone" dataKey="MACDs_12_26_9" stroke="#f472b6" dot={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
                <IndicatorChart
                    data={data}
                    dataKey="STOCHk_14_3_3"
                    color="#34d399"
                    label="Stochastic Oscillator"
                />
            </div>
        </div>
    );
};

export default TechnicalPanel;
