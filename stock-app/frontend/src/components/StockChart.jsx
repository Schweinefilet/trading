import React, { useMemo } from 'react';
import ChartContainer from './ChartContainer';

const StockChart = ({ data, indicatorData, activeIndicators = [], chartType = 'candlestick' }) => {
    console.log("[StockChart] Received data updates", { dataCount: data?.length, indicatorCount: indicatorData?.length });
    const { panes, mainOverlays } = useMemo(() => {
        if (!Array.isArray(indicatorData) || indicatorData.length === 0) return { panes: [], mainOverlays: [] };

        const overlays = [];
        const oscillatorPanes = [];

        const CATEGORIES = {
            sma: { type: 'overlay', color: '#2962ff' },
            ema: { type: 'overlay', color: '#ff9800' },
            rsi: { type: 'oscillator', label: 'RSI (14)', color: '#7e57c2' },
            macd: { type: 'oscillator', label: 'MACD (12, 26, 9)', color: '#26a69a' },
            bbands: { type: 'overlay', color: 'rgba(41, 98, 255, 0.1)' },
            stoch: { type: 'oscillator', label: 'Stochastic', color: '#ff5252' },
            atr: { type: 'oscillator', label: 'ATR (14)', color: '#ffa726' },
            obv: { type: 'oscillator', label: 'OBV', color: '#2962ff' },
            vwap: { type: 'overlay', color: '#6200ea' },
        };

        activeIndicators.forEach(id => {
            const baseId = id.split('_')[0];
            const settings = CATEGORIES[baseId] || { type: 'overlay', color: '#2962ff' };

            const seriesData = indicatorData.map(d => {
                if (!d) return null;
                const timeValue = d.Date || d.time;
                if (!timeValue) return null;

                return {
                    time: timeValue,
                    value: d[id] !== undefined ? d[id] : d[Object.keys(d).find(k => k.toLowerCase().startsWith(id.toLowerCase()))]
                };
            }).filter(d => d && d.value !== null && d.value !== undefined);

            if (seriesData.length === 0) return;

            if (settings.type === 'overlay') {
                overlays.push({ id, data: seriesData, color: settings.color });
            } else {
                oscillatorPanes.push({
                    id,
                    title: settings.label || id.toUpperCase(),
                    data: seriesData,
                    color: settings.color
                });
            }
        });

        return { panes: oscillatorPanes, mainOverlays: overlays };
    }, [indicatorData, activeIndicators]);

    if (!data || data.length === 0) {
        return (
            <div className="w-full h-full flex items-center justify-center bg-[#131722] text-slate-500 border border-[#2a2e39] rounded-lg min-h-[400px]">
                <p>No data available for the selected period</p>
            </div>
        );
    }

    return (
        <div className="w-full h-full">
            <ChartContainer
                data={data}
                chartType={chartType}
                panes={panes}
                overlays={mainOverlays}
            />
        </div>
    );
};

export default StockChart;
