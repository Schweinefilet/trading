import React, { useEffect, useRef, useState, useCallback, forwardRef, useImperativeHandle } from 'react';
import { createChart, CrosshairMode } from 'lightweight-charts';
import DrawingOverlay from './DrawingOverlay';

// Move options out of the component to prevent re-render cycles
const CHART_OPTIONS = {
    layout: {
        background: { color: '#131722' },
        textColor: '#d1d4dc',
        fontSize: 11
    },
    grid: {
        vertLines: { color: '#1e222d' },
        horzLines: { color: '#1e222d' }
    },
    crosshair: { mode: CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#2a2e39', borderVisible: true },
    timeScale: { borderColor: '#2a2e39', borderVisible: true, timeVisible: true, secondsVisible: false },
};

/**
 * Strict data preparation for Lightweight Charts.
 * Ensures unique, sorted timestamps and correct data types.
 */
const prepareData = (rows) => {
    if (!Array.isArray(rows) || rows.length === 0) return [];

    const formatted = rows.map(d => {
        if (!d) return null;
        const t = d.time || d.Date;
        if (!t) return null;

        let timeValue = t;
        if (typeof t === 'string' && (t.includes(':') || t.includes('T') || t.includes(' '))) {
            const date = new Date(t);
            if (!isNaN(date.getTime())) {
                timeValue = Math.floor(date.getTime() / 1000);
            }
        }

        return {
            time: timeValue,
            open: Number(d.open ?? d.Open ?? 0),
            high: Number(d.high ?? d.High ?? 0),
            low: Number(d.low ?? d.Low ?? 0),
            close: Number(d.close ?? d.Close ?? 0),
            value: Number(d.value ?? d.Close ?? d.Volume ?? 0),
        };
    }).filter(d => d !== null && d.time !== undefined);

    // Sort chronologically
    formatted.sort((a, b) => {
        const ta = typeof a.time === 'string' ? new Date(a.time).getTime() : a.time * 1000;
        const tb = typeof b.time === 'string' ? new Date(b.time).getTime() : b.time * 1000;
        return ta - tb;
    });

    // Deduplicate
    const unique = [];
    const seen = new Set();
    for (const row of formatted) {
        if (!seen.has(row.time)) {
            unique.push(row);
            seen.add(row.time);
        }
    }

    return unique;
};

const ChartPane = forwardRef(({
    id,
    data = [],
    type,
    title,
    height,
    isLast,
    syncTimeAxis,
    overlays = [],
    drawings = [],
    onAddDrawing,
    activeTool = 'none'
}, ref) => {
    const containerRef = useRef(null);
    const chartRef = useRef(null);
    const mainSeriesRef = useRef(null);
    const seriesRefs = useRef({});
    const [isReady, setIsReady] = useState(false);

    useImperativeHandle(ref, () => ({
        getChart: () => chartRef.current,
        getSeries: () => mainSeriesRef.current,
    }));

    // Initialize Chart
    useEffect(() => {
        if (!containerRef.current) return;

        console.log(`[ChartContainer] Initializing pane: ${id}`);
        const chart = createChart(containerRef.current, {
            ...CHART_OPTIONS,
            height: height,
            timeScale: {
                ...CHART_OPTIONS.timeScale,
                visible: isLast,
            },
        });
        chartRef.current = chart;

        let series;
        if (type === 'candlestick') {
            series = chart.addCandlestickSeries({
                upColor: '#26a69a', downColor: '#ef5350',
                borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350',
            });
        } else if (type === 'bar') {
            series = chart.addBarSeries({ upColor: '#26a69a', downColor: '#ef5350' });
        } else if (type === 'area') {
            series = chart.addAreaSeries({ lineColor: '#2962ff', topColor: '#2962ff33', bottomColor: '#2962ff00' });
        } else {
            series = chart.addLineSeries({ color: '#2962ff', lineWidth: 2 });
        }

        mainSeriesRef.current = series;
        setIsReady(true);

        const handleTimeChange = (range) => {
            if (syncTimeAxis) syncTimeAxis(range, id);
        };
        chart.timeScale().subscribeVisibleLogicalRangeChange(handleTimeChange);

        return () => {
            console.log(`[ChartContainer] Cleaning up pane: ${id}`);
            setIsReady(false);
            chart.remove();
        };
    }, [id, type, isLast]); // Intentional dependency list to prevent re-creation loops

    // Update Data
    useEffect(() => {
        if (!isReady || !mainSeriesRef.current || !data) return;

        const safeData = prepareData(data);
        console.log(`[ChartContainer] Setting data for ${id}, count: ${safeData.length}`);

        if (safeData.length > 0) {
            mainSeriesRef.current.setData(safeData);
        }

        // Overlays
        overlays.forEach(ov => {
            if (!seriesRefs.current[ov.id]) {
                seriesRefs.current[ov.id] = chartRef.current.addLineSeries({
                    color: ov.color || '#2962ff',
                    lineWidth: 1.5,
                    priceLineVisible: false,
                });
            }
            const ovData = prepareData(ov.data);
            seriesRefs.current[ov.id].setData(ovData);
        });
    }, [data, overlays, isReady]);

    // Handle Resize
    useEffect(() => {
        if (chartRef.current && height && containerRef.current) {
            chartRef.current.resize(containerRef.current.clientWidth, height);
        }
    }, [height]);

    return (
        <div className="relative w-full border-b border-[#2a2e39] bg-[#131722]" style={{ height }}>
            <div className="absolute top-2 left-2 z-10 text-[10px] font-bold text-slate-500 uppercase pointer-events-none">
                {title}
            </div>
            <div ref={containerRef} className="w-full h-full" />
            {id === 'main' && isReady && (
                <DrawingOverlay
                    chart={chartRef.current}
                    mainSeries={mainSeriesRef.current}
                    drawings={drawings}
                    onAddDrawing={onAddDrawing}
                    tool={activeTool}
                />
            )}
        </div>
    );
});

const ChartContainer = ({
    data,
    chartType = 'candlestick',
    panes = [],
    overlays = [],
    drawings = [],
    onAddDrawing,
    activeTool = 'none'
}) => {
    const paneRefs = useRef({});
    const isSyncing = useRef(false);

    const syncTimeAxis = useCallback((range, sourceId) => {
        if (isSyncing.current || !range) return;
        isSyncing.current = true;
        Object.keys(paneRefs.current).forEach(id => {
            if (id !== sourceId && paneRefs.current[id]) {
                const chart = paneRefs.current[id].getChart();
                if (chart) chart.timeScale().setVisibleLogicalRange(range);
            }
        });
        isSyncing.current = false;
    }, []);

    return (
        <div className="flex flex-col w-full h-full bg-[#131722] overflow-hidden">
            <ChartPane
                ref={el => paneRefs.current['main'] = el}
                id="main"
                title="Price"
                data={data}
                type={chartType}
                height={window.innerHeight - 150 - (panes.length * 150)}
                isLast={panes.length === 0}
                syncTimeAxis={syncTimeAxis}
                overlays={overlays}
                drawings={drawings}
                onAddDrawing={onAddDrawing}
                activeTool={activeTool}
            />

            {panes.map((pane, index) => (
                <ChartPane
                    key={pane.id}
                    ref={el => paneRefs.current[pane.id] = el}
                    id={pane.id}
                    title={pane.title}
                    data={pane.data}
                    type="line"
                    height={150}
                    isLast={index === panes.length - 1}
                    syncTimeAxis={syncTimeAxis}
                />
            ))}
        </div>
    );
};

export default ChartContainer;
