import React, { useEffect, useRef, useState, useCallback, useMemo, forwardRef, useImperativeHandle } from 'react';
import { createChart, CrosshairMode, CandlestickSeries, BarSeries, AreaSeries, LineSeries } from 'lightweight-charts';
import { X } from 'lucide-react';
import DrawingOverlay from './DrawingOverlay';

const CHART_OPTIONS = {
    layout: {
        background: { color: '#000000' },
        textColor: '#d1d4dc',
        fontSize: 11
    },
    grid: {
        vertLines: { color: 'rgba(255,255,255,0.06)' },
        horzLines: { color: 'rgba(255,255,255,0.06)' }
    },
    crosshair: { mode: CrosshairMode.Normal },
    rightPriceScale: { borderColor: 'rgba(255,255,255,0.10)', borderVisible: true },
    timeScale: { borderColor: 'rgba(255,255,255,0.10)', borderVisible: true, timeVisible: true, secondsVisible: false },
};

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

    formatted.sort((a, b) => {
        const ta = typeof a.time === 'string' ? new Date(a.time).getTime() : a.time * 1000;
        const tb = typeof b.time === 'string' ? new Date(b.time).getTime() : b.time * 1000;
        return ta - tb;
    });

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

// Strip alpha from hex so it's safe as text/background color
const solidColor = (hex) => {
    if (!hex) return '#2962ff';
    return hex.length > 7 ? hex.slice(0, 7) : hex;
};

// ── ChartPane ─────────────────────────────────────────────────────────────────
const ChartPane = forwardRef(({
    id,
    data = [],
    type,
    title,
    color = '#2962ff',     // line/area series color
    height,
    isLast,
    syncTimeAxis,
    overlays = [],
    drawings = [],
    onAddDrawing,
    activeTool = 'none',
    onCrosshairMove,
    onClose,
    onRemoveOverlay,       // only passed to main price pane
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

    // ── Initialize Chart ───────────────────────────────────────────────────
    useEffect(() => {
        if (!containerRef.current) return;

        const chart = createChart(containerRef.current, {
            ...CHART_OPTIONS,
            height,
            timeScale: { ...CHART_OPTIONS.timeScale, visible: isLast },
        });
        chartRef.current = chart;

        let series;
        if (type === 'candlestick') {
            series = chart.addSeries(CandlestickSeries, {
                upColor: '#26a69a', downColor: '#ef5350',
                borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350',
            });
        } else if (type === 'hollow_candle') {
            series = chart.addSeries(CandlestickSeries, {
                upColor: '#000000',        // hollow body matches chart background
                downColor: '#ef5350',      // filled bearish body
                borderVisible: true,
                borderUpColor: '#26a69a',
                borderDownColor: '#ef5350',
                wickUpColor: '#26a69a',
                wickDownColor: '#ef5350',
            });
        } else if (type === 'bar') {
            series = chart.addSeries(BarSeries, { upColor: '#26a69a', downColor: '#ef5350' });
        } else if (type === 'area') {
            series = chart.addSeries(AreaSeries, {
                lineColor: color,
                topColor: `${solidColor(color)}33`,
                bottomColor: `${solidColor(color)}00`,
            });
        } else {
            // line — used for oscillator panes
            series = chart.addSeries(LineSeries, { color, lineWidth: 2 });
        }

        mainSeriesRef.current = series;
        setIsReady(true);

        chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
            if (syncTimeAxis) syncTimeAxis(range, id);
        });

        if (onCrosshairMove) {
            chart.subscribeCrosshairMove((param) => {
                if (!param || !param.point || !param.seriesData) {
                    onCrosshairMove(null);
                    return;
                }
                const bar = param.seriesData.get(mainSeriesRef.current);
                onCrosshairMove(bar || null);
            });
        }

        return () => {
            setIsReady(false);
            seriesRefs.current = {};
            chart.remove();
        };
    }, [id, type, isLast, color]); // color in deps so oscillator panes use the right color

    // ── Update data + overlays (with stale series cleanup) ─────────────────
    useEffect(() => {
        if (!isReady || !mainSeriesRef.current || !data) return;

        const safeData = prepareData(data);
        if (safeData.length > 0) {
            // LineSeries and AreaSeries only accept {time, value} — strip extra OHLCV fields
            const seriesData = (type === 'line' || type === 'area')
                ? safeData.map(({ time, value }) => ({ time, value }))
                : safeData;
            mainSeriesRef.current.setData(seriesData);
        }

        // Remove series for overlays no longer in the array
        const currentIds = new Set(overlays.map(o => o.id));
        Object.keys(seriesRefs.current).forEach(sid => {
            if (!currentIds.has(sid)) {
                try { chartRef.current.removeSeries(seriesRefs.current[sid]); } catch {}
                delete seriesRefs.current[sid];
            }
        });

        // Add or update overlay series
        overlays.forEach(ov => {
            if (!seriesRefs.current[ov.id]) {
                seriesRefs.current[ov.id] = chartRef.current.addSeries(LineSeries, {
                    color: solidColor(ov.color) || '#2962ff',
                    lineWidth: 1.5,
                    priceLineVisible: false,
                    lastValueVisible: true,
                });
            }
            const ovData = prepareData(ov.data);
            seriesRefs.current[ov.id].setData(ovData);
        });
    }, [data, overlays, isReady, type]);

    // ── Handle container resize ────────────────────────────────────────────
    useEffect(() => {
        if (chartRef.current && height && containerRef.current) {
            chartRef.current.resize(containerRef.current.clientWidth, height);
        }
    }, [height]);

    // ── Overlay legend chips (one per unique sourceId) ─────────────────────
    const overlayChips = useMemo(() => {
        const seen = new Set();
        return overlays.filter(ov => {
            if (!ov.label) return false;
            const key = ov.sourceId ?? ov.id;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });
    }, [overlays]);

    const isOscillator = type === 'line';

    return (
        <div className="relative w-full border-b border-white/5 bg-black flex-shrink-0" style={{ height }}>
            {/* ── Pane header ── */}
            <div className="absolute top-1.5 left-2 z-10 flex items-center gap-2 flex-wrap">
                {/* Title (with colored dot for oscillator panes) */}
                {title && (
                    <span className="text-[10px] font-bold text-slate-500 uppercase flex items-center gap-1 pointer-events-none">
                        {isOscillator && (
                            <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: color }} />
                        )}
                        {title}
                    </span>
                )}

                {/* Overlay legend chips (for main price pane overlays) */}
                {overlayChips.map(ov => (
                    <span
                        key={ov.sourceId ?? ov.id}
                        className="flex items-center gap-0.5 text-[10px] font-bold"
                    >
                        <span
                            className="w-2 h-2 rounded-full flex-shrink-0"
                            style={{ background: solidColor(ov.color) }}
                        />
                        <span className="text-slate-300">{ov.label}</span>
                        {onRemoveOverlay && (
                            <button
                                onClick={() => onRemoveOverlay(ov.sourceId ?? ov.id)}
                                className="ml-0.5 text-slate-600 hover:text-rose-400 transition-colors leading-none"
                                title={`Remove ${ov.label}`}
                            >
                                <X className="w-2.5 h-2.5" />
                            </button>
                        )}
                    </span>
                ))}

                {/* Sub-indicator chips for oscillator panes (no remove button) */}
                {isOscillator && overlays.filter(ov => ov.label).map(ov => (
                    <span key={ov.id} className="flex items-center gap-0.5 text-[10px] font-bold pointer-events-none">
                        <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: solidColor(ov.color) }} />
                        <span className="text-slate-400">{ov.label}</span>
                    </span>
                ))}

                {/* Close button for oscillator panes */}
                {onClose && (
                    <button
                        onClick={onClose}
                        className="text-slate-600 hover:text-slate-300 transition-colors leading-none"
                        title="Hide indicator"
                    >
                        <X className="w-3 h-3" />
                    </button>
                )}
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

// ── ChartContainer ────────────────────────────────────────────────────────────
const ChartContainer = ({
    data,
    chartType = 'candlestick',
    panes = [],
    overlays = [],
    drawings = [],
    onAddDrawing,
    activeTool = 'none',
    onCrosshairMove,
    onRemovePane,
    onRemoveOverlay,
}) => {
    const paneRefs = useRef({});
    const isSyncing = useRef(false);
    const outerRef = useRef(null);
    const [paneHeights, setPaneHeights] = useState({});
    const [containerHeight, setContainerHeight] = useState(600);

    // Measure the actual container height so the main pane fills exactly the
    // remaining space — no gap, no overflow, regardless of layout changes.
    useEffect(() => {
        if (!outerRef.current) return;
        const ro = new ResizeObserver(entries => {
            if (entries[0]) setContainerHeight(entries[0].contentRect.height);
        });
        ro.observe(outerRef.current);
        return () => ro.disconnect();
    }, []);

    const getPaneHeight = (id) => paneHeights[id] ?? 150;
    const totalOscHeight = panes.reduce((sum, p) => sum + getPaneHeight(p.id), 0);
    const dragHandleTotal = panes.length * 4; // each h-1 drag handle = 4px
    const mainHeight = Math.max(200, containerHeight - totalOscHeight - dragHandleTotal);

    const handleDragStart = useCallback((e, paneId) => {
        e.preventDefault();
        const startY = e.clientY;
        const startHeight = paneHeights[paneId] ?? 150;
        const onMove = (ev) => {
            const delta = ev.clientY - startY;
            setPaneHeights(prev => ({ ...prev, [paneId]: Math.max(80, startHeight + delta) }));
        };
        const onUp = () => {
            window.removeEventListener('mousemove', onMove);
            window.removeEventListener('mouseup', onUp);
        };
        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', onUp);
    }, [paneHeights]);

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
        <div ref={outerRef} className="flex flex-col w-full h-full bg-black overflow-hidden">
            {/* ── Main price pane ── */}
            <ChartPane
                ref={el => paneRefs.current['main'] = el}
                id="main"
                title="Price"
                data={data}
                type={chartType}
                height={mainHeight}
                isLast={panes.length === 0}
                syncTimeAxis={syncTimeAxis}
                overlays={overlays}
                drawings={drawings}
                onAddDrawing={onAddDrawing}
                activeTool={activeTool}
                onCrosshairMove={onCrosshairMove}
                onRemoveOverlay={onRemoveOverlay}
            />

            {/* ── Oscillator panes ── */}
            {panes.map((pane, index) => (
                <React.Fragment key={pane.id}>
                    {/* Drag handle */}
                    <div
                        className="w-full h-1 bg-white/8 hover:bg-white/20 cursor-ns-resize flex-shrink-0 transition-colors"
                        onMouseDown={(e) => handleDragStart(e, pane.id)}
                        title="Drag to resize"
                    />
                    <ChartPane
                        ref={el => paneRefs.current[pane.id] = el}
                        id={pane.id}
                        title={pane.title}
                        data={pane.data}
                        type="line"
                        color={pane.color ?? '#2962ff'}
                        height={getPaneHeight(pane.id)}
                        isLast={index === panes.length - 1}
                        syncTimeAxis={syncTimeAxis}
                        overlays={pane.overlays ?? []}
                        onClose={onRemovePane ? () => onRemovePane(pane.id) : undefined}
                    />
                </React.Fragment>
            ))}
        </div>
    );
};

export default ChartContainer;
