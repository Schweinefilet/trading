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
        vertLines: { color: 'rgba(255,255,255,0.22)' },
        horzLines: { color: 'rgba(255,255,255,0.22)' }
    },
    crosshair: { mode: CrosshairMode.Normal },
    rightPriceScale: {
        borderColor: 'rgba(255,255,255,0.10)',
        borderVisible: true,
        minimumWidth: 58,
    },
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

// Format a unix timestamp (seconds) into a readable crosshair label
function formatCrosshairTime(time) {
    if (time === undefined || time === null) return '';
    const date = new Date(
        typeof time === 'string' ? time : time * 1000
    );
    if (isNaN(date.getTime())) return '';
    const now = new Date();
    const isToday = date.toDateString() === now.toDateString();
    const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    if (isToday) return timeStr;
    return `${date.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${timeStr}`;
}

// ── ChartPane ─────────────────────────────────────────────────────────────────
const ChartPane = forwardRef(({
    id,
    data = [],
    type,
    title,
    color = '#2962ff',
    height,
    isLast,
    syncTimeAxis,
    syncCrosshair,       // callback: (param, sourceId) => void
    showTimeLabel,       // show floating time label at bottom of pane
    overlays = [],
    drawings = [],
    onAddDrawing,
    activeTool = 'none',
    onCrosshairMove,
    onClose,
    onRemoveOverlay,
}, ref) => {
    const containerRef = useRef(null);
    const chartRef = useRef(null);
    const mainSeriesRef = useRef(null);
    const seriesRefs = useRef({});
    const [isReady, setIsReady] = useState(false);
    const [timeLabel, setTimeLabel] = useState(null); // { x: number, text: string } | null

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
                upColor: '#000000',
                downColor: '#ef5350',
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
            series = chart.addSeries(LineSeries, { color, lineWidth: 2 });
        }

        mainSeriesRef.current = series;
        setIsReady(true);

        chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
            if (syncTimeAxis) syncTimeAxis(range, id);
        });

        chart.subscribeCrosshairMove((param) => {
            // ── Time label overlay (for panes without a visible time axis) ──
            if (showTimeLabel) {
                if (param?.point && param?.time !== undefined) {
                    setTimeLabel({ x: param.point.x, text: formatCrosshairTime(param.time) });
                } else {
                    setTimeLabel(null);
                }
            }

            // ── Sync crosshair to other panes ──
            if (syncCrosshair) syncCrosshair(param, id);

            // ── OHLCV readout for parent ──
            if (onCrosshairMove) {
                if (!param || !param.point || !param.seriesData) {
                    onCrosshairMove(null);
                    return;
                }
                const bar = param.seriesData.get(mainSeriesRef.current);
                onCrosshairMove(bar || null);
            }
        });

        return () => {
            setIsReady(false);
            setTimeLabel(null);
            seriesRefs.current = {};
            chart.remove();
        };
    }, [id, type, isLast, color]); // eslint-disable-line react-hooks/exhaustive-deps

    // ── Update data + overlays ─────────────────────────────────────────────
    useEffect(() => {
        if (!isReady || !mainSeriesRef.current || !data) return;

        const safeData = prepareData(data);
        if (safeData.length > 0) {
            const seriesData = (type === 'line' || type === 'area')
                ? safeData.map(({ time, value }) => ({ time, value }))
                : safeData;
            mainSeriesRef.current.setData(seriesData);
        }

        const currentIds = new Set(overlays.map(o => o.id));
        Object.keys(seriesRefs.current).forEach(sid => {
            if (!currentIds.has(sid)) {
                try { chartRef.current.removeSeries(seriesRefs.current[sid]); } catch {}
                delete seriesRefs.current[sid];
            }
        });

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

    // ── Overlay legend chips ───────────────────────────────────────────────
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
        <div className="relative w-full border-b border-white/30 bg-black flex-shrink-0" style={{ height }}>
            {/* ── Pane header ── */}
            <div className="absolute top-1.5 left-2 z-10 flex items-center gap-2 flex-wrap">
                {title && (
                    <span className="text-[10px] font-bold text-slate-500 uppercase flex items-center gap-1 pointer-events-none">
                        {isOscillator && (
                            <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: color }} />
                        )}
                        {title}
                    </span>
                )}

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

                {isOscillator && overlays.filter(ov => ov.label).map(ov => (
                    <span key={ov.id} className="flex items-center gap-0.5 text-[10px] font-bold pointer-events-none">
                        <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: solidColor(ov.color) }} />
                        <span className="text-slate-400">{ov.label}</span>
                    </span>
                ))}

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

            {/* ── Floating time label at bottom of pane (when time axis is hidden) ── */}
            {showTimeLabel && timeLabel && (
                <div
                    style={{
                        position: 'absolute',
                        bottom: 0,
                        left: timeLabel.x,
                        transform: 'translateX(-50%)',
                        background: '#1e222d',
                        color: '#d1d4dc',
                        fontSize: '10px',
                        fontFamily: 'monospace',
                        padding: '1px 5px',
                        border: '1px solid rgba(255,255,255,0.25)',
                        borderRadius: '2px',
                        pointerEvents: 'none',
                        zIndex: 20,
                        whiteSpace: 'nowrap',
                        lineHeight: '16px',
                    }}
                >
                    {timeLabel.text}
                </div>
            )}

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
    const isSyncingCrosshair = useRef(false);
    const outerRef = useRef(null);
    const [paneHeights, setPaneHeights] = useState({});
    const [containerHeight, setContainerHeight] = useState(600);

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
    const dragHandleTotal = panes.length * 4;
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

    // Sync visible time range (zoom/scroll) across all panes
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

    // Sync crosshair position across all panes
    const syncCrosshair = useCallback((param, sourceId) => {
        if (isSyncingCrosshair.current) return;
        isSyncingCrosshair.current = true;
        Object.keys(paneRefs.current).forEach(id => {
            if (id === sourceId) return;
            const pane = paneRefs.current[id];
            if (!pane) return;
            const chart = pane.getChart();
            const series = pane.getSeries();
            if (!chart || !series) return;
            if (param?.time !== undefined && param?.point) {
                chart.setCrosshairPosition(0, param.time, series);
            } else {
                chart.clearCrosshairPosition();
            }
        });
        isSyncingCrosshair.current = false;
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
                syncCrosshair={syncCrosshair}
                showTimeLabel={panes.length > 0}
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
                    <div
                        className="w-full h-1 bg-white/30 hover:bg-white/45 cursor-ns-resize flex-shrink-0 transition-colors"
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
                        syncCrosshair={syncCrosshair}
                        showTimeLabel={index !== panes.length - 1}
                        overlays={pane.overlays ?? []}
                        onClose={onRemovePane ? () => onRemovePane(pane.id) : undefined}
                    />
                </React.Fragment>
            ))}
        </div>
    );
};

export default ChartContainer;
