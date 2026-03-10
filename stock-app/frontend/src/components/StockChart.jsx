import React, { useMemo } from 'react';
import ChartContainer from './ChartContainer';

// ── Per-indicator colors ───────────────────────────────────────────────────────
const INDICATOR_COLORS = {
    // SMA (distinct per period)
    sma:     '#2962ff',
    sma_9:   '#FF6B6B',
    sma_20:  '#2962ff',
    sma_50:  '#FF9800',
    sma_100: '#4CAF50',
    sma_200: '#9C27B0',
    // EMA (lighter shades of same palette)
    ema:     '#64B5F6',
    ema_9:   '#FF8A80',
    ema_20:  '#64B5F6',
    ema_50:  '#FFCC02',
    ema_100: '#A5D6A7',
    ema_200: '#CE93D8',
    // Overlays
    vwap:       '#6200EA',
    bbands:     '#2962ff',
    psar:       '#FFEB3B',
    supertrend: '#4CAF50',
    ichimoku:   '#26C6DA',
    // Oscillators
    rsi:   '#7e57c2',
    macd:  '#26a69a',
    stoch: '#ff5252',
    cci:   '#ff6d00',
    atr:   '#ffa726',
    obv:   '#2962ff',
    stdev: '#26c6da',
    cmf:   '#00e676',
};

const INDICATOR_LABELS = {
    sma_9:  'SMA 9',  sma_20:  'SMA 20',  sma_50:  'SMA 50',
    sma_100: 'SMA 100', sma_200: 'SMA 200',
    ema_9:  'EMA 9',  ema_20:  'EMA 20',  ema_50:  'EMA 50',
    ema_100: 'EMA 100', ema_200: 'EMA 200',
    sma: 'SMA', ema: 'EMA',
    vwap: 'VWAP', bbands: 'BB',
    psar: 'PSAR', supertrend: 'ST', ichimoku: 'Ichi',
    rsi: 'RSI (14)', macd: 'MACD (12,26,9)', stoch: 'Stoch',
    cci: 'CCI', atr: 'ATR (14)', obv: 'OBV', stdev: 'StdDev', cmf: 'CMF',
};

// Resolve color/label for any id (with period-aware fallback)
const getColor = (id) => INDICATOR_COLORS[id] ?? INDICATOR_COLORS[id.split('_')[0]] ?? '#2962ff';
const getLabel = (id) => INDICATOR_LABELS[id] ?? INDICATOR_LABELS[id.split('_')[0]] ?? id.toUpperCase();

// ── Which indicators are overlays vs. oscillator panes ───────────────────────
const CATEGORIES = {
    sma:        { type: 'overlay' },
    ema:        { type: 'overlay' },
    bbands:     { type: 'overlay' },
    vwap:       { type: 'overlay' },
    psar:       { type: 'overlay' },
    supertrend: { type: 'overlay' },
    ichimoku:   { type: 'overlay' },
    rsi:        { type: 'oscillator', label: 'RSI (14)' },
    macd:       { type: 'oscillator', label: 'MACD (12,26,9)' },
    stoch:      { type: 'oscillator', label: 'Stochastic' },
    cci:        { type: 'oscillator', label: 'CCI' },
    atr:        { type: 'oscillator', label: 'ATR (14)' },
    obv:        { type: 'oscillator', label: 'OBV' },
    stdev:      { type: 'oscillator', label: 'Std Dev' },
    cmf:        { type: 'oscillator', label: 'CMF' },
};

// ── StockChart ────────────────────────────────────────────────────────────────
const StockChart = ({
    data,
    indicatorData,
    activeIndicators = [],
    chartType = 'candlestick',
    onCrosshairMove,
    onRemovePane,
    onRemoveOverlay,
    drawings = [],
    onAddDrawing,
    activeTool = 'none',
    position,
    tradeMarkers = [],
}) => {
    const { panes, mainOverlays } = useMemo(() => {
        if (!Array.isArray(indicatorData) || indicatorData.length === 0) {
            return { panes: [], mainOverlays: [] };
        }

        const overlays = [];
        const oscillatorPanes = [];
        const sample = indicatorData[0] || {};

        const buildSeries = (col) => indicatorData.map(d => {
            const t = d?.Date || d?.time;
            if (!t || d[col] == null) return null;
            return { time: t, value: d[col] };
        }).filter(Boolean);

        activeIndicators.forEach(id => {
            const baseId = id.split('_')[0];

            // ── Bollinger Bands ─────────────────────────────────────────────
            if (baseId === 'bbands') {
                const bands = [
                    { prefix: 'BBU', color: '#2962ff',    firstLabel: 'BB' },
                    { prefix: 'BBM', color: '#2962ff60',  firstLabel: null },
                    { prefix: 'BBL', color: '#2962ff',    firstLabel: null },
                ];
                bands.forEach(({ prefix, color, firstLabel }) => {
                    const col = Object.keys(sample).find(k => k.toUpperCase().startsWith(prefix));
                    if (!col) return;
                    const sd = buildSeries(col);
                    if (sd.length > 0) {
                        overlays.push({ id: `${id}_${prefix}`, sourceId: id, data: sd, color, label: firstLabel });
                    }
                });
                return;
            }

            // ── MACD: line + signal in one pane ─────────────────────────────
            if (baseId === 'macd') {
                const macdCol   = Object.keys(sample).find(k => /^MACD_\d/.test(k));
                const signalCol = Object.keys(sample).find(k => /^MACDs_/i.test(k));
                const macdData   = macdCol   ? buildSeries(macdCol)   : [];
                const signalData = signalCol ? buildSeries(signalCol) : [];
                if (macdData.length > 0) {
                    oscillatorPanes.push({
                        id: 'macd',
                        title: 'MACD',
                        data: macdData,
                        color: getColor('macd'),
                        overlays: signalData.length > 0
                            ? [{ id: 'macd_signal', sourceId: 'macd', data: signalData, color: '#ef5350', label: 'Signal' }]
                            : [],
                    });
                }
                return;
            }

            // ── Stochastic: %K + %D in one pane ─────────────────────────────
            if (baseId === 'stoch') {
                const kCol = Object.keys(sample).find(k => /^STOCHk_/i.test(k));
                const dCol = Object.keys(sample).find(k => /^STOCHd_/i.test(k));
                const kData = kCol ? buildSeries(kCol) : [];
                const dData = dCol ? buildSeries(dCol) : [];
                if (kData.length > 0) {
                    oscillatorPanes.push({
                        id: 'stoch',
                        title: 'Stochastic',
                        data: kData,
                        color: getColor('stoch'),
                        overlays: dData.length > 0
                            ? [{ id: 'stoch_d', sourceId: 'stoch', data: dData, color: '#ff9800', label: '%D' }]
                            : [],
                    });
                }
                return;
            }

            // ── Ichimoku: 4 lines as overlays ────────────────────────────────
            if (baseId === 'ichimoku') {
                const lines = [
                    { regex: /^ITS_/i,  color: '#ef5350', label: 'Ichi' },
                    { regex: /^IKS_/i,  color: '#2962ff', label: null },
                    { regex: /^ISA_/i,  color: '#4caf5090', label: null },
                    { regex: /^ISB_/i,  color: '#ef535090', label: null },
                ];
                lines.forEach(({ regex, color, label }) => {
                    const col = Object.keys(sample).find(k => regex.test(k));
                    if (!col) return;
                    const sd = buildSeries(col);
                    if (sd.length > 0) {
                        overlays.push({ id: `ichimoku_${col}`, sourceId: id, data: sd, color, label });
                    }
                });
                return;
            }

            // ── SuperTrend ──────────────────────────────────────────────────
            if (baseId === 'supertrend') {
                const col = Object.keys(sample).find(k => /^SUPERT_/i.test(k));
                if (!col) return;
                const sd = buildSeries(col);
                if (sd.length > 0) {
                    overlays.push({ id, sourceId: id, data: sd, color: getColor('supertrend'), label: 'ST' });
                }
                return;
            }

            // ── Generic SMA (no period): draw all found SMA_N periods ────────
            if (id === 'sma') {
                Object.keys(sample)
                    .filter(k => /^SMA_\d+$/i.test(k))
                    .forEach(col => {
                        const period = col.split('_')[1];
                        const fullId = `sma_${period}`;
                        const sd = buildSeries(col);
                        if (sd.length > 0) {
                            overlays.push({ id: fullId, sourceId: 'sma', data: sd, color: getColor(fullId), label: getLabel(fullId) });
                        }
                    });
                return;
            }

            // ── Generic EMA (no period): draw all found EMA_N periods ────────
            if (id === 'ema') {
                Object.keys(sample)
                    .filter(k => /^EMA_\d+$/i.test(k))
                    .forEach(col => {
                        const period = col.split('_')[1];
                        const fullId = `ema_${period}`;
                        const sd = buildSeries(col);
                        if (sd.length > 0) {
                            overlays.push({ id: fullId, sourceId: 'ema', data: sd, color: getColor(fullId), label: getLabel(fullId) });
                        }
                    });
                return;
            }

            // ── Generic fallback ─────────────────────────────────────────────
            const settings = CATEGORIES[baseId] ?? { type: 'overlay' };
            const color = getColor(id);
            const label = getLabel(id);

            const seriesData = indicatorData.map(d => {
                if (!d) return null;
                const timeValue = d.Date || d.time;
                if (!timeValue) return null;
                // 1) Exact key match (e.g. d['rsi'])
                // 2) Exact uppercase column for period-specific ids: ema_50 → EMA_50, sma_200 → SMA_200
                // 3) StartsWith fallback for generic ids like atr → ATRr_14, vwap → VWAP_D
                const parts = id.split('_');
                const exactUpperCol = parts.length === 2
                    ? `${parts[0].toUpperCase()}_${parts[1]}`
                    : null;
                const value = d[id] !== undefined
                    ? d[id]
                    : (exactUpperCol && d[exactUpperCol] !== undefined)
                        ? d[exactUpperCol]
                        : d[Object.keys(d).find(k => k.toLowerCase().startsWith(baseId.toLowerCase()))];
                return { time: timeValue, value };
            }).filter(d => d && d.value != null);

            if (seriesData.length === 0) return;

            if (settings.type === 'overlay') {
                overlays.push({ id, sourceId: id, data: seriesData, color, label });
            } else {
                oscillatorPanes.push({
                    id,
                    title: settings.label || label,
                    data: seriesData,
                    color,
                    overlays: [],
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
                onCrosshairMove={onCrosshairMove}
                onRemovePane={onRemovePane}
                onRemoveOverlay={onRemoveOverlay}
                drawings={drawings}
                onAddDrawing={onAddDrawing}
                activeTool={activeTool}
                position={position}
                tradeMarkers={tradeMarkers}
            />
        </div>
    );
};

export default StockChart;
