export const PORTFOLIO_HISTORY_ANCHOR_DATE = '2026-03-11';
export const PORTFOLIO_HISTORY_ANCHOR_VALUE = 11563.18;
export const PORTFOLIO_HISTORY_VALUE_FLOOR = 10500;

export const MANUAL_PRE_ANCHOR_VALUES = [
    { date: '2026-03-04', value: 12516.08 },
    { date: '2026-03-05', value: 12099.44 },
    { date: '2026-03-06', value: 11438.19 },
    { date: '2026-03-07', value: 11427.99 },
    { date: '2026-03-10', value: 11439.01 },
];

export const buildPortfolioAnalyticsParams = (timeframe, extra = {}) => ({
    timeframe,
    anchor_start_date: PORTFOLIO_HISTORY_ANCHOR_DATE,
    anchor_start_value: PORTFOLIO_HISTORY_ANCHOR_VALUE,
    value_floor: PORTFOLIO_HISTORY_VALUE_FLOOR,
    ...extra,
});

const getTimeframeStartDate = (timeframe) => {
    const now = new Date();
    const start = new Date(now);
    const key = String(timeframe || '').toLowerCase();
    if (key === 'max') return null;
    if (key === '1d') start.setDate(now.getDate() - 0);
    else if (key === '1w') start.setDate(now.getDate() - 6);
    else if (key === '1m') start.setDate(now.getDate() - 30);
    else if (key === '3mo') start.setDate(now.getDate() - 91);
    else if (key === '6mo') start.setDate(now.getDate() - 182);
    else if (key === '1y') start.setDate(now.getDate() - 365);
    else if (key === '2y') start.setDate(now.getDate() - 730);
    else start.setDate(now.getDate() - 365);
    start.setHours(0, 0, 0, 0);
    return start;
};

export const buildSyncedPortfolioHistory = (rawSeries, timeframe) => {
    const safeSeries = Array.isArray(rawSeries) ? rawSeries : [];
    const timeframeStartDate = getTimeframeStartDate(timeframe);

    const manualPointsInWindow = MANUAL_PRE_ANCHOR_VALUES
        .filter((point) => {
            if (!timeframeStartDate) return true;
            const date = new Date(`${point.date}T00:00:00`);
            return date >= timeframeStartDate;
        })
        .map((point) => ({
            date: point.date,
            value: Number(point.value),
            is_inferred: false,
            is_manual: true,
        }));

    const mergedByDate = new Map();
    for (const point of manualPointsInWindow) {
        mergedByDate.set(point.date, point);
    }

    for (const point of safeSeries) {
        const date = String(point?.date || '').slice(0, 10);
        if (!date) continue;
        mergedByDate.set(date, {
            ...point,
            date,
            is_manual: false,
        });
    }

    return Array.from(mergedByDate.values())
        .sort((a, b) => String(a.date).localeCompare(String(b.date)))
        .map((point) => {
            const value = Number(point?.value);
            const date = point?.date;
            if (!date) return null;
            const shortDate = timeframe === '1d'
                ? String(date).slice(11, 16)
                : timeframe === '1w'
                    ? String(date).slice(5, 16)
                    : timeframe === 'max'
                        ? String(date).slice(2, 10)
                        : String(date).slice(5, 10);
            return {
                date,
                shortDate,
                value: Number.isFinite(value) ? value : null,
                isInferred: !!point?.is_inferred,
                isManual: !!point?.is_manual,
            };
        })
        .filter(Boolean);
};
