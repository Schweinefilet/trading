/**
 * computeHeikinAshi - Transforms standard OHLCV data into Heikin Ashi format.
 */
export const computeHeikinAshi = (data) => {
    if (!data || data.length === 0) return [];

    const haData = [];
    let prevHA = null;

    data.forEach((d, i) => {
        const close = (d.Open + d.High + d.Low + d.Close) / 4;
        const open = prevHA ? (prevHA.open + prevHA.close) / 2 : (d.Open + d.Close) / 2;
        const high = Math.max(d.High, open, close);
        const low = Math.min(d.Low, open, close);

        const haPoint = { ...d, Open: open, High: high, Low: low, Close: close };
        haData.push(haPoint);
        prevHA = haPoint;
    });

    return haData;
};
