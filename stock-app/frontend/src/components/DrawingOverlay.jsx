import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';

const DrawingOverlay = ({ chart, mainSeries, drawings = [], onAddDrawing, tool = 'none' }) => {
    const [points, setPoints] = useState([]);
    const [mousePos, setMousePos] = useState(null);
    const svgRef = useRef(null);

    const getPointsCoords = useCallback((pts) => {
        if (!chart || !mainSeries) return [];
        return pts.map(p => {
            const x = chart.timeScale().timeToCoordinate(p.time);
            const y = mainSeries.priceToCoordinate(p.price);
            if (x === null || y === null) return null;
            return { x, y };
        }).filter(p => p !== null);
    }, [chart, mainSeries]);

    const handleMouseDown = (e) => {
        if (tool === 'none' || !chart || !mainSeries) return;

        const rect = svgRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const time = chart.timeScale().coordinateToTime(x);
        const price = mainSeries.coordinateToPrice(y);

        const newPoints = [...points, { time, price }];

        if (tool === 'trendline' && newPoints.length === 2) {
            onAddDrawing({ type: 'trendline', points: newPoints });
            setPoints([]);
        } else if (tool === 'hline') {
            onAddDrawing({ type: 'hline', price });
            setPoints([]);
        } else {
            setPoints(newPoints);
        }
    };

    const currentCoords = useMemo(() => getPointsCoords(points), [points, getPointsCoords]);
    const existingDrawings = useMemo(() => {
        return drawings.map((d, i) => {
            const pts = getPointsCoords(d.points || []);
            if (d.type === 'trendline' && pts.length === 2) {
                return <line key={i} x1={pts[0].x} y1={pts[0].y} x2={pts[1].x} y2={pts[1].y} stroke="#2962ff" strokeWidth="2" />;
            }
            if (d.type === 'hline') {
                const y = mainSeries.priceToCoordinate(d.price);
                return <line key={i} x1="0" y1={y} x2="100%" y2={y} stroke="#2962ff" strokeWidth="1" strokeDasharray="4" />;
            }
            return null;
        });
    }, [drawings, getPointsCoords, mainSeries]);

    return (
        <svg
            ref={svgRef}
            className={`absolute inset-0 z-30 ${tool !== 'none' ? 'cursor-crosshair' : 'pointer-events-none'}`}
            onMouseDown={handleMouseDown}
            onMouseMove={(e) => {
                const rect = svgRef.current.getBoundingClientRect();
                setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
            }}
        >
            {existingDrawings}

            {/* Active Points */}
            {currentCoords.map((p, i) => (
                <circle key={i} cx={p.x} cy={p.y} r="4" fill="#2962ff" />
            ))}

            {/* Preview Line */}
            {tool === 'trendline' && currentCoords.length === 1 && mousePos && (
                <line x1={currentCoords[0].x} y1={currentCoords[0].y} x2={mousePos.x} y2={mousePos.y} stroke="#2962ff" strokeWidth="2" strokeDasharray="4" opacity="0.5" />
            )}
        </svg>
    );
};

export default DrawingOverlay;
