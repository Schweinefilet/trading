import { useEffect } from 'react';

/**
 * Renders a full-viewport canvas background using marcbruederlin/particles.js.
 * Position: fixed so it covers the entire viewport regardless of scroll.
 * Destroyed and removed automatically when the parent page unmounts.
 *
 * @param {string} canvasId - Unique id for the canvas element (one per page).
 */
const ParticlesBg = ({ canvasId = 'particles-bg' }) => {
    useEffect(() => {
        // Create and attach the canvas to body so it's truly behind everything
        const canvas = document.createElement('canvas');
        canvas.id = canvasId;
        canvas.style.cssText = [
            'position:fixed',
            'top:0',
            'left:0',
            'width:100%',
            'height:100%',
            'pointer-events:none',
            'z-index:0',
            'opacity:0.55',
        ].join(';');
        document.body.appendChild(canvas);

        let lib = null;
        let cancelled = false;

        import('particlesjs').then((mod) => {
            if (cancelled) return;
            lib = mod.default ?? mod;
            try {
                lib.init({
                    selector: `#${canvasId}`,
                    maxParticles: 90,
                    color: '#ffffff',
                    connectParticles: true,
                    speed: 0.28,
                    minDistance: 160,
                    sizeVariations: 3,
                });
            } catch (err) {
                console.warn('ParticlesBg:', err);
            }
        });

        return () => {
            cancelled = true;
            try { lib?.destroy?.(); } catch (_) {}
            canvas.remove();
        };
    }, [canvasId]);

    // Nothing to render — canvas is appended directly to <body>
    return null;
};

export default ParticlesBg;
