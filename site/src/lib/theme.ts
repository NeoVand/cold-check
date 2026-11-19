/** The palette for canvases and SVG (layout.css holds the same colours as CSS tokens). */
export const C = {
	paper: '#FFFFFF',
	tint: '#F2F5FB',
	lift: '#E6EBF4',
	ink: '#1F2A44',
	inkSoft: '#4C5B78',
	blueGrey: '#7D8BA8',
	mist: '#BAC5D9',
	edge: '#DDE3EE',
	rose: '#D6336C',
	roseLight: '#F1A3BC',
	roseTint: '#FBE3EB',
	cobalt: '#2557A7',
	cobaltLight: '#9DB8E0',
	cobaltTint: '#E3EBF7',
	wall: '#C3CCDB'
} as const;

/** a canvas sized for the device's pixel ratio; returns the 2D context in CSS pixels */
export function fitCanvas(cv: HTMLCanvasElement, w: number, h: number): CanvasRenderingContext2D {
	const dpr = Math.min(3, window.devicePixelRatio || 1);
	cv.width = Math.round(w * dpr);
	cv.height = Math.round(h * dpr);
	cv.style.width = `${w}px`;
	cv.style.height = `${h}px`;
	const ctx = cv.getContext('2d')!;
	ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
	return ctx;
}

export const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
export const reducedMotion = () => typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** cancelAnimationFrame that is safe during server rendering */
export const stopFrame = (id: number) => {
	if (id && typeof cancelAnimationFrame !== 'undefined') cancelAnimationFrame(id);
};
