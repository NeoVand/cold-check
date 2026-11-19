/** Number formatting for the page (every number shown comes from static/data/*.json). */
const SUP: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };

export const int = (x: number) => Math.round(x).toLocaleString('en-US');

/** 2.0 × 10⁹ style; small numbers as plain integers */
export function sci(x: number, digits = 1, plainBelow = 1e5): string {
	if (x === 0) return '0';
	if (Math.abs(x) < plainBelow) return int(x);
	const e = Math.floor(Math.log10(Math.abs(x)));
	const m = x / 10 ** e;
	return `${m.toFixed(digits)} × 10${String(e)
		.split('')
		.map((c) => SUP[c])
		.join('')}`;
}

/** a duration in seconds, in the unit that reads best */
export function dur(s: number): string {
	if (s === 0) return 'none';
	if (s < 1) return `${+(s * 1e3).toPrecision(2)} ms`;
	if (s < 120) return `${+s.toPrecision(2)} s`;
	if (s < 3600) return `${+(s / 60).toPrecision(2)} min`;
	return `${+(s / 3600).toPrecision(2)} h`;
}

export const pct = (x: number, d = 2) => `${x.toFixed(d)}%`;

export function params(x: number): string {
	if (x >= 1e6) return `${+(x / 1e6).toPrecision(3)}M`;
	if (x >= 1e3) return `${+(x / 1e3).toPrecision(3)}K`;
	return String(x);
}
