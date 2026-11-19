import type { PageLoad } from './$types';
import { asset } from '$app/paths';

/** The demos' data, trimmed to what the page draws (the full files, with the parity cases, stay in static/data). */
export const load: PageLoad = async ({ fetch }) => {
	const get = async (f: string) => (await fetch(asset(`data/${f}` as Parameters<typeof asset>[0]))).json();
	const [S, M, B, R] = await Promise.all([get('sudoku.json'), get('mazes.json'), get('bfs.json'), get('results.json')]);
	const demoIdx = new Set(Object.values(S.demo as Record<string, number>));
	const sudoku = {
		check: S.check,
		demo: S.demo,
		extreme: S.extreme.filter((p: { index: number }) => demoIdx.has(p.index)),
		sizes: Object.fromEntries(Object.entries(S.sizes as Record<string, unknown[]>).map(([k, v]) => [k, v.slice(0, 6)])),
		stats: S.stats,
		source: S.source
	};
	const mazes = {
		check: M.check,
		rule: M.rule,
		hard: M.hard.filter((m: { index: number }) => m.index === 106),
		cram106: M.cram106,
		edge: M.edge.slice(0, 1),
		rooms: M.rooms.filter((r: { extra: number; sym: number }) => r.extra === 2 && r.sym === 1),
		stats: M.stats,
		source: M.source
	};
	const bfs = { w: B.w, wb: B.wb, gate: B.gate, train: B.train, record: { order: { fit_ms: B.record.order.fit_ms } }, source: B.source };
	return { sudoku, mazes, bfs, results: R };
};
