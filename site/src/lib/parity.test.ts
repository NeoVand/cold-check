/**
 * Parity: the page's TypeScript rules give the Python outputs exported by scripts/export_data.py
 * (boards, moves and operation counts), on at least 20 instances per demo.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { build, masks, decode, Solver, type SudokuCheck } from './sudoku/engine.ts';
import { coldUnique, passes, solveHard, lMaze, sym, tokens, labelRule, parseTokens, parseBits } from './mazes/rules.ts';
import { fromEdges, bfs, fit, run, completion, type Graph } from './bfs/rule.ts';

const load = (f: string) => JSON.parse(readFileSync(new URL(`../../static/data/${f}`, import.meta.url), 'utf8'));
const S = load('sudoku.json');
const MZ = load('mazes.json');
const B = load('bfs.json');
const bits = (b: ArrayLike<number> | null) => (b ? Array.from(b, (x) => (x ? '1' : '0')).join('') : null);

describe('sudoku: the fitted check, cold then warm', () => {
	const check = S.check as SudokuCheck;
	const E9 = build(check, 3);
	it('Sudoku-Extreme test puzzles: same board, moves and operation counts', () => {
		expect(S.extreme.length).toBeGreaterThanOrEqual(20);
		for (const p of S.extreme) {
			const s = new Solver(E9, masks(p.puzzle, 9), p.seed).run();
			expect({ i: p.index, status: s.status, moves: s.moves, ops: s.st.ops, cmp: s.st.cmp, cold: s.coldOnly, maxdepth: s.maxDepth }).toEqual({
				i: p.index,
				status: p.py.status,
				moves: p.py.moves,
				ops: p.py.ops,
				cmp: p.py.cmp,
				cold: p.py.cold,
				maxdepth: p.py.maxdepth
			});
			expect(decode(s.dom)).toEqual(p.py.board);
			expect(decode(s.dom)).toEqual(p.solution);
		}
	});
	for (const side of ['4', '16']) {
		it(`${side} x ${side} with the 9 x 9 fit`, () => {
			const b = Math.round(Math.sqrt(Number(side)));
			const E = build(check, b);
			const rows = S.sizes[side];
			expect(rows.length).toBeGreaterThanOrEqual(20);
			for (const p of rows) {
				const s = new Solver(E, masks(p.puzzle, b * b), p.seed).run();
				expect([s.status, s.moves, s.st.ops, s.st.cmp, s.coldOnly]).toEqual([p.py.status, p.py.moves, p.py.ops, p.py.cmp, p.py.cold]);
				expect(decode(s.dom)).toEqual(p.py.board);
			}
		});
	}
	it('25 x 25 with the 9 x 9 fit: the recorded puzzles, move for move', () => {
		const E = build(check, 5);
		const rows = S.sizes['25'];
		expect(rows.length).toBeGreaterThan(0);
		for (const p of rows) {
			const s = new Solver(E, masks(p.puzzle, 25), p.seed).run();
			expect([s.status, s.moves, s.st.ops, s.st.cmp]).toEqual([p.py.orig.status, p.py.orig.moves, p.py.orig.ops, p.py.orig.cmp]);
			if (p.py.orig.board) expect(decode(s.dom)).toEqual(p.py.orig.board);
		}
	});
	it('resuming the warm search in chunks gives the uninterrupted run', () => {
		const p = S.extreme.find((x: { py: { moves: number } }) => x.py.moves > 50);
		const s = new Solver(E9, masks(p.puzzle, 9), p.seed);
		s.cold();
		let k = 0;
		while (!s.warm((k += 7))) {
			/* step */
		}
		expect([s.moves, s.st.ops]).toEqual([p.py.moves, p.py.ops]);
	});
});

describe('mazes: the Maze-Unique check and the Maze-Hard rule', () => {
	it('Maze-Hard test mazes (maze 106 included): same path and counts', () => {
		expect(MZ.hard.length).toBeGreaterThanOrEqual(20);
		for (const m of MZ.hard) {
			const r = solveHard(MZ.rule, parseTokens(m.tokens), 30, 30);
			expect([m.index, bits(r.board), r.ops, r.cmp]).toEqual([m.index, m.py.board, m.py.ops, m.py.cmp]);
			expect(bits(r.board)).toEqual(m.label);
		}
	});
	it('Maze-Unique test mazes, cold derivation alone', () => {
		expect(MZ.unique.length).toBeGreaterThanOrEqual(20);
		for (const m of MZ.unique) {
			const t = parseTokens(m.tokens);
			const r = coldUnique(MZ.check, t, 30, 30);
			expect([bits(r.board), r.ops, r.cmp, r.undecided]).toEqual([m.py.board, m.py.ops, m.py.cmp, m.py.undecided]);
			expect(passes(MZ.check, t, r.board!, 30, 30)).toBe(true);
		}
	});
	it('edge mazes without a wall ring (dead ends on the board edge)', () => {
		expect(MZ.edge.length).toBeGreaterThanOrEqual(20);
		for (const m of MZ.edge) {
			const r = coldUnique(MZ.check, parseTokens(m.tokens), m.side, m.side);
			expect([bits(r.board), r.ops]).toEqual([m.py.board, m.py.ops]);
			expect(bits(r.board)).toEqual(m.label);
		}
	});
	it('room-and-corridor mazes: same builder, same label, same path', () => {
		expect(MZ.rooms.length).toBe(136);
		for (const m of MZ.rooms) {
			const b = sym(lMaze(m.n, m.n, m.extra), m.sym);
			expect([b.H, b.W, b.S, b.G]).toEqual([m.shape[0], m.shape[1], m.S, m.G]);
			const lab = labelRule(b.open, b.H, b.W, b.S[0] * b.W + b.S[1], b.G[0] * b.W + b.G[1]);
			expect(bits(lab)).toEqual(m.label);
			const r = solveHard(MZ.rule, tokens(b), b.H, b.W);
			expect(bits(r.board)).toEqual(m.py.board);
		}
	});
});

describe('bfs: the parent rule fitted at the completed answer', () => {
	const train = B.train.map((t: { n: number; s: number; edges: [number, number][]; par: number[] }) => ({
		g: fromEdges(t.n, t.s, t.edges),
		par: t.par
	}));
	it('the training labels are BFS as the page computes it', () => {
		for (const t of train) expect(Array.from(bfs(t.g).par)).toEqual(t.par);
	});
	it('the fit in TypeScript gives the Python weights', () => {
		const f = fit(train);
		expect(f.gate).toEqual(B.gate);
		f.w.forEach((x, i) => expect(Math.abs(x - B.w[i])).toBeLessThan(1e-6 * (1 + Math.abs(B.w[i]))));
		f.wb.forEach((x, i) => expect(Math.abs(x - B.wb[i])).toBeLessThan(1e-6 * (1 + Math.abs(B.wb[i]))));
	});
	it('run from nothing on graphs of 16 to 1,024 nodes: same parents, same loops', () => {
		expect(B.parity.length).toBeGreaterThanOrEqual(20);
		const f = fit(train);
		for (const p of B.parity) {
			const g: Graph = fromEdges(p.n, p.s, p.edges);
			for (const [w, wb] of [
				[B.w, B.wb],
				[f.w, f.wb]
			]) {
				const r = run(g, w, wb, B.gate);
				expect([p.n, Array.from(r.par), r.loops]).toEqual([p.n, p.py.par, p.py.loops]);
			}
			expect(Array.from(bfs(g).par)).toEqual(p.par);
			expect(completion(p.par, p.s).place.length).toBe(p.n);
		}
	});
});
