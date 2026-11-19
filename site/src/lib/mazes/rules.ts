/**
 * The two fitted maze rules, ported line for line from code/bench/mazes.py:
 * - the Maze-Unique check, run cold: bounds on the kept counts of "on" side and corner neighbours (dead-end filling,
 *   derived from the fit, not written by hand);
 * - the Maze-Hard rule: the cold reach clock from G, then from S the best neighbour that meets the kept conditions.
 * Cells are numbered row by row; tokens 0 wall, 1 open, 2 S, 3 G.
 */

export const SIDE: [number, number][] = [
	[1, 0],
	[0, -1],
	[0, 1],
	[-1, 0]
]; // down, left, right, up
export const CORNER: [number, number][] = [
	[1, 1],
	[1, -1],
	[-1, 1],
	[-1, -1]
];
const RELS = [SIDE, CORNER];

export interface MazeCheck {
	theta: number[];
	kept: number[];
	lo: number[][][]; // [token][own value][relation]
	hi: number[][][];
}
export interface MazeRule {
	w: number[]; // open, nearer, level, farther, unreached, down, left, right, up
	required: number[];
	forbidden: number[];
}
export type Stats = { ops: number; cmp: number };

/** neighbour index per (cell, relation, direction), -1 where it does not exist: nb[(i * 2 + r) * 4 + j] */
export function neighbours(H: number, W: number): Int32Array {
	const nb = new Int32Array(H * W * 8).fill(-1);
	for (let r = 0; r < H; r++)
		for (let c = 0; c < W; c++)
			RELS.forEach((offs, k) =>
				offs.forEach(([dr, dc], j) => {
					const rr = r + dr;
					const cc = c + dc;
					if (rr >= 0 && rr < H && cc >= 0 && cc < W) nb[((r * W + c) * 2 + k) * 4 + j] = rr * W + cc;
				})
			);
	return nb;
}

export function parseTokens(s: string): Int8Array {
	return Int8Array.from(s, (ch) => '#.SG'.indexOf(ch));
}
export function parseBits(s: string): Uint8Array {
	return Uint8Array.from(s, (ch) => (ch === '1' ? 1 : 0));
}

/** mazes.propagate_nbr; onDecide(cell, dom) reports each change of a cell's possible values, in order */
export function propagateNbr(
	dom: Int32Array,
	tok: ArrayLike<number>,
	nb: Int32Array,
	ck: MazeCheck,
	st: Stats,
	onDecide?: (cell: number, d: number) => void
): boolean {
	const n = dom.length;
	const kept = ck.kept;
	const inq = new Uint8Array(n).fill(1);
	const ring = new Int32Array(n);
	for (let i = 0; i < n; i++) ring[i] = i;
	let head = 0;
	let size = n;
	const push = (x: number) => {
		inq[x] = 1;
		ring[(head + size) % n] = x;
		size++;
	};
	while (size > 0) {
		const i = ring[head];
		head = (head + 1) % n;
		size--;
		inq[i] = 0;
		const t = tok[i];
		for (let a = 0; a < 2; a++) {
			if (!((dom[i] >> a) & 1)) continue;
			let bad = false;
			for (const r of kept) {
				const u = 1;
				let L = 0;
				let U = 0;
				for (let j = 0; j < 4; j++) {
					const w = nb[(i * 2 + r) * 4 + j];
					if (w >= 0) {
						st.cmp++;
						if ((dom[w] >> u) & 1) {
							U++;
							st.ops++;
							if (dom[w] === 1 << u) {
								L++;
								st.ops++;
							}
						}
					}
				}
				st.cmp += 2;
				if (L > ck.hi[t][a][r] || U < ck.lo[t][a][r]) bad = true;
			}
			if (bad) {
				dom[i] &= ~(1 << a);
				if (dom[i] === 0) return false;
				onDecide?.(i, dom[i]);
				for (const r of kept)
					for (let j = 0; j < 4; j++) {
						const w = nb[(i * 2 + r) * 4 + j];
						if (w >= 0 && !inq[w]) push(w);
					}
				if (!inq[i]) push(i);
			}
		}
		// forcing on neighbours when my value is fixed
		if (dom[i] === 1 || dom[i] === 2) {
			const a = dom[i] === 1 ? 0 : 1;
			for (const r of kept) {
				const u = 1;
				let L = 0;
				let U = 0;
				for (let j = 0; j < 4; j++) {
					const w = nb[(i * 2 + r) * 4 + j];
					if (w >= 0) {
						if ((dom[w] >> u) & 1) {
							U++;
							st.ops++;
							if (dom[w] === 1 << u) {
								L++;
								st.ops++;
							}
						}
					}
				}
				st.cmp += 2;
				let newv = -1;
				if (U === ck.lo[t][a][r] && L < U) newv = 1 << u;
				else if (L === ck.hi[t][a][r] && U > L) newv = 0;
				if (newv >= 0) {
					for (let j = 0; j < 4; j++) {
						const w = nb[(i * 2 + r) * 4 + j];
						if (w >= 0 && (dom[w] >> u) & 1 && dom[w] !== 1 << u) {
							if (newv > 0) dom[w] = newv;
							else dom[w] &= ~(1 << u);
							if (dom[w] === 0) return false;
							onDecide?.(w, dom[w]);
							if (!inq[w]) push(w);
							// w changed, so every cell that counts w must be looked at again
							for (const r2 of kept)
								for (let j2 = 0; j2 < 4; j2++) {
									const x = nb[(w * 2 + r2) * 4 + j2];
									if (x >= 0 && !inq[x]) push(x);
								}
						}
					}
				}
			}
		}
	}
	return true;
}

export interface ColdResult {
	board: Uint8Array | null;
	ops: number;
	cmp: number;
	undecided: number;
	ok: boolean;
	events: [number, number][];
}

/** mazes.cold_unique: walls start off, S and G on, every open cell with both values possible */
export function coldUnique(ck: MazeCheck, tok: Int8Array, H: number, W: number, record = false): ColdResult {
	const nb = neighbours(H, W);
	const dom = Int32Array.from(tok, (t) => (t === 0 ? 1 : t >= 2 ? 2 : 3));
	const st = { ops: 0, cmp: 0 };
	const events: [number, number][] = [];
	const ok = propagateNbr(dom, tok, nb, ck, st, record ? (c, d) => events.push([c, d]) : undefined);
	let und = 0;
	for (const d of dom) if (d === 3) und++;
	const board = ok && und === 0 ? Uint8Array.from(dom, (d) => (d === 2 ? 1 : 0)) : null;
	return { board, ops: st.ops, cmp: st.cmp, undecided: und, ok, events };
}

/** mazes.passes: every kept count of on neighbours within its fitted range */
export function passes(ck: MazeCheck, tok: Int8Array, board: Uint8Array, H: number, W: number): boolean {
	const nb = neighbours(H, W);
	for (let i = 0; i < H * W; i++)
		for (const r of ck.kept) {
			let c = 0;
			for (let j = 0; j < 4; j++) {
				const w = nb[(i * 2 + r) * 4 + j];
				if (w >= 0) c += board[w];
			}
			const t = tok[i];
			const v = board[i];
			if (c < ck.lo[t][v][r] || c > ck.hi[t][v][r]) return false;
		}
	return true;
}

/** mazes.reach_clock: the loop at which each open cell is first reached from src (-1 never) */
export function reachClock(openm: Uint8Array, nb: Int32Array, r: number, src: number, st: Stats): Int32Array {
	const n = openm.length;
	const clock = new Int32Array(n).fill(-1);
	clock[src] = 0;
	let frontier = [src];
	let t = 0;
	while (frontier.length > 0) {
		t++;
		const nxt: number[] = [];
		for (const i of frontier)
			for (let j = 0; j < 4; j++) {
				const w = nb[(i * 2 + r) * 4 + j];
				st.cmp++;
				if (w >= 0 && openm[w] && clock[w] < 0) {
					clock[w] = t;
					st.ops++;
					nxt.push(w);
				}
			}
		frontier = nxt;
		st.ops++;
	}
	return clock;
}

/** mazes.rule_features: (4, 9) features of the side neighbours of u */
export function ruleFeatures(openm: Uint8Array, clock: Int32Array, nb: Int32Array, u: number): number[][] {
	const F = Array.from({ length: 4 }, () => new Array(9).fill(0));
	for (let j = 0; j < 4; j++) {
		const w = nb[(u * 2 + 0) * 4 + j];
		if (w < 0 || !openm[w]) F[j][4] = w >= 0 && clock[w] < 0 ? 1 : 0;
		if (w >= 0 && openm[w]) {
			F[j][0] = 1;
			if (clock[w] < 0) F[j][4] = 1;
			else if (clock[w] < clock[u]) F[j][1] = 1;
			else if (clock[w] === clock[u]) F[j][2] = 1;
			else F[j][3] = 1;
		}
		F[j][5 + j] = 1;
	}
	return F;
}

export interface HardResult {
	board: Uint8Array | null;
	path: number[];
	clock: Int32Array;
	ops: number;
	cmp: number;
}

/** mazes.solve_hard: the clock from G, then from S the best neighbour meeting the kept conditions, until G */
export function solveHard(rule: MazeRule, tok: Int8Array, H: number, W: number): HardResult {
	const nb = neighbours(H, W);
	const openm = Uint8Array.from(tok, (t) => (t > 0 ? 1 : 0));
	const S = tok.indexOf(2);
	const G = tok.indexOf(3);
	const st = { ops: 0, cmp: 0 };
	const clock = reachClock(openm, nb, 0, G, st);
	let ops = st.ops;
	let cmp = st.cmp;
	const board = new Uint8Array(H * W);
	board[S] = 1;
	const path = [S];
	let u = S;
	for (let step = 0; step < H * W; step++) {
		if (u === G) break;
		const F = ruleFeatures(openm, clock, nb, u);
		const ok = [true, true, true, true];
		for (const f of rule.required) for (let j = 0; j < 4; j++) ok[j] = ok[j] && F[j][f] === 1;
		for (const f of rule.forbidden) for (let j = 0; j < 4; j++) ok[j] = ok[j] && F[j][f] === 0;
		cmp += 4 * (rule.required.length + rule.forbidden.length);
		if (!ok.some(Boolean)) return { board: null, path, clock, ops, cmp };
		let nzmax = 0;
		const s = F.map((row) => {
			let acc = 0;
			let nz = 0;
			for (let f = 0; f < 9; f++) {
				acc += row[f] * rule.w[f];
				if (row[f] !== 0) nz++;
			}
			nzmax = Math.max(nzmax, nz);
			return acc;
		});
		ops += 4 * nzmax;
		let bj = -1;
		for (let j = 0; j < 4; j++) if (ok[j] && (bj < 0 || s[j] > s[bj])) bj = j;
		u = nb[(u * 2 + 0) * 4 + bj];
		board[u] = 1;
		path.push(u);
	}
	if (u !== G) return { board: null, path, clock, ops, cmp };
	return { board, path, clock, ops, cmp };
}

// ------------------------------------------------------------------------- maze builders (mazes.py, verbatim) --
export interface Built {
	open: Uint8Array;
	H: number;
	W: number;
	S: [number, number];
	G: [number, number];
	room?: Uint8Array; // the room's cells (for drawing only)
}

/** mazes.L_maze: an h x w room and a corridor, both from S; `extra` makes the corridor two steps shorter */
export function lMaze(h: number, w: number, extra: number): Built {
	const off = extra ? 1 : 0;
	const H0 = h + 2;
	const W0 = w + 2 + off;
	const g = Array.from({ length: H0 }, () => new Array(W0).fill(false));
	for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) g[r][c] = true;
	g[0][w] = true;
	const S: [number, number] = [0, w + 1];
	for (let r = 0; r < h + 2; r++) g[r][w + 1] = true;
	let G: [number, number];
	if (extra) {
		g[h][0] = true;
		g[h + 1][0] = true;
		G = [h + 1, 1];
		for (let c = 1; c < w + 2; c++) g[h + 1][c] = true;
	} else {
		g[h][0] = true;
		G = [h + 1, 0];
		for (let c = 0; c < w + 2; c++) g[h + 1][c] = true;
	}
	const H = H0 + 2;
	const W = W0 + 2;
	const open = new Uint8Array(H * W);
	const room = new Uint8Array(H * W);
	for (let r = 0; r < H0; r++) for (let c = 0; c < W0; c++) if (g[r][c]) open[(r + 1) * W + c + 1] = 1;
	for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) room[(r + 1) * W + c + 1] = 1;
	return { open, H, W, S: [S[0] + 1, S[1] + 1], G: [G[0] + 1, G[1] + 1], room };
}

/** mazes.sym: np.rot90 k % 4 times (counter-clockwise), then a transpose if k >= 4 */
export function sym(m: Built, k: number): Built {
	let H = m.H;
	let W = m.W;
	let g = m.open;
	let mark = new Uint8Array(H * W);
	mark[m.S[0] * W + m.S[1]] = 1;
	mark[m.G[0] * W + m.G[1]] = 2;
	const rot = (a: Uint8Array, h: number, w: number) => {
		// np.rot90: out[i][j] = a[j][w - 1 - i], shape (w, h)
		const o = new Uint8Array(h * w);
		for (let i = 0; i < w; i++) for (let j = 0; j < h; j++) o[i * h + j] = a[j * w + (w - 1 - i)];
		return o;
	};
	let room = m.room;
	for (let t = 0; t < k % 4; t++) {
		g = rot(g, H, W);
		mark = rot(mark, H, W);
		if (room) room = rot(room, H, W);
		[H, W] = [W, H];
	}
	if (k >= 4) {
		const tr = (a: Uint8Array, h: number, w: number) => {
			const o = new Uint8Array(h * w);
			for (let i = 0; i < w; i++) for (let j = 0; j < h; j++) o[i * h + j] = a[j * w + i];
			return o;
		};
		g = tr(g, H, W);
		mark = tr(mark, H, W);
		if (room) room = tr(room, H, W);
		[H, W] = [W, H];
	}
	const s = mark.indexOf(1);
	const gg = mark.indexOf(2);
	return { open: g, H, W, S: [Math.floor(s / W), s % W], G: [Math.floor(gg / W), gg % W], room };
}

export function tokens(m: Built): Int8Array {
	const t = Int8Array.from(m.open);
	t[m.S[0] * m.W + m.S[1]] = 2;
	t[m.G[0] * m.W + m.G[1]] = 3;
	return t;
}

export function bfsDist(open: ArrayLike<number>, H: number, W: number, src: number): Int32Array {
	const d = new Int32Array(H * W).fill(-1);
	d[src] = 0;
	const q = [src];
	for (let h = 0; h < q.length; h++) {
		const v = q[h];
		const r = Math.floor(v / W);
		const c = v % W;
		for (const [dr, dc] of SIDE) {
			const rr = r + dr;
			const cc = c + dc;
			if (rr >= 0 && rr < H && cc >= 0 && cc < W && open[rr * W + cc] && d[rr * W + cc] < 0) {
				d[rr * W + cc] = d[v] + 1;
				q.push(rr * W + cc);
			}
		}
	}
	return d;
}

/** mazes.label_rule: the benchmark's label (from S, the side neighbour nearest G; ties down, left, right, up) */
export function labelRule(open: ArrayLike<number>, H: number, W: number, S: number, G: number): Uint8Array {
	const dG = bfsDist(open, H, W, G);
	const lab = new Uint8Array(H * W);
	let p = S;
	lab[p] = 1;
	while (p !== G) {
		const r = Math.floor(p / W);
		const c = p % W;
		let best = -1;
		let bk: [number, number] | null = null;
		SIDE.forEach(([dr, dc], k) => {
			const rr = r + dr;
			const cc = c + dc;
			if (rr >= 0 && rr < H && cc >= 0 && cc < W && open[rr * W + cc] && dG[rr * W + cc] >= 0) {
				const key: [number, number] = [dG[rr * W + cc], k];
				if (!bk || key[0] < bk[0] || (key[0] === bk[0] && key[1] < bk[1])) {
					bk = key;
					best = rr * W + cc;
				}
			}
		});
		p = best;
		lab[p] = 1;
	}
	return lab;
}
