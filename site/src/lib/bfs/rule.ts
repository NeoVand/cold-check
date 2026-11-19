/**
 * BFS parents from a rule fitted at the completed answer: a port of code/walkthrough/completed_answer_bfs.py.
 *
 * Completion (from the label alone): each node's place in the order of discovery. The parent rule reads a pair (u, v)
 * through every third node k ("is there a k with a test on (k, u) and a test on (k, v)"), fitted by logistic
 * regression at the answers of 30 small graphs. It is run from nothing: places start at "not yet", a node keeps its
 * parent once it has one, and the kept conditions (an edge, and u placed before v) gate every choice.
 * The fast run below computes the same features as the dense Python ones by looking only at the neighbours of v
 * (every second test needs an edge (k, v)), so it is exact and costs about n^2 times the degree per loop.
 */

export const INF = 1e9;

export interface Graph {
	n: number;
	s: number;
	adj: number[][]; // sorted neighbour lists
}

export function fromEdges(n: number, s: number, edges: [number, number][]): Graph {
	const adj: number[][] = Array.from({ length: n }, () => []);
	for (const [u, v] of edges) {
		adj[u].push(v);
		adj[v].push(u);
	}
	adj.forEach((a) => a.sort((x, y) => x - y));
	return { n, s, adj };
}

/** the label: the neighbour that discovered each node (queue order, neighbours by index); par[s] = s */
export function bfs(g: Graph): { par: Int32Array; order: number[] } {
	const par = new Int32Array(g.n).fill(-1);
	par[g.s] = g.s;
	const q = [g.s];
	for (let h = 0; h < q.length; h++) {
		const u = q[h];
		for (const v of g.adj[u])
			if (par[v] < 0) {
				par[v] = u;
				q.push(v);
			}
	}
	return { par, order: q };
}

/** completion from the label alone: depth (ancestors) and place (nodes before, level by level) */
export function completion(par: ArrayLike<number>, s: number) {
	const n = par.length;
	const depth = new Int32Array(n);
	for (let v = 0; v < n; v++) {
		let u = v;
		while (u !== s) {
			u = par[u];
			depth[v]++;
		}
	}
	const place = new Float64Array(n).fill(INF);
	place[s] = 0;
	const pp = new Float64Array(n).fill(-1);
	let dmax = 0;
	for (const d of depth) dmax = Math.max(dmax, d);
	for (let d = 1; d <= dmax; d++) {
		const lev: number[] = [];
		for (let v = 0; v < n; v++) if (depth[v] === d) lev.push(v);
		for (const v of lev) pp[v] = place[par[v]];
		const order = [...lev].sort((a, b) => pp[a] - pp[b] || a - b);
		let base = 0;
		for (let v = 0; v < n; v++) if (depth[v] < d) base++;
		order.forEach((v, i) => (place[v] = base + i));
	}
	return { depth, place, pp };
}

const lt = (a: number, b: number) => a < b;
const eq = (a: number, b: number) => a === b && a < INF;
const gt = (a: number, b: number) => a > b;

/**
 * The 16 features of one ordered pair (u, v) for register r: own bits [edge, lt, eq, gt, idx_lt], then for each test
 * on (k, u) [lt, eq, gt, eq & lower index, lower index] and each test on (k, v) [edge, edge & k before v]:
 * "is there such a k", then a bias. Same order as features() in Python.
 */
export function pairFeatures(g: Graph, r: ArrayLike<number>, u: number, v: number, out: Float64Array, isEdge: boolean) {
	out[0] = isEdge ? 1 : 0;
	out[1] = lt(r[u], r[v]) ? 1 : 0;
	out[2] = eq(r[u], r[v]) ? 1 : 0;
	out[3] = gt(r[u], r[v]) ? 1 : 0;
	out[4] = u < v ? 1 : 0;
	for (let i = 5; i < 15; i++) out[i] = 0;
	const ru = r[u];
	const rv = r[v];
	for (const k of g.adj[v]) {
		const rk = r[k];
		const before = rk < rv;
		const t0 = rk < ru;
		const t1 = rk === ru && rk < INF;
		const t2 = rk > ru;
		const t3 = t1 && k < u;
		const t4 = k < u;
		if (t0) {
			out[5] = 1;
			if (before) out[6] = 1;
		}
		if (t1) {
			out[7] = 1;
			if (before) out[8] = 1;
		}
		if (t2) {
			out[9] = 1;
			if (before) out[10] = 1;
		}
		if (t3) {
			out[11] = 1;
			if (before) out[12] = 1;
		}
		if (t4) {
			out[13] = 1;
			if (before) out[14] = 1;
		}
	}
	out[15] = 1;
}

const dot = (x: ArrayLike<number>, w: ArrayLike<number>) => {
	let s = 0;
	for (let i = 0; i < w.length; i++) s += x[i] * w[i];
	return s;
};

/** the 'before' rule's 6 features of (a, b) from (parent's place, index) */
export function beforeFeatures(pp: ArrayLike<number>, a: number, b: number, out: Float64Array) {
	const e = eq(pp[a], pp[b]);
	out[0] = lt(pp[a], pp[b]) ? 1 : 0;
	out[1] = e ? 1 : 0;
	out[2] = gt(pp[a], pp[b]) ? 1 : 0;
	out[3] = a < b ? 1 : 0;
	out[4] = e && a < b ? 1 : 0;
	out[5] = 1;
}

// ---------------------------------------------------------------------------------------------------- the fit --
/** the training pairs at the completed answers (dataset('order', ...)) and the 'before' pairs */
export function datasets(train: { g: Graph; par: ArrayLike<number> }[]) {
	const X: number[][] = [];
	const y: number[] = [];
	const Xb: number[][] = [];
	const yb: number[] = [];
	const f = new Float64Array(16);
	const fb = new Float64Array(6);
	for (const { g, par } of train) {
		const { place, pp } = completion(par, g.s);
		const edge = new Set<number>();
		g.adj.forEach((a, u) => a.forEach((v) => edge.add(u * g.n + v)));
		for (let u = 0; u < g.n; u++)
			for (let v = 0; v < g.n; v++) {
				if (u === v) continue;
				beforeFeatures(pp, u, v, fb);
				Xb.push(Array.from(fb));
				yb.push(place[u] < place[v] ? 1 : 0);
				if (v === g.s) continue;
				pairFeatures(g, place, u, v, f, edge.has(u * g.n + v));
				X.push(Array.from(f));
				y.push(par[v] === u ? 1 : 0);
			}
	}
	return { X, y, Xb, yb };
}

/** Newton's method on L2-regularised logistic regression, 60 steps (logreg in Python) */
export function logreg(X: number[][], y: number[], steps = 60, l2 = 1e-3): number[] {
	const d = X[0].length;
	const N = y.length;
	const w = new Array(d).fill(0);
	for (let it = 0; it < steps; it++) {
		const H = Array.from({ length: d }, () => new Array(d).fill(0));
		const g = new Array(d).fill(0);
		for (let i = 0; i < N; i++) {
			const x = X[i];
			const z = Math.max(-30, Math.min(30, dot(x, w)));
			const p = 1 / (1 + Math.exp(-z));
			const q = p * (1 - p);
			for (let a = 0; a < d; a++) {
				if (x[a] === 0) continue;
				g[a] += x[a] * (p - y[i]);
				const xa = x[a] * q;
				for (let b = 0; b < d; b++) H[a][b] += xa * x[b];
			}
		}
		for (let a = 0; a < d; a++) {
			g[a] = g[a] / N + l2 * w[a];
			for (let b = 0; b < d; b++) H[a][b] = H[a][b] / N + (a === b ? l2 : 0);
		}
		const step = solve(H, g);
		for (let a = 0; a < d; a++) w[a] -= step[a];
	}
	return w;
}

/** Gaussian elimination with partial pivoting (LAPACK gesv's method) */
function solve(A: number[][], b: number[]): number[] {
	const n = b.length;
	const M = A.map((r, i) => [...r, b[i]]);
	for (let c = 0; c < n; c++) {
		let p = c;
		for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
		[M[c], M[p]] = [M[p], M[c]];
		for (let r = c + 1; r < n; r++) {
			const f = M[r][c] / M[c][c];
			if (f === 0) continue;
			for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
		}
	}
	const x = new Array(n).fill(0);
	for (let r = n - 1; r >= 0; r--) {
		let s = M[r][n];
		for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
		x[r] = s / M[r][r];
	}
	return x;
}

export interface FitResult {
	w: number[];
	wb: number[];
	gate: number[];
	ms: number;
	pairs: number;
}

/** fit both rules at the completed answers of the training graphs, and the gate (own bits true on every parent) */
export function fit(train: { g: Graph; par: ArrayLike<number> }[]): FitResult {
	const t0 = performance.now();
	const { X, y, Xb, yb } = datasets(train);
	const wb = logreg(Xb, yb);
	const w = logreg(X, y);
	const gate: number[] = [];
	for (let j = 0; j < 5; j++) {
		let all = true;
		for (let i = 0; i < y.length; i++) if (y[i] > 0 && X[i][j] !== 1) all = false;
		if (all) gate.push(j);
	}
	return { w, wb, gate, ms: performance.now() - t0, pairs: y.length };
}

// ---------------------------------------------------------------------------------------------------- the run --
export interface LoopState {
	loop: number;
	par: Int32Array;
	place: Float64Array;
	newly: number[]; // nodes that got their parent in this loop
}

/**
 * run(kind='order', top=True, gate) from Python: places start at "not yet" except the source; a node keeps its parent;
 * the gate keeps only pairs with every gated own bit (edge, u before v). Returns the parents and the last loop index.
 */
export function run(g: Graph, w: number[], wb: number[], gate: number[], onLoop?: (s: LoopState) => void) {
	const n = g.n;
	const s = g.s;
	const par = new Int32Array(n).fill(-1);
	par[s] = s;
	let reg = new Float64Array(n).fill(INF);
	reg[s] = 0;
	const f = new Float64Array(16);
	const fb = new Float64Array(6);
	// The gate requires an edge and u placed before v, so every other u scores -1e9 and can only win when no
	// neighbour passes, in which case the score is not positive and nothing changes. Scanning the neighbours of v in
	// index order with a strict comparison therefore picks the same u as the dense argmax (first index on ties).
	if (!(gate.length === 2 && gate[0] === 0 && gate[1] === 1)) throw new Error('run expects the gate [edge, before]');
	let loop = 0;
	const loops = Math.min(n + 2, 80);
	for (loop = 0; loop < loops; loop++) {
		const newPar = par.slice();
		let changed = false;
		const newly: number[] = [];
		for (let v = 0; v < n; v++) {
			if (v === s || par[v] >= 0) continue;
			let bu = -1;
			let bs = -Infinity;
			for (const u of g.adj[v]) {
				if (!(reg[u] < reg[v])) continue;
				pairFeatures(g, reg, u, v, f, true);
				const sc = dot(f, w);
				if (sc > bs) {
					bs = sc;
					bu = u;
				}
			}
			if (bu >= 0 && bs > 0 && newPar[v] !== bu) {
				newPar[v] = bu;
				changed = true;
				newly.push(v);
			}
		}
		par.set(newPar);
		// place = number of nodes before; 'before' is the fitted rule on (parent's place, index)
		const pp = new Float64Array(n);
		for (let v = 0; v < n; v++) pp[v] = par[v] >= 0 && reg[Math.max(par[v], 0)] < INF ? reg[Math.max(par[v], 0)] : INF;
		pp[s] = -1;
		const nw = new Float64Array(n).fill(INF);
		for (let b = 0; b < n; b++) {
			if (!(pp[b] < INF)) continue;
			let cnt = 0;
			for (let a = 0; a < n; a++) {
				if (a === b || !(pp[a] < INF)) continue;
				beforeFeatures(pp, a, b, fb);
				if (dot(fb, wb) > 0) cnt++;
			}
			nw[b] = cnt;
		}
		nw[s] = 0;
		onLoop?.({ loop, par: par.slice(), place: nw.slice(), newly });
		let same = true;
		for (let v = 0; v < n; v++) if (nw[v] !== reg[v]) same = false;
		if (!changed && same) break;
		reg = nw;
	}
	return { par, loops: loop };
}

// ---------------------------------------------------------------------------------------------- graphs --
/** mulberry32: the page's own seeded generator (the Python graphs come with the data) */
export function rng(seed: number) {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/** ugraph from Python, with the page's generator: a random tree plus about `extra` random edges per node */
export function ugraph(n: number, seed: number, extra = 1.0): Graph {
	const R = rng(seed);
	const order = Array.from({ length: n }, (_, i) => i);
	for (let i = n - 1; i > 0; i--) {
		const j = Math.floor(R() * (i + 1));
		[order[i], order[j]] = [order[j], order[i]];
	}
	const E = new Set<number>();
	const add = (a: number, b: number) => {
		if (a !== b) E.add(Math.min(a, b) * n + Math.max(a, b));
	};
	for (let i = 1; i < n; i++) add(order[i], order[Math.floor(R() * i)]);
	const p = (2 * extra) / n;
	for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) if (R() < p) add(a, b);
	const edges: [number, number][] = [...E].map((k) => [Math.floor(k / n), k % n]);
	return fromEdges(n, Math.floor(R() * n), edges);
}
