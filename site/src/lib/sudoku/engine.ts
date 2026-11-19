/**
 * The fitted Sudoku check, run cold and then warm: a line-for-line port of code/bench/engine.py and
 * code/bench/check.py, so that the page gives the same boards, moves and operation counts as the Python runs.
 *
 * Cold derivation: bounds on the fitted count conditions ("there is one", "at most one") over every unit of the kept
 * relations, repeated until nothing changes. Warm search: the open cell with the fewest values, its values in a
 * random order (xorshift64, seeded as in Python), the cold derivation after each choice, undo on a contradiction.
 * The stop is the fitted check itself: the energy of the full board at or below tau.
 */

export interface SudokuCheck {
	types: number[]; // the 25 pair types seen on 9 x 9 (8-bit agreement patterns)
	w: number[]; // fitted cost per type (>= 0)
	relations: number[][]; // kept relations, as lists of coordinate functions
	conditions: [boolean, boolean][]; // per relation: (n_v >= 1, n_v <= 1)
	tau: number;
}

export interface Engine {
	b: number;
	V: number;
	M: number;
	unitItems: Int32Array[]; // cells of each unit
	itemUnits: Int32Array[]; // units of each cell, in unit order
	ge1: Uint8Array;
	le1: Uint8Array;
	unitRel: Int32Array; // relation index of each unit
	Wi: Int32Array;
	Wj: Int32Array;
	Ww: Float64Array;
	tau: number;
}

/** the eight coordinate functions r, c, r//b, c//b, r%b, c%b, r-c, r+c of every cell */
export function coordFuncs(b: number): Int32Array[] {
	const n = b * b;
	const out: Int32Array[] = [];
	for (let i = 0; i < n * n; i++) {
		const r = Math.floor(i / n);
		const c = i % n;
		out.push(Int32Array.of(r, c, Math.floor(r / b), Math.floor(c / b), r % b, c % b, r - c, r + c));
	}
	return out;
}

export function pairType(fi: Int32Array, fj: Int32Array): number {
	let t = 0;
	for (let k = 0; k < 8; k++) if (fi[k] === fj[k]) t |= 1 << k;
	return t;
}

/** every class of every kept relation, in numpy's order (relation, then class keys sorted lexicographically) */
export function units(b: number, rels: number[][]): { rel: number; cells: number[] }[] {
	const f = coordFuncs(b);
	const out: { rel: number; cells: number[] }[] = [];
	rels.forEach((fs, k) => {
		const keyOf = (i: number) => fs.map((x) => f[i][x]);
		const map = new Map<string, { key: number[]; cells: number[] }>();
		for (let i = 0; i < f.length; i++) {
			const key = keyOf(i);
			const s = key.join(',');
			if (!map.has(s)) map.set(s, { key, cells: [] });
			map.get(s)!.cells.push(i);
		}
		const cls = [...map.values()].sort((a, b2) => {
			for (let j = 0; j < a.key.length; j++) if (a.key[j] !== b2.key[j]) return a.key[j] - b2.key[j];
			return 0;
		});
		for (const c of cls) out.push({ rel: k, cells: c.cells });
	});
	return out;
}

export function build(check: SudokuCheck, b: number): Engine {
	const V = b * b;
	const M = V * V;
	const U = units(b, check.relations);
	const unitItems = U.map((u) => Int32Array.from(u.cells));
	const iu: number[][] = Array.from({ length: M }, () => []);
	U.forEach((u, k) => u.cells.forEach((c) => iu[c].push(k)));
	const ge1 = Uint8Array.from(U.map((u) => (check.conditions[u.rel][0] ? 1 : 0)));
	const le1 = Uint8Array.from(U.map((u) => (check.conditions[u.rel][1] ? 1 : 0)));
	// the check's costs on this board: w of each pair's type (float32, as in Python), 0 for types never fitted
	const wt = new Map<number, number>();
	check.types.forEach((t, k) => wt.set(t, Math.fround(check.w[k])));
	const f = coordFuncs(b);
	const Wi: number[] = [];
	const Wj: number[] = [];
	const Ww: number[] = [];
	for (let i = 0; i < M; i++)
		for (let j = i + 1; j < M; j++) {
			const w = wt.get(pairType(f[i], f[j])) ?? 0;
			if (w > 0) {
				Wi.push(i);
				Wj.push(j);
				Ww.push(w);
			}
		}
	return {
		b,
		V,
		M,
		unitItems,
		itemUnits: iu.map((x) => Int32Array.from(x)),
		ge1,
		le1,
		unitRel: Int32Array.from(U.map((u) => u.rel)),
		Wi: Int32Array.from(Wi),
		Wj: Int32Array.from(Wj),
		Ww: Float64Array.from(Ww),
		tau: check.tau
	};
}

export function popcount(x: number): number {
	let c = 0;
	while (x) {
		x &= x - 1;
		c++;
	}
	return c;
}

export function masks(puzzle: ArrayLike<number>, V: number): Int32Array {
	const full = V >= 31 ? -1 : (1 << V) - 1;
	return Int32Array.from({ length: puzzle.length }, (_, i) => (puzzle[i] > 0 ? 1 << (puzzle[i] - 1) : full));
}

export function decode(dom: Int32Array): number[] {
	return Array.from(dom, (d) => (d && (d & (d - 1)) === 0 ? 31 - Math.clz32(d) + 1 : 0));
}

const MASK64 = (1n << 64n) - 1n;

/** xorshift64 as in engine.py (uint64 arithmetic) */
export class XorShift {
	s: bigint;
	constructor(seed: number) {
		this.s = BigInt.asUintN(64, BigInt(seed) * 2654435761n + 1n);
	}
	pick(c: number): number {
		let s = this.s;
		s = (s ^ (s << 13n)) & MASK64;
		s = s ^ (s >> 7n);
		s = (s ^ (s << 17n)) & MASK64;
		this.s = s;
		return Number(s % BigInt(c));
	}
}

export type Stats = { ops: number; cmp: number };

/**
 * Bounds on the count conditions until nothing changes; a port of engine.propagate (same queue order, same counts).
 * inq: flags of units to scan. onFix(cell) is called when a cell is left with one value.
 */
export function propagate(
	dom: Int32Array,
	E: Engine,
	inq: Uint8Array,
	queue: Int32Array,
	st: Stats,
	onFix?: (cell: number) => void
): boolean {
	const nu = E.unitItems.length;
	let head = 0;
	let tail = 0;
	for (let u = 0; u < nu; u++) if (inq[u]) queue[tail++] = u;
	let size = tail;
	tail = tail % nu;
	const push = (it: number) => {
		const us = E.itemUnits[it];
		for (let j = 0; j < us.length; j++) {
			const w = us[j];
			if (!inq[w]) {
				inq[w] = 1;
				queue[tail] = w;
				tail = (tail + 1) % nu;
				size++;
			}
		}
	};
	while (size > 0) {
		const u = queue[head];
		head = (head + 1) % nu;
		size--;
		inq[u] = 0;
		const items = E.unitItems[u];
		const s = items.length;
		const g = E.ge1[u];
		const l = E.le1[u];
		for (let v = 0; v < E.V; v++) {
			const bit = 1 << v;
			let L = 0;
			let U = 0;
			let last = -1;
			for (let k = 0; k < s; k++) {
				const it = items[k];
				const d = dom[it];
				st.cmp++;
				if (d & bit) {
					U++;
					st.ops++;
					last = it;
					st.cmp++;
					if (d === bit) {
						L++;
						st.ops++;
					}
				}
			}
			st.cmp += 4;
			if (l && L >= 2) return false;
			if (g && U === 0) return false;
			if (l && L === 1 && U > 1) {
				for (let k = 0; k < s; k++) {
					const it = items[k];
					const d = dom[it];
					st.cmp++;
					if (d & bit && d !== bit) {
						const nd = d & ~bit;
						if (nd === 0) return false;
						dom[it] = nd;
						if (onFix && (nd & (nd - 1)) === 0) onFix(it);
						push(it);
					}
				}
			} else if (g && U === 1 && dom[last] !== bit) {
				dom[last] = bit;
				if (onFix) onFix(last);
				push(last);
			}
		}
	}
	return true;
}

export function energy(dom: Int32Array, E: Engine, st: Stats): number {
	let e = 0;
	for (let p = 0; p < E.Wi.length; p++) {
		st.cmp++;
		if (dom[E.Wi[p]] === dom[E.Wj[p]]) {
			e += E.Ww[p];
			st.ops++;
		}
	}
	return e;
}

export type Status = 1 | 0 | -1; // solved, out of budget, no solution

/**
 * engine.solve as a resumable machine: cold() runs the first derivation, warm(pauseAt) advances the depth-first
 * search until `pauseAt` moves (a pause sits at the top of the loop, so resuming is exactly the uninterrupted run).
 */
export class Solver {
	E: Engine;
	n: number;
	given: Int32Array;
	dom: Int32Array; // the latest board (masks)
	coldFixed: number[] = []; // cells the cold stage fixed, in order
	st: Stats = { ops: 0, cmp: 0 };
	moves = 0;
	maxDepth = 0;
	stopTests = 0;
	rejected = 0;
	status: Status | null = null; // null while running
	coldOnly = false;
	private inq: Uint8Array;
	private queue: Int32Array;
	private rng: XorShift;
	private budget: number;
	private stackDom: Int32Array[] = [];
	private stackItem: Int32Array;
	private stackLeft: Int32Array;
	private depth = 0;
	private domDepth = -1; // the search level at which `dom` was made (-1: the cold stage)
	private started = false;

	constructor(E: Engine, puzzleMasks: Int32Array, seed: number, budget = 1e6) {
		this.E = E;
		this.n = puzzleMasks.length;
		this.given = puzzleMasks.slice();
		this.dom = puzzleMasks.slice();
		this.inq = new Uint8Array(E.unitItems.length).fill(1);
		this.queue = new Int32Array(E.unitItems.length);
		this.rng = new XorShift(seed);
		this.budget = budget;
		this.stackItem = new Int32Array(this.n + 1);
		this.stackLeft = new Int32Array(this.n + 1);
	}

	private best(dom: Int32Array, count: boolean): number {
		let best = -1;
		let bc = 1 << 30;
		for (let i = 0; i < this.n; i++) {
			const c = popcount(dom[i]);
			if (count) this.st.cmp++;
			if (c > 1 && c < bc) {
				bc = c;
				best = i;
			}
		}
		return best;
	}

	/** the cold stage; returns true when the run is already over (solved by the cold stage, or no solution) */
	cold(): boolean {
		const ok = propagate(this.dom, this.E, this.inq, this.queue, this.st, (c) => this.coldFixed.push(c));
		if (!ok) {
			this.status = -1;
			return true;
		}
		let done = true;
		for (let i = 0; i < this.n; i++)
			if (popcount(this.dom[i]) > 1) {
				done = false;
				break;
			}
		if (done) {
			this.stopTests++;
			if (energy(this.dom, this.E, this.st) <= this.E.tau) {
				this.status = 1;
				this.coldOnly = true;
				return true;
			}
			this.rejected++;
		}
		const b = this.best(this.dom, true);
		this.stackDom[0] = this.dom.slice();
		this.stackItem[0] = b;
		this.stackLeft[0] = this.dom[b];
		this.depth = 0;
		this.started = true;
		return false;
	}

	/** advance the warm search until `pauseAt` moves or the end; returns true at the end */
	warm(pauseAt = Infinity): boolean {
		if (this.status !== null) return true;
		if (!this.started) throw new Error('cold() first');
		const E = this.E;
		while (this.depth >= 0) {
			if (this.moves >= pauseAt) return false;
			if (this.moves >= this.budget) {
				this.status = 0;
				return true;
			}
			const depth = this.depth;
			const left = this.stackLeft[depth];
			if (left === 0) {
				this.depth--;
				continue;
			}
			const c = popcount(left);
			const pick = this.rng.pick(c);
			let x = left;
			for (let k = 0; k < pick; k++) x &= x - 1;
			const bit = x & -x;
			this.stackLeft[depth] = left & ~bit;
			this.moves++;
			const dom = this.stackDom[depth].slice();
			const it = this.stackItem[depth];
			dom[it] = bit;
			const us = E.itemUnits[it];
			for (let j = 0; j < us.length; j++) this.inq[us[j]] = 1;
			const ok = propagate(dom, E, this.inq, this.queue, this.st);
			this.inq.fill(0);
			this.dom = dom;
			this.domDepth = depth;
			if (!ok) continue;
			const best = this.best(dom, true);
			if (best < 0) {
				this.stopTests++;
				if (energy(dom, E, this.st) <= E.tau) {
					this.status = 1;
					return true;
				}
				this.rejected++;
				continue;
			}
			this.depth = depth + 1;
			if (this.depth + 1 > this.maxDepth) this.maxDepth = this.depth + 1;
			this.stackDom[this.depth] = dom;
			this.stackItem[this.depth] = best;
			this.stackLeft[this.depth] = dom[best];
		}
		this.status = -1;
		return true;
	}

	/** the whole run at once (cold, then warm to the end) */
	run(): this {
		if (!this.cold()) this.warm();
		return this;
	}

	/** the cells whose values were guessed to make the current board (the branch of the search it lies on) */
	branch(): number[] {
		return Array.from(this.stackItem.subarray(0, this.domDepth + 1));
	}

	/** depth of the search stack (0 before the search) */
	get level(): number {
		return Math.max(0, this.depth);
	}
}
