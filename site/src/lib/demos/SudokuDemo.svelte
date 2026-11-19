<script lang="ts">
	import { onDestroy } from 'svelte';
	import { build, masks, popcount, Solver, type Engine, type SudokuCheck } from '#lib/sudoku/engine.ts';
	import Play from '#lib/ui/Play.svelte';
	import { int } from '#lib/format.ts';
	import { whenVisible } from '#lib/ui/reveal.ts';
	import { C, stopFrame } from '#lib/theme.ts';

	type Py = { status: number; moves: number; ops?: number; cold?: boolean; restarts?: number; cold_fixed?: number };
	type Puzzle = { index: number; puzzle: number[]; seed: number; rating?: number; solution?: number[]; py: Py };
	export type SudokuData = {
		check: SudokuCheck;
		demo: { watch: number; median: number; hardest: number };
		extreme: Puzzle[];
		sizes: Record<string, Puzzle[]>;
		stats: {
			extreme: { n_test: number; correct: number; cold_only: number; moves_median: number; fit_ms_warm_median7: number }[];
			side4: { n: number; solved_valid: number };
			side16: { n: number; solved_valid: number; moves_median: number };
			side25: { n: number; solved_valid: number; budget_misses: number; moves_median: number };
		};
	};

	type Mode = 'watch' | 'median' | 'hardest' | '4' | '16' | '25';
	let {
		data,
		initial = 'watch',
		bare = false,
		autoplay = false
	}: { data: SudokuData; initial?: Mode; bare?: boolean; autoplay?: boolean } = $props();

	// svelte-ignore state_referenced_locally
	let mode = $state<Mode>(initial);
	const pick = (m: Mode): Puzzle | null => {
		if (m === 'watch' || m === 'median' || m === 'hardest') return data.extreme.find((p) => p.index === data.demo[m]) ?? null;
		const list = data.sizes[m] ?? [];
		if (m === '16') return list.find((p) => p.py.moves > 100 && p.py.moves < 1000) ?? list[0] ?? null;
		return list[0] ?? null;
	};
	const sideOf = (m: Mode) => (m === '4' ? 4 : m === '16' ? 16 : m === '25' ? 25 : 9);
	const puzzle = $derived(pick(mode));
	const side = $derived(sideOf(mode));
	const b = $derived(Math.round(Math.sqrt(side)));
	const engines: Record<number, Engine> = {};
	const engineFor = (bb: number) => (engines[bb] ??= build(data.check, bb));
	const recorded = $derived(puzzle ? puzzle.py : null);

	// the board shown: value per cell and its kind (0 open, 1 clue, 2 forced cold, 3 set by search)
	let values = $state.raw<number[]>([]);
	let kinds = $state.raw<number[]>([]);
	let phase = $state<'idle' | 'running' | 'done'>('idle');
	let paused = $state(false);
	let moves = $state(0);
	let restarts = $state(0);
	let coldCount = $state(0);
	let warmCount = $state(0); // guessed cells on the current branch
	let derivedCount = $state(0); // cells the cold derivation fixed after those guesses
	let computeMs = $state(0);
	let status = $state('');
	let trace = $state.raw<[number, number, number][]>([]); // (step, open cells, 0 cold / 1 warm)
	let solver: Solver | null = null;
	let raf = 0;

	function reset(p: Puzzle | null) {
		stopFrame(raf);
		solver = null;
		phase = 'idle';
		paused = false;
		moves = 0;
		coldCount = 0;
		restarts = 0;
		warmCount = 0;
		derivedCount = 0;
		computeMs = 0;
		status = '';
		trace = [];
		if (!p) {
			values = [];
			kinds = [];
			return;
		}
		values = p.puzzle.slice();
		kinds = p.puzzle.map((v) => (v > 0 ? 1 : 0));
	}
	function choose(m: Mode) {
		if (m === mode) return;
		mode = m;
		reset(pick(m));
	}
	// svelte-ignore state_referenced_locally
	reset(pick(initial));

	const open = (dom: Int32Array) => {
		let k = 0;
		for (const d of dom) if (popcount(d) > 1) k++;
		return k;
	};

	function show(dom: Int32Array, coldSet: Set<number>, clues: number[], guessed: Set<number>) {
		const v: number[] = new Array(dom.length);
		const k: number[] = new Array(dom.length);
		let w = 0;
		let dv = 0;
		for (let i = 0; i < dom.length; i++) {
			const d = dom[i];
			const single = d && (d & (d - 1)) === 0;
			v[i] = single ? 32 - Math.clz32(d) : 0;
			k[i] = clues[i] > 0 ? 1 : !single ? 0 : coldSet.has(i) ? 2 : guessed.has(i) ? 3 : 4;
			if (k[i] === 3) w++;
			if (k[i] === 4) dv++;
		}
		values = v;
		kinds = k;
		warmCount = w;
		derivedCount = dv;
	}

	function play() {
		if (!puzzle) return;
		if (phase === 'running') {
			paused = !paused;
			if (paused) pausedAt = performance.now();
			else {
				const d = performance.now() - pausedAt;
				tStart += d;
				if (warmStart >= 0) warmStart += d;
				raf = requestAnimationFrame(tick);
			}
			return;
		}
		reset(puzzle);
		const p = puzzle;
		const E = engineFor(b);
		solver = new Solver(E, masks(p.puzzle, side), p.seed);
		const t0 = performance.now();
		const ended = solver.cold();
		computeMs = performance.now() - t0;
		const coldOrder = solver.coldFixed.slice();
		const coldSet = new Set(coldOrder);
		const clues = p.puzzle;
		const startOpen = p.puzzle.filter((v) => v === 0).length;
		trace = [[0, startOpen, 0]];
		phase = 'running';
		// cold stage: reveal the forced cells in the order the derivation fixed them
		const revealMs = Math.min(1600, 40 + 22 * coldOrder.length);
		tStart = performance.now();
		warmStart = -1;
		const total = recorded?.moves ?? 1000;
		// the search is paced by the clock, not by frames: about 3 s for a short search, up to 9 s for a long one
		const warmMs = total < 400 ? 3000 : total < 3000 ? 5000 : 9000;
		let shown = 0;
		const step = () => {
			if (!solver) return;
			if (paused) return;
			const t = performance.now() - tStart;
			if (shown < coldOrder.length) {
				const want = Math.min(coldOrder.length, Math.ceil((t / revealMs) * coldOrder.length));
				const v = values.slice();
				const k = kinds.slice();
				for (; shown < want; shown++) {
					const c = coldOrder[shown];
					v[c] = 32 - Math.clz32(solver.dom[c]);
					k[c] = 2;
				}
				values = v;
				kinds = k;
				coldCount = shown;
				trace = [...trace, [shown, startOpen - shown, 0]];
				raf = requestAnimationFrame(step);
				return;
			}
			if (ended) {
				finish();
				return;
			}
			// warm stage: advance the search a few moves per frame, timing only the solver
			const t1 = performance.now();
			let done = false;
			if (warmStart < 0) warmStart = performance.now();
			const due = Math.max(1, Math.ceil((total * (performance.now() - warmStart)) / warmMs));
			const limit = Math.max(solver.moves + 1, due);
			while (!done && solver.moves < limit && performance.now() - t1 < 14) done = solver.warm(Math.min(limit, solver.moves + 64));
			computeMs += performance.now() - t1;
			moves = solver.moves;
		restarts = solver.restarts;
			restarts = solver.restarts;
			show(solver.dom, coldSet, clues, new Set(solver.branch()));
			const o = open(solver.dom);
			trace = trace.length > 900 ? [...trace.filter((_, i) => i % 2 === 0), [coldOrder.length + moves, o, 1]] : [...trace, [coldOrder.length + moves, o, 1]];
			if (done) finish();
			else raf = requestAnimationFrame(step);
		};
		tick = step;
		raf = requestAnimationFrame(step);
	}
	let tick: () => void = () => {};
	let tStart = 0;
	let warmStart = -1;
	let pausedAt = 0;

	function valid(v: number[]): boolean {
		const n = side;
		const ok = (cells: number[]) => new Set(cells.map((i) => v[i])).size === n && cells.every((i) => v[i] >= 1 && v[i] <= n);
		for (let r = 0; r < n; r++) {
			if (!ok(Array.from({ length: n }, (_, c) => r * n + c))) return false;
			if (!ok(Array.from({ length: n }, (_, c) => c * n + r))) return false;
		}
		for (let br = 0; br < b; br++)
			for (let bc = 0; bc < b; bc++) {
				const cells: number[] = [];
				for (let r = 0; r < b; r++) for (let c = 0; c < b; c++) cells.push((br * b + r) * n + bc * b + c);
				if (!ok(cells)) return false;
			}
		return true;
	}

	function finish() {
		if (!solver || !puzzle) return;
		const coldSet = new Set(solver.coldFixed);
		show(solver.dom, coldSet, puzzle.puzzle, new Set(solver.branch()));
		moves = solver.moves;
		trace = [...trace, [solver.coldFixed.length + moves, 0, moves ? 1 : 0]];
		phase = 'done';
		const sameAsPython = recorded ? solver.moves === recorded.moves : false;
		status =
			solver.status === 1
				? `${valid(values) ? 'Solved: every row, column and box holds each digit once' : 'Stopped'}${sameAsPython ? '; same moves as the Python run' : ''}.`
				: solver.status === 0
					? 'Out of budget.'
					: 'No solution.';
	}

	onDestroy(() => stopFrame(raf));

	// geometry
	const size = 400;
	const cell = $derived(size / side);
	const font = $derived(side <= 9 ? cell * 0.5 : side <= 16 ? cell * 0.52 : cell * 0.5);
	const DERIVED = '#6A8CC8'; // cobalt, lighter: fixed cold after a guess
	const colour = (k: number) => (k === 1 ? C.ink : k === 2 ? C.cobalt : k === 3 ? C.rose : k === 4 ? DERIVED : C.mist);

	const chart = $derived.by(() => {
		if (trace.length < 2) return { cold: '', warm: '', x0: 0 };
		const W = 240;
		const H = 64;
		const maxX = Math.max(1, trace[trace.length - 1][0]);
		const maxY = Math.max(1, trace[0][1]);
		const pt = (s: number, o: number) => `${((s / maxX) * W).toFixed(1)},${(H - (o / maxY) * H).toFixed(1)}`;
		const cold = trace.filter((t) => t[2] === 0).map((t) => pt(t[0], t[1]));
		const lastCold = trace.filter((t) => t[2] === 0).at(-1)!;
		const warm = [lastCold, ...trace.filter((t) => t[2] === 1)].map((t) => pt(t[0], t[1]));
		return { cold: cold.join(' '), warm: warm.length > 1 ? warm.join(' ') : '', x0: (lastCold[0] / maxX) * W };
	});
	const medianMoves = $derived(
		side === 16 ? data.stats.side16.moves_median : side === 25 ? data.stats.side25.moves_median : side === 4 ? null : data.stats.extreme[0].moves_median
	);
	const has25 = $derived((data.sizes['25'] ?? []).length > 0);
</script>

<div class="wrap" {@attach (el) => (autoplay ? whenVisible(el, play) : undefined)}>
	<div class="board">
		<svg viewBox="-1 -1 {size + 2} {size + 2}" role="img" aria-label="Sudoku board {side} by {side}">
			{#each values as v, i (i)}
				{#if v > 0}
					<text
						x={(i % side) * cell + cell / 2}
						y={Math.floor(i / side) * cell + cell / 2}
						font-size={font}
						fill={colour(kinds[i])}
						font-weight={kinds[i] === 3 ? 650 : kinds[i] === 1 ? 520 : 400}
						class={{ pop: kinds[i] === 2 }}
						dominant-baseline="central"
						text-anchor="middle">{v}</text
					>
				{/if}
			{/each}
			{#each Array.from({ length: side - 1 }, (_, k) => k + 1) as k (k)}
				<line x1={k * cell} y1="0" x2={k * cell} y2={size} stroke={k % b === 0 ? C.blueGrey : C.edge} stroke-width={k % b === 0 ? 1 : 0.75} />
				<line y1={k * cell} x1="0" y2={k * cell} x2={size} stroke={k % b === 0 ? C.blueGrey : C.edge} stroke-width={k % b === 0 ? 1 : 0.75} />
			{/each}
			<rect x="0" y="0" width={size} height={size} fill="none" stroke={C.ink} stroke-width="1.2" rx="2" />
		</svg>
	</div>
	<div class="side">
		{#if !bare}<div class="controls">
			<div class="seg" role="group" aria-label="Puzzle">
				<button aria-pressed={mode === 'watch'} onclick={() => choose('watch')}>Test #{data.demo.watch}</button>
				<button aria-pressed={mode === 'median'} onclick={() => choose('median')}>Median</button>
				<button aria-pressed={mode === 'hardest'} onclick={() => choose('hardest')}>Hardest</button>
			</div>
			<div class="seg" role="group" aria-label="Board size">
				<button aria-pressed={mode === '4'} onclick={() => choose('4')}>4 × 4</button>
				<button aria-pressed={['watch', 'median', 'hardest'].includes(mode)} onclick={() => choose('watch')}>9 × 9</button>
				<button aria-pressed={mode === '16'} onclick={() => choose('16')}>16 × 16</button>
				<button aria-pressed={mode === '25'} disabled={!has25} onclick={() => choose('25')}>25 × 25</button>
			</div>
		</div>{/if}
		<div class="play">
			<Play phase={phase === 'running' && paused ? 'idle' : phase} onclick={play} />
			<span class="note">
				{#if puzzle && side === 9}Sudoku-Extreme test puzzle {int(puzzle.index)} of {int(data.stats.extreme[0].n_test)}
				{:else if puzzle}Generated {side} × {side} puzzle {puzzle.index + 1}, same 9 × 9 rule{#if medianMoves !== null}; the benchmark run's median is {int(medianMoves)} moves{/if}{/if}
			</span>
		</div>
		<dl class="readout">
			<dt>Clues</dt>
			<dd>{puzzle ? int(puzzle.puzzle.filter((v) => v > 0).length) : ''} of {int(side * side)}</dd>
			<dt><span class="key" style:background={C.cobalt}></span>Forced cold</dt>
			<dd>{int(coldCount)}{#if recorded?.cold_fixed !== undefined}&nbsp;<span class="py">(Python {int(recorded.cold_fixed)})</span>{/if}</dd>
			<dt><span class="key" style:background={C.rose}></span>Guessed by the search</dt>
			<dd>{int(warmCount)}</dd>
			<dt><span class="key" style:background={DERIVED}></span>Forced cold after a guess</dt>
			<dd>{int(derivedCount)}</dd>
			<dt>Search moves</dt>
			<dd>{int(moves)}{#if recorded}&nbsp;<span class="py">(Python {int(recorded.moves)})</span>{/if}</dd>
			<dt>Restarts</dt>
			<dd>{int(restarts)}{#if recorded?.restarts !== undefined}&nbsp;<span class="py">(Python {int(recorded.restarts)})</span>{/if}</dd>
			<dt>Solver time</dt>
			<dd>{computeMs < 10 ? computeMs.toFixed(2) : int(computeMs)} ms</dd>
		</dl>
		<div class={['chart', { hidden: trace.length < 2 }]}>
			<div class="label">Open cells</div>
			<svg viewBox="-2 -2 244 70" preserveAspectRatio="none" aria-hidden="true">
				<line x1="0" y1="64" x2="240" y2="64" stroke={C.edge} stroke-width="1" vector-effect="non-scaling-stroke" />
				{#if chart.cold}<polyline points={chart.cold} fill="none" stroke={C.cobalt} stroke-width="1.4" vector-effect="non-scaling-stroke" />{/if}
				{#if chart.warm}<polyline points={chart.warm} fill="none" stroke={C.rose} stroke-width="1.2" vector-effect="non-scaling-stroke" />{/if}
			</svg>
			<div class="axis"><span>start</span><span>cold stage, then search moves</span></div>
		</div>
		<p class="status">{status}</p>
	</div>
</div>

<style>
	.wrap {
		display: grid;
		grid-template-columns: minmax(0, 400px) minmax(0, 1fr);
		gap: 28px;
		align-items: start;
	}
	@media (max-width: 720px) {
		.wrap {
			grid-template-columns: minmax(0, 1fr);
			gap: 18px;
		}
	}
	.board svg {
		display: block;
		width: 100%;
		height: auto;
		font-family: var(--font-sans);
		font-variant-numeric: tabular-nums;
	}
	.board text {
		transition: fill 250ms ease;
	}
	.pop {
		animation: fadein 420ms var(--ease);
	}
	@keyframes fadein {
		from {
			opacity: 0;
		}
	}
	.side {
		display: flex;
		flex-direction: column;
		gap: 16px;
		min-width: 0;
	}
	.controls {
		display: flex;
		flex-direction: column;
		gap: 8px;
		align-items: flex-start;
	}
	.seg button:disabled {
		color: var(--color-mist);
		cursor: default;
	}
	.play {
		display: flex;
		align-items: center;
		gap: 12px;
	}
	.note {
		font-size: 13px;
		color: var(--color-ink-soft);
	}
	.key {
		display: inline-block;
		width: 8px;
		height: 8px;
		margin-right: 7px;
		border-radius: 2px;
		vertical-align: 0;
	}
	.py {
		color: var(--color-blue-grey);
	}
	.chart svg {
		display: block;
		width: 100%;
		height: 64px;
		margin-top: 6px;
	}
	.axis {
		display: flex;
		justify-content: space-between;
		margin-top: 4px;
		font-size: 11px;
		color: var(--color-ink-soft);
	}
	.hidden {
		visibility: hidden;
	}
	.status {
		min-height: 1.5em;
		font-size: 13px;
		color: var(--color-ink);
	}
</style>
