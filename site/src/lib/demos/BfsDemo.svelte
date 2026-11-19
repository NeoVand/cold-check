<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { bfs, fit, fromEdges, run, ugraph, type FitResult, type Graph, type LoopState } from '#lib/bfs/rule.ts';
	import Play from '#lib/ui/Play.svelte';
	import IconButton from '#lib/ui/IconButton.svelte';
	import { DiceIcon } from '@hugeicons/core-free-icons';
	import { whenVisible } from '#lib/ui/reveal.ts';
	import { C, ease, fitCanvas, stopFrame } from '#lib/theme.ts';
	import { int } from '#lib/format.ts';

	export type BfsData = {
		w: number[];
		wb: number[];
		gate: number[];
		train: { n: number; s: number; edges: [number, number][]; par: number[] }[];
		record: { order: { fit_ms: number } };
	};
	let {
		data,
		initial = 4,
		autoplay = false
	}: { data: BfsData; /** index into the sizes 16 … 1,024 */ initial?: number; autoplay?: boolean } = $props();

	const SIZES = [16, 32, 64, 128, 256, 512, 1024];
	// svelte-ignore state_referenced_locally
	let sizeI = $state(initial);
	let seed = $state(7);
	const n = $derived(SIZES[sizeI]);

	let fitted = $state.raw<FitResult | null>(null);
	const sameAsPython = $derived(
		fitted ? fitted.w.every((x, i) => Math.abs(x - data.w[i]) < 1e-6 * (1 + Math.abs(data.w[i]))) : false
	);
	function doFit() {
		const train = data.train.map((t) => ({ g: fromEdges(t.n, t.s, t.edges), par: t.par }));
		fit(train); // warm up once, then time the second fit (as the Python timing does)
		fitted = fit(train);
	}
	onMount(doFit);

	const graph = $derived.by((): Graph => ugraph(n, seed));
	const label = $derived(bfs(graph).par);

	// layout: rings by true distance from the source, angles from the label tree (subtree leaf counts)
	const layout = $derived.by(() => {
		const g = graph;
		const { par, order } = bfs(g);
		const kids: number[][] = Array.from({ length: g.n }, () => []);
		for (const v of order) if (v !== g.s) kids[par[v]].push(v);
		const depth = new Int32Array(g.n);
		for (const v of order) if (v !== g.s) depth[v] = depth[par[v]] + 1;
		const leaves = new Float64Array(g.n);
		for (let k = order.length - 1; k >= 0; k--) {
			const v = order[k];
			leaves[v] = kids[v].length ? kids[v].reduce((a, c) => a + leaves[c], 0) : 1;
		}
		const ang = new Float64Array(g.n);
		const span = new Float64Array(g.n);
		ang[g.s] = 0;
		span[g.s] = Math.PI * 2;
		for (const v of order) {
			let a0 = ang[v] - span[v] / 2;
			for (const c of kids[v]) {
				span[c] = (span[v] * leaves[c]) / leaves[v];
				ang[c] = a0 + span[c] / 2;
				a0 += span[c];
			}
		}
		let dmax = 0;
		for (const d of depth) dmax = Math.max(dmax, d);
		return { ang, depth, dmax };
	});

	let loops = $state.raw<LoopState[]>([]);
	let runMs = $state(0);
	let shownLoop = $state(-1); // last loop drawn in full
	let frac = $state(0); // progress within the next loop
	let phase = $state<'idle' | 'running' | 'done'>('idle');
	let raf = 0;
	let box = $state(460);

	function reset() {
		stopFrame(raf);
		loops = [];
		shownLoop = -1;
		frac = 0;
		phase = 'idle';
	}
	function play() {
		if (!fitted || phase === 'running') return;
		reset();
		const states: LoopState[] = [];
		const t0 = performance.now();
		run(graph, fitted.w, fitted.wb, fitted.gate, (s) => states.push(s));
		runMs = performance.now() - t0;
		loops = states;
		phase = 'running';
		const per = 750;
		const tStart = performance.now();
		const step = () => {
			const t = (performance.now() - tStart) / per;
			shownLoop = Math.min(states.length - 1, Math.floor(t) - 1);
			frac = t - Math.floor(t);
			if (Math.floor(t) - 1 < states.length - 1) raf = requestAnimationFrame(step);
			else {
				shownLoop = states.length - 1;
				frac = 0;
				phase = 'done';
			}
		};
		raf = requestAnimationFrame(step);
	}
	onDestroy(() => stopFrame(raf));

	const current = $derived(shownLoop >= 0 ? loops[shownLoop] : null);
	const incoming = $derived(phase === 'running' && shownLoop + 1 < loops.length ? loops[shownLoop + 1] : null);
	const withParent = $derived(current ? current.par.reduce((a, p, v) => a + (p >= 0 && v !== graph.s ? 1 : 0), 0) : 0);
	const right = $derived(current ? current.par.reduce((a, p, v) => a + (v !== graph.s && p === label[v] ? 1 : 0), 0) : 0);

	function paint(cv: HTMLCanvasElement) {
		const size = Math.min(box, 480);
		if (size < 40) return; // not laid out yet (a hidden or zero-width container): nothing to draw, and arc() would throw
		const ctx = fitCanvas(cv, size, size);
		const g = graph;
		const { ang, depth, dmax } = layout;
		const R = size / 2 - 10;
		const ring = (d: number) => (dmax ? (R * d) / dmax : 0);
		const pos = (v: number): [number, number] => [size / 2 + ring(depth[v]) * Math.cos(ang[v]), size / 2 + ring(depth[v]) * Math.sin(ang[v])];
		const P = Array.from({ length: g.n }, (_, v) => pos(v));
		ctx.clearRect(0, 0, size, size);
		// rings (distance from the source; drawing aid)
		ctx.strokeStyle = C.tint;
		ctx.lineWidth = 1;
		for (let d = 1; d <= dmax; d++) {
			ctx.beginPath();
			ctx.arc(size / 2, size / 2, ring(d), 0, Math.PI * 2);
			ctx.stroke();
		}
		// the input graph's edges
		ctx.strokeStyle = C.mist;
		ctx.globalAlpha = g.n > 300 ? 0.28 : 0.55;
		ctx.lineWidth = g.n > 300 ? 0.5 : 0.75;
		ctx.beginPath();
		for (let u = 0; u < g.n; u++)
			for (const v of g.adj[u])
				if (u < v) {
					ctx.moveTo(P[u][0], P[u][1]);
					ctx.lineTo(P[v][0], P[v][1]);
				}
		ctx.stroke();
		ctx.globalAlpha = 1;
		// the rule's parents: drawn in full for finished loops, growing for the loop in progress
		const par = current?.par;
		ctx.strokeStyle = C.cobalt;
		ctx.lineWidth = g.n > 300 ? 0.9 : 1.3;
		ctx.lineCap = 'round';
		ctx.beginPath();
		if (par)
			for (let v = 0; v < g.n; v++) {
				const p = par[v];
				if (p < 0 || v === g.s) continue;
				ctx.moveTo(P[p][0], P[p][1]);
				ctx.lineTo(P[v][0], P[v][1]);
			}
		if (incoming) {
			const f = ease(Math.min(1, frac * 1.4));
			for (const v of incoming.newly) {
				const p = incoming.par[v];
				ctx.moveTo(P[p][0], P[p][1]);
				ctx.lineTo(P[p][0] + (P[v][0] - P[p][0]) * f, P[p][1] + (P[v][1] - P[p][1]) * f);
			}
		}
		ctx.stroke();
		// nodes: hollow until they have a parent
		const r = g.n > 300 ? 1.7 : g.n > 60 ? 2.6 : 3.6;
		for (let v = 0; v < g.n; v++) {
			const has = v === g.s || (par && par[v] >= 0);
			ctx.beginPath();
			ctx.arc(P[v][0], P[v][1], r, 0, Math.PI * 2);
			if (has) {
				ctx.fillStyle = par && v !== g.s && par[v] !== label[v] ? C.rose : C.cobalt;
				ctx.fill();
			} else {
				ctx.fillStyle = C.paper;
				ctx.fill();
				ctx.strokeStyle = C.blueGrey;
				ctx.lineWidth = 0.8;
				ctx.stroke();
			}
		}
		// the source
		ctx.beginPath();
		ctx.arc(size / 2, size / 2, r + 3, 0, Math.PI * 2);
		ctx.strokeStyle = C.ink;
		ctx.lineWidth = 1.2;
		ctx.stroke();
	}

	const bars = $derived.by(() => {
		if (!loops.length) return [];
		return loops.map((l, k) => ({ k, newly: l.newly.length, on: k <= shownLoop }));
	});
	const barMax = $derived(Math.max(1, ...bars.map((b) => b.newly)));
</script>

<div class="wrap" {@attach (el) => (autoplay ? whenVisible(el, () => setTimeout(play, 300)) : undefined)}>
	<div class="stage" bind:clientWidth={box}>
		<canvas {@attach paint} aria-label="A random graph of {n} nodes; the rule's parents drawn as they are found"></canvas>
	</div>
	<div class="side">
		<div class="fit">
			<div class="label">The fit</div>
			<p>
				{#if fitted}
					Fitted in <b class="num">{fitted.ms.toFixed(1)} ms</b> in this browser, at the completed answers of
					{data.train.length} graphs of 8 to 16 nodes ({int(fitted.pairs)} pairs){sameAsPython ? '; the weights match the Python fit' : ''}.
					Python: {data.record.order.fit_ms.toFixed(1)} ms.
				{:else}
					Fitting at the completed answers of {data.train.length} small graphs…
				{/if}
			</p>
		</div>
		<label class="slider">
			<span>Nodes</span>
			<input type="range" min="0" max={SIZES.length - 1} step="1" bind:value={sizeI} oninput={reset} />
			<span class="num">{int(n)}</span>
		</label>
		<div class="play">
			<Play {phase} onclick={play} />
			<IconButton icon={DiceIcon} label="New graph" show onclick={() => ((seed += 1), reset())} />
		</div>
		<dl class="readout">
			<dt>Loop</dt>
			<dd>{current ? current.loop + 1 : 0}{loops.length ? ` of ${loops.length}` : ''}</dd>
			<dt><span class="key" style:background={C.cobalt}></span>Nodes with a parent</dt>
			<dd>{int(withParent)} of {int(n - 1)}</dd>
			<dt>Same parent as BFS</dt>
			<dd>{int(right)} of {int(n - 1)}</dd>
			<dt>Run time</dt>
			<dd>{loops.length ? `${runMs.toFixed(runMs < 10 ? 1 : 0)} ms` : ''}</dd>
		</dl>
		<div class="chart">
			<div class="label">Parents found per loop</div>
			<svg viewBox="0 0 240 56" aria-hidden="true">
				<line x1="0" y1="55.5" x2="240" y2="55.5" stroke={C.edge} />
				{#each bars as b (b.k)}
					{@const bw = Math.min(22, 240 / Math.max(8, bars.length) - 4)}
					<rect
						x={b.k * (bw + 4)}
						y={55 - (b.newly / barMax) * 50}
						width={bw}
						height={(b.newly / barMax) * 50}
						fill={b.on ? C.cobalt : C.lift}
						class="bar"
					/>
				{/each}
			</svg>
		</div>
	</div>
</div>

<style>
	.wrap {
		display: grid;
		grid-template-columns: minmax(0, 480px) minmax(0, 1fr);
		gap: 28px;
		align-items: start;
	}
	@media (max-width: 720px) {
		.wrap {
			grid-template-columns: minmax(0, 1fr);
			gap: 18px;
		}
	}
	.stage {
		width: 100%;
		max-width: 480px;
	}
	canvas {
		display: block;
		max-width: 100%;
	}
	.side {
		display: flex;
		flex-direction: column;
		gap: 14px;
		min-width: 0;
	}
	.fit p {
		margin-top: 4px;
		font-size: 13px;
		line-height: 1.5;
		color: var(--color-ink-soft);
	}
	.fit b {
		color: var(--color-ink);
		font-weight: 600;
	}
	.slider {
		display: flex;
		align-items: center;
		gap: 10px;
		font-size: 13px;
		color: var(--color-ink-soft);
	}
	.slider input {
		width: 150px;
	}
	.play {
		display: flex;
		align-items: center;
		gap: 10px;
	}
	.key {
		display: inline-block;
		width: 8px;
		height: 8px;
		margin-right: 7px;
		border-radius: 2px;
	}
	.chart svg {
		display: block;
		width: 100%;
		max-width: 240px;
		height: 56px;
		margin-top: 6px;
	}
	.bar {
		transition: fill 300ms ease;
	}
</style>
