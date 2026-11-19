<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import type { HeroBoard } from './solve.ts';
	import type { SudokuCheck } from '#lib/sudoku/engine.ts';
	import { C, ease, reducedMotion, stopFrame } from '#lib/theme.ts';

	type Input = { b: number; puzzle: number[]; seed: number };
	let { check, inputs, fitMs }: { check: SudokuCheck | null; inputs: Input[]; fitMs: number } = $props();

	// geometry: the fitted board, then the boards the same rule solves on one baseline (wide: one row; narrow: two)
	let width = $state(900);
	const narrow = $derived(width < 560);
	const G = $derived.by(() => {
		if (!narrow) {
			const BASE = 246;
			const FIT = { x: 28, y: BASE - 150, s: 150 };
			const SIZES = [64, 104, 144, 184];
			const X0 = 300;
			return {
				view: '0 48 920 240',
				BASE,
				FIT,
				X0,
				line0: X0 - 34,
				timer: { x: FIT.x, y: BASE + 30 },
				sub: { x: FIT.x + 92, y: BASE + 30 },
				slots: SIZES.map((s, k) => ({ x: X0 + SIZES.slice(0, k).reduce((a, v) => a + v + 38, 0), y: BASE - s, s }))
			};
		}
		const BASE = 318;
		const FIT = { x: 16, y: 34, s: 132 };
		const SIZES = [44, 70, 98, 128];
		const X0 = 14;
		return {
			view: '0 0 400 346',
			BASE,
			FIT,
			X0,
			line0: X0,
			timer: { x: FIT.x + FIT.s + 22, y: FIT.y + 70 },
			sub: { x: FIT.x + FIT.s + 22, y: FIT.y + 92 },
			slots: SIZES.map((s, k) => ({ x: X0 + SIZES.slice(0, k).reduce((a, v) => a + v + 12, 0), y: BASE - s, s }))
		};
	});
	const FIT = $derived(G.FIT);
	const c = $derived(G.FIT.s / 9);
	const slots = $derived(G.slots);
	const BASE = $derived(G.BASE);

	let boards = $state.raw<(HeroBoard | null)[]>([null, null, null, null]);
	let t = $state(0); // seconds since the cycle began
	let raf = 0;
	let worker: Worker | null = null;
	const FIT_DUR = 2.0;
	const STARTS = [2.3, 3.1, 3.9, 4.8];
	const DURS = [0.7, 1.1, 1.5, 2.0];
	const CYCLE = 9.6;

	onMount(() => {
		if (!check || !inputs.length) return;
		worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
		let k = 0;
		worker.onmessage = (e: MessageEvent<HeroBoard>) => {
			const next = boards.slice();
			next[k++] = e.data;
			boards = next;
		};
		worker.postMessage({ check: $state.snapshot(check), boards: $state.snapshot(inputs) });
		if (reducedMotion()) {
			t = CYCLE - 1.2;
			return;
		}
		const t0 = performance.now();
		const loop = () => {
			t = ((performance.now() - t0) / 1000) % CYCLE;
			raf = requestAnimationFrame(loop);
		};
		raf = requestAnimationFrame(loop);
	});
	onDestroy(() => {
		stopFrame(raf);
		worker?.terminate();
	});

	const clamp = (x: number) => Math.max(0, Math.min(1, x));
	const fade = $derived(1 - clamp((t - (CYCLE - 0.7)) / 0.6));
	const fitP = $derived(clamp(t / FIT_DUR));
	const timer = $derived(fitMs * ease(clamp((t - 0.25) / (FIT_DUR - 0.4))));
	const rel = $derived([clamp((t - 0.45) / 0.45), clamp((t - 0.85) / 0.45), clamp((t - 1.25) / 0.45)]);
	const draw = (k: number) => clamp((t - STARTS[k] + 0.5) / 0.6);

	// cells of each board in the order they appear: clues, the first cold stage, then each guess and what it forces
	const marks = $derived(
		boards.map((bd, k) => {
			if (!bd) return [];
			const p = clamp((t - STARTS[k]) / DURS[k]);
			const order = bd.seq;
			const shown = Math.floor(ease(p) * order.length);
			const clueIn = clamp((t - STARTS[k] + 0.35) / 0.3);
			return [...(clueIn > 0 ? bd.clue.map((c) => [c, 1] as [number, number]) : []), ...order.slice(0, shown)];
		})
	);
	const fill = (kind: number) => (kind === 1 ? C.mist : kind === 2 ? C.cobalt : kind === 3 ? C.rose : C.cobaltLight);
	const lines = (s: number, n: number) => Array.from({ length: n - 1 }, (_, i) => ((i + 1) * s) / n);
</script>

<div class="box" bind:clientWidth={width}>
<svg viewBox={G.view} class="hero" role="img" aria-label="A check fitted in milliseconds at solved 9 by 9 boards, then the same check solving 4 by 4, 9 by 9, 16 by 16 and 25 by 25 Sudoku">
	<g opacity={fade}>
		<!-- the fit -->
		<text x={FIT.x} y={FIT.y - 18} class="lab">FIT AT THE ANSWER</text>
		<g transform="translate({FIT.x} {FIT.y})">
			{#each lines(FIT.s, 9) as v, i (i)}
				<line x1={v} y1="0" x2={v} y2={FIT.s * fitP} stroke={(i + 1) % 3 === 0 ? C.blueGrey : C.edge} stroke-width={(i + 1) % 3 === 0 ? 1 : 0.7} />
				<line y1={v} x1="0" y2={v} x2={FIT.s * fitP} stroke={(i + 1) % 3 === 0 ? C.blueGrey : C.edge} stroke-width={(i + 1) % 3 === 0 ? 1 : 0.7} />
			{/each}
			<!-- the three kept relations of one cell: its row, its column, its box -->
			<rect x="0" y={4 * c} width={FIT.s * rel[0]} height={c} fill={C.cobaltTint} stroke={C.cobalt} stroke-width="0.9" opacity={rel[0] > 0 ? 1 : 0} />
			<rect x={4 * c} y="0" width={c} height={FIT.s * rel[1]} fill={C.cobaltTint} stroke={C.cobalt} stroke-width="0.9" opacity={rel[1] > 0 ? 1 : 0} fill-opacity="0.7" />
			<rect x={3 * c} y={3 * c} width={3 * c} height={3 * c} fill="none" stroke={C.cobalt} stroke-width="1.4" opacity={rel[2]} />
			<rect x={4 * c + c * 0.2} y={4 * c + c * 0.2} width={c * 0.6} height={c * 0.6} rx="1.5" fill={C.ink} opacity={rel[0] > 0 ? 1 : 0} />
			<rect x="0" y="0" width={FIT.s} height={FIT.s} fill="none" stroke={C.ink} stroke-width="1.2" stroke-dasharray={FIT.s * 4} stroke-dashoffset={FIT.s * 4 * (1 - fitP)} />
		</g>
		<text x={G.timer.x} y={G.timer.y} class="timer num">{timer.toFixed(1)} ms</text>
		<text x={G.sub.x} y={G.sub.y} class="sub">on one CPU core</text>

		<!-- the same rule, every size -->
		<line x1={G.line0} y1={BASE + 0.5} x2={G.line0 + (slots[3].x + slots[3].s - G.line0) * clamp((t - 2.0) / 1.2)} y2={BASE + 0.5} stroke={C.edge} />
		{#each slots as sl, k (k)}
			{@const side = [4, 9, 16, 25][k]}
			{@const b = [2, 3, 4, 5][k]}
			{@const cell = sl.s / side}
			{@const p = draw(k)}
			<g transform="translate({sl.x} {sl.y})" opacity={p > 0 ? 1 : 0}>
				{#each marks[k] as [c, kind] (c)}
					<rect
						class="mark"
						x={(c % side) * cell + cell * 0.18}
						y={Math.floor(c / side) * cell + cell * 0.18}
						width={cell * 0.64}
						height={cell * 0.64}
						rx={Math.min(1.5, cell * 0.12)}
						fill={fill(kind)}
					/>
				{/each}
				{#each lines(sl.s, b) as v, i (i)}
					<line x1={v} y1="0" x2={v} y2={sl.s * p} stroke={C.blueGrey} stroke-width="0.8" />
					<line y1={v} x1="0" y2={v} x2={sl.s * p} stroke={C.blueGrey} stroke-width="0.8" />
				{/each}
				<rect x="0" y="0" width={sl.s} height={sl.s} fill="none" stroke={C.ink} stroke-width="1.2" stroke-dasharray={sl.s * 4} stroke-dashoffset={sl.s * 4 * (1 - p)} />
			</g>
			<text x={sl.x + sl.s / 2} y={BASE + 22} class="size" text-anchor="middle" opacity={p}>{side} × {side}</text>
		{/each}
	</g>
</svg>
</div>

<style>
	.hero {
		display: block;
		width: 100%;
		height: auto;
		font-family: var(--font-sans);
	}
	.lab {
		font-size: 10.5px;
		letter-spacing: 0.08em;
		fill: var(--color-ink-soft);
	}
	.box {
		width: 100%;
	}
	.timer {
		font-family: var(--font-mono);
		font-size: 21px;
		fill: var(--color-ink);
	}
	.sub,
	.size {
		font-size: 12px;
		fill: var(--color-ink-soft);
	}
	.mark {
		animation: markin 380ms var(--ease);
	}
	@keyframes markin {
		from {
			opacity: 0;
		}
	}
</style>
