<script lang="ts">
	import { onDestroy } from 'svelte';
	import {
		coldUnique,
		lMaze,
		parseBits,
		parseTokens,
		solveHard,
		sym,
		tokens,
		type MazeCheck,
		type MazeRule
	} from '#lib/mazes/rules.ts';
	import Play from '#lib/ui/Play.svelte';
	import { C, ease, fitCanvas, stopFrame } from '#lib/theme.ts';
	import { int } from '#lib/format.ts';

	type Cram = { equals_label: boolean; valid_path: boolean; length: number; first: string; stopped: boolean };
	export type MazeData = {
		check: MazeCheck;
		rule: MazeRule;
		hard: { index: number; tokens: string; label: string }[];
		cram106: { seed: number; stopped: boolean; diff: { cell: [number, number]; pred: number }[] }[];
		edge: {
			index: number;
			tokens: string;
			label: string;
			side: number;
			cram: { n_diff: number; n_diff_edge: number; edge_tips: number; diff_recorded: { cell: [number, number]; pred: number }[] };
		}[];
		rooms: { extra: number; sym: number; n: number; shortest: number; label_first: string; cram: Record<string, Cram> }[];
		stats: {
			hard_test: { n: number; exact: number };
			unique_test: { n: number; exact: number };
			hard_rooms: { n: number; exact: number };
			unique_generated: Record<string, { side: number; ring: boolean; n: number; exact: number }>;
		};
	};
	let { data }: { data: MazeData } = $props();

	type Mode = 'hard' | 'room' | 'edge';
	let mode = $state<Mode>('hard');
	const ROOM_NS = [12, 16, 18, 20, 22, 24, 26];
	let roomI = $state(5); // 24 x 24
	let showCram = $state(true);

	interface Scene {
		H: number;
		W: number;
		tok: Int8Array;
		label: Uint8Array;
		room?: Uint8Array;
		title: string;
	}
	const scene = $derived.by((): Scene => {
		if (mode === 'hard') {
			const m = data.hard.find((x) => x.index === 106)!;
			return { H: 30, W: 30, tok: parseTokens(m.tokens), label: parseBits(m.label), title: 'Maze-Hard test maze 106' };
		}
		if (mode === 'edge') {
			const m = data.edge[0];
			return { H: m.side, W: m.side, tok: parseTokens(m.tokens), label: parseBits(m.label), title: `A ${m.side} × ${m.side} tree maze without a wall ring` };
		}
		const n = ROOM_NS[roomI];
		const b = sym(lMaze(n, n, 2), 1);
		const t = tokens(b);
		const lab = new Uint8Array(b.H * b.W);
		const r = solveHard(data.rule, t, b.H, b.W); // only used for drawing; the label is checked in the tests
		r.path.forEach((c) => (lab[c] = 1));
		return { H: b.H, W: b.W, tok: t, label: lab, room: b.room, title: `Room ${n} × ${n} with a corridor two steps shorter` };
	});
	const roomRow = $derived(data.rooms.find((r) => r.extra === 2 && r.sym === 1 && r.n === ROOM_NS[roomI])!);
	const cramDiffers = $derived(!Object.values(roomRow.cram).every((x) => x.equals_label));

	// what the run computes, for the current scene
	const ours = $derived.by(() => {
		const { tok, H, W } = scene;
		if (mode === 'edge') {
			const r = coldUnique(data.check, tok, H, W, true);
			return { kind: 'cold' as const, events: r.events, board: r.board, ops: r.ops, path: [] as number[], clock: null };
		}
		const r = solveHard(data.rule, tok, H, W);
		return { kind: 'walk' as const, events: [], board: r.board, ops: r.ops, path: r.path, clock: r.clock };
	});
	const exact = $derived(!!ours.board && ours.board.every((v, i) => v === scene.label[i]));
	const maxClock = $derived(ours.clock ? Math.max(...ours.clock) : 0);

	let phase = $state<'idle' | 'running' | 'done'>('idle');
	let progress = $state(0); // 0..1 across the run
	let cramIn = $state(0); // 0..1 fade of CRAM's overlay
	let raf = 0;
	let box = $state(420);

	function choose(m: Mode) {
		mode = m;
		stop();
	}
	function stop() {
		stopFrame(raf);
		phase = 'idle';
		progress = 0;
		cramIn = 0;
	}
	function play() {
		if (phase === 'running') return;
		stop();
		phase = 'running';
		const dur = mode === 'edge' ? 3600 : 3400;
		const t0 = performance.now();
		const step = () => {
			const t = (performance.now() - t0) / dur;
			progress = Math.min(1, t);
			cramIn = Math.max(0, Math.min(1, (t - 1) / 0.25));
			if (t < 1.25) raf = requestAnimationFrame(step);
			else phase = 'done';
		};
		raf = requestAnimationFrame(step);
	}
	onDestroy(() => stopFrame(raf));

	const cramCells = $derived.by(() => {
		if (mode === 'hard') return data.cram106[0].diff.filter((d) => d.pred === 1).map((d) => d.cell[0] * 30 + d.cell[1]);
		if (mode === 'edge') return data.edge[0].cram.diff_recorded.filter((d) => d.pred === 1).map((d) => d.cell[0] * scene.W + d.cell[1]);
		return [];
	});

	function paint(cv: HTMLCanvasElement) {
		const { H, W, tok, room } = scene;
		const cell = Math.floor(Math.min(box, 460) / Math.max(H, W));
		const w = cell * W;
		const h = cell * H;
		const ctx = fitCanvas(cv, w, h);
		const p = progress;
		const done = phase === 'done';
		ctx.clearRect(0, 0, w, h);
		const xy = (i: number) => [(i % W) * cell, Math.floor(i / W) * cell] as const;
		const centre = (i: number) => [(i % W) * cell + cell / 2, Math.floor(i / W) * cell + cell / 2] as const;
		// walls: pale blocks; open cells: paper
		for (let i = 0; i < H * W; i++) {
			const [x, y] = xy(i);
			if (tok[i] === 0) {
				ctx.fillStyle = C.wall;
				ctx.fillRect(x, y, cell, cell);
			}
		}
		if (ours.kind === 'walk' && ours.clock) {
			// the cold reach clock from G: a wave over the open cells, then the walk from S
			const a = ease(Math.min(1, p / 0.55));
			const front = a * (maxClock + 1);
			for (let i = 0; i < H * W; i++) {
				const c = ours.clock[i];
				if (c < 0 || c > front) continue;
				const [x, y] = xy(i);
				ctx.fillStyle = front - c < 1.5 && a < 1 ? C.cobaltLight : C.cobaltTint;
				ctx.fillRect(x, y, cell, cell);
			}
			if (room && mode === 'room') {
				// the room's outline (drawing aid)
				ctx.strokeStyle = C.mist;
				ctx.lineWidth = 1;
				let r0 = H,
					r1 = 0,
					c0 = W,
					c1 = 0;
				for (let i = 0; i < H * W; i++)
					if (room[i]) {
						r0 = Math.min(r0, Math.floor(i / W));
						r1 = Math.max(r1, Math.floor(i / W));
						c0 = Math.min(c0, i % W);
						c1 = Math.max(c1, i % W);
					}
				if (cramIn > 0 && showCram && cramDiffers) {
					ctx.globalAlpha = cramIn;
					ctx.strokeStyle = C.rose;
					ctx.setLineDash([3, 3]);
					ctx.strokeRect(c0 * cell + 0.5, r0 * cell + 0.5, (c1 - c0 + 1) * cell - 1, (r1 - r0 + 1) * cell - 1);
					ctx.setLineDash([]);
					ctx.globalAlpha = 1;
				}
			}
			const walk = Math.max(0, Math.min(1, (p - 0.55) / 0.45));
			if (walk > 0) {
				const n = ours.path.length;
				const upto = ease(walk) * (n - 1);
				ctx.strokeStyle = C.cobalt;
				ctx.lineWidth = Math.max(1.6, cell * 0.22);
				ctx.lineCap = 'round';
				ctx.lineJoin = 'round';
				ctx.beginPath();
				for (let k = 0; k <= Math.floor(upto); k++) {
					const [cx, cy] = centre(ours.path[k]);
					if (k === 0) ctx.moveTo(cx, cy);
					else ctx.lineTo(cx, cy);
				}
				const f = upto - Math.floor(upto);
				if (f > 0 && Math.floor(upto) + 1 < n) {
					const [ax, ay] = centre(ours.path[Math.floor(upto)]);
					const [bx, by] = centre(ours.path[Math.floor(upto) + 1]);
					ctx.lineTo(ax + (bx - ax) * f, ay + (by - ay) * f);
				}
				ctx.stroke();
			}
			if (mode === 'room' && cramIn > 0 && showCram && cramDiffers) {
				// CRAM's recorded first step: into the room
				const S = tok.indexOf(2);
				const d = { down: [1, 0], left: [0, -1], right: [0, 1], up: [-1, 0] }[roomRow.cram['0'].first] ?? [0, 0];
				const [sx, sy] = centre(S);
				const L = cell * 2.4;
				ctx.globalAlpha = cramIn;
				ctx.strokeStyle = C.rose;
				ctx.fillStyle = C.rose;
				ctx.lineWidth = 1.6;
				const ex = sx + d[1] * L;
				const ey = sy + d[0] * L;
				ctx.beginPath();
				ctx.moveTo(sx + d[1] * cell * 0.5, sy + d[0] * cell * 0.5);
				ctx.lineTo(ex, ey);
				ctx.stroke();
				ctx.beginPath();
				const px = -d[0];
				const py = d[1];
				ctx.moveTo(ex + d[1] * 5, ey + d[0] * 5);
				ctx.lineTo(ex + px * 3.5, ey + py * 3.5);
				ctx.lineTo(ex - px * 3.5, ey - py * 3.5);
				ctx.closePath();
				ctx.fill();
				ctx.globalAlpha = 1;
			}
		} else {
			// Maze-Unique cold derivation: undecided open cells pale; decided off back to paper; on in cobalt
			const dom = Int32Array.from(tok, (t) => (t === 0 ? 1 : t >= 2 ? 2 : 3));
			const k = Math.floor(ease(Math.min(1, p)) * ours.events.length);
			for (let e = 0; e < k; e++) dom[ours.events[e][0]] = ours.events[e][1];
			for (let i = 0; i < H * W; i++) {
				if (tok[i] === 0) continue;
				const [x, y] = xy(i);
				if (dom[i] === 3) {
					ctx.fillStyle = C.tint;
					ctx.fillRect(x, y, cell, cell);
				} else if (dom[i] === 2) {
					ctx.fillStyle = C.cobalt;
					const m = cell * 0.18;
					ctx.fillRect(x + m, y + m, cell - 2 * m, cell - 2 * m);
				}
			}
		}
		// CRAM's released model, where the record has its cells
		if (cramIn > 0 && showCram && cramCells.length) {
			ctx.globalAlpha = cramIn;
			ctx.strokeStyle = C.rose;
			ctx.lineWidth = 1.2;
			for (const i of cramCells) {
				const [x, y] = xy(i);
				ctx.strokeRect(x + 1.5, y + 1.5, cell - 3, cell - 3);
			}
			ctx.globalAlpha = 1;
		}
		// S and G
		for (const [t, filled] of [
			[2, false],
			[3, true]
		] as const) {
			const i = tok.indexOf(t);
			const [cx, cy] = centre(i);
			ctx.beginPath();
			ctx.arc(cx, cy, Math.max(3, cell * 0.32), 0, Math.PI * 2);
			ctx.fillStyle = filled ? C.ink : C.paper;
			ctx.fill();
			ctx.strokeStyle = C.ink;
			ctx.lineWidth = 1.2;
			ctx.stroke();
		}
		ctx.strokeStyle = C.inkSoft;
		ctx.lineWidth = 1;
		ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
		void done;
	}

	const cramLine = $derived.by(() => {
		if (mode === 'hard') {
			const seeds = data.cram106.filter((c) => c.stopped).length;
			return `CRAM's released Maze-Hard models (${seeds} of 3 seeds) stop on this maze with ${cramCells.length} extra cells on, along the board's edge.`;
		}
		if (mode === 'edge') {
			const e = data.edge[0].cram;
			return `CRAM's released Maze-Unique model leaves ${int(e.n_diff)} cells on (${int(e.n_diff_edge)} on the edge); the record lists the first ${e.diff_recorded.length}, drawn here.`;
		}
		const c = roomRow.cram;
		const same = Object.values(c).every((x) => x.equals_label);
		if (same) return `CRAM's released models (3 seeds) return the label's path here too.`;
		const runs = Object.values(c);
		const lens = [...new Set(runs.filter((x) => x.valid_path).map((x) => x.length))].join(', ');
		const valid = runs.filter((x) => x.valid_path).length;
		const firsts = [...new Set(runs.map((x) => x.first))].join(' or ');
		return `CRAM's released models (${runs.length} seeds) step ${firsts} into the room; ${valid} of ${runs.length} return a valid path${lens ? ` of ${lens} steps` : ''}, where the shortest is ${roomRow.shortest}. The record keeps the first step and the length, not the cells.`;
	});
</script>

<div class="wrap">
	<div class="stage" bind:clientWidth={box}>
		<canvas {@attach paint} aria-label={scene.title}></canvas>
	</div>
	<div class="side">
		<div class="seg" role="group" aria-label="Maze">
			<button aria-pressed={mode === 'hard'} onclick={() => choose('hard')}>Maze-Hard #106</button>
			<button aria-pressed={mode === 'room'} onclick={() => choose('room')}>Room</button>
			<button aria-pressed={mode === 'edge'} onclick={() => choose('edge')}>Edge</button>
		</div>
		{#if mode === 'room'}
			<label class="slider">
				<span>Room side</span>
				<input type="range" min="0" max={ROOM_NS.length - 1} step="1" bind:value={roomI} oninput={stop} />
				<span class="num">{ROOM_NS[roomI]}</span>
			</label>
		{/if}
		<div class="play">
			<Play {phase} onclick={play} />
			<span class="note">{scene.title}</span>
		</div>
		<dl class="readout">
			{#if mode === 'edge'}
				<dt>Cells decided cold</dt>
				<dd>{int(ours.events.length)}</dd>
			{:else}
				<dt>Clock loops (from G)</dt>
				<dd>{int(maxClock)}</dd>
				<dt>Path length</dt>
				<dd>{int(ours.path.length - 1)} steps</dd>
			{/if}
			<dt>Operations</dt>
			<dd>{int(ours.ops)}</dd>
			<dt>Matches the label</dt>
			<dd>{exact ? 'yes, every cell' : 'no'}</dd>
		</dl>
		<div class="legend">
			<span><i style:background={C.cobalt}></i>Ours, cold</span>
			{#if mode !== 'edge'}<span><i style:background={C.cobaltTint} style:border="1px solid {C.cobaltLight}"></i>Reach clock</span>{/if}
			<span><i class="ring" style:border-color={C.rose}></i>CRAM</span>
		</div>
		<label class="toggle"><input type="checkbox" bind:checked={showCram} /> Show CRAM's answer after the run</label>
		<p class="cram">{cramLine}</p>
	</div>
</div>

<style>
	.wrap {
		display: grid;
		grid-template-columns: minmax(0, 460px) minmax(0, 1fr);
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
		max-width: 460px;
	}
	canvas {
		display: block;
		max-width: 100%;
	}
	.side {
		display: flex;
		flex-direction: column;
		gap: 14px;
		align-items: flex-start;
		min-width: 0;
	}
	.side > dl {
		align-self: stretch;
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
	.slider {
		display: flex;
		align-items: center;
		gap: 10px;
		font-size: 13px;
		color: var(--color-ink-soft);
	}
	.slider input {
		width: 140px;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 14px;
		font-size: 12px;
		color: var(--color-ink-soft);
	}
	.legend i {
		display: inline-block;
		width: 10px;
		height: 10px;
		margin-right: 6px;
		border-radius: 2px;
		vertical-align: -1px;
	}
	.legend i.ring {
		border: 1.2px solid;
		background: none;
	}
	.toggle {
		font-size: 13px;
		color: var(--color-ink-soft);
		display: flex;
		gap: 7px;
		align-items: center;
	}
	.toggle input {
		accent-color: var(--color-ink);
	}
	.cram {
		font-size: 13px;
		line-height: 1.5;
		color: var(--color-ink);
	}
</style>
