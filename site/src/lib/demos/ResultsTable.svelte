<script lang="ts">
	import { dur, int, params, sci } from '#lib/format.ts';
	import { C } from '#lib/theme.ts';

	type Col = { solved: number; sd: number | null; published: boolean; params: number; flops: number; fit_s?: number };
	type Row2 = { name: string; sudoku: Col; unique: Col };
	type Row8 = Col & { name: string };
	type Ours = { solved: number[]; n: number; params: number; ops: number; ops_cmp: number; fit_s: number[] };
	export type Results = {
		table2: Row2[];
		table8: Row8[];
		ours: { sudoku: Ours; unique: Ours; hard: Ours };
		cram: Record<'sudoku' | 'unique' | 'hard', { solved_pct: number; n: number; params: number; flops: number; fit_s: number; params_nonzero?: number[] }>;
	};
	let { r }: { r: Results } = $props();

	type Bench = 'sudoku' | 'unique' | 'hard';
	let bench = $state<Bench>('sudoku');
	let hl = $state<string | null>('Ours');

	type Line = { name: string; who: 'ours' | 'cram' | 'hand' | 'looped'; solved: string; count?: string; params: string; ops: string; fit: string; hw: string; note?: string };
	const fmtSolved = (x: number, d = 2) => x.toFixed(d);
	const rows = $derived.by((): Line[] => {
		const o = r.ours[bench];
		const c = r.cram[bench];
		const pctOurs = (100 * o.solved.reduce((a, b) => a + b, 0)) / (o.solved.length * o.n);
		const out: Line[] = [
			{
				name: 'Ours',
				who: 'ours',
				solved: fmtSolved(pctOurs, 3),
				count: o.solved.length > 1 ? `${int(o.solved[0])} / ${int(o.n)}, each of ${o.solved.length} fits` : `${int(o.solved[0])} / ${int(o.n)}`,
				params: String(o.params),
				ops: sci(o.ops, 1, 1e3),
				fit: dur(o.fit_s[0]),
				hw: 'one CPU core'
			},
			{
				name: bench === 'hard' ? 'CRAM+z' : 'CRAM',
				who: 'cram',
				solved: fmtSolved(c.solved_pct, 3),
				params: params(c.params_nonzero ? c.params_nonzero[0] : c.params),
				ops: sci(c.flops),
				fit: dur(c.fit_s),
				hw: 'one H200 GPU'
			}
		];
		if (bench === 'hard') {
			for (const x of r.table8.filter((x) => x.name !== 'CRAM+z'))
				out.push({ name: x.name, who: 'looped', solved: fmtSolved(x.solved), params: params(x.params), ops: sci(x.flops), fit: '', hw: '', note: x.published ? 'published' : undefined });
		} else {
			for (const x of r.table2.filter((x) => x.name !== 'CRAM')) {
				const col = x[bench];
				out.push({
					name: x.name === '(hand-set)' ? 'CRAM, hand-set' : x.name,
					who: x.name === '(hand-set)' ? 'hand' : 'looped',
					solved: fmtSolved(col.solved),
					params: params(col.params),
					ops: sci(col.flops),
					fit: dur(col.fit_s ?? 0),
					hw: x.name === '(hand-set)' ? '' : 'one H200 GPU',
					note: col.published ? 'published' : undefined
				});
			}
		}
		return out;
	});
	const dot = (w: Line['who']) => (w === 'ours' ? C.cobalt : w === 'cram' ? C.rose : 'transparent');
</script>

<div class="top">
	<div class="seg" role="group" aria-label="Benchmark">
		<button aria-pressed={bench === 'sudoku'} onclick={() => (bench = 'sudoku')}>Sudoku-Extreme</button>
		<button aria-pressed={bench === 'unique'} onclick={() => (bench = 'unique')}>Maze-Unique</button>
		<button aria-pressed={bench === 'hard'} onclick={() => (bench = 'hard')}>Maze-Hard</button>
	</div>
</div>
<div class="scroll">
	<table>
		<thead>
			<tr>
				<th class="l">Model</th>
				<th>Solved (%)</th>
				<th class="pcol">Parameters</th>
				<th>Operations per puzzle</th>
				<th>Fit time</th>
			</tr>
		</thead>
		<tbody>
			{#each rows as x (x.name)}
				<tr class={[x.who, { hl: hl === x.name }]}>
					<td class="l">
						<button class="name" onclick={() => (hl = hl === x.name ? null : x.name)}>
							<i style:background={dot(x.who)}></i>{x.name}
						</button>
					</td>
					<td class="num">{x.solved}{#if x.note}<sup>†</sup>{/if}{#if x.count}<span class="count">{x.count}</span>{/if}</td>
					<td class="num pcol">{x.params}</td>
					<td class="num">{x.ops}</td>
					<td class="num">{x.fit || '–'}{#if x.who === 'ours'}<span class="count">{x.hw}</span>{/if}</td>
				</tr>
			{/each}
		</tbody>
	</table>
</div>

<style>
	.top {
		margin-bottom: 12px;
	}
	.scroll {
		overflow-x: auto;
	}
	table {
		width: 100%;
		min-width: 560px;
		border-collapse: collapse;
		font-size: 14px;
	}
	th {
		padding: 8px 10px;
		border-bottom: 1px solid var(--color-ink);
		color: var(--color-ink-soft);
		font-weight: 500;
		font-size: 12.5px;
		text-align: right;
		white-space: nowrap;
	}
	td {
		padding: 7px 10px;
		border-bottom: 1px solid var(--color-edge);
		text-align: right;
		vertical-align: top;
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
	}
	tbody tr:last-child td {
		border-bottom: 1px solid var(--color-ink);
	}
	.l {
		text-align: left;
	}
	tr.looped td,
	tr.hand td {
		color: var(--color-ink-soft);
	}
	tr.ours td {
		color: var(--color-ink);
		font-weight: 600;
	}
	tr.ours .count,
	tr.ours .name {
		font-weight: 600;
	}
	tr.hl td {
		background: var(--color-tint);
	}
	tr {
		transition: background 200ms ease;
	}
	.name {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		padding: 0;
		border: 0;
		background: none;
		color: inherit;
		font: inherit;
		text-align: left;
		cursor: pointer;
	}
	.name i {
		display: inline-block;
		width: 7px;
		height: 7px;
		border-radius: 50%;
	}
	.count {
		display: block;
		font-size: 11.5px;
		font-weight: 400;
		color: var(--color-ink-soft);
	}
	@media (max-width: 560px) {
		table {
			min-width: 0;
			font-size: 13px;
		}
		.pcol {
			display: none;
		}
		th {
			white-space: normal;
			padding: 6px 6px;
		}
		td {
			padding: 6px 6px;
			white-space: normal;
		}
		.count {
			white-space: normal;
		}
	}
	sup {
		font-size: 9px;
		margin-left: 1px;
	}
</style>
