<script lang="ts">
	import { int } from '#lib/format.ts';

	export type ClrsRow = {
		key: string;
		task: string;
		tier: number;
		tier_label: string;
		supplied: string | null;
		fit_ms: number;
		fit_from_run: boolean;
		fit_note: string | null;
		test64: number;
		loops_max: number;
		exact_to: number | string;
		heldout_sets: number;
		strings: boolean;
		paper_best: number;
		floydnet: number;
	};
	let { rows, avg }: { rows: ClrsRow[]; avg: { test64: number; paper_best: number; floydnet: number } } = $props();

	const groups = $derived.by(() => {
		const out: { tier: number; label: string; rows: ClrsRow[] }[] = [];
		for (const r of rows) {
			if (!out.length || out[out.length - 1].tier !== r.tier) out.push({ tier: r.tier, label: r.tier_label, rows: [] });
			out[out.length - 1].rows.push(r);
		}
		return out;
	});
	const f2 = (x: number | null) => (x === null ? '–' : x.toFixed(2));
	/** round half to even, as the paper's table (Python's format) does, so the two print the same digits */
	const even = (x: number) => (Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : Math.round(x));
	const fit = (ms: number) => (ms < 1000 ? `${even(ms)} ms` : `${(even(ms / 100) / 10).toFixed(1)} s`);
	const to = (r: ClrsRow) => (typeof r.exact_to === 'number' ? `${int(r.exact_to)}${r.strings ? ' items' : ''}` : r.exact_to);
</script>

<div class="scroll">
	<table>
		<thead>
			<tr>
				<th class="l">Task</th>
				<th class="l sup">Supplied</th>
				<th class="fitc">Fit, one core</th>
				<th>Official 64</th>
				<th class="toc">Exact to</th>
				<th>Our looped blocks</th>
				<th>FloydNet</th>
			</tr>
		</thead>
		{#each groups as g (g.tier)}
			<tbody class={['group', g.tier === 1 ? 'method' : 'ceiling']}>
				<tr class="head">
					<th class="l" colspan="7">
						{g.label}<span class="n">{g.rows.length} tasks</span>
					</th>
				</tr>
				{#each g.rows as r (r.key)}
					<tr>
						<td class="l">
							{r.task}{#if r.supplied}<span class="sub">supplied: {r.supplied}</span>{/if}
						</td>
						<td class="l sup">{r.supplied ?? '–'}</td>
						<td class="fitc" title={r.fit_note ?? undefined}>{fit(r.fit_ms)}{r.fit_from_run ? '*' : ''}</td>
						<td class="ours">{f2(r.test64)}<span class="sub">{typeof r.exact_to === 'number' ? 'to ' : ''}{to(r)}</span></td>
						<td class="ours toc">{to(r)}</td>
						<td class={{ low: r.paper_best < 100 }}>{f2(r.paper_best)}</td>
						<td>{f2(r.floydnet)}</td>
					</tr>
				{/each}
			</tbody>
		{/each}
		<tfoot>
			<tr>
				<td class="l">Average, {rows.length} tasks</td>
				<td class="sup"></td>
				<td class="fitc"></td>
				<td>{f2(avg.test64)}</td>
				<td class="toc"></td>
				<td>{f2(avg.paper_best)}</td>
				<td>{f2(avg.floydnet)}</td>
			</tr>
		</tfoot>
	</table>
</div>

<style>
	.scroll {
		overflow-x: auto;
	}
	table {
		width: 100%;
		min-width: 640px;
		border-collapse: collapse;
		font-size: 13.5px;
		font-variant-numeric: tabular-nums;
	}
	thead th {
		padding: 7px 9px;
		border-bottom: 1px solid var(--color-ink);
		color: var(--color-ink-soft);
		font-weight: 500;
		font-size: 12px;
		text-align: right;
		white-space: nowrap;
	}
	td {
		padding: 5px 9px;
		border-bottom: 1px solid var(--color-edge);
		text-align: right;
		white-space: nowrap;
		color: var(--color-ink-soft);
	}
	.l {
		text-align: left;
	}
	td.l:first-child {
		color: var(--color-ink);
	}
	/* group headings: the method in cobalt; the two ceilings set apart by a dashed rule */
	tr.head th {
		padding: 16px 9px 6px;
		font-size: 12.5px;
		font-weight: 600;
		text-align: left;
		border-bottom: 1px solid var(--color-edge);
	}
	.method tr.head th {
		color: var(--color-cobalt);
	}
	.ceiling tr.head th {
		color: var(--color-ink-soft);
		font-style: italic;
		font-weight: 500;
		border-top: 1px dashed var(--color-mist);
	}
	.n {
		margin-left: 8px;
		font-style: normal;
		font-weight: 400;
		color: var(--color-ink-soft);
	}
	.method .ours {
		color: var(--color-cobalt);
		font-weight: 600;
	}
	/* a ceiling: the same exact scores, shown muted so they never read as the method */
	.ceiling .ours {
		color: var(--color-blue-grey);
		font-weight: 400;
	}
	.ceiling td.l:first-child {
		color: var(--color-ink-soft);
	}
	.low {
		color: var(--color-rose);
	}
	tfoot td {
		padding-top: 7px;
		border-top: 1px solid var(--color-ink);
		border-bottom: 1px solid var(--color-ink);
		color: var(--color-ink-soft);
	}
	.sub {
		display: none;
	}
	@media (max-width: 560px) {
		table {
			min-width: 0;
			font-size: 12.5px;
		}
		.sup,
		.fitc,
		.toc {
			display: none;
		}
		thead th {
			white-space: normal;
			padding: 6px 5px;
		}
		td {
			padding: 5px 5px;
		}
		td.l {
			white-space: normal;
		}
		.sub {
			display: block;
			font-size: 11px;
			font-weight: 400;
			color: var(--color-ink-soft);
		}
	}
</style>
