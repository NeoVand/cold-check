<script lang="ts">
	/** The live check as an instrument: a task picker (a plain list, or the results ledger) beside the running demo. */
	import { HugeiconsIcon } from '@hugeicons/svelte';
	import { Grid3x3Icon, MazeIcon, HierarchyIcon } from '@hugeicons/core-free-icons';
	import SudokuDemo, { type SudokuData } from '#lib/demos/SudokuDemo.svelte';
	import MazeDemo, { type MazeData } from '#lib/demos/MazeDemo.svelte';
	import BfsDemo, { type BfsData } from '#lib/demos/BfsDemo.svelte';
	import type { Task, TaskId } from './tasks.ts';

	let {
		S,
		M,
		B,
		list,
		picker = 'rail',
		start = 's9'
	}: { S: SudokuData; M: MazeData; B: BfsData; list: Task[]; picker?: 'rail' | 'ledger'; start?: TaskId } = $props();

	// svelte-ignore state_referenced_locally
	let id = $state<TaskId>(start);
	const task = $derived(list.find((t) => t.id === id)!);
	const groups = ['Sudoku', 'Mazes', 'Graphs'] as const;
	const icon = { Sudoku: Grid3x3Icon, Mazes: MazeIcon, Graphs: HierarchyIcon };
	function choose(t: TaskId) {
		id = t;
	}
	const sudokuMode = { s4: '4', s9: 'watch', s16: '16', s25: '25' } as const;
	const mazeMode = { mhard: 'hard', mroom: 'room', medge: 'edge' } as const;
</script>

<div class={['inst', picker]}>
	<nav class="pick" aria-label="Tasks">
		<select class="field" value={id} onchange={(e) => choose((e.currentTarget as HTMLSelectElement).value as TaskId)} aria-label="Task">
			{#each groups as g (g)}
				<optgroup label={g}>
					{#each list.filter((t) => t.group === g) as t (t.id)}<option value={t.id}>{g === 'Sudoku' ? `Sudoku ${t.name}` : t.name}</option>{/each}
				</optgroup>
			{/each}
		</select>
		{#if picker === 'rail'}
			<div class="list">
				{#each groups as g (g)}
					<div class="grp">
						<div class="label glab"><HugeiconsIcon icon={icon[g]} size={14} strokeWidth={1.6} />{g}</div>
						{#each list.filter((t) => t.group === g) as t (t.id)}
							<button class="navitem" aria-current={id === t.id} onclick={() => choose(t.id)}>{t.name}</button>
						{/each}
					</div>
				{/each}
			</div>
		{:else}
			<div class="list ledger">
				<div class="lhead"><span></span><span>Ours</span><span>CRAM</span></div>
				{#each groups as g (g)}
					<div class="label glab"><HugeiconsIcon icon={icon[g]} size={14} strokeWidth={1.6} />{g}</div>
					{#each list.filter((t) => t.group === g) as t (t.id)}
						<button class="lrow" aria-current={id === t.id} onclick={() => choose(t.id)}>
							<span class="nm">{t.name}<small>{t.of}</small></span>
							<span class="o num">{t.ours}</span>
							<span class={['c', 'num', { short: t.cramShort }]}>{t.cram}</span>
						</button>
					{/each}
				{/each}
			</div>
		{/if}
	</nav>
	<div class="stage">
		<div class="thead">
			<h3>{task.group === 'Sudoku' ? `Sudoku ${task.name}` : task.name}</h3>
			<p class="hint">{task.asks}</p>
		</div>
		{#key id}
			<div class="demo">
				{#if task.group === 'Sudoku'}
					<SudokuDemo data={S} initial={sudokuMode[id as keyof typeof sudokuMode]} bare autoplay />
				{:else if task.group === 'Mazes'}
					<MazeDemo data={M} initial={mazeMode[id as keyof typeof mazeMode]} bare autoplay />
				{:else}
					<BfsDemo data={B} autoplay />
				{/if}
			</div>
		{/key}
	</div>
</div>

<style>
	.inst {
		display: grid;
		grid-template-columns: 188px minmax(0, 1fr);
		gap: 32px;
		align-items: start;
	}
	.inst.ledger {
		grid-template-columns: 430px minmax(0, 1fr);
		gap: 44px;
	}
	.field {
		display: none;
		width: 100%;
		height: 32px;
		border: 1px solid var(--color-edge);
		border-radius: 6px;
		background: var(--color-paper);
		color: var(--color-ink);
		font: inherit;
		font-size: 14px;
		padding: 0 8px;
	}
	@media (max-width: 860px) {
		.inst,
		.inst.ledger {
			grid-template-columns: minmax(0, 1fr);
			gap: 14px;
		}
		.field {
			display: block;
		}
		.list {
			display: none;
		}
	}
	.grp + .grp {
		margin-top: 14px;
	}
	.glab {
		display: flex;
		align-items: center;
		gap: 6px;
		margin-bottom: 4px;
	}
	.ledger .glab {
		margin: 14px 0 2px;
	}
	.lhead,
	.lrow {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto 124px;
		column-gap: 10px;
		align-items: baseline;
	}
	.lhead {
		padding: 0 8px 6px;
		border-bottom: 1px solid var(--color-ink);
		font-size: 11px;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		color: var(--color-ink-soft);
		text-align: right;
	}
	.lrow {
		width: 100%;
		padding: 6px 8px;
		border: 0;
		border-bottom: 1px solid var(--color-edge);
		border-radius: 0;
		background: none;
		color: var(--color-ink-soft);
		font: inherit;
		font-size: 13px;
		text-align: right;
		cursor: pointer;
		transition:
			background var(--hover) ease,
			color var(--hover) ease;
	}
	.lrow:hover {
		background: var(--color-tint);
	}
	.lrow[aria-current='true'] {
		background: var(--color-tint);
		box-shadow: inset 2px 0 0 var(--color-ink);
		color: var(--color-ink);
	}
	.nm {
		text-align: left;
		color: var(--color-ink);
	}
	.lrow[aria-current='true'] .nm {
		font-weight: 600;
	}
	.nm small {
		display: block;
		font-size: 11.5px;
		font-weight: 400;
		color: var(--color-ink-soft);
	}
	.o,
	.c {
		white-space: nowrap;
	}
	.o {
		color: var(--color-cobalt);
		font-weight: 600;
	}
	.c.short {
		color: var(--color-rose);
	}
	.thead {
		margin-bottom: 14px;
	}
	h3 {
		font-size: 16px;
		font-weight: 600;
	}
	.demo {
		animation: rise 420ms var(--ease);
	}
	@keyframes rise {
		from {
			opacity: 0;
			transform: translateY(6px);
		}
	}
</style>
