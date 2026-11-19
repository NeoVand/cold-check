<script lang="ts">
	import type { PageProps } from './$types';
	import Hero from '#lib/hero/Hero.svelte';
	import Gloss from '#lib/ui/Gloss.svelte';
	import Fold from '#lib/ui/Fold.svelte';
	import SudokuDemo, { type SudokuData } from '#lib/demos/SudokuDemo.svelte';
	import MazeDemo, { type MazeData } from '#lib/demos/MazeDemo.svelte';
	import BfsDemo, { type BfsData } from '#lib/demos/BfsDemo.svelte';
	import ResultsTable, { type Results } from '#lib/demos/ResultsTable.svelte';
	import ClrsTable, { type ClrsRow } from '#lib/demos/ClrsTable.svelte';
	import { int, sci, pct } from '#lib/format.ts';
	import { C } from '#lib/theme.ts';
	import { asset } from '$app/paths';

	let { data }: PageProps = $props();
	const S = $derived(data.sudoku as unknown as SudokuData);
	const M = $derived(data.mazes as unknown as MazeData);
	const B = $derived(data.bfs as unknown as BfsData);
	const R = $derived(data.results as any);
	const RT = $derived(data.results as unknown as Results);

	// every number in the text, read from the result files
	const sx = $derived(R.ours.sudoku);
	const fitMs = $derived(sx.fit_s[0] * 1e3);
	const st = $derived(S.stats);
	const coldFrac = $derived((100 * st.extreme[0].cold_only) / st.extreme[0].n_test);
	const mz = $derived(M.stats as unknown as Record<string, any>);
	const bfs = $derived(R.bfs);
	const bfsAt = (n: string) => bfs.top_gate[n] as number[];
	type CramEdge = { side: number; ring: boolean; n_per_seed: number; solved_per_seed: number[] };
	const cramEdgeNoRing = $derived(Object.values(R.cram_generated.edge as Record<string, CramEdge>).filter((e) => !e.ring));
	/** per seed: "x of n" when every seed agrees, else the range */
	const perSeed = (xs: number[]) => (Math.min(...xs) === Math.max(...xs) ? int(xs[0]) : `${int(Math.min(...xs))}–${int(Math.max(...xs))}`);
	const cramEdgeAgree = $derived(
		Object.values(R.cram_generated.edge as Record<string, CramEdge>).every((e) => Math.min(...e.solved_per_seed) === Math.max(...e.solved_per_seed))
	);
	const cramNoRingSolved = $derived(cramEdgeNoRing[0].solved_per_seed.map((_, k) => sum(cramEdgeNoRing.map((e) => e.solved_per_seed[k]))));
	const ourEdgeNoRing = $derived(
		Object.values(mz.unique_generated as Record<string, { ring: boolean; n: number; exact: number }>).filter((e) => !e.ring)
	);
	const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

	// the hero runs the real rule on one board of each size
	const heroInputs = $derived.by(() => {
		const out: { b: number; puzzle: number[]; seed: number }[] = [];
		const p4 = S.sizes['4']?.[0];
		const p9 = S.extreme.find((p) => p.index === S.demo.watch);
		const p16 = S.sizes['16']?.find((p) => p.py.moves > 100 && p.py.moves < 1000) ?? S.sizes['16']?.[0];
		const p25 = S.sizes['25']?.[0];
		if (p4) out.push({ b: 2, puzzle: p4.puzzle, seed: p4.seed });
		if (p9) out.push({ b: 3, puzzle: p9.puzzle, seed: p9.seed });
		if (p16) out.push({ b: 4, puzzle: p16.puzzle, seed: p16.seed });
		if (p25) out.push({ b: 5, puzzle: p25.puzzle, seed: p25.seed });
		return out;
	});

	// CLRS-30 (null until its results exist)
	const clrs = $derived((R.clrs?.rows ?? null) as ClrsRow[] | null);
	const clrsStats = $derived.by(() => {
		if (!clrs) return null;
		const method = clrs.filter((r) => r.tier === 1);
		const fits = method.map((r) => r.fit_ms);
		const num = (r: ClrsRow) => (typeof r.exact_to === 'number' ? r.exact_to : 0);
		const ceil = clrs.filter((r) => r.tier !== 1);
		return {
			n: clrs.length,
			total: R.clrs.n_tasks_total as number,
			allExact: clrs.every((r) => r.test64 === 100) && R.clrs.heldout.first.all_exact && R.clrs.heldout.other.all_exact,
			method: method.length,
			methodExact: method.filter((r) => r.test64 === 100 && typeof r.exact_to === 'number').length,
			tier2: clrs.filter((r) => r.tier === 2).length,
			tier3: clrs.filter((r) => r.tier === 3).length,
			ceil: ceil.length,
			fitMin: Math.min(...fits),
			fitMax: Math.max(...fits),
			nodes: Math.max(...method.filter((r) => !r.strings).map(num)),
			items: Math.max(...method.filter((r) => r.strings).map(num)),
			avg: R.clrs.avg as { test64: number; paper_best: number; floydnet: number },
			held: R.clrs.heldout as { first: { tasks: number; sets: number }; other: { tasks: number; sets: number } }
		};
	});
	const secs = (ms: number) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toPrecision(2)} s`);

	let copied = $state(false);
	function copy() {
		navigator.clipboard?.writeText(location.href);
		copied = true;
		setTimeout(() => (copied = false), 1600);
	}
</script>

<svelte:head>
	<title>When the Check Is the Algorithm</title>
	<meta name="description" content="Our cold attention layer, fitted at solved examples in milliseconds and run cold: one rule for every board size." />
</svelte:head>

<header class="bar">
	<span class="mark" aria-hidden="true"><i style:background={C.cobalt}></i><i style:background={C.rose}></i></span>
	<span class="brand">Cold attention</span>
</header>

<main>
	<section class="head">
		<p class="meta">November 2026 · Research</p>
		<h1>When the Check Is the Algorithm</h1>
		<p class="sub">We fit our cold attention layer at solved examples, the way CRAM fits its energy, and run it cold. One rule serves every size.</p>
		<nav class="pills" aria-label="Links">
			<a class="pill solid" href={asset('paper/main.pdf' as Parameters<typeof asset>[0])}>
				<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"
					><path d="M3.5 1.5h5l2.5 2.5v8.5h-7.5z M8.5 1.5V4H11 M5.3 7h3.4M5.3 9.4h3.4" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linejoin="round" /></svg
				>Read the paper</a
			>
			<a class="pill" href="https://github.com/NeoVand/cold-check">
				<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"
					><path d="M5 4 2 7l3 3M9 4l3 3-3 3" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round" /></svg
				>Code</a
			>
			<button class="pill ghost" onclick={copy}>
				<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"
					><path d="M6 8a2.5 2.5 0 0 0 3.5 0l2-2A2.5 2.5 0 0 0 8 2.5l-.8.8M8 6a2.5 2.5 0 0 0-3.5 0l-2 2A2.5 2.5 0 0 0 6 11.5l.8-.8" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" /></svg
				>{copied ? 'Copied' : 'Copy link'}</button
			>
		</nav>
	</section>

	<figure class="card hero-card">
		<Hero check={S.check} inputs={heroInputs} fitMs={fitMs} />
		<figcaption class="caption">
			<b>Fitted once in {fitMs.toFixed(0)} ms, then the same rule on every board.</b>
			The fit keeps three relations of a cell: its row, its column and its box (left). Run on boards from 4 × 4 to 25 × 25, the
			rule fixes the cobalt cells cold. Where it stops, the search guesses a rose cell and the cold stage fixes the pale cobalt
			cells that follow from it. Each board is a real run of the fitted rule in this page.
		</figcaption>
	</figure>

	<article class="prose-col">
		<p>
			Neural networks that reason in loops can learn algorithms, but they learn them by training through their loops, for
			hours. Our <Gloss term="cold">cold attention</Gloss> layer is one of them: once trained it is exact at every size, and
			getting there took hours of GPU training per task. <Gloss term="cram">CRAM</Gloss> showed a faster road for puzzles. It
			fits a <Gloss term="check">check</Gloss> at solved examples by <Gloss term="pseudolikelihood">pseudolikelihood</Gloss>,
			in milliseconds and with no loop, then searches for a board that passes it. Its rules are tied to cell positions, so it
			needs one model per board size, and it has nowhere to keep a quantity it computes, such as a distance.
		</p>
		<p>
			We put the two together. We complete each answer from itself, fit our layer's check at the
			<Gloss term="completion">completed answer</Gloss> in milliseconds, keep the one condition the answer cannot show, and run
			the check cold from nothing. Where the cold run leaves something open we search, and the same check says when to stop.
			Nothing in the rule names a position or a size, so one fit serves every size.
		</p>
		<p>
			On CRAM's own benchmarks, a check fitted in {fitMs.toFixed(0)} ms on one CPU core solves all {int(sx.n)}
			<Gloss term="sudokux">Sudoku-Extreme</Gloss> test puzzles (CRAM: {R.cram.sudoku.solved_pct.toFixed(3)}%), with {sci(sx.ops)}
			operations per puzzle where CRAM uses {sci(R.cram.sudoku.flops)}. The same 9 × 9 rule solves 4 × 4, 16 × 16 and 25 × 25
			Sudoku: {int(st.side4.solved_valid)} of {int(st.side4.n)}, {int(st.side16.solved_valid)} of {int(st.side16.n)} and
			{int(st.side25.solved_valid)} of {int(st.side25.n)}. It is exact on
			every <Gloss term="mazehard">Maze-Hard</Gloss> and <Gloss term="mazeunique">Maze-Unique</Gloss> test maze, including the
			room mazes and the mazes with dead ends at the edge where CRAM's released models go wrong. Beyond CRAM's benchmarks, a rule
			fitted in {bfs.fit_ms.toFixed(0)} ms on graphs of up to 16 nodes names every <Gloss term="bfs">BFS parent</Gloss> right on graphs
			of 1,024{#if clrsStats}, and all {clrsStats.n} CLRS-30 algorithms, fitted at the answer, are exact on the official test and
				on every held-out set{/if}.
		</p>

		<Fold title="Attention at a temperature">
			<p>
				Attention gives each item a weighted mix of the others, with weights from a softmax at a temperature. Warm, it blends
				many items. As the temperature falls to zero the softmax becomes a hard choice: the best item, or whether any item passes
				a test. Our earlier paper, <i>Attention at Absolute Zero</i> (arXiv 2611.11937), built a layer that runs at zero temperature
				on pairs of items and proved where such a layer is exact at every size.
			</p>
		</Fold>
		<Fold title="What cold means here">
			<p>
				Cold is a fixed rule applied everywhere until nothing changes, with no noise and no guesses. Its state is bits ("can
				this cell still hold a 7?") and small counts ("how many cells of my row can still hold a 7?"). A pair of items reads every
				third item through its two legs, a <Gloss term="witness">witness</Gloss>, and asks only whether such an item exists, or
				compares two counts. Warm is search: when the cold run stops with cells still open, we guess one, run cold again, and undo
				guesses that lead to a contradiction.
			</p>
		</Fold>
		<Fold title="What CRAM does">
			<p>
				CRAM writes an energy over pairs of cells, as Hopfield and Tank did in the 1980s, and fits its pair costs at solved
				puzzles: each cell is predicted from the other cells of its own solved board. That needs no solver, which is why it
				trains at least 1,000 times faster than <Gloss term="looped">looped models</Gloss>. To answer, it lowers the energy by
				noisy search and stops when the energy falls below a threshold fitted from the training answers. Its costs are tied to
				cell positions, so a 9 × 9 model cannot read a 16 × 16 board, and on Maze-Hard it needs a hand-built extra layer to hold
				distances.
			</p>
		</Fold>

		<h2>The recipe</h2>
		<ol class="steps">
			<li>
				<b>Complete the answer from itself.</b> Add what the label depends on but does not show, computed from the label alone:
				for BFS, each node's place in the order of discovery; for mazes, the loop at which a cold reach from the goal first
				touches each cell.
			</li>
			<li>
				<b>Fit the check at the completed answer,</b> by pseudolikelihood, CRAM's way: no loop, no search, milliseconds.
				Relations are named by what agrees (same row, same column, same box), never by position, so the fit has no size in it.
			</li>
			<li>
				<b>Keep every condition the answers always meet.</b> Some conditions hold at every finished answer, so they teach the
				fit nothing, and the fit drops them; the run from nothing needs them. "The parent is placed before the child" is one, the
				<Gloss term="level">level condition</Gloss>.
			</li>
			<li>
				<b>Run cold, then search the rest.</b> Derive what is forced. If anything is open, guess, derive again, and undo on a
				contradiction. Stop when the fitted check passes everywhere.
			</li>
		</ol>

		<h2>Sudoku at every size</h2>
		<p>
			Fitted on 9 × 9 boards, the check keeps three relations, same row, same column and same box, and with each the two count
			conditions every training answer met: a digit appears at least once and at most once. Run cold, these conditions fix every
			cell that is forced. On {coldFrac.toFixed(1)}% of the test puzzles that is the whole answer; over all of them the search takes
			a median of {int(st.extreme[0].moves_median)} moves. The rule mentions no size, so it reads a 16 × 16 or a 25 × 25 board as
			it reads a 9 × 9 one.
		</p>
	</article>

	<figure class="card">
		<SudokuDemo data={S} />
		<figcaption class="caption">
			<b>One rule, every size.</b> Cobalt digits are forced by the first cold stage. Rose digits are the search's guesses on the
			branch that reached the answer, and lighter cobalt digits were forced cold from those guesses. The run stops when the
			fitted check passes. The 9 × 9 puzzles are Sudoku-Extreme test puzzles; the others come from the benchmark run's
			own generator and seeds. The counts match the Python runs move for move. On 25 × 25 the rule solves
			{int(st.side25.solved_valid)} of {int(st.side25.n)} puzzles and never stops at a wrong board.
		</figcaption>
	</figure>

	<article class="prose-col">
		<h2>Mazes without a hand-built layer</h2>
		<p>
			For mazes the fit keeps side and corner neighbours with ranges of counts; an open cell on the path has exactly two path
			neighbours. Run cold from "every open cell might be on the path", that is dead-end filling, derived rather than written.
			Maze-Hard has loops, so its label is one shortest path picked by a tie-break. CRAM added a hand-built layer to hold
			distances. We use a cold reach from the goal instead: the loop at which a cell is first reached is its distance. The path
			rule fitted at the labels reads: step to a neighbour that is nearer the goal, and among those prefer
			{mz.hard_fit.direction_order.join(', ')}, which is the benchmark's own tie-break, found by the fit.
		</p>
	</article>

	<figure class="card">
		<MazeDemo data={M} />
		<figcaption class="caption">
			<b>Exact where CRAM's released models go wrong.</b> We solve {int(mz.hard_test.exact)} of {int(mz.hard_test.n)} Maze-Hard
			and {int(mz.unique_test.exact)} of {int(mz.unique_test.n)} Maze-Unique test mazes, and {int(mz.hard_rooms.exact)} of
			{int(mz.hard_rooms.n)} room-and-corridor mazes (CRAM: {R.cram_generated.rooms[0].equals_label} of {R.cram_generated.rooms[0].n}).
			Every training maze has a wall ring; without it, CRAM's released Maze-Unique models solve {perSeed(cramNoRingSolved)}
			of {int(sum(cramEdgeNoRing.map((e) => e.n_per_seed)))} tree mazes (each of {cramNoRingSolved.length} seeds), and our check,
			which counts only neighbours that exist, solves
			{int(sum(ourEdgeNoRing.map((e) => e.exact)))} of {int(sum(ourEdgeNoRing.map((e) => e.n)))}.
		</figcaption>
	</figure>

	<article class="prose-col">
		<h2>BFS at any size</h2>
		<p>
			BFS is where the completion matters. The label says which neighbour discovered each node; it does not say when. We
			complete it with each node's place in the order of discovery, computed from the label alone. Depth alone is not enough:
			it gives a valid parent but the wrong one among ties, {pct((100 * bfs.depth_at_answer_64[0]) / bfs.depth_at_answer_64[1], 1)}
			right at 64 nodes. And the fit drops "the parent is placed before the child", which is true at every finished answer; run
			from nothing without it, the rule names {pct((100 * bfs.order_top_no_gate_64[0]) / bfs.order_top_no_gate_64[1], 1)} of
			parents right. Kept, it names all {int(bfsAt('1024')[1])} parents of the 1,024-node test graphs right, in at most
			{bfsAt('1024')[3]} loops.
		</p>
	</article>

	<figure class="card">
		<BfsDemo data={B} />
		<figcaption class="caption">
			<b>Fitted in milliseconds on 16 nodes, exact on 1,024.</b> The page fits the rule in your browser, at the completed answers
			of the 30 training graphs of the Python run, and gets the same weights. Test graphs are drawn fresh in the page: a random
			tree plus about one extra edge per node. Rings mark the distance from the source (drawing aid); cobalt edges are the
			rule's parents, found loop by loop from nothing. A parent that differs from BFS would show in rose.
		</figcaption>
	</figure>

	<article class="prose-col">
		<h2>Results</h2>
		<p>
			Against CRAM and the looped models of CRAM's own tables. Operations are counted the way CRAM counts them: an addition
			counts one, a comparison nothing.
		</p>
	</article>

	<figure class="card">
		<ResultsTable r={RT} />
		<figcaption class="caption">
			<b>Every benchmark solved, from fits of milliseconds.</b> Fit times: ours on one CPU core (the warm median of seven fits),
			the others on one H200 GPU. Operations: the mean over all test puzzles; counting every comparison too, ours is {sci(sx.ops_cmp)} operations per Sudoku-Extreme puzzle. CRAM and the
			looped models: CRAM's paper (Table 2; Table 8 for Maze-Hard). † Accuracy published by the method's authors.
		</figcaption>
	</figure>

	<figure class="card">
		<div class="sizes">
			<div>
				<div class="label">One rule, every size</div>
				<table class="mini">
					<thead><tr><th class="l">Task</th><th>Ours</th><th>CRAM</th></tr></thead>
					<tbody>
						<tr><td class="l">Sudoku 4 × 4</td><td>{int(st.side4.solved_valid)} / {int(st.side4.n)}</td><td>one model per size</td></tr>
						<tr><td class="l">Sudoku 9 × 9 (Extreme)</td><td>{int(sx.solved[0])} / {int(sx.n)}</td><td>{R.cram.sudoku.solved_pct.toFixed(3)}%</td></tr>
						<tr><td class="l">Sudoku 16 × 16</td><td>{int(st.side16.solved_valid)} / {int(st.side16.n)}</td><td>one model per size</td></tr>
						<tr><td class="l">Sudoku 25 × 25</td><td>{int(st.side25.solved_valid)} / {int(st.side25.n)}</td><td>one model per size</td></tr>
						{#each Object.values(mz.unique_generated as Record<string, { side: number; ring: boolean; n: number; exact: number }>) as e (e.side)}
							{@const c = R.cram_generated.edge[`${e.side}_${e.ring ? 'ring' : 'noring'}`]}
							<tr>
								<td class="l">Tree maze {e.side} × {e.side}, {e.ring ? 'wall ring' : 'no ring'}</td>
								<td>{int(e.exact)} / {int(e.n)}</td>
								<td class={{ bad: Math.max(...c.solved_per_seed) < c.n_per_seed }}>{perSeed(c.solved_per_seed)} / {int(c.n_per_seed)}</td>
							</tr>
						{/each}
						{#each Object.entries(mz.hard_loops as Record<string, { n: number; exact: number }>) as [side, e] (side)}
							{@const c = R.cram_generated.loops[side]}
							<tr><td class="l">Looped maze {side} × {side}</td><td>{int(e.exact)} / {int(e.n)}</td><td>{int(c.exact)} / {int(c.n)}</td></tr>
						{/each}
					</tbody>
				</table>
				<p class="foot">
					CRAM: its released models, run on the same mazes; tree mazes per seed{cramEdgeAgree ? ' (all three seeds agree)' : ''},
					looped mazes seed 0.
				</p>
			</div>
			<div>
				<div class="label">BFS parents (the demo's rule), fitted on 8 to 16 nodes</div>
				<table class="mini">
					<thead><tr><th class="l">Nodes</th><th>Parents right</th><th>Loops</th></tr></thead>
					<tbody>
						{#each ['16', '64', '256', '1024'] as n (n)}
							<tr><td class="l">{int(Number(n))}</td><td>{int(bfsAt(n)[0])} / {int(bfsAt(n)[1])}</td><td>≤ {bfsAt(n)[3]}</td></tr>
						{/each}
					</tbody>
				</table>

			</div>
		</div>
	</figure>

	{#if clrs && clrsStats}
		<article class="prose-col">
			<h2>CLRS-30, fitted at the answer</h2>
			<p>
				Fitted the same way, at the finished answer and with no loop, {clrsStats.allExact ? 'all' : ''} {clrsStats.n} <Gloss
					term="clrs">CLRS-30</Gloss
				> algorithms are exact on the official 64-node test and on every held-out set we ran, with the code frozen. For
				{clrsStats.method}, the check uses only comparisons and counts, and each fit takes {secs(clrsStats.fitMin)} to
				{secs(clrsStats.fitMax)} on one laptop core. For the other {clrsStats.ceil} we supply one piece, named in the table: an
				arithmetic primitive (a sum, an orientation test, products and sums) or a piece of algorithmic structure (prefix sums, a
				deletion closure, "open", two passes).
				{#if R.shared}One check also serves five graph algorithms at once: fitted in {secs(R.shared.fit_ms)} on the pooled
					answers of BFS, Prim, Bellman-Ford, Dijkstra and DAG shortest paths, with nothing telling it which task it is
					running, it scores {pct(100 * Math.min(R.shared.official64, R.shared.fresh1024), 0)} on all five, at 64 and at
					1,024 nodes.{/if}
			</p>
		</article>

		<figure class="card">
			<ClrsTable rows={clrs} avg={clrsStats.avg} />
			<figcaption class="caption">
				<b>All {clrsStats.n} exact.</b> Our fitted checks in cobalt where they
				use comparisons and counts only, muted where a primitive or structure was supplied. The best of our earlier
				paper's looped blocks (arXiv 2611.11937, cold or general, trained through their loops) in rose where it missed; FloydNet as
				reported. The first {clrsStats.held.first.tasks} tasks
				passed {clrsStats.held.first.sets} held-out sets and the other {clrsStats.held.other.tasks} passed
				{clrsStats.held.other.sets}, all with frozen code. Exact to: the largest held-out size at which every held-out set is exact.
				Fit times on one laptop core.
			</figcaption>
		</figure>
	{/if}

	<article class="prose-col">
		<h2>Conclusion</h2>
		<p>
			A check fitted at the answer only has to agree with the answer. It produces the answer when its support has a direction,
			and the completed answer supplies that direction. With it, our cold layer trains in milliseconds, CRAM's way, and keeps
			what made it worth training: one rule, exact at every size.
		</p>
		<p class="sources">
			Every number on this page is read from <code>static/data/*.json</code>, exported from the benchmark runs
			(<a href="https://github.com/NeoVand/cold-check/tree/main/code"><code>code/</code></a> in the repository) and from CRAM's released results; each file names its sources.
		</p>
	</article>
</main>

<style>
	.bar {
		display: flex;
		align-items: center;
		gap: 10px;
		max-width: 1080px;
		margin: 0 auto;
		padding: 18px 24px;
		font-size: 13px;
	}
	.mark {
		display: inline-flex;
		gap: 3px;
	}
	.mark i {
		display: block;
		width: 8px;
		height: 8px;
		border-radius: 50%;
	}
	.brand {
		font-weight: 600;
		letter-spacing: 0.01em;
	}
	main {
		padding-bottom: 96px;
	}
	.head {
		max-width: 900px;
		margin: 0 auto;
		padding: 56px 20px 8px;
		text-align: center;
	}
	.meta {
		font-size: 13px;
		color: var(--color-ink-soft);
	}
	h1 {
		margin: 14px auto 0;
		max-width: 820px;
		font-family: var(--font-serif);
		font-size: clamp(36px, 6.2vw, 56px);
		font-weight: 400;
		line-height: 1.06;
		letter-spacing: -0.015em;
		font-variation-settings: 'opsz' 60;
	}
	.sub {
		max-width: 640px;
		margin: 18px auto 0;
		font-size: 19px;
		line-height: 1.45;
		color: var(--color-ink-soft);
	}
	.pills {
		display: flex;
		flex-wrap: wrap;
		justify-content: center;
		gap: 8px;
		margin-top: 26px;
	}
	.pill.ghost {
		border-color: transparent;
		color: var(--color-ink-soft);
	}
	.hero-card {
		margin-top: 36px;
		padding-top: 30px;
	}
	@media (max-width: 640px) {
		.head {
			padding-top: 28px;
		}
		.sub {
			font-size: 17px;
		}
		.hero-card {
			margin-top: 26px;
			padding-top: 16px;
		}
	}
	.steps {
		margin: 0;
		padding-left: 1.3em;
		list-style: decimal;
	}
	.steps li {
		padding-left: 0.3em;
		margin-top: 0.7em;
	}
	.steps li::marker {
		color: var(--color-ink-soft);
		font-variant-numeric: tabular-nums;
	}
	.steps b,
	.prose-col b {
		font-weight: 600;
	}
	.sizes {
		display: grid;
		grid-template-columns: 1.25fr 1fr;
		gap: 36px;
	}
	@media (max-width: 720px) {
		.sizes {
			grid-template-columns: 1fr;
			gap: 28px;
		}
	}
	.mini {
		width: 100%;
		margin-top: 8px;
		border-collapse: collapse;
		font-size: 13.5px;
		font-variant-numeric: tabular-nums;
	}
	.mini th {
		padding: 6px 8px;
		border-bottom: 1px solid var(--color-ink);
		color: var(--color-ink-soft);
		font-weight: 500;
		font-size: 12px;
		text-align: right;
	}
	.mini td {
		padding: 5px 8px;
		border-bottom: 1px solid var(--color-edge);
		text-align: right;
		white-space: nowrap;
	}
	.mini td:nth-child(2) {
		color: var(--color-cobalt);
		font-weight: 600;
	}
	.mini td:nth-child(3) {
		color: var(--color-ink-soft);
	}
	.mini td.bad {
		color: var(--color-rose);
	}
	.mini tbody tr:last-child td {
		border-bottom: 1px solid var(--color-ink);
	}
	.mini .l {
		text-align: left;
		color: var(--color-ink);
		font-weight: 400;
	}
	.foot {
		margin-top: 8px;
		font-size: 12px;
		color: var(--color-ink-soft);
	}
	.sources {
		margin-top: 2.4em !important;
		font-size: 13px !important;
		color: var(--color-ink-soft);
		text-align: left !important;
	}
	code {
		font-family: var(--font-mono);
		font-size: 0.92em;
	}
</style>
