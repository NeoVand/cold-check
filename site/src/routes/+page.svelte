<script lang="ts">
	import type { PageProps } from './$types';
	import Hero from '#lib/hero/Hero.svelte';
	import Gloss from '#lib/ui/Gloss.svelte';
	import Links from '#lib/ui/Links.svelte';
	import CountUp from '#lib/ui/CountUp.svelte';
	import { reveal } from '#lib/ui/reveal.ts';
	import Instrument from '#lib/instrument/Instrument.svelte';
	import SudokuDemo from '#lib/demos/SudokuDemo.svelte';
	import MazeDemo from '#lib/demos/MazeDemo.svelte';
	import BfsDemo from '#lib/demos/BfsDemo.svelte';
	import ResultsTable from '#lib/demos/ResultsTable.svelte';
	import ClrsTable, { type ClrsRow } from '#lib/demos/ClrsTable.svelte';
	import { pageData } from '#lib/pagedata.ts';
	import { int, sci, pct } from '#lib/format.ts';
	import { asset } from '$app/paths';

	let { data }: PageProps = $props();
	const P = $derived(pageData(data));
	const S = $derived(P.S);
	const M = $derived(P.M);
	const R = $derived(P.R);
	const sx = $derived(P.sx);
	const st = $derived(P.st);
	const fitMs = $derived(P.fitMs);

	// every number in the text, read from the result files
	const mz = $derived(M.stats as unknown as Record<string, any>);
	const bfs = $derived(R.bfs);
	const bfsAt = (n: string) => bfs.top_gate[n] as number[];
	type CramEdge = { side: number; ring: boolean; n_per_seed: number; solved_per_seed: number[] };
	const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
	const cramEdgeNoRing = $derived(Object.values(R.cram_generated.edge as Record<string, CramEdge>).filter((e) => !e.ring));
	/** per seed: "x" when every seed agrees, else the range */
	const perSeed = (xs: number[]) => (Math.min(...xs) === Math.max(...xs) ? int(xs[0]) : `${int(Math.min(...xs))}–${int(Math.max(...xs))}`);
	const cramEdgeAgree = $derived(
		Object.values(R.cram_generated.edge as Record<string, CramEdge>).every((e) => Math.min(...e.solved_per_seed) === Math.max(...e.solved_per_seed))
	);
	const cramNoRingSolved = $derived(cramEdgeNoRing[0].solved_per_seed.map((_, k) => sum(cramEdgeNoRing.map((e) => e.solved_per_seed[k]))));
	const ourEdgeNoRing = $derived(
		Object.values(mz.unique_generated as Record<string, { ring: boolean; n: number; exact: number }>).filter((e) => !e.ring)
	);

	// CLRS-30 (null until its results exist)
	const clrs = $derived((R.clrs?.rows ?? null) as ClrsRow[] | null);
	const clrsStats = $derived.by(() => {
		if (!clrs) return null;
		const method = clrs.filter((r) => r.tier === 1);
		const fits = method.map((r) => r.fit_ms);
		return {
			n: clrs.length,
			allExact: clrs.every((r) => r.test64 === 100) && R.clrs.heldout.first.all_exact && R.clrs.heldout.other.all_exact,
			method: method.length,
			ceil: clrs.filter((r) => r.tier !== 1).length,
			fitMin: Math.min(...fits),
			fitMax: Math.max(...fits),
			avg: R.clrs.avg as { test64: number; paper_best: number; floydnet: number },
			held: R.clrs.heldout as { first: { tasks: number; sets: number }; other: { tasks: number; sets: number } }
		};
	});
	const secs = (ms: number) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toPrecision(2)} s`);
	const fig = (f: string) => asset(`figures/${f}` as Parameters<typeof asset>[0]);

	const BIB = `@misc{mohsenvand2026check,
  title  = {When the Check Is the Algorithm},
  author = {Mohsenvand, Mostafa and Jahangiri, Pouyan},
  year   = {2026},
  url    = {https://github.com/NeoVand/cold-check}
}`;
</script>

<svelte:head>
	<title>When the Check Is the Algorithm</title>
	<meta name="description" content="Our cold attention layer, fitted at solved examples in milliseconds and run cold: one rule for every board size." />
</svelte:head>

<div class="page">
	<header class="bar">
		<span class="brand"><span class="mark" aria-hidden="true"><i class="c"></i><i class="r"></i></span>Cold attention</span>
		<Links />
	</header>

	<section class="head">
		<p class="kicker">Paper · November 2026</p>
		<h1>When the Check Is the Algorithm</h1>
		<p class="by">Mostafa 'Neo' Mohsenvand · Pouyan Jahangiri</p>
	</section>

	<div class="grid lead">
		<div class="main">
			<p class="abstract">
				We fit our <Gloss term="cold">cold attention</Gloss> layer at solved examples, the way <Gloss term="cram">CRAM</Gloss> fits its
				energy: in milliseconds, with no loop. Then we run it cold. Nothing in the rule names a position or a size, so one fit serves
				every board, every maze and every graph.
			</p>
		</div>
		<aside class="margin nums" aria-label="Results at a glance">
			<div><b><CountUp value={fitMs} format={(x) => `${x.toFixed(0)} ms`} /></b><span>to fit Sudoku, on one CPU core</span></div>
			<div><b><CountUp value={sx.solved[0]} format={int} /></b><span>of {int(sx.n)} Sudoku-Extreme test puzzles</span></div>
			<div><b>25 × 25</b><span>solved by the 9 × 9 rule</span></div>
			<div><b><CountUp value={bfsAt('1024')[1]} format={int} /></b><span>BFS parents right at 1,024 nodes, fitted on 16</span></div>
		</aside>
	</div>

	<figure class="fig" {@attach reveal}>
		<div class="flabel"><span>Figure 1</span><span class="live">running in this page</span></div>
		<Hero check={S.check} inputs={P.heroInputs} {fitMs} />
		<figcaption>
			<b>Fitted once in {fitMs.toFixed(0)} ms, then the same rule on every board.</b> The fit keeps three relations of a cell: its row, its
			column and its box (left). Run on boards from 4 × 4 to 25 × 25, the rule fixes the cobalt cells cold. Where it stops, the search
			guesses a rose cell and the cold stage fixes the pale cobalt cells that follow from it.
		</figcaption>
	</figure>

	<div class="grid">
		<div class="main">
			<h2><span class="sn">1</span>Two ways to learn an algorithm</h2>
			<p>
				Neural networks that reason in loops can learn algorithms, but they learn them by training through their loops, for hours. Our
				cold attention layer is one of them: once trained it is exact at every size, and getting there took hours of GPU training per
				task. CRAM showed a faster road for puzzles. It fits a <Gloss term="check">check</Gloss> at solved examples by
				<Gloss term="pseudolikelihood">pseudolikelihood</Gloss>, in milliseconds and with no loop, then searches for a board that passes
				it. Its rules are tied to cell positions, so it needs one model per board size, and it has nowhere to keep a quantity it
				computes, such as a distance.
			</p>
			<p>
				We put the two together. We complete each answer from itself, fit our layer's check at the
				<Gloss term="completion">completed answer</Gloss> in milliseconds, keep the one condition the answer cannot show, and run the
				check cold from nothing. Where the cold run leaves something open we search, and the same check says when to stop.
			</p>
		</div>
		<aside class="margin notes">
			<p>
				<b>Attention at a temperature.</b> Attention mixes items with weights from a softmax at a temperature. As the temperature falls to
				zero, the softmax becomes a hard choice: the best item, or whether any item passes a test. Our earlier paper,
				<i>Attention at Absolute Zero</i> (arXiv 2611.11937), built a layer that runs there.
			</p>
			<p>
				<b>What CRAM does.</b> It writes an energy over pairs of cells, as Hopfield and Tank did in the 1980s, and fits its pair costs at
				solved puzzles. To answer, it lowers the energy by noisy search and stops below a fitted threshold.
			</p>
		</aside>
	</div>

	<figure class="fig" {@attach reveal}>
		<div class="flabel"><span>Figure 2</span><span class="ft">The cold block, fitted at the answer</span></div>
		<div class="pan"><img src={fig('fig_arch.svg')} alt="The cold block on a ten-node breadth-first search: the pair state, the fitted check read at zero temperature, the loop; the fit at a completed answer; the cold run loop by loop; the search on a Sudoku board" loading="lazy" /></div>
		<figcaption>
			<b>a,</b> The block's state is one bit for every ordered pair, "u is the parent of v". One loop is the fitted check: the pair's own
			tests and one <Gloss term="witness">witness</Gloss> test through every third node k, read at zero temperature, which keeps the best
			u for every v and how many pass. <b>b,</b> The fit at a completed answer keeps every condition that held at every training parent.
			<b>c,</b> Run cold from nothing, each loop settles the nodes one step further from the source. <b>d,</b> Where bits stay open, the
			search guesses (rose), and the same check stops it. Cobalt: the block's bits, counts and pointers. Graph layouts are drawn for
			clarity.
		</figcaption>
	</figure>

	<div class="grid">
		<div class="main">
			<h2><span class="sn">2</span>The recipe</h2>
			<ol class="steps">
				<li>
					<b>Complete the answer from itself.</b> Add what the label depends on but does not show, computed from the label alone: for BFS,
					each node's place in the order of discovery; for mazes, the loop at which a cold reach from the goal first touches each cell.
				</li>
				<li>
					<b>Fit the check at the completed answer,</b> by pseudolikelihood, CRAM's way: no loop, no search, milliseconds. Relations are
					named by what agrees (same row, same column, same box), never by position, so the fit has no size in it.
				</li>
				<li>
					<b>Keep every condition the answers always meet.</b> A condition that holds at every finished answer teaches the fit nothing,
					so the fit drops it; the run from nothing needs it. "The parent is placed before the child" is one, the
					<Gloss term="level">level condition</Gloss>.
				</li>
				<li>
					<b>Run cold, then search the rest.</b> Derive what is forced. If anything is open, guess, derive again, and undo on a
					contradiction. Stop when the fitted check passes everywhere.
				</li>
			</ol>
		</div>
		<aside class="margin notes">
			<p>
				<b>Cold</b> is a fixed rule applied everywhere until nothing changes, with no noise and no guesses. Its state is bits ("can this
				cell still hold a 7?") and small counts ("how many cells of my row can still hold a 7?").
			</p>
			<p><b>Warm</b> is search: guess one open cell, run cold again, and undo guesses that lead to a contradiction.</p>
		</aside>
	</div>

	<div class="grid">
		<div class="main">
			<h2><span class="sn">3</span>Sudoku at every size</h2>
			<p>
				Fitted on 9 × 9 boards, the check keeps three relations, same row, same column and same box, and with each the two count conditions
				every training answer met: a digit appears at least once and at most once. Run cold, these conditions fix every cell that is
				forced. On {P.coldFrac.toFixed(1)}% of the test puzzles that is the whole answer; over all of them the search takes a median of
				{int(st.extreme[0].moves_median)} moves. The rule mentions no size, so it reads a 16 × 16 or a 25 × 25 board as it reads a 9 × 9 one.
			</p>
		</div>
		<aside class="margin nums small">
			<div><b>{int(st.side4.solved_valid)} / {int(st.side4.n)}</b><span>4 × 4 boards</span></div>
			<div><b>{int(st.side16.solved_valid)} / {int(st.side16.n)}</b><span>16 × 16 boards</span></div>
			<div><b>{int(st.side25.solved_valid)} / {int(st.side25.n)}</b><span>25 × 25 boards</span></div>
			<div><b>0</b><span>wrong stops, any size</span></div>
		</aside>
	</div>

	<figure class="fig" {@attach reveal}>
		<div class="flabel"><span>Figure 3</span><span class="live">live</span></div>
		<SudokuDemo data={S} autoplay />
		<figcaption>
			<b>One rule, every size.</b> Cobalt digits are forced by the first cold stage. Rose digits are the search's guesses on the branch that
			reached the answer, and lighter cobalt digits were forced cold from those guesses. The run stops when the fitted check passes. The 9
			× 9 puzzles are Sudoku-Extreme test puzzles; the others come from the benchmark run's own generator and seeds. The counts match the
			Python runs move for move.
		</figcaption>
	</figure>

	<div class="grid">
		<div class="main">
			<h2><span class="sn">4</span>Mazes without a hand-built layer</h2>
			<p>
				For mazes the fit keeps side and corner neighbours with ranges of counts; an open cell on the path has exactly two path neighbours.
				Run cold from "every open cell might be on the path", that is dead-end filling, derived rather than written. Maze-Hard has loops,
				so its label is one shortest path picked by a tie-break. CRAM added a hand-built layer to hold distances. We use a cold reach from
				the goal instead: the loop at which a cell is first reached is its distance. The path rule fitted at the labels reads: step to a
				neighbour that is nearer the goal, and among those prefer {mz.hard_fit.direction_order.join(', ')}, which is the benchmark's own
				tie-break, found by the fit.
			</p>
		</div>
		<aside class="margin nums small">
			<div><b>{int(mz.hard_test.exact)} / {int(mz.hard_test.n)}</b><span>Maze-Hard test mazes</span></div>
			<div><b>{int(mz.unique_test.exact)} / {int(mz.unique_test.n)}</b><span>Maze-Unique test mazes</span></div>
			<div><b>{int(mz.hard_rooms.exact)} / {int(mz.hard_rooms.n)}</b><span>room mazes; CRAM {R.cram_generated.rooms[0].equals_label}</span></div>
		</aside>
	</div>

	<figure class="fig" {@attach reveal}>
		<div class="flabel"><span>Figure 4</span><span class="live">live</span></div>
		<MazeDemo data={M} autoplay />
		<figcaption>
			<b>Exact where CRAM's released models go wrong.</b> Every training maze has a wall ring; without it, CRAM's released Maze-Unique
			models solve {perSeed(cramNoRingSolved)} of {int(sum(cramEdgeNoRing.map((e) => e.n_per_seed)))} tree mazes (each of
			{cramNoRingSolved.length} seeds), and our check, which counts only neighbours that exist, solves
			{int(sum(ourEdgeNoRing.map((e) => e.exact)))} of {int(sum(ourEdgeNoRing.map((e) => e.n)))}. On the room mazes, CRAM's first
			seed returns the label's path on {R.cram_generated.rooms[0].equals_label} of {R.cram_generated.rooms[0].n}.
		</figcaption>
	</figure>

	<div class="grid">
		<div class="main">
			<h2><span class="sn">5</span>BFS at any size</h2>
			<p>
				BFS is where the completion matters. The label says which neighbour discovered each node; it does not say when. We complete it with
				each node's place in the order of discovery, computed from the label alone. Depth alone is not enough: it gives a valid parent but
				the wrong one among ties, {pct((100 * bfs.depth_at_answer_64[0]) / bfs.depth_at_answer_64[1], 1)} right at 64 nodes. And the fit drops
				"the parent is placed before the child", which is true at every finished answer; run from nothing without it, the rule names
				{pct((100 * bfs.order_top_no_gate_64[0]) / bfs.order_top_no_gate_64[1], 1)} of parents right. Kept, it names all
				{int(bfsAt('1024')[1])} parents of the 1,024-node test graphs right, in at most {bfsAt('1024')[3]} loops.
			</p>
		</div>
		<aside class="margin notes">
			<p>
				<b>The completion</b> is the part of the answer the label leaves out, computed from the label itself. It gives the check a
				direction: the run from nothing can only grow outwards.
			</p>
		</aside>
	</div>

	<figure class="fig" {@attach reveal}>
		<div class="flabel"><span>Figure 5</span><span class="ft">The check is the algorithm</span></div>
		<div class="pan"><img src={fig('fig1.svg')} alt="Breadth-first search: a training graph and its answer, the completion as the order of discovery, the fitted check, and the check run cold on 1,024 nodes with and without the level condition" loading="lazy" /></div>
		<figcaption>
			<b>a,</b> A training graph and its answer, each node's parent. <b>b,</b> The completion: each node's place in the order of discovery;
			arcs above join each parent to its child and always run left to right. <b>c,</b> The fitted check for "u is the parent of v", drawn as
			the cold layer reads it; the bracket is the level condition, which the fit had no reason to use. <b>d,</b> Run cold with the level
			condition kept on a graph of 1,024 nodes: every parent right, each node settled at the loop equal to its distance from the source.
			<b>e,</b> The same check as fitted, on the same graph (rose: wrong parents). Node positions are drawn for clarity; the rings are data.
		</figcaption>
	</figure>

	<figure class="fig" {@attach reveal}>
		<div class="flabel"><span>Figure 6</span><span class="live">live</span></div>
		<BfsDemo data={P.B} autoplay />
		<figcaption>
			<b>Fitted in milliseconds on 16 nodes, exact on 1,024.</b> The page fits the rule in your browser, at the completed answers of the 30
			training graphs of the Python run, and gets the same weights. Test graphs are drawn fresh in the page: a random tree plus about one
			extra edge per node. Rings mark the distance from the source (drawing aid); cobalt edges are the rule's parents, found loop by loop
			from nothing. A parent that differs from BFS would show in rose.
		</figcaption>
	</figure>

	<div class="grid">
		<div class="main">
			<h2><span class="sn">6</span>Results</h2>
			<p>
				Every benchmark, with the check that solves it running live: pick a row. Operations are counted the way CRAM counts them: an
				addition counts one, a comparison nothing.
			</p>
		</div>
	</div>

	<figure class="fig bleed" {@attach reveal}>
		<div class="flabel"><span>Figure 7</span><span class="live">live</span><span class="ft">Pick a row to run it</span></div>
		<Instrument S={S} M={M} B={P.B} list={P.list} picker="ledger" />
	</figure>

	<figure class="fig" {@attach reveal}>
		<div class="flabel"><span>Table 1</span><span class="ft">Against CRAM and the looped models of CRAM's own tables</span></div>
		<ResultsTable r={P.RT} />
		<figcaption>
			Fit times: ours on one CPU core (the warm median of seven fits), the others on one H200 GPU. Operations: the mean over all test
			puzzles; counting every comparison too, ours is {sci(sx.ops_cmp)} operations per Sudoku-Extreme puzzle. CRAM and the looped models:
			CRAM's paper (Table 2; Table 8 for Maze-Hard). † Accuracy published by the method's authors.
		</figcaption>
	</figure>

	<figure class="fig" {@attach reveal}>
		<div class="flabel"><span>Table 2</span><span class="ft">One rule, every size</span></div>
		<div class="sizes">
			<table class="mini">
				<thead><tr><th class="l">Task</th><th>Ours</th><th>CRAM</th></tr></thead>
				<tbody>
					<tr><td class="l">Sudoku 4 × 4</td><td>{int(st.side4.solved_valid)} / {int(st.side4.n)}</td><td>one model per size</td></tr>
					<tr><td class="l">Sudoku 9 × 9 (Extreme)</td><td>{int(sx.solved[0])} / {int(sx.n)}</td><td>{R.cram.sudoku.solved_pct.toFixed(3)}%</td></tr>
					<tr><td class="l">Sudoku 16 × 16</td><td>{int(st.side16.solved_valid)} / {int(st.side16.n)}</td><td>one model per size</td></tr>
					<tr><td class="l">Sudoku 25 × 25</td><td>{int(st.side25.solved_valid)} / {int(st.side25.n)}</td><td>one model per size</td></tr>
					{#each Object.values(mz.unique_generated as Record<string, { side: number; ring: boolean; n: number; exact: number }>) as e (e.side)}
						{@const c = R.cram_generated.edge[`${e.side}_${e.ring ? 'ring' : 'noring'}`]}
						<tr>
							<td class="l">Tree maze {e.side} × {e.side}, {e.ring ? 'wall ring' : 'no ring'}</td>
							<td>{int(e.exact)} / {int(e.n)}</td>
							<td class={{ bad: Math.max(...c.solved_per_seed) < c.n_per_seed }}>{perSeed(c.solved_per_seed)} / {int(c.n_per_seed)}</td>
						</tr>
					{/each}
					{#each Object.entries(mz.hard_loops as Record<string, { n: number; exact: number }>) as [side, e] (side)}
						{@const c = R.cram_generated.loops[side]}
						<tr><td class="l">Looped maze {side} × {side}</td><td>{int(e.exact)} / {int(e.n)}</td><td>{int(c.exact)} / {int(c.n)}</td></tr>
					{/each}
				</tbody>
			</table>
			<table class="mini">
				<thead><tr><th class="l">BFS nodes</th><th>Parents right</th><th>Loops</th></tr></thead>
				<tbody>
					{#each ['16', '64', '256', '1024'] as n (n)}
						<tr><td class="l">{int(Number(n))}</td><td>{int(bfsAt(n)[0])} / {int(bfsAt(n)[1])}</td><td>≤ {bfsAt(n)[3]}</td></tr>
					{/each}
				</tbody>
			</table>
		</div>
		<figcaption>
			CRAM: its released models, run on the same mazes; tree mazes per seed{cramEdgeAgree ? ' (all three seeds agree)' : ''}, looped mazes
			seed 0. BFS: the rule of Figure 6, fitted on 8 to 16 nodes.
		</figcaption>
	</figure>

	{#if clrs && clrsStats}
		<div class="grid">
			<div class="main">
				<h2><span class="sn">7</span>CLRS-30, fitted at the answer</h2>
				<p>
					Fitted the same way, at the finished answer and with no loop, {clrsStats.allExact ? 'all' : ''} {clrsStats.n}
					<Gloss term="clrs">CLRS-30</Gloss> algorithms are exact on the official 64-node test and on every held-out set we ran, with the
					code frozen. For {clrsStats.method}, the check uses only comparisons and counts, and each fit takes {secs(clrsStats.fitMin)} to
					{secs(clrsStats.fitMax)} on one laptop core. For the other {clrsStats.ceil} we supply one piece, named in the table: an arithmetic
					primitive (a sum, an orientation test, products and sums) or a piece of algorithmic structure (prefix sums, a deletion closure,
					"open", two passes).
					{#if R.shared}One check also serves five graph algorithms at once: fitted in {secs(R.shared.fit_ms)} on the pooled answers of BFS,
						Prim, Bellman-Ford, Dijkstra and DAG shortest paths, with nothing telling it which task it is running, it scores
						{pct(100 * Math.min(R.shared.official64, R.shared.fresh1024), 0)} on all five, at 64 and at 1,024 nodes.{/if}
				</p>
			</div>
			<aside class="margin nums">
				<div><b>{clrsStats.n} / {clrsStats.n}</b><span>CLRS-30 algorithms exact</span></div>
				<div><b>{clrsStats.avg.test64.toFixed(2)}%</b><span>average; FloydNet {clrsStats.avg.floydnet.toFixed(2)}%</span></div>
			</aside>
		</div>

		<figure class="fig" {@attach reveal}>
			<div class="flabel"><span>Table 3</span><span class="ft">All {clrsStats.n} exact</span></div>
			<ClrsTable rows={clrs} avg={clrsStats.avg} />
			<figcaption>
				Our fitted checks in cobalt where they use comparisons and counts only, muted where a primitive or structure was supplied. The best
				of our earlier paper's looped blocks (arXiv 2611.11937, cold or general, trained through their loops) in rose where it missed;
				FloydNet as reported. The first {clrsStats.held.first.tasks} tasks passed {clrsStats.held.first.sets} held-out sets and the other
				{clrsStats.held.other.tasks} passed {clrsStats.held.other.sets}, all with frozen code. Exact to: the largest held-out size at which
				every held-out set is exact. Fit times on one laptop core.
			</figcaption>
		</figure>
	{/if}

	<div class="grid">
		<div class="main">
			<h2><span class="sn">{clrs ? 8 : 7}</span>Conclusion</h2>
			<p>
				A check fitted at the answer only has to agree with the answer. It produces the answer when its support has a direction, and the
				completed answer supplies that direction. With it, our cold layer trains in milliseconds, CRAM's way, and keeps what made it worth
				training: one rule, exact at every size.
			</p>
			<h3>Cite</h3>
			<pre class="bib">{BIB}</pre>
			<p class="sources">
				Every number on this page is read from <code>static/data/*.json</code>, exported from the benchmark runs
				(<a href="https://github.com/NeoVand/cold-check/tree/main/code"><code>code/</code></a> in the repository) and from CRAM's released
				results; each file names its sources.
			</p>
		</div>
	</div>
</div>

<style>
	.page {
		max-width: 1080px;
		margin: 0 auto;
		padding: 18px 28px 112px;
	}
	@media (max-width: 640px) {
		.page {
			padding: 14px 18px 80px;
		}
	}
	.bar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 16px;
		font-size: 13px;
	}
	.brand {
		display: inline-flex;
		align-items: center;
		gap: 9px;
		font-weight: 600;
		white-space: nowrap;
	}
	.mark {
		display: inline-flex;
		gap: 3px;
	}
	.mark i {
		width: 8px;
		height: 8px;
		border-radius: 50%;
	}
	.mark .c {
		background: var(--color-cobalt);
	}
	.mark .r {
		background: var(--color-rose);
	}
	.head {
		padding-top: 72px;
	}
	@media (max-width: 640px) {
		.head {
			padding-top: 40px;
		}
	}
	.kicker {
		font-size: 13px;
		color: var(--color-ink-soft);
	}
	h1 {
		margin-top: 10px;
		font-family: var(--font-serif);
		font-size: clamp(36px, 5.6vw, 56px);
		font-weight: 400;
		line-height: 1.04;
		letter-spacing: -0.015em;
		font-variation-settings: 'opsz' 60;
		animation: rise 900ms var(--ease) both;
	}
	.by {
		margin-top: 14px;
		font-size: 15px;
		animation: rise 900ms 120ms var(--ease) both;
	}
	@keyframes rise {
		from {
			opacity: 0;
			transform: translateY(10px);
		}
	}
	/* a text column and a margin column, as in a paper */
	.grid {
		display: grid;
		grid-template-columns: minmax(0, 660px) minmax(0, 1fr);
		gap: 56px;
		margin-top: 72px;
	}
	.grid.lead {
		margin-top: 30px;
		animation: rise 900ms 220ms var(--ease) both;
	}
	@media (max-width: 860px) {
		.grid {
			grid-template-columns: minmax(0, 1fr);
			gap: 22px;
			margin-top: 56px;
		}
	}
	.main p,
	.steps li {
		font-size: 16.5px;
		line-height: 1.62;
	}
	.main p + p {
		margin-top: 0.9em;
	}
	.main .abstract {
		font-size: 19.5px;
		line-height: 1.5;
	}
	.main b {
		font-weight: 600;
	}
	h2 {
		margin-bottom: 0.55em;
		font-family: var(--font-serif);
		font-size: 30px;
		font-weight: 420;
		letter-spacing: -0.01em;
		line-height: 1.15;
		font-variation-settings: 'opsz' 40;
	}
	.sn {
		display: inline-block;
		min-width: 1.15em;
		color: var(--color-mist);
	}
	h3 {
		margin: 2em 0 0.5em;
		font-size: 11px;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		font-weight: 500;
		color: var(--color-ink-soft);
	}
	.steps {
		margin: 0;
		padding-left: 1.4em;
		list-style: decimal;
	}
	.steps li {
		padding-left: 0.3em;
	}
	.steps li + li {
		margin-top: 0.7em;
	}
	.steps li::marker {
		font-family: var(--font-serif);
		color: var(--color-blue-grey);
	}
	/* the margin: key numbers, or notes in place of a fold-out */
	.margin {
		padding-top: 6px;
	}
	.nums {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 18px 20px;
		align-content: start;
	}
	.nums div {
		display: flex;
		flex-direction: column;
		gap: 3px;
	}
	.nums b {
		font-size: 24px;
		font-weight: 600;
		line-height: 1.05;
		color: var(--color-cobalt);
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
	}
	.nums.small b {
		font-size: 18px;
	}
	.nums span {
		font-size: 12px;
		line-height: 1.35;
		color: var(--color-ink-soft);
	}
	@media (max-width: 860px) {
		.grid:not(.lead) .nums {
			grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
		}
	}
	.notes {
		display: flex;
		flex-direction: column;
		gap: 14px;
		padding-left: 14px;
		border-left: 1px solid var(--color-edge);
		align-self: start;
	}
	.notes p {
		font-size: 13px;
		line-height: 1.5;
		color: var(--color-ink-soft);
	}
	.notes b {
		color: var(--color-ink);
		font-weight: 600;
	}
	/* figures: no box; a top rule with the figure's label, the caption below */
	.fig {
		margin: 64px 0 0;
	}
	.fig.bleed {
		width: min(1240px, calc(100vw - 48px));
		margin-left: calc(50% - min(620px, 50vw - 24px));
	}
	@media (max-width: 640px) {
		.fig {
			margin-top: 52px;
		}
		.fig.bleed {
			width: 100%;
			margin-left: 0;
		}
	}
	.flabel {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 12px;
		align-items: baseline;
		margin-bottom: 20px;
		padding-top: 10px;
		border-top: 1px solid var(--color-ink);
		font-size: 12px;
		font-weight: 600;
		letter-spacing: 0.03em;
	}
	.ft {
		font-weight: 400;
		color: var(--color-ink-soft);
	}
	.live {
		font-weight: 500;
		color: var(--color-cobalt);
	}
	.live::before {
		content: '';
		display: inline-block;
		width: 6px;
		height: 6px;
		margin-right: 6px;
		border-radius: 50%;
		background: var(--color-cobalt);
		vertical-align: 1px;
		animation: breathe 2.4s var(--ease) infinite;
	}
	@keyframes breathe {
		50% {
			opacity: 0.3;
		}
	}
	.fig img {
		display: block;
		width: 100%;
		height: auto;
	}
	/* a paper figure on a phone: kept legible, panned sideways */
	@media (max-width: 640px) {
		.pan {
			overflow-x: auto;
			margin: 0 -18px;
			padding: 0 18px 6px;
		}
		.pan img {
			width: 720px;
			max-width: none;
		}
	}
	figcaption {
		max-width: 760px;
		margin-top: 18px;
		font-size: 14px;
		line-height: 1.55;
		color: var(--color-ink-soft);
	}
	figcaption b {
		color: var(--color-ink);
		font-weight: 600;
	}
	.sizes {
		display: grid;
		grid-template-columns: 1.4fr 1fr;
		gap: 48px;
		align-items: start;
	}
	@media (max-width: 760px) {
		.sizes {
			grid-template-columns: minmax(0, 1fr);
			gap: 28px;
		}
	}
	.mini {
		width: 100%;
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
		white-space: normal;
		color: var(--color-ink);
		font-weight: 400;
	}
	.bib {
		overflow-x: auto;
		padding: 14px 16px;
		background: var(--color-tint);
		border-radius: 6px;
		font-family: var(--font-mono);
		font-size: 12.5px;
		line-height: 1.55;
		color: var(--color-ink);
	}
	.sources {
		margin-top: 1.6em !important;
		font-size: 13px !important;
		color: var(--color-ink-soft);
	}
	code {
		font-family: var(--font-mono);
		font-size: 0.92em;
	}
</style>
