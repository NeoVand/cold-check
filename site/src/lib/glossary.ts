/** Plain-words definitions behind the dotted terms of the page. */
export const GLOSSARY: Record<string, { title: string; text: string }> = {
	cold: {
		title: 'Cold attention',
		text: 'Attention run at zero temperature: each item keeps bits and small whole numbers, and a pair of items asks only "is there a third item that passes both tests?" and compares counts. Our earlier paper (arXiv 2611.11937) shows such a layer can be exact at every size.'
	},
	cram: {
		title: 'CRAM',
		text: 'An energy model in the style of Hopfield and Tank (Hoover, Blayney and Whittington, arXiv 2611.17309). It fits pair costs at solved puzzles in milliseconds, then searches for a board of low energy. One model per board size.'
	},
	pseudolikelihood: {
		title: 'Pseudolikelihood',
		text: 'Fitting by predicting each part of a solved example from all the other parts. It needs no solver and no loop, which is why the fit takes milliseconds.'
	},
	check: {
		title: 'Check',
		text: 'A rule that says whether each part of an answer agrees with the rest. It is fitted where agreement is easy to see, at the solved answer, and then used to derive and to stop.'
	},
	completion: {
		title: 'Completed answer',
		text: 'The answer plus quantities computed with closures and counts, from the answer (a node\'s depth, its place in the order of discovery) or from the input (a bottleneck weight, the reach clock). No step of the algorithm is used.'
	},
	level: {
		title: 'Level condition',
		text: '"The parent is placed before the child." It holds at every finished answer, so a fit learns nothing from it and drops it; the run from nothing needs it. We keep every condition that held on all training answers.'
	},
	witness: {
		title: 'Witness',
		text: 'The third item k through which a pair (a, b) is read: the pair asks whether some k passes a test with a and a test with b. Nothing in the test names a position or a size.'
	},
	looped: {
		title: 'Looped models',
		text: 'Networks that apply the same block many times and are trained through those loops (HRM, TRM and others). They take hours of GPU training on these puzzles.'
	},
	sudokux: {
		title: 'Sudoku-Extreme',
		text: 'A benchmark of hard 9 × 9 Sudoku (Sapient). CRAM\'s split tests on 412,786 puzzles.'
	},
	mazehard: {
		title: 'Maze-Hard',
		text: '30 × 30 mazes with loops; the label is one shortest path from S to G, chosen by a fixed tie-break. 1,000 training and 1,000 test mazes.'
	},
	mazeunique: {
		title: 'Maze-Unique',
		text: '30 × 30 mazes with exactly one path from S to G (EqR). A wall ring surrounds every training maze.'
	},
	bfs: {
		title: 'BFS parents',
		text: 'Breadth-first search from a source, labelled as CLRS-30 labels it: each node\'s parent is the neighbour that discovered it, with neighbours scanned in index order.'
	},
	clrs: {
		title: 'CLRS-30',
		text: 'The 30 classical algorithms of the CLRS benchmark (Veličković et al. 2022), tested at 64 nodes and beyond.'
	}
};
