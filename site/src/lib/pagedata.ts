/** What every version of the page reads from the loaded result files. */
import type { SudokuData } from '#lib/demos/SudokuDemo.svelte';
import type { MazeData } from '#lib/demos/MazeDemo.svelte';
import type { BfsData } from '#lib/demos/BfsDemo.svelte';
import type { Results } from '#lib/demos/ResultsTable.svelte';
import { tasks } from '#lib/instrument/tasks.ts';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function pageData(data: { sudoku: any; mazes: any; bfs: any; results: any }) {
	const S = data.sudoku as SudokuData;
	const M = data.mazes as MazeData;
	const B = data.bfs as BfsData;
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	const R = data.results as any;
	const RT = data.results as Results;
	const sx = R.ours.sudoku;
	const fitMs = sx.fit_s[0] * 1e3;
	const st = S.stats;
	const coldFrac = (100 * st.extreme[0].cold_only) / st.extreme[0].n_test;
	const heroInputs: { b: number; puzzle: number[]; seed: number }[] = [];
	const p4 = S.sizes['4']?.[0];
	const p9 = S.extreme.find((p) => p.index === S.demo.watch);
	const p16 = S.sizes['16']?.find((p) => p.py.moves > 100 && p.py.moves < 1000) ?? S.sizes['16']?.[0];
	const p25 = S.sizes['25']?.[0];
	if (p4) heroInputs.push({ b: 2, puzzle: p4.puzzle, seed: p4.seed });
	if (p9) heroInputs.push({ b: 3, puzzle: p9.puzzle, seed: p9.seed });
	if (p16) heroInputs.push({ b: 4, puzzle: p16.puzzle, seed: p16.seed });
	if (p25) heroInputs.push({ b: 5, puzzle: p25.puzzle, seed: p25.seed });
	return { S, M, B, R, RT, sx, fitMs, st, coldFrac, heroInputs, list: tasks(S, M, R) };
}
