import { build, masks, propagate, Solver, type SudokuCheck } from '#lib/sudoku/engine.ts';

/** cell kinds: 1 clue, 2 forced by the first cold stage, 3 guessed by the search, 4 derived cold after a guess */
export type HeroBoard = { side: number; clue: number[]; seq: [number, number][]; moves: number; guesses: number };

/**
 * Solve one board with the fitted check, then replay its final branch: the cold stage's cells in the order it fixed
 * them, then each guess on the branch that reached the answer followed by the cells the cold derivation fixed from it.
 */
export function heroBoard(check: SudokuCheck, b: number, puzzle: number[], seed: number): HeroBoard {
	const side = b * b;
	const E = build(check, b);
	const s = new Solver(E, masks(puzzle, side), seed).run();
	const clue: number[] = [];
	puzzle.forEach((v, i) => v > 0 && clue.push(i));
	const seq: [number, number][] = s.coldFixed.map((c) => [c, 2]);
	const replay = new Solver(E, masks(puzzle, side), seed);
	replay.cold();
	const dom = replay.dom.slice();
	const inq = new Uint8Array(E.unitItems.length);
	const queue = new Int32Array(E.unitItems.length);
	const st = { ops: 0, cmp: 0 };
	const guesses = s.status === 1 ? s.branch() : [];
	for (const g of guesses) {
		if ((dom[g] & (dom[g] - 1)) === 0) continue; // already fixed by an earlier derivation
		dom[g] = s.dom[g];
		seq.push([g, 3]);
		for (const u of E.itemUnits[g]) inq[u] = 1;
		propagate(dom, E, inq, queue, st, (c) => seq.push([c, 4]));
		inq.fill(0);
	}
	return { side, clue, seq, moves: s.moves, guesses: guesses.length };
}
