import { heroBoard } from './solve.ts';
import type { SudokuCheck } from '#lib/sudoku/engine.ts';

self.onmessage = (e: MessageEvent<{ check: SudokuCheck; boards: { b: number; puzzle: number[]; seed: number }[] }>) => {
	const { check, boards } = e.data;
	for (const x of boards) self.postMessage(heroBoard(check, x.b, x.puzzle, x.seed));
};
