/** The tasks the page's instrument can run, each with the result it stands for (read from the result files). */
import { int } from '#lib/format.ts';

export type TaskId = 's4' | 's9' | 's16' | 's25' | 'mhard' | 'mroom' | 'medge' | 'bfs';
export type Task = {
	id: TaskId;
	group: 'Sudoku' | 'Mazes' | 'Graphs';
	name: string;
	asks: string;
	/** our result and CRAM's on the benchmark the task stands for */
	ours: string;
	cram: string;
	/** CRAM fails or cannot run it */
	cramShort: boolean;
	of: string;
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function tasks(S: any, M: any, R: any): Task[] {
	const st = S.stats;
	const mz = M.stats;
	const sx = R.ours.sudoku;
	const noRing = Object.values(mz.unique_generated as Record<string, { ring: boolean; n: number; exact: number }>).filter((e) => !e.ring);
	const cramNoRing = Object.values(R.cram_generated.edge as Record<string, { ring: boolean; n_per_seed: number; solved_per_seed: number[] }>).filter((e) => !e.ring);
	const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
	const room0 = R.cram_generated.rooms[0];
	const b1024 = R.bfs.top_gate['1024'] as number[];
	const perSize = 'one model per size';
	return [
		{ id: 's4', group: 'Sudoku', name: '4 × 4', asks: 'The 9 × 9 rule on 4 × 4 boards.', ours: `${int(st.side4.solved_valid)} / ${int(st.side4.n)}`, cram: perSize, cramShort: true, of: 'generated boards' },
		{
			id: 's9',
			group: 'Sudoku',
			name: '9 × 9, Extreme',
			asks: 'Sudoku-Extreme, the benchmark CRAM reports.',
			ours: `${int(sx.solved[0])} / ${int(sx.n)}`,
			cram: `${R.cram.sudoku.solved_pct.toFixed(3)}%`,
			cramShort: false,
			of: 'test puzzles'
		},
		{ id: 's16', group: 'Sudoku', name: '16 × 16', asks: 'The same 9 × 9 rule, never fitted on a 16 × 16 board.', ours: `${int(st.side16.solved_valid)} / ${int(st.side16.n)}`, cram: perSize, cramShort: true, of: 'generated boards' },
		{ id: 's25', group: 'Sudoku', name: '25 × 25', asks: 'The same 9 × 9 rule on 625 cells.', ours: `${int(st.side25.solved_valid)} / ${int(st.side25.n)}`, cram: perSize, cramShort: true, of: 'generated boards' },
		{
			id: 'mhard',
			group: 'Mazes',
			name: 'Maze-Hard',
			asks: 'A shortest path through a 30 × 30 maze with loops.',
			ours: `${int(mz.hard_test.exact)} / ${int(mz.hard_test.n)}`,
			cram: `${R.cram.hard.solved_pct.toFixed(1)}%`,
			cramShort: false,
			of: 'test mazes'
		},
		{
			id: 'mroom',
			group: 'Mazes',
			name: 'Room mazes',
			asks: 'Rooms and corridors: many shortest paths, one label.',
			ours: `${int(mz.hard_rooms.exact)} / ${int(mz.hard_rooms.n)}`,
			cram: `${int(room0.equals_label)} / ${int(room0.n)}`,
			cramShort: true,
			of: 'room mazes'
		},
		{
			id: 'medge',
			group: 'Mazes',
			name: 'Tree mazes, no ring',
			asks: 'The one path through a tree maze whose dead ends reach the edge.',
			ours: `${int(sum(noRing.map((e) => e.exact)))} / ${int(sum(noRing.map((e) => e.n)))}`,
			cram: `${int(sum(cramNoRing.map((e) => e.solved_per_seed[0])))} / ${int(sum(cramNoRing.map((e) => e.n_per_seed)))}`,
			cramShort: true,
			of: 'tree mazes'
		},
		{
			id: 'bfs',
			group: 'Graphs',
			name: 'BFS parents',
			asks: 'Each node names the neighbour that discovered it. Fitted on 16 nodes.',
			ours: `${int(b1024[0])} / ${int(b1024[1])}`,
			cram: 'not run',
			cramShort: true,
			of: 'parents at 1,024 nodes'
		}
	];
}
