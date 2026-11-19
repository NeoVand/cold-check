# Code

Everything behind the numbers of *When the Check Is the Algorithm*. The solvers are numpy (CLRS-30) and numpy + numba
(puzzles); nothing trains on a GPU. Every result file the paper and the page read is committed next to the script that
wrote it, so each number can be checked without rerunning anything.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python fetch.py          # CRAM's repository at 62bbe83 and the three puzzle datasets (about 770 MB)
```

CLRS-30 needs no download: the official splits are in `clrs/clrs30/` (see below).

## Layout

```
fetch.py        third-party code and data we use but do not redistribute (CRAM's repository; Sudoku-Extreme,
                Maze-Hard, Maze-Unique) -> ext/CRAM/, bench/data/
clrs/           CLRS-30: the cold check (cold.py), the completions (complete.py), the per-task bookkeeping (tasks.py),
                the fit-and-run of the first 19 tasks (run.py), the other 11 (extra.py, dp_extra.py, dfs2.py),
                our generator of the library's inputs and labels at any size (gen.py, gen_extra.py), the finite
                certificates (cert.py), the fitted rules in words (rules.py -> RULES.md), Modal runners, tables
bench/          CRAM's benchmarks in our setting: the role-tied check fitted at the answer (check.py), the cold
                derivation and complete search (engine.py), mazes (mazes.py), N-Queens and colouring (multi.py),
                data loaders written from scratch (data_io.py), finite proofs (proofs.py), runners
walkthrough/    the breadth-first-search walk-through of the paper (Table 1, Figure 2) and the first pilot
released/       tests of CRAM's released models and weights: their code on Modal (modal_cram_check.py), certificates
                and searches on their weights (no torch needed), the hidden-layer pointer on room mazes
```

## The paper's numbers: the final run

Every number in the paper comes from one final run of frozen code, with one search budget (10⁷ moves, restarts on)
for every puzzle:

| What | Command | Writes |
|---|---|---|
| Sudoku-Extreme (three fits, all 412,786 puzzles), the size ladder, mazes, N-Queens, colouring, proofs | `python bench/final/run_final.py sudoku ladder mazes multi proofs` | `bench/final/*.json` |
| Sudoku 25×25 (the same 100 puzzles), on Modal CPU containers | `cd bench && modal run final/modal_final.py` | `bench/final/sudoku_ladder_25.json` |
| The puzzle summary and what changed from the earlier runs | `python bench/final/summarise.py` | `bench/final/summary.json`, `bench/final/DIFF.md` |
| CLRS-30, all 30 tasks: official test, validation, held-out sizes | `cd clrs && modal run modal_final.py` (or `python final_run.py <task>` locally) | `clrs/final/<task>.json` |
| CLRS-30 fit times, one core | `clrs/reproduce.sh times` | `clrs/final/fit_times_m4.json` |
| The level ablation (the fitted check with and without its kept conditions) | `cd clrs && modal run modal_script.py --script ablation_level.py --out ablation_level.json` | `clrs/final/ablation_level.json` |
| The CLRS-30 summary | `cd clrs && python final_summary.py` | `clrs/final/summary.json` |

`bench/final/run_final.py` checks the engine files against `bench/final/ENGINE_SHA256.json` before it runs, and
`clrs/final/MANIFEST.json` lists the frozen CLRS files. The copies here differ from the files that ran only in comments,
path constants and Modal app names; the hashes as run are in `bench/final/ENGINE_SHA256_AS_RUN.json` and in
`MANIFEST.json` (`frozen_sha256`).

The sections below are the earlier runs, kept with their result files; the final run supersedes their numbers.

## One command per result

Run from `code/` unless a command says `cd`. Fit times in the paper are wall clock on one laptop core (an Apple M4),
with one BLAS thread (`OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`).

### CRAM's benchmarks (Table 2, Figure 1)

| Result | Command | Writes |
|---|---|---|
| Sudoku-Extreme: three fits (train-s0, s1, s2), each on all 412,786 test puzzles | `python bench/run_sudoku.py` (`python bench/run_sudoku.py 2000` for a smoke run) | `bench/sudoku_extreme.json` |
| The same rule at 4×4, 9×9 and 16×16 | `python bench/run_sudoku_sizes.py 2:1000 3:200 4:1000` | `bench/sudoku_sizes_local.json` |
| 25×25, as registered (100 puzzles) | `cd bench && modal run modal_bench.py::sudoku25 --n 100` | `bench/sudoku_sizes.json` |
| 25×25, the registered retest with unit pairs | `cd bench && modal run modal_bench.py::sudoku25b --n 100` | `bench/sudoku25b.json` |
| Unit pairs on all 412,786 puzzles (not registered) | `python bench/run_sudoku_pairs.py` | `bench/sudoku_extreme_pairs.json` |
| Vocabulary ablation, 12 distractor relations (A1, A1b) | `python bench/ablation_vocab.py` | `bench/ablation_vocab.json` |
| CRAM's noisy search with and without our cold stage (A2) | `python bench/gibbs.py 2000` | `bench/gibbs.json` |
| Maze-Unique, Maze-Hard, edge mazes, room mazes, looped mazes | `python bench/run_mazes.py` | `bench/mazes.json` |
| N-Queens (fit on 8×8, run at 80×80 and 100×100) and 3-colouring | `python bench/run_multi.py` | `bench/multi.json` |
| Finite checks behind the every-size statements (Sudoku types, stop, maze ranges over 108,084 tree mazes) | `python bench/proofs.py` | `bench/proofs.json` |

`bench/run_sudoku.py` checks every split against the content hashes in CRAM's split index before it fits.

### CRAM's released models

| Result | Command | Writes |
|---|---|---|
| CRAM's own test scores, room mazes, edge mazes, looped mazes, Maze-Hard maze 106 (their code and checkpoints on CPU) | `modal run released/modal_cram_check.py --which all`, then `--which stage3` | `released/run_released_*.json` |
| Summary of those runs | `python released/run_released_summary.py > released/run_released_summary.txt` | |
| Which cells the edge fault predicts wrong | `python released/run_released_edge_rule.py` | (prints) |
| CRAM+z's hidden layer and pointer on room mazes (their released numbers, numpy only) | `python released/pointer_test.py`; `python released/pointer_test2.py` | `released/pointer_room.json`, `released/pointer_L.json` |
| Is CRAM's Sudoku stop sound, from its weights | `python released/sudoku_stop_cert.py`; `python released/sudoku_stop_cert2.py`; `python released/sudoku_stop_search.py` | `released/sudoku_stop_*.json` |
| CRAM's Maze-Unique cell rule, from its weights | `python released/maze_unique_cert.py`; `python released/maze_unique_cert2.py` | `released/maze_unique_cert.json` |

These read the checkpoints in `ext/CRAM/checkpoints/` as raw storages (no unpickling); only `modal_cram_check.py`
runs CRAM's own code, with torch, in a Modal image.

### Breadth-first search, the walk-through (Table 1, Figure 2)

| Result | Command | Writes |
|---|---|---|
| Completion by depth or by order of discovery; read at the answer or run cold; with and without the level condition | `python walkthrough/completed_answer_bfs.py` | `walkthrough/completed_answer_bfs.json` |
| The first pilot: reachability and BFS parents fitted with no completion | `python walkthrough/fit_at_answer.py` | `walkthrough/fit_at_answer.json` |

### CLRS-30 (Table 3, Box 1)

`clrs/reproduce.sh <section>` runs each group; it holds the exact task groups and sizes of our runs.

| Result | Command | Writes |
|---|---|---|
| First 19 tasks: fit, official test, larger sizes from our generator | `clrs/reproduce.sh final` (Modal) or `cd clrs && python run.py bfs --big 256:32,1024:8` per task | `clrs/results/<task>.json` |
| Their frozen-code held-out batteries (val, fresh 64, 256, 1,024, sparse 1,024, 4,096) | `clrs/reproduce.sh heldout` | `clrs/results_heldout/`, `clrs/results_heldout2/` |
| The other 11 tasks (official test, val, held-out sizes) | `clrs/reproduce.sh extra` | `clrs/results_extra_heldout/` |
| Fit times, one core, best of three | `clrs/reproduce.sh times` | `clrs/fit_times_m4.json`, `clrs/fit_times_m4_extra.json` |
| 10 training inputs instead of 100 | `clrs/reproduce.sh few` | `clrs/logs/few_data.log` |
| Keep only the target's own conditions, or every condition | `clrs/reproduce.sh gate` | `clrs/sweep/` |
| Finite certificates (sorts, minimum, Kruskal, BFS) and the rules in words | `clrs/reproduce.sh proofs` | `clrs/cert.json`, `clrs/RULES.md` |
| Our generator reproduces every official test split bit for bit | `clrs/reproduce.sh generator` | `clrs/gen_verify.json`, `clrs/gen_extra_verify*.json` |
| The registered DFS test at the completed answer | `clrs/reproduce.sh dfs` | `clrs/dfs_test.json` |
| The tables | `clrs/reproduce.sh tables` | (prints) |

`clrs/results_superseded/` keeps the first 19 tasks' numbers from earlier code versions, for the record. Some choices
of the method were made after seeing official test scores (the paper lists them); the held-out batteries were run
after them, with the code frozen.

## Data and attribution

- **CLRS-30** (Veličković et al., 2022; DeepMind's `dm-clrs` 2.0.3, Apache 2.0). `clrs/clrs30/` holds the official
  test and validation splits and the training pools (sizes 4, 7, 11, 13, 16; the string matchers at 20) as padded
  numpy arrays without hints, made with the library by `witness_clrs/generate.py` of
  [NeoVand/cold-attention](https://github.com/NeoVand/cold-attention); each file's `.json` manifest holds the sha256 of
  its arrays. Three tasks' training splits that folder lacked were made by `clrs/make_train.py` with our generator
  (`clrs/data/`). `clrs/witness_clrs/` is a copy of that package's task list and scorer (no jax).
- **CRAM** (Hoover, Blayney and Whittington, arXiv:2611.17309, Apache 2.0, [Prefrontal-AI/CRAM](https://github.com/Prefrontal-AI/CRAM)).
  Not redistributed: `fetch.py` clones it at commit 62bbe83 into `ext/CRAM/`. We read its Sudoku-Extreme split index,
  its released checkpoints and its result files; our solvers do not import its code.
- **Sudoku-Extreme** (`sapientinc/sudoku-extreme` at 58942f96), **Maze-Hard** (`sapientinc/maze-30x30-hard-1k` at
  549754de) and **Maze-Unique** (`locuslab/EqR-data` at bcb7d45c) are fetched from Hugging Face by `fetch.py`, not
  redistributed.
