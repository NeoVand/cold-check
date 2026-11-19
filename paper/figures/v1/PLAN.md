# Figures for "When the Check Is the Algorithm"

Build from this folder: `./run.sh figN.py` (each script asserts its printed numbers against the data and prints the
style kit's geometry checks), then `./run.sh check_numbers.py`, `./run.sh check_balance.py fig1 fig2 fig3 fig4 fig5`,
`./run.sh zoomall.py figN`, `./run.sh contact.py`. Every result file is read under the repository's `code/` tree
(`kit.CODE`; set `COLDCHECK_DATA` to read another copy); the final results are read in one place, `results.py`
(`code/bench/final/summary.json`, `code/clrs/final/`). Boards extracted from runs (with the final engine) and the glyph
inputs are in `data/`.
Style: `kit.py` and `house/` (Helvetica Neue 9/8/6.5 pt, navy hairlines, cobalt and rose; the system font's faces are
extracted into `house/.fonts`, which is not committed).

**Colour, the same in every figure.** Cobalt: our fitted check run cold, and our results. Rose: what it is compared
with: the check as fitted without the kept condition (Figs. 1, 2), the search on cells the cold stage leaves open,
CRAM and FloydNet. Pale rose: the looped models. Navy: inputs, givens and answers drawn as objects. Grey: checks read at
the answer (Fig. 2b) and a graph's other edges.

| Fig. | Story | Panels | Data |
|---|---|---|---|
| 1 | The check is the algorithm (breadth-first search with queue-order parents, end to end) | a training graph and answer; b completion (order of discovery); c the fitted check as pair and witness, level condition bracketed; d, e the 1,024-node run from nothing, kept and as fitted, as a radial map | `data/bfs_map.json` (`data/bfs_map.py`, which re-runs `code/walkthrough/completed_answer_bfs.py` and checks it against `data/bfs_run.json` and the record) |
| 2 | A check is not yet a procedure | a loop-by-loop run on the training graph; b parents right at 64 nodes (Table 1); c parents right per loop at 1,024 nodes; d loops against levels on 223 graphs | the same, `code/walkthrough/*.json` |
| 3 | One fitted rule on CRAM's benchmarks | a Sudoku 4 × 4 to 25 × 25 (given, set cold, set by the search); b edge maze against CRAM's released model; c room maze against CRAM; d N-Queens 100 × 100; e operations per puzzle against solved | `data/puzzle_boards.json` (`data/puzzle_boards.py`, the final engine), `data/released_boards.json` (`data/modal_released_boards.py`, CRAM's released checkpoints with their own code), `code/bench/final/summary.json`, `code/ext/cram.txt` |
| 4 | CLRS-30, fitted at the answer | a thirty glyphs in three groups by what is supplied, fit time and largest size exact under each; b official test against our looped blocks and FloydNet | `code/clrs/final/` (`results.py`); groups and comparators from `paper/gen/make_clrs_table.py`; glyph inputs `data/glyphs/` (`tiles.py`) |
| 5 | Fitted in milliseconds, exact far beyond training | a fit time per benchmark; b training size to the largest size exact | `costdata.py` (CRAM's tables from `code/ext/cram.txt`; ours from the final results) |

## Layout choices
- Fig. 1 map: a radial tree (chosen), a spring layout of the whole graph, an area-even radial
  (`candidates/contact_fig1.png`). The radial tree's rings are the loops (a node settles in the loop equal to its
  distance, asserted), so settling level by level is read off the map; the spring layout was a tangle.
- Fig. 1c: one pair-and-witness glyph with the level condition as a bracket, rather than two glyphs that differed by
  one label.
- Fig. 5: two aligned row panels (chosen) rather than one plane of fit time against size, which crowded every
  millisecond fit into one corner (`candidates/contact_fig5.png`).
- Fig. 4: three groups side by side (5, 3 and 2 columns), the tasks given a piece on a pale band.
- Fig. 3: the searched Sudoku cells keep rose digits without a cell tint; maze walls are pale.

## Checks
- Every panel zoomed at 600 dpi (`out/zoom/`); geometry checks clean apart from labels inside the marks they name.
- Numbers asserted in each script; `check_numbers.py` checks the ten that the text also prints against
  `paper/numbers.tex` (all agree). Captions print numbers through `\nm{}` where `numbers.tex` has a key.
- Colour balance in `out/balance.txt`. Figs. 4 and 5 are mostly cobalt because most of their data are ours.
- Schematic marks are named in each caption.
