# Fig. 5 caption. File: out/fig5.pdf (183 x 86 mm). Script: fig5.py; data: costdata.py (code/bench/final, code/clrs/final,
# code/ext/cram.txt)

\textbf{Fig. 5 | Fitted in milliseconds, exact far beyond training.} Cobalt: ours (open: CLRS-30 tasks given a piece by
hand); rose: CRAM; pale rose: the looped models. Both axes are logarithmic. \textbf{a}, The time to fit one
rule: ours on one laptop core (CRAM's benchmarks: the median of warm fits; CLRS-30: the best of three), CRAM and the looped models on one H200 GPU as CRAM reports them (its Tables~2 and~3; no fit times
are reported for Maze-Hard, and N-Queens is CRAM's 8\,$\times$\,8 model). Within a row, marks that would overlap are
stacked; only their horizontal position is data. \textbf{b}, The sizes, in nodes or cells, at which one fitted rule is
exact: a line from our largest training size (tick) to the largest held-out size at which every test was exact (dot).
CRAM fits and tests each model at one size (rose); the looped models are trained and tested at the same sizes. Sudoku:
fitted at 9\,$\times$\,9, exact at 4\,$\times$\,4, 16\,$\times$\,16 and 25\,$\times$\,25 (625 cells). Mazes: exact up to 121
cells a side. N-Queens: fitted on 8\,$\times$\,8, exact at 100\,$\times$\,100. CLRS-30: one line per task, trained at
16 items (the string matchers at 20); segments intersect, at a fixed size of four points, is not drawn.
