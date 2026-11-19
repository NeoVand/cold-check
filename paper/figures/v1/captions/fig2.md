# Fig. 2 caption. File: out/fig2.pdf (183 x 104 mm). Script: fig2.py; data: data/bfs_map.json, data/bfs_run.json,
# code/walkthrough/completed_answer_bfs.json, code/walkthrough/fit_at_answer.json

\textbf{Fig. 2 | A check is not yet a procedure.} Breadth-first search with queue-order parents (the textbook
labelling), as in Fig.~1. Cobalt: the check run with the level condition kept; rose: the check
as fitted, and the fit with no completion; grey: checks read at the completed answer, not runs. \textbf{a}, The training
graph of Fig.~1a run from nothing, loop by loop. A node is filled once it has a parent and the parent is right, a ring
if the parent is wrong, white while unsettled; arrows are the parents named so far. As fitted, the check names parents
for nodes two and three steps from the source in its first loop, before anything below them is placed, and three of
them are wrong. With the level condition kept it settles one distance per loop, all right. \textbf{b}, Parents right on
100 test graphs of 64 nodes (\nm{bfs.parents64} parents): with no completion, the fit run from nothing gets 45.9\%; with depth as the
completion, the check read at the answer gets \nm{bfs.depth.64}\% (every parent it names is a valid shortest-path parent, but depth
does not hold the order of discovery that breaks the label's ties); with the order of discovery it gets 100\% read at
the answer, \nm{bfs.asfitted.64}\% run from ``not yet'' as fitted, and 100\% run with the level condition kept (also at 256 and 1,024
nodes). \textbf{c}, The 1,024-node graph of Fig.~1d,e: parents right after each loop, and the nodes the as-fitted run has
settled. Kept, every parent is right by loop \nm{bfs.run.loops}; as fitted, \nm{bfs.run.first} nodes are settled in loop 1 and
\nm{bfs.run.final} parents are right when the run stops at loop \nm{bfs.run.stop}. \textbf{d}, Loops to rest against the number of levels in the answer (the distance of
the farthest node) on all 220 test graphs of 16, 64 and 256 nodes and the three of 1,024 (disc area: number of graphs, key at right):
the kept run stops after exactly as many loops as there are levels, as Proposition~2 bounds. Schematic: the node
positions in a (rows: distance from the source).
