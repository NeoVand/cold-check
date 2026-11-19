# Fig. 1 caption. File: out/fig1.pdf (183 x 147 mm). Script: fig1.py; data: data/bfs_map.json (data/bfs_map.py)

\textbf{Fig. 1 | The check is the algorithm.} Breadth-first search with queue-order parents (the textbook labelling), from one finished answer to a cold run on a graph 64
times larger than the largest training graph (16 nodes). Cobalt: the fitted check run cold with the level condition kept; rose: the same check as
fitted. \textbf{a}, One of the 30 training graphs (12 nodes) and its answer, each node's parent (arrows; the source
filled; grey, the graph's other edges). \textbf{b}, The completion, computed from the answer alone: each node's place in
the order of discovery, which is also its position on the line. Arcs above join each parent to its child and always
run left to right; arcs below are the other edges. \textbf{c}, The check the fit returned for ``$u$ is the parent of
$v$'', in \nm{bfs.fit} ms on one laptop core, drawn as the cold layer reads it: the pair $(u, v)$ joined by an edge, and a witness
test through a third node $k$, ``no neighbour $k$ of $v$ placed before $u$'' (the largest weights of the fit). The
bracket is the level condition, ``$u$ placed before $v$'': it held at every training parent, so a fit at the answer has
no reason to use it, and we keep it. \textbf{d}, The check run from ``not yet'' on a test graph of 1,024 nodes, with the
level condition kept: every node's parent is right, and each node is settled in the loop equal to its distance from the
source, the ring it sits on (scale below; the last at loop \nm{bfs.run.loops}). \textbf{e}, The same check as fitted, on the same graph:
it settles \nm{bfs.run.first} nodes in its first loop, \nm{bfs.run.firstwrong} of them with a wrong parent (rose chords, from the parent named to the
node), and when it stops, at loop \nm{bfs.run.stop}, \nm{bfs.run.final} of the \nm{bfs.run.n} parents are right (cobalt) and 49 nodes are unsettled (open
circles). Schematic: the node positions in a, b and the radial layout of d and e (the source at the centre, ring $r$ at
distance $r$, each subtree a wedge); the rings and their distances are data.
