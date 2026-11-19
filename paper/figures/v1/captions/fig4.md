# Fig. 4 caption (FILE numbering: fig4.md is "Fig. 4", cross-references too; printed as Fig. 5). File: out/fig4.pdf (183 x 160 mm). Script: fig4.py (glyphs: tiles.py); data: code/clrs/final/
# (results.py), groups and comparators from paper/gen/make_clrs_table.py; glyph inputs data/glyphs/

\textbf{Fig. 4 | CLRS-30, fitted at the answer.} Cobalt: our fitted checks; pale cobalt: our looped blocks of the
earlier paper; rose: FloydNet. \textbf{a}, The thirty tasks, in three groups by what we supply by hand. Left,
\nm{clrs.g1.n} tasks learned with comparisons, counts and the completion of the answer alone. On a pale band:
\nm{clrs.g2.n} tasks given one arithmetic primitive and \nm{clrs.g3.n} given a piece of algorithmic structure (named
under each). Each glyph is a small input of the benchmark's own generator with its true answer drawn in navy over the
input in grey, in the layouts of our earlier paper's Fig.~5a; every fitted check returns the true answer on every
official and held-out test input. Under each task: the time to fit its rule on one laptop core (topological
sort and strong components reuse the depth-first check), and the largest held-out
size at which every input was exact (segments intersect always has four points). \textbf{b}, Percent right on the
official 64-node test: our fitted check is exact on all thirty (filled: comparisons and counts only; open: a piece supplied), against the
better of our general and cold looped blocks and FloydNet's reported score, both from our earlier paper's
Supplementary Table~4. Schematic: the glyphs' layouts (node order around each ring, the arcs of the sorting tiles, the
tints of the split tables, the placement of the 24-node graph).
