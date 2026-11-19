"""Import the figure captions written beside the figures (figures/v1/captions/figN.md) into sections/cap_figN.tex.
The leading "\\textbf{Fig. N | Title.}" becomes \\figlabel{Fig. \\thefigure}{Title.} so numbering follows the paper.
A figure without a caption file gets a red placeholder. Run: python3 paper/gen/make_captions.py"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
CAP = os.path.join(PAPER, 'figures', 'v1', 'captions')
OUT = os.path.join(PAPER, 'sections')
PLACEHOLDER = {
    3: 'One rule for every board.',
    4: 'Thirty algorithms from their answers.',
    5: 'Fitted in milliseconds, exact far beyond training.',
}
LABEL = {1: 'fig:idea', 2: 'fig:theory', 3: 'fig:puzzles', 4: 'fig:clrs', 5: 'fig:cost'}


def header_number(name):
    m = re.search(r'\\textbf\{Fig\. *(\d+)', open(os.path.join(CAP, name)).read())
    return int(m.group(1))


# The caption files may number figures by file (fig1.md is "Fig. 1") or as printed (fig1.md is "Fig. 2", the
# architecture figure being Fig. 1). Read the convention from fig1.md's own header, and check every file agrees.
OFFSET = header_number('fig1.md') - 1 if os.path.exists(os.path.join(CAP, 'fig1.md')) else 0
assert OFFSET in (0, 1), OFFSET
for n in range(1, 6):
    if os.path.exists(os.path.join(CAP, f'fig{n}.md')):
        assert header_number(f'fig{n}.md') == n + OFFSET, (n, OFFSET)


def refs(text):
    # point cross-references at labels so the printed numbers follow the paper; "paper's Fig." is the earlier paper
    def to_label(m):
        k = int(m.group(1)) - OFFSET
        return 'Fig.~\\ref{' + ('fig:arch' if k == 0 else LABEL[k]) + '}'
    return re.sub(r"(?<!paper's )Fig\.~(\d)(?=[a-z,]|\b)", to_label, text)


PLACEHOLDER['_arch'] = 'The cold attention block.'
for n in [1, 2, 3, 4, 5, '_arch']:
    src = os.path.join(CAP, f'fig{n}.md')
    if os.path.exists(src):
        lines = [l for l in open(src).read().split('\n') if not l.startswith('#')]
        body = '\n'.join(lines).strip()
        m = re.match(r'\\textbf\{Fig\. *\d+ *\| *(.+?)\}\s*', body, re.S)
        assert m, src
        title, rest = m.group(1), body[m.end():]
        if n == '_arch':                     # the BFS fit time as a number key, checked against numbers.tex
            num = dict(re.findall(r'\\nmdef\{([^}]*)\}\{([^}]*)\}', open(os.path.join(PAPER, 'numbers.tex')).read()))
            for lit in re.findall(r'(\d+) ms', rest):
                assert lit == num['bfs.clrs.fit'], (lit, num['bfs.clrs.fit'])
            rest = re.sub(r'\b(\d+) ms', r'\\nm{bfs.clrs.fit} ms', rest)
        tex = f'\\figlabel{{Fig. \\thefigure}}{{{title}}} {refs(rest)}\n'
    else:
        tex = (f'\\figlabel{{Fig. \\thefigure}}{{{PLACEHOLDER[n]}}} '
               f'\\textcolor{{red}}{{[caption arrives with the figure]}}\n')
    open(os.path.join(OUT, f'cap_fig{n}.tex'), 'w').write(tex)
    print(n, 'caption file' if os.path.exists(src) else 'placeholder')
