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
for n in range(1, 6):
    src = os.path.join(CAP, f'fig{n}.md')
    if os.path.exists(src):
        lines = [l for l in open(src).read().split('\n') if not l.startswith('#')]
        body = '\n'.join(lines).strip()
        m = re.match(r'\\textbf\{Fig\. *\d+ *\| *(.+?)\}\s*', body, re.S)
        assert m, src
        title, rest = m.group(1), body[m.end():]
        tex = f'\\figlabel{{Fig. \\thefigure}}{{{title}}} {rest}\n'
    else:
        tex = (f'\\figlabel{{Fig. \\thefigure}}{{{PLACEHOLDER[n]}}} '
               f'\\textcolor{{red}}{{[caption arrives with the figure]}}\n')
    open(os.path.join(OUT, f'cap_fig{n}.tex'), 'w').write(tex)
    print(n, 'caption file' if os.path.exists(src) else 'placeholder')
