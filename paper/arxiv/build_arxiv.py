"""Build the arXiv source bundle of "When the Check Is the Algorithm" (run from anywhere: python3 -I build_arxiv.py).

1. Compiles ../main.tex with tectonic, keeping main.bbl.
2. Flattens every \\input, replaces every \\nm{key} by its value from ../numbers.tex, inlines the bibliography (.bbl),
   points every figure at figures/figN.pdf (copied from ../figures/v1/out/), strips every comment, and maps the few
   non-ASCII characters to LaTeX so that pdfLaTeX reads the file as arXiv will.
3. Writes src/main.tex and src/figures/, and when-the-check-is-the-algorithm-arxiv.tar.gz.
Nothing here uploads anything. Check the bundle under pdfLaTeX with check_pdflatex.py.
"""
import pathlib
import re
import shutil
import subprocess
import tarfile

HERE = pathlib.Path(__file__).resolve().parent
PAPER = HERE.parent
SRC = HERE / 'src'
TGZ = HERE / 'when-the-check-is-the-algorithm-arxiv.tar.gz'

subprocess.run(['tectonic', '-X', 'compile', '--keep-intermediates', 'main.tex'], cwd=PAPER, check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

NUM = {}
for line in (PAPER / 'numbers.tex').read_text().splitlines():
    m = re.match(r'\\nmdef\{([^}]+)\}\{(.*)\}\s*$', line)
    if m:
        NUM[m.group(1)] = m.group(2)


def strip_comments(t):
    out = []
    for line in t.split('\n'):
        m = re.search(r'(?<!\\)%', line)
        if m is None:
            out.append(line)
            continue
        head = line[:m.start()]
        if head.strip() == '':
            continue                                   # a comment line: dropped, so it makes no paragraph break
        out.append(head.rstrip() + '%' if not head.endswith(' ') else head.rstrip())
    return '\n'.join(out)


def flatten(path):
    t = strip_comments(path.read_text())

    def inc(m):
        name = m.group(1)
        if name.endswith('numbers.tex'):
            return ''
        p = PAPER / name                               # TeX resolves \input from the main file's folder
        if p.suffix != '.tex':
            p = p.with_suffix('.tex')
        return flatten(p.resolve()).strip()
    return re.sub(r'\\input\{([^}]+)\}', inc, t)


t = flatten(PAPER / 'main.tex')

for pat in (r'\\newcommand\{\\nmdef\}[^\n]*\n', r'\\newcommand\{\\nm\}[^\n]*\n', r'\\newcommand\{\\figat\}[^\n]*\n',
            r'\\newcommand\{\\fig\}[^\n]*\n', r'\\newcommand\{\\figw\}[^\n]*\n'):
    t, n = re.subn(pat, '', t)
    assert n == 1, pat
if '\\pend{' not in t.split('\\newcommand{\\pend}', 1)[1]:
    t, n = re.subn(r'\\newcommand\{\\pend\}[^\n]*\n', '', t)
    assert n == 1
missing = sorted(set(re.findall(r'\\nm\{([^}]+)\}', t)) - set(NUM))
assert not missing, missing
t = re.sub(r'\\nm\{([^}]+)\}', lambda m: NUM[m.group(1)], t)
assert '[caption arrives with the figure]' not in t, 'a figure caption is still a placeholder'

if SRC.exists():
    shutil.rmtree(SRC)
(SRC / 'figures').mkdir(parents=True)
FIGS = []


def figure(name):
    src = PAPER / 'figures' / 'v1' / 'out' / f'{name}.pdf'
    assert src.exists(), src
    FIGS.append(src)
    shutil.copyfile(src, SRC / 'figures' / f'{name}.pdf')
    return r'\includegraphics[width=\linewidth]{figures/' + name + '.pdf}'


t = re.sub(r'\\fig\{([^}]+)\}', lambda m: figure(m.group(1)), t)
t = re.sub(r'\\figw\{([^}]+)\}\{([^}]+)\}',
           lambda m: figure(m.group(2)).replace('width=\\linewidth', 'width=' + m.group(1) + '\\linewidth'), t)

bbl = (PAPER / 'main.bbl').read_text()
t, n = re.subn(r'\\bibliographystyle\{[^}]+\}\s*\\bibliography\{[^}]+\}', lambda m: bbl.strip(), t)
assert n == 1

for a, b in (('\u2212', '$-$'), ('\u00d7', r'$\times$'), ('\u2265', r'$\geq$'), ('\u2264', r'$\leq$'),
             ('\u2192', r'$\to$'), ('\u2018', '`'), ('\u2019', "'"), ('\u201c', '``'), ('\u201d', "''"),
             ('\u2013', '--'), ('\u2014', '---'), ('\u2026', r'\ldots{}'), ('\u00a0', '~')):
    t = t.replace(a, b)
nonascii = sorted({c for c in t if ord(c) > 127})
t = re.sub(r'\n{3,}', '\n\n', t)
assert not re.search(r'(?<![\w.])/(?:Users|home)/', t), 'an absolute local path in the source'
(SRC / 'main.tex').write_text(t)

with tarfile.open(TGZ, 'w:gz') as tar:
    tar.add(SRC / 'main.tex', arcname='main.tex')
    for f in sorted((SRC / 'figures').iterdir()):
        tar.add(f, arcname=f'figures/{f.name}')
print(f'{len(NUM)} numbers flattened; {len(FIGS)} figures; non-ASCII left: {"".join(nonascii) or "none"}')
print('wrote', TGZ)
