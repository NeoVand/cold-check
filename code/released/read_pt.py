"""Read a torch .pt zip without unpickling: list the pickle's opcodes (pickletools, no execution) and the raw storages."""
import sys, zipfile, pickletools, io, numpy as np
p = sys.argv[1]
z = zipfile.ZipFile(p)
names = z.namelist()
pk = [n for n in names if n.endswith('data.pkl')][0]
out = io.StringIO(); pickletools.dis(z.read(pk), out=out)
keep = [l for l in out.getvalue().splitlines() if any(t in l for t in ('BINUNICODE', 'BININT', 'TUPLE', 'GLOBAL', 'STACK_GLOBAL'))]
print('\n'.join(l.strip()[:110] for l in keep[:200]))
for n in sorted(names):
    if '/data/' in n and not n.endswith('.pkl'):
        b = z.read(n)
        if len(b) <= 4096:
            print(n, len(b), 'f32', np.frombuffer(b, '<f4') if len(b) % 4 == 0 else None)
            if len(b) % 8 == 0: print('   f64', np.frombuffer(b, '<f8'))
        else: print(n, len(b), 'bytes')
