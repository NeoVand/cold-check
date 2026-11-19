"""Does the edge rule (an open cell stays on only with two on side neighbours, except at the board edge) predict which
cells CRAM's released Maze-Unique model gets wrong on the job_pcd mazes? Reads run_released_pcd_*.json; runs no model.
The modal module is stubbed so modal_cram_check's maze builders import without Modal."""
import sys, types, json
import numpy as np
m = types.ModuleType('modal')
class _A:
    def __init__(s,*a,**k): pass
    def function(s,*a,**k): return lambda f: f
    def local_entrypoint(s,*a,**k): return lambda f: f
class _I:
    def __getattr__(s,n): return lambda *a,**k: s
m.App=_A; m.Image=_I()
sys.modules['modal']=m
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import modal_cram_check as C
W=str(HERE)+'/'
res={}
for h in (15,30,60):
    rng=np.random.default_rng(1000+h); mazes=[]
    for i in range(200):
        g=C.tree_maze(h,rng); op=np.argwhere(g); a,b=rng.choice(len(op),2,replace=False)
        mazes.append((g,tuple(int(v) for v in op[a]),tuple(int(v) for v in op[b])))
    for seed in range(3):
        d=json.load(open(W+f'run_released_pcd_{seed}_{h}.json'))['result']['variants']
        for var in ('ring','noring'):
            agree=0; nd=[]
            for (g,S,G),row in zip(mazes,d[var]['mazes']):
                if var=='noring': g=g[1:-1,1:-1]; S=(S[0]-1,S[1]-1); G=(G[0]-1,G[1]-1)
                H,Wd=g.shape
                on=g.copy()
                edge=np.zeros(g.shape,bool); edge[0]=edge[-1]=True; edge[:,0]=edge[:,-1]=True
                while True:
                    n=C.side_open(on)
                    off=on & (n<2) & ~edge
                    off[S]=off[G]=False
                    if not off.any(): break
                    on&=~off
                lab=np.zeros(g.shape,bool)
                for c in C.path_cells(g,S,G): lab[c]=True
                pred_diff=int((on&~lab).sum())
                got=[tuple(x['cell']) for x in row['diff']]
                mine=sorted(map(tuple,np.argwhere(on&~lab).tolist()))
                ok = pred_diff==row['n_diff'] and sorted(got)==mine[:len(got)] if len(got)<60 else pred_diff==row['n_diff'] and set(got)<=set(mine)
                agree+=ok; nd.append(row['n_diff'])
            res[(h,seed,var)]=(agree, float(np.mean(nd)), int(min(nd)))
for k,v in res.items(): print(k,'rule-predicted wrong set matches',v[0],'/200; mean wrong cells',v[1],'min',v[2])
