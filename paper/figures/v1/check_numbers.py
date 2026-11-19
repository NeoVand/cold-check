"""Every number printed in a figure that the paper also prints, asserted against ../../numbers.tex (the writer's
generated numbers), so a figure and the text cannot disagree. Run after the figures: ./run.sh check_numbers.py"""
import json
import re
from pathlib import Path
import numpy as np

NUM = dict(re.findall(r"\\nmdef\{([^}]*)\}\{([^}]*)\}", (Path(__file__).resolve().parents[2] / "numbers.tex").read_text()))
D = json.load(open("data/bfs_map.json"))
g = D["graph1024"]
af = g["as_fitted"]
checks = {
    "bfs.run.first": f"{af['settled'][0]}", "bfs.run.firstwrong": f"{af['settled'][0] - af['right'][0]}",
    "bfs.run.final": f"{af['right'][-1]}", "bfs.run.stop": f"{len(af['settled'])}",
    "bfs.run.loops": f"{max(g['kept']['settle_loop'])}", "bfs.run.n": f"{g['n'] - 1:,}",
}
import sys; sys.path.insert(0, "."); import kit
REC = json.load(open(kit.CODE / "walkthrough" / "completed_answer_bfs.json"))
checks["bfs.depth.64"] = f"{100 * REC['depth']['64']['at_answer'][0] / REC['depth']['64']['at_answer'][1]:.1f}"
checks["bfs.asfitted.64"] = f"{100 * REC['order']['64']['top'][0] / REC['order']['64']['top'][1]:.1f}"
checks["bfs.parents64"] = f"{REC['order']['64']['top'][1]:,}"
RB = json.load(open("data/released_boards.json"))
checks["rel.noring.solved"] = f"{RB['edge']['0']['solved_all']}"
bad = 0
for k, v in checks.items():
    ok = NUM.get(k) == v
    bad += not ok
    print(("ok   " if ok else "DIFF ") + f"{k}: figure {v}, numbers.tex {NUM.get(k)}")
assert bad == 0
print(len(checks), "numbers agree with numbers.tex")
