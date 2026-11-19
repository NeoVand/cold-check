"""The final results, read in one place (paths relative to the repository's code/ tree, kit.CODE):
    code/bench/final/summary.json      every puzzle benchmark, one frozen engine (unit pairs, Luby restarts, 10^7 moves)
    code/clrs/final/summary.json       CLRS-30: per task, % right on the official test and every held-out set, fit time
    code/clrs/final/fit_times_m4.json  fit times on one laptop core (best of three; Floyd-Warshall and DFS family: one)
"""
import json

import kit

BENCH = kit.CODE / "bench" / "final"
CLRSF = kit.CODE / "clrs" / "final"
PUZ = json.loads((BENCH / "summary.json").read_text())
CL = json.loads((CLRSF / "summary.json").read_text())
CL_FIT = json.loads((CLRSF / "fit_times_m4.json").read_text())["fits"]
assert CL["_totals"]["tasks"] == 30 and CL["_totals"]["below_100"] == 0


def clrs(task):
    """fit_ms, exact_to (largest held-out size, every set exact), official score, loops."""
    d = CL[task]
    assert all(v == 1.0 for v in d["scores"].values()), task
    sizes = [int("".join(c for c in k.split("_")[0] if c.isdigit())) for k in d["scores"] if k.startswith("gen")]
    fit = CL_FIT[task]["best_ms"] if task in CL_FIT else d["fit_ms_m4"]
    return dict(fit_ms=fit, exact_to=max(sizes), test64=100.0 * d["scores"]["test64"], loops=d["loops_max"])
