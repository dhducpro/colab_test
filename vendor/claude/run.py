"""Preliminary 2x2 campaign: {GA, TS} x {generic, grid}. Results appended to results.jsonl."""
import json, sys, time
from pdpcr import *
from algos import *

def instances():
    for n, T in ((50, 15), (100, 30)):
        for s in (1, 2):
            yield make_open_grid(n, 10, 50, 50, 100 * n + s), T
            yield make_aisle_grid(n, 10, 12, 40, 100 * n + s), T

done = set()
try:
    for line in open("results.jsonl"):
        d = json.loads(line); done.add((d["instance"], d["algo"], d["grid"], d["seed"]))
except FileNotFoundError:
    pass
for inst, T in instances():
    sp = Spatial(inst, 10)
    lb = inst.lower_bound()
    for seed in (0, 1):
        for algo in ("GA", "TS"):
            for grid in (False, True):
                if (inst.name, algo, grid, seed) in done:
                    continue
                tr = []
                f = genetic if algo == "GA" else tabu_search
                t0 = time.time()
                s, e, it = f(inst, sp, grid, T, seed, trace=tr)
                cm = check_feasible(inst, s)
                rec = dict(instance=inst.name, layout=inst.layout, n=inst.n, algo=algo, grid=grid,
                           seed=seed, T=T, cmax=cm, init=tr[0][1], lb=lb, iters=it,
                           time=round(time.time() - t0, 1), trace=tr)
                with open("results.jsonl", "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(inst.name, algo, grid, seed, cm, flush=True)
print("DONE", flush=True)
