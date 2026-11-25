"""Euclidean Li&Lim-like instances, generic variants, long budgets with anytime traces."""
import json, math, random, time
from pdpcr import *
from algos import *

def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def make_lilim_like(n, m, kind, seed, rmax=5):
    rng = random.Random(seed)
    centers = [(rng.randint(10, 90), rng.randint(10, 90)) for _ in range(8)]
    def pt(clustered):
        if clustered:
            cx, cy = rng.choice(centers)
            return (min(100, max(0, round(rng.gauss(cx, 6)))), min(100, max(0, round(rng.gauss(cy, 6)))))
        return (rng.randint(0, 100), rng.randint(0, 100))
    P, D, R = [], [], []
    for _ in range(n):
        cl = kind == "C" or (kind == "RC" and rng.random() < 0.5)
        p = pt(cl)
        while True:
            d = pt(cl)
            if d != p:
                break
        P.append(p); D.append(d); R.append(rng.randint(1, rmax))
    return Instance(f"L{kind}_n{n}_s{seed}", "euclid-" + kind, 101, 101, (50, 50), P, D, R, m, euclid)

VARIANTS = {"TS": dict(f=tabu_search, kw={}),
            "GA": dict(f=genetic, kw={}),
            "GA+": dict(f=genetic, kw=dict(pl=0.8)),
            "GAc": dict(f=genetic, kw=dict(crit_mut=True, pm=0.8))}

T = 120
done = set()
try:
    for l in open("results2.jsonl"):
        d = json.loads(l); done.add((d["instance"], d["algo"], d["seed"]))
except FileNotFoundError:
    pass
for kind in ("R", "C", "RC"):
    inst = make_lilim_like(100, 10, kind, 7)
    sp = Spatial(inst, 10)
    for seed in (0, 1):
        for name, v in VARIANTS.items():
            if (inst.name, name, seed) in done:
                continue
            tr = []
            s, e, it = v["f"](inst, sp, False, T, seed, trace=tr, **v["kw"])
            cm = check_feasible(inst, s)
            with open("results2.jsonl", "a") as fh:
                fh.write(json.dumps(dict(instance=inst.name, algo=name, seed=seed, T=T, cmax=cm,
                                         init=tr[0][1], iters=it, trace=tr)) + "\n")
            print(inst.name, name, seed, round(cm, 1), it, flush=True)
print("DONE", flush=True)
