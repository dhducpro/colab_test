import json, statistics as st
from collections import defaultdict
R = [json.loads(l) for l in open("results2.jsonl")]
best = defaultdict(lambda: 1e18)
for r in R: best[r["instance"]] = min(best[r["instance"]], r["cmax"])
def at(tr, t):
    v = tr[0][1]
    for (tt, c) in tr:
        if tt <= t: v = c
    return v
algos = ["TS", "GA", "GA+", "GAc"]
insts = sorted(best)
print("Final gap (%) to best found, per instance (mean over 2 seeds)")
for a in algos:
    row = [st.mean(100*(r["cmax"]-best[i])/best[i] for r in R if r["instance"]==i and r["algo"]==a) for i in insts]
    print(f"{a:4s}", " ".join(f"{x:5.1f}" for x in row), " mean", round(st.mean(row),1))
print(insts)
print("\nAnytime mean gap (%) over all instances and seeds")
cps = [6, 12, 30, 60, 120]
curves = {}
for a in algos:
    vals = []
    for t in cps:
        vals.append(st.mean(100*(at(r["trace"], t)-best[r["instance"]])/best[r["instance"]] for r in R if r["algo"]==a))
    curves[a] = vals
    print(f"{a:4s}", " ".join(f"{t}s:{v:5.1f}" for t,v in zip(cps, vals)))
print("\nInitial gap", {a: round(st.mean(100*(r["init"]-best[r["instance"]])/best[r["instance"]] for r in R if r["algo"]==a),1) for a in algos})
print("Pairwise: every TS run better than every run of", {a: all(max(r["cmax"] for r in R if r["instance"]==i and r["algo"]=="TS") < min(r["cmax"] for r in R if r["instance"]==i and r["algo"]==a) for i in insts) for a in algos[1:]})
json.dump({"cps": cps, "curves": curves}, open("curves.json","w"))
