import json, statistics as st
from collections import defaultdict
R = [json.loads(l) for l in open("results.jsonl")]
best = defaultdict(lambda: 10**9)
for r in R: best[r["instance"]] = min(best[r["instance"]], r["cmax"])
V = [("GA", False), ("GA", True), ("TS", False), ("TS", True)]
name = {("GA",False):"GA-gen",("GA",True):"GA-grid",("TS",False):"TS-gen",("TS",True):"TS-grid"}
print("Mean gap (%) to best value found, by class")
classes = sorted({(r["layout"], r["n"]) for r in R}, key=lambda x:(x[1],x[0]))
for cl in classes:
    row = []
    for v in V:
        g = [100*(r["cmax"]-best[r["instance"]])/best[r["instance"]] for r in R if (r["layout"],r["n"])==cl and (r["algo"],r["grid"])==v]
        row.append(f"{name[v]} {st.mean(g):5.2f} (n={len(g)})" if g else f"{name[v]}   -  ")
    print(cl, " | ".join(row))
print("\nMean gap (%) to lower bound, construction gap, iterations")
for v in V:
    rs=[r for r in R if (r["algo"],r["grid"])==v]
    if rs:
        print(name[v], "gapLB", round(st.mean(100*(r["cmax"]-r["lb"])/r["lb"] for r in rs),1),
              "init gap to best", round(st.mean(100*(r["init"]-best[r["instance"]])/best[r["instance"]] for r in rs),1),
              "iters", round(st.mean(r["iters"] for r in rs)))
# wins per instance (mean over seeds)
print("\nBest mean per instance:")
for inst in sorted(best):
    m = {v: st.mean(r["cmax"] for r in R if r["instance"]==inst and (r["algo"],r["grid"])==v) for v in V
         if any(r["instance"]==inst and (r["algo"],r["grid"])==v for r in R)}
    print(inst, {name[v]: round(x,1) for v,x in m.items()})
