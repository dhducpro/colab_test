"""Check redistribute() on Example 3 of Nguyen et al.: Cmax 24 -> 22 at cut zeta=4."""
import random
from pdpcr import *

class Mini: pass
inst = Mini()
inst.n = 5; inst.m = 7
S,A,B,C,D,E,T = range(7)
inst.r = [7,5,2,2,4,3,7]
inst.c = [0,1,2,4,3,1,0]
BIG = 100
inst.t = [[BIG]*7 for _ in range(7)]
for (i,j,v) in [(S,A,4),(S,B,7),(A,C,5),(B,C,3),(D,T,2),(E,T,3),
                (A,D,10),(A,E,4),(B,D,2),(B,E,3),(C,D,1),(C,E,4)]:
    inst.t[i][j] = v
sol = empty_sol(5)
for (i,j,f) in [(S,A,5),(S,B,2),(A,D,4),(A,C,1),(B,C,1),(B,E,1),(C,E,2),(D,T,4),(E,T,3)]:
    add_flow(sol,i,j,f)
ev = evaluate(inst, sol)
print("initial Cmax", ev.cmax)
ev.order = [S,B,A,C,D,E,T]
for k,v in enumerate(ev.order): ev.pos[v]=k
for grid in (False, True):
    class SP: pass
    sp = SP(); sp.cand = [set(range(1,7)) for _ in range(7)]
    new = redistribute(inst, sol, ev, 4, grid, random.Random(0), sp)
    e2 = evaluate(inst, new)
    print("grid" if grid else "generic", "-> Cmax", e2.cmax,
          {(a,b):f for a in (A,B,C) for b,f in new[0][a].items() if b in (D,E)})
