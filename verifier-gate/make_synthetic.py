"""Build a COMPLETE, VALID synthetic bo3 cache (no 0.5s) so the gate can be
proven to pass on good data and scream on each way of breaking it."""
import sys, os, json, random
ROOT = os.environ.get("LLMV_ROOT") or os.getcwd()   # point at your llm-as-a-verifier checkout
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,"scripts"))
os.chdir(ROOT)
from llm_verifier import pivot_tournament as ppt
from llm_verifier.fine_grained_reward import cache_key
from llm_verifier.benchmarks import BENCHMARKS
import run as R

cfg=BENCHMARKS["terminal_bench_2.1"]; crit=list(cfg.criteria); REPS=2
tasks,_=R.LOADERS[cfg.loader](cfg.data,ROOT); tasks={t:v[:3] for t,v in tasks.items()}
allp,swing=R.classify(tasks)
rng=random.Random(cfg.seed); rings={t:ppt.ring_cycle(len(tasks[t]),rng) for t in swing}
jitter=random.Random(1234)
def score():
    v=round(jitter.uniform(0.05,0.95),4)
    return v if v!=0.5 else 0.61          # never an exact 0.5
cache={}
def put(task,a,b):
    # NEVER overwrite: score_directed_pairs only scores keys NOT already cached
    # (fine_grained_reward.py:827). Overwriting re-randomizes the ring/pivot
    # duplicate AFTER pivots were chosen, leaving a self-inconsistent cache.
    for cid in crit:
        for rep in range(REPS):
            k=cache_key(cid,task,a,b,rep)
            if k not in cache:
                cache[k]={"score_A":score(),"score_B":score()}
for t in swing:
    n=len(tasks[t])
    for (a,b) in rings[t]: put(t,a,b)
    w,c=[0.0]*n,[0]*n
    for (x,y) in rings[t]:
        ra=sum(cache[cache_key(cid,t,x,y,r)]["score_A"] for cid in crit for r in range(REPS))/(len(crit)*REPS)
        rb=sum(cache[cache_key(cid,t,x,y,r)]["score_B"] for cid in crit for r in range(REPS))/(len(crit)*REPS)
        p=ppt.bradley_terry(ra,rb); w[x]+=p; c[x]+=1; w[y]+=1-p; c[y]+=1
    for (a,b) in ppt.pivot_round_pairs(n,ppt.select_pivots(w,c,1)): put(t,a,b)
S=os.environ.get("LLMV_OUT") or os.path.dirname(os.path.abspath(__file__))
json.dump(cache,open(os.path.join(S,"synthetic_good_bo3.json"),"w"))
print("synthetic entries:",len(cache),"swing:",len(swing))
