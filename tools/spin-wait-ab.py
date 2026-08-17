#!/usr/bin/env python3
"""Before/after harness for the vLLM spin-wait change (`busy_loop_s`).

WHY A DEDICATED HARNESS. The change being tested claims "no performance loss" while cutting
CPU burn. Both halves of that need measuring, at the same time, with the same prompts, on
the same clock state — otherwise the comparison is between two different experiments.

⚠️ RECORDS CLOCK STATE IN THE OUTPUT and refuses to pretend it does not matter. A capped
and an uncapped run are not comparable, and this deployment has already had a thermal event
silently re-cap it. If the two runs disagree on clock state, the comparison is void and the
report says so.

⚠️ MEASURES THE COST SIDE TOO. CPU% comes from /proc utime+stime deltas, NOT `ps -eo pcpu`
— that reports a LIFETIME AVERAGE and reads the same under every condition. That mistake
was made here first and nearly became a conclusion.

Run once before the change, once after:

    python3 tools/spin-wait-ab.py --label before --out results/spin-before.json
    python3 tools/spin-wait-ab.py --label after  --out results/spin-after.json
    python3 tools/spin-wait-ab.py --compare results/spin-before.json results/spin-after.json
"""
import argparse, json, statistics as st, subprocess, sys, threading, time, urllib.request

DECODE_PROMPT = ("Write a detailed technical explanation of how a B-tree index works in a "
                 "relational database, including node structure, splits, and lookup cost.")
PREFILL_FILLER = ("The quarterly maintenance review covers throughput, seasonal demand and "
                  "component availability across the distribution network. ")

CPU_DELTA = r'''
import os,sys,time
W=float(sys.argv[1]); HZ=os.sysconf("SC_CLK_TCK")
def pids():
    o=[]
    for p in filter(str.isdigit, os.listdir("/proc")):
        try: c=open("/proc/%s/cmdline"%p,"rb").read().decode("utf8","ignore")
        except Exception: continue
        if "vllm" in c.lower(): o.append(p)
    return o
def t(p):
    try:
        f=open("/proc/%s/stat"%p).read(); x=f[f.rindex(")")+2:].split()
        return int(x[11])+int(x[12])
    except Exception: return None
P=pids(); a={p:t(p) for p in P}; s=time.time(); time.sleep(W); b={p:t(p) for p in P}; el=time.time()-s
tot=sum((b[p]-a[p])/HZ/el*100 for p in P if a.get(p) is not None and b.get(p) is not None)
z=[]
for d in os.listdir("/sys/class/thermal"):
    if d.startswith("thermal_zone"):
        try: z.append(int(open("/sys/class/thermal/%s/temp"%d).read())//1000)
        except Exception: pass
print(json.dumps({"cpu_pct":round(tot,1),"zone_max":max(z) if z else None})) if False else print(
    '{"cpu_pct": %.1f, "zone_max": %s}' % (tot, max(z) if z else "null"))
'''


def ssh(node, script, arg):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", node,
                        "python3 - %s" % arg], input=script, capture_output=True,
                       text=True, timeout=60)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {"cpu_pct": None, "zone_max": None, "error": (r.stderr or "")[:90]}


def clocks(node):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", node,
                        "nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=25)
    try:
        sm, tc = [x.strip() for x in r.stdout.strip().split(",")[:2]]
        return float(sm), float(tc)
    except Exception:
        return None, None


def inflight(base):
    """Requests the engine is already serving. Foreign traffic makes a timing trial
    meaningless, and this deployment has watchdog crons firing real generations every few
    minutes — one landed mid-trial and produced a 10.4 tok/s reading among 47s."""
    try:
        raw = urllib.request.urlopen(base + "/metrics", timeout=8).read().decode()
        for line in raw.splitlines():
            if line.startswith("vllm:num_requests_running"):
                return float(line.rsplit(" ", 1)[1])
    except Exception:
        pass
    return None


def clean_trial(base, model, prompt, max_tokens, tries=4):
    """Run a timing trial, DISCARDING any that overlapped foreign traffic.

    Silencing the watchdogs would also work and the tooling exists for it — but quieting
    monitoring to make a benchmark look tidy is the wrong trade, especially on a night that
    has already had a thermal event. Detect contention instead and refuse the sample.
    """
    for _ in range(tries):
        if (inflight(base) or 0) > 0:
            time.sleep(8)
            continue
        r = gen(base, model, prompt, max_tokens)
        if (inflight(base) or 0) > 0:      # someone joined while we ran
            time.sleep(8)
            continue
        return r
    return None


def gen(base, model, prompt, max_tokens, temperature=0.0):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "stream": False}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=600))
    el = time.time() - t0
    u = d.get("usage", {})
    return {"secs": el, "completion_tokens": u.get("completion_tokens", 0),
            "prompt_tokens": u.get("prompt_tokens", 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://spark-1:8888")
    ap.add_argument("--model", default="deepseek-v4-flash-dspark")
    ap.add_argument("--nodes", default="spark-1,spark-2")
    ap.add_argument("--label", default="run")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--decode-tokens", type=int, default=400)
    ap.add_argument("--prefill-chars", type=int, default=56000, help="~14k tokens")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    a = ap.parse_args()

    if a.compare:
        return compare(a.compare[0], a.compare[1])

    nodes = [n.strip() for n in a.nodes.split(",") if n.strip()]
    base = a.base.rstrip("/")
    res = {"label": a.label, "model": a.model, "trials": a.trials,
           "decode_tokens": a.decode_tokens}

    print("=== %s ===" % a.label)
    print("  warming (cold engine is ~30%% slower and it returns after idle)")
    for _ in range(2):
        gen(base, a.model, DECODE_PROMPT, 200)

    # --- decode, single stream ---
    dec, skipped = [], 0
    for i in range(a.trials):
        r = clean_trial(base, a.model, DECODE_PROMPT, a.decode_tokens)
        if r is None:
            skipped += 1
            print("  decode trial %d: SKIPPED — engine busy with other traffic" % (i + 1))
            continue
        tps = r["completion_tokens"] / r["secs"] if r["secs"] else 0
        dec.append(tps)
        print("  decode trial %d: %5.1f tok/s (%d tokens)" % (i + 1, tps, r["completion_tokens"]))
    res["decode_skipped"] = skipped
    res["decode_tok_s"] = {"values": [round(x, 2) for x in dec],
                           "mean": round(st.mean(dec), 2) if dec else None,
                           "median": round(st.median(dec), 2) if dec else None,
                           "sd": round(st.stdev(dec), 2) if len(dec) > 1 else 0.0}
    if len(dec) > 1 and st.stdev(dec) > 0.10 * st.mean(dec):
        print("  ⚠️  sd is >10%% of the mean — this baseline cannot resolve a small effect.")

    # --- prefill ---
    big = PREFILL_FILLER * (a.prefill_chars // len(PREFILL_FILLER))
    pre = []
    print("  (prefill trial 1 is discarded: first touch of a new prompt is uncached)")
    for i in range(4):
        r = clean_trial(base, a.model, big + "\n\nSummarise the above in one sentence.", 24)
        if r is None:
            print("  prefill trial %d: SKIPPED — busy" % (i + 1)); continue
        if i == 0:
            print("  prefill trial 1: %6.0f prompt tok/s  [discarded, cold]"
                  % (r["prompt_tokens"] / r["secs"] if r["secs"] else 0)); continue
        pts = r["prompt_tokens"] / r["secs"] if r["secs"] else 0
        pre.append(pts)
        print("  prefill trial %d: %6.0f prompt tok/s (%d prompt tokens)" % (i + 1, pts, r["prompt_tokens"]))
    res["prefill_tok_s"] = {"values": [round(x, 1) for x in pre],
                            "mean": round(st.mean(pre), 1) if pre else None,
                            "note": "trial 1 discarded as cold"}

    # --- CPU + thermals DURING load, and clock state ---
    print("  sampling CPU/thermals under load...")
    hold = {"stop": False}

    def loader():
        while not hold["stop"]:
            try:
                gen(base, a.model, DECODE_PROMPT, 300)
            except Exception:
                return

    t = threading.Thread(target=loader, daemon=True)
    t.start()
    time.sleep(6)
    cpu = {}
    for n in nodes:
        cpu[n] = ssh(n, CPU_DELTA, "5")
        sm, tc = clocks(n)
        cpu[n]["clock_mhz"] = sm
        cpu[n]["gpu_temp_c"] = tc
        print("    %-8s cpu=%s%%  zone_max=%sC  clock=%s MHz  gpu=%sC"
              % (n, cpu[n].get("cpu_pct"), cpu[n].get("zone_max"), sm, tc))
    hold["stop"] = True
    time.sleep(2)
    res["under_load"] = cpu

    # --- idle CPU, for the poll-path contrast ---
    print("  waiting for quiet, then sampling idle CPU...")
    time.sleep(25)
    res["idle"] = {n: ssh(n, CPU_DELTA, "4") for n in nodes}
    for n in nodes:
        print("    %-8s idle cpu=%s%%" % (n, res["idle"][n].get("cpu_pct")))

    res["clock_state"] = {n: cpu[n].get("clock_mhz") for n in nodes}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print("  saved -> %s" % a.out)
    return 0


def compare(p_before, p_after):
    b, a = json.load(open(p_before)), json.load(open(p_after))
    print("=== %s  ->  %s ===\n" % (b["label"], a["label"]))

    # ⚠️ clock state gates the whole comparison
    cb, ca = b.get("clock_state", {}), a.get("clock_state", {})
    same = all(cb.get(k) is not None and ca.get(k) is not None
               and abs(cb[k] - ca[k]) <= 200 for k in cb) and set(cb) == set(ca)
    print("  clock state before: %s" % cb)
    print("  clock state after : %s" % ca)
    if not same:
        print("\n  *** COMPARISON VOID: the two runs were at different clock states. ***")
        print("  A capped and an uncapped run are not comparable. Re-run both at the same")
        print("  clock policy before drawing any conclusion.\n")

    def row(name, bv, av, unit="", better_is="higher"):
        if bv in (None, 0) or av is None:
            print("  %-26s %10s -> %10s" % (name, bv, av)); return
        d = (av - bv) / bv * 100
        verdict = ""
        if abs(d) < 2:
            verdict = "(within noise)"
        elif (d > 0) == (better_is == "higher"):
            verdict = "BETTER"
        else:
            verdict = "*** WORSE ***"
        print("  %-26s %10.1f%s -> %10.1f%s  %+6.1f%%  %s" % (name, bv, unit, av, unit, d, verdict))

    row("decode tok/s", b["decode_tok_s"]["mean"], a["decode_tok_s"]["mean"])
    row("prefill tok/s", b["prefill_tok_s"]["mean"], a["prefill_tok_s"]["mean"])
    for n in b.get("under_load", {}):
        row("CPU%% under load %s" % n, b["under_load"][n].get("cpu_pct"),
            a["under_load"].get(n, {}).get("cpu_pct"), "%", better_is="lower")
        row("zone max %s" % n, b["under_load"][n].get("zone_max"),
            a["under_load"].get(n, {}).get("zone_max"), "C", better_is="lower")
    print("\n  decode sd: before %.2f, after %.2f — a change smaller than these is not a result."
          % (b["decode_tok_s"].get("sd", 0), a["decode_tok_s"].get("sd", 0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
