#!/usr/bin/env python3
"""Measure what best-of-N verification actually costs, end to end, with provenance.

Every throughput number here names its PROMPT, TOKEN COUNT and CLOCK STATE, per the
repo rule — a number missing those is not comparable to anything.

Writes one JSON with per-request usage so the published figures can be re-derived
rather than taken on trust.
"""
import argparse, json, os, statistics as st, subprocess, sys, threading, time, urllib.request

PROMPT = ("Write a Python function that merges two sorted lists into one sorted list. "
          "Explain the time complexity in one sentence.")


def clocks(nodes=("spark-1", "spark-2")):
    out = {}
    for n in nodes:
        try:
            r = subprocess.run(["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", n,
                                "nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu "
                                "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=20)
            out[n] = r.stdout.strip() or "UNVERIFIED"
        except Exception:
            out[n] = "UNVERIFIED"
    return out


def gen(base, model, prompt, max_tokens, temperature=0.7):
    t0 = time.time()
    req = urllib.request.Request(base + "/chat/completions", method="POST",
        data=json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": max_tokens, "temperature": temperature,
                         "stream": False}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    except Exception as e:
        return {"ok": False, "error": type(e).__name__, "seconds": time.time() - t0}
    u = d.get("usage") or {}
    return {"ok": True, "seconds": round(time.time() - t0, 3),
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "text_chars": len(d["choices"][0]["message"]["content"] or "")}


def burst(base, model, n, max_tokens):
    """n requests fired simultaneously. Returns rows + wall time."""
    rows, lock = [], threading.Lock()
    def w():
        r = gen(base, model, PROMPT, max_tokens)
        with lock: rows.append(r)
    t0 = time.time()
    th = [threading.Thread(target=w) for _ in range(n)]
    [x.start() for x in th]; [x.join() for x in th]
    return rows, round(time.time() - t0, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("DSPARK_BASE", "http://spark-1:8888/v1"))
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--levels", default="1,3,6")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    base = a.base.rstrip("/")
    model = json.loads(urllib.request.urlopen(base + "/models", timeout=15).read())["data"][0]["id"]

    pre = clocks()
    print(f"  model={model}  max_tokens={a.max_tokens}  temperature=0.7")
    print(f"  prompt ({len(PROMPT)} chars): {PROMPT[:70]}...")
    for n, c in pre.items(): print(f"  {n} clocks(sm,max,tempC) pre-run: {c}")
    print()

    gen(base, model, PROMPT, a.max_tokens)          # warm
    results = []
    for lvl in [int(x) for x in a.levels.split(",")]:
        rows, wall = burst(base, model, lvl, a.max_tokens)
        ok = [r for r in rows if r["ok"]]
        if not ok:
            print(f"  c={lvl}: ALL FAILED"); continue
        toks = sum(r["completion_tokens"] or 0 for r in ok)
        lat = st.mean(r["seconds"] for r in ok)
        agg = round(toks / wall, 1)
        results.append({"concurrency": lvl, "wall_s": wall, "mean_latency_s": round(lat, 2),
                        "completion_tokens": toks, "aggregate_tok_s": agg,
                        "clocks_after": clocks(), "requests": ok})
        print(f"  c={lvl}: {len(ok)} req in {wall:5.1f}s | mean latency {lat:5.1f}s | "
              f"{toks:4d} tok | aggregate {agg:6.1f} tok/s")

    out = {"model": model, "prompt": PROMPT, "prompt_chars": len(PROMPT),
           "max_tokens": a.max_tokens, "temperature": 0.7,
           "clocks_pre_run": pre, "levels": results,
           "note": "short-prompt concurrency curve; every row carries per-request usage"}
    path = a.out or f"results/{time.strftime('%Y-%m-%d')}/verify-cost.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(out, f, indent=2)
    print(f"\n  saved -> {path}")
    return 0 if results else 2


if __name__ == "__main__":
    sys.exit(main())
