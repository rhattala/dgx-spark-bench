#!/usr/bin/env python3
"""Prefill throughput vs prompt depth, single request, cold prefix.

WHY: the llm-as-a-verifier workload is prefill-dominated (46k-275k-token prompts),
so decode tok/s says nothing useful about how long it will take. This measures the
quantity that actually governs the run.

Every number is labelled with prompt, token count and clock state, per the standing
rule that a number without those three is not comparable to anything.

Each trial uses a UNIQUE random preamble so it cannot hit the prefix cache; the
reported figure is therefore cold-prefill, the worst case and the one that sizes the run.
"""
import argparse, json, os, random, string, sys, time, urllib.request

def post(base, path, payload, timeout=1800):
    req = urllib.request.Request(
        base + path, method="POST",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def get(base, path, timeout=15):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return r.read().decode()

def prompt_counter(base):
    """Cumulative prompt tokens served, by NAME (positional parsing mislabels).

    The before/after gauge cannot see a request that starts AND finishes inside
    a trial -- e.g. the 10-minute keepalive, whose 1-token request completes long
    before the post-check. A counter delta catches any intruder regardless of
    timing.
    """
    try:
        body = get(base, "/metrics")
    except Exception:
        return None
    for line in body.splitlines():
        if line.startswith("vllm:prompt_tokens_total{"):
            return float(line.rsplit(" ", 1)[1])
    return None


def inflight(base):
    """Requests currently running+waiting, or None if unreadable.

    Returns None rather than 0 on failure: a check that cannot verify must say
    UNVERIFIED, never 'idle'.
    """
    try:
        body = get(base, "/metrics")
    except Exception:
        return None
    tot = 0.0
    seen = False
    for line in body.splitlines():
        for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
            if line.startswith(name + "{"):
                tot += float(line.rsplit(" ", 1)[1]); seen = True
    return tot if seen else None

def filler(n_words, rng):
    """Unique pseudo-text: defeats the prefix cache without being degenerate."""
    return " ".join(
        "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 9)))
        for _ in range(n_words))

def trial(base, model, target_tokens, rng):
    # ~0.75 words/token for this kind of text; corrected by the server's own count.
    body = filler(int(target_tokens * 0.75), rng)
    prompt = (f"Below is a log excerpt. After reading it, reply with exactly the "
              f"word DONE and nothing else.\n\n{body}\n\nReply with exactly: DONE")
    t0 = time.time()
    r = post(base, "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4, "temperature": 0.0, "stream": False})
    dt = time.time() - t0
    u = r.get("usage") or {}
    pt = u.get("prompt_tokens")
    ct = u.get("completion_tokens")
    if not pt:
        return None
    if dt <= 0:
        return None          # cannot form a rate; drop the row rather than format None
    return {"prompt_tokens": pt, "completion_tokens": ct,
            "seconds": round(dt, 2),
            "prefill_tok_s": round(pt / dt, 1)}

def clock_state(nodes):
    out = {}
    for n in nodes:
        # `timeout 15` on the LOCAL side: ConnectTimeout bounds only the connect,
        # so a node that wedges AFTER accept would hang os.popen().read() forever
        # and freeze the harness before the first trial.
        rc = os.popen(
            f"timeout 15 ssh -o ConnectTimeout=8 -o BatchMode=yes {n} "
            f"'nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu "
            f"--format=csv,noheader' 2>/dev/null").read().strip()
        out[n] = rc or "UNVERIFIED"
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get(
        "DSPARK_BASE", "http://spark-1:8888"))
    ap.add_argument("--depths", default="25000,50000,100000,200000")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    base = a.base.rstrip("/")
    model = json.loads(get(base, "/v1/models"))["data"][0]["id"]
    rng = random.Random(a.seed)

    before = inflight(base)
    if before is None:
        print("UNVERIFIED: cannot read /metrics — refusing to measure", file=sys.stderr)
        return 2
    if before > 0:
        print(f"UNVERIFIED: {before:.0f} request(s) in flight — refusing to measure",
              file=sys.stderr)
        return 2

    clocks = clock_state(["spark-1", "spark-2"])
    print(f"model={model}")
    for n, c in clocks.items():
        print(f"  {n} clocks(sm,max,tempC) = {c}")
    print("  prompt=unique-random-filler (cold prefix, cache-defeating)  temperature=0.0"
          "  max_tokens=4  stream=false")

    rows = []
    for depth in [int(x) for x in a.depths.split(",")]:
        for i in range(a.repeats):
            # NB: `inflight() or 1` is WRONG — a legitimate 0.0 is falsy and
            # becomes 1, so every trial is skipped as "contended". None (could
            # not read) and 0.0 (genuinely idle) must be distinguished
            # explicitly: unreadable counts as contended, idle does not.
            n = inflight(base)
            if n is None or n > 0:
                print(f"  depth={depth} rep={i+1}: SKIPPED — "
                      f"{'metrics unreadable' if n is None else f'{n:.0f} in flight'}")
                continue
            r = trial(base, model, depth, rng)
            if not r:
                print(f"  depth={depth} rep={i+1}: FAILED — no usage returned")
                continue
            n = inflight(base)
            if n is None or n > 0:
                print(f"  depth={depth} rep={i+1}: DISCARDED — contended during trial")
                continue
            r["target_depth"] = depth
            rows.append(r)
            print(f"  depth~{depth:>7,} -> {r['prompt_tokens']:>7,} prompt tok  "
                  f"{r['seconds']:>7.2f} s  {r['prefill_tok_s']:>9,.1f} prompt tok/s")

    # Per-depth completeness. prefill tok/s FALLS with depth, so a run that
    # silently lost only the deepest points yields an OPTIMISTIC fit -- wrong in
    # the dangerous direction. Any requested depth with zero accepted rows is a
    # failed run, not a partial one.
    depths = [int(x) for x in a.depths.split(",")]
    per_depth = {d: sum(1 for r in rows if r["target_depth"] == d) for d in depths}
    empty = [d for d, n in per_depth.items() if n == 0]
    out = {"model": model, "clocks_pre_run_idle": clocks, "rows": rows,
           "per_depth_accepted": per_depth,
           "note": ("cold prefill, single request, unique filler prompt. "
                    "JUDGE ROWS BY prompt_tokens, NOT target_depth: gibberish "
                    "tokenizes ~2-4x heavier than the 0.75 words/token estimate, "
                    "so actual depth overshoots the target substantially. "
                    "Clocks above are PRE-RUN IDLE. Measures the PREFILL "
                    "component only -- decode must be added separately when "
                    "sizing a request timeout.")}
    if a.out:
        d = os.path.dirname(a.out)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  saved -> {a.out}")
    if empty:
        print(f"\n  UNVERIFIED: no accepted rows at requested depth(s) {empty} — "
              f"the deepest points set the duration model and the timeout, so a "
              f"shallow-only curve is optimistic in the dangerous direction.")
        return 2
    return 0 if rows else 2

if __name__ == "__main__":
    sys.exit(main())
