#!/usr/bin/env python3
"""Sustained-load soak test with thermal telemetry.

WHY. Every performance number this project holds comes from runs of a few minutes. The
question a soak answers is different and more useful: **does throughput HOLD under
sustained load, or does it decay?** Decay is what thermal throttling looks like from the
outside, and a short benchmark cannot see it.

⚠️ RUN ONLY WITH A THERMAL GUARD ACTIVE ON EVERY NODE. spark-1 has R26; spark-2 got one on
2026-08-16 specifically so this could run. This script REFUSES to start otherwise — a soak
on an unguarded, uncapped node is the one experiment that can damage hardware rather than
just produce a bad number.

WHAT IT RECORDS, per 30s window:
  * completed requests, tokens, aggregate tok/s
  * latency p50/p95, and errors BY TYPE (an error rate that climbs is the real signal)
  * temperature and achieved clock on BOTH nodes
so the write-up can answer "did it get slower, and was it heat?" rather than guessing.

⚠️ THROUGHPUT DECAY IS NOT AUTOMATICALLY THERMAL. It can also be KV-cache pressure, memory
fragmentation, or a scheduler effect. This records temperature and clock ALONGSIDE
throughput so the two can be correlated instead of assumed — the correlation is the
evidence, not the decay on its own.
"""
import argparse, json, statistics, subprocess, sys, threading, time, urllib.request

STOP = threading.Event()
LOCK = threading.Lock()
EVENTS = []          # (t_done, secs, tokens, error_or_None)
TELEM = []           # per-sample node telemetry

PROMPT = ("Write a Python function that merges overlapping intervals, with docstring, "
          "type hints, and three edge cases handled explicitly.")


def node_probe(nodes, interval=10):
    while not STOP.is_set():
        row = {"t": time.time()}
        for n in nodes:
            try:
                cmd = "nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader,nounits"
                full = cmd if n == "local" else ["ssh", "-o", "BatchMode=yes", n, cmd]
                r = (subprocess.run(full, shell=True, capture_output=True, text=True, timeout=20)
                     if n == "local" else
                     subprocess.run(full, capture_output=True, text=True, timeout=20))
                t, sm = [x.strip() for x in r.stdout.strip().split(",")[:2]]
                row[n] = {"temp": float(t), "sm": float(sm)}
            except Exception:
                row[n] = {"temp": None, "sm": None}
        with LOCK:
            TELEM.append(row)
        STOP.wait(interval)


def worker(base, model, max_tokens):
    while not STOP.is_set():
        body = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": max_tokens, "temperature": 0.0, "stream": False}
        req = urllib.request.Request(base + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            d = json.load(urllib.request.urlopen(req, timeout=300))
            tok = d.get("usage", {}).get("completion_tokens", 0)
            with LOCK:
                EVENTS.append((time.time(), time.time() - t0, tok, None))
        except Exception as e:
            with LOCK:
                EVENTS.append((time.time(), time.time() - t0, 0, type(e).__name__))


def guards_ok(nodes):
    """Refuse to soak an unguarded node. This is the precondition, not a nicety."""
    ok = True
    for n in nodes:
        if n == "spark-1":
            r = subprocess.run(["ssh", n, "systemctl is-active dspark-thermal-guard"],
                               capture_output=True, text=True, timeout=25)
            alive = r.stdout.strip() == "active"
            print("  %-8s R26 thermal guard: %s" % (n, r.stdout.strip()))
        else:
            r = subprocess.run(["ssh", n, "$HOME/dspark-thermal-guard-spark2 --status"],
                               capture_output=True, text=True, timeout=25)
            alive = r.returncode == 0
            print("  %-8s guard: %s" % (n, r.stdout.strip()[:80]))
        ok &= alive
    return ok


def window_stats(t0, t1):
    with LOCK:
        ev = [e for e in EVENTS if t0 <= e[0] < t1]
        tl = [r for r in TELEM if t0 <= r["t"] < t1]
    okev = [e for e in ev if e[3] is None]
    lat = sorted(e[1] for e in okev)
    tok = sum(e[2] for e in okev)
    errs = {}
    for e in ev:
        if e[3]:
            errs[e[3]] = errs.get(e[3], 0) + 1
    out = {"completed": len(okev), "errors": errs, "tokens": tok,
           "tok_s": tok / (t1 - t0) if t1 > t0 else 0,
           "p50": lat[len(lat)//2] if lat else None,
           "p95": lat[int(len(lat)*0.95)] if len(lat) > 1 else (lat[0] if lat else None)}
    for node in ("spark-1", "spark-2"):
        temps = [r[node]["temp"] for r in tl if r.get(node, {}).get("temp") is not None]
        clks = [r[node]["sm"] for r in tl if r.get(node, {}).get("sm") is not None]
        out[node] = {"temp_max": max(temps) if temps else None,
                     "clk_median": round(statistics.median(clks)) if clks else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://spark-1:8888")
    ap.add_argument("--model", default="deepseek-v4-flash-dspark")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--window", type=float, default=30)
    ap.add_argument("--nodes", default="spark-1,spark-2")
    ap.add_argument("--label", default="soak")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force-no-guard", action="store_true",
                    help="run WITHOUT a thermal guard. Do not use on spark-2.")
    a = ap.parse_args()
    nodes = [n for n in a.nodes.split(",") if n]

    print("=== PRECONDITION: thermal guards ===")
    if not guards_ok(nodes):
        if not a.force_no_guard:
            print("\nABORT: a node has no live thermal guard. A sustained soak on an "
                  "unguarded, uncapped GPU is the one test that can damage hardware "
                  "rather than merely produce a bad number. Start the guard, or pass "
                  "--force-no-guard if you accept that.")
            return 2
        print("\n⚠️  PROCEEDING WITHOUT A GUARD because --force-no-guard was given.")

    print("\n=== SOAK: c%d for %.0f min, %d max_tokens ===" % (a.concurrency, a.minutes, a.max_tokens))
    print("  PROMPT: fixed interval-merge code prompt")
    probe = threading.Thread(target=node_probe, args=(nodes,), daemon=True)
    probe.start()
    ths = [threading.Thread(target=worker, args=(a.base, a.model, a.max_tokens), daemon=True)
           for _ in range(a.concurrency)]
    t_start = time.time()
    for t in ths:
        t.start()

    windows = []
    try:
        while time.time() - t_start < a.minutes * 60:
            w0 = time.time()
            time.sleep(min(a.window, max(0, a.minutes * 60 - (time.time() - t_start))))
            w = window_stats(w0, time.time())
            w["t_rel"] = round(w0 - t_start)
            windows.append(w)
            print("  t+%4ds  %3d done  %6.1f tok/s  p50 %5.2fs p95 %5.2fs  "
                  "s1 %sC/%sMHz  s2 %sC/%sMHz  errors=%s"
                  % (w["t_rel"], w["completed"], w["tok_s"],
                     w["p50"] or 0, w["p95"] or 0,
                     w["spark-1"]["temp_max"], w["spark-1"]["clk_median"],
                     w["spark-2"]["temp_max"], w["spark-2"]["clk_median"],
                     w["errors"] or "none"), flush=True)
    except KeyboardInterrupt:
        print("\n  interrupted — stopping cleanly")
    STOP.set()
    time.sleep(2)

    # ⚠️ EXCLUDE THE FIRST WINDOW: it contains ramp-up (workers starting, engine warming),
    # so comparing it to the last window reports a warm-up artifact as a throughput gain.
    # Measured 2026-08-16: first window 106.6 vs steady state 133-160 tok/s -> a spurious
    # "+25.4%". Steady state is what a soak is for.
    good = [w for w in windows if w["completed"] > 0][1:]
    if len(good) >= 2:
        first, last = good[0]["tok_s"], good[-1]["tok_s"]
        drift = (last - first) / first * 100 if first else 0
        peak1 = max((w["spark-1"]["temp_max"] or 0) for w in good)
        peak2 = max((w["spark-2"]["temp_max"] or 0) for w in good)
        print("\n" + "=" * 66)
        print("  throughput  first window %.1f -> last %.1f tok/s  (%+.1f%%)" % (first, last, drift))
        print("  peak temps  spark-1 %.0fC   spark-2 %.0fC" % (peak1, peak2))
        toterr = sum(sum(w["errors"].values()) for w in windows)
        print("  errors      %d total" % toterr)
        if drift < -10:
            print("  ⚠️  THROUGHPUT DECAYED >10%. Correlate with the temp/clock columns above "
                  "before calling it thermal — KV pressure and fragmentation look the same "
                  "from the outside.")
        else:
            print("  throughput HELD (no decay beyond 10%).")

    if a.out:
        json.dump({"label": a.label, "concurrency": a.concurrency, "minutes": a.minutes,
                   "max_tokens": a.max_tokens, "prompt": "fixed interval-merge code prompt",
                   "windows": windows, "telemetry": TELEM[-600:]}, open(a.out, "w"), indent=1)
        print("  saved -> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
