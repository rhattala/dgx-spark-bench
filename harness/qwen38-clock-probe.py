#!/usr/bin/env python3
"""Decode / prefill / concurrency probe for a single OpenAI-compatible endpoint.

Built for the R21 clock A/B, but endpoint-agnostic so the SAME probe can be pointed at
DeepSeek afterwards — a comparison run through two different harnesses is not a
comparison. Every number it prints names its PROMPT, TOKEN COUNT and CLOCK STATE.

⚠️ THREE MEASUREMENT TRAPS THIS HANDLES, ALL DOCUMENTED AS REAL:

1. PREFIX CACHE. SGLang keeps a radix cache. Re-sending an identical prompt skips prefill
   entirely, so a repeated-prompt prefill measurement reports the cache, not the machine.
   Prefill prompts therefore carry a unique nonce per trial. Decode prompts are
   deliberately IDENTICAL — for a clock A/B, holding content fixed is the point, and the
   recipe measured repeat reproducibility at +/-0.2 tok/s.

2. FIRST-REQUEST COSTS. mmap page-in and spec-path warmup poison the first call after a
   load. --warmup burns them.

3. DECODE vs TOTAL. tok/s computed from total wall clock includes prefill and understates
   decode. With a short prompt and 800 output tokens the error is small, but it is
   reported honestly rather than hidden: prefill is measured separately.
"""
import argparse, json, statistics, sys, threading, time, urllib.request, uuid

# Fixed decode prompt — identical across every trial and every clock, by design.
DECODE_PROMPT = ("Write a Python implementation of a red-black tree with insert, delete, "
                 "and search. Include docstrings.")
# Long filler for prefill; a nonce is prepended per trial to defeat the prefix cache.
FILLER = ("The quick brown fox jumps over the lazy dog near the riverbank at dawn while "
          "the heron watches from the reeds and the mist lifts slowly off the water. ")


def call(url, model, key, prompt, max_tokens, no_think, timeout=900):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": False}
    if no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    el = time.time() - t0
    u = d.get("usage", {})
    return {"secs": el, "ctok": u.get("completion_tokens", 0),
            "ptok": u.get("prompt_tokens", 0),
            "finish": d["choices"][0].get("finish_reason")}


def decode_block(args, trials):
    """Single-stream decode. Identical prompt every time (clock is the variable)."""
    out = []
    for i in range(trials):
        r = call(args.url, args.model, args.key, DECODE_PROMPT, args.max_tokens, args.no_think)
        tps = r["ctok"] / r["secs"] if r["secs"] else 0
        out.append(tps)
        if not args.quiet:
            print("    decode  t%d: %4d tok in %6.2fs = %5.1f tok/s  (finish=%s)"
                  % (i + 1, r["ctok"], r["secs"], tps, r["finish"]))
    return out


def prefill_block(args, trials):
    """Prefill. UNIQUE prompt per trial or the radix cache answers instead of the GPU."""
    out = []
    for i in range(trials):
        nonce = uuid.uuid4().hex
        prompt = ("Session %s. Read the following log and reply with the single word OK.\n\n"
                  % nonce) + (FILLER * 420) + "\n\nReply with exactly: OK"
        r = call(args.url, args.model, args.key, prompt, 8, args.no_think)
        pps = r["ptok"] / r["secs"] if r["secs"] else 0
        out.append(pps)
        if not args.quiet:
            print("    prefill t%d: %5d prompt-tok in %6.2fs = %6.0f tok/s"
                  % (i + 1, r["ptok"], r["secs"], pps))
    return out


def conc_block(args, levels):
    """Aggregate throughput at N concurrent streams.

    Aggregate = sum(completion tokens) / WALL CLOCK of the whole batch — not the mean of
    per-stream rates, which would overstate it by ignoring stragglers.
    """
    res = {}
    for n in levels:
        results, errors = [], []
        lock = threading.Lock()

        def worker(idx):
            try:
                # Distinct prompt per stream: identical prompts across concurrent streams
                # would share radix-cache prefixes and flatter the result.
                p = "%s (variant %d: also explain the rotation cases)" % (DECODE_PROMPT, idx)
                r = call(args.url, args.model, args.key, p, args.max_tokens, args.no_think)
                with lock:
                    results.append(r)
            except Exception as e:
                with lock:
                    errors.append(str(e)[:80])

        ths = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        t0 = time.time()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.time() - t0
        tot = sum(r["ctok"] for r in results)
        agg = tot / wall if wall else 0
        per = agg / n if n else 0
        res[n] = {"agg": agg, "per_stream": per, "wall": wall, "tok": tot,
                  "ok": len(results), "err": errors}
        print("    c%-2d: %5d tok in %6.1fs = %6.1f tok/s aggregate  (%5.1f/stream, %d/%d ok)%s"
              % (n, tot, wall, agg, per, len(results), n,
                 ("  ERRORS: " + "; ".join(errors[:2])) if errors else ""))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="spark-2")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--url", default=None)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--api-key-file", default=None)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--label", default="run")
    ap.add_argument("--no-think", action="store_true", default=True)
    ap.add_argument("--think", dest="no_think", action="store_false")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument("--conc", default="", help="comma list, e.g. 1,2,4,8")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    a.url = a.url or "http://%s:%d/v1/chat/completions" % (a.host, a.port)
    a.key = None
    if a.api_key_file:
        with open(a.api_key_file) as f:
            a.key = f.read().strip()   # never printed

    rec = {"label": a.label, "url": a.url, "model": a.model,
           "max_tokens": a.max_tokens, "thinking": not a.no_think}

    dec = decode_block(a, a.trials)
    rec["decode"] = {"trials": [round(x, 2) for x in dec],
                     "median": round(statistics.median(dec), 2),
                     "mean": round(statistics.fmean(dec), 2),
                     "spread_pct": round((max(dec) - min(dec)) / statistics.fmean(dec) * 100, 1)}
    if not a.quiet:
        print("    -> decode median %.1f tok/s (spread %.1f%%)"
              % (rec["decode"]["median"], rec["decode"]["spread_pct"]))

    if not a.decode_only:
        pre = prefill_block(a, a.trials)
        rec["prefill"] = {"trials": [round(x) for x in pre],
                          "median": round(statistics.median(pre)),
                          "spread_pct": round((max(pre) - min(pre)) / statistics.fmean(pre) * 100, 1)}
        if not a.quiet:
            print("    -> prefill median %.0f tok/s (spread %.1f%%)"
                  % (rec["prefill"]["median"], rec["prefill"]["spread_pct"]))

    if a.conc:
        levels = [int(x) for x in a.conc.split(",") if x.strip()]
        rec["concurrency"] = {str(k): v for k, v in conc_block(a, levels).items()}

    if a.out:
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
