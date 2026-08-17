#!/usr/bin/env python3
"""Needle-in-a-Haystack: does long context actually RETRIEVE, or just accept tokens?

WHY. This cluster serves a 1M-token context window. Accepting a large prompt and *using*
it are different things, and nothing in this suite previously distinguished them — a model
that silently attends to only the first fraction of a long prompt looks identical from the
outside until an answer quietly comes back wrong.

METHOD. Plant a unique sentence (the "needle") at a controlled depth inside filler text,
then ask a question only that sentence can answer. Pass = the exact token appears in the
reply.

⚠️ THE NEEDLE MUST BE UNGUESSABLE. It is a random token generated per run, so the model
cannot answer from prior knowledge, and a pass cannot be luck. A needle like "the capital
of France" would be answerable without reading the haystack at all.

⚠️ PREFIX CACHING WOULD FAKE THIS. Every prompt carries a unique session id at the very
front, so a cached prefix cannot serve the answer.

⚠️ REPORTS UNVERIFIED, NOT FAILURE, when a probe errors or the context is refused — a
rejected request is not the same as a wrong answer, and conflating them would overstate a
capability limit as a retrieval failure.
"""
import argparse, difflib, json, random, string, sys, time, urllib.request

FILLER = ("The maintenance schedule for the regional distribution network is reviewed each "
          "quarter by the operations committee, which considers throughput, seasonal demand, "
          "and the availability of replacement components across the supply chain. ")


def make_needle(rng):
    return "SPARK-NIAH-" + "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(8))


def build(prompt_tokens, depth, needle, rng):
    """~4 chars/token is a rough but stable approximation for this filler."""
    target_chars = int(prompt_tokens * 4)
    reps = max(1, target_chars // len(FILLER))
    hay = FILLER * reps
    cut = int(len(hay) * depth)
    sentence = (" The authorisation code for the northern relay station is %s. " % needle)
    doc = hay[:cut] + sentence + hay[cut:]
    # unique session id FIRST so no prefix cache can answer this
    head = ("Session %s. Read the following document carefully.\n\n"
            % "".join(rng.choice(string.hexdigits.lower()) for _ in range(16)))
    tail = ("\n\nQuestion: What is the authorisation code for the northern relay station? "
            "Reply with only the code.")
    return head + doc + tail


NEEDLE_PREFIX = "SPARK-NIAH-"


def near_miss(needle, ans):
    """(similarity, is_near_miss). A near-miss is a TRANSCRIPTION slip on a needle that was
    FOUND — not a retrieval failure. Grading those as failures understates long-context
    capability and points debugging at the wrong thing.

    ⚠️ ANCHOR ON THE PREFIX INSIDE THE ANSWER. This used to compare against `ans[:len+6]`,
    anchored at position 0, so it only fired on a bare answer: the same one-character error
    behind a preamble ("The code is SPARK-NIAH-MKCBSUN") scored 0.59 and graded a flat FAIL.
    ⚠️ AND IT MUST BE ABLE TO FAIL: a DIFFERENT needle must not pass. The 0.85 threshold on
    an 8-char random suffix needs ~6 of 8 characters right, which luck does not supply.
    """
    if not ans:
        return 0.0, False
    starts = []
    # Every occurrence, not just the first: "Not SPARK-NIAH-XXXXXXXX; the code is
    # SPARK-NIAH-MKCBSUN" anchored on the decoy and scored 0.50.
    i = ans.find(NEEDLE_PREFIX)
    while i >= 0:
        starts.append(i)
        i = ans.find(NEEDLE_PREFIX, i + 1)
    # ⚠️ THE CORRUPTION CAN HIT THE ANCHOR ITSELF. The real 248k slip was
    # SPARK-NIAH- -> SPARK-NIA0-, so anchoring on the full prefix misses exactly the case
    # this function exists for. Fall back to a shorter stem, then to the head.
    if not starts:
        j = ans.find(NEEDLE_PREFIX[:8])
        starts = [j] if j >= 0 else [0]
    # Trailing prose inflates the window and drags similarity down ("...MKCBSUN, as stated
    # in the document" scored 0.818 and graded a flat FAIL), so try both window lengths.
    sim = max(difflib.SequenceMatcher(None, needle, ans[st:st + L]).ratio()
              for st in starts for L in (len(needle), len(needle) + 6))
    return sim, (needle not in ans) and sim >= 0.85


def probe(url, model, prompt, timeout):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64, "temperature": 0.0, "stream": False}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=timeout))
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, str(e)[:110]), "secs": time.time() - t0}
    el = time.time() - t0
    ch = d["choices"][0]
    # ⚠️ THE FIELD NAME IS NOT PORTABLE. This build emits the thinking channel as
    # `reasoning`; other vLLM/SGLang builds emit `reasoning_content`. Reading only one
    # name returns an empty string forever on the other, with no error — read BOTH.
    m = ch["message"]
    txt = " ".join(x for x in (m.get("content"), m.get("reasoning"),
                               m.get("reasoning_content")) if x)
    return {"answer": txt.strip()[:120], "secs": el,
            "prompt_tokens": d.get("usage", {}).get("prompt_tokens"),
            "finish": ch.get("finish_reason")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://spark-1:8888/v1/chat/completions")
    ap.add_argument("--model", default="deepseek-v4-flash-dspark")
    ap.add_argument("--sizes", default="8000,32000,124000,248000",
                    help="approximate prompt token counts")
    ap.add_argument("--depths", default="0.05,0.5,0.95",
                    help="fractional position of the needle")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--label", default="niah")
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    rng = random.Random()

    if a.self_test:
        # Prove the grader can fail: a needle that is NOT in the reply must not pass.
        print("=== SELF-TEST: the grader must be able to fail ===")
        n = make_needle(rng)
        cases = [("reply contains the needle", "the code is %s ok" % n, True),
                 ("reply omits the needle", "I could not find the code", False),
                 ("reply has a DIFFERENT needle", "the code is SPARK-NIAH-ZZZZZZZZ", False)]
        ok = True
        for label, reply, want in cases:
            got = n in reply
            good = got == want
            ok &= good
            print("  %-34s got=%-5s want=%-5s %s" % (label, got, want, "ok" if good else "*** BROKEN ***"))
        print("\n  near_miss() — a transcription slip must PASS, a wrong needle must FAIL")
        nm_cases = [
            ("one char dropped, bare",        "SPARK-NIAH-MKCBSUN",                    True),
            ("one char dropped, with preamble",
             "The authorisation code is SPARK-NIAH-MKCBSUN.",                          True),
            ("one char substituted",          "SPARK-NIA0-MKCBSUN6",                   True),
            ("A DIFFERENT NEEDLE must fail",  "SPARK-NIAH-ZZZZZZZZ",                   False),
            ("refusal must fail",             "I could not find the code",             False),
            ("empty answer must fail",        "",                                      False),
            ("prefix only, no suffix",        "SPARK-NIAH-",                           False),
            # --- second-pass review: all three of these graded FAIL before ---
            ("slip with TRAILING prose",
             "The code is SPARK-NIAH-MKCBSUN, as stated in the document.",             True),
            ("CORRUPTED PREFIX (the real 248k slip)",
             "The code is SPARK-NIA0-MKCBSUN6, per the relay station",                 True),
            ("decoy needle first, real one second",
             "Not SPARK-NIAH-XXXXXXXX; the code is SPARK-NIAH-MKCBSUN",                True),
            ("decoy only, no real needle",
             "Not SPARK-NIAH-XXXXXXXX; I could not find it",                           False),
        ]
        fixed = "SPARK-NIAH-MKCBSUN6"
        for label, reply, want in nm_cases:
            _, got = near_miss(fixed, reply)
            good = got == want
            ok &= good
            print("    %-34s got=%-5s want=%-5s %s" % (label, got, want,
                                                       "ok" if good else "*** BROKEN ***"))

        # and the builder must actually place the needle
        p = build(2000, 0.5, n, rng)
        placed = n in p
        ok &= placed
        print("  %-34s got=%-5s want=True  %s" % ("builder plants the needle", placed,
                                                  "ok" if placed else "*** BROKEN ***"))
        print("\n  %s\n" % ("HARNESS TRUSTWORTHY" if ok else "*** BROKEN ***"))
        return 0 if ok else 1

    sizes = [int(x) for x in a.sizes.split(",")]
    depths = [float(x) for x in a.depths.split(",")]
    rows = []
    print("=== NIAH: %d sizes x %d depths = %d probes ===" % (len(sizes), len(depths),
                                                              len(sizes) * len(depths)))
    print("  needle is RANDOM per probe (unguessable); each prompt carries a unique session id")
    print("  %-9s %-6s %-9s %-9s %s" % ("target", "depth", "actual", "secs", "result"))
    for size in sizes:
        for depth in depths:
            needle = make_needle(rng)
            prompt = build(size, depth, needle, rng)
            r = probe(a.url, a.model, prompt, a.timeout)
            if "error" in r:
                rows.append({"target": size, "depth": depth, "unverified": True,
                             "detail": r["error"], "secs": round(r["secs"], 1)})
                print("  %-9d %-6.2f %-9s %-9.1f UNVERIFIED (%s)"
                      % (size, depth, "-", r["secs"], r["error"][:44]))
                continue
            ans = (r["answer"] or "").strip()
            ok = needle in ans
            # ⚠️ DISTINGUISH RETRIEVAL FROM TRANSCRIPTION. Measured 2026-08-16: both
            # "failures" at 74k and 148k tokens were 1-2 CHARACTER errors
            # (SPARK-NIAH-MKCBSUN6 -> ...MKCBSUN, SPARK-NIAH-... -> SPARK-NIA0-...).
            # The model FOUND the needle every time. Grading those as retrieval failures
            # would understate long-context capability and point debugging at the wrong
            # thing entirely.
            sim, near = near_miss(needle, ans)
            rows.append({"target": size, "depth": depth, "pass": ok, "near_miss": near,
                         "similarity": round(sim, 3),
                         "prompt_tokens": r["prompt_tokens"], "secs": round(r["secs"], 1),
                         "answer": ans[:60], "needle": needle})
            print("  %-9d %-6.2f %-9s %-9.1f %s%s"
                  % (size, depth, r["prompt_tokens"], r["secs"],
                     "PASS" if ok else ("NEAR-MISS" if near else "FAIL"),
                     "" if ok else "  got=%r (%.0f%% match)" % (ans[:34], sim * 100)))

    graded = [x for x in rows if "pass" in x]
    p = sum(1 for x in graded if x["pass"])
    nm = sum(1 for x in graded if x.get("near_miss"))
    u = len(rows) - len(graded)
    print("\n" + "=" * 60)
    print("  EXACT MATCH   %d/%d" % (p, len(graded)))
    print("  RETRIEVAL     %d/%d   (exact + near-miss: the needle was FOUND)"
          % (p + nm, len(graded)))
    if nm:
        print("  -> %d near-miss(es): 1-2 character transcription errors, not retrieval"
              % nm)
    if u:
        print("  UNVERIFIED    %d" % u)
    if graded:
        print("  deepest passing context: %s prompt tokens"
              % max((x["prompt_tokens"] or 0) for x in graded if x["pass"]) if p else "none")
    if a.out:
        json.dump({"label": a.label, "model": a.model, "passed": p, "graded": len(graded),
                   "unverified": u, "probes": rows}, open(a.out, "w"), indent=1)
        print("  saved -> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
