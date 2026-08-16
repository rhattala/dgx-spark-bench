#!/usr/bin/env python3
"""Speculative-decoding health probe: pinned prompt, pinned sampling, warm engine.

WHY A BARE ACCEPTANCE PERCENTAGE IS NOT A HEALTH METRIC. Acceptance swings ~20 pp on
prompt alone. Measured on one healthy cluster, minutes apart, with the loader patch
verified PRESENT in the serving image:

    prose benchmark prompt .... 25.5%    <- would scream "loader bug is live"
    code-review prompt ........ 45.7%    <- would fail a >=55% gate

Both readings are false alarms about a system that was fine. A scalar threshold means
nothing unless the prompt that produced it is fixed, so this probe pins the prompt, the
token count and the sampling, and warms the engine first (~30% cold penalty, and it
returns after idle).

THE PER-POSITION PROFILE IS THE REAL SIGNAL. A healthy k=5 drafter decays monotonically
across draft positions. A drafter running with uninitialised weights does not — it
collapses roughly flat. Measured healthy profile, 6 runs:

    pos0 74-79%   pos1 52-60%   pos2 38-46%   pos3 24-31%   pos4 15-20%

Shape is far more diagnostic than the headline number, which a hard prompt depresses on
its own. This probe grades the shape FIRST and the percentage second.

⚠️ PARSE /metrics BY NAME, NEVER BY POSITION. The endpoint emits drafts, draft_tokens
and accepted in that order; a positional parse silently mislabels them and produces a
plausible-looking acceptance rate that is arithmetic on the wrong counters.
`--self-test` includes a negative control for exactly this.

Exit codes:  0 healthy · 1 below floor · 2 profile looks broken · 3 probe error
"""
import argparse, json, os, sys, time, urllib.request, uuid

PROMPT = """Review this function for correctness and concurrency problems, then rewrite it.

def reconcile_ledger(entries, opening_balance, *, tolerance=Decimal("0.01")):
    balance = Decimal(opening_balance)
    discrepancies = []
    for idx, e in enumerate(entries):
        if e.currency != BASE_CURRENCY:
            rate = fx_rate(e.currency, BASE_CURRENCY, on=e.posted_at)
            amount = (e.amount * rate).quantize(Decimal("0.01"), ROUND_HALF_EVEN)
        else:
            amount = e.amount
        balance += amount if e.direction is Direction.CREDIT else -amount
        if e.expected_balance is not None:
            delta = abs(balance - e.expected_balance)
            if delta > tolerance:
                discrepancies.append(Discrepancy(index=idx, entry=e, delta=delta))
    return balance, discrepancies
"""

# How far a position may RISE above its predecessor before the profile stops counting as
# decaying. Small positive slack absorbs sampling noise without admitting a flat profile.
MONOTONIC_SLACK_PP = 6
# A healthy drafter's first position is strong. Below this, the drafter itself is weak —
# the signature of a loader/weight-mapping fault rather than a hard prompt.
POS0_MIN_PCT = 55


def parse_metrics(raw):
    """Prometheus text -> counters. BY NAME. See the module docstring."""
    out = {"pos": {}}
    for line in raw.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if "spec_decode_num_draft_tokens_total" in line:
            out["draft"] = float(line.rsplit(" ", 1)[1])
        elif "spec_decode_num_accepted_tokens_per_pos_total" in line:
            out["pos"][int(line.split('position="')[1].split('"')[0])] = \
                float(line.rsplit(" ", 1)[1])
        elif "spec_decode_num_accepted_tokens_total" in line:
            out["accepted"] = float(line.rsplit(" ", 1)[1])
        elif "spec_decode_num_drafts_total" in line:
            out["drafts"] = float(line.rsplit(" ", 1)[1])
    return out


def grade(rate, prof, floor):
    """Pure verdict. (exit_code, [lines]). Shape is judged BEFORE the headline number."""
    lines = []
    if len(prof) >= 5 and prof[0] < POS0_MIN_PCT:
        lines.append("UNHEALTHY: first draft position only %.0f%% — the drafter itself is "
                     "weak. That is a loader/weight-mapping signature, not a hard prompt."
                     % prof[0])
        return 2, lines
    monotonic = all(prof[i] >= prof[i + 1] - MONOTONIC_SLACK_PP
                    for i in range(len(prof) - 1)) if prof else False
    if not monotonic:
        lines.append("UNHEALTHY: per-position profile is not decaying. A healthy drafter "
                     "loses ground at every successive position; a broken one is flat.")
        return 2, lines
    if rate < floor:
        lines.append("BELOW FLOOR: %.1f%% < %.0f%%. Profile shape is fine, so suspect load "
                     "or thermal throttling before suspecting the loader." % (rate, floor))
        return 1, lines
    lines.append("HEALTHY")
    return 0, lines


def _headers(api_key):
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = "Bearer %s" % api_key
    return h


def metrics(base, api_key):
    req = urllib.request.Request(base + "/metrics", headers=_headers(api_key))
    return parse_metrics(urllib.request.urlopen(req, timeout=20).read().decode())


def ask(base, model, api_key, max_tokens, temperature, top_p):
    body = {"model": model, "stream": False, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p,
            # unique preamble defeats prefix caching, which would serve the answer
            "messages": [{"role": "user",
                          "content": "// probe %s\n%s" % (uuid.uuid4().hex, PROMPT)}]}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=_headers(api_key))
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=300))
    return d.get("usage", {}).get("completion_tokens", 0), time.time() - t0


def self_test():
    """Every case here must return the code stated. If nothing fails, suspect the harness."""
    print("=== SELF-TEST: the grader and the parser must both be able to FAIL ===\n")
    ok = True

    print("  grade() — profile shape decides before the percentage")
    cases = [
        ("healthy decay, good rate",      45.0, [76, 55, 42, 28, 18], 0),
        ("healthy decay, rate too low",   20.0, [76, 55, 42, 28, 18], 1),
        ("FLAT profile (broken drafter)", 30.0, [30, 29, 31, 30, 29], 2),
        ("INVERTED profile",              40.0, [20, 40, 60, 70, 80], 2),
        ("strong pos0 but rising after",  45.0, [70, 80, 60, 40, 20], 2),
        ("noisy but still decaying",      45.0, [76, 58, 60, 30, 18], 0),
    ]
    for label, rate, prof, want in cases:
        got, _ = grade(rate, prof, 33.0)
        good = got == want
        ok &= good
        print("    %-32s rc=%d want=%d  %s" % (label, got, want,
                                               "ok" if good else "*** BROKEN ***"))

    print("\n  parse_metrics() — negative control for the positional-parse trap")
    # Emitted in the order drafts, draft_tokens, accepted. A positional parse reads them
    # as draft=100, accepted=500 and reports 500% acceptance, or worse, something plausible.
    raw = (
        "# HELP vllm:spec_decode_num_drafts_total drafts\n"
        'vllm:spec_decode_num_drafts_total{model_name="m"} 100.0\n'
        'vllm:spec_decode_num_draft_tokens_total{model_name="m"} 500.0\n'
        'vllm:spec_decode_num_accepted_tokens_total{model_name="m"} 225.0\n'
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="m",position="0"} 76.0\n'
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="m",position="1"} 55.0\n'
    )
    m = parse_metrics(raw)
    checks = [("drafts == 100", m.get("drafts") == 100.0),
              ("draft_tokens == 500", m.get("draft") == 500.0),
              ("accepted == 225", m.get("accepted") == 225.0),
              ("per-pos not folded into accepted", m.get("accepted") == 225.0
               and m["pos"].get(0) == 76.0),
              ("draft is NOT the drafts counter", m.get("draft") != m.get("drafts"))]
    for label, good in checks:
        ok &= good
        print("    %-32s %s" % (label, "ok" if good else "*** BROKEN ***"))

    print("\n  %s\n" % ("HARNESS TRUSTWORTHY" if ok else "*** BROKEN — DO NOT TRUST RESULTS ***"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=os.environ.get("PROBE_BASE", "http://127.0.0.1:8888"))
    ap.add_argument("--model", default=os.environ.get("PROBE_MODEL", "deepseek-v4-flash-dspark"))
    ap.add_argument("--n", type=int, default=int(os.environ.get("PROBE_N", "5")))
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("PROBE_MAX_TOKENS", "400")))
    # Floor set ~2 sigma below a measured production mean of 42.8% on this pinned prompt.
    # It is NOT portable to another prompt — that is the whole point of the tool.
    ap.add_argument("--floor", type=float, default=float(os.environ.get("PROBE_FLOOR", "33")))
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--api-key-file", default=None,
                    help="path to a file holding a bearer token (never pass the key itself)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    api_key = None
    if a.api_key_file:
        api_key = open(os.path.expanduser(a.api_key_file)).read().strip()
        if not api_key:
            raise SystemExit("ABORT: --api-key-file %s is empty" % a.api_key_file)

    base = a.base.rstrip("/")
    try:
        for _ in range(2):
            ask(base, a.model, api_key, 300, a.temperature, a.top_p)   # warm
        before = metrics(base, api_key)
        tok, secs = 0, 0.0
        for _ in range(a.n):
            c, s = ask(base, a.model, api_key, a.max_tokens, a.temperature, a.top_p)
            tok += c
            secs += s
        after = metrics(base, api_key)
    except Exception as e:
        print("PROBE ERROR: %s: %s" % (type(e).__name__, e))
        return 3

    if "draft" not in before or "draft" not in after:
        print("PROBE ERROR: no spec-decode counters on /metrics — is speculative decoding on?")
        return 3

    dd = after["draft"] - before["draft"]
    da = after["accepted"] - before["accepted"]
    dr = after["drafts"] - before["drafts"]
    if dd <= 0:
        print("PROBE ERROR: no draft tokens observed — is speculative decoding on?")
        return 3

    rate = 100.0 * da / dd
    prof = [100.0 * (after["pos"].get(p, 0) - before["pos"].get(p, 0)) / dr
            for p in sorted(after["pos"])] if dr else []

    print("acceptance %.1f%%  (%.0f/%.0f, mean %.2f accepted per draft)  %.1f tok/s  floor %.0f%%"
          % (rate, da, dd, da / dr if dr else 0, tok / secs, a.floor))
    if prof:
        print("per-position: %s" % "  ".join("pos%d %.0f%%" % (i, v) for i, v in enumerate(prof)))
    rc, lines = grade(rate, prof, a.floor)
    for ln in lines:
        print(ln)
    return rc


if __name__ == "__main__":
    sys.exit(main())
