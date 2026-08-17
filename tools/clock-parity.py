#!/usr/bin/env python3
"""Assert every node in a tensor-parallel cluster runs the SAME GPU clock policy.

WHY THIS EXISTS. We measured a GPU clock cap costing +9.5% prefill / +1.7% decode and
removed it on both nodes. But a boot-time service re-applied the cap at every start, so
**live state is not boot state**: whichever node reboots comes back capped while the other
stays uncapped.

That asymmetry is worse than either state on its own. Tensor parallelism runs in lockstep
across nodes, so every layer proceeds at the SLOWEST node's pace — the entire gain
evaporates, throughput quietly returns to capped levels, and nothing reports it. The
cluster looks healthy the whole time.

⚠️ IT MUST BE RUN UNDER LOAD OR IT LIES. Verified on this hardware: no `nvidia-smi` field
reports an `-lgc` lock. `Applications Clocks` reads a fixed value, `Max Clocks` another,
and neither moves when the lock is applied or released. At idle a capped and an uncapped
GPU can report the SAME clock. **The clock ACHIEVED while the GPU is working is the only
evidence of clock policy that exists.** So this tool drives a real generation and samples
during it; with no engine to load the GPUs it reports UNVERIFIED — never "ok".

Proven by execution, not by reading: a cap was re-applied to one node on purpose and this
reported a 539 MHz spread.

Exit codes:  0 = parity OK   1 = MISMATCH (actionable)   2 = UNVERIFIED (could not tell)
"""
import argparse, json, subprocess, sys, threading, time, urllib.request

DEFAULT_TOLERANCE_MHZ = 200      # generous: DVFS jitters; we are catching ~500 MHz gaps


def verdict(peaks, nodes, tolerance, uncapped_above):
    """Pure. peaks: {node: (mhz, temp)}. Returns (exit_code, [lines])."""
    lines = []
    missing = [n for n in nodes if n not in peaks]
    if missing:
        lines.append("UNVERIFIED: no clock samples from %s. A node that cannot be sampled "
                     "is not a node that is fine." % ", ".join(missing))
        return 2, lines

    for n in nodes:
        lines.append("  %-10s achieved %4.0f MHz under load, %2.0f C"
                     % (n, peaks[n][0], peaks[n][1]))

    vals = [peaks[n][0] for n in nodes]
    spread = max(vals) - min(vals)
    if spread > tolerance:
        slow = min(nodes, key=lambda n: peaks[n][0])
        lines += [
            "",
            "MISMATCH: %.0f MHz spread between nodes (tolerance %d)." % (spread, tolerance),
            "  '%s' is the SLOW node at %.0f MHz. Tensor parallelism runs in lockstep, so"
            % (slow, peaks[slow][0]),
            "  the WHOLE cluster runs at its pace and the uncapped gain is gone.",
            "  Most likely cause: '%s' rebooted and a boot-time service re-applied a clock"
            % slow,
            "  cap — live state is not boot state.",
            "  Fix (apply to EVERY node, symmetry is the point):  sudo nvidia-smi -rgc",
        ]
        return 1, lines

    state = "UNCAPPED" if min(vals) > uncapped_above else "CAPPED (or thermally limited)"
    lines.append("")
    lines.append("PARITY OK: all %d nodes %s (spread %.0f MHz, tolerance %d)."
                 % (len(nodes), state, spread, tolerance))
    return 0, lines


def sample(node, local_node):
    cmd = "nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader,nounits"
    argv = (["bash", "-c", cmd] if node == local_node else
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", node, cmd])
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=25)
        if r.returncode != 0:
            return None, None
        sm, temp = [x.strip() for x in r.stdout.strip().split(",")[:2]]
        return float(sm), float(temp)
    except Exception:
        return None, None


def engine_up(base):
    try:
        urllib.request.urlopen(base + "/metrics", timeout=8).read()
        return True
    except Exception:
        return False


def generate(base, model, tokens, out):
    """A short real generation — the load that makes achieved clocks meaningful.

    ⚠️ REPORTS WHETHER IT ACTUALLY RAN. This used to end in `except Exception: pass`, so a
    404 from a wrong model name returned instantly, created NO load, and the tool still
    printed "achieved N MHz under load ... PARITY OK". A confident verdict with no evidence
    its own precondition held — which is the exact thing this tool exists to catch.
    """
    body = {"model": model, "max_tokens": tokens, "temperature": 0.0, "stream": False,
            "messages": [{"role": "user",
                          "content": "Write a detailed paragraph about compiler optimization."}]}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=180))
        out["tokens"] = d.get("usage", {}).get("completion_tokens", 0) or 0
    except Exception as e:
        out["error"] = "%s: %s" % (type(e).__name__, str(e)[:120])


def self_test():
    """Every case must return the code stated. If nothing fails, suspect the harness."""
    print("=== SELF-TEST: the verdict must be able to FAIL ===\n")
    ok = True
    cases = [
        ("both uncapped, in parity",
         {"a": (2528, 60), "b": (2574, 62)}, ["a", "b"], 0),
        ("both capped, still in parity",
         {"a": (1980, 55), "b": (1995, 57)}, ["a", "b"], 0),
        ("ONE NODE CAPPED (the real failure)",
         {"a": (2519, 61), "b": (1980, 54)}, ["a", "b"], 1),
        ("spread just inside tolerance",
         {"a": (2500, 60), "b": (2320, 60)}, ["a", "b"], 0),
        ("spread just outside tolerance",
         {"a": (2500, 60), "b": (2290, 60)}, ["a", "b"], 1),
        ("a node did not answer -> UNVERIFIED, not ok",
         {"a": (2500, 60)}, ["a", "b"], 2),
        ("three nodes, one slow",
         {"a": (2500, 60), "b": (2510, 61), "c": (1990, 52)}, ["a", "b", "c"], 1),
    ]
    for label, peaks, nodes, want in cases:
        got, _ = verdict(peaks, nodes, DEFAULT_TOLERANCE_MHZ, 2200)
        good = got == want
        ok &= good
        print("  %-38s rc=%d want=%d  %s" % (label, got, want,
                                             "ok" if good else "*** BROKEN ***"))
    print("\n  %s\n" % ("HARNESS TRUSTWORTHY" if ok else "*** BROKEN — DO NOT TRUST RESULTS ***"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nodes", default="spark-1,spark-2",
                    help="comma-separated hostnames, reachable by ssh")
    ap.add_argument("--local-node", default=None,
                    help="name of the node this runs on, if any (skips ssh for it)")
    ap.add_argument("--base", default="http://spark-1:8888",
                    help="engine base URL used to create load")
    ap.add_argument("--model", default="deepseek-v4-flash-dspark")
    ap.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE_MHZ)
    ap.add_argument("--uncapped-above", type=int, default=2200,
                    help="achieved MHz above which the cluster is reported UNCAPPED")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=220, help="generation length for the load")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    nodes = [n.strip() for n in a.nodes.split(",") if n.strip()]
    base = a.base.rstrip("/")

    if not engine_up(base):
        print("UNVERIFIED: engine not reachable at %s — cannot create load, so achieved "
              "clocks would be meaningless.\n  (A capped and an uncapped GPU look identical "
              "at idle. This is why there is no idle fallback.)" % base)
        return 2

    load = {}
    t = threading.Thread(target=generate, args=(base, a.model, a.tokens, load), daemon=True)
    t.start()
    time.sleep(3)

    peaks = {}
    for _ in range(a.samples):
        for n in nodes:
            sm, temp = sample(n, a.local_node)
            if sm is None:
                continue
            if n not in peaks or sm > peaks[n][0]:
                peaks[n] = (sm, temp)
        time.sleep(2)
    t.join(timeout=180)

    # The load is this tool's PRECONDITION, not a side effect. If it did not run, every
    # clock reading below is an idle reading, and a capped and an uncapped GPU are
    # indistinguishable at idle. Say UNVERIFIED; never print "under load" on no load.
    if load.get("tokens", 0) <= 0:
        print("UNVERIFIED: the load generation did not run (%s), so no clock was sampled "
              "under load.\n  A capped and an uncapped GPU read the same at idle, so any "
              "verdict here would be meaningless.\n  Check --model matches a served model "
              "name." % (load.get("error") or "no completion tokens returned"))
        return 2

    rc, lines = verdict(peaks, nodes, a.tolerance, a.uncapped_above)
    for ln in lines:
        print(ln)
    return rc


if __name__ == "__main__":
    sys.exit(main())
