#!/usr/bin/env python3
"""Completeness + integrity gate for a bo3 verifier run.

WHY THIS EXISTS
---------------
A naive "did the cache reach N entries?" gate is NOT a gate here, for three
independent reasons found in review:

  1. ARITHMETIC. ring(3) + pivot_round(k=1) = 5 pairs, but they ALWAYS overlap
     by exactly one directed pair, so only 4 are distinct. Expected entries are
     24 swing x 4 pairs x 3 criteria x 2 reps = 576, not 720. A count gate set
     to 720 fails a flawless run.

  2. POISONED ENTRIES COUNT. `_score_tags_by_prefill` swallows any exception and
     returns tag-less; `extract_score` then returns 0.5 as a NORMAL value, so
     `score_directed_pairs` writes {"score_A": 0.5, "score_B": 0.5} INTO THE
     CACHE as if it were a real measurement. Only calls that raise are excluded.
     A run in which every prefill call failed produces a complete-looking cache
     of ties that passes any count-based check.

  3. ORPHANS INFLATE. Resuming after a partial phase A can select a different
     pivot, leaving phase-B entries for pairs no longer needed. Count goes UP
     while completeness goes DOWN.

So this checks SET MEMBERSHIP against the deterministically-recomputed key set,
plus an exact-0.5 detector, plus the call budget.

Exit codes: 0 = pass, 1 = FAIL, 2 = UNVERIFIED (cannot check -- never "ok").
"""
import argparse, json, math, os, random, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))

from llm_verifier import pivot_tournament as ppt          # noqa: E402
from llm_verifier.fine_grained_reward import (            # noqa: E402
    cache_key, directed_reward)
from llm_verifier.benchmarks import BENCHMARKS            # noqa: E402
import run as R                                           # noqa: E402

HTTP_PER_SCORING_CALL = 3      # 1 generation + 2 constrained prefill calls


def directed_from_cache(cache, crit_ids, task, a, b, n_reps):
    """(score_A, score_B) for a directed pair, or None if any entry is absent.

    Delegates the AGGREGATION to the production `directed_reward` rather than
    reimplementing it, so this gate cannot drift from the code it audits. It
    only adds the presence check `directed_reward` deliberately lacks: that
    function defaults missing entries to 0.5, which is right for scoring and
    exactly wrong for a completeness gate.
    """
    for cid in crit_ids:
        for rep in range(n_reps):
            if cache_key(cid, task, a, b, rep) not in cache:
                return None
    return directed_reward(cache, task, a, b, crit_ids, n_reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="terminal_bench_2.1")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--n-trials", type=int, default=3)
    ap.add_argument("--pivots", type=int, default=1)
    ap.add_argument("--n-reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--strict", action="store_true",
                    help="orphan entries FAIL instead of WARN. Use for the "
                         "acceptance run: the protocol is a cache created empty, "
                         "so any orphan is contamination or a pivot that moved "
                         "on resume, and needs a written explanation.")
    a = ap.parse_args()

    cfg = BENCHMARKS[a.benchmark]
    cache_path = a.cache or os.path.join(
        ROOT, cfg.cache.replace(".json", "_bo3.json"))
    seed = a.seed if a.seed is not None else cfg.seed

    if not os.path.exists(cache_path):
        print(f"UNVERIFIED: no cache at {cache_path}"); return 2
    try:
        with open(cache_path) as f:
            cache = json.load(f)
    except Exception as e:
        print(f"UNVERIFIED: cache unreadable ({e})"); return 2
    if not isinstance(cache, dict):
        print("UNVERIFIED: cache is not a dict"); return 2

    tasks, _ = R.LOADERS[cfg.loader](cfg.data, ROOT)
    tasks = {t: v[:a.n_trials] for t, v in tasks.items()}
    all_pass, swing = R.classify(tasks)
    crit_ids = list(cfg.criteria)

    # Rings are drawn deterministically from the seed, in run.py's order.
    rng = random.Random(seed)
    rings = {t: ppt.ring_cycle(len(tasks[t]), rng) for t in swing}

    needed, missing, unresolved = set(), [], []
    for task in swing:
        n = len(tasks[task])
        for pair in rings[task]:
            for cid in crit_ids:
                for rep in range(a.n_reps):
                    needed.add(cache_key(cid, task, pair[0], pair[1], rep))
        # pivots depend on the ring scores that must already be cached
        w, c = [0.0] * n, [0] * n
        ok = True
        for (x, y) in rings[task]:
            d = directed_from_cache(cache, crit_ids, task, x, y, a.n_reps)
            if d is None:
                ok = False; break
            p = ppt.bradley_terry(*d)
            w[x] += p; c[x] += 1; w[y] += 1.0 - p; c[y] += 1
        if not ok:
            unresolved.append(task)      # phase A incomplete -> pivots unknowable
            continue
        for pair in ppt.pivot_round_pairs(n, ppt.select_pivots(w, c, a.pivots)):
            for cid in crit_ids:
                for rep in range(a.n_reps):
                    needed.add(cache_key(cid, task, pair[0], pair[1], rep))

    missing = sorted(needed - set(cache))
    orphans = sorted(set(cache) - needed)
    # An exact 0.5 is the fallback signature. The scale is integers 1..20
    # mapped by (v-1)/19, so the regex fallback can NEVER return exactly 0.5
    # ((10-1)/19 = 0.4737, (11-1)/19 = 0.5263); only an error tie or a tag-less
    # prefill result produces it. A genuine logprob expectation hitting exactly
    # 10.5/20 in float64 is possible but vanishingly unlikely, and both sides
    # doing so simultaneously more so — hence the both/one breakdown, so a
    # flagged entry is diagnosable rather than merely rejected.
    ties_both = sorted(k for k, v in cache.items()
                       if v.get("score_A") == 0.5 and v.get("score_B") == 0.5)
    ties_one = sorted(k for k, v in cache.items()
                      if (v.get("score_A") == 0.5) != (v.get("score_B") == 0.5))
    ties = ties_both + ties_one

    print(f"cache            : {cache_path}")
    print(f"swing tasks      : {len(swing)}  (all-pass {len(all_pass)})")
    print(f"expected entries : {len(needed)}")
    print(f"present entries  : {len(cache)}")
    print(f"missing          : {len(missing)}")
    print(f"orphan (unneeded): {len(orphans)}")
    print(f"exact-0.5 (both) : {len(ties_both)}   <-- error tie / both tags lost")
    print(f"exact-0.5 (one)  : {len(ties_one)}    <-- second prefill call failed")
    print(f"implied HTTP     : {len(needed) * HTTP_PER_SCORING_CALL}")

    # ORDER MATTERS. Definite evidence of failure outranks "cannot verify":
    # a cache that is BOTH incomplete AND poisoned must report the poison, not
    # merely decline to judge. Both codes are unhealthy, so nothing is wrongly
    # accepted either way -- but "I found poison" is a stronger, more actionable
    # claim than "I could not check", and the earlier version threw it away.
    fail = False
    if unresolved:
        print(f"\nNOTE: phase A is incomplete for {len(unresolved)} task(s), so their "
              f"pivots cannot be derived and their phase-B keys are absent from the "
              f"needed set. CONSEQUENCE: `missing` is UNDERSTATED and `orphan` is "
              f"OVERSTATED by those tasks: {unresolved[:5]}")
    if missing:
        print(f"\nFAIL: {len(missing)} needed score(s) absent — those comparisons "
              f"were silently scored 0.5/0.5 for this run. e.g. {missing[:3]}")
        fail = True
    if ties:
        print(f"\nFAIL: {len(ties)} cached entr(ies) hold an exact 0.5 "
              f"({len(ties_both)} both-sided, {len(ties_one)} one-sided). The "
              f"regex fallback cannot produce exactly 0.5 on a 1..20 scale, so "
              f"these are error ties or tag-less prefill results recorded AS "
              f"DATA. e.g. {ties[:3]}")
        fail = True
    if orphans:
        lvl = "FAIL" if a.strict else "WARN"
        print(f"\n{lvl}: {len(orphans)} cached entr(ies) are not needed by this "
              f"configuration (stale cache, contamination, or a pivot that moved "
              f"on resume).")
        if a.strict:
            fail = True

    if fail:
        print("\nRESULT: FAIL")
        return 1
    if unresolved:
        print("\nRESULT: UNVERIFIED — nothing provably wrong, but completeness "
              "cannot be established while phase A is incomplete.")
        return 2
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
