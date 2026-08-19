# verifier-gate

Part of [dual-dgx-spark-bench](https://github.com/rhattala/dual-dgx-spark-bench) —
tooling for a 2-node NVIDIA DGX Spark (GB10) cluster.

A completeness-and-integrity gate for [llm-as-a-verifier](https://github.com/llm-as-a-verifier/llm-as-a-verifier)
runs, and the negative controls that prove it works.

Built while running the Terminal-Bench 2.1 self-verification benchmark on a self-hosted
2× DGX Spark cluster. Full write-up: `../docs/verifying-the-verifier.pdf`.

## Why this exists

**A benchmark run against a completely dead endpoint scores 79.8% and exits 0.**

```
Method                               Score     Rate
------------------------------------------------------------------------
Pass@1                         70.67/89    79.4%
LLM-as-a-Verifier                 71/89    79.8%   ← made of nothing
Oracle (Bo3)                      82/89    92.1%
Verifier tokens (0 verifier calls)
```

Every one of the 576 scoring calls failed with `Connection error`. The framework scores a
failed call as a 0.5/0.5 tie by default, so the run completed normally and printed a plausible
result sitting exactly where a real one would — above the floor, below the ceiling. The only
honest tell is `0 verifier calls`, buried in a token-accounting block.

That is not a criticism unique to this framework. **Any harness that degrades failures into
neutral scores can manufacture a believable number from nothing.** If you run benchmarks, point
yours at a closed port and see whether it notices.

## What the gate checks

Counting cache entries is not enough, for three independent reasons:

1. **The expected count is not obvious.** For best-of-3 the ring pass and the pivot round
   *always* share exactly one directed pair, so there are 4 distinct pairs per task, not 5 —
   576 entries, not the naive 720. A count gate set to 720 fails a flawless run.
2. **Poisoned entries count.** A failure inside the score-extraction path returns 0.5 as a
   *normal* value, which then gets written to the cache. Only calls that *raise* are excluded.
3. **Orphans inflate.** Resuming after a partial phase A can select a different pivot, leaving
   entries for pairs no longer needed — the count goes up while completeness goes down.

So `verify_run.py` re-derives the needed key set deterministically, delegates score aggregation
to the framework's own `directed_reward` (so it cannot drift from the code it audits), and
flags any exact 0.5.

**Why exact 0.5 is a valid signature:** the 1–20 scale maps by `(v−1)/19`, and no value maps to
exactly 0.5 — the nearest are 0.4737 and 0.5263. The letter path therefore *cannot* emit 0.5;
only an error tie or a tag-less prefill can.

Exit codes: `0` pass · `1` FAIL · `2` UNVERIFIED. **It never reports "ok" when it cannot check.**

## Proven by making it fail

A check nobody has watched fail is not yet a check. Every row was constructed and executed:

| cache under test | verdict |
|---|---|
| valid synthetic (576 entries) | PASS (0) |
| one entry = exact 0.5, both sides | FAIL (1) |
| one entry = 0.5, one side | FAIL (1) |
| one needed entry deleted | FAIL (1) |
| incomplete **and** poisoned | FAIL (1), reporting both |
| contaminated with an unneeded key | WARN (0) / `--strict` FAIL (1) |
| full run against a dead endpoint | FAIL (1) |
| no cache at all | UNVERIFIED (2) |

## The acceptance signature

The final step replays the completed cache through the framework's **own** code with the
endpoint pointed at a dead port. A complete cache makes **zero** API calls (`LazyClient` never
constructs a client); an incomplete one fails loudly against the closed socket.

This removes the last piece of reimplementation that could drift from what it audits — the
pivots are derived by `run.py` itself, not by us.

## Files

| file | purpose |
|---|---|
| `verify_run.py` | the gate |
| `run_and_gate.sh` | preconditions → run → gate → remediation loop → acceptance replay |
| `repro_tag_lookup.py` | minimal repro of the off-by-one below (no API key needed) |
| `test_tag_lookup_whitespace.py` | 4 regression tests; 3 fail without the fix |
| `make_synthetic.py` | builds a valid synthetic cache for the controls |

## Usage

```bash
cp verify_run.py run_and_gate.sh /path/to/llm-as-a-verifier/
cd /path/to/llm-as-a-verifier

WORKERS=4 ./run_and_gate.sh          # fresh run
RESUME=1 WORKERS=4 ./run_and_gate.sh # resume; refuses if any cached 0.5 survives
python verify_run.py --strict        # gate an existing cache
```

`RESUME=1` deliberately refuses a poisoned cache: resume only re-scores *missing* keys, so a
cached tie would survive to the final result.

## The bug this found

Upstream [issue #10](https://github.com/llm-as-a-verifier/llm-as-a-verifier/issues/10) ·
[PR #11](https://github.com/llm-as-a-verifier/llm-as-a-verifier/pull/11).

`_find_tag_logprobs` keeps the last position whose accumulated text ends with the score tag. A
whitespace-only token leaves that text unchanged after `rstrip()`, so the tag matches twice and
the lookup advances one slot past the real distribution onto the closing tag's placeholder.

It fires whenever the grammar-constrained prefill samples a **bare space** — a legal prefix of
`" A"` — which strips to `""`. Measured on a self-hosted DeepSeek-V4-Flash: **78 of 357 cached
scores (22%) became exact 0.5 ties**, with no exception raised.

The distribution is present and correct in every case. It is retrieved, then overwritten.

**What it costs:** re-poisoning a completed 576-entry cache at the observed rate and re-scoring
200× gives 0.8 pp (uniform) to 1.1 pp (clustered as observed) off the mean — less than 22%
suggests, because the tournament aggregates 6 scoring calls over 4 distinct pairs per task.
**The bug inflates variance far more than it biases the mean**: the spread runs 80.9%–88.8%, so
a single run can land three points either side on luck alone. That, rather than any shift in
the average, is the reason to fix it.
