# verifier-gate

**Run a benchmark, and refuse to believe the number unless it survives a check.**

```bash
./verify.sh /path/to/llm-as-a-verifier
```

Exit 0 = the score is real. Anything else = do not publish it.

## Why you want this

A run against a **dead port** scores **79.8%** and exits cleanly:

```
Pass@1                         70.67/89    79.4%
LLM-as-a-Verifier                 71/89    79.8%   <- made of nothing
Oracle (Bo3)                      82/89    92.1%
Verifier tokens (0 verifier calls)
```

Failed calls become 0.5 ties, so the run completes and prints a plausible number. Most
harnesses do some version of this. Point yours at a closed port and find out.

Using it caught a real bug: **22% of every score was being silently destroyed**
([issue #10](https://github.com/llm-as-a-verifier/llm-as-a-verifier/issues/10),
[PR #11](https://github.com/llm-as-a-verifier/llm-as-a-verifier/pull/11)).

---

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


## What it costs (measured, 4 runs)

End-to-end best-of-3 on short outputs, 2× DGX Spark GB10. Prompt: a 119-char coding question,
`max_tokens=300`, `temperature=0.7`, clocks 2405/3003 MHz at ~40 °C.
Raw per-request data: [`results/2026-08-18/verify-cost*.json`](../results/2026-08-18/).

| concurrency | aggregate | mean latency |
|---|---|---|
| 1 | 70 tok/s (67–72, sd 2) | 4.1 s |
| 3 | 160 tok/s (142–185, sd 16) | 5.5 s |
| 6 | **237 tok/s** (226–256, sd 12) | 7.2 s |

**3.4× the throughput for ~75% more latency.** Generating N candidates in parallel is therefore
cheap — 3 candidates cost ~5.5 s, not 3 × 4.1 s.

Verification itself measured at ~35 s for 12 scoring calls at c=12, giving roughly **41 s
end-to-end** against ~4 s for a single unverified answer.

⚠️ Run-to-run spread is large — the c=3 figure ranged 142–185 tok/s across four runs. An earlier
version of this table quoted single-run numbers (73/142/268); the 268 was the top of the range,
not the centre. **Do not quote a single run of this.**

On long agent traces (~95k tokens) verification is ~21 min/task at K=1: a deliberate
"do this properly" invocation, or an overnight batch. Not every turn.

## Configuration — WITHDRAWN pending honest re-measurement

This section previously recommended K=1 with all three criteria, based on an ablation run
offline against a completed 576-entry cache.

**That ablation was invalid and the recommendation is withdrawn.** The cache holds only **4 of
the 6** possible directed pairs per task — the tournament scores only the pairs it needs.
Re-shuffling the ring to average over seeds demands pairs that were never scored, and
`directed_reward` silently defaults a missing entry to **0.5**. Audited directly: **6,000 of
28,800 lookups (20.8%) were fabricated 0.5/0.5 ties.**

That is precisely the failure this gate exists to catch, committed in our own follow-up
analysis and published as advice. It is left described here rather than deleted, because the
lesson is the point: **an offline ablation over a cache is only valid for configurations whose
keys the cache actually contains.**

The one claim that survived every reconstruction method is that **K=1 performs about the same as
K=2**. Whether a single criterion is cheaper-but-worse is genuinely unresolved: different
reconstructions disagree on the sign.

To settle it honestly, score the ~288 missing reversed-ring entries once — roughly half a run —
after which every configuration and shuffling is computable offline with no fabrication.

