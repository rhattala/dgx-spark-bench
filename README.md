# spark-bench

Benchmarks and operational harnesses for a **2× NVIDIA DGX Spark (GB10)** cluster running
local LLM inference. Built while deploying and debugging a real system, not as a synthetic
exercise — most of these harnesses exist because something silently gave a wrong answer
and the harness is what caught it.

## Hardware

| | |
|---|---|
| Nodes | 2 × DGX Spark, GB10, **121 GB unified memory each** (273 GB/s bandwidth) |
| Interconnect | ConnectX-7, RoCEv2, MTU 9000, ~13.9 GB/s busbw — **not NVLink** |
| Primary stack | DeepSeek-V4-Flash-0731, vLLM, TP=2 across both nodes, 1M context, fp8 KV |
| Alt stack | Qwen3.8-27B-NVFP4 + DSpark drafter, SGLang, TP=1 on one node |
| Clocks | **uncapped** (~2500 MHz achieved; 2000 MHz cap removed after measurement) |

⚠️ **121 GB is memory CAPACITY. 273 GB/s is BANDWIDTH.** They get conflated constantly and
the distinction changes every sizing calculation.

## What's here

| harness | what it measures |
|---|---|
| `harness/dspark-bench.py` | 44-task machine-graded quality suite (math, code-that-executes, format, tool selection) |
| `harness/dspark-frontend-bench.py` | 4-tier front-end generation, sequential or **parallel**, with rendered artifacts |
| `harness/api_suite.py` | **the server, not the model** — OpenAI-compat contract, streaming, tool calls, error handling, security posture |
| `harness/soak.py` | sustained-load soak with per-node thermal + clock telemetry |
| `harness/qwen38-clock-probe.py` | decode / prefill / concurrency probe, endpoint-agnostic |
| `harness/niah.py` | long-context retrieval — random unguessable needles, cache-defeating |
| `tools/acceptance-probe.py` | speculative-decoding health by **per-position profile**, not by a bare percentage |
| `tools/clock-parity.py` | asserts every node shares one clock policy — **must run under load** |
| `tools/thermal-guard.py` | cap-on-hot / release-on-cool, with ownership, self-limit and liveness |

## Operational checks

Three of these are not benchmarks. They are checks that each caught something a reasonable
person had already concluded was fine, and each is useful to anyone running the same class
of system rather than this specific one.

### `tools/acceptance-probe.py` — a bare acceptance % is not a health metric

Draft acceptance swings **~20 pp on prompt alone**. Measured on one healthy cluster,
minutes apart, loader patch verified present: the prose prompt read **25.5%** (would
scream "loader bug live") and a code prompt read **45.7%** (would fail a ≥55% gate). Both
false alarms about a system that was fine.

So the probe pins prompt, sampling and token count, warms the engine, and grades the
**shape** before the number — a healthy k=5 drafter decays monotonically across draft
positions; one running on badly-loaded weights collapses roughly flat.

```
acceptance 50.9%  (1437/2825, mean 2.54 accepted per draft)  49.0 tok/s  floor 33%
per-position: pos0 79%  pos1 62%  pos2 48%  pos3 37%  pos4 28%
HEALTHY
```

⚠️ **The floor is not portable.** 33% sits below a measured mean of 42.8% *on this exact
prompt* (n=3: 43.0/40.2/45.3, sd 2.6) — that is ~3.8σ, not the "~2σ" an earlier version of
this file claimed. Conservative in the safe direction, but the σ figure was simply wrong.
Change the prompt and the floor is meaningless, which is the entire point of the tool. The
baseline also predates the 0731 checkpoint, which reads ~50.9%: re-baseline before treating
33% as tuned rather than merely loose.

⚠️ **Shape grading needs BOTH rules.** A "no big rises" test alone cannot see a flat
profile — and flat is the fault it exists to catch. `[56,55,56,55,56]` and even
`[56,61,66,71,76]` passed it. The self-test missed this because its flat case used a weak
`pos0`, which a *different* arm rejects: the case was over-determined, so it passed while
the arm under test did nothing. There is now a minimum decay-per-position requirement and
cases that must fail without it.

⚠️ Parse `/metrics` **by name**. The endpoint emits drafts, draft_tokens and accepted in
that order, and a positional parse silently mislabels them into a plausible-looking rate.
`--self-test` carries a negative control for precisely this.

### `tools/clock-parity.py` — asymmetry is worse than either state

We uncapped the GPU clock on both nodes, but a boot-time service re-applied the cap at
every start, so **live state is not boot state**: whichever node reboots comes back capped
while the other stays uncapped. Tensor parallelism runs in lockstep, so the whole cluster
then runs at the slow node's pace, the measured gain evaporates, and nothing reports it.

⚠️ **It must run under load or it lies.** No `nvidia-smi` field reports an `-lgc` lock —
`Applications Clocks` and `Max Clocks` are both fixed and neither moves when the lock is
applied or released. At idle, a capped and an uncapped GPU can read the **same** clock.
The clock *achieved while working* is the only evidence that exists, so the tool drives a
real generation and samples during it. With no engine to load the GPUs it returns
**UNVERIFIED (rc=2)** — there is deliberately no idle fallback.

### `tools/thermal-guard.py` — a loop that re-asserts state needs an owner

One job: cap the clock above `--trip`, release it after `--clear-cycles` samples below
`--clear`. It never touches the engine or any service — a guard that can stop your serving
stack is a bigger hazard than the heat it prevents.

It exists in this shape because of one incident: a verify-and-reapply loop re-asserted an
admission gate every 5 s, correctly and tirelessly, **on behalf of a test unit that should
have been dead**. The loop wasn't buggy — it had no concept of "I should not be running."
So this one carries four defences: an flock (one owner, ever), a `--max-runtime` self-limit,
a heartbeat whose `--status` says **UNVERIFIED (rc=2)** rather than "ok" when stale, and
release-only-what-it-applied. ⚠️ That last defence has a limit worth stating: **it
cannot detect a cap that was already there** — no `nvidia-smi` field reports an `-lgc` lock,
which is why the parity tool samples clocks under load. Cap a node deliberately, let the
guard trip, and it will record that cap as its own and release *yours* on exit. Don't run it
on a node you have capped by hand.

Proven by execution on a live node in `--dry-run` (no clocks touched):

| test | result |
|---|---|
| trip fires (`--test-trip 30` at 70 °C) | applied ✅ |
| second instance | refused, rc=1 ✅ |
| `--status` while alive | OK, rc=0 ✅ |
| SIGTERM | released what it applied, rc=1 STOPPED ✅ |
| **SIGKILL then wait** | UNVERIFIED rc=2 — "you are not protected" ✅ |

⚠️ **That last test also exposed the honest limit:** staleness detection has a blind
window of `interval × 6`. A hard-killed guard read "OK" for ~12 s at `--interval 2`, ~30 s
at the default. One OK is not proof of protection during an incident. And the interval used
is the one the **running guard** recorded — judging by the caller's `--interval` produced
both a false alarm (60 s guard called dead by a 5 s check) and, worse, a **false OK on a
dead guard**, which silently extended that window.

All three tools were verified by **execution, not inspection**: the parity check was proven by
re-applying a cap to one node on purpose (caught a 539 MHz spread), and both `--self-test`
suites were re-run against deliberately-broken copies to confirm they fail loudly.

## Rules every harness here follows

These are not style preferences. Each one exists because its absence produced a wrong
result that was believed for a while.

1. **Every check must be able to FAIL.** Suites carry `--self-test` with deliberately-wrong
   fixtures. *If nothing in a run failed, suspect the harness before believing the result.*
2. **Verify by execution, not by reading.** "I read the code and it looks right" is not a
   result. Break the thing the check checks, and confirm it screams.
3. **A check that cannot verify says UNVERIFIED — never "ok".** `2>/dev/null` and `|| true`
   convert failure into silence.
4. **Every throughput number names its PROMPT, TOKEN COUNT and CLOCK STATE.** All three move
   the result by more than the effects people try to measure with them.
5. **Parse metrics BY NAME, never positionally.**
6. **Verify identity from the artifact, never the label.** A model name in a config proves
   nothing about what answered.

## Findings

### Clock cap cost — measured on both stacks

Interleaved ABBA design, n=8 pairs per arm, paired test governs, 2% practical-significance
floor declared **before** the data existed. Achieved clock recorded per trial, because no
nvidia-smi field reports an `-lgc` lock.

| stack | decode | prefill (~14k tok) |
|---|---|---|
| Qwen3.8-27B / SGLang / 1 node | +1.0% *(below floor)* | +5.7% |
| DeepSeek-V4-Flash / vLLM / TP=2 | +1.7% *(below floor)* | **+9.5%** |

**SM clock rose 28% and decode moved 1%** — decode is memory-bandwidth-bound. Prefill is
compute-bound and gains materially.

⚠️ A pre-registered prediction that DeepSeek's prefill gain would be *smaller* than Qwen's
(NCCL allreduce being clock-insensitive) **was wrong** — it came in at nearly double.
Recorded as a miss rather than quietly reframed.

⚠️ There is **no headroom beyond uncapped**: forcing a 2800 MHz floor yields the same
2574 MHz, with `nvidia-smi` reporting "All done" while changing nothing. No settable power
limit exists on GB10.

### Sustained load is thermally fine

10 min at concurrency 4, uncapped, both nodes: **0 errors, temps plateau 68–71 °C, clocks
hold 2476/2515 MHz with no droop, throughput flat.** Guard trips at 88 °C.

### Reasoning effort is worth 9 checks — all of them in maths

| | reasoning OFF | reasoning HIGH |
|---|---|---|
| maths / logic | 6/15 | **15/15** |
| code | 12/12 | 12/12 |
| format | 4/4 | 4/4 |
| **total** | 35/44 | **44/44** |
| wall clock | **45 s** | 145 s (3.2×) |

**Rule that falls out:** reasoning ON for maths and logic; OFF for code (identical score,
3.2× faster). And **OFF for long code generation** — at `high`, both DeepSeek and Qwen burn
the entire 12k-token budget in the reasoning channel and return **zero bytes**.

⚠️ **Qwen3.x ignores `reasoning_effort` entirely** — it gates thinking via
`chat_template_kwargs.enable_thinking`. The OpenAI-style knob is accepted and silently
does nothing.

### Static checks saturate and then INVERT the answer

Both models scored 17–18/18 on the hardest front-end task under static checks. Driving the
same pages in a real browser (`harness/functional_check.py`) reverses the ranking:

**Both models, 3 runs per tier, 12 pages each:**

| | static checks | functional checks |
|---|---|---|
| DeepSeek-V4-Flash | 54.3/63 | **75/75 (100%)** |
| Qwen3.8-27B | **61.3/63** | 66/75 (88%) |

**Static says Qwen. Functional says DeepSeek.** And the failure is reproducible: Qwen's
kanban board failed `add_card_EXISTS` and `add_card_works` on **all three runs**, while
scoring **18/18** on static checks every time — because every string the static suite greps
for was present in the source. One run also threw an uncaught `Unexpected token 'null'`.

**A check suite both candidates max out has stopped measuring**, and worse, it can rank a
broken page above a working one with near-zero run-to-run variance. Presence in the source
is not function. `functional_check.py` carries a `--self-test` that runs it against a
deliberately broken fixture AND a working one, because a functional suite that passes
everything is the same trap it was built to escape.

## Results format

`results/<date>/*.json`, one file per run, each carrying its own provenance: label, model,
endpoint, clock state, reasoning setting, token budgets, and per-check detail. Numbers from
different prompts or clock states are **not** comparable and the files say so.

## Reproducing

```bash
python3 harness/api_suite.py --self-test          # prove the graders can fail
python3 tools/acceptance-probe.py --self-test
python3 tools/clock-parity.py --self-test
python3 tools/thermal-guard.py --self-test
python3 tools/acceptance-probe.py --base http://<head>:8888 --model <name>
python3 tools/clock-parity.py --nodes <n1>,<n2> --base http://<head>:8888 --model <name>
python3 harness/api_suite.py --base http://<host>:8888 --model <name>
python3 harness/dspark-bench.py --reasoning high --repeat 3
python3 harness/soak.py --concurrency 4 --minutes 10     # refuses without a thermal guard
```

## Honest limits

- **Harness precision, measured (2026-08-16, 4 tiers x 3 back-to-back runs):** widest check
  spread on any single task was **1**; three of four tasks had **zero** variance. An earlier
  claim of "+/-2 checks" was wrong — it came from runs spread across HOURS under different
  load, not a controlled repeat. Across sessions the drift is larger than within one.
- ⚠️ **Precision is not accuracy.** The same harness with near-zero variance ranked a kanban
  board with **no add-card feature** above one that implemented it three times. A repeatable
  instrument measuring the wrong thing returns the wrong answer, repeatably. Treat check
  counts as a floor, and verify function in a browser before drawing quality conclusions.
- Concurrency figures are **one-shot batch** (fire N, stop the clock when the last finishes),
  not steady-state. They are not comparable to published aggregate figures measured differently.
- The two stacks **cannot run simultaneously** — they contend for the same hardware — so every
  cross-stack comparison is sequential, hours apart, on a shared machine.
