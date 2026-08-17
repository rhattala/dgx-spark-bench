# dgx-spark-bench

Benchmarks and operational harnesses for a **2× NVIDIA DGX Spark (GB10)** cluster running
local LLM inference. Built while deploying and debugging a real system, not as a synthetic
exercise — most of these harnesses exist because something silently gave a wrong answer
and the harness is what caught it.

> ### Read this before quoting any number here
>
> **This is a case study on one machine, not a model ranking.** Every measurement comes
> from a single 2-node cluster, on one day, with n=3–8 depending on the test. The two
> stacks compared cannot even run at the same time — they contend for the same hardware —
> so every cross-stack comparison is sequential and hours apart.
>
> If you take one thing from this repo, take the **method**, not the leaderboard: two
> automated suites here reach *opposite* conclusions about the same twelve pages, and the
> more confident one is wrong. The numbers are evidence about this cluster. The traps are
> general.

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
| `harness/niah.py` | long-context **retrieval** — random unguessable needles, cache-defeating |
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
acceptance 45.2%  (1391/3075, mean 2.26 accepted per draft)  45.2 tok/s  floor 38%
per-position: pos0 76%  pos1 56%  pos2 42%  pos3 30%  pos4 22%
HEALTHY
```

⚠️ **The floor is not portable, and it is deliberately loose.** Re-baselined on the 0731
checkpoint (n=6, this pinned prompt, uncapped clocks): **mean 46.3%, sd 1.61**. The previous
floor of 33% came from the preview checkpoint and sat **8.3σ** below that — it would have
fired only on a catastrophe. It is now **38%**.

38 is *not* mean − 3sd (that would be 41.5). The sd is **within-session**: one day, one clock
state, one load condition — and this repo has already measured that across-session drift
exceeds within-session drift. A floor tuned to within-session variance would false-alarm on
an ordinary busy afternoon. 38 sits ~5sd below the mean, below the lowest reading ever
recorded on *either* checkpoint, and still halves the old gap. Tighten it only with a
multi-day baseline. Change the prompt and it means nothing at all.

⚠️ **A self-test must prove each ARM, not just each case.** A "no big rises" rule alone
cannot see a flat profile — `[56,55,56,55,56]` and even a *rising* `[56,61,66,71,76]` both
passed it — so a minimum decay-per-position rule was added. But this defect was then
committed twice. First the decay rule was unproven, because the flat test case also had a
weak `pos0` that a *different* arm rejected. Fixing that made the **pos0 arm** unproven — with
the stronger decay rule in place, deleting the pos0 comparison entirely left the whole
suite green. A case two arms can each reject proves neither, and adding an arm can silently
orphan an existing one. So coverage is now **measured, not asserted**: the self-test
disables each arm in turn and fails if no case changes verdict.

⚠️ **Scoped to k=5, and it says so.** An earlier version claimed per-step scaling made the
decay rule "valid for k != 5". It isn't — healthy decay is convex, so the per-step average
falls as k grows and an extrapolated healthy k=10 profile grades FLAT. Any other k is now
UNVERIFIED rather than confidently wrong. The decay test also compares **endpoints**: the
middle of the profile is unconstrained, so `[79,40,39,38,37]` passes despite pos1 sitting
far outside every healthy sample.

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

### The finding, in two screenshots

Same prompt, same day, both asked for a kanban board with add / edit / delete / drag / filter
/ undo. Both scored **18/18 on static checks, in all three runs.**

**DeepSeek — three columns, and an "Add card…" input with a + button in every one:**

![DeepSeek kanban](docs/img/kanban-deepseek.png)

**Qwen — arguably the nicer board. Drag handles, status pills, card IDs, filter, undo,
dark-mode toggle, keyboard legend. And no way to add a card. Anywhere:**

![Qwen kanban](docs/img/kanban-qwen.png)

That is the whole argument for functional testing. The prettier page is the broken one, and
the static suite ranked it *higher* precisely because it contained more of the strings the
suite greps for. Presence in the source is not function.

⚠️ **A detail that matters for honesty:** this is Qwen's run 2. Run 1 threw an uncaught
`Unexpected token 'null'` and rendered nothing at all
([screenshot](docs/img/kanban-qwen-run1-jserror.png)) — using *that* as the comparison would
imply the board is always blank, which is false. The missing add-card control is present in
**3 of 3** runs; the total render failure was 1 of 3.

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
### Known limitations, carried deliberately

These survived four adversarial review passes and are recorded rather than fixed, because
the honest answer in each case is that fixing them costs more than the risk they carry.

- **acceptance-probe** compares profile ENDPOINTS, so the middle is unconstrained by
  anything but the no-big-rises rule: `[79,40,39,38,37]` passes despite pos1 sitting far
  outside every healthy sample. It is scoped to **k=5** and returns UNVERIFIED for any other
  k — healthy decay is convex, so per-step thresholds do not transfer, and no healthy k=10
  profile has ever been measured. Whether a fresh engine pre-registers all five per-position
  counters is unknown; learning it requires a restart.
- **thermal-guard cannot detect a pre-existing operator cap.** No `nvidia-smi` field reports
  an `-lgc` lock, so this is physically undetectable, not an oversight. Do not run it on a
  node you have deliberately capped.
- **clock-parity** proves its load generation *completed*, not that the sampling window
  overlapped it. A request queued behind a busy engine could generate after sampling ends.
  It fails safe today; reproducing the failure needs a saturated engine.
- **niah's near-miss means "a near-copy of the needle appears in the answer"** — retrieval
  evidence, not answer correctness. A reply naming a variant while denying it has the code
  counts. Retrieval and exact-match are therefore always reported separately.
- **Notify-failure paths are unexercised by any test.** Forcing one risks a real page; a
  harness would need an injectable notify command, which is a design change rather than a
  patch.

- The two stacks **cannot run simultaneously** — they contend for the same hardware — so every
  cross-stack comparison is sequential, hours apart, on a shared machine.
