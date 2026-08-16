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

### Machine checks saturate, and then they lie

Both models scored 17–18/18 on the hardest front-end task. Functional testing in a real
browser found the higher-scoring page **had no way to add a card at all** — a feature the
prompt explicitly required, which the lower-scoring page implemented three times over.

**A check suite both candidates max out has stopped measuring.** Presence-in-the-source is
not function. This is why `results/` records what was verified functionally and what was not.

## Results format

`results/<date>/*.json`, one file per run, each carrying its own provenance: label, model,
endpoint, clock state, reasoning setting, token budgets, and per-check detail. Numbers from
different prompts or clock states are **not** comparable and the files say so.

## Reproducing

```bash
python3 harness/api_suite.py --self-test          # prove the graders can fail
python3 harness/api_suite.py --base http://<host>:8888 --model <name>
python3 harness/dspark-bench.py --reasoning high --repeat 3
python3 harness/soak.py --concurrency 4 --minutes 10     # refuses without a thermal guard
```

## Honest limits

- Several headline numbers are **n=1**. Re-running one front-end task varied the score by
  about ±2 checks; timing by a few percent. Timing conclusions survive that spread, check-count
  conclusions often do not.
- Concurrency figures are **one-shot batch** (fire N, stop the clock when the last finishes),
  not steady-state. They are not comparable to published aggregate figures measured differently.
- The two stacks **cannot run simultaneously** — they contend for the same hardware — so every
  cross-stack comparison is sequential, hours apart, on a shared machine.
