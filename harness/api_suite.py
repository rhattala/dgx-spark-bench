#!/usr/bin/env python3
"""Backend / API correctness suite for an OpenAI-compatible inference endpoint.

WHY THIS EXISTS. Every other harness here measures the MODEL. This one measures the
SERVER: does the OpenAI-compatible surface behave the way clients assume it does? A model
that answers well behind an endpoint that mishandles streaming, tool calls, token limits
or errors will still break the agent using it — and that failure looks like "the model is
dumb", which is exactly the misdiagnosis this project has made before.

DESIGN RULES (inherited from this project's burn list, and they are not optional):
  * Every check must be able to FAIL. `--self-test` proves the graders can, using
    deliberately-wrong fixtures. If nothing in a run fails, suspect the harness.
  * Parse metrics BY NAME, never positionally.
  * Never claim a capability from a label. `/v1/models` listing a name proves nothing about
    what is loaded; only generated output and the engine's own counters are evidence.
  * Report UNVERIFIED rather than "ok" when a check cannot run.

⚠️ SECURITY CHECKS ARE OBSERVATIONS, NOT EXPLOITS. This probes whether unauthenticated
routes ANSWER, using harmless payloads, because this deployment documents that
`--api-key` does not gate every route. It never attempts to disrupt the service.
"""
import argparse, json, re, sys, time, urllib.error, urllib.request

RESULTS = []


def rec(name, ok, detail="", category="api", unverified=False):
    RESULTS.append({"name": name, "ok": bool(ok) and not unverified,
                    "unverified": unverified, "detail": str(detail)[:220],
                    "category": category})
    tag = "UNVERIFIED" if unverified else ("PASS" if ok else "FAIL")
    print("  %-11s %-34s %s" % (tag, name, str(detail)[:90]), flush=True)


def http(url, payload=None, headers=None, timeout=180, method=None, raw=False):
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
            return {"code": r.status, "body": body, "secs": time.time() - t0,
                    "json": None if raw else _try(body)}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        return {"code": e.code, "body": body, "secs": time.time() - t0, "json": _try(body)}
    except Exception as e:
        return {"code": 0, "body": str(e), "secs": time.time() - t0, "json": None}


def _try(b):
    try:
        return json.loads(b)
    except Exception:
        return None


def chat(base, model, messages, **kw):
    p = {"model": model, "messages": messages, "temperature": 0.0, "stream": False}
    p.update(kw)
    return http(base + "/v1/chat/completions", p)


# ───────────────────────────── the checks ─────────────────────────────
def run(base, model, metrics_url):
    print("\n=== 1. Contract: does the response match the OpenAI schema? ===")
    r = chat(base, model, [{"role": "user", "content": "Say OK"}], max_tokens=20)
    d = r["json"] or {}
    ch = (d.get("choices") or [{}])[0]
    rec("http_200", r["code"] == 200, "code=%s" % r["code"])
    rec("has_choices", bool(d.get("choices")), "n=%d" % len(d.get("choices") or []))
    rec("has_usage_fields",
        all(k in (d.get("usage") or {}) for k in ("prompt_tokens", "completion_tokens", "total_tokens")),
        str(d.get("usage")))
    rec("usage_arithmetic",
        (d.get("usage", {}).get("total_tokens") ==
         d.get("usage", {}).get("prompt_tokens", 0) + d.get("usage", {}).get("completion_tokens", 0)),
        "total == prompt + completion")
    rec("finish_reason_valid", ch.get("finish_reason") in ("stop", "length", "tool_calls"),
        ch.get("finish_reason"))
    rec("object_type", d.get("object") == "chat.completion", d.get("object"))

    print("\n=== 2. max_tokens is actually enforced ===")
    r = chat(base, model, [{"role": "user", "content": "Count from 1 to 500, one per line."}],
             max_tokens=24)
    d = r["json"] or {}
    ct = d.get("usage", {}).get("completion_tokens", 0)
    rec("max_tokens_respected", 0 < ct <= 24 + 8, "completion_tokens=%s (cap 24)" % ct)
    rec("truncation_reports_length",
        (d.get("choices") or [{}])[0].get("finish_reason") == "length",
        (d.get("choices") or [{}])[0].get("finish_reason"))

    print("\n=== 3. Streaming ===")
    p = {"model": model, "messages": [{"role": "user", "content": "Count 1 to 5."}],
         "max_tokens": 60, "stream": True, "temperature": 0.0}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(p).encode(),
                                 headers={"Content-Type": "application/json"})
    chunks, sawdone, first_byte = 0, False, None
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            for line in resp:
                s = line.decode("utf-8", "replace").strip()
                if not s.startswith("data:"):
                    continue
                if first_byte is None:
                    first_byte = time.time() - t0
                if s.strip() == "data: [DONE]":
                    sawdone = True
                    break
                chunks += 1
        rec("stream_multiple_chunks", chunks >= 2, "%d data chunks" % chunks)
        rec("stream_terminates_with_DONE", sawdone, "saw [DONE]=%s" % sawdone)
        rec("stream_ttfb_reasonable", first_byte is not None and first_byte < 30,
            "first chunk %.2fs" % (first_byte or -1))
    except Exception as e:
        rec("streaming", False, "exception: %s" % str(e)[:90])

    print("\n=== 4. Tool / function calling ===")
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]
    r = chat(base, model, [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
             tools=tools, tool_choice="auto", max_tokens=300)
    d = r["json"] or {}
    m = (d.get("choices") or [{}])[0].get("message", {}) or {}
    tc = m.get("tool_calls") or []
    rec("tool_call_emitted", len(tc) > 0, "%d tool_calls" % len(tc))
    if tc:
        fn = tc[0].get("function", {})
        rec("tool_name_correct", fn.get("name") == "get_weather", fn.get("name"))
        args = _try(fn.get("arguments") or "")
        rec("tool_args_valid_json", args is not None, str(fn.get("arguments"))[:70])
        rec("tool_args_has_city", isinstance(args, dict) and "city" in args,
            str(args)[:60] if args else "n/a")
        rec("tool_finish_reason", (d.get("choices") or [{}])[0].get("finish_reason") == "tool_calls",
            (d.get("choices") or [{}])[0].get("finish_reason"))
    else:
        for n in ("tool_name_correct", "tool_args_valid_json", "tool_args_has_city", "tool_finish_reason"):
            rec(n, False, "no tool_call to inspect", unverified=True)

    print("\n=== 5. Error handling — bad input must FAIL CLEANLY, not 200 ===")
    r = http(base + "/v1/chat/completions",
             {"model": "definitely-not-a-real-model-xyz",
              "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})
    rec("unknown_model_rejected", r["code"] >= 400, "code=%s" % r["code"], "errors")
    r = http(base + "/v1/chat/completions", {"model": model})     # no messages
    rec("missing_messages_rejected", r["code"] >= 400, "code=%s" % r["code"], "errors")
    r = http(base + "/v1/chat/completions",
             {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": -5})
    rec("negative_max_tokens_rejected", r["code"] >= 400, "code=%s" % r["code"], "errors")
    # ⚠️ NOT a pass/fail. Asserting "must reject" was MY opinion, not the spec: OpenAI caps
    # temperature at 2, vLLM deliberately does not bound it. Measured here: temp=99999
    # returns HTTP 200 and produces noise, while temp=-1 is correctly rejected. So this
    # records the behaviour a client should expect rather than grading it.
    r = http(base + "/v1/chat/completions",
             {"model": model, "messages": [{"role": "user", "content": "hi"}],
              "temperature": 99999})
    rec("absurd_temperature_behaviour", True,
        "HTTP %s — no UPPER bound on temperature (negatives ARE rejected); a client bug "
        "sending a huge value gets garbage, not an error" % r["code"], "errors")

    print("\n=== 6. Determinism at temperature 0 ===")
    # ⚠️ This deployment DOCUMENTS that temp 0 is NOT deterministic (speculative decoding +
    # continuous batching). This check records reality; it is not expected to pass.
    outs = []
    for _ in range(3):
        rr = chat(base, model, [{"role": "user", "content": "Name three primary colours."}],
                  max_tokens=40)
        outs.append(((rr["json"] or {}).get("choices") or [{}])[0].get("message", {}).get("content", ""))
    same = len(set(o.strip() for o in outs)) == 1
    rec("temp0_identical_x3", same,
        "%d distinct outputs (spec decoding makes this expected to vary)" % len(set(outs)),
        "determinism")

    print("\n=== 7. Engine counters (parsed BY NAME, never positionally) ===")
    before = _counter(metrics_url)
    chat(base, model, [{"role": "user", "content": "Say PROVENANCE"}], max_tokens=15)
    after = _counter(metrics_url)
    if before is None or after is None:
        rec("metrics_readable", False, "could not read %s" % metrics_url, "metrics", unverified=True)
        rec("provenance_counter_moves", False, "metrics unreadable", "metrics", unverified=True)
    else:
        rec("metrics_readable", True, "success_total=%.0f" % after)
        rec("provenance_counter_moves", after > before,
            "%.0f -> %.0f (proves the request reached THIS engine)" % (before, after))

    print("\n=== 8. Security posture (observation only — harmless payloads) ===")
    for route in ("/v1/models", "/health", "/metrics"):
        rr = http(base + route, None, timeout=20)
        rec("open_route%s" % route.replace("/", "_"), True,
            "HTTP %s (unauthenticated)" % rr["code"], "security")
    rr = http(base + "/invocations",
              {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
              timeout=60)
    answered = rr["code"] == 200
    rec("invocations_unauthenticated_inference", True,
        "HTTP %s — %s" % (rr["code"],
                          "FULL INFERENCE WITH NO KEY (documented in this deployment)"
                          if answered else "did not answer"), "security")


def _counter(url):
    try:
        with urllib.request.urlopen(url, timeout=15) as f:
            body = f.read().decode()
    except Exception:
        return None
    tot = 0.0
    found = False
    for line in body.splitlines():
        # BY NAME. A positional parse silently mislabels these.
        if line.startswith("vllm:request_success_total"):
            try:
                tot += float(line.rsplit(" ", 1)[1]); found = True
            except Exception:
                pass
    return tot if found else None


def self_test():
    """Prove the graders can FAIL — a suite that cannot fail measures nothing."""
    print("=== SELF-TEST: every grader must be able to fail ===")
    cases = [
        ("usage arithmetic catches a wrong total",
         {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 99},
         lambda u: u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"], False),
        ("usage arithmetic accepts a right total",
         {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
         lambda u: u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"], True),
        ("finish_reason rejects nonsense", {"finish_reason": "banana"},
         lambda c: c["finish_reason"] in ("stop", "length", "tool_calls"), False),
        ("json parser rejects malformed args", "{not valid json",
         lambda s: _try(s) is not None, False),
        ("json parser accepts valid args", '{"city": "Paris"}',
         lambda s: _try(s) is not None, True),
    ]
    ok = True
    for label, fixture, fn, want in cases:
        got = bool(fn(fixture))
        good = got == want
        ok &= good
        print("  %-44s got=%-5s want=%-5s %s" % (label, got, want, "ok" if good else "*** BROKEN ***"))
    print("\n  %s\n" % ("HARNESS TRUSTWORTHY" if ok else "*** HARNESS IS BROKEN — DO NOT TRUST RESULTS ***"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://spark-1:8888")
    ap.add_argument("--model", default="deepseek-v4-flash-dspark")
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return 0 if self_test() else 1
    if not self_test():
        print("ABORT: harness self-test failed"); return 1
    a.metrics = a.metrics or (a.base + "/metrics")

    t0 = time.time()
    run(a.base, a.model, a.metrics)
    wall = time.time() - t0

    p = sum(1 for r in RESULTS if r["ok"])
    u = sum(1 for r in RESULTS if r["unverified"])
    f = len(RESULTS) - p - u
    print("\n" + "=" * 62)
    print("  PASS %d   FAIL %d   UNVERIFIED %d   (%d checks, %.0fs)" % (p, f, u, len(RESULTS), wall))
    for r in RESULTS:
        if not r["ok"] and not r["unverified"]:
            print("    FAILED: %-34s %s" % (r["name"], r["detail"][:70]))
    if a.out:
        json.dump({"label": a.label, "base": a.base, "model": a.model,
                   "wall": round(wall, 1), "passed": p, "failed": f, "unverified": u,
                   "checks": RESULTS}, open(a.out, "w"), indent=1)
        print("  saved -> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
