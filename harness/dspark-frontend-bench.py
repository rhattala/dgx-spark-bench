#!/usr/bin/env python3
"""Front-end coding benchmark — three tiers, visually reviewable output.

WHY THIS EXISTS: the 44-task suite is saturated and text-only. It cannot tell you whether
the model writes GOOD FRONT-END CODE — that needs (a) something you can look at, and
(b) checks that a plausible-looking page actually fails.

Each task produces ONE self-contained .html file. Two kinds of grading:

  MACHINE  — objective, pass/fail, no judgement. Renders? Valid JS? Responsive meta?
             Semantic landmarks? Keyboard-reachable? No external requests? etc.
  VISUAL   — you and I look at it. Machine checks cannot see "ugly" or "unusable".

⚠️ Machine checks here are NECESSARY, NOT SUFFICIENT. A page can pass every one and
still look terrible. That is exactly why the artifacts get rendered for review.
"""
import argparse, json, os, re, subprocess, sys, tempfile, threading, time, urllib.request

URL_DEFAULT = "http://spark-1:8888/v1/chat/completions"
MODEL_DEFAULT = "deepseek-v4-flash-dspark"

# Set from CLI in main(). Kept module-level so ask() needs no extra threading.
URL = URL_DEFAULT
MODEL = MODEL_DEFAULT
# Bearer token for endpoints that require one (SGLang runs with --api-key).
# Supplied by PATH or env only — never as a CLI argument, which would land in shell
# history and every `ps` listing. ⚠️ NEVER print this value.
API_KEY = None
# Qwen-style thinking off-switch (see ask()). Set by --no-think.
NO_THINK = False

COMMON = (
    "Output ONE complete self-contained HTML file. All CSS in a <style> tag and all JS in "
    "a <script> tag — no external CDNs, no external fonts, no frameworks, no build step. "
    "It must work opened directly in a browser. Output ONLY the HTML in a single ```html "
    "code fence, no commentary."
)

TASKS = [
    ("simple", "pricing-table",
     "Build a responsive pricing table with three tiers: Starter $9, Pro $29 (visually "
     "highlighted as the recommended plan), and Enterprise $99. Each tier lists 4 features "
     "and has a call-to-action button. It must look clean and modern, work down to a 320px "
     "wide phone, and support both light and dark colour schemes via prefers-color-scheme. "
     + COMMON),

    ("medium", "task-board",
     "Build an interactive task board. Requirements: add a task via an input; each task can "
     "be marked complete (with a visible completed style) and deleted; filter buttons for "
     "All / Active / Completed; a live count of remaining tasks; tasks persist across page "
     "reloads using localStorage; the empty state shows a friendly message. Must be fully "
     "keyboard operable (Enter adds a task, all controls reachable by Tab with visible focus "
     "styles) and responsive to 320px. "
     + COMMON),

    ("hard", "sales-dashboard",
     "Build a sales dashboard with NO charting libraries — draw the chart yourself with "
     "inline SVG. Requirements: (1) four KPI stat tiles at the top; (2) a bar chart of "
     "monthly revenue for 12 months, drawn as SVG with axis labels and hover tooltips; "
     "(3) a table of 8 sales records that can be sorted by clicking any column header "
     "(ascending/descending, with an indicator) and filtered by a search input; (4) a "
     "light/dark theme toggle that persists in localStorage. Seed it with realistic sample "
     "data defined in JS. Must be responsive to 320px, and the table must scroll "
     "horizontally on small screens rather than breaking the layout. "
     + COMMON),
("extra-hard", "kanban-board",
     "Build a kanban board with NO libraries and NO frameworks. Requirements: (1) three "
     "columns - To Do, In Progress, Done - each showing a live card count; (2) cards can be "
     "moved between columns by DRAG AND DROP using the native HTML5 drag events, with a "
     "visible drop indicator on the target column; (3) every drag-and-drop action must ALSO "
     "be achievable from the keyboard alone (select a card, then move it left/right with "
     "arrow keys or buttons) with visible focus styles; (4) add a card via an input, edit a "
     "card title by double-clicking it, delete a card; (5) an UNDO button that reverts the "
     "last action (move, add, edit or delete) - keep a history stack; (6) a search box that "
     "filters cards live across all columns; (7) the whole board persists in localStorage "
     "and restores on reload; (8) a light/dark theme toggle that also persists. Seed it with "
     "6 realistic sample cards. Must be usable down to 320px, where the columns stack "
     "vertically. "
     + COMMON),
]


def ask(prompt, reasoning="high", max_tokens=12000, timeout=1800):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.2, "stream": False}
    # ⚠️ Qwen3.x does NOT honour `reasoning_effort` as an off-switch — its thinking is
    # gated by the chat template's `enable_thinking`. Measured 2026-08-15 on Qwen3.8-27B:
    # with reasoning_effort high OR omitted, long front-end prompts burned the ENTIRE
    # 12k budget in the reasoning channel and returned ZERO bytes of content
    # (finish=length). With enable_thinking=False, reasoning_tokens drops to 0 and real
    # HTML appears. Sending the OpenAI-style knob to a Qwen model looks like it works and
    # silently does nothing — same shape as this project's other label-vs-artifact traps.
    if NO_THINK:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    # Only send reasoning_effort when set. An empty string is NOT "off" — vLLM 400s on it.
    # Measured 2026-08-15: on long code generation, omitting it entirely finishes in ~3.9k
    # tokens / 56 s, while effort=high burns the full 12k budget and truncates.
    if reasoning:
        body["reasoning_effort"] = reasoning
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = "Bearer " + API_KEY
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers=headers)
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    ch = d["choices"][0]
    return {"content": (ch["message"].get("content") or ""),
            "finish": ch.get("finish_reason"),
            "ctok": d.get("usage", {}).get("completion_tokens", 0),
            "secs": time.time() - t0}


def extract_html(txt):
    m = re.findall(r"```(?:html)?\s*\n(.*?)```", txt, re.S)
    if m:
        return max(m, key=len).strip()
    i = txt.lower().find("<!doctype")
    if i == -1:
        i = txt.lower().find("<html")
    return txt[i:].strip() if i != -1 else txt.strip()


# ─────────────── machine checks (objective; each must be able to FAIL) ───────────────
def check(html, tier):
    h = html.lower()
    c = {}
    c["has_doctype"]     = h.lstrip().startswith("<!doctype html")
    c["has_title"]       = "<title>" in h
    c["viewport_meta"]   = bool(re.search(r'<meta[^>]+name=["\']viewport["\']', h))
    c["no_external_req"] = not bool(re.search(r'(src|href)=["\']https?://', h))
    c["semantic_html"]   = sum(t in h for t in ("<main", "<header", "<section", "<nav", "<footer", "<table")) >= 2
    c["responsive_css"]  = "@media" in h or "clamp(" in h or "minmax(" in h
    c["dark_mode"]       = "prefers-color-scheme" in h or "data-theme" in h
    c["focus_styles"]    = ":focus" in h
    c["no_inline_onclick"] = not bool(re.search(r'\son(click|input|change|submit)\s*=', h))
    # balanced tags — crude but catches truncation
    for tag in ("html", "body", "style"):
        c["closes_%s" % tag] = ("</%s>" % tag) in h
    if tier in ("medium", "hard"):
        c["has_js"]        = "<script" in h and len(re.findall(r"function|=>|addeventlistener", h)) >= 3
        c["localstorage"]  = "localstorage" in h
        c["aria_or_label"] = "aria-" in h or "<label" in h
    if tier == "hard":
        c["inline_svg"]    = "<svg" in h and ("<rect" in h or "<path" in h or "<line" in h)
        c["sortable"]      = "sort" in h
        c["theme_toggle"]  = "theme" in h
    if tier == "extra-hard":
        c["dragdrop"]      = "dragstart" in h and ("drop" in h and "dragover" in h)
        c["keyboard_move"] = "keydown" in h or "arrowright" in h or "arrowleft" in h
        c["undo"]          = "undo" in h
        c["search_filter"] = "input" in h and ("filter" in h or "search" in h)
        c["theme_toggle"]  = "theme" in h
        c["localstorage2"] = h.count("localstorage") >= 2
    return c


def js_syntax_ok(html):
    """Extract <script> bodies and syntax-check with node if available."""
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S | re.I)
    body = "\n".join(s for s in scripts if s.strip())
    if not body.strip():
        return None, "no script"
    if not shutil_which("node"):
        return None, "node unavailable"
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(body); p = f.name
    try:
        r = subprocess.run(["node", "--check", p], capture_output=True, text=True, timeout=20)
        return r.returncode == 0, (r.stderr.strip().splitlines() or [""])[0][:90]
    except Exception as e:
        return None, str(e)[:60]
    finally:
        try: os.unlink(p)
        except Exception: pass


def shutil_which(x):
    from shutil import which
    return which(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/Users/randomllama/spark-deploy/evidence/frontend-bench")
    ap.add_argument("--url", default=URL_DEFAULT)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--api-key-file", default=None,
                    help="path to a file holding a bearer token (SGLang needs one; "
                         "vLLM does not). Falls back to $DSPARK_BENCH_API_KEY. "
                         "Never pass the key itself on the command line.")
    ap.add_argument("--reasoning", default="high")
    ap.add_argument("--no-think", action="store_true",
                    help="send chat_template_kwargs.enable_thinking=false (Qwen3.x; "
                         "reasoning_effort does NOT disable Qwen thinking)")
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--label", default="0731")
    ap.add_argument("--parallel", action="store_true",
                    help="fire ALL tiers concurrently and report total wall clock. This is "
                         "the only way a concurrency advantage shows up: a sequential run "
                         "measures one stream and cannot see it. Simulates multi-agent use.")
    a = ap.parse_args()

    global URL, MODEL, API_KEY, NO_THINK
    URL, MODEL = a.url, a.model
    NO_THINK = a.no_think
    if a.api_key_file:
        with open(os.path.expanduser(a.api_key_file)) as f:
            API_KEY = f.read().strip()
        if not API_KEY:
            raise SystemExit("ABORT: --api-key-file %s is empty" % a.api_key_file)
    else:
        API_KEY = os.environ.get("DSPARK_BENCH_API_KEY") or None

    os.makedirs(a.outdir, exist_ok=True)

    results = []

    if a.parallel:
        # Multi-agent simulation: every tier in flight at once. Total wall clock is the
        # number that matters here — NOT per-task time, which is expected to be worse.
        print("=== PARALLEL: all %d tiers concurrently (multi-agent simulation) ===" % len(TASKS),
              flush=True)
        lock = threading.Lock()
        bucket = {}

        def one(tier, name, prompt):
            try:
                r = ask(prompt, a.reasoning, max_tokens=a.max_tokens)
            except Exception as e:
                with lock:
                    bucket[name] = {"tier": tier, "name": name, "error": str(e)[:150]}
                return
            with lock:
                bucket[name] = {"r": r, "tier": tier, "name": name}

        ths = [threading.Thread(target=one, args=t) for t in TASKS]
        t0 = time.time()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.time() - t0

        for tier, name, _ in TASKS:
            b = bucket.get(name, {})
            if "error" in b:
                print("  %-11s %-16s REQUEST FAILED" % (tier, name), flush=True)
                results.append({"tier": tier, "name": name, "error": b["error"]}); continue
            r = b["r"]
            html = extract_html(r["content"])
            path = os.path.join(a.outdir, "%s-%s.html" % (tier, name))
            open(path, "w", encoding="utf-8").write(html)
            c = check(html, tier); ok, jserr = js_syntax_ok(html)
            passed = sum(1 for v in c.values() if v)
            print("  %-11s %-16s checks %2d/%-2d js=%-7s %6dtok %5.0fs" % (
                tier, name, passed, len(c),
                {True: "OK", False: "INVALID", None: "n/a"}[ok], r["ctok"], r["secs"]), flush=True)
            results.append({"tier": tier, "name": name, "path": path, "bytes": len(html),
                            "ctok": r["ctok"], "secs": round(r["secs"], 1), "finish": r["finish"],
                            "checks": c, "checks_passed": passed, "checks_total": len(c),
                            "js_syntax": ok, "js_err": jserr})
        tot = sum(r.get("ctok", 0) for r in results)
        print("\n  TOTAL WALL CLOCK: %.0fs   aggregate %.1f tok/s   (%d tokens across %d streams)"
              % (wall, tot / wall if wall else 0, tot, len(TASKS)), flush=True)
        out = os.path.join(a.outdir, "results-%s.json" % a.label)
        json.dump({"label": a.label, "reasoning": a.reasoning, "url": URL, "model": MODEL,
                   "auth": bool(API_KEY), "no_think": NO_THINK, "max_tokens": a.max_tokens,
                   "mode": "parallel", "wall": round(wall, 1),
                   "aggregate_tok_s": round(tot / wall, 1) if wall else 0,
                   "results": results}, open(out, "w"), indent=1)
        print("  saved -> %s" % out)
        return

    for tier, name, prompt in TASKS:
        print("=== %-6s %s ===" % (tier, name), flush=True)
        try:
            r = ask(prompt, a.reasoning, max_tokens=a.max_tokens)
        except Exception as e:
            print("  REQUEST FAILED: %s" % str(e)[:100], flush=True)
            results.append({"tier": tier, "name": name, "error": str(e)[:150]}); continue
        html = extract_html(r["content"])
        path = os.path.join(a.outdir, "%s-%s.html" % (tier, name))
        open(path, "w", encoding="utf-8").write(html)

        c = check(html, tier)
        ok, jserr = js_syntax_ok(html)
        passed = sum(1 for v in c.values() if v)
        print("  %d tok in %.0fs (finish=%s), %d bytes" % (r["ctok"], r["secs"], r["finish"], len(html)), flush=True)
        print("  machine checks: %d/%d" % (passed, len(c)), flush=True)
        for k, v in c.items():
            if not v:
                print("    FAIL %s" % k, flush=True)
        print("  js syntax: %s%s" % ({True: "OK", False: "INVALID", None: "n/a"}[ok],
                                     "" if ok is not False else " — " + jserr), flush=True)
        print("  -> %s\n" % path, flush=True)
        results.append({"tier": tier, "name": name, "path": path, "bytes": len(html),
                        "ctok": r["ctok"], "secs": round(r["secs"], 1), "finish": r["finish"],
                        "checks": c, "checks_passed": passed, "checks_total": len(c),
                        "js_syntax": ok, "js_err": jserr})

    out = os.path.join(a.outdir, "results-%s.json" % a.label)
    # Record url+model, not just the label: a label naming our model has been wrong
    # three times in this project (Cursor->Fireworks, Hermes->Ollama, MoA->Codex).
    json.dump({"label": a.label, "reasoning": a.reasoning,
               "url": URL, "model": MODEL, "auth": bool(API_KEY),
               "no_think": NO_THINK, "max_tokens": a.max_tokens,
               "results": results},
              open(out, "w"), indent=1)
    print("=" * 60)
    for r in results:
        if "error" in r:
            print("  %-6s %-16s REQUEST FAILED" % (r["tier"], r["name"])); continue
        print("  %-6s %-16s checks %2d/%-2d  js=%-7s %5dtok %5.0fs" % (
            r["tier"], r["name"], r["checks_passed"], r["checks_total"],
            {True: "OK", False: "INVALID", None: "n/a"}[r["js_syntax"]], r["ctok"], r["secs"]))
    print("\n  ⚠️ Machine checks are NECESSARY, NOT SUFFICIENT — a page can pass all of")
    print("     them and still look terrible. Render them and look.")
    print("  saved -> %s" % out)


if __name__ == "__main__":
    main()
