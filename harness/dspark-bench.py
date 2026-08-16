#!/usr/bin/env python3
"""Consolidated quality + agentic benchmark for the DGX Spark serving endpoint.

⚠️ LIVES IN spark-deploy/tools/ ON PURPOSE. The first version of this harness was
built in the session scratchpad under /tmp and was deleted by a tmp clean, taking the
whole suite with it mid-upgrade. Benchmarks are regression baselines — they are
project artifacts, not scratch.

DESIGN RULE: every task is machine-graded — exact string, parsed number, valid JSON,
or CODE THAT IS EXECUTED against assertions. No human judgement, so runs compare.

⚠️ Reads the reasoning channel, under BOTH field names. This build returns it as
`reasoning`; other builds return `reasoning_content`. A grader that reads only
`content` scores a thinking model as empty, and one that reads only the wrong
channel name records "no reasoning" forever without ever erroring.

⚠️ Temperature 0 is NOT deterministic here (speculative decoding + continuous
batching). Single-trial deltas are noise. Use --repeat for anything you intend to
act on.

Usage:
    dspark-bench.py --self-test            # prove the graders can FAIL
    dspark-bench.py --label baseline --out base.json
    dspark-bench.py --label 0731 --reasoning high --out new.json
    dspark-bench.py --compare base.json new.json
"""
import argparse, json, os, re, subprocess, sys, tempfile, time, urllib.request

URL_DEFAULT = "http://spark-1:8888/v1/chat/completions"
MODEL_DEFAULT = "deepseek-v4-flash-dspark"

# Bearer token for endpoints that require one (SGLang runs with --api-key; the vLLM
# lane does not). Set via --api-key-file or DSPARK_BENCH_API_KEY — never as a CLI
# argument, which would land in shell history and in every `ps` listing on the box.
# ⚠️ NEVER print this value. Failures report only whether a key was SENT.
API_KEY = None


def _load_api_key(path):
    """Read a bearer token from a file. Returns None if no path given."""
    if not path:
        return os.environ.get("DSPARK_BENCH_API_KEY") or None
    with open(os.path.expanduser(path)) as f:
        k = f.read().strip()
    if not k:
        raise SystemExit("ABORT: --api-key-file %s is empty" % path)
    return k


# ─────────────────────────── transport ───────────────────────────
def ask(prompt, url, model, max_tokens=3000, temperature=0.0, reasoning=None,
        tools=None, timeout=600):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "stream": False}
    if reasoning:
        body["reasoning_effort"] = reasoning
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = "Bearer " + API_KEY
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers=headers)
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=timeout))
    except Exception as e:
        return {"ok": False, "err": str(e)[:120], "content": "", "calls": [],
                "secs": time.time() - t0, "ctok": 0, "finish": "error"}
    ch = d["choices"][0]; m = ch["message"]
    calls = []
    for tc in (m.get("tool_calls") or []):
        fn = tc.get("function", {})
        try: args = json.loads(fn.get("arguments") or "{}")
        except Exception: args = {}
        calls.append({"name": fn.get("name"), "args": args})
    return {"ok": True, "err": None,
            "content": (m.get("content") or "").strip(),
            # ⚠️ READ BOTH NAMES. This build emits `reasoning`; other vLLM/SGLang builds
            # emit `reasoning_content`. Reading one name returns "" forever on the other,
            # silently — a thinking model then records as having thought nothing.
            "reasoning": ((m.get("reasoning") or m.get("reasoning_content") or "").strip()),
            "calls": calls, "finish": ch.get("finish_reason"),
            "ctok": d.get("usage", {}).get("completion_tokens", 0),
            "secs": time.time() - t0}


# ─────────────────────────── graders ───────────────────────────
def has_number(txt, want):
    """Unit-aware, tolerance-aware numeric grader.

    Two bugs were fixed here on 2026-08-15, both of which scored CORRECT answers as
    failures and nearly became published findings:
      1. expected 0.05, model said "5 cents" -> naive parse read 5. Now unit-aware.
      2. expected 5.33, model said "5.333" (more correct) -> exact match failed 3/3
         at every reasoning level. Now 0.5% tolerance for NON-INTEGER expectations.
    Integers stay EXACT so 392-vs-391 still fails; a blanket tolerance would make the
    grader unable to fail at all, which is worse than the bug it fixes.
    """
    clean = txt.replace(",", "")
    nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", clean)]

    def match(n, w):
        if float(w).is_integer():
            return abs(n - w) < 1e-9
        return abs(n - w) <= abs(w) * 0.005

    if any(match(n, want) for n in nums):
        return True
    if re.search(r"\bcents?\b", clean, re.I) and any(match(n / 100.0, want) for n in nums):
        return True
    return False


def extract_code(txt):
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", txt, re.S)
    return m[0] if m else txt


def run_code(code, asserts, timeout=25):
    src = code + "\n\n" + asserts + "\nprint('PASS')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        return "PASS" in p.stdout, (p.stderr or p.stdout)[-140:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)[:100]
    finally:
        try: os.unlink(path)
        except Exception: pass


def try_json(t):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip()).strip()
    try: return json.loads(t)
    except Exception: return None


# ─────────────────────────── tasks ───────────────────────────
NUMERIC = [
    # (id, tier, prompt, expected)
    ("bat_ball",  1, "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Answer with just the amount.", 0.05),
    ("widgets",   1, "If 5 machines take 5 minutes to make 5 widgets, how long would 100 machines take to make 100 widgets? Answer with just the number of minutes.", 5),
    ("lilypad",   1, "A lily patch doubles in size every day. It covers the whole lake on day 48. On what day was it half covered? Answer with just the day number.", 47),
    ("sheep",     1, "A farmer has 17 sheep. All but 9 die. How many sheep are left? Answer with just the number.", 9),
    ("socks",     1, "A drawer has 10 black and 10 white socks, mixed, in the dark. How many socks must you take to GUARANTEE a matching pair? Answer with just the number.", 3),
    ("mult",      1, "What is 17 * 23? Reply with just the number.", 391),
    ("order",     1, "Compute 144 / 12 + 7 * 3. Reply with just the number.", 33),
    ("pow",       1, "What is 2 to the power of 10? Reply with just the number.", 1024),
    ("pct",       1, "What is 15% of 240? Reply with just the number.", 36),
    ("bird",      2, "Two trains are 120 miles apart on the same track, heading toward each other at 60 mph and 40 mph. A bird flies at 100 mph back and forth between them until they collide. How far does the bird travel in total? Answer with just the number of miles.", 120),
    ("clock",     2, "In a 24-hour period, how many times do the hour and minute hands of an analog clock overlap exactly? Answer with just the number.", 22),
    ("painters",  2, "If 4 painters can paint a house in 8 hours, how many hours would 6 painters take, working at the same rate? Answer with just the number of hours, as a decimal if needed.", 5.33),
    ("handshake", 2, "At a party, every person shakes hands with every other person exactly once. There were 66 handshakes total. How many people were at the party? Answer with just the number.", 12),
    ("wine",      2, "You have 8 bottles, exactly one is poisoned. A taster who drinks poison dies after exactly 1 hour. You have 1 hour. What is the MINIMUM number of tasters needed to guarantee identifying the poisoned bottle? Answer with just the number.", 3),
    ("ages",      2, "A father is 4 times as old as his son. In 20 years he will be twice as old as his son. How old is the father now? Answer with just the number.", 40),
    ("lockers",   3, "100 lockers all closed. Student 1 toggles every locker, student 2 every 2nd, student 3 every 3rd, ... student 100 every 100th. How many lockers are open at the end? Answer with just the number.", 10),
    ("eggs",      3, "You have 2 identical eggs and a 100-floor building. Find the highest floor an egg survives, minimising the WORST-CASE number of drops. What is that minimum worst-case number of drops? Answer with just the number.", 14),
    ("horses",    3, "25 horses, no timer, a track that races 5 at a time. What is the MINIMUM number of races needed to find the fastest 3? Answer with just the number.", 7),
    ("josephus",  3, "41 people stand in a circle. Counting around, every 3rd person is eliminated, continuing until one remains. Counting starts at person 1, so person 3 is eliminated first. What is the position number of the survivor? Answer with just the number.", 31),
    ("sum7",      3, "What is the sum of all 3-digit positive integers divisible by 7? Answer with just the number.", 70336),
    ("weigh",     3, "You have 12 balls, identical except one is a different weight (unknown heavier or lighter). Using a balance scale, what is the MINIMUM number of weighings to guarantee finding it AND whether it is heavy or light? Answer with just the number.", 3),
]

FORMAT = [
    ("one_word",   "Reply with exactly one word: BLUE. No punctuation, no explanation.",
     lambda t: t.strip().strip(".").upper() == "BLUE"),
    ("three_csv",  "List exactly three fruits, comma-separated, lowercase, nothing else.",
     lambda t: len([x for x in t.strip().rstrip(".").split(",") if x.strip()]) == 3),
    ("json_only",  'Output ONLY valid JSON, no markdown fence, with exactly these keys: name (string "test"), count (number 42), ok (boolean true).',
     lambda t: (lambda o: isinstance(o, dict) and o.get("name") == "test"
                and o.get("count") == 42 and o.get("ok") is True)(try_json(t))),
    ("no_yapping", "Reply with only the capital of France. One word.",
     lambda t: t.strip().rstrip(".").lower() == "paris"),
]

CODE = [
    ("reverse",  "Write a Python function `rev(s)` returning the reverse of string s. Code only, in a python fence.",
     "assert rev('hello')=='olleh'\nassert rev('')==''\nassert rev('a')=='a'"),
    ("fizzbuzz", "Write a Python function `fb(n)` returning a list of length n: for i from 1..n give 'Fizz' if divisible by 3, 'Buzz' if by 5, 'FizzBuzz' if both, else str(i). Code only, in a python fence.",
     "r=fb(15)\nassert len(r)==15 and r[2]=='Fizz' and r[4]=='Buzz' and r[14]=='FizzBuzz' and r[0]=='1'"),
    ("bsearch",  "Write a Python function `bs(arr, t)` doing binary search on a sorted list, returning the index of t or -1. Code only, in a python fence.",
     "assert bs([1,3,5,7,9],7)==3 and bs([1,3,5,7,9],4)==-1 and bs([],1)==-1 and bs([2],2)==0"),
    ("parens",   "Write a Python function `bal(s)` returning True if brackets ()[]{} in s are balanced and correctly nested, else False. Code only, in a python fence.",
     "assert bal('([]{})') and not bal('([)]') and bal('') and not bal('(')"),
    ("lru",      "Write a Python class `LRU` with __init__(self, cap), get(key) returning -1 if absent, and put(key,value), evicting least-recently-used past cap. Code only, in a python fence.",
     "c=LRU(2)\nc.put(1,1); c.put(2,2)\nassert c.get(1)==1\nc.put(3,3)\nassert c.get(2)==-1 and c.get(3)==3"),
    ("dedupe",   "Write a Python function `dedupe(xs)` returning xs with duplicates removed, preserving first-occurrence order. Code only, in a python fence.",
     "assert dedupe([3,1,3,2,1])==[3,1,2] and dedupe([])==[] and dedupe([1,1,1])==[1]"),
    ("merge_iv", "Write a Python function `merge(intervals)` merging overlapping [start,end] pairs, returning them sorted by start. Handle unsorted input and touching intervals like [1,4],[4,5] (which merge). Code only, in a python fence.",
     "assert merge([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\nassert merge([[1,4],[4,5]])==[[1,5]]\nassert merge([])==[]\nassert merge([[5,6],[1,2]])==[[1,2],[5,6]]"),
    ("edit_dist","Write a Python function `ed(a,b)` returning the Levenshtein edit distance between strings a and b. Code only, in a python fence.",
     "assert ed('kitten','sitting')==3 and ed('','abc')==3 and ed('abc','abc')==0 and ed('flaw','lawn')==2"),
    ("valid_bst","Write a Python function `is_bst(node)` validating a binary search tree. Nodes are dicts {'val':v,'left':n_or_None,'right':n_or_None}. Must enforce the FULL range constraint, not just parent-child. Code only, in a python fence.",
     "t={'val':5,'left':{'val':3,'left':None,'right':None},'right':{'val':8,'left':None,'right':None}}\nassert is_bst(t)\nbad={'val':5,'left':{'val':3,'left':None,'right':{'val':6,'left':None,'right':None}},'right':None}\nassert not is_bst(bad)\nassert is_bst(None)"),
    ("lcs",      "Write a Python function `lcs(a,b)` returning the LENGTH of the longest common subsequence of strings a and b. Code only, in a python fence.",
     "assert lcs('abcde','ace')==3 and lcs('abc','abc')==3 and lcs('abc','def')==0 and lcs('','a')==0"),
    ("spiral",   "Write a Python function `spiral(m)` returning elements of a 2D list m in clockwise spiral order from top-left. Handle non-square and empty. Code only, in a python fence.",
     "assert spiral([[1,2,3],[4,5,6],[7,8,9]])==[1,2,3,6,9,8,7,4,5]\nassert spiral([[1,2],[3,4],[5,6]])==[1,2,4,6,5,3]\nassert spiral([])==[] and spiral([[1]])==[1]"),
    ("debug",    "This function should return the SECOND largest DISTINCT value in a list, or None if there isn't one. It has bugs. Return a corrected version, same name `second(xs)`. Code only, in a python fence.\n\n```python\ndef second(xs):\n    xs.sort()\n    return xs[-2]\n```",
     "assert second([1,5,3])==3\nassert second([5,5,5]) is None\nassert second([1]) is None\nassert second([]) is None\nassert second([2,2,9])==2\norig=[3,1,2]\nsecond(orig)\nassert orig==[3,1,2], 'must not mutate caller list'"),
]

TOOLS = [
    {"type":"function","function":{"name":"read_file","description":"Read the contents of a file at a path.",
     "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
    {"type":"function","function":{"name":"write_file","description":"Write content to a file at a path.",
     "parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
    {"type":"function","function":{"name":"run_tests","description":"Run the test suite and return results.",
     "parameters":{"type":"object","properties":{"pattern":{"type":"string"}},"required":[]}}},
    {"type":"function","function":{"name":"search_code","description":"Search the repository for a regex pattern.",
     "parameters":{"type":"object","properties":{"pattern":{"type":"string"},"max_results":{"type":"integer"}},"required":["pattern"]}}},
]

AGENTIC = [
    ("sel_read",   "Show me what's in src/main.py",
     lambda c: len(c)==1 and c[0]["name"]=="read_file" and c[0]["args"].get("path")=="src/main.py"),
    ("sel_search", "Find every place in the repo where we call `deprecated_api`.",
     lambda c: len(c)==1 and c[0]["name"]=="search_code" and "deprecated_api" in str(c[0]["args"].get("pattern",""))),
    ("sel_tests",  "Run the test suite.",
     lambda c: len(c)==1 and c[0]["name"]=="run_tests"),
    ("sel_write",  "Create a file called README.md containing exactly: # Hello",
     lambda c: len(c)==1 and c[0]["name"]=="write_file" and c[0]["args"].get("path")=="README.md"),
    ("sel_typed",  "Search for the pattern TODO and cap it at 5 results.",
     lambda c: len(c)==1 and c[0]["name"]=="search_code" and c[0]["args"].get("max_results")==5),
    ("restraint1", "Hi there!",              lambda c: len(c)==0),
    ("restraint2", "In one sentence, what does the acronym API stand for?", lambda c: len(c)==0),
]


# ─────────────────────────── self-test ───────────────────────────
def self_test():
    """A harness is unproven until it has been SEEN TO FAIL."""
    cases = [
        ("has_number('10 cents',0.05)", has_number("10 cents", 0.05), False),
        ("has_number('5 cents',0.05)",  has_number("5 cents", 0.05), True),
        ("has_number('5.333',5.33)",    has_number("5.333", 5.33), True),
        ("has_number('392',391)",       has_number("392", 391), False),
        ("has_number('70337',70336)",   has_number("70337", 70336), False),
        ("run_code(BROKEN)",  run_code("def rev(s):\n    return s", "assert rev('ab')=='ba'")[0], False),
        ("run_code(CORRECT)", run_code("def rev(s):\n    return s[::-1]", "assert rev('ab')=='ba'")[0], True),
        ("run_code(SYNTAX)",  run_code("def rev(s) return s", "assert rev('a')=='a'")[0], False),
        ("debug asserts reject original buggy impl",
         run_code("def second(xs):\n    xs.sort()\n    return xs[-2]",
                  [a for n, _, a in CODE if n == "debug"][0])[0], False),
    ]
    d = {n: f for n, _, f in FORMAT}
    cases += [("one_word('BLUE')", d["one_word"]("BLUE"), True),
              ("one_word('blue.')", d["one_word"]("The answer is blue."), False),
              ("json_only(count=41)", d["json_only"]('{"name":"test","count":41,"ok":true}'), False)]
    bad = 0
    print("=== SELF-TEST: the harness must be able to FAIL ===")
    for name, got, want in cases:
        ok = (bool(got) == want); bad += not ok
        print("  %-46s got=%-5s want=%-5s %s" % (name, bool(got), want, "ok" if ok else "*** HARNESS BUG ***"))
    print("\n  %s\n" % ("HARNESS TRUSTWORTHY" if bad == 0 else "%d BUGS - do not trust results" % bad))
    return bad == 0


# ─────────────────────────── run ───────────────────────────
def run(url, model, reasoning, max_tokens, repeat, verbose=True):
    rows = []

    def rec(cat, name, passed, r, detail=""):
        rows.append({"cat": cat, "name": name, "pass": bool(passed),
                     "secs": round(r["secs"], 1), "ctok": r["ctok"], "finish": r["finish"]})
        if verbose:
            print("  %-7s %-11s %s %6.1fs %5dtok %s" % (
                cat, name, "PASS" if passed else "FAIL", r["secs"], r["ctok"], detail[:40]), flush=True)

    for name, tier, prompt, want in NUMERIC:
        best = False; last = None
        for _ in range(repeat):
            r = ask(prompt, url, model, max_tokens, 0.0, reasoning); last = r
            if r["ok"] and has_number(r["content"], want): best = True; break
        rec("num%d" % tier, name, best, last, "" if best else repr(last["content"][:30]))

    for name, prompt, check in FORMAT:
        r = ask(prompt, url, model, max_tokens, 0.0, reasoning)
        try: ok = r["ok"] and bool(check(r["content"]))
        except Exception: ok = False
        rec("format", name, ok, r)

    for name, prompt, asserts in CODE:
        r = ask(prompt, url, model, max_tokens, 0.0, reasoning)
        if not r["ok"]:
            rec("code", name, False, r, r["err"]); continue
        ok, err = run_code(extract_code(r["content"]), asserts)
        rec("code", name, ok, r, "" if ok else err.replace("\n", " ")[:38])

    for name, prompt, check in AGENTIC:
        r = ask(prompt, url, model, max_tokens, 0.0, reasoning, tools=TOOLS)
        try: ok = r["ok"] and bool(check(r["calls"]))
        except Exception: ok = False
        rec("agent", name, ok, r, ",".join(c["name"] for c in r["calls"]))

    p = sum(1 for x in rows if x["pass"])
    by = {}
    for x in rows:
        d = by.setdefault(x["cat"], [0, 0]); d[1] += 1; d[0] += x["pass"]
    return {"rows": rows, "passed": p, "total": len(rows),
            "pct": round(100.0 * p / len(rows), 1), "by_cat": by}


def compare(a_path, b_path):
    a = json.load(open(a_path)); b = json.load(open(b_path))
    am = {r["name"]: r["pass"] for r in a["rows"]}
    bm = {r["name"]: r["pass"] for r in b["rows"]}
    print("=== %s  ->  %s ===" % (a.get("label", a_path), b.get("label", b_path)))
    print("  %s: %d/%d (%.1f%%)   ->   %s: %d/%d (%.1f%%)" % (
        a.get("label", "A"), a["passed"], a["total"], a["pct"],
        b.get("label", "B"), b["passed"], b["total"], b["pct"]))
    reg = [k for k in am if am[k] and not bm.get(k, False)]
    fix = [k for k in am if not am[k] and bm.get(k, False)]
    print("\n  REGRESSIONS (was pass, now fail): %s" % (", ".join(reg) or "none"))
    print("  FIXED       (was fail, now pass): %s" % (", ".join(fix) or "none"))
    print("\n  ⚠️ temp 0 is NOT deterministic here — treat single-task flips as noise")
    print("     unless they reproduce across --repeat runs.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL_DEFAULT)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--api-key-file", default=None,
                    help="path to a file holding a bearer token (SGLang needs one; "
                         "vLLM does not). Falls back to $DSPARK_BENCH_API_KEY. "
                         "Never pass the key itself on the command line.")
    ap.add_argument("--reasoning", default="high")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--repeat", type=int, default=1, help="retries per numeric task before scoring FAIL")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "NEW"))
    a = ap.parse_args()

    global API_KEY
    API_KEY = _load_api_key(a.api_key_file)

    if a.self_test:
        return 0 if self_test() else 1
    if a.compare:
        compare(*a.compare); return 0

    if not self_test():
        print("ABORT: harness self-test failed"); return 1

    print("=== %s | model=%s reasoning=%s max_tokens=%d repeat=%d ===" % (
        a.label, a.model, a.reasoning, a.max_tokens, a.repeat), flush=True)
    t0 = time.time()
    res = run(a.url, a.model, a.reasoning, a.max_tokens, a.repeat)
    res.update({"label": a.label, "model": a.model, "reasoning": a.reasoning,
                "max_tokens": a.max_tokens, "repeat": a.repeat,
                "wall": round(time.time() - t0, 1)})
    print("\n  " + "  ".join("%s %d/%d" % (c, v[0], v[1]) for c, v in sorted(res["by_cat"].items())))
    print("  TOTAL %d/%d = %.1f%%   wall %.1fs" % (res["passed"], res["total"], res["pct"], res["wall"]))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print("  saved -> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
