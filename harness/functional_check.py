#!/usr/bin/env python3
"""Functional checks for generated web pages — does the page WORK, not does it CONTAIN.

WHY THIS EXISTS. The static check suite in `dspark-frontend-bench.py` is precise and wrong.
Measured 2026-08-16: it has near-zero run-to-run variance (3 of 4 tiers at ZERO spread), and
it ranked a kanban board with **no way to add a card** ABOVE one that implemented adding
three times over. Both scored 17-18/18. The prompt explicitly required the feature.

**A repeatable instrument measuring the wrong thing returns the wrong answer, repeatably**
— and the tight confidence interval makes it read as more credible, not less. Static checks
confirm a string appears in the source. That is a floor, not a verdict.

This loads each page in a real browser and DRIVES it:
  * JS console errors and uncaught exceptions (a broken page can still pass every regex)
  * required controls actually EXIST as interactive elements
  * required interactions actually DO something (add an item, filter, undo, persist)
  * localStorage actually gets written
  * the page survives a reload with its state intact

⚠️ THESE CHECKS MUST BE ABLE TO FAIL — and they are proven to, by `--self-test`, which runs
them against deliberately broken fixtures. A functional suite that passes everything is the
same trap as a static suite that passes everything.

⚠️ IT CANNOT SEE "UGLY". Nothing here judges visual design. It answers "does the thing the
prompt asked for actually function", which is the question the static checks silently
stopped answering.
"""
import argparse, json, os, sys, tempfile, time

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("playwright missing. Install:  ./.venv/bin/pip install playwright", file=sys.stderr)
    sys.exit(3)


# ── per-tier functional expectations, derived from the PROMPTS, not from the output ──
SPEC = {
    "pricing-table": {
        "must_render": True,
        "min_text_len": 120,
        "expect": [
            ("three_tier_prices", "page shows $9, $29 and $99",
             lambda p: all(x in p.inner_text("body") for x in ("9", "29", "99"))),
            ("has_cta_buttons", ">=3 clickable call-to-action elements",
             lambda p: len(p.query_selector_all("button, a.btn, a[class*=cta], a[href]")) >= 3),
            ("responsive_at_320", "no horizontal overflow at 320px",
             lambda p: _no_hscroll(p, 320)),
        ],
    },
    "task-board": {
        "must_render": True, "min_text_len": 60,
        "expect": [
            ("add_task_works", "typing + Enter/Add actually adds a visible task",
             lambda p: _add_item(p, "PROBE_TASK_A")),
            ("localStorage_written", "state persisted to localStorage",
             lambda p: len(p.evaluate("Object.keys(localStorage)")) > 0),
            ("survives_reload", "added task still present after reload",
             lambda p: _reload_and_find(p, "PROBE_TASK_A")),
            ("responsive_at_320", "no horizontal overflow at 320px",
             lambda p: _no_hscroll(p, 320)),
        ],
    },
    "sales-dashboard": {
        "must_render": True, "min_text_len": 100,
        "expect": [
            ("svg_chart_drawn", "inline SVG with actual drawn geometry",
             lambda p: p.evaluate(
                 "document.querySelectorAll('svg rect, svg path, svg line, svg circle').length") >= 3),
            ("table_has_rows", "data table has >=4 body rows",
             lambda p: p.evaluate("document.querySelectorAll('tbody tr').length") >= 4),
            ("sort_changes_order", "clicking a header REORDERS the rows",
             lambda p: _sort_reorders(p)),
            ("responsive_at_320", "no horizontal overflow at 320px",
             lambda p: _no_hscroll(p, 320)),
        ],
    },
    "kanban-board": {
        "must_render": True, "min_text_len": 80,
        "expect": [
            # This is the exact check the static suite missed.
            ("add_card_EXISTS", "an input or button for adding a card exists at all",
             lambda p: _has_add_affordance(p)),
            ("add_card_works", "adding a card actually adds a visible card",
             lambda p: _add_item(p, "PROBE_CARD_A")),
            ("undo_control", "an undo control exists",
             lambda p: _find_btn(p, r"undo") is not None),
            ("search_filters", "typing in search reduces visible cards",
             lambda p: _search_filters(p)),
            ("localStorage_written", "board state persisted",
             lambda p: len(p.evaluate("Object.keys(localStorage)")) > 0),
            ("responsive_at_320", "no horizontal overflow at 320px",
             lambda p: _no_hscroll(p, 320)),
        ],
    },
}


# ───────────────────────────── interaction helpers ─────────────────────────────
def _no_hscroll(p, width):
    p.set_viewport_size({"width": width, "height": 900})
    p.wait_for_timeout(180)
    over = p.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth + 2")
    p.set_viewport_size({"width": 1280, "height": 900})
    p.wait_for_timeout(120)
    return not over


def _find_btn(p, pattern):
    import re
    for b in p.query_selector_all("button, a[role=button], [role=button]"):
        try:
            if re.search(pattern, (b.inner_text() or "") + " " + (b.get_attribute("aria-label") or ""),
                         re.I):
                return b
        except Exception:
            pass
    return None


def _has_add_affordance(p):
    """An input that is NOT the search box, or a button that says add/new/+."""
    for i in p.query_selector_all("input, textarea"):
        try:
            meta = " ".join(filter(None, [i.get_attribute("type"), i.get_attribute("placeholder"),
                                          i.get_attribute("id"), i.get_attribute("class"),
                                          i.get_attribute("aria-label")])).lower()
            if "search" in meta or "filter" in meta or i.get_attribute("type") == "search":
                continue
            return True
        except Exception:
            pass
    return _find_btn(p, r"\badd\b|\bnew\b|^\+$") is not None


def _add_item(p, text):
    """Type into the first non-search input and commit via Enter, form submit, or an Add button."""
    target = None
    for i in p.query_selector_all("input, textarea"):
        try:
            meta = " ".join(filter(None, [i.get_attribute("type"), i.get_attribute("placeholder"),
                                          i.get_attribute("id"), i.get_attribute("class")])).lower()
            if "search" in meta or "filter" in meta or i.get_attribute("type") == "search":
                continue
            target = i
            break
        except Exception:
            pass
    if target is None:
        return False
    try:
        target.fill(text)
        target.press("Enter")
        p.wait_for_timeout(250)
        if text in p.inner_text("body"):
            return True
        btn = _find_btn(p, r"\badd\b|\bnew\b|^\+$")
        if btn:
            btn.click()
            p.wait_for_timeout(250)
        return text in p.inner_text("body")
    except Exception:
        return False


def _reload_and_find(p, text):
    try:
        p.reload()
        p.wait_for_timeout(500)
        return text in p.inner_text("body")
    except Exception:
        return False


def _sort_reorders(p):
    try:
        before = p.evaluate("[...document.querySelectorAll('tbody tr')].map(r=>r.innerText).join('|')")
        th = p.query_selector_all("thead th, th")
        if not th:
            return False
        for h in th[:4]:
            h.click()
            p.wait_for_timeout(220)
            after = p.evaluate("[...document.querySelectorAll('tbody tr')].map(r=>r.innerText).join('|')")
            if after != before and after:
                return True
        return False
    except Exception:
        return False


def _search_filters(p):
    try:
        box = None
        for i in p.query_selector_all("input"):
            meta = " ".join(filter(None, [i.get_attribute("type"), i.get_attribute("placeholder"),
                                          i.get_attribute("id"), i.get_attribute("class")])).lower()
            if "search" in meta or "filter" in meta:
                box = i
                break
        if box is None:
            return False
        before = p.evaluate(
            "[...document.querySelectorAll('*')].filter(e=>e.offsetParent!==null).length")
        box.fill("zzzzqqqqnomatch")
        p.wait_for_timeout(350)
        after = p.evaluate(
            "[...document.querySelectorAll('*')].filter(e=>e.offsetParent!==null).length")
        box.fill("")
        p.wait_for_timeout(200)
        return after < before
    except Exception:
        return False


# ───────────────────────────────── driver ─────────────────────────────────
def check_page(pw, path, tier_key, verbose=True):
    spec = SPEC.get(tier_key)
    out = {"file": os.path.basename(path), "tier": tier_key, "console_errors": [],
           "checks": {}, "passed": 0, "total": 0}
    if spec is None:
        out["error"] = "no spec for %s" % tier_key
        return out
    b = pw.chromium.launch(channel="chrome", headless=True)
    ctx = b.new_context(viewport={"width": 1280, "height": 900})
    p = ctx.new_page()
    errs = []
    p.on("console", lambda m: errs.append(m.text[:160]) if m.type == "error" else None)
    p.on("pageerror", lambda e: errs.append("UNCAUGHT: " + str(e)[:160]))
    try:
        p.goto("file://" + os.path.abspath(path), wait_until="load", timeout=30000)
        p.wait_for_timeout(700)
        body = p.inner_text("body")
        out["checks"]["renders_content"] = len(body) >= spec["min_text_len"]
        for name, desc, fn in spec["expect"]:
            try:
                out["checks"][name] = bool(fn(p))
            except Exception as e:
                out["checks"][name] = False
                if verbose:
                    print("      (%s raised %s)" % (name, type(e).__name__))
        # console errors judged LAST: interactions above are what surface them
        out["console_errors"] = errs[:5]
        out["checks"]["no_js_errors"] = len(errs) == 0
    except Exception as e:
        out["error"] = str(e)[:180]
    finally:
        ctx.close(); b.close()
    out["total"] = len(out["checks"])
    out["passed"] = sum(1 for v in out["checks"].values() if v)
    return out


BROKEN = """<!doctype html><html><head><title>x</title></head><body>
<h1>Broken fixture</h1><input type="search" placeholder="Search cards"><div>only a search box</div>
<script>window.addEventListener('load',function(){ undefinedFunction(); });</script>
</body></html>"""

WORKING = """<!doctype html><html><head><title>ok</title><meta name="viewport" content="width=device-width"></head>
<body><h1>Kanban</h1><input id="new" placeholder="New card"><button id="add">Add</button>
<button id="undo">Undo</button><input id="search" placeholder="Search cards" type="search">
<div id="list"></div><script>
const L=document.getElementById('list');
function render(){L.innerHTML='';(JSON.parse(localStorage.cards||'[]')).forEach(c=>{
 const d=document.createElement('div');d.className='card';d.textContent=c;L.appendChild(d);});}
document.getElementById('add').onclick=()=>{const v=document.getElementById('new').value;
 if(!v)return;const a=JSON.parse(localStorage.cards||'[]');a.push(v);localStorage.cards=JSON.stringify(a);render();};
document.getElementById('new').addEventListener('keydown',e=>{if(e.key==='Enter')document.getElementById('add').click();});
document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();
 [...L.children].forEach(c=>c.style.display=c.textContent.toLowerCase().includes(q)?'':'none');});
render();</script></body></html>"""


def self_test(pw):
    """Prove these checks can FAIL. A functional suite that passes everything is the same
    trap as the static suite it replaces."""
    print("=== SELF-TEST: functional checks must be able to fail ===")
    ok = True
    with tempfile.TemporaryDirectory() as d:
        bp = os.path.join(d, "broken.html"); open(bp, "w").write(BROKEN)
        wp = os.path.join(d, "working.html"); open(wp, "w").write(WORKING)
        rb = check_page(pw, bp, "kanban-board", verbose=False)
        rw = check_page(pw, wp, "kanban-board", verbose=False)
        cases = [
            ("broken: add_card_EXISTS is False", rb["checks"].get("add_card_EXISTS"), False),
            ("broken: add_card_works is False", rb["checks"].get("add_card_works"), False),
            ("broken: no_js_errors is False", rb["checks"].get("no_js_errors"), False),
            ("working: add_card_EXISTS is True", rw["checks"].get("add_card_EXISTS"), True),
            ("working: add_card_works is True", rw["checks"].get("add_card_works"), True),
            ("working: search_filters is True", rw["checks"].get("search_filters"), True),
            ("working: localStorage_written is True", rw["checks"].get("localStorage_written"), True),
        ]
        for label, got, want in cases:
            good = bool(got) == want
            ok &= good
            print("  %-44s got=%-5s want=%-5s %s" % (label, got, want, "ok" if good else "*** BROKEN ***"))
    print("\n  %s\n" % ("FUNCTIONAL HARNESS TRUSTWORTHY" if ok
                        else "*** HARNESS BROKEN — DO NOT TRUST RESULTS ***"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="directory of generated .html pages")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    with sync_playwright() as pw:
        if a.self_test:
            return 0 if self_test(pw) else 1
        if not self_test(pw):
            print("ABORT: functional self-test failed"); return 1
        if not a.dir:
            print("need --dir"); return 2

        files = sorted(f for f in os.listdir(a.dir) if f.endswith(".html"))
        results = []
        print("=== FUNCTIONAL CHECKS: %d page(s) ===" % len(files))
        for f in files:
            key = next((k for k in SPEC if k in f), None)
            if key is None:
                continue
            r = check_page(pw, os.path.join(a.dir, f), key)
            results.append(r)
            print("\n  %-42s %d/%d" % (f, r["passed"], r["total"]))
            for n, v in r["checks"].items():
                if not v:
                    print("      FAIL %s" % n)
            for e in r["console_errors"]:
                print("      JS ERROR: %s" % e)

        print("\n" + "=" * 64)
        tot_p = sum(r["passed"] for r in results)
        tot_t = sum(r["total"] for r in results)
        print("  FUNCTIONAL TOTAL: %d/%d across %d pages" % (tot_p, tot_t, len(results)))
        print("  ⚠️ This measures whether the page WORKS. It cannot see whether it looks good.")
        if a.out:
            json.dump({"label": a.label, "dir": a.dir, "passed": tot_p, "total": tot_t,
                       "pages": results}, open(a.out, "w"), indent=1)
            print("  saved -> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
