#!/usr/bin/env python3
"""Screenshot the generated front-end artifacts, for the README and write-ups.

Uses the same path the functional suite uses — real system Chrome via Playwright — because
the point of these images is to show what the pages ACTUALLY render, not what a headless
shell approximates. `file://` is fine here: Playwright opens local files directly, which
the browser extension cannot.

The default targets are the extra-hard kanban boards from both models, because that pair
IS the finding: both scored 18/18 on static checks; only one of them can add a card.
"""
import argparse, os, sys

DEFAULT = [
    ("results/2026-08-16/frontend-repeat/extra-hard-kanban-board-run1.html",
     "docs/img/kanban-deepseek.png"),
    # ⚠️ run2, NOT run1. Run 1 threw an uncaught "Unexpected token 'null'" and rendered
    # nothing at all — using it would imply Qwen's board is always blank, which is false.
    # Runs 2 and 3 render perfectly well and STILL have no add-card control, which is both
    # the honest picture and the more interesting one.
    ("results/2026-08-16/qwen-repeat/extra-hard-kanban-board-run2.html",
     "docs/img/kanban-qwen.png"),
    ("results/2026-08-16/qwen-repeat/extra-hard-kanban-board-run1.html",
     "docs/img/kanban-qwen-run1-jserror.png"),
    ("results/2026-08-16/frontend-repeat/hard-sales-dashboard-run1.html",
     "docs/img/dashboard-deepseek.png"),
    ("results/2026-08-16/qwen-repeat/hard-sales-dashboard-run1.html",
     "docs/img/dashboard-qwen.png"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=860)
    ap.add_argument("--full-page", action="store_true")
    a = ap.parse_args()

    from playwright.sync_api import sync_playwright
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    made = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True)
        for src, dst in DEFAULT:
            s = os.path.join(root, src)
            d = os.path.join(root, dst)
            if not os.path.exists(s):
                print("  SKIP (missing): %s" % src)
                continue
            os.makedirs(os.path.dirname(d), exist_ok=True)
            pg = b.new_page(viewport={"width": a.width, "height": a.height},
                            device_scale_factor=2)
            pg.goto("file://" + s)
            pg.wait_for_timeout(1200)          # let any JS render
            pg.screenshot(path=d, full_page=a.full_page)
            pg.close()
            made.append(dst)
            print("  %-46s -> %s" % (os.path.basename(src), dst))
        b.close()
    print("\n  %d image(s) written" % len(made))
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
