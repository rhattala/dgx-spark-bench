#!/usr/bin/env python3
"""Minimal GPU thermal guard: cap the clock when hot, release when cool, do nothing else.

WHAT IT DOES. One job. At or above TRIP_C, apply a clock cap. Once the GPU has been below
CLEAR_C for CLEAR_CYCLES consecutive samples, release it. It does **not** touch the
inference engine, admission control, or any service — a guard that can stop your serving
stack is a bigger hazard than the heat it prevents.

⚠️ DESIGNED AGAINST ONE SPECIFIC INCIDENT: "a self-healing loop has no concept of 'I
should not be running'." A verify-and-reapply loop re-asserted state every 5 s, correctly
and tirelessly, on behalf of a test unit that should have been dead. The loop was not
buggy — it had no ownership or liveness check, so it could not distinguish legitimate
operation from being an orphan. Anything that re-asserts state needs an owner check, a
lease, or a self-limit; a reapply loop alone is not enough. Four defences here:

  1. LOCKFILE (flock) — exactly one guard per node, ever. A second copy exits immediately
     rather than fighting the first over the clock.
  2. SELF-LIMIT — MAX_RUNTIME (default 24 h). It cannot outlive its purpose by weeks.
  3. HEARTBEAT — state written every cycle. `--status` reports **UNVERIFIED (rc=2)**, never
     "ok", when the heartbeat is stale. A guard that is silently dead is worse than no
     guard, because you believe you are protected.
  4. RELEASES ONLY THE CAP IT APPLIED — with a limit it cannot escape. ⚠️ It CANNOT tell
     whether a cap was already there: no `nvidia-smi` field reports an `-lgc` lock (that is
     why the parity checker samples clocks under load instead). So if you have capped a node
     deliberately and this guard then trips, its `set_cap(True)` is a hardware no-op, it
     records the cap as its own, and on exit it releases YOURS. **Do not run this guard on a
     node you have deliberately capped.** The honest claim is "it releases the cap it
     applied, and cannot distinguish that from someone else's."

⚠️ AN UNREADABLE SENSOR IS NOT A COOL GPU. If `nvidia-smi` fails, the cycle records
`sensor: UNREADABLE` and takes no action — it never writes a reassuring temperature.

⚠️ HONEST LIMIT — STALENESS DETECTION HAS A BLIND WINDOW. `--status` decides liveness from
heartbeat age, so for up to `interval * 6` seconds after a SIGKILL it still reports "alive".
Measured: with `--interval 2`, a hard-killed guard read OK for ~12 s before flipping to
UNVERIFIED. At the 5 s default that window is ~30 s. If you need faster detection, shorten
the interval; do not read a single OK as proof of protection during an incident.

⚠️ The interval used is the one the RUNNING GUARD recorded in its heartbeat, not whatever
`--interval` you happen to pass to `--status`. Those diverged before: a guard running at 60 s
was called dead by a status check defaulting to 5 s (false alarm), and — worse — a guard
killed 25 s ago after running at 2 s read "alive" to a status check defaulting to 5 s (false
OK, silently extending the very window disclosed above). A heartbeat written by an older
build carries no interval; `--status` then falls back to the passed value and SAYS SO.

Exit codes (--status):  0 alive · 1 stopped cleanly · 2 UNVERIFIED (stale/missing/unreadable)
"""
import argparse, atexit, fcntl, json, os, signal, subprocess, sys, time

DEFAULT_TRIP_C = 88.0
DEFAULT_CLEAR_C = 78.0
DEFAULT_CLEAR_CYCLES = 6
DEFAULT_INTERVAL = 5.0
DEFAULT_MAX_RUNTIME = 24 * 3600.0
DEFAULT_CAP_MHZ = 2000
STALE_FACTOR = 6          # heartbeat older than INTERVAL * this == not alive
# The ONE range the loop and --status agree on. Validated at LAUNCH as well as on read: a
# value only the writer accepts is a value the reader must refuse, and refusing it after
# the fact is how a live guard gets called dead, or a dead one alive.
INTERVAL_MIN, INTERVAL_MAX = 1.0, 3600.0

_state = {"capped_by_us": False, "dry_run": False, "cap_mhz": DEFAULT_CAP_MHZ}


# ───────────────────────────── pure decision logic ─────────────────────────────
def decide(temp, capped_by_us, cool_streak, trip, clear, clear_cycles):
    """Pure. -> (action, new_cool_streak) where action is 'cap' | 'release' | 'hold'.

    Separated from the loop so it can be fed deliberately-wrong data by --self-test.
    Reading this function and believing it is not evidence; the self-test is.
    """
    if temp is None:
        return "hold", cool_streak            # unreadable sensor: never act, never reassure
    if temp >= trip and not capped_by_us:
        return "cap", 0
    if capped_by_us:
        if temp < clear:
            cool_streak += 1
            if cool_streak >= clear_cycles:
                return "release", 0
            return "hold", cool_streak
        return "hold", 0                      # a single hot sample resets the cool streak
    return "hold", cool_streak


def sane_interval(v):
    """(value, reason_or_None). Clamps into the range --status will accept.

    ⚠️ `--interval` was a bare float at launch while --status trusted only [1,3600], so the
    two disagreed about the same number. Executed: a guard legitimately started at 0.5 s and
    SIGKILLed 10 s earlier read **OK: guard alive** — the exact false-OK this mechanism
    exists to prevent, resurrected for sub-second intervals; and one started at 7200 s read
    UNVERIFIED forever while perfectly healthy. Clamp where the value is SET, so read-side
    clamping only ever fires on genuine corruption.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL, "not a number, using %g" % DEFAULT_INTERVAL
    if f < INTERVAL_MIN:
        return INTERVAL_MIN, "%g is below the %g s floor --status can verify" % (f, INTERVAL_MIN)
    if f > INTERVAL_MAX:
        return INTERVAL_MAX, "%g exceeds the %g s ceiling --status can verify" % (f, INTERVAL_MAX)
    return f, None


def status_from(d, now, interval):
    """Pure. -> (exit_code, message). `d` is the parsed heartbeat, or None if absent.

    The guard's OWN interval governs. See the module docstring: judging by the caller's
    --interval produced both a false alarm and a false OK on a dead guard.
    """
    if d is None:
        return 2, "UNVERIFIED: no heartbeat file — the guard has never run on this node"
    # ⚠️ THE HEARTBEAT IS UNTRUSTED INPUT even though this program wrote it. A string,
    # zero, or an absurd value all arrived here uncaught: {"interval":"abc"} raised
    # TypeError and the process exited rc=1 -- which this contract defines as "STOPPED
    # CLEANLY", so a CORRUPT heartbeat read as an orderly shutdown. And interval=1e9 made
    # a 30-day-old heartbeat report "alive", i.e. one fat-fingered flag blinds --status
    # permanently. Validate and clamp; say so when the recorded value is unusable.
    own, src = None, "caller (heartbeat records none)"
    try:
        cand = float(d.get("interval"))
        if INTERVAL_MIN <= cand <= INTERVAL_MAX:
            own, src = cand, "guard"
        elif cand == cand:                       # a real number, just not a sane one
            src = "caller (heartbeat interval %g is out of range)" % cand
    except (TypeError, ValueError):
        if d.get("interval") is not None:
            src = "caller (heartbeat interval is not a number)"
    interval = own or interval
    try:
        age = now - float(d.get("ts", 0))
    except (TypeError, ValueError):
        return 2, "UNVERIFIED: heartbeat timestamp is unreadable — treat as NOT protected"
    if not d.get("running"):
        return 1, ("STOPPED: guard exited cleanly %.0fs ago (capped_by_us=%s)"
                   % (age, d.get("capped_by_us")))
    if age > interval * STALE_FACTOR:
        return 2, ("UNVERIFIED: heartbeat is %.0fs old (interval %.0fs, from %s) — the guard "
                   "is NOT alive. You are not protected." % (age, interval, src))
    t = d.get("temp")
    return 0, ("OK: guard alive, last %.0fs ago (interval %.0fs from %s) | temp %s | "
               "capped_by_us=%s"
               % (age, interval, src, ("%.0fC" % t) if t is not None else "UNREADABLE",
                  d.get("capped_by_us")))


# ───────────────────────────── side-effecting parts ─────────────────────────────
def gpu():
    """(temp_c, sm_mhz) or (None, None). Never raises, and never lies: a failure is None,
    which the caller must treat as UNVERIFIED rather than as 'cool'."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return None, None
        t, sm = [x.strip() for x in r.stdout.strip().split(",")[:2]]
        return float(t), float(sm)
    except Exception:
        return None, None


def set_cap(on):
    if _state["dry_run"]:
        print("[guard] DRY-RUN: would %s cap" % ("apply %d MHz" % _state["cap_mhz"] if on
                                                 else "release"), flush=True)
        return True
    arg = ["-lgc", "0,%d" % _state["cap_mhz"]] if on else ["-rgc"]
    r = subprocess.run(["sudo", "-n", "nvidia-smi"] + arg, capture_output=True, text=True)
    return r.returncode == 0


def write_state(path, **kw):
    kw["ts"] = time.time()
    try:
        with open(path, "w") as f:
            json.dump(kw, f)
    except Exception:
        pass


def self_test():
    """Cases that MUST fail are the point. If nothing fails, suspect the harness."""
    print("=== SELF-TEST: the guard's decisions must be able to FAIL ===\n")
    ok = True

    print("  decide() — trip, hold, and release")
    cases = [
        # label,                          temp  capped streak -> action,   streak
        ("hot, not yet capped -> cap",      90, False, 0, "cap",     0),
        ("hot, already capped -> hold",     90, True,  0, "hold",    0),
        ("cool but NOT capped -> hold",     40, False, 0, "hold",    0),
        ("cool, capped, streak building",   40, True,  0, "hold",    1),
        ("cool, capped, streak completes",  40, True,  5, "release", 0),
        ("HOT AGAIN mid-streak resets",     85, True,  4, "hold",    0),
        ("exactly at trip -> cap",          88, False, 0, "cap",     0),
        ("just below trip -> hold",       87.9, False, 0, "hold",    0),
        ("SENSOR UNREADABLE -> never act", None, False, 0, "hold",   0),
        ("unreadable while capped -> hold", None, True, 3, "hold",   3),
    ]
    for label, temp, capped, streak, want_a, want_s in cases:
        a, s = decide(temp, capped, streak, 88.0, 78.0, 6)
        good = (a == want_a and s == want_s)
        ok &= good
        print("    %-34s -> %-7s streak=%d  want %-7s %d  %s"
              % (label, a, s, want_a, want_s, "ok" if good else "*** BROKEN ***"))

    print("\n  status_from() — a dead guard must NOT report ok")
    now = 1000.0
    scases = [
        ("no heartbeat at all",       None,                                          2),
        ("fresh heartbeat",           {"ts": now - 2, "running": True, "temp": 55},   0),
        ("STALE heartbeat (dead)",    {"ts": now - 400, "running": True, "temp": 55}, 2),
        ("clean exit",                {"ts": now - 9, "running": False},              1),
        ("alive, sensor unreadable",  {"ts": now - 2, "running": True, "temp": None}, 0),
        # --- added after review: the caller's --interval must NOT govern ---
        ("slow guard (60s), 45s old -> alive, not a false alarm",
         {"ts": now - 45, "running": True, "temp": 55, "interval": 60}, 0),
        ("fast guard (2s) DEAD 25s -> must NOT read OK",
         {"ts": now - 25, "running": True, "temp": 55, "interval": 2}, 2),
        ("legacy heartbeat with no interval falls back to caller's",
         {"ts": now - 400, "running": True, "temp": 55}, 2),
        # --- second-pass review: the heartbeat is untrusted input ---
        ("interval is a STRING -> must not crash into rc=1",
         {"ts": now - 400, "running": True, "temp": 55, "interval": "abc"}, 2),
        ("absurd interval must NOT make a 30-day heartbeat 'alive'",
         {"ts": now - 2592000, "running": True, "temp": 55, "interval": 1e9}, 2),
        ("negative interval falls back, stays loud",
         {"ts": now - 400, "running": True, "temp": 55, "interval": -5}, 2),
        ("ts is a string -> UNVERIFIED, not a crash",
         {"ts": "nope", "running": True, "temp": 55, "interval": 5}, 2),
    ]
    for label, d, want in scases:
        rc, msg = status_from(d, now, 5.0)
        good = rc == want
        ok &= good
        print("    %-34s rc=%d want=%d  %s" % (label, rc, want,
                                               "ok" if good else "*** BROKEN ***"))

    print("\n  sane_interval() — the launcher must not write what --status refuses")
    for label, given, want in [("sub-second clamps up", 0.5, INTERVAL_MIN),
                               ("huge clamps down", 7200, INTERVAL_MAX),
                               ("in range passes through", 7.0, 7.0),
                               ("junk falls back", "abc", DEFAULT_INTERVAL),
                               ("negative clamps up", -5, INTERVAL_MIN)]:
        got, _ = sane_interval(given)
        good = got == want
        ok &= good
        print("    %-26s %-6s -> %-7s want %-7s %s" % (label, given, got, want,
                                                       "ok" if good else "*** BROKEN ***"))

    print("\n  %s\n" % ("HARNESS TRUSTWORTHY" if ok else "*** BROKEN — DO NOT TRUST IT ***"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default=os.environ.get("GUARD_NAME", "gpu"),
                    help="label for this node; namespaces the lock and heartbeat files")
    ap.add_argument("--trip", type=float, default=float(os.environ.get("GUARD_TRIP_C", DEFAULT_TRIP_C)))
    ap.add_argument("--clear", type=float, default=float(os.environ.get("GUARD_CLEAR_C", DEFAULT_CLEAR_C)))
    ap.add_argument("--clear-cycles", type=int, default=DEFAULT_CLEAR_CYCLES)
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    ap.add_argument("--cap-mhz", type=int, default=DEFAULT_CAP_MHZ)
    ap.add_argument("--max-runtime", type=float, default=DEFAULT_MAX_RUNTIME,
                    help="seconds; the guard exits rather than outliving its purpose")
    ap.add_argument("--run-dir", default=os.environ.get("GUARD_RUN_DIR", "/tmp"))
    ap.add_argument("--dry-run", action="store_true",
                    help="log clock actions instead of performing them")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--test-trip", type=float, default=None,
                    help="override the trip point to PROVE the guard fires (e.g. --test-trip 30)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    heartbeat = os.path.join(a.run_dir, "thermal-guard-%s.state" % a.name)
    lockfile = os.path.join(a.run_dir, "thermal-guard-%s.lock" % a.name)

    if a.status:
        d = None
        if os.path.exists(heartbeat):
            try:
                d = json.load(open(heartbeat))
            except Exception as e:
                print("UNVERIFIED: heartbeat unreadable (%s)" % e)
                return 2
        try:
            rc, msg = status_from(d, time.time(), sane_interval(a.interval)[0])
        except Exception as e:
            # Never let an unexpected shape exit through a code that means something else.
            print("UNVERIFIED: could not evaluate the heartbeat (%s) — treat as NOT "
                  "protected" % type(e).__name__)
            return 2
        print(msg)
        return rc

    a.interval, why = sane_interval(a.interval)
    if why:
        print("[guard] interval clamped to %g s: %s" % (a.interval, why), flush=True)

    _state["dry_run"] = a.dry_run
    _state["cap_mhz"] = a.cap_mhz
    trip = a.test_trip if a.test_trip is not None else a.trip
    clear = (trip - 10) if a.test_trip is not None else a.clear

    lock = open(lockfile, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another guard already holds %s — exiting (one owner only)" % lockfile)
        return 1

    def release_on_exit():
        if _state["capped_by_us"]:
            set_cap(False)
            print("[guard] exiting — released the cap we applied", flush=True)
        write_state(heartbeat, running=False, capped_by_us=_state["capped_by_us"],
                    interval=a.interval)

    atexit.register(release_on_exit)
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, lambda *_: sys.exit(0))

    t_start = time.time()
    cool_streak = 0
    print("[guard] %s up | trip %.0fC clear %.0fC | interval %.0fs | self-limit %.1fh%s"
          % (a.name, trip, clear, a.interval, a.max_runtime / 3600,
             " | DRY-RUN" if a.dry_run else ""), flush=True)

    while True:
        if time.time() - t_start > a.max_runtime:
            print("[guard] self-limit reached — exiting rather than running forever",
                  flush=True)
            return 0
        temp, sm = gpu()
        action, cool_streak = decide(temp, _state["capped_by_us"], cool_streak,
                                     trip, clear, a.clear_cycles)
        if action == "cap":
            applied = set_cap(True)
            _state["capped_by_us"] = applied
            print("[guard] %.0fC >= %.0fC — %s %d MHz cap"
                  % (temp, trip, "APPLIED" if applied else "FAILED TO APPLY", a.cap_mhz),
                  flush=True)
        elif action == "release":
            if set_cap(False):
                _state["capped_by_us"] = False
                print("[guard] %.0fC < %.0fC for %d cycles — cap released"
                      % (temp, clear, a.clear_cycles), flush=True)

        write_state(heartbeat, running=True, temp=temp, sm=sm, interval=a.interval,
                    capped_by_us=_state["capped_by_us"],
                    sensor="ok" if temp is not None else "UNREADABLE")
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
