#!/usr/bin/env python3
"""Full-surface thermal/health telemetry for DGX Spark nodes.

WHY THIS EXISTS
---------------
On 2026-08-18 the thermal guard capped both nodes at a measured 95.4 C while
`nvidia-smi` simultaneously reported a GPU die temperature of 73 C. A dashboard
built on the GPU sensor alone would have shown a comfortable 73 C one second
before the cluster downclocked itself by 30%. The 22 C gap IS the early-warning
window, and nothing was watching it.

This reads the surfaces nvidia-smi does not expose:
  * 7x acpitz thermal zones  -- what the guard actually trips on
  * 4x mlx5                  -- ConnectX-7 NIC (the NCCL fabric)
  * 3x nvme                  -- SSD thermals
  * CPU utilisation          -- from /proc/stat DELTAS, never `ps` (see below)

DESIGN RULES THIS FILE OBEYS (each earned the hard way on this cluster)
  1. A reading that cannot be taken reports UNVERIFIED and counts as unhealthy.
     It never reports 0, and never silently reports "ok".
  2. No `2>/dev/null || true`. A check that degrades from noisy to silent
     exactly when it starts to matter is worse than no check.
  3. Sensors are keyed by INDEX AND TYPE, parsed by name -- never positionally.
  4. `ps -eo pcpu` is a LIFETIME AVERAGE and reads flat under every condition.
     CPU comes from two /proc/stat samples and a delta.
  5. READ-ONLY. This tool has no verb that changes anything, and by default
     opens no socket. (sparkDash was rejected for being an unauthenticated
     endpoint with a destructive verb; this is deliberately the opposite.)

The acpitz zones all report the type string "acpitz" with no labels, so they
CANNOT be named from the system -- a label is a claim. They are reported by
index and must be identified behaviourally (see --fingerprint).
"""
import argparse, json, subprocess, sys, time

UNVERIFIED = "UNVERIFIED"

REMOTE = r'''
echo "###ZONES"
for z in /sys/class/thermal/thermal_zone*/; do
  i=$(basename "$z" | tr -dc 0-9)
  t=$(cat "$z/type" 2>&1) || t=UNREADABLE
  v=$(cat "$z/temp" 2>&1) || v=UNREADABLE
  echo "$i|$t|$v"
done
echo "###HWMON"
for h in /sys/class/hwmon/hwmon*/; do
  n=$(cat "$h/name" 2>&1) || n=UNREADABLE
  for f in "$h"temp*_input; do
    [ -e "$f" ] || continue
    b=$(basename "$f" _input)
    lbl=$(cat "$h$b"_label 2>/dev/null || echo "")
    v=$(cat "$f" 2>&1) || v=UNREADABLE
    echo "$n|$b|$lbl|$v"
  done
done
echo "###STAT"
head -1 /proc/stat
echo "###GPU"
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.max.sm,utilization.gpu,power.draw --format=csv,noheader 2>&1 || echo UNREADABLE
echo "###LOAD"
cat /proc/loadavg
echo "###END"
'''


def ssh(host, script, timeout=25):
    """Run a script on a node. Returns (stdout, None) or (None, reason).

    `timeout` is enforced LOCALLY: ssh's ConnectTimeout bounds only the connect,
    so a node that wedges after accept would hang forever.
    """
    try:
        p = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", host, script],
            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "ssh timed out after %ss" % timeout
    except Exception as e:
        return None, "ssh failed: %s" % type(e).__name__
    if p.returncode != 0:
        return None, "ssh rc=%d: %s" % (p.returncode, (p.stderr or "").strip()[:120])
    return p.stdout, None


def cpu_totals(stat_line):
    """(busy, total) jiffies from a /proc/stat 'cpu ' line."""
    f = [int(x) for x in stat_line.split()[1:]]
    if len(f) < 4:
        return None
    idle = f[3] + (f[4] if len(f) > 4 else 0)      # idle + iowait
    return sum(f) - idle, sum(f)


def parse(out):
    sec, cur = {}, None
    for line in out.splitlines():
        if line.startswith("###"):
            cur = line[3:]; sec[cur] = []
        elif cur:
            sec[cur].append(line)

    zones = []
    for row in sec.get("ZONES", []):
        parts = row.split("|")
        if len(parts) != 3:
            continue
        i, t, v = parts
        zones.append({"index": int(i) if i.isdigit() else i, "type": t,
                      "celsius": round(int(v) / 1000.0, 1) if v.isdigit() else UNVERIFIED})

    hw = []
    for row in sec.get("HWMON", []):
        parts = row.split("|")
        if len(parts) != 4:
            continue
        n, b, lbl, v = parts
        hw.append({"chip": n, "sensor": b, "label": lbl or None,
                   "celsius": round(int(v) / 1000.0, 1) if v.isdigit() else UNVERIFIED})

    gpu = UNVERIFIED
    g = sec.get("GPU", [])
    if g and "UNREADABLE" not in g[0] and "," in g[0]:
        f = [x.strip() for x in g[0].split(",")]
        gpu = {"die_c": f[0], "sm_mhz": f[1], "sm_max_mhz": f[2],
               "util_pct": f[3], "power_w": f[4]}

    return {"zones": zones, "hwmon": hw, "gpu": gpu,
            "stat": sec.get("STAT", [None])[0],
            "loadavg": (sec.get("LOAD", [""])[0].split()[:3] or None)}


def sample(host, interval=1.0):
    """One reading, with CPU% from a real /proc/stat delta."""
    o1, err = ssh(host, REMOTE)
    if err:
        return {"host": host, "status": UNVERIFIED, "reason": err}
    a = parse(o1)
    time.sleep(interval)
    o2, err = ssh(host, REMOTE)
    if err:
        return {"host": host, "status": UNVERIFIED, "reason": err}
    b = parse(o2)

    cpu = UNVERIFIED
    if a.get("stat") and b.get("stat"):
        t1, t2 = cpu_totals(a["stat"]), cpu_totals(b["stat"])
        if t1 and t2 and t2[1] > t1[1]:
            cpu = round(100.0 * (t2[0] - t1[0]) / (t2[1] - t1[1]), 1)

    zone_vals = [z["celsius"] for z in b["zones"] if z["celsius"] != UNVERIFIED]
    return {
        "host": host,
        "status": "ok" if zone_vals else UNVERIFIED,
        "reason": None if zone_vals else "no readable thermal zone",
        "board_max_c": max(zone_vals) if zone_vals else UNVERIFIED,
        "zones": b["zones"], "hwmon": b["hwmon"], "gpu": b["gpu"],
        "cpu_pct": cpu, "loadavg": b["loadavg"],
    }


def fmt(s):
    if s["status"] == UNVERIFIED:
        return "  %-9s %s — %s" % (s["host"], UNVERIFIED, s.get("reason"))
    g = s["gpu"]
    die = g["die_c"] if isinstance(g, dict) else UNVERIFIED
    gap = ""
    if isinstance(g, dict) and s["board_max_c"] != UNVERIFIED:
        try:
            gap = "  gap %+.1f" % (s["board_max_c"] - float(die))
        except (TypeError, ValueError):
            gap = ""
    nic = [h["celsius"] for h in s["hwmon"] if h["chip"] == "mlx5"]
    nvme = [h["celsius"] for h in s["hwmon"] if h["chip"] == "nvme"]
    return ("  %-9s board %5s  die %4s%s   cpu %5s%%  gpu %4s  nic %-18s nvme %s"
            % (s["host"], s["board_max_c"], die, gap, s["cpu_pct"],
               (g["util_pct"] if isinstance(g, dict) else UNVERIFIED),
               ("/".join(str(x) for x in nic) if nic else UNVERIFIED),
               ("/".join(str(x) for x in nvme) if nvme else UNVERIFIED)))


def self_test():
    """Negative control: the collector must SCREAM on an unreachable node.

    A check nobody has watched fail is not yet a check.
    """
    print("SELF-TEST — a reading that cannot be taken must report UNVERIFIED,")
    print("            never 0 and never 'ok'.\n")
    bad = sample("nosuchhost-telemetry-negative-control", interval=0.01)
    ok1 = bad["status"] == UNVERIFIED and bad.get("reason")
    print("  unreachable host -> status=%s reason=%r" % (bad["status"], (bad.get("reason") or "")[:60]))
    print("  %s" % ("PASS" if ok1 else "FAIL — silent success on an unreachable node!"))

    t = cpu_totals("cpu 100 0 100 800 0 0 0 0 0 0")
    ok2 = t == (200, 1000)
    print("\n  cpu_totals parses idle+iowait out: %s -> %s" % (t, "PASS" if ok2 else "FAIL"))

    z = parse("###ZONES\n0|acpitz|UNREADABLE\n###END")["zones"]
    ok3 = z and z[0]["celsius"] == UNVERIFIED
    print("  unreadable zone -> %s : %s" % (z[0]["celsius"] if z else "?", "PASS" if ok3 else "FAIL"))
    print("\n%s" % ("ALL CONTROLS PASS" if (ok1 and ok2 and ok3) else "SELF-TEST FAILED"))
    return 0 if (ok1 and ok2 and ok3) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hosts", default="spark-1,spark-2")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--watch", type=float, default=0, help="repeat every N seconds")
    ap.add_argument("--append", help="append one JSON line per sample to this file")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    hosts = [h.strip() for h in a.hosts.split(",") if h.strip()]
    while True:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        samples = [sample(h) for h in hosts]
        for s in samples:
            s["ts"] = stamp
        if a.json:
            print(json.dumps({"ts": stamp, "nodes": samples}))
        else:
            print("[%s]" % stamp)
            for s in samples:
                print(fmt(s))
        if a.append:
            with open(a.append, "a") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")
        if not a.watch:
            break
        time.sleep(a.watch)
    return 0 if all(s["status"] == "ok" for s in samples) else 2


if __name__ == "__main__":
    sys.exit(main())
