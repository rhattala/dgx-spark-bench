#!/usr/bin/env python3
"""Identify DGX Spark's unlabelled acpitz thermal zones BEHAVIOURALLY.

WHY: all 7 zones report the type string "acpitz" with no label, so they cannot be
named from the system. A dashboard that prints "CPU temp" beside one of them is
asserting something it never measured. The only honest identification is to watch
which zones move under which load.

MEASURED 2026-08-19 on two supposedly identical GB10 nodes:

  CPU-only load   every zone rose within 1.5 C of every other
                  -> there is no CPU-specific rail
  GPU-only load   zones separate into two families ~9.5 C apart

  both nodes GPU-adjacent: 0, 2, 4, 5   distant: 1, 3, 6

A short probe once suggested the two nodes differed here. They do not — that was a
thermal transient caught mid-rise, and the claim was retracted. Classify from
STEADY-STATE samples, not a single 90 s burst: zone 2 on one node needed several
minutes to reach its plateau and looked "distant" until it did.

Read max() across all zones rather than trusting an index, which is what the
thermal guard already does.

Usage:  zone-map.py spark-1 spark-2
"""
import subprocess, sys, time

READ = ('for z in /sys/class/thermal/thermal_zone*/temp; do '
        'printf "%s " $(cat $z); done')


def zones(host):
    try:
        out = subprocess.run(["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes",
                              host, READ], capture_output=True, text=True,
                             timeout=25).stdout.split()
        return [int(x) / 1000.0 for x in out if x.isdigit()]
    except Exception:
        return None


def spin(host, secs, threads=20):
    subprocess.run(["ssh", "-o", "ConnectTimeout=8", host,
                    f'for i in $(seq 1 {threads}); do (timeout {secs} bash -c '
                    f'"while :; do :; done" &); done'],
                   capture_output=True, timeout=25)


def families(pre, load):
    """Split zones by how much they moved. Returns (deltas, labels)."""
    d = [b - a for a, b in zip(pre, load)]
    mid = (max(d) + min(d)) / 2
    return d, ["GPU-adj" if x > mid else "distant" for x in d]


def main():
    hosts = sys.argv[1:] or ["spark-1", "spark-2"]
    print("Phase 1 — CPU-only load (no GPU work)\n")
    for h in hosts:
        a = zones(h)
        if not a:
            print(f"  {h}: UNVERIFIED — cannot read zones"); continue
        spin(h, 75)
        time.sleep(70)
        b = zones(h)
        d, _ = families(a, b)
        print(f"  {h}: idle {['%.1f'%x for x in a]}")
        print(f"  {h}: load {['%.1f'%x for x in b]}")
        print(f"  {h}: rise {['%+.1f'%x for x in d]}  spread {max(d)-min(d):.1f} C")
        print(f"       -> {'uniform: no CPU-specific rail' if max(d)-min(d) < 3 else 'NON-uniform: a CPU rail may exist'}\n")

    print("Phase 2 — GPU classification is deliberately NOT automated here: this")
    print("          script must not decide how to load your serving stack. Put the")
    print("          engine under sustained GPU load for several MINUTES (not one")
    print("          burst), sample zones repeatedly, and compare medians.\n")
    print("Reference (2026-08-18, 924 hot vs 38 idle samples per node, 2x GB10):")
    print("  GPU-adjacent 0,2,4,5   distant 1,3,6   -- identical on both nodes")


if __name__ == "__main__":
    sys.exit(main())
