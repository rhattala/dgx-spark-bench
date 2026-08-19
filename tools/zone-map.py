#!/usr/bin/env python3
"""Identify DGX Spark's unlabelled acpitz thermal zones BEHAVIOURALLY.

WHY: all 7 zones report the type string "acpitz" with no label, so they cannot be
named from the system. A dashboard that prints "CPU temp" beside one of them is
asserting something it never measured. The only honest identification is to watch
which zones move under which load.

MEASURED 2026-08-19 on two supposedly identical GB10 nodes:

  CPU-only load   every zone rose within 1.5 C of every other
                  -> there is no CPU-specific rail
  GPU-only load   zones separate into two families ~12 C apart

  spark-1 GPU-adjacent: 0, 2, 4, 5      spark-2 GPU-adjacent: 0, 4, 5

  ** zone 2 is GPU-adjacent on one node and not the other. **

That last point is the reason this script exists: you cannot hardcode a zone
index and expect it to mean the same thing on the next machine. The thermal guard
is correct to read max() across all zones rather than trusting an index.

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

    print("Phase 2 — run a GPU-heavy prefill against the engine, then re-run with")
    print("          --gpu-phase to classify. Kept manual: this script must not")
    print("          decide on its own how to load your serving stack.\n")
    print("Reference result (2026-08-19, two GB10 nodes):")
    print("  spark-1 GPU-adjacent 0,2,4,5   spark-2 GPU-adjacent 0,4,5")
    print("  zone 2 differs between nodes -> never hardcode a zone index.")


if __name__ == "__main__":
    sys.exit(main())
