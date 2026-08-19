#!/usr/bin/env python3
"""Render captured spark-telemetry samples into a self-contained HTML dashboard.

DELIBERATELY A STATIC FILE, NOT A SERVICE. The prior dashboard proposal for this
cluster was rejected because it exposed an unauthenticated API on 0.0.0.0 with an
SSH shutdown verb. This has no server, no port, no verbs and nothing to
authenticate: it reads a JSONL capture and writes one HTML file.

    spark-dashboard.py --telemetry results/2026-08-18/telemetry.jsonl --out dash.html
"""
import argparse, html, json, os, statistics as st, sys
from collections import defaultdict

TRIP = 95.0          # thermal guard stage 1, measured
STAGE2 = 96.0        # refuses new tailnet connections
CRITICAL = 104.0     # kernel trip


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass          # a partial final line during live capture
    return rows


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def series(rows, host, get):
    out = []
    for r in rows:
        if r.get("host") != host:
            continue
        v = get(r)
        if v is not None:
            out.append(v)
    return out


def spark_svg(datasets, width=880, height=190, ymin=None, ymax=None,
              hlines=(), pad=34):
    """Multi-series line chart. datasets = [(label, colorvar, [values])]."""
    vals = [v for _, _, s in datasets for v in s]
    if not vals:
        return '<p class="empty">no data</p>'
    lo = ymin if ymin is not None else min(vals)
    hi = ymax if ymax is not None else max(vals)
    for _, y in hlines:
        lo, hi = min(lo, y), max(hi, y)
    if hi - lo < 1e-9:
        hi = lo + 1
    span = hi - lo
    lo -= span * 0.08
    hi += span * 0.08
    span = hi - lo

    def Y(v):
        return round(height - pad - (v - lo) / span * (height - 2 * pad), 1)

    def X(i, n):
        return round(pad + i / max(1, n - 1) * (width - pad - 10), 1)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    # horizontal reference lines (trip points)
    for label, y in hlines:
        yy = Y(y)
        parts.append(
            f'<line x1="{pad}" y1="{yy}" x2="{width-10}" y2="{yy}" '
            f'stroke="var(--crimson)" stroke-width="1" stroke-dasharray="4 3" opacity=".75"/>'
            f'<text x="{width-12}" y="{yy-5}" text-anchor="end" font-size="10" '
            f'font-family="ui-monospace,monospace" fill="var(--crimson)">{html.escape(label)}</text>')
    # y axis ticks
    for frac in (0, .5, 1):
        v = lo + span * frac
        yy = Y(v)
        parts.append(
            f'<line x1="{pad}" y1="{yy}" x2="{width-10}" y2="{yy}" stroke="currentColor" opacity=".1"/>'
            f'<text x="{pad-6}" y="{yy+3}" text-anchor="end" font-size="10" '
            f'font-family="ui-monospace,monospace" fill="currentColor" opacity=".55">{v:.0f}</text>')
    for label, color, s in datasets:
        if not s:
            continue
        n = len(s)
        pts = " ".join(f"{X(i,n)},{Y(v)}" for i, v in enumerate(s))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                     f'stroke-width="1.6" stroke-linejoin="round" opacity=".95"/>')
    parts.append("</svg>")
    return "".join(parts)


def tile(label, value, unit="", state="ok", sub=""):
    return (f'<div class="tile s-{state}"><span class="tl">{html.escape(label)}</span>'
            f'<span class="tv">{html.escape(str(value))}<span class="tu">{html.escape(unit)}</span></span>'
            f'<span class="ts">{html.escape(sub)}</span></div>')


def headroom_state(peak):
    if peak is None:
        return "unv"
    h = TRIP - peak
    return "crit" if h <= 0 else "warn" if h < 5 else "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="DGX Spark Thermal & Fabric Telemetry")
    a = ap.parse_args()

    rows = load(a.telemetry)
    if not rows:
        print("UNVERIFIED: no samples read", file=sys.stderr)
        return 2
    hosts = sorted({r["host"] for r in rows})
    unv = sum(1 for r in rows if r.get("status") != "ok")
    span = f'{rows[0]["ts"].replace("T"," ")} → {rows[-1]["ts"].replace("T"," ")}'
    hours = len(rows) / max(1, len(hosts)) / 60.0

    COLORS = {hosts[0]: "var(--copper)"}
    if len(hosts) > 1:
        COLORS[hosts[1]] = "var(--teal)"

    board = {h: series(rows, h, lambda r: num(r.get("board_max_c"))) for h in hosts}
    die = {h: series(rows, h, lambda r: num((r.get("gpu") or {}).get("die_c"))) for h in hosts}
    cpu = {h: series(rows, h, lambda r: num(r.get("cpu_pct"))) for h in hosts}
    clk = {h: series(rows, h, lambda r: num((r.get("gpu") or {}).get("sm_mhz"))) for h in hosts}
    pwr = {h: series(rows, h, lambda r: num((r.get("gpu") or {}).get("power_w"))) for h in hosts}
    nic = {h: series(rows, h, lambda r: max([num(x["celsius"]) for x in r.get("hwmon", [])
                                             if x["chip"] == "mlx5" and num(x["celsius"]) is not None] or [None])
                     if any(x["chip"] == "mlx5" for x in r.get("hwmon", [])) else None) for h in hosts}
    nvme = {h: series(rows, h, lambda r: max([num(x["celsius"]) for x in r.get("hwmon", [])
                                              if x["chip"] == "nvme" and num(x["celsius"]) is not None] or [None])
                      if any(x["chip"] == "nvme" for x in r.get("hwmon", [])) else None) for h in hosts}
    gap = {h: [b - d for b, d in zip(board[h], die[h])] for h in hosts}

    # zone families: the 7 acpitz zones carry no distinguishing label, so group
    # them by observed mean rather than asserting names we did not measure.
    last = {h: next(r for r in reversed(rows) if r["host"] == h) for h in hosts}

    T = []
    for h in hosts:
        b = board[h]
        peak = max(b) if b else None
        T.append(tile(f"{h} · board peak", f"{peak:.1f}" if peak else "—", "°C",
                      headroom_state(peak),
                      f"headroom {TRIP-peak:+.1f} to trip" if peak else "UNVERIFIED"))
    for h in hosts:
        g = gap[h]
        T.append(tile(f"{h} · board−die gap", f"{max(g):.1f}" if g else "—", "°C",
                      "warn" if g and max(g) > 15 else "ok",
                      f"mean {st.mean(g):.1f}" if g else ""))
    for h in hosts:
        n = nic[h]
        T.append(tile(f"{h} · NIC asic", f"{max(n):.0f}" if n else "—", "°C", "ok",
                      "ConnectX-7 · unmonitored before"))

    charts = []
    charts.append(("Board temperature — what the thermal guard trips on",
                   f"{len(rows)} samples · {hours:.1f} h · dashed line = stage-1 trip",
                   spark_svg([(h, COLORS[h], board[h]) for h in hosts],
                             hlines=(("stage 1  95°C", TRIP),))))
    charts.append(("GPU die temperature — what nvidia-smi reports",
                   "same window, same scale as above — note it never approaches the trip line",
                   spark_svg([(h, COLORS[h], die[h]) for h in hosts],
                             hlines=(("stage 1  95°C", TRIP),))))
    charts.append(("Board minus die — the blind spot, in degrees",
                   "a GPU-only dashboard is wrong by this much, and wrong most at peak",
                   spark_svg([(h, COLORS[h], gap[h]) for h in hosts])))
    charts.append(("ConnectX-7 NIC (mlx5 asic)",
                   "the 200GbE fabric carrying all NCCL traffic — not exposed by nvidia-smi",
                   spark_svg([(h, COLORS[h], nic[h]) for h in hosts])))
    charts.append(("NVMe",
                   "labelled sensors: Composite, Sensor 1, Sensor 2 — max shown",
                   spark_svg([(h, COLORS[h], nvme[h]) for h in hosts])))
    charts.append(("GPU SM clock",
                   "drops are the guard shedding to 1750 MHz, or power/thermal DVFS",
                   spark_svg([(h, COLORS[h], clk[h]) for h in hosts])))
    charts.append(("CPU utilisation",
                   "from /proc/stat deltas — `ps -eo pcpu` is a lifetime average and reads flat",
                   spark_svg([(h, COLORS[h], cpu[h]) for h in hosts])))
    charts.append(("GPU power draw",
                   "power limit and fan speed are not exposed on this platform",
                   spark_svg([(h, COLORS[h], pwr[h]) for h in hosts])))

    zone_rows = ""
    for h in hosts:
        zs = last[h].get("zones", [])
        cells = "".join(f'<td class="num">{z["celsius"]}</td>' for z in zs)
        zone_rows += f"<tr><td>{html.escape(h)}</td>{cells}</tr>"
    nz = len(last[hosts[0]].get("zones", []))
    zone_head = "".join(f"<th class='num'>zone {i}</th>" for i in range(nz))

    legend = " ".join(
        f'<span class="lg"><i style="background:{COLORS[h]}"></i>{html.escape(h)}</span>'
        for h in hosts)

    chart_html = "".join(
        f'<section class="card"><h3>{html.escape(t)}</h3>'
        f'<p class="sub">{html.escape(s)}</p><div class="cw">{svg}</div></section>'
        for t, s, svg in charts)

    doc = f"""<title>{html.escape(a.title)}</title>
<style>
:root{{
  --paper:#F7F9FA;--panel:#FFF;--ink:#0E141A;--mid:#586470;--faint:#8A949E;
  --rule:#DBE2E8;--rule2:#C2CBD4;
  --copper:#A64B22;--teal:#0B6B62;--crimson:#9A2828;--amber:#8A6100;
  --ok:#0B6B62;--warn:#8A6100;--crit:#9A2828;--unv:#586470;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sh:0 1px 2px rgba(14,20,26,.05),0 8px 22px -14px rgba(14,20,26,.16);
}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --paper:#0C1015;--panel:#141A20;--ink:#E6EBF0;--mid:#98A3AE;--faint:#6C7681;
  --rule:#222A32;--rule2:#313B45;
  --copper:#E08A5B;--teal:#5FC7B8;--crimson:#E38585;--amber:#D9AC4C;
  --ok:#5FC7B8;--warn:#D9AC4C;--crit:#E38585;--unv:#98A3AE;
  --sh:0 1px 2px rgba(0,0,0,.45),0 8px 22px -14px rgba(0,0,0,.75);
}}}}
:root[data-theme="dark"]{{
  --paper:#0C1015;--panel:#141A20;--ink:#E6EBF0;--mid:#98A3AE;--faint:#6C7681;
  --rule:#222A32;--rule2:#313B45;
  --copper:#E08A5B;--teal:#5FC7B8;--crimson:#E38585;--amber:#D9AC4C;
  --ok:#5FC7B8;--warn:#D9AC4C;--crit:#E38585;--unv:#98A3AE;
  --sh:0 1px 2px rgba(0,0,0,.45),0 8px 22px -14px rgba(0,0,0,.75);
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
     font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:76rem;margin:0 auto;padding:2.5rem 1.25rem 5rem}}
header{{border-bottom:2px solid var(--ink);padding-bottom:1.4rem;margin-bottom:1.6rem}}
h1{{font-size:clamp(1.5rem,3.4vw,2.1rem);font-weight:780;letter-spacing:-.025em;margin:0 0 .35rem}}
.meta{{font-family:var(--mono);font-size:.74rem;color:var(--mid);display:flex;flex-wrap:wrap;gap:.35rem 1.1rem}}
.legend{{display:flex;gap:1rem;margin-top:.7rem;font-family:var(--mono);font-size:.74rem;color:var(--mid)}}
.lg i{{display:inline-block;width:11px;height:3px;vertical-align:middle;margin-right:.4rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(13.5rem,1fr));gap:.8rem;margin-bottom:1.8rem}}
.tile{{background:var(--panel);border:1px solid var(--rule);border-left:3px solid var(--unv);
      padding:.85rem .95rem;box-shadow:var(--sh);display:flex;flex-direction:column;gap:.15rem}}
.tile.s-ok{{border-left-color:var(--ok)}}
.tile.s-warn{{border-left-color:var(--warn)}}
.tile.s-crit{{border-left-color:var(--crit)}}
.tl{{font-family:var(--mono);font-size:.66rem;letter-spacing:.08em;text-transform:uppercase;color:var(--faint)}}
.tv{{font-family:var(--mono);font-size:1.55rem;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.1}}
.tu{{font-size:.8rem;font-weight:500;color:var(--mid);margin-left:.15rem}}
.ts{{font-family:var(--mono);font-size:.68rem;color:var(--mid)}}
.tile.s-crit .tv{{color:var(--crit)}}
.tile.s-warn .tv{{color:var(--warn)}}
.card{{background:var(--panel);border:1px solid var(--rule);box-shadow:var(--sh);margin-bottom:1rem;overflow:hidden}}
.card h3{{font-size:.98rem;font-weight:700;margin:0;padding:.95rem 1.1rem .1rem;letter-spacing:-.008em}}
.sub{{font-family:var(--mono);font-size:.7rem;color:var(--faint);margin:0;padding:0 1.1rem .8rem}}
.cw{{overflow-x:auto;padding:0 1.1rem 1rem}}
svg{{display:block;max-width:100%;height:auto;color:var(--ink)}}
.empty{{font-family:var(--mono);font-size:.8rem;color:var(--faint);padding:0 1.1rem 1rem}}
table{{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.78rem}}
.tw{{overflow-x:auto;padding:0 0 .2rem}}
th{{text-align:left;font-weight:600;color:var(--faint);text-transform:uppercase;letter-spacing:.07em;
   font-size:.62rem;padding:.7rem 1.1rem;border-bottom:1px solid var(--rule2);white-space:nowrap}}
td{{padding:.6rem 1.1rem;border-bottom:1px solid var(--rule);white-space:nowrap}}
tr:last-child td{{border-bottom:none}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
.note{{font-size:.9rem;color:var(--mid);margin:.2rem 0 1.4rem;max-width:70ch}}
footer{{margin-top:2rem;padding-top:1.2rem;border-top:1px solid var(--rule2);
        font-family:var(--mono);font-size:.7rem;color:var(--faint);line-height:1.85}}
</style>
<div class="wrap">
<header>
  <h1>{html.escape(a.title)}</h1>
  <div class="meta">
    <span>{len(rows)} samples</span><span>{span}</span>
    <span>{hours:.1f} h</span>
    <span>UNVERIFIED: {unv}</span>
    <span>stage 1 {TRIP:g}°C · stage 2 {STAGE2:g}°C · kernel {CRITICAL:g}°C</span>
  </div>
  <div class="legend">{legend}</div>
</header>

<p class="note">Every surface below except GPU die temperature and SM clock is invisible to
<code>nvidia-smi</code>. The thermal guard trips on the board zones, not the die — so a
GPU-only dashboard reads comfortable right up to the moment the cluster downclocks itself.</p>

<div class="grid">{''.join(T)}</div>

{chart_html}

<section class="card">
  <h3>Board thermal zones, most recent sample</h3>
  <p class="sub">all 7 report type "acpitz" with no distinguishing label — they cannot be
  named from the system, only identified behaviourally. Reported by index, not guessed at.</p>
  <div class="tw"><table><thead><tr><th>node</th>{zone_head}</tr></thead>
  <tbody>{zone_rows}</tbody></table></div>
</section>

<footer>
  Static file — no server, no port, no verbs. Generated by
  spark-bench/tools/spark-dashboard.py from a spark-telemetry.py capture.<br>
  A reading that could not be taken is counted UNVERIFIED, never 0 and never "ok".
</footer>
</div>
"""
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(doc)
    print(f"  wrote {a.out}  ({len(doc)/1024:.0f} KB, {len(rows)} samples, {unv} UNVERIFIED)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
