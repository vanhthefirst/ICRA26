"""
Sketch-Prompted VLA — read the target-displacement probe (stdlib + matplotlib;
no simulator, no numpy).

Turns the probe's rollouts into the one figure the position claim actually needs:
success against how far the target was moved from where LIBERO places it. A
dataset-level pair of numbers says "moving objects hurts"; a curve says how much,
from where, and whether the fall is graded or a cliff.

Rates are computed per SCENE and then averaged, with the spread taken across
scenes at that radius. Rollouts within a scene share a pinned initial state and
come out close to all-or-nothing, so pooling the raw trials would report an
interval several times tighter than the evidence supports.

    python scripts/analyze_displacement.py --run pi05_displacement
    python scripts/analyze_displacement.py --run pi05_displacement \
        --compare pi05_displacement_univla        # once a second policy exists

Writes `analysis.json` beside the run's results.csv and
`report/displacement/fig_displacement.png`.
"""

import os, csv, json, math, argparse, collections

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SET_ROOT = os.path.join(_REPO, "outputs", "validation_set_displacement")
RUN_ROOT = os.path.join(_REPO, "outputs", "rollouts")
FIG_DIR = os.path.join(_REPO, "report", "displacement")

# Validated light-mode categorical pair (dataviz validator: all checks pass,
# adjacent CVD dE 29.4). One accent carries the trend; direction is encoded by
# MARKER SHAPE, not by a fifth and sixth hue, so the figure stays readable in
# greyscale print and under any CVD.
ACCENT = "#1f6feb"
ACCENT_2 = "#d1682a"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e4e4e2"
MARKERS = {"origin": "o", "xpos": "^", "xneg": "v", "ypos": "s", "yneg": "D"}


def load_meta():
    meta = {}
    for d in sorted(os.listdir(SET_ROOT)):
        p = os.path.join(SET_ROOT, d, "meta.json")
        if os.path.isfile(p):
            meta[d] = json.load(open(p))
    if not meta:
        raise SystemExit("no scenes in %s — run build_displacement_probe.py first" % SET_ROOT)
    return meta


def per_scene_rate(run, meta, key="success_sustained"):
    path = os.path.join(RUN_ROOT, run, "results.csv")
    if not os.path.isfile(path):
        raise SystemExit("no results at %s" % path)
    hits = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r.get("skipped") == "True" or r["dir"] not in meta:
            continue
        hits[r["dir"]].append(r[key] == "True")
    return {k: sum(v) / len(v) for k, v in hits.items() if v}, \
           {k: len(v) for k, v in hits.items() if v}


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def summarise(rates, meta):
    by_r = collections.defaultdict(list)
    by_rd = collections.defaultdict(list)
    for d, v in rates.items():
        m = meta[d]
        by_r[m["radius_m"]].append(v)
        by_rd[(m["radius_m"], m["direction"])].append(v)
    radii = sorted(by_r)
    rows = []
    for r in radii:
        mu, se = mean_se(by_r[r])
        rows.append(dict(radius_m=r, n_scenes=len(by_r[r]),
                         success=round(mu, 4), se=round(se, 4)))
    return rows, by_r, by_rd


def cliff(rows):
    """Largest single-step drop, and where it happens. A graded decline and a
    cliff imply different mechanisms — smooth degradation looks like a policy
    interpolating badly, a cliff looks like one that only ever learned a spot."""
    worst = None
    for a, b in zip(rows, rows[1:]):
        d = a["success"] - b["success"]
        if worst is None or d > worst[0]:
            worst = (d, a["radius_m"], b["radius_m"])
    return worst


def plot(rows, rates, meta, out_path, label, compare=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    ax.set_facecolor("white")

    # individual scenes behind the trend: the reader sees the spread the error
    # bars summarise, and can tell 4 scenes from 40 at a glance.
    seen = set()
    for d, v in sorted(rates.items()):
        m = meta[d]
        ax.plot(m["radius_m"] * 100, v * 100, MARKERS.get(m["direction"], "o"),
                ms=5, mfc="none", mec=MUTED, mew=1.0, alpha=0.75, zorder=2,
                label="_scene")
        seen.add(m["direction"])

    x = [r["radius_m"] * 100 for r in rows]
    y = [r["success"] * 100 for r in rows]
    e = [r["se"] * 100 for r in rows]
    ax.errorbar(x, y, yerr=e, color=ACCENT, lw=2.0, marker="o", ms=7,
                capsize=3, zorder=3, label=label)
    # Up-and-right, not straight up: the trend descends left to right, so the
    # space above-right of each point is empty while the space directly above it
    # holds that point's error-bar cap.
    for xi, yi in zip(x, y):
        ax.annotate("%.0f%%" % yi, (xi, yi), textcoords="offset points",
                    xytext=(7, 7), ha="left", fontsize=8, color=INK,
                    clip_on=False)

    if compare:
        crows, clabel = compare
        cx = [r["radius_m"] * 100 for r in crows]
        cy = [r["success"] * 100 for r in crows]
        ce = [r["se"] * 100 for r in crows]
        ax.errorbar(cx, cy, yerr=ce, color=ACCENT_2, lw=2.0, marker="s", ms=7,
                    capsize=3, zorder=3, label=clabel)

    ax.set_xlabel("target displacement from its LIBERO position (cm)", color=INK)
    ax.set_ylabel("success rate (%)", color=INK)
    ax.set_ylim(-4, 108)
    ax.set_xlim(-0.6, max(x) + 1.2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    shapes = "  ".join("%s %s" % (MARKERS[d], d) for d in sorted(seen) if d in MARKERS)
    ax.set_title("Same task, same caption, nothing added — only the bowl moved",
                 color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK,
              loc="lower left" if y[0] > 50 else "upper right")
    fig.text(0.01, 0.005, "open markers: one scene   " + shapes,
             fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run-id under outputs/rollouts/")
    ap.add_argument("--compare", default=None, help="second run-id, e.g. another policy")
    ap.add_argument("--label", default=None)
    ap.add_argument("--compare-label", default=None)
    args = ap.parse_args()

    meta = load_meta()
    rates, ns = per_scene_rate(args.run, meta)
    rows, by_r, by_rd = summarise(rates, meta)

    print("scenes scored: %d   rollouts/scene: %s"
          % (len(rates), sorted(set(ns.values()))))
    print()
    print("%-10s %8s %10s %8s" % ("offset cm", "scenes", "success", "se"))
    for r in rows:
        print("%-10.1f %8d %9.1f%% %7.1f" % (r["radius_m"] * 100, r["n_scenes"],
                                             100 * r["success"], 100 * r["se"]))
    d, a, b = cliff(rows)
    print("\nlargest single-step drop: %.1f points between %.0f and %.0f cm"
          % (100 * d, 100 * a, 100 * b))
    print("origin -> furthest       : %.1f%% -> %.1f%% (%.1f points)"
          % (100 * rows[0]["success"], 100 * rows[-1]["success"],
             100 * (rows[0]["success"] - rows[-1]["success"])))

    print("\nby direction (success %, scenes in brackets)")
    dirs = sorted({d for (_, d) in by_rd})
    print("%-10s %s" % ("offset cm", "  ".join("%-12s" % d for d in dirs)))
    for r in sorted(by_r):
        cells = []
        for dd in dirs:
            v = by_rd.get((r, dd))
            cells.append("%-12s" % ("--" if not v else
                                    "%.0f%% [%d]" % (100 * sum(v) / len(v), len(v))))
        print("%-10.1f %s" % (100 * r, "  ".join(cells)))

    out = dict(run=args.run, n_scenes=len(rates), rows=rows,
               by_direction={"%.2f|%s" % k: v for k, v in by_rd.items()},
               largest_drop_points=round(100 * d, 2),
               largest_drop_between_cm=[round(100 * a, 1), round(100 * b, 1)])
    apath = os.path.join(RUN_ROOT, args.run, "analysis.json")
    json.dump(out, open(apath, "w"), indent=2)
    print("\nwrote", apath)

    compare = None
    if args.compare:
        crates, _ = per_scene_rate(args.compare, meta)
        crows, _, _ = summarise(crates, meta)
        compare = (crows, args.compare_label or args.compare)
    plot(rows, rates, meta,
         os.path.join(FIG_DIR, "fig_displacement.png"),
         args.label or args.run, compare)


if __name__ == "__main__":
    main()
