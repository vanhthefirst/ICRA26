"""
Sketch-Prompted VLA — read the displacement probes (stdlib + matplotlib; no
simulator, no numpy).

Turns the probe rollouts into the one figure the position claim actually needs:
success against how far an object was moved from where LIBERO places it. A
dataset-level pair of numbers says "moving objects hurts"; a curve says how much,
from where, and whether the fall is graded or a cliff.

Two arms go on one axis. `--set` picks the scene folder a run was built from, so
the target arm and the destination arm can be drawn together even though they are
different scene sets:

    # target arm alone
    python scripts/analyze_displacement.py --run pi05_displacement

    # both arms, one panel — the figure the position argument reduces to
    python scripts/analyze_displacement.py \
        --run pi05_displacement  --set validation_set_displacement \
        --label "target moved" \
        --compare pi05_destination --compare-set validation_set_destination \
        --compare-label "destination moved"

Rates are computed per SCENE and then averaged, with the spread taken across
scenes at that radius. Rollouts within a scene share a pinned initial state and
come out close to all-or-nothing, so pooling the raw trials would report an
interval several times tighter than the evidence supports.

TERMINAL DISTANCE is reported alongside success, and in the destination arm it is
the sharper measurement of the two. `terminal_dist_xy` is how far the carried
bowl ended from the plate. If the policy has memorised a drop point rather than
reading the scene, then moving the plate by r leaves the bowl about r away from
it, so the terminal distance should track the displacement roughly one for one.
Success can only say "no"; this says where it went instead.

Writes `analysis.json` beside the run's results.csv and a figure under
`report/displacement/`.
"""

import os, csv, json, math, argparse, collections

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(_REPO, "outputs")
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


def load_meta(set_name):
    root = os.path.join(OUT_ROOT, set_name)
    meta = {}
    if os.path.isdir(root):
        for d in sorted(os.listdir(root)):
            p = os.path.join(root, d, "meta.json")
            if os.path.isfile(p):
                meta[d] = json.load(open(p))
    if not meta:
        raise SystemExit("no scenes in %s — run build_displacement_probe.py first" % root)
    return meta


def read_rows(run, meta):
    path = os.path.join(RUN_ROOT, run, "results.csv")
    if not os.path.isfile(path):
        raise SystemExit("no results at %s" % path)
    rows = [r for r in csv.DictReader(open(path))
            if r.get("skipped") != "True" and r["dir"] in meta]
    if not rows:
        raise SystemExit("no rows of %s match the scenes in the given --set; "
                         "the run and the scene set do not belong together" % run)
    return rows


def per_scene_rate(rows, key="success_sustained"):
    hits = collections.defaultdict(list)
    for r in rows:
        hits[r["dir"]].append(r[key] == "True")
    return {k: sum(v) / len(v) for k, v in hits.items() if v}, \
           {k: len(v) for k, v in hits.items() if v}


def per_scene_dist(rows):
    """Mean terminal distance from the destination, per scene. Rollouts that
    never lifted anything have no terminal distance and are left out rather than
    counted as zero."""
    d = collections.defaultdict(list)
    for r in rows:
        v = r.get("terminal_dist_xy")
        if v not in (None, "", "None"):
            d[r["dir"]].append(float(v))
    return {k: sum(v) / len(v) for k, v in d.items() if v}


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def summarise(rates, meta, dists=None):
    by_r = collections.defaultdict(list)
    by_rd = collections.defaultdict(list)
    by_r_dist = collections.defaultdict(list)
    for d, v in rates.items():
        m = meta[d]
        by_r[m["radius_m"]].append(v)
        by_rd[(m["radius_m"], m["direction"])].append(v)
        if dists and d in dists:
            by_r_dist[m["radius_m"]].append(dists[d])
    rows = []
    for r in sorted(by_r):
        mu, se = mean_se(by_r[r])
        row = dict(radius_m=r, n_scenes=len(by_r[r]),
                   success=round(mu, 4), se=round(se, 4))
        if by_r_dist.get(r):
            dm, dse = mean_se(by_r_dist[r])
            row.update(terminal_dist_m=round(dm, 4), terminal_dist_se=round(dse, 4))
        rows.append(row)
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


def report(name, rows, by_r, by_rd, n_scenes, ns):
    print("== %s ==" % name)
    print("scenes scored: %d   rollouts/scene: %s" % (n_scenes, sorted(set(ns.values()))))
    print()
    has_d = any("terminal_dist_m" in r for r in rows)
    head = "%-10s %8s %10s %8s" % ("offset cm", "scenes", "success", "se")
    if has_d:
        head += " %14s %10s" % ("term dist cm", "dist/offset")
    print(head)
    for r in rows:
        line = "%-10.1f %8d %9.1f%% %7.1f" % (r["radius_m"] * 100, r["n_scenes"],
                                              100 * r["success"], 100 * r["se"])
        if has_d and "terminal_dist_m" in r:
            ratio = (r["terminal_dist_m"] / r["radius_m"]) if r["radius_m"] else float("nan")
            line += " %13.1f %10s" % (100 * r["terminal_dist_m"],
                                      "--" if r["radius_m"] == 0 else "%.2f" % ratio)
        print(line)
    d, a, b = cliff(rows)
    print("\nlargest single-step drop: %.1f points between %.0f and %.0f cm"
          % (100 * d, 100 * a, 100 * b))
    print("origin -> furthest       : %.1f%% -> %.1f%% (%.1f points)"
          % (100 * rows[0]["success"], 100 * rows[-1]["success"],
             100 * (rows[0]["success"] - rows[-1]["success"])))

    print("\nby direction (success %, scenes in brackets)")
    dirs = sorted({dd for (_, dd) in by_rd})
    print("%-10s %s" % ("offset cm", "  ".join("%-12s" % dd for dd in dirs)))
    for r in sorted(by_r):
        cells = []
        for dd in dirs:
            v = by_rd.get((r, dd))
            cells.append("%-12s" % ("--" if not v else
                                    "%.0f%% [%d]" % (100 * sum(v) / len(v), len(v))))
        print("%-10.1f %s" % (100 * r, "  ".join(cells)))
    print()
    return (d, a, b)


def plot(rows, rates, meta, out_path, label, title, compare=None):
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
    # Up-and-right, not straight up: a descending trend leaves the space
    # above-right of each point empty while the space directly above it holds
    # that point's error-bar cap.
    for xi, yi in zip(x, y):
        ax.annotate("%.0f%%" % yi, (xi, yi), textcoords="offset points",
                    xytext=(7, 7), ha="left", fontsize=8, color=INK,
                    clip_on=False)

    xmax = max(x)
    if compare:
        crows, clabel, crates, cmeta = compare
        for d, v in sorted(crates.items()):
            m = cmeta[d]
            ax.plot(m["radius_m"] * 100, v * 100, MARKERS.get(m["direction"], "o"),
                    ms=5, mfc="none", mec=MUTED, mew=1.0, alpha=0.4, zorder=2,
                    label="_scene")
        cx = [r["radius_m"] * 100 for r in crows]
        cy = [r["success"] * 100 for r in crows]
        ce = [r["se"] * 100 for r in crows]
        ax.errorbar(cx, cy, yerr=ce, color=ACCENT_2, lw=2.0, marker="s", ms=7,
                    capsize=3, zorder=3, label=clabel)
        # Below-LEFT for the compare series. A second curve is only worth drawing
        # when it descends where the first does not, so the space below-right of
        # each of its points is where the rest of that curve goes. The origin is
        # skipped because both arms share it and the two labels would collide.
        for xi, yi in zip(cx, cy):
            if xi == 0:
                continue
            ax.annotate("%.0f%%" % yi, (xi, yi), textcoords="offset points",
                        xytext=(-8, -13), ha="right", fontsize=8, color=INK,
                        clip_on=False)
        xmax = max(xmax, max(cx))

    ax.set_xlabel("displacement from the LIBERO position (cm)", color=INK)
    ax.set_ylabel("success rate (%)", color=INK)
    ax.set_ylim(-4, 108)
    ax.set_xlim(-0.6, xmax + 1.2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    shapes = "  ".join("%s %s" % (MARKERS[d], d) for d in sorted(seen) if d in MARKERS)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK,
              loc="lower left" if y[0] > 50 and not compare else "center left")
    fig.text(0.01, 0.005, "open markers: one scene   " + shapes,
             fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run-id under outputs/rollouts/")
    ap.add_argument("--set", dest="set_name", default="validation_set_displacement",
                    help="scene folder under outputs/ the run was built from")
    ap.add_argument("--compare", default=None, help="second run-id, e.g. the other arm")
    ap.add_argument("--compare-set", dest="compare_set", default=None,
                    help="scene folder for --compare (defaults to --set)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--compare-label", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None, help="figure filename under report/displacement/")
    args = ap.parse_args()

    meta = load_meta(args.set_name)
    rows_csv = read_rows(args.run, meta)
    rates, ns = per_scene_rate(rows_csv)
    dists = per_scene_dist(rows_csv)
    rows, by_r, by_rd = summarise(rates, meta, dists)
    d, a, b = report(args.label or args.run, rows, by_r, by_rd, len(rates), ns)

    out = dict(run=args.run, scene_set=args.set_name, n_scenes=len(rates), rows=rows,
               by_direction={"%.2f|%s" % k: v for k, v in by_rd.items()},
               largest_drop_points=round(100 * d, 2),
               largest_drop_between_cm=[round(100 * a, 1), round(100 * b, 1)])

    compare = None
    if args.compare:
        cmeta = load_meta(args.compare_set or args.set_name)
        crows_csv = read_rows(args.compare, cmeta)
        crates, cns = per_scene_rate(crows_csv)
        cdists = per_scene_dist(crows_csv)
        crows, cby_r, cby_rd = summarise(crates, cmeta, cdists)
        cd, ca, cb = report(args.compare_label or args.compare, crows, cby_r, cby_rd,
                            len(crates), cns)
        compare = (crows, args.compare_label or args.compare, crates, cmeta)
        out["compare"] = dict(run=args.compare,
                              scene_set=args.compare_set or args.set_name,
                              n_scenes=len(crates), rows=crows,
                              largest_drop_points=round(100 * cd, 2),
                              largest_drop_between_cm=[round(100 * ca, 1),
                                                       round(100 * cb, 1)])

    apath = os.path.join(RUN_ROOT, args.run, "analysis.json")
    json.dump(out, open(apath, "w"), indent=2)
    print("wrote", apath)

    default_title = ("Same task, same caption, nothing added — only one object moved"
                     if compare else
                     "Same task, same caption, nothing added — only the bowl moved")
    plot(rows, rates, meta,
         os.path.join(FIG_DIR, args.out or
                      ("fig_displacement_both.png" if compare else "fig_displacement.png")),
         args.label or args.run, args.title or default_title, compare)


if __name__ == "__main__":
    main()
