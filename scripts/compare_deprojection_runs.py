"""
Sketch-Prompted VLA — side-by-side headline gap under the two pixel->world
deprojection rungs (issue 2).

Reads two rollout run directories (by convention full_run_plane and
full_run_depth, produced by the same four rollout_sketch.py invocations
under --deproject plane / depth) and prints the auto-minus-text_only gap under
each, overall and per suite x tier.

Reports grasped_correct beside grasped_any everywhere. grasped_any counts
lifting ANY object, so a baseline guessing uniformly among candidates can beat a
policy that aims correctly and misses; read alone it inverts the result. See
ROLLOUT.md, "grasped_any is not a targeting metric".

    python scripts/compare_deprojection_runs.py
    python scripts/compare_deprojection_runs.py --runs full_run_plane full_run_depth
"""

import argparse
import collections
import csv
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.path.join(_REPO, "outputs", "rollouts")
SUCCESS_FIELD = "success_sustained"


def load(run_id):
    path = os.path.join(RUN_ROOT, run_id, "results.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r.get("skipped") != "True"]


def pct(rows, field):
    rows = list(rows)
    if not rows:
        return None
    return 100.0 * sum(1 for r in rows if r.get(field) == "True") / len(rows)


def fmt(v):
    return "  n/a" if v is None else f"{v:5.1f}"


BASELINE = "text_only"


def gap(rows, field, cond="auto"):
    """`cond` minus text_only, in percentage points."""
    a = pct([r for r in rows if r["condition"] == cond], field)
    t = pct([r for r in rows if r["condition"] == BASELINE], field)
    if a is None or t is None:
        return None, a, t
    return a - t, a, t


def sketch_conditions(runs, explicit=None):
    """Which sketch conditions to report against the baseline.

    Defaults to every non-baseline condition present, `auto` first. Written for
    the human depth pass: `human_r1` carries auto, human:<name> and
    human_consensus alongside text_only, and reporting only `auto` — which is
    what this script used to do — would silently drop the two columns the run
    exists to compare."""
    if explicit:
        return explicit.split(",")
    present = {r["condition"] for rows in runs.values() for r in rows} - {BASELINE}
    return (["auto"] if "auto" in present else []) + sorted(present - {"auto"})


def upper_bound_flag(run_id, rows):
    cfg = os.path.join(RUN_ROOT, run_id, "run_config.json")
    if os.path.exists(cfg):
        c = json.load(open(cfg))
        if c.get("upper_bound"):
            return True
    return any(r.get("deproject") == "depth" for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs=2, default=["full_run_plane", "full_run_depth"])
    ap.add_argument("--conditions", default=None,
                    help="comma list of sketch conditions to report against "
                         "text_only; default every non-baseline condition in "
                         "the two runs, auto first")
    args = ap.parse_args()

    runs = {}
    for rid in args.runs:
        rows = load(rid)
        if rows is None:
            print(f"[skip] no results.csv for run_id {rid!r}")
            continue
        runs[rid] = rows

    conds = sketch_conditions(runs, args.conditions)
    primary = conds[0] if conds else "auto"

    for rid, rows in runs.items():
        ub = upper_bound_flag(rid, rows)
        tag = "  ** UPPER BOUND (privileged depth) **" if ub else "  (non-privileged)"
        print(f"\n{'='*78}\nrun {rid}   n={len(rows)} rows{tag}")

        for cond in conds:
            g, a, t = gap(rows, SUCCESS_FIELD, cond)
            print(f"  HEADLINE {SUCCESS_FIELD}: {cond:<18s} {fmt(a)}%  "
                  f"text_only {fmt(t)}%  gap {'n/a' if g is None else f'{g:+.1f}pp'}")
        for field in ("grasped_any", "correct_instance_grasped", "correct_destination"):
            label = "grasped_correct" if field == "correct_instance_grasped" else field
            for cond in conds:
                g2, a2, t2 = gap(rows, field, cond)
                print(f"  {label:>22s}: {cond:<18s} {fmt(a2)}%  text_only {fmt(t2)}%  "
                      f"gap {'n/a' if g2 is None else f'{g2:+.1f}pp'}")

        if len(conds) > 1:
            print(f"\n  suite/tier table below is `{primary}` only "
                  f"(of {', '.join(conds)}) -- one table per condition would be "
                  f"unreadable; pass --conditions to change which one.")
        by = collections.defaultdict(list)
        for r in rows:
            by[(r["suite"], r["tier"])].append(r)
        print(f"\n  {'suite/tier':22s} {'succ_auto':>9s} {'succ_text':>9s} {'gap':>7s} "
              f"{'gAny_a':>7s} {'gAny_t':>7s} {'gCorr_a':>8s} {'gCorr_t':>8s}")
        for k in sorted(by):
            rs = by[k]
            g3, a3, t3 = gap(rs, SUCCESS_FIELD, primary)
            _, ga, gt = gap(rs, "grasped_any", primary)
            _, ca, ct = gap(rs, "correct_instance_grasped", primary)
            print(f"  {k[0]+'/'+k[1]:22s} {fmt(a3):>9s} {fmt(t3):>9s} "
                  f"{('n/a' if g3 is None else f'{g3:+.1f}'):>7s} "
                  f"{fmt(ga):>7s} {fmt(gt):>7s} {fmt(ca):>8s} {fmt(ct):>8s}")

    if len(runs) == 2:
        a_id, b_id = args.runs
        print(f"\n{'='*78}\nplane -> depth, headline gap (condition minus text_only)")
        print(f"  {'condition':<20s} {a_id:>18s} {b_id:>18s}")
        for cond in conds:
            ga = gap(runs[a_id], SUCCESS_FIELD, cond)[0] if a_id in runs else None
            gb = gap(runs[b_id], SUCCESS_FIELD, cond)[0] if b_id in runs else None
            print(f"  {cond:<20s} {('n/a' if ga is None else f'{ga:+.1f}pp'):>18s} "
                  f"{('n/a' if gb is None else f'{gb:+.1f}pp'):>18s}")
        print("\n  The plane column is the one to quote when quoting one number;\n"
              "  the depth column is an upper bound (privileged z).")


if __name__ == "__main__":
    main()
