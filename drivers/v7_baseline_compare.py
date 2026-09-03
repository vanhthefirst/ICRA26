#!/usr/bin/env python3
"""Put the baseline and the fine-tune on one table, on one set of layouts.

THE QUESTION THIS CLOSES
    The gap between this repo's baseline (stock pi0.5-LIBERO, 40.3% explicit /
    36.5% ambiguous) and the SketchPromptVLA fine-tune (2.44% at
    checkpoint-29999, 85.5% at rg_v7_paired/2999) has never been a single
    measurement. The two numbers come from different scene sets, different
    episode counts, different captions and different success criteria. Three
    causes of the ORIGINAL gap were found and fixed -- a 180-degree orientation
    fault worth the entire score, a sketch/frame augmentation desynchronisation,
    and a corpus in which one layout carried one sketch -- but "fixed" was never
    demonstrated on a shared design.

    This script requires a shared design. It refuses to score the fine-tune
    without the baseline arm, for the same reason score_referent_following.py
    refuses to summarise a real arm without its swap: the single arm reads as a
    result and is not one.

WHAT IT REPORTS
    task success and P(grasp bowl_2) per arm, with Wilson intervals, and then
    the three differences that carry the argument:

      ceiling gap    pi05+explicit  ->  v7+real     is the fine-tune competitive
                                                     with a policy that is TOLD
                                                     the referent?
      floor gap      pi05+stored    ->  v7+real     does the sketch buy anything
                                                     over the same task with the
                                                     referent withheld?
      sketch dose    v7 blank/real/swap on p_bowl2   does the mark MOVE the
                                                     referent, or only decorate?

    The dose is the only one of the three that can distinguish grounding from a
    spatial habit, and it is the one a real arm alone hides.

USAGE
    v7_baseline_compare.py --v7-real R.json [R2.json ...] --v7-swap S.json ...
                           --v7-blank B.json ... --pi05-explicit E.json ...
                           --pi05-stored T.json ... [--out compare.json]

    A merged artifact (v7_paired_step2999_merged_all400.json) may be passed to
    --v7-real and --v7-swap; rows are filtered by their own `mode` field, so
    passing it to both is correct and does not double count.
"""
import argparse
import json
import math
import pathlib
from collections import defaultdict

BOWL2 = "akita_black_bowl_2"


def wilson(k, n, z=1.96):
    """Wilson score interval. Normal-approximation intervals are wrong at the
    rates this project reports -- a 0/20 task is not [0, 0]."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def load(paths, mode=None):
    """Rows from one or more evaluator outputs, optionally filtered by mode."""
    rows = []
    for path in paths or []:
        blob = json.load(open(path))
        for row in blob["rows"]:
            if mode is None or row.get("mode") == mode:
                row = dict(row)
                row["_file"] = str(path)
                rows.append(row)
    return rows


def tkey(task):
    return int(task[1:]) if task[1:].isdigit() else 0


class Arm:
    def __init__(self, label, rows, note="", score_success=True):
        self.label = label
        self.rows = rows
        self.note = note
        # The swap arm stops at the first bowl lift and its BDDL goal still
        # describes the ORIGINAL referent, so its task-success number is a
        # truncated episode scored against the wrong goal. It is not a rate;
        # it is not printed. p_bowl2 is the swap arm's only score.
        self.score_success = score_success
        self.n = len(rows)
        self.success = sum(bool(r["success"]) for r in rows)
        self.sustained = sum(bool(r.get("sustained_success", r["success"])) for r in rows)
        self.bowl2 = sum(r["grasped"] == BOWL2 for r in rows)
        self.nograsp = sum(r["grasped"] in (None, "", "none") for r in rows)
        self.tasks = sorted({r["task"] for r in rows}, key=tkey)
        self.windows = sorted({r.get("success_window", 1) for r in rows})

    def rate(self, field):
        return getattr(self, field) / self.n if self.n else float("nan")

    def ci(self, field):
        return wilson(getattr(self, field), self.n)

    def per_task(self, field):
        out = {}
        for task in self.tasks:
            rows = [r for r in self.rows if r["task"] == task]
            if field == "success":
                k = sum(bool(r["success"]) for r in rows)
            else:
                k = sum(r["grasped"] == BOWL2 for r in rows)
            out[task] = (k, len(rows))
        return out


def diff_line(name, a, b, field):
    """b minus a, with a pooled-normal interval on the difference."""
    pa, pb = a.rate(field), b.rate(field)
    se = math.sqrt(pa * (1 - pa) / a.n + pb * (1 - pb) / b.n) if a.n and b.n else float("nan")
    d = pb - pa
    lo, hi = d - 1.96 * se, d + 1.96 * se
    crosses = lo <= 0 <= hi
    return ("%-28s %+7.3f  [%+.3f, %+.3f]  %s"
            % (name, d, lo, hi, "consistent with zero" if crosses else "excludes zero"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v7-real", nargs="+", required=True)
    ap.add_argument("--v7-swap", nargs="+", required=True)
    ap.add_argument("--v7-blank", nargs="+")
    ap.add_argument("--pi05-explicit", nargs="+", required=True,
                    help="stock pi0.5-LIBERO, BDDL caption -- the ceiling arm")
    ap.add_argument("--pi05-stored", nargs="+", required=True,
                    help="stock pi0.5-LIBERO, referent-free caption -- the floor arm")
    ap.add_argument("--out")
    args = ap.parse_args()

    arms = {
        "pi05+explicit": Arm("pi05+explicit", load(args.pi05_explicit),
                             "baseline, told the referent"),
        "pi05+stored": Arm("pi05+stored", load(args.pi05_stored),
                           "baseline, referent withheld"),
        "v7+blank": Arm("v7+blank", load(args.v7_blank, "blank"),
                        "fine-tune, no sketch"),
        "v7+real": Arm("v7+real", load(args.v7_real, "real"),
                       "fine-tune, sketch on bowl_1"),
        "v7+swap": Arm("v7+swap", load(args.v7_swap, "swap"),
                       "fine-tune, sketch on bowl_2", score_success=False),
    }
    arms = {k: v for k, v in arms.items() if v.n}
    for required in ("pi05+explicit", "pi05+stored", "v7+real", "v7+swap"):
        if required not in arms:
            raise SystemExit("ABORT: no rows for %s. This table is not readable "
                             "without it." % required)

    # A shared design is the entire point; check it rather than assume it.
    layouts = {name: set(arm.tasks) for name, arm in arms.items()}
    shared = set.intersection(*layouts.values())
    mismatched = {n: sorted(t - shared, key=tkey) for n, t in layouts.items() if t - shared}
    windows = {n: a.windows for n, a in arms.items()}

    print("ARMS -- all on the same paired layouts, demos and step budget")
    print("%-16s%6s%10s%22s%10s%20s  %s"
          % ("arm", "n", "success", "95% CI", "p_bowl2", "95% CI", "what it is"))
    print("-" * 120)
    for name, arm in arms.items():
        blo, bhi = arm.ci("bowl2")
        if arm.score_success:
            slo, shi = arm.ci("success")
            success = "%10.3f    [%.3f, %.3f]" % (arm.rate("success"), slo, shi)
        else:
            success = "%28s" % "n/a (episode truncated)"
        print("%-16s%6d%s%10.3f    [%.3f, %.3f]  %s"
              % (name, arm.n, success, arm.rate("bowl2"), blo, bhi, arm.note))

    print()
    print("THE THREE DIFFERENCES  (task success unless noted)")
    print(diff_line("ceiling gap  expl -> real", arms["pi05+explicit"], arms["v7+real"], "success"))
    print(diff_line("floor gap    stor -> real", arms["pi05+stored"], arms["v7+real"], "success"))
    if "v7+blank" in arms:
        print(diff_line("sketch, closed loop", arms["v7+blank"], arms["v7+real"], "success"))
    print(diff_line("DOSE  p_bowl2 real->swap", arms["v7+real"], arms["v7+swap"], "bowl2"))
    if "v7+blank" in arms:
        print(diff_line("DOSE  p_bowl2 blank->swap", arms["v7+blank"], arms["v7+swap"], "bowl2"))
        print(diff_line("DOSE  p_bowl2 blank->real", arms["v7+blank"], arms["v7+real"], "bowl2"))
    print(diff_line("prior  pi05 vs v7 blank" if "v7+blank" in arms
                    else "prior  pi05 vs v7 real",
                    arms["pi05+stored"], arms.get("v7+blank", arms["v7+real"]), "bowl2"))

    print()
    print("PER-LAYOUT TASK SUCCESS")
    header = "%-6s" % "task" + "".join("%16s" % n for n in arms)
    print(header)
    print("-" * len(header))
    per = {n: a.per_task("success" if a.score_success else "bowl2")
           for n, a in arms.items()}
    for task in sorted(shared, key=tkey):
        line = "%-6s" % task
        for name in arms:
            k, n = per[name].get(task, (0, 0))
            line += "%16s" % ("%d/%d" % (k, n) if n else "-")
        print(line)
    print("%-6s%s" % ("", "".join("%16s" % ("p_bowl2" if not a.score_success else "")
                                  for a in arms.values())))

    print()
    print("READING GUARDS")
    if mismatched:
        print("  LAYOUTS DIFFER between arms -- the table above is NOT like-for-like:")
        for name, extra in mismatched.items():
            print("    %s carries %s that another arm does not" % (name, ",".join(extra)))
    else:
        print("  layouts        : identical across arms (%s)"
              % ",".join(sorted(shared, key=tkey)))
    if len({tuple(w) for w in windows.values()}) > 1:
        print("  SUCCESS WINDOWS DIFFER -- a windowed rate and an instantaneous one "
              "are different measurements: %s" % windows)
    else:
        print("  success window : %s across all arms"
              % ",".join(str(w) for w in next(iter(windows.values()))))
    print("  the ceiling gap is the mismatch question. The dose is the grounding")
    print("  question. A fine-tune can close the first and still fail the second.")

    if args.out:
        blob = {
            "arms": {name: {
                "n": arm.n,
                "success": round(arm.rate("success"), 4),
                "success_ci": [round(v, 4) for v in arm.ci("success")],
                "sustained_success": round(arm.rate("sustained"), 4),
                "p_bowl2": round(arm.rate("bowl2"), 4),
                "p_bowl2_ci": [round(v, 4) for v in arm.ci("bowl2")],
                "no_grasp": round(arm.rate("nograsp"), 4),
                "success_windows": arm.windows,
                "tasks": arm.tasks,
                "files": sorted({r["_file"] for r in arm.rows}),
            } for name, arm in arms.items()},
            "shared_layouts": sorted(shared, key=tkey),
            "layout_mismatch": mismatched,
            "like_for_like": not mismatched and len({tuple(w) for w in windows.values()}) == 1,
        }
        pathlib.Path(args.out).write_text(json.dumps(blob, indent=1))
        print("\nwrote", args.out)


if __name__ == "__main__":
    main()
