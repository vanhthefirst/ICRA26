#!/usr/bin/env python3
"""Turn the two anchored V7 run directories into the write-up, in one command.

WHAT THIS IS FOR
    v7_anchored_arm.sh already calls score_referent_following.py, which answers
    the GROUNDING question: does moving the mark move the grasp onto the marked
    object. It does not answer the MISMATCH question: how does the fine-tune
    compare with the baseline where the baseline was measured. Answering that by
    hand means remembering that the anchored arms run on 26 of the 37 scenes and
    that the published 40.3 / 36.5 are over all 37 -- and reading a 26-scene
    number against a 37-scene baseline is the exact error this branch exists to
    stop. So the comparison is computed, not recalled.

    Every comparator is restricted to the SAME scenes the V7 arms ran, off the
    CSVs committed in this repo. Nothing here is typed in from a runbook.

    python drivers/v7_anchored_report.py \
        --real outputs/rollouts/sketchvla_rg_v7_ambiguous_sketch \
        --swap outputs/rollouts/sketchvla_rg_v7_ambiguous_swap
"""
import argparse
import csv
import math
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_LIST = os.path.join(REPO, "outputs", "validation_set_spatial", "swap_scene_list.txt")
COMPARATORS = [
    ("pi05 baseline, explicit", "pi05_anchored_explicit_518"),
    ("pi05 baseline, ambiguous", "pi05_anchored_ambiguous_518"),
    ("overlay_v6, ambiguous", "sketchvla_input_overlay_ambiguous_sketch"),
]
FIELDS = [("success_sustained", "task success"),
          ("correct_instance_grasped", "correct object")]


def wilson(k, n, z=1.96):
    if not n:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def read(run_dir):
    path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(path):
        raise SystemExit("no results.csv in %s" % run_dir)
    return [r for r in csv.DictReader(open(path)) if r.get("skipped") != "True"]


def rate(rows, field):
    rows = [r for r in rows if r.get(field) not in (None, "")]
    k = sum(r[field] == "True" for r in rows)
    return k, len(rows)


def diff(k1, n1, k2, n2):
    """k2/n2 minus k1/n1, with a normal interval on the difference."""
    if not (n1 and n2):
        return float("nan"), float("nan"), float("nan")
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p2 - p1
    return d, d - 1.96 * se, d + 1.96 * se


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True, help="V7 real-sketch run directory")
    ap.add_argument("--swap", required=True, help="V7 swap-sketch run directory")
    ap.add_argument("--rollouts-root", default=os.path.join(REPO, "outputs", "rollouts"))
    ap.add_argument("--label", default="V7 rg_v7_paired/2999")
    ap.add_argument("--skip-scorer", action="store_true",
                    help="do not shell out to score_referent_following.py")
    args = ap.parse_args()

    real, swap = read(args.real), read(args.swap)
    real_scenes = {r["dir"] for r in real}
    swap_scenes = {r["dir"] for r in swap}

    print("=" * 78)
    print("ANCHORED ARM -- %s" % args.label)
    print("=" * 78)

    # ---- the guards, before any number is shown ---------------------------
    problems = []
    if real_scenes != swap_scenes:
        problems.append("the two arms cover DIFFERENT scenes (%d real, %d swap, "
                        "%d shared) -- the pairing is not a pairing"
                        % (len(real_scenes), len(swap_scenes),
                           len(real_scenes & swap_scenes)))
    if os.path.exists(SCENE_LIST):
        want = {t.split("/", 1)[1] for t in
                open(SCENE_LIST).read().strip().split(",") if t}
        if real_scenes != want:
            extra = sorted(real_scenes - want)
            missing = sorted(want - real_scenes)
            problems.append(
                "the run does not match swap_scene_list.txt (%d expected): "
                "%d extra %s, %d missing %s"
                % (len(want), len(extra), extra[:4], len(missing), missing[:4]))
            if extra:
                problems.append("EXTRA scenes are ones whose swap ring was "
                                "rejected as clipped or ambiguous -- their swap "
                                "rows do not mean what the metric assumes")
    else:
        problems.append("no swap_scene_list.txt -- run "
                        "scripts/build_anchored_swap_sketches.py")
    for p in problems:
        print("  WARNING: %s" % p)
    if not problems:
        print("  scenes: %d, identical across both arms and matching the "
              "audited list" % len(real_scenes))
    print("  rows  : %d real, %d swap" % (len(real), len(swap)))
    print()

    # ---- the mismatch question --------------------------------------------
    print("AGAINST THE BASELINE, ON THESE SCENES ONLY")
    print("  Every comparator is recomputed on the same %d scenes. The published"
          % len(real_scenes))
    print("  40.3%/36.5% are over all 37 and are NOT the number to use here.")
    print()
    for field, label in FIELDS:
        k, n = rate(real, field)
        lo, hi = wilson(k, n)
        print("  %s" % label.upper())
        print("    %-28s%8.4f  [%.3f, %.3f]  n=%d"
              % (args.label, k / n if n else float("nan"), lo, hi, n))
        for name, run in COMPARATORS:
            path = os.path.join(args.rollouts_root, run)
            if not os.path.isdir(path):
                print("    %-28s(missing %s)" % (name, run))
                continue
            rows = [r for r in read(path) if r["dir"] in real_scenes]
            ck, cn = rate(rows, field)
            if not cn:
                print("    %-28s(no rows on these scenes)" % name)
                continue
            d, dlo, dhi = diff(ck, cn, k, n)
            verdict = "excludes zero" if not (dlo <= 0 <= dhi) else "consistent with zero"
            print("    %-28s%8.4f  n=%d   delta %+.4f [%+.4f, %+.4f]  %s"
                  % (name, ck / cn, cn, d, dlo, dhi, verdict))
        print()

    # ---- the grounding question -------------------------------------------
    print("DOES THE MARK STEER THE REFERENT")
    print("  The question above is not this one. A fine-tune can match the")
    print("  baseline on task success and still not read the sketch; overlay_v6")
    print("  did exactly that. Both or neither.")
    print()
    if args.skip_scorer:
        print("  (skipped)")
        return
    scorer = os.path.join(REPO, "scripts", "score_referent_following.py")
    proc = subprocess.run([sys.executable, scorer, "--real", args.real,
                           "--swap", args.swap], capture_output=True, text=True)
    for line in (proc.stdout or proc.stderr).splitlines():
        print("  " + line)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
