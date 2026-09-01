#!/usr/bin/env python3
"""Score a real/swap rollout pair. The only sanctioned way to report a sketch arm.

THE REPORTING HAZARD THIS SCRIPT EXISTS TO REMOVE
-------------------------------------------------
`overlay_v6`'s real arm alone reads as a breakthrough: 59.0% -> 78.7% on taking
the goal bowl, +19.7 points. It is not one. Moving the circle onto the distractor
cost the goal bowl 15.5 points, so the sketch is causal -- but the freed mass did
not go to the circled object. The circled bowl gained 3.3 points (~1.6 sigma)
while an UNCIRCLED, irrelevant bowl gained 7.8 (~4 sigma). That is a marker
degrading the scene representation, not one designating a referent, and no number
computed from the real arm alone can tell the two apart.

So this script takes two run directories and refuses to produce a summary from
one. `--real` and `--swap` are both required.

THE ESTIMAND
------------
For each object o, paired by scene:

    effect[o] = P(grasp o | circle on the distractor) - P(grasp o | circle on the target)

which is the causal effect of moving the mark onto the distractor, on taking
object o. Read it as a shape, not a single number:

    effect[distractor]  should be large and positive -- the mark moved, the grasp
                        followed. This is `referent_following`.
    effect[target]      should be large and negative, by roughly the same amount.
    effect[everything   should be ~0. Any uncircled object rising is
     else]              `specificity_violation`, and if it exceeds
                        effect[distractor] the model is being disturbed rather
                        than steered. That is the v6 signature.

`grasped_any` is not a targeting metric and is reported only as a health check:
a run that stops grasping altogether can post a flattering effect on every
object at once.

    python scripts/score_referent_following.py \
        --real outputs/rollouts/sketchvla_rg_v7_ambiguous_sketch \
        --swap outputs/rollouts/sketchvla_rg_v7_ambiguous_swap \
        --out outputs/referent_following_v7.json
"""

import argparse
import collections
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import provenance

SIGMA_FOR_CLAIM = 2.0


def read_rows(run_dir):
    path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(path):
        raise SystemExit(f"[error] no results.csv in {run_dir}")
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("skipped", "False") != "True"]
    if not rows:
        raise SystemExit(f"[error] {path} has no unskipped rows")
    return rows


def grasp_rates(rows):
    """Per scene: the share of GRASPS that took each object.

    Conditioned on grasping, as the published tables are, so that a change in how
    often the policy grasps at all does not masquerade as a change in what it
    grasps.
    """
    by_scene = collections.defaultdict(list)
    for r in rows:
        by_scene[(r["suite"], r["dir"])].append(r)
    out = {}
    for scene, rs in by_scene.items():
        grasps = [r for r in rs if r.get("grasped_instance")]
        if not grasps:
            continue
        counts = collections.Counter(r["grasped_instance"] for r in grasps)
        out[scene] = ({obj: n / len(grasps) for obj, n in counts.items()},
                      len(grasps), len(rs))
    return out


def paired_effect(real, swap, objects):
    """Scene-paired difference per object, with a paired standard error."""
    scenes = sorted(set(real) & set(swap))
    if not scenes:
        raise SystemExit("[error] the two arms share no scene; they are not a pair")
    effects = {}
    for obj in objects:
        diffs = [swap[s][0].get(obj, 0.0) - real[s][0].get(obj, 0.0) for s in scenes]
        mean = sum(diffs) / len(diffs)
        if len(diffs) > 1:
            var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
            se = math.sqrt(var / len(diffs))
        else:
            se = float("nan")
        effects[obj] = {"points": round(100 * mean, 2), "se_points": round(100 * se, 2),
                        "sigma": round(mean / se, 2) if se else None, "n_scenes": len(diffs)}
    return effects, scenes


def verdict(effects, circled_in_swap, circled_in_real):
    """Grounded, disturbed, or null -- stated as a rule, not left to the reader."""
    follow = effects.get(circled_in_swap, {})
    others = {o: e for o, e in effects.items() if o not in (circled_in_swap, circled_in_real)}
    worst = max(others.items(), key=lambda kv: kv[1]["points"], default=(None, {"points": 0.0}))
    sigma = follow.get("sigma") or 0.0
    if sigma < SIGMA_FOR_CLAIM:
        label = "null: moving the mark did not move the grasp onto the marked object"
    elif worst[1]["points"] >= follow["points"]:
        label = ("disturbance: an uncircled object gained at least as much as the circled "
                 "one -- the mark degrades the scene rather than designating a referent")
    else:
        label = "grounded: the circled object gained, and gained more than any uncircled one"
    return {"label": label, "referent_following": follow,
            "specificity_violation": {"object": worst[0], **worst[1]}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--real", required=True, help="run dir of the circle-on-target arm")
    ap.add_argument("--swap", required=True, help="run dir of the circle-on-distractor arm")
    ap.add_argument("--target", default="akita_black_bowl_1")
    ap.add_argument("--distractor", default="akita_black_bowl_2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    real_rows, swap_rows = read_rows(args.real), read_rows(args.swap)
    for name, rows, expected in (("real", real_rows, "native"), ("swap", swap_rows, "swap")):
        routes = {r.get("sketch_route") for r in rows}
        if routes != {expected}:
            raise SystemExit(
                f"[error] --{name} is {sorted(routes)}, expected {expected!r}. "
                "Passing the same arm twice would report a null as a null for the wrong reason.")

    real, swap = grasp_rates(real_rows), grasp_rates(swap_rows)
    objects = sorted({o for s in list(real.values()) + list(swap.values()) for o in s[0]})
    effects, scenes = paired_effect(real, swap, objects)

    def arm_block(rows, rates):
        return {"n_rollouts": len(rows),
                "grasped_any": round(sum(1 for r in rows if r.get("grasped_instance")) / len(rows), 4),
                "success_sustained": round(sum(r["success_sustained"] == "True" for r in rows) / len(rows), 4),
                "grasp_share": {o: round(sum(v[0].get(o, 0.0) for v in rates.values()) / len(rates), 4)
                                for o in objects}}

    result = {
        "arms": {"real": arm_block(real_rows, real), "swap": arm_block(swap_rows, swap)},
        "effect_of_moving_the_circle": effects,
        "verdict": verdict(effects, args.distractor, args.target),
        "n_paired_scenes": len(scenes),
        "run_dirs": {"real": args.real, "swap": args.swap},
    }
    if args.out:
        provenance.write_json(args.out, result)
        print("wrote", args.out)

    print("\neffect of moving the circle onto the distractor (points of grasps, paired by scene)")
    for obj, e in sorted(effects.items(), key=lambda kv: -kv[1]["points"]):
        mark = "  <- circled in swap" if obj == args.distractor else (
            "  <- circled in real" if obj == args.target else "")
        print(f"  {obj:28s} {e['points']:+7.2f}  se {e['se_points']:5.2f}  "
              f"{e['sigma'] if e['sigma'] is not None else float('nan'):+6.2f} sigma{mark}")
    print("\nVERDICT:", result["verdict"]["label"])


if __name__ == "__main__":
    main()
