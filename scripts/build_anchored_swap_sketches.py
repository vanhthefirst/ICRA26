#!/usr/bin/env python3
"""The swap sketch for the 37 anchored Spatial scenes: the circle on the distractor.

WHY THIS HAS TO EXIST BEFORE V7 CAN BE RUN ON THESE SCENES

`scripts/score_referent_following.py` refuses to summarise a real arm without
its swap arm, and it is right to: `overlay_v6`'s real arm alone read as +19.7
points on taking the goal bowl and was not a result -- the mark was degrading
the scene representation, not designating a referent, and only the swap arm
could tell the two apart. So a V7 run on the anchored suite needs a sketch with
the circle on the DISTRACTOR, scene for scene, and the anchored export
(`outputs/validation_set_spatial/scene_*/tokens.json`) carries only the real one.

WHAT IS DRAWN, AND WHY IT IS NOT A NEW SKETCH LANGUAGE

The same two functions the anchored builder used, at the same per-scene seed:
`draw_circle` for the ring and `draw_arrow` for the target->destination stroke.
Same jitter draw, same wobble, same stroke widths. Only the circle's CENTRE and
the arrow's TAIL move, from the target to the distractor; the destination is
untouched, because the swap moves the referent and not the goal. Anything else
would make the two arms differ in more than the one variable they are supposed
to differ in.

WHICH OBJECT IS THE DISTRACTOR

`akita_black_bowl_2`, uniformly. It ships in every stock libero_spatial task,
it is present in all 37 scenes, and it is the object v5's and v6's swap arms
circled -- so this arm stays comparable with the two that came before it
(STATE.md's `bowl_2 (distractor)` column). Scenes where it is absent are
skipped rather than re-pointed at a sibling; a swap arm whose referent changes
between scenes is not one condition.

THE RADIUS IS DERIVED, NOT INVENTED

The builder sized the ring from the target's projected geom spread, which needs
a live simulator. `meta.json` does not carry that spread, but it carries
`px_extent` for every instance and the `radius` actually used for the target, so
the distractor's radius is the target's scaled by the ratio of their extents,
under the builder's own clamp. The scaling is checked rather than trusted: every
scene must place the distractor inside the drawn ring and every other object
outside it by the builder's own `ELLIPSE_MIN` margin, or the scene is rejected.
A rejected scene is a scene whose swap arm would be ambiguous, and one of those
silently included is worth more than the whole run.

    python scripts/build_anchored_swap_sketches.py
    python scripts/build_anchored_swap_sketches.py --suite spatial --strict
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_validation_set_spatial_anchored as A
import provenance

DISTRACTOR = "akita_black_bowl_2"
SUITE_DIR = {"spatial": "validation_set_spatial"}
# `draw_circle` clips its ring points into the canvas, so a ring that overruns
# the frame is flattened against the border rather than lost. A few percent is
# the stock LIBERO framing and is what the real sketches already live with; a
# ring mostly outside the frame is not a mark anyone could read.
MAX_OUTSIDE = 0.15


def fraction_outside_frame(tok, samples=720):
    """Share of the drawn ellipse's outline that falls outside the canvas."""
    ang = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    xs = tok["cx"] + tok["rx"] * np.cos(ang)
    ys = tok["cy"] + tok["ry"] * np.sin(ang)
    inside = ((xs >= 1) & (xs <= A.IMG_W - 2) & (ys >= 1) & (ys <= A.IMG_H - 2))
    return 1.0 - float(inside.mean())


def swap_tokens(tokens, meta, distractor=DISTRACTOR):
    """The evaluator's own construction, reproduced exactly.

    `sketch_eval_common.swap_tokens` on `SketchPromptVLA-Pi@feat/eval-harness`
    keeps the authored `rx`/`ry` and moves only the circle's CENTRE onto the
    distractor's pixel, with the arrow running distractor -> destination. That
    is deliberate and it is right: holding the ring's shape constant leaves
    POSITION as the only difference between the two arms, which is what makes
    the contrast paired. Redrawing the ring with fresh jitter, or resizing it to
    the distractor's own extent, would put a second variable into a
    one-variable experiment.

    Reproduced here rather than imported because the evaluator lives in the
    other repository and in a python 3.8 venv. `--check-parity` diffs this
    against that file when a checkout is reachable, so the copy cannot drift
    silently -- an audit of tokens the run will not use is worth nothing.
    """
    circle = (tokens or {}).get("circle")
    arrow = (tokens or {}).get("arrow")
    px = (meta or {}).get("all_pixels") or {}
    target = meta.get("target")
    dest = meta.get("destination")
    if not (circle and arrow and dest in px and distractor in px
            and distractor != target):
        return None, None
    dx, dy = px[distractor]
    ex, ey = px[dest]
    return ({"cx": float(dx), "cy": float(dy),
             "rx": float(circle["rx"]), "ry": float(circle["ry"])},
            {"x0": float(dx), "y0": float(dy), "x1": float(ex), "y1": float(ey)})


def build_scene(root, distractor=DISTRACTOR):
    """Returns (tokens, report). `tokens` is None when the scene is rejected."""
    meta = json.load(open(os.path.join(root, "meta.json")))
    target = meta["target"]
    pix = {k: tuple(v) for k, v in meta["all_pixels"].items()}
    report = {"scene": os.path.basename(root), "tier": meta["tier"],
              "target": target, "distractor": distractor}

    if distractor not in pix:
        report["reject"] = "distractor_absent"
        return None, report
    if distractor == target:
        report["reject"] = "distractor_is_target"
        return None, report

    authored = json.load(open(os.path.join(root, "tokens.json")))["symbolic_tokens"]
    tok_circle, tok_arrow = swap_tokens(authored, meta, distractor)
    if tok_circle is None:
        report["reject"] = "no_usable_distractor"
        return None, report
    tokens = {"circle": tok_circle, "arrow": tok_arrow}

    # `ellipse_norm` measures from the tokens ACTUALLY drawn, so it sees the
    # jitter rather than the request.
    norms = {name: round(float(A.ellipse_norm(tok_circle, p)), 3)
             for name, p in pix.items()}
    outside = round(fraction_outside_frame(tok_circle), 4)
    report["rx"] = round(tok_circle["rx"], 3)
    report["ry"] = round(tok_circle["ry"], 3)
    report["ellipse_norm"] = norms
    report["fraction_outside_frame"] = outside
    report["circle"] = tok_circle
    report["arrow"] = tok_arrow

    # The gates are the BUILDER'S, applied to the swapped ring. It gated
    # ELLIPSE_MIN on same-category instances -- `meta["ellipse_norm"]` on an
    # accepted scene carries the sibling bowls and nothing else -- because the
    # failure it guards against is the ring being read as marking a different
    # GRASP CANDIDATE. A ramekin 1.2 radii out is not a candidate; a second
    # bowl is. Non-bowls are recorded as advisory so a scene can still be
    # inspected, never as a silent pass.
    same_category = [n for n in pix if n.rsplit("_", 1)[0] == distractor.rsplit("_", 1)[0]]
    if norms[distractor] >= 1.0:
        report["reject"] = "distractor_outside_ring_%.2f" % norms[distractor]
        return None, report
    intruders = {n: norms[n] for n in same_category
                 if n != distractor and norms[n] < A.ELLIPSE_MIN}
    if intruders:
        report["reject"] = "ambiguous_ring_" + ",".join(
            "%s@%.2f" % (n, v) for n, v in sorted(intruders.items()))
        return None, report
    if outside > MAX_OUTSIDE:
        report["reject"] = "ring_%.0f%%_outside_frame" % (100 * outside)
        return None, report
    # A non-candidate merely NEAR the ring is advisory; one strictly inside it
    # is a second reading of the mark, and the swap arm's whole job is to make
    # the mark's referent unambiguous.
    enclosed = {n: norms[n] for n in pix
                if n not in same_category and norms[n] < 1.0}
    if enclosed:
        report["reject"] = "ring_encloses_" + ",".join(
            "%s@%.2f" % (n, v) for n, v in sorted(enclosed.items()))
        return None, report

    report["near_non_candidates"] = {n: v for n, v in norms.items()
                                     if n not in same_category and v < A.ELLIPSE_MIN}
    report["separation_to_target"] = norms[target]
    return tokens, report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="spatial", choices=sorted(SUITE_DIR))
    ap.add_argument("--distractor", default=DISTRACTOR)
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any scene is rejected")
    ap.add_argument("--out", default=None,
                    help="manifest path; default outputs/<set>/swap_manifest.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only; write no manifest and no scene lists")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    set_dir = os.path.join(repo, "outputs", SUITE_DIR[args.suite])
    scenes = sorted(d for d in os.listdir(set_dir) if d.startswith("scene_"))
    if not scenes:
        raise SystemExit("no scenes under %s" % set_dir)

    reports, written, rejected = [], 0, []
    for scene in scenes:
        root = os.path.join(set_dir, scene)
        tokens, report = build_scene(root, args.distractor)
        reports.append(report)
        # No asset is written. `eval_sketchvla.py --sketch-mode swap` builds the
        # swap tokens itself from tokens.json + meta.json at run time, so a
        # tokens_swap.json on disk would be a second source of truth that the
        # run ignores -- and the first thing to go stale. An earlier revision of
        # this script wrote one; it is removed here.
        stale = os.path.join(root, "tokens_swap.json")
        if os.path.exists(stale) and not args.dry_run:
            os.remove(stale)
            report["removed_stale_asset"] = True
        if tokens is None:
            rejected.append((scene, report["reject"]))
            continue
        written += 1

    out = args.out or os.path.join(set_dir, "swap_manifest.json")
    manifest = {
        "suite": args.suite,
        "distractor": args.distractor,
        "n_scenes": len(scenes),
        "n_accepted": written,
        "n_rejected": len(rejected),
        "rejected": [{"scene": s, "reason": r} for s, r in rejected],
        "ellipse_min": A.ELLIPSE_MIN,
        "scenes": reports,
    }
    # score_referent_following.py pairs the two arms BY SCENE, so the real arm
    # has to be restricted to the scenes the swap arm can cover. Emitting the
    # list here is the difference between a paired estimand and a comparison of
    # two different scene sets -- which is the error this whole line of work has
    # been unpicking.
    accepted_list = ",".join("%s/%s" % (args.suite, r["scene"])
                             for r in reports if "reject" not in r)
    if not args.dry_run:
        provenance.write_json(out, manifest)
        with open(os.path.join(set_dir, "swap_scene_list.txt"), "w") as fh:
            fh.write(accepted_list + "\n")
        # eval_sketchvla.py's --scenes takes BARE scene dirs; this repo's
        # rollout_sketch.py takes suite/dir. Both spellings, so neither runner
        # needs anyone to reformat a 26-entry list by hand.
        with open(os.path.join(set_dir, "swap_scene_list_bare.txt"), "w") as fh:
            fh.write(",".join(r["scene"] for r in reports
                              if "reject" not in r) + "\n")

    print("%d scenes: %d admit a swap ring, %d rejected%s"
          % (len(scenes), written, len(rejected), " (dry run)" if args.dry_run else ""))
    for scene, reason in rejected:
        print("  REJECT %s  %s" % (scene, reason))
    accepted = [r for r in reports if "reject" not in r]
    if accepted:
        sep = [r["separation_to_target"] for r in accepted]
        print("separation of the TARGET from the swap ring, in ring radii: "
              "min %.2f  median %.2f  (must exceed %.2f)"
              % (min(sep), float(np.median(sep)), A.ELLIPSE_MIN))
    print("RUN BOTH ARMS ON THESE SCENES ONLY -- the pairing is by scene:")
    print("  --scenes %s" % accepted_list)
    if not args.dry_run:
        print("wrote", out)
        print("wrote", os.path.join(set_dir, "swap_scene_list.txt"))
        print("wrote", os.path.join(set_dir, "swap_scene_list_bare.txt"))
    if rejected and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
