"""
Sketch-Prompted VLA — target-displacement probe (needs the libero env).

Answers one question the validation set cannot: **how much of the policy's
success comes from the target being where the demonstrations put it?**

The anchored Spatial set holds every object at its shipped position, so it is the
control for that question, not the experiment. The evidence for a position effect
so far is indirect — the anchored set beats the from-scratch set by 13-17 points,
and inside the anchored set a goal on the shipped `plate_1` scores 64% against
12% for a goal on an injected plate. Both are suggestive. Neither varies position
and nothing else.

This does. One shipped task, one caption, no objects added, no objects moved
except the target, which is translated in fixed steps away from where LIBERO
places it. Success against displacement is a dose-response curve, which is a far
stronger claim than any pair of dataset-level numbers.

DESIGN, and the one deliberate departure from stock.

`akita_black_bowl_2` is REMOVED here. Every shipped `libero_spatial` task carries
two identical bowls, so with a category caption ("the black bowl") a scene has
two valid referents and target selection becomes a coin flip that would swamp the
displacement signal. Deleting it makes the caption resolve to exactly one object
at every offset. That contradicts the anchoring rule the validation set follows,
and it is correct here for the same reason the rule exists there: isolate one
variable. The deletion is constant across every offset including zero, so it
shifts the whole curve and cannot manufacture its shape. The offset-0 point is
the reference the rest is read against, not an independent claim about stock
LIBERO.

Everything else is stock: the plate, the cookies, the ramekin, the cabinet and
the stove stay exactly where they ship, and the goal predicate is unchanged.

The caption is the category-explicit one, NOT the stock caption. Stock captions
name the target by where it is ("the black bowl **between the plate and the
ramekin**"), so translating the bowl would make the sentence false and confound
displacement with a caption that no longer describes the scene.

GATES. An offset is skipped, and recorded as skipped, when the bowl would leave
the reachable band, land within `D_CROSS` of another object, fall off the table,
project outside the frame, or fail the teleport oracle. Skips are part of the
result: a curve with holes at large offsets is still informative, a curve that
silently drops them is not.

Each scene pins its own `init_state.npz` in-place — `capture_scene_init_states.py`
knows only the three validation suites, and there is no reason to teach it a
fourth for a probe.

    source $OPENPI/examples/libero/.venv/bin/activate
    export PYTHONPATH=$PYTHONPATH:$OPENPI/third_party/libero
    export MUJOCO_GL=egl
    export LIBERO_SPATIAL_BDDL=$OPENPI/third_party/libero/libero/libero/bddl_files/libero_spatial
    cd $REPO
    python scripts/build_displacement_probe.py 2>&1 \
        | tee outputs/validation_set_displacement/build_log.txt

Then run it through the normal harness (`rollout_sketch.py` reads extra suites
from `EXTRA_SUITES`) and plot with `scripts/analyze_displacement.py`.
"""

import os, re, json, gc, copy
import numpy as np
import cv2

import build_validation_set_spatial_anchored as A

OUT_ROOT = os.path.join(A._REPO, "outputs", "validation_set_displacement")

# Two tasks, because no single one clears all four directions. Measured against
# the reachability gates below: `table_center` clears 12 of 17 offsets (it owns
# the open middle of the table), `between_plate_ramekin` clears 10 and covers the
# +y/-y span the other truncates. Together they give 22 scenes and every
# direction at full radius from at least one base. `next_to_ramekin` clears 6 and
# `next_to_cookie_box` none — its bowl starts in the front-right corner with the
# table edge immediately beyond.
TASKS = ["table_center", "between_plate_ramekin"]

# Metres from the shipped position. 0.12 m is roughly the width of the reachable
# band, so the sweep spans "where it always is" to "the far side of the table".
RADII = [0.03, 0.06, 0.09, 0.12]
DIRECTIONS = {"xpos": (1, 0), "xneg": (-1, 0), "ypos": (0, 1), "yneg": (0, -1)}
N_ROLLOUT_HINT = 14                      # recorded in the manifest, not used here

CAPTION = "pick up the black bowl and place it on the plate"
CAPTION_AMBIGUOUS = "pick this up and place it on that"
DROP_INSTANCE = "akita_black_bowl_2"


def offsets():
    yield ("origin", 0.0, 0.0, 0.0)
    for name, (ux, uy) in DIRECTIONS.items():
        for r in RADII:
            yield (name, r, ux * r, uy * r)


def reachable_rect(parsed):
    """Where the target may be placed.

    `WORKSPACE_X/Y` in the anchored builder bounds where INJECTED duplicates may
    go; it is deliberately conservative and at least one shipped bowl starts
    outside it (`next_to_cookie_box` sits at y = -0.07 against a floor of -0.06).
    A stock position is reachable by definition — LIBERO demonstrates it — so the
    probe takes the union of that rectangle with the stock objects' own bounding
    box, padded. Using the tighter rectangle would reject a task's own origin."""
    xs, ys = [], []
    for inst in parsed["objs"]:
        _, xy = A.init_region_of(parsed, inst)
        if xy is not None:
            xs.append(xy[0]); ys.append(xy[1])
    if not xs:
        return A.WORKSPACE_X, A.WORKSPACE_Y
    pad = 0.02
    return ((min(A.WORKSPACE_X[0], min(xs) - pad), max(A.WORKSPACE_X[1], max(xs) + pad)),
            (min(A.WORKSPACE_Y[0], min(ys) - pad), max(A.WORKSPACE_Y[1], max(ys) + pad)))


def strip_instance(text, inst):
    """Remove one object from (:objects) and its (:init) placement.

    Rebuilds the whole (:objects) block from LIBERO's own category grouping
    rather than deleting a token in place: the parser assigns by category, so an
    edit that leaves a category line malformed drops every instance in it."""
    objs = A._libero_typed(A._tag_block(text, "objects"), ":objects")
    cats = copy.deepcopy(objs)
    for cat in list(cats):
        cats[cat] = [i for i in cats[cat] if i != inst]
        if not cats[cat]:
            del cats[cat]
    lines = "\n".join("    " + " ".join(v) + f" - {c}" for c, v in cats.items())
    text = A._replace_block(text, A._tag_block(text, "objects"),
                            f"(:objects\n{lines}\n  )")

    init = A._tag_block(text, "init")
    kept = [sx for sx in A._top_sexprs(A._block_inner(init, "init"))
            if not re.search(r"\(\s*[A-Za-z]+\s+%s\b" % re.escape(inst), sx)]
    text = A._replace_block(text, init,
                            "(:init\n" + "\n".join("    " + " ".join(s.split())
                                                   for s in kept) + "\n  )")
    return text


def move_target(text, parsed, dx, dy):
    """Re-centre the target's own placement region by (dx, dy).

    Editing that region's rectangle in place, rather than adding a new one, keeps
    the (:init) line and every other region byte-identical, so the only thing
    that differs between two offsets is four numbers."""
    rn, xy = A.init_region_of(parsed, A.TARGET)
    if rn is None or xy is None:
        raise ValueError("target has no table-plane placement region")
    cx, cy = xy[0] + dx, xy[1] + dy
    blk = A._tag_block(text, "regions")
    for sx in A._top_sexprs(A._block_inner(blk, "regions")):
        if not re.match(r"\(\s*%s\b" % re.escape(rn), sx):
            continue
        ri = sx.find("(:ranges")
        old = A._balanced(sx, ri)
        new = ("(:ranges (\n              "
               f"({cx-A.HALF_BOX:.4f} {cy-A.HALF_BOX:.4f} "
               f"{cx+A.HALF_BOX:.4f} {cy+A.HALF_BOX:.4f})\n            )\n          )")
        return text.replace(sx, sx.replace(old, new), 1), (cx, cy)
    raise ValueError("region %s not found in text" % rn)


def build_scene(task_key, tag, radius, dx, dy, idx, seed, scene_dir):
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils import camera_utils as CU

    src = os.path.join(A.BDDL_ROOT, A.BASE_TASKS[task_key])
    parsed = A.parse_stock(src)
    text = strip_instance(parsed["text"], DROP_INSTANCE)
    text, (cx, cy) = move_target(text, parsed, dx, dy)
    text = A._replace_block(text, A._tag_block(text, "obj_of_interest"),
                            f"(:obj_of_interest\n    {A.TARGET}\n    plate_1\n  )")
    text = re.sub(r"\(:language\s+.*?\)", f"(:language {CAPTION})", text,
                  count=1, flags=re.S)
    A.verify_injected_bddl(text, [A.TARGET, "plate_1"])

    rx, ry = reachable_rect(parsed)
    if not (rx[0] <= cx <= rx[1] and ry[0] <= cy <= ry[1]):
        return None, "outside_reach_%.3f_%.3f" % (cx, cy)
    for inst in parsed["objs"]:
        if inst in (A.TARGET, DROP_INSTANCE):
            continue
        _, xy = A.init_region_of(parsed, inst)
        if xy is not None and np.hypot(cx - xy[0], cy - xy[1]) < A.D_CROSS:
            return None, "too_close_to_%s" % inst

    os.makedirs(scene_dir, exist_ok=True)
    bpath = os.path.join(scene_dir, "scene.bddl")
    open(bpath, "w").write(text)

    np.random.seed(seed)
    env = OffScreenRenderEnv(bddl_file_name=bpath, camera_heights=A.IMG_H,
                             camera_widths=A.IMG_W, camera_names=[A.CAMERA])
    try:
        obs = A.settle(env)
        model, data = env.sim.model, env.sim.data
        frame = A.frame_obs(obs)
        W2P = CU.get_camera_transform_matrix(sim=env.sim, camera_name=A.CAMERA,
                                             camera_height=A.IMG_H,
                                             camera_width=A.IMG_W)
        instances = [i for v in A._libero_typed(A._tag_block(text, "objects"),
                                                ":objects").values() for i in v]
        for i in instances:
            if float(A.vcenter(model, data, A.bid_of(model, i))[2]) < A.TABLE_Z_MIN:
                return None, "fell_%s" % i

        pix = {i: A.project([A.vcenter(model, data, A.bid_of(model, i))], W2P)[0]
               for i in instances}
        ext = {i: A.px_extent(model, data, W2P, i) for i in instances}
        m = int(np.ceil(ext[A.TARGET])) + 2
        if not (m <= pix[A.TARGET][0] <= A.IMG_W - 1 - m
                and m <= pix[A.TARGET][1] <= A.IMG_H - 1 - m):
            return None, "target_off_frame_pix%s_ext%.0f" % (pix[A.TARGET], ext[A.TARGET])

        if A.success(env) is not False:
            return None, "pre_solved"
        if A.teleport_on(env, model, data, A.TARGET, "plate_1") is not True:
            return None, "oracle_false"

        np.random.seed(seed)
        obs = A.settle(env)
        model, data = env.sim.model, env.sim.data
        grasp = A.scripted_grasp(env, model, data, A.TARGET, obs)

        # Pin the state behind this exact frame, in place. Re-settling from the
        # same seed is what the capture script's first rung does; doing it here
        # avoids teaching that script a fourth suite for a probe.
        np.random.seed(seed)
        A.settle(env)
        flat = np.asarray(env.get_sim_state(), dtype=np.float64).ravel()
        nq, nv = env.sim.model.nq, env.sim.model.nv
        np.savez(os.path.join(scene_dir, "init_state.npz"),
                 time=flat[:1], qpos=flat[1:1 + nq], qvel=flat[1 + nq:1 + nq + nv])

        A.safe_imwrite(os.path.join(scene_dir, "frame0.png"),
                       cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        meta = dict(
            schema_version="1.0", suite="displacement", tier="displacement",
            task=task_key, base_file=A.BASE_TASKS[task_key],
            direction=tag, radius_m=round(radius, 4),
            offset_xy=[round(dx, 4), round(dy, 4)],
            target_xy=[round(cx, 4), round(cy, 4)],
            target=A.TARGET, destination="plate_1", destination_region="plate_1",
            goal_predicate="On", instruction=CAPTION,
            instruction_explicit=CAPTION, instruction_ambiguous=CAPTION_AMBIGUOUS,
            prompt_bucket="two_clause_On",
            dropped_instance=DROP_INSTANCE, seed=seed,
            all_pixels={k: list(v) for k, v in pix.items()},
            px_extent={k: round(v, 2) for k, v in ext.items()},
            pick_px=list(pix[A.TARGET]), place_px=list(pix["plate_1"]),
            camera_matrix=W2P.tolist(), oracle_success=True, grasp=grasp,
            anchored=True, n_rollout_hint=N_ROLLOUT_HINT,
            target_bowl=A.TARGET, target_plate="plate_1")
        json.dump(meta, open(os.path.join(scene_dir, "meta.json"), "w"), indent=2)
        json.dump({k: meta[k] for k in
                   ("instruction", "instruction_explicit", "instruction_ambiguous",
                    "prompt_bucket", "suite", "tier", "target", "destination",
                    "destination_region", "goal_predicate", "target_bowl",
                    "target_plate")},
                  open(os.path.join(scene_dir, "tokens.json"), "w"), indent=2)
        return meta, "ok"
    finally:
        try:
            env.close()
        except Exception:
            pass
        gc.collect()


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    plan = [(t, tag, r, dx, dy) for t in TASKS for (tag, r, dx, dy) in offsets()]
    print("MODE: %d task(s) x %d offsets = %d scenes"
          % (len(TASKS), len(list(offsets())), len(plan)))
    print("  radii %s m, directions %s" % (RADII, list(DIRECTIONS)))

    manifest, skipped = [], []
    for idx, (task, tag, r, dx, dy) in enumerate(plan):
        sd = os.path.join(OUT_ROOT, "scene_%04d" % idx)
        made = None
        for attempt in range(8):
            seed = 20000 + idx * 100 + attempt
            try:
                made, why = build_scene(task, tag, r, dx, dy, idx, seed, sd)
            except Exception as e:
                made, why = None, "error:%s:%s" % (type(e).__name__, e)
            print("  scene_%04d %-22s %-5s r=%.2f -> %s" % (idx, task, tag, r, why))
            if made:
                break
        if made:
            made["dir"] = "scene_%04d" % idx
            manifest.append({k: made.get(k) for k in
                             ("dir", "task", "direction", "radius_m", "offset_xy",
                              "target_xy", "tier", "target", "destination",
                              "instruction", "seed", "grasp")})
        else:
            skipped.append(dict(dir="scene_%04d" % idx, task=task, direction=tag,
                                radius_m=round(r, 4), reason=why))

    json.dump(manifest, open(os.path.join(OUT_ROOT, "manifest.json"), "w"), indent=2)
    json.dump(skipped, open(os.path.join(OUT_ROOT, "skipped.json"), "w"), indent=2)
    print("\n%d scene(s) built, %d offset(s) unreachable" % (len(manifest), len(skipped)))
    for s in skipped:
        print("   skipped %-5s r=%.2f  %s" % (s["direction"], s["radius_m"], s["reason"]))
    print("\nNext:")
    print("  SCENES=$(python -c \"import json;print(','.join('displacement/'+e['dir'] "
          "for e in json.load(open('outputs/validation_set_displacement/manifest.json'))))\")")
    print("  EXTRA_SUITES=displacement python scripts/rollout_sketch.py --policy pi05 \\")
    print("      --conditions text_only --prompt-type explicit "
          "--run-id pi05_displacement --n-rollouts %d --scenes \"$SCENES\" --resume"
          % N_ROLLOUT_HINT)
    print("  python scripts/analyze_displacement.py --run pi05_displacement")


if __name__ == "__main__":
    main()
