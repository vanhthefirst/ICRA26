"""
Sketch-Prompted VLA — stage 1 of the RLDS export: render each scene at the RLDS
resolution and dump the arrays (needs the libero env).

The export is two scripts because its two halves cannot share an interpreter.
Rendering needs mujoco/robosuite/libero, which live in the Python 3.8 client venv;
writing RLDS needs tensorflow and tensorflow_datasets, which do not. This half
produces a plain `.npz` per scene and imports no tensorflow at all;
`export_rlds_pack.py` turns those into the dataset.

WHAT THE TARGET SCHEMA IS, AND HOW I KNOW

`src/sketchvla/utils/convert_libero_data_to_lerobot.py` in the model repo is the
consumer: it reads the RLDS and writes the LeRobot dataset openpi trains on. It
touches exactly these fields, so they are the specification —

    step["observation"]["image"]         (256, 256, 3) uint8
    step["observation"]["wrist_image"]   (256, 256, 3) uint8
    step["observation"]["state"]         (8,)          float32
    step["action"]                       (7,)          float32
    step["language_instruction"]         text
    step["sketch"]["circle"]             (256, 256, 1) uint8
    step["sketch"]["arrow"]              (256, 256, 1) uint8
    step["sketch"]["target"]             (256, 256, 1) uint8
    step["sketch"]["circle_meta"]        (3,)          float32
    step["sketch"]["arrow_start"]        (2,)          float32
    step["sketch"]["arrow_end"]          (2,)          float32

Writing to the consumer's access pattern is the point of the exercise: the
colleague runs `convert_libero_data_to_lerobot.py` unchanged and no one hand-
converts anything. `export_rlds_pack.py --verify` re-reads the emitted dataset
and asserts every one of those names, shapes and dtypes.

Two assumptions I could not check against `sketch_rlds_dataset.py`, which was not
reachable when this was written. Both are flagged in the manifest so they are easy
to correct:

  * `circle_meta` is written as (cx, cy, r) with r the mean of the recorded rx/ry.
  * `state` follows openpi's LIBERO convention — eef position (3), eef
    orientation as axis-angle (3), gripper joint positions (2) — which is what
    `examples/libero/main.py` builds for the policy server.

RESOLUTION. The scenes were built and pinned at 128; the RLDS is 256. Frames are
RE-RENDERED at 256 from the pinned state, never upscaled, so the pixels are real.
The sketch masks are re-drawn from the recorded `symbolic_tokens` scaled by 2 —
the tokens, not the PNG, are the canonical annotation, and `sketch.png` at 128 is
the same geometry at half scale. Re-running the annotator's wobble at 256 would
have produced different high-frequency detail for the same circle, so the mask is
a clean stroke through the recorded geometry instead.

ACTIONS ARE ZERO. These scenes have no demonstrations — that is the point of a
validation set built from BDDLs rather than from demos, and it is already stated
in `sketchvla/utils/validation_data.py`. The `action` field exists because the
schema requires it. Nothing should train on it or score an L1 against it, and
`episode_metadata.has_demonstration` is False on every episode so a consumer can
assert rather than assume.

    source $OPENPI/examples/libero/.venv/bin/activate
    export PYTHONPATH=$PYTHONPATH:$OPENPI/third_party/libero
    export MUJOCO_GL=egl
    cd $REPO
    python scripts/export_rlds_frames.py --suite spatial
    python scripts/export_rlds_frames.py --suite spatial --scenes scene_0000  # smoke
"""

import os, sys, json, gc, argparse

import numpy as np
import cv2

import build_validation_set_spatial_anchored as A

RES = 256
# Every helper borrowed from the builder reads these module constants — `project`
# for its row flip, `px_extent` and `visibility` for their render size. The whole
# export runs at one resolution, so setting them once here is the honest way to
# retarget them; the alternative is a second copy of each function.
A.IMG_H = A.IMG_W = RES

SUITE_DIR = {"spatial": "validation_set_spatial",
             "object": "validation_set_object",
             "goal": "validation_set_goal",
             "displacement": "validation_set_displacement"}
OUT_ROOT = os.path.join(A._REPO, "outputs", "rlds_frames")

CIRCLE_THICKNESS = 3          # px at 256; 1-2 px at 128 scaled up and rounded
ARROW_THICKNESS = 3


def scale_tokens(tok, factor):
    return {k: (v * factor if isinstance(v, (int, float)) else v) for k, v in tok.items()}


def circle_mask(tok):
    """Stroke, not disc. The consumer thresholds at >127 and paints those pixels,
    so a filled circle would paint over the object it is meant to ring."""
    m = np.zeros((RES, RES, 1), np.uint8)
    cv2.ellipse(m, (int(round(tok["cx"])), int(round(tok["cy"]))),
                (max(1, int(round(tok["rx"]))), max(1, int(round(tok["ry"])))),
                0, 0, 360, 255, CIRCLE_THICKNESS, cv2.LINE_8)
    return m


def arrow_mask(tok):
    m = np.zeros((RES, RES, 1), np.uint8)
    cv2.arrowedLine(m, (int(round(tok["x0"])), int(round(tok["y0"]))),
                    (int(round(tok["x1"])), int(round(tok["y1"]))),
                    255, ARROW_THICKNESS, cv2.LINE_8, tipLength=0.35)
    return m


def state_vector(obs):
    """openpi's LIBERO convention: eef position, eef orientation as axis-angle,
    gripper joint positions. Same 8 numbers `examples/libero/main.py` sends the
    policy server, so a model trained on this sees what it sees at rollout."""
    from robosuite.utils.transform_utils import quat2axisangle
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"], np.float32),
        np.asarray(quat2axisangle(obs["robot0_eef_quat"]), np.float32),
        np.asarray(obs["robot0_gripper_qpos"], np.float32)]).astype(np.float32)


def as_uint8(img):
    f = np.asarray(img)
    if f.dtype != np.uint8:
        f = np.clip(f * 255.0 if f.max() <= 1 + 1e-6 else f, 0, 255).astype(np.uint8)
    return f.copy()


def wrist_image(obs):
    for k in ("robot0_eye_in_hand_image", "eye_in_hand_image"):
        if k in obs:
            return as_uint8(obs[k])
    return None


def export_scene(suite, scene_dir, out_dir):
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils import camera_utils as CU

    meta = json.load(open(os.path.join(scene_dir, "meta.json")))
    tokens = json.load(open(os.path.join(scene_dir, "tokens.json")))
    bddl = os.path.join(scene_dir, "scene.bddl")
    pin = os.path.join(scene_dir, "init_state.npz")
    if not os.path.exists(pin):
        return None, "no_init_state"

    z = np.load(pin)
    flat = np.concatenate([np.atleast_1d(z["time"]).ravel(),
                          z["qpos"].ravel(), z["qvel"].ravel()])

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES,
                             camera_widths=RES,
                             camera_names=["agentview", "robot0_eye_in_hand"])
    try:
        env.reset()
        obs = env.set_init_state(flat)
        if obs is None or "agentview_image" not in obs:
            obs = env.env._get_observations() if hasattr(env.env, "_get_observations") \
                else env.step(np.zeros(A.ADIM))[0]

        model, data = env.sim.model, env.sim.data
        image = A.frame_obs(obs)
        wrist = wrist_image(obs)
        if wrist is None:
            return None, "no_wrist_camera"

        # Geometry check: the pin is only worth anything if it puts the objects
        # back where the recorded pixels say they were. Compare at 128, the space
        # `all_pixels` is recorded in.
        W2P = CU.get_camera_transform_matrix(sim=env.sim, camera_name=A.CAMERA,
                                             camera_height=RES, camera_width=RES)
        worst = 0.0
        for name, want in meta["all_pixels"].items():
            got = A.project([A.vcenter(model, data, A.bid_of(model, name))], W2P)[0]
            worst = max(worst, float(np.hypot(got[0] / 2.0 - want[0],
                                              got[1] / 2.0 - want[1])))
        if worst > 3.0:
            return None, "pin_mismatch_%.1fpx" % worst

        instances = list(meta["all_pixels"])
        vis, vismask = A.visibility(env, model, data, instances, meta["target"])
        target_mask = vismask.reshape(RES, RES, 1).astype(np.uint8)

        tok = meta["symbolic_tokens"]
        c = scale_tokens(tok["circle"], RES / 128.0)
        a = scale_tokens(tok["arrow"], RES / 128.0)

        np.savez_compressed(
            out_dir + ".npz",
            image=image, wrist_image=wrist, state=state_vector(obs),
            action=np.zeros(7, np.float32),
            circle=circle_mask(c), arrow=arrow_mask(a), target=target_mask,
            circle_meta=np.array([c["cx"], c["cy"], (c["rx"] + c["ry"]) / 2.0], np.float32),
            arrow_start=np.array([a["x0"], a["y0"]], np.float32),
            arrow_end=np.array([a["x1"], a["y1"]], np.float32))

        row = dict(suite=suite, dir=os.path.basename(scene_dir),
                   npz=os.path.basename(out_dir + ".npz"),
                   tier=meta.get("tier"), task=meta.get("task"),
                   target=meta["target"], destination=meta["destination"],
                   goal_predicate=meta.get("goal_predicate", "On"),
                   instruction_explicit=tokens.get("instruction_explicit",
                                                   meta.get("instruction")),
                   instruction_ambiguous=tokens.get("instruction_ambiguous"),
                   out_of_focus=meta.get("out_of_focus", []),
                   anchored=bool(meta.get("anchored", False)),
                   seed=meta.get("seed"), pin_residual_px=round(worst, 3),
                   target_visibility=vis["visibility"])
        return row, "ok"
    finally:
        try:
            env.close()
        except Exception:
            pass
        gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="spatial", choices=sorted(SUITE_DIR))
    ap.add_argument("--scenes", default=None,
                    help="comma list of scene dirs; default every scene in the manifest")
    args = ap.parse_args()

    set_root = os.path.join(A._REPO, "outputs", SUITE_DIR[args.suite])
    manifest = json.load(open(os.path.join(set_root, "manifest.json")))
    dirs = [e["dir"] for e in manifest]
    if args.scenes:
        want = {s.strip() for s in args.scenes.split(",")}
        dirs = [d for d in dirs if d in want]
    out_root = os.path.join(OUT_ROOT, args.suite)
    os.makedirs(out_root, exist_ok=True)

    print("suite %s: %d scene(s) at %dx%d -> %s" % (args.suite, len(dirs), RES, RES, out_root))
    rows, fails = [], []
    for d in dirs:
        row, why = export_scene(args.suite, os.path.join(set_root, d),
                                os.path.join(out_root, d))
        print("  %-12s %s" % (d, why))
        (rows if row else fails).append(row or dict(dir=d, reason=why))

    out = dict(suite=args.suite, resolution=RES, n_scenes=len(rows),
               scenes=rows, failed=fails,
               action_is_zero=True,
               action_note="no demonstration exists for these scenes; the field "
                           "is schema-required and must not be trained or scored on",
               state_convention="eef_pos(3) + eef_axisangle(3) + gripper_qpos(2), "
                                "openpi examples/libero/main.py",
               circle_meta_convention="(cx, cy, mean(rx, ry)) in 256-px image space",
               masks_from="symbolic_tokens scaled by 256/128; sketch.png is the "
                          "same geometry at half scale")
    json.dump(out, open(os.path.join(out_root, "frames_manifest.json"), "w"), indent=2)
    print("\n%d exported, %d failed -> %s"
          % (len(rows), len(fails), os.path.join(out_root, "frames_manifest.json")))
    for f in fails:
        print("   FAILED %s: %s" % (f["dir"], f["reason"]))
    if rows:
        print("worst pin residual: %.2f px" % max(r["pin_residual_px"] for r in rows))


if __name__ == "__main__":
    sys.exit(main())
