#!/usr/bin/env python3
"""Evaluate real and swapped sketches on the exact ten paired LIBERO layouts.

Two policies, one evaluator. `--policy sketchvla` sends the sketch to a
SketchPromptVLA server; `--policy pi05` sends nothing to a STOCK pi0.5-LIBERO
server, on the same layouts, the same demo indices, the same donor poses, the
same step budget and the same success test. That second arm is the only way to
read the fine-tune's number against the baseline's: every previously published
baseline (40.3% explicit / 36.5% ambiguous) was measured on 37 anchored Spatial
scenes with a sustained-5 criterion, and no arm of it has ever been run on
these ten paired layouts. Comparing across those two designs is not a
comparison.

`--caption explicit` replaces the corpus's referent-free caption ("do this")
with the layout's own BDDL wording, which names the target bowl uniquely even
after the distractor is relocated. So:

    pi05      + explicit  -- how hard the layouts are for a text-only policy
                             that is TOLD the referent: the ceiling.
    pi05      + stored    -- the same policy denied the referent: the floor,
                             and the prior a sketch has to beat.
    sketchvla + stored    -- blank / real / swap, the three sketch doses.

Nothing here is a fair number on its own. Report the block.
"""

import argparse
import collections
import json
import os
import pathlib
import sys
import time

import h5py
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

HERE = pathlib.Path(__file__).resolve().parent
TRAIN_REPO = pathlib.Path(os.environ.get("REPO", "/workspace/SketchPromptVLA-Pi"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TRAIN_REPO / "examples" / "libero"))

import build_paired_corpus as B  # noqa: E402
import eval_sketchvla as E  # noqa: E402
import provenance  # noqa: E402
import sketch_eval_common as sec  # noqa: E402


BOWLS = ("akita_black_bowl_1", "akita_black_bowl_2")
LIFT_TH = 0.03
WRIST_KEY = "robot0_eye_in_hand_image"


def _unrotate_image(value):
    value = np.asarray(value)
    if value.ndim == 3:
        value = value[..., 0]
    return np.ascontiguousarray(np.rot90(value, 2))


def _unrotate_xy(value, canvas=256):
    value = np.asarray(value, dtype=np.float32)
    return np.array([canvas - 1 - value[0], canvas - 1 - value[1]], np.float32)


class StoredSketch:
    """One corpus sketch, turned back to simulator orientation before E.build_element."""

    def __init__(self, episode_path, mode):
        if mode not in ("real", "swap", "blank"):
            raise ValueError("mode must be real, swap or blank")
        self.mode = mode
        with np.load(episode_path) as episode:
            if mode in ("real", "blank"):
                self.circle = _unrotate_image(episode["circle"])
                self.arrow = _unrotate_image(episode["arrow"])
                circle_meta = np.asarray(episode["circle_meta"], np.float32)
                arrow_start = episode["arrow_start"]
                arrow_end = episode["arrow_end"]
            else:
                self.circle = _unrotate_image(episode["circle_swap"])
                self.arrow = _unrotate_image(episode["arrow_swap"])
                circle_meta = np.asarray(episode["distractor_meta"], np.float32)
                arrow_start = episode["arrow_swap_start"]
                arrow_end = episode["arrow_swap_end"]
            self.caption = str(episode["caption"].item())
            self.reference_frame = np.asarray(episode["images"][0]).copy()
        center = _unrotate_xy(circle_meta[:2])
        self.circle_meta = np.array([center[0], center[1], circle_meta[2]], np.float32)
        self.arrow_start = _unrotate_xy(arrow_start)
        self.arrow_end = _unrotate_xy(arrow_end)

    def payload(self, frame_rgb, resize_fn, to_uint8):
        if self.mode == "blank":
            # Same convention as validation_data.ablate_sketch: empty masks and
            # zeroed geometry, and the overlay is the CLEAN frame rather than a
            # black square, so the input stays on the "no sketch was drawn"
            # distribution instead of introducing a second shift.
            empty = np.zeros_like(sec._resize_mask(
                self.circle, sec.RESIZE_SIZE, resize_fn, to_uint8))
            return {
                "observation/arrow": empty,
                "observation/circle": empty.copy(),
                "observation/sketch_overlay": to_uint8(
                    resize_fn(frame_rgb, sec.RESIZE_SIZE, sec.RESIZE_SIZE)),
                "observation/arrow_start": np.zeros(2, np.float32),
                "observation/arrow_end": np.zeros(2, np.float32),
                "observation/circle_meta": np.zeros(3, np.float32),
            }
        circle = sec._resize_mask(self.circle, sec.RESIZE_SIZE, resize_fn, to_uint8)
        arrow = sec._resize_mask(self.arrow, sec.RESIZE_SIZE, resize_fn, to_uint8)
        overlay = sec.overlay_sketch(frame_rgb, self.circle > 127, self.arrow > 127)
        overlay = to_uint8(resize_fn(overlay, sec.RESIZE_SIZE, sec.RESIZE_SIZE))
        return {
            "observation/arrow": arrow,
            "observation/circle": circle,
            "observation/sketch_overlay": overlay,
            "observation/arrow_start": self.arrow_start,
            "observation/arrow_end": self.arrow_end,
            "observation/circle_meta": self.circle_meta,
        }


def explicit_caption(task_key):
    """The layout's own BDDL wording, which names the target bowl uniquely.

    The pairing moves the DISTRACTOR into the partner task's target region, so
    the target's own descriptor ("the black bowl next to the ramekin") still
    picks out exactly one bowl -- that admissibility is what
    check_sketch_necessity.py certified before the corpus was built. Do not use
    this arm on a layout that has not been through that check.
    """
    return B.T[task_key].replace("_", " ")


def quat2axisangle(quat):
    """Verbatim from robosuite, via openpi's examples/libero/main.py.

    Reproduced rather than imported so the observation this file builds cannot
    drift with whatever robosuite the libero venv happens to pin.
    """
    quat = np.asarray(quat, dtype=float).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if abs(den) < 1e-12:
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def _rot180(image):
    return np.ascontiguousarray(np.asarray(image)[::-1, ::-1])


class Pi05Policy:
    """Stock pi0.5-LIBERO behind an openpi websocket server -- the baseline arm.

    pi0.5 has no sketch channel, so "pi0.5 without the sketch" is pi0.5 in its
    native mode: base image, wrist image, 8-D state, instruction. There is
    nothing to ablate, which is why this arm accepts only --sketch-modes blank.

    The metadata guard is not decoration. Pointing this class at the
    SketchPromptVLA server would produce a fine-tune number wearing a baseline
    label, and every earlier round of this investigation lost time to exactly
    that class of mistake.
    """

    def __init__(self, host, port):
        from openpi_client import image_tools
        from openpi_client import websocket_client_policy

        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)
        metadata = self.client.get_server_metadata() or {}
        if metadata.get("model_variant") or metadata.get("checkpoint_dir"):
            raise RuntimeError(
                "port %d is serving a SketchPromptVLA checkpoint (%r); the "
                "baseline arm needs serve_policy.py --env LIBERO" % (port, metadata))
        # openpi renders LIBERO at 256 and pad-resizes to 224; pi0.5-LIBERO has
        # never seen another input size, and a wrong one degrades the score
        # silently instead of raising. sec.RESIZE_SIZE is the harness's constant,
        # not openpi's, so it is checked rather than trusted.
        if sec.RESIZE_SIZE != 224:
            raise RuntimeError(
                "sketch_eval_common.RESIZE_SIZE is %r; pi0.5-LIBERO expects 224. "
                "Fix the constant or the baseline arm is not the baseline."
                % (sec.RESIZE_SIZE,))
        self.metadata = metadata
        self.resize_fn = image_tools.resize_with_pad
        self.to_uint8 = image_tools.convert_to_uint8
        self.plan = collections.deque()

    def reset(self, instruction, sketch):
        if sketch is not None and sketch.mode != "blank":
            raise RuntimeError("pi05 cannot be handed a sketch (mode=%r)" % sketch.mode)
        self.instruction = instruction
        self.plan.clear()

    def element(self, obs):
        image = _rot180(obs[sec.AGENTVIEW_KEY])
        wrist = _rot180(obs[WRIST_KEY])
        state = np.concatenate((
            np.asarray(obs["robot0_eef_pos"], dtype=float),
            quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=float),
        ))
        if state.shape[0] != 8:
            raise RuntimeError("state is %d-D, pi0.5-LIBERO expects 8" % state.shape[0])
        return {
            "observation/image": self.to_uint8(
                self.resize_fn(image, sec.RESIZE_SIZE, sec.RESIZE_SIZE)),
            "observation/wrist_image": self.to_uint8(
                self.resize_fn(wrist, sec.RESIZE_SIZE, sec.RESIZE_SIZE)),
            "observation/state": state,
            "prompt": self.instruction,
        }

    def act(self, obs):
        if not self.plan:
            actions = np.asarray(self.client.infer(self.element(obs))["actions"])
            if len(actions) < sec.REPLAN_STEPS:
                raise RuntimeError("policy returned only %d actions" % len(actions))
            self.plan.extend(actions[:sec.REPLAN_STEPS])
        return np.asarray(self.plan.popleft(), dtype=float)


class SketchVlaPolicy:
    def __init__(self, host, port, variant, checkpoint):
        from openpi_client import image_tools
        from openpi_client import websocket_client_policy

        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)
        metadata = self.client.get_server_metadata() or {}
        if metadata.get("model_variant") != variant:
            raise RuntimeError("server variant mismatch: %r" % metadata)
        served = metadata.get("checkpoint_dir")
        if served and os.path.realpath(served) != os.path.realpath(checkpoint):
            raise RuntimeError("server checkpoint mismatch: %s != %s" % (served, checkpoint))
        self.resize_fn = image_tools.resize_with_pad
        self.to_uint8 = image_tools.convert_to_uint8
        self.plan = collections.deque()

    def reset(self, instruction, sketch):
        self.instruction = instruction
        self.sketch = sketch
        self.plan.clear()

    def act(self, obs):
        if not self.plan:
            element = E.build_element(
                obs, self.sketch, self.instruction, self.resize_fn, self.to_uint8,
                sec.RESIZE_SIZE, rotate180=True,
            )
            actions = np.asarray(self.client.infer(element)["actions"])
            if len(actions) < sec.REPLAN_STEPS:
                raise RuntimeError("policy returned only %d actions" % len(actions))
            self.plan.extend(actions[:sec.REPLAN_STEPS])
        return np.asarray(self.plan.popleft(), dtype=float)


def bowl_z(sim, name):
    return float(sim.data.body_xpos[B.A.bid_of(sim.model, name), 2])


def run_episode(env, policy, sketch, caption, mode, max_steps, success_window=1):
    """One episode. `success_window` consecutive successful steps end it.

    window=1 is the criterion every V7 chunk was already run under: stop at the
    first step `_check_success()` returns True. The anchored baselines in the
    evaluation repo used sustained-5 instead, so both are recorded on every row
    -- `success` is the instantaneous test and `sustained_success` the windowed
    one -- and a run at window=1 reports `sustained_success` only for episodes
    that happened to hold it anyway. To compare against a sustained-5 number,
    run the arm at --success-window 5 and read the sustained column; do not mix
    a windowed number with a window=1 one.
    """
    obs = env.env._get_observations()
    sim = env.env.sim
    z0 = {b: bowl_z(sim, b) for b in BOWLS}
    lift = {b: 0.0 for b in BOWLS}
    policy.reset(caption, sketch)
    task_success = False
    sustained = False
    streak = 0
    for step in range(max_steps):
        obs, *_ = env.step(policy.act(obs).tolist())
        for bowl in BOWLS:
            lift[bowl] = max(lift[bowl], bowl_z(sim, bowl) - z0[bowl])
        now = bool(env.env._check_success())
        task_success = task_success or now
        streak = streak + 1 if now else 0
        sustained = sustained or streak >= success_window
        if mode == "swap":
            if max(lift.values()) > LIFT_TH:
                break
        elif sustained:
            break
    grasped = max(lift, key=lift.get)
    if lift[grasped] <= LIFT_TH:
        grasped = None
    desired = BOWLS[1] if mode == "swap" else BOWLS[0]
    return {
        "success": task_success,
        "sustained_success": sustained,
        "referent_success": grasped == desired,
        "wrong_bowl": grasped is not None and grasped != desired,
        "grasped": grasped,
        "steps": step + 1,
        "lift": {b: round(v, 4) for b, v in lift.items()},
    }


def summarise(rows):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row["mode"], row["task"])].append(row)
        grouped[(row["mode"], "ALL")].append(row)
    out = {}
    for (mode, task), values in sorted(grouped.items()):
        n = len(values)
        out["%s|%s" % (mode, task)] = {
            "n": n,
            "success": round(sum(v["success"] for v in values) / n, 4),
            "sustained_success": round(
                sum(v.get("sustained_success", False) for v in values) / n, 4),
            "p_bowl2": round(sum(v["grasped"] == BOWLS[1] for v in values) / n, 4),
            "referent_success": round(sum(v["referent_success"] for v in values) / n, 4),
            "wrong_bowl": round(sum(v["wrong_bowl"] for v in values) / n, 4),
            "grasped_any": round(sum(v["grasped"] is not None for v in values) / n, 4),
        }
    return out


def write_result(path, rows, args, started):
    result = {
        "rows": rows,
        "summary": summarise(rows),
        "config": vars(args),
        "wall_s": round(time.time() - started, 1),
    }
    provenance.write_json(path, result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bddl-dir", required=True)
    ap.add_argument("--demo-dir", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--policy", default="sketchvla", choices=("sketchvla", "pi05"),
                    help="sketchvla: the fine-tune, sketch delivered. "
                         "pi05: stock pi0.5-LIBERO, no sketch -- the baseline arm.")
    ap.add_argument("--checkpoint", default=None,
                    help="required for --policy sketchvla; the server must serve it")
    ap.add_argument("--variant", default="referent_grounding")
    ap.add_argument("--caption", default="stored", choices=("stored", "explicit"),
                    help="stored: the corpus's referent-free caption. "
                         "explicit: the layout's BDDL wording, which names the bowl.")
    ap.add_argument("--success-window", type=int, default=1,
                    help="consecutive successful steps that end an episode; "
                         "1 reproduces every V7 chunk, 5 the anchored baselines")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tasks", default=",".join(B.T))
    ap.add_argument("--sketch-modes", default="real,swap")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=520)
    ap.add_argument("--max-frame-error", type=float, default=5.0,
                    help="Maximum mean RGB error versus the packed frame-0 reference")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    modes = [mode for mode in args.sketch_modes.split(",") if mode]
    if set(modes) - {"real", "swap", "blank"}:
        raise SystemExit("--sketch-modes accepts only real,swap,blank")
    if args.success_window < 1:
        raise SystemExit("--success-window must be at least 1")
    if args.policy == "pi05":
        # A "real" or "swap" arm on pi0.5 would be the identical input scored
        # against two different desired referents -- the same 200 episodes run
        # twice and reported as an effect. Only blank is meaningful.
        if modes != ["blank"]:
            raise SystemExit("--policy pi05 has no sketch channel: use "
                             "--sketch-modes blank")
        if args.checkpoint:
            raise SystemExit("--checkpoint is meaningless for the stock baseline")
        policy = Pi05Policy(args.host, args.port)
    else:
        if not args.checkpoint:
            raise SystemExit("--policy sketchvla requires --checkpoint")
        policy = SketchVlaPolicy(args.host, args.port, args.variant, args.checkpoint)
    rows = []
    started = time.time()

    from libero.libero.envs import OffScreenRenderEnv

    for mode in modes:
        for task_key in [key for key in args.tasks.split(",") if key]:
            task = B.T[task_key]
            donor_key, fixture = B.PAIRING[task_key]
            episode_paths = sorted(
                pathlib.Path(args.frames_dir, task_key).glob("demo_*.npz"),
                key=lambda path: int(path.stem.split("_")[1]),
            )[:args.episodes]
            if len(episode_paths) != args.episodes:
                raise RuntimeError("%s has %d usable episodes, wanted %d" %
                                   (task_key, len(episode_paths), args.episodes))
            env = OffScreenRenderEnv(
                bddl_file_name=os.path.join(args.bddl_dir, task + ".bddl"),
                camera_heights=256, camera_widths=256,
                camera_names=["agentview", "robot0_eye_in_hand"],
            )
            try:
                with h5py.File(os.path.join(args.demo_dir, task + "_demo.hdf5"), "r") as source:
                    donor = (h5py.File(os.path.join(args.demo_dir, B.T[donor_key] + "_demo.hdf5"), "r")
                             if donor_key else None)
                    donor_keys = (sorted(donor["data"], key=lambda key: int(key.split("_")[1]))
                                  if donor else [])
                    for episode_path in episode_paths:
                        demo = episode_path.stem
                        index = int(demo.split("_")[1])
                        flat = source["data"][demo]["states"][0]
                        env.reset()
                        env.set_init_state(flat)
                        sim = env.env.sim
                        if donor is not None:
                            donor_state = donor["data"][donor_keys[index % len(donor_keys)]]["states"][0]
                            address = sim.model.get_joint_qpos_addr("akita_black_bowl_1_joint0")
                            donor_pose = np.asarray(donor_state[1 + address[0]:1 + address[1]], float)
                            pose = B.map_donor_pose(sim, donor_pose, fixture, donor_state, flat)
                            bowl2 = sim.model.get_joint_qpos_addr("akita_black_bowl_2_joint0")
                            qpos = sim.data.qpos.copy()
                            qpos[bowl2[0]:bowl2[1]] = pose
                            sim.data.qpos[:] = qpos
                            sim.forward()
                        for _ in range(5):
                            obs, *_ = env.step(np.zeros(7, np.float32))

                        sketch = StoredSketch(str(episode_path), mode)
                        live = np.ascontiguousarray(np.rot90(obs[sec.AGENTVIEW_KEY], 2))
                        mean_frame_error = float(np.abs(live.astype(float) - sketch.reference_frame).mean())
                        if mean_frame_error > args.max_frame_error:
                            raise RuntimeError("%s frame mismatch %.3f" % (episode_path, mean_frame_error))
                        caption = (explicit_caption(task_key)
                                   if args.caption == "explicit" else sketch.caption)
                        row = run_episode(env, policy, sketch, caption, mode,
                                          args.max_steps, args.success_window)
                        row.update(task=task_key, mode=mode, demo=demo, donor=donor_key,
                                   policy=args.policy, caption_mode=args.caption,
                                   caption=caption, success_window=args.success_window,
                                   frame_error=round(mean_frame_error, 5))
                        rows.append(row)
                        write_result(args.out, rows, args, started)
                        print("%s/%s/%s success=%s referent=%s grasped=%s frame_err=%.4f" %
                              (mode, task_key, demo, row["success"], row["referent_success"],
                               row["grasped"], mean_frame_error), flush=True)
                    if donor is not None:
                        donor.close()
            finally:
                env.close()

    write_result(args.out, rows, args, started)
    print(json.dumps(summarise(rows), indent=2), flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
