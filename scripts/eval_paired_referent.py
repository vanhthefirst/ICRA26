#!/usr/bin/env python3
"""Evaluate real and swapped sketches on the exact ten paired LIBERO layouts."""

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


def run_episode(env, policy, sketch, mode, max_steps):
    obs = env.env._get_observations()
    sim = env.env.sim
    z0 = {b: bowl_z(sim, b) for b in BOWLS}
    lift = {b: 0.0 for b in BOWLS}
    policy.reset(sketch.caption, sketch)
    task_success = False
    for step in range(max_steps):
        obs, *_ = env.step(policy.act(obs).tolist())
        for bowl in BOWLS:
            lift[bowl] = max(lift[bowl], bowl_z(sim, bowl) - z0[bowl])
        task_success = bool(env.env._check_success())
        if (mode in ("real", "blank") and task_success) or (mode == "swap" and max(lift.values()) > LIFT_TH):
            break
    grasped = max(lift, key=lift.get)
    if lift[grasped] <= LIFT_TH:
        grasped = None
    desired = BOWLS[1] if mode == "swap" else BOWLS[0]
    return {
        "success": task_success,
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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--variant", default="referent_grounding")
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
                        row = run_episode(env, policy, sketch, mode, args.max_steps)
                        row.update(task=task_key, mode=mode, demo=demo, donor=donor_key,
                                   caption=sketch.caption, frame_error=round(mean_frame_error, 5))
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
