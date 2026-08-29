#!/usr/bin/env python3
"""Build the paired upright Spatial corpus: stage 1, replay + render (no TF).

WHY THIS CORPUS
  pcla_v2/v3/v4 proved the sketch pathway works and the corpus is the fault:
  every LIBERO-Spatial layout but one carries a single sketch, so the image
  alone determines the action and ignoring the sketch costs nothing
  (STATE.md 28 Aug). And the corpus was 180 degrees from pi0.5's world —
  measured fatal on 29 Aug (stock pi0.5: 96.7% upright, 0.0% inverted).
  This builder fixes both at once: five layout pairs where only the sketch
  says which bowl to take, rendered upright.

MECHANISM (all measured, see outputs/paired_spatial_bddl/replay_pair_matrix.log)
  For pair (Ti, Tj): replay Ti's shipped demos with Ti's DISTRACTOR bowl set
  to the pose of Tj's TARGET bowl, taken from Tj's demo of the same index
  (full 7-DoF free-joint qpos out of the donor's states[0]). Both scenes then
  show bowls at {rI, rJ}; the trajectory still reaches Ti's target. Episodes
  are kept ONLY if the goal predicate still succeeds after open-loop replay,
  so every kept episode is a verified (image, sketch, action) triple.

  Fixture-attached donor poses (drawer, cabinet top, stove) are mapped
  through the fixture's base pose in each episode's qpos when the fixture has
  a free joint; otherwise the donor pose is used as-is. This is the fix for
  the one weak matrix direction (bowl into the closed drawer: 12/25 raw).

OUTPUT
  One .npz per kept episode under --out/<task>/demo_<i>.npz:
    images (T,256,256,3) u8, wrist (T,256,256,3) u8, states (T,8) f32,
    actions (T,7) f32, circle/arrow/target (256,256,1) u8 static masks,
    circle_meta (3,) arrow_start/end (2,) f32, plus caption + episode_key.
  Everything already rotated 180 into the modified_libero_rlds orientation.
  Stage 2 (scripts/pack_paired_corpus.py, TF env) writes the RLDS.

RUNS IN the py3.8 LIBERO client venv (or any robosuite-1.4/mujoco-2.3.7 env):
    python scripts/build_paired_corpus.py --demo-dir <hdf5 dir> --out <dir> \
        [--tasks t6,...] [--limit N] [--captions captions.json]
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_validation_set_spatial_anchored as A
import export_rlds_frames as EF

RES = 256
A.IMG_H = A.IMG_W = RES

T = {
    "t1": "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "t2": "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "t3": "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    "t4": "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "t5": "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    "t6": "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "t7": "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "t8": "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "t9": "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    "t10": "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
}
# replay task -> (donor task or None, fixture whose frame the donor pose lives in
# or None). Donor bowl is always the donor task's TARGET, akita_black_bowl_1.
# None donor = the shipped bowl_2 position already matches the pair layout.
PAIRING = {
    "t1": ("t2", None),                  # bowl_2 -> table_center
    "t2": ("t1", None),                  # bowl_2 -> between_plate_ramekin
    "t3": ("t4", None),                  # bowl_2 -> next_to_box
    "t4": ("t3", "wooden_cabinet_1"),    # bowl_2 -> INTO top drawer, cabinet-mapped
    "t5": (None, None),
    "t6": ("t5", None),                  # bowl_2 -> next_to_plate
    "t7": ("t8", "glazed_rim_porcelain_ramekin_1"),  # bowl_2 -> stacked on ramekin
    "t8": (None, None),
    "t9": (None, None),                  # the shipped ambiguous pair
    "t10": (None, None),
}
DEFAULT_CAPTIONS = ["do this", "grab this", "put this over there"]

TARGET = "akita_black_bowl_1"
DEST = "plate_1"


def quat_mul(q, p):  # wxyz
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = p
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rot(q, v):
    return quat_mul(quat_mul(q, np.concatenate([[0.0], v])), quat_conj(q))[1:]


def joint_qpos_addr(sim, name):
    try:
        adr = sim.model.get_joint_qpos_addr(name)
    except Exception:
        return None
    return adr if isinstance(adr, tuple) else None  # free joints only


def fixture_pose_from_state(sim, fixture, flat_state):
    """Fixture base pose (pos, wxyz quat) out of a flattened MjSimState, or None."""
    adr = joint_qpos_addr(sim, f"{fixture}_joint0")
    if adr is None:
        return None
    q = flat_state[1 + adr[0]: 1 + adr[1]]
    return np.asarray(q[:3], float), np.asarray(q[3:7], float)


def map_donor_pose(sim, donor_pose7, fixture, donor_state, replay_state):
    """Donor bowl pose re-expressed in the replay episode's fixture frame."""
    if fixture is None:
        return donor_pose7
    d = fixture_pose_from_state(sim, fixture, donor_state)
    r = fixture_pose_from_state(sim, fixture, replay_state)
    if d is None or r is None:
        return donor_pose7  # fixture is welded; poses identical across episodes
    dp, dq = d
    rp, rq = r
    bp, bq = np.asarray(donor_pose7[:3], float), np.asarray(donor_pose7[3:7], float)
    local_p = quat_rot(quat_conj(dq), bp - dp)
    local_q = quat_mul(quat_conj(dq), bq)
    out_p = rp + quat_rot(rq, local_p)
    out_q = quat_mul(rq, local_q)
    return np.concatenate([out_p, out_q])


def sketch_from_frame0(env, margin=1.35):
    """Circle around the target bowl, arrow to the plate, target visibility mask.
    Computed in raw (projection-path) orientation; caller rotates."""
    from robosuite.utils import camera_utils as CU
    sim = env.env.sim
    model, data = sim.model, sim.data
    W2P = CU.get_camera_transform_matrix(sim=sim, camera_name=A.CAMERA,
                                         camera_height=RES, camera_width=RES)
    cx, cy = A.project([A.vcenter(model, data, A.bid_of(model, TARGET))], W2P)[0]
    # px_extent returns the half-extent radius in pixels (a float)
    r = A.px_extent(model, data, W2P, TARGET)
    rx = ry = max(10.0, float(r) * margin)
    dx, dy = A.project([A.vcenter(model, data, A.bid_of(model, DEST))], W2P)[0]

    circle_tok = {"cx": float(cx), "cy": float(cy), "rx": float(rx), "ry": float(ry)}
    # arrow: circle edge -> destination centre; if the two nearly touch, start
    # from the circle centre so the arrow stays legible instead of degenerating
    v = np.array([dx - cx, dy - cy], float)
    n = np.linalg.norm(v) or 1.0
    u = v / n
    start = np.array([cx, cy]) + u * max(rx, ry)
    end = np.array([dx, dy], float)
    if np.linalg.norm(end - start) < 12.0:
        start = np.array([cx, cy], float)
    arrow_tok = {"x0": float(start[0]), "y0": float(start[1]),
                 "x1": float(end[0]), "y1": float(end[1])}

    circle_px = EF.circle_mask(circle_tok)
    arrow_px = EF.arrow_mask(arrow_tok)
    inst = [n for n in (TARGET, "akita_black_bowl_2") if True]
    _, vismask = A.visibility(env, model, data, inst, TARGET)
    target_mask = vismask.reshape(RES, RES, 1).astype(np.uint8)
    meta = np.array([circle_tok["cx"], circle_tok["cy"],
                     (circle_tok["rx"] + circle_tok["ry"]) / 2.0], np.float32)
    a0 = np.array([arrow_tok["x0"], arrow_tok["y0"]], np.float32)
    a1 = np.array([arrow_tok["x1"], arrow_tok["y1"]], np.float32)
    return circle_px, arrow_px, target_mask, meta, a0, a1


def rot_all(images, wrist, circle, arrow, target, meta, a0, a1):
    images = np.ascontiguousarray(images[:, ::-1, ::-1])
    wrist = np.ascontiguousarray(wrist[:, ::-1, ::-1])
    circle, arrow, target = (EF.rot180_image(m) for m in (circle, arrow, target))
    cx, cy = EF.rot180_xy(meta[0], meta[1])
    meta = np.array([cx, cy, meta[2]], np.float32)
    a0 = np.array(EF.rot180_xy(*a0), np.float32)
    a1 = np.array(EF.rot180_xy(*a1), np.float32)
    return images, wrist, circle, arrow, target, meta, a0, a1


def build_task(tkey, bddl_dir, demo_dir, out_dir, captions, limit, rng):
    from libero.libero.envs import OffScreenRenderEnv
    task = T[tkey]
    donor_key, fixture = PAIRING[tkey]
    demo_path = os.path.join(demo_dir, task + "_demo.hdf5")
    bddl = os.path.join(bddl_dir, task + ".bddl")
    os.makedirs(os.path.join(out_dir, tkey), exist_ok=True)

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES,
                             camera_widths=RES,
                             camera_names=["agentview", "robot0_eye_in_hand"])
    kept = tried = 0
    try:
        with h5py.File(demo_path, "r") as f:
            donor_f = h5py.File(os.path.join(
                demo_dir, T[donor_key] + "_demo.hdf5"), "r") if donor_key else None
            keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            if limit:
                keys = keys[:limit]
            dkeys = (sorted(donor_f["data"].keys(), key=lambda k: int(k.split("_")[1]))
                     if donor_f else [])
            for i, k in enumerate(keys):
                tried += 1
                d = f["data"][k]
                flat = d["states"][0]
                env.reset()
                env.set_init_state(flat)
                sim = env.env.sim
                if donor_f is not None:
                    dstate = donor_f["data"][dkeys[i % len(dkeys)]]["states"][0]
                    da = sim.model.get_joint_qpos_addr("akita_black_bowl_1_joint0")
                    pose = np.asarray(dstate[1 + da[0]: 1 + da[1]], float)
                    pose = map_donor_pose(sim, pose, fixture, dstate, flat)
                    b2 = sim.model.get_joint_qpos_addr("akita_black_bowl_2_joint0")
                    q = sim.data.qpos.copy()
                    q[b2[0]:b2[1]] = pose
                    sim.data.qpos[:] = q
                    sim.forward()
                # settle physics briefly so a mapped pose can seat itself
                for _ in range(5):
                    obs, *_ = env.step(np.zeros(7, np.float32))

                circle, arrow, target, meta, a0, a1 = sketch_from_frame0(env)

                images, wrists, states, joints, acts = [], [], [], [], []
                obs = env.env._get_observations()
                for a in d["actions"][:]:
                    images.append(EF.as_uint8(obs["agentview_image"]))
                    wrists.append(EF.as_uint8(obs["robot0_eye_in_hand_image"]))
                    states.append(EF.state_vector(obs))
                    joints.append(np.asarray(obs["robot0_joint_pos"], np.float32))
                    acts.append(np.asarray(a, np.float32))
                    obs, *_ = env.step(a)
                if not env.env._check_success():
                    print(f"  {tkey}/{k}: goal failed after replay, dropped", flush=True)
                    continue

                images = np.stack(images)
                wrists = np.stack(wrists)
                images, wrists, circle_r, arrow_r, target_r, meta_r, a0_r, a1_r = \
                    rot_all(images, wrists, circle, arrow, target, meta, a0, a1)
                cap = captions[rng.integers(len(captions))]
                np.savez_compressed(
                    os.path.join(out_dir, tkey, f"{k}.npz"),
                    images=images, wrist=wrists,
                    states=np.stack(states), joint_states=np.stack(joints),
                    actions=np.stack(acts),
                    circle=circle_r, arrow=arrow_r, target=target_r,
                    circle_meta=meta_r, arrow_start=a0_r, arrow_end=a1_r,
                    caption=np.array(cap), episode_key=np.array(f"paired_spatial/{tkey}/{k}"))
                kept += 1
                print(f"  {tkey}/{k}: kept ({images.shape[0]} steps)", flush=True)
            if donor_f is not None:
                donor_f.close()
    finally:
        env.close()
    return kept, tried


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", required=True, help="shipped libero_spatial BDDLs")
    ap.add_argument("--demo-dir", required=True, help="dir of *_demo.hdf5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=",".join(T),
                    help="comma list of t1..t10")
    ap.add_argument("--limit", type=int, default=0, help="demos per task, 0=all")
    ap.add_argument("--captions", default=None,
                    help="json list of deictic captions; default the known three")
    args = ap.parse_args()

    captions = (json.load(open(args.captions)) if args.captions
                else DEFAULT_CAPTIONS)
    rng = np.random.default_rng(7)
    total_kept = total_tried = 0
    for tkey in args.tasks.split(","):
        tkey = tkey.strip()
        print(f"== {tkey}: {T[tkey]}", flush=True)
        kept, tried = build_task(tkey, args.bddl_dir, args.demo_dir, args.out,
                                 captions, args.limit, rng)
        total_kept += kept
        total_tried += tried
        print(f"== {tkey}: kept {kept}/{tried}", flush=True)
    print(f"\nTOTAL kept {total_kept}/{total_tried} -> {args.out}")


if __name__ == "__main__":
    main()
