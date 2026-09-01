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
    Since 1 Sep the counterfactual half as well: circle_swap/arrow_swap drawn
    round the DISTRACTOR bowl, its visibility mask `distractor`, and
    distractor_meta / arrow_swap_start / arrow_swap_end. Same code, same stroke,
    same frame -- these are what makes the corpus sketch-necessary at the level
    of the loss rather than only at the level of the BDDL goal.
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
    "t1": ("t5", None),                  # bowl_2 -> next_to_plate
    "t2": ("t1", None),                  # bowl_2 -> between_plate_ramekin
    "t3": ("t4", None),                  # bowl_2 -> next_to_box
    "t4": ("t3", "wooden_cabinet_1"),    # bowl_2 -> INTO top drawer, cabinet-mapped
    "t5": (None, None),
    "t6": ("t4", None),                  # bowl_2 -> next_to_box
    "t7": ("t8", "glazed_rim_porcelain_ramekin_1"),  # bowl_2 -> stacked on ramekin
    "t8": (None, None),
    "t9": (None, None),                  # the shipped ambiguous pair
    "t10": (None, None),
}
DEFAULT_CAPTIONS = ["do this", "grab this", "put this over there"]

TARGET = "akita_black_bowl_1"
DISTRACTOR = "akita_black_bowl_2"
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


def _marks_for(model, data, W2P, inst, margin):
    """Circle around `inst` and the arrow from it to the destination."""
    cx, cy = A.project([A.vcenter(model, data, A.bid_of(model, inst))], W2P)[0]
    r = A.px_extent(model, data, W2P, inst)
    rx = ry = max(10.0, float(r) * margin)
    dx, dy = A.project([A.vcenter(model, data, A.bid_of(model, DEST))], W2P)[0]

    circle_tok = {"cx": float(cx), "cy": float(cy), "rx": float(rx), "ry": float(ry)}
    v = np.array([dx - cx, dy - cy], float)
    n = np.linalg.norm(v) or 1.0
    u = v / n
    start = np.array([cx, cy]) + u * max(rx, ry)
    end = np.array([dx, dy], float)
    if np.linalg.norm(end - start) < 12.0:
        start = np.array([cx, cy], float)
    arrow_tok = {"x0": float(start[0]), "y0": float(start[1]),
                 "x1": float(end[0]), "y1": float(end[1])}

    meta = np.array([circle_tok["cx"], circle_tok["cy"],
                     (circle_tok["rx"] + circle_tok["ry"]) / 2.0], np.float32)
    return (EF.circle_mask(circle_tok), EF.arrow_mask(arrow_tok), meta,
            np.array([arrow_tok["x0"], arrow_tok["y0"]], np.float32),
            np.array([arrow_tok["x1"], arrow_tok["y1"]], np.float32))


def sketch_from_frame0(env, margin=1.35):
    """Sketch marks and object masks for BOTH bowls, in raw (projection-path)
    orientation; the caller rotates.

    The distractor half is what `referent_grounding` (v7) trains against. Two
    things it needs and the target-only version could not give it: a dense
    grounding label -- which pixels the circled object occupies -- and a
    COUNTERFACTUAL, the same scene with the circle drawn round the other bowl.
    Without the counterfactual a grounding head can score perfectly by learning
    "the circled object is the one I was going to reach for anyway", which is
    the failure mode every run of this project has produced so far.

    Both bowls are the same asset, so the distractor's marks are rendered by the
    same code at the same stroke -- pixel-exact with the real circle rather than
    an approximation drawn later from `circle_meta`.
    """
    from robosuite.utils import camera_utils as CU
    sim = env.env.sim
    model, data = sim.model, sim.data
    W2P = CU.get_camera_transform_matrix(sim=sim, camera_name=A.CAMERA,
                                         camera_height=RES, camera_width=RES)
    circle_px, arrow_px, meta, a0, a1 = _marks_for(model, data, W2P, TARGET, margin)
    dis_circle, dis_arrow, dis_meta, dis_a0, dis_a1 = _marks_for(
        model, data, W2P, DISTRACTOR, margin)

    inst = [TARGET, DISTRACTOR]
    _, target_vis = A.visibility(env, model, data, inst, TARGET)
    _, distractor_vis = A.visibility(env, model, data, inst, DISTRACTOR)
    return dict(
        circle=circle_px, arrow=arrow_px,
        target=target_vis.reshape(RES, RES, 1).astype(np.uint8),
        circle_meta=meta, arrow_start=a0, arrow_end=a1,
        circle_swap=dis_circle, arrow_swap=dis_arrow,
        distractor=distractor_vis.reshape(RES, RES, 1).astype(np.uint8),
        distractor_meta=dis_meta, arrow_swap_start=dis_a0, arrow_swap_end=dis_a1,
    )


_MASK_KEYS = ("circle", "arrow", "target", "circle_swap", "arrow_swap", "distractor")
_XY_KEYS = ("arrow_start", "arrow_end", "arrow_swap_start", "arrow_swap_end")
_META_KEYS = ("circle_meta", "distractor_meta")


def rot_all(images, wrist, sketch):
    """Rotate frames and every sketch field into the modified_libero_rlds
    orientation. Every field turns together or none does -- a mask left in the
    projection orientation is silently 180 degrees from the pixels it labels."""
    images = np.ascontiguousarray(images[:, ::-1, ::-1])
    wrist = np.ascontiguousarray(wrist[:, ::-1, ::-1])
    out = dict(sketch)
    for k in _MASK_KEYS:
        out[k] = EF.rot180_image(sketch[k])
    for k in _META_KEYS:
        cx, cy = EF.rot180_xy(sketch[k][0], sketch[k][1])
        out[k] = np.array([cx, cy, sketch[k][2]], np.float32)
    for k in _XY_KEYS:
        out[k] = np.array(EF.rot180_xy(*sketch[k]), np.float32)
    return images, wrist, out


def build_task(tkey, bddl_dir, demo_dir, out_dir, captions, limit, rng,
               offset=0, stride=1, resume=False):
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
            # one task can be split across processes: every stride-th demo from
            # offset. The donor index below is i's position in the FULL list, so
            # a demo draws the same donor whether or not the task was split.
            keys = list(enumerate(keys))[offset::stride]
            if resume:
                before = len(keys)
                keys = [(i, k) for i, k in keys
                        if not os.path.exists(
                            os.path.join(out_dir, tkey, f"{k}.npz"))]
                print(f"  {tkey}: resume, {before - len(keys)} already built",
                      flush=True)
            dkeys = (sorted(donor_f["data"].keys(), key=lambda k: int(k.split("_")[1]))
                     if donor_f else [])
            for i, k in keys:
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

                sketch = sketch_from_frame0(env)

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
                images, wrists, sketch_r = rot_all(images, wrists, sketch)
                cap = captions[rng.integers(len(captions))]
                np.savez_compressed(
                    os.path.join(out_dir, tkey, f"{k}.npz"),
                    images=images, wrist=wrists,
                    states=np.stack(states), joint_states=np.stack(joints),
                    actions=np.stack(acts),
                    caption=np.array(cap),
                    episode_key=np.array(f"paired_spatial/{tkey}/{k}"),
                    **sketch_r)
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
    ap.add_argument("--resume", action="store_true",
                    help="skip demos whose npz already exists (a demo that was "
                         "dropped by the success filter has no npz and is "
                         "retried, which is cheap and deterministic)")
    ap.add_argument("--offset", type=int, default=0,
                    help="start at this demo index (with --stride, splits one "
                         "task across processes)")
    ap.add_argument("--stride", type=int, default=1,
                    help="take every stride-th demo")
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
                                 captions, args.limit, rng,
                                 args.offset, args.stride, args.resume)
        total_kept += kept
        total_tried += tried
        print(f"== {tkey}: kept {kept}/{tried}", flush=True)
    print(f"\nTOTAL kept {total_kept}/{total_tried} -> {args.out}")


if __name__ == "__main__":
    main()
