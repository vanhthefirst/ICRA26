"""
Hardening PROBE (run in WSL2, libero env). Validates two new checks on a few
existing validation scenes before we fold them into the builder as gates:

  (A) target-visibility  : segmentation-based occlusion fraction of the target bowl
  (B) scripted-grasp     : can a simple top-down OSC grasp lift the target bowl?

Also reports target->nearest-neighbour clearance. Writes debug images and a
JSON so we can calibrate thresholds. NON-destructive: reads existing scenes,
writes only into outputs/validation_set/_harden_probe/.

    conda activate <libero env>
    cd /mnt/c/Users/Admin/sketch_vla
    python scripts/test_harden_wsl.py 2>&1 | tee outputs/validation_set/_harden_probe/log.txt
"""

import os, json, glob
import numpy as np
import cv2

ROOT = "/mnt/c/Users/Admin/sketch_vla/outputs/validation_set"
DBG  = os.path.join(ROOT, "_harden_probe")
os.makedirs(DBG, exist_ok=True)
IMG_H = IMG_W = 128
CAMERA = "agentview"


def frame_from_obs(obs):
    key = "agentview_image" if "agentview_image" in obs else \
          [k for k in obs if "agentview" in k and "image" in k][0]
    f = np.asarray(obs[key])
    if f.dtype != np.uint8:
        f = np.clip(f * 255.0 if f.max() <= 1.0 + 1e-6 else f, 0, 255).astype(np.uint8)
    return f.copy()


def visual_center(model, data, bid):
    gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
    ref = [g for g in gids if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
    return data.geom_xpos[ref[0]].copy() if ref else data.body_xpos[bid].copy()


def body_id(model, name):
    for c in (name, f"{name}_main"):
        if c in model.body_names:
            return model.body_name2id(c)
    raise KeyError(name)


def free_joint_of(model, prefix):
    for j in range(model.njnt):
        nm = model.joint_id2name(j)
        if nm and nm.startswith(prefix) and "joint" in nm:
            return j
    return None


# ─────────────────────── (A) visibility via segmentation ───────────────────────
def target_geom_ids(model, bid):
    return set(g for g in range(model.ngeom) if model.geom_bodyid[g] == bid)


def seg_count(env, tgt_geom_ids):
    """Render instance/geom segmentation and count target-geom pixels.
    Robosuite render(segmentation=True) -> (H,W,2); channel 0 is geom id
    (may be offset by +1). We match by set membership after trying offsets."""
    seg = env.sim.render(width=IMG_W, height=IMG_H, camera_name=CAMERA, segmentation=True)
    ch0 = seg[..., 0].astype(np.int64)
    # try direct and -1 offset; pick whichever hits target geoms
    best = 0; best_ids = ch0
    for off in (0, -1, 1):
        ids = ch0 + off
        c = int(np.isin(ids, list(tgt_geom_ids)).sum())
        if c > best:
            best, best_ids = c, ids
    mask = np.isin(best_ids, list(tgt_geom_ids))
    return int(mask.sum()), mask, seg


def visibility(env, model, data, tb_bid, other_bodies):
    tgt_geoms = target_geom_ids(model, tb_bid)
    v_occ, mask_occ, _ = seg_count(env, tgt_geoms)
    # hide other movable objects far below, re-render
    saved = {}
    for name in other_bodies:
        j = free_joint_of(model, name)
        if j is None:
            continue
        qa = model.jnt_qposadr[j]
        saved[qa] = data.qpos[qa:qa + 3].copy()
        data.qpos[qa:qa + 3] = [data.qpos[qa], data.qpos[qa + 1], -1.0]
    env.sim.forward()
    v_full, _, _ = seg_count(env, tgt_geoms)
    for qa, v in saved.items():
        data.qpos[qa:qa + 3] = v
    env.sim.forward()
    vis = v_occ / max(v_full, 1)
    return dict(v_occluded=v_occ, v_full=v_full, visibility=round(vis, 3)), mask_occ


# ─────────────────────── (B) scripted top-down grasp ───────────────────────
def eef(obs):
    for k in ("robot0_eef_pos", "eef_pos"):
        if k in obs:
            return np.asarray(obs[k])
    return None


def servo(env, obs, goal_xyz, grip, steps, gain=8.0):
    for _ in range(steps):
        e = eef(obs)
        act = np.zeros(env_action_dim)
        if e is not None:
            act[:3] = np.clip((goal_xyz - e) * gain, -1, 1)
        act[-1] = grip
        obs, _, _, _ = env.step(act)
    return obs


def scripted_grasp(env, obs, model, data, tb_bid, tb_name):
    """Open, hover, descend, close, lift. Success if bowl rises > 3 cm.
    Tries gripper close=+1 first, then -1 if that fails."""
    global env_action_dim
    j = free_joint_of(model, tb_name)
    qa = model.jnt_qposadr[j]
    z0 = float(data.qpos[qa + 2])
    tc = visual_center(model, data, tb_bid)
    for close_sign in (+1.0, -1.0):
        # reset arm state cheaply by re-settling is expensive; just re-run sequence
        obs = servo(env, obs, tc + [0, 0, 0.12], -close_sign, 30)   # open, hover
        obs = servo(env, obs, tc + [0, 0, 0.005], -close_sign, 25)  # descend
        obs = servo(env, obs, tc + [0, 0, 0.005], close_sign, 12)   # close
        obs = servo(env, obs, tc + [0, 0, 0.18], close_sign, 30)    # lift
        z1 = float(data.qpos[qa + 2])
        if z1 - z0 > 0.03:
            return dict(grasp_success=True, lift=round(z1 - z0, 3), close_sign=close_sign), obs
    return dict(grasp_success=False, lift=round(float(data.qpos[qa + 2]) - z0, 3)), obs


env_action_dim = 7


def run_scene(scene_dir):
    global env_action_dim
    from libero.libero.envs import OffScreenRenderEnv
    meta = json.load(open(os.path.join(scene_dir, "meta.json")))
    bddl = os.path.join(scene_dir, "scene.bddl")
    tb, tp = meta["target_bowl"], meta["target_plate"]
    name = os.path.basename(scene_dir)
    out = {"scene": name, "tier": meta["tier"], "target_bowl": tb}

    # --- grasp pass (fresh env) ---
    np.random.seed(meta["seed"])
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=IMG_H,
                             camera_widths=IMG_W, camera_names=[CAMERA])
    obs = env.reset()
    env_action_dim = getattr(env, "action_dim", 7)
    for _ in range(20):
        obs, _, _, _ = env.step(np.zeros(env_action_dim))
    model, data = env.sim.model, env.sim.data
    tb_bid = body_id(model, tb)

    # clearance
    tpos = visual_center(model, data, tb_bid)
    others = [i for i in meta["instances"] if i != tb]
    dists = [float(np.linalg.norm(visual_center(model, data, body_id(model, o))[:2] - tpos[:2]))
             for o in others]
    out["clearance_xy"] = round(min(dists), 3) if dists else None

    g, obs = scripted_grasp(env, obs, model, data, tb_bid, tb)
    out.update(g)
    cv2.imwrite(os.path.join(DBG, f"{name}_postgrasp.png"),
                cv2.cvtColor(frame_from_obs(obs), cv2.COLOR_RGB2BGR))
    env.close()

    # --- visibility pass (fresh env) ---
    np.random.seed(meta["seed"])
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=IMG_H,
                             camera_widths=IMG_W, camera_names=[CAMERA])
    obs = env.reset()
    for _ in range(20):
        obs, _, _, _ = env.step(np.zeros(env_action_dim))
    model, data = env.sim.model, env.sim.data
    tb_bid = body_id(model, tb)
    try:
        vis, mask = visibility(env, model, data, tb_bid, others)
        out.update(vis)
        m = (mask.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(DBG, f"{name}_targetmask.png"), m)
    except Exception as e:
        out["visibility_error"] = str(e)
    env.close()
    return out


def main():
    scenes = sorted(glob.glob(os.path.join(ROOT, "scene_*")))
    if not scenes:
        print("no scenes found"); return
    # pick easy / mid / dense
    picks = [scenes[0], scenes[len(scenes) // 2], scenes[-1]]
    results = []
    for sd in picks:
        print("\n=== probing", os.path.basename(sd), "===")
        r = run_scene(sd)
        print(json.dumps(r, indent=2))
        results.append(r)
    json.dump(results, open(os.path.join(DBG, "probe_results.json"), "w"), indent=2)
    print("\nDONE. Send back outputs/validation_set/_harden_probe/ "
          "(probe_results.json, *_postgrasp.png, *_targetmask.png, log.txt)")


if __name__ == "__main__":
    main()
