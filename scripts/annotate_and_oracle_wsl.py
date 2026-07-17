"""
Vertical-slice step 2: draw the sketch on the augmented scene + run the
teleport oracle. Run in WSL2 (libero env), AFTER gen_augmented_scene_wsl.py
has produced outputs/vslice/scene_0001.bddl + scene_0001_meta.json.

Produces:
  outputs/vslice/scene_0001_sketch.png   -- frame0 + green circle (target bowl)
                                            + red arrow (-> target plate)
  outputs/vslice/scene_0001_tokens.json  -- symbolic circle/arrow tokens
  outputs/vslice/scene_0001_oracle_before.png / _after.png
  console: success-check result of placing GT bowl on GT plate

Draw functions/colours are copied verbatim from generate_dataset_wsl.py so the
validation sketches match the training distribution exactly.

    conda activate <libero env>
    cd /mnt/c/Users/Admin/sketch_vla
    python scripts/annotate_and_oracle_wsl.py 2>&1 | tee outputs/vslice/oracle_log.txt
"""

import os, json, re
import numpy as np
import cv2

OUT_DIR = "/mnt/c/Users/Admin/sketch_vla/outputs/vslice"
BDDL    = os.path.join(OUT_DIR, "scene_0001.bddl")
META    = os.path.join(OUT_DIR, "scene_0001_meta.json")
IMG_H = IMG_W = 128
CAMERA = "agentview"


# ───────── draw funcs (verbatim from generate_dataset_wsl.py) ─────────
def draw_circle(img, center_px, radius, seed):
    rng = np.random.default_rng(seed)
    out = img.copy()
    cx = center_px[0] + int(rng.integers(-3, 4))
    cy = center_px[1] + int(rng.integers(-3, 4))
    rx = radius + float(rng.uniform(-2, 3))
    ry = radius + float(rng.uniform(-2, 3))
    n = 80
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    wobble = (rng.uniform(0.5, 1.0) * np.sin(3 * angles + rng.uniform(0, np.pi))
              + rng.uniform(0.3, 0.6) * np.sin(7 * angles + rng.uniform(0, np.pi)))
    wobble_scale = float(rng.uniform(0.8, 1.8))
    pts = []
    for i, a in enumerate(angles):
        s = 1.0 + wobble[i] * wobble_scale / max(rx, ry)
        x = int(np.clip(cx + rx * s * np.cos(a), 1, IMG_W - 2))
        y = int(np.clip(cy + ry * s * np.sin(a), 1, IMG_H - 2))
        pts.append((x, y))
    for i in range(len(pts)):
        cv2.line(out, pts[i], pts[(i + 1) % len(pts)],
                 color=(0, 200, 0), thickness=int(rng.integers(1, 3)), lineType=cv2.LINE_AA)
    return out, {"cx": cx, "cy": cy, "rx": float(rx), "ry": float(ry)}


def draw_destination_arrow(img, pick_px, place_px, seed):
    rng = np.random.default_rng(seed + 1)
    out = img.copy()
    x1 = pick_px[0] + int(rng.integers(-2, 3)); y1 = pick_px[1] + int(rng.integers(-2, 3))
    x2 = place_px[0] + int(rng.integers(-2, 3)); y2 = place_px[1] + int(rng.integers(-2, 3))
    mx = (x1 + x2) // 2 + int(rng.integers(-7, 8)); my = (y1 + y2) // 2 + int(rng.integers(-7, 8))
    thick = int(rng.integers(1, 3))
    cv2.line(out, (x1, y1), (mx, my), color=(200, 50, 50), thickness=thick, lineType=cv2.LINE_AA)
    cv2.arrowedLine(out, (mx, my), (x2, y2), color=(200, 50, 50), thickness=thick,
                    line_type=cv2.LINE_AA, tipLength=0.35)
    return out, {"x0": x1, "y0": y1, "x1": x2, "y1": y2}


# ───────── projection helpers ─────────
def project_points(world_positions, W2P):
    from robosuite.utils import camera_utils as CU
    if len(world_positions) == 0:
        return []
    pix = CU.project_points_from_world_to_camera(
        points=np.array(world_positions), world_to_camera_transform=W2P,
        camera_height=IMG_H, camera_width=IMG_W)
    return [(int(p[1]), (IMG_H - 1) - int(p[0])) for p in pix]


def visual_center(model, data, body_id):
    gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]
    ref = [g for g in gids if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
    return data.geom_xpos[ref[0]].copy() if ref else data.body_xpos[body_id].copy()


def frame_from_obs(obs):
    key = "agentview_image" if "agentview_image" in obs else \
          [k for k in obs if "agentview" in k and "image" in k][0]
    f = np.asarray(obs[key])
    if f.dtype != np.uint8:
        f = np.clip(f * 255.0 if f.max() <= 1.0 + 1e-6 else f, 0, 255).astype(np.uint8)
    return f.copy()


def check_success(env):
    """Try the various success accessors LIBERO/robosuite expose; report all."""
    results = {}
    targets = {"env": env, "env.env": getattr(env, "env", None),
               "env.env.env": getattr(getattr(env, "env", None), "env", None)}
    for label, obj in targets.items():
        if obj is None:
            continue
        for m in ("check_success", "_check_success"):
            fn = getattr(obj, m, None)
            if callable(fn):
                try:
                    results[f"{label}.{m}()"] = bool(fn())
                except Exception as e:
                    results[f"{label}.{m}()"] = f"ERR {e}"
    return results


def main():
    meta = json.load(open(META))
    target_bowl, target_plate = meta["target_bowl"], meta["target_plate"]
    print("target bowl :", target_bowl)
    print("target plate:", target_plate)

    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils import camera_utils as CU

    env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=IMG_H,
                             camera_widths=IMG_W, camera_names=[CAMERA])
    obs = env.reset()
    try:
        adim = env.action_dim
    except AttributeError:
        adim = 7
    for _ in range(20):
        obs, _, _, _ = env.step(np.zeros(adim))

    frame = frame_from_obs(obs)
    sim = env.sim; model, data = sim.model, sim.data
    W2P = CU.get_camera_transform_matrix(sim=sim, camera_name=CAMERA,
                                         camera_height=IMG_H, camera_width=IMG_W)

    def body_id(name):
        for c in (name, f"{name}_main"):
            if c in model.body_names:
                return model.body_name2id(c)
        raise KeyError(name)

    # circle center + radius from the target bowl's geoms
    tb = body_id(target_bowl)
    pick_px = project_points([visual_center(model, data, tb)], W2P)[0]
    gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == tb]
    gpx = project_points([data.geom_xpos[g].copy() for g in gids], W2P)
    if gpx:
        cols, rows = [p[0] for p in gpx], [p[1] for p in gpx]
        radius = max(5, min(int(np.hypot(max(cols) - min(cols), max(rows) - min(rows)) / 2 * 1.5), 40))
    else:
        radius = 10

    tp = body_id(target_plate)
    place_px = project_points([visual_center(model, data, tp)], W2P)[0]

    # ── draw sketch (green circle -> target bowl, red arrow -> target plate) ──
    seed = 12345
    img_c, tok_circle = draw_circle(frame, pick_px, radius, seed)
    img_ca, tok_arrow = draw_destination_arrow(img_c, pick_px, place_px, seed)
    cv2.imwrite(os.path.join(OUT_DIR, "scene_0001_sketch.png"),
                cv2.cvtColor(img_ca, cv2.COLOR_RGB2BGR))
    json.dump({"instruction": meta["language"], "target_bowl": target_bowl,
               "target_plate": target_plate,
               "symbolic_tokens": {"circle": tok_circle, "arrow": tok_arrow},
               "pick_px": list(pick_px), "place_px": list(place_px), "radius": radius},
              open(os.path.join(OUT_DIR, "scene_0001_tokens.json"), "w"), indent=2)
    print(f"sketch: circle@{pick_px} r={radius}  arrow->{place_px}")

    # ── teleport oracle: place GT bowl on GT plate, settle, check goal ──
    cv2.imwrite(os.path.join(OUT_DIR, "scene_0001_oracle_before.png"),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print("\nsuccess BEFORE teleport:", check_success(env))

    # find the target bowl's free joint
    jname = None
    for j in range(model.njnt):
        nm = model.joint_id2name(j)
        if nm and nm.startswith(target_bowl) and "joint" in nm:
            jname = nm; break
    if jname is None:
        print("[!] could not find free joint for", target_bowl); env.close(); return
    jid = model.joint_name2id(jname)
    qadr = model.jnt_qposadr[jid]; vadr = model.jnt_dofadr[jid]
    plate_pos = visual_center(model, data, tp)
    data.qpos[qadr:qadr + 3] = [plate_pos[0], plate_pos[1], plate_pos[2] + 0.04]
    data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
    data.qvel[vadr:vadr + 6] = 0
    sim.forward()
    for _ in range(40):
        obs, _, _, _ = env.step(np.zeros(adim))

    frame_after = frame_from_obs(obs)
    cv2.imwrite(os.path.join(OUT_DIR, "scene_0001_oracle_after.png"),
                cv2.cvtColor(frame_after, cv2.COLOR_RGB2BGR))
    print("success AFTER teleport :", check_success(env))

    # ambiguity guarantee
    nb = meta["counts"]["akita_black_bowl"]; npl = meta["counts"]["plate"]
    print(f"\nambiguity check: {nb} bowls, {npl} plates -> vague caption "
          f"'{meta['language']}' matches multiple; text alone insufficient: "
          f"{nb > 1 or npl > 1}")
    print("\nDONE. Send back scene_0001_sketch.png, oracle_before/after pngs, "
          "tokens.json, oracle_log.txt")
    env.close()


if __name__ == "__main__":
    main()
