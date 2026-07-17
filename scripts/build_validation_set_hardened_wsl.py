"""
DrawVLA validation set — HARDENED builder (run in WSL2, libero env).

Adds three inline checks to the base builder, run on the SAME deterministic env
state that produced each scene (placement is seeded => reproducible):
  - clearance      : target -> nearest-neighbour xy distance (recorded)
  - visibility     : RGB-difference occlusion fraction of the target (gate)
  - scripted grasp : top-down OSC grasp must lift the target > 3 cm (gate)
plus the existing gates (settled on table, target in frame, teleport oracle).

SMOKE=True -> 3 scenes (calibrate thresholds). Flip to False to regenerate the
full ~38-scene hardened set into outputs/validation_set_hardened/.

    conda activate <libero env>
    cd /mnt/c/Users/Admin/sketch_vla
    python scripts/build_validation_set_hardened_wsl.py 2>&1 | tee outputs/validation_set_hardened/build_log.txt
"""

import os, json, gc
import numpy as np
import cv2

SMOKE = False
OUT_ROOT = "/mnt/c/Users/Admin/sketch_vla/outputs/validation_set_hardened"
IMG_H = IMG_W = 128
CAMERA = "agentview"

RECT_X = (-0.22, 0.20); RECT_Y = (-0.06, 0.32)
MIN_DIST = 0.088; HALF_BOX = 0.012; TABLE_Z_MIN = 0.85
VIS_MIN = 0.35            # min target visibility to keep a scene
DIFF_TH = 30             # per-pixel RGB-sum diff threshold for silhouette counting
CAPTIONS = ["pick up the black bowl and place it on the plate",
            "pick up the bowl and put it on the plate",
            "grab the black bowl and place it on the plate"]


def tier_specs(smoke):
    if smoke:
        return [("control", [(1, 1, 1, 1)]),
                ("referential", [(3, 1, 1, 1)]),
                ("both", [(5, 4, 2, 2)])]
    specs = [("control", [(1, 1, 1, 1)] * 5)]
    ref = [];  [ref.extend([(n, 1, 1, 1)] * 3) for n in (2, 3, 4, 5)]
    specs.append(("referential", ref))
    dr = [];   [dr.extend([(1, m, 1, 1)] * 3) for m in (2, 3, 4)]
    specs.append(("directional", dr))
    bo = [];   [bo.extend([(n, m, 2, 2)] * 3) for (n, m) in ((3, 2), (4, 3), (5, 3), (4, 4))]
    specs.append(("both", bo))
    return specs


# ---- draw funcs (match training) ----
def draw_circle(img, c, radius, seed):
    rng = np.random.default_rng(seed); out = img.copy()
    cx = c[0] + int(rng.integers(-3, 4)); cy = c[1] + int(rng.integers(-3, 4))
    rx = radius + float(rng.uniform(-2, 3)); ry = radius + float(rng.uniform(-2, 3))
    ang = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    wob = (rng.uniform(0.5, 1.0) * np.sin(3 * ang + rng.uniform(0, np.pi))
           + rng.uniform(0.3, 0.6) * np.sin(7 * ang + rng.uniform(0, np.pi)))
    ws = float(rng.uniform(0.8, 1.8)); pts = []
    for i, a in enumerate(ang):
        s = 1 + wob[i] * ws / max(rx, ry)
        pts.append((int(np.clip(cx + rx * s * np.cos(a), 1, IMG_W - 2)),
                    int(np.clip(cy + ry * s * np.sin(a), 1, IMG_H - 2))))
    for i in range(len(pts)):
        cv2.line(out, pts[i], pts[(i + 1) % len(pts)], (0, 200, 0), int(rng.integers(1, 3)), cv2.LINE_AA)
    return out, {"cx": cx, "cy": cy, "rx": float(rx), "ry": float(ry)}


def draw_arrow(img, p, q, seed):
    rng = np.random.default_rng(seed + 1); out = img.copy()
    x1 = p[0] + int(rng.integers(-2, 3)); y1 = p[1] + int(rng.integers(-2, 3))
    x2 = q[0] + int(rng.integers(-2, 3)); y2 = q[1] + int(rng.integers(-2, 3))
    mx = (x1 + x2) // 2 + int(rng.integers(-7, 8)); my = (y1 + y2) // 2 + int(rng.integers(-7, 8))
    th = int(rng.integers(1, 3))
    cv2.line(out, (x1, y1), (mx, my), (200, 50, 50), th, cv2.LINE_AA)
    cv2.arrowedLine(out, (mx, my), (x2, y2), (200, 50, 50), th, cv2.LINE_AA, tipLength=0.35)
    return out, {"x0": x1, "y0": y1, "x1": x2, "y1": y2}


# ---- helpers ----
def project(world, W2P):
    from robosuite.utils import camera_utils as CU
    if len(world) == 0:
        return []
    pix = CU.project_points_from_world_to_camera(points=np.array(world),
          world_to_camera_transform=W2P, camera_height=IMG_H, camera_width=IMG_W)
    return [(int(p[1]), (IMG_H - 1) - int(p[0])) for p in pix]


def vcenter(model, data, bid):
    gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
    ref = [g for g in gids if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
    return data.geom_xpos[ref[0]].copy() if ref else data.body_xpos[bid].copy()


def bid_of(model, name):
    for c in (name, f"{name}_main"):
        if c in model.body_names:
            return model.body_name2id(c)
    raise KeyError(name)


def jnt_of(model, prefix):
    for j in range(model.njnt):
        nm = model.joint_id2name(j)
        if nm and nm.startswith(prefix) and "joint" in nm:
            return j
    return None


def frame_obs(obs):
    k = "agentview_image" if "agentview_image" in obs else \
        [x for x in obs if "agentview" in x and "image" in x][0]
    f = np.asarray(obs[k])
    if f.dtype != np.uint8:
        f = np.clip(f * 255.0 if f.max() <= 1 + 1e-6 else f, 0, 255).astype(np.uint8)
    return f.copy()


def success(env):
    for o in (env, getattr(env, "env", None)):
        if o is None:
            continue
        for m in ("check_success", "_check_success"):
            fn = getattr(o, m, None)
            if callable(fn):
                try:
                    return bool(fn())
                except Exception:
                    pass
    return None


def sample_positions(n, rng, tries=3000):
    pts = []
    for _ in range(tries):
        if len(pts) == n:
            break
        x = rng.uniform(*RECT_X); y = rng.uniform(*RECT_Y)
        if all(np.hypot(x - a, y - b) >= MIN_DIST for a, b in pts):
            pts.append((float(x), float(y)))
    if len(pts) < n:
        raise RuntimeError("placement failed")
    return pts


def build_bddl(counts, language, seed):
    rng = np.random.default_rng(seed)
    inst = []
    for c, n in counts.items():
        inst += [(c, f"{c}_{i}") for i in range(1, n + 1)]
    pos = sample_positions(len(inst), rng)
    place = {i: pos[k] for k, (_, i) in enumerate(inst)}
    t = int(rng.integers(1, counts["akita_black_bowl"] + 1))
    d = int(rng.integers(1, counts["plate"] + 1))
    tb, tp = f"akita_black_bowl_{t}", f"plate_{d}"

    def reg(i):
        cx, cy = place[i]
        return (f"      ({i}_region\n          (:target main_table)\n          (:ranges (\n"
                f"              ({cx-HALF_BOX:.4f} {cy-HALF_BOX:.4f} {cx+HALF_BOX:.4f} {cy+HALF_BOX:.4f})\n"
                f"            )\n          )\n      )\n")
    regions = "".join(reg(i) for _, i in inst)
    objs = "\n".join("    " + " ".join(f"{c}_{k}" for k in range(1, n + 1)) + f" - {c}"
                     for c, n in counts.items())
    inits = "".join(f"    (On {i} main_table_{i}_region)\n" for _, i in inst)
    bddl = f"""(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language {language})
    (:regions
{regions}    )

  (:fixtures
    main_table - table
  )

  (:objects
{objs}
  )

  (:obj_of_interest
    {tb}
    {tp}
  )

  (:init
{inits}  )

  (:goal
    (And (On {tb} {tp}))
  )

)
"""
    meta = {"counts": counts, "language": language, "seed": seed, "target_bowl": tb,
            "target_plate": tp, "placements": place, "instances": [i for _, i in inst]}
    return bddl, meta


# ---- RGB-diff visibility ----
def rgb(env):
    return env.sim.render(width=IMG_W, height=IMG_H, camera_name=CAMERA).astype(np.int32)


def visibility(env, model, data, instances, target):
    saved = {}
    for i in instances:
        j = jnt_of(model, i); qa = model.jnt_qposadr[j]
        saved[i] = (qa, data.qpos[qa:qa + 7].copy())

    def hide(names):
        for n in names:
            qa = saved[n][0]; data.qpos[qa + 2] = -1.0

    def restore(names):
        for n in names:
            qa, v = saved[n]; data.qpos[qa:qa + 7] = v
    others = [i for i in instances if i != target]

    hide(others); env.sim.forward(); R_t = rgb(env)
    hide([target]); env.sim.forward(); R_e = rgb(env)
    A = int((np.abs(R_t - R_e).sum(-1) > DIFF_TH).sum())
    restore(instances); env.sim.forward(); R_f = rgb(env)
    hide([target]); env.sim.forward(); R_nt = rgb(env)
    Vocc = int((np.abs(R_f - R_nt).sum(-1) > DIFF_TH).sum())
    restore(instances); env.sim.forward()
    vis_mask = ((np.abs(R_f - R_nt).sum(-1) > DIFF_TH) * 255).astype(np.uint8)
    return dict(v_visible=Vocc, v_full=A, visibility=round(Vocc / max(A, 1), 3)), vis_mask


# ---- scripted grasp ----
def eef(obs):
    for k in ("robot0_eef_pos", "eef_pos"):
        if k in obs:
            return np.asarray(obs[k])
    return None


def scripted_grasp(env, adim, model, data, tb, obs):
    j = jnt_of(model, tb); qa = model.jnt_qposadr[j]
    z0 = float(data.qpos[qa + 2]); tc = vcenter(model, data, bid_of(model, tb))

    def servo(obs, goal, grip, steps, gain=8.0):
        for _ in range(steps):
            e = eef(obs); a = np.zeros(adim)
            if e is not None:
                a[:3] = np.clip((goal - e) * gain, -1, 1)
            a[-1] = grip
            obs, _, _, _ = env.step(a)
        return obs
    for cs in (-1.0, 1.0):
        obs = servo(obs, tc + [0, 0, 0.12], -cs, 30)
        obs = servo(obs, tc + [0, 0, 0.005], -cs, 25)
        obs = servo(obs, tc + [0, 0, 0.005], cs, 12)
        obs = servo(obs, tc + [0, 0, 0.18], cs, 30)
        if float(data.qpos[qa + 2]) - z0 > 0.03:
            return dict(grasp_success=True, lift=round(float(data.qpos[qa + 2]) - z0, 3), close_sign=cs)
    return dict(grasp_success=False, lift=round(float(data.qpos[qa + 2]) - z0, 3))


def build_scene(combo, tier, seed, scene_dir):
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils import camera_utils as CU
    rng = np.random.default_rng(seed)
    counts = {"akita_black_bowl": combo[0], "plate": combo[1],
              "glazed_rim_porcelain_ramekin": combo[2], "cookies": combo[3]}
    lang = CAPTIONS[int(rng.integers(0, len(CAPTIONS)))]
    bddl, meta = build_bddl(counts, lang, seed); meta["tier"] = tier
    os.makedirs(scene_dir, exist_ok=True)
    bpath = os.path.join(scene_dir, "scene.bddl"); open(bpath, "w").write(bddl)

    np.random.seed(seed)                        # deterministic placement
    env = OffScreenRenderEnv(bddl_file_name=bpath, camera_heights=IMG_H,
                             camera_widths=IMG_W, camera_names=[CAMERA])
    try:
        obs = env.reset(); adim = getattr(env, "action_dim", 7)
        for _ in range(20):
            obs, _, _, _ = env.step(np.zeros(adim))
        frame = frame_obs(obs); model, data = env.sim.model, env.sim.data
        W2P = CU.get_camera_transform_matrix(sim=env.sim, camera_name=CAMERA,
                                             camera_height=IMG_H, camera_width=IMG_W)
        tb, tp = meta["target_bowl"], meta["target_plate"]
        for i in meta["instances"]:
            if vcenter(model, data, bid_of(model, i))[2] < TABLE_Z_MIN:
                return None, "fell_off"
        pick = project([vcenter(model, data, bid_of(model, tb))], W2P)[0]
        place = project([vcenter(model, data, bid_of(model, tp))], W2P)[0]
        if not (3 <= pick[0] <= 124 and 3 <= pick[1] <= 124):
            return None, "off_frame"
        gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid_of(model, tb)]
        gpx = project([data.geom_xpos[g].copy() for g in gids], W2P)
        cs_ = [p[0] for p in gpx]; rs_ = [p[1] for p in gpx]
        radius = max(5, min(int(np.hypot(max(cs_) - min(cs_), max(rs_) - min(rs_)) / 2 * 1.5), 40))

        img_c, tok_c = draw_circle(frame, pick, radius, seed)
        img_ca, tok_a = draw_arrow(img_c, pick, place, seed)

        # clearance
        tpos = vcenter(model, data, bid_of(model, tb))
        others = [i for i in meta["instances"] if i != tb]
        clr = min(float(np.linalg.norm(vcenter(model, data, bid_of(model, o))[:2] - tpos[:2]))
                  for o in others) if others else None

        # visibility (perturb+restore) BEFORE oracle/grasp
        vis, vismask = visibility(env, model, data, meta["instances"], tb)
        if vis["visibility"] < VIS_MIN:
            return None, f"low_vis_{vis['visibility']}"

        # teleport oracle
        j = jnt_of(model, tb); qa = model.jnt_qposadr[j]; va = model.jnt_dofadr[j]
        pp = vcenter(model, data, bid_of(model, tp))
        data.qpos[qa:qa + 3] = [pp[0], pp[1], pp[2] + 0.04]; data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
        data.qvel[va:va + 6] = 0; env.sim.forward()
        for _ in range(40):
            obs, _, _, _ = env.step(np.zeros(adim))
        if success(env) is not True:
            return None, "oracle_false"

        # grasp gate (fresh deterministic reset)
        np.random.seed(seed); obs = env.reset()
        for _ in range(20):
            obs, _, _, _ = env.step(np.zeros(adim))
        model, data = env.sim.model, env.sim.data
        g = scripted_grasp(env, adim, model, data, tb, obs)
        if not g["grasp_success"]:
            return None, f"ungraspable_lift{g['lift']}"

        # passed all gates -> save
        cv2.imwrite(os.path.join(scene_dir, "frame0.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(scene_dir, "sketch.png"), cv2.cvtColor(img_ca, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(scene_dir, "target_vismask.png"), vismask)
        meta.update(dict(pick_px=list(pick), place_px=list(place), radius=radius,
                         symbolic_tokens={"circle": tok_c, "arrow": tok_a},
                         camera_matrix=W2P.tolist(), oracle_success=True,
                         clearance_xy=round(clr, 3) if clr else None,
                         visibility=vis, grasp=g))
        json.dump(meta, open(os.path.join(scene_dir, "meta.json"), "w"), indent=2)
        json.dump({"instruction": lang, "target_bowl": tb, "target_plate": tp,
                   "symbolic_tokens": meta["symbolic_tokens"]},
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
    manifest = []; idx = 0
    for tier, combos in tier_specs(SMOKE):
        for combo in combos:
            made = None
            for attempt in range(10):
                seed = 2000 + idx * 100 + attempt
                sd = os.path.join(OUT_ROOT, f"scene_{idx:04d}")
                try:
                    made, why = build_scene(combo, tier, seed, sd)
                except Exception as e:
                    made, why = None, f"error:{e}"
                print(f"  scene_{idx:04d} {tier} {combo} seed={seed} -> {why}")
                if made:
                    break
            if made:
                made["dir"] = f"scene_{idx:04d}"
                manifest.append({k: made[k] for k in
                                 ("dir", "tier", "counts", "seed", "target_bowl",
                                  "target_plate", "language", "clearance_xy",
                                  "visibility", "grasp")})
                print(f"[ok] scene_{idx:04d} {tier} vis={made['visibility']['visibility']} "
                      f"lift={made['grasp']['lift']} clr={made['clearance_xy']}")
            else:
                print(f"[FAIL] scene_{idx:04d} {tier} {combo}")
            idx += 1
    json.dump(manifest, open(os.path.join(OUT_ROOT, "manifest.json"), "w"), indent=2)

    # contact sheet of sketches
    imgs = []
    for e in manifest:
        im = cv2.imread(os.path.join(OUT_ROOT, e["dir"], "sketch.png"))
        if im is None:
            continue
        im = cv2.resize(im, (128, 128), interpolation=cv2.INTER_NEAREST)
        cv2.putText(im, f"{e['tier'][:4]} {e['counts']['akita_black_bowl']}b{e['counts']['plate']}p"
                    f" v{e['visibility']['visibility']:.2f}", (2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
        imgs.append(im)
    if imgs:
        cols = min(8, len(imgs)); rows = (len(imgs) + cols - 1) // cols
        sheet = np.full((rows * 130, cols * 130, 3), 40, np.uint8)
        for k, im in enumerate(imgs):
            r, c = divmod(k, cols)
            sheet[r * 130 + 1:r * 130 + 129, c * 130 + 1:c * 130 + 129] = im
        cv2.imwrite(os.path.join(OUT_ROOT, "contact_sheet.png"), sheet)

    print(f"\nSMOKE={SMOKE}  {len(manifest)} scenes. Metrics:")
    for e in manifest:
        print(f"  {e['dir']} {e['tier']:11s} vis={e['visibility']['visibility']:.2f} "
              f"lift={e['grasp']['lift']:.3f} clr={e['clearance_xy']}")


if __name__ == "__main__":
    main()
