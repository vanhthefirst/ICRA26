"""
DrawVLA validation set — LIBERO-SPATIAL builder, HARDENED v2 (run in WSL2, libero).

Backports the Object suite's stronger gate stack onto the Spatial (table / `On`)
task, so all suites share one standard. Structurally parallel to
build_validation_set_object_wsl.py on purpose -- this file, that file, and the
future Goal builder are the same template with a different predicate.

What v2 adds over the original hardened Spatial set:
  * NEGATIVE ORACLES THAT GATE. `On(bowl, plate)` is True when the bowl is within
    3 cm (xy) of the plate AND in contact. So two plates closer than ~6 cm make a
    bowl on the WRONG plate also satisfy `On bowl target_plate` -> a silently
    broken directional scene. v2 requires:
        target_bowl  -> every OTHER plate  => False   (directional axis)
        every SIBLING bowl -> target_plate => False   (referential axis)
    These certify the scene is genuinely unsolvable without the sketch, and the
    directional one catches plates placed too close.
  * PIXEL-SEPARATION resolvability gates from projected extents (not constants):
        circle: dist(target, sibling) >= drawn_radius + 0.5*ext(sibling) + 3 px
        arrow : dist(target_plate, rival plate) >= ext(target_plate) + 4 px
  * RESTART-based rejection sampling (the original had the doomed-prefix bug).
  * Emits the canonical schema (SCHEMA.md v1.0) directly: suite/target/
    destination/destination_region/goal_predicate.

Predicate note: Spatial goal is `(On bowl_t plate_d)`, so the destination IS an
object instance -> destination_region == destination == plate_d (unlike Object's
`_contain_region`). Oracle uses `On` semantics (rest the bowl on the plate).

SMOKE=True -> 4 scenes at the extremes + prints measured px_extent per category
(use it to set spacing / per-category N caps, as we did for Object). Flip to
False for the full 38-scene run.

    conda activate libero
    cd /mnt/c/Users/Admin/sketch_vla
    mkdir -p outputs/validation_set_hardened_v2
    python scripts/build_validation_set_spatial_v2_wsl.py 2>&1 | tee outputs/validation_set_hardened_v2/build_log.txt

Writes to validation_set_hardened_v2 (NOT the live set). After it verifies as
good, replace: mv validation_set_hardened validation_set_hardened_v1_backup &&
mv validation_set_hardened_v2 validation_set_hardened, then re-run
normalize_validation_schema.py.
"""

import os, json, gc
import numpy as np
import cv2

SMOKE = False
OUT_ROOT = "/mnt/c/Users/Admin/sketch_vla/outputs/validation_set_hardened_v2"
IMG_H = IMG_W = 128
CAMERA = "agentview"
ADIM = 7

# table workspace (from the original hardened builder; single rectangle, not banded)
RECT_X = (-0.22, 0.20); RECT_Y = (-0.06, 0.32)
TABLE_Z_MIN = 0.85                 # below this a body has fallen off the table
HALF_BOX = 0.012
# same-category / cross spacing. Bowls carry a big drawn circle (~r15-20), so the
# circle gate needs generous sibling spacing; plates must clear On's 3 cm xy rule
# AND be arrow-resolvable. Tune from the SMOKE px_extent dump if rejections spike.
D_BOWL = 0.165      # bowl  <-> bowl   (siblings, referential axis; smoke: bowl
                    # ext ~14 px, gate needs ~26 px, 0.150 was borderline -> churn.
                    # 0.165 gives ~2 px margin; dense both still fits 97.7%/restart)
D_PLATE = 0.150     # plate <-> plate  (destinations, directional axis)
D_CROSS = 0.095     # any other pair

VIS_MIN = 0.35
DIFF_TH = 30
PX_MARGIN = 3
PX_MARGIN_PLATE = 4

TARGET_CAT = "akita_black_bowl"
DEST_CAT = "plate"
DISTRACTOR_CATS = ["glazed_rim_porcelain_ramekin", "cookies"]
CAPTIONS = ["pick up the black bowl and place it on the plate",
            "pick up the bowl and put it on the plate",
            "grab the black bowl and place it on the plate"]


def tier_specs(smoke):
    """(n_bowl, n_plate, n_distractor_each)"""
    if smoke:
        return [("control", [(1, 1, 1)]),
                ("referential", [(5, 1, 1)]),
                ("directional", [(1, 3, 1)]),
                ("both", [(4, 3, 2)])]
    specs = [("control", [(1, 1, 1)] * 5)]
    ref = []
    for n in (2, 3, 4, 5):
        ref += [(n, 1, 1)] * 3                         # 12
    specs.append(("referential", ref))
    dr = [(1, 2, 1)] * 3 + [(1, 3, 1)] * 3 + [(1, 4, 1)] * 3   # 9
    specs.append(("directional", dr))
    bo = []
    for (n, m) in ((3, 2), (4, 2), (3, 3), (5, 3)):
        bo += [(n, m, 2)] * 3                          # 12
    specs.append(("both", bo))
    return specs


# ----------------------------------------------------------------- io / util --
def safe_imwrite(path, img):
    ok = cv2.imwrite(path, img)
    n = os.path.getsize(path) if os.path.exists(path) else 0
    tail = False
    if n:
        with open(path, "rb") as f:
            f.seek(-8, 2); tail = b"IEND" in f.read()
    back = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    good = bool(ok and n > 0 and tail and back is not None and back.shape[:2] == img.shape[:2])
    if not good:
        raise IOError(f"corrupt write {path}: ok={ok} bytes={n} iend={tail}")
    return True


def project(world, W2P):
    from robosuite.utils import camera_utils as CU
    if len(world) == 0:
        return []
    pix = CU.project_points_from_world_to_camera(
        points=np.array(world), world_to_camera_transform=W2P,
        camera_height=IMG_H, camera_width=IMG_W)
    return [(int(p[1]), (IMG_H - 1) - int(p[0])) for p in pix]


def bid_of(model, name):
    for c in (name, f"{name}_main"):
        if c in model.body_names:
            return model.body_name2id(c)
    raise KeyError(name)


def vcenter(model, data, bid):
    gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
    ref = [g for g in gids if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
    return data.geom_xpos[ref[0]].copy() if ref else data.body_xpos[bid].copy()


def jnt_of(model, prefix):
    for j in range(model.njnt):
        if model.joint_id2name(j) == f"{prefix}_joint0":
            return j
    for j in range(model.njnt):
        nm = model.joint_id2name(j)
        if nm and nm.startswith(prefix) and "joint" in nm:
            return j
    return None


def px_extent(model, data, W2P, name):
    """Half-size in pixels of the object's projected bounding box (8 geom-box
    corners rotated by geom_xmat). Gating only; drawn radius keeps its own formula."""
    b = bid_of(model, name)
    pts = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != b:
            continue
        hs = np.asarray(model.geom_size[g], float)
        R = np.asarray(data.geom_xmat[g], float).reshape(3, 3)
        c = np.asarray(data.geom_xpos[g], float)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    pts.append(c + R.dot(np.array([sx, sy, sz]) * hs))
    if not pts:
        return 5.0
    p = project(pts, W2P)
    xs = [q[0] for q in p]; ys = [q[1] for q in p]
    return float(max(max(xs) - min(xs), max(ys) - min(ys)) / 2.0)


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


def settle(env, n=20):
    obs = env.reset()
    for _ in range(n):
        obs, _, _, _ = env.step(np.zeros(ADIM))
    return obs


# ------------------------------------------------------------- sketch drawing --
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
        cv2.line(out, pts[i], pts[(i + 1) % len(pts)], (0, 200, 0),
                 int(rng.integers(1, 3)), cv2.LINE_AA)
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


# ------------------------------------------------------ placement (restarts!) --
def sample_positions(instances, rng, restarts=400, tries_per=600):
    """instances: list of (name, category). Per-category self-spacing (D_BOWL for
    bowls, D_PLATE for plates), D_CROSS otherwise. Rejection sampling WITH RESTARTS."""
    def dmin(ca, cb):
        if ca == cb == TARGET_CAT:
            return D_BOWL
        if ca == cb == DEST_CAT:
            return D_PLATE
        return D_CROSS
    for _ in range(restarts):
        placed = []          # (x, y, cat)
        ok = True
        for name, cat in instances:
            got = False
            for _ in range(tries_per):
                x = rng.uniform(*RECT_X); y = rng.uniform(*RECT_Y)
                if all(np.hypot(x - a, y - b) >= dmin(cat, c) for a, b, c in placed):
                    placed.append((float(x), float(y), cat)); got = True; break
            if not got:
                ok = False; break
        if ok:
            return {instances[k][0]: (placed[k][0], placed[k][1])
                    for k in range(len(instances))}
    raise RuntimeError(f"placement failed after {restarts} restarts "
                       f"({len(instances)} objects)")


def build_bddl(counts, tgt_idx, dest_idx, language, seed):
    rng = np.random.default_rng(seed)
    instances = []
    for c, n in counts.items():
        instances += [(f"{c}_{i}", c) for i in range(1, n + 1)]
    place = sample_positions(instances, rng)
    target = f"{TARGET_CAT}_{tgt_idx}"; dest = f"{DEST_CAT}_{dest_idx}"

    def reg(name):
        cx, cy = place[name]
        return (f"      ({name}_region\n          (:target main_table)\n          (:ranges (\n"
                f"              ({cx-HALF_BOX:.4f} {cy-HALF_BOX:.4f} "
                f"{cx+HALF_BOX:.4f} {cy+HALF_BOX:.4f})\n"
                f"            )\n          )\n      )\n")
    names = [n for n, _ in instances]
    regions = "".join(reg(n) for n in names)
    objs = "\n".join("    " + " ".join(f"{c}_{k}" for k in range(1, n + 1)) + f" - {c}"
                     for c, n in counts.items())
    inits = "".join(f"    (On {n} main_table_{n}_region)\n" for n in names)
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
    {target}
    {dest}
  )

  (:init
{inits}  )

  (:goal
    (And (On {target} {dest}))
  )

)
"""
    meta = dict(counts=counts, target=target, dest=dest, language=language,
                seed=seed, instances=names, placements=place,
                siblings=[f"{TARGET_CAT}_{i}" for i in range(1, counts[TARGET_CAT] + 1)
                          if f"{TARGET_CAT}_{i}" != target],
                other_plates=[f"{DEST_CAT}_{i}" for i in range(1, counts[DEST_CAT] + 1)
                              if f"{DEST_CAT}_{i}" != dest])
    return bddl, meta


# ------------------------------------------------------------------- metrics --
def rgb(env):
    return env.sim.render(width=IMG_W, height=IMG_H, camera_name=CAMERA).astype(np.int32)


def visibility(env, model, data, instances, target):
    saved = {}
    for i in instances:
        j = jnt_of(model, i)
        if j is None:
            continue
        qa = model.jnt_qposadr[j]
        saved[i] = (qa, data.qpos[qa:qa + 7].copy())
    inst = list(saved.keys())

    def hide(names):
        for n in names:
            data.qpos[saved[n][0] + 2] = -1.0

    def restore(names):
        for n in names:
            qa, v = saved[n]; data.qpos[qa:qa + 7] = v
    others = [i for i in inst if i != target]
    hide(others); env.sim.forward(); R_t = rgb(env)
    hide([target]); env.sim.forward(); R_e = rgb(env)
    A = int((np.abs(R_t - R_e).sum(-1) > DIFF_TH).sum())
    restore(inst); env.sim.forward(); R_f = rgb(env)
    hide([target]); env.sim.forward(); R_nt = rgb(env)
    V = int((np.abs(R_f - R_nt).sum(-1) > DIFF_TH).sum())
    restore(inst); env.sim.forward()
    mask = ((np.abs(R_f - R_nt).sum(-1) > DIFF_TH) * 255).astype(np.uint8)
    return dict(v_visible=V, v_full=A, visibility=round(V / max(A, 1), 3)), mask


def teleport_on(env, model, data, bowl, plate, steps=40):
    """Rest `bowl` on `plate` and return check_success(). `On` compares body_xpos
    (bowl above plate, contact, xy<3cm); we drop the bowl at the plate's xy just
    above its top so it settles into contact."""
    pc = vcenter(model, data, bid_of(model, plate))
    j = jnt_of(model, bowl); qa = model.jnt_qposadr[j]; va = model.jnt_dofadr[j]
    data.qpos[qa:qa + 3] = [pc[0], pc[1], pc[2] + 0.04]
    data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
    data.qvel[va:va + 6] = 0
    env.sim.forward()
    for _ in range(steps):
        env.step(np.zeros(ADIM))
    return success(env)


def eef(obs):
    for k in ("robot0_eef_pos", "eef_pos"):
        if k in obs:
            return np.asarray(obs[k])
    return None


def scripted_grasp(env, model, data, tgt, obs):
    j = jnt_of(model, tgt); qa = model.jnt_qposadr[j]
    z0 = float(data.qpos[qa + 2]); tc = vcenter(model, data, bid_of(model, tgt))

    def servo(obs, goal, grip, steps, gain=8.0):
        for _ in range(steps):
            e = eef(obs); a = np.zeros(ADIM)
            if e is not None:
                a[:3] = np.clip((goal - e) * gain, -1, 1)
            a[-1] = grip
            obs, _, _, _ = env.step(a)
        return obs
    best = -9.0
    for dz in (0.005, 0.03):
        for cs in (-1.0, 1.0):
            obs = servo(obs, tc + [0, 0, 0.12], -cs, 30)
            obs = servo(obs, tc + [0, 0, dz], -cs, 25)
            obs = servo(obs, tc + [0, 0, dz], cs, 12)
            obs = servo(obs, tc + [0, 0, 0.18], cs, 30)
            lift = float(data.qpos[qa + 2]) - z0
            best = max(best, lift)
            if lift > 0.03:
                return dict(grasp_success=True, lift=round(lift, 3),
                            close_sign=cs, approach_dz=dz)
    return dict(grasp_success=False, lift=round(best, 3))


# ------------------------------------------------------------------ one scene --
def build_scene(combo, tier, seed, scene_dir, dump_ext=False):
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils import camera_utils as CU
    n_bowl, n_plate, n_dis = combo
    rng = np.random.default_rng(seed)
    counts = {TARGET_CAT: n_bowl, DEST_CAT: n_plate}
    for d in DISTRACTOR_CATS:
        counts[d] = n_dis
    tgt_idx = int(rng.integers(1, n_bowl + 1))
    dest_idx = int(rng.integers(1, n_plate + 1))
    lang = CAPTIONS[int(rng.integers(0, len(CAPTIONS)))]

    bddl, meta = build_bddl(counts, tgt_idx, dest_idx, lang, seed)
    meta["tier"] = tier
    os.makedirs(scene_dir, exist_ok=True)
    bpath = os.path.join(scene_dir, "scene.bddl")
    open(bpath, "w").write(bddl)

    np.random.seed(seed)
    env = OffScreenRenderEnv(bddl_file_name=bpath, camera_heights=IMG_H,
                             camera_widths=IMG_W, camera_names=[CAMERA])
    try:
        obs = settle(env)
        model, data = env.sim.model, env.sim.data
        frame = frame_obs(obs)
        W2P = CU.get_camera_transform_matrix(sim=env.sim, camera_name=CAMERA,
                                             camera_height=IMG_H, camera_width=IMG_W)
        tgt, dest = meta["target"], meta["dest"]

        for i in meta["instances"]:
            if float(vcenter(model, data, bid_of(model, i))[2]) < TABLE_Z_MIN:
                return None, f"fell_{i}"

        pix = {i: project([vcenter(model, data, bid_of(model, i))], W2P)[0]
               for i in meta["instances"]}
        ext = {i: px_extent(model, data, W2P, i) for i in meta["instances"]}
        if dump_ext:
            byc = {}
            for i in meta["instances"]:
                byc.setdefault(i.rsplit("_", 1)[0], []).append(ext[i])
            print("   [ext px] " + "  ".join(f"{c}:{np.mean(v):.1f}" for c, v in byc.items()))

        def inframe(p):
            return 3 <= p[0] <= IMG_W - 4 and 3 <= p[1] <= IMG_H - 4
        if not inframe(pix[tgt]):
            return None, "target_off_frame"
        for p in meta["other_plates"] + [dest]:
            if not inframe(pix[p]):
                return None, f"{p}_off_frame"

        gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid_of(model, tgt)]
        gpx = project([data.geom_xpos[g].copy() for g in gids], W2P)
        cs_ = [p[0] for p in gpx]; rs_ = [p[1] for p in gpx]
        radius = max(5, min(int(np.hypot(max(cs_) - min(cs_),
                                         max(rs_) - min(rs_)) / 2 * 1.5), 40))

        # pixel resolvability
        px_plate = px_plate_req = None
        if meta["other_plates"]:
            px_plate_req = ext[dest] + PX_MARGIN_PLATE
            px_plate = min(float(np.hypot(pix[dest][0] - pix[p][0], pix[dest][1] - pix[p][1]))
                           for p in meta["other_plates"])
            if px_plate < px_plate_req:
                return None, f"plates_overlap_{px_plate:.0f}px_need{px_plate_req:.0f}"
        px_sib = px_sib_req = None
        if meta["siblings"]:
            worst = min(((float(np.hypot(pix[tgt][0] - pix[s][0],
                                         pix[tgt][1] - pix[s][1])) - ext[s]), s)
                        for s in meta["siblings"])
            s_worst = worst[1]
            px_sib = float(np.hypot(pix[tgt][0] - pix[s_worst][0],
                                    pix[tgt][1] - pix[s_worst][1]))
            px_sib_req = radius + 0.5 * ext[s_worst] + PX_MARGIN
            if px_sib < px_sib_req:
                return None, f"siblings_overlap_{px_sib:.0f}px_need{px_sib_req:.0f}_r{radius}"

        vis, vismask = visibility(env, model, data, meta["instances"], tgt)
        if vis["visibility"] < VIS_MIN:
            return None, f"low_vis_{vis['visibility']}"

        if success(env) is not False:
            return None, "pre_solved"

        if teleport_on(env, model, data, tgt, dest) is not True:
            return None, "oracle_false"

        neg = {}
        for p in meta["other_plates"]:                    # directional axis
            settle(env); model, data = env.sim.model, env.sim.data
            r = teleport_on(env, model, data, tgt, p)
            neg[f"{tgt}->{p}"] = r
            if r is not False:
                return None, f"neg_dest_true_{p}"
        for s in meta["siblings"]:                         # referential axis
            settle(env); model, data = env.sim.model, env.sim.data
            r = teleport_on(env, model, data, s, dest)
            neg[f"{s}->{dest}"] = r
            if r is not False:
                return None, f"neg_obj_true_{s}"

        np.random.seed(seed); obs = settle(env)
        model, data = env.sim.model, env.sim.data
        g = scripted_grasp(env, model, data, tgt, obs)
        if not g["grasp_success"]:
            return None, f"ungraspable_lift{g['lift']}"

        others = [i for i in meta["instances"] if i != tgt]
        tp = vcenter(model, data, bid_of(model, tgt))[:2]
        clr = min(float(np.linalg.norm(vcenter(model, data, bid_of(model, o))[:2] - tp))
                  for o in others) if others else None

        img_c, tok_c = draw_circle(frame, pix[tgt], radius, seed)
        img_ca, tok_a = draw_arrow(img_c, pix[tgt], pix[dest], seed)
        safe_imwrite(os.path.join(scene_dir, "frame0.png"),
                     cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        safe_imwrite(os.path.join(scene_dir, "sketch.png"),
                     cv2.cvtColor(img_ca, cv2.COLOR_RGB2BGR))
        safe_imwrite(os.path.join(scene_dir, "target_vismask.png"), vismask)

        # canonical schema (SCHEMA.md v1.0)
        meta.update(dict(
            schema_version="1.0", suite="spatial", goal_predicate="On",
            target=tgt, destination=dest, destination_region=dest,
            instruction=lang,
            pick_px=list(pix[tgt]), place_px=list(pix[dest]), radius=radius,
            all_pixels={k: list(v) for k, v in pix.items()},
            px_extent={k: round(v, 2) for k, v in ext.items()},
            symbolic_tokens={"circle": tok_c, "arrow": tok_a},
            camera_matrix=W2P.tolist(), oracle_success=True, oracle_negatives=neg,
            px_sep_plates=px_plate, px_req_plates=px_plate_req,
            px_sep_siblings=px_sib, px_req_siblings=px_sib_req,
            clearance_xy=round(clr, 3) if clr else None, visibility=vis, grasp=g,
            # legacy aliases kept for the original loader
            target_bowl=tgt, target_plate=dest))
        json.dump(meta, open(os.path.join(scene_dir, "meta.json"), "w"), indent=2)
        json.dump({"instruction": lang, "suite": "spatial", "tier": tier,
                   "target": tgt, "destination": dest, "destination_region": dest,
                   "goal_predicate": "On", "symbolic_tokens": meta["symbolic_tokens"],
                   "target_bowl": tgt, "target_plate": dest},
                  open(os.path.join(scene_dir, "tokens.json"), "w"), indent=2)
        return meta, "ok"
    finally:
        try:
            env.close()
        except Exception:
            pass
        gc.collect()


# ----------------------------------------------------------------------- main --
def write_datasheet(manifest, fails):
    from collections import Counter
    n = len(manifest)
    tiers = Counter(e["tier"] for e in manifest)
    vis = [e["visibility"]["visibility"] for e in manifest]
    lift = [e["grasp"]["lift"] for e in manifest]

    def stat(v, f="{:.3f}"):
        return "n/a" if not v else (f + " / " + f + " / " + f).format(min(v), float(np.mean(v)), max(v))
    md = f"""# DrawVLA validation set — LIBERO-**Spatial** suite (hardened v2)

{n} scenes. `scripts/build_validation_set_spatial_v2_wsl.py` (SMOKE={SMOKE}).
Canonical schema v1.0 (see SCHEMA.md). Backports the Object suite's negative-
oracle, pixel-separation, and restart-sampling gates onto the `On` task.

## Composition

| tier | scenes | bowls N | plates M | meaning |
|---|---|---|---|---|
| control | {tiers.get('control',0)} | 1 | 1 | unambiguous |
| referential | {tiers.get('referential',0)} | 2-5 | 1 | which bowl |
| directional | {tiers.get('directional',0)} | 1 | 2-4 | which plate |
| both | {tiers.get('both',0)} | 3-5 | 2-3 | both |

Goal `(On akita_black_bowl_t plate_d)`; destination is an object instance, so
`destination_region == destination == plate_d`.

## Gates (all must pass)

settled (on table) · target+plates in frame · pixel separation (circle vs
sibling bowls, arrow vs rival plates, from projected extents) · visibility >=
{VIS_MIN} · not pre-solved · positive oracle (bowl rests On plate) · **negative
oracles**: target bowl -> every other plate False (directional; also catches
plates within On's 3 cm rule), every sibling bowl -> target plate False
(referential) · graspable (lift > 3 cm).

## Measured (min / mean / max)

| metric | value |
|---|---|
| visibility | {stat(vis)} |
| grasp lift (m) | {stat(lift)} |

Rejections: {len(fails)} ({', '.join(f'{k} x{v}' for k, v in Counter(w.split('_')[0] for w in fails).most_common())}).

## Notes

- `On(bowl, plate)` is True within 3 cm xy + contact; the directional negative
  oracle is what guarantees a bowl on the wrong plate does NOT satisfy the goal.
- Ambiguity is object/destination multiplicity; agentview rarely occludes, so the
  visibility gate seldom binds.
"""
    open(os.path.join(OUT_ROOT, "DATASHEET.md"), "w").write(md)


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    manifest = []; idx = 0; fails = []
    for tier, combos in tier_specs(SMOKE):
        for combo in combos:
            made = None
            for attempt in range(24):
                seed = 2000 + idx * 100 + attempt
                sd = os.path.join(OUT_ROOT, f"scene_{idx:04d}")
                try:
                    made, why = build_scene(combo, tier, seed, sd,
                                            dump_ext=(SMOKE and attempt == 0))
                except Exception as e:
                    made, why = None, f"error:{type(e).__name__}:{e}"
                print(f"  scene_{idx:04d} {tier} N{combo[0]}/M{combo[1]} seed={seed} -> {why}")
                if made:
                    break
                fails.append(why)
            if made:
                made["dir"] = f"scene_{idx:04d}"
                manifest.append({k: made[k] for k in
                                 ("dir", "tier", "counts", "seed", "target", "dest",
                                  "language", "clearance_xy", "px_sep_plates",
                                  "px_sep_siblings", "visibility", "grasp")})
                print(f"[ok] scene_{idx:04d} {tier} vis={made['visibility']['visibility']} "
                      f"lift={made['grasp']['lift']} pxP={made['px_sep_plates']} "
                      f"pxS={made['px_sep_siblings']}")
            else:
                print(f"[FAIL] scene_{idx:04d} {tier} {combo}")
            idx += 1
    json.dump(manifest, open(os.path.join(OUT_ROOT, "manifest.json"), "w"), indent=2)

    imgs = []
    for e in manifest:
        im = cv2.imread(os.path.join(OUT_ROOT, e["dir"], "sketch.png"))
        if im is None:
            continue
        im = cv2.resize(im, (128, 128), interpolation=cv2.INTER_NEAREST)
        cv2.putText(im, f"{e['tier'][:4]} {e['counts'][TARGET_CAT]}b{e['counts'][DEST_CAT]}p"
                    f" v{e['visibility']['visibility']:.2f}", (2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
        imgs.append(im)
    if imgs:
        cols = min(8, len(imgs)); rows = (len(imgs) + cols - 1) // cols
        sheet = np.full((rows * 130, cols * 130, 3), 40, np.uint8)
        for k, im in enumerate(imgs):
            r, c = divmod(k, cols)
            sheet[r * 130 + 1:r * 130 + 129, c * 130 + 1:c * 130 + 129] = im
        safe_imwrite(os.path.join(OUT_ROOT, "contact_sheet.png"), sheet)
    write_datasheet(manifest, fails)

    from collections import Counter
    print(f"\nSMOKE={SMOKE}  {len(manifest)} scenes kept, {len(fails)} rejections")
    for why, c in Counter(w.split("_")[0] for w in fails).most_common():
        print(f"   reject[{why}] x{c}")
    for e in manifest:
        print(f"  {e['dir']} {e['tier']:11s} N={e['counts'][TARGET_CAT]} "
              f"M={e['counts'][DEST_CAT]} vis={e['visibility']['visibility']:.2f} "
              f"lift={e['grasp']['lift']:.3f} pxP={e['px_sep_plates']} pxS={e['px_sep_siblings']}")


if __name__ == "__main__":
    main()
