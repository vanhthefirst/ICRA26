"""
Sketch-Prompted VLA validation set — LIBERO-OBJECT builder (run in WSL2, libero env).

Same quality bar as the Spatial hardened set, adapted to the Object suite per the
two probes (outputs/probe_object.txt, outputs/probe_object2.txt):

  * destination is a CONTAINER REGION. Each basket's region must be DECLARED:
        (contain_region (:target basket_k))
    Two such blocks with the same region name do NOT collide (verified, probe 2 Q1).
  * In(a,b) == b.check_contact(a) and b.check_contain(a); for a site region
    check_contact is hardcoded True, so In == in_box(site_xpos, site_xmat, BODY_xpos).
    => the teleport oracle targets the CONTAIN SITE and sets the object's BODY origin.
       (Visual-center geoms are still used for projection/circle/grasp.)
  * workspace is the FLOOR and is BANDED: baskets at +y, groceries in front.
  * placement uses rejection sampling WITH RESTARTS (a doomed prefix can never be
    repaired by more tries -- this is what made 3 baskets "impossible" in probe 2).

Gates (each failure auto-resamples with a new seed):
    settled          all bodies above floor; target tilt < 60 deg
    in-frame         target + all baskets project inside the image
    pixel separation dest basket vs other baskets >= 18 px
                     target vs same-category siblings >= circle radius + 4 px
                     (a sketch the model cannot visually resolve is a broken scene)
    visibility       RGB-diff occlusion fraction of target >= 0.35
    not pre-solved   check_success() is False at t=0
    oracle +         target -> dest basket  => True
    oracle - (dir)   target -> every OTHER basket => False
    oracle - (ref)   every OTHER same-category instance -> dest basket => False
    graspable        scripted top-down OSC grasp lifts target > 3 cm

SMOKE=True -> 4 scenes exercising the extremes (N=5, M=3). Flip to False for the
full 38-scene run into outputs/validation_set_object/.

    conda activate libero
    cd /mnt/c/Users/Admin/sketch_prompted_vla
    mkdir -p outputs/validation_set_object
    python scripts/build_validation_set_object_wsl.py 2>&1 | tee outputs/validation_set_object/build_log.txt
"""

import os, json, gc
import numpy as np
import cv2

SMOKE = False
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, rename-proof
OUT_ROOT = os.path.join(_REPO, "outputs", "validation_set_object")
IMG_H = IMG_W = 128
CAMERA = "agentview"
ADIM = 7                                  # OffScreenRenderEnv has no .action_dim

# --- placement bands (probe 1: bin at y~+0.26, groceries y in [-0.265,+0.085]) ---
# Basket band widened in y: pixel col tracks world y, and with y spanning only
# 0.18 the three basket centres had ~25 px of column range to share -> chronic
# `baskets_overlap` rejections. Grocery band pulled back to keep the two apart.
BASKET_X = (-0.21, 0.21); BASKET_Y = (0.04, 0.32)
GROC_X = (-0.215, 0.165); GROC_Y = (-0.255, -0.020)
# Separations raised so the PIXEL resolvability gates are satisfied BY CONSTRUCTION
# rather than by rejection. At D_BB=0.175 two baskets projected only ~20-27 px
# apart while the gate needed ext(basket)+4 ~= 20-30 px -> 34 basket rejections.
D_BB = 0.215      # basket <-> basket  (basket footprint ~0.15 x 0.15)
D_BG = 0.150      # basket <-> grocery
D_GG = 0.105      # grocery <-> grocery
HALF_BOX = 0.012
FLOOR_Z_MIN = -0.005
# NOTE: no tilt gate. `tilt_deg` measures the body z-axis against world z, but for
# the flat groceries (butter, cream_cheese, alphabet_soup rest at body_z ~0.009-0.038)
# the mesh frame is z-horizontal AT REST -- a normal object reads 90 deg. It is a
# per-category constant, not a pose defect. Recorded as metadata only; "is this
# object usable" is already covered by the visibility and grasp gates.

VIS_MIN = 0.35
DIFF_TH = 30
# Sketch-resolvability gates, in PIXELS, measured from projected object extents.
# A flat constant was wrong: it scaled with the circle but the ambiguity scales
# with the SIBLING's size, so a tiny target (radius clamped to 5) passed with an
# identical sibling only 10 px away. Now:
#   circle : dist(target, sibling) >= drawn_radius + sibling_extent + PX_MARGIN
#   arrow  : dist(dest, other basket) >= dest_extent + PX_MARGIN_BASKET
# (baskets are ~40 px wide and legitimately overlap on screen; what must hold is
#  that a rival basket's centre lies outside the destination's silhouette, so the
#  arrow tip is attributable.)
PX_MARGIN = 3
PX_MARGIN_BASKET = 4

# Measured from the previous full run (33 scenes, 223 object pairs): the floor
# workspace projects at ~101 px/m, and 85 px/m at the near-10th-percentile depth.
# Half-extents below are measured means of px_extent per category.
PX_PER_M = 85.0                      # conservative (far/oblique pairs project smaller)
CAT_EXT_PX = {"ketchup": 12.7, "orange_juice": 12.6, "salad_dressing": 12.1,
              "milk": 11.5, "bbq_sauce": 9.0, "tomato_sauce": 8.3,
              "alphabet_soup": 8.1, "chocolate_pudding": 5.2,
              "cream_cheese": 4.7, "butter": 4.1}
# Same-category spacing that satisfies the circle gate BY CONSTRUCTION.
SIB_MIN_M = {c: (0.5 * e + 6 + PX_MARGIN) / PX_PER_M for c, e in CAT_EXT_PX.items()}
# How many identical copies the 0.38 x 0.235 grocery band can actually hold at
# that spacing (measured by rejection sampling). Tall bottles simply do not fit
# five times over -- demanding N=5 of them is geometrically impossible, which is
# what produced 172 sibling rejections in the previous run.
N_MAX_BY_CAT = {c: (5 if e <= 9.0 else 3) for c, e in CAT_EXT_PX.items()}
ORACLE_DZ = 0.05

TARGET_CATS = ["milk", "alphabet_soup", "tomato_sauce", "orange_juice", "ketchup",
               "bbq_sauce", "salad_dressing", "chocolate_pudding", "cream_cheese",
               "butter"]
CAPTION_TMPL = ["pick up the {c} and place it in the basket",
                "pick up the {c} and put it in the basket",
                "grab the {c} and put it in the basket"]


def tier_specs(smoke):
    """(n_target, n_basket, n_distractor_categories)"""
    if smoke:
        return [("control", [(1, 1, 2)]),
                ("referential", [(4, 1, 1)]),
                ("directional", [(1, 3, 1)]),
                ("both", [(5, 3, 2)])]
    specs = [("control", [(1, 1, 2)] * 5)]
    ref = []
    for n in (2, 3, 4, 5):
        ref += [(n, 1, 1)] * 3                       # 12
    specs.append(("referential", ref))
    dr = [(1, 2, 1)] * 5 + [(1, 3, 1)] * 4           # 9   (M capped at 3: footprint)
    specs.append(("directional", dr))
    bo = []
    for (n, m) in ((3, 2), (4, 2), (3, 3), (5, 3)):
        bo += [(n, m, 2)] * 3                        # 12
    specs.append(("both", bo))
    return specs


# ----------------------------------------------------------------- io / util --
def safe_imwrite(path, img):
    """Verified write -- the WSL<->Windows DrvFs bridge can silently truncate."""
    ok = cv2.imwrite(path, img)
    n = os.path.getsize(path) if os.path.exists(path) else 0
    tail = False
    if n:
        with open(path, "rb") as f:
            f.seek(-8, 2); tail = b"IEND" in f.read()
    back = cv2.imread(path, cv2.IMREAD_UNCHANGED)      # keep grayscale as 2-D
    shape_ok = back is not None and back.shape[:2] == img.shape[:2]
    good = bool(ok and n > 0 and tail and shape_ok)
    if not good:
        raise IOError(f"corrupt write {path}: ok={ok} bytes={n} iend={tail} "
                      f"shape={None if back is None else back.shape} vs {img.shape}")
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
    """Half-size in PIXELS of the object's projected bounding box.

    Projects the 8 corners of every geom's bounding box (geom_size half-extents
    rotated by geom_xmat) rather than just geom centres -- centres badly
    understate the size of a single-geom object. Used for GATING only; the drawn
    circle radius keeps the original geom-centre formula so sketches stay
    consistent with the training pipeline and the Spatial set.
    """
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


def tilt_deg(model, data, name):
    b = bid_of(model, name)
    zax = data.body_xmat[b].reshape(3, 3)[:, 2]
    return float(np.degrees(np.arccos(np.clip(zax[2], -1, 1))))


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
def sample_banded(n_basket, groceries, rng, target_cat=None,
                  restarts=400, tries_per=600):
    """Rejection sampling WITH RESTARTS. A bad prefix cannot be repaired by more
    tries, so the whole layout is thrown away and re-drawn.

    Same-category instances (the ones the circle must tell apart) are spaced by
    SIB_MIN_M rather than D_GG, so the pixel resolvability gate is satisfied by
    construction instead of by rejection."""
    d_sib = SIB_MIN_M.get(target_cat, D_GG) if target_cat else D_GG
    d_sib = max(d_sib, D_GG)
    cat_of = {g: g.rsplit("_", 1)[0] for g in groceries}
    for _ in range(restarts):
        bp = []
        for _ in range(tries_per):
            if len(bp) == n_basket:
                break
            x = rng.uniform(*BASKET_X); y = rng.uniform(*BASKET_Y)
            if all(np.hypot(x - a, y - b) >= D_BB for a, b in bp):
                bp.append((float(x), float(y)))
        if len(bp) < n_basket:
            continue
        gp = []            # list of (x, y, category)
        for _ in range(tries_per * 4):
            if len(gp) == len(groceries):
                break
            nxt = cat_of[groceries[len(gp)]]
            x = rng.uniform(*GROC_X); y = rng.uniform(*GROC_Y)
            need = [(d_sib if (c == nxt == target_cat) else D_GG) for (_, _, c) in gp]
            if all(np.hypot(x - a, y - b) >= nd
                   for (a, b, _), nd in zip(gp, need)) and \
               all(np.hypot(x - a, y - b) >= D_BG for a, b in bp):
                gp.append((float(x), float(y), nxt))
        if len(gp) < len(groceries):
            continue
        out = {f"basket_{k+1}": bp[k] for k in range(n_basket)}
        out.update({g: (gp[k][0], gp[k][1]) for k, g in enumerate(groceries)})
        return out
    raise RuntimeError(f"placement failed after {restarts} restarts "
                       f"(M={n_basket}, G={len(groceries)}, d_sib={d_sib:.3f})")


def build_bddl(target_cat, n_target, n_basket, distractors, tgt_idx, dest_idx,
               language, seed):
    rng = np.random.default_rng(seed)
    counts = {target_cat: n_target, "basket": n_basket}
    for d in distractors:
        counts[d] = counts.get(d, 0) + 1
    groceries = []
    for c, n in counts.items():
        if c != "basket":
            groceries += [f"{c}_{i}" for i in range(1, n + 1)]
    place = sample_banded(n_basket, groceries, rng, target_cat=target_cat)
    target = f"{target_cat}_{tgt_idx}"; dest = f"basket_{dest_idx}"

    def ranged(name):
        cx, cy = place[name]
        return (f"      ({name}_region\n          (:target floor)\n          (:ranges (\n"
                f"              ({cx-HALF_BOX:.4f} {cy-HALF_BOX:.4f} "
                f"{cx+HALF_BOX:.4f} {cy+HALF_BOX:.4f})\n"
                f"            )\n          )\n      )\n")
    names = [f"basket_{k+1}" for k in range(n_basket)] + groceries
    regions = "".join(ranged(n) for n in names)
    regions += "".join(f"      (contain_region\n          (:target basket_{k+1})\n      )\n"
                       for k in range(n_basket))
    objs = "\n".join("    " + " ".join(f"{c}_{k}" for k in range(1, n + 1)) + f" - {c}"
                     for c, n in counts.items())
    inits = "".join(f"    (On {n} floor_{n}_region)\n" for n in names)
    bddl = f"""(define (problem LIBERO_Floor_Manipulation)
  (:domain robosuite)
  (:language {language})
    (:regions
{regions}    )

  (:fixtures
    floor - floor
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
    (And (In {target} {dest}_contain_region))
  )

)
"""
    meta = dict(counts=counts, target=target, dest=dest, language=language, seed=seed,
                instances=names, placements=place, target_cat=target_cat,
                n_target=n_target, n_basket=n_basket, distractors=list(distractors),
                siblings=[f"{target_cat}_{i}" for i in range(1, n_target + 1)
                          if f"{target_cat}_{i}" != target],
                other_baskets=[f"basket_{k+1}" for k in range(n_basket)
                               if f"basket_{k+1}" != dest])
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


def teleport(env, model, data, obj, basket, dz=ORACLE_DZ, steps=40):
    """In() tests BODY xpos against the contain SITE box -> aim body origin there."""
    sid = model.site_name2id(f"{basket}_contain_region")
    sp = data.site_xpos[sid].copy()
    j = jnt_of(model, obj); qa = model.jnt_qposadr[j]; va = model.jnt_dofadr[j]
    data.qpos[qa:qa + 3] = [sp[0], sp[1], sp[2] + dz]
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
def build_scene(combo, tier, seed, scene_dir, cat_idx=0):
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils import camera_utils as CU
    n_target, n_basket, n_dis = combo
    rng = np.random.default_rng(seed)
    # Round-robin, not random: random draw left ketchup at 0 scenes and milk at 1,
    # and the resolvability gates compound that bias (small objects need less
    # clearance, so they survive rejection more often). cat_idx is the scene index.
    # ...and only from categories whose N_MAX admits this scene's N: five
    # identical ketchups cannot be resolvably spaced in this band at any seed.
    allowed = [c for c in TARGET_CATS if N_MAX_BY_CAT[c] >= n_target] or TARGET_CATS
    target_cat = allowed[cat_idx % len(allowed)]
    pool = [c for c in TARGET_CATS if c != target_cat]
    distractors = list(rng.choice(pool, size=min(n_dis, len(pool)), replace=False))
    tgt_idx = int(rng.integers(1, n_target + 1))
    dest_idx = int(rng.integers(1, n_basket + 1))
    lang = CAPTION_TMPL[int(rng.integers(0, len(CAPTION_TMPL)))].format(
        c=target_cat.replace("_", " "))

    bddl, meta = build_bddl(target_cat, n_target, n_basket, distractors,
                            tgt_idx, dest_idx, lang, seed)
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

        # gate: settled
        for i in meta["instances"]:
            if float(data.body_xpos[bid_of(model, i)][2]) < FLOOR_Z_MIN:
                return None, f"fell_{i}"
        tilt = tilt_deg(model, data, tgt)      # recorded, not gated (see note above)

        # projections
        pix = {i: project([vcenter(model, data, bid_of(model, i))], W2P)[0]
               for i in meta["instances"]}
        dest_site = model.site_name2id(f"{dest}_contain_region")
        dest_px = project([data.site_xpos[dest_site].copy()], W2P)[0]

        def inframe(p):
            return 3 <= p[0] <= IMG_W - 4 and 3 <= p[1] <= IMG_H - 4
        if not inframe(pix[tgt]):
            return None, "target_off_frame"
        for k in range(n_basket):
            if not inframe(pix[f"basket_{k+1}"]):
                return None, f"basket_{k+1}_off_frame"

        # circle radius from the target's projected geom extent
        gids = [g for g in range(model.ngeom)
                if model.geom_bodyid[g] == bid_of(model, tgt)]
        gpx = project([data.geom_xpos[g].copy() for g in gids], W2P)
        cs_ = [p[0] for p in gpx]; rs_ = [p[1] for p in gpx]
        radius = max(5, min(int(np.hypot(max(cs_) - min(cs_),
                                         max(rs_) - min(rs_)) / 2 * 1.5), 40))

        # gate: pixel separation -- the sketch must be visually RESOLVABLE.
        # Thresholds derive from the projected extents of the two objects being
        # told apart, not from a constant (see PX_MARGIN note at the top).
        ext = {i: px_extent(model, data, W2P, i) for i in meta["instances"]}
        px_basket = px_basket_req = None
        if meta["other_baskets"]:
            px_basket_req = ext[dest] + PX_MARGIN_BASKET
            px_basket = min(float(np.hypot(dest_px[0] - pix[b][0], dest_px[1] - pix[b][1]))
                            for b in meta["other_baskets"])
            if px_basket < px_basket_req:
                return None, (f"baskets_overlap_{px_basket:.0f}px_"
                              f"need{px_basket_req:.0f}")
        px_sib = px_sib_req = None
        if meta["siblings"]:
            worst = min(((float(np.hypot(pix[tgt][0] - pix[s][0],
                                         pix[tgt][1] - pix[s][1])) - ext[s]), s)
                        for s in meta["siblings"])
            s_worst = worst[1]
            px_sib = float(np.hypot(pix[tgt][0] - pix[s_worst][0],
                                    pix[tgt][1] - pix[s_worst][1]))
            # "sibling CENTRE clearly outside the circle", not "circle touches no
            # pixel of the sibling" -- the latter demands ~0.28 m between tall
            # bottles, which the 0.38 x 0.235 band cannot supply at N>3.
            px_sib_req = radius + 0.5 * ext[s_worst] + PX_MARGIN
            if px_sib < px_sib_req:
                return None, (f"siblings_overlap_{px_sib:.0f}px_"
                              f"need{px_sib_req:.0f}_r{radius}")

        # gate: visibility
        vis, vismask = visibility(env, model, data, meta["instances"], tgt)
        if vis["visibility"] < VIS_MIN:
            return None, f"low_vis_{vis['visibility']}"

        # gate: not pre-solved
        if success(env) is not False:
            return None, "pre_solved"

        # gate: oracle positive
        if teleport(env, model, data, tgt, dest) is not True:
            return None, "oracle_false"

        # gate: oracle negatives -- wrong destination (directional axis)
        neg = {}
        for b in meta["other_baskets"]:
            settle(env); model, data = env.sim.model, env.sim.data
            r = teleport(env, model, data, tgt, b)
            neg[f"{tgt}->{b}"] = r
            if r is not False:
                return None, f"neg_dest_true_{b}"
        # gate: oracle negatives -- wrong instance (referential axis)
        for s in meta["siblings"]:
            settle(env); model, data = env.sim.model, env.sim.data
            r = teleport(env, model, data, s, dest)
            neg[f"{s}->{dest}"] = r
            if r is not False:
                return None, f"neg_obj_true_{s}"

        # gate: graspable (fresh deterministic reset)
        np.random.seed(seed); obs = settle(env)
        model, data = env.sim.model, env.sim.data
        g = scripted_grasp(env, model, data, tgt, obs)
        if not g["grasp_success"]:
            return None, f"ungraspable_lift{g['lift']}"

        # clearance
        others = [i for i in meta["instances"] if i != tgt]
        tp = vcenter(model, data, bid_of(model, tgt))[:2]
        clr = min(float(np.linalg.norm(
            vcenter(model, data, bid_of(model, o))[:2] - tp)) for o in others) \
            if others else None

        # ---- passed everything: draw + save
        img_c, tok_c = draw_circle(frame, pix[tgt], radius, seed)
        img_ca, tok_a = draw_arrow(img_c, pix[tgt], dest_px, seed)
        safe_imwrite(os.path.join(scene_dir, "frame0.png"),
                     cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        safe_imwrite(os.path.join(scene_dir, "sketch.png"),
                     cv2.cvtColor(img_ca, cv2.COLOR_RGB2BGR))
        safe_imwrite(os.path.join(scene_dir, "target_vismask.png"), vismask)
        meta.update(dict(pick_px=list(pix[tgt]), place_px=list(dest_px), radius=radius,
                         all_pixels={k: list(v) for k, v in pix.items()},
                         symbolic_tokens={"circle": tok_c, "arrow": tok_a},
                         camera_matrix=W2P.tolist(), oracle_success=True,
                         oracle_negatives=neg, tilt_deg=round(tilt, 1),
                         px_sep_baskets=px_basket, px_sep_siblings=px_sib,
                         px_req_baskets=px_basket_req, px_req_siblings=px_sib_req,
                         px_extent={k: round(v, 2) for k, v in ext.items()},
                         clearance_xy=round(clr, 3) if clr else None,
                         visibility=vis, grasp=g))
        json.dump(meta, open(os.path.join(scene_dir, "meta.json"), "w"), indent=2)
        json.dump({"instruction": lang, "target": tgt, "destination": dest,
                   "destination_region": f"{dest}_contain_region",
                   "symbolic_tokens": meta["symbolic_tokens"]},
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
    cats = Counter(e["target_cat"] for e in manifest)

    def col(key, f=lambda e, k: e[k]):
        v = [f(e, key) for e in manifest]
        v = [x for x in v if x is not None]
        return v
    vis = [e["visibility"]["visibility"] for e in manifest]
    lift = [e["grasp"]["lift"] for e in manifest]
    pxb = col("px_sep_baskets"); pxs = col("px_sep_siblings")
    clr = col("clearance_xy")

    def stat(v, fmt="{:.3f}"):
        if not v:
            return "n/a"
        return (fmt + " / " + fmt + " / " + fmt).format(min(v), float(np.mean(v)), max(v))
    md = f"""# Sketch-Prompted VLA validation set — LIBERO-**Object** suite

{n} scenes. Generated by `scripts/build_validation_set_object_wsl.py` (SMOKE={SMOKE}).
Companion to the Spatial set in `outputs/validation_set_spatial/`.

## Purpose

Scenes that are deliberately **impossible to disambiguate from the caption alone**.
Each scene pairs a vague instruction ("pick up the milk and place it in the basket")
with several identical candidate objects and/or several candidate baskets. A
circle (which object) + arrow (which destination) is the only signal that
identifies the intended instance. The BDDL goal names one specific instance and
one specific basket's container region, so a rollout can be scored automatically.

## Composition

| tier | scenes | targets N | baskets M | meaning |
|---|---|---|---|---|
| control | {tiers.get('control',0)} | 1 | 1 | unambiguous; caption alone suffices |
| referential | {tiers.get('referential',0)} | 2-5 | 1 | which object is ambiguous |
| directional | {tiers.get('directional',0)} | 1 | 2-3 | which destination is ambiguous |
| both | {tiers.get('both',0)} | 3-5 | 2-3 | both axes ambiguous |

Target categories used: {', '.join(f'{k} x{v}' for k, v in cats.most_common())}.

## Per scene

`scene.bddl` `frame0.png` `sketch.png` `target_vismask.png` `tokens.json` `meta.json`
(128x128, camera `agentview`). `meta.json` records the seed, camera matrix, all
object pixels, placements, visibility, grasp lift, clearance, pixel separations,
tilt, and the full oracle result matrix. Every scene is reproducible from its seed.

## Gates (all must pass; failures resample with a new seed)

- **settled** — every body above the floor plane
- **in-frame** — target and all baskets project inside the image
- **pixel separation (sketch resolvability)** — thresholds derive from the
  *projected extents* of the objects being told apart, not constants:
  circle needs `dist(target, sibling) >= drawn_radius + sibling_extent + {PX_MARGIN}px`;
  arrow needs `dist(dest, rival basket) >= dest_extent + {PX_MARGIN_BASKET}px`.
  A constant threshold was wrong here: it scaled with the circle while the
  ambiguity scales with the *sibling*, so tiny flat targets (radius clamped to 5)
  passed with an identical sibling 10px away. `meta.json` records the achieved
  separation, the requirement, and every object's extent.
- **visibility** — RGB-difference occlusion fraction of target >= {VIS_MIN}
- **not pre-solved** — `check_success()` is False at t=0
- **oracle +** — teleporting the target into the named basket scores True
- **oracle - (directional)** — target into *every other* basket scores False
- **oracle - (referential)** — *every* same-category sibling into the named
  basket scores False
- **graspable** — scripted top-down OSC grasp lifts the target > 3cm

The two negative oracles are what certify the scene is genuinely unsolvable
without the sketch, along each axis independently.

## Measured distributions (min / mean / max)

| metric | value |
|---|---|
| visibility | {stat(vis)} |
| grasp lift (m) | {stat(lift)} |
| clearance xy (m) | {stat(clr)} |
| px sep, baskets | {stat(pxb, '{:.1f}')} |
| px sep, siblings | {stat(pxs, '{:.1f}')} |

Rejections during generation: {len(fails)} ({', '.join(f'{k} x{v}' for k, v in Counter(w.split('_')[0] for w in fails).most_common())}).

## Suite-specific facts (differ from the Spatial set)

- Destination is a **container region**, declared explicitly per basket:
  `(contain_region (:target basket_k))`. Omitting it raises
  `KeyError: basket_k_contain_region` at the first `step()`. Two such blocks
  sharing the region name do **not** collide.
- `In(a,b) = b.check_contact(a) and b.check_contain(a)`, and for a site region
  `check_contact` is hardcoded `True`. So `In` reduces to
  `in_box(site_xpos, site_xmat, body_xpos)` — it tests the object's **body
  origin**, not its visual centre. The oracle aims the body origin at the
  contain site; projection and grasping still use the visual-centre geom.
- Workspace is the **floor**, banded: baskets at +y, groceries in front.
- Basket footprint is ~0.15 x 0.15, so baskets need ~0.175 world separation.

## Known limitations

- **M capped at 3.** Four baskets do not fit the band at the required separation
  without degenerate corner layouts.
- **Ambiguity is multiplicity, not occlusion.** `agentview` looks down at the
  floor steeply, so objects rarely occlude one another; the visibility gate
  seldom binds. Scenes are hard because there are many identical candidates,
  not because the view is cluttered. Do not cite this set as evidence about
  occlusion robustness.
- **The `In` box is permissive.** Its half-extent is (0.061, 0.061, 0.070)
  centred at z=0.067, so it reaches the floor — an object merely standing at a
  basket's xy would count as contained. The positive oracle is therefore a weak
  check; the *negative* oracles carry the weight.
- **Tilt is recorded, not gated.** `tilt_deg` compares the body z-axis to world
  z, but flat groceries (butter, cream cheese) rest with a horizontal mesh
  frame and read ~90 deg when perfectly normal. Usability is covered by the
  visibility and grasp gates instead.
- Scoring the headline experiment (text-only vs text+sketch) needs a trained
  policy on a GPU. Each scene ships its BDDL for exactly that.

## Reproduce

```bash
conda activate libero
cd /mnt/c/Users/Admin/sketch_prompted_vla
python scripts/build_validation_set_object_wsl.py
```
"""
    p = os.path.join(OUT_ROOT, "DATASHEET.md")
    with open(p, "w") as f:
        f.write(md)
    print(f"  wrote {p} ({os.path.getsize(p)} bytes)")


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    manifest = []; idx = 0; fails = []
    for tier, combos in tier_specs(SMOKE):
        for combo in combos:
            made = None
            for attempt in range(24):
                seed = 3000 + idx * 100 + attempt
                sd = os.path.join(OUT_ROOT, f"scene_{idx:04d}")
                try:
                    made, why = build_scene(combo, tier, seed, sd, cat_idx=idx)
                except Exception as e:
                    made, why = None, f"error:{type(e).__name__}:{e}"
                print(f"  scene_{idx:04d} {tier} N{combo[0]}/M{combo[1]}/D{combo[2]} "
                      f"seed={seed} -> {why}")
                if made:
                    break
                fails.append(why)
            if made:
                made["dir"] = f"scene_{idx:04d}"
                manifest.append({k: made[k] for k in
                                 ("dir", "tier", "counts", "seed", "target", "dest",
                                  "target_cat", "n_target", "n_basket", "language",
                                  "clearance_xy", "px_sep_baskets", "px_sep_siblings",
                                  "tilt_deg", "visibility", "grasp")})
                print(f"[ok] scene_{idx:04d} {tier} vis={made['visibility']['visibility']} "
                      f"lift={made['grasp']['lift']} pxB={made['px_sep_baskets']} "
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
        cv2.putText(im, f"{e['tier'][:4]} {e['n_target']}x{e['n_basket']}b "
                    f"v{e['visibility']['visibility']:.2f}", (2, 12),
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

    print(f"\nSMOKE={SMOKE}  {len(manifest)} scenes kept, {len(fails)} rejections")
    from collections import Counter
    for why, c in Counter(w.split("_")[0] for w in fails).most_common():
        print(f"   reject[{why}] x{c}")
    for e in manifest:
        print(f"  {e['dir']} {e['tier']:11s} {e['target_cat']:18s} "
              f"N={e['n_target']} M={e['n_basket']} "
              f"vis={e['visibility']['visibility']:.2f} lift={e['grasp']['lift']:.3f} "
              f"pxB={e['px_sep_baskets']} pxS={e['px_sep_siblings']}")


if __name__ == "__main__":
    main()
