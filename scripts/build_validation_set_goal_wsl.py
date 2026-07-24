"""
DrawVLA validation set — LIBERO-GOAL builder (run in WSL2, libero env).

Structurally parallel to build_validation_set_object_wsl.py and
build_validation_set_spatial_v2_wsl.py — SAME gate stack, SAME canonical schema
(SCHEMA.md v1.0), SAME sketch drawing. What is different about Goal, and why this
file is longer than its two siblings:

  * Goal has NO single flat workspace we author from scratch. Each task ships a
    bespoke scene with real fixtures (wooden_cabinet, flat_stove, wine_rack) and
    affordance regions (flat_stove_1_cook_region, wine_rack_1_top_region,
    wooden_cabinet_1_top_side/top_region, main_table_stove_front_region). We must
    therefore START FROM THE ORIGINAL BDDL and INJECT duplicate instances +
    placement regions, keeping every fixture, region and (:init) line intact and
    only retargeting the (:goal) to the chosen instance.
  * The destination is EITHER an object instance (plate_1 / akita_black_bowl_1 —
    DUPLICABLE -> directional/both tiers) OR a fixed affordance region (NOT
    duplicable -> referential-only). Probe (outputs/probe_goal.txt, A2) measured
    the split: 2 object-dest tasks, 6 region-dest tasks. This builder implements
    OPTION 1: every usable task feeds control+referential (duplicate the TARGET,
    arrow -> its fixed region or the destination); the 2 object-dest tasks
    additionally feed directional+both (duplicate the DESTINATION).
  * GRASP IS RECORDED, NOT GATED (Aaron's call, this session). The scripted
    top-down grasp fails on wine_bottle and plate (probe B: lift ~0.02 m / 0.0) —
    a scripted-oracle limitation, not a scene defect, exactly like tilt in the
    Object suite (SUITE_FACTS 3.6). oracle_success (teleport) is what certifies a
    scene, and it is independent of the scripted grasp. So we compute grasp, store
    it in meta['grasp'], and NEVER reject on grasp failure. Keeping wine_bottle and
    plate keeps all 8 usable tasks and full referential diversity.

===========================================================================
ASSUMPTIONS THE FIRST WSL RUN (VSLICE) MUST CONFIRM — this file is written
against the two siblings + the probe, but four things were never observed
directly and the VSLICE dump is here to surface them loudly:
  A1. Original-BDDL region syntax: each (:regions) entry is
      (name (:target FIXTURE) (:ranges ((x0 y0 x1 y1) ...)) [ (:yaw ..) ]).
      A region declared with (:target main_table) is referenced elsewhere as
      main_table_<name> (LIBERO prefixes the fixture) — confirmed by the probe's
      object_states_dict keys (main_table_plate_region, ...). We rely on this when
      naming injected duplicate regions.
  A2. [RESOLVED] On-region oracle now confirmed: the vslice built bowl_on_stove
      (On -> flat_stove_1_cook_region) successfully, so teleport_dest onto an
      affordance site scores True. (In-region on the drawer does NOT — see A4.)
  A3. Fixture footprints come from the original (:regions) rectangles. If a fixture
      has NO init region declared, we cannot rect-avoid it; we mitigate by sampling
      duplicates inside the bounding box of the original OBJECT regions (objects sit
      in the reachable front band, away from back fixtures). The settled/in-frame/
      oracle gates catch gross collisions; the VSLICE dump prints every parsed rect.
  A4. [RESOLVED -> DROPPED] The drawer task (open_the_top_drawer_and_put_the_bowl_
      inside) failed all 24 seeds with oracle_false: the drawer starts CLOSED so the
      In-region site is retracted and no teleport satisfies In. It needs OPEN then
      INSERT (two actions a single circle+arrow cannot express — the libero_10
      rationale), so it is removed from GOAL_TASKS. Roster is now 7 usable tasks,
      all On: 2 object-dest + 5 region-dest.
===========================================================================

Modes:
  VSLICE=True  -> ONE referential scene of put_the_bowl_on_the_plate (2 bowls,
                  1 plate) + a full structure dump of the parsed original BDDL.
                  This is the probe-and-slice: run it FIRST, paste it back.
  SMOKE=True   -> one scene per tier across a couple of tasks (extremes).
  both False   -> full 38-scene run into outputs/validation_set_goal/.

    conda activate libero
    cd /mnt/c/Users/Admin/sketch_vla
    mkdir -p outputs/validation_set_goal
    python scripts/build_validation_set_goal_wsl.py 2>&1 | tee outputs/validation_set_goal/build_log.txt
"""

import os, re, json, gc
import numpy as np
import cv2

# ----------------------------------------------------------------- run mode ---
VSLICE = False         # rack seating oracle FIXED (wine_on_rack 2/2 ok, no regression).
SMOKE = False          # <- FULL run: 38 scenes, all 7 tasks, into outputs/validation_set_goal/.
BDDL_ROOT = os.environ.get(
    "LIBERO_GOAL_BDDL", "/root/LIBERO/libero/libero/bddl_files/libero_goal")
OUT_ROOT = "/mnt/c/Users/Admin/sketch_vla/outputs/validation_set_goal"
IMG_H = IMG_W = 128
CAMERA = "agentview"
ADIM = 7
TABLE_Z_MIN = 0.80                 # goal table sits ~0.90; below this a body fell
HALF_BOX = 0.012                   # injected placement region half-size (world m)
HALF_OBJ = 0.045                   # object footprint half-size for rect-avoidance
VIS_MIN = 0.35
DIFF_TH = 30
PX_MARGIN = 3
PX_MARGIN_DEST = 4
ORACLE_DZ = 0.05

# ---------------------------------------------------------------- task roster --
# The 8 usable On/In tasks (probe section A). Each: filename, target instance
# (always index 1 = the ORIGINAL, canonically placed), target category, the
# destination as it appears as the goal's 2nd arg, the destination KIND, the
# owning object/fixture (destination for the schema), goal predicate, and a vague
# caption that never names an instance. Object-dest tasks are marked duplicable.
GOAL_TASKS = {
    "bowl_on_plate": dict(
        file="put_the_bowl_on_the_plate.bddl", predicate="On",
        target_cat="akita_black_bowl", dest_kind="OBJECT",
        dest_inst="plate_1", dest_cat="plate", dest_region="plate_1",
        caption="put the bowl on the plate"),
    "cheese_in_bowl": dict(
        file="put_the_cream_cheese_in_the_bowl.bddl", predicate="On",
        target_cat="cream_cheese", dest_kind="OBJECT",
        dest_inst="akita_black_bowl_1", dest_cat="akita_black_bowl",
        dest_region="akita_black_bowl_1",
        caption="put the cream cheese in the bowl"),
    "bowl_on_stove": dict(
        file="put_the_bowl_on_the_stove.bddl", predicate="On",
        target_cat="akita_black_bowl", dest_kind="REGION",
        dest_inst="flat_stove_1", dest_cat="flat_stove",
        dest_region="flat_stove_1_cook_region",
        caption="put the bowl on the stove"),
    "bowl_on_cabinet": dict(
        file="put_the_bowl_on_top_of_the_cabinet.bddl", predicate="On",
        target_cat="akita_black_bowl", dest_kind="REGION",
        dest_inst="wooden_cabinet_1", dest_cat="wooden_cabinet",
        dest_region="wooden_cabinet_1_top_side",
        caption="put the bowl on top of the cabinet"),
    "plate_to_stove_front": dict(
        file="push_the_plate_to_the_front_of_the_stove.bddl", predicate="On",
        target_cat="plate", dest_kind="REGION",
        dest_inst="main_table", dest_cat="table",
        dest_region="main_table_stove_front_region",
        caption="push the plate to the front of the stove"),
    "wine_on_rack": dict(
        file="put_the_wine_bottle_on_the_rack.bddl", predicate="On",
        target_cat="wine_bottle", dest_kind="REGION",
        dest_inst="wine_rack_1", dest_cat="wine_rack",
        dest_region="wine_rack_1_top_region",
        caption="put the wine bottle on the rack"),
    "wine_on_cabinet": dict(
        file="put_the_wine_bottle_on_top_of_the_cabinet.bddl", predicate="On",
        target_cat="wine_bottle", dest_kind="REGION",
        dest_inst="wooden_cabinet_1", dest_cat="wooden_cabinet",
        dest_region="wooden_cabinet_1_top_side",
        caption="put the wine bottle on top of the cabinet"),
    # DROPPED: open_the_top_drawer_and_put_the_bowl_inside. Its goal is
    # In(bowl, wooden_cabinet_1_top_region) only, but the drawer starts CLOSED, so
    # the region site is retracted inside the cabinet and no teleport can satisfy
    # In — the vslice confirmed this (all 24 seeds -> oracle_false). Achieving it
    # requires OPEN then INSERT: two actions a single circle+arrow cannot express.
    # That is precisely why libero_10 was postponed (SUITE_FACTS / handoff). So we
    # drop it here for the same reason, leaving 7 usable tasks (all On). If we ever
    # want it, the scene must ship with the drawer pre-opened (init Open), which
    # changes the task, and the oracle must set the drawer joint before teleport.
}
OBJECT_DEST_TASKS = [k for k, v in GOAL_TASKS.items() if v["dest_kind"] == "OBJECT"]
REGION_DEST_TASKS = [k for k, v in GOAL_TASKS.items() if v["dest_kind"] == "REGION"]

# same-category sibling spacing (world m) so the circle gate holds BY CONSTRUCTION.
# From Spatial/Object: bowls carry a big circle, bottles are tall, cheese is tiny.
SIB_SPACING = {"akita_black_bowl": 0.150, "plate": 0.150,
               "wine_bottle": 0.150, "cream_cheese": 0.110}
# provisional per-category copy caps for the goal table (fixtures eat space); the
# full run will refine these the way Object's N_MAX_BY_CAT was measured.
N_MAX = {"akita_black_bowl": 4, "plate": 3, "wine_bottle": 3, "cream_cheese": 4}


def tier_specs(vslice, smoke):
    """Yield (task_key, tier, n_target, n_dest). n_dest>1 only for object-dest."""
    if vslice:
        # Targeted re-test of the wine-rack seating oracle: the ONLY task that
        # failed the full run. wine_on_cabinet is the flat-surface control (must
        # still pass -> proves no regression from the new seat search).
        return [("wine_on_rack", "control", 1, 1),
                ("wine_on_rack", "referential", 2, 1),
                ("wine_on_cabinet", "control", 1, 1)]
    if smoke:
        return [("bowl_on_stove", "control", 1, 1),
                ("bowl_on_stove", "referential", 4, 1),
                ("bowl_on_plate", "directional", 1, 3),
                ("bowl_on_plate", "both", 3, 2)]
    specs = []
    # control (5): spread across usable tasks, unambiguous (1 target, 1 dest)
    ctrl_tasks = ["bowl_on_plate", "bowl_on_stove", "wine_on_rack",
                  "plate_to_stove_front", "cheese_in_bowl"]
    specs += [(t, "control", 1, 1) for t in ctrl_tasks]
    # referential (12): duplicate the TARGET, across all usable tasks
    ref_plan = [("bowl_on_plate", 3), ("bowl_on_stove", 4), ("bowl_on_cabinet", 2),
                ("wine_on_rack", 3), ("wine_on_cabinet", 2), ("plate_to_stove_front", 2),
                ("cheese_in_bowl", 4), ("bowl_on_plate", 2), ("wine_on_rack", 2),
                ("bowl_on_stove", 2), ("cheese_in_bowl", 3), ("bowl_on_cabinet", 3)]
    specs += [(t, "referential", n, 1) for (t, n) in ref_plan]  # 12 (7 usable tasks)
    # directional (9): duplicate the DESTINATION -> object-dest tasks only
    dir_plan = [("bowl_on_plate", 2), ("bowl_on_plate", 3), ("bowl_on_plate", 3),
                ("cheese_in_bowl", 2), ("cheese_in_bowl", 3), ("cheese_in_bowl", 3),
                ("bowl_on_plate", 2), ("cheese_in_bowl", 2), ("bowl_on_plate", 3)]
    specs += [(t, "directional", 1, m) for (t, m) in dir_plan]  # 9
    # both (12): duplicate target AND destination -> object-dest tasks only
    both_plan = [("bowl_on_plate", 2, 2), ("bowl_on_plate", 3, 2), ("bowl_on_plate", 3, 3),
                 ("cheese_in_bowl", 2, 2), ("cheese_in_bowl", 3, 2), ("cheese_in_bowl", 3, 3),
                 ("bowl_on_plate", 2, 3), ("cheese_in_bowl", 2, 3), ("bowl_on_plate", 4, 2),
                 ("cheese_in_bowl", 4, 2), ("bowl_on_plate", 3, 2), ("cheese_in_bowl", 3, 2)]
    specs += [(t, "both", n, m) for (t, n, m) in both_plan]  # 12
    return specs


# ============================ shared helpers (identical to the two siblings) ==
def safe_imwrite(path, img):
    ok = cv2.imwrite(path, img)
    n = os.path.getsize(path) if os.path.exists(path) else 0
    tail = False
    if n:
        with open(path, "rb") as f:
            f.seek(-8, 2); tail = b"IEND" in f.read()
    back = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    good = bool(ok and n > 0 and tail and back is not None
                and back.shape[:2] == img.shape[:2])
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
    if target not in saved:
        return dict(v_visible=0, v_full=0, visibility=0.0), \
            np.zeros((IMG_H, IMG_W), np.uint8)

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


def eef(obs):
    for k in ("robot0_eef_pos", "eef_pos"):
        if k in obs:
            return np.asarray(obs[k])
    return None


def scripted_grasp(env, model, data, tgt, obs):
    """RECORDED, NOT GATED. Returned dict is stored in meta['grasp']; callers must
    never reject on grasp_success being False (Aaron's call: scripted-grasp
    failure on wine_bottle/plate is an oracle limitation, not a scene defect)."""
    j = jnt_of(model, tgt)
    if j is None:
        return dict(grasp_success=False, lift=0.0, note="no_free_joint")
    qa = model.jnt_qposadr[j]
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


# ===================================== original-BDDL parsing + injection =======
def _balanced(txt, start):
    """Return substring txt[start:end] where txt[start]=='(' and parens balance."""
    depth = 0
    for j in range(start, len(txt)):
        if txt[j] == "(":
            depth += 1
        elif txt[j] == ")":
            depth -= 1
            if depth == 0:
                return txt[start:j + 1]
    return txt[start:]


def _tag_block(txt, tag):
    """Full '(:tag ... )' sexpr with balanced parens, or ''."""
    i = txt.find(f"(:{tag}")
    return _balanced(txt, i) if i >= 0 else ""


def _top_sexprs(inner):
    out = []; depth = 0; start = None
    for k, ch in enumerate(inner):
        if ch == "(":
            if depth == 0:
                start = k
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                out.append(inner[start:k + 1])
    return out


def _libero_typed(block, head):
    """Reproduce LIBERO's robosuite_parse_problem `:objects`/`:fixtures` loop
    EXACTLY (bddl_utils.py): tokenise, then walk `name... - category`, assigning
    the accumulated names to that category. Returns {category: [instances]} — the
    same grouping LIBERO uses to build objects_dict/fixtures_dict. Using this (not
    a bespoke line parser) guarantees our instance view can never diverge from
    LIBERO's, which is what silently dropped an object and caused the KeyError."""
    if not block:
        return {}
    i = block.find(head)
    inner = block[i + len(head):]
    inner = inner[:inner.rfind(")")]                      # drop the block's close
    toks = inner.replace("(", " ").replace(")", " ").split()
    out = {}
    acc = []
    while toks:
        if toks[0] == "-":
            toks.pop(0)
            cat = toks.pop(0)
            out[cat] = acc              # ASSIGN — faithful to LIBERO. A second
            acc = []                    # group for the same category OVERWRITES
        else:                           # the first (the exact KeyError trigger),
            acc.append(toks.pop(0))     # so the verifier can detect the loss.
    return out


def _init_place_states(text):
    """[(pred, obj, region), ...] from (:init), matching LIBERO's predicate view.
    The object being placed is arg 1 — exactly the key LIBERO looks up in
    objects_dict during placement, so this is what a preflight must validate."""
    blk = _tag_block(text, "init")
    return re.findall(
        r"\(([A-Za-z]+)\s+([A-Za-z0-9_]+)(?:\s+([A-Za-z0-9_]+))?\)",
        " ".join(blk.split()))


def verify_injected_bddl(text):
    """Refuse to hand LIBERO a BDDL it will choke on. Reproduces LIBERO's object
    parse on the FINAL text and asserts every object placed in (:init) and every
    movable object named in the (:goal) is actually declared. Raises ValueError
    with a precise message instead of letting LIBERO die deep inside a KeyError."""
    objs = _libero_typed(_tag_block(text, "objects"), ":objects")
    fixts = _libero_typed(_tag_block(text, "fixtures"), ":fixtures")
    declared = {i for v in objs.values() for i in v} | {i for v in fixts.values() for i in v}
    # every category must keep all its instances after LIBERO's assign-by-category
    dup_seen = {}
    for cat, insts in objs.items():
        if len(insts) != len(set(insts)):
            raise ValueError(f"category '{cat}' has duplicate instance names: {insts}")
    for (p, a, b) in _init_place_states(text):
        if p.lower() in ("on", "in") and a not in declared:
            raise ValueError(
                f"(:init) places undeclared object '{a}' — objects_dict would "
                f"KeyError. Declared movable/fixture instances: {sorted(declared)}")
    return declared


def parse_original(path):
    """Parse the shipped goal BDDL: objects, fixtures, regions(+ranges), goal,
    and the raw text of the :regions / :objects / :init blocks so we can rebuild
    them with duplicates appended."""
    txt = open(path).read()
    flat = " ".join(txt.split())
    lang = re.search(r"\(:language\s+(.*?)\)", txt, re.S)
    lang = " ".join(lang.group(1).split()) if lang else ""

    # Parse (:objects) / (:fixtures) with LIBERO's EXACT token algorithm so our
    # view of the instances is byte-identical to objects_dict/fixtures_dict. A
    # hand-rolled line parser can diverge on a differently-formatted task file and
    # silently drop an object -> the deep KeyError. See _libero_typed.
    obj_block = _tag_block(txt, "objects")
    fx_block = _tag_block(txt, "fixtures")
    objs_by_cat = _libero_typed(obj_block, ":objects")
    fixtures_by_cat = _libero_typed(fx_block, ":fixtures")
    objs = {inst: cat for cat, insts in objs_by_cat.items() for inst in insts}
    fixtures = {inst: cat for cat, insts in fixtures_by_cat.items() for inst in insts}

    reg_block = _tag_block(txt, "regions")
    regions = {}       # name -> dict(target=..., ranges=[(x0,y0,x1,y1),...])
    # content strictly between "(:regions" and its matching final ")", so each
    # top-level sexpr is exactly one region (no outer wrap -> no single-blob bug).
    _rhead = reg_block.find(":regions")
    _first = reg_block.find("(", _rhead + len(":regions")) if _rhead >= 0 else -1
    inner = reg_block[_first:reg_block.rfind(")")] if _first >= 0 else ""
    for sx in _top_sexprs(inner):
        name_m = re.match(r"\(\s*([A-Za-z0-9_]+)", sx)
        if not name_m or name_m.group(1) in (":target", ":ranges", ":yaw"):
            continue
        name = name_m.group(1)
        tgt_m = re.search(r"\(:target\s+([A-Za-z0-9_]+)\)", sx)
        # bound the number scrape to the balanced (:ranges ...) sub-sexpr so a
        # trailing (:yaw ...) can never leak its numbers into the rectangles.
        ranges = []
        ri = sx.find("(:ranges")
        if ri >= 0:
            nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", _balanced(sx, ri))]
            ranges = [tuple(nums[i:i + 4]) for i in range(0, len(nums) - 3, 4)]
        regions[name] = dict(target=tgt_m.group(1) if tgt_m else None, ranges=ranges)

    goal = _tag_block(txt, "goal")
    gflat = " ".join(goal.split())
    preds = re.findall(r"\(([A-Za-z]+)\s+([A-Za-z0-9_]+)(?:\s+([A-Za-z0-9_]+))?\)", gflat)
    preds = [(p, a, b) for (p, a, b) in preds if p.lower() != "and"]
    return dict(text=txt, lang=lang, objs=objs, objs_by_cat=objs_by_cat,
                fixtures=fixtures, fixtures_by_cat=fixtures_by_cat, regions=regions,
                preds=preds, obj_block=obj_block, reg_block=reg_block,
                init_block=_tag_block(txt, "init"))


def occupied_rects(parsed):
    """All (x0,y0,x1,y1) rectangles declared in the original :regions (objects and
    fixtures alike) — everything a duplicate must avoid."""
    rects = []
    for name, r in parsed["regions"].items():
        for (x0, y0, x1, y1) in r["ranges"]:
            rects.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
    return rects


def _rect_of(cx, cy, half):
    return (cx - half, cy - half, cx + half, cy + half)


def _overlap(a, b, pad=0.0):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or
                a[3] + pad < b[1] or b[3] + pad < a[1])


def sample_free(parsed, target_cat, n_extra_target, dest_cat, n_extra_dest,
                orig_target_xy, orig_dest_xy, rng, restarts=500, tries=800):
    """Place duplicate instances in free table space. Sampling rect = bounding box
    of the original OBJECT regions (reachable front band), inflated a little; every
    candidate must avoid all original rects and prior duplicates with per-category
    spacing. RESTARTS, not just retries (SUITE_FACTS 10)."""
    occ = occupied_rects(parsed)
    # sampling rectangle from original object placements
    obj_rects = []
    for name, r in parsed["regions"].items():
        base = name.split("_region")[0]
        if any(base.startswith(o) or o.startswith(base) for o in parsed["objs"]):
            obj_rects += r["ranges"]
    if obj_rects:
        xs = [x for rr in obj_rects for x in (rr[0], rr[2])]
        ys = [y for rr in obj_rects for y in (rr[1], rr[3])]
        RX = (min(xs) - 0.10, max(xs) + 0.10); RY = (min(ys) - 0.10, max(ys) + 0.10)
    else:                                   # fallback: central band
        RX = (-0.15, 0.15); RY = (-0.20, 0.15)

    want = [(target_cat, orig_target_xy)] * 0  # placeholder for readability
    want = [("T", target_cat)] * n_extra_target + [("D", dest_cat)] * n_extra_dest
    for _ in range(restarts):
        placed = []     # (x, y, cat, role)  seeded with the originals
        placed.append((orig_target_xy[0], orig_target_xy[1], target_cat, "T"))
        if orig_dest_xy is not None:
            placed.append((orig_dest_xy[0], orig_dest_xy[1], dest_cat, "D"))
        ok = True
        for role, cat in want:
            got = False
            for _ in range(tries):
                x = rng.uniform(*RX); y = rng.uniform(*RY)
                sp = SIB_SPACING.get(cat, 0.13)
                if any(np.hypot(x - a, y - b) < sp for a, b, c, _ in placed if c == cat):
                    continue
                if any(_overlap(_rect_of(x, y, HALF_OBJ), o, pad=0.01) for o in occ):
                    continue
                if any(np.hypot(x - a, y - b) < 0.09 for a, b, _, _ in placed):
                    continue
                placed.append((float(x), float(y), cat, role)); got = True; break
            if not got:
                ok = False; break
        if ok:
            tdup = [(a, b) for a, b, c, r in placed if r == "T"][1:]   # skip original
            ddup = [(a, b) for a, b, c, r in placed if r == "D"][1:]
            return tdup, ddup
    raise RuntimeError(f"placement failed after {restarts} restarts "
                       f"(nT+{n_extra_target} nD+{n_extra_dest})")


def build_bddl_goal(task_key, tier, n_target, n_dest, dest_idx, seed, orig_xy):
    """Return (bddl_text, meta). Keep the original scene verbatim; append duplicate
    objects + their init regions; retarget the goal to the chosen instance."""
    T = GOAL_TASKS[task_key]
    parsed = orig_xy["parsed"]
    tcat = T["target_cat"]
    target = f"{tcat}_1"                      # circle always on the original
    rng = np.random.default_rng(seed)

    n_extra_t = n_target - 1
    n_extra_d = (n_dest - 1) if T["dest_kind"] == "OBJECT" else 0
    tdup, ddup = sample_free(parsed, tcat, n_extra_t,
                             T["dest_cat"], n_extra_d,
                             orig_xy["target_xy"], orig_xy.get("dest_xy"),
                             rng)

    # --- category -> [instances], seeded from LIBERO's OWN parse (objs_by_cat).
    # CRITICAL: LIBERO's parser keys (:objects) by CATEGORY and *assigns* (not
    # appends), so every instance of a category MUST live on ONE "a b c - cat"
    # line. A separate "cat_2 - cat" line silently overwrites cat_1 (this is the
    # bug that KeyError'd akita_black_bowl_1). So we REBUILD the whole block from
    # objs_by_cat (identical to LIBERO's grouping) and add duplicates in-place.
    import copy as _copy
    cat_insts = _copy.deepcopy(parsed["objs_by_cat"])

    # --- duplicate objects: append instances to their category, add init regions
    dup_inits = []
    dup_regions = []
    sib_names = []
    for k, (x, y) in enumerate(tdup, start=2):
        nm = f"{tcat}_{k}"; sib_names.append(nm)
        cat_insts.setdefault(tcat, []).append(nm)
        rn = f"{nm}_dup"
        dup_regions.append(_region_text(rn, x, y))
        dup_inits.append(f"    (On {nm} main_table_{rn})")
    dest_dup_names = []
    if T["dest_kind"] == "OBJECT":
        dcat = T["dest_cat"]
        for k, (x, y) in enumerate(ddup, start=2):
            nm = f"{dcat}_{k}"; dest_dup_names.append(nm)
            cat_insts.setdefault(dcat, []).append(nm)
            rn = f"{nm}_dup"
            dup_regions.append(_region_text(rn, x, y))
            dup_inits.append(f"    (On {nm} main_table_{rn})")

    # --- destination instance for the goal
    if T["dest_kind"] == "OBJECT":
        dcat = T["dest_cat"]
        dest_inst = f"{dcat}_{dest_idx}"
        dest_region = dest_inst
        destination = dest_inst
        other_dests = [f"{dcat}_1"] + dest_dup_names
        other_dests = [d for d in other_dests if d != dest_inst]
    else:
        dest_inst = T["dest_inst"]
        dest_region = T["dest_region"]
        destination = T["dest_inst"]
        other_dests = []

    pred = T["predicate"]
    new_goal = f"    (And ({pred} {target} {dest_region}))"

    # --- rebuild the blocks. Objects: REPLACE (grouped per category). Regions and
    # init: append duplicates (regions keyed by unique name, init is a flat list —
    # neither has the category-overwrite hazard). Goal: replace.
    obj_lines = "\n".join("    " + " ".join(insts) + f" - {cat}"
                          for cat, insts in cat_insts.items())
    new_obj_block = f"(:objects\n{obj_lines}\n  )"
    text = parsed["text"]
    text = _replace_block(text, parsed["obj_block"], new_obj_block)
    text = _inject_before_close(text, parsed["reg_block"], dup_regions)
    text = _inject_before_close(text, parsed["init_block"], dup_inits)
    text = _replace_block(text, _tag_block(text, "goal"),
                          f"(:goal\n{new_goal}\n  )")

    # preflight: parse the FINAL text the way LIBERO will and refuse to emit it if
    # any (:init)/(:goal) object is undeclared. Turns a deep LIBERO KeyError into a
    # precise, local failure we can see immediately.
    verify_injected_bddl(text)

    meta = dict(task=task_key, file=T["file"], tier=tier, predicate=pred,
                target=target, target_cat=tcat, destination=destination,
                destination_region=dest_region, dest_kind=T["dest_kind"],
                dest_inst=dest_inst, other_dests=other_dests, siblings=sib_names,
                n_target=n_target, n_dest=n_dest, seed=seed, language=T["caption"],
                dup_target_xy=tdup, dup_dest_xy=ddup)
    return text, meta


def _region_text(name, cx, cy):
    return (f"      ({name}\n          (:target main_table)\n          (:ranges (\n"
            f"              ({cx-HALF_BOX:.4f} {cy-HALF_BOX:.4f} "
            f"{cx+HALF_BOX:.4f} {cy+HALF_BOX:.4f})\n            )\n          )\n      )")


def _inject_before_close(text, block, new_lines):
    """Insert new_lines just before the final ')' of `block` inside `text`."""
    if not new_lines or not block:
        return text
    idx = text.find(block)
    if idx < 0:
        return text
    close = block.rfind(")")
    injected = block[:close] + "\n" + "\n".join(new_lines) + "\n" + block[close:]
    return text[:idx] + injected + text[idx + len(block):]


def _replace_block(text, block, replacement):
    idx = text.find(block)
    if idx < 0:
        return text
    return text[:idx] + replacement + text[idx + len(block):]


# ===================================================================== oracle ==
# Orientations to try when SEATING an object into a region (existential search for
# a valid success pose). Upright is first so flat surfaces (stove cook-top, cabinet
# top) still pass on attempt 1 — no regression. The two horizontals lay a bottle
# along the rack's cradle axis; On(bottle, rack) needs BOTH the region box AND
# physical contact with the rack, which an upright bottle loses by rolling off.
_QUAT_UP = [1.0, 0.0, 0.0, 0.0]
_QUAT_HX = [0.70710678, 0.70710678, 0.0, 0.0]      # +90 deg about x -> lie down
_QUAT_HY = [0.70710678, 0.0, 0.70710678, 0.0]      # +90 deg about y -> lie down


def _place_and_settle(env, model, data, qa, va, xyz, quat, steps):
    data.qpos[qa:qa + 3] = xyz
    data.qpos[qa + 3:qa + 7] = quat
    data.qvel[va:va + 6] = 0
    env.sim.forward()
    for _ in range(steps):
        env.step(np.zeros(ADIM))
    return success(env)


def teleport_dest(env, model, data, obj, task, dest_inst, dest_region,
                  dz=ORACLE_DZ, steps=45, seat=False):
    """Move `obj` to its destination and return check_success().
      OBJECT dest -> rest obj on the dest object's visual centre (On semantics).
      REGION dest -> place obj's body origin at the region SITE, settle.
    Works for On and In because we set qpos and let check_success decide.

    seat=True (POSITIVE oracle only): existential search over orientation x dz for a
    pose that scores True — needed for cradle-style regions like the wine rack where
    a single upright drop never contacts the rack. Safe for negatives, which pass
    seat=False: they teleport a NON-target object and check the goal on the UNMOVED
    target, so they are structurally False regardless of pose."""
    T = GOAL_TASKS[task]
    j = jnt_of(model, obj)
    if j is None:
        return None
    qa = model.jnt_qposadr[j]; va = model.jnt_dofadr[j]

    if T["dest_kind"] == "OBJECT":
        pc = vcenter(model, data, bid_of(model, dest_inst))
        return _place_and_settle(env, model, data, qa, va,
                                 [pc[0], pc[1], pc[2] + 0.04], _QUAT_UP, steps)

    # REGION destination
    try:
        sid = model.site_name2id(dest_region)
        sp = data.site_xpos[sid].copy()
    except Exception:
        sp = vcenter(model, data, bid_of(model, T["dest_inst"]))

    if not seat:
        return _place_and_settle(env, model, data, qa, va,
                                 [sp[0], sp[1], sp[2] + dz], _QUAT_UP, steps)

    # positive oracle: try flat-surface upright first, then cradle seatings.
    last = None
    for quat in (_QUAT_UP, _QUAT_HY, _QUAT_HX):
        for ddz in (0.05, 0.02, 0.0, -0.02):
            last = _place_and_settle(env, model, data, qa, va,
                                     [sp[0], sp[1], sp[2] + ddz], quat, max(steps, 55))
            if last is True:
                return True
    return last


def dest_pixel(model, data, W2P, task, dest_inst, dest_region):
    """Arrow tip: the dest object's visual centre, or the region site's xpos."""
    T = GOAL_TASKS[task]
    if T["dest_kind"] == "OBJECT":
        return project([vcenter(model, data, bid_of(model, dest_inst))], W2P)[0]
    try:
        sid = model.site_name2id(dest_region)
        return project([data.site_xpos[sid].copy()], W2P)[0]
    except Exception:
        return project([vcenter(model, data, bid_of(model, T["dest_inst"]))], W2P)[0]


# ================================================================= one scene ===
def _dump_structure(parsed, task_key):
    T = GOAL_TASKS[task_key]
    print("\n" + "-" * 70)
    print(f"STRUCTURE DUMP — {T['file']}")
    print("-" * 70)
    print(f"  lang     : {parsed['lang']}")
    print(f"  objects  : {parsed['objs']}")
    print(f"  fixtures : {parsed['fixtures']}")
    print(f"  goal     : {parsed['preds']}")
    print("  regions  :")
    for n, r in parsed["regions"].items():
        print(f"     {n:38s} target={r['target']}  ranges={r['ranges']}")


def build_scene(task_key, tier, n_target, n_dest, seed, scene_dir, dump=False):
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils import camera_utils as CU
    T = GOAL_TASKS[task_key]
    src = os.path.join(BDDL_ROOT, T["file"])
    parsed = parse_original(src)
    if dump:
        _dump_structure(parsed, task_key)

    # locate the original target/dest placement (region rectangle centre) so
    # duplicates cluster around the real objects. Object init regions are named
    # "<category>_region" (one per object), so match on category.
    def region_centre(cat):
        r = parsed["regions"].get(f"{cat}_region")
        if r and r["ranges"]:
            x0, y0, x1, y1 = r["ranges"][0]
            return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        return (0.0, 0.0)
    tgt_xy = region_centre(T["target_cat"])
    dest_xy = region_centre(T["dest_cat"]) if T["dest_kind"] == "OBJECT" else None
    orig_xy = dict(parsed=parsed, target_xy=tgt_xy, dest_xy=dest_xy)

    rng = np.random.default_rng(seed)
    dest_idx = int(rng.integers(1, n_dest + 1)) if T["dest_kind"] == "OBJECT" else 1

    bddl, meta = build_bddl_goal(task_key, tier, n_target, n_dest, dest_idx, seed, orig_xy)
    os.makedirs(scene_dir, exist_ok=True)
    bpath = os.path.join(scene_dir, "scene.bddl")
    open(bpath, "w").write(bddl)
    if dump:
        print(f"\n  [wrote injected bddl -> {bpath}] "
              f"n_target={n_target} n_dest={n_dest} dest_idx={dest_idx}")

    np.random.seed(seed)
    env = OffScreenRenderEnv(bddl_file_name=bpath, camera_heights=IMG_H,
                             camera_widths=IMG_W, camera_names=[CAMERA])
    try:
        obs = settle(env)
        model, data = env.sim.model, env.sim.data
        frame = frame_obs(obs)
        W2P = CU.get_camera_transform_matrix(sim=env.sim, camera_name=CAMERA,
                                             camera_height=IMG_H, camera_width=IMG_W)
        tgt = meta["target"]
        instances = [tgt] + meta["siblings"]
        if T["dest_kind"] == "OBJECT":
            instances += [meta["dest_inst"]] + [d for d in meta["other_dests"]]
        instances = list(dict.fromkeys(instances))     # de-dup, keep order

        # gate: settled
        for i in instances:
            try:
                if float(vcenter(model, data, bid_of(model, i))[2]) < TABLE_Z_MIN:
                    return None, f"fell_{i}"
            except KeyError:
                return None, f"missing_body_{i}"

        # projections
        pix = {i: project([vcenter(model, data, bid_of(model, i))], W2P)[0]
               for i in instances}
        dpx = dest_pixel(model, data, W2P, task_key, meta["dest_inst"],
                         meta["destination_region"])

        def inframe(p):
            return 3 <= p[0] <= IMG_W - 4 and 3 <= p[1] <= IMG_H - 4
        if not inframe(pix[tgt]):
            return None, "target_off_frame"
        if not inframe(dpx):
            return None, "dest_off_frame"

        # circle radius from target projected geom extent
        gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid_of(model, tgt)]
        gpx = project([data.geom_xpos[g].copy() for g in gids], W2P)
        cs_ = [p[0] for p in gpx]; rs_ = [p[1] for p in gpx]
        radius = max(5, min(int(np.hypot(max(cs_) - min(cs_),
                                         max(rs_) - min(rs_)) / 2 * 1.5), 40))

        # gate: pixel resolvability (siblings vs circle; rival dests vs arrow)
        ext = {i: px_extent(model, data, W2P, i) for i in instances}
        px_sib = px_sib_req = None
        if meta["siblings"]:
            s_worst = min(meta["siblings"],
                          key=lambda s: np.hypot(pix[tgt][0] - pix[s][0],
                                                 pix[tgt][1] - pix[s][1]) - ext[s])
            px_sib = float(np.hypot(pix[tgt][0] - pix[s_worst][0],
                                    pix[tgt][1] - pix[s_worst][1]))
            px_sib_req = radius + 0.5 * ext[s_worst] + PX_MARGIN
            if px_sib < px_sib_req:
                return None, f"siblings_overlap_{px_sib:.0f}px_need{px_sib_req:.0f}_r{radius}"
        px_dest = px_dest_req = None
        if meta["other_dests"]:
            px_dest_req = ext[meta["dest_inst"]] + PX_MARGIN_DEST
            px_dest = min(float(np.hypot(dpx[0] - pix[d][0], dpx[1] - pix[d][1]))
                          for d in meta["other_dests"])
            if px_dest < px_dest_req:
                return None, f"dests_overlap_{px_dest:.0f}px_need{px_dest_req:.0f}"

        # gate: visibility
        vis, vismask = visibility(env, model, data, instances, tgt)
        if vis["visibility"] < VIS_MIN:
            return None, f"low_vis_{vis['visibility']}"

        # gate: not pre-solved
        if success(env) is not False:
            return None, "pre_solved"

        # gate: positive oracle (seat=True -> search for a valid success pose;
        # matters for cradle regions like the wine rack, no-op for flat surfaces)
        if teleport_dest(env, model, data, tgt, task_key,
                         meta["dest_inst"], meta["destination_region"],
                         seat=True) is not True:
            return None, "oracle_false"

        # gate: negative oracles
        neg = {}
        for d in meta["other_dests"]:                  # directional axis
            settle(env); model, data = env.sim.model, env.sim.data
            r = teleport_dest(env, model, data, tgt, task_key, d, d)
            neg[f"{tgt}->{d}"] = r
            if r is not False:
                return None, f"neg_dest_true_{d}"
        for s in meta["siblings"]:                     # referential axis
            settle(env); model, data = env.sim.model, env.sim.data
            r = teleport_dest(env, model, data, s, task_key,
                              meta["dest_inst"], meta["destination_region"])
            neg[f"{s}->{meta['dest_inst']}"] = r
            if r is not False:
                return None, f"neg_obj_true_{s}"

        # grasp — RECORDED, NOT GATED
        np.random.seed(seed); obs = settle(env)
        model, data = env.sim.model, env.sim.data
        grasp = scripted_grasp(env, model, data, tgt, obs)

        # clearance
        others = [i for i in instances if i != tgt]
        tp = vcenter(model, data, bid_of(model, tgt))[:2]
        clr = min(float(np.linalg.norm(vcenter(model, data, bid_of(model, o))[:2] - tp))
                  for o in others) if others else None

        # ---- passed the GATING checks (grasp excluded): draw + save
        img_c, tok_c = draw_circle(frame, pix[tgt], radius, seed)
        img_ca, tok_a = draw_arrow(img_c, pix[tgt], dpx, seed)
        safe_imwrite(os.path.join(scene_dir, "frame0.png"),
                     cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        safe_imwrite(os.path.join(scene_dir, "sketch.png"),
                     cv2.cvtColor(img_ca, cv2.COLOR_RGB2BGR))
        safe_imwrite(os.path.join(scene_dir, "target_vismask.png"), vismask)

        meta.update(dict(
            schema_version="1.0", suite="goal", goal_predicate=T["predicate"],
            instruction=T["caption"],
            pick_px=list(pix[tgt]), place_px=list(dpx), radius=radius,
            all_pixels={k: list(v) for k, v in pix.items()},
            px_extent={k: round(v, 2) for k, v in ext.items()},
            symbolic_tokens={"circle": tok_c, "arrow": tok_a},
            camera_matrix=W2P.tolist(), oracle_success=True, oracle_negatives=neg,
            px_sep_siblings=px_sib, px_req_siblings=px_sib_req,
            px_sep_dests=px_dest, px_req_dests=px_dest_req,
            clearance_xy=round(clr, 3) if clr is not None else None,
            visibility=vis, grasp=grasp))
        json.dump(meta, open(os.path.join(scene_dir, "meta.json"), "w"), indent=2)
        json.dump({"instruction": T["caption"], "suite": "goal", "tier": tier,
                   "target": tgt, "destination": meta["destination"],
                   "destination_region": meta["destination_region"],
                   "goal_predicate": T["predicate"],
                   "symbolic_tokens": meta["symbolic_tokens"]},
                  open(os.path.join(scene_dir, "tokens.json"), "w"), indent=2)
        return meta, "ok"
    finally:
        try:
            env.close()
        except Exception:
            pass
        gc.collect()


# ===================================================================== main ====
def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    specs = tier_specs(VSLICE, SMOKE)
    print(f"MODE: VSLICE={VSLICE} SMOKE={SMOKE}  BDDL_ROOT={BDDL_ROOT}")
    print(f"  {len(specs)} scene(s) planned. Option-1 roster: "
          f"object-dest tasks={OBJECT_DEST_TASKS}, region-dest tasks={REGION_DEST_TASKS}")
    manifest = []; fails = []; idx = 0
    for (task_key, tier, n_target, n_dest) in specs:
        made = None
        for attempt in range(24):
            seed = 4000 + idx * 100 + attempt
            sd = os.path.join(OUT_ROOT, f"scene_{idx:04d}")
            try:
                made, why = build_scene(task_key, tier, n_target, n_dest, seed, sd,
                                        dump=((VSLICE or SMOKE) and attempt == 0))
            except Exception as e:
                import traceback
                made, why = None, f"error:{type(e).__name__}:{e}"
                if VSLICE:
                    traceback.print_exc()
            print(f"  scene_{idx:04d} {task_key} {tier} N{n_target}/D{n_dest} "
                  f"seed={seed} -> {why}")
            if made:
                break
            fails.append(why)
        if made:
            made["dir"] = f"scene_{idx:04d}"
            manifest.append({k: made.get(k) for k in
                             ("dir", "task", "tier", "predicate", "target",
                              "target_cat", "destination", "destination_region",
                              "dest_kind", "n_target", "n_dest", "seed",
                              "clearance_xy", "px_sep_siblings", "px_sep_dests",
                              "visibility", "grasp")})
            print(f"[ok] scene_{idx:04d} {task_key} {tier} "
                  f"vis={made['visibility']['visibility']} "
                  f"grasp={made['grasp']['grasp_success']}/{made['grasp']['lift']} "
                  f"pxS={made['px_sep_siblings']} pxD={made['px_sep_dests']}")
        else:
            print(f"[FAIL] scene_{idx:04d} {task_key} {tier} N{n_target}/D{n_dest}")
        idx += 1

    json.dump(manifest, open(os.path.join(OUT_ROOT, "manifest.json"), "w"), indent=2)
    from collections import Counter
    print(f"\nMODE VSLICE={VSLICE} SMOKE={SMOKE}: {len(manifest)} kept, "
          f"{len(fails)} rejections")
    for why, c in Counter(w.split("_")[0] for w in fails).most_common():
        print(f"   reject[{why}] x{c}")
    for e in manifest:
        print(f"  {e['dir']} {e['task']:20s} {e['tier']:11s} "
              f"pred={e['predicate']} dest_kind={e['dest_kind']} "
              f"vis={e['visibility']['visibility']:.2f} "
              f"grasp={e['grasp']['grasp_success']} lift={e['grasp']['lift']:.3f}")
    if VSLICE:
        print("\nVSLICE done. Paste this whole log back. Confirms: original-BDDL "
              "parse (A1), region-oracle semantics (A2), fixture rects (A3). Then "
              "flip VSLICE=False (SMOKE first) for the tiered run.")


if __name__ == "__main__":
    main()
