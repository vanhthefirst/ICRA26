"""
Sketch-Prompted VLA validation set — LIBERO-SPATIAL, layout-anchored builder
(needs the libero env: `conda activate libero`).

Replaces the from-scratch Spatial builder for the paper's evaluation set. The
from-scratch builder samples EVERY object's position inside a table rectangle,
so its scenes are layouts the fine-tuned policy has never seen. Whatever such a
scene loses against the standard suites mixes two causes: the caption cannot
resolve the target (the effect I want) and the layout is out of distribution
(the effect I do not). This builder removes the second cause by construction:
it reads a shipped `libero_spatial` BDDL, keeps every object, fixture, region
and `(:init)` line verbatim, and only ADDS duplicate instances. The stock
objects never move.

Same parse-and-inject pattern as `build_validation_set_goal.py`, with two
Spatial-specific departures.

DEPARTURE 1 — region naming. Goal's object init regions are `<category>_region`,
so that builder finds an object's placement by category. Spatial's are semantic
(`between_plate_ramekin_region`, `next_to_box_region`, `table_center`), so a
category lookup finds nothing. I resolve every object's placement through its
`(:init)` line instead — `(On akita_black_bowl_1 main_table_X)` -> region `X` —
which is how LIBERO itself decides where the object goes and therefore cannot
diverge from it.

DEPARTURE 2 — `akita_black_bowl_2`. Every shipped `libero_spatial` task carries
TWO black bowls, and the goal is always `(On akita_black_bowl_1 plate_1)`. The
stock caption resolves that by spatial phrase ("between the plate and the
ramekin"); my caption names the category only, so bowl_2 stays visually
indistinguishable from the target. Per the decision recorded in
`claude/eval_layout_anchoring.md`, bowl_2 is kept in place and declared OUT OF
FOCUS: it is not counted as a tier candidate, it is named explicitly in
`meta['out_of_focus']` so no downstream reader has to infer it, and the sketch
is still held responsible for excluding it — the drawn circle must not enclose
its centre. It gets that weaker criterion rather than the strict pixel gate
because its position is stock and a failure cannot be answered by resampling;
see the two-criteria note at the sibling gate.

The consequence is recorded in that same file and is not a bug: `control x
explicit` on this suite is not a language-only ceiling, because two identical
bowls are present and the caption names a category. The ceiling for these
layouts is the stock-caption number already measured in
`outputs/rollouts/openpi_repro_500/` (spatial 98.0), and the gap between that
number and `control x explicit` is what the spatial phrase was buying.

BASE ROSTER. Of the 10 shipped tasks I use the 5 whose bowl_1 starts on a
`main_table` region. Dropped: `in_the_top_drawer_of_the_wooden_cabinet`
(bowl_1 is `In` a CLOSED drawer — the retracted region site defeats the
teleport oracle, the same reason the Goal builder dropped
`open_the_top_drawer_and_put_the_bowl_inside`). Held back, buildable but not in
the 38: `on_the_cookie_box`, `on_the_ramekin`, `on_the_stove`,
`on_the_wooden_cabinet` — bowl_1 starts stacked or on a fixture, which makes
the circle geometry and the scripted grasp behave differently enough that I
would want them measured as their own group rather than mixed in.

GRASP IS RECORDED, NOT GATED — unlike the from-scratch builder. There, a failed
scripted grasp was answered by resampling the layout; here the layout is fixed,
so gating on grasp would silently delete stock base tasks. Same rule the Goal
builder already follows (SUITE_FACTS 3b.5).

Emits canonical schema v1.0 (SCHEMA.md) into `outputs/validation_set_spatial/`,
the name every downstream script hardcodes. Move the from-scratch set aside
first; it is not overwritten in place.

    # pod: the LIBERO venv, not conda — see RUNBOOK_BASELINES.md part B
    source $OPENPI/examples/libero/.venv/bin/activate
    export PYTHONPATH=$PYTHONPATH:$OPENPI/third_party/libero
    export MUJOCO_GL=egl
    export LIBERO_SPATIAL_BDDL=$OPENPI/third_party/libero/libero/libero/bddl_files/libero_spatial
    cd $REPO
    mv outputs/validation_set_spatial outputs/validation_set_spatial_fromscratch
    mkdir -p outputs/validation_set_spatial
    python scripts/build_validation_set_spatial_anchored.py 2>&1 \
        | tee outputs/validation_set_spatial/build_log.txt
    python scripts/normalize_validation_schema.py

SMOKE=True -> one scene per tier, plus the per-category projected extents used
to set spacing. DUMP=True -> parsed structure of each base BDDL.
"""

import os, re, json, gc, copy
import numpy as np
import cv2

SMOKE = False
DUMP = False

BDDL_ROOT = os.environ.get(
    "LIBERO_SPATIAL_BDDL", "/root/LIBERO/libero/libero/bddl_files/libero_spatial")
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Defaults to the directory every downstream script already hardcodes
# (`capture_scene_init_states.py`, `normalize_validation_schema.py`,
# `build_prompt_variants.py`, `audit_validation_sets.py`, `rollout_sketch.py`,
# `analyze_baselines.py`, and `sketch_eval_common.py` in the eval repo all name
# `validation_set_spatial` literally). Emitting anywhere else means patching
# seven files across two repos, so the from-scratch set gets renamed out of the
# way instead and this one takes the name. `SPATIAL_SET=` overrides for a
# side-by-side build.
OUT_ROOT = os.path.join(_REPO, "outputs",
                        os.environ.get("SPATIAL_SET", "validation_set_spatial"))

IMG_H = IMG_W = 128
CAMERA = "agentview"
ADIM = 7
TABLE_Z_MIN = 0.85
HALF_BOX = 0.012                  # injected placement region half-size, world m
HALF_OBJ = 0.045                  # footprint half-size for rect-avoidance
VIS_MIN = 0.35
DIFF_TH = 30
PX_MARGIN = 3
PX_MARGIN_PLATE = 4
ELLIPSE_MIN = 1.25                # sibling centre, in drawn-circle radii

TARGET_CAT = "akita_black_bowl"
DEST_CAT = "plate"
TARGET = f"{TARGET_CAT}_1"        # LIBERO's own goal instance in all 10 tasks
OUT_OF_FOCUS = f"{TARGET_CAT}_2"  # stock second bowl — kept, not counted

# Same three captions the from-scratch builder used, so the ambiguous-caption
# bank in build_prompt_variants.py still keys them to bucket `two_clause_On`.
CAPTIONS = ["pick up the black bowl and place it on the plate",
            "pick up the bowl and put it on the plate",
            "grab the black bowl and place it on the plate"]

# Same-category spacing for INJECTED copies (world m), from the from-scratch
# builder's measured values: bowls carry a large drawn circle and need the most
# room, plates must additionally clear On's 3 cm xy rule.
SIB_SPACING = {TARGET_CAT: 0.165, DEST_CAT: 0.150}
D_CROSS = 0.095

# Reachable table band, from the from-scratch builder. The sampling rectangle is
# the stock object regions' bounding box inflated by 0.10 and then intersected
# with this. The intersection is what keeps duplicates off the cabinet (y~-0.27)
# and the stove (y~-0.14): both declare only a 2 cm init rectangle, far smaller
# than their real footprint, so rect-avoidance alone would not clear them
# (the Goal builder hit the same limitation, probe note A3).
WORKSPACE_X = (-0.22, 0.20)
WORKSPACE_Y = (-0.06, 0.32)

BASE_TASKS = {
    "between_plate_ramekin": "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl",
    "table_center": "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl",
    "next_to_plate": "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate.bddl",
    "next_to_ramekin": "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate.bddl",
    "next_to_cookie_box": "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate.bddl",
}
BASE_ORDER = list(BASE_TASKS)


def tier_specs(smoke):
    """(task_key, tier, n_target_infocus, n_dest).

    n_target_infocus counts CANDIDATES the caption's category words match and I
    hold the sketch responsible for resolving: the stock bowl_1 plus injected
    copies. The stock bowl_2 is physically present in every scene and excluded
    from this count — see DEPARTURE 2."""
    if smoke:
        return [("between_plate_ramekin", "control", 1, 1),
                ("table_center", "referential", 3, 1),
                ("next_to_plate", "directional", 1, 3),
                ("next_to_ramekin", "both", 3, 2)]
    B = BASE_ORDER
    specs = [(t, "control", 1, 1) for t in B]                              # 5
    ref = [2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3]
    specs += [(B[i % 5], "referential", n, 1) for i, n in enumerate(ref)]  # 12
    dirn = [2, 3, 2, 3, 2, 3, 2, 3, 2]
    specs += [(B[i % 5], "directional", 1, m) for i, m in enumerate(dirn)]  # 9
    both = [(2, 2), (3, 2), (2, 3), (3, 3), (2, 2), (3, 2),
            (2, 3), (3, 3), (2, 2), (3, 2), (2, 3), (3, 2)]
    specs += [(B[i % 5], "both", n, m) for i, (n, m) in enumerate(both)]   # 12
    return specs


# ======================================================== shared sim helpers ===
def safe_imwrite(path, img):
    ok = cv2.imwrite(path, img)
    n = os.path.getsize(path) if os.path.exists(path) else 0
    tail = False
    if n:
        with open(path, "rb") as f:
            f.seek(-8, 2); tail = b"IEND" in f.read()
    back = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if not (ok and n > 0 and tail and back is not None
            and back.shape[:2] == img.shape[:2]):
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
    ref = [g for g in gids
           if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
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


def rgb(env):
    return env.sim.render(width=IMG_W, height=IMG_H,
                          camera_name=CAMERA).astype(np.int32)


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
    """Rest `bowl` on `plate`, return check_success(). `On` wants the bowl above
    the plate, in contact, within 3 cm xy; dropping it just over the plate's
    visual centre settles into that."""
    pc = vcenter(model, data, bid_of(model, plate))
    j = jnt_of(model, bowl)
    if j is None:
        return None
    qa = model.jnt_qposadr[j]; va = model.jnt_dofadr[j]
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


# ============================================================ sketch drawing ===
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


def ellipse_norm(tok, p):
    """Where `p` sits relative to the circle actually drawn, in units of that
    circle's radius along p's direction: <1 inside, 1 on the stroke. Computed
    from the emitted `rx`/`ry` tokens rather than the requested radius, because
    the drawer jitters both centre and radii — checking the request instead of
    the stroke is how a sibling ends up inside a circle that "passed"."""
    dx = (p[0] - tok["cx"]) / max(tok["rx"], 1e-6)
    dy = (p[1] - tok["cy"]) / max(tok["ry"], 1e-6)
    return float(np.hypot(dx, dy))


def draw_arrow(img, p, q, seed):
    rng = np.random.default_rng(seed + 1); out = img.copy()
    x1 = p[0] + int(rng.integers(-2, 3)); y1 = p[1] + int(rng.integers(-2, 3))
    x2 = q[0] + int(rng.integers(-2, 3)); y2 = q[1] + int(rng.integers(-2, 3))
    mx = (x1 + x2) // 2 + int(rng.integers(-7, 8))
    my = (y1 + y2) // 2 + int(rng.integers(-7, 8))
    th = int(rng.integers(1, 3))
    cv2.line(out, (x1, y1), (mx, my), (200, 50, 50), th, cv2.LINE_AA)
    cv2.arrowedLine(out, (mx, my), (x2, y2), (200, 50, 50), th, cv2.LINE_AA,
                    tipLength=0.35)
    return out, {"x0": x1, "y0": y1, "x1": x2, "y1": y2}


# =========================================== stock-BDDL parsing and injection ===
def _balanced(txt, start):
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
    i = txt.find(f"(:{tag}")
    return _balanced(txt, i) if i >= 0 else ""


def _block_inner(block, tag):
    """Contents of a `(:tag ...)` block with the wrapper stripped, so
    `_top_sexprs` returns one entry per member rather than the block itself."""
    if not block:
        return ""
    head = block.find(f":{tag}")
    first = block.find("(", head + len(tag) + 1) if head >= 0 else -1
    return block[first:block.rfind(")")] if first >= 0 else ""


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
    """Reproduce LIBERO's `robosuite_parse_problem` :objects/:fixtures loop
    exactly: tokenise, walk `name... - category`, ASSIGN the accumulated names to
    that category. Assignment, not appending, is why every instance of a category
    must live on one line — a second `cat_2 - cat` line deletes the first."""
    if not block:
        return {}
    i = block.find(head)
    inner = block[i + len(head):]
    inner = inner[:inner.rfind(")")]
    toks = inner.replace("(", " ").replace(")", " ").split()
    out = {}; acc = []
    while toks:
        if toks[0] == "-":
            toks.pop(0)
            out[toks.pop(0)] = acc
            acc = []
        else:
            acc.append(toks.pop(0))
    return out


def _init_states(text):
    blk = _tag_block(text, "init")
    return re.findall(
        r"\(([A-Za-z]+)\s+([A-Za-z0-9_]+)(?:\s+([A-Za-z0-9_]+))?\)",
        " ".join(blk.split()))


def parse_stock(path):
    txt = open(path).read()
    lang = re.search(r"\(:language\s+(.*?)\)", txt, re.S)
    lang = " ".join(lang.group(1).split()) if lang else ""

    obj_block = _tag_block(txt, "objects")
    fx_block = _tag_block(txt, "fixtures")
    objs_by_cat = _libero_typed(obj_block, ":objects")
    fixtures_by_cat = _libero_typed(fx_block, ":fixtures")
    objs = {i: c for c, insts in objs_by_cat.items() for i in insts}
    fixtures = {i: c for c, insts in fixtures_by_cat.items() for i in insts}

    reg_block = _tag_block(txt, "regions")
    regions = {}
    head = reg_block.find(":regions")
    first = reg_block.find("(", head + len(":regions")) if head >= 0 else -1
    inner = reg_block[first:reg_block.rfind(")")] if first >= 0 else ""
    for sx in _top_sexprs(inner):
        m = re.match(r"\(\s*([A-Za-z0-9_]+)", sx)
        if not m or m.group(1) in (":target", ":ranges", ":yaw", ":yaw_rotation"):
            continue
        name = m.group(1)
        tgt = re.search(r"\(:target\s+([A-Za-z0-9_]+)\)", sx)
        ranges = []
        ri = sx.find("(:ranges")
        if ri >= 0:
            nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", _balanced(sx, ri))]
            ranges = [tuple(nums[i:i + 4]) for i in range(0, len(nums) - 3, 4)]
        regions[name] = dict(target=tgt.group(1) if tgt else None, ranges=ranges)

    return dict(text=txt, lang=lang, objs=objs, objs_by_cat=objs_by_cat,
                fixtures=fixtures, fixtures_by_cat=fixtures_by_cat,
                regions=regions, obj_block=obj_block, reg_block=reg_block,
                init_block=_tag_block(txt, "init"),
                ooi_block=_tag_block(txt, "obj_of_interest"),
                goal_block=_tag_block(txt, "goal"))


def init_region_of(parsed, inst):
    """Region an object is placed on, read from `(:init)`.

    Spatial's region names are semantic, so the Goal builder's `<category>_region`
    lookup finds nothing here. `(:init)` is what LIBERO itself consumes, so
    resolving through it cannot disagree with where the object actually goes.
    Returns (region_name, (cx, cy)) or (region_name, None) when the region
    carries no rectangle — an object placed on a fixture affordance site
    (`flat_stove_1_cook_region`) or stacked on another object has no table-plane
    centre to reason about."""
    for (pred, a, b) in _init_states(parsed["text"]):
        if a != inst or not b:
            continue
        for owner in ["main_table"] + list(parsed["fixtures"]):
            if b.startswith(owner + "_"):
                rn = b[len(owner) + 1:]
                r = parsed["regions"].get(rn)
                if r and r["ranges"]:
                    x0, y0, x1, y1 = r["ranges"][0]
                    return rn, ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
                return rn, None
        r = parsed["regions"].get(b)
        if r and r["ranges"]:
            x0, y0, x1, y1 = r["ranges"][0]
            return b, ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        return b, None            # `(On bowl_1 cookies_1)` — stacked on an object
    return None, None


def occupied_rects(parsed):
    rects = []
    for r in parsed["regions"].values():
        for (x0, y0, x1, y1) in r["ranges"]:
            rects.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
    return rects


def _rect_of(cx, cy, half):
    return (cx - half, cy - half, cx + half, cy + half)


def _overlap(a, b, pad=0.0):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or
                a[3] + pad < b[1] or b[3] + pad < a[1])


def sampling_rect(parsed):
    """Bounding box of the stock MOVABLE objects' placement rectangles, inflated,
    then intersected with the reachable band. The intersection is doing real
    work — see the WORKSPACE_* comment."""
    rects = []
    for inst in parsed["objs"]:
        rn, _ = init_region_of(parsed, inst)
        r = parsed["regions"].get(rn) if rn else None
        if r:
            rects += r["ranges"]
    if not rects:
        return WORKSPACE_X, WORKSPACE_Y
    xs = [x for rr in rects for x in (rr[0], rr[2])]
    ys = [y for rr in rects for y in (rr[1], rr[3])]
    rx = (max(min(xs) - 0.10, WORKSPACE_X[0]), min(max(xs) + 0.10, WORKSPACE_X[1]))
    ry = (max(min(ys) - 0.10, WORKSPACE_Y[0]), min(max(ys) + 0.10, WORKSPACE_Y[1]))
    return rx, ry


def sample_duplicates(parsed, n_extra_target, n_extra_dest, rng,
                      restarts=500, tries=800):
    """Positions for injected copies only. Every stock object with a table-plane
    centre seeds the occupancy list, so a duplicate keeps its distance from the
    originals as well as from other duplicates. Restarts rather than more tries:
    a doomed prefix cannot be repaired (SUITE_FACTS 10)."""
    occ = occupied_rects(parsed)
    RX, RY = sampling_rect(parsed)
    seeds = []
    for inst, cat in parsed["objs"].items():
        _, xy = init_region_of(parsed, inst)
        if xy is not None:
            seeds.append((xy[0], xy[1], cat))
    want = [TARGET_CAT] * n_extra_target + [DEST_CAT] * n_extra_dest
    if not want:
        return [], []
    for _ in range(restarts):
        placed = list(seeds)
        got_all = True
        out = []
        for cat in want:
            got = False
            for _ in range(tries):
                x = float(rng.uniform(*RX)); y = float(rng.uniform(*RY))
                sp = SIB_SPACING.get(cat, D_CROSS)
                if any(np.hypot(x - a, y - b) < (sp if c == cat else D_CROSS)
                       for a, b, c in placed):
                    continue
                if any(_overlap(_rect_of(x, y, HALF_OBJ), o, pad=0.01) for o in occ):
                    continue
                placed.append((x, y, cat)); out.append((cat, x, y)); got = True
                break
            if not got:
                got_all = False; break
        if got_all:
            return ([(x, y) for c, x, y in out if c == TARGET_CAT],
                    [(x, y) for c, x, y in out if c == DEST_CAT])
    raise RuntimeError(f"placement failed after {restarts} restarts "
                       f"(+{n_extra_target} bowls, +{n_extra_dest} plates)")


def _region_text(name, cx, cy):
    return (f"      ({name}\n          (:target main_table)\n          (:ranges (\n"
            f"              ({cx-HALF_BOX:.4f} {cy-HALF_BOX:.4f} "
            f"{cx+HALF_BOX:.4f} {cy+HALF_BOX:.4f})\n            )\n          )\n      )")


def _inject_before_close(text, block, new_lines):
    if not new_lines or not block:
        return text
    idx = text.find(block)
    if idx < 0:
        return text
    close = block.rfind(")")
    body = block[:close].rstrip()
    indent = block[:close][len(block[:close].rstrip("\t ")):]
    injected = body + "\n" + "\n".join(new_lines) + "\n" + indent + block[close:]
    return text[:idx] + injected + text[idx + len(block):]


def _replace_block(text, block, replacement):
    idx = text.find(block)
    return text if idx < 0 else text[:idx] + replacement + text[idx + len(block):]


def verify_injected_bddl(text, must_declare):
    """Refuse to hand LIBERO a BDDL it will choke on. Reparses the FINAL text the
    way LIBERO will and asserts every object placed in `(:init)` and every
    instance the goal names is declared, turning a deep KeyError inside env
    construction into a local, readable failure."""
    objs = _libero_typed(_tag_block(text, "objects"), ":objects")
    fixts = _libero_typed(_tag_block(text, "fixtures"), ":fixtures")
    declared = {i for v in objs.values() for i in v} | {i for v in fixts.values() for i in v}
    for cat, insts in objs.items():
        if len(insts) != len(set(insts)):
            raise ValueError(f"category '{cat}' has duplicate instances: {insts}")
    for (p, a, b) in _init_states(text):
        if p.lower() in ("on", "in") and a not in declared:
            raise ValueError(f"(:init) places undeclared object '{a}'; "
                             f"declared={sorted(declared)}")
    for name in must_declare:
        if name not in declared:
            raise ValueError(f"'{name}' vanished from (:objects) — the "
                             f"category-assign hazard bit again")
    return declared


def verify_stock_preserved(stock_text, new_text):
    """The claim this whole builder exists to support: no stock object moved.

    Every `(:init)` line and every region s-expression of the shipped file must
    still be present verbatim. Injection only ever appends, so anything that
    fails here is a bug in the injection, not a judgement call."""
    flat = " ".join(new_text.split())
    for sx in _top_sexprs(_block_inner(_tag_block(stock_text, "init"), "init")):
        if " ".join(sx.split()) not in flat:
            raise ValueError(f"stock (:init) line lost or altered: "
                             f"{' '.join(sx.split())}")
    for sx in _top_sexprs(_block_inner(_tag_block(stock_text, "regions"), "regions")):
        if " ".join(sx.split()) not in flat:
            raise ValueError(f"stock region lost or altered: "
                             f"{' '.join(sx.split())[:60]}")
    return True


def build_bddl(task_key, tier, n_infocus, n_dest, dest_idx, caption, seed):
    """Stock scene verbatim, plus duplicate instances, plus a retargeted goal."""
    path = os.path.join(BDDL_ROOT, BASE_TASKS[task_key])
    parsed = parse_stock(path)
    rng = np.random.default_rng(seed)

    if parsed["objs_by_cat"].get(TARGET_CAT, [])[:2] != [TARGET, OUT_OF_FOCUS]:
        raise ValueError(f"{task_key}: expected stock bowls [{TARGET}, "
                         f"{OUT_OF_FOCUS}], got "
                         f"{parsed['objs_by_cat'].get(TARGET_CAT)}")

    tdup, ddup = sample_duplicates(parsed, n_infocus - 1, n_dest - 1, rng)

    cat_insts = copy.deepcopy(parsed["objs_by_cat"])
    dup_regions, dup_inits, siblings, dest_dups = [], [], [], []
    # Stock occupies _1 and _2 for bowls and _1 for plates, so injected instances
    # start at _3 and _2 respectively.
    for k, (x, y) in enumerate(tdup, start=3):
        nm = f"{TARGET_CAT}_{k}"; siblings.append(nm)
        cat_insts[TARGET_CAT].append(nm)
        rn = f"{nm}_dup"
        dup_regions.append(_region_text(rn, x, y))
        dup_inits.append(f"    (On {nm} main_table_{rn})")
    for k, (x, y) in enumerate(ddup, start=2):
        nm = f"{DEST_CAT}_{k}"; dest_dups.append(nm)
        cat_insts[DEST_CAT].append(nm)
        rn = f"{nm}_dup"
        dup_regions.append(_region_text(rn, x, y))
        dup_inits.append(f"    (On {nm} main_table_{rn})")

    dest = f"{DEST_CAT}_{dest_idx}"
    other_dests = [f"{DEST_CAT}_1"] + dest_dups
    other_dests = [d for d in other_dests if d != dest]

    obj_lines = "\n".join("    " + " ".join(v) + f" - {c}"
                          for c, v in cat_insts.items())
    text = parsed["text"]
    text = _replace_block(text, parsed["obj_block"], f"(:objects\n{obj_lines}\n  )")
    text = _inject_before_close(text, parsed["reg_block"], dup_regions)
    text = _inject_before_close(text, _tag_block(text, "init"), dup_inits)
    text = _replace_block(text, _tag_block(text, "obj_of_interest"),
                          f"(:obj_of_interest\n    {TARGET}\n    {dest}\n  )")
    text = _replace_block(text, _tag_block(text, "goal"),
                          f"(:goal\n    (And (On {TARGET} {dest}))\n  )")
    text = re.sub(r"\(:language\s+.*?\)", f"(:language {caption})", text,
                  count=1, flags=re.S)

    verify_injected_bddl(text, [TARGET, OUT_OF_FOCUS, dest] + siblings + dest_dups)
    verify_stock_preserved(parsed["text"], text)

    meta = dict(
        task=task_key, base_file=BASE_TASKS[task_key], tier=tier,
        target=TARGET, destination=dest, destination_region=dest,
        goal_predicate="On", instruction=caption,
        stock_language=parsed["lang"], stock_goal_destination=f"{DEST_CAT}_1",
        out_of_focus=[OUT_OF_FOCUS],
        siblings=siblings, other_plates=other_dests,
        n_target_infocus=n_infocus, n_target_physical=n_infocus + 1,
        n_dest=n_dest, seed=seed,
        instances=[i for v in cat_insts.values() for i in v],
        dup_target_xy=tdup, dup_dest_xy=ddup,
        anchored=True)
    return text, meta, parsed


# ================================================================= one scene ===
def build_scene(task_key, tier, n_infocus, n_dest, seed, scene_dir, dump_ext=False):
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils import camera_utils as CU

    rng = np.random.default_rng(seed)
    caption = CAPTIONS[int(rng.integers(0, len(CAPTIONS)))]
    dest_idx = int(rng.integers(1, n_dest + 1))

    bddl, meta, parsed = build_bddl(task_key, tier, n_infocus, n_dest,
                                    dest_idx, caption, seed)
    if DUMP:
        print(f"\n  [{task_key}] stock lang: {parsed['lang']}")
        print(f"  [{task_key}] objects   : {parsed['objs']}")
        for n, r in parsed["regions"].items():
            print(f"      {n:34s} target={r['target']} ranges={r['ranges']}")

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
        tgt, dest = meta["target"], meta["destination"]
        instances = meta["instances"]

        # The circle has to be readable against the stock second bowl exactly as
        # it does against an injected one, so bowl_2 is gated as a sibling even
        # though it is not a tier candidate.
        gate_sibs = meta["siblings"] + meta["out_of_focus"]

        for i in instances:
            try:
                if float(vcenter(model, data, bid_of(model, i))[2]) < TABLE_Z_MIN:
                    return None, f"fell_{i}"
            except KeyError:
                return None, f"missing_body_{i}"

        pix = {i: project([vcenter(model, data, bid_of(model, i))], W2P)[0]
               for i in instances}
        ext = {i: px_extent(model, data, W2P, i) for i in instances}
        if dump_ext:
            byc = {}
            for i in instances:
                byc.setdefault(i.rsplit("_", 1)[0], []).append(ext[i])
            print("   [ext px] " + "  ".join(f"{c}:{np.mean(v):.1f}"
                                             for c, v in byc.items()))

        def inframe(p, e=0.0):
            m = int(np.ceil(e)) + 2
            return m <= p[0] <= IMG_W - 1 - m and m <= p[1] <= IMG_H - 1 - m

        if not inframe(pix[tgt], ext[tgt]):
            return None, "target_off_frame"
        for p in meta["other_plates"] + [dest]:
            if not inframe(pix[p], ext[p]):
                return None, f"{p}_off_frame"

        gids = [g for g in range(model.ngeom)
                if model.geom_bodyid[g] == bid_of(model, tgt)]
        gpx = project([data.geom_xpos[g].copy() for g in gids], W2P)
        cs_ = [p[0] for p in gpx]; rs_ = [p[1] for p in gpx]
        radius = max(5, min(int(np.hypot(max(cs_) - min(cs_),
                                         max(rs_) - min(rs_)) / 2 * 1.5), 40))

        px_plate = px_plate_req = None
        if meta["other_plates"]:
            px_plate_req = ext[dest] + PX_MARGIN_PLATE
            px_plate = min(float(np.hypot(pix[dest][0] - pix[p][0],
                                          pix[dest][1] - pix[p][1]))
                           for p in meta["other_plates"])
            if px_plate < px_plate_req:
                return None, f"plates_overlap_{px_plate:.0f}px_need{px_plate_req:.0f}"

        img_c, tok_c = draw_circle(frame, pix[tgt], radius, seed)

        # Two separation criteria, because I control only one of the two kinds of
        # sibling. INJECTED copies get the strict by-construction gate — if they
        # sit too close I resample them. The stock `akita_black_bowl_2` cannot be
        # moved without defeating the point of this builder, so it gets the
        # weaker criterion the repo already settled on for the case where
        # geometry is not mine to choose (SUITE_FACTS 4): its centre must fall
        # clear of the circle actually drawn. ELLIPSE_MIN leaves room for the
        # drawer's wobble, which the tokens do not capture.
        px_sib = px_sib_req = None
        if meta["siblings"]:
            s_worst = min(meta["siblings"],
                          key=lambda s: float(np.hypot(pix[tgt][0] - pix[s][0],
                                                       pix[tgt][1] - pix[s][1])) - ext[s])
            px_sib = float(np.hypot(pix[tgt][0] - pix[s_worst][0],
                                    pix[tgt][1] - pix[s_worst][1]))
            px_sib_req = radius + 0.5 * ext[s_worst] + PX_MARGIN
            if px_sib < px_sib_req:
                return None, (f"siblings_overlap_{px_sib:.0f}px_"
                              f"need{px_sib_req:.0f}_r{radius}_{s_worst}")

        enorm = {s: round(ellipse_norm(tok_c, pix[s]), 3) for s in gate_sibs}
        e_worst = min(enorm, key=enorm.get)
        if enorm[e_worst] < ELLIPSE_MIN:
            return None, (f"circleenclosed_{e_worst}_norm{enorm[e_worst]:.2f}"
                          f"_need{ELLIPSE_MIN}")

        vis, vismask = visibility(env, model, data, instances, tgt)
        if vis["visibility"] < VIS_MIN:
            return None, f"low_vis_{vis['visibility']}"
        if (vismask[0, :].any() or vismask[-1, :].any()
                or vismask[:, 0].any() or vismask[:, -1].any()):
            return None, "target_clipped_by_frame"

        if success(env) is not False:
            return None, "pre_solved"

        if teleport_on(env, model, data, tgt, dest) is not True:
            return None, "oracle_false"

        neg = {}
        for p in meta["other_plates"]:
            settle(env); model, data = env.sim.model, env.sim.data
            r = teleport_on(env, model, data, tgt, p)
            neg[f"{tgt}->{p}"] = r
            if r is not False:
                return None, f"neg_dest_true_{p}"
        for s in gate_sibs:
            settle(env); model, data = env.sim.model, env.sim.data
            r = teleport_on(env, model, data, s, dest)
            neg[f"{s}->{dest}"] = r
            if r is not False:
                return None, f"neg_obj_true_{s}"

        np.random.seed(seed); obs = settle(env)
        model, data = env.sim.model, env.sim.data
        grasp = scripted_grasp(env, model, data, tgt, obs)

        others = [i for i in instances if i != tgt]
        tp = vcenter(model, data, bid_of(model, tgt))[:2]
        clr = min(float(np.linalg.norm(
            vcenter(model, data, bid_of(model, o))[:2] - tp)) for o in others) \
            if others else None

        img_ca, tok_a = draw_arrow(img_c, pix[tgt], pix[dest], seed)
        safe_imwrite(os.path.join(scene_dir, "frame0.png"),
                     cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        safe_imwrite(os.path.join(scene_dir, "sketch.png"),
                     cv2.cvtColor(img_ca, cv2.COLOR_RGB2BGR))
        safe_imwrite(os.path.join(scene_dir, "target_vismask.png"), vismask)

        meta.update(dict(
            schema_version="1.0", suite="spatial",
            pick_px=list(pix[tgt]), place_px=list(pix[dest]), radius=radius,
            all_pixels={k: list(v) for k, v in pix.items()},
            px_extent={k: round(v, 2) for k, v in ext.items()},
            symbolic_tokens={"circle": tok_c, "arrow": tok_a},
            camera_matrix=W2P.tolist(), oracle_success=True, oracle_negatives=neg,
            px_sep_plates=px_plate, px_req_plates=px_plate_req,
            px_sep_siblings=px_sib, px_req_siblings=px_sib_req,
            ellipse_norm=enorm, ellipse_norm_min=ELLIPSE_MIN,
            clearance_xy=round(clr, 3) if clr else None,
            visibility=vis, grasp=grasp,
            target_bowl=tgt, target_plate=dest))
        json.dump(meta, open(os.path.join(scene_dir, "meta.json"), "w"), indent=2)
        json.dump({"instruction": caption, "suite": "spatial", "tier": tier,
                   "target": tgt, "destination": dest, "destination_region": dest,
                   "goal_predicate": "On", "out_of_focus": meta["out_of_focus"],
                   "symbolic_tokens": meta["symbolic_tokens"],
                   "target_bowl": tgt, "target_plate": dest},
                  open(os.path.join(scene_dir, "tokens.json"), "w"), indent=2)
        return meta, "ok"
    finally:
        try:
            env.close()
        except Exception:
            pass
        gc.collect()


# ====================================================================== main ===
def write_datasheet(manifest, fails):
    from collections import Counter
    tiers = Counter(e["tier"] for e in manifest)
    vis = [e["visibility"]["visibility"] for e in manifest]
    lift = [e["grasp"]["lift"] for e in manifest]
    nograsp = [e["dir"] for e in manifest if not e["grasp"]["grasp_success"]]

    def stat(v, f="{:.3f}"):
        return "n/a" if not v else (f + " / " + f + " / " + f).format(
            min(v), float(np.mean(v)), max(v))

    md = f"""# Sketch-Prompted VLA validation set — LIBERO-**Spatial**, layout-anchored

{len(manifest)} scenes. `scripts/build_validation_set_spatial_anchored.py`
(SMOKE={SMOKE}). Canonical schema v1.0 (SCHEMA.md).

Every scene is a shipped `libero_spatial` BDDL with duplicate instances added
and nothing else changed: same objects, same placement regions, same `(:init)`.
`verify_stock_preserved()` asserts this on the emitted text, so the claim is
checked per scene rather than asserted. What differs from stock is the caption
(category-explicit instead of the stock spatial phrase), the goal's destination
instance in the directional and both tiers, and the added copies.

## Base tasks

{chr(10).join('- `' + k + '` -> ' + v for k, v in BASE_TASKS.items())}

Dropped: `in_the_top_drawer_of_the_wooden_cabinet` (bowl_1 starts `In` a closed
drawer). Held back for separate measurement: `on_the_cookie_box`,
`on_the_ramekin`, `on_the_stove`, `on_the_wooden_cabinet` (bowl_1 starts stacked
or on a fixture).

## Composition

| tier | scenes | in-focus bowls | physical bowls | plates |
|---|---|---|---|---|
| control | {tiers.get('control',0)} | 1 | 2 | 1 |
| referential | {tiers.get('referential',0)} | 2-3 | 3-4 | 1 |
| directional | {tiers.get('directional',0)} | 1 | 2 | 2-3 |
| both | {tiers.get('both',0)} | 2-3 | 3-4 | 2-3 |

`akita_black_bowl_2` ships in every stock task and stays where stock puts it. It
is out of focus: not a tier candidate, but gated like a sibling and named in
`meta['out_of_focus']`. Read `control` here as "one candidate the sketch is
responsible for", not as a language-only ceiling — the ceiling for these layouts
is the stock-caption number in `outputs/rollouts/openpi_repro_500/`.

## Gates

settled · target and plates fully in frame · strict pixel separation for
INJECTED siblings · drawn-circle clearance (centre outside {ELLIPSE_MIN}x the
stroke) for every sibling INCLUDING `akita_black_bowl_2` · arrow vs rival plates
· visibility >= {VIS_MIN} · not pre-solved · positive oracle (bowl rests On the
destination plate) · negative oracles both axes.

Two separation criteria because only one kind of sibling is mine to move. An
injected copy that crowds the target is resampled, so it gets the strict gate.
`akita_black_bowl_2` is stock and fixed, so it gets the criterion the repo
settled on for geometry I do not choose: its centre must fall clear of the
circle actually drawn, measured from the emitted `rx`/`ry` tokens.

Grasp is recorded, not gated: the layout is stock and cannot be resampled, so
gating on the scripted grasp would delete base tasks rather than improve them.
{len(nograsp)} scene(s) ship `grasp_success=False`{': ' + ', '.join(nograsp) if nograsp else ''}.

## Measured (min / mean / max)

| metric | value |
|---|---|
| visibility | {stat(vis)} |
| grasp lift (m) | {stat(lift)} |

Rejections: {len(fails)} ({', '.join(f'{k} x{v}' for k, v in Counter(w.split('_')[0] for w in fails).most_common())}).
"""
    open(os.path.join(OUT_ROOT, "DATASHEET.md"), "w").write(md)


def main():
    only = os.environ.get("ONLY_SCENES")
    only = {int(x) for x in only.split(",")} if only else None
    os.makedirs(OUT_ROOT, exist_ok=True)
    specs = tier_specs(SMOKE)
    print(f"MODE: SMOKE={SMOKE} DUMP={DUMP}  BDDL_ROOT={BDDL_ROOT}")
    print(f"  {len(specs)} scene(s) planned over {len(BASE_TASKS)} base tasks.")

    manifest = []; fails = []
    for idx, (task_key, tier, n_infocus, n_dest) in enumerate(specs):
        if only is not None and idx not in only:
            continue
        made = None
        for attempt in range(24):
            seed = 7000 + idx * 100 + attempt
            sd = os.path.join(OUT_ROOT, f"scene_{idx:04d}")
            try:
                made, why = build_scene(task_key, tier, n_infocus, n_dest, seed, sd,
                                        dump_ext=(SMOKE and attempt == 0))
            except Exception as e:
                made, why = None, f"error:{type(e).__name__}:{e}"
            print(f"  scene_{idx:04d} {task_key:22s} {tier:11s} "
                  f"N{n_infocus}/M{n_dest} seed={seed} -> {why}")
            if made:
                break
            fails.append(why)
        if made:
            made["dir"] = f"scene_{idx:04d}"
            manifest.append({k: made.get(k) for k in
                             ("dir", "task", "tier", "target", "destination",
                              "out_of_focus", "siblings", "other_plates",
                              "n_target_infocus", "n_target_physical", "n_dest",
                              "seed", "instruction", "clearance_xy",
                              "px_sep_plates", "px_sep_siblings", "ellipse_norm",
                              "visibility", "grasp")})
            print(f"[ok] scene_{idx:04d} {tier} vis={made['visibility']['visibility']} "
                  f"lift={made['grasp']['lift']} pxP={made['px_sep_plates']} "
                  f"pxS={made['px_sep_siblings']}")
        else:
            print(f"[FAIL] scene_{idx:04d} {task_key} {tier} N{n_infocus}/M{n_dest}")

    mpath = os.path.join(OUT_ROOT, "manifest.json")
    if only is not None and os.path.exists(mpath):
        keep = {e["dir"]: e for e in json.load(open(mpath))}
        for e in manifest:
            keep[e["dir"]] = e
        manifest = [keep[k] for k in sorted(keep)]
    json.dump(manifest, open(mpath, "w"), indent=2)

    imgs = []
    for e in manifest:
        im = cv2.imread(os.path.join(OUT_ROOT, e["dir"], "sketch.png"))
        if im is None:
            continue
        im = cv2.resize(im, (128, 128), interpolation=cv2.INTER_NEAREST)
        cv2.putText(im, f"{e['tier'][:4]} {e['n_target_infocus']}b{e['n_dest']}p",
                    (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1,
                    cv2.LINE_AA)
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


if __name__ == "__main__":
    main()
