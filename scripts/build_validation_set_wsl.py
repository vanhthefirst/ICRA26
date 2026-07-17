"""
DrawVLA VALIDATION SET builder (run in WSL2, libero env).

Produces a graded-ambiguity, simulator-grounded validation set:
each scene = an augmented LIBERO tabletop with duplicate bowls / plates /
ramekins / cookie boxes, a deliberately vague caption, a synthetic
circle+arrow sketch (overlay PNG + symbolic tokens), and a BDDL goal naming a
specific target bowl + destination plate. Every scene is validated by the
teleport oracle (GT bowl -> GT plate must satisfy the goal predicate) and
discarded+resampled if it fails.

Tiers (isolate the two ambiguity axes, span the 90%->25% curve):
  control      : 1 bowl , 1 plate            (text-solvable; no-regression)
  referential  : N bowls, 1 plate            (which OBJECT)
  directional  : 1 bowl , M plates           (which DESTINATION)
  both         : N bowls, M plates + clutter (hardest; the 25% case)

Output tree:
  outputs/validation_set/
    scene_XXXX/ {scene.bddl, frame0.png, sketch.png, tokens.json, meta.json}
    manifest.json
    contact_sheet.png

    conda activate <libero env>
    cd /mnt/c/Users/Admin/sketch_vla
    python scripts/build_validation_set_wsl.py 2>&1 | tee outputs/validation_set/build_log.txt
"""

import os, json, gc
import numpy as np
import cv2

# ─────────────────────────────── CONFIG ───────────────────────────────
SMOKE = False    # True: ~6 representative scenes for a quick check.
                 # False: the full ~38-scene set.

OUT_ROOT = "/mnt/c/Users/Admin/sketch_vla/outputs/validation_set"
IMG_H = IMG_W = 128
CAMERA = "agentview"

# reachable tabletop rectangle (table-frame metres), from shipped spatial scenes
RECT_X = (-0.20, 0.18)
RECT_Y = (-0.05, 0.30)
MIN_DIST = 0.095          # min centre-to-centre spacing (m)
HALF_BOX = 0.012          # bddl region half-extent
TABLE_Z_MIN = 0.85        # settled objects must stay above this (else "fell off")

CAPTIONS = [
    "pick up the black bowl and place it on the plate",
    "pick up the bowl and put it on the plate",
    "grab the black bowl and place it on the plate",
]

# tier -> list of (n_bowl, m_plate, n_ramekin, n_cookie)
def tier_specs(smoke):
    if smoke:
        return [
            ("control",     [(1, 1, 1, 1)]),
            ("referential", [(3, 1, 1, 1)]),
            ("directional", [(1, 3, 1, 1)]),
            ("both",        [(4, 3, 2, 1), (5, 4, 2, 2)]),
            ("both",        [(4, 2, 1, 1)]),
        ]
    specs = []
    specs.append(("control",     [(1, 1, 1, 1)] * 5))
    ref = []
    for n in (2, 3, 4, 5):
        ref += [(n, 1, 1, 1)] * 3
    specs.append(("referential", ref))                       # 12
    dr = []
    for m in (2, 3, 4):
        dr += [(1, m, 1, 1)] * 3
    specs.append(("directional", dr))                        # 9
    bo = []
    for (n, m) in ((3, 2), (4, 3), (5, 3), (4, 4)):
        bo += [(n, m, 2, 2)] * 3
    specs.append(("both", bo))                               # 12
    return specs                                             # ~38


# ─────────────────── draw funcs (match training pipeline) ───────────────────
def draw_circle(img, center_px, radius, seed):
    rng = np.random.default_rng(seed); out = img.copy()
    cx = center_px[0] + int(rng.integers(-3, 4)); cy = center_px[1] + int(rng.integers(-3, 4))
    rx = radius + float(rng.uniform(-2, 3)); ry = radius + float(rng.uniform(-2, 3))
    angles = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    wob = (rng.uniform(0.5, 1.0) * np.sin(3 * angles + rng.uniform(0, np.pi))
           + rng.uniform(0.3, 0.6) * np.sin(7 * angles + rng.uniform(0, np.pi)))
    ws = float(rng.uniform(0.8, 1.8)); pts = []
    for i, a in enumerate(angles):
        s = 1.0 + wob[i] * ws / max(rx, ry)
        pts.append((int(np.clip(cx + rx * s * np.cos(a), 1, IMG_W - 2)),
                    int(np.clip(cy + ry * s * np.sin(a), 1, IMG_H - 2))))
    for i in range(len(pts)):
        cv2.line(out, pts[i], pts[(i + 1) % len(pts)], (0, 200, 0),
                 int(rng.integers(1, 3)), cv2.LINE_AA)
    return out, {"cx": cx, "cy": cy, "rx": float(rx), "ry": float(ry)}


def draw_arrow(img, pick_px, place_px, seed):
    rng = np.random.default_rng(seed + 1); out = img.copy()
    x1 = pick_px[0] + int(rng.integers(-2, 3)); y1 = pick_px[1] + int(rng.integers(-2, 3))
    x2 = place_px[0] + int(rng.integers(-2, 3)); y2 = place_px[1] + int(rng.integers(-2, 3))
    mx = (x1 + x2) // 2 + int(rng.integers(-7, 8)); my = (y1 + y2) // 2 + int(rng.integers(-7, 8))
    th = int(rng.integers(1, 3))
    cv2.line(out, (x1, y1), (mx, my), (200, 50, 50), th, cv2.LINE_AA)
    cv2.arrowedLine(out, (mx, my), (x2, y2), (200, 50, 50), th, cv2.LINE_AA, tipLength=0.35)
    return out, {"x0": x1, "y0": y1, "x1": x2, "y1": y2}


# ─────────────────────────── helpers ───────────────────────────
def project_points(world, W2P):
    from robosuite.utils import camera_utils as CU
    if len(world) == 0:
        return []
    pix = CU.project_points_from_world_to_camera(
        points=np.array(world), world_to_camera_transform=W2P,
        camera_height=IMG_H, camera_width=IMG_W)
    return [(int(p[1]), (IMG_H - 1) - int(p[0])) for p in pix]


def visual_center(model, data, bid):
    gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
    ref = [g for g in gids if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
    return data.geom_xpos[ref[0]].copy() if ref else data.body_xpos[bid].copy()


def frame_from_obs(obs):
    key = "agentview_image" if "agentview_image" in obs else \
          [k for k in obs if "agentview" in k and "image" in k][0]
    f = np.asarray(obs[key])
    if f.dtype != np.uint8:
        f = np.clip(f * 255.0 if f.max() <= 1.0 + 1e-6 else f, 0, 255).astype(np.uint8)
    return f.copy()


def check_success(env):
    for obj in (env, getattr(env, "env", None)):
        if obj is None:
            continue
        for m in ("check_success", "_check_success"):
            fn = getattr(obj, m, None)
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
        if all(np.hypot(x - px, y - py) >= MIN_DIST for px, py in pts):
            pts.append((float(x), float(y)))
    if len(pts) < n:
        raise RuntimeError(f"placement failed: {len(pts)}/{n}")
    return pts


def build_bddl(counts, language, seed):
    rng = np.random.default_rng(seed)
    instances = []
    for cat, n in counts.items():
        instances += [(cat, f"{cat}_{i}") for i in range(1, n + 1)]
    pos = sample_positions(len(instances), rng)
    placements = {inst: pos[k] for k, (_, inst) in enumerate(instances)}
    t = int(rng.integers(1, counts["akita_black_bowl"] + 1))
    d = int(rng.integers(1, counts["plate"] + 1))
    tb, tp = f"akita_black_bowl_{t}", f"plate_{d}"

    def region(inst):
        cx, cy = placements[inst]
        return (f"      ({inst}_region\n          (:target main_table)\n"
                f"          (:ranges (\n              "
                f"({cx-HALF_BOX:.4f} {cy-HALF_BOX:.4f} {cx+HALF_BOX:.4f} {cy+HALF_BOX:.4f})\n"
                f"            )\n          )\n      )\n")
    regions = "".join(region(i) for _, i in instances)
    objs = "\n".join(f"    " + " ".join(f"{c}_{k}" for k in range(1, n + 1)) + f" - {c}"
                     for c, n in counts.items())
    inits = "".join(f"    (On {i} main_table_{i}_region)\n" for _, i in instances)
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
    meta = {"counts": counts, "language": language, "seed": seed,
            "target_bowl": tb, "target_plate": tp,
            "placements": placements, "instances": [i for _, i in instances]}
    return bddl, meta


def build_one_scene(spec_counts, tier, seed, scene_dir):
    """Returns meta dict on success, or None if invalid (caller retries)."""
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils import camera_utils as CU

    rng = np.random.default_rng(seed)
    counts = {"akita_black_bowl": spec_counts[0], "plate": spec_counts[1],
              "glazed_rim_porcelain_ramekin": spec_counts[2], "cookies": spec_counts[3]}
    language = CAPTIONS[int(rng.integers(0, len(CAPTIONS)))]
    bddl, meta = build_bddl(counts, language, seed)
    meta["tier"] = tier
    os.makedirs(scene_dir, exist_ok=True)
    bddl_path = os.path.join(scene_dir, "scene.bddl")
    open(bddl_path, "w").write(bddl)

    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=IMG_H,
                             camera_widths=IMG_W, camera_names=[CAMERA])
    try:
        obs = env.reset()
        adim = getattr(env, "action_dim", 7)
        for _ in range(20):
            obs, _, _, _ = env.step(np.zeros(adim))
        frame = frame_from_obs(obs)
        sim = env.sim; model, data = sim.model, sim.data
        W2P = CU.get_camera_transform_matrix(sim=sim, camera_name=CAMERA,
                                             camera_height=IMG_H, camera_width=IMG_W)

        def bid(name):
            for c in (name, f"{name}_main"):
                if c in model.body_names:
                    return model.body_name2id(c)
            raise KeyError(name)

        tb, tp = meta["target_bowl"], meta["target_plate"]
        # validity: all objects settled on table
        for inst in meta["instances"]:
            if visual_center(model, data, bid(inst))[2] < TABLE_Z_MIN:
                print(f"    invalid (fell off): {inst}"); return None

        pick_px = project_points([visual_center(model, data, bid(tb))], W2P)[0]
        place_px = project_points([visual_center(model, data, bid(tp))], W2P)[0]
        if not (3 <= pick_px[0] <= 124 and 3 <= pick_px[1] <= 124):
            print("    invalid (target off-frame)"); return None

        gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid(tb)]
        gpx = project_points([data.geom_xpos[g].copy() for g in gids], W2P)
        cols, rows = [p[0] for p in gpx], [p[1] for p in gpx]
        radius = max(5, min(int(np.hypot(max(cols) - min(cols), max(rows) - min(rows)) / 2 * 1.5), 40))

        # sketch
        img_c, tok_c = draw_circle(frame, pick_px, radius, seed)
        img_ca, tok_a = draw_arrow(img_c, pick_px, place_px, seed)
        cv2.imwrite(os.path.join(scene_dir, "frame0.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(scene_dir, "sketch.png"), cv2.cvtColor(img_ca, cv2.COLOR_RGB2BGR))

        # teleport oracle
        jname = next((model.joint_id2name(j) for j in range(model.njnt)
                      if model.joint_id2name(j) and model.joint_id2name(j).startswith(tb)
                      and "joint" in model.joint_id2name(j)), None)
        oracle = None
        if jname:
            jid = model.joint_name2id(jname)
            qa = model.jnt_qposadr[jid]; va = model.jnt_dofadr[jid]
            pp = visual_center(model, data, bid(tp))
            data.qpos[qa:qa + 3] = [pp[0], pp[1], pp[2] + 0.04]
            data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
            data.qvel[va:va + 6] = 0
            sim.forward()
            for _ in range(40):
                obs, _, _, _ = env.step(np.zeros(adim))
            oracle = check_success(env)
        if oracle is not True:
            print(f"    invalid (oracle={oracle})"); return None

        meta.update({"pick_px": list(pick_px), "place_px": list(place_px),
                     "radius": radius, "symbolic_tokens": {"circle": tok_c, "arrow": tok_a},
                     "camera_matrix": W2P.tolist(), "oracle_success": True,
                     "object_pixels": {i: list(project_points([visual_center(model, data, bid(i))], W2P)[0])
                                       for i in meta["instances"]}})
        json.dump(meta, open(os.path.join(scene_dir, "meta.json"), "w"), indent=2)
        json.dump({"instruction": language, "target_bowl": tb, "target_plate": tp,
                   "symbolic_tokens": meta["symbolic_tokens"]},
                  open(os.path.join(scene_dir, "tokens.json"), "w"), indent=2)
        return meta
    finally:
        try:
            env.close()
        except Exception:
            pass
        gc.collect()


def contact_sheet(manifest, path):
    imgs = []
    for e in manifest:
        p = os.path.join(OUT_ROOT, e["dir"], "sketch.png")
        im = cv2.imread(p)
        if im is None:
            continue
        im = cv2.resize(im, (128, 128), interpolation=cv2.INTER_NEAREST)
        cv2.putText(im, f"{e['tier'][:4]} {e['counts']['akita_black_bowl']}b{e['counts']['plate']}p",
                    (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1, cv2.LINE_AA)
        imgs.append(im)
    if not imgs:
        return
    cols = min(8, len(imgs)); rows = (len(imgs) + cols - 1) // cols
    sheet = np.full((rows * 130, cols * 130, 3), 40, np.uint8)
    for k, im in enumerate(imgs):
        r, c = divmod(k, cols)
        sheet[r * 130 + 1:r * 130 + 129, c * 130 + 1:c * 130 + 129] = im
    cv2.imwrite(path, sheet)


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    specs = tier_specs(SMOKE)
    manifest = []
    idx = 0
    base_seed = 1000
    for tier, combos in specs:
        for combo in combos:
            # retry up to 6 seeds until a valid scene is produced
            made = None
            for attempt in range(6):
                seed = base_seed + idx * 100 + attempt
                scene_dir = os.path.join(OUT_ROOT, f"scene_{idx:04d}")
                try:
                    made = build_one_scene(combo, tier, seed, scene_dir)
                except Exception as e:
                    print(f"  scene {idx:04d} attempt {attempt} error: {e}")
                    made = None
                if made:
                    break
            if made:
                made["dir"] = f"scene_{idx:04d}"
                manifest.append({"dir": made["dir"], "tier": tier,
                                 "counts": made["counts"], "seed": made["seed"],
                                 "target_bowl": made["target_bowl"],
                                 "target_plate": made["target_plate"],
                                 "language": made["language"]})
                print(f"[ok] scene_{idx:04d}  {tier:11s} "
                      f"{combo[0]}bowl {combo[1]}plate  target={made['target_bowl']}->{made['target_plate']}")
            else:
                print(f"[FAIL] scene_{idx:04d} {tier} {combo} — no valid scene in 6 tries")
            idx += 1

    json.dump(manifest, open(os.path.join(OUT_ROOT, "manifest.json"), "w"), indent=2)
    contact_sheet(manifest, os.path.join(OUT_ROOT, "contact_sheet.png"))
    by_tier = {}
    for e in manifest:
        by_tier[e["tier"]] = by_tier.get(e["tier"], 0) + 1
    print(f"\nSMOKE={SMOKE}  produced {len(manifest)} valid scenes: {by_tier}")
    print("contact_sheet.png + manifest.json written to", OUT_ROOT)


if __name__ == "__main__":
    main()
