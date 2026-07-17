"""
DrawVLA validation set — augmented-scene generator + vertical-slice smoke test.
Run in WSL2 (libero env). Simulator-only augmentation: we emit a real LIBERO
BDDL file (same grammar as the shipped libero_spatial tasks) with EXTRA
duplicate bowls / plates / ramekins / cookie boxes, and a goal that names ONE
specific target bowl + ONE specific destination plate.

This script:
  1. Builds one augmented BDDL by direct text templating (no procedural-API
     fragility) into outputs/vslice/scene_0001.bddl
  2. Loads it with libero's OffScreenRenderEnv, resets (objects settle),
     renders agentview frame 0 -> outputs/vslice/scene_0001_frame0.png
  3. Reads every object's settled world position, projects to pixels, and
     writes outputs/vslice/scene_0001_meta.json + a projection-check overlay
     (small dots on every object, red dot = GT target bowl, blue = GT plate).

Nothing under data/ is touched. Safe to run.

    conda activate <libero env>
    cd /mnt/c/Users/Admin/sketch_vla
    python scripts/gen_augmented_scene_wsl.py 2>&1 | tee outputs/vslice/gen_log.txt

Then send me outputs/vslice/scene_0001_frame0.png, the projection-check png,
scene_0001_meta.json, scene_0001.bddl and gen_log.txt.
"""

import os, json, re
import numpy as np
import cv2

OUT_DIR = "/mnt/c/Users/Admin/sketch_vla/outputs/vslice"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_H = IMG_W = 128
CAMERA = "agentview"

# ---- scene composition for the vertical slice --------------------------------
# modest counts to prove the loop; we scale up (3-5 bowls, 3-4 plates, etc.)
# once this renders cleanly.
COUNTS = {
    "akita_black_bowl": 3,               # near-identical -> referential ambiguity
    "plate": 2,                          # multiple destinations -> directional ambiguity
    "glazed_rim_porcelain_ramekin": 1,   # clutter
    "cookies": 1,                        # clutter
}
LANGUAGE = "pick up the black bowl and place it on the plate"   # deliberately vague
SEED = 1
KEEP_BIG_FIXTURES = False   # drop wooden_cabinet + flat_stove to free tabletop space

# reachable tabletop rectangle (table-frame metres), inferred from the shipped
# libero_spatial region centroids.
X_RANGE = (-0.20, 0.16)
Y_RANGE = (0.00, 0.30)
GRID_NX, GRID_NY = 4, 4
JITTER = 0.015          # +/- within a grid cell
HALF_BOX = 0.015        # region half-extent written into the bddl


# ============================ BDDL text builder ===============================
def make_grid_cells():
    xs = np.linspace(X_RANGE[0], X_RANGE[1], GRID_NX)
    ys = np.linspace(Y_RANGE[0], Y_RANGE[1], GRID_NY)
    cells = [(float(x), float(y)) for x in xs for y in ys]
    return cells


def build_bddl(counts, language, seed):
    rng = np.random.default_rng(seed)
    # flat list of instances
    instances = []
    for cat, n in counts.items():
        for i in range(1, n + 1):
            instances.append((cat, f"{cat}_{i}"))

    cells = make_grid_cells()
    if len(instances) > len(cells):
        raise ValueError(f"{len(instances)} objects > {len(cells)} grid cells; "
                         f"increase grid or reduce counts.")
    chosen = rng.choice(len(cells), size=len(instances), replace=False)

    placements = {}  # inst -> (cx, cy)
    for (cat, inst), ci in zip(instances, chosen):
        cx, cy = cells[int(ci)]
        cx += float(rng.uniform(-JITTER, JITTER))
        cy += float(rng.uniform(-JITTER, JITTER))
        placements[inst] = (cx, cy)

    # ground-truth target bowl + destination plate
    n_bowl = counts["akita_black_bowl"]
    n_plate = counts["plate"]
    t = int(rng.integers(1, n_bowl + 1))
    d = int(rng.integers(1, n_plate + 1))
    target_bowl = f"akita_black_bowl_{t}"
    target_plate = f"plate_{d}"

    # ---- assemble text ----
    def region_block(inst):
        cx, cy = placements[inst]
        x1, y1, x2, y2 = cx - HALF_BOX, cy - HALF_BOX, cx + HALF_BOX, cy + HALF_BOX
        return (f"      ({inst}_region\n"
                f"          (:target main_table)\n"
                f"          (:ranges (\n"
                f"              ({x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f})\n"
                f"            )\n"
                f"          )\n"
                f"      )\n")

    regions = "".join(region_block(inst) for _, inst in instances)

    # objects grouped by category
    obj_lines = []
    for cat, n in counts.items():
        names = " ".join(f"{cat}_{i}" for i in range(1, n + 1))
        obj_lines.append(f"    {names} - {cat}")
    objects_str = "\n".join(obj_lines)

    init_lines = "".join(
        f"    (On {inst} main_table_{inst}_region)\n" for _, inst in instances
    )

    bddl = f"""(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language {language})
    (:regions
{regions}    )

  (:fixtures
    main_table - table
  )

  (:objects
{objects_str}
  )

  (:obj_of_interest
    {target_bowl}
    {target_plate}
  )

  (:init
{init_lines}  )

  (:goal
    (And (On {target_bowl} {target_plate}))
  )

)
"""
    meta = {
        "counts": counts, "language": language, "seed": seed,
        "target_bowl": target_bowl, "target_plate": target_plate,
        "placements": placements,
        "instances": [inst for _, inst in instances],
    }
    return bddl, meta


# ============================ projection helpers ==============================
def project_points(world_positions, world_to_pixel_tf):
    from robosuite.utils import camera_utils as CU
    if len(world_positions) == 0:
        return []
    pts = np.array(world_positions)
    pix = CU.project_points_from_world_to_camera(
        points=pts, world_to_camera_transform=world_to_pixel_tf,
        camera_height=IMG_H, camera_width=IMG_W)
    res = []
    for p in pix:
        col = int(p[1]); row = (IMG_H - 1) - int(p[0])
        res.append((col, row))
    return res


def get_visual_center(model, data, body_id):
    geom_ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]
    ref = [g for g in geom_ids
           if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
    if ref:
        return data.geom_xpos[ref[0]].copy()
    return data.body_xpos[body_id].copy()


# ================================== main ======================================
def main():
    bddl, meta = build_bddl(COUNTS, LANGUAGE, SEED)
    bddl_path = os.path.join(OUT_DIR, "scene_0001.bddl")
    with open(bddl_path, "w") as f:
        f.write(bddl)
    print("Wrote", bddl_path)
    print("\n----- BDDL -----\n", bddl)

    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils import camera_utils as CU

    env = OffScreenRenderEnv(bddl_file_name=bddl_path,
                             camera_heights=IMG_H, camera_widths=IMG_W,
                             camera_names=[CAMERA])
    obs = env.reset()
    # action dim varies by wrapper/robosuite version; resolve robustly
    try:
        adim = env.action_dim
    except AttributeError:
        try:
            low, _ = env.action_spec
            adim = int(np.asarray(low).shape[0])
        except Exception:
            adim = 7                          # Panda OSC_POSE (6) + gripper (1)
    print("action dim:", adim)
    for _ in range(20):                       # let objects settle
        obs, _, _, _ = env.step(np.zeros(adim))

    # frame: agentview_image is ALREADY correctly oriented -- do NOT flip it.
    # (per verify_object_projection_wsl.py: the projection mirrors the ROW via
    # (H-1)-row instead; flipping the frame double-mirrors and misaligns.)
    img_key = "agentview_image" if "agentview_image" in obs else \
              [k for k in obs if "agentview" in k and "image" in k][0]
    frame = np.asarray(obs[img_key])
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0 if frame.max() <= 1.0 + 1e-6 else frame,
                        0, 255).astype(np.uint8)
    frame = frame.copy()
    cv2.imwrite(os.path.join(OUT_DIR, "scene_0001_frame0.png"),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    sim = env.sim
    model, data = sim.model, sim.data
    W2P = CU.get_camera_transform_matrix(sim=sim, camera_name=CAMERA,
                                         camera_height=IMG_H, camera_width=IMG_W)

    def body_id(name):
        for cand in (name, f"{name}_main"):
            if cand in model.body_names:
                return model.body_name2id(cand)
        return None

    obj_world, obj_px = {}, {}
    for inst in meta["instances"]:
        bid = body_id(inst)
        if bid is None:
            print("  [!] body not found for", inst); continue
        pos = get_visual_center(model, data, bid)
        obj_world[inst] = pos.tolist()
        obj_px[inst] = project_points([pos], W2P)[0]

    meta["object_world_pos"] = obj_world
    meta["object_pixels"] = {k: list(v) for k, v in obj_px.items()}
    meta["camera_matrix"] = W2P.tolist()
    with open(os.path.join(OUT_DIR, "scene_0001_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # projection-check overlay
    check = frame.copy()
    for inst, (cx, cy) in obj_px.items():
        if inst == meta["target_bowl"]:
            color = (255, 0, 0)         # red = GT target bowl
        elif inst == meta["target_plate"]:
            color = (0, 0, 255)         # blue = GT destination plate
        else:
            color = (0, 255, 0)         # green = other objects
        cv2.circle(check, (cx, cy), 4, color, 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT_DIR, "scene_0001_projcheck.png"),
                cv2.cvtColor(check, cv2.COLOR_RGB2BGR))

    print("\nTarget bowl :", meta["target_bowl"], "px", obj_px.get(meta["target_bowl"]))
    print("Target plate:", meta["target_plate"], "px", obj_px.get(meta["target_plate"]))
    print("All objects :")
    for inst in meta["instances"]:
        print(f"   {inst:34s} world={np.round(obj_world.get(inst,[]),3)} px={obj_px.get(inst)}")
    print("\nDONE. Send back the 3 pngs + json + bddl + gen_log.txt from", OUT_DIR)
    env.close()


if __name__ == "__main__":
    main()
