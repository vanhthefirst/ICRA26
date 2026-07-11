"""
Dataset generation pipeline for DrawVLA (runs in WSL2).
V3: Adds Modality Mixing and Motion Arrows.
"""

import h5py, json, os, re, hashlib
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

import libero.libero.envs
import robosuite
import robomimic.utils.file_utils as FileUtils
from robosuite.utils import camera_utils as CU

DATA_DIR   = "/mnt/c/Users/Admin/sketch_vla/data/libero_spatial"
OUTPUT_DIR = "/mnt/c/Users/Admin/sketch_vla/outputs/dataset_poc"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_W, IMG_H = 128, 128
CAMERA_NAME  = "agentview"

import time

# PNG end-of-file marker (IEND chunk + its fixed CRC) -- every valid PNG
# ends with exactly these 8 bytes. A truncated write is missing this.
_PNG_IEND = b"IEND\xaeB`\x82"

# ── Utilities ──────────────────────────────────────────────────────────────────
def safe_imwrite(filepath, img, retries=5, wait=0.5):
    """
    Durably write `img` (BGR, uint8) to `filepath`.

    The previous version called cv2.imwrite() then immediately cv2.imread()
    in the same process to "verify" the write. That doesn't actually prove
    anything across the WSL2<->Windows DrvFs bridge: a same-process
    read-back can be satisfied entirely by the local OS page cache before
    the bytes are durably flushed to the underlying Windows-visible file --
    which is exactly why 4/5 motion-arrow PNGs passed that check yet were
    still truncated (missing their IEND chunk) when read from a completely
    separate mount of the same path. This version instead:
      1. Encodes the PNG fully in memory first (cv2.imencode), so we know
         exactly how many bytes SHOULD land on disk.
      2. Writes via a raw file handle and explicitly calls f.flush() +
         os.fsync(fd) to force the write out of the OS buffer and across
         the bridge, instead of trusting cv2.imwrite's internal buffering.
      3. Re-opens the file fresh (new fd, new read) and checks: exact byte
         length match, PNG IEND footer present, AND it decodes to the
         expected shape -- three independent signals instead of one.
      4. Retries with a short delay if any check fails.
    """
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise IOError(f"cv2.imencode failed for {filepath}")
    expected_bytes = encoded.tobytes()
    expected_len = len(expected_bytes)

    for attempt in range(1, retries + 1):
        with open(filepath, "wb") as f:
            f.write(expected_bytes)
            f.flush()
            os.fsync(f.fileno())

        try:
            with open(filepath, "rb") as f:
                on_disk = f.read()
        except Exception:
            on_disk = b""

        size_ok = len(on_disk) == expected_len
        iend_ok = on_disk.endswith(_PNG_IEND)
        decode_ok = False
        if size_ok and iend_ok:
            test_img = cv2.imread(filepath)
            decode_ok = test_img is not None and test_img.shape == img.shape

        if size_ok and iend_ok and decode_ok:
            return True

        print(f"    Warning: write verification failed for {filepath} "
              f"(size_ok={size_ok}, iend_ok={iend_ok}, decode_ok={decode_ok}), "
              f"retrying (attempt {attempt}/{retries})...")
        time.sleep(wait)

    raise IOError(f"Failed to durably write valid image to {filepath} after {retries} attempts.")
def detect_grasp_release(actions, ee_pos):
    gripper_cmd = actions[:, -1]
    T = len(gripper_cmd)
    grasp_f = release_f = None
    for i in range(1, T):
        if gripper_cmd[i - 1] < 0 and gripper_cmd[i] > 0 and grasp_f is None:
            grasp_f = i
        if gripper_cmd[i - 1] > 0 and gripper_cmd[i] < 0 and grasp_f is not None:
            release_f = i
            break
    if grasp_f is None:
        for i in range(1, T):
            if gripper_cmd[i - 1] > 0 and gripper_cmd[i] < 0 and grasp_f is None:
                grasp_f = i
            if gripper_cmd[i - 1] < 0 and gripper_cmd[i] > 0 and grasp_f is not None:
                release_f = i
                break
    used_fallback = False
    if grasp_f is None:
        used_fallback = True
        half      = T // 2
        grasp_f   = int(np.argmin(ee_pos[:half, 2]))
        release_f = half + int(np.argmin(ee_pos[half:, 2]))
    return grasp_f, release_f, used_fallback

def degrade_language_referential(lang):
    """Circle-only variant (answers 'which'). Needs destination manually specified in a real dataset, 
    but for POC we use a simple degraded placeholder like 'pick up this one'."""
    return "pick up this one"

def degrade_language_directional(lang):
    """Circle+Arrow variant. The drawing answers 'which' and 'where', so text can be maximally ambiguous."""
    return "move this there"


# ── Annotation Drawing ────────────────────────────────────────────────────────
def draw_circle(img, center_px, radius, seed):
    rng = np.random.default_rng(seed)
    out = img.copy()

    cx = center_px[0] + int(rng.integers(-3, 4))
    cy = center_px[1] + int(rng.integers(-3, 4))
    rx = radius + float(rng.uniform(-2, 3))
    ry = radius + float(rng.uniform(-2, 3))

    n      = 80
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
    
    tokens = {"cx": cx, "cy": cy, "rx": float(rx), "ry": float(ry)}
    return out, tokens

def draw_destination_arrow(img, pick_px, place_px, seed):
    rng = np.random.default_rng(seed + 1)
    out = img.copy()

    x1 = pick_px[0]  + int(rng.integers(-2, 3))
    y1 = pick_px[1]  + int(rng.integers(-2, 3))
    x2 = place_px[0] + int(rng.integers(-2, 3))
    y2 = place_px[1] + int(rng.integers(-2, 3))

    mx = (x1 + x2) // 2 + int(rng.integers(-7, 8))
    my = (y1 + y2) // 2 + int(rng.integers(-7, 8))
    thick = int(rng.integers(1, 3))
    
    cv2.line(out, (x1, y1), (mx, my), color=(200, 50, 50), thickness=thick, lineType=cv2.LINE_AA)
    cv2.arrowedLine(out, (mx, my), (x2, y2), color=(200, 50, 50), thickness=thick,
                    line_type=cv2.LINE_AA, tipLength=0.35)
    
    tokens = {"x0": x1, "y0": y1, "x1": x2, "y1": y2}
    return out, tokens

def draw_motion_arrow(img, waypoints_2d, place_px, seed):
    rng = np.random.default_rng(seed + 2)
    out = img.copy()
    
    if len(waypoints_2d) < 2:
        return out, []
        
    thick = int(rng.integers(1, 3))
    
    noisy_wps = []
    for p in waypoints_2d:
        nx = p[0] + int(rng.integers(-3, 4))
        ny = p[1] + int(rng.integers(-3, 4))
        noisy_wps.append((nx, ny))
        
    # To fix the short motion arrow issue, force the final waypoint to hit the destination place_px
    nx = place_px[0] + int(rng.integers(-3, 4))
    ny = place_px[1] + int(rng.integers(-3, 4))
    noisy_wps.append((nx, ny))
        
    for i in range(len(noisy_wps) - 2):
        cv2.line(out, noisy_wps[i], noisy_wps[i+1], color=(200, 50, 50), thickness=thick, lineType=cv2.LINE_AA)
    
    p2_last = noisy_wps[-2]
    p_last = noisy_wps[-1]
    cv2.arrowedLine(out, p2_last, p_last, color=(200, 50, 50), thickness=thick, line_type=cv2.LINE_AA, tipLength=0.35)
    
    tokens = [{"x": p[0], "y": p[1]} for p in noisy_wps]
    return out, tokens

# ── Robosuite setup ────────────────────────────────────────────────────────────
def build_asset_index(roots):
    index = {}
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != ".git"]
            for fn in filenames:
                index.setdefault(fn, os.path.join(dirpath, fn))
    return index

def get_visual_center_world_pos(model, data, body_id):
    geom_ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]
    ref_geoms = [g for g in geom_ids
                 if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
    if ref_geoms:
        return data.geom_xpos[ref_geoms[0]].copy()
    return data.body_xpos[body_id].copy()


def project_points(world_positions, world_to_pixel_tf):
    if len(world_positions) == 0: return []
    pts = np.array(world_positions)
    pix = CU.project_points_from_world_to_camera(
        points=pts, world_to_camera_transform=world_to_pixel_tf,
        camera_height=IMG_H, camera_width=IMG_W,
    )
    res = []
    for p in pix:
        col = int(p[1])
        row = (IMG_H - 1) - int(p[0])
        res.append((col, row))
    return res


def main():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".hdf5"))
    if not files: raise FileNotFoundError(f"No .hdf5 files found in {DATA_DIR}")

    asset_index = build_asset_index([os.path.dirname(robosuite.__file__), "/root/LIBERO"])
    def postprocess_model_xml(xml_str):
        def fix(m):
            p = m.group(1)
            if os.path.exists(p): return f'file="{p}"'
            new_p = asset_index.get(os.path.basename(p))
            return f'file="{new_p}"' if new_p else m.group(0)
        return re.sub(r'file="([^"]+)"', fix, xml_str)

    metadata = []
    
    for file_idx, filename in enumerate(files[:1]):
        path = os.path.join(DATA_DIR, filename)
        print(f"\nProcessing {filename}...")
        
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=path)
        env_kwargs = dict(env_meta["env_kwargs"])
        new_bddl = f"/root/LIBERO/libero/libero/bddl_files/libero_spatial/{filename.replace('_demo.hdf5', '.bddl')}"
        env_kwargs["bddl_file_name"] = new_bddl
        env_kwargs["camera_names"] = ["agentview"]
        env_kwargs["camera_heights"] = IMG_H
        env_kwargs["camera_widths"] = IMG_W
        
        try:
            robo_env = robosuite.make(env_meta["env_name"], **env_kwargs)
            robo_env.reset()
        except Exception as e:
            print(f"Error making env: {e}")
            continue

        with h5py.File(path, "r") as f:
            demo_keys = list(f["data"].keys())
            
            for demo_key in demo_keys[:5]:
                print(f"  -> {demo_key}")
                # Generate per-demo reproducible seed
                demo_seed = int(hashlib.sha256(demo_key.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)
                
                demo = f[f"data/{demo_key}"]
                rgb = demo["obs/agentview_rgb"][:]
                ee_pos = demo["obs/ee_pos"][:]
                actions = demo["actions"][:]
                states = demo["states"][:]
                lang = json.loads(f["data"].attrs["problem_info"])["language_instruction"]
                
                xml = postprocess_model_xml(demo.attrs["model_file"].decode("utf-8") if isinstance(demo.attrs["model_file"], bytes) else demo.attrs["model_file"])
                robo_env.reset_from_xml_string(xml)
                
                goal = robo_env.parsed_problem["goal_state"][0]
                target_obj_name = goal[1]
                dest_name = goal[2]
                
                robo_env.sim.set_state_from_flattened(states[0])
                robo_env.sim.forward()
                world_to_pixel_tf = CU.get_camera_transform_matrix(sim=robo_env.sim, camera_name="agentview", camera_height=IMG_H, camera_width=IMG_W)
                
                model = robo_env.sim.model
                data = robo_env.sim.data
                
                def resolve_body_name(name, body_names):
                    if name in body_names: return name
                    if f"{name}_main" in body_names: return f"{name}_main"
                    return None

                target_body_name = resolve_body_name(target_obj_name, model.body_names)
                target_id = model.body_name2id(target_body_name)
                target_pos = get_visual_center_world_pos(model, data, target_id)
                pick_px = project_points([target_pos], world_to_pixel_tf)[0]
                
                geom_ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == target_id]
                geom_centers = [data.geom_xpos[g].copy() for g in geom_ids]
                if geom_centers:
                    geom_pxs = project_points(geom_centers, world_to_pixel_tf)
                    cols, rows = [p[0] for p in geom_pxs], [p[1] for p in geom_pxs]
                    radius_px = max(5, min(int((np.sqrt((max(cols)-min(cols))**2 + (max(rows)-min(rows))**2) / 2.0) * 1.5), 40))
                else:
                    radius_px = 10
                
                dest_body_name = resolve_body_name(dest_name, model.body_names)
                if dest_name in model.site_names:
                    dest_pos = data.site_xpos[model.site_name2id(dest_name)].copy()
                elif dest_body_name is not None:
                    dest_pos = get_visual_center_world_pos(model, data, model.body_name2id(dest_body_name))
                else:
                    dest_pos = np.array([0,0,0])
                    
                place_px = project_points([dest_pos], world_to_pixel_tf)[0]
                
                grasp_f, release_f, used_fallback = detect_grasp_release(actions, ee_pos)
                if used_fallback:
                    print(f"    WARNING: {demo_key} used min-z fallback for grasp/release.")
                
                # Compute Motion Arrow 2D Waypoints
                if not used_fallback and release_f > grasp_f:
                    ee_traj = ee_pos[grasp_f:release_f]
                    indices = np.linspace(0, len(ee_traj)-1, 5, dtype=int)
                    sparse_ee_3d = ee_traj[indices]
                    motion_wps_2d = project_points(sparse_ee_3d, world_to_pixel_tf)
                else:
                    motion_wps_2d = None
                
                base_img = rgb[0].copy()

                # Variant 1: Clean (No Draw)
                meta_clean = {
                    "file": filename, "demo": demo_key, "variant": "clean",
                    "instruction": lang, "used_grasp_fallback": used_fallback,
                    "symbolic_tokens": {}, "image": None
                }
                metadata.append(meta_clean)

                # Variant 2: Circle Only
                img_v2, tok_circle = draw_circle(base_img, pick_px, radius_px, demo_seed)
                out_v2 = f"demo_{file_idx}_{demo_key}_v2_circle.png"
                safe_imwrite(os.path.join(OUTPUT_DIR, out_v2), cv2.cvtColor(img_v2, cv2.COLOR_RGB2BGR))
                meta_circle = {
                    "file": filename, "demo": demo_key, "variant": "circle_only",
                    "instruction": degrade_language_referential(lang),
                    "used_grasp_fallback": used_fallback,
                    "symbolic_tokens": {"circle": tok_circle},
                    "image": out_v2
                }
                metadata.append(meta_circle)

                # Variant 3: Circle + Dest Arrow
                img_v3, tok_dest_arrow = draw_destination_arrow(img_v2, pick_px, place_px, demo_seed)
                out_v3 = f"demo_{file_idx}_{demo_key}_v3_dest_arrow.png"
                safe_imwrite(os.path.join(OUTPUT_DIR, out_v3), cv2.cvtColor(img_v3, cv2.COLOR_RGB2BGR))
                meta_dest_arrow = {
                    "file": filename, "demo": demo_key, "variant": "circle_dest_arrow",
                    "instruction": degrade_language_directional(lang),
                    "used_grasp_fallback": used_fallback,
                    "symbolic_tokens": {"circle": tok_circle, "arrow": tok_dest_arrow},
                    "image": out_v3
                }
                metadata.append(meta_dest_arrow)

                # Variant 4: Circle + Motion Arrow
                if motion_wps_2d is not None:
                    img_v4, tok_motion_arrow = draw_motion_arrow(img_v2, motion_wps_2d, place_px, demo_seed)
                    out_v4 = f"demo_{file_idx}_{demo_key}_v4_motion_arrow.png"
                    safe_imwrite(os.path.join(OUTPUT_DIR, out_v4), cv2.cvtColor(img_v4, cv2.COLOR_RGB2BGR))
                    meta_motion_arrow = {
                        "file": filename, "demo": demo_key, "variant": "circle_motion_arrow",
                        "instruction": degrade_language_directional(lang),
                        "used_grasp_fallback": used_fallback,
                        "symbolic_tokens": {"circle": tok_circle, "motion_arrow": tok_motion_arrow},
                        "image": out_v4
                    }
                    metadata.append(meta_motion_arrow)

    with open(os.path.join(OUTPUT_DIR, "dataset_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Exported to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
