"""
Proof-of-concept annotation for DrawVLA.

Given one LIBERO-Spatial HDF5 demo episode:
  1. Loads the first frame of demo_0
  2. Detects grasp / release frames from gripper_states
  3. Projects EEF 3D positions to image-plane pixels (exact camera params,
     extracted from the episode's own MuJoCo XML via extract_camera_from_xml.py)
  4. Draws a clean standard circle + straight arrow
  5. Draws a human-like wobbled circle + curved arrow
  6. Saves three-panel comparison → outputs/annotation_poc.png
"""

import h5py, json, os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

from mujoco_object_state import (
    build_model, free_object_bodies, body_positions_at,
    identify_manipulated_object,
)


# NOTE: this project is run from WSL2 (conda env: libero), same as the
# verify_*_wsl.py scripts -- use the /mnt/c/... form, not a native Windows
# path (C:\...), which WSL2's Python can't resolve.
DATA_DIR   = "/mnt/c/Users/Admin/sketch_vla/data/libero_spatial"
OUTPUT_DIR = "/mnt/c/Users/Admin/sketch_vla/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_W, IMG_H = 128, 128
CAMERA_NAME  = "agentview"

# ── Camera math ────────────────────────────────────────────────────────────────
# NOTE (this session): previously this file hand-transcribed CAM_EYE /
# CAM_LOOKAT / CAM_UP / CAM_FOV as constants and used a hand-rolled
# look-at/perspective projection (build_view_matrix / world_to_pixel). That
# was a real transcription-error risk, and a CAM_UP axis-labeling bug in an
# earlier session caused real damage (see git history). Since mujoco_object_
# state.build_model() already compiles a real mujoco.MjModel for this exact
# episode, we now read the camera pose LIVE from the compiled model every
# run (cam_xpos/cam_xmat/cam_fovy -- no transcription possible) and project
# using the exact convention from robosuite.utils.camera_utils
# (get_camera_extrinsic_matrix / get_camera_intrinsic_matrix /
# project_points_from_world_to_camera, verified against the current
# ARISE-Initiative/robosuite source), reimplemented here with plain numpy so
# this script has no robosuite/WSL2 dependency. This guarantees annotate_poc.py
# and the WSL2 verify_*_wsl.py scripts agree by construction.


def get_camera_matrix(model, data, camera_name, img_w, img_h):
    """
    Build the 4x4 world->pixel projection matrix for @camera_name, using the
    exact same convention as robosuite.utils.camera_utils.get_camera_transform_matrix:
      1. Extrinsic: MuJoCo's raw cam_xpos/cam_xmat pose, corrected onto the
         standard (x=right, y=down, z=forward) computer-vision camera
         convention via the [1,-1,-1] diagonal correction.
      2. Intrinsic: standard pinhole K from cam_fovy.
      3. Combined: K_exp @ inverse(extrinsic).
    Requires mujoco.mj_forward(model, data) to have been called already so
    cam_xpos/cam_xmat are populated (camera is static/world-fixed here, so
    any valid forward pass works -- it does not depend on which qpos frame
    was used).
    """
    cam_id = model.camera(camera_name).id
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_rot = data.cam_xmat[cam_id].reshape(3, 3).copy()
    fovy    = float(model.cam_fovy[cam_id])

    # Extrinsic (camera pose in world), MuJoCo -> CV axis correction.
    R = np.eye(4)
    R[:3, :3] = cam_rot
    R[:3, 3]  = cam_pos
    axis_correction = np.diag([1.0, -1.0, -1.0, 1.0])
    R = R @ axis_correction

    # Intrinsic.
    f = 0.5 * img_h / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, img_w / 2], [0, f, img_h / 2], [0, 0, 1]])
    K_exp = np.eye(4)
    K_exp[:3, :3] = K

    R_inv = np.linalg.inv(R)
    return K_exp @ R_inv, fovy


def world_to_pixel(pt3d, camera_matrix, W, H):
    """
    Project a 3D world-space point to (col, row) pixel coordinates, using
    the same convention as robosuite.utils.camera_utils.
    project_points_from_world_to_camera. Returns None if the point projects
    behind the camera (non-positive depth).

    NOTE: the raw row from this convention (K_exp @ inverse-extrinsic, same
    as robosuite.utils.camera_utils) comes out vertically mirrored for this
    camera/mujoco setup -- confirmed empirically this session by comparing
    robosuite's own project_points_from_world_to_camera against a MuJoCo-
    rendered marker ground truth (verify_markers_in_scene_wsl.py): printed
    rows matched exactly (H-1)-row of the true positions, while the frame
    itself needed no adjustment. Mirroring the row back here keeps this
    script's output consistent with the verified WSL2 pipeline. If this
    project's camera setup ever changes and pixels look upside-down again,
    re-check against a marker-in-scene ground truth before re-deriving this.
    """
    p = camera_matrix @ np.array([*pt3d, 1.0])
    if p[2] <= 0:
        return None                                  # point is behind camera
    x = p[0] / p[2]
    y = p[1] / p[2]
    col = int(np.clip(round(x), 0, W - 1))
    row = int(np.clip(round(y), 0, H - 1))
    row = (H - 1) - row
    return (col, row)


# ── Grasp / release detection ──────────────────────────────────────────────────
def detect_grasp_release(actions, ee_pos):
    """
    Uses actions[:, -1] — the gripper command in robosuite OSC_POSE:
      -1.0 = open (release)
      +1.0 = close (grasp)

    Falls back to minimum-z heuristic if no clean transition is found.
    """
    gripper_cmd = actions[:, -1]
    T = len(gripper_cmd)

    print(f"\nGripper command  min={gripper_cmd.min():.2f}  max={gripper_cmd.max():.2f}")
    print(f"Unique values: {np.unique(gripper_cmd.round(1))}")

    grasp_f = release_f = None

    # Convention A: -1=open → +1=close means grasp
    for i in range(1, T):
        if gripper_cmd[i - 1] < 0 and gripper_cmd[i] > 0 and grasp_f is None:
            grasp_f = i
        if gripper_cmd[i - 1] > 0 and gripper_cmd[i] < 0 and grasp_f is not None:
            release_f = i
            break

    # Convention B (reverse): +1=open → -1=close means grasp
    if grasp_f is None:
        print("Convention A found no transition. Trying convention B (+1→-1 = grasp).")
        for i in range(1, T):
            if gripper_cmd[i - 1] > 0 and gripper_cmd[i] < 0 and grasp_f is None:
                grasp_f = i
            if gripper_cmd[i - 1] < 0 and gripper_cmd[i] > 0 and grasp_f is not None:
                release_f = i
                break

    # Fallback: use minimum EEF-z in first/second half as proxy
    if grasp_f is None:
        print("No gripper transition found in actions. Using min-z fallback.")
        half      = T // 2
        grasp_f   = int(np.argmin(ee_pos[:half, 2]))
        release_f = half + int(np.argmin(ee_pos[half:, 2]))

    return grasp_f, release_f


# ── Standard (clean/perfect) annotation ───────────────────────────────────────
def draw_standard(img, pick_px, place_px, radius=10):
    """
    Perfect circle at pick location.
    Straight arrow from pick to place location.
    """
    out = img.copy()
    cv2.circle(out, pick_px, radius,
               color=(0, 220, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.arrowedLine(out, pick_px, place_px,
                    color=(220, 50, 50), thickness=2,
                    line_type=cv2.LINE_AA, tipLength=0.25)
    return out


# ── Human-like (augmented) annotation ─────────────────────────────────────────
def draw_human(img, pick_px, place_px, radius=10, seed=42):
    """
    Wobbled ellipse at pick location (simulates shaky hand).
    Two-segment curved arrow to place location (simulates rough stroke).
    Both have jittered endpoints and variable stroke width.
    """
    rng = np.random.default_rng(seed)
    out = img.copy()

    # ── Circle ────────────────────────────────────────────────────────────────
    cx = pick_px[0] + int(rng.integers(-3, 4))    # jitter the centre
    cy = pick_px[1] + int(rng.integers(-3, 4))
    rx = radius + float(rng.uniform(-2, 3))        # slightly different x/y radii
    ry = radius + float(rng.uniform(-2, 3))        # → elliptical instead of circular

    n      = 80
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Wobble = two sine waves at incommensurate frequencies (mimics hand tremor)
    wobble = (rng.uniform(0.5, 1.0) * np.sin(3 * angles + rng.uniform(0, np.pi))
            + rng.uniform(0.3, 0.6) * np.sin(7 * angles + rng.uniform(0, np.pi)))
    wobble_scale = float(rng.uniform(0.8, 1.8))

    pts = []
    for i, a in enumerate(angles):
        s = 1.0 + wobble[i] * wobble_scale / max(rx, ry)
        x = int(np.clip(cx + rx * s * np.cos(a), 1, IMG_W - 2))
        y = int(np.clip(cy + ry * s * np.sin(a), 1, IMG_H - 2))
        pts.append((x, y))

    # Draw as short connected segments, each with a random thickness
    for i in range(len(pts)):
        cv2.line(out, pts[i], pts[(i + 1) % len(pts)],
                 color=(0, 200, 0),
                 thickness=int(rng.integers(1, 3)),
                 lineType=cv2.LINE_AA)

    # ── Arrow ─────────────────────────────────────────────────────────────────
    x1 = pick_px[0]  + int(rng.integers(-2, 3))   # jitter start
    y1 = pick_px[1]  + int(rng.integers(-2, 3))
    x2 = place_px[0] + int(rng.integers(-2, 3))   # jitter end
    y2 = place_px[1] + int(rng.integers(-2, 3))

    # Offset midpoint → gentle curve (two-segment approximation)
    mx = (x1 + x2) // 2 + int(rng.integers(-7, 8))
    my = (y1 + y2) // 2 + int(rng.integers(-7, 8))

    thick = int(rng.integers(1, 3))
    cv2.line(out, (x1, y1), (mx, my),
             color=(200, 50, 50), thickness=thick, lineType=cv2.LINE_AA)
    cv2.arrowedLine(out, (mx, my), (x2, y2),
                    color=(200, 50, 50), thickness=thick,
                    line_type=cv2.LINE_AA, tipLength=0.35)
    return out


def draw_trajectory(img, ee_pos, camera_matrix, grasp_f, release_f):
    """Overlay full EEF trajectory. Shows approach, hold, and return phases."""
    out = img.copy()
    prev = None
    for t in range(len(ee_pos)):
        px = world_to_pixel(ee_pos[t], camera_matrix, IMG_W, IMG_H)
        if px is None:
            continue
        if t < grasp_f:
            col = (255, 140, 0)    # orange — approaching object
        elif t <= release_f:
            col = (0, 210, 80)     # green  — holding object
        else:
            col = (80, 80, 255)    # blue   — returning
        if prev is not None:
            cv2.line(out, prev, px, col, 1, cv2.LINE_AA)
        if t == grasp_f:
            cv2.circle(out, px, 3, (0, 255, 0), -1)   # green dot = grasp
        if t == release_f:
            cv2.circle(out, px, 3, (0, 0, 255), -1)   # blue dot = release
        prev = px
    return out


def main():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".hdf5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in {DATA_DIR}")

    path = os.path.join(DATA_DIR, files[0])
    print(f"File  : {files[0]}\n")

    with h5py.File(path, "r") as f:
        demo           = f["data/demo_0"]
        rgb            = demo["obs/agentview_rgb"][:]     # (T, 128, 128, 3)
        ee_pos         = demo["obs/ee_pos"][:]            # (T, 3) world coords
        actions = demo["actions"][:]    # (T, 7) gripper command scalar
        states  = demo["states"][:]     # (T, 92) = [time, qpos(48), qvel(43)]
        model_file_xml = demo.attrs["model_file"]
        lang = json.loads(
            f["data"].attrs["problem_info"])["language_instruction"]

    T = len(rgb)
    print(f"Task   : {lang}")
    print(f"Frames : {T}")
    print(f"\n{'Frame':>5}  {'x':>8}  {'y':>8}  {'z':>8}  {'gripper_cmd':>12}")
    print("─" * 50)
    for t in range(0, T, 5):
        print(f"{t:>5}  {ee_pos[t,0]:>8.4f}  {ee_pos[t,1]:>8.4f}  "
              f"{ee_pos[t,2]:>8.4f}  {actions[t,-1]:>12.1f}")

    # Detect grasp / release
    grasp_f, release_f = detect_grasp_release(actions, ee_pos)
    if grasp_f is None:
        grasp_f = T // 3
        print(f"Warning: no grasp detected → using frame {grasp_f} as fallback")
    if release_f is None:
        release_f = (T * 2) // 3
        print(f"Warning: no release detected → using frame {release_f} as fallback")

    print(f"\nGrasp frame   : {grasp_f:3d}  →  ee_pos = {ee_pos[grasp_f].round(4)}")
    print(f"Release frame : {release_f:3d}  →  ee_pos = {ee_pos[release_f].round(4)}")

    # ── True object position (not EEF) via replayed MuJoCo forward kinematics ──
    # The EEF doesn't hold objects exactly at its own reference point, so using
    # ee_pos directly leaves a small but visible offset from the object's real
    # centroid. Instead: rebuild the episode's own MuJoCo model from its
    # embedded model_file XML, replay the saved qpos at the grasp/release
    # frames, and read out the actual rigid-body world position of whichever
    # free-floating object is closest to the gripper at the grasp frame (the
    # object being manipulated).
    model = build_model(model_file_xml)
    data  = mujoco.MjData(model)
    candidate_objects = free_object_bodies(model)

    # Camera pose is static/world-fixed, so any valid forward pass works --
    # use frame 0's qpos. mj_forward must be called at least once with SOME
    # valid qpos before cam_xpos/cam_xmat are populated.
    data.qpos[:] = states[0, 1:1 + model.nq]
    mujoco.mj_forward(model, data)
    camera_matrix, cam_fovy = get_camera_matrix(model, data, CAMERA_NAME, IMG_W, IMG_H)
    print(f"\nLive camera params read from compiled model ('{CAMERA_NAME}'):")
    print(f"  cam_fovy = {cam_fovy:.4f} deg")

    obj_name, pick_pos3d = identify_manipulated_object(
        model, data, states[grasp_f], ee_pos[grasp_f], candidate_objects)
    place_positions = body_positions_at(
        model, data, states[release_f, 1:1 + model.nq], [obj_name])
    place_pos3d = place_positions[obj_name]

    print(f"\nManipulated object (closest to EEF at grasp): {obj_name}")
    print(f"  object pos @ grasp   = {pick_pos3d.round(4)}   (ee_pos was {ee_pos[grasp_f].round(4)})")
    print(f"  object pos @ release = {place_pos3d.round(4)}   (ee_pos was {ee_pos[release_f].round(4)})")

    # Project TRUE object 3D positions → 2D pixels (EEF trajectory panel still
    # uses ee_pos directly -- that panel is about gripper motion, not objects).
    pick_px  = world_to_pixel(pick_pos3d,  camera_matrix, IMG_W, IMG_H)
    place_px = world_to_pixel(place_pos3d, camera_matrix, IMG_W, IMG_H)

    print(f"\nProjected pick  → pixel {pick_px}")
    print(f"Projected place → pixel {place_px}")

    if pick_px  is None: pick_px  = (64, 72);  print("Fallback pick pixel used.")
    if place_px is None: place_px = (90, 45);  print("Fallback place pixel used.")

    # Build annotated images
    base      = rgb[0]
    img_traj  = draw_trajectory(base, ee_pos, camera_matrix, grasp_f, release_f)
    img_std   = draw_standard(base, pick_px, place_px)
    img_human = draw_human(base,    pick_px, place_px)

    # ── Four-panel figure ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(17, 5))

    panels = [
        (base,      "Original Frame"),
        (img_traj,  "EEF Trajectory\n(orange=approach  green=holding  blue=return)"),
        (img_std,   "Standard Annotation\n(perfect circle + straight arrow)"),
        (img_human, "Human-like Annotation\n(wobbled circle + curved arrow)"),
    ]
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
        ax.axis("off")
        ax.plot(*pick_px,  "g+", markersize=12, markeredgewidth=2,
                label="pick  (grasp)")
        ax.plot(*place_px, "r+", markersize=12, markeredgewidth=2,
                label="place (release)")

    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f'"{lang}"', fontsize=9, style="italic", y=1.01)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "annotation_poc.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {out}")

if __name__ == "__main__":
    main()
