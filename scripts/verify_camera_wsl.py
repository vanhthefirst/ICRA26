"""
Camera calibration verification — run inside WSL2 (conda env: libero).

Loads demo_0 of the same HDF5 that annotate_poc.py uses, builds the REAL
robosuite/LIBERO env (real meshes/textures this time, not the dummy assets
from mujoco_object_state.py), replays the episode's own model_file + state
at frame 0, and:

  1. Reads MuJoCo's actual 'agentview' camera pose (cam_xpos / cam_xmat /
     cam_fovy) straight from the compiled model.
  2. Converts that into the same (eye, lookat, up, fov) representation used
     by CAM_EYE / CAM_LOOKAT / CAM_UP / CAM_FOV in annotate_poc.py, and
     diffs them directly.
  3. Renders frame 0 offscreen and saves it next to the frame actually
     stored in the HDF5 (obs/agentview_rgb[0]), plus a pixel-diff, so you
     can compare by eye and by number.

Run from WSL2:
    conda activate libero
    cd ~/LIBERO   # or anywhere; paths below are absolute
    python /mnt/c/Users/Admin/sketch_vla/scripts/verify_camera_wsl.py
"""

import os
import re
import json
import h5py
import numpy as np

DATA_DIR   = "/mnt/c/Users/Admin/sketch_vla/data/libero_spatial"
OUTPUT_DIR = "/mnt/c/Users/Admin/sketch_vla/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_W, IMG_H = 128, 128

# Current hand-derived values from annotate_poc.py — edit here if you change them there.
CAM_EYE_HAND    = np.array([0.658613, 0.000000, 1.610350])
CAM_LOOKAT_HAND = np.array([-0.119385, 0.000000, 0.982084])
CAM_UP_HAND     = np.array([0.0, 1.0, 0.0])
CAM_FOV_HAND    = 45.0


def main():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".hdf5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in {DATA_DIR}")
    path = os.path.join(DATA_DIR, files[0])
    print(f"File: {files[0]}\n")

    with h5py.File(path, "r") as f:
        demo = f["data/demo_0"]
        rgb0 = demo["obs/agentview_rgb"][0]           # (128,128,3) stored frame
        states = demo["states"][:]
        model_file_xml = demo.attrs["model_file"]
        if isinstance(model_file_xml, bytes):
            model_file_xml = model_file_xml.decode("utf-8")

        env_args_raw = f["data"].attrs.get("env_args", None)

    # ── Build the real robosuite/LIBERO env ─────────────────────────────────
    # robomimic's EnvRobosuite hard-imports legacy mujoco_py (not installed —
    # this setup uses the native mujoco bindings robosuite 1.4 ships with),
    # so we build the env directly via robosuite.make() instead, and fix up
    # any stale absolute asset paths in the demo's model_file XML ourselves
    # by resolving broken file="..." references against the local robosuite
    # package and LIBERO repo, indexed by filename.
    import libero.libero.envs
    import robosuite
    import robomimic.utils.file_utils as FileUtils  # only used for metadata parsing, no mujoco_py needed here

    bddl_path = (
        "/root/LIBERO/libero/libero/bddl_files/libero_spatial/"
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl"
    )

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=path)
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["bddl_file_name"] = bddl_path
    env_kwargs["camera_names"] = ["agentview"]
    env_kwargs["camera_heights"] = IMG_H
    env_kwargs["camera_widths"] = IMG_W
    print(f"env_name: {env_meta.get('env_name')}")
    print(f"env_kwargs: {env_kwargs}\n")

    robo_env = robosuite.make(env_meta["env_name"], **env_kwargs)
    robo_env.reset()

    # ── Build a filename → real local path index (robosuite assets + LIBERO repo) ──
    def build_asset_index(roots):
        index = {}
        for root in roots:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d != ".git"]
                for fn in filenames:
                    index.setdefault(fn, os.path.join(dirpath, fn))
        return index

    asset_roots = [os.path.dirname(robosuite.__file__), "/root/LIBERO"]
    print(f"Indexing local assets under: {asset_roots} ...")
    asset_index = build_asset_index(asset_roots)
    print(f"Indexed {len(asset_index)} files.\n")

    def postprocess_model_xml(xml_str):
        unresolved = []

        def fix(match):
            path = match.group(1)
            if os.path.exists(path):
                return f'file="{path}"'
            base = os.path.basename(path)
            new_path = asset_index.get(base)
            if new_path:
                return f'file="{new_path}"'
            unresolved.append(path)
            return match.group(0)

        xml_str = re.sub(r'file="([^"]+)"', fix, xml_str)
        if unresolved:
            print(f"WARNING: {len(unresolved)} asset path(s) could not be resolved locally:")
            for p in unresolved[:10]:
                print(f"    {p}")
        return xml_str

    # Replay this exact demo's model + frame-0 state (real object placement,
    # real assets — no dummy substitution this time).
    xml = postprocess_model_xml(model_file_xml)
    robo_env.reset_from_xml_string(xml)
    robo_env.sim.set_state_from_flattened(states[0])
    robo_env.sim.forward()

    # ── 1. Real camera pose straight from MuJoCo ────────────────────────────
    cam_id = robo_env.sim.model.camera_name2id("agentview")
    cam_xpos = robo_env.sim.data.cam_xpos[cam_id].copy()
    cam_xmat = robo_env.sim.data.cam_xmat[cam_id].reshape(3, 3).copy()
    cam_fovy = float(robo_env.sim.model.cam_fovy[cam_id])

    # MuJoCo camera-frame convention: columns of cam_xmat are the camera's
    # local (right, up, backward) axes expressed in world coordinates.
    # The camera looks down its own -z, so forward = -backward.
    right_world   = cam_xmat[:, 0]
    up_world      = cam_xmat[:, 1]
    backward_world = cam_xmat[:, 2]
    forward_world = -backward_world

    cam_eye_real    = cam_xpos
    cam_lookat_real = cam_xpos + forward_world   # unit-distance target, matches annotate_poc's convention
    cam_up_real     = up_world
    cam_fov_real    = cam_fovy

    print("=" * 70)
    print("CAMERA PARAM COMPARISON  (real MuJoCo  vs  hand-derived in annotate_poc.py)")
    print("=" * 70)
    print(f"EYE      real = {cam_eye_real.round(6)}")
    print(f"         hand = {CAM_EYE_HAND.round(6)}")
    print(f"         diff = {(cam_eye_real - CAM_EYE_HAND).round(6)}  (norm={np.linalg.norm(cam_eye_real - CAM_EYE_HAND):.6f})\n")

    print(f"LOOKAT   real = {cam_lookat_real.round(6)}")
    print(f"         hand = {CAM_LOOKAT_HAND.round(6)}")
    fwd_hand = (CAM_LOOKAT_HAND - CAM_EYE_HAND); fwd_hand /= np.linalg.norm(fwd_hand)
    cos_angle = np.clip(np.dot(forward_world, fwd_hand), -1, 1)
    print(f"         forward-direction angle diff = {np.degrees(np.arccos(cos_angle)):.3f} deg\n")

    print(f"UP       real = {cam_up_real.round(6)}")
    print(f"         hand = {CAM_UP_HAND.round(6)}")
    cos_up = np.clip(np.dot(cam_up_real / np.linalg.norm(cam_up_real), CAM_UP_HAND), -1, 1)
    print(f"         up-direction angle diff = {np.degrees(np.arccos(cos_up)):.3f} deg\n")

    print(f"FOV      real = {cam_fov_real:.4f}")
    print(f"         hand = {CAM_FOV_HAND:.4f}")
    print(f"         diff = {cam_fov_real - CAM_FOV_HAND:.4f} deg\n")

    # Also print right_world for completeness (helps if up/lookat look swapped)
    print(f"(right_world = {right_world.round(6)}, for reference)\n")

    # ── 2. Real offscreen render vs stored HDF5 frame ───────────────────────
    obs = robo_env._get_observations(force_update=True)
    rendered = np.asarray(obs["agentview_image"])
    if rendered.dtype != np.uint8:
        rendered = np.clip(rendered * 255.0 if rendered.max() <= 1.0 + 1e-6 else rendered, 0, 255).astype(np.uint8)

    # robosuite/mujoco offscreen renders come out flipped vertically relative
    # to how they're typically stored after collection -- test both and
    # report which orientation actually matches, rather than assuming.
    rendered_flip = np.flipud(rendered)

    def mse(a, b):
        return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))

    mse_asis = mse(rendered, rgb0)
    mse_flip = mse(rendered_flip, rgb0)
    print("=" * 70)
    print("RENDER COMPARISON  (frame 0, offscreen render vs stored obs/agentview_rgb[0])")
    print("=" * 70)
    print(f"MSE as-is        = {mse_asis:.2f}")
    print(f"MSE flipped(v)   = {mse_flip:.2f}")
    best = "as-is" if mse_asis <= mse_flip else "flipped vertically"
    print(f"→ Rendered image matches stored frame ({best}).\n")

    # Save a side-by-side PNG for visual inspection.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_rendered = rendered if mse_asis <= mse_flip else rendered_flip
    diff_img = np.abs(best_rendered.astype(np.int16) - rgb0.astype(np.int16)).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, im, title in zip(
        axes,
        [rgb0, best_rendered, diff_img],
        ["Stored HDF5 frame 0", f"Real MuJoCo render ({best})", "Abs diff"],
    ):
        ax.imshow(im)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "camera_verification.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison image → {out_path}")


if __name__ == "__main__":
    main()
