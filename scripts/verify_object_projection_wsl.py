"""
Ground-truth object projection check — run inside WSL2 (conda env: libero).

Uses robosuite's own tested projection utilities (robosuite.utils.camera_utils)
instead of any hand-rolled view-matrix code, and reads the manipulated object's
position directly from the REAL environment (real assets, real body tree) at
the grasp/release frames -- no dummy mujoco_object_state.py reconstruction
involved. Draws the projected pixel directly on the REAL rendered frame for
that exact state, so there is no ambiguity left: if the marker lands on the
bowl in this image, both the camera math and the object identification are
confirmed correct simultaneously.

Run from WSL2:
    conda activate libero
    python /mnt/c/Users/Admin/sketch_vla/scripts/verify_object_projection_wsl.py
"""

import os
import re
import h5py
import numpy as np

DATA_DIR   = "/mnt/c/Users/Admin/sketch_vla/data/libero_spatial"
OUTPUT_DIR = "/mnt/c/Users/Admin/sketch_vla/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_W, IMG_H = 128, 128


def detect_grasp_release(actions):
    g = actions[:, -1]
    T = len(g)
    grasp_f = release_f = None
    for i in range(1, T):
        if g[i - 1] < 0 and g[i] > 0 and grasp_f is None:
            grasp_f = i
        if g[i - 1] > 0 and g[i] < 0 and grasp_f is not None:
            release_f = i
            break
    return grasp_f, release_f


def main():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".hdf5"))
    path = os.path.join(DATA_DIR, files[0])
    print(f"File: {files[0]}\n")

    with h5py.File(path, "r") as f:
        demo = f["data/demo_0"]
        rgb = demo["obs/agentview_rgb"][:]
        ee_pos = demo["obs/ee_pos"][:]
        actions = demo["actions"][:]
        states = demo["states"][:]
        model_file_xml = demo.attrs["model_file"]
        if isinstance(model_file_xml, bytes):
            model_file_xml = model_file_xml.decode("utf-8")

    grasp_f, release_f = detect_grasp_release(actions)
    print(f"grasp_f={grasp_f}  release_f={release_f}")

    import libero.libero.envs
    import robosuite
    import robomimic.utils.file_utils as FileUtils
    from robosuite.utils import camera_utils as CU

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

    robo_env = robosuite.make(env_meta["env_name"], **env_kwargs)
    robo_env.reset()

    def build_asset_index(roots):
        index = {}
        for root in roots:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d != ".git"]
                for fn in filenames:
                    index.setdefault(fn, os.path.join(dirpath, fn))
        return index

    asset_index = build_asset_index([os.path.dirname(robosuite.__file__), "/root/LIBERO"])

    def postprocess_model_xml(xml_str):
        def fix(m):
            p = m.group(1)
            if os.path.exists(p):
                return f'file="{p}"'
            new_p = asset_index.get(os.path.basename(p))
            return f'file="{new_p}"' if new_p else m.group(0)
        return re.sub(r'file="([^"]+)"', fix, xml_str)

    xml = postprocess_model_xml(model_file_xml)
    robo_env.reset_from_xml_string(xml)

    # World-to-pixel transform for 'agentview' -- computed once via robosuite's
    # own tested utility (camera pose is static across frames/resets).
    world_to_pixel_tf = CU.get_camera_transform_matrix(
        sim=robo_env.sim, camera_name="agentview",
        camera_height=IMG_H, camera_width=IMG_W,
    )

    def visual_center_world_pos(model, data, body_id):
        # FIX (this session, see inspect_bowl_geoms_wsl.py): use the world
        # position of the body's contype=0/conaffinity=0 reference geom (its
        # actual visual mesh) instead of raw body_xpos, which for the bowls
        # sits at the bottom of the object rather than its center.
        geom_ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]
        ref_geoms = [g for g in geom_ids
                     if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
        if ref_geoms:
            return data.geom_xpos[ref_geoms[0]].copy()
        return data.body_xpos[body_id].copy()

    def render_and_project(frame_idx, body_name, label):
        robo_env.sim.set_state_from_flattened(states[frame_idx])
        robo_env.sim.forward()

        body_id = robo_env.sim.model.body_name2id(body_name)
        obj_pos = visual_center_world_pos(robo_env.sim.model, robo_env.sim.data, body_id)

        pix = CU.project_points_from_world_to_camera(
            points=obj_pos[None, :],
            world_to_camera_transform=world_to_pixel_tf,
            camera_height=IMG_H, camera_width=IMG_W,
        )[0]
        # project_points_from_world_to_camera returns (row, col)
        row, col = int(pix[0]), int(pix[1])
        # FIX (this session): the returned row is vertically mirrored for
        # this camera/robosuite/mujoco combination -- confirmed numerically
        # against the MuJoCo-rendered marker ground truth
        # (verify_markers_in_scene_wsl.py): printed rows matched exactly
        # (IMG_H-1)-row of the true positions. The frame itself is already
        # correctly oriented (do NOT flip it -- that was tried and it just
        # makes the picture upside-down while coincidentally re-aligning
        # with the still-wrong row value). Mirror the row back instead.
        row = (IMG_H - 1) - row

        obs = robo_env._get_observations(force_update=True)
        frame = np.asarray(obs["agentview_image"])
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255.0 if frame.max() <= 1.0 + 1e-6 else frame, 0, 255).astype(np.uint8)

        print(f"{label}: frame={frame_idx}  body={body_name}  world_pos={obj_pos.round(4)}  -> pixel (col={col}, row={row})")
        return frame, (col, row), obj_pos

    frame_grasp, px_grasp, pos_grasp = render_and_project(grasp_f, "akita_black_bowl_1_main", "GRASP")
    frame_release, px_release, pos_release = render_and_project(release_f, "akita_black_bowl_1_main", "RELEASE")
    frame0, px0, pos0 = render_and_project(0, "akita_black_bowl_1_main", "FRAME0(rest)")

    # Cross-check hypothesis: is the grasp-frame object position projected
    # onto the FRAME-0 image (mismatch, what annotate_poc.py currently does)
    # actually off the visible bowl -- vs. projecting the frame-0 position
    # onto frame-0's own image (should land cleanly if states are consistent)?
    px_grasp_on_frame0 = CU.project_points_from_world_to_camera(
        points=pos_grasp[None, :], world_to_camera_transform=world_to_pixel_tf,
        camera_height=IMG_H, camera_width=IMG_W,
    )[0]
    # Same row-mirror fix as in render_and_project() above.
    col_grasp_on_frame0 = int(px_grasp_on_frame0[1])
    row_grasp_on_frame0 = (IMG_H - 1) - int(px_grasp_on_frame0[0])
    print(f"\n[cross-check] grasp-frame object pos projected -> pixel (col={col_grasp_on_frame0}, row={row_grasp_on_frame0})  (would be drawn on FRAME 0 image by current annotate_poc.py)")
    print(f"[cross-check] frame-0 object pos projected        -> pixel (col={px0[0]}, row={px0[1]})  (drawn on FRAME 0 image, own frame -- should land on the bowl if consistent)")

    # Also project ee_pos at the same frames, straight from the HDF5, for a
    # sanity cross-check against the object position.
    pix_ee_grasp = CU.project_points_from_world_to_camera(
        points=ee_pos[grasp_f][None, :], world_to_camera_transform=world_to_pixel_tf,
        camera_height=IMG_H, camera_width=IMG_W,
    )[0]
    col_ee_grasp = int(pix_ee_grasp[1])
    row_ee_grasp = (IMG_H - 1) - int(pix_ee_grasp[0])
    print(f"ee_pos@grasp {ee_pos[grasp_f].round(4)} -> pixel (col={col_ee_grasp}, row={row_ee_grasp})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, frame, px, title in [
        (axes[0], frame0, px0, f"FRAME 0 (rest)\nown-frame pos {pos0.round(3)}"),
        (axes[1], frame0, (col_grasp_on_frame0, row_grasp_on_frame0),
         "FRAME 0 image\n+ GRASP-frame object pos\n(what annotate_poc.py currently does)"),
        (axes[2], frame_grasp, px_grasp, f"GRASP frame {grasp_f}\nown-frame pos {pos_grasp.round(3)}"),
        (axes[3], frame_release, px_release, f"RELEASE frame {release_f}\nown-frame pos {pos_release.round(3)}"),
    ]:
        ax.imshow(frame)
        ax.plot(*px, "g+", markersize=16, markeredgewidth=2)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "object_projection_check.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
