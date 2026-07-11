"""
Project ALL named scene objects onto frame 0, labeled -- run inside WSL2.

Decisive test: plate and ramekin are big, unambiguous, easy to eyeball. If
those land exactly on their visible blobs, the camera + projection pipeline
is fully vindicated and any remaining issue is specifically about which body
name corresponds to "the bowl actually being grasped" (there are two:
akita_black_bowl_1 and akita_black_bowl_2). If even the plate is off, the
problem is upstream of object identification.

Run from WSL2:
    conda activate libero
    python /mnt/c/Users/Admin/sketch_vla/scripts/verify_all_objects_wsl.py
"""

import os
import re
import h5py
import numpy as np

DATA_DIR   = "/mnt/c/Users/Admin/sketch_vla/data/libero_spatial"
OUTPUT_DIR = "/mnt/c/Users/Admin/sketch_vla/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_W, IMG_H = 128, 128


def main():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".hdf5"))
    path = os.path.join(DATA_DIR, files[0])
    print(f"File: {files[0]}\n")

    with h5py.File(path, "r") as f:
        demo = f["data/demo_0"]
        states = demo["states"][:]
        model_file_xml = demo.attrs["model_file"]
        if isinstance(model_file_xml, bytes):
            model_file_xml = model_file_xml.decode("utf-8")

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
    robo_env.sim.set_state_from_flattened(states[0])
    robo_env.sim.forward()

    world_to_pixel_tf = CU.get_camera_transform_matrix(
        sim=robo_env.sim, camera_name="agentview",
        camera_height=IMG_H, camera_width=IMG_W,
    )

    # Every free-jointed (movable) body in the scene.
    body_names = []
    for j in range(robo_env.sim.model.njnt):
        if robo_env.sim.model.jnt_type[j] == 0:  # mjJNT_FREE == 0
            body_id = robo_env.sim.model.jnt_bodyid[j]
            body_names.append(robo_env.sim.model.body_id2name(body_id))
    print("Free bodies found:", body_names, "\n")

    obs = robo_env._get_observations(force_update=True)
    frame = np.asarray(obs["agentview_image"])
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0 if frame.max() <= 1.0 + 1e-6 else frame, 0, 255).astype(np.uint8)
    # NOTE (this session): the frame itself does NOT need flipping -- it was
    # already correctly oriented (confirmed: flipping it "fixed" marker
    # alignment only by coincidentally mirroring the image to match an
    # already-wrong row value, which made the picture visually upside-down).
    # The real bug is in the projected ROW (see the +1 correction below).

    def visual_center_world_pos(model, data, body_id):
        # FIX (this session, see inspect_bowl_geoms_wsl.py): body_xpos is
        # wherever the asset author put the body/free-joint origin, which
        # for the akita_black_bowl_* objects is the BOTTOM of the bowl, not
        # its visual center (~half the bowl's height off). Each object body
        # has exactly one "reference" geom with contype=0 and conaffinity=0
        # (a MESH-type geom for the object's actual visual volume); that
        # geom's own world position is what actually sits at the object's
        # visual center. Falls back to raw body_xpos if none found.
        geom_ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]
        ref_geoms = [g for g in geom_ids
                     if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
        if ref_geoms:
            return data.geom_xpos[ref_geoms[0]].copy()
        return data.body_xpos[body_id].copy()

    results = []
    for name in body_names:
        body_id = robo_env.sim.model.body_name2id(name)
        pos = visual_center_world_pos(robo_env.sim.model, robo_env.sim.data, body_id)
        pix = CU.project_points_from_world_to_camera(
            points=pos[None, :], world_to_camera_transform=world_to_pixel_tf,
            camera_height=IMG_H, camera_width=IMG_W,
        )[0]
        row, col = int(pix[0]), int(pix[1])
        # FIX (this session): project_points_from_world_to_camera's returned
        # row is vertically mirrored for this camera/robosuite/mujoco
        # version combination -- confirmed numerically against the MuJoCo-
        # rendered marker ground truth (verify_markers_in_scene_wsl.py):
        # printed rows matched exactly (IMG_H-1)-row of the true positions,
        # and that corrected ordering matched each object's real
        # depth-from-camera order exactly. The frame is fine as-is; only
        # the analytic row needs mirroring back.
        row = (IMG_H - 1) - row
        print(f"{name:35s} world_pos={pos.round(4)}  -> pixel (col={col}, row={row})")
        results.append((name, (col, row)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(frame)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for (name, (col, row)), c in zip(results, colors):
        ax.plot(col, row, "+", color=c, markersize=14, markeredgewidth=2)
        ax.annotate(name.replace("_main", ""), (col, row), color=c, fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.axis("off")
    ax.set_title("All free bodies projected onto frame 0", fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "all_objects_projection.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
