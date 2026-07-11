"""
Decisive ground-truth test — run inside WSL2 (conda env: libero).

Instead of trusting any analytic 3D->2D projection formula (ours or
robosuite.utils.camera_utils -- both under suspicion after the last test),
inject small colored sphere markers directly into the MuJoCo scene XML at
each object's real body_xpos, then let MuJoCo's OWN renderer draw them --
the exact same rendering pipeline that already proved pixel-perfect against
the stored HDF5 frame (MSE=44.65). If the markers don't land exactly on
their objects, the object POSITIONS themselves must be wrong (not the
projection math, which is bypassed entirely here). If they land exactly on
target, positions are correct and camera_utils' projection function is the
culprit.

FIXED THIS SESSION: markers previously compiled into the model at the
correct xpos but never appeared in the render. Root cause was NOT a
mjvScene/maxgeom sizing issue (robosuite's reset_from_xml_string() builds a
brand-new MjSim + fresh offscreen render context every call, verified
against robosuite/environments/base.py -- so the context is always sized
for the current model). The real cause: MuJoCo geoms with no explicit
group= attribute default to group 0, and robosuite's _reset_internal() sets
vopt.geomgroup[0] = 1 if render_collision_mesh else 0 (default False ->
group 0 HIDDEN) and vopt.geomgroup[1] = 1 if render_visual_mesh else 0
(default True -> shown). Our marker geoms had no group=, so they silently
landed in the hidden "collision mesh" bucket -- no error, nothing rendered.
Fix: markers now set group="1" explicitly (see marker_xml_blocks below).
We also print sim.model.ngeom and the render context's maxgeom (if exposed
by this mujoco binding) purely as a sanity diagnostic, in case a second,
independent issue is also present.

Run from WSL2:
    conda activate libero
    python /mnt/c/Users/Admin/sketch_vla/scripts/verify_markers_in_scene_wsl.py
"""

import os
import re
import h5py
import numpy as np

DATA_DIR   = "/mnt/c/Users/Admin/sketch_vla/data/libero_spatial"
OUTPUT_DIR = "/mnt/c/Users/Admin/sketch_vla/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_W, IMG_H = 128, 128

MARKER_COLORS = {
    "akita_black_bowl_1_main":             "1 0 0 1",    # red
    "akita_black_bowl_2_main":             "0 1 0 1",    # green
    "cookies_1_main":                      "0 0 1 1",    # blue
    "glazed_rim_porcelain_ramekin_1_main": "1 1 0 1",    # yellow
    "plate_1_main":                        "1 0 1 1",    # magenta
}


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

    # ── First pass: load without markers, replay frame 0, read real body positions ──
    robo_env.reset_from_xml_string(xml)
    robo_env.sim.set_state_from_flattened(states[0])
    robo_env.sim.forward()

    def visual_center_world_pos(model, data, body_id):
        # FIX (this session, see inspect_bowl_geoms_wsl.py): use the world
        # position of the body's contype=0/conaffinity=0 reference geom (its
        # actual visual mesh) instead of raw body_xpos, which for the bowls
        # sits at the bottom of the object rather than its center. Applying
        # this here too keeps this script's ground truth consistent with
        # verify_all_objects_wsl.py / verify_object_projection_wsl.py.
        geom_ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]
        ref_geoms = [g for g in geom_ids
                     if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
        if ref_geoms:
            return data.geom_xpos[ref_geoms[0]].copy()
        return data.body_xpos[body_id].copy()

    positions = {}
    for name in MARKER_COLORS:
        body_id = robo_env.sim.model.body_name2id(name)
        positions[name] = visual_center_world_pos(robo_env.sim.model, robo_env.sim.data, body_id)
        print(f"{name:35s} world_pos={positions[name].round(4)}")

    # ── Second pass: inject a fixed (no-joint) sphere body at each position,
    # recompile, replay the SAME frame-0 state, and render. Injected bodies
    # add no new DOFs, so state indices are unaffected. ──
    #
    # ROOT CAUSE (found this session): MuJoCo geoms without an explicit
    # `group=` attribute default to group 0. robosuite's _reset_internal()
    # sets the offscreen render context's visibility mask via
    #   vopt.geomgroup[0] = 1 if render_collision_mesh else 0   (default: 0 -> HIDDEN)
    #   vopt.geomgroup[1] = 1 if render_visual_mesh    else 0   (default: 1 -> shown)
    # i.e. group 0 is robosuite's "collision mesh" bucket and is invisible by
    # default; group 1 is the "visual mesh" bucket. Our injected marker geoms
    # had no group= attribute -> silently defaulted to group 0 -> silently
    # culled from the render, with zero error, exactly matching what we saw
    # (bodies compile in, xpos is correct, but nothing appears on screen).
    # Fix: explicitly put markers in group="1" (the visual bucket).
    marker_xml_blocks = []
    for name, pos in positions.items():
        color = MARKER_COLORS[name]
        x, y, z = pos
        marker_xml_blocks.append(
            f'<body name="marker_{name}" pos="{x} {y} {z}">'
            f'<geom type="sphere" size="0.012" rgba="{color}" '
            f'group="1" contype="0" conaffinity="0"/>'
            f'</body>'
        )
    markers_str = "\n".join(marker_xml_blocks)
    n_occurrences = xml.count("</worldbody>")
    print(f"\nDIAG: '</worldbody>' occurrences in xml = {n_occurrences}")
    xml_with_markers = xml.replace("</worldbody>", markers_str + "\n</worldbody>")
    print(f"DIAG: xml length before={len(xml)}  after={len(xml_with_markers)}  (should differ by ~{len(markers_str)})")

    robo_env.reset_from_xml_string(xml_with_markers)
    print(f"DIAG: model.nbody after reset = {robo_env.sim.model.nbody}")
    print(f"DIAG: model.ngeom after reset  = {robo_env.sim.model.ngeom}")
    # Sanity-check for a stale/undersized render scene buffer, in case that's
    # ALSO a factor on top of the group= bug. Attribute name/location varies
    # by mujoco binding version, so probe a few plausible spots and don't
    # fail if none exist.
    ctx = robo_env.sim._render_context_offscreen
    maxgeom = None
    for attr_path in ("scn.maxgeom", "con.maxgeom"):
        obj = ctx
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            maxgeom = obj
            print(f"DIAG: render context {attr_path} = {maxgeom}")
        except AttributeError:
            continue
    if maxgeom is None:
        print("DIAG: could not locate a maxgeom attribute on the render context "
              "(binding-dependent) -- not necessarily a problem, just couldn't probe it.")
    elif robo_env.sim.model.ngeom > maxgeom:
        print(f"DIAG: *** ngeom ({robo_env.sim.model.ngeom}) > maxgeom ({maxgeom}) "
              f"-- scene buffer IS undersized, geoms will be silently dropped! ***")
    else:
        print(f"DIAG: ngeom ({robo_env.sim.model.ngeom}) <= maxgeom ({maxgeom}) -- buffer size is fine.")

    for name in MARKER_COLORS:
        marker_body = f"marker_{name}"
        try:
            bid = robo_env.sim.model.body_name2id(marker_body)
            mpos = robo_env.sim.data.body_xpos[bid].copy()
            print(f"DIAG: {marker_body} exists, body_id={bid}, xpos(pre-forward)={mpos.round(4)}")
        except Exception as e:
            print(f"DIAG: {marker_body} NOT FOUND -- {e}")

    robo_env.sim.set_state_from_flattened(states[0])
    robo_env.sim.forward()

    for name in MARKER_COLORS:
        marker_body = f"marker_{name}"
        bid = robo_env.sim.model.body_name2id(marker_body)
        mpos = robo_env.sim.data.body_xpos[bid].copy()
        print(f"DIAG: {marker_body} xpos(post-forward)={mpos.round(4)}  target_was={positions[name].round(4)}")

    obs = robo_env._get_observations(force_update=True)
    frame = np.asarray(obs["agentview_image"])
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0 if frame.max() <= 1.0 + 1e-6 else frame, 0, 255).astype(np.uint8)
    # NOTE (this session): confirmed the frame does NOT need flipping -- it
    # was already correctly oriented, which is exactly why the markers (drawn
    # directly by MuJoCo's own rasterizer, bypassing analytic projection
    # entirely) landed correctly here with no adjustment. The bug turned out
    # to be a vertically-mirrored row in camera_utils.project_points_from_
    # world_to_camera's output specifically -- see the row-mirror fix in
    # verify_all_objects_wsl.py / verify_object_projection_wsl.py. This
    # marker frame was the ground truth that exposed it.

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(frame)
    ax.axis("off")
    ax.set_title("Markers rendered BY MUJOCO ITSELF at each body's real xpos\n"
                 "red=bowl1 green=bowl2 blue=cookies yellow=ramekin magenta=plate", fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "markers_in_scene.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
