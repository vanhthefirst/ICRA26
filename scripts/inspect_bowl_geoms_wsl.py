"""
Diagnose the body-origin vs. visual-centroid offset for the two akita_black_bowl
bodies (and, for comparison, plate_1_main / cookies_1_main / the ramekin, which
already project correctly) -- run inside WSL2 (conda env: libero).

For each of the 5 movable objects, lists every geom attached to that body:
  - geom name
  - local pos (geom's offset from the body origin, in the body's own frame)
  - geom type + size
  - world-space geom_xpos (computed via forward kinematics)
Then computes two candidate "visual centroid" positions per body:
  1. simple mean of attached geoms' world xpos
  2. bounding-box center of attached geoms' world xpos +/- their size extents
     (rough AABB center, good enough for near-spherical/box-ish shapes)
and prints the offset of each candidate from the current body_xpos, so we can
see exactly how far off (and in which direction) the bowls' body origin is
from where a human would call "the center of the bowl".

Run from WSL2:
    conda activate libero
    python /mnt/c/Users/Admin/sketch_vla/scripts/inspect_bowl_geoms_wsl.py
"""

import os
import re
import h5py
import numpy as np

DATA_DIR = "/mnt/c/Users/Admin/sketch_vla/data/libero_spatial"
OBJECTS = [
    "akita_black_bowl_1_main",
    "akita_black_bowl_2_main",
    "cookies_1_main",
    "glazed_rim_porcelain_ramekin_1_main",
    "plate_1_main",
]


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
    env_kwargs["camera_heights"] = 128
    env_kwargs["camera_widths"] = 128

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

    model = robo_env.sim.model
    data = robo_env.sim.data

    for body_name in OBJECTS:
        body_id = model.body_name2id(body_name)
        body_pos = data.body_xpos[body_id].copy()
        print("=" * 90)
        print(f"{body_name}  (body_id={body_id})")
        print(f"  body_xpos (world) = {body_pos.round(4)}")

        # Find every geom whose parent body is this body.
        geom_ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]
        if not geom_ids:
            print("  (no geoms directly on this body -- may be parented under a child body)")
            continue

        world_positions = []
        for g in geom_ids:
            gname = model.geom_id2name(g) or f"geom_{g}"
            local_pos = model.geom_pos[g].copy()
            gtype = model.geom_type[g]
            gsize = model.geom_size[g].copy()
            world_pos = data.geom_xpos[g].copy()
            contype = model.geom_contype[g]
            conaffinity = model.geom_conaffinity[g]
            print(f"  geom '{gname}': type={gtype} size={gsize.round(4)} "
                  f"local_pos={local_pos.round(4)} world_pos={world_pos.round(4)} "
                  f"contype={contype} conaffinity={conaffinity}")
            world_positions.append(world_pos)

        world_positions = np.array(world_positions)
        mean_centroid = world_positions.mean(axis=0)
        bbox_min = world_positions.min(axis=0)
        bbox_max = world_positions.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2.0

        print(f"  --> mean-of-geoms centroid = {mean_centroid.round(4)}   "
              f"offset from body_xpos = {(mean_centroid - body_pos).round(4)}")
        print(f"  --> bbox-of-geoms   center = {bbox_center.round(4)}   "
              f"offset from body_xpos = {(bbox_center - body_pos).round(4)}")
        print()

    print("=" * 90)
    print("Done. Compare the 'offset from body_xpos' rows above: for plate/cookies/")
    print("ramekin (already visually centered) this offset should be small/near-zero;")
    print("for the bowls, whichever candidate has the LARGEST offset in the expected")
    print("direction is the one that explains what you're seeing on screen.")


if __name__ == "__main__":
    main()
