"""
Reusable helper: reconstruct a compilable MuJoCo model from a LIBERO/robosuite
episode's embedded `model_file` XML string (HDF5 demo attr), so we can replay
any frame's saved `states` (full sim qpos+qvel) and read out TRUE object body
positions -- not just the end-effector position.

Why this exists: the raw model_file XML references absolute mesh/texture
paths from the original authors' machines (e.g. /Users/yifengz/workspace/...)
which don't exist on this machine. None of that geometry affects the
quantity we actually need (rigid body world positions), so we:
  1. Strip purely-visual mesh geoms (contype=0 conaffinity=0, name has
     "_vis") that reference .obj/.msh files MuJoCo can't easily fake.
  2. Rewrite remaining asset file="" paths to just their basename.
  3. Supply dummy-but-valid STL (tetrahedron) / PNG (1x1) bytes for every
     remaining referenced asset via MuJoCo's in-memory asset dict, so the
     model compiles without touching disk.
This preserves the full kinematic tree (joint order, body nesting) exactly
as recorded, so xpos/qpos indices line up correctly.
"""

import re
import struct
import base64
import os
import numpy as np
import mujoco


# Known local-frame offset (in the object body's own coordinate frame, not
# world) from the body/free-joint origin to the object's true visual center
# -- i.e. each object's "*_g0" geom's local_pos, measured once against the
# REAL robosuite/WSL2 env via inspect_bowl_geoms_wsl.py, where that geom
# survives untouched (unlike here, where build_model() strips it -- see
# visual_center_world_pos() below for why). These offsets are fixed
# properties of the underlying mesh asset, not of any particular demo/frame,
# so they're safe to reuse across episodes as long as the object TYPE
# matches. Keyed by the body name with its "_<N>_main" instance suffix
# stripped (e.g. "akita_black_bowl_1_main" -> "akita_black_bowl").
KNOWN_VISUAL_CENTER_LOCAL_OFFSET = {
    "akita_black_bowl":              np.array([0.0004, -0.0003, 0.0284]),
    "cookies":                       np.array([-0.0001, 0.0005, 0.0003]),
    "glazed_rim_porcelain_ramekin":  np.array([0.0, -0.0002, 0.0202]),
    "plate":                         np.array([0.0001, -0.0001, 0.0083]),
}


def _object_type_from_body_name(body_name: str) -> str:
    """'akita_black_bowl_1_main' -> 'akita_black_bowl'; 'plate_1_main' -> 'plate'."""
    name = re.sub(r"_main$", "", body_name)
    name = re.sub(r"_\d+$", "", name)
    return name


def _dummy_stl_tetrahedron():
    v0, v1, v2, v3 = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
    faces = [(v0, v2, v1), (v0, v1, v3), (v0, v3, v2), (v1, v2, v3)]
    header = b"\x00" * 80
    count = struct.pack("<I", len(faces))
    body = b"".join(
        struct.pack("<12fH", 0, 0, 0, *a, *b, *c, 0) for a, b, c in faces
    )
    return header + count + body


_DUMMY_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY"
    "42YAAAAASUVORK5CYII="
)


def build_model(model_file_xml: str) -> mujoco.MjModel:
    """
    Given the raw model_file XML string stored in an episode's HDF5
    (data/demo_N attrs['model_file']), return a compiled MjModel with all
    kinematics/joints intact but visual mesh assets faked out.
    """
    xml = model_file_xml

    # Drop <mesh> assets that point at .obj/.msh files (high-poly visual
    # meshes MuJoCo's native STL loader can't read) ...
    xml = re.sub(r'<mesh[^>]*file="[^"]*\.(obj|msh)"[^>]*/>\s*', "", xml)
    # ... and the <geom mesh="..._vis...".../> elements that reference them.
    # These are always visual-only (contype="0" conaffinity="0") in LIBERO
    # scenes, so removing them does not change any physics/kinematics.
    xml = re.sub(r'<geom[^>]*mesh="[^"]*_vis[^"]*"[^>]*/>\s*', "", xml)

    # Collapse remaining asset paths to their basename (MuJoCo's VFS/asset
    # dict keys on basename), and collect the unique set.
    files = sorted(set(re.findall(r'file="([^"]+)"', xml)))
    xml = re.sub(r'file="([^"]+)"', lambda m: f'file="{os.path.basename(m.group(1))}"', xml)

    assets = {}
    for fpath in files:
        key = os.path.basename(fpath)
        if key.lower().endswith(".stl"):
            assets[key] = _dummy_stl_tetrahedron()
        elif key.lower().endswith(".png"):
            assets[key] = _DUMMY_PNG_1X1

    return mujoco.MjModel.from_xml_string(xml, assets)


def free_object_bodies(model: mujoco.MjModel):
    """
    Names of all bodies whose (own) joint is a free joint -- i.e. movable
    rigid objects in the scene (bowls, plates, cabinets, etc.), as opposed
    to the robot arm (hinge joints) or the static table/walls (no joint).
    """
    names = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            body_id = model.jnt_bodyid[j]
            names.append(model.body(body_id).name)
    return names


def visual_center_world_pos(model: mujoco.MjModel, data: mujoco.MjData, body_id: int,
                             body_name: str = None):
    """
    Return the world position of an object's true visual/volumetric center,
    not its raw body origin (data.xpos).

    Found this session (via inspect_bowl_geoms_wsl.py, run against the REAL
    robosuite/WSL2 env, not this dummy reconstruction): for these LIBERO
    object assets, body_xpos is wherever the asset author put the body/free-
    joint origin -- for the akita_black_bowl_* objects specifically, that's
    the BOTTOM of the bowl, not its center, offset by ~half the bowl's
    height. Each object body has exactly one "reference" geom with
    contype=0 and conaffinity=0 (a MESH-type geom for the object's actual
    visual bounding volume); that geom's own world position is what
    actually sits at the object's visual center.

    HOWEVER: build_model() above deliberately strips exactly this kind of
    geom (any <geom mesh="..._vis..."> reference) because MuJoCo's native
    loader can't read the high-poly .obj/.msh visual mesh it points to --
    the geom's own *name* (e.g. "akita_black_bowl_1_g0") doesn't contain
    "_vis", but the *mesh asset it references* does, so it gets stripped in
    THIS dummy-reconstructed model even though it survives untouched in the
    real WSL2 env. So on this path the ideal contype=0/conaffinity=0 geom
    is essentially never present.

    Priority order:
      1. If that ideal geom happens to survive (e.g. a future asset that
         isn't stripped), use its exact world position.
      2. Else, if body_name's object TYPE is in KNOWN_VISUAL_CENTER_LOCAL_
         OFFSET (measured once against the real WSL2 env), apply that exact
         known local-frame offset, rotated into world space by the body's
         current orientation. This is exact, not an approximation.
      3. Else, fall back to the centroid of whatever collision geoms
         (contype=1/conaffinity=1) DO survive stripping -- recovers most,
         not all, of the true offset; only used for object types we haven't
         measured yet.
      4. Else (no geoms at all -- shouldn't happen for a real object),
         raw body origin.
    """
    geom_ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]

    ref_geoms = [g for g in geom_ids
                 if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
    if ref_geoms:
        return data.geom_xpos[ref_geoms[0]].copy()

    if body_name is not None:
        obj_type = _object_type_from_body_name(body_name)
        local_offset = KNOWN_VISUAL_CENTER_LOCAL_OFFSET.get(obj_type)
        if local_offset is not None:
            body_rot = data.xmat[body_id].reshape(3, 3)
            return data.xpos[body_id] + body_rot @ local_offset

    if geom_ids:
        return np.mean([data.geom_xpos[g] for g in geom_ids], axis=0)

    return data.xpos[body_id].copy()


def body_positions_at(model: mujoco.MjModel, data: mujoco.MjData,
                       qpos_row: np.ndarray, body_names):
    """
    Set qpos to the given row (length model.nq), forward-kinematics it, and
    return {body_name: visual-center world position (3,)} for the requested
    bodies. See visual_center_world_pos() for why this isn't just data.xpos.
    """
    data.qpos[:] = qpos_row
    mujoco.mj_forward(model, data)
    return {name: visual_center_world_pos(model, data, model.body(name).id, body_name=name)
            for name in body_names}


def identify_manipulated_object(model, data, states_row, ee_pos_xyz, candidate_names):
    """
    Heuristic: at the grasp frame, the object actually being picked up is
    whichever free-floating body is closest to the end-effector position.
    Returns (name, position).
    """
    qpos = states_row[1:1 + model.nq]
    positions = body_positions_at(model, data, qpos, candidate_names)
    best_name = min(positions, key=lambda n: np.linalg.norm(positions[n] - ee_pos_xyz))
    return best_name, positions[best_name]
