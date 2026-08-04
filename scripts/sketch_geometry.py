"""
Sketch-Prompted VLA — shared world<->pixel projection (issues 2 and 4 of
prompt_libero_rollout_harness.md).

Pure numpy. Must NOT import robosuite / mujoco / libero: this is the seam a GPU
machine later plugs a learned policy into, and it must be importable and testable
without a simulator.

Convention, asserted once here rather than hand-rolled a second time
(SUITE_FACTS.md S1.1): every pixel this module accepts or returns is
`pixel_xy = (col, row)` -- the SAME convention as `pick_px` / `place_px` /
`symbolic_tokens.circle{cx,cy}` / `symbolic_tokens.arrow{x0,y0,x1,y1}` in every
scene's tokens.json. This is NOT robosuite's native convention.
`camera_utils.project_points_from_world_to_camera` returns `(row, col)` in a frame
that every builder then flips once more (`row = (IMG_H-1) - raw_row`) to match the
CORRECTLY ORIENTED `agentview_image` (do not flip the image itself). The two
functions below reproduce that exact pipeline in closed form -- including the
integer rounding and boundary clipping the builders' `project()` performs -- so a
pixel computed here is byte-identical to one computed by a builder, and so nobody
has to re-derive the row-flip a second time (it has already cost time once).

Round-trip self-test: `python scripts/sketch_geometry.py`. It deprojects every
`all_pixels` / `pick_px` / `place_px` entry in a sample of real scenes (all three
suites) back to world xy at several candidate support heights and reprojects,
asserting recovery of the original pixel to within 1.0 px -- the same tolerance
issue 1's initial-state pinning is held to, since both use this routine.
"""

import numpy as np

IMG_H = IMG_W = 128


def project_world_to_pixel_xy(world_xyz, camera_matrix, img_h=IMG_H, img_w=IMG_W):
    """world_xyz: (..., 3) world-frame point(s). camera_matrix: (4, 4) world->pixel
    transform (a scene's meta['camera_matrix'] / camera_matrix, robosuite's
    get_camera_transform_matrix output).

    Returns pixel_xy: (..., 2) int array, (col, row) -- the pick_px/place_px/
    all_pixels convention, not robosuite's native (row, col).
    """
    world_xyz = np.asarray(world_xyz, dtype=float)
    single = world_xyz.ndim == 1
    if single:
        world_xyz = world_xyz[None, :]
    M = np.asarray(camera_matrix, dtype=float)
    ones = np.ones(world_xyz.shape[:-1] + (1,))
    pts = np.concatenate([world_xyz, ones], axis=-1)          # (..., 4)
    raw = np.einsum("ij,...j->...i", M, pts)                  # (..., 4) = (A, B, C, D)
    u = raw[..., 0] / raw[..., 2]
    v = raw[..., 1] / raw[..., 2]
    u_r = np.clip(np.round(u), 0, img_w - 1).astype(int)      # robosuite's own round+clip
    v_r = np.clip(np.round(v), 0, img_h - 1).astype(int)
    col = u_r
    row = (img_h - 1) - v_r                                   # the repo's established flip
    pix = np.stack([col, row], axis=-1)
    return pix[0] if single else pix


def deproject_pixel_xy_to_world_xy(pixel_xy, z_world, camera_matrix, img_h=IMG_H, img_w=IMG_W):
    """Inverse of project_world_to_pixel_xy at a KNOWN support-plane height
    z_world. Issue 2: one pixel is a camera ray, not a 3D point -- ray-height
    intersection needs a known z (table/floor + a per-category resting offset);
    deprojecting to an unknown/moving z is under-determined and out of scope here.

    pixel_xy: (col, row), the pick_px convention. z_world: float, world-frame
    height of the support plane the object rests on.

    Two linear constraints (the pixel's col and row) in two unknowns (world x, y)
    at fixed z: solved directly from camera_matrix, no iteration. Also the exact
    routine issue 1's pinning ladder (rung 2) uses to recover a duplicate
    instance's xy from its recorded meta['all_pixels'] centre.
    """
    col, row = float(pixel_xy[0]), float(pixel_xy[1])
    u = col
    v = (img_h - 1) - row                                     # undo the established flip
    M = np.asarray(camera_matrix, dtype=float)
    m00, m01, m02, m03 = M[0]
    m10, m11, m12, m13 = M[1]
    m20, m21, m22, m23 = M[2]
    z = float(z_world)
    A_ = np.array([[u * m20 - m00, u * m21 - m01],
                   [v * m20 - m10, v * m21 - m11]])
    b_ = np.array([(m02 * z + m03) - u * (m22 * z + m23),
                   (m12 * z + m13) - v * (m22 * z + m23)])
    x, y = np.linalg.solve(A_, b_)
    return float(x), float(y)


# ------------------------------------------------------------------- self-test --
def _round_trip_check(camera_matrix, pixel_xy, z_candidates, img_h=IMG_H, img_w=IMG_W, tol=1.0):
    """Deproject pixel_xy at each candidate z, reproject, and check recovery
    within `tol` px on both axes. Boundary pixels are excluded by the caller --
    clipping at the frame edge is lossy by construction and not a projection bug.
    Returns the worst-case (dx, dy) magnitude across the candidate heights."""
    worst = (0.0, 0.0)
    for z in z_candidates:
        x, y = deproject_pixel_xy_to_world_xy(pixel_xy, z, camera_matrix, img_h, img_w)
        col2, row2 = project_world_to_pixel_xy(np.array([x, y, z]), camera_matrix, img_h, img_w)
        dx, dy = abs(col2 - pixel_xy[0]), abs(row2 - pixel_xy[1])
        worst = (max(worst[0], dx), max(worst[1], dy))
        if dx > tol or dy > tol:
            raise AssertionError(
                f"round-trip failed at z={z}: pixel={pixel_xy} -> world=({x:.4f},{y:.4f}) "
                f"-> pixel=({col2},{row2}), off by ({dx},{dy}) px > tol={tol}")
    return worst


def _self_test():
    import glob, json, os

    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    Z_CANDIDATES = [0.0, 0.83, 0.90, 1.10]     # floor .. table .. tall-object cases
    suites = ["validation_set_spatial", "validation_set_object", "validation_set_goal"]
    n_scenes = 0
    n_pixels = 0
    worst_dx = worst_dy = 0.0
    for suite in suites:
        pattern = os.path.join(_REPO, "outputs", suite, "scene_*", "meta.json")
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"  [skip] no scenes found for {suite} ({pattern})")
            continue
        # every 4th scene: representative coverage without re-checking all 114 x 4 heights
        for p in paths[::4]:
            meta = json.load(open(p))
            cam = meta["camera_matrix"]
            pixels = dict(meta.get("all_pixels", {}))
            pixels["_pick"] = meta["pick_px"]
            pixels["_place"] = meta["place_px"]
            n_scenes += 1
            for name, px in pixels.items():
                if px[0] <= 1 or px[0] >= IMG_W - 2 or px[1] <= 1 or px[1] >= IMG_H - 2:
                    continue    # near-boundary: clipping makes the inverse lossy by construction
                dx, dy = _round_trip_check(cam, px, Z_CANDIDATES)
                worst_dx = max(worst_dx, dx)
                worst_dy = max(worst_dy, dy)
                n_pixels += 1
    print(f"round-trip: {n_scenes} scenes, {n_pixels} pixel entries x {len(Z_CANDIDATES)} "
          f"heights, worst offset ({worst_dx:.3f}, {worst_dy:.3f}) px, tol=1.0 px -> PASS")


if __name__ == "__main__":
    _self_test()
