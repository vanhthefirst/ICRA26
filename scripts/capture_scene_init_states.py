"""
Sketch-Prompted VLA — pin and persist the annotated initial state (needs
libero env). Resolves issue 1 of prompt_libero_rollout_harness.md, a hard blocker:
`scene.bddl` only pins a +-1.2cm placement BOX (HALF_BOX), not an exact pose, and
LIBERO's placement sampler redraws inside that box (plus a random yaw) at every
`reset()`. The exact state behind every `frame0.png` existed only as a PNG until
this script runs -- and a rollout that resets a scene without pinning it would
score against a DIFFERENT, undocumented layout than the one every sketch (auto and
human) was drawn against.

Ladder, stopping at the first rung that meets tolerance (every instance's
re-projected visual centre within TOL_PX of meta['all_pixels'], Euclidean):

  1. resample  -- np.random.seed(scene's own build seed), then reset() up to
                  N_RESAMPLE times, keeping the lowest-max-error draw. This
                  replays the exact RNG sequence the builder consumed before it
                  captured frame0.png (a single global np.random.seed() call
                  followed by the FIRST reset() -- build_bddl's own placement
                  sampling uses an INDEPENDENT np.random.default_rng(seed), so it
                  does not consume the global stream). Measured on the real 114:
                  every scene reproduces at attempt 1, error 0.0px -- see
                  outputs/rollouts/init_state_capture_report.json.
  2. solve     -- deproject each instance's recorded pixel onto the support plane
                  at the Z the best resample draw already settled to (Z is not
                  recoverable from a pixel; a projected centre doesn't move with
                  yaw, so yaw is left as the resample draw gave it), write the
                  solved xy into that instance's free joint, forward(), a short
                  re-settle, re-verify.
  3. give up   -- record the scene + residual in nonreproducible.json; the
                  rollout harness skips it with that reason in results.csv.

Hard constraints:
  * frame0.png is NEVER regenerated. This script changes the SIMULATOR to match
    the PNG that already exists, never the other way around.
  * The 36 human-study-subset scenes (outputs/human_study/scene_subset.json) are
    captured FIRST -- if one turns out non-reproducible it should be swappable out
    of the roster before further annotation, not discovered after.

Persistence: env.get_sim_state() (LIBERO's own `ControlEnv.get_sim_state`, a thin
wrapper over `self.env.sim.get_state().flatten()` -> [time, qpos, qvel]) into each
scene's `init_state.npz`. Restored at rollout with `env.set_init_state(flat)`
(`ControlEnv.set_init_state` -> `regenerate_obs_from_state` -> `sim.set_state_from_
flattened` + `sim.forward()` + `check_success()` + observable refresh) -- this
IS the LIBERO-idiomatic route (confirmed present on this install, so used in
preference to hand-rolling `sim.set_state()` + `sim.forward()`), the same
mechanism LIBERO's own eval harness uses via `init_files/*.pruned_init`.

    conda activate libero
    cd /mnt/c/Users/Admin/sketch_prompted_vla
    python scripts/capture_scene_init_states.py --smoke      # 3 scenes, one per suite
    python scripts/capture_scene_init_states.py               # 36-subset, then remaining 78
    python scripts/capture_scene_init_states.py --subset-only
"""

import os, sys, json, gc, argparse, time
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sketch_geometry import project_world_to_pixel_xy, deproject_pixel_xy_to_world_xy

IMG_H = IMG_W = 128
CAMERA = "agentview"
ADIM = 7                              # OffScreenRenderEnv has no .action_dim
N_RESAMPLE = 50                       # rung 1: reset attempts before falling back
TOL_PX = 1.0                          # issue 1's stated tolerance, Euclidean px
RESETTLE_STEPS = 8                    # rung 2: brief re-settle after the qpos edit
                                       # (small compared to the initial 20 -- the
                                       # edit only nudges xy inside a box already
                                       # physically stable from rung 1's best draw)

SUITE_DIR = {"spatial": "validation_set_spatial",
             "object": "validation_set_object",
             "goal": "validation_set_goal"}
RUN_ROOT = os.path.join(_REPO, "outputs", "rollouts")
NONREPRO_PATH = os.path.join(RUN_ROOT, "nonreproducible.json")
REPORT_PATH = os.path.join(RUN_ROOT, "init_state_capture_report.json")
SUBSET_PATH = os.path.join(_REPO, "outputs", "human_study", "scene_subset.json")
MANIFEST_PATH = os.path.join(_REPO, "outputs", "validation_manifest_all.json")


def scene_root(suite, dir_):
    return os.path.join(_REPO, "outputs", SUITE_DIR[suite], dir_)


# ---- the small set of live-sim helpers every builder already duplicates; kept
# byte-identical here so a live measurement matches what built frame0.png ----
def bid_of(model, name):
    for c in (name, f"{name}_main"):
        if c in model.body_names:
            return model.body_name2id(c)
    raise KeyError(name)


def vcenter(model, data, bid):
    gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
    ref = [g for g in gids if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
    return data.geom_xpos[ref[0]].copy() if ref else data.body_xpos[bid].copy()


def jnt_of(model, prefix):
    for j in range(model.njnt):
        if model.joint_id2name(j) == f"{prefix}_joint0":
            return j
    for j in range(model.njnt):
        nm = model.joint_id2name(j)
        if nm and nm.startswith(prefix) and "joint" in nm:
            return j
    return None


def settle(env, n=20):
    obs = env.reset()
    for _ in range(n):
        obs, _, _, _ = env.step(np.zeros(ADIM))
    return obs


def reproj_errors(model, data, W2P, all_pixels):
    errs = {}
    for name, px in all_pixels.items():
        try:
            bid = bid_of(model, name)
        except KeyError:
            continue
        wc = vcenter(model, data, bid)
        proj = project_world_to_pixel_xy(wc, W2P)
        errs[name] = float(np.hypot(proj[0] - px[0], proj[1] - px[1]))
    return errs


# ------------------------------------------------------------------ one scene --
def capture_scene(suite, dir_):
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils import camera_utils as CU

    root = scene_root(suite, dir_)
    meta = json.load(open(os.path.join(root, "meta.json")))
    all_pixels = meta["all_pixels"]
    bddl_path = os.path.join(root, "scene.bddl")

    np.random.seed(meta["seed"])
    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=IMG_H,
                              camera_widths=IMG_W, camera_names=[CAMERA])
    try:
        W2P = None
        best_state = None
        best_err = None
        best_errs = None
        n_attempts = 0
        for attempt in range(N_RESAMPLE):
            settle(env)
            n_attempts = attempt + 1
            model, data = env.sim.model, env.sim.data
            if W2P is None:
                W2P = CU.get_camera_transform_matrix(
                    sim=env.sim, camera_name=CAMERA, camera_height=IMG_H, camera_width=IMG_W)
            errs = reproj_errors(model, data, W2P, all_pixels)
            m = max(errs.values()) if errs else 0.0
            if best_err is None or m < best_err:
                best_err, best_errs, best_state = m, errs, env.sim.get_state()
            if best_err <= TOL_PX:
                break

        # restore the winning resample draw regardless of which attempt found it
        env.sim.set_state(best_state)
        env.sim.forward()
        model, data = env.sim.model, env.sim.data
        rung = "resample"

        if best_err > TOL_PX:
            # rung 2: solve xy per instance at the Z this draw already settled to;
            # Z and quaternion are NOT touched (Z isn't recoverable from a single
            # pixel; yaw doesn't move a projected centre -- see module docstring).
            for name, px in all_pixels.items():
                j = jnt_of(model, name)
                if j is None:
                    continue
                qa = model.jnt_qposadr[j]
                z0 = float(data.qpos[qa + 2])
                x, y = deproject_pixel_xy_to_world_xy(px, z0, W2P)
                data.qpos[qa + 0] = x
                data.qpos[qa + 1] = y
            env.sim.forward()
            for _ in range(RESETTLE_STEPS):
                env.step(np.zeros(ADIM))
            model, data = env.sim.model, env.sim.data
            errs2 = reproj_errors(model, data, W2P, all_pixels)
            m2 = max(errs2.values()) if errs2 else 0.0
            if m2 <= TOL_PX:
                rung, best_err, best_errs = "solve", m2, errs2
            else:
                rung, best_err, best_errs, best_state = "giveup", m2, errs2, None

        result = dict(suite=suite, dir=dir_, rung=rung, n_attempts_rung1=n_attempts,
                      residual_px=round(best_err, 4),
                      per_instance_px={k: round(v, 4) for k, v in best_errs.items()})

        if best_state is None:
            return result

        flat = env.get_sim_state()                 # [time, qpos..., qvel...]
        nq, nv = model.nq, model.nv
        t, qpos, qvel = flat[0], flat[1:1 + nq], flat[1 + nq:1 + nq + nv]
        npz_path = os.path.join(root, "init_state.npz")
        np.savez(npz_path, qpos=qpos, qvel=qvel, time=np.array([t]))
        # verified write: reload from disk, restore via the LIBERO-idiomatic
        # route, and re-check tolerance -- catches a truncated DrvFs write too.
        chk = np.load(npz_path)
        flat2 = np.concatenate([chk["time"], chk["qpos"], chk["qvel"]])
        env.set_init_state(flat2)
        model, data = env.sim.model, env.sim.data
        errs3 = reproj_errors(model, data, W2P, all_pixels)
        m3 = max(errs3.values()) if errs3 else 0.0
        result["reload_residual_px"] = round(m3, 4)
        result["reload_ok"] = bool(m3 <= TOL_PX)
        if not result["reload_ok"]:
            result["rung"] = "giveup"
            result["giveup_reason"] = "reload_mismatch"
            os.remove(npz_path)
        return result
    finally:
        try:
            env.close()
        except Exception:
            pass
        gc.collect()


# ----------------------------------------------------------------------- main --
def build_roster(subset_only, all_only):
    subset = json.load(open(SUBSET_PATH))
    subset_pairs = [(s["suite"], s["dir"]) for s in subset["scenes"]]
    if subset_only:
        return subset_pairs
    manifest = json.load(open(MANIFEST_PATH))
    all_pairs = [(e["suite"], e["dir"]) for e in manifest]
    if all_only:
        return all_pairs
    rest = [p for p in all_pairs if p not in set(subset_pairs)]
    return subset_pairs + rest       # 36-subset FIRST, per the hard constraint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3 scenes, one per suite")
    ap.add_argument("--subset-only", action="store_true", help="the 36 human-study scenes only")
    ap.add_argument("--all", action="store_true", help="all 114 in manifest order, subset not prioritised")
    args = ap.parse_args()
    os.makedirs(RUN_ROOT, exist_ok=True)

    if args.smoke:
        roster = [("spatial", "scene_0000"), ("object", "scene_0000"), ("goal", "scene_0000")]
    else:
        roster = build_roster(args.subset_only, args.all)

    results, nonrepro = [], []
    t0 = time.time()
    for i, (suite, dir_) in enumerate(roster):
        r = capture_scene(suite, dir_)
        results.append(r)
        tag = "OK" if r["rung"] != "giveup" else "GIVEUP"
        print(f"  [{i+1}/{len(roster)}] {suite}/{dir_} rung={r['rung']:8s} "
              f"residual={r['residual_px']:.3f}px attempts={r['n_attempts_rung1']} -> {tag}")
        if r["rung"] == "giveup":
            nonrepro.append(r)

    by_rung = {}
    for r in results:
        by_rung[r["rung"]] = by_rung.get(r["rung"], 0) + 1
    n = len(results)
    print(f"\n{n} scenes attempted in {time.time()-t0:.1f}s")
    for rung in ("resample", "solve", "giveup"):
        c = by_rung.get(rung, 0)
        print(f"  {rung:10s} {c:3d} ({100.0*c/n:.1f}%)")

    # merge with any existing report/nonreproducible list, so --subset-only
    # followed by a run over the rest ACCUMULATES rather than clobbers.
    def merge_by_key(prior_list, new_list):
        new_keys = {(x["suite"], x["dir"]) for x in new_list}
        kept = [r for r in prior_list if (r["suite"], r["dir"]) not in new_keys]
        return kept + new_list

    merged_results = results
    if os.path.exists(REPORT_PATH) and not args.smoke:
        prior = json.load(open(REPORT_PATH))
        merged_results = merge_by_key(prior.get("results", []), results)
    by_rung_m = {}
    for r in merged_results:
        by_rung_m[r["rung"]] = by_rung_m.get(r["rung"], 0) + 1
    report = dict(n_scenes=len(merged_results), tol_px=TOL_PX, n_resample=N_RESAMPLE,
                  by_rung={k: by_rung_m.get(k, 0) for k in ("resample", "solve", "giveup")},
                  results=merged_results)

    merged_nonrepro = nonrepro
    if os.path.exists(NONREPRO_PATH) and not args.smoke:
        prior_nr = json.load(open(NONREPRO_PATH))
        merged_nonrepro = merge_by_key(prior_nr, nonrepro)

    if not args.smoke:
        json.dump(report, open(REPORT_PATH, "w"), indent=2)
        json.dump(merged_nonrepro, open(NONREPRO_PATH, "w"), indent=2)
        print(f"\nwrote {REPORT_PATH} ({report['n_scenes']} scenes total)")
        print(f"wrote {NONREPRO_PATH} ({len(merged_nonrepro)} scene(s) total)")


if __name__ == "__main__":
    main()
