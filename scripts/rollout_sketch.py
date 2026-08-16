"""
Sketch-Prompted VLA — LIBERO rollout + scoring harness (needs the libero env).

Consumes what scripts/capture_scene_init_states.py pinned: resets each scene
to its EXACT annotated initial state (init_state.npz), hands a SketchPolicy
(scripts/sketch_policies.py) the live obs plus a static sketch prompt, steps the
simulator, and scores success by the scene's own BDDL goal predicate
(env.check_success()), sustained over a window (issue 5).

Conditions are (label, sketch_root_template) pairs -- adding a source is a CLI
argument, not a code change:

    text_only        no sketch; RestrictedPrompt (issue 6 firewall)
    auto             outputs/validation_set_<suite>/<dir>/
    human:<name>     outputs/human_study/rendered/<name>/validation_set_<suite>/<dir>/
    human_consensus  outputs/human_study/rendered/consensus/validation_set_<suite>/<dir>/

Human sketches exist for the 36-scene study subset only (outputs/human_study/
scene_subset.json); a run whose --conditions includes any human:* / human_
consensus condition is scoped to that subset (intersected with --scenes and with
the reproducible roster). A run of text_only + auto alone covers all 114.

`--prompt-type {explicit,ambiguous}` selects which caption the policy is given
and is orthogonal to both `--conditions` and the scene's tier
(PROMPT_TAXONOMY.md). It is ONE choice per invocation and needs its own
`--run-id`: both arms write the same condition label and the same rollout
indices, so pointing them at one run-id would make the second arm resume as
already-done. `stable_seed` does not read the prompt type, so rollout k of a
scene starts from the same rng draw in both arms and the two are paired scene by
scene and rollout by rollout.

Scene order is SCENE-MAJOR (issue 8): one OffScreenRenderEnv is built per scene
and reused across every condition x rollout for that scene, restoring
init_state.npz between them -- never rebuilt per rollout. env.close() +
gc.collect() between scenes, exactly like the builders. A partial run therefore
yields complete rows for the scenes it reached, and --resume skips any
(suite, dir, condition, rollout_idx) triple already in results.csv.

`--policy` is ONE choice for the whole invocation, and the scripted policies read
different prompt types: ScriptedSketchOracle reads `symbolic_tokens` off a
`Prompt`, TextOnlyGuessPolicy reads candidate pixels off a `RestrictedPrompt`.
So `text_only` can never share an invocation with a sketch condition — passing
`--conditions text_only,auto --policy oracle` hands the oracle a RestrictedPrompt
and dies with `AttributeError: 'RestrictedPrompt' object has no attribute
'symbolic_tokens'`. A full run is TWO invocations sharing one `--run-id`; both
append to the same results.csv and their configs accumulate into
run_config.json's `invocations` list. `full_run_plane/results.csv` is exactly
this: 114 (auto, oracle) rows and 342 (text_only, text_guess) rows.

    conda activate libero
    cd /mnt/c/Users/Admin/sketch_prompted_vla
    python scripts/rollout_sketch.py --smoke

    # all 114, both halves of the headline gap, one run id
    python scripts/rollout_sketch.py --conditions auto --policy oracle \
        --scenes all --run-id r1
    python scripts/rollout_sketch.py --conditions text_only --policy text_guess \
        --scenes all --n-rollouts 3 --run-id r1 --resume

    # adding human sketches: they are sketch-bearing, so they ride with `auto`
    python scripts/rollout_sketch.py \
        --conditions auto,human:aaron,human_consensus --policy oracle \
        --scenes subset --run-id r2
    python scripts/rollout_sketch.py --conditions text_only --policy text_guess \
        --scenes subset --n-rollouts 3 --run-id r2 --resume

`--policy pi05` is the learned, sketch-free baseline: pi0.5-LIBERO served by
openpi over a websocket (scripts/pi05_policy.py, prompt_pi05_baseline.md). It
runs `text_only` and nothing else -- the checkpoint has no sketch channel -- and
opens the scene env at 256 with the wrist camera, since that is what it was
fine-tuned on. Requires a policy server already running on a GPU:

    python scripts/rollout_sketch.py --policy pi05 --scenes all --run-id pi05_baseline
"""

import os, sys, json, gc, csv, time, argparse, subprocess, hashlib
import numpy as np
import cv2

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sketch_geometry import project_world_to_pixel_xy
from sketch_policies import (Prompt, RestrictedPrompt, ScriptedSketchOracle,
                              TextOnlyGuessPolicy, NoOpPolicy)
# constants only -- pi05_policy imports nothing beyond numpy at module level, so
# this is safe on a machine with no openpi-client. The Pi05ServerPolicy class
# itself is imported lazily in make_policy().
from pi05_policy import (LIBERO_ENV_RESOLUTION,
                          RESIZE_SIZE as PI05_RESIZE_SIZE)

IMG_H = IMG_W = 128
CAMERA = "agentview"
ADIM = 7
SUCCESS_WINDOW = 5          # consecutive steps check_success() must hold (issue 5)
GRASP_LIFT_TH = 0.03        # m; matches the builders' own graspable gate
WRONG_DEST_APPROX_TH = 0.08  # m; xy proximity used for the approximate post-hoc
                             # "which destination did the released object end up
                             # near" attribution -- NOT the reported success
                             # number, which always comes from env.check_success()
MAX_STEPS_DEFAULT = 200
PI05_MAX_STEPS_CEILING = 320   # ceiling for --policy pi05; the per-scene budget
                               # is the suite's own (220/280/300, openpi's
                               # examples/libero/main.py) applied through
                               # policy.episode_len in run_rollout.
DEPTH_PATCH_K = 3           # k x k median patch for --deproject depth; a single
                            # 128x128 pixel is not reliably ON a flat grocery box

SUITE_DIR = {"spatial": "validation_set_spatial",
             "object": "validation_set_object",
             "goal": "validation_set_goal"}
OUT_ROOT = os.path.join(_REPO, "outputs", "validation_set_%s")
RUN_ROOT = os.path.join(_REPO, "outputs", "rollouts")
SUBSET_PATH = os.path.join(_REPO, "outputs", "human_study", "scene_subset.json")
MANIFEST_PATH = os.path.join(_REPO, "outputs", "validation_manifest_all.json")
NONREPRO_PATH = os.path.join(RUN_ROOT, "nonreproducible.json")

RESULT_FIELDS = [
    "suite", "dir", "tier", "condition", "policy", "sketch_route", "rollout_idx",
    "skipped", "skip_reason", "n_steps",
    "success_final", "success_sustained", "first_success_step",
    "grasp_success_flag", "grasped_any", "lifted", "grasped_instance",
    "correct_instance_grasped", "nearest_destination", "correct_destination",
    "terminal_dist_xy", "sketch_referent_object", "sketch_referent_destination",
    "sketch_fidelity_object", "sketch_fidelity_destination",
    "deproject", "z_pick", "z_place",
    # Determinism fingerprints. Two identical prompts run twice ought to give
    # identical rows; on human_r1 they did not (ROLLOUT.md, "A determinism
    # failure"), and outcome columns alone cannot say WHERE the two runs parted.
    # These three split the question into its three answers:
    #   init_state_hash differs  -> set_init_state did not restore the scene, so
    #                               the rollout started somewhere else. The
    #                               residual-state hypothesis, confirmed.
    #   init_warmstart_hash only -> qpos/qvel restored but qacc_warmstart was
    #                               not. That alone does not prove the warm
    #                               start is what carries the difference into
    #                               the trajectory -- qacc_warmstart is a
    #                               SOLVER OUTPUT, recomputed by forward() from
    #                               whatever inputs are live, so this pattern
    #                               only shows something upstream (data.ctrl,
    #                               the robosuite controller goal) was not
    #                               restored either. `determinism_r2` tested
    #                               zeroing qacc_warmstart alone and it did not
    #                               remove the flips -- see clear_warmstart().
    #   final_state_hash differs -> identical start, divergent trajectory: the
    #                               step itself is not reproducible.
    #   all three equal          -> the sim matched and only the scoring flipped.
    "init_state_hash", "init_warmstart_hash", "final_state_hash",
    # pi0.5 server diagnostics (empty for every scripted policy). A collapsed
    # n_infer_calls is the signature of an episode that died early on an
    # exception rather than one that genuinely failed the task.
    "n_infer_calls", "infer_ms_mean",
    # How the sketch reached pi0.5 (none|overlay|language). Empty for every
    # scripted policy. Recorded per ROW because `condition` alone cannot
    # distinguish an overlay run from a language run -- both are "auto".
    "pi05_sketch_mode",
    # Which caption the policy was given (explicit|ambiguous), per
    # PROMPT_TAXONOMY.md. Orthogonal to `tier`: a control scene can carry either
    # caption, and so can a `both` scene. Recorded per ROW so an explicit arm and
    # an ambiguous arm remain separable if they are ever merged into one file.
    "prompt_type",
]


def scene_root(suite, dir_):
    return os.path.join(_REPO, "outputs", SUITE_DIR[suite], dir_)


def stable_seed(*parts):
    """Deterministic per-(scene, condition, rollout) seed.

    This used to be `hash((suite, dir_, label, r_idx)) & 0xFFFFFFFF`. Python
    SALTS `hash()` of str per process (PYTHONHASHSEED randomisation), so the
    only stochastic policy here -- TextOnlyGuessPolicy, which picks uniformly
    among candidate pixels -- drew a DIFFERENT sample on every invocation. The
    baseline was therefore irreproducible, and since a full run is two separate
    invocations, two runs of the identical configuration disagreed: `full_run`
    scored text_only at 21.9% and an otherwise byte-identical re-run scored it
    at 18.4%, moving the headline gap from +15.8pp to +19.3pp with no code
    change. The `auto` half, driven by the deterministic oracle, matched
    bit-for-bit across both -- which is what isolated the cause.

    blake2b is stable across processes, machines and Python versions."""
    key = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % (2**32)


# ---- live-sim helpers, byte-identical to the builders' (see SUITE_FACTS.md) --
def bid_of(model, name):
    for c in (name, f"{name}_main"):
        if c in model.body_names:
            return model.body_name2id(c)
    raise KeyError(name)


def vcenter(model, data, bid):
    gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
    ref = [g for g in gids if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
    return data.geom_xpos[ref[0]].copy() if ref else data.body_xpos[bid].copy()


def success(env):
    for o in (env, getattr(env, "env", None)):
        if o is None:
            continue
        for m in ("check_success", "_check_success"):
            fn = getattr(o, m, None)
            if callable(fn):
                try:
                    return bool(fn())
                except Exception:
                    pass
    return None


def frame_obs(obs):
    k = "agentview_image" if "agentview_image" in obs else \
        [x for x in obs if "agentview" in x and "image" in x][0]
    f = np.asarray(obs[k])
    if f.dtype != np.uint8:
        f = np.clip(f * 255.0 if f.max() <= 1 + 1e-6 else f, 0, 255).astype(np.uint8)
    return f.copy()


def nearest_instance(px, all_pixels):
    if not all_pixels:
        return None
    best, best_d = None, None
    for name, p in all_pixels.items():
        d = float(np.hypot(px[0] - p[0], px[1] - p[1]))
        if best_d is None or d < best_d:
            best, best_d = name, d
    return best


# ------------------------------------------------------------- roster / io ----
def load_manifest():
    return json.load(open(MANIFEST_PATH))


def load_subset_pairs():
    subset = json.load(open(SUBSET_PATH))
    return [(s["suite"], s["dir"]) for s in subset["scenes"]]


def load_nonreproducible():
    if not os.path.exists(NONREPRO_PATH):
        return {}
    entries = json.load(open(NONREPRO_PATH))
    return {(e["suite"], e["dir"]): e for e in entries}


def parse_scene_filter(spec):
    """--scenes all | subset | comma list of suite/dir"""
    if spec in ("all", None):
        return None
    if spec == "subset":
        return set(load_subset_pairs())
    out = set()
    for tok in spec.split(","):
        suite, dir_ = tok.strip().split("/")
        out.add((suite, dir_))
    return out


# --------------------------------------------------------- condition sources --
def resolve_conditions(labels):
    conds = []
    for label in labels:
        if label == "text_only":
            conds.append((label, None))
        elif label == "auto":
            conds.append((label, os.path.join(_REPO, "outputs", "validation_set_{suite}", "{dir}")))
        elif label == "human_consensus":
            conds.append((label, os.path.join(_REPO, "outputs", "human_study", "rendered",
                                               "consensus", "validation_set_{suite}", "{dir}")))
        elif label.startswith("human:"):
            annot = label.split(":", 1)[1]
            conds.append((label, os.path.join(_REPO, "outputs", "human_study", "rendered",
                                               annot, "validation_set_{suite}", "{dir}")))
        else:
            raise ValueError(f"unknown condition {label!r}")
    return conds


def load_sketch(root_template, suite, dir_, sketch_route):
    """Returns (sketch_rgb_or_None, symbolic_tokens_or_None, ok). ok=False means
    this (annotator, scene) cell is ABSENT (skipped by the annotator, or not in
    that annotator's export) -- issue 7: excluded, never imputed."""
    root = root_template.format(suite=suite, dir=dir_)
    tok_path, sk_path = os.path.join(root, "tokens.json"), os.path.join(root, "sketch.png")
    if not os.path.exists(tok_path):
        return None, None, False
    tok = json.load(open(tok_path))["symbolic_tokens"]
    sketch_rgb = None
    if sketch_route == "overlay":
        img = cv2.imread(sk_path)
        if img is None:
            return None, None, False
        sketch_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return sketch_rgb, tok, True


def candidate_pixels(meta):
    """Unlabelled candidate pick / place pixels for the RestrictedPrompt (issue
    6): every same-category instance (target + siblings) for pick, every
    destination-typed instance for place, falling back to place_px when the
    destination is a region absent from all_pixels (same substitution
    score_human_sketches.py makes, HUMAN_STUDY.md 'Region destinations')."""
    all_px = meta.get("all_pixels", {})
    target, dest = meta["target"], meta["destination"]
    siblings = meta.get("siblings", [])
    pick_names = [target] + list(siblings)
    picks = [all_px[n] for n in pick_names if n in all_px]
    other_dests = (meta.get("other_baskets") or meta.get("other_plates") or
                   meta.get("other_dests") or [])
    dest_names = [dest] + list(other_dests)
    places = [all_px[n] for n in dest_names if n in all_px]
    if not places:
        places = [meta["place_px"]]
    return picks or [meta["pick_px"]], places


def instruction_for(meta, prompt_type):
    """The caption for this arm. `explicit` names the object and the destination
    by category, `ambiguous` names neither -- see PROMPT_TAXONOMY.md. Both keys
    are written by scripts/build_prompt_variants.py; a scene set that predates it
    carries neither, and this raises rather than falling back to `instruction`,
    because a silent fallback would run an ambiguous arm on explicit captions and
    report it as an ambiguous result."""
    key = f"instruction_{prompt_type}"
    if key not in meta:
        raise KeyError(
            f"{key} missing from meta.json -- run scripts/build_prompt_variants.py")
    return meta[key]


def build_prompt(condition_label, root_template, suite, dir_, tier, meta, sketch_route, rng,
                 z_at_pixel=None, prompt_type="explicit"):
    cam = meta["camera_matrix"]
    grasp = meta.get("grasp", {})
    scene_meta = dict(suite=suite, dir=dir_, tier=tier, camera_matrix=cam,
                      grasp_close_sign=grasp.get("close_sign"),
                      grasp_approach_dz=grasp.get("approach_dz"),
                      z_at_pixel=z_at_pixel)   # None => support-plane fallback
    instruction = instruction_for(meta, prompt_type)
    if condition_label == "text_only":
        picks, places = candidate_pixels(meta)
        picks = [list(p) for p in picks]; places = [list(p) for p in places]
        rng.shuffle(picks); rng.shuffle(places)
        return RestrictedPrompt(instruction=instruction, scene_meta=scene_meta,
                                  candidate_pick_px=picks, candidate_place_px=places), True
    sketch_rgb, tok, ok = load_sketch(root_template, suite, dir_, sketch_route)
    if not ok:
        return None, False
    return Prompt(instruction=instruction, sketch_rgb=sketch_rgb,
                  symbolic_tokens=tok, scene_meta=scene_meta), True


# --------------------------------------------------------------- depth (opt) --
def render_depth(env):
    """Real (metric) depth map for the current sim state, in the SAME row
    orientation as the corrected `agentview_image` -- see depth_z_at_pixel."""
    from robosuite.utils import camera_utils as CU
    _, depth = env.sim.render(width=IMG_W, height=IMG_H, camera_name=CAMERA, depth=True)
    depth = CU.get_real_depth_map(sim=env.sim, depth_map=depth)
    return depth[..., 0] if depth.ndim == 3 else depth


def depth_z_at_pixel(env, px, depth=None, k=DEPTH_PATCH_K):
    """issue 2 rung 2, the oracle affordance: true world z at a pixel from
    rendered depth. Any run using it is an upper bound (--deproject depth stamps
    this into run_config.json).

    TWO row conventions meet here and they are NOT the same one -- this cost a
    silently wrong number once, so both are named explicitly:

      * SAMPLING the depth map uses `row` UNFLIPPED. `get_real_depth_map` comes
        back already in the corrected orientation, matching `agentview_image`
        and therefore matching pick_px/all_pixels directly.
      * BACK-PROJECTING through the intrinsics uses `(IMG_H-1) - row`, the raw
        OpenGL row the intrinsic matrix is defined against -- the same flip
        sketch_geometry documents.

    Applying the flip to BOTH (the original form of this function) samples the
    mirrored row and silently reads background: the Spatial bowl of
    scene_0000 came back at z=-0.65m and the Object chocolate_pudding at
    z=-0.28m, both far-plane hits. It went unnoticed because the object first
    spot-checked, plate_1, happens to sit at row 64, whose mirror is 63.

    A single pixel is still fragile -- at 128x128 a flat grocery box does not
    reliably cover its own centre pixel -- so the estimate is the MEDIAN world z
    over a k x k patch, which rejects the occasional ray that slips past the
    object into the background."""
    from robosuite.utils import camera_utils as CU
    if depth is None:
        depth = render_depth(env)
    K = CU.get_camera_intrinsic_matrix(sim=env.sim, camera_name=CAMERA,
                                        camera_height=IMG_H, camera_width=IMG_W)
    Rext = CU.get_camera_extrinsic_matrix(sim=env.sim, camera_name=CAMERA)
    f = K[0, 0]
    col, row = int(px[0]), int(px[1])
    h = k // 2
    zs = []
    for dr in range(-h, h + 1):
        for dc in range(-h, h + 1):
            rr = int(np.clip(row + dr, 0, IMG_H - 1))
            cc = int(np.clip(col + dc, 0, IMG_W - 1))
            Zc = float(depth[rr, cc])                  # sample: UNFLIPPED row
            v_raw = (IMG_H - 1) - rr                   # geometry: FLIPPED row
            Xc = (cc - K[0, 2]) * Zc / f
            Yc = (v_raw - K[1, 2]) * Zc / f
            zs.append(float((Rext @ np.array([Xc, Yc, Zc, 1.0]))[2]))
    return float(np.median(zs))


def make_z_at_pixel(env, depth):
    """The callable handed to every policy through `scene_meta['z_at_pixel']`
    when --deproject depth is active. Policies call it AFTER choosing their own
    pixel, so the sketch-reading oracle and the firewalled text-only baseline
    are measured under the SAME deprojection -- otherwise 'depth' would be a
    property of the sketch condition rather than of the deprojection method,
    and the headline gap would silently absorb it. It also lets pick and place
    take their own z, which a single scene-level z_override could not."""
    return lambda px: depth_z_at_pixel(env, px, depth=depth)


# --------------------------------------------------------------- one rollout --
def state_hash(*arrays):
    """Short blake2b over raw float bytes -- a fingerprint of simulator state.

    Bit-exact by design. Anything that is genuinely restored comes back
    bit-identical (issue 1 measured 0.000px residual on all 114 scenes), so a
    difference here is a real difference, not float noise. Arrays that this
    MuJoCo build does not expose are skipped rather than faked, which keeps the
    hash comparable across rows of one run; it is NOT comparable across builds,
    and nothing reads it that way.
    """
    h = hashlib.blake2b(digest_size=8)
    for a in arrays:
        if a is None:
            continue
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()


_CLOCK_WARNED = False


def reset_episode_clock(env):
    """Zero robosuite's per-EPISODE step counter and terminated flag.

    `set_init_state` restores the physical state and nothing else. robosuite
    counts steps on the env OBJECT -- `self.timestep`, incremented in
    `_post_action`, which sets `self.done` once it reaches `self.horizon`, after
    which `step()` raises `ValueError: executing action in terminated episode`.
    Issue 8 reuses one env across every condition x rollout of a scene, so that
    counter accumulates across rollouts that are supposed to be independent
    episodes.

    Invisible until now purely because of run shapes: the largest scene so far
    was 4 conditions x 1 rollout x 200 steps = 800, under the horizon. The
    determinism check is 2 conditions x 10 rollouts = 4000 steps on one env and
    died inside its first scene.

    Deliberately narrow. `cur_time` and the observable clocks are left running,
    because they were also running through `full_run_plane` and `human_r1` --
    zeroing them here would make a determinism measurement of a harness that is
    not the one that produced the numbers being measured.
    """
    global _CLOCK_WARNED
    node, hops = env, 0
    while node is not None and hops < 8:
        if hasattr(node, "timestep") and hasattr(node, "done"):
            node.timestep = 0
            node.done = False
            return True
        node = getattr(node, "env", None)
        hops += 1
    if not _CLOCK_WARNED:
        print("[warn] no robosuite episode clock found to reset; a long "
              "multi-rollout run may die with 'executing action in terminated "
              "episode' once the horizon is reached.")
        _CLOCK_WARNED = True
    return False


def clear_warmstart(env, data):
    """Zero MuJoCo's constraint-solver warm start before a rollout.

    `determinism_r1` measured the cause of the flips: `init_state_hash` is
    identical on 72/72 repeat groups, so `set_init_state` restores qpos/qvel
    perfectly -- and `init_warmstart_hash` DIFFERS on 72/72. `qacc_warmstart` is
    not part of the flattened state `init_state.npz` pins, so it survives from
    whatever the previous rollout left behind, and it never settles: no group
    has two rollouts sharing one warm start.

    It is only the solver's initial guess, so the converged contact forces
    should agree to solver tolerance either way. In contact-rich manipulation
    they do not agree exactly, and 72/72 groups end in a different final state
    because of it.

    OFF BY DEFAULT, and that is deliberate. Every number in ROLLOUT.md was
    produced with the warm start running on, so switching this on silently would
    make new runs incomparable with `full_run_plane`, `full_run_depth` and
    `human_r1` without anything in the run recording why. `--reset-warmstart`
    stamps itself into run_config.json.

    MEASURED INSUFFICIENT -- `determinism_r2` ran with this flag and every
    10-rollout group still held TEN distinct warm starts, exactly as without it.
    The `sim.forward()` below re-runs the constraint solver and writes the
    converged `qacc` straight back into `qacc_warmstart`, and it computes that
    solution from inputs `set_init_state` also does not restore: `data.ctrl`
    still holds the previous rollout's last commanded torques, and robosuite's
    controller still holds its previous goal. Zeroing one downstream buffer and
    then asking the solver to refill it from stale upstream inputs cannot work.
    Use `--isolate-rollouts` instead; this flag is kept only because
    `determinism_r2` was run with it and its provenance must stay readable.
    """
    ws = getattr(data, "qacc_warmstart", None)
    if ws is None:
        return False
    ws[:] = 0.0
    try:
        env.sim.forward()
    except Exception:
        pass
    return True


def isolate_rollout(env, meta, flat_init):
    """Start a rollout from a genuinely clean episode, not merely a restored one.

    `determinism_r2` established that zeroing `qacc_warmstart` is not enough,
    because the residual state that drives it lives further upstream: `ctrl`,
    `qfrc_applied` / `xfrc_applied`, and the robosuite controller's own goal,
    which is Python state and not in mjData at all. Only `reset()` clears the
    controller.

    So: re-seed, reset, restore. That is the ordering
    `capture_scene_init_states.py` and `open_scene_env` already use once per
    scene (issue 1) -- the seed is replayed so the non-qpos visual draw some
    Goal arenas take from the global stream lands identically, which is the
    whole reason the throwaway reset exists. Applying it per ROLLOUT rather than
    per scene is the only change.

    `hard_reset` is forced off first. robosuite's default `reset()` tears the
    sim down and rebuilds the model, which would both cost seconds per rollout
    and invalidate the `model` / `data` handles the caller is holding. The soft
    path runs `mj_resetData` plus `_reset_internal`, which resets the
    controllers and keeps the data object identity.

    Returns the fresh obs. The caller MUST re-read `env.sim.model` /
    `env.sim.data` afterwards rather than trusting handles taken before it.
    """
    node, hops = env, 0
    while node is not None and hops < 8:
        if hasattr(node, "hard_reset"):
            node.hard_reset = False
            break
        node = getattr(node, "env", None)
        hops += 1
    np.random.seed(meta["seed"])
    env.reset()
    obs = env.set_init_state(flat_init)
    # Belt and braces on the buffers a soft reset may leave alone. These are
    # solver INPUTS; zeroing them is meaningful, unlike zeroing the warm start
    # the solver is about to overwrite.
    for name in ("ctrl", "qfrc_applied", "xfrc_applied", "act", "act_dot"):
        arr = getattr(env.sim.data, name, None)
        if arr is not None and getattr(arr, "size", 0):
            arr[:] = 0.0
    return obs


def run_rollout(env, model, data, flat_init, meta, policy, prompt, max_steps,
                video_path=None, reset_warmstart=False, isolate=False):
    if isolate:
        obs = isolate_rollout(env, meta, flat_init)
        # a reset can hand back different handles; never trust the caller's
        model, data = env.sim.model, env.sim.data
    else:
        obs = env.set_init_state(flat_init)
    reset_episode_clock(env)
    if reset_warmstart:
        clear_warmstart(env, data)
    # Fingerprint the restored state BEFORE the first step. Taken here and not
    # in main() because this is the only place the restore happens, and the
    # question these answer is whether two rollouts of one prompt began from the
    # same place -- see RESULT_FIELDS.
    h_init = state_hash(data.qpos, data.qvel)
    h_warm = state_hash(getattr(data, "qacc_warmstart", None), getattr(data, "act", None))
    policy.reset(prompt)
    all_px = meta.get("all_pixels", {})
    instances = list(all_px.keys())
    init_z = {n: float(vcenter(model, data, bid_of(model, n))[2]) for n in instances}
    max_z = dict(init_z)     # running peak height per instance -- a released
                             # object settles back near its resting height, so
                             # comparing only initial vs FINAL z misses every
                             # grasp that ended in a successful placement

    vw = None
    if video_path:
        # frame size comes from the actual render, not the 128 constant -- a
        # pi0.5 run opens the env at 256 and a hardcoded (128,128) silently
        # writes an unreadable video.
        fh, fw = frame_obs(obs).shape[:2]
        vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (fw, fh))

    success_hist = []
    ep_len = getattr(policy, "episode_len", max_steps)
    n_steps = min(max_steps, ep_len + 20)
    for t in range(n_steps):
        a = policy.act(obs, t)
        obs, _, _, _ = env.step(a)
        success_hist.append(bool(success(env)))
        for n in instances:
            z = float(vcenter(model, data, bid_of(model, n))[2])
            if z > max_z[n]:
                max_z[n] = z
        if vw is not None:
            vw.write(cv2.cvtColor(frame_obs(obs), cv2.COLOR_RGB2BGR))
    if vw is not None:
        vw.release()

    # sustained-success window (issue 5)
    first_success_step, sustained = None, False
    for i, s in enumerate(success_hist):
        if s and first_success_step is None:
            first_success_step = i
        if s and all(success_hist[i:i + SUCCESS_WINDOW]) and i + SUCCESS_WINDOW <= len(success_hist):
            sustained = True
            first_success_step = i
            break

    # failure attribution. "lifted" uses the PEAK height reached during the
    # episode, not the final one -- a successfully placed object settles back
    # near its resting height, which would otherwise look identical to "never
    # grasped". Instance identity (which object) still comes from the terminal
    # xy position, below.
    lift = {n: max_z[n] - init_z[n] for n in instances}
    grasped_instance = max(lift, key=lift.get) if lift else None
    grasped_any = grasped_instance is not None and lift[grasped_instance] > GRASP_LIFT_TH
    if not grasped_any:
        grasped_instance = None

    nearest_dest, terminal_dist = None, None
    ref_name = grasped_instance or meta["target"]
    if ref_name in all_px:
        pos_xy = vcenter(model, data, bid_of(model, ref_name))[:2]
        dest_candidates = [meta["destination"]] + list(
            meta.get("other_baskets") or meta.get("other_plates") or meta.get("other_dests") or [])
        dists = {}
        for d in dest_candidates:
            if d in all_px:
                dpos = vcenter(model, data, bid_of(model, d))[:2]
                dists[d] = float(np.linalg.norm(pos_xy - dpos))
        if dists:
            closest = min(dists, key=dists.get)
            terminal_dist = dists[closest]
            # only credit a destination if the object actually ended up NEAR one
            # (issue 5: a scene "merely standing at" a region can satisfy the
            # permissive In() box test without meaning anything was delivered)
            if terminal_dist <= WRONG_DEST_APPROX_TH:
                nearest_dest = closest

    # sketch-following fidelity (issue 7): what the SKETCH pointed to, vs what
    # actually happened -- independent of whether the sketch was "correct"
    tok = prompt.symbolic_tokens if isinstance(prompt, Prompt) else None
    sketch_ref_obj = sketch_ref_dest = None
    if tok is not None and all_px:
        sketch_ref_obj = nearest_instance((tok["circle"]["cx"], tok["circle"]["cy"]), all_px)
        sketch_ref_dest = nearest_instance((tok["arrow"]["x1"], tok["arrow"]["y1"]), all_px)

    zp, zq = getattr(policy, "last_z_pick", None), getattr(policy, "last_z_place", None)
    return dict(
        n_steps=n_steps,
        init_state_hash=h_init, init_warmstart_hash=h_warm,
        final_state_hash=state_hash(data.qpos, data.qvel),
        z_pick=round(zp, 4) if zp is not None else None,
        z_place=round(zq, 4) if zq is not None else None,
        success_final=bool(success_hist[-1]) if success_hist else False,
        success_sustained=sustained, first_success_step=first_success_step,
        grasped_any=grasped_any, lifted=grasped_any, grasped_instance=grasped_instance,
        correct_instance_grasped=(grasped_instance == meta["target"]) if grasped_any else False,
        nearest_destination=nearest_dest,
        correct_destination=(nearest_dest == meta["destination"]) if nearest_dest else False,
        terminal_dist_xy=round(terminal_dist, 4) if terminal_dist is not None else None,
        sketch_referent_object=sketch_ref_obj, sketch_referent_destination=sketch_ref_dest,
        sketch_fidelity_object=(grasped_instance == sketch_ref_obj) if (grasped_any and sketch_ref_obj) else None,
        sketch_fidelity_destination=(nearest_dest == sketch_ref_dest) if (nearest_dest and sketch_ref_dest) else None,
        n_infer_calls=getattr(policy, "n_infer_calls", None),
        infer_ms_mean=(round(policy.infer_ms_mean, 1)
                       if getattr(policy, "infer_ms_mean", None) is not None else None),
    )


# --------------------------------------------------------------- one scene ----
def open_scene_env(suite, dir_, depth=False, render_size=IMG_H, wrist=False):
    """`render_size` / `wrist` exist for the pi0.5 baseline and change nothing
    for the scripted policies, which keep the 128x128 agentview-only default.

    pi0.5-LIBERO was fine-tuned on 256-rendered LIBERO with a wrist camera and
    has never seen a 128x128 frame, so a pi05 run opens the SAME scene.bddl and
    restores the SAME init_state.npz at a different camera configuration. That
    is safe: init_state.npz is `time + qpos + qvel`, pure sim state, and carries
    no camera information -- the initial physical state is identical across the
    two, which is what makes the conditions comparable at all.

    The 128-space sketch geometry (pick_px, symbolic_tokens, the depth
    intrinsics) is NOT valid at 256 and is not used on the pi05 path."""
    from libero.libero.envs import OffScreenRenderEnv
    root = scene_root(suite, dir_)
    meta = json.load(open(os.path.join(root, "meta.json")))
    npz = np.load(os.path.join(root, "init_state.npz"))
    flat = np.concatenate([npz["time"], npz["qpos"], npz["qvel"]])
    cams = [CAMERA] + (["robot0_eye_in_hand"] if wrist else [])
    kwargs = dict(bddl_file_name=os.path.join(root, "scene.bddl"),
                  camera_heights=render_size, camera_widths=render_size,
                  camera_names=cams)
    if depth:
        kwargs["camera_depths"] = [True] * len(cams)
    np.random.seed(meta["seed"])
    env = OffScreenRenderEnv(**kwargs)
    env.reset()   # ONE throwaway reset: locks the seeded non-qpos visual draw
                  # (texture/material) some Goal scenes carry -- see
                  # capture_scene_init_states.py's docstring / ROLLOUT.md
                  # issue-1 write-up. set_init_state below never touches it.
    return env, meta, flat


_PI05_SINGLETON = None


def make_policy(policy_name, rng_seed, args=None):
    if policy_name == "oracle":
        return ScriptedSketchOracle()
    if policy_name == "text_guess":
        return TextOnlyGuessPolicy(rng_seed=rng_seed)
    if policy_name == "noop":
        return NoOpPolicy()
    if policy_name == "pi05":
        # imported lazily: openpi-client is only needed on the pi05 path, and a
        # machine running the scripted policies should not have to install it.
        #
        # The instance is CACHED across rollouts, unlike the scripted policies.
        # Those are cheap dataclass-ish objects; this one owns a websocket, and
        # rebuilding it per rollout would open ~114 connections per condition to
        # gain nothing -- reset() already clears every piece of episode state
        # (instruction, action plan, latency counters).
        global _PI05_SINGLETON
        if _PI05_SINGLETON is None:
            from pi05_policy import Pi05ServerPolicy
            _PI05_SINGLETON = Pi05ServerPolicy(
                host=args.pi05_host, port=args.pi05_port,
                replan_steps=args.pi05_replan,
                rotate180=not args.pi05_no_rotate180,
                num_steps_wait=args.pi05_wait_steps,
                sketch_mode=args.pi05_sketch_mode)
                # dump_first_frame_to is set per rollout by main(), so a
                # multi-scene run dumps one frame per scene rather than
                # overwriting a single file 114 times.
        return _PI05_SINGLETON
    raise ValueError(f"unknown policy {policy_name!r}")


# ----------------------------------------------------------------------- main --
def load_done(results_path):
    done = set()
    if not os.path.exists(results_path):
        return done
    with open(results_path) as f:
        for row in csv.DictReader(f):
            if row["skipped"] == "True":
                continue
            done.add((row["suite"], row["dir"], row["condition"], int(row["rollout_idx"])))
    return done


def append_rows(results_path, rows, write_header):
    with open(results_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="text_only,auto")
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--policy", default="oracle",
                    choices=["oracle", "text_guess", "noop", "pi05"])
    ap.add_argument("--sketch-route", default="tokens", choices=["overlay", "tokens"])
    ap.add_argument("--n-rollouts", type=int, default=1)
    ap.add_argument("--prompt-type", default="explicit", choices=["explicit", "ambiguous"],
                    help="which caption to feed the policy (PROMPT_TAXONOMY.md). "
                         "`explicit` names the object and the destination by category, "
                         "`ambiguous` names neither. One per invocation, and one "
                         "--run-id per arm.")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="default %d for the scripted policies; %d for --policy "
                         "pi05, whose per-suite budget (220/280/300, openpi's) is "
                         "then applied per scene via policy.episode_len"
                         % (MAX_STEPS_DEFAULT, PI05_MAX_STEPS_CEILING))
    # ---- pi0.5 baseline (see scripts/pi05_policy.py, prompt_pi05_baseline.md) --
    ap.add_argument("--pi05-host", default="0.0.0.0",
                    help="openpi policy server host. Use the GPU machine's IP if "
                         "the server is not on this box.")
    ap.add_argument("--pi05-port", type=int, default=8000)
    ap.add_argument("--pi05-replan", type=int, default=5,
                    help="actions consumed per inferred chunk before re-querying "
                         "(openpi's replan_steps)")
    ap.add_argument("--pi05-wait-steps", type=int, default=0,
                    help="dummy settling steps at episode start. openpi uses 10; "
                         "0 here because init_state.npz is already settled.")
    ap.add_argument("--pi05-no-rotate180", action="store_true",
                    help="DEBUG ONLY: skip the 180-degree rotation openpi applies "
                         "to match training preprocessing. Expect a near-zero "
                         "success rate; useful only to confirm the rotation is "
                         "the thing that matters.")
    ap.add_argument("--pi05-dump-frame", default=None, metavar="DIR",
                    help="write each episode's FIRST frame, exactly as the model "
                         "receives it (rotated, pad-resized to 224), into DIR as "
                         "<suite>_<dir>_<condition>_<rollout>.png. The orientation "
                         "check of prompt_pi05_baseline.md section 5.2: compare it "
                         "against the scene's frame0.png before trusting any "
                         "success rate.")
    ap.add_argument("--pi05-sketch-mode", default="none",
                    choices=["none", "overlay", "language"],
                    help="how the sketch reaches a model that has no sketch "
                         "channel. 'none': the sketch-free baseline of "
                         "report/pi05_baseline. 'overlay': circle and arrow "
                         "composited onto the image the model sees. 'language': "
                         "the same geometry described in words and appended to "
                         "the instruction -- the modality control that bounds "
                         "what any prompt could recover. Both read symbolic_"
                         "tokens only, never the ground-truth referent. See "
                         "scripts/pi05_sketch.py and prompt_pi05_recovery.md.")
    ap.add_argument("--deproject", default="plane", choices=["plane", "depth"],
                    help="pixel->world z source. 'plane': per-suite SUPPORT_Z constant, "
                         "the non-privileged default a RGB-only policy could also use. "
                         "'depth': true rendered depth at the chosen pixel -- an UPPER "
                         "BOUND, since a real RGB-only policy has no depth channel.")
    ap.add_argument("--depth-deproject", action="store_true",
                    help="deprecated alias for --deproject depth")
    ap.add_argument("--reset-warmstart", action="store_true",
                    help="zero MuJoCo's qacc_warmstart before every rollout. "
                         "determinism_r1 identified it as the sole difference "
                         "between repeats of an identical prompt. OFF by "
                         "default so new runs stay comparable with the existing "
                         "ones, which all ran with it on; recorded in "
                         "run_config.json when used. MEASURED INSUFFICIENT by "
                         "determinism_r2 -- prefer --isolate-rollouts.")
    ap.add_argument("--isolate-rollouts", action="store_true",
                    help="re-seed, reset and restore before EVERY rollout "
                         "instead of once per scene, so each rollout is a "
                         "genuinely fresh episode: clears data.ctrl, the "
                         "applied forces and the robosuite controller goal, "
                         "none of which set_init_state restores. Slower, "
                         "recorded in run_config.json, and changes the numbers "
                         "relative to every existing run.")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.depth_deproject:
        args.deproject = "depth"
    use_depth = args.deproject == "depth"

    is_pi05 = args.policy == "pi05"
    if args.max_steps is None:
        args.max_steps = PI05_MAX_STEPS_CEILING if is_pi05 else MAX_STEPS_DEFAULT
    if is_pi05:
        # pi0.5 chooses its own pixels internally -- there is nothing here to
        # deproject, and the depth intrinsics are written against IMG_H=128
        # while a pi05 env renders at 256, so the combination would be wrong
        # rather than merely useless.
        if use_depth:
            print("[error] --policy pi05 does not deproject pixels; "
                  "--deproject depth is meaningless here and its 128x128 "
                  "intrinsics are invalid at the 256 render. Drop the flag.")
            sys.exit(1)
        if args.conditions == "text_only,auto":
            # In `none` mode text_only is the only condition pi0.5 can run; in a
            # sketch mode it is `auto`, since the run's whole purpose is to
            # deliver a sketch.
            args.conditions = ("text_only" if args.pi05_sketch_mode == "none"
                               else "auto")

    if args.smoke:
        # policy is a single choice for the whole run (a scripted-oracle pass
        # and a text-only pass are two separate invocations, e.g.
        # `--policy oracle --conditions auto` then `--policy text_guess
        # --conditions text_only`) -- text_only needs a policy that reads
        # RestrictedPrompt, oracle needs one that reads symbolic_tokens, so the
        # smoke default exercises the sketch-bearing path with whatever
        # --policy was actually passed (default oracle) over "auto" only.
        if args.conditions == "text_only,auto":
            args.conditions = "text_only" if args.policy == "text_guess" else "auto"
        args.n_rollouts = 1
        scene_pairs = [("spatial", "scene_0000"), ("object", "scene_0000"), ("goal", "scene_0000")]
    else:
        scene_pairs = None

    if is_pi05:
        # The pairing rule, in both directions. pi0.5 has no sketch channel, so
        # a condition that supplies a sketch is only honest when a sketch mode
        # is actually delivering it, and a sketch mode is only meaningful when
        # a condition actually supplies one. Either mismatch would produce a row
        # whose label does not describe what the model received.
        conds = args.conditions.split(",")
        if args.pi05_sketch_mode == "none":
            bad = [c for c in conds if c != "text_only"]
            if bad:
                print("[error] --policy pi05 with --pi05-sketch-mode none is the "
                      "SKETCH-FREE baseline: the sketch would be loaded and then "
                      "discarded, producing a text-only number wearing a sketch "
                      "condition's label. Refusing %s -- either use --conditions "
                      "text_only, or pass --pi05-sketch-mode overlay|language."
                      % (",".join(bad),))
                sys.exit(1)
        else:
            bad = [c for c in conds if c == "text_only"]
            if bad:
                print("[error] --pi05-sketch-mode %s needs a condition that "
                      "carries a sketch; text_only carries none, so those rows "
                      "would silently be the baseline again. Use --conditions "
                      "auto (or a human:* source)." % (args.pi05_sketch_mode,))
                sys.exit(1)

    # --- policy/condition pairing for the SCRIPTED policies -------------------
    # `--policy` is one choice for the whole invocation, but the scripted
    # policies read different prompt types: ScriptedSketchOracle wants
    # `symbolic_tokens` off a Prompt, TextOnlyGuessPolicy wants candidate pixels
    # off a RestrictedPrompt. Mixing text_only with a sketch condition under one
    # policy therefore cannot work, and used to surface ~500 lines later as
    # `AttributeError: 'RestrictedPrompt' object has no attribute
    # 'symbolic_tokens'` — after the env was built and the first scene had run.
    # Fail here instead, with the two-invocation recipe. (Checked after the
    # smoke rewrite so `--smoke`'s own default remains legal.)
    if not is_pi05:
        conds = args.conditions.split(",")
        sketchy = [c for c in conds if c != "text_only"]
        texty = [c for c in conds if c == "text_only"]
        if args.policy == "oracle" and texty:
            print("[error] --policy oracle reads the sketch's symbolic_tokens, and "
                  "text_only carries no sketch. Split into two invocations sharing "
                  "one --run-id; both append to the same results.csv:\n"
                  "    ... --conditions %s --policy oracle      --run-id RID\n"
                  "    ... --conditions text_only --policy text_guess --run-id RID --resume"
                  % (",".join(sketchy) or "auto",))
            sys.exit(1)
        if args.policy == "text_guess" and sketchy:
            print("[error] --policy text_guess reads RestrictedPrompt candidate pixels "
                  "and never looks at a sketch, so %s would be relabelled baseline "
                  "rows. Run those under --policy oracle in a separate invocation "
                  "with the same --run-id." % (",".join(sketchy),))
            sys.exit(1)

    conditions = resolve_conditions(args.conditions.split(","))
    has_human = any(c.startswith("human") for c in args.conditions.split(","))

    manifest = load_manifest()
    all_pairs = [(e["suite"], e["dir"]) for e in manifest]
    tier_of = {(e["suite"], e["dir"]): e["tier"] for e in manifest}
    nonrepro = load_nonreproducible()

    if scene_pairs is None:
        roster = all_pairs
        wanted = parse_scene_filter(args.scenes)
        if wanted is not None:
            roster = [p for p in roster if p in wanted]
        if has_human:
            subset = set(load_subset_pairs())
            roster = [p for p in roster if p in subset]
        scene_pairs = roster

    run_id = args.run_id or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUN_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.csv")
    video_dir = os.path.join(run_dir, "videos")
    if args.video:
        os.makedirs(video_dir, exist_ok=True)
    if args.pi05_dump_frame:
        os.makedirs(args.pi05_dump_frame, exist_ok=True)

    if os.path.exists(results_path) and not args.resume and not args.smoke:
        print(f"[error] {results_path} exists; pass --resume to continue it "
              f"or a different --run-id to start fresh.")
        sys.exit(1)

    # Appending into a results.csv written under a DIFFERENT schema silently
    # misaligns it: DictWriter emits values in RESULT_FIELDS order regardless of
    # what header the file already carries, so resuming a run written before the
    # determinism fingerprints were added would give those rows three extra
    # trailing values the header cannot name. Refuse instead of corrupting a
    # finished run -- every pre-existing run in outputs/rollouts/ is in exactly
    # this position.
    if os.path.exists(results_path):
        with open(results_path) as f:
            old_header = next(csv.reader(f), [])
        if old_header and old_header != RESULT_FIELDS:
            missing = [c for c in RESULT_FIELDS if c not in old_header]
            extra = [c for c in old_header if c not in RESULT_FIELDS]
            print(f"[error] {results_path} was written under a different results "
                  f"schema and cannot be appended to.\n"
                  f"        columns this build adds:  {missing or 'none'}\n"
                  f"        columns this build drops: {extra or 'none'}\n"
                  f"        Use a new --run-id. The old run stays readable as it is.")
            sys.exit(1)

    if os.path.exists(results_path):
        # Same trap as --pi05-sketch-mode below, one axis over: the two prompt
        # arms write the same condition label and the same rollout indices, so
        # resuming an explicit run under an ambiguous flag would skip every
        # rollout as already-done and leave the file looking like a complete
        # ambiguous arm. One --run-id per prompt type.
        with open(results_path) as f:
            prior = {r.get("prompt_type", "") for r in csv.DictReader(f)}
        prior.discard("")
        other = prior - {args.prompt_type}
        if other:
            print("[error] %s already holds rows from --prompt-type %s. Both arms "
                  "write the same condition label, so resuming here would skip "
                  "this arm's rollouts as already done. Use a separate --run-id "
                  "per prompt type." % (results_path, "/".join(sorted(other))))
            sys.exit(1)

    if is_pi05 and os.path.exists(results_path):
        # overlay and language both write condition="auto", so they share a
        # resume key (suite, dir, condition, rollout_idx). Pointed at one
        # --run-id, the second mode would be skipped as already-done and the
        # comparison would silently be one arm run twice. One run-id per mode.
        with open(results_path) as f:
            prior = {r.get("pi05_sketch_mode", "") for r in csv.DictReader(f)}
        prior.discard("")
        other = prior - {args.pi05_sketch_mode}
        if other:
            print("[error] %s already holds rows from --pi05-sketch-mode %s, and "
                  "every mode writes the same condition label, so resuming here "
                  "would skip this mode's rollouts as already done. Use a "
                  "separate --run-id per mode."
                  % (results_path, "/".join(sorted(other))))
            sys.exit(1)

    done = load_done(results_path) if args.resume else set()

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO
                                          ).decode().strip()
    except Exception:
        git_sha = None
    run_config = dict(conditions=args.conditions.split(","), scenes_requested=args.scenes,
                      policy=args.policy, sketch_route=args.sketch_route,
                      n_rollouts=args.n_rollouts, max_steps=args.max_steps,
                      prompt_type=args.prompt_type,
                      deproject=args.deproject, depth_deproject=use_depth,
                      depth_patch_k=DEPTH_PATCH_K if use_depth else None,
                      # the honesty stamp: a depth run reads true rendered z at
                      # the chosen pixel, which no RGB-only policy can do.
                      upper_bound=use_depth,
                      upper_bound_reason=("rendered-depth deprojection: privileged "
                                          "z a real RGB-only policy has no access to"
                                          if use_depth else None),
                      video=args.video, reset_warmstart=args.reset_warmstart,
                      isolate_rollouts=args.isolate_rollouts,
                      git_sha=git_sha, n_scenes=len(scene_pairs), run_id=run_id,
                      success_window=SUCCESS_WINDOW, grasp_lift_th=GRASP_LIFT_TH,
                      # pi0.5 provenance: without these a success rate cannot be
                      # attributed to a checkpoint or a preprocessing choice.
                      pi05=(dict(host=args.pi05_host, port=args.pi05_port,
                                 replan_steps=args.pi05_replan,
                                 wait_steps=args.pi05_wait_steps,
                                 rotate180=not args.pi05_no_rotate180,
                                 render_size=LIBERO_ENV_RESOLUTION,
                                 resize_size=PI05_RESIZE_SIZE,
                                 sketch_mode=args.pi05_sketch_mode,
                                 wrist_camera=True) if is_pi05 else None))
    # A full run is TWO invocations (--policy oracle over `auto`, --policy
    # text_guess over `text_only`), so a plain overwrite here records only the
    # last one -- which is why full_run/run_config.json claims conditions
    # ["auto"] while its results.csv holds both. Accumulate instead, and hoist
    # upper_bound to the run level: if ANY invocation used depth, the run's
    # numbers are an upper bound.
    cfg_path = os.path.join(run_dir, "run_config.json")
    invocations = []
    if os.path.exists(cfg_path):
        prev = json.load(open(cfg_path))
        invocations = prev.get("invocations", [prev])
    invocations.append(run_config)
    json.dump(dict(run_id=run_id,
                   deproject=sorted({i.get("deproject", "plane") for i in invocations}),
                   upper_bound=any(i.get("upper_bound") for i in invocations),
                   conditions=sorted({c for i in invocations for c in i.get("conditions", [])}),
                   invocations=invocations),
              open(cfg_path, "w"), indent=2)

    write_header = not os.path.exists(results_path)
    n_written = 0
    for si, (suite, dir_) in enumerate(scene_pairs):
        tier = tier_of.get((suite, dir_), "?")
        if (suite, dir_) in nonrepro:
            row = {k: "" for k in RESULT_FIELDS}
            row.update(suite=suite, dir=dir_, tier=tier, condition="", policy=args.policy,
                      sketch_route=args.sketch_route, rollout_idx=-1, skipped=True,
                      prompt_type=args.prompt_type,
                      skip_reason=f"nonreproducible:{nonrepro[(suite,dir_)].get('residual_px')}px")
            append_rows(results_path, [row], write_header); write_header = False
            print(f"[{si+1}/{len(scene_pairs)}] {suite}/{dir_} SKIP (nonreproducible)")
            continue

        env, meta, flat = open_scene_env(
            suite, dir_, depth=use_depth,
            render_size=(LIBERO_ENV_RESOLUTION if is_pi05 else IMG_H),
            wrist=is_pi05)
        model, data = env.sim.model, env.sim.data
        # one depth render per SCENE, taken at the pinned initial state (before
        # any policy has moved anything) -- the sketch is drawn on that same
        # initial frame, so this is the frame its pixels refer to. Shared by
        # every condition x rollout for the scene, matching the scene-major
        # env reuse (issue 8).
        z_at_pixel = None
        if use_depth:
            env.set_init_state(flat)
            z_at_pixel = make_z_at_pixel(env, render_depth(env))
        rows = []
        try:
            for label, root_template in conditions:
                for r_idx in range(args.n_rollouts):
                    key = (suite, dir_, label, r_idx)
                    if key in done:
                        continue
                    rng = np.random.default_rng(stable_seed(suite, dir_, label, r_idx))
                    prompt, ok = build_prompt(label, root_template, suite, dir_, tier, meta,
                                              args.sketch_route, rng, z_at_pixel=z_at_pixel,
                                              prompt_type=args.prompt_type)
                    if not ok:
                        row = {k: "" for k in RESULT_FIELDS}
                        row.update(suite=suite, dir=dir_, tier=tier, condition=label,
                                  policy=args.policy, sketch_route=args.sketch_route,
                                  rollout_idx=r_idx, skipped=True, skip_reason="sketch_absent",
                                  prompt_type=args.prompt_type)
                        rows.append(row); continue
                    policy = make_policy(args.policy, rng_seed=int(rng.integers(0, 2**31)),
                                         args=args)
                    if is_pi05 and args.pi05_dump_frame:
                        policy.dump_first_frame_to = os.path.join(
                            args.pi05_dump_frame, f"{suite}_{dir_}_{label}_{r_idx}.png")
                    video_path = (os.path.join(video_dir, f"{suite}_{dir_}_{label}_{r_idx}.mp4")
                                  if args.video else None)
                    res = run_rollout(env, model, data, flat, meta, policy, prompt,
                                      args.max_steps, video_path,
                                      reset_warmstart=args.reset_warmstart,
                                      isolate=args.isolate_rollouts)
                    # --isolate-rollouts resets the env, so the scene-level
                    # handles taken at open_scene_env may no longer be current.
                    model, data = env.sim.model, env.sim.data
                    row = {k: "" for k in RESULT_FIELDS}
                    row.update(suite=suite, dir=dir_, tier=tier, condition=label,
                              policy=args.policy, sketch_route=args.sketch_route,
                              rollout_idx=r_idx, skipped=False, skip_reason="",
                              deproject=args.deproject, prompt_type=args.prompt_type,
                              pi05_sketch_mode=(args.pi05_sketch_mode if is_pi05 else ""),
                              grasp_success_flag=meta.get("grasp", {}).get("grasp_success"))
                    row.update(res)
                    rows.append(row)
        finally:
            try:
                env.close()
            except Exception:
                pass
            gc.collect()

        if rows:
            append_rows(results_path, rows, write_header)
            write_header = False
            n_written += len(rows)
        ok_rows = [r for r in rows if not r["skipped"]]
        succ = sum(1 for r in ok_rows if r.get("success_sustained"))
        print(f"[{si+1}/{len(scene_pairs)}] {suite}/{dir_} tier={tier} "
              f"{len(ok_rows)} rollout(s), {succ} sustained-success")

    write_summary(run_dir, results_path)
    print(f"\nwrote {n_written} new row(s) to {results_path}")


def write_summary(run_dir, results_path):
    import collections
    rows = list(csv.DictReader(open(results_path))) if os.path.exists(results_path) else []
    rows = [r for r in rows if r["skipped"] != "True"]

    def frac(rs, field):
        rs = list(rs)
        if not rs:
            return None
        return round(sum(1 for r in rs if r[field] == "True") / len(rs), 4)

    def rate(rs):
        return frac(rs, "success_sustained")

    def grasp_block(rs):
        """grasped_any is NOT a targeting metric and must never be read as one.
        It counts lifting ANY object, so a baseline guessing uniformly among
        candidates can beat a policy that aims correctly but misses -- which is
        exactly what Object showed: random targeting sometimes lands on a
        graspable tall bottle, while the oracle aims true at a flat box and
        misses it. grasped_correct (the target instance, and only it) is the
        one that answers 'did the policy go for the right thing', so the two
        are always reported together."""
        return dict(grasped_any=frac(rs, "grasped_any"),
                    grasped_correct=frac(rs, "correct_instance_grasped"),
                    correct_destination=frac(rs, "correct_destination"),
                    n=len(list(rs)))

    by_cond = collections.defaultdict(list)
    by_cond_suite_tier = collections.defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
        by_cond_suite_tier[(r["condition"], r["suite"], r["tier"])].append(r)

    deprojects = sorted({r.get("deproject", "") for r in rows} - {""})
    summary = dict(
        n_rows=len(rows),
        deproject=deprojects,
        upper_bound=("depth" in deprojects),
        upper_bound_reason=("rendered-depth deprojection: privileged z a real "
                            "RGB-only policy has no access to"
                            if "depth" in deprojects else None),
        success_rate_by_condition={c: rate(rs) for c, rs in by_cond.items()},
        success_rate_by_condition_suite_tier={
            f"{c}|{s}|{t}": rate(rs) for (c, s, t), rs in by_cond_suite_tier.items()},
        grasp_by_condition={c: grasp_block(rs) for c, rs in by_cond.items()},
        grasp_by_condition_suite_tier={
            f"{c}|{s}|{t}": grasp_block(rs) for (c, s, t), rs in by_cond_suite_tier.items()},
    )
    a = summary["success_rate_by_condition"]
    if "auto" in a and "text_only" in a and a["auto"] is not None and a["text_only"] is not None:
        summary["headline_auto_minus_text_only"] = round(a["auto"] - a["text_only"], 4)
    human_conds = [c for c in a if c.startswith("human")]
    if human_conds and "text_only" in a:
        best = max((c for c in human_conds if a[c] is not None), key=lambda c: 0, default=None)
        if best:
            summary["headline_human_minus_text_only"] = {
                c: round(a[c] - a["text_only"], 4) for c in human_conds if a[c] is not None}
    if human_conds and "auto" in a and a["auto"] is not None:
        summary["headline_human_minus_auto"] = {
            c: round(a[c] - a["auto"], 4) for c in human_conds if a[c] is not None}

    grasp_strat = collections.defaultdict(list)
    for r in rows:
        grasp_strat[(r["condition"], r["grasp_success_flag"])].append(r)
    summary["success_rate_by_condition_x_grasp_success_flag"] = {
        f"{c}|{g}": rate(rs) for (c, g), rs in grasp_strat.items()}

    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
    print(f"wrote {os.path.join(run_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
