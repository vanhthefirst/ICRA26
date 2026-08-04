"""
Sketch-Prompted VLA — policy protocol + baselines (issues 3 and 6 of
prompt_libero_rollout_harness.md).

Pure numpy: no robosuite / mujoco / libero import. This is the seam a GPU
machine later plugs a learned (UniVLA) policy into in place of the three
implementations here.

Issue 3 (static prompt, separate channel): `Prompt` carries the sketch on its
own `sketch_rgb` / `symbolic_tokens` fields, drawn once at reset() and held for
the whole episode -- `act()` never receives it again, only the live `obs`.

Issue 6 (baseline firewall): `RestrictedPrompt` has no `sketch_rgb`,
`symbolic_tokens`, `target`, `destination`, `pick_px` or `place_px` attribute at
all. A policy that tries to read one gets an `AttributeError` at the first
rollout, not a quietly optimistic number -- the same rigour
build_human_study_bundle.py applies to its browser payload, enforced here by
the type itself rather than by a runtime scan.

Every policy exposes the same protocol so the harness never special-cases one:

    class SketchPolicy(Protocol):
        def reset(self, prompt) -> None: ...
        def act(self, obs: dict, t: int) -> np.ndarray: ...   # shape (7,)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from sketch_geometry import deproject_pixel_xy_to_world_xy

ADIM = 7

# Per-suite support-plane height (world z, m) a deprojected pixel is intersected
# against (issue 2 rung 1, the honest default) -- the height of the VISUAL
# CENTRE the pixel was originally projected from, not the floor/table plane
# itself. Measured directly off the pinned init states (vcenter z, one scene
# per suite, spatial/scene_0000, object/scene_0000, goal/scene_0000): spatial
# bowl 0.927m, object milk/basket/ketchup 0.06-0.07m (grocery visual centres
# sit well above the floor plane -- 0.03 was measured wrong on the first pass
# and undershot every grasp), goal bowl 0.927m. A single suite-wide constant is
# enough: the motion's two-stage approach (HOVER_DZ then APPROACH_DZ) already
# absorbs a few cm of z error, the same tolerance the builders' own
# scripted_grasp relies on (SUITE_FACTS.md S1 #3).
SUPPORT_Z = {"spatial": 0.92, "object": 0.065, "goal": 0.92}

# KNOWN LIMITATION of the plane route, measured not assumed. Object's 0.065 was
# fitted to the tall groceries and is wrong for the flat ones: deprojecting each
# instance's recorded all_pixels centre at 0.065 and comparing against its true
# meta['placements'] xy (225 instances, all 38 Object scenes) gives ~1cm for
# basket/milk/orange_juice/salad_dressing/ketchup but 7.5-8.2cm for
# chocolate_pudding/butter/cream_cheese, whose visual centres sit at z~0.010-0.016.
# 5.6cm of z error becomes ~8cm of LATERAL error at this camera angle, which the
# motion's HOVER_DZ/APPROACH_DZ do NOT absorb -- those absorb error along z.
# Those three categories appear only in the non-control Object tiers, which is
# why Object control ties between conditions while the other tiers do not.
# The fix is NOT a per-category table: category identity is ground truth, the
# same leak class as a nearest-instance snap, and Goal metas carry no
# `placements` to validate it against. Use --deproject depth (z_at_pixel below).


def resolve_z(scene_meta, pixel_xy, suite):
    """The single place a pixel's world z is decided, for EVERY policy.

    `scene_meta['z_at_pixel']` is present only under --deproject depth; it reads
    true rendered depth at the pixel the policy actually chose. Absent, this
    falls back to the per-suite SUPPORT_Z constant -- the non-privileged route,
    the one an RGB-only policy could also run.

    Both routes are resolved per PIXEL, so pick and place each get their own
    height: the arrow head usually lands on a basket or plate at a different
    height from the circled object, and forcing one z on both was wrong even
    when that z was right for the pick."""
    fn = scene_meta.get("z_at_pixel")
    if fn is not None:
        return float(fn(pixel_xy))
    return SUPPORT_Z.get(suite, 0.90)

# Category keyword -> canonical name, for the (recorded, not action-critical --
# see TextOnlyGuessPolicy) instruction parse. Covers every target category
# across all three suites (SCHEMA.md, SUITE_FACTS.md S2).
CATEGORY_KEYWORDS = {
    "black bowl": "akita_black_bowl", "bowl": "akita_black_bowl",
    "plate": "plate", "ramekin": "glazed_rim_porcelain_ramekin", "cookies": "cookies",
    "milk": "milk", "alphabet soup": "alphabet_soup", "tomato sauce": "tomato_sauce",
    "orange juice": "orange_juice", "ketchup": "ketchup", "bbq sauce": "bbq_sauce",
    "salad dressing": "salad_dressing", "chocolate pudding": "chocolate_pudding",
    "cream cheese": "cream_cheese", "butter": "butter", "basket": "basket",
    "wine bottle": "wine_bottle", "wine": "wine_bottle", "moka pot": "moka_pot",
}


def parse_category(instruction):
    """Best-effort category name from a vague instruction. Longer keywords are
    checked first so "black bowl" wins over the bare "bowl". Used for
    record-keeping only -- TextOnlyGuessPolicy acts on the harness-supplied
    candidate pixels, not on this parse, since there is no real perception
    model here to turn a category name into a location."""
    low = instruction.lower()
    for kw in sorted(CATEGORY_KEYWORDS, key=len, reverse=True):
        if kw in low:
            return CATEGORY_KEYWORDS[kw]
    return None


@dataclass
class Prompt:
    """The full prompt a sketch-conditioned policy receives at reset(). Which
    of sketch_rgb / symbolic_tokens is populated depends on --sketch-route."""
    instruction: str
    sketch_rgb: Optional[np.ndarray] = None            # (128,128,3) uint8 or None
    symbolic_tokens: Optional[Dict[str, Any]] = None   # {circle:{cx,cy,rx,ry}, arrow:{x0,y0,x1,y1}}
    scene_meta: Optional[Dict[str, Any]] = None        # {suite, dir, tier, camera_matrix}


@dataclass
class RestrictedPrompt:
    """issue 6: the text-only baseline's prompt. Deliberately has no
    sketch_rgb / symbolic_tokens / target / destination / pick_px / place_px
    attribute -- a leak is an AttributeError, not a silently optimistic number.

    candidate_pick_px / candidate_place_px are UNLABELLED, harness-shuffled
    pixel centres: every same-category object, every destination-typed
    instance. That is the minimum a policy perceiving the raw RGB frame would
    already know; it is not the sketch, and it never says which entry is the
    intended one."""
    instruction: str
    scene_meta: Optional[Dict[str, Any]] = None        # {suite, dir, tier, camera_matrix}
    candidate_pick_px: Optional[List[Tuple[int, int]]] = None
    candidate_place_px: Optional[List[Tuple[int, int]]] = None


class SketchPolicy(Protocol):
    def reset(self, prompt) -> None: ...
    def act(self, obs: dict, t: int) -> np.ndarray: ...


def _eef(obs):
    for k in ("robot0_eef_pos", "eef_pos"):
        if k in obs:
            return np.asarray(obs[k])
    return None


class _PickPlaceMotion:
    """Shared open-loop pick-transport-place state machine, driven purely by
    (obs, t) -- the servo() recipe build_validation_set_object_wsl.py's
    scripted_grasp uses at build time, restructured as a per-step function
    since SketchPolicy.act() is called once per environment step rather than
    given a loop to run.

    Gripper close_sign / approach_dz default to -1.0 / 0.005 (the builders'
    first (dz, cs) try, SUITE_FACTS.md S1 #3) but are overridable per call --
    the harness passes the scene's OWN measured meta['grasp'] values when
    known (a mechanical property of that object/gripper pair, not the sketch's
    disambiguating identity, so reusing it is not a ground-truth leak for the
    firewalled baseline either). Some objects (wine_bottle, plate) fail to
    lift under this recipe regardless -- a known, recorded limitation, the
    same asymmetry Goal's own DATASHEET records for 6 of 38 scenes, not
    something this harness papers over."""
    GAIN = 8.0
    CLOSE_SIGN = -1.0
    APPROACH_DZ = 0.005
    RELEASE_DZ = 0.04
    LIFT_DZ = 0.18
    HOVER_DZ = 0.12

    def __init__(self, pick_xyz, place_xyz, close_sign=None, approach_dz=None):
        pick = np.asarray(pick_xyz, dtype=float)
        place = np.asarray(place_xyz, dtype=float)
        close_sign = self.CLOSE_SIGN if close_sign is None else close_sign
        approach_dz = self.APPROACH_DZ if approach_dz is None else approach_dz
        above_pick = pick + [0, 0, self.HOVER_DZ]
        at_pick = pick + [0, 0, approach_dz]
        lifted = pick + [0, 0, self.LIFT_DZ]
        above_place = np.array([place[0], place[1], lifted[2]])
        at_place = place + [0, 0, self.RELEASE_DZ]
        retreat = np.array([at_place[0], at_place[1], lifted[2]])
        g_open, g_close = -close_sign, close_sign
        # (goal_xyz, grip, n_steps) -- mirrors scripted_grasp's servo() calls
        self.phases = [
            (above_pick, g_open, 30), (at_pick, g_open, 25),
            (at_pick, g_close, 12), (lifted, g_close, 30),
            (above_place, g_close, 40), (at_place, g_close, 25),
            (at_place, g_open, 12), (retreat, g_open, 20),
        ]
        self._bounds = np.cumsum([0] + [p[2] for p in self.phases])

    def act(self, obs, t):
        idx = min(int(np.searchsorted(self._bounds, t, side="right") - 1), len(self.phases) - 1)
        goal, grip, _ = self.phases[max(idx, 0)]
        a = np.zeros(ADIM)
        e = _eef(obs)
        if e is not None:
            a[:3] = np.clip((np.asarray(goal) - e) * self.GAIN, -1, 1)
        a[-1] = grip
        return a

    @property
    def episode_len(self):
        return int(self._bounds[-1])


class ScriptedSketchOracle:
    """Upper bound: how much of the task is solvable given a PERFECT reading
    of the sketch. Deprojects the circle centre (pick) and arrow head (place)
    via sketch_geometry against the suite's support-plane height, then runs
    the shared pick-transport-place motion. Proves the loop end to end without
    a GPU -- this is what a learned policy is later compared against."""

    def __init__(self):
        self._motion = None
        self.last_pick_xy = self.last_place_xy = None
        self.last_z_pick = self.last_z_place = None

    def reset(self, prompt):
        tok = prompt.symbolic_tokens
        if tok is None:
            raise ValueError("ScriptedSketchOracle needs symbolic_tokens "
                              "(run with --sketch-route tokens)")
        meta = prompt.scene_meta or {}
        suite = meta.get("suite", "spatial")
        cam = meta["camera_matrix"]
        pick_px = (tok["circle"]["cx"], tok["circle"]["cy"])
        place_px = (tok["arrow"]["x1"], tok["arrow"]["y1"])     # arrow head = destination
        # issue 2: plane constant or true rendered depth, decided by the harness
        # (--deproject) and applied identically to this policy and the
        # firewalled text-only baseline, so the method never confounds the gap.
        z_pick = resolve_z(meta, pick_px, suite)
        z_place = resolve_z(meta, place_px, suite)
        pick_xy = deproject_pixel_xy_to_world_xy(pick_px, z_pick, cam)
        place_xy = deproject_pixel_xy_to_world_xy(place_px, z_place, cam)
        self.last_pick_xy, self.last_place_xy = pick_xy, place_xy
        self.last_z_pick, self.last_z_place = z_pick, z_place
        self._motion = _PickPlaceMotion((pick_xy[0], pick_xy[1], z_pick),
                                        (place_xy[0], place_xy[1], z_place),
                                        close_sign=meta.get("grasp_close_sign"),
                                        approach_dz=meta.get("grasp_approach_dz"))

    def act(self, obs, t):
        return self._motion.act(obs, t)

    @property
    def episode_len(self):
        return self._motion.episode_len


class TextOnlyGuessPolicy:
    """issue 6 baseline. RestrictedPrompt only: no sketch, no truth fields.
    Picks uniformly at random among the harness's unlabelled candidate pick /
    place pixels (the minimum a policy perceiving the raw RGB frame would
    already have), deprojects the CHOSEN pixel, runs the same motion. This is
    the denominator every headline gap in this project is measured against --
    ~25% expected on a 4-bowl referential scene."""

    def __init__(self, rng_seed=0):
        self._rng = np.random.default_rng(rng_seed)
        self._motion = None
        self.last_category = None
        self.last_pick_px = self.last_place_px = None
        self.last_z_pick = self.last_z_place = None

    def reset(self, prompt):
        self.last_category = parse_category(prompt.instruction)
        meta = prompt.scene_meta or {}
        suite = meta.get("suite", "spatial")
        cam = meta["camera_matrix"]
        picks = prompt.candidate_pick_px or [(64, 64)]
        places = prompt.candidate_place_px or [(64, 64)]
        pick_px = picks[int(self._rng.integers(0, len(picks)))]
        place_px = places[int(self._rng.integers(0, len(places)))]
        self.last_pick_px, self.last_place_px = pick_px, place_px
        # same deprojection as the oracle, resolved AFTER the random choice, so
        # this baseline is never handicapped by a worse z than the sketch route.
        z_pick = resolve_z(meta, pick_px, suite)
        z_place = resolve_z(meta, place_px, suite)
        pick_xy = deproject_pixel_xy_to_world_xy(pick_px, z_pick, cam)
        place_xy = deproject_pixel_xy_to_world_xy(place_px, z_place, cam)
        self.last_z_pick, self.last_z_place = z_pick, z_place
        self._motion = _PickPlaceMotion((pick_xy[0], pick_xy[1], z_pick),
                                        (place_xy[0], place_xy[1], z_place),
                                        close_sign=meta.get("grasp_close_sign"),
                                        approach_dz=meta.get("grasp_approach_dz"))

    def act(self, obs, t):
        return self._motion.act(obs, t)

    @property
    def episode_len(self):
        return self._motion.episode_len


class NoOpPolicy:
    """Harness sanity check: zero actions for the whole episode. Every scene
    passed the builders' not-pre-solved gate, so this must score False
    everywhere -- a True here means the harness is wrong, not the scene."""

    def reset(self, prompt):
        pass

    def act(self, obs, t):
        return np.zeros(ADIM)

    @property
    def episode_len(self):
        return 1
