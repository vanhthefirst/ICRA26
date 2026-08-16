"""
Sketch-Prompted VLA -- pi0.5-LIBERO as the learned, sketch-free baseline.

This is the seam scripts/sketch_policies.py reserved for "a GPU machine later
plugs a learned policy in here", filled with Physical Intelligence's
pi0.5-LIBERO checkpoint (`gs://openpi-assets/checkpoints/pi05_libero`, openpi
config `pi05_libero`). It implements the same SketchPolicy protocol as the
scripted baselines, so scripts/rollout_sketch.py drives it unchanged:

    class SketchPolicy(Protocol):
        def reset(self, prompt) -> None: ...
        def act(self, obs: dict, t: int) -> np.ndarray: ...   # shape (7,)

WHY THIS IS THE NO-SKETCH BASELINE, WITH NO EDITS TO THE MODEL
--------------------------------------------------------------
pi0.5 consumes (base image, wrist image, proprioceptive state, language string)
and emits an action chunk. It has no sketch channel to remove. "pi0.5 without
the sketch" is therefore simply pi0.5 in its native mode, fed `meta
['instruction']` -- the same instruction string the sketch conditions get. There
is nothing to ablate; the honest baseline is the stock checkpoint.

The issue-6 firewall is preserved for free: this class reads exactly one field
off the prompt, `instruction`, plus `scene_meta['suite']` for the step budget.
It never touches `pick_px`, `place_px`, `target`, `destination`, `sketch_rgb`
or `symbolic_tokens`, and `require_text_only=True` (the default) makes it REFUSE
a sketch-bearing Prompt outright -- so a pi0.5 number reported as a baseline
cannot have been produced with a sketch in the loop by accident.

ARCHITECTURE: SERVER + CLIENT
-----------------------------
openpi holds the model in a separate process behind a websocket
(`scripts/serve_policy.py`), because the model wants a modern JAX/PyTorch stack
and LIBERO wants python 3.8 + mujoco. That split is convenient here rather than
incidental: the simulator keeps running in the `libero` conda env exactly as it
does for the scripted policies, and only `openpi-client` (a thin, dependency-
light package) is installed alongside it.

    Terminal 1 (GPU, openpi repo, python 3.11):
        uv run scripts/serve_policy.py --env LIBERO
    Terminal 2 (conda activate libero, this repo):
        python scripts/rollout_sketch.py --policy pi05 --conditions text_only ...

FIVE PREPROCESSING FACTS THAT ARE NOT OPTIONAL
---------------------------------------------
Each of these silently degrades the success rate rather than raising an error,
which is why they are stated here and verified by the standard-suite
reproduction run described in brief_pi05_baseline.md before any number off my
own scenes is believed.

 1. RESOLUTION. openpi renders LIBERO at 256x256 and pad-resizes to 224 before
    inference; pi0.5-LIBERO never saw a 128x128 frame. My suites render at 128
    because that is the sketch canvas. The two are decoupled: the harness opens
    a 256 env for a pi0.5 run, and the sketch's 128-space pixel coordinates are
    not consulted by this policy at all.
 2. 180-DEGREE ROTATION. openpi applies `img[::-1, ::-1]` to both cameras, to
    match the orientation of the OpenVLA `modified_libero_rlds` data pi0.5-LIBERO
    was fine-tuned on. SUITE_FACTS.md's "obs['agentview_image'] is already
    correctly oriented -- do not flip it" governs the PROJECTION path, where
    pixels must line up with frame0.png; it does not govern what the model is
    fed. Both are true at once and they are about different things.
 3. WRIST CAMERA. pi0.5-LIBERO takes `robot0_eye_in_hand_image` as well as
    `agentview_image`. My harness has never rendered the wrist camera, because
    no scripted policy needed it.
 4. STATE IS 8-D: eef position (3), eef quaternion as axis-angle (3), gripper
    qpos (2). Not the 7-D action vector, and the quaternion conversion is
    robosuite's, reproduced verbatim below.
 5. CHUNKED CONTROL. The model returns an action chunk; openpi executes
    `replan_steps` of it (5) and re-queries. Consuming only the first action, or
    the whole chunk, are both wrong.

Legacy note: this project was once called DrawVLA in some script headers. The
name is Sketch-Prompted VLA.
"""

import math
import time

import numpy as np

ADIM = 7

# openpi's per-suite step budgets (examples/libero/main.py), set from the longest
# training demo in each suite. My scenes are single pick-and-place and closer to
# the spatial/object end, but the budget is a ceiling, not a target, and cutting
# it turns a slow success into a recorded failure.
PI05_MAX_STEPS = {"spatial": 220, "object": 280, "goal": 300}
PI05_MAX_STEPS_DEFAULT = 300

# The resolution openpi renders LIBERO at, and the resolution it feeds the model.
LIBERO_ENV_RESOLUTION = 256
RESIZE_SIZE = 224
REPLAN_STEPS = 5

AGENTVIEW_KEY = "agentview_image"
WRIST_KEY = "robot0_eye_in_hand_image"


def quat2axisangle(quat):
    """Verbatim from robosuite (via openpi's examples/libero/main.py).

    Reproduced rather than imported so that the observation this policy builds
    is defined in one file I control, and cannot drift if the robosuite pinned
    in the `libero` env differs from the one openpi was written against."""
    quat = np.asarray(quat, dtype=float).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _rot180(img):
    """openpi's `np.ascontiguousarray(img[::-1, ::-1])` -- a 180-degree rotation
    (both axes), NOT the single vertical flip that appears elsewhere in this
    repo's geometry code."""
    return np.ascontiguousarray(np.asarray(img)[::-1, ::-1])


class Pi05ServerPolicy:
    """pi0.5-LIBERO behind an openpi websocket policy server.

    Parameters mirror openpi's `examples/libero/main.py` Args so that a
    discrepancy between my harness and the reference eval loop is visible as a
    parameter difference rather than buried in code.

    Diagnostics recorded per episode (read by the harness and written into
    results.csv): `n_infer_calls`, `infer_ms_mean`, `infer_ms_max`.
    """

    def __init__(self, host="0.0.0.0", port=8000, replan_steps=REPLAN_STEPS,
                 resize_size=RESIZE_SIZE, rotate180=True, num_steps_wait=0,
                 require_text_only=True, dump_first_frame_to=None,
                 sketch_mode="none"):
        if sketch_mode not in ("none", "overlay", "language"):
            raise ValueError("sketch_mode must be none|overlay|language, got %r"
                             % (sketch_mode,))
        # "none" is the baseline reported in report/pi05_baseline: the stock
        # checkpoint, text instruction only. "overlay" and "language" are the
        # two zero-shot recovery arms -- see scripts/pi05_sketch.py for why they
        # are a matched pair rather than alternatives.
        self.sketch_mode = sketch_mode
        self._tokens = None
        self.host = host
        self.port = port
        self.replan_steps = int(replan_steps)
        self.resize_size = int(resize_size)
        self.rotate180 = bool(rotate180)
        # openpi steps 10 dummy actions at episode start to let dropped objects
        # settle. My scenes do not need it: capture_scene_init_states.py
        # pinned each init_state.npz AFTER settling, and frame0.png is that
        # settled frame -- the same frame the sketch was drawn on. Defaulting
        # this to 0 keeps every condition starting from the identical state.
        self.num_steps_wait = int(num_steps_wait)
        self.require_text_only = bool(require_text_only)
        self.dump_first_frame_to = dump_first_frame_to

        self._client = None
        self._resize_with_pad = None
        self._convert_to_uint8 = None

        self.instruction = None
        self._logged_instruction = None
        self.episode_len = PI05_MAX_STEPS_DEFAULT
        self._plan = []
        self.n_infer_calls = 0
        self._infer_ms = []
        self._dumped = False

    # ------------------------------------------------------------ connection --
    def _connect(self):
        """Lazy import + connect, so this module stays importable (and syntax-
        checkable) on a machine with no openpi-client and no GPU."""
        if self._client is not None:
            return
        try:
            from openpi_client import image_tools
            from openpi_client import websocket_client_policy
        except ImportError as e:
            raise ImportError(
                "openpi-client is not installed in this environment. From the "
                "openpi checkout, inside the `libero` env:\n"
                "    uv pip install -e packages/openpi-client\n"
                "(or `pip install -e /path/to/openpi/packages/openpi-client`)\n"
                "The MODEL does not run here -- only the websocket client does; "
                "the checkpoint is served by scripts/serve_policy.py on the GPU "
                "machine.\noriginal error: %s" % (e,))
        # openpi's own resize is used rather than a cv2 equivalent: it is
        # PIL-bilinear with zero padding, and the model is sensitive to the
        # interpolation it was trained under. A visually identical cv2 resize is
        # not the same function.
        self._resize_with_pad = image_tools.resize_with_pad
        self._convert_to_uint8 = image_tools.convert_to_uint8
        self._client = websocket_client_policy.WebsocketClientPolicy(self.host, self.port)
        meta = None
        try:
            meta = self._client.get_server_metadata()
        except Exception:
            pass
        self.server_metadata = meta

    # ----------------------------------------------------------------- reset --
    def reset(self, prompt):
        """In `none` mode: reads `prompt.instruction` and
        `prompt.scene_meta['suite']`, nothing else. In the two sketch modes it
        additionally reads `prompt.symbolic_tokens` -- the drawn geometry, and
        still never `target`, `destination`, `pick_px` or `place_px`. See the
        firewall note in the module docstring."""
        if self.sketch_mode == "none" and self.require_text_only:
            leaked = [f for f in ("sketch_rgb", "symbolic_tokens")
                      if getattr(prompt, f, None) is not None]
            if leaked:
                raise ValueError(
                    "Pi05ServerPolicy was handed a sketch-bearing prompt (%s) "
                    "while sketch_mode='none'. pi0.5-LIBERO has no sketch "
                    "channel, so this run would be a text-only number wearing a "
                    "sketch condition's label. Either use --conditions "
                    "text_only, or pass --pi05-sketch-mode overlay|language to "
                    "actually deliver the sketch."
                    % (", ".join(leaked),))
        self._connect()
        self.instruction = str(prompt.instruction)
        self._tokens = None

        if self.sketch_mode != "none":
            tok = getattr(prompt, "symbolic_tokens", None)
            if not tok:
                raise ValueError(
                    "sketch_mode=%r but the prompt carries no symbolic_tokens. "
                    "A silently text-only episode inside a sketch condition "
                    "would understate the very effect this run measures."
                    % (self.sketch_mode,))
            if self.sketch_mode == "overlay":
                self._tokens = tok          # composited per frame in _element
            else:
                from pi05_sketch import describe_tokens
                # The description is of the frame the MODEL sees, so it is
                # rotated iff the image is. Passing self.rotate180 through keeps
                # the two consistent even under --pi05-no-rotate180.
                self.instruction = describe_tokens(
                    self.instruction, tok, rotate180=self.rotate180)

        # Echo the string the server will actually receive, whenever it changes.
        # The explicit and ambiguous arms differ ONLY in this string, and both
        # write the same condition label, so a mislabelled arm is invisible in
        # every other column of results.csv. Logging on change keeps it to a
        # couple of lines for the ambiguous arm and one per caption for the
        # explicit one, rather than one per rollout.
        if self.instruction != self._logged_instruction:
            self._logged_instruction = self.instruction
            print(f"    prompt -> {self.instruction!r}", flush=True)

        scene_meta = getattr(prompt, "scene_meta", None) or {}
        suite = scene_meta.get("suite")
        self.episode_len = PI05_MAX_STEPS.get(suite, PI05_MAX_STEPS_DEFAULT)
        self._plan = []
        self.n_infer_calls = 0
        self._infer_ms = []
        self._dumped = False

    # ------------------------------------------------------------ obs -> dict --
    def _element(self, obs):
        img = np.asarray(obs[AGENTVIEW_KEY])
        wrist = np.asarray(obs[WRIST_KEY])
        if img.shape[0] != LIBERO_ENV_RESOLUTION:
            raise ValueError(
                "agentview is %dx%d; pi0.5-LIBERO expects the env opened at %d "
                "(openpi renders at 256 and pad-resizes to 224). Open the scene "
                "env with camera_heights/widths=%d for a pi05 run."
                % (img.shape[0], img.shape[1], LIBERO_ENV_RESOLUTION,
                   LIBERO_ENV_RESOLUTION))
        if self._tokens is not None:
            # BEFORE the rotation and before the resize: tokens are in
            # frame0's orientation at 128, the raw frame is that same
            # orientation at 256, so the only transform needed is the scale.
            # Rotating afterwards carries the marks along with the pixels they
            # annotate. Only the base camera is annotated -- the wrist view is a
            # moving close-up the sketch says nothing about.
            from pi05_sketch import draw_tokens
            img = draw_tokens(img, self._tokens)

        if self.rotate180:
            img, wrist = _rot180(img), _rot180(wrist)
        img = self._convert_to_uint8(
            self._resize_with_pad(img, self.resize_size, self.resize_size))
        wrist = self._convert_to_uint8(
            self._resize_with_pad(wrist, self.resize_size, self.resize_size))

        if self.dump_first_frame_to and not self._dumped:
            # One-off orientation check: this is the exact array the model sees.
            # Eyeball it against frame0.png before trusting any success rate.
            # A swallowed failure here is worse than useless: the dump exists to
            # CERTIFY the orientation, so a silently absent file would leave the
            # check looking done when it never ran.
            try:
                import cv2
                if not cv2.imwrite(self.dump_first_frame_to,
                                   cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
                    raise IOError("cv2.imwrite returned False")
            except Exception as e:
                print("[warn] could not dump model-input frame to %s: %s"
                      % (self.dump_first_frame_to, e))
            self._dumped = True

        state = np.concatenate((
            np.asarray(obs["robot0_eef_pos"], dtype=float),
            quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=float),
        ))
        if state.shape[0] != 8:
            raise ValueError("state is %d-D; pi0.5-LIBERO expects 8 "
                             "(eef_pos 3 + axis-angle 3 + gripper_qpos 2)"
                             % (state.shape[0],))
        return {
            "observation/image": img,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": self.instruction,
        }

    # ------------------------------------------------------------------- act --
    def act(self, obs, t):
        if t < self.num_steps_wait:
            a = np.zeros(ADIM)
            a[-1] = -1.0            # openpi's LIBERO_DUMMY_ACTION
            return a
        if not self._plan:
            t0 = time.time()
            chunk = self._client.infer(self._element(obs))["actions"]
            self._infer_ms.append((time.time() - t0) * 1000.0)
            self.n_infer_calls += 1
            chunk = np.asarray(chunk)
            if len(chunk) < self.replan_steps:
                raise ValueError(
                    "replan_steps=%d but the policy returned a chunk of %d"
                    % (self.replan_steps, len(chunk)))
            self._plan = [np.asarray(a, dtype=float) for a in chunk[:self.replan_steps]]
        a = self._plan.pop(0)
        if a.shape[0] != ADIM:
            raise ValueError("action is %d-D, expected %d" % (a.shape[0], ADIM))
        return a

    # ---------------------------------------------------------- diagnostics --
    @property
    def infer_ms_mean(self):
        return float(np.mean(self._infer_ms)) if self._infer_ms else None

    @property
    def infer_ms_max(self):
        return float(np.max(self._infer_ms)) if self._infer_ms else None
