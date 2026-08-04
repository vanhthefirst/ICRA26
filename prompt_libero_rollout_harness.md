# Prompt — running human sketches inside LIBERO (rollout + scoring harness)

Paste everything below the line into a fresh session with the
`sketch_prompted_vla` folder attached. Run it in **WSL2, `conda activate
libero`** — this is the first work in this project that needs the simulator since
the suites were built.

---

## Role and objective

You are working in the **Sketch-Prompted VLA** repository at
`C:\Users\Admin\sketch_prompted_vla` (`/mnt/c/Users/Admin/sketch_prompted_vla`
from WSL). Read `CLAUDE.md`, `README.md`, `SCHEMA.md`, `SUITE_FACTS.md` and
`outputs/human_study/HUMAN_STUDY.md` before writing code. `SUITE_FACTS.md` in
particular holds LIBERO facts that were expensive to derive — do not re-derive
them.

The human-sketch **collection** platform is finished and works. What does not
exist is anything that **consumes** a sketch inside the LIBERO simulator. Build
that: a rollout harness that resets a LIBERO environment from a scene's BDDL,
hands a policy the observation plus a sketch, steps the simulator, and scores
success by the BDDL goal predicate.

**Collection stays offline. Do not build a live drawing loop.** The existing
`sketch_tool.html` renders pre-exported PNGs, the annotator draws in a browser on
Windows, and returns a JSON file. That round trip is complete and correct, and
coupling a browser canvas to a live MuJoCo process would add a server, a port
bridge and a WSLg display dependency to buy nothing — the scenes are static
initial states, so there is no live state for the annotator to react to. The
sketch is drawn once, offline, on `frame0.png`, and replayed into the simulator
afterwards. If you find yourself writing a Flask endpoint, stop.

**Section 3 of this brief lists eight known issues that will break this harness
if they are not addressed. Resolve them in the order given. They are not
hypothetical: several were found by inspecting the existing builders, and issue 1
is currently a hard blocker.**

## 1. What already exists (do not rebuild any of it)

**The validation suites** — `outputs/validation_set_{spatial,object,goal}/`, 38
scenes each, **114 total**, canonical schema v1.0. Each `scene_NNNN/` holds
`scene.bddl`, `frame0.png` (128×128 agentview, settled, no sketch), `sketch.png`
(auto circle + arrow), `tokens.json`, `meta.json`, `target_vismask.png`. Audited
clean, 114/114, by `scripts/audit_validation_sets.py`.

**The human-sketch study** — `scripts/build_human_study_bundle.py` emits a single
self-contained `outputs/human_study/sketch_tool.html` over a seeded, stratified
36-scene subset (seed `20260802`, 12 per suite, tiers 2/4/3/3). Annotators draw
offline and return JSON into `outputs/human_study/responses/`.
`scripts/score_human_sketches.py` scores human-vs-auto agreement **and — this is
the part you consume — exports the human sketches into scene folders that mirror
the validation suites**:

```
outputs/human_study/rendered/<annotator>/validation_set_<suite>/<dir>/
    sketch.png     128x128 RGB, auto colour scheme, format-identical to the auto sketch
    tokens.json    canonical shape, symbolic_tokens filled from the HUMAN geometry
outputs/human_study/rendered/consensus/validation_set_<suite>/<dir>/
    the medoid annotator's sketch, plus the group median geometry
```

That layout is deliberate: **a loader is pointed at a different root and runs
unchanged.** Your harness must exploit this — a sketch source is a root path plus
a label, not a special case in the code. `frame0.png`, `scene.bddl` and
`meta.json` are not duplicated into the export; join back to
`outputs/validation_set_<suite>/<dir>/` on `(suite, dir)`.

## 2. Facts already established — take them, do not re-derive them

- Images are **128×128**, camera `agentview`. `pick_px` / `place_px` are
  `[col, row]`. `symbolic_tokens.circle` is `{cx, cy, rx, ry}`,
  `symbolic_tokens.arrow` is `{x0, y0, x1, y1}` (tail → head).
- `obs["agentview_image"]` is already correctly oriented — **do not flip it**.
  `project_points_from_world_to_camera` returns `(row, col)`; use
  `col = int(p[1])`, `row = (IMG_H - 1) - int(p[0])`.
- `OffScreenRenderEnv` has **no** `.action_dim` — use `ADIM = 7` — and no
  `._get_observations()`.
- Success is `env.check_success()`, falling back to `env.env._check_success()`.
- Object position for projection is the `contype=0 / conaffinity=0` reference
  geom's `xpos` (the visual centre), **not** `body_xpos`. The exception is
  predicate evaluation: `In(a, b)` tests the object's **body origin** against the
  contain site.
- Free joints are `<instance>_joint0`; set pose via `model.jnt_qposadr` /
  `jnt_dofadr`. `env.sim.forward()` after writing `qpos`.
- Scripted-grasp gripper close sign is `-1.0` for these objects.
- The workspace projects at roughly **101 px/m**.
- **Writing across the WSL↔Windows DrvFs bridge can silently truncate.** Verify
  every write: bytes > 0, PNG `IEND` present, re-decodes.
- `outputs/validation_manifest_all.json` rows MUST be keyed on `(suite, dir)` —
  `dir` and `seed` both collide across suites.
- A working scripted top-down OSC grasp already exists in
  `build_validation_set_object_wsl.py` (`scripted_grasp`, used by the `graspable`
  gate) and a working teleport oracle beside it (`teleport`). Reuse both.

## 3. Known issues to resolve before the first full run

### Issue 1 — the annotated initial state was never persisted (blocker)

`scene.bddl` does not pin object poses. The builders sampled a position, then
wrote a **±1.2 cm box** around it into the BDDL (`HALF_BOX = 0.012`), and
LIBERO's placement sampler draws uniformly inside that box at every `reset()`,
with yaw randomisation on top. The builders never called `sim.get_state()`, so
**the exact state behind every `frame0.png` exists only as a PNG.**

At ~101 px/m the positional spread is only about ±1.2 px, which sounds harmless
until you remember the sketch is a *pixel-space* annotation: a circle with radius
13 px around a target that has moved 2 px, in a scene whose whole design is that
a same-category sibling sits just outside the circle, is a materially different
annotation. Yaw randomisation is the larger effect and changes the silhouette
outright. Worse, a failure caused by scene drift is indistinguishable from a
policy failure in the results table.

**Resolve it with this ladder, stopping at the first rung that meets tolerance.**
Tolerance: every instance's re-projected visual centre within **1.0 px** of the
`meta['all_pixels']` value recorded at build time. Report the measured rate at
each rung.

1. **Resample.** Reset up to *N* times (start at 50) and keep the draw with the
   lowest re-projection error. Cheap, and with a ±1.2 cm box it may well suffice
   on its own. Measure before assuming it does not.
2. **Solve for the pose directly.** `meta['all_pixels']` gives every instance's
   projected pixel centre and `meta['camera_matrix']` gives the world→pixel
   transform. An object resting on a known support plane has two unknowns, `x`
   and `y`, and two constraints, `col` and `row` — so **the world xy of every
   object is recoverable by deprojecting its recorded pixel centre onto the
   support plane**. Write those xy values into each free joint's `qpos`, keep `z`
   and the quaternion from the reset draw, `sim.forward()`, settle, re-verify.
   Yaw is not recoverable this way, but yaw does not move a projected centre, and
   the sketch is keyed to centres.
3. **Give up honestly.** A scene that still misses tolerance goes into
   `nonreproducible.json` with its residual, and the harness skips it with that
   reason recorded in `results.csv`. A scene whose initial state cannot be pinned
   cannot carry a pixel-space sketch, and shipping it silently would poison the
   headline number.

Then **persist** the result: `env.sim.get_state()` flattened into
`init_state.npz` (`qpos`, `qvel`, `time`) in each scene folder, restored at
rollout with `env.sim.set_state(...)` then `env.sim.forward()`. This is the same
mechanism LIBERO's own evaluation harness uses via its `init_files/*.pruned_init`
states, so it is the idiomatic route, not an invention. If the installed LIBERO
exposes `env.set_init_state()`, prefer it and say so.

Two hard constraints:

- **`frame0.png` is authoritative and must never be regenerated.** Every human
  sketch collected so far is keyed to those exact pixels. The pinning step
  changes the *simulator state* to match the PNG, never the PNG to match the
  simulator.
- **Run the capture over the 36 study-subset scenes first**, before any further
  annotator is sent the tool. If a subset scene turns out to be non-reproducible,
  it should be swapped out of the roster rather than annotated and then thrown
  away.

Update `SCHEMA.md` with the new `init_state.npz` entry, and extend
`audit_validation_sets.py` to check its presence and shape.

### Issue 2 — pixel → world deprojection is under-determined

A sketch gives image-plane coordinates. Grasping needs a 3D point. One pixel is a
ray, not a point, and the proposal names this explicitly: image-plane arrow
length is ambiguous about true 3D magnitude, which is the same ambiguity that
motivates the depth-augmented latent-action line of work.

Three ways to close it, and you should implement the first two:

1. **Support-plane intersection** (the honest default). Intersect the camera ray
   with the known table or floor height for that suite, plus a per-category
   resting-height offset. Needs no privileged information a real deployment would
   lack.
2. **Rendered depth** (the oracle affordance). LIBERO can render depth; construct
   the env with depth enabled and read the true `z` at the pixel. Exact, but a
   real policy conditioned on RGB does not have it — so any result using it must
   be labelled an upper bound, in `ROLLOUT.md` and in `run_config.json`.
3. **Nearest-instance snap** — deproject, then snap to the nearest known object
   centre. **Do not use this for the sketch path.** It reads `all_pixels`, which
   is ground truth, so it silently converts "follow the sketch" into "look up the
   answer". It is legitimate only inside the failure-attribution logic, after the
   rollout, where reading ground truth is the point.

Put the projection and deprojection in **one shared module** and give it a
round-trip test: deproject every entry in `meta['all_pixels']` to world, reproject
it, and assert it returns the original pixel. That single test also validates the
pinning arithmetic in issue 1, since both use the same routine.

### Issue 3 — nothing yet defines how the sketch reaches the policy over time

The sketch is drawn once on `frame0.png`. At step 200 the gripper has moved and
the scene has changed. Three questions have no answer in the repo, and the
harness cannot be written without settling them. Settle them this way, and state
the decision in `ROLLOUT.md`:

- **The sketch is a static prompt, not an observation.** It is drawn at *t = 0*
  and held constant for the whole episode. It is an instruction — the visual half
  of "put the bowl on the plate" — and instructions do not update mid-episode.
  (The alternative, re-rendering the overlay onto every live frame, is
  PIVOT-style interactive redrawing; it is explicitly out of scope here.)
- **The overlay goes on a separate prompt image, not burned into the live
  observation.** Burning green and red strokes into a 128×128 observation
  occludes the very object being grasped, and it corrupts the live stream a
  learned policy is trained on. Proposal §2.5 already anticipates keeping the
  annotation on a separate prompt-image channel — do that. The policy receives
  `obs` (live, clean) and `prompt.sketch_rgb` (static, annotated).
- **Both representation routes are required, not optional.** Proposal §2.5
  ablates the rendered overlay against the symbolic tokens, and the suites ship
  both for exactly that reason. A `--sketch-route {overlay,tokens}` flag selects
  which the prompt carries.

### Issue 4 — coordinate conventions are a repeated foot-gun

`SUITE_FACTS.md` §1.1 records that flipping the frame double-mirrors everything,
and that projection returns `(row, col)` while the sketch geometry is
`(col, row)`. This has already cost time once. The mitigation is structural, not
a comment: **one projection module, one convention, asserted**. Every function
signature in it names the convention (`xy_px` vs `rowcol`), and the round-trip
test from issue 2 fails loudly on a transposition. Do not hand-roll the mapping a
second time inside the harness.

### Issue 5 — `In` is permissive, so success can be spurious

`SUITE_FACTS.md` §3.2: for a site region `check_contact` is hardcoded `True`, so
`In(a, b)` reduces to a box test on the object's body origin, and the box reaches
the floor — an object merely *standing at* a basket's xy counts as contained. A
policy that shoves the target in roughly the right direction can score a success
it did not earn, and the scripted oracle can too.

Two mitigations, both cheap:

- **Score the negative oracles at rollout time as well.** Every scene was
  certified at build time such that the wrong object or the wrong destination
  scores False. Re-running that check against the *terminal* state turns a bare
  boolean into a statement about which goal was actually satisfied.
- **Require the sustained window.** Success must hold for a run of consecutive
  steps (LIBERO's standard criterion), not merely be true on one frame. Define
  the window as a named constant, state its value in `ROLLOUT.md`, and record
  both the first step at which success was reached and whether it held.

### Issue 6 — the text-only baseline can leak ground truth

The forced-guess baseline is the denominator for every headline number in this
project. If it can see `target`, `destination`, `pick_px`, `place_px` or the
sketch, the measured gap collapses and the collapse is invisible in the output.

Enforce it **structurally, not by discipline**: the baseline is constructed with a
restricted prompt object that does not carry those fields at all, so a leak is an
`AttributeError` at the first rollout rather than a quietly optimistic number.
Mirror what `build_human_study_bundle.py` already does — it audits its browser
payload against an allow-list and aborts the build on a violation. Same idea, same
rigour.

### Issue 7 — human sketches are incomplete, and sometimes wrong

Two cases the harness will hit on the first real run and must not treat as
crashes:

- **Skipped scenes.** An annotator may press Skip, leaving no geometry. Those
  `(annotator, scene)` cells are absent, not zero. Exclude them from that
  annotator's rate and report the exclusion count; never impute.
- **Sketches that circle the wrong object.** Rolling these out is not a bug, it
  is one of the more interesting measurements available — but only if the harness
  reports **two** numbers rather than one: *task success* (did the BDDL goal
  hold) and *sketch-following fidelity* (did the policy act on the instance the
  sketch actually indicated, whether or not that was the intended one). A policy
  that faithfully executes a wrong sketch is a good policy given a bad prompt,
  and collapsing that into a single failure rate hides it.

`score_human_sketches.py` already computes referential and directional
correctness per scene; join to it rather than recomputing.

### Issue 8 — the run matrix is large, and env construction is slow and leaky

114 scenes × conditions × rollouts, each building an `OffScreenRenderEnv`. The
builders already carry the scar: both call `env.close()` and then `gc.collect()`
per scene, because MuJoCo rendering contexts leak.

- Build the env **once per scene** and reuse it across that scene's conditions
  and rollouts, restoring `init_state.npz` between them. Do not rebuild per
  rollout.
- `env.close()` + `gc.collect()` when moving to the next scene, exactly as the
  builders do.
- Make the run **resumable**: append to `results.csv` as you go and skip
  `(scene, condition, rollout_idx)` triples already present. A run that dies at
  scene 90 should not restart at scene 0.
- Order the matrix scene-major, so a partial run still yields complete rows for
  the scenes it reached rather than a partial column for every scene.

## 4. Deliverables

Write these, and nothing else:

```
scripts/sketch_geometry.py                 # shared projection / deprojection, no libero import
scripts/capture_scene_init_states_wsl.py   # issue 1: pin and persist the annotated state
scripts/sketch_policies.py                 # policy protocol + baselines, no libero import
scripts/rollout_sketch_wsl.py              # the harness
outputs/rollouts/ROLLOUT.md                # protocol document
outputs/rollouts/<run_id>/                 # results, written by the harness
```

`sketch_geometry.py` and `sketch_policies.py` must import only numpy — they are
the seam a GPU machine will later plug UniVLA into, and they should be readable
and testable without a simulator. The two `*_wsl.py` scripts may import
`robosuite` / `mujoco` / `libero`, and by the repo's naming convention must carry
the `_wsl` suffix.

### `sketch_policies.py`

```python
class SketchPolicy(Protocol):
    def reset(self, prompt: Prompt) -> None: ...
    def act(self, obs: dict, t: int) -> np.ndarray: ...   # shape (7,)
```

`Prompt` carries `instruction`, `sketch_rgb` (128×128×3 `uint8` or `None`),
`symbolic_tokens` (`{circle, arrow}` or `None`) and `scene_meta` (suite, dir,
tier). Per issue 6, the text-only baseline gets a restricted variant that lacks
the sketch and truth fields entirely.

Three implementations:

- **`ScriptedSketchOracle`** — deprojects the circle centre and the arrow head via
  `sketch_geometry`, then runs the existing `scripted_grasp` motion, lifts,
  transports, releases. This is the upper bound: how much of the task is solvable
  given a perfect *reading* of the sketch. It is what proves the loop end to end
  without a GPU.
- **`TextOnlyGuessPolicy`** — parses the object category from the instruction,
  picks **uniformly at random** among instances of that category and among
  candidate destinations, then executes the same motion. On a 4-bowl referential
  scene its expected success is ~25%, which is exactly the floor the meeting
  script quotes and the denominator the sketch gap is measured against.
- **`NoOpPolicy`** — zero actions. A harness sanity check: every scene must score
  False, because every scene passed the builders' *not pre-solved* gate. If one
  scores True, the harness is wrong, not the scene.

### `rollout_sketch_wsl.py`

Conditions are `(label, sketch_root_or_None)` pairs, so adding a source is a CLI
argument rather than a code change:

| condition | sketch source |
|---|---|
| `text_only` | none — prompt carries `instruction` only |
| `auto` | `outputs/validation_set_<suite>/<dir>/` |
| `human:<annotator>` | `outputs/human_study/rendered/<annotator>/validation_set_<suite>/<dir>/` |
| `human_consensus` | `outputs/human_study/rendered/consensus/validation_set_<suite>/<dir>/` |

Scope a run to the intersection of the requested conditions — human sketches
exist for the 36-scene subset only, so a run including a `human:*` condition
scores those 36 across every condition, while `text_only` + `auto` alone can
cover all 114. Report which scene set was used.

**Failure attribution.** Record per rollout whether the object was grasped at
all, whether it was lifted, whether the *correct* instance was grasped, whether
it was released nearest the *correct* destination, and the terminal distance to
the destination region. Six Goal scenes carry `grasp['grasp_success'] = False`
from the build-time probe, so stratify on it and report rates both ways.

**Outputs**, into `outputs/rollouts/<run_id>/`:

- `results.csv` — one row per `(suite, dir, condition, rollout_idx)`, appended as
  the run proceeds
- `summary.json` — success rate by condition, and by condition × suite × tier
- the headline numbers, printed and stored: **`auto − text_only`** (the measured
  value of sketch prompting), **`human − text_only`** (the value of a real
  sketch), **`human − auto`** (the sim-to-human transfer gap the suite datasheets
  list as an open limitation)
- `run_config.json` — every flag, the git SHA, the policy class, the sketch
  route, whether the depth affordance was used, seeds
- optional `--video` writing an MP4 per rollout for eye-checking failures

Flags: `--conditions`, `--scenes`, `--policy`, `--sketch-route
{overlay,tokens}`, `--n-rollouts`, `--max-steps`, `--depth-deproject`, `--video`,
`--resume`, `--smoke`.

### `outputs/rollouts/ROLLOUT.md`

The protocol document, in the style of the suite `DATASHEET.md` files and
`HUMAN_STUDY.md`. It must record, for each of the eight issues above, **what was
actually measured and which resolution was taken** — the initial-state
reproduction rate and which rung of the ladder it needed, which deprojection
method was used and whether it is an upper bound, the static-prompt decision, the
sustained-success window, how the baseline is firewalled, how skipped and
incorrect human sketches were handled, and the run's wall-clock cost. Then the
condition matrix, the failure decomposition, how to run every stage, and the
limitations. Keep it to what a reader needs to reproduce and interpret the run —
no padding.

## 5. Repository conventions (from `CLAUDE.md`)

- The project is **Sketch-Prompted VLA**. **Never write "DrawVLA"** — a dead
  working title that leaked into some legacy script headers. Do not introduce it
  anywhere new.
- Author line on any document is **`Aaron`** — nothing else. No full legal name,
  no title, no affiliation.
- **First person singular — I / my.** Aaron works alone; there is no team. Never
  write "we", "our" or "us" in documents, docstrings or commit messages.
  Impersonal and passive constructions are preferred for method ("A rollout is
  scored only if…"); use "I" for claims and decisions.
- Today's date must be confirmed rather than assumed.
- Match the existing scripts' style: a module docstring stating what the script
  does, what it assumes, and the exact commands to run it; a clearly separated
  constants block with the reasoning for each non-obvious value in a comment; and
  stdout progress worth reading.

## 6. How to work

Follow the repo's standing build order: **vertical slice → smoke (3–4 scenes) →
full run.** It has caught a real bug at every stage of this project. Concretely:

1. `sketch_geometry.py` with its round-trip test passing, before anything imports
   it.
2. Pin the initial states for three scenes and confirm the re-render matches
   `frame0.png`, before capturing all 114 — and capture the 36 study-subset
   scenes before the rest.
3. One scene rolling out end to end under `NoOpPolicy`, scoring False, before
   wiring the oracle.
4. The oracle on one scene per suite, before the full matrix.

Issue 1 is a blocker and its measured reproduction rate determines whether the
rest of the plan holds. Surface that number as soon as you have it rather than at
the end — if it is low, the harness design needs revisiting before more code is
written.

Everything specified here runs on the laptop; the GPU-dependent part of this
project is the learned policy, not the harness, and the scripted oracle exists so
the loop can be proven now. If any of it turns out to need a GPU, say so rather
than stubbing it out silently.

Deliver what is specified, at the scope specified. Make routine judgment calls
yourself and keep going; check in only where two readings of this brief would
lead to materially different work. If something here is mistaken or a better
approach exists, say so in a sentence and continue with the task as written
rather than quietly reshaping it.

Say in one sentence what you are about to do before your first tool call, then
work. While working, give a brief update only when you find something important
or change direction. When you finish, lead with the outcome: the reproduction
rate, whether the loop runs end to end, and what the oracle scored.

Do not spawn subagents; this is a single coherent track of work. Match
written-file length to what the task needs; do not pad with redundant summaries
or boilerplate sections.
