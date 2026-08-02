# Prompt — human sketch collection UI + manual-vs-auto comparison harness


## Role and objective

You are working in the **Sketch-Prompted VLA** repository at
`C:\Users\Admin\sketch_prompted_vla`. Read `CLAUDE.md`, `README.md`,
`SCHEMA.md` and `SUITE_FACTS.md` before writing code; they hold conventions and
hard-won facts that are expensive to re-derive.

Build two things:

1. **A browser-based sketch collection tool** that shows a human annotator the
   raw `frame0.png` of a validation scene plus its vague instruction, and
   records the circle and arrow they draw.
2. **A comparison harness** that scores those human sketches against the
   existing auto-generated (OpenCV) sketches and answers the question the
   validation-set datasheets currently list as an open limitation: *how large is
   the synthetic-to-human gap, and is the auto-annotator's imprecision
   augmentation wide enough to cover real human strokes?*

## What already exists (do not rebuild it)

`outputs/` holds three built validation suites — `validation_set_spatial`,
`validation_set_object`, `validation_set_goal` — 38 scenes each, **114 total**,
all normalised to canonical schema v1.0. Each `scene_NNNN/` directory contains:

| file | contents |
|---|---|
| `frame0.png` | 128×128 RGB agentview, physics settled, **no sketch** |
| `sketch.png` | `frame0` + auto-drawn circle and arrow |
| `tokens.json` | `instruction`, `target`, `destination`, `destination_region`, `goal_predicate`, `suite`, `tier`, `symbolic_tokens` |
| `meta.json` | superset of `tokens.json` + `pick_px`, `place_px`, `radius`, `all_pixels`, `px_extent`, `camera_matrix`, gate measurements |
| `scene.bddl` | the LIBERO task |
| `target_vismask.png` | grayscale silhouette of the target |

Facts you will need, verified against the files — do not re-derive them:

- Images are **128×128**. `pick_px` and `place_px` are `[col, row]` integers in
  that space.
- `symbolic_tokens.circle` is `{cx, cy, rx, ry}`; `symbolic_tokens.arrow` is
  `{x0, y0, x1, y1}` (tail → head). All in 128-space pixels.
- In `sketch.png` as read in RGB, the auto circle is drawn `(0, 200, 0)` green
  and the auto arrow `(200, 50, 50)` red. Match these colours exactly when you
  render human strokes, so a human sketch is a drop-in replacement for an auto
  one.
- The auto-drawer's imprecision augmentation (in
  `scripts/build_validation_set_object_wsl.py`, `draw_circle` / `draw_arrow`)
  jitters the circle centre by integers in `[-3, 3]`, the radii by
  `uniform(-2, 3)`, both arrow endpoints by integers in `[-2, 2]`, and bends the
  arrow through a midpoint offset by integers in `[-7, 7]`. **These four ranges
  are the hypothesis the comparison harness tests.**
- `outputs/validation_manifest_all.json` has 114 rows. Rows MUST be keyed on the
  pair `(suite, dir)` — `dir` and `seed` both collide across suites.
- `meta.json['all_pixels']` maps every named instance in the scene to its pixel
  centre. This is what lets you decide whether a human circled the *right*
  object and whether their arrow points at the *right* destination.

## Deliverables

Write these files, and nothing else:

```
scripts/build_human_study_bundle.py   # samples scenes, emits the single-file HTML
scripts/score_human_sketches.py       # comparison harness
outputs/human_study/
    sketch_tool.html                  # generated; the artefact annotators open
    scene_subset.json                 # the sampled scenes, seeded and reproducible
    responses/                        # where returned annotator JSON files land
outputs/human_study/HUMAN_STUDY.md    # protocol + how to run both scripts
```

`score_human_sketches.py` writes its results into
`outputs/human_study/comparison/` (a JSON of all metrics, a per-scene CSV, and
matplotlib figures). Both scripts must be **pure stdlib + numpy + Pillow +
matplotlib** — they run on a Windows laptop outside WSL, so no `robosuite`,
`mujoco` or `libero` imports.

## 1. Scene sampling (`build_human_study_bundle.py`)

Draw a **stratified subset of 36 scenes** — 12 per suite, allocated across the
four tiers in proportion to each suite's `control 5 / referential 12 /
directional 9 / both 12` split, so roughly 2 control, 4 referential, 3
directional, 3 both per suite. Seed the sample and record the seed. Every
annotator sees the identical 36 scenes in an identical order, because
inter-annotator agreement is only computable on shared scenes.

Write the chosen `(suite, dir)` pairs to `scene_subset.json` with each scene's
`instruction`, `tier`, and the ground-truth fields the scorer needs. Do not put
the auto sketch geometry in the bundle the annotator opens — see below.

## 2. The sketch tool (`sketch_tool.html`)

**Single self-contained HTML file.** No build step, no server, no CDN imports.
`frame0.png` for each sampled scene is **base64-embedded** into the generated
HTML by the builder script — a `file://` page cannot `fetch()` sibling PNGs, and
requiring `python -m http.server` defeats the point of handing one file to a
collaborator. Thirty-six 128×128 PNGs base64-encode to roughly 1 MB, which is
fine.

**Critical experimental constraint: the annotator must never see the auto
sketch, `pick_px`, `place_px`, `target` or `destination`.** Embed only
`frame0.png`, the vague `instruction`, the suite and the scene id. If the
ground truth is in the HTML the annotator can read it, and the referential- and
directional-accuracy numbers below become worthless. The scorer joins back to
the truth from disk afterwards, on `(suite, dir)`.

Interaction:

- Display the 128×128 frame upscaled **4× to 512×512**, `image-rendering:
  pixelated`, so the annotator sees exactly the pixels the policy sees. Map
  every recorded coordinate back into 128-space as a float — do not round to the
  display grid.
- Two tools, **circle** and **arrow**, both freehand pointer strokes (mouse,
  trackpad and stylus/touch via Pointer Events). One of each per scene. Drawing
  a second circle replaces the first.
- Circle rendered green, arrow red, matching the auto colours.
- For the arrow, take the tail as the stroke's first point and the head as its
  last; render an arrowhead at the head so the annotator sees the direction they
  gave.
- Controls: undo, clear-current-stroke, clear-all, next, previous. Keyboard
  shortcuts (`c` circle, `a` arrow, `u` undo, `Enter` next). A progress
  indicator showing *n of 36*.
- A **skip** button that records the scene with `skipped: true` and a
  short reason the annotator picks (`cannot tell which object`, `cannot tell
  where`, `image unclear`, `other`). A skip is a finding, not a failure — if
  humans cannot resolve a scene from the frame alone, that is data about the
  scene.
- Persist progress to `localStorage` after every scene, keyed by annotator id,
  so a closed tab does not lose the session.
- An opening screen: a short plain-language explanation of the task ("circle the
  object you would move, draw an arrow to where you would move it"), one
  **practice scene** whose result is recorded but flagged `practice: true`, and
  a field for a free-form annotator id.
- A closing screen with a **Download JSON** button.

## 3. Annotation output schema

One JSON file per annotator, downloaded from the tool and dropped into
`outputs/human_study/responses/`. Shape:

```json
{
  "schema_version": "1.0",
  "annotator_id": "a01",
  "subset_seed": 20260802,
  "user_agent": "...",
  "started_utc": "...", "finished_utc": "...",
  "annotations": [
    {
      "suite": "goal", "dir": "scene_0000", "tier": "control",
      "practice": false, "skipped": false, "skip_reason": null,
      "circle": {
        "points": [[x, y], ...],
        "fit": {"cx": 0.0, "cy": 0.0, "rx": 0.0, "ry": 0.0, "theta_deg": 0.0}
      },
      "arrow": {
        "points": [[x, y], ...],
        "x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0
      },
      "effort": {
        "time_to_first_stroke_ms": 0,
        "time_total_ms": 0,
        "n_undo": 0,
        "n_redraw": 0,
        "n_points_circle": 0,
        "n_points_arrow": 0
      }
    }
  ]
}
```

All coordinates are floats in 128-space. Keep the raw `points` lists — they are
cheap and they permit richer stroke analysis later. Fit the circle ellipse in
the browser (least-squares over the stroke points, allowing rotation
`theta_deg`), so `fit` is directly comparable to the auto
`symbolic_tokens.circle`.

## 4. Comparison harness (`score_human_sketches.py`)

Loads every file in `responses/`, joins to disk truth on `(suite, dir)`, and
reports four families of metric. Report each broken down by **suite** and by
**tier**, not only pooled — the tiers are what make the benchmark diagnostic.

**(a) Human correctness — the number that matters most.** Whether a real human
can supply the disambiguating signal at all.

- *Referential accuracy*: the human circle's fitted ellipse contains
  `pick_px` and contains no other instance's pixel from `all_pixels`. Report the
  rate, plus a breakdown of failures into "contains the wrong object", "contains
  several", "contains none".
- *Directional accuracy*: the human arrow head's nearest candidate in
  `all_pixels` is the true `destination`. Report the rate.
- *Joint accuracy*: both correct.
- Skip rate, with reasons.

**(b) Human-vs-auto geometric deviation.** Per scene, against the auto
`symbolic_tokens`:

- circle centre offset in px, and normalised by the auto `rx`
- circle radius ratio `human_r / auto_r` (use the mean of `rx, ry`)
- circle ellipse IoU
- arrow tail offset, head offset (px)
- arrow angle difference (degrees, signed and absolute)
- arrow length ratio
- arrow path curvature: maximum deviation of the human stroke from the straight
  tail→head chord, in px — directly comparable to the auto drawer's `[-7, 7]`
  midpoint bend

**(c) The augmentation-calibration verdict.** For each of the four auto jitter
parameters, plot the human empirical distribution against the auto uniform range
and report what fraction of human sketches fall **outside** it. State plainly
whether the current augmentation is too tight, about right, or too loose, and
give a recommended range per parameter. This is the actionable output — it feeds
straight back into the builders' `draw_circle` / `draw_arrow`.

**(d) Inter-annotator agreement and effort.** Only computable with ≥2
annotators; degrade gracefully to a note when only one response file is present.

- pairwise circle IoU and arrow angle difference between annotators on the same
  scene
- agreement on *which instance* was circled (fraction of scene-pairs where both
  annotators selected the same object), and on which destination
- median time-to-draw per scene, stroke counts, undo counts — the low-effort
  claim in the proposal needs a measured number

Also render, for each annotated scene, a **`sketch_human.png`** at 128×128 in
the auto colour scheme into
`outputs/human_study/rendered/<annotator>/<suite>/<dir>.png`, plus a
side-by-side contact sheet of auto vs human for visual inspection. The rendered
human sketches must be format-identical to the auto ones so a later GPU rollout
can swap them in with no code change.

## 5. Protocol document

`outputs/human_study/HUMAN_STUDY.md` — the datasheet for this study, in the
style of the existing suite `DATASHEET.md` files. Cover: the sampling scheme and
seed, the annotator-facing instructions verbatim, what the annotator was and was
not shown and why, the output schema, how to run both scripts, and the metric
definitions. Keep it to what a reader needs to reproduce and interpret the
study — no padding.

## Repository conventions (from `CLAUDE.md`)

- The project is **Sketch-Prompted VLA**. **Never write "DrawVLA"** — it is a
  dead working title that leaked into some legacy script headers. Do not
  introduce it anywhere new.
- Author line on any document is **`Aaron`** — nothing else. No full legal name,
  no title, no affiliation.
- **First person singular throughout — I / my.** Aaron works alone; there is no
  team. Never write "we", "our" or "us" in documents, docstrings or commit
  messages. Impersonal and passive constructions are preferred for describing
  method ("A scene is kept only if…"); use "I" for claims and decisions.
- Today's date must be confirmed rather than assumed.
- Follow the existing scripts' style: a module docstring stating what the script
  does and how to run it, a clearly separated constants block, and stdout
  progress that is worth reading.

## How to work

Build in the order the repo already favours: **vertical slice → smoke → full
run.** Get the bundle builder emitting an HTML that works for 2–3 scenes and
confirm the round trip end-to-end — draw, export, score — before generating the
full 36-scene bundle. That ordering has caught a real bug at every stage of this
project.

Deliver what is specified, at the scope specified. Make routine judgment calls
yourself and keep going; check in only where two readings of this brief would
lead to materially different work. If something here is mistaken or a better
approach exists, say so in a sentence and continue with the task as written
rather than quietly reshaping it.

Say in one sentence what you are about to do before your first tool call, then
work. While working, give a brief update only when you find something important
or change direction. When you finish, lead with the outcome — what you built and
what the round-trip test showed — with detail after it.

Do not spawn subagents for this; it is a single coherent track of work you can
finish yourself. Match written-file length to what the task needs; do not pad
with redundant summaries or boilerplate sections.
