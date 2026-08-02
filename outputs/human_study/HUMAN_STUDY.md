# Sketch-Prompted VLA — human sketch collection study

Aaron — 2 August 2026

Datasheet for the human-sketch agreement study. Companion to the three suite
`DATASHEET.md` files; canonical schema v1.0 (see `SCHEMA.md`).

Built by `scripts/build_human_study_bundle.py`, scored by
`scripts/score_human_sketches.py`. Both are pure stdlib + numpy + Pillow +
matplotlib and run on Windows outside WSL — no `robosuite`, `mujoco` or `libero`.

## Purpose

Every suite datasheet lists the same open item: the validation sketches are
auto-drawn, and nothing yet establishes that they resemble what a person would
draw. This study closes that gap. It answers two questions:

1. **Can a human supply the disambiguating signal at all?** The scenes are
   deliberately impossible to resolve from the caption. If a person cannot pick
   the right object and the right destination from a 128×128 frame either, the
   benchmark is measuring something other than what it claims to.
2. **Is the auto-annotator's imprecision augmentation wide enough to cover real
   human strokes?** The auto drawer jitters the circle centre, the radii, the
   arrow endpoints and the arrow bend by four fixed ranges. Those four ranges are
   an untested guess. This study measures the human distributions against them
   and recommends replacements.

The second question is the actionable one — its output feeds straight back into
`draw_circle` / `draw_arrow` in the three builders.

## The four ranges under test

From `draw_circle` / `draw_arrow`. All three builders
(`build_validation_set_{spatial,object,goal}_wsl.py`) carry **byte-identical**
jitter code, verified 2026-08-02, so one hypothesis covers all three suites.

| parameter | auto code | range | anchor it jitters away from |
|---|---|---|---|
| circle centre | `rng.integers(-3, 4)` | integers [−3, 3] per axis | `pick_px` |
| circle radii | `rng.uniform(-2, 3)` | continuous [−2, 3) per axis | `radius` |
| arrow endpoints | `rng.integers(-2, 3)` | integers [−2, 2] per axis | `pick_px`, `place_px` |
| arrow bend | `rng.integers(-7, 8)` | integers [−7, 7] per axis on the midpoint | the tail→head chord |

## Sampling

**36 scenes, seed `20260802`**, drawn from the 114 in
`outputs/validation_manifest_all.json`. Twelve per suite, allocated across tiers
**2 control / 4 referential / 3 directional / 3 both**.

Exact proportional allocation of 12 against each suite's
`control 5 / referential 12 / directional 9 / both 12` split is
1.58 / 3.79 / 2.84 / 3.79, and largest-remainder rounding would give 1 control.
Control is bumped to 2 at the expense of `both`: control scenes are the only ones
where a wrong circle is unambiguously annotator error rather than genuine scene
ambiguity, so they are the calibration baseline every other tier is read against,
and one per suite is too thin to read.

Sampling runs on a sorted candidate list, so the roster depends only on the seed.
The 36 are then shuffled with the same seed so the suites interleave rather than
arriving in blocks. **Every annotator sees the identical 36 scenes in the
identical order** — inter-annotator agreement is only computable on shared
scenes. Rebuilding with the same seed reproduces the roster and its order
exactly; this was checked rather than assumed.

The roster, with each scene's instruction, tier and ground truth, is
`scene_subset.json`.

### Practice scene

One extra scene precedes the 36: **`spatial / scene_0002`**, control tier, drawn
from outside the subset. It is recorded and flagged `practice: true`, and the
scorer discards it. Taking it from outside the subset keeps every scored scene
free of a warm-up annotation; taking it from the control tier teaches the
controls without also teaching the annotator that ambiguity is expected.

## What the annotator is shown, and what is withheld

**Shown:** the raw `frame0.png`, the vague instruction, the suite name and the
scene id.

**Withheld:** the auto `sketch.png`, `pick_px`, `place_px`, `target`,
`destination`, `all_pixels` — and also the **tier**. Tier is withheld even though
it appears in the output schema, because "control" tells the annotator the scene
is unambiguous and "both" tells them to expect two ambiguities, which is exactly
the judgement being measured. The tool therefore writes `"tier": null` and the
scorer fills it in from disk on `(suite, dir)`.

This is enforced, not merely intended. `build_human_study_bundle.py` audits the
browser payload before writing it: every key must be on an allow-list, and the
`target`, `destination` and `destination_region` strings for all 37 scenes are
searched for in the serialised bundle. Either check failing aborts the build.

`frame0.png` is base64-embedded rather than referenced, because a `file://` page
cannot `fetch()` sibling PNGs and requiring `python -m http.server` defeats the
point of handing one file to a collaborator. Thirty-seven frames cost 1.20 MB.

## Annotator-facing instructions (verbatim)

> You are looking at what a robot's camera sees: a small 128×128 photo of a table
> or floor with a few objects on it, shown here blown up four times so the
> individual pixels are visible. It will look blocky. That is the real resolution
> the robot works from, so it is the resolution you should judge from too.
>
> Underneath each picture is a short instruction, such as "put the bowl on the
> plate". The instruction is deliberately vague. Often there will be several
> bowls, or several plates, and the words alone will not tell you which one is
> meant. That is the point of the study.
>
> **Your job is to answer the instruction with two marks:**
>
> 1. **Draw a circle** around the one object you would pick up.
> 2. **Draw an arrow** from that object to the place you would put it. Start the
>    arrow at the object and finish it where the object should end up — the
>    arrowhead appears wherever you lift your finger or mouse.
>
> There is no correct answer to look up and nothing is being tested about you. If
> the instruction is ambiguous, pick whichever object or destination seems most
> natural to you and move on. Draw the way you would draw on a whiteboard:
> quickly and roughly. A wobbly circle is a good circle. Do not zoom in, count
> pixels or deliberate — how fast and how loosely people do this is one of the
> things being measured.
>
> If you truly cannot tell what is being asked, press **Skip** and say why. A
> skip is a useful result, not a failure; it tells me the picture alone is not
> enough for that scene.
>
> There are 36 scenes, preceded by one practice scene to get the feel of the
> controls. Your progress is saved in this browser after every scene, so you can
> close the tab and come back. At the end you will get a **Download JSON**
> button — send me that file.
>
> Keyboard: `c` circle tool, `a` arrow tool, `u` undo, `Enter` next scene.

The wording is deliberately non-technical and deliberately discourages care:
the proposal's low-effort claim is only testable if annotators are not asked to
be precise.

## The tool

`sketch_tool.html` — one self-contained file, no build step, no server, no CDN.

- The frame is displayed at **4× (512×512), `image-rendering: pixelated`**, so
  the annotator sees exactly the pixels the policy sees. Recorded coordinates are
  mapped back to 128-space as **floats**; rounding to the display grid would
  quantise every stroke to 0.25 px and make human strokes look crisper than they
  are.
- Two freehand tools, circle and arrow, both via **Pointer Events** — mouse,
  trackpad and stylus/touch all work. One stroke of each per scene; a second
  stroke of the same tool replaces the first and increments `n_redraw`.
- Circle drawn `rgb(0,200,0)`, arrow `rgb(200,50,50)` — the auto colours.
- The arrow tail is the stroke's first point, the head its last, and an arrowhead
  is rendered at the head so the annotator sees the direction they gave.
- Controls: undo, clear-current-stroke, clear-all, previous, next, skip.
  Shortcuts `c` / `a` / `u` / `Enter`. Progress reads *n of 36*.
- **Skip** records `skipped: true` with one of `cannot tell which object`,
  `cannot tell where`, `image unclear`, `other`.
- Progress is written to `localStorage` after every scene, keyed by annotator id
  and subset seed, so a closed tab does not lose the session.
- A closing screen offers **Download JSON**.

### Ellipse fit

The circle stroke's ellipse is fitted **in the browser**, so `fit` is directly
comparable to the auto `symbolic_tokens.circle`. The conic
`Ax² + Bxy + Cy² + Dx + Ey + F = 0` is fitted by least squares over the stroke
points with `F` fixed to −1, after the points are centred and scaled to unit RMS
radius; without that normalisation the `F = −1` constraint is badly conditioned
for strokes far from the origin. Semi-axes and rotation come from the eigen-
decomposition of `[[A, B/2], [B/2, C]]`. A polygon-moment fit (Green's theorem,
parameterisation-independent) is the fallback when the solve is singular, the
discriminant is non-elliptic, or the conic runs away — which happens on
near-straight or self-crossing strokes.

`rx` is reported as the semi-axis nearer the **horizontal**, so `theta_deg` lands
in [−45, 45] and `rx` / `ry` line up with the axis-aligned auto pair. Auto
circles are near-round (`radius ± 2..3 px`), so this never has to arbitrate
between a long and a short axis.

Verified against synthetic ellipses with known parameters, including rotated and
noisy ones and a 300° open arc — the realistic hand-drawn case, since people
rarely close a loop. Recovery is exact on noiseless input.

## Delivery and collection

`sketch_tool.html` has no build step, no server dependency and no CDN imports, so
the study is run by handing over one file. An annotator needs no clone, no editor
and no Python.

**Send** the 1.21 MB HTML, zipped — some mail clients flag a bare `.html`
attachment, and anyone sharing it through Drive must *download* rather than
*preview* it, since Drive will not run HTML in the browser.

**Collect** by hand: at the end the annotator presses Download JSON and sends back
`human_sketches_<name>.json`. Drop the files into
`outputs/human_study/responses/` and run the scorer. Filenames do not matter — it
reads every `.json` in that directory and keys on the `annotator_id` inside.

Everyone must receive the **same build**. The roster is fixed by the seed, and
inter-annotator agreement is only computable on shared scenes.

There is no hosted deployment and no submission endpoint. The study is small
enough that collecting a handful of files by hand is less work than standing one
up, and keeping the tool offline-only means no annotation data leaves a
participant's machine until they choose to send it.

## Who annotated, and filtering bad work

The tool asks for a **name**, not a code. It is normalised to plain Latin rather
than rejected — NFD decomposition strips combining accents, and Vietnamese
`Đ`/`đ`, `Æ`/`æ`, `Ø`/`ø` and apostrophes are mapped explicitly — so
`Đỗ Việt Anh` is accepted and stored as `Do Viet Anh`. The response carries both
`annotator_name` (as normalised) and `annotator_id`, a lowercase underscored slug
used for the filename, the `localStorage` key and the export directory. Digits,
punctuation and non-Latin scripts are refused with a plain-language message.

This identifies submitters; it does not authenticate them. Anyone who receives the
file can submit under any name. That is a deliberate choice for a study of this size,
and it means **quality has to be filtered after the fact rather than gated at the
door**. Three things in the design exist for that:

- **The 6 control-tier scenes are attention checks.** A control scene has exactly
  one candidate object and one destination, so a wrong circle there is inattention
  rather than genuine ambiguity. `metrics.json` reports referential accuracy by
  tier and by annotator; an annotator well below ceiling on control is grounds to
  discard their whole file, not just those scenes.
- **Per-scene timings.** `time_to_first_stroke_ms` and `time_total_ms` are
  recorded for every scene. A run completed implausibly fast is visible in
  `fig_effort.png` before it reaches the headline numbers.
- **The practice scene**, which is recorded and flagged, so a run that is erratic
  only at the start can be told apart from one that is erratic throughout.

## Output schema

One JSON per annotator, downloaded from the tool, dropped into
`outputs/human_study/responses/`. All coordinates are floats in 128-space.

```json
{
  "schema_version": "1.0",
  "annotator_id": "alex_smith",
  "annotator_name": "Alex Smith",
  "subset_seed": 20260802,
  "bundle_generated_utc": "...",
  "user_agent": "...",
  "started_utc": "...", "finished_utc": "...",
  "annotations": [
    {
      "suite": "goal", "dir": "scene_0010", "tier": null,
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
        "time_to_first_stroke_ms": 0, "time_total_ms": 0,
        "n_undo": 0, "n_redraw": 0,
        "n_points_circle": 0, "n_points_arrow": 0
      }
    }
  ]
}
```

`tier` is always `null` from the tool — see *What the annotator is shown*. Raw
`points` lists are kept; they are cheap and they permit richer stroke analysis
later. On a skipped scene `circle` and `arrow` are `null`.

## Metrics

All four families are reported **pooled, by suite and by tier**. The tiers are
what make the benchmark diagnostic, so a pooled number alone is not enough.

### (a) Human correctness

The number that matters most: whether a real person can supply the signal.

- **Referential accuracy** — the fitted ellipse contains `pick_px` and contains
  no other instance centre from `all_pixels`. Failures split into *contains the
  wrong object* (exactly one instance, not the target), *contains several*, and
  *contains none*. A looser companion rate, `contains_target`, is reported
  alongside; it ignores whether other instances were also enclosed.
- **Directional accuracy** — the candidate in `all_pixels` nearest the arrow head
  is the true `destination`.
- **Joint accuracy** — both.
- **Skip rate**, broken down by reason.

### (b) Human-vs-auto geometric deviation

Per scene, against the auto `symbolic_tokens`: circle centre offset (px, and
normalised by the auto `rx`), radius ratio `human_r / auto_r` on the mean of
`rx, ry`, ellipse IoU (rasterised at 4× supersampling, rotation-aware), arrow
tail and head offsets, arrow angle difference (signed and absolute), arrow length
ratio, and arrow path curvature — the maximum perpendicular deviation of the
human stroke from its own tail→head chord.

### (c) Augmentation-calibration verdict

For each of the four ranges, the human empirical distribution is plotted against
the auto uniform range and the fraction of human strokes falling **outside** it
is reported, with a verdict of *too tight* (>25% outside), *too loose* (human
5–95 interval narrower than half the auto range) or *about right*, and a
recommended replacement range taken from the human 5th–95th percentiles, printed
as the literal `rng.integers(...)` / `rng.uniform(...)` call to paste back into
the builders.

Two things about this family are worth stating.

**Only correct strokes feed it.** A circle drawn around the wrong object is not
an imprecise circle, it is a different intent; letting it in would recommend
widening the augmentation to cover annotator error, the opposite of what the
augmentation is for. Circle parameters therefore use referentially correct scenes
and arrow parameters directionally correct ones. Excluded counts are reported.

**The bend range is reconstructed, not read.** `symbolic_tokens.arrow` records
only the endpoints, not the midpoint the auto drawer bent the path through, so
the auto chord-deviation distribution cannot be read off disk. It is instead
Monte-Carlo simulated (4000 draws per scene, seed `20260802`) from the drawer's
own [−7, 7] midpoint law on each annotated scene's actual chord. Because the auto
path is two straight segments, its maximum deviation is exactly at the midpoint,
so the simulated quantity is the same quantity measured on the human stroke. The
recommended half-width is the `k` whose simulated 95th percentile best matches
the human 95th percentile on those same chords.

### (d) Inter-annotator agreement and effort

Computable only with ≥2 response files; with one, the scorer prints a note and
every other family is unaffected.

Pairwise on shared, non-skipped scenes: circle IoU, absolute arrow angle
difference, agreement on **which instance** was circled (nearest `all_pixels`
candidate to the fitted centre — more robust than containment, which can be
empty), and agreement on which destination. Effort: median time per scene, median
time to first stroke, stroke point counts, undo and redraw counts — the
proposal's low-effort claim needs a measured number.

## Region destinations — a substitution the scorer makes

`meta['all_pixels']` enumerates **movable instances only**. Eleven of the 114
scenes, all Goal suite and all control or referential tier, name a *fixture or
region* as the destination — `flat_stove_1`, `wine_rack_1`, `wooden_cabinet_1`,
`main_table` — which is therefore absent from `all_pixels`. On those scenes the
nearest-candidate directional test is undefined and would score 0% for reasons
unrelated to the annotator.

The scorer adds the destination to the candidate set at `place_px`, the projected
region point the auto arrow already aims at, and flags the scene
`dest_from_place_px` in `per_scene.csv`. **Four of the 36 sampled scenes** are
affected: `goal/scene_0001`, `goal/scene_0002`, `goal/scene_0010`,
`goal/scene_0013`. The anchor is a region point rather than an object centroid,
so those four directional results are a slightly different kind of measurement;
the flag is there so they can be excluded if that matters.

## How to run

```powershell
cd C:\Users\Admin\sketch_prompted_vla

# 1. build the bundle (already done; rerun only to change the seed or the quota)
python scripts\build_human_study_bundle.py
python scripts\build_human_study_bundle.py --smoke    # 3-scene vertical slice

# 2. send outputs\human_study\sketch_tool.html to each annotator, alone.
#    Drop the returned JSON into outputs\human_study\responses\

# 3. score
python scripts\score_human_sketches.py
python scripts\score_human_sketches.py --smoke
```

The scorer writes into `outputs/human_study/comparison/`:

| file | contents |
|---|---|
| `metrics.json` | every number the script computes |
| `per_scene.csv` | one row per `(annotator, suite, dir)` |
| `fig_accuracy.png` | (a) by suite and tier |
| `fig_calibration.png` | (c) human distribution against each auto range |
| `fig_geometry.png` | (b) deviation distributions |
| `fig_effort.png` | (d) time and stroke counts |
| `contact_sheet_<id>.png` | auto sketch beside human sketch, per scene |

The scorer cross-checks each response against `scene_subset.json` — seed
mismatch, off-roster scenes, missing roster scenes and any roster/disk truth
disagreement are all reported.

## Where the human sketches are stored, and how to evaluate against them

Three tiers, in decreasing order of how much you should care about losing them.

**1. Raw strokes — the source of truth.**
`outputs/human_study/responses/human_sketches_<annotator>.json`, one file per
annotator, dropped in by hand from the tool's Download button. Every stroke's
full point list in 128-space floats, the fitted ellipse, the arrow endpoints,
skip flags and timings. Nothing is derived, so this is the copy to keep. Both
tiers below are regenerated from it on every scorer run.

**2. Scene folders that mirror the validation suites — the model-facing export.**

```
outputs/human_study/rendered/
    <annotator>/validation_set_<suite>/<dir>/
        sketch.png     128x128 RGB, auto colour scheme
        tokens.json    canonical shape, symbolic_tokens = the HUMAN geometry
    consensus/validation_set_<suite>/<dir>/
        sketch.png     the medoid annotator's stroke for that scene
        tokens.json    same, plus the group median geometry and the vote record
```

The layout deliberately matches `outputs/validation_set_<suite>/<dir>/`, so an
evaluation harness is pointed at a different root and runs unchanged. `sketch.png`
is byte-format identical to the auto one — same size, mode and two colours. It is
drawn at 4× and box-downsampled: a 4× box filter over a nearest-upscaled frame
reproduces the original pixels exactly wherever no stroke was laid down, so the
frame is untouched while the strokes pick up the same kind of antialiasing
`cv2.LINE_AA` gives the auto sketches.

`tokens.json` carries the full canonical contract — `instruction`, `target`,
`destination`, `destination_region`, `goal_predicate`, `suite`, `tier`,
`symbolic_tokens`, `radius`, `pick_px`, `place_px`, `seed` — with
`symbolic_tokens` filled from the human fit rather than the auto drawer, plus
`sketch_source` and `annotator_id` for provenance. The circle carries an extra
`theta_deg`; the auto ellipse is axis-aligned and has no rotation to record, so
this is additive in the same way the legacy suite keys are.

**`frame0.png`, `scene.bddl`, `target_vismask.png` and `meta.json` are not
copied.** They are identical to the originals, and duplicating 36 BDDLs per
annotator buys nothing. Join back to `outputs/validation_set_<suite>/<dir>/` on
`(suite, dir)` for those — the same key everything else in this repo uses.

A text-only vs auto-sketch vs human-sketch comparison is then three runs of the
same harness over three roots, with the BDDL and the frame shared.

**3. Metrics** — `comparison/metrics.json`, `per_scene.csv`, the four figures and
the contact sheets. Fully derived; delete freely.

### The consensus scene set

With several annotators there are several sketches per scene, so
`rendered/consensus/` holds one representative per scene for a single headline
run.

The consensus is the **medoid**, not a synthesised average: the real annotator
whose geometry sits closest to the group median. An averaged ellipse and a
straight averaged arrow would be a third stroke distribution, neither auto nor
human, which is the one thing this export must not ship. Taking a real stroke
keeps `sketch.png` and `tokens.json` consistent with each other and with what a
person actually drew. The median is recorded alongside as `median_geometry` for
anyone who wants it.

Only annotators who agree on **which instance** was circled vote, since averaging
across annotators who resolved the ambiguity differently would produce a circle
enclosing neither object. A tie means the scene is genuinely contested, no
consensus is written, and it is listed under `consensus.contested` in
`metrics.json` with the vote breakdown. With one annotator the consensus is that
annotator, recorded as `consensus_n_annotators: 1`.

## Known limitations

- **No result yet.** This is the instrument, not the finding. The numbers appear
  once response files exist.
- **`n` per cell is small.** Thirty-six scenes across four tiers and three suites
  is three scenes per (suite, tier) cell. Tier-level rates are indicative;
  suite-level and pooled rates are the ones to quote. The recommended
  augmentation ranges come from 5th–95th percentiles and want several annotators
  before they are worth pasting into a builder.
- **Ellipse containment is a proxy for enclosure.** Referential accuracy tests
  the *fitted ellipse* against instance *centres*, not the drawn stroke against
  object silhouettes. A stroke that visibly encloses the target but fits badly —
  a figure-of-eight, say — can score wrong. `target_vismask.png` ships with every
  scene if a silhouette-based check is ever wanted.
- **`all_pixels` includes the destination instance.** A circle drawn generously
  enough to also enclose the destination scores *contains several*, which is the
  specified reading but is stricter than "circled the right object". The
  `contains_target` column is the looser reading.
- **The four ranges are tested independently.** The auto drawer samples them
  jointly, and the recommendation treats each margin separately; correlations
  between, say, radius inflation and centre offset are visible in
  `per_scene.csv` but are not modelled.
- **Self-selected annotators, self-reported effort.** Timing comes from
  `performance.now()` inside the tool, so it measures wall-clock time on the
  scene, including any time the annotator spent looking away.

## Round-trip verification

Before the full bundle was generated, the pipeline was checked end to end on a
3-scene vertical slice — build, draw, export, score — per the repo's standing
build order. Two real defects surfaced and were fixed:

- the in-browser linear solver indexed a matrix row as though it were the matrix
  (`r[i][i]` where `r` is already row `i`), returning `NaN` for every system, so
  every ellipse fit was silently falling through to the polygon fallback;
- `render()` could be entered from an image `onload` handler before `start()` had
  built the per-scene state array, throwing and killing the handler.

The live browser was not available for the round trip, so the tool's **actual**
`<script>` was extracted from the generated HTML and executed under Node against
a DOM shim, driven by synthetic pointer events through the real `pointerdown` /
`pointermove` / `pointerup` handlers. `toImg`, `commit`, `fitEllipse`, the
`localStorage` save/restore path and `buildExport` are the shipped code, not
reimplementations. Canvas rendering is the one path that was verified by eye
instead, from the contact sheets. The synthetic responses were deleted afterwards
so nothing in `responses/` is fabricated.
