# Sketch-Prompted VLA — human sketch collection study

Aaron — 2 August 2026

Datasheet for the human-sketch agreement study. Companion to the three suite
`DATASHEET.md` files; canonical schema v1.0 (see `SCHEMA.md`).

Built by `scripts/build_human_study_bundle.py`, scored by
`scripts/score_human_sketches.py`. Both are pure stdlib + numpy + Pillow +
matplotlib, and need no simulator — no `robosuite`, `mujoco` or `libero`.

## Purpose

Every suite datasheet lists the same open item: the validation sketches are
auto-drawn, and nothing yet establishes that they resemble what a person would
draw. This study closes that gap. It answers two questions:

1. **Can a human supply a usable disambiguating signal at all?** The scenes are
   deliberately impossible to resolve from the caption. The measurable form of
   this is *legibility and decisiveness*, not agreement with the BDDL: does a
   person produce one clean circle and one clean arrow, quickly, without
   skipping? Whether they land on the same instance the BDDL happens to name is
   a separate and much weaker question — see *What "correct" means* below.
2. **Is the auto-annotator's imprecision augmentation wide enough to cover real
   human strokes?** The auto drawer jitters the circle centre, the radii, the
   arrow endpoints and the arrow bend by four fixed ranges. Those four ranges are
   an untested guess. This study measures the human distributions against them
   and recommends replacements.

The second question is the actionable one — its output feeds straight back into
`draw_circle` / `draw_arrow` in the three builders.

## What "correct" means, and why the BDDL is not the referee

This section governs how every number below is read. It was written after the
first pass of the study design, and it corrects a framing error in that design.

**The problem.** On a scene with two akita bowls and two plates, "put the bowl on
the plate" names neither. The BDDL names one pair — `(On akita_black_bowl_1
plate_2)` on `goal/scene_0026` — but that pair was fixed by the builder when it
injected the duplicates, and nothing in the 128×128 frame distinguishes bowl_1
from bowl_2. An annotator who circles bowl_2 has not made an error. They have
resolved an ambiguity differently.

In the 36 sampled scenes:

| | scenes |
|---|---|
| target has a same-category clone in frame | 21 |
| destination has a same-category clone | 12 |
| neither — fully determined by the picture | 9 |

Over all 114, 90 have at least one same-category decoy.

**The consequence.** Treating the BDDL pair as the correct answer makes a
correct-but-different sketch look like annotator error, and it does so on 21 of
36 scenes. Scoring a *rollout* that way is worse: a policy that follows a bowl_2
sketch perfectly scores `check_success() == False`, so the human-sketch condition
would lose to the auto-sketch condition for reasons that have nothing to do with
sketch quality. That would invalidate the three-way comparison this study exists
to enable.

**The resolution — the sketch is the specification, not a hint toward a hidden
answer.** This follows from the project's own premise: the caption is
insufficient *by design*, and the drawn referent is what supplies the missing
argument. Under that reading, "grasp the circled object and put it where the
arrow points" is the task, and any (bowl, plate) pair is a legitimate one. The
BDDL predicate is one instance of the task, not the definition of it.

**Is that too permissive?** It would be if the sketch were decoration on top of a
fixed goal — then anything goes and nothing is tested. It is not, because the
sketch is a *hard constraint the policy must read off an image*. What is being
tested moves from "did the sketch pick the blessed object" to "**was the sketch
legible enough that the policy did what it said**". A model that grasps the
uncircled bowl fails, whichever bowl was circled. That is a real and demanding
criterion.

**Does it collapse human-sketch and auto-sketch into the same condition?** No,
and this is the important consequence. Once both conditions are scored against
their own sketch, the referent choice drops out and what remains is exactly the
difference that matters: **legibility**. Auto strokes are crisp, near-round,
centred within a few pixels of the centroid, with two-segment arrows. Human
strokes wobble, inflate, run as open arcs rather than closed loops, sometimes
enclose two instances, and land the arrow head at a different offset. If
fidelity under human sketches is materially lower than under auto sketches, the
augmentation is too narrow and the policy is brittle to real strokes — which is
question 2, measured end to end instead of by proxy. If it is not lower, the
auto sketches are an adequate stand-in and the synthetic-to-human gap the suite
datasheets flag is closed.

**What is genuinely lost.** Question 1 in its original form. "Can a human pick
the right object" is unanswerable on a clone scene, and a pooled referential
accuracy near 50% there would measure the clone rate, not the annotator. Its
salvageable replacements, both already computed:

- **control-tier accuracy** (6 scenes, one candidate each) — the honest accuracy
  check, and the attention check for filtering bad submissions;
- **inter-annotator agreement** — if several people independently circle the same
  bowl, the scene has a natural reading; if they split, it is genuinely
  contested, which `consensus.contested` already records with the vote
  breakdown;
- **skip rate**, by reason.

**Why no BDDL variants are enumerated.** An obvious alternative is to author one
BDDL per admissible pair and score against whichever the sketch selected. It is
tempting and it does not scale in the form it first appears: 2 bowls × 2 plates
is 4 goals, 3 × 3 is 9, and each variant needs the builders' full gate stack
re-run — graspable, reachable, visible, oracle-positive — none of which the
duplicate instances ever passed. `oracle_negatives` establishes only that the
other pairs score False *under the original goal*; it says nothing about whether
they are achievable.

The combinatorial framing is also the wrong one. A sketch selects exactly one
pair at rollout time, so at most one variant per (scene, sketch) is ever needed —
linear in the number of sketches collected, not quadratic in the object count,
and smaller still where annotators agree. Lazily emitting and gating that single
variant is a viable route if a BDDL-scored number is ever wanted per sketch. It
is not the route taken here, because `sketch_fidelity_object` /
`sketch_fidelity_destination` in `rollout_sketch.py` already measure the same
thing without authoring or re-gating anything.

**How this is reported.** Every rollout condition carries two numbers:

| number | source | reading |
|---|---|---|
| `success_sustained` | `env.check_success()` on the scene's own BDDL | intended-pair success; unfair to any sketch that resolved the ambiguity differently, and reported for continuity with the existing runs |
| `sketch_fidelity_object` / `_destination` | nearest `all_pixels` instance to the policy's grasped object / delivery point, against the instance the sketch pointed at | did the policy do what the picture said — the headline number for the human-vs-auto comparison |

The gap between them is not noise to be minimised. It is the **cost of
ambiguity**: how often a perfectly executed sketch disagrees with the benchmark's
arbitrary choice. Joining those columns to this scorer's per-scene referent
labels on `(suite, dir)` is the outstanding work — `ROLLOUT.md` issue 7.

**A confound to control.** Not every pair is equally hard. A plate resting on the
flat table and a plate on the stove are different placement problems, so
comparing human-sketch fidelity against auto-sketch fidelity across *different*
pairs mixes prompt legibility with task difficulty. Two mitigations, both cheap:
report fidelity restricted to scenes where the annotator and the auto drawer
chose the same referent (a clean within-pair comparison, `n` permitting), and
report the difficulty spread across pairs from the control-tier scenes where no
choice exists.

## The four ranges under test

From `draw_circle` / `draw_arrow`. All three builders
(`build_validation_set_{spatial,object,goal}.py`) carry **byte-identical**
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

### Census mode — annotating all 114

`python scripts\build_human_study_bundle.py --all` skips the sampling entirely
and emits every scene on disk: **113 scored + 1 practice**, into
`outputs/human_study/full114/`, scored with
`score_human_sketches.py --study-dir outputs/human_study/full114`.

Four things differ from the sampled bundle, all deliberate.

- **Separate directory, not an overwrite.** `full114/` carries its own
  `scene_subset.json`, `sketch_tool.html` and `responses/`. The sampled 36-scene
  roster may already be in someone's hands, and a response scored against the
  wrong roster is silently wrong rather than loudly wrong. Both can exist side by
  side; the scorer picks one with `--study-dir`.
- **The practice scene is held out, not drawn from outside.** With the census
  taken there is no "outside", so `--all` picks the practice scene *first* and
  takes the remaining 113 as the scored set. `spatial/scene_0002` at seed
  `20260802`, the same scene the sampled bundle uses. The invariant that no
  scored scene carries a warm-up annotation is preserved; the cost is that one
  control scene is never scored.
- **No tier quota.** The census inherits the suites' natural tier distribution —
  spatial `4/12/9/12`, object and goal `5/12/9/12` — rather than the 2/4/3/3
  balance. Control is 14 of 113 (12%) instead of 6 of 36 (17%), so the
  attention-check baseline is proportionally thinner but absolutely larger.
- **3.68 MB instead of 1.21 MB.** Still one file, still offline, still no server.
  Zip it before mailing.

**When the census is worth it.** Every cell in the (suite × tier) breakdown goes
from 3 scenes to 9–12, which is the difference between indicative and quotable —
the *n per cell is small* limitation below is the single biggest weakness of the
sampled design. It also removes any sampling question from the calibration
percentiles feeding back into the builders.

**What it costs.** Roughly three times the annotator effort. If several people
are being asked to annotate, the sampled 36 is still the right roster for them:
inter-annotator agreement needs shared scenes, not many scenes, and asking a
collaborator for 113 is a good way to get a rushed file. A workable split is the
census for my own pass and the sampled 36 for everyone else — the 36 are a subset
of the 114, so agreement is computable on the overlap by joining on
`(suite, dir)`.

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

All four families are computed **pooled, by suite and by tier**. The tiers are
what make the benchmark diagnostic, so a pooled number alone is not enough — and
for family (a) the pooled number is actively misleading and must not be quoted at
all, for the reason given in *What "correct" means*. It stays in `metrics.json`
only so the by-tier blocks have something to sum to.

### (a) Agreement with the BDDL referent — NOT "correctness"

The scorer's field names (`referential_ok`, `directional_ok`, `joint_ok`) and
`metrics.json`'s `accuracy` block predate *What "correct" means* above. The
names are kept so existing consumers do not break; the reading is not. On a
scene whose target has a same-category clone these measure **agreement with the
instance the BDDL happens to name**, which is a coin flip by construction, not
accuracy. Quote them by tier, never pooled.

- **Referential agreement** — the fitted ellipse contains `pick_px` and contains
  no other instance centre from `all_pixels`. Failures split into *contains the
  wrong object* (exactly one instance, not the target), *contains several*, and
  *contains none*. A looser companion rate, `contains_target`, is reported
  alongside; it ignores whether other instances were also enclosed.
- **Directional agreement** — the candidate in `all_pixels` nearest the arrow
  head is the true `destination`.
- **Joint** — both.
- **Skip rate**, broken down by reason.

How to read the tiers:

| tier | n | reading |
|---|---|---|
| `control` | 6 | one candidate, one destination — **genuine accuracy**, and the attention check for discarding a submission |
| `referential`, `both` | 21 | target has a clone; a miss is scene ambiguity, not annotator error |
| `directional`, `both` | 12 | destination has a clone; same |

*Contains several* and *contains none* are the two failure modes that remain
diagnostic on every tier, since neither depends on which instance was intended:
the first is a circle drawn too loosely to specify anything, the second a stroke
that fits so badly it encloses nothing. Both are legibility defects and both
would fail a policy regardless of referent.

### (b) Human-vs-auto geometric deviation — reported twice

Per scene, against the auto `symbolic_tokens`: circle centre offset (px, and
normalised by the auto `rx`), radius ratio `human_r / auto_r` on the mean of
`rx, ry`, ellipse IoU (rasterised at 4× supersampling, rotation-aware), arrow
tail and head offsets, arrow angle difference (signed and absolute), arrow length
ratio, and arrow path curvature — the maximum perpendicular deviation of the
human stroke from its own tail→head chord.

**Two blocks, `same_referent` and `all`.** A stroke aimed at a different instance
than the auto drawer contributes the **distance between two objects**, not the
annotator's imprecision — 26 px on `goal/scene_0026`, against a real hand wobble
of 2–3 px. Pooling those into one median reads as human imprecision and is not.
`same_referent` keeps only strokes that point at the auto drawer's instance, so
the residual is stroke noise alone; **this is the block to quote for
imprecision**, and it is the same basis calibration (c) uses. `all` is retained
because the gap between the two is itself informative — it measures how far apart
the candidates sit in the multi-candidate tiers.

The filter is applied **per family, not jointly**: circle metrics gate on
`referential_ok`, arrow metrics on `directional_ok`. Circle and arrow correctness
are independent — an annotator can circle the intended bowl and still send the
arrow to the other plate, and on many scenes only one of the two is ambiguous at
all — so a joint gate would discard a sound arrow because of its circle, and the
reverse. `n_same_referent` and `n_different_referent` are reported per family so
the basis of each block is visible.

`fig_geometry.png` overlays both: light histogram all strokes, dark histogram
same-referent, a median line for each.

This correction was applied after the fact. The first implementation filtered
only skipped rows, which left family (b) pooling both populations while family
(c) — with an explicit comment giving this exact reasoning — filtered correctly.
The inconsistency was the defect, not the reasoning.

### (c) Augmentation-calibration verdict

For each of the four ranges, the human empirical distribution is plotted against
the auto uniform range and the fraction of human strokes falling **outside** it
is reported, with a verdict of *too tight* (>25% outside), *too loose* (human
5–95 interval narrower than half the auto range) or *about right*, and a
recommended replacement range taken from the human 5th–95th percentiles, printed
as the literal `rng.integers(...)` / `rng.uniform(...)` call to paste back into
the builders.

**Two recommendations, raw and robust.** The 5th–95th percentiles are not
resistant to a minority of strokes drawn with a different *convention* rather
than a shakier hand. Measured on the first response file: 21% of arrow-endpoint
residuals sat beyond 3 median-absolute-deviations, almost all of them arrow
**tails** — a human starts the stroke at the edge of the circle just drawn,
while the auto drawer anchors at the centroid. Those 21% dragged the raw
recommendation to `rng.integers(-14, 19)`, ±15% of a 128 px frame. Splitting the
two ends confirms it: tails run p05/p95 of −16.1/+19.4 raw against −0.8/+4.2
trimmed, while heads are −3.2/+9.4 raw against −2.1/+4.5 trimmed.

`recommended_robust` therefore reports the same percentiles after dropping
values more than `ROBUST_K = 3` MAD-SDs from the median, alongside the raw
`recommended`. On the first file that turns `rng.integers(-14, 19)` into
`rng.integers(-2, 6)`. Neither figure is authoritative: the raw one treats an
edge-anchored arrow tail as imprecision to reproduce, the robust one treats it as
behaviour to exclude. That is a modelling decision about what the augmentation is
*for*, and it must be recorded rather than left to whichever number the scorer
printed. `robust.frac_dropped` is reported so the size of the disagreement is
always visible.

For the circle families the two barely differ (0% and 2% dropped), which is the
expected result and a useful check that the trim is not silently reshaping
well-behaved distributions.

Two further things about this family are worth stating.

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
python scripts\build_human_study_bundle.py            # stratified 36
python scripts\build_human_study_bundle.py --smoke    # 3-scene vertical slice
python scripts\build_human_study_bundle.py --all      # census, 113 + 1 practice

# 2. send outputs\human_study\sketch_tool.html to each annotator, alone.
#    Drop the returned JSON into outputs\human_study\responses\
#    Census bundle lives in outputs\human_study\full114\ — tool and responses both.

# 3. score
python scripts\score_human_sketches.py
python scripts\score_human_sketches.py --smoke
python scripts\score_human_sketches.py --study-dir outputs\human_study\full114

# 4. AFTER annotating, to inspect the ground truth of a scene by eye
python scripts\show_scene_truth.py goal 26          # one scene, printed + labelled PNG
python scripts\show_scene_truth.py --subset         # answer-key HTML, the 36 study scenes
python scripts\show_scene_truth.py --all            # answer-key HTML, all 114
python scripts\show_scene_truth.py --all-ambiguous  # which scenes carry decoys
```

`show_scene_truth.py` is read-only. It re-reads each scene's BDDL goal and
`meta.json` and renders `frame0.png` with every candidate ringed and named —
green for `target`, red for `destination`, blue for same-category decoys — so an
instance name can be matched to a blob in the image without reading pixel
coordinates out of a JSON. Output goes to
`outputs/human_study/truth_overlays/`.

**`(:obj_of_interest ...)` in the BDDL is not authoritative.** It is inherited
from the stock LIBERO task and was never rewritten when the duplicate instances
were injected, so on **15 of the 114 scenes** it names a different instance than
the goal actually scored — `goal/scene_0026` lists `plate_1` while the goal is
`(On akita_black_bowl_1 plate_2)`. Only `(:goal ...)` counts, which is what
`audit_validation_sets.py` and `show_scene_truth.py` both parse.

Nothing in this section may be run against a scene before that scene has been
annotated — see *Annotator self-contamination* under *Known limitations*.

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
- **The sketch-fidelity join is not done.** *What "correct" means* nominates
  `sketch_fidelity_object` / `_destination` as the headline number, but those
  columns live in `rollout_sketch.py`'s `results.csv` and are not yet joined
  to this scorer's per-scene referent labels. Until that join exists, the
  human-vs-auto rollout comparison can only be read off BDDL success, which is
  the biased number. This is the single largest outstanding item.
- **Task difficulty is not controlled across referent choices.** See the
  confound note at the end of *What "correct" means*.
- **Annotator self-contamination.** `show_scene_truth.py` makes the ground truth
  easy to look at, which is useful for auditing and fatal for annotating. Any
  scene whose answer was inspected before it was drawn must be excluded from
  that annotator's file, or the annotator excluded entirely.
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

A third surfaced later, during live annotation, and is the reason the round trip
is not sufficient on its own:

- **the practice badge showed on every scored scene.** `showScene()` sets
  `$("practice-flag").hidden = !sc.practice`, which is correct, but the
  `hidden` attribute is honoured only through the UA rule
  `[hidden] { display: none }` — and that loses to any *author* rule setting
  `display` on the same element, because a class selector outranks the UA sheet.
  `.practice-flag { display: inline-block }` therefore pinned the badge visible
  and `hidden` did nothing. Fixed by restating `[hidden] { display: none
  !important }` in the tool's stylesheet.

  **Cosmetic only.** `buildExport()` writes `practice: !!sc.practice` from the
  bundle, never from the DOM, and the progress line reads
  `sc.practice ? "Practice scene" : "Scene n of N"` from the same field — which
  is why the badge and the counter contradicted each other on screen. No response
  file was ever mislabelled and no scored scene was ever discarded. The
  `.practice-flag` element is the only one the JS toggles via `hidden` that also
  carries an author `display` rule, so nothing else was affected.

  It escaped the round trip because that check runs the tool's `<script>` under a
  DOM shim, which has no CSS cascade — an attribute set on a stub object always
  "works". Anything whose failure mode is *rendered appearance* has to be checked
  in a real browser; the round trip covers logic, not layout.

  **Rebuilding is safe mid-session.** `lsKey()` is
  `LS_PREFIX + subset_seed + ":" + annotatorId` and `restore()` accepts any saved
  state whose length matches the roster, so a rebuild at the same seed — which
  reproduces the roster and its order exactly — reloads a part-finished session
  untouched. Verified by fingerprinting the roster before and after the fix
  (`41aa52ca…`, identical). Same browser, same profile, same annotator name, and
  do not clear site data.

The live browser was not available for the round trip, so the tool's **actual**
`<script>` was extracted from the generated HTML and executed under Node against
a DOM shim, driven by synthetic pointer events through the real `pointerdown` /
`pointermove` / `pointerup` handlers. `toImg`, `commit`, `fitEllipse`, the
`localStorage` save/restore path and `buildExport` are the shipped code, not
reimplementations. Canvas rendering is the one path that was verified by eye
instead, from the contact sheets. The synthetic responses were deleted afterwards
so nothing in `responses/` is fabricated.
