# Sketch-Prompted VLA — implementation plan

Aaron — 16 August 2026

Where the project stands, what the next run is, and what the finished thing
looks like. Companion to `sketch_prompted_vla_proposal.md` (the research
argument), `PROMPT_TAXONOMY.md` (the vocabulary), `outputs/rollouts/ROLLOUT.md`
(what the harness measured) and `report/` (the write-ups).

This document is a plan, not a result. Every number in it is copied from an
artefact already in the repo, and the path to each is given so it can be
checked.

---

## Vocabulary — read this first

Two things can make a task hard here, and until 16 August I called both
"ambiguous". They are now separate axes and the whole document uses them
strictly. Full definitions in `PROMPT_TAXONOMY.md`.

**Prompt type** — a property of the caption, read with the picture covered up.

| | the caption | example |
|---|---|---|
| **explicit** | names the target and the destination by category | *"pick up the black bowl and place it on the plate"* |
| **ambiguous** | names neither; pointing words only | *"pick this up and place it on that"* |

**Tier** — a property of the scene, counted with the caption covered up:
`control` (one candidate to one), `referential` (many to one), `directional`
(one to many), `both` (many to many).

They are independent. Adding objects to a scene changes the tier and never the
prompt type. A `control` scene with an ambiguous caption and a `both` scene with
an explicit caption are ordinary rows.

**Every scene carries both captions**, so the benchmark is **228 evaluation
rows** over 114 physical scenes — same BDDL, same `init_state.npz`, same
sketches, same tiers.

**Reading the older numbers.** Everything measured before 16 August used the
authored captions, which are *explicit* captions under this vocabulary. So the
34.5% below is an explicit-prompt number at 3 rollouts per scene. No result was
invalidated by the redefinition; they were renamed.

---

## The state in one paragraph

The dataset, the harness and the sketch-free baseline are finished, and the
published-benchmark reproduction has run at full trial count. π₀.₅-LIBERO scores
**34.5%** on the 114 scenes with explicit captions. The same pipeline, the same
checkpoint and the same text-only input scores **97.0%** on the four standard
LIBERO suites. The control tier splits that gap into about 25 points of
distribution shift and about 45 points of referential ambiguity, so the headroom
a sketch could recover is measured, and it is large. Two things have never been
run: the ambiguous-caption arm, which is the condition a sketch is meant to
rescue, and any test of whether a real model can *read* a sketch — every sketch
number in the repo so far comes from a scripted oracle handed the geometry as
parsed numbers, not from a policy looking at a drawing.

### No sketch has touched any of those numbers

This is the easiest thing in the project to misread, so it is stated once,
plainly.

π₀.₅ takes a base image, a wrist image, an 8-D state and a language string. That
is all. **It has no sketch channel and cannot be given one.** So neither 34.5%
nor 97.0% involves a drawing of any kind. Both are text-only.

What differs between them is the **scenes**, not the input:

| | which scenes | the caption | the input |
|---|---|---|---|
| **97.0%** | the four standard LIBERO suites — what the checkpoint was fine-tuned on | explicit, one right answer | text only |
| **34.5%** | my 114 synthesised scenes | explicit, but several candidates match it | text only |

The 62-point drop is therefore **"familiar scenes with one candidate vs
synthesised scenes with several"**, not "with sketch vs without sketch". Reading
it the second way would credit the sketch with a gain nobody has measured yet.

One detail for precision: 97.0% is the average over all four standard suites.
The **98.2%** used in the gap decomposition below is that same run restricted to
the three suites my scenes derive from (Spatial, Object, Goal), so both sides of
the subtraction come from comparable scenes.

## Final goal

An ICRA-style paper whose central table is:

| | explicit caption | ambiguous caption |
|---|---|---|
| **text only** | **34.5%** ✅ measured at 3 rollouts | ⬜ stage 4c |
| **+ sketch, zero-shot** | ⬜ stage 5 | ⬜ stage 5 |
| **+ sketch, fine-tuned** | not my track | not my track |

Every cell is the same 114 physical scenes, so the columns are directly
comparable and can be compared scene by scene — the harness seeds rollout *k* of
a scene identically in both arms, because `stable_seed` does not read the prompt
type.

The third row — sketch-conditioned fine-tuning, the proposal's §4 method — is
**someone else's track** and is deliberately out of scope for this plan. My
deliverable is the first two rows plus the supporting apparatus: the
human-in-the-loop protocol, the auto-vs-human agreement study, and the
referential/directional disambiguation benchmark itself as a released artefact.

The four contributions listed in the proposal, §6, still stand. Nothing in this
plan changes them.

---

## Stage board

| # | Stage | State | Artefact |
|---|---|---|---|
| 0 | Three validation suites, 114 scenes, schema v1.0 | **done** | `outputs/validation_set_*`, `report/validation_suites/` |
| 1 | Rollout harness, eight open issues resolved | **done** | `scripts/rollout_sketch.py`, `outputs/rollouts/ROLLOUT.md` |
| 2 | Scripted-oracle sketch-vs-text gap | **done** | `full_run_plane`, `full_run_depth` |
| 3 | Human sketch tool + first annotator | **partly done** | `outputs/human_study/`, `human_r1`, `human_r1_depth` |
| 4 | π₀.₅-LIBERO sketch-free baseline, explicit captions, 3 rollouts | **done** | `outputs/rollouts/pi05_baseline/`, `report/pi05_baseline/` |
| 4a | 500-trial reproduction of the published numbers | **done** 13 Aug | `RUNBOOK_REPRO_500.md`, `outputs/rollouts/openpi_repro_500/` |
| 4b | Prompt taxonomy + ambiguous captions, 228 evaluation rows | **done** 16 Aug | `PROMPT_TAXONOMY.md`, `scripts/build_prompt_variants.py`, `outputs/evaluation_rows_all.json` |
| 4c | Both caption arms at 532 trials/suite | **next run** | `RUNBOOK_BASELINES.md`, `scripts/run_baselines.sh`, `report/prompt_baselines/` |
| 5 | Zero-shot sketch arms: overlay + language | code written, never run | `scripts/pi05_sketch.py`, `brief_pi05_recovery.md` |
| 6 | Error bars, human study at scale, paper | not started | — |

Sketch-conditioned fine-tuning and the training-data export are **not on this
board**. They belong to another track and I am not planning them here.

---

## What is done

### Stage 0 — the scenes

114 scenes across three suites, 38 each, synthesised as real LIBERO BDDL by
direct string templating. Tier split per suite: control 5, referential 12,
directional 9, both 12. Every scene passes the full gate stack and was
re-verified independently of its build log.

The non-control tiers are built so that the **explicit** caption cannot pick out
the referent — duplicate near-identical objects, several plausible destinations —
so the sketch is the only disambiguating signal available. That is a property of
the scene, which is why it is the tier and not the caption that encodes it.

`libero_10` is postponed on purpose: its goals need two or three predicates and
one circle plus one arrow cannot express two actions.

### Stage 1 — the harness

`scripts/rollout_sketch.py` drives any policy satisfying the `SketchPolicy`
protocol over the suites, and resolves the eight issues raised in
`brief_libero_rollout_harness.md`. Three facts worth carrying forward:

- **All 114 scenes reproduce their annotated initial state at 0.000 px
  residual.** No scene needed the fallback rungs. `init_state_capture_report.json`.
- **The harness has a measured noise floor and it is not zero.** `determinism_r1`
  puts the outcome flip rate at 4.2% [1.4%, 11.5%] and the spread on a 36-scene
  run at 5.6 pp, caused by a solver warm start that is not restored between
  rollouts. `--reset-warmstart` (`determinism_r2`) did not remove it. The
  decision was to report the band rather than chase the harness to zero — but no
  number in the repo currently carries that band. See stage 6.
- **`--prompt-type {explicit,ambiguous}`** selects the caption and is recorded in
  a `prompt_type` column on every results row. The harness raises rather than
  falling back if the caption keys are missing, and refuses to let two arms share
  a `--run-id`.

### Stage 2 — the scripted oracle

Non-privileged support-plane deprojection, all 114 scenes: **auto 37.7% vs
text_only 18.1%, a +19.6 pp gap.** Control tiers show no gap, as they must — a
control scene has one candidate, so the text-only guess is already right and the
sketch adds nothing. The gap appears exactly on the tiers engineered to put
several candidates in front of the gripper.

This establishes the sketch is an *executable specification*. It does not
establish that a model can read one. That distinction is the whole reason stage 5
exists.

The known wart: no single deprojection reads the whole benchmark. The plane route
mis-places Object's flat groceries by up to 8 cm laterally (the `SUPPORT_Z`
constant was fitted to the tall ones); the depth route fixes Object and breaks
Goal, because a region-typed destination has no object surface for the ray to
land on — `wine_bottle → REGION` reads z = 0.707, over 20 cm below the table.
**This limits the scripted-oracle track only.** π₀.₅ acts on images directly and
never touches the deprojection, so stages 4–6 are unaffected by it.

### Stage 3 — the human study, first pass

The drawing tool, the protocol and the scorer all exist and have been exercised
against real data once: one annotator, the 36-scene study subset, scored as
`human_r1` (plane) and `human_r1_depth`. Read naively the human sketch loses to
text-only; `ROLLOUT.md` explains why that reading is wrong.

**One annotator is not a study.** Twelve scenes per suite puts every per-suite
figure inside its own error bars (binomial SE ≈ 14 pp at n = 12), so only the
pooled figure and the matched-referent decomposition are quotable, and
`human_consensus` currently means nothing because there is nothing to reach
consensus with.

### Stage 4a — the pipeline certification

openpi's own unmodified LIBERO eval loop on the standard suites, **50 trials per
task, 500 episodes per suite, 2,000 episodes, zero exceptions**:

| suite | published | observed |
|---|---|---|
| LIBERO-Spatial | 98.8 | 98.0 |
| LIBERO-Object | 98.2 | 98.6 |
| LIBERO-Goal | 98.0 | 98.0 |
| LIBERO-10 | 92.4 | 93.4 |
| **average** | **96.85** | **97.0** |

Every suite within 1.0 point of published; binomial SE at 500 episodes is about
1 point, so the residuals are sampling noise. Until this ran, "my scenes are hard
for π₀.₅" and "my observation pipeline is subtly wrong" were indistinguishable,
and there are at least four ways to be subtly wrong that all present as a merely
low success rate — resolution, the 180° rotation, the wrist camera, and chunked
control. `outputs/rollouts/pi05_baseline/openpi_reference.json`.

One task is worth recording: within LIBERO-10, *"put both moka pots on the
stove"* scores **56.0%** while the other nine average 97.6%. It is the only task
in the four standard suites whose instruction refers to two identical instances
of one object — the ambiguity class this project is about — and it is where the
checkpoint is weakest. A single task is not a controlled result, but it is a
suggestive one.

### Stage 4 — the sketch-free baseline

Stock `pi05_libero` checkpoint, **explicit captions**, text only, 114 scenes × 3
rollouts = 342 rows, zero skips. No sketch anywhere.

| | value | source |
|---|---|---|
| Sustained success | **34.5%** | `summary.json` |
| Same pipeline, same input, three standard suites | 98.2% | `openpi_reference.json` |
| Control tier | 73.3% (15 scenes, 45 rollouts) | `analysis.json` |
| Other three tiers | 28.6% (99 scenes, 297 rollouts) | `analysis.json` |
| ⇒ distribution shift | ≈ 24.9 pp | 98.2 − 73.3 |
| ⇒ referential ambiguity | ≈ 44.7 pp | 73.3 − 28.6 |
| Lifted *something* | 97.1% of all rollouts | `summary.json` |
| Wrong grasps that were same-category siblings | 73.2% of 112 | `analysis.json` |

Two tier-level findings sharpen it, and they are where stages 4c and 5 have to
show an effect if they show one anywhere:

- **Directional tier: 40.7% correct destination against a 39.8% chance floor.**
  That is chance to within a point. The model is not weakly disambiguating
  destinations on these scenes; it is not disambiguating them at all.
- **Referential tier: 50.0% correct object against a 34.8% floor.** Above chance
  but far from resolved — some captions carry weak spatial cues and it exploits
  them partially.

**And I know which object it grabbed each time, by name.** `results.csv` carries
a `grasped_instance` column, filled in for all 332 grasps with no blanks:

| scene | asked for | actually grabbed |
|---|---|---|
| `spatial/scene_0012` | `akita_black_bowl_4` | `akita_black_bowl_3` |
| `spatial/scene_0016` | `akita_black_bowl_2` | `akita_black_bowl_4` |
| `spatial/scene_0009` | `akita_black_bowl_2` | `glazed_rim_porcelain_ramekin_1` |

The model reaches confidently for a black bowl — just not the one asked for.
That is referential confusion identified by instance, not inferred from a low
score. It is also why the sibling rate is the sharpest test in stage 5: a sketch
that works should collapse that 73.2% specifically.

What is **not** measured is *why* it picks the sibling it picks — nearest, most
central, best lit. Nobody has looked. A nice-to-have ablation, not a blocker.

Nothing needed ablating to get this. `scripts/pi05_policy.py` reads one caption
field and refuses a sketch-bearing prompt outright, so a number reported as a
baseline cannot have been produced with a sketch in the loop by accident.

Cost: 19,608 inference calls at 86.6 ms mean latency, roughly 50 minutes on one
RTX 4090.

Two caveats travel with every number above and must keep travelling:
`correct_destination` is an xy-proximity proxy (0.08 m threshold), not LIBERO's
contact solver, and is systematically pessimistic for Goal's region-typed
destinations; only `success_sustained` comes from `env.check_success()` and only
it should be quoted as a success rate.

### Stage 4b — the prompt taxonomy

One word, "ambiguous", was doing two jobs: describing a caption that does not
name its referents, and describing a scene with several candidates. That made
"an ambiguous scene with an explicit prompt" sound contradictory when it is
ordinary, and it meant no table could separate the two effects. Split into the
two axes at the top of this document.

Every scene gained `instruction_explicit` and `instruction_ambiguous`; the legacy
`instruction` key still equals the explicit one, so older consumers keep working.
Two manifests with two jobs, not two versions of one file:
`validation_manifest_all.json` lists the 114 physical scenes,
`evaluation_rows_all.json` the 228 evaluation rows.

The ambiguous captions are a **bank of 20 wordings in 5 buckets**, not one string
per predicate. Each scene draws only from the bucket matching its own explicit
caption's shape and verb, so the two arms differ in referring expressions and
nothing else — the two `push` scenes get push wordings, not "pick this up".
Assignment is round-robin within (bucket, suite, tier) with a blake2b-derived
per-group offset, so no template can concentrate in one tier and confound the
tier comparison. `prompt_template_id` on every row lets the analysis check
whether wording mattered at all.

Verified: audit clean 114/114, caption check clean 228/228, and zero captions
containing any of the 41 words that name something in a scene — the check builds
that vocabulary from the object instances themselves rather than a hand-kept
denylist.

---

## The next run — stage 4c, both caption arms at full trial count

Runbook: `RUNBOOK_BASELINES.md`. Runner: `scripts/run_baselines.sh`.

Two arms on the stock checkpoint, no sketch, **14 rollouts × 38 scenes = 532
trials per suite**, 1,596 per arm:

- **`explicit`** — re-measures stage 4's 34.5% at 4.7× the rollouts.
- **`ambiguous`** — the same scenes with the names removed.

500 does not divide by 38, so this is 500 rounded up to a whole rollout per
scene; every scene gets equal coverage. ~4 hours per arm on one RTX 4090.

### How I will read it

- **`control` × `ambiguous` is the cell to watch.** One candidate object, one
  candidate destination, so the *scene* answers the question the caption
  refuses. If the policy still fails there, the failure is about the words, not
  about choosing between candidates — and that is a different deficit from the
  one stage 5 is designed to fix.
- **Ambiguous should be at or below explicit in every tier.** If it beats
  explicit anywhere, suspect a mislabelled arm before believing it; the Part C
  smoke gate exists for exactly that.
- **Per-template spread.** `analyze_baselines.py --arms` reports success per
  caption template. A template far off the others is a lexical artefact, not a
  finding about ambiguity.
- **Paired, not pooled.** Same scenes and same rollout seeds across arms, so
  report the per-scene paired difference and how many scenes flipped each way. A
  pooled −4 pp built from 30 regressing and 26 improving is a different finding
  from 30 regressing and none improving.

Output: `report/prompt_baselines/report.tex`, two pages, five tables. Its numbers
come from a generated `tables.tex` and are never typed by hand.

---

## After that — stage 5, the zero-shot sketch arms

**This is the outstanding test of the project's actual claim.** Brief:
`brief_pi05_recovery.md`. Code: `scripts/pi05_sketch.py` plus
`--pi05-sketch-mode {none,overlay,language}` — written on a laptop with no GPU,
verified only against a stub, **never run against a live policy server**. Treat
it as a considered draft; where it contradicts the machine, the machine wins.

Two arms on the stock checkpoint, all 114 scenes, no training:

- **`overlay`** — the circle and arrow composited into the image the model sees.
- **`language`** — the same circle and arrow described in words, appended to the
  caption.

### Why both, and how I will read the result

They are a matched pair, not alternatives. π₀.₅ never saw annotation marks in an
image, so a null from `overlay` alone is uninterpretable. `language` speaks in a
modality the model was trained on, and therefore bounds what *any* prompt could
recover on these scenes.

| | overlay works | overlay fails |
|---|---|---|
| **language works** | both channels usable | the information suffices, the **modality** is the barrier — the finding the fine-tuning track needs from me |
| **language fails** | marks beat words — surprising, check for a leak | the bottleneck is not referential, and the ≈45 pp attribution needs revisiting |

All four cells are publishable. I am not going to treat a null as a failure to
engineer away, and I am not going to tune the prompt wording until the number
improves — an alternative phrasing is a separate labelled arm, not a replacement.

**Which caption do the sketch arms sit on?** Both, eventually; `ambiguous` first
if budget forces a choice, because that is the condition a sketch is meant to
rescue and it is where the headroom is largest.

### The two traps

**The firewall.** Both adapters read `prompt.symbolic_tokens` and nothing else.
Neither ever sees `meta['target']`, `meta['destination']`, `pick_px` or
`place_px`. The language string is derived from the *sketch's own geometry*, so
it carries exactly what an annotator drew and not one bit of ground truth beyond
it. A paraphrase built from the target's name would be a different and dishonest
experiment.

**The rotation, which cuts the two arms opposite ways.** The model is fed a
180°-rotated frame — settled by measurement in the baseline. `overlay` draws on
the raw 256 frame *before* the rotation, so the marks rotate with the pixels they
annotate and stay attached for free. `language` must describe the frame **as the
model sees it**: a circle at the top-left of `frame0` is at the bottom-right of
the model's view. Describing the unrotated frame hands the model a confidently
inverted instruction, which is worse than no instruction, and it fails silently.
`describe_tokens(..., rotate180=True)` handles it and `spatial/scene_0000` is the
worked example — circle at (92, 41) in frame0 space reads "top-right", but
(35, 86) in the model's view reads **"bottom-left"**, which is what the prompt
must say.

**Word order matters now.** Every ambiguous template puts the target first and
the destination second. The language adapter appends geometry to that string and
depends on the order holding.

### Run order

Vertical slice → smoke → full, which has caught a real bug at every stage.

1. **Pod up.** RunPod, RTX 4090 or A40, same network volume, EU-RO-1.
   `bash scripts/pod_bootstrap.sh`; `RUNPOD_SETUP.md` is the full sequence.
   `MUJOCO_GL=egl`, no nested Docker.
2. **Orientation check (the gate).** Smoke `overlay` with `--pi05-dump-frame`,
   open the dumped frames, confirm the circle sits on an object and the arrow
   points at a destination *in the rotated view*. Then smoke `language` and read
   the prompt strings out of the log. **Do not start the full runs until both
   hold.**
3. **Full runs**, one run-id each. ~50 minutes each at 3 rollouts.
4. **Analysis.** Extend `scripts/analyze_baselines.py` rather than writing a
   second script, so every arm comes out of one code path. **Do not re-run the
   text-only arms** — stage 4c's results are the comparison.
5. **Report** `report/pi05_recovery/`, with the overlay panel from step 2 as a
   figure — a reader must be able to see what the model was shown.

---

## Stage 6 — what the paper needs that the repo does not have

- **Second and third annotators.** One annotator over 36 scenes is not a study.
  Until there are two, `human_consensus` is a column with no meaning and the
  inter-annotator agreement claim in contribution 4 is unsupported. Cheap, no
  GPU, should start now rather than wait.
- **Error bars on everything.** The 5.6 pp determinism spread is measured and
  written into no number. Every gap I quote inherits it. A presentation change,
  not new experiments.
- **The deprojection honesty note**, or a region-aware `z_place`. A per-category
  `SUPPORT_Z` table is *not* the fix — category identity is ground truth, the
  same leak class the project is built to avoid. Scripted-oracle track only.
- **Bibliography check.** Two December 2025 arXiv ids in the proposal's related
  work are flagged as unverified and must be confirmed before they go in.

---

## Risks, honestly

1. **`language` fails as well.** Then the ≈45 pp attributed to referential
   ambiguity is not referential, the baseline decomposition needs revisiting, and
   the project's premise weakens. This is the outcome that would cost the most
   and it is a live possibility. It is also why the arm exists.
2. **The ambiguous arm lands near the explicit arm.** That would mean π₀.₅ was
   never using the object names much on these scenes — plausible given the
   directional tier already sits at chance — and it would make the caption axis
   less interesting than the scene axis. Still publishable, but it changes what
   the sketch is for.
3. **Object stays uninterpretable.** Neither deprojection reads it and Goal
   together. Affects the scripted-oracle table, not the π₀.₅ arms.
4. **The overlay marks land wrong and nobody notices.** The rotation cuts the two
   arms opposite ways and both fail silently. Step 2 of the run order is the only
   thing between me and a confidently wrong result, which is why it is a hard
   gate and not a sanity check.

## What I need to confirm before committing to a schedule

**The submission deadline and the venue.** The repo is named `ICRA26` but I
should confirm which call this is actually aimed at before laying dates against
the stages above. This plan deliberately carries none.

---

## Reproducing anything above

| what | where |
|---|---|
| Vocabulary and the caption bank | `PROMPT_TAXONOMY.md`, `scripts/build_prompt_variants.py` |
| Suites and gates | `scripts/build_validation_set_*.py`, `scripts/audit_validation_sets.py` |
| Hard-won LIBERO facts | `SUITE_FACTS.md`, `SCHEMA.md` — read before touching a builder |
| Harness protocol and measurements | `outputs/rollouts/ROLLOUT.md` |
| Sketch-free baseline | `brief_pi05_baseline.md`, `report/pi05_baseline/` |
| Both caption arms | `RUNBOOK_BASELINES.md`, `scripts/run_baselines.sh`, `report/prompt_baselines/` |
| 500-trial reproduction | `RUNBOOK_REPRO_500.md`, `scripts/repro_500.sh` |
| Sketch arms brief and steps | `brief_pi05_recovery.md`, `RUNBOOK_PI05_RECOVERY.md` |
| Pod setup | `RUNPOD_SETUP.md`, `scripts/pod_bootstrap.sh` |
| Human study | `outputs/human_study/HUMAN_STUDY.md` |
| Which reports are current | `report/README.md` |
