# Prompt — closing the human-sketch loop end to end

Paste everything below the line into a fresh Claude Code session in WSL2 with
`conda activate libero`, from `/mnt/c/Users/Admin/sketch_prompted_vla`.

**Aaron does one thing by hand in the middle of this** — draws the 36 scenes.
Step 3 below says when.

---

## Role and objective

You are working in the **Sketch-Prompted VLA** repository. Read `CLAUDE.md`,
`SCHEMA.md`, `SUITE_FACTS.md`, `outputs/human_study/HUMAN_STUDY.md` and
`outputs/rollouts/ROLLOUT.md` before writing code. Do not re-derive facts those
documents already hold.

Every component of this project now exists, but **one path has never been
executed**: a real human stroke has never travelled from the drawing tool through
to a scored LIBERO rollout. `outputs/human_study/responses/` is empty, so the
`human:*` and `human_consensus` conditions have never run on real data. The whole
point of the project is that a person's circle and arrow resolve ambiguity a
caption cannot — and that claim is still untested with an actual person.

Close that loop. When you are done, `outputs/rollouts/` must contain a run whose
conditions include `text_only`, `auto` and `human:<annotator>` over the same
36-scene subset, scored inside LIBERO.

## Where things stand

- 114 scenes, three suites, schema v1.0, audit clean.
- `init_state.npz` captured for all 114; **`nonreproducible.json` is empty**, so
  the 36-scene study roster (seed `20260802`) stands as built and needs no swap.
- `outputs/human_study/sketch_tool.html` is built and validated under a Node DOM
  shim. `scene_subset.json` holds the roster. `responses/` is empty.
- `score_human_sketches.py` computes agreement metrics **and** exports
  `rendered/<annotator>/validation_set_<suite>/<dir>/` mirroring the suites —
  that export is what `rollout_sketch.py` consumes as a `human:*` condition.
- `full_run` holds the first 114-scene rollout (plane deprojection, pre-fix).
  `full_run_plane` and the depth passes are the corrected re-run.

## 1. Loose ends from the deprojection work — clear these first

Three items, all small, all raised in review:

1. **The depth round-trip is untested.** The row-flip bug in `depth_z_at_pixel`
   survived because the round-trip assertion in `sketch_geometry.py` covers only
   the projection path. Extend it: for every entry in `meta['all_pixels']` on a
   sample of scenes across all three suites, sample depth at that pixel,
   back-project, and assert the recovered world point matches. That bug must not
   be reintroducible. Record in `ROLLOUT.md` that the `plate_1` spot-check passed
   under the buggy code because row 64 flips to 63 — a near-symmetric pixel is a
   bad probe, and the next person deserves the warning.
2. **An internal contradiction in the Object `grasped_correct` write-up.** The
   comparison reported the Object referential inversion as "narrowing sharply"
   while quoting `0.0 vs 8.3` for *both* `grasped_any` and `grasped_correct`, and
   explained it as a grasp of the *wrong* object — which cannot hold if
   `grasped_correct` is also 8.3. Recompute, state the correct numbers, and
   correct anything already written into `ROLLOUT.md`.
3. **`full_run/run_config.json` is known-inaccurate** — it records only `auto`
   because of the overwrite bug since fixed. `full_run` is being kept as the
   archival record of the first method, so write a short note into that run
   directory saying what the config omits, rather than leaving a file that
   quietly misdescribes its own run.

## 2. Choose the deprojection method on evidence

When the four-pass re-run lands, report the headline `auto − text_only` under
**both** `plane` and `depth`, broken down by suite and tier, using
`grasped_correct` rather than `grasped_any` for targeting.

Then pick one as the default and say why in `ROLLOUT.md`. Note that the
acceptance test measuring lateral error against `placements` compares to the
object's **base centroid**, whereas a top-down grasp wants the **visible top
surface** — so depth's apparently worse overall figure (4.30 vs 3.63 cm) may be
the metric penalising it for returning the point that is actually wanted. Let
rollout success arbitrate, not the lateral-error table.

Whichever wins, depth remains an oracle affordance a real RGB policy lacks. Keep
it stamped as an upper bound in `run_config.json`, `summary.json` and
`ROLLOUT.md`.

## 3. Prove the human path before Aaron spends time drawing

Do not send Aaron to the tool until the consumption path is known to work.
Vertical slice first, as always in this repo.

Generate a **synthetic annotator response** — take the auto `symbolic_tokens` for
the 36 subset scenes, perturb them well beyond the auto jitter ranges (a real
hand is looser), and emit a file in the exact response schema, including a
skipped scene and a scene whose circle deliberately encloses the wrong instance,
so both edge cases in issue 7 are exercised.

Run it all the way through: `score_human_sketches.py` → check the
`rendered/<annotator>/` export appears in suite-mirroring layout →
`rollout_sketch.py --conditions text_only,auto,human:<synthetic>` on 3 scenes
→ confirm rows appear with the human condition scored.

**Then delete the synthetic file and its derived outputs.** `HUMAN_STUDY.md`
already records that nothing in `responses/` is fabricated, and that must stay
true. Write the synthetic response outside `responses/` if that is cleaner.

Report what broke, because something will. Two things I expect to bite: the
`rendered/` export path may not yet be wired as a condition root in the harness's
scene-source resolution, and a skipped scene has `circle: null`, which the
rollout must treat as "no row for this cell" rather than crashing or scoring a
zero.

## 4. Hand off to Aaron

When step 3 is green, stop and tell him, in a short message, exactly:

- the path to open — `outputs/human_study/sketch_tool.html` — and that it opens
  by double-clicking, no server
- roughly how long 36 scenes takes
- that he should draw quickly and roughly, not carefully, because time-to-draw
  and stroke looseness are measured quantities and deliberating corrupts them
- where to put the downloaded JSON: `outputs/human_study/responses/`
- that he is annotator 1 of ideally 2–3; inter-annotator agreement needs at least
  a second person on the identical roster

Do not proceed past this point without a real response file. Do not simulate one
and present the numbers as a result.

## 5. Score the real thing

Once a real response lands:

1. `score_human_sketches.py` — report referential accuracy, directional accuracy,
   joint accuracy, skip rate, the four-parameter augmentation calibration verdict
   with recommended ranges, and median time-to-draw. With one annotator,
   inter-annotator agreement degrades to a note; everything else stands.
2. `rollout_sketch.py` over the 36-scene subset with conditions
   `text_only`, `auto`, `human:<annotator>`, under the deprojection method chosen
   in step 2, into a new `run_id`.
3. Report the three headline numbers: **`auto − text_only`**,
   **`human − text_only`**, and **`human − auto`** — the last being the
   sim-to-human transfer gap that all three suite datasheets list as an open
   limitation. Break down by tier.
4. Write the results into `ROLLOUT.md` and `HUMAN_STUDY.md`, replacing the
   "no result yet" placeholders.

Interpret honestly. With one annotator and 36 scenes the per-tier cells are tiny;
say so. If `human − auto` is negative, that is a real and publishable finding
about the synthetic-to-human gap, not a failure to explain away. If the auto
augmentation ranges turn out too tight, give the recommended replacement as the
literal `rng.integers(...)` / `rng.uniform(...)` call to paste into the builders.

## Conventions (from `CLAUDE.md`)

- The project is **Sketch-Prompted VLA**. Never write "DrawVLA".
- Author line is **`Aaron`** — nothing else.
- **First person singular, I / my.** Never "we", "our", "us". Passive and
  impersonal preferred for method; "I" for claims and decisions.
- Confirm today's date rather than assuming it.

## How to work

Vertical slice → smoke → full run, as throughout this project.

Long runs must be detached (`setsid nohup ... &` or `tmux`) so they survive the
session ending; a 36-scene three-condition run is short, but do not leave a
multi-hour job as a child of this session.

Deliver what is specified, at the scope specified. Make routine judgment calls
yourself; check in only where two readings would lead to materially different
work. If something here is mistaken, say so in a sentence and continue as asked.

One sentence before your first tool call, brief updates only on findings or
changes of direction, and lead with the outcome when you finish. No subagents.
Match written-file length to what the task needs.
