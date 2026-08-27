# Runbook — reconciling the two pi0.5 baselines

Aaron, 26 Aug 2026.

My text-only pi0.5 baseline over the 114 validation scenes is **34.5%**
(`outputs/rollouts/pi05_baseline/`, 342 rollouts). The parallel run in
`ductaingn/SketchPromptVLA-Pi@main` returned roughly **0%**. This runbook records
why, and how to make the second number reproduce the first.

The short version: the gap is not a rollout-harness bug on either side. The two
runs were not evaluating the same thing. Mine serves the stock pi0.5-LIBERO
checkpoint; his serves a LoRA fine-tune of pi0.5 **base** trained on my
validation export, whose action field is identically zero.

## 1. What the evidence says

**My harness is certified.** `outputs/rollouts/pi05_baseline/openpi_reference.json`
records openpi's own unmodified eval loop at the published trial count: 98.0 /
98.6 / 98.0 / 93.4 across the four standard suites, 2,000 episodes, average 97.0
against 96.85 published. The observation pipeline is not the variable.

**A rotation fault cannot produce either symptom.** `pi05_smoke_norot`
(`rotate180: false`) scores 0% with `grasped_any 0.00` — the arm never reaches an
object. `pi05_smoke` (`rotate180: true`) scores 33% with `grasped_any 1.00`. The
displacement probe sits at the opposite extreme: `grasped_any 1.00`,
`grasped_correct 1.00`, 99.32% over 294 rollouts. A wrong rotation pins a run to
the floor; it cannot hold one at the ceiling, so it does not explain the flat
displacement curve either.

**RETRACTED — the zero-action theory.** I first concluded that his fine-tune
had been trained on my validation export, whose action field is identically
zero, and that this alone produced the 0%. That is wrong. It held for MY export
(74 episodes, one step each, `max |action| = 0.0`, `has_demonstration` False
throughout — still true, and still a reason nothing may train on it), but it is
not what he trained on. Audited on the real corpus,
`huggingface.co/datasets/ductaingn/sketch_libero_rlds` (public, 450 train + 50
val episodes):

    steps per episode                 87 / 129 / 197   (min / mean / max)
    episodes with all-zero actions    0 / 25
    global max |action|               0.938

Real trajectories, real actions. The action signal is fine.

**What actually explains the 0%.** Every caption in that corpus is deictic and
object-free — 12 distinct wordings across the 25 val episodes, `"do this"` (x7),
`"grab this"` (x4), `"put this over there"` (x3), and no wording anywhere names
an object. Set that beside the finding already recorded on my own
`feat/eval-harness` branch in `docs/HANDOFF.md`: `pcla_v2`'s `sketch_l2` is flat
at ~0.006 across steps 1000-5000, an order of magnitude below the abort band —
"the model trains well and ignores the sketch".

So the language carries no task information and the sketch pathway never came
online. The policy is being asked to pick an unspecified object out of a
cluttered scene. Near-zero is the correct output of that model, not a fault in
his eval loop.

This also means **his number was never a parity target for mine.** Different
checkpoint (LoRA on pi0.5-base vs stock pi0.5-LIBERO), different training
corpus, different caption register. The comparison worth making is his
sketch-conditioned run against my AMBIGUOUS arm — see §8.

## 2. A real orientation bug, which is mine and is separate

The export was written in raw robosuite orientation — arm at the bottom of the
frame — while `modified_libero_rlds`, the data pi0.5-LIBERO was fine-tuned on,
sits 180 degrees from that. Neither `convert_libero_data_to_lerobot.py` nor
`LeRobotSketchLiberoDataConfig` rotates, so my orientation propagated all the way
to the model. Measured over all 74 episodes: 0/74 in the orientation pi0.5
expects before the fix, 74/74 after.

SUITE_FACTS.md §1.1 ("already correctly oriented — do not flip it") governs the
PROJECTION path, where a pixel must line up with `frame0.png`. It does not govern
what a model is fed. Both facts hold at once; the note already stands in
`scripts/pi05_policy.py`. The export is a model-facing artefact and was filed
under the wrong one of the two.

Fixed in `scripts/export_rlds_frames.py`: image, wrist, all three masks and the
`circle_meta` / `arrow_start` / `arrow_end` coordinates now rotate together, at
one point, after the pin check. `--no-rotate180` reproduces the old output.
`export_rlds_pack.py --verify` now fails on an inverted frame and on a
`circle_meta` that has parted company with its mask (checked at 0.33 px worst
drift across the 74).

This is worth fixing so a future training run inherits pi0.5's pretraining
instead of fighting it. It is **not** what caused the 0%, and fixing it alone
moves that run from 0% to 0%.

## 3. The parity run

Re-exporting is not required for this: a text-only baseline reads no sketch and
touches no RLDS.

Patch: `outputs/sketchvla_baseline_parity.patch`, against
`ductaingn/SketchPromptVLA-Pi@main` (`f48b23a`). Apply on a clean checkout —
a Windows/DrvFs clone shows every file as modified through CRLF alone, so
generate any further diffs with `| sed 's/\r$//'`.

    git apply sketchvla_baseline_parity.patch

**Terminal 1 — the stock checkpoint, not the fine-tune.** This is the whole
point: 34.5% is a pi0.5-LIBERO number.

    uv run scripts/serve_policy.py --env LIBERO

**Terminal 2 — one invocation per suite.**

    python examples/libero/main_sketchvla.py \
        --args.task-suite-name sketch_spatial \
        --args.sketch-export-dir /path/to/sketch_prompted_vla/outputs/libero_export \
        --args.sketch-mode none \
        --args.num-trials-per-task 3 \
        --args.num-steps-wait 0 \
        --args.max-steps-override 320 \
        --args.success-window 5 \
        --args.replan-steps 5 \
        --args.resize-size 224

Repeat with `sketch_object` and `sketch_goal`. The suites must be installed
first — copy `outputs/libero_export/bddl_files/<pf>` and `init_files/<pf>` into
the trees `get_libero_path("bddl_files")` and `get_libero_path("init_states")`
return, per `outputs/libero_export/README.md`.

Each flag is one of my run's recorded parameters
(`outputs/rollouts/pi05_baseline/run_config.json`):

| flag | value | why |
|---|---|---|
| `sketch-mode` | `none` | text-only baseline; no sketch keys are sent, so the stock LIBERO server accepts the observation |
| `rotate-180` | `true` (default) | pi0.5-LIBERO expects the `modified_libero_rlds` orientation |
| `num-steps-wait` | `0` | my init states were pinned AFTER settling; 10 dummy steps would start each episode somewhere else |
| `max-steps-override` | `320` | my per-episode budget; the suite table has no entry for `sketch_*` |
| `success-window` | `5` | my harness requires the goal predicate to hold for 5 consecutive steps, so a placement that rolls off is not banked |
| `num-trials-per-task` | `3` | 38 scenes x 3 = 114 rollouts per suite, 342 total |
| `replan-steps` / `resize-size` | `5` / `224` | openpi reference values, unchanged |

`--args.sketch-export-dir` also brings in the per-task reset seeds. Some Goal
arenas draw material state during `reset()` from the global numpy stream, so
without seeding there the physics is right and the render is wrong — measured at
158/255 max channel difference on `goal/scene_0000`.

## 4. What to expect

From `outputs/rollouts/pi05_baseline/results.csv`, condition `text_only`:

| suite | n | success |
|---|---|---|
| spatial | 114 | 24.6% |
| object | 114 | 27.2% |
| goal | 114 | 51.8% |
| **all** | **342** | **34.5%** |

At 114 rollouts per suite the binomial standard error is about 4 points, so a
suite landing within roughly 8 points of its row is agreement. Anything near 0%
means the stock checkpoint is not what is being served.

The eval loop now logs its own parity-relevant configuration on startup:

    Eval config: sketch_mode=none rotate_180=True success_window=5 \
        num_steps_wait=0 replan_steps=5 resize_size=224 max_steps=320

Check that line before reading any success rate off the run.

## 5. Guard against the recurrence

`convert_libero_data_to_lerobot.py` now refuses to emit a training set whose
episodes are all zero-action or all flagged `has_demonstration=False`, and prints
the tally either way:

    Converted 74 episode(s), 74 step(s); 74 episode(s) with all-zero actions,
    74 flagged has_demonstration=False.
    ZeroActionDatasetError: This RLDS is a validation set, not a training set ...

`allow_zero_actions=True` overrides it, for the case where that is genuinely
intended.

## 6. What to run, and where

Three scripts. Each states its environment in its own header.

### Now, locally — no GPU, no setup

    python scripts/audit_rlds_export.py outputs/rlds/sketch_libero_val_spatial_anchored

Runs anywhere (stdlib + numpy + Pillow). It parses the TFRecords by hand rather
than through tfds, because tensorflow is in neither the base nor the `libero`
env. This is the evidence base for §1 and it exits non-zero on a bad export.
Current output on the shipped files: 74/74 zero-action, 0/74 correctly oriented.

Ask him to run the same line against `/data/ductaingn/DrawVLA/output/rlds/spatial`
— the dataset his fine-tune actually consumed. That confirms the diagnosis on his
copy rather than by inference from mine.

### The parity run — needs a GPU

    # Terminal 1 (openpi main venv)
    bash scripts/run_parity_baseline.sh server
    # Terminal 2 (libero client env)
    bash scripts/run_parity_baseline.sh client

Set `SKETCHVLA_REPO` and `SPVLA_REPO` first. The patch must be applied to the
former, and the sketch suites installed into LIBERO's BDDL/init trees.

### Before any future training run — no GPU inference

    SUITE=spatial bash scripts/reexport_rlds.sh

Stage 1 needs the `libero` env; stage 2 needs tensorflow + tensorflow_datasets,
which that env does not have. Not required for the parity run.

## 7. GPU

One GPU, not eight. The 8x A40 in `RUNBOOK_EVAL_POD.md` was for running several
policy servers in parallel to cut wall clock; a single 342-rollout baseline does
not need that.

- **Ampere is the safe choice** — A40 48 GB, A6000, or A100. The reason is in
  `RUNBOOK_EVAL_POD.md` §B: `examples/libero/requirements.txt` pins
  `torch==1.11.0+cu113`, which carries kernels for `sm_86` and nothing newer.
  4090 and L40S are `sm_89`, H100 is `sm_90`.
- **Caveat, stated because it is a real inconsistency in my own records:**
  `openpi_reference.json` logs the 2,000-episode reproduction as having run on an
  RTX 4090. Either the pin does not bite for this workload (the client's torch
  may never touch CUDA — the simulator renders through EGL and the model runs in
  JAX on the server side) or that entry is wrong. I have not resolved it, so A40
  is what I would book.
- **VRAM:** pi0.5 in bfloat16 is comfortable in 24 GB; an A40's 46 GB is ample.
- **Disk:** ~50 GB for the checkpoint (~10 GB), the two venvs (~20-25 GB) and the
  LIBERO assets.
- **Time:** measured from `pi05_baseline/results.csv` — 19,608 inference calls at
  87 ms mean is ~28 minutes of pure inference, ~1.5 h wall clock with simulator
  stepping and model load.

If only an Ada/Hopper instance is available, it is still worth trying: the
failure mode of the `sm_86` pin is a loud CUDA kernel error at import, not a
quietly wrong number, so it will announce itself in the first seconds.

## 8. Evaluating his checkpoint (the comparison that is actually meaningful)

His model is not a second measurement of my baseline; it is a different system
answering the project's real question — *does the sketch recover what an
ambiguous caption destroys?* My ambiguous arm is the floor it has to beat.

    SUITE=sketch_spatial CHECKPOINT_DIR=<his checkpoint> \
        bash scripts/run_sketchvla_eval.sh server     # terminal 1
        bash scripts/run_sketchvla_eval.sh client     # terminal 2

Three flags differ from the parity run in §3, and each one invalidates the
number if it is wrong:

| flag | value | why |
|---|---|---|
| `--args.no-rotate-180` | set | his checkpoint trained on UNROTATED frames — 0/25 episodes of `sketch_libero_rlds` sit in the `modified_libero_rlds` orientation. Rotating feeds it an upside-down world. My patch defaults `rotate_180=True`, which is right for stock pi0.5-LIBERO and wrong for him. |
| `--args.sketch-mode` | `dataset` | his training captions are `"do this"`. Text-only measures nothing. |
| `--args.dataset-dir` | the VALIDATION export | the sketch must belong to the scene being run, not the training corpus |

`tensorflow_datasets` must be installed **in the libero client venv** — the
sketch is read by the rollout loop, not by the server. The script checks and
fails rather than running blank.

**Before trusting the result**, read the `Successfully cached N sketches for M
tasks` line. N must equal the scene count (37 for the anchored Spatial set).

A silent mis-pairing was live here until 27 Aug 2026:
`load_sketches_from_dataset` keyed sketches by `episode_metadata/episode_key`,
which his corpus carries and my export does not. Mine keys by `scene`, so the
lookup fell through to `language_instruction` — and the 37 anchored scenes share
about six captions between them, so every scene would have been handed some
other scene's drawing, with a plausible number at the end of it. Patched to
prefer `scene`, which is exactly the `task.name` the rollout loop looks up.

### The bar

| scene set | trials/arm | explicit | **ambiguous** |
|---|---|---|---|
| 37 anchored, Spatial only (`pi05_anchored_*_518`) | 518 | 40.3% | **36.5%** |
| 114, all three suites (`pi05_*_532`) | 1,596 | 33.8% | **17.0%** |

Both pairs are valid and cover different scene sets — see the project memory
note `prompt-baselines-scene-coverage`. Explicit is the WRONG comparison: those
captions name the object, which his model never saw.
