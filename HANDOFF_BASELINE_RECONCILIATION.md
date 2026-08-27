# Handoff — reconciling my baseline with the colleague's 0%

Aaron, 27 August 2026. Companion to `RUNBOOK_BASELINE_PARITY.md` (the how).
This file is the *where I am*. Nothing has been committed and no GPU run has
happened yet — everything below is desk work, waiting on a GPU.

---

## Bottom line

**My baseline is sound. His 0% was never a measurement of the same thing.**

He serves a LoRA fine-tune of pi0.5-**base**, trained on a corpus whose captions
are all deictic (`"do this"`, `"grab this"`), with a sketch pathway that his own
diagnosis says never came online. The language carries no object information and
the sketch is ignored, so the policy cannot know what to pick. Near-zero is the
correct output of that model. My number is stock pi0.5-**LIBERO** on explicit
captions. Two different systems; the gap was never a bug to close.

Nothing needs fixing in how I built the baseline. Several things needed fixing
in his harness before his checkpoint can be evaluated fairly, and they are done.

## What I got wrong mid-session, and corrected

I first concluded his fine-tune had trained on my validation export, whose action
field is identically zero, and that this produced the 0%. **Retracted.** That is
true of my export (74 episodes, one step each, `max |action| = 0.0`) but he
trains on `ductaingn/sketch_libero_rlds` — 450 train + 50 val episodes, 87-197
steps each, `max |action| = 0.938`. I audited the real bytes off HuggingFace.
`RUNBOOK_BASELINE_PARITY.md` §1 carries the retraction.

## Verified facts (audited, not inferred)

| fact | evidence |
|---|---|
| My pipeline is certified | `openpi_repro_500`: 98.0 / 98.6 / 98.0 / 93.4, 2,000 episodes, avg 97.0 vs 96.85 published |
| A rotation fault floors a run, never ceilings it | `pi05_smoke_norot` 0% / `grasped_any 0.00` vs `pi05_smoke` 33% / `grasped_any 1.00` |
| The flat displacement curve is real, not an artifact | `grasped_any 1.00`, `grasped_correct 1.00`, 99.32% over 294 rollouts, `rotate180: true` |
| His training corpus has real actions | 450+50 episodes, 87/129/197 steps, `max abs 0.938` |
| Every caption in it is object-free | 12 distinct wordings / 25 val episodes, none naming an object |
| His corpus is upside-down for pi0.5 | 0/25 in the `modified_libero_rlds` orientation |
| **My export is upside-down too** | 0/74 before the fix, 74/74 after, coordinate drift 0.33 px worst |
| His eval had the 180 rotation commented out | `main_sketchvla.py`, from the file's first commit — and correctly so, given his data |

## Changes made — all uncommitted

**My repo** (`sketch_prompted_vla`):

- `scripts/export_rlds_frames.py` — rotates image, wrist, all three masks and the
  `circle_meta` / `arrow_start` / `arrow_end` coordinates together, once, after
  the pin check. `--no-rotate180` reproduces the old output.
- `scripts/export_rlds_pack.py` — `--verify` now fails on an inverted frame and
  on a `circle_meta` adrift from its mask. Both were silent.
- `scripts/audit_rlds_export.py` — **new.** Audits any RLDS for action signal and
  orientation without TensorFlow. Runs anywhere.
- `scripts/run_parity_baseline.sh` — **new.** Stock pi0.5-LIBERO through his
  harness, to check the harness.
- `scripts/run_sketchvla_eval.sh` — **new.** His checkpoint with the sketch on,
  against my ambiguous arm.
- `scripts/reexport_rlds.sh` — **new.** Re-export in the correct orientation.
- `SUITE_FACTS.md` §1.1 — cross-reference marking that rule projection-path-only.
  Reading it as "never rotate" is what shipped the export inverted.
- `RUNBOOK_BASELINE_PARITY.md` — **new.** The full reconciliation.

**His repo**, as `outputs/sketchvla_baseline_parity.patch` (verified to apply and
compile on a clean checkout):

- rotation restored behind `--args.rotate-180`, default on
- `--args.sketch-mode none` for a genuine text-only baseline
- `--args.success-window` for my sustained-5 criterion
- `--args.max-steps-override`, moved before the suite table (which raises on
  `sketch_*`)
- `--args.sketch-export-dir` to use my `register_sketch_suites.py` and its
  per-task reset seeds
- sketch keying prefers `episode_metadata/scene` — **this one was a silent
  scene/sketch mis-pairing**, see below
- `initial_states[episode_idx]` IndexError fixed; `.pruned_init` filename
  resolved against the filesystem (mine carry a `_demo` suffix)
- `convert_libero_data_to_lerobot.py` refuses a zero-action training set

## The trap worth remembering

`load_sketches_from_dataset` keyed sketches by `episode_metadata/episode_key`.
His corpus has that field; my export does not (it keys by `scene`). The lookup
fell through to `language_instruction`, and my 37 anchored scenes share about six
captions — so every scene would have been handed another scene's drawing. The run
completes and prints a plausible number. **Always read the `Successfully cached N
sketches for M tasks` line; N must equal the scene count.**

## SUPERSEDED — the A/B/C plan and the main_sketchvla.py patch

Both are dead. Written before I looked at the `feat/eval-harness` branch.

**The experiment was already run.** `examples/libero/eval_sketchvla.py` on that
branch is a purpose-built replacement for `main_sketchvla.py`, and
`outputs/rollouts/sketchvla_pcla_*` holds four completed arms against
`checkpoint-29999` — 2,128 rollouts with a matched sketch-withheld control:

| arm | n | success | 95% CI | grasped_any | target_grasped |
|---|---|---|---|---|---|
| explicit + sketch | 532 | 2.44% | [1.43, 4.14] | 52.8% | 20.1% |
| explicit + blank | 532 | 2.82% | [1.72, 4.60] | 49.6% | 19.2% |
| ambiguous + sketch | 532 | 2.63% | [1.57, 4.37] | 52.6% | 17.7% |
| ambiguous + blank | 532 | 2.63% | [1.57, 4.37] | 55.5% | 17.7% |

**The confidence intervals overlap completely.** The sketch changes nothing, and
neither does the caption. The model is not frozen — it grasps something half the
time and the right thing 20% of the time — and it succeeds 18.6% on the control
tier but **0.0% on referential, directional and both**. Against pi0.5-LIBERO on
the same suite: 40.3% explicit, 36.5% ambiguous.

`eval_sketchvla.py`'s own docstring names the defects I had independently
rediscovered in `main_sketchvla.py`, including "looks sketches up in the TRAINING
TFDS dataset, where none of the 114 validation scenes appear, falling through to
blank masks with no error". `outputs/sketchvla_baseline_parity.patch`,
`scripts/run_parity_baseline.sh` and `scripts/run_sketchvla_eval.sh` are aimed at
a script that was already retired. Keep them only as a record; do not run them.

## Next — v3, the SigLIP freeze

The live plan is `docs/HANDOFF.md` §"v3" on `feat/eval-harness`, not this file.
`pcla_v2` failed because the sketch pathway never carried signal, and the last
unexcluded cause is that `get_freeze_filter` froze only `.*llm.*`, leaving the
414.8M-param SigLIP tower training at 5e-5.

Applied on the pod (backup at `/workspace/pi0_types.py.bak`):

    # src/sketchvla/utils/pi0_types.py:463
    gemma_params_filter = nnx_utils.PathRegex(".*(llm|img).*")

Run it as a **5,000-step diagnostic**, not 30k — `sketch_l2` answers by step 1000
and is unambiguous by 5000. `save_interval=500`. Decision criterion: `sketch_l2`
must rise clearly above the ~0.006 floor; 0.03611 is the reference for "the
pathway is carrying signal". Ignore `language_l2`.

    source /workspace/env.sh
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 nohup uv run src/sketchvla/fine_tune.py \
      sketchvla=prompt_conditioned_latent_action \
      fine_tune.exp_name=pcla_v3 \
      fine_tune.rlds_data_dir=/workspace/data/sketch_libero_rlds \
      fine_tune.num_train_steps=5000 fine_tune.save_interval=500 \
      > /workspace/logs/pcla_v3.log 2>&1 &

`source /workspace/env.sh` is mandatory (WANDB_API_KEY), and
`fine_tune.rlds_data_dir` is mandatory — without it the LeRobot path runs and
dies on the `ductaingn/sketch_libero` 404. No LeRobot build is needed at all.

**Deliberately NOT paired with the orientation fix.** `sketch_l2` measures
whether the pathway carries signal, not task success, and image and sketch are
inverted consistently — so orientation should not gate pathway aliveness.
Pairing would spend the single-variable attribution for no gain on the question
being asked. Fix orientation before the next full 30k run.

**Deferred:** re-export both corpora in the corrected orientation
(`scripts/reexport_rlds.sh`; his `sketch_libero_rlds` is 500/500 inverted too).
Re-exporting invalidates checkpoints trained on the old orientation.

## GPU to book

One Ampere card — A40 48 GB, A6000 or A100. Not eight; the 8x A40 in
`RUNBOOK_EVAL_POD.md` was for parallel servers. ~24 GB VRAM, ~50 GB disk, ~1.5 h
per 500-trial run (measured: 19,608 inference calls at 87 ms).

`examples/libero/requirements.txt` pins `torch==1.11.0+cu113` — kernels to
`sm_86` and no further, so 4090 / L40S (`sm_89`) and H100 (`sm_90`) are out.
Caveat I could not resolve: `openpi_reference.json` logs the 2,000-episode
reproduction as having run on a 4090. Either the pin does not bite for this
workload or that entry is wrong. Book the A40. If only Ada/Hopper is free it is
still worth trying — that pin fails as a loud CUDA error at import, not a quietly
wrong number.

`tensorflow_datasets` must be in the **libero client venv** for step 3.
