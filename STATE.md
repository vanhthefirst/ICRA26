# State of play — Sketch-Prompted VLA

Aaron, 27 August 2026.

One page, two threads, and a pointer to the authoritative file for each. Written
because the project runs across two repositories whose handoffs do not know about
each other: training lives in `SketchPromptVLA-Pi`, evaluation lives here, and
asking "which document do I trust" had no good answer.

**This file is an index, not a fourth handoff.** When it disagrees with the file
it points at, the pointed-at file wins.

---

## Thread 1 — the sketch pathway (training)

**Authoritative:** `SketchPromptVLA-Pi@feat/eval-harness:docs/HANDOFF.md`, written
24 Aug 04:56. Still current: only one commit on that branch postdates it
(`a1ce381`, `validate: --asset-id`), and nothing since has touched the diagnosis.

| run | verdict |
|---|---|
| original (`anhdao69/sketchprompt`, 30k) | `tanh(attn_gate)` 1.784e-4 — pathway never opened |
| `pcla_v2` (LR + batch fix) | **failed**; `sketch_l2` flat at ~0.006 through step 5000 while loss fell 0.0975 → 0.0140. The model trains well and ignores the sketch. |
| **`pcla_v3`** | **FAILED, 27 Aug.** SigLIP tower frozen. `sketch_l2` 0.00839 / 0.00678 / 0.00700 at steps 500 / 1000 / 1500 — flat, inside `pcla_v2`'s band, 5x below the 0.03611 signal reference. Stopped at 1500; checkpoint kept. |

**What v3 changes,** and the only thing it changes:

    # src/sketchvla/utils/pi0_types.py:463   (backup: /workspace/pi0_types.py.bak)
    gemma_params_filter = nnx_utils.PathRegex(".*(llm|img).*")

`get_freeze_filter` froze only `.*llm.*`, so the 414.8M-param `PaliGemma/img`
tower trained at the full 5e-5. Verified on the pod by counting the initialised
train state, not by reading the regex:

    PaliGemma/img   414.8 M  bfloat16 (frozen)
    trainable f32    91.9 M            (was 506.7 M; 506.7 - 414.8 = 91.9)

The sketch pathway is untouched by the filter — the pcla model's own modules are
`sketch_encoder`, `attention`, `dense_proj`, `proj`, `cls_token`,
`pos_embedding`, `norm`, none of which contain `img` or `llm`. Freezing the
sketch encoder by accident would have silently destroyed the experiment.

**Form:** 5,000 steps, batch 32, `save_interval=500`, `rlds_data_dir` set (the
LeRobot path is bypassed entirely and needs no build). ~4.7 h on one A100 80 GB.
Log `/workspace/logs/pcla_v3.log`.

**Verdict.** The freeze was a real defect fixed — 414.8M params were training at
5e-5 that should not have been — but it was **not the cause**. That was the last
substantial hypothesis in HANDOFF's list, and it is now excluded. Three retrains
have been spent without measuring where the signal actually dies.

**Decision criterion (unmet):** `sketch_l2` must rise clearly above the ~0.006
floor and keep rising; 0.03611 is the reference for "the pathway is carrying
signal". It answers by step 1000. **Ignore `language_l2`** — its decay is correct on a
dataset whose captions are object-free. Smoke test before launch read
`language L2 0.91237, sketch L2 0.00538` at step 10, which is the expected
starting position.

**Next — instrument, do not retrain.** With `output = base + tanh(g) *
sketch_branch` and `g` initialised at 0, the gate still receives gradient
(`dL/dg = dL/doutput * sech^2(g) * sketch_branch`) **unless `sketch_branch` is
itself ~0**. That fork is minutes on CPU with `scripts/probe_sketch_gates.py`:

  * encoder output ~0 -> `SketchEncoder` produces nothing; the gate is innocent
    and the bug is upstream. "Initialise the gate slightly open" would paper over
    a dead encoder.
  * encoder output healthy, gate still stuck -> gradient dies between them, and
    that is an architecture bug.

The mask-format diff remains **untested rather than excluded** — it is gated on
`|tanh(attn_gate)| > 0.01` and the gate has never opened, so that check has never
run. The delivery side is already cleared: eval masks were re-rendered through
the exact code path (circle ring 1,392 px, arrow 613 px, training palette).

---

## Thread 2 — baselines and probes (evaluation)

**Authoritative:** this repo. `report/prompt_baselines/` and the git log since
24 Aug. `docs/HANDOFF.md` in the other repo predates all of it.

### pi0.5-LIBERO, stock — the reference model

Pipeline certified against openpi's own eval loop: 98.0 / 98.6 / 98.0 / 93.4 over
2,000 episodes, average 97.0 vs 96.85 published.

| run | scenes | trials | explicit | ambiguous |
|---|---|---|---|---|
| `pi05_anchored_*_518` | 37, **Spatial only** | 518 | **40.3%** | **36.5%** |
| `pi05_*_532` | 114, all three suites | 1,596 | **33.8%** | **17.0%** |

Both pairs are valid and cover different scene sets — see the memory note
`prompt-baselines-scene-coverage`. The Spatial-only figures sit higher because
Object and Goal are harder; expect the eventual full-set anchored numbers below
40.3 / 36.5. `pi05_baseline` (342 rows, 34.5%) is the older 3-rollout run,
superseded by `pi05_explicit_532`.

Note how differently ambiguity bites on the two sets: on 114 scenes it halves
performance (control 73.8% → 25.2%); on the 37 anchored Spatial scenes it barely
moves (85.7% → 75.0%).

### Sketch-VLA fine-tune (`checkpoint-29999`), 2x2, 2,128 rollouts

| arm | n | success | 95% CI |
|---|---|---|---|
| explicit + sketch | 532 | 2.44% | [1.43, 4.14] |
| explicit + blank | 532 | 2.82% | [1.72, 4.60] |
| ambiguous + sketch | 532 | 2.63% | [1.57, 4.37] |
| ambiguous + blank | 532 | 2.63% | [1.57, 4.37] |

**The confidence intervals overlap completely.** The sketch changes nothing and
neither does the caption. The model is not frozen — `grasped_any` 52.8%,
`target_grasped` 20.1% — and it scores 18.6% on the control tier but **0.0% on
referential, directional and both**. Run with `examples/libero/eval_sketchvla.py`
on `feat/eval-harness`, which replaced `main_sketchvla.py` for evaluation.

### Position probes — does moving an object break the policy?

The premise being tested was "moving the target object degrades most VLAs".

| probe | scenes | rollouts | result |
|---|---|---|---|
| target moves | 21 | 294 | **flat at 99.3%** out to 12 cm; largest drop 1.02 pts |
| destination moves | 23 | 322 | 95.0% overall, but **100% → 83.3% at 12 cm**; largest drop 14.88 pts between 9 and 12 cm |

So the premise fails on the target side and holds on the destination side. That
asymmetry is a finding, not an artifact: `grasped_any` and `grasped_correct` are
both 1.00 across the target arm, and a preprocessing fault would floor a run
rather than hold it at the ceiling.

---

## Data — one defect spans both repos

Both RLDS corpora are **180 degrees from the orientation pi0.5 expects**
(`openvla/modified_libero_rlds`: arm hanging from the top, lit table below).

| corpus | episodes | actions | orientation |
|---|---|---|---|
| my `sketch_libero_val_spatial_anchored` | 74, single-step | all zero *(by construction)* | 0/74 — **fixed 27 Aug** |
| `hf.co/datasets/ductaingn/sketch_libero_rlds` | 500, 75-197 steps | real, max abs 1.0 | 0/500 — **not fixed** |

The validation export is a scoring set and must never be trained on; its zero
actions are correct. The training corpus is fine as data and only wrong way up.

`scripts/export_rlds_frames.py` now rotates image, wrist, all three masks and the
`circle_meta` / `arrow_start` / `arrow_end` coordinates together;
`export_rlds_pack.py --verify` fails on an inverted frame. `SUITE_FACTS.md` §1.1
now says explicitly that its "do not flip" rule governs the projection path only.

**Not paired with v3 on purpose.** `sketch_l2` measures whether the pathway
carries signal, and image and sketch are inverted consistently, so orientation
should not gate pathway aliveness. Pairing would spend the single-variable
attribution for no gain. Fix it before the next full 30k run.

---

## 28 Aug — the gate is open, the sketch is still ignored, and we know why

`pcla_v4_gate` (1500 steps, A100, one variable against v3: `attn_gate_init`
0.0 -> 0.1) **failed the criterion, and in failing it closed the thread.**

| step | v4 `sketch_l2` | v3 `sketch_l2` | v4 `tanh(attn_gate)` |
|---|---|---|---|
| 500 | 0.00714 | 0.00839 | 0.0979 |
| 1000 | 0.00665 | 0.00678 | 0.0918 |
| 1499 | 0.00578 | 0.00700 | 0.0864 |

The gate change worked *mechanically* and changed nothing about the outcome.
Measured on the block: at `attn_gate_init=0.0` the gradient into `attn/to_kv` --
the only path the sketch enters by -- is **identically 0.0**, not merely small,
so the sketch encoder had never received a learning signal in any run. At 0.1 it
is 1.97e-4. The gate then held open all 1500 steps (`probe_sketch_gates` on the
checkpoint: `tanh = +0.084823`, vs +1.784e-4 for the original 30k run).

**Struck: the auxiliary loss.** It was for the case where the optimiser slams the
gate shut. Per-step `gate/attn_tanh` logging shows that never happens -- 0.0997
-> 0.0864 is a slow drift. It would answer a question the data has closed.

**Struck: the mask-format diff — now excluded, not untested.** It was gated on
`|tanh(attn_gate)| > 0.01`, which no run had ever satisfied. At 0.085 it ran: a
real-mask ablation on held-out `val` agrees with the synthetic-ring probe
(delta arm L1 **-0.00001**, SKETCH IGNORED). `sketch_l2` was right all along.

**Root cause — the sketch is redundant with the IMAGE.** Measured on the bytes of
`sketch_libero_rlds`:

  * captions are deictic and object-free (`'do this'`, `'grab this'`,
    `'put this over there'`) -- so language cannot identify the target. STATE's
    earlier reading was right; `RUNBOOK_RETRAIN_PROBE`'s cause #2 is superseded
    *on the caption half only*.
  * 450 episodes = **10 LIBERO-Spatial scenes x 45 demos**. Within a scene the
    circle centre occupies **1-4 cells** at 16 px quantisation across all 45
    episodes -- it marks the same object every time, moving only with pose
    randomisation.

So **image -> scene -> target**, and the sketch adds nothing the image lacks.
The *training loss* can therefore be minimised without ever reading the sketch,
which is exactly what every run did. (Careful with the ablation's arm L1 of
0.099: that is chunk-regression error, **not** task competence -- the same
model scores 2.4% success in the 2x2. Low L1 here means "plausible-looking
motion", and it is consistent with a policy that solves nothing.) It is also why
the 2x2's arms were identical and referential/directional scored 0.0%: the
corpus never required the skill those tiers test. **No architectural change
fixes this.** The next move is data -- scenes with several candidate objects
where the sketch is the only disambiguator -- not another retrain.

**Tooling defect found and fixed** (`SketchPromptVLA-Pi@16a28b3`):
`validate.py --ablate-sketch` never pinned the flow-matching noise.
`SketchVLAPolicy.infer` splits its RNG per call, so the with- and without-sketch
passes gave each sample different noise and the delta was sampler variance.
Caught by running the ablation on **v3 as a control**: its gate is an exact
arithmetic no-op, yet it reported `delta +0.00085, SKETCH HELPS`. `verdict`
thresholds on `delta > 0`, so it would call SKETCH HELPS on any checkpoint whose
noise landed positive. **Any earlier conclusion drawn from `--ablate-sketch` is
contaminated.** With noise pinned, v3 reports `-0.00000` and v4 `-0.00001`.

---

## Open, in priority order

1. ~~Run the gate probe.~~ **DONE 27 Aug — the gate is the fault, not the
   encoder.** `pcla_v3` @1500: `attn_gate -0.000184`, `ff_gate +0.000022`;
   original 30k run: `+0.000178`. `tanh(g) ~ 0` makes the block an exact no-op,
   which is why the 2x2's sketch and blank arms were statistically identical.
   The encoder is healthy: 29.26M float params, rms 0.045, max 1.313, no NaN.
   (The probe's `l2=nan` and `l2=2.6e9` were artifacts — `None` biases and a
   uint32 PRNG key; fixed in `probe_sketch_gates.py`, uncommitted.)
2. ~~Change the gate initialisation.~~ **DONE 28 Aug, and it failed the
   criterion — see the section above.** `attn_gate_init: 0.1` is now in
   `conf/sketchvla/prompt_conditioned_latent_action.yaml`; keep it, since the
   zero-gradient finding means 0.0 was never defensible, but do not expect it to
   buy anything on this corpus. The auxiliary loss is struck.

2b. **NEXT — make Spatial teachable; do not switch suites.** The project
   constraint is that the colleague's work covers Spatial only, so results must
   stay on Spatial to remain comparable with the anchored baselines
   (40.3% / 36.5% over 518 trials), the 2x2, the position probes and
   `checkpoint-29999`. An earlier draft of this section recommended switching
   the corpus to LIBERO-Goal; that is **withdrawn** — it rested on an unverified
   assumption about Spatial's layouts and under-weighted losing comparability.

   `check_sketch_necessity.py` on the shipped suites:

   | suite | layouts | sketches per layout |
   |---|---|---|
   | libero_spatial | 9 | {1: 8, 2: 1} |
   | libero_goal | 1 (8 usable tasks) | {8: 1} — admissible |
   | libero_object | 10 | {1: 10} |
   | libero_10 | 9 | {1: 8, 2: 1} |

   **No object needs injecting — measured 28 Aug.** All ten shipped tasks
   place the fixtures and clutter identically (plate, cookies, ramekin,
   cabinet, stove: 10/10 files each); a scene is fully determined by the SET
   of regions the two identical black bowls occupy. The stove/cabinet pair is
   ambiguous for exactly that reason — both put bowls at
   `{flat_stove_1_cook_region, wooden_cabinet_1_top_side}` and swap which
   instance is the target. The template generalises: for each task pair, move
   each task's DISTRACTOR bowl to the partner's target region. Demos and
   stored sim-state dimensions are untouched; replay needs only a 7-DoF pose
   override on bowl_2's free joint.

   `scripts/pair_spatial_bddls.py` implements it — 6 one-line `(:init)` edits
   across 5 pairs — and `check_sketch_necessity` on the output
   (`outputs/paired_spatial_bddl/`) reads **ADMISSIBLE, 5/5 layouts x 2
   sketches**. The drawer pair (`in_top_drawer` <-> `next_to_cookie_box`) is
   provisional: both scenes show one visible bowl and a closed cabinet, so
   the sketch must be able to mark the drawer face.

   **Replay feasibility — measured, 50 shipped demos, worst-case edit.**
   Task `next_to_ramekin` with bowl_2 moved to `next_to_plate_region` (11 cm
   from the destination plate, in the carry path — the highest collision-risk
   placement of the six). CPU MuJoCo 2.3.7 + robosuite 1.4.0, open-loop from
   `states[0]`:

   | arm | success |
   |---|---|
   | control (unmodified replay) | 45/50 |
   | bowl_2 moved | 35/50 — **35/45 = 78% of replayable demos** |

   Moved-arm failures correlate with large bowl_2 displacement during the
   episode (0.13–1.59 m — the trajectory clips it), so the corpus builder
   must **replay each episode and keep only those whose goal still succeeds**.
   Every kept episode is then a verified-consistent (image, sketch, action)
   triple; the worst pair yields ~35 of 45 usable demos, and the other pairs'
   placements are farther from the carry path. The 5 control failures are
   open-loop replay nondeterminism (mujoco-version sensitivity), the known
   reason LIBERO ships regenerated datasets. Goal's single-layout analysis
   (`outputs/probe_goal.txt`) is kept as a fallback.

3. **Gate every future corpus on `scripts/check_sketch_necessity.py`.** It parses
   BDDL files with stdlib only -- no environment, no GPU, no conversion -- and
   groups scenes by layout (`(:objects)` categories + `(:init)`) against sketch
   (`(:goal)` target, dest). A layout carrying one sketch teaches the model that
   the image alone determines the action. Run it *before* building a corpus;
   running it before `sketch_libero_rlds` would have saved three retrains and an
   architecture change. Two of its own bugs were found in use and are fixed:
   the layout key originally used instance names (hiding the stove/cabinet pair
   entirely), and the `(:objects)` parse assumed one instance per line, silently
   dropping `akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl`.

4. **Before any further training:** re-export both corpora upright
   (`scripts/reexport_rlds.sh`). Re-exporting invalidates checkpoints trained on
   the old orientation — coordinate before re-uploading. One cheap check to run
   alongside: since pcla_v3 the SigLIP tower is frozen, so a frozen pretrained
   encoder is being fed upside-down scenes — that could depress absolute success
   (the ~2.4%) while explaining nothing about the referential-tier 0%. Eval one
   existing checkpoint on corrected vs. inverted frames to measure it.
5. **Deferred:** anchored arms across Object and Goal, for the full-set numbers.

## Retracted, so it is not re-derived

- *"His fine-tune trained on my zero-action validation export."* False. It trains
  on the 450-episode corpus with real actions. Audited on the bytes.
- *"The colleague's ~0% is a harness bug."* No. Object-free captions
  (`"do this"`, `"grab this"`) plus a sketch pathway that never opened. Near-zero
  is the correct output of that model.
- *"A parity run would reconcile the two numbers."* They were never measuring the
  same thing. `outputs/sketchvla_baseline_parity.patch`,
  `scripts/run_parity_baseline.sh` and `scripts/run_sketchvla_eval.sh` target a
  script already retired by `eval_sketchvla.py`. Kept as record; do not run.
