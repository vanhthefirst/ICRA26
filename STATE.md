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

## Open, in priority order

1. **Run the gate probe** (`scripts/probe_sketch_gates.py`) on `checkpoint-29999`
   or the `pcla_v3` step-1500 checkpoint. CPU, minutes. Decides whether the
   sketch encoder or the gating is at fault — the question three retrains have
   not answered.
2. **Only then** choose a remedy: a dead encoder needs fixing upstream; a live
   encoder behind a stuck gate justifies initialising the gate open or adding an
   auxiliary loss (predict circled-object position from sketch tokens).
3. **Before any further training:** re-export both corpora upright
   (`scripts/reexport_rlds.sh`). Re-exporting invalidates checkpoints trained on
   the old orientation — coordinate before re-uploading.
4. **Deferred:** anchored arms across Object and Goal, for the full-set numbers.

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
