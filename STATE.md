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

## 1 Sep — v7 implemented: a grounding objective, a counterfactual, and the two blockers closed

Code only; nothing has been run. Authoritative:
`SketchPromptVLA-Pi@feat/eval-harness:docs/RUNBOOK_V7_REFERENT_GROUNDING.md`.

**Blocker A (provenance) is closed in code.** `sketchvla/provenance.py` and its
twin `scripts/provenance.py` stamp every result file with both repos' shas, a
`tree_digest` over the modified and untracked files (so two hand-shipped trees at
one sha are still distinguishable), and any stashes present.
`make_pod_payload.sh` writes a `VERSION` into `/workspace/harness_repo`, which
was never a git repository. `scripts/pod_provenance_setup.sh` gives a pod a
working `git fetch` and archives the three stale stashes instead of dropping
them.

**Blocker B has a probe.** `scripts/probe_relocation_floor.py`: stock
`pi05_libero`, explicit captions, three arms — shipped layout, distractor
relocated (the exact intervention that builds the paired corpus, so its success
rate is the CEILING for any model on it), and target relocated. Not the same
question as the displacement probe, which removed the distractor and translated
in xy; this imports the builder's own donor-pose teleport.

**A defect found while implementing, and it would have broken v7 silently.**
`preprocess_observation` augmented the frame and the sketch masks with
INDEPENDENT crops and rotations: `augmax.Chain` splits its key across its
transforms and the RGB chain carries a fourth (`ColorJitter`) the mask chain does
not, so `split(rng, 4)` and `split(rng, 3)` gave `RandomCrop` and `Rotate`
different subkeys. Relative displacement on the order of the ring's own radius,
every step, since the first run. Invisible to `pcla` (mean-pools its sketch
tokens) and to `input_overlay` (one pre-composited frame); fatal to anything that
binds the mark to a position. Fixed by stacking the frame and every sketch plane
on the channel axis and applying one chain with one key.
`scripts/check_aug_alignment.py` is the regression test.

**v7 `referent_grounding`.** The measurements say the objective, not the wiring,
starved the pathway: the sketch's share of a loss averaged over ~140 frames is
under one percent, so the optimiser shrinking it is correct behaviour and will
happen wherever it is attached. So v7 gives the sketch its own dense objective —
a per-patch cross entropy against the corpus's visibility mask for the circled
object — plus a counterfactual: a quarter of the batch is served with the marks
round the DISTRACTOR, fitted against the distractor's mask, carrying no action
loss. Without it a pointer scores perfectly by learning "the bowl I was going to
take anyway", which describes every result to date. The grounded region then
enters as two PREFIX tokens rather than a vector added after the whole PaliGemma
forward.

The counterfactual needs labels the corpus does not carry, so
`build_paired_corpus.py` now emits `circle_swap`, `arrow_swap`, `distractor`,
`distractor_meta` and the swap arrow endpoints — same renderer, same stroke, same
frame — and `pack_paired_corpus.py --require-counterfactual` declares them into
`features.json`. **The corpus must be rebuilt into a new directory before v7 can
run.** The old one stays as it is: v5 and v6 must remain reproducible.

**`sketch_l2` is not replaced by another magnitude.** The gates are
`grounding/point_hit_real` against `grounding/point_hit_swap`, `follow_ratio`,
and `gate/swap_delta` against `gate/blank_delta` — each a pair, none readable
alone. `scripts/probe_grounding.py` gives the same shape of answer offline in
minutes, so a checkpoint that fails it never earns a 518-rollout eval.

**Reporting is now enforced.** `scripts/score_referent_following.py` takes a real
and a swap run directory, refuses to summarise one, and reports the scene-paired
effect of moving the circle on the grasp rate of every object, with a verdict of
grounded / disturbance / null. Run on `overlay_v6`'s two arms it reproduces this
file's numbers — bowl_1 -15.96 pts, bowl_2 (circled in swap) +3.11 at 0.65 sigma,
bowl_3 (uncircled) +7.64 at 2.25 sigma — and returns **null**, which is the check
that the metric detects the failure it was written for.

---

## 1 Sep — input_overlay moves the needle and still does not ground the sketch

Full account in `SketchPromptVLA-Pi@feat/eval-harness:docs/SESSION_2026-09-01.md`. `overlay_v6_paired`: the sketch drawn
into the frame the frozen SigLIP reads (`base_0_rgb` <- `sketch_overlay`), same
paired corpus, batch 32, same LR schedule, 3000 steps, 2 h 23 m on one A100,
loss 0.0697 -> 0.0120. At matched step 1400 it is 0.0170 against v5's 0.0167, so
the variants fit equally well and the difference is in what was fitted.

Spatial, ambiguous captions, 37 scenes x 14 rollouts per arm, `--rotate180`.
Percentages are of grasps; v5's rows reproduce its published table under the
same code.

| arm | success | grasped | bowl_1 (goal) | bowl_2 (distractor) | bowl_3 |
|---|---|---|---|---|---|
| v5 pcla, real *(circle on bowl_1)* | 38.6% | 95.6% | 59.0% | 15.2% | 11.1% |
| v5 pcla, swap *(circle on bowl_2)* | 39.6% | 94.0% | 59.8% | 14.2% | 11.7% |
| v6 overlay, real | 40.3% | 89.6% | **78.7%** | 10.1% | 5.6% |
| v6 overlay, swap | 35.5% | 88.0% | **63.2%** | 13.4% | 13.4% |

**The sketch is finally causal and still not referential.** Moving the circle off
the goal bowl costs it 15.5 points (78.7 -> 63.2, ~5 sigma) where v5's equivalent
was +0.8. But the freed mass does not go to the circled object: bowl_2 gains 3.3
points (~1.6 sigma, marginal) while the uncircled, irrelevant bowl_3 gains 7.8
(~4 sigma). A pointer would drive the circled object up sharply. This drives an
unrelated one up harder — the signature of a marker that degrades the scene
representation rather than designating a referent. Offline, the frame-0 ablation
delta is +0.03652 against v5's +0.00044, but for this variant emptying the sketch
changes `base_0_rgb` itself, which the model never saw unmarked, so some of that
is distribution shock.

**The reporting hazard.** The real arm alone reads as a breakthrough (+19.7
points on taking the goal bowl) and is not one. `sketch_l2` was the first number
in this project that moved for the wrong reason; this is the second. Never report
the real arm without the swap arm.

---

## 31 Aug — the referent is not lost anywhere; it arrives and is unusable

Three probes, run against `pcla_v5_paired/1499` on the CA-MTL-3 volume. Code:
`SketchPromptVLA-Pi@feat/eval-harness` `scripts/probe_sketch_readout.py` and
`scripts/probe_sketch_attention.py` (commits `265fe94`..`fd77035`). Raw numbers:
`outputs/readout_v5.json`, `outputs/attention_v5.json`,
`outputs/readout_v5_step3.json`.

**1. The sketch encoder keeps the circle.** Ridge from pooled encoder tokens onto
the circle centre, fit on the corpus's 377 train episodes and scored on its 41
val episodes -- disjoint by construction, and the labels are the builder's own
`sketch/circle_meta`, not the pixels.

| read from | R^2 | median err | p90 |
|---|---|---|---|
| the input mask itself (control) | 0.9985 | 1.2 px | 3.9 px |
| mean of all 99 sketch tokens | 0.9785 | 2.1 px | 7.7 px |
| the 49 circle-branch tokens | 0.9802 | 2.8 px | 7.6 px |
| cls token | 0.9815 | 3.0 px | 9.1 px |
| **the vector `modality_attn` adds to every action token** | **0.9737** | **4.0 px** | 8.4 px |

Pixels are at the 256 render, where circle centres spread 67 px in x. The
encoder is within 1.6 px of the raw-mask control. On 2000 synthetic rings at
uniformly random centres the same readouts give 0.911 / 12.5 px for the mean
token and 0.823 / 18.7 px for the added vector, so `to_kv`->`to_out` is a lossier
carrier off the corpus's circle placements -- in-distribution it is not.

Do not read `arrow_mean` as evidence about the arrow. The encoder's two
transformer layers run *after* the branches are concatenated, so the arrow-branch
tokens are not arrow-only: they score 0.935 on the synthetic set, where the arrow
channel is empty.

**2. The cross-attention is uniform.** 41 val scenes, one frame each.

| measure | value | uniform would be |
|---|---|---|
| tanh(attn_gate) | 0.0879 | -- |
| branch share of the residual stream (rms) | 0.090 | -- |
| attention entropy | 6.6288 bits | 6.6294 |
| mass on cls / arrow / circle | 0.0102 / 0.4959 / 0.4939 | 0.0101 / 0.4949 / 0.4949 |
| attention centroid across scenes | x 0.5020 +- 0.0019 | 0.5 |
| true circle centre across scenes | x 0.4718 +- 0.2661 | -- |
| follow ratio | 0.00057 | 0 |

Every scene is at uniform individually (6.6271-6.6293), so this is not an
averaging artefact. Move the ring 40% of the frame across otherwise identical
queries and the attention centroid moves 0.02% of it.

**Not because there is nothing to attend to.** The sketch tokens differ from one
another by more than their own mean magnitude (`media_spread` 1.34), and so do
their keys (1.24). The logits are simply tiny: a random unit query sees a spread
of 0.066 across the 99 keys, and the real logits are near 0.03 (inferred from the
entropy deficit, 0.00059 bits ~ Var/2 in nats -- a derivation, not a
measurement). A softmax over 99 tokens at that logit scale is uniform to a few
percent. The block is doing mean-pool-then-project, and mean pooling happens to
preserve the position linearly.

**3. What the architecture can express, given that.** `modality_attn` is applied
to `suffix_out` -- *after* the whole PaliGemma forward -- and only
`action_out_proj`, one Linear, follows it. With the softmax uniform the branch
output is `to_out(mean_j v_j)`, which is the same vector for every action-token
query and does not depend on the queries at all. So the sketch reaches the action
head as **one vector added identically at every horizon step**, with no spatial
and no temporal structure. It can translate the predicted action chunk; it cannot
change its shape. "Take the left bowl instead of the right one" is a shape change.

**Measured 31 Aug, and it is WRONG.** Serving one observation twice with the
ring 0.40 of the frame apart, on pinned noise, 41 val scenes, arm dimensions
only (`outputs/chunk_structure.json`):

| difference | rms | not-constant-across-horizon fraction |
|---|---|---|
| ring left vs ring right | 0.00202 | 0.326 |
| two captions naming different objects | 0.1235 | 0.426 |

The sketch's effect is *not* a constant offset -- a third of it varies across the
horizon, much like the language pathway's 0.43. The channel can reshape the
chunk. What it cannot do is matter: the language pathway moves the arm actions
**55x further** (median over scenes; never less than 12x, `outputs/chunk_structure.json`),
and ground-truth arm deltas have std ~0.39, so the sketch moves the actions by
about 0.5% of their scale.

So this is a magnitude problem, not an expressivity one, and the architecture is
not the constraint. The gate agrees: `tanh(attn_gate)` went 0.0997 -> 0.0890 over
the run. Gradient descent *shrank* the pathway, which is the correct response to
(4) -- the loss barely rewards it.

**4. And it is not used.** `validate.py --ablate-sketch` on the paired corpus's
own val split, sketch channels emptied, flow-matching noise pinned per sample so
the two passes are comparable:

| frames scored | arm L1 with sketch | without | delta |
|---|---|---|---|
| 4 per episode, 164 samples | 0.067515 | 0.067648 | +0.000133 (0.20%) |
| frame 0 only, 41 samples | 0.108839 | 0.109281 | +0.000442 (0.41%) |

Gripper sign accuracy is identical to four decimals in both (0.9860 / 0.9860 and
1.0000 / 1.0000). Frame 0 is scored separately on purpose: from the second frame
on, proprioception already says which bowl the arm is heading for, so the sketch
is redundant there and an episode-averaged ablation understates it. At frame 0
the sketch is the *only* disambiguator — and deleting it costs 0.4% of the action
error, on the corpus built so that within a layout only the sketch says which
bowl. Raw: `outputs/ablate_paired.json`, `outputs/ablate_frame0.json`.

**Verdict.** Not the encoder, and not the conditioning path: both carry the
referent to the action tokens with room to spare. The signal is delivered at
4.0 px and the model does not act on it. The optimiser never learned to
use it, and (4) says the training loss barely noticed: on the corpus designed to
require the sketch, deleting it costs 0.4% of the action error. The structural explanation --
that the channel cannot express object choice -- was measured and refuted the
same day. It can; it is 55x too weak to matter, and the optimiser shrank it
because the loss did not want it. That moves the fault to the training signal:
the question is no longer what the model can express but why a corpus built to
require the sketch does not, in practice, require it.

Numbers are deterministic: the corpus arm came out bit-identical across two runs.

---

## 30 Aug — the swap test: the sketch does not steer the referent (verdict (a))

The 2x2 null could not tell a sketch-ignoring model from a sketch-following one,
because the circle always marked the scene's target -- which is also the BDDL
goal object and also what the policy's prior reaches for. `sketch_mode=swap`
re-anchors the circle onto a same-type distractor so the two disagree. Ambiguous
captions, 37 scenes x 14, upright frames:

| ambiguous arm | grasps | took circled bowl | took goal bowl | success |
|---|---|---|---|---|
| real (circle on goal bowl) | 495 | 292 (59.0%) | 292 (59.0%) | 38.6% |
| blank (no circle) | 491 | -- | 293 (59.7%) | 38.8% |
| **swap (circle on distractor)** | 487 | **69 (14.2%)** | **291 (59.8%)** | 39.6% |

Paired by scene, goal-bowl rate real - swap: **-1.0 pts, SE 1.2, t=-0.90**. Moving
the circle onto the distractor does not move the grasp. The 14.2% is the rate at
which the policy takes that bowl anyway, not sketch-following.

**The finding, stated precisely.** The sketch is not ignored -- serving the same
observation with the sketch zeroed moves the action chunk (mean |d| 0.047-0.069),
and `sketch_l2` rose 5x against v4. It changes the trajectory without carrying
the referent. Object choice is independent of where the circle is.

`explicit x swap` was not run: under (a) it is predicted-null by construction,
since the sketch fails where it is the ONLY disambiguator and cannot then
override a caption that names the target.

**So `sketch_l2` is not a success criterion.** It measures that the sketch
pathway carries signal, not that the signal is referentially useful. Every
training decision to date was steered by it.

**Next is diagnosis inside the model, not more rollouts.** Two probes localise
the break: (1) can a linear readout of the sketch encoder's tokens recover the
circled object's position -- if not, the referent is discarded upstream of the
action head; (2) does the action expert put attention mass on sketch tokens --
`attn_tanh` held at 0.089 all run, so the gate was open, and the question is
whether anything the action path used flowed through it. Which of the two fails
decides whether the fix is the encoder, the conditioning path, or the objective.

---

## 30 Aug — the 2x2: the sketch changes the actions and not the outcome

`pcla_v5_paired` step 1499, spatial, 37 scenes x 14 rollouts per arm, upright
frames (`--rotate180`), sustained success:

| | real sketch | blank sketch |
|---|---|---|
| explicit | 45.6% (236/518) | 45.4% (235/518) |
| ambiguous | 38.6% (200/518) | 38.8% (201/518) |

Paired by scene: explicit real-blank **+0.2 pts, SE 1.1, t=0.17**; ambiguous
**-0.2 pts, SE 0.9, t=-0.22**. Thirty of 37 ambiguous scenes score *identically*
with and without the sketch. Zeroing the sketch channel changes nothing.

**It is not a plumbing bug.** `sketch_libero_policy` reads all six sketch keys,
and probing the served checkpoint with the same observation twice -- real sketch
vs zeroed -- moves the action chunk by mean |d| 0.047-0.069, max 0.18-0.50 over
three scenes. The model conditions on the sketch. The conditioning just does not
change *which bowl it goes for*.

**It is not a harness limitation.** Every scene's goal is
`(On akita_black_bowl_1 plate_1)` -- a specific instance -- with
`akita_black_bowl_2` present as a distractor, so the wrong bowl fails. The eval
can see referential error; there is none to see, in either direction.

**What the model is doing instead.** Ambiguous+blank at 38.8% is far above the
~23% that guessing between two identical bowls would give (0.5 x explicit's
45.6%). The policy resolves a deictic caption with a positional prior, and the
sketch rides along without steering it.

**So `sketch_l2` was the wrong success criterion.** It measures that the sketch
pathway carries signal -- which is true, and rose 5x against v4 -- not that the
signal is referentially useful. The 1500-step run answered what it could; a
longer run optimises the same quantity and cannot close this gap.

**Next, and it is not more training:** a sketch-SWAP arm. Circle the distractor
`akita_black_bowl_2` and record which bowl is picked. Real-vs-blank can only ask
whether the sketch helps on average; the swap asks whether the sketch *controls
the referent*, and separates "follows sketches, prior agreed anyway" from "does
not follow sketches at all".

---

## 30 Aug — v5 reads the sketch and ignores it: the 2x2 is null

`pcla_v5_paired` (1500 steps, A100 80 GB, corpus `sketch_libero_rlds_paired`,
418 episodes / 377 train + 41 val). **Passed.**

| step | v5 `sketch_l2` | v4 | v3 |
|---|---|---|---|
| 500 | 0.00740 | 0.00714 | 0.00839 |
| 1000 | **0.01914** | 0.00665 | 0.00678 |
| final | **0.02939** | 0.00578 | 0.00700 |

Against the 0.03611 signal reference that is 81%, rising monotonically, where
every previous run sat flat on the ~0.006 floor and drifted *down*. Corroborated
by two independent readings: `gate/attn_tanh` held 0.0997 -> 0.0890 (the gate
never closed), and `gate/language_l2` fell 0.5418 -> 0.2672 over the same steps
-- weight moving off the deictic caption and onto the sketch, which is the
behaviour the paired corpus was built to force. Loss 0.0229 -> 0.0167.
W&B run `8654dz2b`; checkpoint at step 1499.

**Not single-variable, by construction.** Corpus content, world orientation and
the open gate all changed against v4. This is the existence test -- the
architecture *can* learn to use the sketch -- and it does not apportion the gain
among the three. Attribution needs a run on the paired corpus in the old
inverted orientation, and one with the gate at 0.0.

**Eval must flip.** v5 trained upright, so `eval_sketchvla.py` needs
`--rotate180` (added 30 Aug, training repo `0ebd281`); it rotates frames and
sketch marks together and maps the pixel-space geometry vectors with them.
Evaluating v5 without it measures orientation, not sketch-following.

### What the build cost, and why (worth not re-learning)

Three environmental defects, all now fixed in `scripts/`:

1. **MuJoCo was rendering on the CPU.** The H200 pod carried only
   `50_mesa.json`, so `MUJOCO_GL=egl` resolved to llvmpipe. The A100 pod has
   `10_nvidia.json` but lacked the `libegl1` dispatcher. Install it and pin
   `__EGL_VENDOR_LIBRARY_FILENAMES` to the nvidia ICD -- `apt` installs the mesa
   vendor beside it -- and a frame renders in 0.8 ms. Four starved shards went
   from ~10 min/episode to finishing 145 episodes in ~3 min.
2. **`nproc` lies inside the container**: 252 host cores against a `cpu.max`
   quota of 26.35 CPUs. Sizing a fan-out from `nproc` is how llvmpipe came to
   run ~640 worker threads on ~26 usable CPUs.
3. **`np.load` on a compressed npz is lazy.** `pack_paired_corpus` indexed
   `z["images"][t]` in a loop, re-inflating the whole frame stack once per
   frame: 584 GB read from a 7.25 GB corpus, 80 minutes with no shard written.
   Hoisting the arrays packs the same 418 episodes in ~5 min.

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

0. **THE MISMATCH IS ONE ARM FROM CLOSED.** Full account in
   `docs/V7_BASELINE_PARITY_CLOSEOUT.md`. The rendered layer (sketch keying,
   re-rendered masks) and the visual layer (180 degrees, worth the entire score
   -- 96.7% upright vs 0.0% inverted for stock pi0.5 -- and the frame/mask
   augmentation desynchronisation) are both fixed and demonstrated, and the
   corpus redundancy that starved the objective is fixed too. What is left is
   not a defect, it is a **design mismatch**: 40.3% is 37 anchored Spatial
   scenes at 14 rollouts with explicit captions and sustained-5 success, and
   V7's 85.5% is 10 paired layouts at 20 episodes with referent-free captions
   and instantaneous success. **No arm of the baseline has ever been run on the
   paired layouts.** Until one has, "V7 beats the baseline" is two designs
   compared, which is the same error as reporting a real arm without its swap.
   `scripts/eval_paired_referent.py --policy pi05` and
   `drivers/v7_baseline_arm.sh` run it; `drivers/v7_baseline_compare.py`
   scores it and refuses to print without both baseline arms.

   The mirror-image arm -- V7 on the anchored scenes -- is specified in
   `docs/V7_ANCHORED_ARM.md` and driven by `drivers/v7_anchored_arm.sh`.
   `eval_sketchvla.py` already has a `--sketch-mode swap`; what it never had is
   a check that the moved ring still means one thing, and **only 26 of the 37
   scenes pass one**: for 8 the ring runs off the frame, for 2 it also contains
   `bowl_1`, for 1 it encloses a plate. v5's and v6's swap arms ran the same
   unchecked construction over all 37, so both included clipped or ambiguous
   rings. Re-scored on the clean 26, v6 is still null and sharper: bowl_1
   -20.36 (was -15.96), bowl_2 still +3.85 at 0.57 sigma, uncircled bowl_3
   +9.07. Cleaning the rings did not rescue v6, so the defect is a
   reporting-precision problem and not the explanation of the null. Those 26 scenes are
   harder than the full set, so the baseline that arm must be read against is
   **31.0% explicit / 27.8% ambiguous**, not 40.3 / 36.5
   (`scripts/analysis/anchored_subset`). Both arms must run on the same 26, or
   `score_referent_following.py`'s scene pairing is not a pairing.

0bis. **Two things never done, and they gate the interpretation of everything
   above.** Detail and proposed tests in
      `SketchPromptVLA-Pi@feat/eval-harness:docs/SESSION_2026-09-01.md`.

   a. **Neither repo's version is pinned at the point a number is produced**, so
      the same nominal script gives different results in different places.
      Measured on the pod: the training repo sits at `265fe94` while origin is at
      `7d3527c`, because pods cannot `git fetch` non-interactively (https remote,
      interactive credentials) and code arrives as base64 over the tty; the
      working tree carried four modified and four untracked files; `git stash
      list` holds three stashes from three different sessions; and
      `/workspace/harness_repo` is not a git repository at all, so a rollout's
      harness has no version identity. Fix: write both SHAs into every result
      file, give the harness bundle a VERSION file, give pods a fetchable remote,
      clear the stale stashes.

   b. **"Moving the target object degrades almost every VLA" has never been
      tested.** The paired corpus is built by relocating a bowl, so if relocation
      alone is expensive then the ~40% ceiling every arm hits may be a relocation
      artifact and not a grounding failure — and every sketch number in this file
      is being read against a floor nobody has established. Test it with stock
      `pi05_libero` (96.7% upright) on shipped scenes versus the same scenes with
      the target relocated, explicit captions, no sketch pathway in the loop.


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

   **Full matrix — measured 29 Aug, all five pairs, 25 demos each**
   (`scripts/replay_donor_pose.py`, log
   `outputs/paired_spatial_bddl/replay_pair_matrix.log`). The distractor pose
   is taken from a shipped episode of the partner task (bowl_1's free-joint
   qpos out of the donor's `states[0]`), which is what handles stacked and
   in-drawer placements and is the mechanism the corpus builder should use.

   | replay task <- donor pose | control | moved |
   |---|---|---|
   | next_to_ramekin <- next_to_plate (xy, 50 demos) | 45/50 | 35/50 |
   | between_plate_ramekin <- table_center | 23/25 | 22/25 |
   | table_center <- between_plate_ramekin | 22/25 | 20/25 |
   | in_top_drawer <- next_to_box | 18/25 | 23/25 |
   | next_to_box <- in_top_drawer (into closed drawer) | 23/25 | **12/25** |
   | on_cookie_box <- on_ramekin (stacked) | 20/25 | 20/25 |

   Four of five pairs survive at 78–100% of their control rate; stacking on
   the ramekin costs nothing. The one weak direction is teleporting a bowl
   INTO the closed drawer: the donor pose is absolute, the cabinet pose varies
   per episode, so the bowl clips the drawer walls. Fix in the builder: map
   the donor bowl pose into the donor episode's cabinet frame and re-apply it
   in the replay episode's cabinet frame — or drop that one direction and
   keep the drawer pair one-sided (its partner direction works at 23/25).

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

4. ~~Re-export the training corpus upright, and measure what inversion costs.~~
   **DONE 29 Aug, on the pod.** Two results:

   * **The corpus is upright now.** `SketchPromptVLA-Pi:scripts/rotate_sketch_rlds.py`
     (commit `08e7e00`) rewrites the TFRecord protos directly — images/masks
     `[::-1, ::-1]`, points `(255-x, 255-y)`, actions untouched — so no
     re-render and no schema change. All 500 episodes (450 train + 50 val) at
     `/workspace/data/sketch_libero_rlds_upright/`, verified (coords exact,
     jpeg round-trip mean diff 1.07). The original is untouched; the HF
     re-upload still needs coordinating since old checkpoints saw the
     inverted world.

   * **Inversion alone is fatal — measured, not argued.** Stock pi0.5-LIBERO
     through the shipped eval loop on libero_spatial: **96.7% upright (29/30)
     vs 0.0% inverted (0/50)** (`examples/libero/main.py --args.no-rotate180`,
     commit `08e7e00`; logs `/workspace/logs/probe_{upright,inverted}.log`).
     A frozen-SigLIP policy scores ZERO when its frames are upside-down. The
     fine-tune trained and evaluated on an inverted world with SigLIP frozen
     since v3, so its trainable layers were compensating for a vision tower
     emitting out-of-distribution features. This is the leading explanation
     for the ~2.4% absolute floor, and it says nothing about the
     referential-tier 0% — that remains the corpus redundancy above. **Train
     nothing more on the inverted corpus.**

5. **Deferred:** anchored arms across Object and Goal, for the full-set numbers.

## Retracted, so it is not re-derived

- *"The sketch is destroyed at the cross-attention."* Written on 31 Aug on the
  strength of the uniform attention, and wrong. A uniform softmax returns the
  MEAN of the value vectors, and `CrossAttention.to_kv` carries no bias, so the
  branch output is a linear function of the mean sketch token -- which encodes
  the circle centre at R^2 0.974. Uniform attention does not select; it does not
  discard. The referent reaches the action expert.

- *"His fine-tune trained on my zero-action validation export."* False. It trains
  on the 450-episode corpus with real actions. Audited on the bytes.
- *"The colleague's ~0% is a harness bug."* No. Object-free captions
  (`"do this"`, `"grab this"`) plus a sketch pathway that never opened. Near-zero
  is the correct output of that model.
- *"A parity run would reconcile the two numbers."* They were never measuring the
  same thing. `outputs/sketchvla_baseline_parity.patch`,
  `scripts/run_parity_baseline.sh` and `scripts/run_sketchvla_eval.sh` target a
  script already retired by `eval_sketchvla.py`. Kept as record; do not run.
