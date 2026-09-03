# Closing the baseline / fine-tune mismatch — what is fixed, and the one arm left

Written 3 September 2026, on `claude/sketchpromptvla-mismatch-fix-240ubd`.
Companion to `docs/V7_HANDOFF_CHECKPOINT2999_ROLLOUT80.md` (the rollout) and
`HANDOFF_BASELINE_RECONCILIATION.md` (where the question started).

---

## The question

The evaluation repo reported stock pi0.5-LIBERO at **40.3% explicit / 36.5%
ambiguous**. The training repo's fine-tune reported **2.44%**. A year of this
project has been spent on that gap. It was never one defect, and it was never
one measurement.

## What the gap was made of, and what remains

| layer | what it was | status |
|---|---|---|
| **rendered** | `load_sketches_from_dataset` keyed sketches by `episode_metadata/episode_key`, absent from the evaluation export, so the lookup fell through to `language_instruction` and ~37 anchored scenes sharing six captions were each handed another scene's drawing. The run completes and prints a plausible number. | **fixed** — key on `scene`; assert `N == scene count` |
| **rendered** | Eval masks were not drawn by the training renderer. | **fixed** — re-rendered through the exact code path (circle ring 1,392 px, arrow 613 px, training palette) |
| **visual** | Both RLDS corpora sat 180° from the orientation pi0.5 expects. Measured, not argued: stock pi0.5-LIBERO scores **96.7% upright and 0.0% inverted** on libero_spatial. The fine-tune trained and evaluated in an inverted world with SigLIP frozen. | **fixed** — `rotate_sketch_rlds.py`, all 500 episodes; leading explanation of the 2.44% floor |
| **visual** | `preprocess_observation` gave the frame and the sketch masks *independent* crops and rotations — `augmax.Chain` splits its key per transform and the RGB chain carries a `ColorJitter` the mask chain does not. Displacement on the order of the ring's own radius, every step, since the first run. | **fixed** — one chain, one key, channel-stacked; `check_aug_alignment.py` is the regression test (worst error 0.05 px) |
| **objective** | One layout carried one sketch, so the image alone determined the action and the sketch's share of the loss was under one percent. | **fixed** — paired corpus, `check_sketch_necessity.py` gate, V7's dense per-patch grounding loss plus the distractor counterfactual |
| **design** | **The two numbers still do not measure the same thing.** 40.3% is 37 anchored Spatial scenes, 14 rollouts, explicit captions, sustained-5 success. 85.5% is 10 paired layouts, 20 episodes, referent-free captions, instantaneous success, 520 steps. Different scenes, different captions, different criterion. | **OPEN — this is the whole of what is left** |

Everything above the last row is closed and demonstrated. The last row cannot be
closed by reasoning about it. **No arm of the baseline has ever been run on the
ten paired layouts**, so "V7 beats the baseline" is at present a comparison of
0.855 on one design against 0.403 on another. That is the same category of error
as reporting the real arm without the swap arm, and this project has already
been burned by it twice (`sketch_l2`, and `overlay_v6`'s +19.7 points).

## What this branch adds

Three pieces, all offline work, all verified to compile and — for the scorer —
to run end to end on synthetic rows.

**`scripts/eval_paired_referent.py` gains a baseline policy.** `--policy pi05`
drives a stock pi0.5-LIBERO server (`serve_policy.py --env LIBERO`) through the
identical loop: same layouts, same demo indices, same donor poses, same
`map_donor_pose`, same frame-reconstruction guard, same 520-step budget, same
`_check_success()`. pi0.5 has no sketch channel, so the arm accepts only
`--sketch-modes blank`, and it refuses a server that reports a `checkpoint_dir`
— pointing it at port 8200 would produce a fine-tune number wearing a baseline
label, which is precisely the mistake that started this file.

**`--caption explicit`** swaps the corpus's referent-free caption for the
layout's own BDDL wording. The pairing moves the *distractor* into the partner
task's target region, so the target's own descriptor still names exactly one
bowl — that admissibility is what `check_sketch_necessity.py` certified before
the corpus was built. Without this arm there is no ceiling to read the fine-tune
against.

**`--success-window`** records both criteria on every row. `success` is the
instantaneous test that every V7 chunk already ran under (`window=1`, the
default, is byte-for-byte the old behaviour); `sustained_success` is the
window the anchored baselines used. Mixing a windowed rate with an
instantaneous one is a third design difference, and now it is visible in the
row rather than in a runbook.

**`drivers/v7_baseline_arm.sh`** runs the two baseline arms in 100-row chunks
with the same row-count / traceback / SHA-256 discipline as the rollout driver.
**`drivers/v7_baseline_compare.py`** prints the block, with Wilson intervals,
and refuses to print at all if either baseline arm is missing.

## The run that closes it

The pod, the V7 server on 8200 and the checkpoint are all still required. The
baseline needs a **second** server on 8300, in the openpi venv:

```bash
cd /workspace/SketchPromptVLA-Pi && uv run scripts/serve_policy.py --env LIBERO --port 8300
```

Then, from `/workspace/eval_scripts`:

```bash
bash drivers/v7_baseline_arm.sh          # 400 episodes: explicit x10, stored x10
```

and finally, once the V7 rollout's own 400 rows and the 200 blank rows are in:

```bash
python drivers/v7_baseline_compare.py \
  --v7-real  outputs/v7_paired_step2999_merged_all400.json \
  --v7-swap  outputs/v7_paired_step2999_merged_all400.json \
  --v7-blank outputs/v7_blank_step2999_blank_t1_t5.json \
             outputs/v7_blank_step2999_blank_t6_t10.json \
  --pi05-explicit outputs/v7_baseline_pi05_explicit_t1_t5.json \
                  outputs/v7_baseline_pi05_explicit_t6_t10.json \
  --pi05-stored   outputs/v7_baseline_pi05_stored_t1_t5.json \
                  outputs/v7_baseline_pi05_stored_t6_t10.json \
  --out outputs/v7_baseline_parity_compare.json
```

Passing the merged artifact to both `--v7-real` and `--v7-swap` is correct: rows
are filtered on their own `mode` field and are not double counted.

## How to read the result, decided in advance

- **Ceiling gap** — `pi05+explicit` minus `v7+real` on task success. If the
  interval crosses zero, the fine-tune matches a policy that is *told* the
  referent on these layouts, and the historic 40.3-versus-2.44 mismatch is
  closed and quantified. If pi0.5+explicit lands far below 0.855, the paired
  layouts are *easier* than the anchored suite and V7's number is partly design,
  not capability — say so before anyone reads 0.855 against 0.403.
- **Floor gap** — `pi05+stored` minus `v7+real`. This is the honest version of
  the original comparison: same layouts, same referent-free captions, one model
  can see the mark and the other cannot.
- **Dose** — `p_bowl2` across blank / real / swap. This is the grounding
  question and it is *separate*. The numbers already on the pod
  (0.190 blank, 0.135 real, 0.315 swap; 0.795 blank vs 0.855 real on task
  success) say the mark moves the referent in both directions from the prior,
  which is real evidence — and that swap tops out at 0.315, which is weak
  grounding, not solved grounding. **A fine-tune can close the mismatch and
  still fail the dose.** Report both or neither.

## What is not in scope here, and should not be smuggled in

Running V7 on the 37 anchored Spatial scenes would answer the mirror-image
question ("what does the fine-tune score where the baseline was measured").
It needs sketches for scenes the paired corpus does not cover, and it is a
second corpus build, not a flag. It is the natural follow-up; it is not what
closes this.
