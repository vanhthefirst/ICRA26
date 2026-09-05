# V7 on the 37 anchored Spatial scenes — what it needs, and what the number must be read against

3 September 2026, branch `claude/sketchpromptvla-mismatch-fix-240ubd`.
Mirror-image of `docs/V7_BASELINE_PARITY_CLOSEOUT.md`: that one brings the
baseline onto V7's ground, this one brings V7 onto the baseline's.

---

## The short version

V7 cannot be run on all 37 anchored scenes, and the number it produces must not
be read against 40.3% / 36.5%.

- A V7 arm is unreportable without its swap arm — `score_referent_following.py`
  refuses one arm, and it is right to: `overlay_v6`'s real arm alone read as
  +19.7 points and was a marker degrading the scene representation, not a
  pointer. So the anchored arm needs a distractor-circled sketch per scene.
- `eval_sketchvla.py --sketch-mode swap` already builds one, at run time, from
  `tokens.json` + `meta.json`. It keeps the authored `rx`/`ry` and moves only the
  circle's centre onto `akita_black_bowl_2`, arrow running distractor →
  destination. That convention is right and this repo defers to it: holding the
  ring's shape constant leaves POSITION as the only difference between the arms.
- What it does **not** do is check that the moved ring still means one thing.
  `scripts/build_anchored_swap_sketches.py` audits it scene by scene.
  **Only 26 of the 37 pass.** The other 11 are rejected, with reasons.
- The 26 are **harder than the 37**, so the baseline to compare against is
  **31.0% explicit / 27.8% ambiguous**, not 40.3 / 36.5.

## Why 13 scenes cannot carry a swap arm

The distractor is `akita_black_bowl_2` — `swap_tokens` picks the first
`out_of_focus` instance of the target's category, which is bowl_2 in all 37
scenes, and it is the object v5's and v6's swap arms circled, so this arm stays
comparable with both. The anchored scenes were built so a ring around the TARGET
is unambiguous. Nothing ever required that of a ring around bowl_2, and for 11
scenes it does not hold:

| reason | n | what it is |
|---|---:|---|
| ring falls off the frame | 8 | bowl_2 sits at x=112–124 on a 128 canvas and keeps the target's radius; the ring runs 22–46% past the border. The DATASHEET already documents this hazard for `next_to_the_plate`, where it disqualified the TARGET |
| ring also contains bowl_1 | 2 | the two bowls are 17–18 px apart; a ring on one sits 1.15–1.22 radii from the other, inside the builder's own 1.25 margin |
| ring encloses a plate | 1 | `plate_2` strictly inside the ring on scene_0032, so the mark has a second reading |

**This is retrospective on v5 and v6 too.** Their swap arms ran the same
unchecked construction over all 37 scenes, so both included rings that were
clipped or ambiguous. **Re-scored on the clean 26, v6 is still null** — and a
sharper null than the published one:

| object | all 37 | the 26 |
|---|---:|---:|
| bowl_1, circled in real | −15.96 (3.54σ) | **−20.36 (3.30σ)** |
| bowl_2, circled in swap | +3.11 (0.65σ) | **+3.85 (0.57σ)** |
| bowl_3, uncircled | +7.64 (2.25σ) | **+9.07 (1.93σ)** |

The mark still moves the grasp OFF the object it leaves and still fails to move
it ONTO the object it lands on, with an uncircled bowl gaining more than the
circled one. Cleaning the rings did not rescue v6; it made the signature
clearer. So the ring defect is a reporting-precision problem, not the
explanation of the null.

The gates are the builder's own, applied to the swapped ring: `ELLIPSE_MIN`
(1.25 radii) against same-category instances, because the failure being guarded
against is the ring reading as marking a different *grasp candidate*. A ramekin
1.2 radii out is not a candidate and is recorded as advisory; a second bowl is,
and rejects. A non-candidate strictly *inside* the ring rejects too — that is a
second reading, not a near miss. On the 26 that pass, the target sits 1.29 radii
clear of the swap ring at worst and 2.81 at the median.

Full per-scene numbers in `outputs/validation_set_spatial/swap_manifest.json`.
The scene list is written twice: `swap_scene_list.txt` in this repo's
`suite/dir` spelling, `swap_scene_list_bare.txt` in the bare spelling
`eval_sketchvla.py --scenes` takes.

**No sketch asset is written.** The evaluator builds the swap tokens itself at
run time, so a `tokens_swap.json` on disk would be a second source of truth that
the run ignores — and the first thing to go stale. An earlier revision of this
script wrote one; it does not any more, and it deletes any it finds.

## The subset is not a random sample — measured

`scripts/analysis/anchored_subset` re-rates any anchored run on the 24. The
scenes where a ring fits cleanly around bowl_2 are the ones where bowl_2 is well
inside the frame and well separated, which goes with a busier layout:

| run | all 37 | the 24 | delta |
|---|---:|---:|---:|
| pi05 anchored, explicit | 0.4035 | **0.3104** | −0.093 |
| pi05 anchored, ambiguous | 0.3649 | **0.2775** | −0.087 |
| overlay_v6, ambiguous, real arm | 0.4035 | 0.3352 | −0.068 |

Reading a 26-scene V7 number against the 37-scene baseline would credit it with
nine points it did not earn. That is the same error as the paired-layouts
comparison in the closeout doc, one level down, and it is why the numbers above
are written here before the run rather than after it.

## The driver

`drivers/v7_anchored_arm.sh`. It runs the real arm, then the swap arm on the
same scene list, then scores the pair — scoring is part of the run, not a
follow-up somebody may or may not get to. It refuses to start without the scene
list, and it aborts if the swap arm fails, because the real arm alone is not
reportable.

Flags verified against `Args` in `examples/libero/eval_sketchvla.py` on
`feat/eval-harness` (repo attached to the session, `a026d38`):
`--sketch-mode swap` exists; `--scenes` takes bare scene dirs; a single-shard
run merges into `results.csv` and writes `summary.json` by itself.

Two things the driver gets right that are easy to get wrong by hand:

- **`--rotate180` is mandatory.** The evaluator defaults it OFF, which matches
  sketch_libero as shipped and every checkpoint up to pcla_v4. V7 trained on the
  upright corpus. Stock pi0.5-LIBERO measures 96.7% upright against 0.0%
  inverted, so getting this wrong measures orientation and nothing else.
- **`--model_variant` on the server.** Without it `serve_policy_sketchvla.py`
  silently defaults to `input_overlay` and loads the wrong architecture with no
  error. The server already running on 8200 was started correctly, so the run
  should reuse it rather than start a second one.

`referent_grounding` needs nothing extra at inference: the model's optional
`target` / `distractor` / `circle_swap` / `arrow_swap` inputs are training
labels, and `sketch_libero_policy` only forwards them when the corpus carries
them, so the element `build_element` already sends is complete.

## Rebuilding the assets

```bash
python scripts/build_anchored_swap_sketches.py            # writes 24 tokens_swap.json
python scripts/build_anchored_swap_sketches.py --strict   # non-zero exit if any scene rejects
python scripts/analysis/anchored_subset outputs/rollouts/pi05_anchored_explicit_518
```

It is deterministic — same seeds, same tokens — and it deletes the asset from a
scene it now rejects, so a tightened gate cannot leave a stale sketch behind for
a runner that reads the directory rather than the manifest.

## When the run happens

Both arms on the same 24 scenes (`--scenes` from `swap_scene_list.txt`),
ambiguous captions, 14 rollouts, `--rotate180`, into
`outputs/rollouts/sketchvla_rg_v7_ambiguous_sketch` and `..._swap` — the two
directory names `score_referent_following.py` already names in its own usage
example. Then:

```bash
python scripts/score_referent_following.py \
  --real outputs/rollouts/sketchvla_rg_v7_ambiguous_sketch \
  --swap outputs/rollouts/sketchvla_rg_v7_ambiguous_swap \
  --out outputs/referent_following_v7.json
```

Read the verdict, not the real arm: `effect[bowl_2]` large and positive,
`effect[bowl_1]` large and negative by about as much, everything else ~0. v6
returned **null** on exactly this test with a real arm that looked like a
breakthrough.

For the write-up, `drivers/v7_anchored_report.py` does both halves in one
command — the baseline comparison recomputed on whatever scenes the run
actually covered, then the referent verdict — and warns before printing a
number if the two arms cover different scenes or the run strays from the
audited list.

```bash
python drivers/v7_anchored_report.py \
  --real outputs/rollouts/sketchvla_rg_v7_ambiguous_sketch \
  --swap outputs/rollouts/sketchvla_rg_v7_ambiguous_swap
```

Rehearsed on v6's pair, and the rehearsal is itself the warning this file keeps
making. On the 26 scenes v6 is **indistinguishable from the baseline on task
success** (0.3352 vs 0.3104 explicit, +0.025, interval crosses zero) and
**clearly better at grasping the right object** (0.6181 vs 0.4698, +0.148,
excludes zero) — while the referent test on the same rows returns null. Picking
the correct bowl more often is not evidence of reading the mark. Both halves or
neither.
