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
- The anchored export never carried one. `scripts/build_anchored_swap_sketches.py`
  builds it, with the anchored builder's own `draw_circle` / `draw_arrow` at each
  scene's own seed — same jitter, same wobble, same strokes. Only the circle's
  centre and the arrow's tail move; the destination is untouched.
- **Only 24 of the 37 scenes admit one.** The other 13 are rejected, with reasons.
- The 24 are **harder than the 37**, so the baseline to compare against is
  **29.5% explicit / 25.9% ambiguous**, not 40.3 / 36.5.

## Why 13 scenes cannot carry a swap arm

The distractor is `akita_black_bowl_2` in every scene — the object v5's and v6's
swap arms circled, so this arm stays comparable with both. The anchored scenes
were built so a ring around the TARGET is unambiguous. Nothing ever required
that of a ring around bowl_2, and for 13 scenes it does not hold:

| reason | n | what it is |
|---|---:|---|
| ring falls off the frame | 8 | bowl_2 sits at x=112–124 on a 128 canvas; a radius-13 ring runs past the border. The DATASHEET already documents this hazard for `next_to_the_plate`, where it disqualified the TARGET |
| ring also contains bowl_1 | 2 | the two bowls are 17–18 px apart with radius 13; a ring on one contains the other |
| ring encloses a non-bowl | 3 | cookie box or plate strictly inside the ring, so the mark has a second reading |

The gates are the builder's own, applied to the swapped ring: `ELLIPSE_MIN`
(1.25 radii) against same-category instances, because the failure being guarded
against is the ring reading as marking a different *grasp candidate*. A ramekin
1.2 radii out is not a candidate and is recorded as advisory; a second bowl is,
and rejects. Full per-scene numbers in
`outputs/validation_set_spatial/swap_manifest.json`; the runnable list in
`swap_scene_list.txt`.

The radius for the distractor is derived, not invented: the builder sized the
ring from the target's projected geom spread, which needs a live simulator, so
the distractor's radius is the target's scaled by the ratio of the two
`px_extent` values, under the builder's own clamp — and then *checked*, scene by
scene, by the gates above. The target ends up 1.32 radii clear of the swap ring
at worst, 3.75 at the median.

## The subset is not a random sample — measured

`scripts/analysis/anchored_subset` re-rates any anchored run on the 24. The
scenes where a ring fits cleanly around bowl_2 are the ones where bowl_2 is well
inside the frame and well separated, which goes with a busier layout:

| run | all 37 | the 24 | delta |
|---|---:|---:|---:|
| pi05 anchored, explicit | 0.4035 | **0.2946** | −0.109 |
| pi05 anchored, ambiguous | 0.3649 | **0.2589** | −0.106 |
| overlay_v6, ambiguous, real arm | 0.4035 | 0.3214 | −0.082 |

Reading a 24-scene V7 number against the 37-scene baseline would credit it with
eleven points it did not earn. That is the same error as the paired-layouts
comparison in the closeout doc, one level down, and it is why the numbers above
are written here before the run rather than after it.

## What is still blocking the run itself

Two things, neither of which is in this repository:

1. **The pod.** No GPU and no `ssh` in the session this was written from. The V7
   server for the run already exists — port 8200, variant `referent_grounding`,
   checkpoint `rg_v7_paired/2999`.
2. **The evaluator's flags.** The anchored CSVs in `outputs/rollouts/` were
   written by `examples/libero/eval_sketchvla.py` on
   `SketchPromptVLA-Pi@feat/eval-harness` — `scripts/analyze_sketchvla.py` says
   so, and the 518-row shape (37 scenes x 14) matches. That script is not
   readable from here (the repository is outside this session's scope, and
   `add_repo` was refused), so the exact flags for the anchored suite and for
   pointing the swap arm at `tokens_swap.json` are unverified. **A driver
   written on a guessed flag set would be worse than none**, so none is
   committed. Paste `eval_sketchvla.py --help` — or grant the repo — and the
   driver is a short follow-up.

`serve_policy_sketchvla.py` must always be given `--model_variant`; without it
it silently defaults to `input_overlay` and loads the wrong architecture with no
error (`RUNBOOK_EVAL_POD.md`). The server already running was started correctly.

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
