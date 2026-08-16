# Sketch-Prompted VLA validation set — LIBERO-**Spatial** suite

38 scenes. `scripts/build_validation_set_spatial.py` (SMOKE=False).
Canonical schema v1.0 (see SCHEMA.md). Backports the Object suite's negative-
oracle, pixel-separation, and restart-sampling gates onto the `On` task.

## Composition

| tier | scenes | bowls N | plates M | meaning |
|---|---|---|---|---|
| control | 5 | 1 | 1 | one-to-one |
| referential | 12 | 2-5 | 1 | many-to-one — which bowl |
| directional | 9 | 1 | 2-4 | one-to-many — which plate |
| both | 12 | 3-5 | 2-3 | many-to-many |

Tier is candidate multiplicity, written as (targets to destinations). It says nothing
about the caption -- see `PROMPT_TAXONOMY.md`.

Goal `(On akita_black_bowl_t plate_d)`; destination is an object instance, so
`destination_region == destination == plate_d`.

## Gates (all must pass)

settled (on table) · target+plates in frame · pixel separation (circle vs
sibling bowls, arrow vs rival plates, from projected extents) · visibility >=
0.35 · not pre-solved · positive oracle (bowl rests On plate) · **negative
oracles**: target bowl -> every other plate False (directional; also catches
plates within On's 3 cm rule), every sibling bowl -> target plate False
(referential) · graspable (lift > 3 cm).

## Measured (min / mean / max)

| metric | value |
|---|---|
| visibility | 0.963 / 0.999 / 1.025 |
| grasp lift (m) | 0.140 / 0.162 / 0.166 |

Rejections: 1 (target x1).

## Notes

- `On(bowl, plate)` is True within 3 cm xy + contact; the directional negative
  oracle is what guarantees a bowl on the wrong plate does NOT satisfy the goal.
- Tier is object/destination multiplicity; agentview rarely occludes, so the
  visibility gate seldom binds.

## Prompt types

Every scene carries two captions, and the tier above describes the **scene**,
not either caption. Definitions in `PROMPT_TAXONOMY.md`.

| prompt type | key in `tokens.json` | example |
|---|---|---|
| explicit | `instruction_explicit` | the authored caption, naming target and destination by category |
| ambiguous | `instruction_ambiguous` | `move this onto that` / `move this into that` |

So this suite is **38 scenes x 2 captions = 76 evaluation rows**, listed in
`evaluation_rows.json`. Select an arm with
`rollout_sketch.py --prompt-type {explicit,ambiguous}`.
