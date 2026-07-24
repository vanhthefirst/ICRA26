# DrawVLA validation set — LIBERO-**Spatial** suite (hardened v2)

38 scenes. `scripts/build_validation_set_spatial_v2_wsl.py` (SMOKE=False).
Canonical schema v1.0 (see SCHEMA.md). Backports the Object suite's negative-
oracle, pixel-separation, and restart-sampling gates onto the `On` task.

## Composition

| tier | scenes | bowls N | plates M | meaning |
|---|---|---|---|---|
| control | 5 | 1 | 1 | unambiguous |
| referential | 12 | 2-5 | 1 | which bowl |
| directional | 9 | 1 | 2-4 | which plate |
| both | 12 | 3-5 | 2-3 | both |

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
| grasp lift (m) | 0.140 / 0.161 / 0.166 |

Rejections: 13 (siblings x7, ungraspable x4, fell x1, plates x1).

## Notes

- `On(bowl, plate)` is True within 3 cm xy + contact; the directional negative
  oracle is what guarantees a bowl on the wrong plate does NOT satisfy the goal.
- Ambiguity is object/destination multiplicity; agentview rarely occludes, so the
  visibility gate seldom binds.
