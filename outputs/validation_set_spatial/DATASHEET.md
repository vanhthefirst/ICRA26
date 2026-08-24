# Sketch-Prompted VLA validation set — LIBERO-**Spatial**, layout-anchored

37 scenes. `scripts/build_validation_set_spatial_anchored.py`
(SMOKE=False). Canonical schema v1.0 (SCHEMA.md).

Every scene is a shipped `libero_spatial` BDDL with duplicate instances added
and nothing else changed: same objects, same placement regions, same `(:init)`.
`verify_stock_preserved()` asserts this on the emitted text, so the claim is
checked per scene rather than asserted. What differs from stock is the caption
(category-explicit instead of the stock spatial phrase), the goal's destination
instance in the directional and both tiers, and the added copies.

## Base tasks

- `between_plate_ramekin` -> pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl
- `table_center` -> pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl
- `next_to_ramekin` -> pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate.bddl
- `next_to_cookie_box` -> pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate.bddl

Dropped: `in_the_top_drawer_of_the_wooden_cabinet` (bowl_1 starts `In` a closed
drawer, no teleport satisfies the goal) and `next_to_the_plate` (bowl_1 projects
to pixel (116, 44) with half-extent 13, so its silhouette touches the right
border and a circle around it would be clipped by the frame — stock LIBERO's own
framing, and only a problem because that bowl is the target). Held back for
separate measurement: `on_the_cookie_box`, `on_the_ramekin`, `on_the_stove`,
`on_the_wooden_cabinet` (bowl_1 starts stacked or on a fixture).

## Composition

| tier | scenes | in-focus bowls | physical bowls | plates |
|---|---|---|---|---|
| control | 4 | 1 | 2 | 1 |
| referential | 12 | 2-3 | 3-4 | 1 |
| directional | 9 | 1 | 2 | 2-3 |
| both | 12 | 2-3 | 3-4 | 2-3 |

`akita_black_bowl_2` ships in every stock task and stays where stock puts it. It
is out of focus: not a tier candidate, but gated like a sibling and named in
`meta['out_of_focus']`. Read `control` here as "one candidate the sketch is
responsible for", not as a language-only ceiling — the ceiling for these layouts
is the stock-caption number in `outputs/rollouts/openpi_repro_500/`.

## Gates

settled · target and plates fully in frame · strict pixel separation for
INJECTED siblings · drawn-circle clearance (centre outside 1.25x the
stroke) for every sibling INCLUDING `akita_black_bowl_2` · arrow vs rival plates
· visibility >= 0.35 · not pre-solved · positive oracle (bowl rests On the
destination plate) · negative oracles both axes.

Two separation criteria because only one kind of sibling is mine to move. An
injected copy that crowds the target is resampled, so it gets the strict gate.
`akita_black_bowl_2` is stock and fixed, so it gets the criterion the repo
settled on for geometry I do not choose: its centre must fall clear of the
circle actually drawn, measured from the emitted `rx`/`ry` tokens.

Grasp is recorded, not gated: the layout is stock and cannot be resampled, so
gating on the scripted grasp would delete base tasks rather than improve them.
0 scene(s) ship `grasp_success=False`.

## Measured (min / mean / max)

| metric | value |
|---|---|
| visibility | 0.881 / 0.995 / 1.020 |
| grasp lift (m) | 0.162 / 0.165 / 0.170 |

Rejections: 17 (plate x9, circleenclosed x6, fell x1, siblings x1).
