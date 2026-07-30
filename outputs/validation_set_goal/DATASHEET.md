# Sketch-Prompted VLA validation set — LIBERO-**Goal** suite

38 scenes. Built by `scripts/build_validation_set_goal_wsl.py` (VSLICE=False,
SMOKE=False); `DATASHEET.md` and `contact_sheet.png` written by
`scripts/package_goal_suite.py`. Canonical schema v1.0 (see SCHEMA.md).
Companion to `outputs/validation_set_spatial/` and `outputs/validation_set_object/`.

## Purpose

Scenes that are deliberately **impossible to disambiguate from the caption alone**.
Each scene pairs a vague instruction with several identical candidate objects
and/or several candidate destinations. A circle (which object) + arrow (which
destination) is the only signal that identifies the intended instance. The BDDL
goal names one specific instance, so a rollout can be scored automatically.

What makes Goal different from its two siblings: it has **no single flat
workspace**. Each LIBERO-Goal task ships a bespoke scene with real fixtures
(`wooden_cabinet`, `flat_stove`, `wine_rack`) and affordance regions, so the
builder starts from the ORIGINAL BDDL and injects duplicate instances, keeping
every fixture, region and `(:init)` line intact and retargeting only `(:goal)`.

## Composition

| tier | scenes | targets N | dests M | meaning |
|---|---|---|---|---|
| control | 5 | 1 | 1 | unambiguous; caption alone suffices |
| referential | 12 | 2-4 | 1 | which object is ambiguous |
| directional | 9 | 1 | 2-3 | which destination is ambiguous |
| both | 12 | 2-4 | 2-3 | both axes ambiguous |

### Tasks covered

| task | scenes | destination kind |
|---|---|---|
| `bowl_on_plate` | 14 | OBJECT |
| `cheese_in_bowl` | 13 | OBJECT |
| `bowl_on_stove` | 3 | REGION |
| `wine_on_rack` | 3 | REGION |
| `plate_to_stove_front` | 2 | REGION |
| `bowl_on_cabinet` | 2 | REGION |
| `wine_on_cabinet` | 1 | REGION |

Destination kind overall: OBJECT 27, REGION 11.
Goal predicate: On 38.
Target categories: akita_black_bowl x19, cream_cheese x13, wine_bottle x4, plate x2.

**Why the tier/task shape is uneven.** A Goal destination is either an OBJECT
instance (duplicable → supports directional/both) or a fixed affordance REGION
(not duplicable → referential-only). The probe (`outputs/probe_goal.txt`, §A2)
measured the split as 2 object-dest vs 6 region-dest tasks, and the builder
implements **Option 1**: every usable task feeds control+referential, while the
2 object-dest tasks additionally feed directional+both. The drawer task was
dropped after all 24 seeds scored `oracle_false` — the drawer starts closed, so
the `In` region site is retracted and no teleport satisfies it. Opening then
inserting is two actions, which one circle+arrow cannot express (the same
rationale that postpones `libero_10`). Roster: 7 usable tasks, all `On`.

## Per scene

`scene.bddl` `frame0.png` `sketch.png` `tokens.json` `meta.json`
(128x128, camera `agentview`). `meta.json` records the seed, camera matrix, all
object pixels, placements, visibility, grasp, clearance, pixel separations, and
the full oracle result matrix. Every scene is reproducible from its seed.

## Gates (all must pass; failures resample with a new seed)

- **settled** — every body above the floor plane
- **in-frame** — target and all destinations project inside the image
- **pixel separation (sketch resolvability)** — thresholds derive from the
  *projected extents* of the objects being told apart, not constants
- **visibility** — RGB-difference occlusion fraction of target >= 0.35
- **not pre-solved** — `check_success()` is False at t=0
- **oracle +** — teleporting the target to the named destination scores True
- **oracle - (directional)** — target into *every other* destination scores False
- **oracle - (referential)** — *every* same-category sibling into the named
  destination scores False

The two negative oracles are what certify a scene is genuinely unsolvable
without the sketch, along each axis independently. Negative oracles per scene:
0→5 scenes, 1→10 scenes, 2→11 scenes, 3→8 scenes, 4→4 scenes
(control scenes have none by construction — nothing to disambiguate).

**Grasp is RECORDED, NOT GATED in this suite** — see Known limitations.

## Measured distributions (min / mean / max)

| metric | value |
|---|---|
| visibility | 0.652 / 0.988 / 1.005 |
| grasp lift (m) | 0.000 / 0.142 / 0.167 |
| clearance xy (m) | 0.079 / 0.160 / 0.396 |
| px sep, dests | 15.5 / 32.4 / 54.1 |
| px sep, siblings | 14.6 / 29.9 / 60.9 |

Positive oracle: 38/38 True.
Rejections during generation: 17 (siblings x9, oracle x4, fell x3, dest x1).

## Known limitations

- **Grasp is recorded, not gated — 6 of 38 scenes have
  `grasp_success: False`** (`wine_on_rack` x3, `plate_to_stove_front` x2, `wine_on_cabinet` x1). This is a deliberate decision, not a
  defect. See the dedicated section below.
- **Directional and both tiers rest on only 2 tasks** (`bowl_on_plate`,
  `cheese_in_bowl`), the only ones with a duplicable object destination.
  Region-destination tasks cannot express "which destination?" ambiguity.
- **Ambiguity is multiplicity, not occlusion.** `agentview` looks down steeply,
  so objects rarely occlude one another and the visibility gate seldom binds.
  Do not cite this set as evidence about occlusion robustness.
- **All 38 scenes use the `On` predicate.** The one `In` task (the drawer) was
  dropped, so this suite does not exercise `In` at all — the Object suite is
  where `In` coverage lives.
- Scoring the headline experiment (text-only vs text+sketch) needs a trained
  policy on a GPU. Each scene ships its BDDL for exactly that.

## The grasp gate: why this suite differs from Spatial and Object

Spatial and Object both **reject** a scene when the scripted top-down grasp fails.
Goal records the result and keeps the scene. That is a real divergence, and the
justification rests on what the gate is measuring in each suite.

| target category | scenes | grasp True | grasp False |
|---|---|---|---|
| `akita_black_bowl` | 19 | 19 | 0 |
| `cream_cheese` | 13 | 13 | 0 |
| `wine_bottle` | 4 | 0 | 4 |
| `plate` | 2 | 0 | 2 |

**The split is perfectly categorical — zero within-category variance.** Every
`akita_black_bowl` and `cream_cheese` scene grasps; no `wine_bottle` or `plate`
scene ever does. So in this suite a grasp failure is a property of the *object
category* under a scripted top-down gripper, not of a particular sampled
placement. Resampling the seed cannot fix it.

Contrast Object: all 38 of its kept scenes grasp successfully, and its 15
`ungraspable` rejections were all *graspable* grocery categories — those were bad
placements, and resampling did fix them. **The same gate therefore does different
work in the two suites:** in Object it filters unlucky samples; in Goal it would
not filter anything, it would delete the `wine_on_rack`, `wine_on_cabinet` and
`plate_to_stove_front` tasks outright, taking the suite from 7 usable tasks to 4.

What certifies a scene is `oracle_success` — teleporting the target to the named
destination scores the BDDL goal True — and that is independent of the scripted
grasp. All 38/38 scenes pass it.

### Consequence for the headline experiment (read this before scoring)

Keeping these scenes has a cost: a policy that cannot physically lift a wine
bottle will score near zero on those 6 scenes in **both** the
text-only and the text+sketch condition. The comparison stays valid — both arms
face identical physical difficulty — but the floor effect compresses the measured
effect size and could mask a real difference.

Recommended reporting: **stratify.** Headline number on the 32 grasp-True
scenes, with the 6 grasp-False scenes reported separately as a
manipulation-limited subset. Filter with
`meta['grasp']['grasp_success']`; those scenes are flagged `g!` on the contact
sheet. Do not silently pool them.

## Reproduce

```bash
conda activate libero
cd /mnt/c/Users/Admin/sketch_prompted_vla
python scripts/build_validation_set_goal_wsl.py     # rebuilds all 38 scenes
python scripts/package_goal_suite.py                # this file + contact sheet
python scripts/normalize_validation_schema.py       # refresh canonical manifests
python scripts/audit_validation_sets.py             # read-only verification
```
