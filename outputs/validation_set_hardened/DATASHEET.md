# DrawVLA Validation Set (HARDENED) — Datasheet

**Built:** 2026-07-16 · **Scenes:** 38 · **Source:** augmented LIBERO-Spatial (simulator-grounded)
**Supersedes:** `outputs/validation_set/` (adds reproducibility + grasp/visibility gates)

## Purpose
A held-out evaluation set whose scenes are **un-disambiguable by text alone**. Each scene
augments a LIBERO tabletop with duplicate bowls / plates / ramekins / cookie boxes, pairs
it with a **vague caption**, and provides a human-style **circle** (which object) + **arrow**
(which destination). Designed to show text is insufficient (and UniVLA's spatial reasoning
is not enough) — a sketch modality is needed.

## Construction (fully automatic, per scene)
1. Author a real LIBERO **BDDL task** with N `akita_black_bowl`, M `plate`, plus
   `glazed_rim_porcelain_ramekin` / `cookies` clutter; placement by **seeded** rejection
   sampling (min 8.8 cm spacing) → scenes are **reproducible from the recorded seed**.
2. Load in MuJoCo/robosuite, settle, render 128×128 `agentview`.
3. Project ground-truth poses to pixels (verified pixel-accurate); draw a wobbled **green
   circle** on the target bowl and **red arrow** to the target plate — same draw functions
   as training, so train/val distributions match.
4. BDDL goal names a **specific instance**: `(And (On akita_black_bowl_t plate_d))`.

## Validity gates (a scene is discarded + resampled unless ALL pass)
| Gate | Check | Result |
|---|---|---|
| settled | every object stays on the table (z > 0.85) | enforced |
| in-frame | target bowl projects inside the image | enforced |
| **oracle** | teleport GT bowl → GT plate satisfies the goal predicate | 38/38 |
| **visibility** | RGB-diff occlusion fraction of target ≥ 0.35 | min 0.77, ~0.99 typical |
| **graspable** | scripted top-down OSC grasp lifts the target > 3 cm | 38/38 (lift ~0.16 m) |

Gates actively filter: multiple attempts were rejected as `ungraspable`, `fell_off`, or
`oracle_false` and resampled (the hardest 4-bowl/4-plate scene needed 8 tries).

## Tiers (isolate the two ambiguity axes; span the 90%→25% curve)
| Tier | Composition | Isolates | Count |
|---|---|---|---|
| control | 1 bowl, 1 plate | text-solvable baseline (no-regression) | 5 |
| referential | 2–5 bowls, 1 plate | which **object** | 12 |
| directional | 1 bowl, 2–4 plates | which **destination** | 9 |
| both | 3–5 bowls, 2–4 plates + clutter | both (hard "25%" case) | 12 |

## Recorded metrics (per scene, in `meta.json` + `manifest.json`)
- `visibility`  — {v_visible, v_full, visibility}. Ratios marginally >1.0 in a few scenes
  are a benign RGB-diff shadow/contact-edge artifact (≈ fully visible; treat as 1.0).
- `grasp`       — {grasp_success, lift, close_sign}.
- `clearance_xy`— target → nearest-neighbour distance (m).
- `oracle_success`, camera matrix, target ids, all object pixels, placements, seed.

## Guarantees / QA
- **Oracle-solvable:** 38/38.  **Graspable:** 38/38.  **Visibility ≥0.35:** 38/38.
- **Ambiguity:** every non-control scene has >1 candidate object and/or destination.
- **No positional shortcut:** target instance random, placed at a random location.
- **Reproducible:** each scene regenerates exactly from its recorded seed.
- **Distribution match:** 128×128 agentview, identical sketch rendering to training.

## Per-scene files (`scene_XXXX/`)
`scene.bddl` (enables GPU-side rollout scoring) · `frame0.png` · `sketch.png` ·
`target_vismask.png` · `tokens.json` (symbolic circle/arrow + caption + target ids) ·
`meta.json`. Top level: `manifest.json`, `contact_sheet.png`, this datasheet.

## Remaining / deferred
- **Headline metric** (text-only vs text+sketch accuracy) needs the trained policy rolled
  out GPU-side; each scene ships its BDDL for exactly that.
- **Real-human sketch core** (Phase 4): a small subset you hand-draw + synthetic-vs-human
  agreement check.
- **UniVLA/RLDS export:** convert per-scene folders into the loader format the training
  side consumes.
