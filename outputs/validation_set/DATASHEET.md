# DrawVLA Validation Set — Datasheet

**Built:** 2026-07-16 · **Scenes:** 38 · **Source:** augmented LIBERO-Spatial (simulator-grounded)

## Purpose
A held-out evaluation set whose scenes are **deliberately un-disambiguable by text alone**.
Each scene augments a LIBERO tabletop with duplicate bowls / plates / ramekins / cookie
boxes, pairs it with a **vague caption**, and provides a human-style **circle** (which
object) + **arrow** (which destination). The set is designed to show that text is
insufficient (and UniVLA's spatial reasoning is not enough) — a sketch modality is needed.

## How each scene is built (fully automatic)
1. Author a real LIBERO **BDDL task** with N `akita_black_bowl`, M `plate`, plus
   `glazed_rim_porcelain_ramekin` / `cookies` clutter; objects placed by rejection
   sampling (min 9.5 cm spacing) in the reachable tabletop rectangle.
2. Load in the MuJoCo/robosuite simulator, settle, render 128×128 `agentview`.
3. Read ground-truth object poses; project to pixels via robosuite's camera matrix
   (projection verified pixel-accurate). Draw a wobbled **green circle** on the target
   bowl and a **red arrow** to the target plate — same draw functions/colours as the
   training pipeline, so train/val distributions match.
4. The BDDL goal names a **specific instance**: `(And (On akita_black_bowl_t plate_d))`.
5. **Teleport oracle:** place the GT bowl on the GT plate, settle, assert the goal
   predicate fires — proving the scene is solvable and the goal is instance-specific.
   Scenes failing any validity gate (object fell off table / target off-frame / oracle
   False) are discarded and resampled.

## Tiers (isolate the two ambiguity axes)
| Tier | Composition | Isolates | Count |
|---|---|---|---|
| control | 1 bowl, 1 plate | text-solvable baseline (no-regression) | 5 |
| referential | 2–5 bowls, 1 plate | which **object** | 12 |
| directional | 1 bowl, 2–4 plates | which **destination** | 9 |
| both | 3–5 bowls, 2–4 plates + clutter | both (the hard "25%" case) | 12 |

## Guarantees / QA (all passing)
- **Oracle-solvable:** 38/38 scenes.
- **Ambiguity:** every non-control scene has >1 candidate object and/or destination.
- **No positional shortcut:** target instance is random and placed at a random location
  (target ≠ instance_1 in 16/24 multi-bowl and 16/21 multi-plate scenes).
- **Distribution match:** 128×128 agentview, identical sketch rendering to training.

## Per-scene files (`scene_XXXX/`)
- `scene.bddl` — the LIBERO task (self-contained; enables GPU-side rollout scoring).
- `frame0.png` — clean initial observation.
- `sketch.png` — observation + circle + arrow (rendered overlay).
- `tokens.json` — symbolic tokens: `circle=(cx,cy,rx,ry)`, `arrow=(x0,y0,x1,y1)` + caption + target ids.
- `meta.json` — counts, tier, target ids, placements, all object pixels, camera matrix, oracle result.
- Top level: `manifest.json`, `contact_sheet.png`, this datasheet.

## Known limitations (by design / deferred)
- **Headline metric needs the model:** text-only vs text+sketch accuracy is scored by
  rolling out the trained policy (GPU-side); each scene ships its BDDL for exactly that.
- **Graspability:** the oracle teleports the bowl; it does not prove the arm can grasp
  the target through clutter. Dense scenes being hard is intended; a scripted-grasp
  reachability gate can be added if truly-impossible grasps appear.
- **Occlusion:** in the densest scenes the target may be partly occluded; the circle
  still marks its projected centre. A visibility metric is a possible refinement.
- **Format:** stored as per-scene folders + manifest; an RLDS/HDF5 export for the
  UniVLA loader is the remaining packaging step.
