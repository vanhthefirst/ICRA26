# CONTEXT HANDOFF — DrawVLA validation set: GOAL suite (next)

## WHO / WHAT
Aaron (Do Viet Anh), RA at SUTD. Project: DrawVLA / sketch-prompted VLA — a human
draws a CIRCLE (which object) + ARROW (which destination) on the robot camera view
to resolve ambiguity language alone cannot. We build VALIDATION scenes that are
deliberately impossible to disambiguate from a vague caption: many near-identical
objects and/or multiple plausible destinations, where circle+arrow is the ONLY
disambiguating signal. Each scene ships its BDDL so a GPU rollout can score it later
(the headline text-only vs text+sketch experiment needs a trained policy on a GPU —
not runnable on Aaron's 4GB laptop).

## ENVIRONMENT (unchanged, important)
- Repo: `C:\Users\Admin\sketch_vla` == `/mnt/c/Users/Admin/sketch_vla` in WSL2.
- WSL2 + conda env `libero`, Python 3.8. LIBERO source at `/root/LIBERO`. All suite
  BDDL templates ship in the package — NO large downloads; we synthesise scenes.
- The assistant CANNOT run WSL. It writes scripts into the repo; Aaron runs them and
  pastes back output/files. Do NOT modify `data/` (read-only).
- Method: author a real LIBERO BDDL as TEXT (direct string templating), add duplicate
  instances, load with OffScreenRenderEnv, settle, render, project GT to pixels, auto
  draw circle+arrow, validate with gates, package. Vertical-slice -> smoke(~3-4) ->
  full run, always. This has caught a real bug at every stage.

=============================================================================
## STATUS
=============================================================================
- **SPATIAL**: done + hardened + backported to the shared gate stack (v2).
  `outputs/validation_set_hardened/` — 38 scenes. (Old v1 saved as
  `outputs/validation_set_hardened_v1_backup/`.)
- **OBJECT**: DONE. `outputs/validation_set_object/` — 38 scenes.
- **GOAL**: NEXT. Probe written and about to be run (see below).
- **LONG / libero_10**: still POSTPONED (2-3 predicate goals; one circle+arrow can't
  express two actions; Turnon/Close/Open have nothing to circle).

Both finished suites: tiers control 5 / referential 12 / directional 9 / both 12,
each scene passing every gate, independently re-verified (not trusting the build log).
Both carry the canonical schema v1.0.

### What was completed THIS phase
1. **Object suite finished** (`scripts/build_validation_set_object_wsl.py`), 38/38,
   0 verification errors. Key suite facts: container regions must be DECLARED per
   basket `(contain_region (:target basket_k))`; `In` tests the object's BODY origin
   against the contain SITE box; floor workspace is banded; camera ~101 px/m; object
   size caps how many identical copies fit (tall bottles max 3, compact goods 5);
   tilt is a per-category constant, not a pose defect (recorded, not gated).
2. **Cross-checked Object vs Spatial** and unified them to ONE canonical schema
   (`SCHEMA.md` v1.0). Normaliser: `scripts/normalize_validation_schema.py`
   (additive, idempotent, verified). Canonical fields: suite, tier, target,
   destination, destination_region, goal_predicate, instruction, symbolic_tokens,
   pick_px/place_px/radius, camera_matrix, visibility, grasp, clearance_xy, seed,
   oracle_success. `destination_region` = exact 2nd arg of the BDDL goal (Spatial On:
   == destination; Object In: destination + "_contain_region"). Combined manifest
   `outputs/validation_manifest_all.json` (76 scenes) — MUST be keyed by
   `(suite, dir)`, never `dir` or `seed` alone (both collide across suites).
3. **Backported Spatial's gates** — rebuilt Spatial with the Object gate stack
   (`scripts/build_validation_set_spatial_v2_wsl.py`): NEGATIVE ORACLES that gate
   (target->wrong plate must be False — this also catches plates within On's 3cm rule;
   sibling bowl->target plate must be False), pixel-separation resolvability from
   projected extents, restart-based rejection sampling. 38/38, 0 errors. Promoted to
   the live `validation_set_hardened/`.

### Durable knowledge already written down (READ THESE FIRST next time)
- `SUITE_FACTS.md` — the 10 general hard-won facts + Object specifics + the design
  principle "satisfy a gate by construction, not by rejection".
- `SCHEMA.md` — the canonical schema v1.0 contract + rules for the Goal builder to
  stay in-format + the (suite, dir) keying note.
- The Object and Spatial-v2 builders are LINE-FOR-LINE PARALLEL — Goal is the same
  template with per-task On/In and region-typed destinations handled.

=============================================================================
## GOAL SUITE — THE OPEN DESIGN DECISION
=============================================================================
libero_goal has 10 tasks, MIXED predicates (7 On, 1 In, 1 Open, 1 Turnon), same four
object categories every scene (akita_black_bowl, cream_cheese, wine_bottle, plate) but
different fixtures per task (table, wooden_cabinet, flat_stove, wine_rack). Usable for
circle+arrow = the 8 On/In pick-and-place tasks (Open/Turnon have nothing to circle).

The crux: a Goal task's destination is EITHER
  - an OBJECT INSTANCE (e.g. plate_1) -> can be DUPLICATED -> directional tier OK, OR
  - a FIXED REGION (e.g. main_table_stove_front_region, wooden_cabinet_1_top_region)
    -> CANNOT be duplicated -> no "which destination?" ambiguity for that task.
How the 8 usable tasks split between these two is the fact that decides the design.
DO NOT assume the split from memory — the probe measures it.

### THREE OPTIONS UNDER CONSIDERATION
1. **Referential-only for region tasks** (expected pick): region-destination tasks
   feed control + referential tiers (duplicate the TARGET object; arrow points at the
   fixed region). Object-destination tasks additionally feed directional + both
   (duplicate the destination object). All four tiers exist; directional/both lean on
   the object-destination subset.
2. **Object-destination tasks only**: drop region tasks; build all four tiers from
   duplicable-destination tasks only. Uniform and clean, but if only ~2 tasks qualify
   the suite collapses toward a re-run of Spatial. Low value if the split is lopsided.
3. **Decide after the probe** (what we are doing): run the probe, read the actual
   split, then choose 1 vs 2 with real counts.

=============================================================================
## THE PROBE — `scripts/probe_goal_wsl.py`  (run, output -> outputs/probe_goal.txt)
=============================================================================
Run: `conda activate libero && cd /mnt/c/Users/Admin/sketch_vla && \
python scripts/probe_goal_wsl.py 2>&1 | tee outputs/probe_goal.txt`

It has three parts:
- **Section A** — pure BDDL parse of all 10 tasks: language, goal predicate(s),
  target object (+category), destination, USABLE? (On/In only), and destination class
  = OBJECT / OBJECT_CONTAIN (duplicable) vs REGION (fixed).
- **Section A2** — the DECISION TABLE: count of usable tasks, destination-kind counts,
  target-category counts, and explicit lists of object-destination vs
  region-destination task names. This is what picks Option 1 vs 2.
- **Section B** — light SIM confirmation on one object-dest and one region-dest task:
  destination key present in object_states_dict (so the oracle can score it), which
  instances have free joints (what's duplicable), fixtures present (they eat table
  space -> per-task placement rect), and rest height + scripted grasp per pickable
  object (esp. wine_bottle, likely awkward top-down).

=============================================================================
## FIRST ACTION FOR NEXT SESSION
=============================================================================
1. Read `SUITE_FACTS.md` and `SCHEMA.md`.
2. Read the pasted `outputs/probe_goal.txt`. From Section A2 pick Option 1 vs 2 with
   real counts (no memory assumptions). Report the recommendation with the numbers.
3. Then build the Goal vertical slice using the Object/Spatial-v2 builder as the
   template: emit canonical schema (suite="goal", goal_predicate per task), reuse the
   hardened gate stack incl. negative oracles, handle region-typed destinations
   (destination == destination_region set equal for object dests; the exact region arg
   for region dests). Vertical-slice one scene -> smoke -> full 38 -> re-run
   `normalize_validation_schema.py` (add "goal" to SETS) to refresh the combined
   manifest.

### STILL OPEN (not now)
- Backport the same gate stack is DONE for Spatial; Long/libero_10 remains deferred.
- Real-human sketch agreement check; UniVLA/RLDS export — both later.
